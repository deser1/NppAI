#include "NppAIEngine.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cstdint>
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")

static bool sha256(const std::vector<char>& data, std::array<unsigned char, 32>& digest) {
    BCRYPT_ALG_HANDLE algorithm = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    DWORD objectBytes = 0, resultBytes = 0;
    std::vector<unsigned char> object;
    bool ok = false;

    if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr, 0) < 0)
        goto cleanup;
    if (BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                          reinterpret_cast<PUCHAR>(&objectBytes), sizeof(objectBytes),
                          &resultBytes, 0) < 0)
        goto cleanup;
    object.resize(objectBytes);
    if (BCryptCreateHash(algorithm, &hash, object.data(), objectBytes, nullptr, 0, 0) < 0)
        goto cleanup;
    if (!data.empty() &&
        BCryptHashData(hash, reinterpret_cast<PUCHAR>(const_cast<char*>(data.data())),
                       static_cast<ULONG>(data.size()), 0) < 0)
        goto cleanup;
    if (BCryptFinishHash(hash, digest.data(), static_cast<ULONG>(digest.size()), 0) < 0)
        goto cleanup;
    ok = true;

cleanup:
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    return ok;
}

static bool writeTinyModelV2(const std::filesystem::path& path, bool corruptAfterHash = false) {
    const int32_t dim = 1, hidden = 1, layers = 1, context = 8, vocab = 301;
    const std::size_t floatCount =
        vocab * dim + context * dim +
        layers * (2 * dim + 4 * dim * dim + 3 * dim * hidden) +
        dim + dim * vocab;
    std::vector<float> weights(floatCount, 0.0f);

    std::size_t offset = 0;
    weights[offset + 66] = -1.0f;
    weights[offset + 300] = 1.0f;
    offset += vocab;
    offset += context;
    weights[offset++] = 1.0f;
    offset += 4;
    weights[offset++] = 1.0f;
    offset += 3;
    weights[offset++] = 1.0f;
    weights[offset + static_cast<unsigned char>('X')] = 10.0f;
    weights[offset + static_cast<unsigned char>('Y')] = -10.0f;

    std::vector<char> payload(weights.size() * sizeof(float));
    std::memcpy(payload.data(), weights.data(), payload.size());
    std::array<unsigned char, 32> digest{};
    if (!sha256(payload, digest))
        return false;
    if (corruptAfterHash && !payload.empty())
        payload[0] ^= 1;

    const char magic[8] = {'N','P','P','A','I','\0','\0','\0'};
    const uint32_t version = 2;
    const int32_t dimensions[5] = {dim, hidden, layers, context, vocab};
    const uint64_t payloadBytes = static_cast<uint64_t>(payload.size());

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&payloadBytes), sizeof(payloadBytes));
    out.write(reinterpret_cast<const char*>(digest.data()), digest.size());
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    return static_cast<bool>(out);
}

int main() {
    const auto root = std::filesystem::temp_directory_path() / "nppai_pipeline_test";
    const auto mergedDir = root / "merged";
    const auto fallbackDir = root / "fallback";
    const auto mergedModel = mergedDir / "tiny.nppai";
    const auto fallbackModel = fallbackDir / "tiny.nppai";
    std::filesystem::remove_all(root);
    std::filesystem::create_directories(mergedDir);
    std::filesystem::create_directories(fallbackDir);

    if (!writeTinyModelV2(mergedModel) || !writeTinyModelV2(fallbackModel)) {
        std::cerr << "FAIL: could not create tiny model fixtures\n";
        std::filesystem::remove_all(root);
        return 1;
    }

    {
        std::ofstream bpe(mergedDir / "bpe_merges.txt");
        if (!bpe) {
            std::cerr << "FAIL: could not create BPE fixture\n";
            std::filesystem::remove_all(root);
            return 1;
        }
        bpe << "65 66 300\n";
    }

    const auto corruptModel = root / "corrupt.nppai";
    if (!writeTinyModelV2(corruptModel, true)) {
        std::cerr << "FAIL: could not create corrupt v2 fixture\n";
        std::filesystem::remove_all(root);
        return 1;
    }
    NppAIEngine corruptEngine;
    if (corruptEngine.loadModel(corruptModel.string())) {
        std::cerr << "FAIL: corrupted v2 payload passed SHA-256 validation\n";
        std::filesystem::remove_all(root);
        return 1;
    }

    NppAIEngine mergedEngine;
    NppAIEngine fallbackEngine;
    if (!mergedEngine.loadModel(mergedModel.string()) ||
        !fallbackEngine.loadModel(fallbackModel.string())) {
        std::cerr << "FAIL: tiny model could not be loaded\n";
        std::filesystem::remove_all(root);
        return 1;
    }

    const std::string mergedResult =
        mergedEngine.generate("AB", 1, [](char, bool) {}, [](int) {});
    const std::string fallbackResult =
        fallbackEngine.generate("AB", 1, [](char, bool) {}, [](int) {});
    std::filesystem::remove_all(root);

    if (mergedResult != "ABX") {
        std::cerr << "FAIL: merged-token generation did not select X\n";
        return 1;
    }
    if (fallbackResult != "ABY") {
        std::cerr << "FAIL: byte-fallback generation did not select Y\n";
        return 1;
    }

    std::cout << "Generation pipeline test passed.\n";
    return 0;
}
