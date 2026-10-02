#include "NppAIEngine.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

static bool writeTinyModel(const std::filesystem::path& path) {
    const int dim = 1;
    const int hidden = 1;
    const int layers = 1;
    const int context = 8;
    const int vocab = 301;
    const int header[5] = {dim, hidden, layers, context, vocab};

    const std::size_t floatCount =
        vocab * dim + context * dim +
        layers * (2 * dim + 4 * dim * dim + 3 * dim * hidden) +
        dim + dim * vocab;
    std::vector<float> weights(floatCount, 0.0f);

    std::size_t offset = 0;
    // Token embeddings: merged token 300 and byte fallback token 66 carry
    // opposite signs so the generation path can distinguish them.
    weights[offset + 66] = -1.0f;
    weights[offset + 300] = 1.0f;
    offset += vocab;

    offset += context;       // positional embeddings
    weights[offset++] = 1.0f; // attention RMS weight
    offset += 4;             // q, k, v, o
    weights[offset++] = 1.0f; // FFN RMS weight
    offset += 3;             // gate, up, down
    weights[offset++] = 1.0f; // output RMS weight

    // Positive hidden state strongly selects X; negative strongly selects Y.
    weights[offset + static_cast<unsigned char>('X')] = 10.0f;
    weights[offset + static_cast<unsigned char>('Y')] = -10.0f;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(weights.data()),
              static_cast<std::streamsize>(weights.size() * sizeof(float)));
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

    if (!writeTinyModel(mergedModel) || !writeTinyModel(fallbackModel)) {
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
