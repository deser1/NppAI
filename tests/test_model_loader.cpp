#include "NppAIEngine.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include <cstdint>

static bool expectRejected(NppAIEngine& engine, const std::string& path,
                           const char* label) {
    if (engine.loadModel(path)) {
        std::cerr << "FAIL: " << label << " was accepted\n";
        std::remove(path.c_str());
        return false;
    }
    std::remove(path.c_str());
    return true;
}


static void writeV2Model(const std::string& path, uint32_t version,
                         const std::vector<char>& payload,
                         const std::array<unsigned char, 32>& sha256) {
    const char magic[8] = {'N','P','P','A','I','\0','\0','\0'};
    const int32_t dimensions[5] = {1, 1, 1, 1, 1};
    const uint64_t payloadBytes = static_cast<uint64_t>(payload.size());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(magic, sizeof(magic));
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(dimensions), sizeof(dimensions));
    out.write(reinterpret_cast<const char*>(&payloadBytes), sizeof(payloadBytes));
    out.write(reinterpret_cast<const char*>(sha256.data()), sha256.size());
    out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
}

int main() {
    {
        const std::string path = "nppai_test_invalid_model.nppai";

        // Header claims an impossible dimension. Loader must reject it before
        // allocating model tensors.
        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            int header[5] = {1 << 30, 64, 1, 16, 256};
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
        }

        NppAIEngine engine;
        if (!expectRejected(engine, path, "invalid model dimensions"))
            return 1;
    }

    {
        const std::string path = "nppai_test_truncated_model.nppai";

        // For a 1x1, 1-layer model the loader expects 13 FP32 values after
        // the 20-byte header. Write one byte less than the complete payload.
        const int header[5] = {1, 1, 1, 1, 1};
        const std::size_t expectedPayloadBytes = 13 * sizeof(float);
        std::vector<char> truncatedPayload(expectedPayloadBytes - 1, 0);

        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            out.write(truncatedPayload.data(),
                      static_cast<std::streamsize>(truncatedPayload.size()));
        }

        NppAIEngine engine;
        if (!expectRejected(engine, path, "truncated model payload"))
            return 1;
    }

    {
        const std::string path = "nppai_test_trailing_payload_model.nppai";

        // Exact payload validation must reject appended/trailing bytes instead
        // of silently accepting data outside the documented model format.
        const int header[5] = {1, 1, 1, 1, 1};
        const std::size_t expectedPayloadBytes = 13 * sizeof(float);
        std::vector<char> payload(expectedPayloadBytes + 1, 0);

        {
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(header), sizeof(header));
            out.write(payload.data(),
                      static_cast<std::streamsize>(payload.size()));
        }

        NppAIEngine engine;
        if (!expectRejected(engine, path, "model payload with trailing data"))
            return 1;
    }

    {
        const std::string path = "nppai_test_v2_valid.nppai";
        std::vector<char> payload(13 * sizeof(float), 0);
        const std::array<unsigned char, 32> zeroPayloadSha256 = {
            0x79,0x55,0xcb,0x2d,0xe9,0x0d,0xd9,0xef,
            0xc6,0xdf,0x9f,0xdb,0xf5,0xf5,0xd1,0x0c,
            0x11,0x4f,0x41,0x35,0xa9,0xa6,0xb5,0x2d,
            0xb1,0x00,0x3b,0xe7,0x49,0xe3,0x2f,0x7a
        };
        writeV2Model(path, 2, payload, zeroPayloadSha256);
        NppAIEngine engine;
        if (!engine.loadModel(path)) {
            std::cerr << "FAIL: valid v2 model was rejected\n";
            std::remove(path.c_str());
            return 1;
        }
        std::remove(path.c_str());
    }

    {
        const std::string path = "nppai_test_v2_corrupted.nppai";
        std::vector<char> payload(13 * sizeof(float), 0);
        const std::array<unsigned char, 32> originalSha256 = {
            0x79,0x55,0xcb,0x2d,0xe9,0x0d,0xd9,0xef,
            0xc6,0xdf,0x9f,0xdb,0xf5,0xf5,0xd1,0x0c,
            0x11,0x4f,0x41,0x35,0xa9,0xa6,0xb5,0x2d,
            0xb1,0x00,0x3b,0xe7,0x49,0xe3,0x2f,0x7a
        };
        payload[0] = 1; // Hash still describes the original all-zero payload.
        writeV2Model(path, 2, payload, originalSha256);
        NppAIEngine engine;
        if (!expectRejected(engine, path, "v2 model with corrupted payload"))
            return 1;
    }

    {
        const std::string path = "nppai_test_v2_unsupported.nppai";
        std::vector<char> payload(13 * sizeof(float), 0);
        std::array<unsigned char, 32> sha256{};
        writeV2Model(path, 999, payload, sha256);
        NppAIEngine engine;
        if (!expectRejected(engine, path, "unsupported model format version"))
            return 1;
    }

    std::cout << "Model loader validation passed.\n";
    return 0;
}
