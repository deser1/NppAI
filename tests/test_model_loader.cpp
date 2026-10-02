#include "NppAIEngine.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

    std::cout << "Model loader validation passed.\n";
    return 0;
}
