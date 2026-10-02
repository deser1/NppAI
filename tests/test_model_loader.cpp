#include "NppAIEngine.h"
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>

int main() {
    const std::string path = "nppai_test_invalid_model.nppai";

    // Header claims an impossible dimension. Loader must reject it before
    // allocating model tensors.
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        int header[5] = {1 << 30, 64, 1, 16, 256};
        out.write(reinterpret_cast<const char*>(header), sizeof(header));
    }

    NppAIEngine engine;
    const bool loaded = engine.loadModel(path);
    std::remove(path.c_str());

    if (loaded) {
        std::cerr << "FAIL: invalid model was accepted\n";
        return 1;
    }

    std::cout << "Model loader validation passed.\n";
    return 0;
}
