#include "src/NppAIEngine.h"
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    const std::string modelPath =
        argc > 1 ? argv[1] : "models/NppAI-model-v1.nppai";

    NppAIEngine engine;

    if (!engine.loadModel(modelPath)) {
        std::cerr << "ERROR: failed to load model: " << modelPath << '\n';
        return 1;
    }

    std::cout << "Model loaded successfully. Starting generation...\n\n";

    const std::string result = engine.generate(
        "[USER]: Lista todo w Vue\n[AI]:\n",
        256,
        [](char, bool) {},
        [](int) {}
    );

    if (result.empty()) {
        std::cerr << "ERROR: generation returned an empty result.\n";
        return 2;
    }

    std::ofstream out("output_test.txt", std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: cannot create output_test.txt\n";
        return 3;
    }

    out << result;

    std::cout << "\nGenerated " << result.length()
              << " characters successfully.\n";

    return 0;
}
