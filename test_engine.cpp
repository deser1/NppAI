#include "src/NppAIEngine.h"
#include <fstream>
#include <cstdlib>
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

    // Optional evaluation prompt and output path; keep legacy smoke-test defaults.
    const std::string prompt = argc > 2 ? argv[2] : "[USER]: Lista todo w Vue\n[AI]:\n";
    const std::string outputPath = argc > 3 ? argv[3] : "output_test.txt";
    const std::string result = engine.generate(
        prompt,
        256,
        [](char, bool) {},
        [](int) {}
    );

    if (result.empty()) {
        std::cerr << "ERROR: generation returned an empty result.\n";
        return 2;
    }

    std::ofstream out(outputPath, std::ios::binary);
    if (!out) {
        std::cerr << "ERROR: cannot create generation output file\n";
        return 3;
    }

    out << result;

    std::cout << "\nGenerated " << result.length()
              << " characters successfully.\n";

    return 0;
}
