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

    // The engine returns prompt + completion; score only generated text.
    if (result.compare(0, prompt.size(), prompt) != 0) {
        std::cerr << "ERROR: generated result does not start with the prompt\n";
        return 4;
    }
    const std::string completion = result.substr(prompt.size());
    if (completion.find_first_not_of(" \t\r\n") == std::string::npos) {
        std::cerr << "ERROR: no non-whitespace completion was generated\n";
        return 5;
    }
    out << completion;

    std::cout << "\nGenerated " << completion.length()
              << " completion characters successfully.\n";

    return 0;
}
