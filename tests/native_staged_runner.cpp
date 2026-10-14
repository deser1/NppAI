#include "NppAIEngine.h"
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <string>
#include <iterator>

int main(int argc, char** argv) {
    // Optional evaluation-only token budget; keep legacy defaults unchanged.
    int generationLimit = 160;
    if (argc > 5) {
        char* end = nullptr;
        const long parsed = std::strtol(argv[5], &end, 10);
        if (argv[5][0] == '\0' || *end != '\0' || parsed < 1 || parsed > 256) {
            std::cerr << "ERROR: generation token budget must be 1..256\n";
            return 7;
        }
        generationLimit = static_cast<int>(parsed);
    }
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
    std::string assistantPrefix;
    if (argc > 4) {
        std::ifstream prefixFile(argv[4], std::ios::binary);
        if (!prefixFile) return 6;
        char prefixBuffer[4097];
        prefixFile.read(prefixBuffer, sizeof(prefixBuffer));
        if (prefixFile.bad()) return 6;
        assistantPrefix.assign(prefixBuffer, static_cast<std::size_t>(prefixFile.gcount()));
        if (assistantPrefix.size() > 4096) return 6;
    }
    const std::string generationPrompt = prompt + assistantPrefix;
    const std::string result = engine.generate(
        generationPrompt,
        generationLimit,
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
    if (result.compare(0, generationPrompt.size(), generationPrompt) != 0) {
        std::cerr << "ERROR: generated result does not start with the prompt\n";
        return 4;
    }
    const std::string completion = result.substr(prompt.size());
    const std::string generatedBody = result.substr(generationPrompt.size());
    if (generatedBody.find_first_not_of(" \t\r\n") == std::string::npos) {
        std::cerr << "ERROR: no non-whitespace completion was generated\n";
        return 5;
    }
    out << completion;

    std::cout << "\nGenerated " << completion.length()
              << " completion characters successfully.\n";

    return 0;
}
