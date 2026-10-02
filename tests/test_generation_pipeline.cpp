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
    // Include BPE merge token 300 in the model vocabulary.
    const int vocab = 301;
    const int header[5] = {dim, hidden, layers, context, vocab};

    // Payload order mirrors NppAIEngine::loadModel.
    const std::size_t floatCount =
        vocab * dim + context * dim +
        layers * (2 * dim + 4 * dim * dim + 3 * dim * hidden) +
        dim + dim * vocab;
    std::vector<float> weights(floatCount, 0.0f);

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out)
        return false;
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(weights.data()),
              static_cast<std::streamsize>(weights.size() * sizeof(float)));
    return static_cast<bool>(out);
}

int main() {
    const auto dir = std::filesystem::temp_directory_path() / "nppai_pipeline_test";
    const auto modelPath = dir / "tiny.nppai";
    std::filesystem::create_directories(dir);

    if (!writeTinyModel(modelPath)) {
        std::cerr << "FAIL: could not create tiny model fixture\n";
        std::filesystem::remove_all(dir);
        return 1;
    }

    // loadModel() discovers the tokenizer next to the model file.
    // Use a merge whose output ID is intentionally unrelated to its rank.
    {
        std::ofstream bpe(dir / "bpe_merges.txt");
        if (!bpe) {
            std::cerr << "FAIL: could not create BPE fixture\n";
            std::filesystem::remove_all(dir);
            return 1;
        }
        bpe << "65 66 300\n";
    }

    NppAIEngine engine;
    if (!engine.loadModel(modelPath.string())) {
        std::cerr << "FAIL: tiny model could not be loaded\n";
        std::filesystem::remove_all(dir);
        return 1;
    }

    // Zero weights make logits equal, so generation remains safe and bounded.
    // This exercises model loading, BPE tokenization with token 300 inside
    // the model vocabulary, forward propagation, sampling and detokenization
    // as one pipeline.
    const std::string prompt = "AB";
    const std::string result = engine.generate(prompt, 1, [](char, bool) {}, [](int) {});
    std::filesystem::remove_all(dir);

    if (result.empty() || result.rfind(prompt, 0) != 0) {
        std::cerr << "FAIL: BPE generation pipeline did not preserve the prompt\n";
        return 1;
    }
    if (result.size() > prompt.size() + 1) {
        std::cerr << "FAIL: generation exceeded requested token bound\n";
        return 1;
    }

    std::cout << "Generation pipeline test passed.\n";
    return 0;
}
