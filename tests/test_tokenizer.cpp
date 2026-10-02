#include "NppAIEngine.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

class NppAITest {
public:
    static bool tokenizerFallback() {
        NppAIEngine engine;
        const auto tokens = engine.tokenize("Hello");
        if (tokens.size() != 5 || tokens[0] != 'H' || tokens[4] != 'o') {
            std::cerr << "FAIL: byte tokenizer fallback\n";
            return false;
        }
        const auto text = engine.detokenize(tokens);
        if (text != "Hello") {
            std::cerr << "FAIL: byte detokenizer fallback\n";
            return false;
        }
        return true;
    }

    static bool tokenizerBPE() {
        NppAIEngine engine;
        const auto path =
            std::filesystem::temp_directory_path() / "nppai_test_bpe_merges.txt";

        {
            std::ofstream file(path);
            if (!file.is_open()) {
                std::cerr << "FAIL: could not create BPE fixture\n";
                return false;
            }

            // Build deterministic merges for "Hello":
            // H + e -> He -> Hel -> Hell -> Hello.
            file << "72 101 256\n";
            file << "256 108 257\n";
            file << "257 108 258\n";
            file << "258 111 259\n";
        }

        const bool loaded = engine.loadBPETokenizer(path.string());
        if (!loaded) {
            std::cerr << "FAIL: BPE fixture could not be loaded\n";
            std::filesystem::remove(path);
            return false;
        }

        const auto first = engine.tokenize("Hello");
        const auto second = engine.tokenize("Hello");

        const std::vector<int> expected = {259};
        if (first != expected || second != expected || first != second) {
            std::cerr << "FAIL: BPE tokenization is not deterministic\n";
            std::filesystem::remove(path);
            return false;
        }

        if (engine.detokenize(first) != "Hello") {
            std::cerr << "FAIL: BPE round-trip detokenization\n";
            std::filesystem::remove(path);
            return false;
        }

        std::filesystem::remove(path);
        return true;
    }
};

int main() {
    if (!NppAITest::tokenizerFallback())
        return 1;
    if (!NppAITest::tokenizerBPE())
        return 1;

    std::cout << "Tokenizer tests passed.\n";
    return 0;
}
