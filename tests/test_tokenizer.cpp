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

    static bool tokenizerRejectsInvalidMerge() {
        NppAIEngine engine;
        const auto path =
            std::filesystem::temp_directory_path() / "nppai_test_invalid_bpe_merges.txt";

        {
            std::ofstream file(path);
            if (!file.is_open()) {
                std::cerr << "FAIL: could not create invalid BPE fixture\n";
                return false;
            }
            // Token 999 is not defined yet, so this merge must be rejected.
            file << "72 999 256\n";
        }

        const bool loaded = engine.loadBPETokenizer(path.string());
        const auto tokens = engine.tokenize("Hello");
        const bool validFallback =
            !loaded && tokens.size() == 5 && tokens[0] == 'H' && tokens[4] == 'o';

        std::filesystem::remove(path);
        if (!validFallback) {
            std::cerr << "FAIL: invalid BPE merge was not rejected safely\n";
            return false;
        }
        return true;
    }

    static bool tokenizerRejectsDuplicateOutputId() {
        NppAIEngine engine;
        const auto path =
            std::filesystem::temp_directory_path() / "nppai_test_duplicate_bpe_merges.txt";

        {
            std::ofstream file(path);
            if (!file.is_open()) {
                std::cerr << "FAIL: could not create duplicate BPE fixture\n";
                return false;
            }
            file << "72 101 256\n";
            file << "108 108 256\n";
        }

        const bool loaded = engine.loadBPETokenizer(path.string());
        const auto tokens = engine.tokenize("Hello");
        const bool validFallback =
            !loaded && tokens.size() == 5 && tokens[0] == 'H' && tokens[4] == 'o';

        std::filesystem::remove(path);
        if (!validFallback) {
            std::cerr << "FAIL: duplicate BPE output ID was not rejected safely\n";
            return false;
        }
        return true;
    }

    static bool tokenizerRejectsDuplicatePair() {
        NppAIEngine engine;
        const auto path =
            std::filesystem::temp_directory_path() / "nppai_test_duplicate_bpe_pair.txt";

        {
            std::ofstream file(path);
            if (!file.is_open()) {
                std::cerr << "FAIL: could not create duplicate BPE pair fixture\n";
                return false;
            }
            file << "72 101 256\n";
            file << "72 101 257\n";
        }

        const bool loaded = engine.loadBPETokenizer(path.string());
        const auto tokens = engine.tokenize("Hello");
        const bool validFallback =
            !loaded && tokens.size() == 5 && tokens[0] == 'H' && tokens[4] == 'o';

        std::filesystem::remove(path);
        if (!validFallback) {
            std::cerr << "FAIL: duplicate BPE pair was not rejected safely\n";
            return false;
        }
        return true;
    }

    static bool tokenizerUsesMergeRankNotTokenId() {
        NppAIEngine engine;
        const auto path =
            std::filesystem::temp_directory_path() / "nppai_test_bpe_rank.txt";

        {
            std::ofstream file(path);
            if (!file.is_open()) {
                std::cerr << "FAIL: could not create BPE rank fixture\n";
                return false;
            }

            // File order defines rank. The first merge intentionally has a
            // larger output token ID than the second merge.
            file << "97 98 300\n";
            file << "98 99 256\n";
        }

        const bool loaded = engine.loadBPETokenizer(path.string());
        const auto tokens = engine.tokenize("abc");
        std::filesystem::remove(path);

        const std::vector<int> expected = {300, 'c'};
        if (!loaded || tokens != expected) {
            std::cerr << "FAIL: BPE merge priority followed token ID instead of rank\n";
            return false;
        }

        if (engine.detokenize(tokens) != "abc") {
            std::cerr << "FAIL: ranked BPE merge did not round-trip\n";
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
    if (!NppAITest::tokenizerUsesMergeRankNotTokenId())
        return 1;
    if (!NppAITest::tokenizerRejectsInvalidMerge())
        return 1;
    if (!NppAITest::tokenizerRejectsDuplicateOutputId())
        return 1;
    if (!NppAITest::tokenizerRejectsDuplicatePair())
        return 1;

    std::cout << "Tokenizer tests passed.\n";
    return 0;
}
