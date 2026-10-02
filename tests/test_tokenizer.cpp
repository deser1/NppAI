#include "NppAIEngine.h"
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
};

int main() {
    if (!NppAITest::tokenizerFallback())
        return 1;
    std::cout << "Tokenizer tests passed.\n";
    return 0;
}
