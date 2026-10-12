#include "NppAIEngine.h"
#include <fstream>
#include <iomanip>
#include <vector>

class NppAITest {
public:
    static Tensor logits(NppAIEngine& engine, const std::vector<int>& tokens) {
        return engine.forward(tokens);
    }
};

int main(int argc, char** argv) {
    if (argc != 4) return 2;
    NppAIEngine engine;
    Tensor::setGpuOverrideForTesting(0);
    if (!engine.loadModel(argv[1])) return 3;
    std::ifstream input(argv[2]);
    std::vector<int> tokens;
    int token;
    while (input >> token) tokens.push_back(token);
    if (tokens.empty() || !input.eof()) return 4;
    Tensor logits = NppAITest::logits(engine, tokens);
    std::ofstream output(argv[3]);
    output << std::setprecision(9);
    for (float value : logits.data) output << value << '\n';
    return output ? 0 : 5;
}
