#include "NppAIEngine.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static bool closeEnough(float a, float b) { return std::fabs(a - b) < 1e-5f; }

int main() {
    Tensor e0({1, 2}); e0.data = {2.0f, 4.0f};
    Tensor e1({1, 2}); e1.data = {10.0f, -2.0f};
    Tensor e2({1, 2}); e2.data = {-4.0f, 8.0f};
    const std::vector<Tensor> experts = {e0, e1, e2};

    const auto routes = MoERouter::topK({3.0f, 2.0f, 1.0f}, 2);
    const Tensor output = MoEInference::combineExpertOutputs(experts, routes);
    const float w0 = 1.0f / (1.0f + std::exp(-1.0f));
    const float w1 = 1.0f - w0;
    if (!closeEnough(output.data[0], w0 * 2.0f + w1 * 10.0f) ||
        !closeEnough(output.data[1], w0 * 4.0f + w1 * -2.0f)) {
        std::cerr << "FAIL: weighted expert combination is incorrect\n"; return 1;
    }

    std::vector<int> loads = {1, 0, 0};
    const auto fallbackRoutes = MoERouter::topKWithCapacity({3.0f, 2.0f, 1.0f}, 2, loads, 1);
    const Tensor fallback = MoEInference::combineExpertOutputs(experts, fallbackRoutes);
    const float fw1 = 1.0f / (1.0f + std::exp(-1.0f));
    const float fw2 = 1.0f - fw1;
    if (!closeEnough(fallback.data[0], fw1 * 10.0f + fw2 * -4.0f) ||
        !closeEnough(fallback.data[1], fw1 * -2.0f + fw2 * 8.0f)) {
        std::cerr << "FAIL: capacity fallback expert combination is incorrect\n"; return 1;
    }

    const auto routesAgain = MoERouter::topK({3.0f, 2.0f, 1.0f}, 2);
    const Tensor outputAgain = MoEInference::combineExpertOutputs(experts, routesAgain);
    if (outputAgain.data != output.data) {
        std::cerr << "FAIL: identical MoE inference input was not deterministic\n"; return 1;
    }

    bool rejected = false;
    try { (void)MoEInference::combineExpertOutputs(experts, {{0, 0.4f, false}, {1, 0.4f, false}}); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { std::cerr << "FAIL: non-normalized route weights accepted\n"; return 1; }

    Tensor wrong({1, 3}); wrong.data = {1.0f, 2.0f, 3.0f};
    rejected = false;
    try { (void)MoEInference::combineExpertOutputs({e0, wrong}, {{0, 0.5f, false}, {1, 0.5f, false}}); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { std::cerr << "FAIL: mismatched expert shapes accepted\n"; return 1; }

    std::cout << "MoE inference correctness validation passed.\n";
    return 0;
}
