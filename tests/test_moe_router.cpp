#include "NppAIEngine.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

static bool closeEnough(float a, float b) {
    return std::fabs(a - b) < 1e-6f;
}

int main() {
    const auto routes = MoERouter::topK({0.5f, 2.0f, 2.0f, -1.0f}, 2);
    if (routes.size() != 2 || routes[0].expertIndex != 1 || routes[1].expertIndex != 2) {
        std::cerr << "FAIL: deterministic tie-break did not prefer lower expert index\n";
        return 1;
    }
    if (!closeEnough(routes[0].weight, 0.5f) || !closeEnough(routes[1].weight, 0.5f)) {
        std::cerr << "FAIL: selected expert weights were not normalized\n";
        return 1;
    }

    const auto ranked = MoERouter::topK({1.0f, 3.0f, 2.0f}, 2);
    const float expected0 = 1.0f / (1.0f + std::exp(-1.0f));
    if (ranked[0].expertIndex != 1 || ranked[1].expertIndex != 2 ||
        !closeEnough(ranked[0].weight, expected0) ||
        !closeEnough(ranked[0].weight + ranked[1].weight, 1.0f)) {
        std::cerr << "FAIL: top-k ordering or softmax normalization is incorrect\n";
        return 1;
    }

    bool rejected = false;
    try { MoERouter::topK({1.0f, 2.0f}, 0); } catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { std::cerr << "FAIL: k=0 was accepted\n"; return 1; }

    rejected = false;
    try { MoERouter::topK({1.0f, std::numeric_limits<float>::quiet_NaN()}, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected) { std::cerr << "FAIL: non-finite router logits were accepted\n"; return 1; }

    std::cout << "MoE top-k routing validation passed.\n";
    return 0;
}
