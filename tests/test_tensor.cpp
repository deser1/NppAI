#include "NppAIEngine.h"
#include <cmath>
#include <iostream>

namespace {
bool nearlyEqual(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) <= eps;
}

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}
}

int main() {
    bool ok = true;

    Tensor a({2, 3});
    a.data = {1, 2, 3, 4, 5, 6};

    Tensor b({3, 2});
    b.data = {7, 8, 9, 10, 11, 12};

    Tensor c = Tensor::matmul(a, b);
    ok &= check(c.shape == std::vector<int>({2, 2}), "matmul shape");
    ok &= check(nearlyEqual(c.at(0, 0), 58.0f), "c[0,0]");
    ok &= check(nearlyEqual(c.at(0, 1), 64.0f), "c[0,1]");
    ok &= check(nearlyEqual(c.at(1, 0), 139.0f), "c[1,0]");
    ok &= check(nearlyEqual(c.at(1, 1), 154.0f), "c[1,1]");

    Tensor bt = Tensor::matmul(a, a, true);
    ok &= check(bt.shape == std::vector<int>({2, 2}), "transpose matmul shape");
    ok &= check(nearlyEqual(bt.at(0, 0), 14.0f), "transpose c[0,0]");
    ok &= check(nearlyEqual(bt.at(0, 1), 32.0f), "transpose c[0,1]");
    ok &= check(nearlyEqual(bt.at(1, 1), 77.0f), "transpose c[1,1]");

    Tensor activation({1, 2});
    activation.data = {0.0f, 1.0f};
    activation.applySiLU();
    ok &= check(nearlyEqual(activation.data[0], 0.0f), "SiLU(0)");
    ok &= check(nearlyEqual(activation.data[1], 0.7310586f, 1e-4f), "SiLU(1)");

    if (!ok)
        return 1;

    std::cout << "Tensor tests passed.\n";
    return 0;
}
