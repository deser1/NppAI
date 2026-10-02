#include "NppAIEngine.h"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
std::string tempPath(const char* name) {
#ifdef _WIN32
    return std::string(name);
#else
    return std::string("/tmp/") + name;
#endif
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

    Tensor norm({1, 2});
    norm.data = {1.0f, 2.0f};
    Tensor weights({2});
    weights.data = {1.0f, 1.0f};
    norm.applyRMSNorm(weights);
    ok &= check(nearlyEqual(norm.data[0], 0.632453f, 1e-4f), "RMSNorm[0]");
    ok &= check(nearlyEqual(norm.data[1], 1.264906f, 1e-4f), "RMSNorm[1]");

    const std::string qPath = tempPath("nppai_test_tensor.bin");
    {
        std::ofstream out(qPath, std::ios::binary | std::ios::trunc);
        float values[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        out.write(reinterpret_cast<const char*>(values), sizeof(values));
    }
    {
        Tensor quantized({5});
        std::ifstream in(qPath, std::ios::binary);
        ok &= check(quantized.readFromFile(in, true), "quantized read");
        ok &= check(quantized.data.empty(), "quantized frees fp32 data");
        ok &= check(quantized.data_q8.size() == 5, "quantized size");
        ok &= check(nearlyEqual(quantized.get(0), -2.0f, 0.02f), "quantized -2");
        ok &= check(nearlyEqual(quantized.get(4), 2.0f, 0.02f), "quantized +2");
    }
    std::remove(qPath.c_str());

    const std::string truncatedPath = tempPath("nppai_test_tensor_truncated.bin");
    {
        std::ofstream out(truncatedPath, std::ios::binary | std::ios::trunc);
        float value = 1.0f;
        out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    }
    {
        Tensor truncated({2});
        std::ifstream in(truncatedPath, std::ios::binary);
        ok &= check(!truncated.readFromFile(in, false), "truncated tensor rejected");
    }
    std::remove(truncatedPath.c_str());

    if (!ok)
        return 1;

    std::cout << "Tensor tests passed.\n";
    return 0;
}
