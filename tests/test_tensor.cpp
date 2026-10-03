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

    // Runtime dispatch parity: scalar is always exercised; SIMD is compared
    // when the current runner supports AVX2/FMA and OS YMM state.
    Tensor parityA({1, 16});
    Tensor parityB({2, 16});
    for (int i = 0; i < 16; ++i) {
        parityA.data[i] = std::sin(static_cast<float>(i) * 0.17f);
        parityB.data[i] = std::cos(static_cast<float>(i) * 0.11f);
        parityB.data[16 + i] = std::sin(static_cast<float>(i) * 0.07f);
    }

    Tensor::setSimdOverrideForTesting(0);
    Tensor scalarFp32 = Tensor::matmul(parityA, parityB, true);

    Tensor parityQ8({2, 16});
    parityQ8.scale_q8 = 0.01f;
    parityQ8.data.clear();
    parityQ8.data_q8.resize(32);
    for (int i = 0; i < 32; ++i)
        parityQ8.data_q8[i] = static_cast<int8_t>(std::round(parityB.data[i] / parityQ8.scale_q8));
    Tensor scalarInt8 = Tensor::matmul(parityA, parityQ8, true);

    Tensor scalarNorm({1, 16});
    Tensor parityWeight({16});
    for (int i = 0; i < 16; ++i) {
        scalarNorm.data[i] = parityA.data[i] + 0.25f;
        parityWeight.data[i] = 0.5f + static_cast<float>(i) * 0.03f;
    }
    scalarNorm.applyRMSNorm(parityWeight);

    if (Tensor::simdAvailableForTesting()) {
        Tensor::setSimdOverrideForTesting(1);
        Tensor simdFp32 = Tensor::matmul(parityA, parityB, true);
        Tensor simdInt8 = Tensor::matmul(parityA, parityQ8, true);
        Tensor simdNorm({1, 16});
        for (int i = 0; i < 16; ++i)
            simdNorm.data[i] = parityA.data[i] + 0.25f;
        simdNorm.applyRMSNorm(parityWeight);

        for (int i = 0; i < 2; ++i) {
            ok &= check(nearlyEqual(scalarFp32.data[i], simdFp32.data[i], 1e-4f),
                        "scalar/SIMD FP32 parity");
            ok &= check(nearlyEqual(scalarInt8.data[i], simdInt8.data[i], 1e-4f),
                        "scalar/SIMD INT8 parity");
        }
        for (int i = 0; i < 16; ++i)
            ok &= check(nearlyEqual(scalarNorm.data[i], simdNorm.data[i], 1e-4f),
                        "scalar/SIMD RMSNorm parity");
    }
    Tensor::setSimdOverrideForTesting(-1);

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
