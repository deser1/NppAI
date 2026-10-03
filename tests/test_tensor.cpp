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

    // Invalid shapes and backing storage must fail deterministically instead
    // of indexing outside tensor buffers.
    auto expectInvalidMatmul = [&](const Tensor& left, const Tensor& right,
                                   bool transpose, const char* message) {
        try {
            (void)Tensor::matmul(left, right, transpose);
            return check(false, message);
        } catch (const std::invalid_argument&) {
            return true;
        }
    };
    Tensor rankOne({3});
    ok &= expectInvalidMatmul(rankOne, b, false, "rank-1 left tensor rejected");
    Tensor incompatible({4, 2});
    ok &= expectInvalidMatmul(a, incompatible, false, "incompatible dimensions rejected");
    Tensor badStorage({3, 2});
    badStorage.data.pop_back();
    ok &= expectInvalidMatmul(a, badStorage, false, "short FP32 storage rejected");
    Tensor badQ8({2, 3});
    badQ8.data.clear();
    badQ8.data_q8.resize(5);
    badQ8.scale_q8 = 0.1f;
    ok &= expectInvalidMatmul(a, badQ8, true, "short INT8 storage rejected");

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

    // Quantization accuracy: the real read/quantize/dequantize path should
    // preserve values within one quantization step and remain numerically
    // close to FP32 when used by transpose matmul.
    const std::string accuracyPath = tempPath("nppai_test_quantization_accuracy.bin");
    const std::vector<float> accuracyValues = {
        -3.25f, -1.75f, -0.5f, -0.01f, 0.0f, 0.02f, 0.75f, 2.9f
    };
    {
        std::ofstream out(accuracyPath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(accuracyValues.data()),
                  static_cast<std::streamsize>(accuracyValues.size() * sizeof(float)));
    }
    Tensor accuracyQ8({2, 4});
    {
        std::ifstream in(accuracyPath, std::ios::binary);
        ok &= check(accuracyQ8.readFromFile(in, true), "accuracy quantized read");
    }
    ok &= check(accuracyQ8.scale_q8 > 0.0f, "quantization scale is positive");
    for (size_t i = 0; i < accuracyValues.size(); ++i) {
        const float error = std::fabs(accuracyQ8.get(static_cast<int>(i)) - accuracyValues[i]);
        ok &= check(error <= accuracyQ8.scale_q8 + 1e-6f,
                    "dequantization error bounded by one quantization step");
    }

    Tensor accuracyFp32({2, 4});
    accuracyFp32.data = accuracyValues;
    Tensor accuracyInput({1, 4});
    accuracyInput.data = {0.5f, -0.25f, 0.75f, -1.0f};

    Tensor::setSimdOverrideForTesting(0);
    Tensor fp32Output = Tensor::matmul(accuracyInput, accuracyFp32, true);
    Tensor int8Output = Tensor::matmul(accuracyInput, accuracyQ8, true);
    for (int i = 0; i < 2; ++i) {
        const float outputError = std::fabs(fp32Output.data[i] - int8Output.data[i]);
        const float inputL1 = 0.5f + 0.25f + 0.75f + 1.0f;
        ok &= check(outputError <= inputL1 * accuracyQ8.scale_q8 + 1e-5f,
                    "INT8 matmul error bounded by quantization step");
    }
    Tensor::setSimdOverrideForTesting(-1);
    std::remove(accuracyPath.c_str());

    const std::string zeroPath = tempPath("nppai_test_quantization_zero.bin");
    {
        std::ofstream out(zeroPath, std::ios::binary | std::ios::trunc);
        const float zeros[] = {0.0f, 0.0f, 0.0f, 0.0f};
        out.write(reinterpret_cast<const char*>(zeros), sizeof(zeros));
    }
    {
        Tensor zeroQ8({4});
        std::ifstream in(zeroPath, std::ios::binary);
        ok &= check(zeroQ8.readFromFile(in, true), "zero tensor quantized read");
        for (int i = 0; i < 4; ++i)
            ok &= check(nearlyEqual(zeroQ8.get(i), 0.0f), "zero tensor remains zero");
    }
    std::remove(zeroPath.c_str());

    // Quantization edge cases: asymmetric ranges, exact INT8 extrema and
    // non-multiple-of-8 widths exercise rounding plus the scalar SIMD tail.
    const std::string edgePath = tempPath("nppai_test_quantization_edges.bin");
    const std::vector<float> edgeValues = {
        -127.0f, -63.5f, -1.0f, 0.0f, 1.0f, 31.75f, 63.5f, 95.25f, 127.0f
    };
    {
        std::ofstream out(edgePath, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(edgeValues.data()),
                  static_cast<std::streamsize>(edgeValues.size() * sizeof(float)));
    }
    Tensor edgeQ8({1, 9});
    {
        std::ifstream in(edgePath, std::ios::binary);
        ok &= check(edgeQ8.readFromFile(in, true), "edge quantized read");
    }
    ok &= check(edgeQ8.data_q8.front() == -127, "negative INT8 extreme preserved");
    ok &= check(edgeQ8.data_q8.back() == 127, "positive INT8 extreme preserved");
    for (size_t i = 0; i < edgeValues.size(); ++i) {
        const float error = std::fabs(edgeQ8.get(static_cast<int>(i)) - edgeValues[i]);
        ok &= check(error <= edgeQ8.scale_q8 * 0.51f + 1e-6f,
                    "edge dequantization stays within half a quantization step");
    }

    Tensor edgeInput({1, 9});
    edgeInput.data = {1.0f, -0.5f, 0.25f, -0.125f, 0.0625f,
                      -0.03125f, 0.015625f, -0.0078125f, 0.00390625f};
    Tensor edgeFp32({1, 9});
    edgeFp32.data = edgeValues;
    Tensor::setSimdOverrideForTesting(0);
    const Tensor edgeFp32Out = Tensor::matmul(edgeInput, edgeFp32, true);
    const Tensor edgeInt8Out = Tensor::matmul(edgeInput, edgeQ8, true);
    float edgeInputL1 = 0.0f;
    for (float value : edgeInput.data)
        edgeInputL1 += std::fabs(value);
    ok &= check(std::fabs(edgeFp32Out.data[0] - edgeInt8Out.data[0]) <=
                    edgeInputL1 * edgeQ8.scale_q8 * 0.51f + 1e-5f,
                "odd-width INT8 matmul error respects rounding bound");
    Tensor::setSimdOverrideForTesting(-1);
    std::remove(edgePath.c_str());

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
