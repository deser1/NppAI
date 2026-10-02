#include "NppAIEngine.h"
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

static double benchmark(const Tensor& a, const Tensor& b, int iterations) {
    volatile float sink = 0.0f;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        Tensor result = Tensor::matmul(a, b, true);
        sink += result.get(0, 0);
    }
    const auto end = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::milli>(end - start).count() /
           iterations;
}

int main() {
    constexpr int size = 256;
    constexpr int iterations = 20;

    Tensor a({1, size});
    Tensor fp32({size, size});
    Tensor int8({size, size});

    for (int i = 0; i < size; ++i)
        a.data[i] = std::sin(static_cast<float>(i) * 0.01f);

    int8.scale_q8 = 1.0f / 127.0f;
    int8.data_q8.resize(size * size);
    int8.data.clear();

    for (int r = 0; r < size; ++r) {
        for (int c = 0; c < size; ++c) {
            const float value = std::sin(static_cast<float>(r + c) * 0.01f);
            fp32.data[r * size + c] = value;
            int8.data_q8[r * size + c] =
                static_cast<int8_t>(std::round(value / int8.scale_q8));
        }
    }

    // Warm up caches and any runtime dispatch before measuring.
    Tensor::matmul(a, fp32, true);
    Tensor::matmul(a, int8, true);

    const double fp32Ms = benchmark(a, fp32, iterations);
    const double int8Ms = benchmark(a, int8, iterations);

    std::cout << "NppAI Tensor matmul benchmark (" << size << "x" << size
              << ", " << iterations << " iterations)\n";
    std::cout << "FP32: " << fp32Ms << " ms/op\n";
    std::cout << "INT8: " << int8Ms << " ms/op\n";
    if (int8Ms > 0.0)
        std::cout << "FP32/INT8 ratio: " << (fp32Ms / int8Ms) << "x\n";

    return 0;
}
