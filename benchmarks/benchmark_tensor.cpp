#include "NppAIEngine.h"
#include <algorithm>
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

static double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    if (samples.size() % 2 == 0)
        return (samples[middle - 1] + samples[middle]) * 0.5;
    return samples[middle];
}

static void runBenchmark(int size, int warmupIterations, int iterations, int series) {
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

    benchmark(a, fp32, warmupIterations);
    benchmark(a, int8, warmupIterations);

    std::vector<double> fp32Samples;
    std::vector<double> int8Samples;
    fp32Samples.reserve(series);
    int8Samples.reserve(series);

    for (int i = 0; i < series; ++i) {
        if (i % 2 == 0) {
            fp32Samples.push_back(benchmark(a, fp32, iterations));
            int8Samples.push_back(benchmark(a, int8, iterations));
        } else {
            int8Samples.push_back(benchmark(a, int8, iterations));
            fp32Samples.push_back(benchmark(a, fp32, iterations));
        }
    }

    const double fp32Ms = median(fp32Samples);
    const double int8Ms = median(int8Samples);

    std::cout << "SIZE " << size << "x" << size
              << " (median of " << series << " series, " << iterations
              << " iterations/series)\n";
    std::cout << "FP32: " << fp32Ms << " ms/op\n";
    std::cout << "INT8: " << int8Ms << " ms/op\n";
    if (int8Ms > 0.0)
        std::cout << "FP32/INT8 ratio: " << (fp32Ms / int8Ms) << "x\n";
}

int main() {
    constexpr int series = 7;

    std::cout << "NppAI Tensor matmul multi-size benchmark\n";
    runBenchmark(128, 50, 400, series);
    runBenchmark(256, 50, 200, series);
    runBenchmark(512, 25, 100, series);
    runBenchmark(1024, 10, 50, series);

    return 0;
}
