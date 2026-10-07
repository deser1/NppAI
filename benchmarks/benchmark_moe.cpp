#include "NppAIEngine.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

static double median(std::vector<double> samples) {
    std::sort(samples.begin(), samples.end());
    const std::size_t middle = samples.size() / 2;
    if (samples.size() % 2 == 0)
        return (samples[middle - 1] + samples[middle]) * 0.5;
    return samples[middle];
}

static Tensor denseCombine(const std::vector<Tensor>& experts) {
    Tensor output(experts.front().shape);
    const float weight = 1.0f / static_cast<float>(experts.size());
    for (const Tensor& expert : experts) {
        for (std::size_t i = 0; i < output.data.size(); ++i)
            output.data[i] += expert.data[i] * weight;
    }
    return output;
}

static double benchmarkDense(const std::vector<Tensor>& experts, int iterations) {
    volatile float sink = 0.0f;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        const Tensor output = denseCombine(experts);
        sink += output.data[static_cast<std::size_t>(i) % output.data.size()];
    }
    const auto end = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

static double benchmarkMoE(const std::vector<Tensor>& experts,
                           const std::vector<float>& logits,
                           int topK,
                           int iterations) {
    volatile float sink = 0.0f;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        const auto routes = MoERouter::topK(logits, topK);
        const Tensor output = MoEInference::combineExpertOutputs(experts, routes);
        sink += output.data[static_cast<std::size_t>(i) % output.data.size()];
    }
    const auto end = std::chrono::steady_clock::now();
    (void)sink;
    return std::chrono::duration<double, std::micro>(end - start).count() / iterations;
}

int main() {
    constexpr int expertCount = 8;
    constexpr int topK = 2;
    constexpr int width = 4096;
    constexpr int warmupIterations = 50;
    constexpr int iterations = 500;
    constexpr int series = 7;

    std::vector<Tensor> experts;
    experts.reserve(expertCount);
    std::vector<float> logits;
    logits.reserve(expertCount);

    for (int expert = 0; expert < expertCount; ++expert) {
        Tensor output({1, width});
        for (int i = 0; i < width; ++i)
            output.data[i] = std::sin(static_cast<float>(expert * width + i) * 0.001f);
        experts.push_back(std::move(output));
        logits.push_back(std::cos(static_cast<float>(expert) * 0.37f));
    }

    benchmarkDense(experts, warmupIterations);
    benchmarkMoE(experts, logits, topK, warmupIterations);

    std::vector<double> denseSamples;
    std::vector<double> moeSamples;
    denseSamples.reserve(series);
    moeSamples.reserve(series);

    for (int i = 0; i < series; ++i) {
        if (i % 2 == 0) {
            denseSamples.push_back(benchmarkDense(experts, iterations));
            moeSamples.push_back(benchmarkMoE(experts, logits, topK, iterations));
        } else {
            moeSamples.push_back(benchmarkMoE(experts, logits, topK, iterations));
            denseSamples.push_back(benchmarkDense(experts, iterations));
        }
    }

    const double denseUs = median(denseSamples);
    const double moeUs = median(moeSamples);

    std::cout << "NppAI MoE vs dense expert-combination benchmark\n";
    std::cout << "EXPERTS: " << expertCount << "\n";
    std::cout << "TOP_K: " << topK << "\n";
    std::cout << "WIDTH: " << width << "\n";
    std::cout << "DENSE: " << denseUs << " us/op\n";
    std::cout << "MOE: " << moeUs << " us/op\n";
    if (moeUs > 0.0)
        std::cout << "DENSE/MOE ratio: " << (denseUs / moeUs) << "x\n";
    return 0;
}
