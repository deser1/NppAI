#include "RAGManager.h"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <filesystem>
#include <numeric>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    const bool realRepository = argc > 1 && std::string(argv[1]) == "--repo";
    const int documents = (argc > 1 && !realRepository) ? std::stoi(argv[1]) : 1000;
    if (realRepository && (argc != 3 || !std::filesystem::is_directory(argv[2]))) return 2;
    if (!realRepository && (documents <= 0 || documents > 10000)) return 2;
    auto& rag = RAGManager::getInstance();
    size_t indexedFiles = 0;
    if (realRepository) {
        indexedFiles = rag.indexRepository(argv[2]);
        if (indexedFiles == 0) return 1;
    } else for (int i = 0; i < documents; ++i) {
        const std::string source = "src/module_" + std::to_string(i) + ".cpp";
        const std::string text = "class Module" + std::to_string(i) +
            " { public: int calculateChecksum(int input) { return input + " +
            std::to_string(i) + "; } };\n";
        rag.addDocument(text, source, "cpp");
    }

    constexpr int warmup = 3;
    constexpr int iterations = 11;
    std::vector<double> samples;
    samples.reserve(iterations);
    size_t observedBytes = 0;
    for (int i = 0; i < warmup + iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        const std::string result = rag.retrieveContextRanked(
            realRepository ? "RAGManager indexRepository retrieveContext" : "calculateChecksum module input", 3, "", realRepository ? "" : "cpp");
        const auto end = std::chrono::steady_clock::now();
        observedBytes += result.size();
        if (i >= warmup)
            samples.push_back(std::chrono::duration<double, std::milli>(end - start).count());
    }
    if (observedBytes == 0) {
        std::cerr << "No retrieval result\n";
        return 1;
    }
    std::sort(samples.begin(), samples.end());
    const double median = samples[samples.size() / 2];
    const double p95 = samples[static_cast<size_t>((samples.size() - 1) * 0.95)];
    std::cout << "{\"mode\":\"" << (realRepository ? "real_repository" : "synthetic")
              << "\",\"documents\":" << (realRepository ? indexedFiles : static_cast<size_t>(documents))
              << ",\"retrieval_median_ms\":" << median
              << ",\"retrieval_p95_ms\":" << p95
              << ",\"warmup\":" << warmup
              << ",\"samples\":" << iterations
              << ",\"observed_result_bytes\":" << observedBytes
              << "}" << std::endl;
    return 0;
}
