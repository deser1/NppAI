#include "RAGManager.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

static size_t peakRssBytes() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(),
            reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)))
        return static_cast<size_t>(pmc.PeakWorkingSetSize);
    return 0;
#else
    struct rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
#ifdef __APPLE__
    return static_cast<size_t>(usage.ru_maxrss);
#else
    return static_cast<size_t>(usage.ru_maxrss) * 1024;
#endif
#endif
}

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const int files = argc > 1 ? std::stoi(argv[1]) : 1000;
    if (files <= 0 || files > 10000) return 2;
    const fs::path root = fs::temp_directory_path() / "nppai_rag_index_scale_fixture";
    fs::remove_all(root);
    fs::create_directories(root);
    struct Cleanup { fs::path p; ~Cleanup() { std::error_code ec; fs::remove_all(p, ec); } } cleanup{root};

    for (int i = 0; i < files; ++i) {
        std::ofstream out(root / ("source_" + std::to_string(i) + ".cpp"), std::ios::binary);
        if (!out) return 3;
        out << "// deterministic RAG indexing fixture " << i << "\n";
        for (int j = 0; j < 24; ++j)
            out << "int function_" << i << "_" << j
                << "(int value) { return value + " << (i + j) << "; }\n";
    }

    auto& rag = RAGManager::getInstance();
    const size_t before = peakRssBytes();
    const auto start = std::chrono::steady_clock::now();
    const size_t indexed = rag.indexRepository(root.string());
    const auto end = std::chrono::steady_clock::now();
    const size_t after = peakRssBytes();
    if (indexed != static_cast<size_t>(files)) {
        std::cerr << "Indexed " << indexed << " of " << files << " files\n";
        return 1;
    }
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "{\"files\":" << files
              << ",\"index_ms\":" << milliseconds
              << ",\"ms_per_file\":" << milliseconds / files
              << ",\"peak_rss_bytes\":" << after
              << ",\"peak_rss_growth_bytes\":" << (after >= before ? after - before : 0)
              << ",\"memory_metric\":\"process_peak_rss\"}" << std::endl;
    return 0;
}
