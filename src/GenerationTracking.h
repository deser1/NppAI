#pragma once
#include <string>

struct GenerationTrackingResult {
    std::string prompt;
    std::string generated;
    int startLine;
    int endLine;
    std::string filePath;
};

namespace GenerationTracking {
GenerationTrackingResult build(const std::string& prompt,
                               const std::string& generated,
                               int startLine,
                               int endLine,
                               const std::string& filePath);
}
