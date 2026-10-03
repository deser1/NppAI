#include "GenerationTracking.h"
#include <algorithm>

namespace GenerationTracking {
GenerationTrackingResult build(const std::string& prompt,
                               const std::string& generated,
                               int startLine,
                               int endLine,
                               const std::string& filePath) {
    GenerationTrackingResult result{prompt, generated, startLine, endLine, filePath};
    if (result.startLine < 0)
        result.startLine = 0;
    if (result.endLine < result.startLine)
        result.endLine = result.startLine;
    return result;
}
}
