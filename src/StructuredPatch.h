#pragma once
#include <cstddef>
#include <string>
#include <vector>

enum class PatchLineType { Context, Added, Removed };

struct PatchLine {
    PatchLineType type;
    std::size_t oldLine;
    std::size_t newLine;
    std::string text;
};

struct StructuredPatch {
    std::string path;
    std::string originalText;
    std::string proposedText;
    std::vector<PatchLine> lines;

    bool empty() const;
    std::string unifiedDiff() const;
};

class StructuredPatchBuilder {
public:
    static StructuredPatch build(const std::string& path,
                                 const std::string& originalText,
                                 const std::string& proposedText);
};
