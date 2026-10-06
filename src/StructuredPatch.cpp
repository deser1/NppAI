#include "StructuredPatch.h"
#include <sstream>

namespace {
std::vector<std::string> splitLines(const std::string& text) {
    std::vector<std::string> lines;
    std::istringstream input(text);
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        lines.push_back(line);
    }
    if (!text.empty() && text.back() == '\n' && (lines.empty() || !lines.back().empty()))
        lines.emplace_back();
    return lines;
}
}

bool StructuredPatch::empty() const {
    for (const auto& line : lines)
        if (line.type != PatchLineType::Context) return false;
    return true;
}

std::string StructuredPatch::unifiedDiff() const {
    if (empty()) return {};
    std::ostringstream out;
    out << "--- a/" << path << "\n+++ b/" << path << "\n";
    out << "@@ -1," << splitLines(originalText).size()
        << " +1," << splitLines(proposedText).size() << " @@\n";
    for (const auto& line : lines) {
        const char prefix = line.type == PatchLineType::Added ? '+' :
                            line.type == PatchLineType::Removed ? '-' : ' ';
        out << prefix << line.text << "\n";
    }
    return out.str();
}

StructuredPatch StructuredPatchBuilder::build(const std::string& path,
                                               const std::string& originalText,
                                               const std::string& proposedText) {
    StructuredPatch patch{path, originalText, proposedText, {}};
    const auto oldLines = splitLines(originalText);
    const auto newLines = splitLines(proposedText);
    const std::size_t n = oldLines.size(), m = newLines.size();

    std::vector<std::vector<std::size_t>> lcs(n + 1, std::vector<std::size_t>(m + 1));
    for (std::size_t i = n; i-- > 0;)
        for (std::size_t j = m; j-- > 0;)
            lcs[i][j] = oldLines[i] == newLines[j] ? 1 + lcs[i + 1][j + 1]
                                                   : std::max(lcs[i + 1][j], lcs[i][j + 1]);

    std::size_t i = 0, j = 0, oldNo = 1, newNo = 1;
    while (i < n || j < m) {
        if (i < n && j < m && oldLines[i] == newLines[j]) {
            patch.lines.push_back({PatchLineType::Context, oldNo++, newNo++, oldLines[i++]});
            ++j;
        } else if (j < m && (i == n || lcs[i][j + 1] >= lcs[i + 1][j])) {
            patch.lines.push_back({PatchLineType::Added, 0, newNo++, newLines[j++]});
        } else {
            patch.lines.push_back({PatchLineType::Removed, oldNo++, 0, oldLines[i++]});
        }
    }
    return patch;
}
