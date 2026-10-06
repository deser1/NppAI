#include "StructuredPatch.h"
#include <iostream>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << "\n";
    return condition;
}
}

int main() {
    bool ok = true;
    const auto patch = StructuredPatchBuilder::build(
        "src/example.cpp",
        "int value = 1;\nreturn value;\n",
        "int value = 2;\nreturn value;\n");

    ok &= check(!patch.empty(), "changed text produces a non-empty patch");
    ok &= check(patch.path == "src/example.cpp", "patch retains repository-relative path");
    ok &= check(patch.originalText.find("value = 1") != std::string::npos &&
                    patch.proposedText.find("value = 2") != std::string::npos,
                "patch retains before and after text for review");

    bool removed = false, added = false;
    for (const auto& line : patch.lines) {
        removed |= line.type == PatchLineType::Removed && line.text == "int value = 1;";
        added |= line.type == PatchLineType::Added && line.text == "int value = 2;";
    }
    ok &= check(removed && added, "structured patch identifies removed and added lines");

    const auto diff = patch.unifiedDiff();
    ok &= check(diff.find("--- a/src/example.cpp") != std::string::npos &&
                    diff.find("+++ b/src/example.cpp") != std::string::npos &&
                    diff.find("-int value = 1;") != std::string::npos &&
                    diff.find("+int value = 2;") != std::string::npos,
                "patch renders a reviewable unified diff");

    const auto unchanged = StructuredPatchBuilder::build("same.txt", "same\n", "same\n");
    ok &= check(unchanged.empty() && unchanged.unifiedDiff().empty(),
                "unchanged text produces no review diff");

    return ok ? 0 : 1;
}
