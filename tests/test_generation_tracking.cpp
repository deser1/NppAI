#include "GenerationTracking.h"
#include <iostream>

static bool check(bool value, const char* message) {
    if (!value) std::cerr << "FAIL: " << message << "\n";
    return value;
}

int main() {
    bool ok = true;
    const auto normal = GenerationTracking::build(
        "make function", "int f() { return 1; }", 4, 7, "sample.cpp");
    ok &= check(normal.prompt == "make function", "prompt is preserved");
    ok &= check(normal.generated == "int f() { return 1; }", "generated code is preserved");
    ok &= check(normal.startLine == 4 && normal.endLine == 7, "editor line range is preserved");
    ok &= check(normal.filePath == "sample.cpp", "file path is preserved");

    const auto reversed = GenerationTracking::build("p", "g", 8, 3, "x.cpp");
    ok &= check(reversed.startLine == 8 && reversed.endLine == 8,
                "end line cannot precede generation start");

    const auto negative = GenerationTracking::build("p", "g", -2, -1, "x.cpp");
    ok &= check(negative.startLine == 0 && negative.endLine == 0,
                "invalid negative line range is normalized");
    return ok ? 0 : 1;
}
