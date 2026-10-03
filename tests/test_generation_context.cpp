#include "GenerationContext.h"
#include <iostream>
#include <string>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}
}

int main() {
    bool ok = true;
    ok &= check(GenerationContext::select("vector hit", "legacy") == "vector hit",
                "retrieved RAG context has priority");
    ok &= check(GenerationContext::select("", "legacy memory") == "legacy memory",
                "legacy memory is used when retrieval is empty");
    ok &= check(GenerationContext::select("", "") == "",
                "empty sources produce empty context");

    const std::string legacy = "0123456789";
    ok &= check(GenerationContext::select("", legacy, 4) == "6789",
                "legacy fallback keeps the newest tail within the limit");
    ok &= check(GenerationContext::select("hit", legacy, 0) == "hit",
                "retrieved context is not truncated by the legacy limit");

    if (!ok)
        return 1;
    std::cout << "Generation context integration tests passed.\n";
    return 0;
}
