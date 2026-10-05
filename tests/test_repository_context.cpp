#include "RepositoryContext.h"
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
    ok &= check(RepositoryContext::buildQuery("fix authentication", "") == "fix authentication",
                "query without current file remains unchanged");
    ok &= check(RepositoryContext::buildQuery("fix authentication", "C:/repo/src/auth.cpp") ==
                    "fix authentication\ncurrent file auth.cpp",
                "current filename enriches repository retrieval query");
    ok &= check(RepositoryContext::limit("abcdef", 4) == "abcd",
                "repository context respects byte budget");
    ok &= check(RepositoryContext::limit("short", 20) == "short",
                "short repository context is preserved");
    const std::string utf8 = std::string("abc") + "\xC5\xBC" + "def";
    ok &= check(RepositoryContext::limit(utf8, 4) == "abc",
                "context limit does not split UTF-8 characters");
    ok &= check(RepositoryContext::limit("abc", 0).empty(),
                "zero context budget returns empty context");
    if (!ok) return 1;
    std::cout << "Repository context tests passed.\n";
    return 0;
}
