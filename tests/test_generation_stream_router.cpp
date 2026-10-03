#include "GenerationStreamRouter.h"
#include <iostream>
#include <string>

static bool check(bool value, const char* message) {
    if (!value) std::cerr << "FAIL: " << message << "\n";
    return value;
}

int main() {
    bool ok = true;
    std::string code;
    std::string visibleThought;
    int removed = 0;

    GenerationStreamRouter router(
        [&](const std::string& thought) { visibleThought = thought; },
        [&](char c) { code += c; },
        [&](int count) {
            removed += count;
            while (count-- > 0 && !code.empty()) code.pop_back();
        });

    router.onToken('a', false);
    router.onToken('b', false);
    router.onToken('x', true);
    router.onToken('\n', true);
    router.onToken('c', false);
    router.onRemove(1);

    ok &= check(code == "ab", "code tokens and backtracking are routed to editor callbacks");
    ok &= check(router.thoughtBuffer() == "x\n", "thought tokens stay out of generated code");
    ok &= check(visibleThought == "x\n", "thought callback flushes on newline");
    ok &= check(removed == 1, "remove callback receives requested count");

    std::string twenty;
    GenerationStreamRouter cadence(
        [&](const std::string& thought) { twenty = thought; },
        [](char) {}, [](int) {});
    for (int i = 0; i < 20; ++i) cadence.onToken('t', true);
    ok &= check(twenty.size() == 20, "thought callback flushes every 20 characters");

    return ok ? 0 : 1;
}
