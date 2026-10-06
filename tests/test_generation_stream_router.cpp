#include "GenerationStreamRouter.h"
#include <iostream>
#include <string>
#include <vector>

static bool check(bool value, const char* message) {
    if (!value) std::cerr << "FAIL: " << message << "\n";
    return value;
}

int main() {
    bool ok = true;
    std::string code;
    std::string visibleThought;
    int removed = 0;
    std::vector<GenerationProgressEvent> progress;

    GenerationStreamRouter router(
        [&](const std::string& thought) { visibleThought = thought; },
        [&](char c) { code += c; },
        [&](int count) {
            removed += count;
            while (count-- > 0 && !code.empty()) code.pop_back();
        },
        [&](const GenerationProgressEvent& event) { progress.push_back(event); });

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

    router.onProgress(GenerationProgressEvent::Type::TaskStarted, "Preparing coding task");
    router.onProgress(GenerationProgressEvent::Type::ContextReady, "Repository context ready");
    router.onProgress(GenerationProgressEvent::Type::GenerationStarted);
    router.onProgress(GenerationProgressEvent::Type::GenerationCompleted, "Response ready");
    router.onProgress(GenerationProgressEvent::Type::FileRead, "src/main.cpp");
    router.onProgress(GenerationProgressEvent::Type::FileChanged, "src/main.cpp");
    router.onProgress(GenerationProgressEvent::Type::BuildStarted, "Building workspace");
    router.onProgress(GenerationProgressEvent::Type::BuildPassed, "Build passed");
    router.onProgress(GenerationProgressEvent::Type::TestStarted, "Running tests");
    router.onProgress(GenerationProgressEvent::Type::TestPassed, "Tests passed");
    router.onProgress(GenerationProgressEvent::Type::TaskCompleted, "Task completed");
    ok &= check(progress.size() == 11, "observable progress events reach the UI-facing callback");
    ok &= check(progress[0].type == GenerationProgressEvent::Type::TaskStarted &&
                    progress[1].type == GenerationProgressEvent::Type::ContextReady &&
                    progress[2].type == GenerationProgressEvent::Type::GenerationStarted &&
                    progress[3].type == GenerationProgressEvent::Type::GenerationCompleted &&
                    progress[4].type == GenerationProgressEvent::Type::FileRead &&
                    progress[5].type == GenerationProgressEvent::Type::FileChanged &&
                    progress[6].type == GenerationProgressEvent::Type::BuildStarted &&
                    progress[7].type == GenerationProgressEvent::Type::BuildPassed &&
                    progress[8].type == GenerationProgressEvent::Type::TestStarted &&
                    progress[9].type == GenerationProgressEvent::Type::TestPassed &&
                    progress[10].type == GenerationProgressEvent::Type::TaskCompleted,
                "progress event ordering is preserved");
    ok &= check(progress[1].message == "Repository context ready",
                "progress events carry concise observable status messages");

    std::string twenty;
    GenerationStreamRouter cadence(
        [&](const std::string& thought) { twenty = thought; },
        [](char) {}, [](int) {});
    for (int i = 0; i < 20; ++i) cadence.onToken('t', true);
    ok &= check(twenty.size() == 20, "thought callback flushes every 20 characters");

    return ok ? 0 : 1;
}
