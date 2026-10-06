#include "AgentFeedbackLoop.h"
#include <iostream>
#include <vector>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << "\n";
    return condition;
}
}

int main() {
    bool ok = true;
    std::vector<GenerationProgressEvent::Type> events;
    GenerationStreamRouter router(nullptr, nullptr, nullptr,
        [&](const GenerationProgressEvent& event) { events.push_back(event.type); });
    AgentFeedbackLoop loop(&router);

    int testsCalled = 0;
    auto buildFailure = loop.validate(
        [] { return AgentCommandResult{1, "compiler: missing symbol"}; },
        [&] { ++testsCalled; return AgentCommandResult{0, {}}; });
    ok &= check(!buildFailure.succeeded && testsCalled == 0,
                "tests do not run after a failed build");
    ok &= check(buildFailure.repairFeedback == "compiler: missing symbol",
                "compiler diagnostics are preserved for repair");
    ok &= check(events.size() == 2 &&
                    events[0] == GenerationProgressEvent::Type::BuildStarted &&
                    events[1] == GenerationProgressEvent::Type::BuildFailed,
                "failed build emits observable lifecycle events");

    events.clear();
    auto testFailure = loop.validate(
        [] { return AgentCommandResult{0, {}}; },
        [] { return AgentCommandResult{2, "test: expected 2 got 3"}; });
    ok &= check(!testFailure.succeeded &&
                    testFailure.repairFeedback == "test: expected 2 got 3",
                "test diagnostics are preserved for repair");
    ok &= check(events.size() == 4 &&
                    events.back() == GenerationProgressEvent::Type::TestFailed,
                "test failure emits build and test lifecycle events");

    events.clear();
    auto success = loop.validate(
        [] { return AgentCommandResult{0, {}}; },
        [] { return AgentCommandResult{0, {}}; });
    ok &= check(success.succeeded && success.repairFeedback.empty(),
                "successful build and tests complete validation");
    ok &= check(events.size() == 5 &&
                    events.back() == GenerationProgressEvent::Type::TaskCompleted,
                "successful validation completes observable task");

    return ok ? 0 : 1;
}
