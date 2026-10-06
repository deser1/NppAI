#include "BoundedRepairLoop.h"
#include <iostream>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << "\n";
    return condition;
}
}

int main() {
    bool ok = true;
    AgentFeedbackLoop validator;

    int buildCalls = 0, repairCalls = 0;
    BoundedRepairLoop succeedsAfterRepair(3);
    const auto recovered = succeedsAfterRepair.run(
        validator,
        [&] {
            ++buildCalls;
            return buildCalls < 2 ? AgentCommandResult{1, "compile error A"}
                                  : AgentCommandResult{0, {}};
        },
        [] { return AgentCommandResult{0, {}}; },
        [&](std::size_t attempt, const std::string& diagnostics) {
            ++repairCalls;
            ok &= check(attempt == 1 && diagnostics == "compile error A",
                        "repair receives failed-attempt diagnostics");
        });
    ok &= check(recovered.succeeded && !recovered.exhausted &&
                    recovered.attempts == 2 && repairCalls == 1,
                "loop stops immediately after repaired validation succeeds");
    ok &= check(recovered.history.size() == 1 &&
                    recovered.history[0].diagnostics == "compile error A",
                "failed diagnostics remain in history after recovery");

    int failedBuilds = 0, failedRepairs = 0;
    BoundedRepairLoop bounded(2);
    const auto exhausted = bounded.run(
        validator,
        [&] {
            ++failedBuilds;
            return AgentCommandResult{1, failedBuilds == 1 ? "error one" : "error two"};
        },
        [] { return AgentCommandResult{0, {}}; },
        [&](std::size_t, const std::string&) { ++failedRepairs; });
    ok &= check(!exhausted.succeeded && exhausted.exhausted &&
                    exhausted.attempts == 2 && failedBuilds == 2,
                "validation cannot exceed configured attempt limit");
    ok &= check(failedRepairs == 1,
                "repair is not invoked after the final exhausted attempt");
    ok &= check(exhausted.history.size() == 2 &&
                    exhausted.history[0].diagnostics == "error one" &&
                    exhausted.history[1].diagnostics == "error two",
                "all failed diagnostics are preserved in order");

    int zeroBuilds = 0;
    BoundedRepairLoop disabled(0);
    const auto zero = disabled.run(
        validator,
        [&] { ++zeroBuilds; return AgentCommandResult{0, {}}; },
        [] { return AgentCommandResult{0, {}}; },
        nullptr);
    ok &= check(zero.exhausted && zero.attempts == 0 && zeroBuilds == 0,
                "zero-attempt policy executes no commands");

    return ok ? 0 : 1;
}
