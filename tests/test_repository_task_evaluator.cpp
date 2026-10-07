#include "RepositoryTaskEvaluator.h"
#include <cmath>
#include <iostream>
#include <vector>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << "\n";
    return condition;
}
bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }
}

int main() {
    bool ok = true;

    const auto empty = RepositoryTaskEvaluator::summarize({});
    ok &= check(empty.taskCount == 0 && near(empty.taskSuccessRate, 0.0),
                "empty evaluation is deterministic");

    const std::vector<RepositoryTaskEvaluation> tasks = {
        {"cross-file API change", true, true, true, true, 0},
        {"compile-error repair", true, true, true, true, 1},
        {"unsafe stale patch", false, false, false, true, 2},
        {"test regression", true, true, false, true, 1},
    };

    const auto summary = RepositoryTaskEvaluator::summarize(tasks);
    ok &= check(summary.taskCount == 4, "counts repository tasks");
    ok &= check(summary.successfulTasks == 2, "counts end-to-end successes");
    ok &= check(near(summary.taskSuccessRate, 0.5), "computes task success rate");
    ok &= check(near(summary.patchAccuracy, 0.75), "computes patch accuracy");
    ok &= check(near(summary.buildPassRate, 0.75), "computes build pass rate");
    ok &= check(near(summary.testPassRate, 0.5), "computes test pass rate");
    ok &= check(near(summary.workspaceSafetyRate, 1.0), "computes workspace safety rate");
    ok &= check(near(summary.averageRepairAttempts, 1.0), "computes repair effort");

    return ok ? 0 : 1;
}
