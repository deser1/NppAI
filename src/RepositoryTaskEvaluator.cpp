#include "RepositoryTaskEvaluator.h"

RepositoryEvaluationSummary RepositoryTaskEvaluator::summarize(
    const std::vector<RepositoryTaskEvaluation>& tasks) {
    RepositoryEvaluationSummary summary;
    summary.taskCount = tasks.size();
    if (tasks.empty())
        return summary;

    std::size_t patchMatches = 0;
    std::size_t buildsPassed = 0;
    std::size_t testsPassed = 0;
    std::size_t safeWorkspaces = 0;
    std::size_t repairAttempts = 0;

    for (const auto& task : tasks) {
        patchMatches += task.patchMatchesExpected ? 1 : 0;
        buildsPassed += task.buildPassed ? 1 : 0;
        testsPassed += task.testsPassed ? 1 : 0;
        safeWorkspaces += task.workspacePreserved ? 1 : 0;
        repairAttempts += task.repairAttempts;
        if (task.patchMatchesExpected && task.buildPassed && task.testsPassed &&
            task.workspacePreserved)
            ++summary.successfulTasks;
    }

    const double count = static_cast<double>(summary.taskCount);
    summary.taskSuccessRate = summary.successfulTasks / count;
    summary.patchAccuracy = patchMatches / count;
    summary.buildPassRate = buildsPassed / count;
    summary.testPassRate = testsPassed / count;
    summary.workspaceSafetyRate = safeWorkspaces / count;
    summary.averageRepairAttempts = static_cast<double>(repairAttempts) / count;
    return summary;
}
