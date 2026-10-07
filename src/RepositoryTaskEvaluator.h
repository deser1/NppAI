#pragma once
#include <cstddef>
#include <string>
#include <vector>

struct RepositoryTaskEvaluation {
    std::string name;
    bool patchMatchesExpected = false;
    bool buildPassed = false;
    bool testsPassed = false;
    bool workspacePreserved = false;
    std::size_t repairAttempts = 0;
};

struct RepositoryEvaluationSummary {
    std::size_t taskCount = 0;
    std::size_t successfulTasks = 0;
    double taskSuccessRate = 0.0;
    double patchAccuracy = 0.0;
    double buildPassRate = 0.0;
    double testPassRate = 0.0;
    double workspaceSafetyRate = 0.0;
    double averageRepairAttempts = 0.0;
};

class RepositoryTaskEvaluator {
public:
    static RepositoryEvaluationSummary summarize(
        const std::vector<RepositoryTaskEvaluation>& tasks);
};
