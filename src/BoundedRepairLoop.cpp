#include "BoundedRepairLoop.h"

BoundedRepairLoop::BoundedRepairLoop(std::size_t maxAttempts)
    : maxAttempts_(maxAttempts) {}

RepairLoopResult BoundedRepairLoop::run(const AgentFeedbackLoop& validator,
                                        const AgentFeedbackLoop::Command& build,
                                        const AgentFeedbackLoop::Command& tests,
                                        const Repair& repair) const {
    RepairLoopResult result;
    if (maxAttempts_ == 0) {
        result.exhausted = true;
        return result;
    }

    for (std::size_t attempt = 1; attempt <= maxAttempts_; ++attempt) {
        result.attempts = attempt;
        result.validation = validator.validate(build, tests);
        if (result.validation.succeeded) {
            result.succeeded = true;
            return result;
        }

        result.history.push_back({attempt, result.validation.repairFeedback});
        if (attempt < maxAttempts_ && repair)
            repair(attempt, result.validation.repairFeedback);
    }

    result.exhausted = true;
    return result;
}
