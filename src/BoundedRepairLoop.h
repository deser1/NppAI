#pragma once
#include "AgentFeedbackLoop.h"
#include <cstddef>
#include <functional>
#include <string>
#include <vector>

struct RepairAttempt {
    std::size_t attempt = 0;
    std::string diagnostics;
};

struct RepairLoopResult {
    bool succeeded = false;
    bool exhausted = false;
    std::size_t attempts = 0;
    std::vector<RepairAttempt> history;
    AgentValidationResult validation;
};

class BoundedRepairLoop {
public:
    using Repair = std::function<void(std::size_t attempt, const std::string& diagnostics)>;

    explicit BoundedRepairLoop(std::size_t maxAttempts);
    RepairLoopResult run(const AgentFeedbackLoop& validator,
                         const AgentFeedbackLoop::Command& build,
                         const AgentFeedbackLoop::Command& tests,
                         const Repair& repair) const;

private:
    std::size_t maxAttempts_;
};
