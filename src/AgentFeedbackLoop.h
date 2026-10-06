#pragma once
#include "GenerationStreamRouter.h"
#include <functional>
#include <string>

struct AgentCommandResult {
    int exitCode = -1;
    std::string diagnostics;
    bool succeeded() const { return exitCode == 0; }
};

struct AgentValidationResult {
    bool succeeded = false;
    AgentCommandResult build;
    AgentCommandResult tests;
    std::string repairFeedback;
};

class AgentFeedbackLoop {
public:
    using Command = std::function<AgentCommandResult()>;

    explicit AgentFeedbackLoop(GenerationStreamRouter* progress = nullptr);
    AgentValidationResult validate(const Command& build, const Command& tests) const;

private:
    GenerationStreamRouter* progress_;
};
