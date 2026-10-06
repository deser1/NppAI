#pragma once
#include "BoundedRepairLoop.h"
#include "StructuredPatch.h"
#include <cstddef>
#include <string>
#include <vector>

enum class AgentTaskStatus { Succeeded, Failed, RolledBack };

struct AgentTaskOutcome {
    std::string task;
    AgentTaskStatus status = AgentTaskStatus::Failed;
    std::size_t attempts = 0;
    std::string patch;
    std::vector<std::string> diagnostics;

    std::string toJsonLine() const;
};

class AgentTaskOutcomeBuilder {
public:
    static AgentTaskOutcome fromRepairResult(const std::string& task,
                                             const StructuredPatch& patch,
                                             const RepairLoopResult& result,
                                             bool rolledBack);
};
