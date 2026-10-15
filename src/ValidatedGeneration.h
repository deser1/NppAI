#pragma once
#include "AgentTaskOutcome.h"
#include "PatchReview.h"
#include <functional>

struct ValidatedGenerationRequest {
    std::string task, context, path, original;
    std::size_t selectionStart = 0, selectionEnd = 0;
    bool allowOnlineEscalation = false;
};
struct GeneratedCandidate { std::string code; bool online = false; std::string reason; };
struct ValidatedGenerationResult {
    StructuredPatch patch;
    RepairLoopResult repair;
    AgentTaskOutcome outcome;
    bool blocked = false, cancelled = false;
    std::string error;
    std::vector<std::string> candidateBackends;
    bool ready() const { return repair.succeeded && !blocked && !cancelled && !patch.empty(); }
};
class ValidatedGeneration {
public:
    using Generate = std::function<GeneratedCandidate(const std::string&, const std::string&, bool)>;
    using Validate = std::function<AgentCommandResult(const std::string&, bool tests)>;
    using Cancelled = std::function<bool()>;
    static ValidatedGenerationResult run(const ValidatedGenerationRequest& request,
        const Generate& generate, const Validate& validate, const Cancelled& cancelled,
        GenerationStreamRouter* progress = nullptr);
};
