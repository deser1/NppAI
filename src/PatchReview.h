#pragma once
#include "StructuredPatch.h"
#include <filesystem>
#include <string>

enum class PatchReviewDecision { Pending, Accepted, Rejected };

struct PatchApplyResult {
    bool applied;
    std::string error;
};

class PatchReview {
public:
    PatchReview(std::filesystem::path workspaceRoot, StructuredPatch patch);

    const StructuredPatch& patch() const;
    PatchReviewDecision decision() const;
    void accept();
    void reject();
    PatchApplyResult apply() const;

private:
    std::filesystem::path workspaceRoot_;
    StructuredPatch patch_;
    PatchReviewDecision decision_ = PatchReviewDecision::Pending;
};
