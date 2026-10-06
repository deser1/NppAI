#include "PatchReview.h"
#include "WorkspaceFileReader.h"
#include <fstream>

namespace {
bool isWithin(const std::filesystem::path& root, const std::filesystem::path& candidate) {
    auto rootIt = root.begin();
    auto candidateIt = candidate.begin();
    for (; rootIt != root.end(); ++rootIt, ++candidateIt) {
        if (candidateIt == candidate.end() || *rootIt != *candidateIt) return false;
    }
    return true;
}
}

PatchReview::PatchReview(std::filesystem::path workspaceRoot, StructuredPatch patch)
    : workspaceRoot_(std::filesystem::weakly_canonical(std::move(workspaceRoot))),
      patch_(std::move(patch)) {}

const StructuredPatch& PatchReview::patch() const { return patch_; }
PatchReviewDecision PatchReview::decision() const { return decision_; }
void PatchReview::accept() { decision_ = PatchReviewDecision::Accepted; }
void PatchReview::reject() { decision_ = PatchReviewDecision::Rejected; }

PatchApplyResult PatchReview::apply() const {
    if (decision_ != PatchReviewDecision::Accepted)
        return {false, decision_ == PatchReviewDecision::Rejected
                           ? "Patch was rejected."
                           : "Patch requires explicit acceptance before apply."};
    if (patch_.path.empty() || patch_.empty())
        return {false, "Patch has no applicable changes."};

    WorkspaceFileReader reader(workspaceRoot_);
    std::string current, error;
    if (!reader.read(patch_.path, current, error))
        return {false, error};
    if (current != patch_.originalText)
        return {false, "Workspace file changed after the patch was prepared."};

    std::error_code ec;
    const auto target = std::filesystem::weakly_canonical(workspaceRoot_ / patch_.path, ec);
    if (ec || !isWithin(workspaceRoot_, target))
        return {false, "Patch target is outside the workspace."};

    std::ofstream output(target, std::ios::binary | std::ios::trunc);
    if (!output) return {false, "Patch target could not be opened for writing."};
    output.write(patch_.proposedText.data(),
                 static_cast<std::streamsize>(patch_.proposedText.size()));
    if (!output) return {false, "Failed while writing the accepted patch."};
    return {true, {}};
}
