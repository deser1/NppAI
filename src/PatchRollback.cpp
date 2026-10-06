#include "PatchRollback.h"
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

PatchRollback::PatchRollback(std::filesystem::path workspaceRoot, StructuredPatch patch)
    : workspaceRoot_(std::filesystem::weakly_canonical(std::move(workspaceRoot))),
      patch_(std::move(patch)) {}

PatchRollbackResult PatchRollback::restore() const {
    if (patch_.path.empty() || patch_.empty())
        return {false, "Patch has no changes to roll back."};

    WorkspaceFileReader reader(workspaceRoot_);
    std::string current, error;
    if (!reader.read(patch_.path, current, error))
        return {false, error};
    if (current != patch_.proposedText)
        return {false, "Workspace file changed after the patch was applied; rollback refused."};

    std::error_code ec;
    const auto target = std::filesystem::weakly_canonical(workspaceRoot_ / patch_.path, ec);
    if (ec || !isWithin(workspaceRoot_, target))
        return {false, "Rollback target is outside the workspace."};

    std::ofstream output(target, std::ios::binary | std::ios::trunc);
    if (!output) return {false, "Rollback target could not be opened for writing."};
    output.write(patch_.originalText.data(),
                 static_cast<std::streamsize>(patch_.originalText.size()));
    if (!output) return {false, "Failed while restoring the original file content."};
    return {true, {}};
}
