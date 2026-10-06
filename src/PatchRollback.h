#pragma once
#include "StructuredPatch.h"
#include <filesystem>
#include <string>

struct PatchRollbackResult {
    bool restored = false;
    std::string error;
};

class PatchRollback {
public:
    PatchRollback(std::filesystem::path workspaceRoot, StructuredPatch patch);
    PatchRollbackResult restore() const;

private:
    std::filesystem::path workspaceRoot_;
    StructuredPatch patch_;
};
