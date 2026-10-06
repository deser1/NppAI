#include "PatchReview.h"
#include "PatchRollback.h"
#include "StructuredPatch.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << "\n";
    return condition;
}
std::string readAll(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
}

int main() {
    bool ok = true;
    const auto base = std::filesystem::temp_directory_path() / "nppai_patch_rollback";
    const auto workspace = base / "workspace";
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(workspace);
    const auto file = workspace / "code.cpp";

    { std::ofstream(file, std::ios::binary) << "original\n"; }
    const auto patch = StructuredPatchBuilder::build("code.cpp", "original\n", "proposed\n");
    PatchReview review(workspace, patch);
    review.accept();
    ok &= check(review.apply().applied && readAll(file) == "proposed\n",
                "fixture applies accepted patch before rollback");

    PatchRollback rollback(workspace, patch);
    ok &= check(rollback.restore().restored && readAll(file) == "original\n",
                "rollback restores exact original content");

    { std::ofstream(file, std::ios::binary | std::ios::trunc) << "proposed\n"; }
    { std::ofstream(file, std::ios::binary | std::ios::trunc) << "user edit\n"; }
    const auto changed = rollback.restore();
    ok &= check(!changed.restored && readAll(file) == "user edit\n",
                "rollback refuses to overwrite a later user change");

    { std::ofstream(base / "outside.cpp", std::ios::binary) << "proposed\n"; }
    const auto outsidePatch = StructuredPatchBuilder::build("../outside.cpp", "original\n", "proposed\n");
    PatchRollback outside(workspace, outsidePatch);
    ok &= check(!outside.restore().restored && readAll(base / "outside.cpp") == "proposed\n",
                "rollback cannot escape workspace boundary");

    std::filesystem::remove_all(base);
    return ok ? 0 : 1;
}
