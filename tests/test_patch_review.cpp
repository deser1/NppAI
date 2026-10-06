#include "PatchReview.h"
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
    const auto base = std::filesystem::temp_directory_path() / "nppai_patch_review";
    const auto workspace = base / "workspace";
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(workspace);
    const auto file = workspace / "code.cpp";
    { std::ofstream(file, std::ios::binary) << "old\n"; }

    auto patch = StructuredPatchBuilder::build("code.cpp", "old\n", "new\n");
    PatchReview pending(workspace, patch);
    ok &= check(!pending.apply().applied && readAll(file) == "old\n",
                "pending patch cannot modify workspace");

    PatchReview rejected(workspace, patch);
    rejected.reject();
    ok &= check(!rejected.apply().applied && readAll(file) == "old\n",
                "rejected patch cannot modify workspace");

    PatchReview accepted(workspace, patch);
    accepted.accept();
    ok &= check(accepted.apply().applied && readAll(file) == "new\n",
                "accepted patch applies proposed content");

    { std::ofstream(file, std::ios::binary | std::ios::trunc) << "external\n"; }
    PatchReview stale(workspace, patch);
    stale.accept();
    ok &= check(!stale.apply().applied && readAll(file) == "external\n",
                "stale accepted patch does not overwrite newer content");

    { std::ofstream(base / "outside.cpp", std::ios::binary) << "outside\n"; }
    auto outsidePatch = StructuredPatchBuilder::build("../outside.cpp", "outside\n", "bad\n");
    PatchReview outside(workspace, outsidePatch);
    outside.accept();
    ok &= check(!outside.apply().applied && readAll(base / "outside.cpp") == "outside\n",
                "accepted patch cannot escape workspace boundary");

    std::filesystem::remove_all(base);
    return ok ? 0 : 1;
}
