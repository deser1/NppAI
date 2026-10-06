#include "AgentFeedbackLoop.h"
#include "PatchReview.h"
#include "PatchRollback.h"
#include "StructuredPatch.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << "\n";
    return condition;
}
std::string readAll(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
void writeAll(const std::filesystem::path& path, const std::string& text) {
    std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}
}

int main() {
    bool ok = true;
    const auto base = std::filesystem::temp_directory_path() / "nppai_agent_workflow";
    const auto workspace = base / "workspace";
    const auto file = workspace / "task.cpp";
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(workspace);

    // Successful task: review -> apply -> build -> test -> keep patch.
    writeAll(file, "int value() { return 1; }\n");
    const auto goodPatch = StructuredPatchBuilder::build(
        "task.cpp",
        "int value() { return 1; }\n",
        "int value() { return 2; }\n");
    PatchReview goodReview(workspace, goodPatch);
    goodReview.accept();
    ok &= check(goodReview.apply().applied, "accepted successful-task patch applies");

    AgentFeedbackLoop validator;
    const auto success = validator.validate(
        [&] {
            return AgentCommandResult{
                readAll(file).find("return 2") != std::string::npos ? 0 : 1,
                "build expected patched source"};
        },
        [&] {
            return AgentCommandResult{
                readAll(file) == goodPatch.proposedText ? 0 : 2,
                "test expected exact proposed content"};
        });
    ok &= check(success.succeeded && readAll(file) == goodPatch.proposedText,
                "successful validation preserves applied patch");

    // Failing task: review -> apply -> failed build -> diagnostics -> rollback.
    writeAll(file, "int value() { return 2; }\n");
    const auto badPatch = StructuredPatchBuilder::build(
        "task.cpp",
        "int value() { return 2; }\n",
        "int value() { return broken; }\n");
    PatchReview badReview(workspace, badPatch);
    badReview.accept();
    ok &= check(badReview.apply().applied, "accepted failing-task patch applies");

    int testsCalled = 0;
    const auto failure = validator.validate(
        [] { return AgentCommandResult{1, "compiler: 'broken' was not declared"}; },
        [&] { ++testsCalled; return AgentCommandResult{0, {}}; });
    ok &= check(!failure.succeeded && testsCalled == 0 &&
                    failure.repairFeedback == "compiler: 'broken' was not declared",
                "failed build preserves diagnostics and prevents tests");

    PatchRollback recovery(workspace, badPatch);
    const auto rollback = recovery.restore();
    ok &= check(rollback.restored && readAll(file) == badPatch.originalText,
                "failed task rolls workspace back to pre-patch content");

    std::filesystem::remove_all(base);
    return ok ? 0 : 1;
}
