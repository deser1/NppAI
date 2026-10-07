#include "AgentTaskOutcome.h"
#include <iostream>
#include <string>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << "\n";
    return condition;
}
}

int main() {
    bool ok = true;
    const auto patch = StructuredPatchBuilder::build(
        "src/a.cpp", "int x = 1;\n", "int x = 2;\n");

    RepairLoopResult success;
    success.succeeded = true;
    success.attempts = 2;
    success.history.push_back({1, "compiler: expected \";\"\nline 4"});
    const auto good = AgentTaskOutcomeBuilder::fromRepairResult(
        "Fix \"x\"\nvalue", patch, success, false);
    ok &= check(good.status == AgentTaskStatus::Succeeded &&
                    good.attempts == 2 && good.diagnostics.size() == 1,
                "successful repair result becomes a successful task outcome");
    const auto json = good.toJsonLine();
    ok &= check(json.find(R"json("task":"Fix \\"x\\"\\nvalue")json") != std::string::npos,
                "JSONL escapes task text deterministically");
    ok &= check(json.find(R"json(compiler: expected \\";\\"\\nline 4)json") != std::string::npos,
                "JSONL escapes diagnostic text deterministically");
    ok &= check(json.find(R"json("status":"succeeded")json") != std::string::npos &&
                    json.find(R"json("patch":)json") != std::string::npos,
                "JSONL contains status and structured patch diff");

    RepairLoopResult failed;
    failed.exhausted = true;
    failed.attempts = 3;
    failed.history.push_back({1, "error one"});
    failed.history.push_back({2, "error two"});
    failed.history.push_back({3, "error three"});
    const auto rolledBack = AgentTaskOutcomeBuilder::fromRepairResult(
        "repair task", patch, failed, true);
    ok &= check(rolledBack.status == AgentTaskStatus::RolledBack &&
                    rolledBack.diagnostics.size() == 3,
                "exhausted rolled-back task preserves all failed diagnostics");

    const auto plainFailure = AgentTaskOutcomeBuilder::fromRepairResult(
        "repair task", patch, failed, false);
    ok &= check(plainFailure.status == AgentTaskStatus::Failed,
                "unrecovered exhausted task remains failed");

    return ok ? 0 : 1;
}
