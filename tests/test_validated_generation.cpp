#include "ValidatedGeneration.h"
#include <iostream>
#include <stdexcept>
#include <vector>
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main() {
    try {
        ValidatedGenerationRequest request{"@python-signature add(a, b)\nAdd.", "context", "task.py", "old\n", 0, 4, true};
        int generated = 0, builds = 0, tests = 0;
        std::vector<bool> forced;
        auto result = ValidatedGeneration::run(request,
            [&](const std::string& task, const std::string& context, bool online) {
                check(task == request.task, "repair must preserve original task/signature");
                if (generated) check(context.find("Diagnostics:") != std::string::npos, "repair carries actual diagnostics");
                forced.push_back(online); ++generated;
                return GeneratedCandidate{generated == 3 ? "fixed\n" : "broken\n", online};
            }, [&](const std::string& code, bool isTest) {
                if (isTest) { ++tests; return AgentCommandResult{0, "authored tests passed"}; }
                ++builds; return AgentCommandResult{code == "fixed\n" ? 0 : 1, "SyntaxError: broken"};
            }, [] { return false; });
        check(result.ready() && result.repair.attempts == 3 && generated == 3 && builds == 3 && tests == 1,
              "three candidates; syntax failure prevents tests; final proposal passes");
        check(forced == std::vector<bool>({false, false, true}), "two failed local candidates trigger final online escalation");
        check(request.original == "old\n", "validation must not modify original source");
        PatchReview review(".", result.patch);
        check(!review.checkBuffer(request.original).applied, "review requires explicit acceptance");
        review.accept();
        check(review.checkBuffer(request.original).applied, "accepted verified snapshot can apply");
        check(!review.checkBuffer("user edited\n").applied, "stale buffer blocks apply");
        review.reject(); check(!review.checkBuffer(request.original).applied, "rejection blocks apply");
        check(result.outcome.diagnostics.size() == 2 && result.outcome.patch.find("+fixed") != std::string::npos,
              "outcome includes failures and final review diff");

        request.allowOnlineEscalation = false; generated = builds = tests = 0;
        result = ValidatedGeneration::run(request,
            [&](const auto&, const auto&, bool online) { check(!online, "local policy must never escalate"); ++generated; return GeneratedCandidate{"bad\n", false}; },
            [&](const auto&, bool isTest) { if (isTest) ++tests; else ++builds; return AgentCommandResult{isTest ? 1 : 0, "assertion failed"}; },
            [] { return false; });
        check(!result.ready() && result.repair.exhausted && generated == 3 && builds == 3 && tests == 3,
              "failed tests exhaust finite budget without touching source");
        generated = 0;
        result = ValidatedGeneration::run(request,
            [&](const auto&, const auto&, bool) { ++generated; return GeneratedCandidate{"code\n", false}; },
            [](const auto&, bool) { return AgentCommandResult{2, "Docker unavailable"}; }, [] { return false; });
        check(result.blocked && generated == 1 && result.repair.attempts == 1 && !result.ready(),
              "infrastructure failure stops repairs immediately");
        generated = 0;
        result = ValidatedGeneration::run(request,
            [&](const auto&, const auto&, bool) { ++generated; return GeneratedCandidate{"def silnia(n):\n    return 1\n", true}; },
            [](const auto&, bool) -> AgentCommandResult { throw std::runtime_error("Configure Python validation: missing setup"); },
            [] { return false; });
        check(generated == 1 && result.blocked && !result.ready() &&
              result.patch.unifiedDiff().find("+def silnia(n)") != std::string::npos,
              "missing validator preserves the actual model proposal but cannot enable Apply");
        generated = 0;
        result = ValidatedGeneration::run(request,
            [&](const auto&, const auto&, bool) { ++generated; return GeneratedCandidate{"code", false}; },
            [](const auto&, bool) { return AgentCommandResult{0, "pass"}; }, [] { return true; });
        check(result.cancelled && generated == 0 && !result.ready(), "cancelled request never invokes a backend");
        request.selectionEnd = 99;
        result = ValidatedGeneration::run(request, {}, {}, [] { return false; });
        check(result.blocked && !result.ready(), "invalid selection fails before generation");
        request.selectionEnd = 4;
        result = ValidatedGeneration::run(request,
            [](const auto&, const auto&, bool) { return GeneratedCandidate{std::string(131073, 'x'), false}; }, {}, [] { return false; });
        check(result.blocked && result.patch.empty(), "oversized candidate never enters diff construction");
        std::cout << "Validated generation, escalation, review and cancellation tests passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
