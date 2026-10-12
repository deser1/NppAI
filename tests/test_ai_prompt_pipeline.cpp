#include "AIPromptPipeline.h"
#include <iostream>
#include <string>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}
}

int main() {
    bool ok = true;

    const std::string noContext = AIPromptPipeline::buildPrompt("write a loop", "");
    ok &= check(noContext == "[USER]: write a loop\n[AI]:\n",
                "prompt without context");

    const std::string withContext =
        AIPromptPipeline::buildPrompt("fix it", "int value = 1;");
    ok &= check(withContext ==
                    "[SYSTEM]: Kontekst poprzednich modyfikacji dla tego pliku:\n"
                    "int value = 1;\n\n[USER]: fix it\n[AI]:\n",
                "prompt with context");

    ok &= check(AIPromptPipeline::extractResponse(
                    noContext + "for (;;) {}", noContext) == "for (;;) {}",
                "generated prompt prefix removed");

    ok &= check(AIPromptPipeline::extractResponse(
                    "independent output", noContext) == "independent output",
                "independent output preserved");

    ok &= check(AIPromptPipeline::extractResponse(noContext, noContext).empty(),
                "prompt-only generation becomes empty response");

    const auto plain = AIPromptPipeline::prepare("write a loop", "");
    ok &= check(plain.error.empty() && plain.assistantPrefix.empty() && plain.inferencePrompt == noContext,
                "ordinary prompts retain their original generation path");
    const auto guided = AIPromptPipeline::prepare(
        "@python-signature fn_add_13(a,b)\r\nReturn the sum of a and b.", "project context");
    ok &= check(guided.error.empty() && guided.assistantPrefix == "def fn_add_13(a, b):\n    ",
                "explicit task signature is normalized with LF");
    ok &= check(guided.inferencePrompt == guided.basePrompt + guided.assistantPrefix &&
                guided.basePrompt.find("project context") != std::string::npos &&
                guided.basePrompt.find("@python-signature") == std::string::npos,
                "guided prompt retains RAG context but excludes directive syntax");
    ok &= check(AIPromptPipeline::extractResponse(guided.inferencePrompt + "return a + b\n", guided.basePrompt)
                == "def fn_add_13(a, b):\n    return a + b\n",
                "returned response includes the requested signature exactly once");
    for (const std::string& invalid : {
        "@python-signature f(x,x)\nTask", "@python-signature f(x=1)\nTask",
        "@python-signature f(for)\nTask", "@python-signature class(x)\nTask",
        "@python-signature f(g(x))\nTask", "@python-signature f(x) g(y)\nTask",
        "@python-signature f(x,)\nTask", "@python-signature f(x)",
        "@python-signaturef(x)\nTask"}) {
        const auto rejected = AIPromptPipeline::prepare(invalid, "");
        int callbacks = 0;
        AIPromptPipeline::streamPrefix(rejected, [&](char, bool) { ++callbacks; });
        ok &= check(!rejected.error.empty() && callbacks == 0,
                    "malformed signature is rejected without streaming");
    }

    if (!ok)
        return 1;

    std::cout << "AI prompt pipeline tests passed.\n";
    return 0;
}
