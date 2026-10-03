#include "AIPromptPipeline.h"
#include "GenerationContext.h"
#include "PluginPromptActions.h"

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

    const std::string selection = "int add(int a, int b) { return a + b; }";
    const std::string editorPrompt =
        PluginPromptActions::appendSelection("Explain and improve this code", selection);

    const std::string context = GenerationContext::select(
        "Project convention: validate arguments before computation.",
        "stale legacy memory");

    const std::string formatted = AIPromptPipeline::buildPrompt(editorPrompt, context);

    const std::string expected =
        "[SYSTEM]: Kontekst poprzednich modyfikacji dla tego pliku:\n"
        "Project convention: validate arguments before computation.\n\n"
        "[USER]: Explain and improve this code\r\n"
        "```\r\nint add(int a, int b) { return a + b; }\r\n```\r\n"
        "\n[AI]:\n";

    ok &= check(formatted == expected,
                "selection, RAG context and model prompt compose end to end");

    const std::string generated = formatted + "int add(int a, int b) { return a + b; }";
    ok &= check(AIPromptPipeline::extractResponse(generated, formatted) ==
                    "int add(int a, int b) { return a + b; }",
                "model response is separated from the composed prompt");

    const std::string fallback = GenerationContext::select("", "old-1\nold-2\n", 6);
    const std::string fallbackPrompt = AIPromptPipeline::buildPrompt(
        PluginPromptActions::appendSelection("", "x++;"), fallback);

    ok &= check(fallback == "old-2\n",
                "legacy fallback keeps the newest bounded context");
    ok &= check(fallbackPrompt.find("[SYSTEM]:") == 0 &&
                    fallbackPrompt.find("```\r\nx++;\r\n```") != std::string::npos,
                "legacy fallback also reaches the model prompt");

    if (!ok)
        return 1;

    std::cout << "Plugin generation flow integration tests passed.\n";
    return 0;
}