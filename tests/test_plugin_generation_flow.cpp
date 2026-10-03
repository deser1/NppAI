#include "AIPromptPipeline.h"
#include "GenerationContext.h"
#include "GenerationStreamRouter.h"
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

    std::string editorOutput;
    std::string thoughtOutput;
    int removedCharacters = 0;
    GenerationStreamRouter streamRouter(
        [&](const std::string& thought) { thoughtOutput = thought; },
        [&](char c) { editorOutput += c; },
        [&](int count) {
            removedCharacters += count;
            while (count-- > 0 && !editorOutput.empty())
                editorOutput.pop_back();
        });

    const std::string streamedCode = "int sum = a + c;";
    for (char c : streamedCode)
        streamRouter.onToken(c, false);

    const std::string streamedThought = "checking arguments\n";
    for (char c : streamedThought)
        streamRouter.onToken(c, true);

    streamRouter.onRemove(2);
    streamRouter.onToken('b', false);
    streamRouter.onToken(';', false);

    ok &= check(editorOutput == "int sum = a + b;",
                "streamed model code and backtracking reach the editor output");
    ok &= check(streamRouter.thoughtBuffer() == streamedThought &&
                    thoughtOutput == streamedThought,
                "reasoning stream stays separate from editor output");
    ok &= check(removedCharacters == 2,
                "model backtracking reaches the editor removal callback");

    if (!ok)
        return 1;

    std::cout << "Plugin generation flow integration tests passed.\n";
    return 0;
}