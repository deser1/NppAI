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

    if (!ok)
        return 1;

    std::cout << "AI prompt pipeline tests passed.\n";
    return 0;
}
