#include "AIPromptPipeline.h"

namespace AIPromptPipeline {

std::string buildPrompt(const std::string& prompt, const std::string& currentContext) {
    std::string formattedPrompt;
    if (!currentContext.empty()) {
        formattedPrompt +=
            "[SYSTEM]: Kontekst poprzednich modyfikacji dla tego pliku:\n" +
            currentContext + "\n\n";
    }
    formattedPrompt += "[USER]: " + prompt + "\n[AI]:\n";
    return formattedPrompt;
}

std::string extractResponse(const std::string& generatedText, const std::string& formattedPrompt) {
    if (generatedText.rfind(formattedPrompt, 0) == 0)
        return generatedText.substr(formattedPrompt.length());
    return generatedText;
}

}
