#pragma once
#include <string>

namespace AIPromptPipeline {
std::string buildPrompt(const std::string& prompt, const std::string& currentContext);
std::string extractResponse(const std::string& generatedText, const std::string& formattedPrompt);
}
