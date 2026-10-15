#pragma once
#include <string>
#include <functional>

namespace AIPromptPipeline {
struct GenerationRequest {
    std::string task;
    std::string basePrompt;
    std::string inferencePrompt;
    std::string assistantPrefix;
    std::string error;
};
GenerationRequest prepare(const std::string& prompt, const std::string& currentContext);
void streamPrefix(const GenerationRequest& request, const std::function<void(char, bool)>& onToken);
std::string buildPrompt(const std::string& prompt, const std::string& currentContext);
std::string extractResponse(const std::string& generatedText, const std::string& formattedPrompt);
}
