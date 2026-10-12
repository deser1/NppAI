#include "AIPromptPipeline.h"
#include <regex>
#include <set>
#include <sstream>

namespace AIPromptPipeline {

namespace {
std::string trim(const std::string& text) {
    const auto first = text.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    return text.substr(first, text.find_last_not_of(" \t\r\n") - first + 1);
}
bool identifier(const std::string& name) {
    static const std::regex pattern("[A-Za-z_][A-Za-z_0-9]*");
    static const std::set<std::string> keywords = {
        "False", "None", "True", "and", "as", "assert", "async", "await",
        "break", "class", "continue", "def", "del", "elif", "else", "except",
        "finally", "for", "from", "global", "if", "import", "in", "is",
        "lambda", "nonlocal", "not", "or", "pass", "raise", "return",
        "try", "while", "with", "yield"};
    return std::regex_match(name, pattern) && !keywords.count(name);
}
}

GenerationRequest prepare(const std::string& task, const std::string& context) {
    GenerationRequest request;
    std::string instruction = task;
    const std::string directive = "@python-signature";
    if (task.rfind(directive, 0) == 0) {
        const auto newline = task.find('\n');
        const std::string signature = trim(task.substr(directive.size(), newline == std::string::npos ? std::string::npos : newline - directive.size()));
        const std::string description = newline == std::string::npos ? "" : trim(task.substr(newline + 1));
        std::smatch match;
        static const std::regex pattern("([A-Za-z_][A-Za-z_0-9]*)\\(([^()]*)\\)");
        auto invalid = [&request]() {
            request.error = "Use @python-signature name(arg1, arg2) on the first line, then describe the Python function. Plain distinct parameter names are required.";
            return request;
        };
        if (task.size() <= directive.size() || (task[directive.size()] != ' ' && task[directive.size()] != '\t') || signature.size() > 4096 || description.empty() || !std::regex_match(signature, match, pattern) || !identifier(match[1].str()))
            return invalid();
        const std::string arguments = trim(match[2].str());
        std::string normalized;
        std::set<std::string> seen;
        if (!arguments.empty()) {
            if (arguments.back() == ',') return invalid();
            std::istringstream input(arguments);
            std::string argument;
            while (std::getline(input, argument, ',')) {
                argument = trim(argument);
                if (!identifier(argument) || !seen.insert(argument).second) return invalid();
                if (!normalized.empty()) normalized += ", ";
                normalized += argument;
            }
        }
        const std::string declaration = match[1].str() + "(" + normalized + ")";
        instruction = "Implement " + declaration + ". " + description;
        request.assistantPrefix = "def " + declaration + ":\n    ";
    }
    request.basePrompt = buildPrompt(instruction, context);
    request.inferencePrompt = request.basePrompt + request.assistantPrefix;
    return request;
}

void streamPrefix(const GenerationRequest& request, const std::function<void(char, bool)>& onToken) {
    if (request.error.empty() && onToken)
        for (char c : request.assistantPrefix) onToken(c, false);
}

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
