#include "TelemetryRedaction.h"
#include <regex>

namespace TelemetryRedaction {
std::string redact(const std::string& input) {
    std::string safe = input;
    safe = std::regex_replace(safe, std::regex(R"(sk-[a-zA-Z0-9]{20,})"), "[REDACTED_API_KEY]");
    safe = std::regex_replace(safe, std::regex(R"(\b(?:\d{1,3}\.){3}\d{1,3}\b)"), "[REDACTED_IP]");
    safe = std::regex_replace(safe, std::regex(R"(gh[pousr]_[a-zA-Z0-9_]{20,})"), "[REDACTED_GH_TOKEN]");
    safe = std::regex_replace(safe, std::regex(R"([C-Z]:\\[^\r\n\"']+)"), "[REDACTED_PATH]");
    return safe;
}

std::string escapeJson(const std::string& input) {
    std::string output;
    for (char c : input) {
        switch (c) {
        case '"': output += "\\\""; break;
        case '\\': output += "\\\\"; break;
        case '\b': output += "\\b"; break;
        case '\f': output += "\\f"; break;
        case '\n': output += "\\n"; break;
        case '\r': output += "\\r"; break;
        case '\t': output += "\\t"; break;
        default: output += c; break;
        }
    }
    return output;
}

std::string buildLearningPayload(const std::string& prompt,
                                 const std::string& generatedCode,
                                 const std::string& userModifiedCode) {
    const std::string safePrompt = redact(prompt);
    const std::string safeGenerated = redact(generatedCode);
    const std::string safeModified = redact(userModifiedCode);
    return "{ \"prompt\": \"" + escapeJson(safePrompt) +
           "\", \"generated_code\": \"" + escapeJson(safeGenerated) +
           "\", \"final_code\": \"" + escapeJson(safeModified) +
           "\", \"thought_process\": \"\" }";
}
}
