#pragma once
#include <string>

namespace TelemetryRedaction {
std::string redact(const std::string& input);
std::string escapeJson(const std::string& input);
std::string buildLearningPayload(const std::string& prompt,
                                 const std::string& generatedCode,
                                 const std::string& userModifiedCode);
}
