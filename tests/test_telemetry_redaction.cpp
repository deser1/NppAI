#include "TelemetryRedaction.h"
#include <iostream>
#include <string>

static bool check(bool value, const char* message) {
    if (!value) std::cerr << "FAIL: " << message << "\n";
    return value;
}

int main() {
    bool ok = true;
    const std::string apiKey = std::string("sk") + "-" + std::string(24, 'A');
    const std::string ghToken = std::string("gh") + "p_" + std::string(36, 'B');
    const std::string input = "key=" + apiKey + " token=" + ghToken +
        " host=192.168.10.42 path=C:\\Users\\Example\\secret.cpp";

    const auto redacted = TelemetryRedaction::redact(input);
    ok &= check(redacted.find(apiKey) == std::string::npos, "API key is removed");
    ok &= check(redacted.find(ghToken) == std::string::npos, "GitHub token is removed");
    ok &= check(redacted.find("192.168.10.42") == std::string::npos, "IP address is removed");
    ok &= check(redacted.find("C:\\Users\\Example") == std::string::npos, "local path is removed");

    const auto payload = TelemetryRedaction::buildLearningPayload(
        "say \"hello\" " + apiKey, "generated\ncode", "final\tcode");
    ok &= check(payload.find(apiKey) == std::string::npos, "payload redacts prompt");
    ok &= check(payload.find("\\\"hello\\\"") != std::string::npos, "JSON quotes are escaped");
    ok &= check(payload.find("generated\\ncode") != std::string::npos, "JSON newline is escaped");
    ok &= check(payload.find("\"generated_code\"") != std::string::npos, "generated code is preserved");
    return ok ? 0 : 1;
}
