#include "TelemetryManager.h"
#include <thread>
#include <iostream>
#include <fstream>
#include <chrono>

std::string TelemetryManager::anonymizeCode(const std::string& inputCode) {
    std::string safeCode = inputCode;

    // 1. Usuwanie kluczy API (np. OpenAI sk-...)
    std::regex openAiKey(R"(sk-[a-zA-Z0-9]{20,})");
    safeCode = std::regex_replace(safeCode, openAiKey, "[REDACTED_API_KEY]");

    // 2. Usuwanie adresów IP
    std::regex ipAddress(R"(\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b)");
    safeCode = std::regex_replace(safeCode, ipAddress, "[REDACTED_IP]");

    // 3. Usuwanie tokenów GitHub (ghp_...)
    std::regex githubToken(R"(ghp_[a-zA-Z0-9]{36})");
    safeCode = std::regex_replace(safeCode, githubToken, "[REDACTED_GH_TOKEN]");

    // 4. Usuwanie ścieżek z dysku C: (np. C:\Users\Nazwa)
    std::regex localPaths(R"([C-Z]:\\[\w\\]+)");
    safeCode = std::regex_replace(safeCode, localPaths, "[REDACTED_PATH]");

    return safeCode;
}

void TelemetryManager::queueLearningData(const std::string& prompt, const std::string& generatedCode, const std::string& userModifiedCode) {
    // Najpierw anonimizujemy dane
    std::string safePrompt = anonymizeCode(prompt);
    std::string safeGenerated = anonymizeCode(generatedCode);
    std::string safeModified = anonymizeCode(userModifiedCode);

    // Na potrzeby tego prototypu zapiszemy to lokalnie do pliku JSON Lines (JSONL)
    // W pełnej wersji tutaj będzie zapytanie HTTP POST do Twojego serwera NppAI
    
    std::thread([safePrompt, safeGenerated, safeModified]() {
        // Symulacja wysyłania asynchronicznego
        std::ofstream logFile("C:\\NppAI_Telemetry.jsonl", std::ios::app);
        if (logFile.is_open()) {
            logFile << "{ \"prompt\": \"" << safePrompt 
                    << "\", \"ai_code\": \"" << safeGenerated 
                    << "\", \"user_code\": \"" << safeModified << "\" }\n";
            logFile.close();
        }
    }).detach();
}

void TelemetryManager::processQueueAsync() {
    // Miejsce na implementację kolejki z WinHTTP / libcurl
}
