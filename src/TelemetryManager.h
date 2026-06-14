#pragma once
#include <string>
#include <regex>

class TelemetryManager {
public:
    static TelemetryManager& getInstance() {
        static TelemetryManager instance;
        return instance;
    }

    // Funkcja filtrująca prywatne dane z kodu (API keys, IP, itp.)
    std::string anonymizeCode(const std::string& inputCode);

    // Zapisuje log do wysłania (Prompt + AI Code + Poprawiony przez usera)
    void queueLearningData(const std::string& prompt, const std::string& generatedCode, const std::string& userModifiedCode);

    // Funkcja wysyłająca dane asynchronicznie w tle (do implementacji WinHTTP)
    void processQueueAsync();

private:
    TelemetryManager() = default;
    ~TelemetryManager() = default;
    TelemetryManager(const TelemetryManager&) = delete;
    TelemetryManager& operator=(const TelemetryManager&) = delete;

    // TODO: Kolejka wątków i dane logowania
};
