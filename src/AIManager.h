#pragma once
#include <string>
#include "NppAIEngine.h"

class AIManager {
public:
    static AIManager& getInstance() {
        static AIManager instance;
        return instance;
    }

    // Ładuje nasz własny model GGUF (lub customowy binarny)
    void loadModel(const std::string& modelPath);

    // Funkcja wywołująca model AI
    std::string generateCode(const std::string& prompt, const std::string& currentContext);

    // Funkcje śledzące dla Diff Trackera (uczenie z zachowania usera)
    void startTracking(const std::string& prompt, const std::string& generatedCode, int startLine, int endLine);
    void checkModifications(const std::string& currentTextInEditor);

private:
    AIManager() = default;
    ~AIManager() = default;
    AIManager(const AIManager&) = delete;
    AIManager& operator=(const AIManager&) = delete;

    // Zmienne stanu śledzenia
    bool isTracking = false;
    std::string lastPrompt;
    std::string lastGeneratedCode;
    int trackedStartLine = 0;
    int trackedEndLine = 0;

    // Nasz autorski silnik AI
    NppAIEngine engine;
};
