#pragma once
#include <string>
#include <vector>
#include <functional>
#include <chrono>
#include <windows.h>
#include "NppAIEngine.h"

class AIManager {
public:
    static AIManager& getInstance() {
        static AIManager instance;
        return instance;
    }

    // Ładuje nasz własny model GGUF (lub customowy binarny)
    void loadModel(const std::string& modelPath);

    // Funkcja wywołująca model AI z obsługą strumieniowania do interfejsu (char, isThought)
    std::string generateCode(const std::string& prompt, const std::string& currentContext, std::function<void(char, bool)> onToken = nullptr, std::function<void(int)> onRemove = nullptr);

    // Zatrzymuje działanie generatora
    void stopGeneration() { engine.stopGeneration(); }

    // Funkcje śledzące dla Diff Trackera (uczenie z zachowania usera)
    void startTracking(const std::string& prompt, const std::string& generatedCode, int startLine, int endLine, const std::string& filePath = "");
    void checkModifications(const std::string& currentTextInEditor);
    void onEditorModified(HWND hwnd, int position, int linesAdded);

    // Sprawdza i pobiera nowe wagi modelu z chmury
    void checkAndDownloadModelUpdate();

private:
    AIManager() = default;
    ~AIManager() = default;
    AIManager(const AIManager&) = delete;
    AIManager& operator=(const AIManager&) = delete;

    // Zmienne stanu śledzenia
    bool isTracking = false;
    std::string lastPrompt;
    std::string lastGeneratedCode;
    std::string trackedFilePath;
    int trackedStartLine = 0;
    int trackedEndLine = 0;
    HWND trackedHwnd = nullptr;

    // Debouncing dla "Cichego Obserwatora"
    std::chrono::steady_clock::time_point lastModificationTime;
    bool isDebounceThreadRunning = false;

    // Nasz autorski silnik AI
    NppAIEngine engine;
};
