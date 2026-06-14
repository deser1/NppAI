#include "AIManager.h"
#include "TelemetryManager.h"
#include "PluginDefinition.h"
#include <chrono>
#include <windows.h>
#include <shlwapi.h>

extern NppData nppData;

std::string getPluginDllDirectory() {
    char path[MAX_PATH];
    HMODULE hm = NULL;
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
          (LPCSTR)&getPluginDllDirectory, &hm)) {
        GetModuleFileNameA(hm, path, sizeof(path));
        PathRemoveFileSpecA(path);
        return std::string(path);
    }
    return "";
}

void AIManager::loadModel(const std::string& modelPath) {
    // Inicjalizacja naszego autorskiego silnika AI
    std::string fullPath = modelPath;
    if (modelPath.find(":") == std::string::npos) { // Jeśli to ścieżka względna
        fullPath = getPluginDllDirectory() + "\\" + modelPath;
    }
    engine.loadModel(fullPath);
}

std::string AIManager::generateCode(const std::string& prompt, const std::string& currentContext) {
    // Formatowanie promptu do formatu "Instruct", którego uczy się model
    std::string formattedPrompt = "[USER]: " + prompt + "\n[AI]:\n";
    
    // Generowanie kodu za pomocą naszego własnego silnika Transformera
    // Zwiększamy limit tokenów do 4096, by model mógł wypisać dłuższą odpowiedź HTML/Vue
    std::string generatedCode = engine.generate(formattedPrompt, 4096);
    
    // Usunięcie wpisanego promptu (z tagami), aby do edytora trafiła sama wygenerowana odpowiedź AI
    if (generatedCode.find(formattedPrompt) == 0) {
        generatedCode = generatedCode.substr(formattedPrompt.length());
    }
    
    return generatedCode;
}

void AIManager::startTracking(const std::string& prompt, const std::string& generatedCode, int startLine, int endLine) {
    this->isTracking = true;
    this->lastPrompt = prompt;
    this->lastGeneratedCode = generatedCode;
    this->trackedStartLine = startLine;
    this->trackedEndLine = endLine;
}

void AIManager::checkModifications(const std::string& currentTextInEditor) {
    if (!isTracking) return;

    // Gdy użytkownik edytuje, sprawdzamy czy kod różni się od wygenerowanego
    // W uproszczeniu pobieramy fragment kodu z edytora i wysyłamy jako telemetrię
    
    // Zatrzymujemy śledzenie po wysłaniu lekcji
    isTracking = false;

    // Wysyłamy dane do telemetrii
    TelemetryManager::getInstance().queueLearningData(
        lastPrompt, 
        lastGeneratedCode, 
        currentTextInEditor // Tutaj w pełnej wersji przekazujemy tylko wyodrębniony zmieniony fragment
    );
}
