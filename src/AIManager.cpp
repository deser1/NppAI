#include "AIManager.h"
#include "TelemetryManager.h"
#include <chrono>

void AIManager::loadModel(const std::string& modelPath) {
    // Inicjalizacja naszego autorskiego silnika AI
    engine.loadModel(modelPath);
}

std::string AIManager::generateCode(const std::string& prompt, const std::string& currentContext) {
    // Generowanie kodu za pomocą naszego własnego silnika Transformera
    std::string generatedCode = engine.generate(prompt, 64);
    
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
