#include "AIManager.h"
#include "TelemetryManager.h"
#include "PluginDefinition.h"
#include "Scintilla.h"
#include <chrono>
#include <windows.h>
#include <shlwapi.h>
#include <thread>
#include <fstream>

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

std::string AIManager::generateCode(const std::string& prompt, const std::string& currentContext, std::function<void(char, bool)> onToken, std::function<void(int)> onRemove) {
    // Formatowanie promptu do formatu "Instruct", którego uczy się model
    std::string formattedPrompt = "[USER]: " + prompt + "\n[AI]:\n";
    
    // Generowanie kodu za pomocą naszego własnego silnika Transformera
    // Zwiększamy limit tokenów do 4096, by model mógł wypisać dłuższą odpowiedź HTML/Vue
    std::string generatedCode = engine.generate(formattedPrompt, 4096, onToken, onRemove);
    
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
    this->lastModificationTime = std::chrono::steady_clock::now();
    this->isDebounceThreadRunning = false;
    
    int which = -1;
    ::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0, (LPARAM)&which);
    if (which != -1) {
        this->trackedHwnd = (which == 0) ? nppData._scintillaMainHandle : nppData._scintillaSecondHandle;
    }
}

void AIManager::onEditorModified(HWND hwnd, int position, int linesAdded) {
    if (!isTracking || hwnd != trackedHwnd) return;

    // Przesuń śledzone linie, jeśli dodano/usunięto tekst przed lub w trakcie naszego bloku
    int modLine = (int)::SendMessage(hwnd, SCI_LINEFROMPOSITION, position, 0);
    
    if (linesAdded != 0) {
        if (modLine < trackedStartLine) {
            trackedStartLine += linesAdded;
            trackedEndLine += linesAdded;
        } else if (modLine >= trackedStartLine && modLine <= trackedEndLine) {
            trackedEndLine += linesAdded;
        }
    }

    lastModificationTime = std::chrono::steady_clock::now();

    if (!isDebounceThreadRunning) {
        isDebounceThreadRunning = true;
        std::thread([this, hwnd]() {
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                if (!isTracking) {
                    isDebounceThreadRunning = false;
                    break;
                }

                auto now = std::chrono::steady_clock::now();
                if (std::chrono::duration_cast<std::chrono::seconds>(now - lastModificationTime).count() >= 5) {
                    // Minęło 5 sekund bez edycji, wyciągamy kod i zapisujemy do bazy wiedzy!
                    
                    int startPos = (int)::SendMessage(hwnd, SCI_POSITIONFROMLINE, trackedStartLine, 0);
                    int endPos = (int)::SendMessage(hwnd, SCI_GETLINEENDPOSITION, trackedEndLine, 0);
                    
                    if (startPos >= 0 && endPos > startPos) {
                        int length = endPos - startPos;
                        std::string modifiedCode(length + 1, '\0');
                        
                        // SCI_GETTEXTRANGEFULL wymaga struktury Sci_TextRangeFull
                        Sci_TextRangeFull tr;
                        tr.chrg.cpMin = startPos;
                        tr.chrg.cpMax = endPos;
                        tr.lpstrText = modifiedCode.data();
                        
                        ::SendMessage(hwnd, SCI_GETTEXTRANGEFULL, 0, (LPARAM)&tr);
                        modifiedCode.resize(length); // Pozbycie się zbędnych znaków
                        
                        // Zapis do pliku instruct_dataset.txt
                        std::string datasetPath = getPluginDllDirectory() + "\\datasets\\instruct_dataset.txt";
                        if (!PathFileExistsA(datasetPath.c_str())) {
                            datasetPath = getPluginDllDirectory() + "\\..\\..\\datasets\\instruct_dataset.txt";
                        }
                        
                        // Wyciągnięcie bloku <THINK> z oryginalnie wygenerowanego kodu
                        std::string thinkBlock = "";
                        size_t thinkStart = lastGeneratedCode.find("<THINK>");
                        size_t thinkEnd = lastGeneratedCode.find("</THINK>");
                        if (thinkStart != std::string::npos && thinkEnd != std::string::npos) {
                            thinkBlock = lastGeneratedCode.substr(thinkStart, thinkEnd - thinkStart + 8) + "\n";
                        }
                        
                        std::ofstream datasetFile(datasetPath, std::ios::app);
                        if (datasetFile.is_open()) {
                            datasetFile << "\n[USER]: " << lastPrompt << "\n[AI]:\n" << thinkBlock << modifiedCode << "\n";
                            datasetFile.close();
                        }
                        
                        // Wyślij do telemetrii
                        TelemetryManager::getInstance().queueLearningData(lastPrompt, lastGeneratedCode, modifiedCode);
                    }
                    
                    isTracking = false;
                    isDebounceThreadRunning = false;
                    break;
                }
            }
        }).detach();
    }
}

void AIManager::checkModifications(const std::string& currentTextInEditor) {
    // Zachowujemy tę metodę na wypadek zapisu pliku (fallback)
    if (!isTracking) return;
    
    // Ręczne wyzwolenie zapisu, ustawienie timera w przeszłości, by wymusić zapis
    lastModificationTime = std::chrono::steady_clock::now() - std::chrono::seconds(10);
}
