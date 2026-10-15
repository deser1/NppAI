#include "AIManager.h"
#include "AIPromptPipeline.h"
#include "AutomaticModelRouting.h"
#include "RemoteModelClient.h"
#include <algorithm>
#include <stdexcept>
#include "PluginDefinition.h"
#include "RAGManager.h"
#include "Scintilla.h"
#include "TelemetryManager.h"
#include <chrono>
#include <fstream>
#include <shlwapi.h>
#include <thread>
#include <windows.h>
#include <wininet.h>

#pragma comment(lib, "wininet.lib")

extern NppData nppData;

// Funkcja pomocnicza do bezpiecznego escape'owania znaków do JSON
std::string escapeJsonString(const std::string &input) {
  std::string output;
  for (char c : input) {
    if (c == '"')
      output += "\\\"";
    else if (c == '\\')
      output += "\\\\";
    else if (c == '\b')
      output += "\\b";
    else if (c == '\f')
      output += "\\f";
    else if (c == '\n')
      output += "\\n";
    else if (c == '\r')
      output += "\\r";
    else if (c == '\t')
      output += "\\t";
    else
      output += c;
  }
  return output;
}

// Funkcja komunikująca się z naszym backendem FastAPI
void sendToCloudBackend(const std::string &jsonPayload) {
  HINTERNET hSession = InternetOpenA(
      "NppAI Plugin", INTERNET_OPEN_TYPE_PRECONFIG, NULL, NULL, 0);
  if (hSession) {
    HINTERNET hConnect = InternetConnectA(hSession, "127.0.0.1", 8000, NULL,
                                          NULL, INTERNET_SERVICE_HTTP, 0, 1);
    if (hConnect) {
      HINTERNET hRequest = HttpOpenRequestA(
          hConnect, "POST", "/api/submit_knowledge", NULL, NULL, NULL, 0, 1);
      if (hRequest) {
        std::string headers = "Content-Type: application/json\r\n";
        HttpSendRequestA(hRequest, headers.c_str(), (DWORD)headers.length(),
                         (LPVOID)jsonPayload.c_str(),
                         (DWORD)jsonPayload.length());
        InternetCloseHandle(hRequest);
      }
      InternetCloseHandle(hConnect);
    }
    InternetCloseHandle(hSession);
  }
}

std::string getPluginDllDirectory() {
  char path[MAX_PATH];
  HMODULE hm = NULL;
  if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                             GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                         (LPCSTR)&getPluginDllDirectory, &hm)) {
    GetModuleFileNameA(hm, path, sizeof(path));
    PathRemoveFileSpecA(path);
    return std::string(path);
  }
  return "";
}

void AIManager::loadModel(const std::string &modelPath) {
  // Inicjalizacja naszego autorskiego silnika AI
  std::string fullPath = modelPath;
  if (modelPath.find(":") == std::string::npos) { // Jeśli to ścieżka względna
    fullPath = getPluginDllDirectory() + "\\" + modelPath;
  }
  std::lock_guard<std::mutex> lock(generationMutex);
  localReady = engine.loadModel(fullPath);
}

namespace {
std::string environment(const char* name) {
  DWORD size = GetEnvironmentVariableA(name, nullptr, 0);
  if (!size) return "";
  std::string value(size, '\0');
  DWORD copied = GetEnvironmentVariableA(name, &value[0], size);
  if (!copied || copied >= size) return "";
  value.resize(copied);
  return value;
}
bool hasCode(const std::string& text) {
  return text.find_first_not_of(" \t\n\r\v\f") != std::string::npos;
}
}

std::string AIManager::generateCode(const std::string &prompt,
                                    const std::string &currentContext,
                                    std::function<void(char, bool)> onToken,
                                    std::function<void(int)> onRemove,
                                    std::function<void(const std::string&)> onStatus) {
  std::lock_guard<std::mutex> lock(generationMutex);
  if (generationCancelled) throw std::runtime_error("Generation cancelled");
  const auto request = AIPromptPipeline::prepare(prompt, currentContext);
  if (!request.error.empty()) throw std::runtime_error(request.error);
  const auto mode = AutomaticModelRouting::parseMode(environment("NPPAI_PLUGIN_MODE"));
  const auto endpoint = environment("NPPAI_GENERATION_URL");
  const auto key = environment("NPPAI_API_KEY");
  const bool remote = environment("NPPAI_ALLOW_REMOTE") == "1" && !endpoint.empty() && !key.empty();
  const auto decision = AutomaticModelRouting::select(mode, localReady,
      request.inferencePrompt.size(), localReady ? engine.contextCapacity() : 0, remote);
  if (onStatus) onStatus(decision.reason);
  if (decision.backend == AutomaticModelRouting::Backend::Unavailable)
    throw std::runtime_error(decision.reason);
  const int budget = 512;
  auto online = [&]() {
    const auto json = AutomaticModelRouting::encodeRequest(request.task, currentContext,
                                                          request.assistantPrefix, budget);
    const auto result = RemoteModelClient::generate(endpoint, key, json, budget, generationCancelled);
    if (!request.assistantPrefix.empty() && result.compare(0, request.assistantPrefix.size(), request.assistantPrefix) != 0)
      throw std::runtime_error("Online model did not preserve requested function prefix");
    if (!hasCode(result.substr(request.assistantPrefix.size())))
      throw std::runtime_error("Online model returned an empty completion");
    if (generationCancelled) throw std::runtime_error("Generation cancelled");
    if (onToken) for (char c : result) {
      if (generationCancelled) throw std::runtime_error("Generation cancelled");
      onToken(c, false);
    }
    return result;
  };
  if (decision.backend == AutomaticModelRouting::Backend::Online) return online();
  // Hold only the prefix and initial whitespace. Once visible code starts,
  // preserve native streaming; never retry remotely after publishing code.
  std::string pending = request.assistantPrefix;
  bool published = false;
  const auto generated = engine.generate(request.inferencePrompt, budget,
      [&](char c, bool thought) {
        if (generationCancelled) return;
        if (published) { if (onToken) onToken(c, thought); }
        else if (!thought) {
          pending += c;
          if (hasCode(std::string(1, c))) {
            if (onToken) for (char byte : pending) onToken(byte, false);
            pending.clear(); published = true;
          }
        }
      }, [&](int count) {
        if (generationCancelled) return;
        if (published) { if (onRemove) onRemove(count); }
        else if (count > 0) pending.erase(pending.size() - (std::min)(pending.size(), std::size_t(count)));
      });
  if (generationCancelled) throw std::runtime_error("Generation cancelled");
  const auto output = AIPromptPipeline::extractResponse(generated, request.basePrompt);
  if (!hasCode(output.substr((std::min)(output.size(), request.assistantPrefix.size())))) {
    if (!published && mode == AutomaticModelRouting::Mode::Automatic && remote) {
      if (onStatus) onStatus("Online: local model returned an empty completion");
      return online();
    }
    throw std::runtime_error("Local model returned an empty completion");
  }
  return output;
}

void AIManager::startTracking(const std::string &prompt,
                              const std::string &generatedCode, int startLine,
                              int endLine, const std::string &filePath) {
  this->isTracking = true;
  this->lastPrompt = prompt;
  this->lastGeneratedCode = generatedCode;
  this->trackedFilePath = filePath;
  this->trackedStartLine = startLine;
  this->trackedEndLine = endLine;
  this->lastModificationTime = std::chrono::steady_clock::now();
  this->isDebounceThreadRunning = false;

  int which = -1;
  ::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0,
                (LPARAM)&which);
  if (which != -1) {
    this->trackedHwnd = (which == 0) ? nppData._scintillaMainHandle
                                     : nppData._scintillaSecondHandle;
  }
}

void AIManager::onEditorModified(HWND hwnd, int position, int linesAdded) {
  if (!isTracking || hwnd != trackedHwnd)
    return;

  // Przesuń śledzone linie, jeśli dodano/usunięto tekst przed lub w trakcie
  // naszego bloku
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
        if (std::chrono::duration_cast<std::chrono::seconds>(
                now - lastModificationTime)
                .count() >= 5) {
          // Minęło 5 sekund bez edycji, wyciągamy kod i zapisujemy do bazy
          // wiedzy!

          int startPos = (int)::SendMessage(hwnd, SCI_POSITIONFROMLINE,
                                            trackedStartLine, 0);
          int endPos = (int)::SendMessage(hwnd, SCI_GETLINEENDPOSITION,
                                          trackedEndLine, 0);

          if (startPos >= 0 && endPos > startPos) {
            int length = endPos - startPos;
            std::string modifiedCode(length + 1, '\0');

            // SCI_GETTEXTRANGEFULL wymaga struktury Sci_TextRangeFull
            Sci_TextRangeFull tr;
            tr.chrg.cpMin = startPos;
            tr.chrg.cpMax = endPos;
            tr.lpstrText = &modifiedCode[0];

            ::SendMessage(hwnd, SCI_GETTEXTRANGEFULL, 0, (LPARAM)&tr);
            modifiedCode.resize(length); // Pozbycie się zbędnych znaków

            // Wyciągnięcie bloku <THINK> z oryginalnie wygenerowanego kodu
            std::string thinkBlock = "";
            size_t thinkStart = lastGeneratedCode.find("<THINK>");
            size_t thinkEnd = lastGeneratedCode.find("</THINK>");
            if (thinkStart != std::string::npos &&
                thinkEnd != std::string::npos) {
              thinkBlock = lastGeneratedCode.substr(thinkStart,
                                                    thinkEnd - thinkStart + 8) +
                           "\n";
            }

            // Przygotuj JSON dla Cloud Backend (Federated Learning)
            std::string jsonPayload = "{";
            jsonPayload +=
                "\"prompt\":\"" + escapeJsonString(lastPrompt) + "\",";
            jsonPayload +=
                "\"thought_process\":\"" + escapeJsonString(thinkBlock) + "\",";
            jsonPayload +=
                "\"final_code\":\"" + escapeJsonString(modifiedCode) + "\",";
            jsonPayload += "\"user_id\":\"programmer_" +
                           std::to_string(GetCurrentProcessId()) + "\"";
            jsonPayload += "}";

            // Wątek wysyłający do chmury (aby nie blokować timera)
            std::thread([jsonPayload]() {
              sendToCloudBackend(jsonPayload);
            }).detach();

            // Zapis do lokalnej pamięci pliku (RAG)
            if (!trackedFilePath.empty()) {
              std::string memPath = trackedFilePath + ".nppai_mem";
              std::ofstream memFile(memPath, std::ios::app);
              if (memFile.is_open()) {
                memFile << "\n[USER]: " << lastPrompt << "\n[AI]:\n"
                        << modifiedCode << "\n";
                memFile.close();
              }

              // NOWE: Integracja z VectorDB dla inteligentnego wyszukiwania
              // (RAG)
              std::string docContent =
                  "[USER]: " + lastPrompt + "\n[AI]:\n" + modifiedCode;
              RAGManager::getInstance().addDocument(docContent);
              RAGManager::getInstance().saveDatabase(trackedFilePath +
                                                     ".rag_db");
            }

            // Wyślij do telemetrii
            TelemetryManager::getInstance().queueLearningData(
                lastPrompt, lastGeneratedCode, modifiedCode);
          }

          isTracking = false;
          isDebounceThreadRunning = false;
          break;
        }
      }
    }).detach();
  }
}

void AIManager::checkModifications() {
  // Zachowujemy tę metodę na wypadek zapisu pliku (fallback)
  if (!isTracking)
    return;

  // Ręczne wyzwolenie zapisu, ustawienie timera w przeszłości, by wymusić zapis
  lastModificationTime =
      std::chrono::steady_clock::now() - std::chrono::seconds(10);
}

void AIManager::checkAndDownloadModelUpdate() {
  std::thread([this]() {
    HINTERNET hSession = InternetOpenA(
        "NppAI Updater", INTERNET_OPEN_TYPE_PRECONFIG, NULL, NULL, 0);
    if (!hSession)
      return;

    HINTERNET hConnect = InternetConnectA(hSession, "127.0.0.1", 8000, NULL,
                                          NULL, INTERNET_SERVICE_HTTP, 0, 1);
    if (!hConnect) {
      InternetCloseHandle(hSession);
      return;
    }

    // Pobieramy czas modyfikacji obecnego pliku modelu
    std::string modelPath =
        getPluginDllDirectory() + "\\models\\NppAI-model-v1.nppai";
    std::string versionPath =
        getPluginDllDirectory() + "\\models\\NppAI-model-v1.version";
    std::string clientVersion = "0";

    std::ifstream verFile(versionPath);
    if (verFile.is_open()) {
      std::getline(verFile, clientVersion);
      verFile.close();
    }

    std::string url = "/api/check_model_update?client_version=" + clientVersion;
    HINTERNET hRequest =
        HttpOpenRequestA(hConnect, "GET", url.c_str(), NULL, NULL, NULL, 0, 1);
    if (hRequest) {
      if (HttpSendRequestA(hRequest, NULL, 0, NULL, 0)) {
        char buffer[1024];
        DWORD bytesRead;
        std::string response = "";
        while (InternetReadFile(hRequest, buffer, sizeof(buffer) - 1,
                                &bytesRead) &&
               bytesRead > 0) {
          buffer[bytesRead] = '\0';
          response += buffer;
        }

        // Proste parsowanie JSONa
        std::string versionStr = "";
        size_t verPos = response.find("\"version\":\"");
        if (verPos != std::string::npos) {
          verPos += 11;
          size_t verEnd = response.find("\"", verPos);
          if (verEnd != std::string::npos) {
            versionStr = response.substr(verPos, verEnd - verPos);
          }
        }

        if (response.find("\"update_available\":true") != std::string::npos ||
            response.find("\"update_available\": true") != std::string::npos) {
          // Jest aktualizacja! Pobieramy nowy plik modelu
          HINTERNET hDlRequest = HttpOpenRequestA(
              hConnect, "GET", "/api/download_model", NULL, NULL, NULL, 0, 1);
          if (hDlRequest) {
            if (HttpSendRequestA(hDlRequest, NULL, 0, NULL, 0)) {
              std::string newModelPath = modelPath + ".new";
              std::ofstream outFile(newModelPath, std::ios::binary);
              if (outFile.is_open()) {
                // Pobieramy bezpiecznie w pętli do bufora RAM, unikając
                // problemów z siecią WinINet
                char buffer[4096];
                DWORD bytesRead = 0;
                while (InternetReadFile(hDlRequest, buffer, sizeof(buffer),
                                        &bytesRead) &&
                       bytesRead > 0) {
                  outFile.write(buffer, bytesRead);
                }
                outFile.close();

                // Podmień stary model na nowy (wymaga przeładowania wtyczki by
                // zaczął działać)
                DeleteFileA(modelPath.c_str());
                MoveFileA(newModelPath.c_str(), modelPath.c_str());

                // Zapisz nową wersję
                if (!versionStr.empty()) {
                  std::ofstream vFile(versionPath);
                  vFile << versionStr;
                }

                // Zasygnalizuj silnikowi by przeładował plik wag
                loadModel(modelPath);
              }
            }
            InternetCloseHandle(hDlRequest);
          }
        }
      }
      InternetCloseHandle(hRequest);
    }
    InternetCloseHandle(hConnect);
    InternetCloseHandle(hSession);
  }).detach();
}
