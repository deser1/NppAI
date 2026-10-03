#include "TelemetryManager.h"
#include "TelemetryRedaction.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <thread>
#include <windows.h>
#include <winhttp.h>

#pragma comment(lib, "winhttp.lib")

std::string TelemetryManager::anonymizeCode(const std::string &inputCode) {
  return TelemetryRedaction::redact(inputCode);
}

// Funkcja pomocnicza do escape'owania znaków do JSON
static std::string escapeJsonString(const std::string &input) {
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

void TelemetryManager::queueLearningData(const std::string &prompt,
                                         const std::string &generatedCode,
                                         const std::string &userModifiedCode) {
  // Najpierw anonimizujemy dane
  std::string safePrompt = anonymizeCode(prompt);
  std::string safeGenerated = anonymizeCode(generatedCode);
  std::string safeModified = anonymizeCode(userModifiedCode);

  // Wysyłanie asynchroniczne via WinHTTP
  std::thread([safePrompt, safeModified]() {
    HINTERNET hSession =
        WinHttpOpen(L"NppAI Telemetry/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY,
                    WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
    if (!hSession)
      return;

    HINTERNET hConnect = WinHttpConnect(hSession, L"localhost", 8000, 0);
    if (hConnect) {
      HINTERNET hRequest = WinHttpOpenRequest(
          hConnect, L"POST", L"/api/submit_knowledge", NULL, WINHTTP_NO_REFERER,
          WINHTTP_DEFAULT_ACCEPT_TYPES, 0);
      if (hRequest) {
        // Budowanie JSONa
        std::string jsonBody =
            "{ \"prompt\": \"" + escapeJsonString(safePrompt) +
            "\", \"final_code\": \"" + escapeJsonString(safeModified) +
            "\", \"thought_process\": \"\", \"user_id\": \"nppai_user\" }";

        LPCWSTR additionalHeaders = L"Content-Type: application/json\r\n";
        DWORD headersLength = -1;

        BOOL bResults = WinHttpSendRequest(
            hRequest, additionalHeaders, headersLength,
            (LPVOID)jsonBody.c_str(), jsonBody.length(), jsonBody.length(), 0);

        if (bResults) {
          WinHttpReceiveResponse(hRequest, NULL);
        }
        WinHttpCloseHandle(hRequest);
      }
      WinHttpCloseHandle(hConnect);
    }
    WinHttpCloseHandle(hSession);
  }).detach();
}

void TelemetryManager::processQueueAsync() {
  // Pusta metoda - wysyłanie zaimplementowano per zapytanie w queueLearningData
  // z WinHTTP
}
