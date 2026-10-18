// this file is part of notepad++
// Copyright (C)2022 Don HO <don.h@free.fr>
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU General Public License
// as published by the Free Software Foundation; either
// version 2 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software
// Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

#include "PluginDefinition.h"
#include "AIManager.h"
#include "GenerationContext.h"
#include "ValidatedGeneration.h"
#include "PythonValidationProcess.h"
#include <algorithm>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include "GenerationStreamRouter.h"
#include "AIPromptPipeline.h"
#include "GenerationTracking.h"
#include "PluginPromptActions.h"
#include "DockingFeature/Docking.h"
#include "Notepad_plus_msgs.h"
#include "RAGManager.h"
#include "RepositoryContext.h"
#include "TelemetryManager.h"
#include "menuCmdID.h"
#include <atomic>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern NppData nppData;

// Okienko wprowadzania promptu (Dockable Panel)
HWND g_hAIPanel = NULL;
HWND g_hEdit = NULL;
HWND g_hBtn = NULL;
HWND g_hHistory = NULL;     // Nowe pole do wyswietlania myslenia
HWND g_hStatusLabel = NULL; // Dodane: etykieta postępu trenowania
bool isPanelRegistered = false;
std::atomic<bool> isGenerating{false};

// Funkcja pomocnicza do konwersji UTF-8 na UTF-16
std::wstring Utf8ToUtf16(const std::string &utf8) {
  if (utf8.empty())
    return std::wstring();
  int size_needed =
      MultiByteToWideChar(CP_UTF8, 0, &utf8[0], (int)utf8.size(), NULL, 0);
  std::wstring utf16(size_needed, 0);
  MultiByteToWideChar(CP_UTF8, 0, &utf8[0], (int)utf8.size(), &utf16[0],
                      size_needed);
  return utf16;
}

bool IsPolishLanguage() {
  // Unlike most Notepad++ text messages, this API explicitly returns char*,
  // independent of UNICODE. Reading it as TCHAR corrupts language detection.
  const auto required = ::SendMessageW(nppData._nppHandle, NPPM_GETNATIVELANGFILENAME, 0, 0);
  if (required <= 0 || required > 32767) return false;
  std::string name(static_cast<std::size_t>(required) + 1, '\0');
  const auto copied = ::SendMessageW(nppData._nppHandle, NPPM_GETNATIVELANGFILENAME,
      name.size(), reinterpret_cast<LPARAM>(&name[0]));
  if (copied <= 0 || static_cast<std::size_t>(copied) >= name.size()) return false;
  name.resize(static_cast<std::size_t>(copied));
  std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return char(std::tolower(c)); });
  const auto slash = name.find_last_of("/\\");
  return name.substr(slash == std::string::npos ? 0 : slash+1) == "polish.xml";
}

std::wstring Loc(const std::wstring &pl, const std::wstring &en) {
  return IsPolishLanguage() ? pl : en;
}

std::string Loc(const std::string &pl, const std::string &en) {
  return IsPolishLanguage() ? pl : en;
}

namespace {
std::string panelPromptUtf8() {
  wchar_t buffer[4096]{};
  const int length = GetWindowTextW(g_hEdit, buffer, 4096);
  if (!length) return "";
  const int bytes = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, buffer, length, nullptr, 0, nullptr, nullptr);
  if (!bytes) throw std::runtime_error("Invalid Unicode prompt");
  std::string result(bytes, '\0');
  if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, buffer, length, &result[0], bytes, nullptr, nullptr))
    throw std::runtime_error("Invalid Unicode prompt");
  return result;
}

std::wstring generationErrorText(const std::string& error) {
  if (error.find("Online model is not configured") == 0 ||
      error.find("Local model unavailable; no permitted online backend") == 0)
    return Loc(L"Brak modelu NppAI lub konfiguracji serwera. Uruchom scripts/start_nppai.py z --model i --notepad. Szczegóły: docs/notepad-setup-pl.md.",
               L"NppAI model/server is not configured. Run scripts/start_nppai.py with --model and --notepad. See docs/notepad-setup-pl.md.");
  if (error.find("Online model connection failed") == 0)
    return Loc(L"Nie można połączyć się z serwerem NppAI lub upłynął limit czasu. Sprawdź, czy serwer działa, oraz NPPAI_GENERATION_URL. Uruchom Notepad++ ponownie przez scripts/start_nppai.py.",
               L"Cannot reach NppAI or the request timed out. Check the server and NPPAI_GENERATION_URL. Restart Notepad++ through scripts/start_nppai.py.");
  if (error.find("Online model returned HTTP 401") == 0 || error.find("Online model returned HTTP 403") == 0)
    return Loc(L"Serwer NppAI odrzucił klucz. NPPAI_API_KEY musi być taki sam dla serwera i Notepad++. Uruchom oba ponownie przez scripts/start_nppai.py.",
               L"NppAI rejected authentication. Server and Notepad++ need the same NPPAI_API_KEY. Restart both through scripts/start_nppai.py.");
  if (error.find("Online model returned HTTP 503") == 0)
    return Loc(L"Serwer NppAI działa, ale model jest niedostępny lub zajęty. Sprawdź NPPAI_INFERENCE_MODEL i okno serwera; model musi być zgodnym plikiem .nppai.",
               L"NppAI is reachable, but the model is unavailable or busy. Check NPPAI_INFERENCE_MODEL and the server console; a compatible .nppai file is required.");
  if (error.find("Online model returned HTTP 422") == 0)
    return Loc(L"Zapytanie lub kontekst przekracza możliwości modelu NppAI. Skróć kontekst albo użyj zgodnego modelu z większym oknem kontekstu.",
               L"The request/context exceeds NppAI model capacity. Reduce context or use a compatible model with a larger context window.");
  if (error.find("Configure Python validation:") == 0)
    return Loc(L"Odpowiedź NppAI jest w podglądzie. Aby włączyć Zastosuj, skonfiguruj walidację: NPPAI_VALIDATION_PYTHON, NPPAI_VALIDATION_IMAGE i NPPAI_TEST_CASES. Instrukcja: docs/notepad-setup-pl.md.",
               L"NppAI response is in the preview. To enable Apply, configure NPPAI_VALIDATION_PYTHON, NPPAI_VALIDATION_IMAGE and NPPAI_TEST_CASES. See docs/notepad-setup-pl.md.");
  if (error == "Validated generation currently requires a UTF-8 Python document")
    return Loc(L"Zapisz plik Python (.py) w kodowaniu UTF-8 i spróbuj ponownie.", L"Save a Python (.py) file as UTF-8 and try again.");
  if (error == "This validation profile supports Python (.py or @python-signature)")
    return Loc(L"Zapisz dokument jako plik .py lub użyj @python-signature w pierwszej linii zapytania.",
               L"Save the document as .py or use @python-signature on the first prompt line.");
  return Utf8ToUtf16(error);
}
constexpr UINT ValidationStatusMessage = WM_APP + 61;
constexpr UINT ValidationReadyMessage = WM_APP + 62;
struct PendingGeneration {
  ValidatedGenerationRequest request;
  ValidatedGenerationResult result;
  HWND editor = nullptr;
  LPARAM document = 0;
  UINT_PTR buffer = 0;
  unsigned int epoch = 0;
};
HWND g_hApply = nullptr, g_hReject = nullptr;
std::atomic<unsigned int> generationEpoch{0};
std::mutex generationMailbox;
std::unique_ptr<PendingGeneration> completedGeneration, pendingGeneration;
std::string latestGenerationStatus;
unsigned int latestStatusEpoch = 0;

void postGenerationStatus(HWND panel, unsigned int epoch, const std::string& status) {
  std::lock_guard<std::mutex> lock(generationMailbox);
  if (generationEpoch != epoch) return;
  latestGenerationStatus = status; latestStatusEpoch = epoch;
  PostMessageW(panel, ValidationStatusMessage, 0, 0);
}
std::string editorText(HWND editor) {
  const auto length = SendMessage(editor, SCI_GETLENGTH, 0, 0);
  if (length < 0 || length > 131072) throw std::runtime_error("Review supports documents up to 128 KiB");
  std::vector<char> bytes(static_cast<std::size_t>(length) + 1);
  SendMessage(editor, SCI_GETTEXT, bytes.size(), reinterpret_cast<LPARAM>(bytes.data()));
  return std::string(bytes.data(), static_cast<std::size_t>(length));
}
std::string currentFilePath() {
  wchar_t path[MAX_PATH]{};
  SendMessage(nppData._nppHandle, NPPM_GETFULLCURRENTPATH, MAX_PATH, reinterpret_cast<LPARAM>(path));
  int size = WideCharToMultiByte(CP_UTF8, 0, path, -1, nullptr, 0, nullptr, nullptr);
  if (size <= 1) return {};
  std::string result(size, '\0');
  WideCharToMultiByte(CP_UTF8, 0, path, -1, &result[0], size, nullptr, nullptr);
  result.resize(size - 1); return result;
}
void applyPendingGeneration() {
  if (!pendingGeneration || isGenerating || !pendingGeneration->result.ready()) return;
  try {
    auto& pending = *pendingGeneration;
    int view = -1;
    SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0, reinterpret_cast<LPARAM>(&view));
    HWND editor = view == 0 ? nppData._scintillaMainHandle : view == 1 ? nppData._scintillaSecondHandle : nullptr;
    if (editor != pending.editor ||
        static_cast<UINT_PTR>(SendMessage(nppData._nppHandle, NPPM_GETCURRENTBUFFERID, 0, 0)) != pending.buffer ||
        SendMessage(editor, SCI_GETDOCPOINTER, 0, 0) != pending.document ||
        SendMessage(editor, SCI_GETREADONLY, 0, 0))
      throw std::runtime_error("Activate the original writable document before applying this proposal");
    PatchReview review(std::filesystem::current_path(), pending.result.patch);
    review.accept();
    const auto check = review.checkBuffer(editorText(editor));
    if (!check.applied) { EnableWindow(g_hApply, FALSE); throw std::runtime_error(check.error); }
    EnableWindow(g_hApply, FALSE);
    const auto& patch = pending.result.patch;
    SendMessage(editor, SCI_BEGINUNDOACTION, 0, 0);
    SendMessage(editor, SCI_SETTARGETRANGE, 0, patch.originalText.size());
    const auto inserted = SendMessage(editor, SCI_REPLACETARGET, patch.proposedText.size(),
                                      reinterpret_cast<LPARAM>(patch.proposedText.data()));
    SendMessage(editor, SCI_ENDUNDOACTION, 0, 0);
    if (inserted != static_cast<LRESULT>(patch.proposedText.size()))
      throw std::runtime_error("Editor could not apply the accepted proposal; use Undo to inspect the change");
    const auto replacementSize = patch.proposedText.size() -
        (patch.originalText.size() - (pending.request.selectionEnd - pending.request.selectionStart));
    const auto startLine = SendMessage(editor, SCI_LINEFROMPOSITION, pending.request.selectionStart, 0);
    const auto endLine = SendMessage(editor, SCI_LINEFROMPOSITION, pending.request.selectionStart + replacementSize, 0);
    if (!pending.request.path.empty())
      AIManager::getInstance().startTracking(pending.request.task,
          patch.proposedText.substr(pending.request.selectionStart, replacementSize),
          int(startLine), int(endLine), pending.request.path);
    SetWindowTextW(g_hStatusLabel, Loc(L"Zastosowano. Możesz cofnąć zmianę.", L"Applied. The change can be undone.").c_str());
    pendingGeneration.reset(); EnableWindow(g_hApply, FALSE); EnableWindow(g_hReject, FALSE);
  } catch (const std::exception& error) {
    SetWindowTextW(g_hStatusLabel, generationErrorText(error.what()).c_str());
  }
}
}

void ExecuteAIGeneration() {
  if (!g_hEdit) return;
  if (isGenerating) {
    AIManager::getInstance().stopGeneration();
    SetWindowTextW(g_hBtn, Loc(L"Zatrzymywanie...", L"Stopping...").c_str());
    return;
  }
  try {
    const auto task = panelPromptUtf8();
    if (task.empty()) return;
    pendingGeneration.reset(); EnableWindow(g_hApply, FALSE); EnableWindow(g_hReject, FALSE);
    const auto parsed = AIPromptPipeline::prepare(task, "");
    if (!parsed.error.empty()) throw std::runtime_error(parsed.error);
    int view = -1;
    SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0, reinterpret_cast<LPARAM>(&view));
    if (view != 0 && view != 1) throw std::runtime_error("No active editor");
    const HWND editor = view == 0 ? nppData._scintillaMainHandle : nppData._scintillaSecondHandle;
    if (SendMessage(editor, SCI_GETCODEPAGE, 0, 0) != SC_CP_UTF8)
      throw std::runtime_error("Validated generation currently requires a UTF-8 Python document");
    auto pending = std::make_unique<PendingGeneration>();
    pending->editor = editor;
    pending->document = SendMessage(editor, SCI_GETDOCPOINTER, 0, 0);
    pending->buffer = static_cast<UINT_PTR>(SendMessage(nppData._nppHandle, NPPM_GETCURRENTBUFFERID, 0, 0));
    pending->request.task = task;
    pending->request.path = currentFilePath();
    std::string extension = std::filesystem::u8path(pending->request.path).extension().u8string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return char(std::tolower(c)); });
    if (extension != ".py" && parsed.assistantPrefix.empty())
      throw std::runtime_error("This validation profile supports Python (.py or @python-signature)");
    pending->request.original = editorText(editor);
    pending->request.selectionStart = static_cast<std::size_t>(SendMessage(editor, SCI_GETSELECTIONSTART, 0, 0));
    pending->request.selectionEnd = static_cast<std::size_t>(SendMessage(editor, SCI_GETSELECTIONEND, 0, 0));
    if (pending->request.selectionStart > pending->request.selectionEnd || pending->request.selectionEnd > pending->request.original.size())
      throw std::runtime_error("Selection changed while capturing the document");
    pending->request.allowOnlineEscalation = AIManager::getInstance().mayEscalateOnline();
    const auto validatorScript = std::filesystem::path(getPluginDllDirectory()) / "sandbox_validate.py";
    pending->epoch = ++generationEpoch;
    const auto epoch = pending->epoch;
    const HWND panel = g_hAIPanel;
    pendingGeneration.reset(); EnableWindow(g_hApply, FALSE); EnableWindow(g_hReject, FALSE);
    isGenerating = true; AIManager::getInstance().beginGeneration();
    SetWindowTextW(g_hBtn, Loc(L"Zatrzymaj", L"Stop").c_str());
    SetWindowTextW(g_hHistory, Loc(L"Generowanie propozycji; dokument pozostaje bez zmian.",
                                 L"Generating a proposal; the document remains unchanged.").c_str());
    std::thread([pending = std::move(pending), validatorScript, panel, epoch]() mutable {
      try {
        std::shared_ptr<PythonValidationProcess> validator;
        if (!pending->request.path.empty()) {
          RAGManager::getInstance().loadDatabase(pending->request.path + ".rag_db");
          pending->request.context = RepositoryContext::limit(RAGManager::getInstance().retrieveContext(
              RepositoryContext::buildQuery(pending->request.task, pending->request.path), 5));
        }
        const auto start = pending->request.selectionStart;
        const auto count = pending->request.selectionEnd - start;
        if (!pending->request.original.empty())
          pending->request.context += "\n[Captured Python document excerpt around the original selection]\n" +
              pending->request.original.substr(start > 2048 ? start - 2048 : 0, (std::min)(count + 4096, std::size_t(12288)));
        GenerationStreamRouter progress(nullptr, nullptr, nullptr,
            [panel, epoch](const GenerationProgressEvent& event) { postGenerationStatus(panel, epoch, event.message); });
        auto cancelled = [epoch] { return generationEpoch != epoch || AIManager::getInstance().isGenerationCancelled(); };
        pending->result = ValidatedGeneration::run(pending->request,
            [panel, epoch](const std::string& task, const std::string& context, bool forceOnline) {
              bool online = false; std::string reason;
              const auto code = AIManager::getInstance().generateCode(task, context, nullptr, nullptr,
                  [panel, epoch, &online, &reason](const std::string& status) {
                    reason = status; online = status.find("Online:") == 0; postGenerationStatus(panel, epoch, status);
                  }, forceOnline);
              return GeneratedCandidate{code, online, reason};
            }, [&validator, validatorScript, cancelled](const std::string& code, bool tests) {
              // Keep a generated proposal visible even when validation is not
              // configured. Validation failure still keeps Apply disabled.
              if (!validator) validator = std::make_shared<PythonValidationProcess>(validatorScript);
              return validator->run(code, tests, cancelled);
            },
            cancelled, &progress);
      } catch (const std::exception& error) {
        pending->result.blocked = true; pending->result.error = error.what();
      }
      std::lock_guard<std::mutex> lock(generationMailbox);
      if (generationEpoch != epoch) { isGenerating = false; return; }
      completedGeneration = std::move(pending);
      if (!PostMessageW(panel, ValidationReadyMessage, 0, 0)) { completedGeneration.reset(); isGenerating = false; }
    }).detach();
  } catch (const std::exception& error) {
    isGenerating = false;
    SetWindowTextW(g_hBtn, Loc(L"Wygeneruj Kod AI", L"Generate AI Code").c_str());
    SetWindowTextW(g_hStatusLabel, generationErrorText(error.what()).c_str());
    SetWindowTextW(g_hHistory, generationErrorText(error.what()).c_str());
  }
}

#ifdef NPPAI_TESTING
extern "C" __declspec(dllexport) const char* panelPromptForTesting() {
  static thread_local std::string result;
  result = panelPromptUtf8();
  return result.c_str();
}
extern "C" __declspec(dllexport) void stageBufferReviewForTesting(HWND editor, const char* replacement) {
  pendingGeneration = std::make_unique<PendingGeneration>();
  auto& pending = *pendingGeneration;
  pending.editor = editor;
  pending.document = SendMessage(editor, SCI_GETDOCPOINTER, 0, 0);
  pending.buffer = static_cast<UINT_PTR>(SendMessage(nppData._nppHandle, NPPM_GETCURRENTBUFFERID, 0, 0));
  pending.request.original = editorText(editor);
  pending.request.selectionEnd = pending.request.original.size();
  pending.result.patch = StructuredPatchBuilder::build("fixture.py", pending.request.original, replacement);
  pending.result.repair.succeeded = true;
}
extern "C" __declspec(dllexport) void applyBufferReviewForTesting() { applyPendingGeneration(); }
extern "C" __declspec(dllexport) void rejectBufferReviewForTesting() { pendingGeneration.reset(); }
#endif

LRESULT CALLBACK AIPanelProc(HWND hwnd, UINT msg, WPARAM wParam,
                             LPARAM lParam) {
  switch (msg) {
  case ValidationStatusMessage: {
    std::lock_guard<std::mutex> lock(generationMailbox);
    if (latestStatusEpoch == generationEpoch && isGenerating)
      SetWindowTextW(g_hStatusLabel, Utf8ToUtf16(latestGenerationStatus).c_str());
    return 0;
  }
  case ValidationReadyMessage: {
    {
      std::lock_guard<std::mutex> lock(generationMailbox);
      if (!completedGeneration || completedGeneration->epoch != generationEpoch) return 0;
      pendingGeneration = std::move(completedGeneration);
    }
    isGenerating = false;
    SetWindowTextW(g_hBtn, Loc(L"Wygeneruj Kod AI", L"Generate AI Code").c_str());
    const auto& result = pendingGeneration->result;
    std::string report = result.ready() ? "Validation passed. Review the diff before Apply.\r\n" :
        result.cancelled ? "Cancelled. Original document unchanged.\r\n" : "Validation failed/blocked. Apply disabled.\r\n";
    report += "Candidates validated: " + std::to_string(result.repair.attempts) + "/3\r\n";
    for (std::size_t i = 0; i < result.candidateBackends.size(); ++i)
      report += "Candidate " + std::to_string(i + 1) + ": " + result.candidateBackends[i] + "\r\n";
    if (!result.error.empty()) report += result.error + "\r\n";
    for (const auto& attempt : result.repair.history)
      report += "Attempt " + std::to_string(attempt.attempt) + ": " + attempt.diagnostics + "\r\n";
    report += result.repair.validation.build.diagnostics + "\r\n" + result.repair.validation.tests.diagnostics + "\r\n";
    report += result.patch.unifiedDiff();
    SetWindowTextW(g_hHistory, Utf8ToUtf16(report).c_str());
    SetWindowTextW(g_hStatusLabel, Loc(result.ready() ? L"Sprawdź różnice i zastosuj lub odrzuć." : L"Walidacja niepotwierdzona. Dokument bez zmian.",
                                      result.ready() ? L"Review, then apply or reject." : L"Validation not confirmed. Document unchanged.").c_str());
    if (!result.error.empty()) {
      const auto message = generationErrorText(result.error);
      SetWindowTextW(g_hStatusLabel, message.c_str());
      const auto detailed = message + L"\r\n\r\n" + Utf8ToUtf16(report);
      SetWindowTextW(g_hHistory, detailed.c_str());
    }
    EnableWindow(g_hApply, result.ready()); EnableWindow(g_hReject, TRUE);
    return 0;
  }
  case WM_CREATE: {
    // Pole jawnej odpowiedzi AI na samej górze
    g_hHistory = CreateWindowExW(
        WS_EX_CLIENTEDGE, L"EDIT",
        Loc(L"Panel odpowiedzi AI gotowy.\r\nSerwer własnego modelu: scripts/start_nppai.py --model PLIK.nppai --notepad notepad++.exe. Instrukcja: docs/notepad-setup-pl.md.\r\nTryb funkcji Python: wpisz @python-signature nazwa(argumenty) w pierwszej linii, a opis w kolejnej.\r\n",
            L"AI response panel ready.\r\nOwn-model server: scripts/start_nppai.py --model FILE.nppai --notepad notepad++.exe. Guide: docs/notepad-setup-pl.md.\r\nPython function mode: enter @python-signature name(arguments) on the first line, then the task description.\r\n")
            .c_str(),
        WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOVSCROLL | ES_MULTILINE |
            ES_READONLY,
        0, 0, 100, 50, hwnd, (HMENU)3, NULL, NULL);

    SendMessage(g_hHistory, EM_SETLIMITTEXT, 1024 * 1024, 0);

    // Główne pole zapytania użytkownika
    g_hEdit = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"",
                              WS_CHILD | WS_VISIBLE | WS_BORDER |
                                  ES_AUTOVSCROLL | ES_MULTILINE | ES_WANTRETURN,
                              0, 0, 100, 100, hwnd, (HMENU)1, NULL, NULL);

    SendMessageW(g_hEdit, EM_SETLIMITTEXT, 4095, 0);
    HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    SendMessage(g_hHistory, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));
    SendMessage(g_hEdit, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));

    g_hBtn = CreateWindowExW(
        0, L"BUTTON", Loc(L"Wygeneruj Kod AI", L"Generate AI Code").c_str(),
        WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 0, 0, 100, 30, hwnd, (HMENU)2,
        NULL, NULL);
    SendMessage(g_hBtn, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));

    g_hApply = CreateWindowExW(0, L"BUTTON", Loc(L"Zastosuj", L"Apply").c_str(), WS_CHILD | WS_VISIBLE,
        0, 0, 100, 30, hwnd, (HMENU)5, nullptr, nullptr);
    g_hReject = CreateWindowExW(0, L"BUTTON", Loc(L"Odrzuć", L"Reject").c_str(), WS_CHILD | WS_VISIBLE,
        0, 0, 100, 30, hwnd, (HMENU)6, nullptr, nullptr);
    SendMessage(g_hApply, WM_SETFONT, (WPARAM)hFont, TRUE);
    SendMessage(g_hReject, WM_SETFONT, (WPARAM)hFont, TRUE);
    EnableWindow(g_hApply, FALSE); EnableWindow(g_hReject, FALSE);

    // Etykieta statusu trenowania
    g_hStatusLabel = CreateWindowExW(
        0, L"STATIC",
        Loc(L"Podgląd i walidacja przed zastosowaniem", L"Review and validate before applying").c_str(),
        WS_CHILD | WS_VISIBLE, 0, 0, 100, 30, hwnd, (HMENU)4, NULL, NULL);
    SendMessage(g_hStatusLabel, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));

    // Timer do sprawdzania aktualizacji chmurowych (co 60 sekund)
    SetTimer(hwnd, 2, 60000, NULL);

    return 0;
  }
  case WM_TIMER: {
    if (wParam == 2) {
      // Nowy timer: Sprawdza aktualizacje z chmury co 60 sekund
      AIManager::getInstance().checkAndDownloadModelUpdate();
    }
    return 0;
  }
  case WM_SIZE: {
    int width = LOWORD(lParam);
    int height = HIWORD(lParam);
    int btnHeight = 30;
    int historyHeight = (std::max)(40, (height - 95) * 2 / 3); // Więcej miejsca na strumieniowaną odpowiedź

    // Okno odpowiedzi AI na samej górze
    MoveWindow(g_hHistory, 5, 5, width - 10, historyHeight, TRUE);
    // Pole tekstowe zajmuje przestrzeń pod oknem myślenia
    MoveWindow(g_hEdit, 5, historyHeight + 10, width - 10,
               (std::max)(30, height - historyHeight - btnHeight * 2 - 25), TRUE);
    // Etykieta statusu po lewej na dole
    MoveWindow(g_hStatusLabel, 5, height - btnHeight * 2 - 10, (std::max)(20, width - 140),
               btnHeight, TRUE);
    // Przycisk po prawej na dole
    MoveWindow(g_hBtn, width - 130, height - btnHeight * 2 - 10, 120, btnHeight,
               TRUE);
    MoveWindow(g_hApply, 5, height - btnHeight - 5, (width - 15) / 2, btnHeight, TRUE);
    MoveWindow(g_hReject, 10 + (width - 15) / 2, height - btnHeight - 5, (width - 15) / 2, btnHeight, TRUE);
    return 0;
  }
  case WM_COMMAND: {
    if (LOWORD(wParam) == 5) applyPendingGeneration();
    if (LOWORD(wParam) == 6 && !isGenerating) {
      pendingGeneration.reset(); EnableWindow(g_hApply, FALSE); EnableWindow(g_hReject, FALSE);
      SetWindowTextW(g_hHistory, Loc(L"Odrzucono. Dokument bez zmian.", L"Rejected. Document unchanged.").c_str());
    }
    if (LOWORD(wParam) == 2) {
      ExecuteAIGeneration();
    }
    return 0;
  }
  case WM_DESTROY: {
    AIManager::getInstance().stopGeneration(); ++generationEpoch;
    { std::lock_guard<std::mutex> lock(generationMailbox);
      if (completedGeneration) isGenerating = false;
      completedGeneration.reset();
    }
    pendingGeneration.reset();
    KillTimer(hwnd, 2);
    g_hAIPanel = NULL;
    return 0;
  }
  }
  return DefWindowProcW(hwnd, msg, wParam, lParam);
}

void InitAIPanel() {
  if (g_hAIPanel != NULL)
    return;

  WNDCLASSW wc = {0};
  wc.lpfnWndProc = AIPanelProc;
  wc.hInstance = GetModuleHandle(NULL);
  wc.lpszClassName = L"NppAI_PanelClass";
  wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
  RegisterClassW(&wc);

  g_hAIPanel =
      CreateWindowExW(0, L"NppAI_PanelClass", L"NppAI - Chat",
                      WS_CHILD | WS_CLIPCHILDREN | WS_CLIPSIBLINGS, 0, 0, 400,
                      200, nppData._nppHandle, NULL, wc.hInstance, NULL);

  DockedWidgetData tbData = {0};
  tbData.hClient = g_hAIPanel;
  tbData.pszName = L"NppAI - Prompt";
  tbData.dlgID = 0;
  tbData.uMask = DWS_DF_CONT_BOTTOM | DWS_ICONTAB | DWS_ICONBAR;
  tbData.pszModuleName = L"NppAI";

  SendMessage(nppData._nppHandle, NPPM_DMMREGASDCKDLG, 0, (LPARAM)&tbData);
  isPanelRegistered = true;
}

//
// The plugin data that Notepad++ needs
//
FuncItem funcItem[nbFunc];

//
// The data of Notepad++ that you can use in your plugin commands
//
NppData nppData;

//
// Initialize your plugin data here
// It will be called while plugin loading
void pluginInit(HANDLE /*hModule*/) {}

//
// Here you can do the clean up, save the parameters (if any) for the next
// session
//
void pluginCleanUp() {}

//
// Initialization of your plugin commands
// You should fill your plugins commands here
void commandMenuInit() {

  //--------------------------------------------//
  //-- STEP 3. CUSTOMIZE YOUR PLUGIN COMMANDS --//
  //--------------------------------------------//
  // with function :
  // setCommand(int index,                      // zero based number to indicate
  // the order of command
  //            const wchar_t *commandName,     // the command name that you
  //            want to see in plugin menu PFUNCPLUGINCMD functionPointer, //
  //            the symbol of function (function pointer) associated with this
  //            command. The body should be defined below. See Step 4.
  //            ShortcutKey *shortcut,          // optional. Define a shortcut
  //            to trigger this command bool check0nInit                //
  //            optional. Make this menu item be checked visually
  //            );
  setCommand(0, L"Pokaż Panel AI", toggleAIPanel, NULL, false);
  setCommand(1, L"Zapytaj AI o zaznaczony kod", sendSelectionToChat, NULL,
             false);
  setCommand(2, L"Zmień status telemetrii", toggleTelemetry, NULL, false);
}

//
// Here you can do the clean up (especially for the shortcut)
//
void commandMenuCleanUp() {
  // Don't forget to deallocate your shortcut here
}

//
// This function help you to initialize your plugin commands
//
bool setCommand(size_t index, const wchar_t *cmdName, PFUNCPLUGINCMD pFunc,
                ShortcutKey *sk, bool check0nInit) {
  if (index >= nbFunc)
    return false;

  if (!pFunc)
    return false;

  lstrcpyW(funcItem[index]._itemName, cmdName);
  funcItem[index]._pFunc = pFunc;
  funcItem[index]._init2Check = check0nInit;
  funcItem[index]._pShKey = sk;

  return true;
}

void UpdateLocalization() {
  HMENU hPluginMenu = (HMENU)::SendMessage(
      nppData._nppHandle, NPPM_GETMENUHANDLE, 0, 0); // 0 is NPPPLUGINMENU
  if (hPluginMenu) {
    std::wstring cmd0 = Loc(L"Pokaż Panel AI", L"Show AI Panel");
    std::wstring cmd1 =
        Loc(L"Zapytaj AI o zaznaczony kod", L"Ask AI about selected code");
    std::wstring cmd2 =
        Loc(L"Zmień status telemetrii", L"Toggle telemetry status");

    lstrcpyW(funcItem[0]._itemName, cmd0.c_str());
    lstrcpyW(funcItem[1]._itemName, cmd1.c_str());
    lstrcpyW(funcItem[2]._itemName, cmd2.c_str());

    MENUITEMINFOW mii = {sizeof(MENUITEMINFOW)};
    mii.fMask = MIIM_STRING;

    mii.dwTypeData = const_cast<LPWSTR>(cmd0.c_str());
    SetMenuItemInfoW(hPluginMenu, funcItem[0]._cmdID, FALSE, &mii);

    mii.dwTypeData = const_cast<LPWSTR>(cmd1.c_str());
    SetMenuItemInfoW(hPluginMenu, funcItem[1]._cmdID, FALSE, &mii);

    mii.dwTypeData = const_cast<LPWSTR>(cmd2.c_str());
    SetMenuItemInfoW(hPluginMenu, funcItem[2]._cmdID, FALSE, &mii);
  }

  if (g_hAIPanel) {
    if (g_hBtn) {
      if (isGenerating) {
        SetWindowTextW(g_hBtn, Loc(L"Zatrzymaj", L"Stop").c_str());
      } else {
        SetWindowTextW(g_hBtn,
                       Loc(L"Wygeneruj Kod AI", L"Generate AI Code").c_str());
      }
    }
    if (g_hStatusLabel) {
      SetWindowTextW(g_hStatusLabel, Loc(L"NppAI gotowe. Połączenie z serwerem jest sprawdzane przy generowaniu.",
                                         L"NppAI ready. Server connection is checked when generating.")
                                         .c_str());
    }
  }
}
void generateAICode() {
  // Nie jest już bezpośrednio wywoływana z menu. Zastąpiona przez
  // toggleAIPanel.
}

void toggleAIPanel() {
  if (!isPanelRegistered) {
    InitAIPanel();
  } else {
    // Jeśli już zarejestrowany, pokazujemy go
    ::SendMessage(nppData._nppHandle, NPPM_DMMSHOW, 0, (LPARAM)g_hAIPanel);
  }
}

std::string buildSelectionPrompt(HWND curScintilla) {
  auto selLen = ::SendMessage(curScintilla, SCI_GETSELTEXT, 0, 0);
  if (selLen <= 1)
    return {};

  std::vector<char> selText(selLen);
  ::SendMessage(curScintilla, SCI_GETSELTEXT, 0, (LPARAM)selText.data());
  std::string selection(selText.begin(), selText.end() - 1);
  return PluginPromptActions::appendSelection("", selection);
}

#ifdef NPPAI_TESTING
extern "C" __declspec(dllexport) const char *buildSelectionPromptForTesting(
    HWND curScintilla) {
  static std::string prompt;
  prompt = buildSelectionPrompt(curScintilla);
  return prompt.c_str();
}
#endif

void sendSelectionToChat() {
  // Pobranie uchwytu Scintilli
  int which = -1;
  ::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0,
                (LPARAM)&which);
  if (which == -1)
    return;
  HWND curScintilla = (which == 0) ? nppData._scintillaMainHandle
                                   : nppData._scintillaSecondHandle;

  // Pobranie zaznaczonego tekstu i przygotowanie go do pola promptu.
  std::string prompt = buildSelectionPrompt(curScintilla);
  if (prompt.empty())
    return;

  // Pokaż panel
  if (!isPanelRegistered) {
    InitAIPanel();
  } else {
    ::SendMessage(nppData._nppHandle, NPPM_DMMSHOW, 0, (LPARAM)g_hAIPanel);
  }

  // Dodaj zaznaczony kod do pola textarea
  const int textLen = GetWindowTextLengthW(g_hEdit);
  std::vector<wchar_t> buffer(static_cast<std::size_t>(textLen) + 1);
  GetWindowTextW(g_hEdit, buffer.data(), textLen + 1);
  std::wstring currentText(buffer.data());
  if (!currentText.empty()) currentText += L"\r\n";
  const auto newText = currentText + Utf8ToUtf16(prompt);
  SetWindowTextW(g_hEdit, newText.c_str());

  // Ustaw kursor na samym początku by użytkownik wpisał polecenie
  SendMessage(g_hEdit, EM_SETSEL, 0, 0);
  SetFocus(g_hEdit);
}

void toggleTelemetry() {
  ::MessageBoxW(NULL,
                Loc(L"Telemetria i uczenie globalne: AKTYWNE\nTwój kod będzie "
                    L"anonimizowany przed wysłaniem.",
                    L"Telemetry and global learning: ACTIVE\nYour code will be "
                    L"anonymized before sending.")
                    .c_str(),
                Loc(L"NppAI Ustawienia", L"NppAI Settings").c_str(), MB_OK);
}
