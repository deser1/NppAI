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
#include "DockingFeature/Docking.h"
#include "Notepad_plus_msgs.h"
#include "RAGManager.h"
#include "TelemetryManager.h"
#include "menuCmdID.h"
#include <atomic>
#include <cwctype>
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
  TCHAR langFile[MAX_PATH] = {0};
  ::SendMessage(nppData._nppHandle, NPPM_GETNATIVELANGFILENAME, MAX_PATH,
                (LPARAM)langFile);

  std::basic_string<TCHAR> path(langFile);
  for (auto &c : path) {
#ifdef UNICODE
    c = std::towlower(c);
#else
    c = std::tolower(c);
#endif
  }

  return path.find(TEXT("polish.xml")) != std::basic_string<TCHAR>::npos;
}

std::wstring Loc(const std::wstring &pl, const std::wstring &en) {
  return IsPolishLanguage() ? pl : en;
}

std::string Loc(const std::string &pl, const std::string &en) {
  return IsPolishLanguage() ? pl : en;
}

// Funkcja odpowiedzialna za wykonanie akcji generowania po wpisaniu tekstu
void ExecuteAIGeneration() {
  if (!g_hEdit)
    return;

  if (isGenerating) {
    // Jeśli generujemy, przerwijmy to!
    AIManager::getInstance().stopGeneration();
    SetWindowTextW(g_hBtn,
                   Loc(L"Wygeneruj Kod AI", L"Generate AI Code").c_str());
    isGenerating = false;
    return;
  }

  char buf[4096] = {0};
  GetWindowTextA(g_hEdit, buf, sizeof(buf));
  std::string prompt = buf;
  if (prompt.empty())
    return;

  // Wyczyść pole po pobraniu tekstu
  SetWindowTextW(g_hEdit, L"");

  // Wyczyść historię myślenia
  SetWindowTextW(g_hHistory, Loc(L"Czekam na odpowiedź...\r\n",
                                 L"Waiting for response...\r\n")
                                 .c_str());

  // Pobranie uchwytu Scintilli
  int which = -1;
  ::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0,
                (LPARAM)&which);
  if (which == -1)
    return;
  HWND curScintilla = (which == 0) ? nppData._scintillaMainHandle
                                   : nppData._scintillaSecondHandle;

  // Uruchomienie trackera
  int startLine = (int)::SendMessage(
      curScintilla, SCI_LINEFROMPOSITION,
      (WPARAM)::SendMessage(curScintilla, SCI_GETCURRENTPOS, 0, 0), 0);

  // Zmień przycisk na "Zatrzymaj"
  isGenerating = true;
  SetWindowTextW(g_hBtn, Loc(L"Zatrzymaj", L"Stop").c_str());

  // Pobranie ścieżki pliku (dla pamięci RAG)
  TCHAR currentPath[MAX_PATH] = {0};
  ::SendMessage(nppData._nppHandle, NPPM_GETFULLCURRENTPATH, MAX_PATH,
                (LPARAM)currentPath);
  std::string currentFilePath = "";
#ifdef UNICODE
  int len =
      WideCharToMultiByte(CP_UTF8, 0, currentPath, -1, NULL, 0, NULL, NULL);
  if (len > 0) {
    std::vector<char> buf(len);
    WideCharToMultiByte(CP_UTF8, 0, currentPath, -1, buf.data(), len, NULL,
                        NULL);
    currentFilePath = buf.data();
  }
#else
  currentFilePath = currentPath;
#endif

  // Odczytanie kontekstu pamięci z pliku .nppai_mem
  std::string currentContext = "";
  if (!currentFilePath.empty()) {
    // Nowy mechanizm RAG (Retrieval-Augmented Generation) oparty na lokalnej
    // wektorowej bazie
    RAGManager::getInstance().loadDatabase(currentFilePath + ".rag_db");
    currentContext = RAGManager::getInstance().retrieveContext(prompt, 3);

    // Fallback: jeśli wektorowa baza jest pusta, używamy starego mechanizmu
    // tekstowego
    if (currentContext.empty()) {
      std::string memPath = currentFilePath + ".nppai_mem";
      std::ifstream memFile(memPath);
      if (memFile.is_open()) {
        std::string line;
        while (std::getline(memFile, line)) {
          currentContext += line + "\n";
        }
        memFile.close();

        // Ograniczenie kontekstu, jeśli rozrósł się za bardzo
        if (currentContext.length() > 1000) {
          currentContext =
              currentContext.substr(currentContext.length() - 1000);
        }
      }
    }
  }

  // Wygenerowanie kodu w osobnym wątku, by nie blokować interfejsu (Notepad++
  // brak odpowiedzi)
  std::thread([prompt, currentContext, currentFilePath, curScintilla,
               startLine]() {
    // Zmienne do obsługi tagu <THINK>
    std::string think_buffer = "";
    bool is_thinking = false;

    // Wygenerowanie kodu strumieniowo
    std::string generated = AIManager::getInstance().generateCode(
        prompt, currentContext,
        [&is_thinking, &think_buffer, curScintilla](char c, bool isThought) {
          if (isThought) {
            // Jesteśmy w trakcie myślenia
            think_buffer += c;
            // Możemy aktualizować pole historii na żywo (wymaga konwersji
            // std::string -> LPCSTR) Ze względów wydajnościowych robimy to co
            // kilka znaków lub na nowej linii
            if (c == '\n' || think_buffer.length() % 20 == 0) {
              std::wstring w_think = Utf8ToUtf16(think_buffer);
              ::SetWindowTextW(g_hHistory, w_think.c_str());
            }
          } else {
            // Jesteśmy w trakcie generowania faktycznego kodu, wypisujemy go do
            // edytora
            std::string s(1, c);
            ::SendMessage(curScintilla, SCI_REPLACESEL, 0, (LPARAM)s.c_str());
          }
        },
        [curScintilla](int count) {
          // Ta funkcja jest wywoływana, gdy silnik AI "cofnie" się o kilka
          // znaków
          for (int i = 0; i < count; i++) {
            ::SendMessage(curScintilla, SCI_DELETEBACK, 0, 0);
          }
        });

    // Ustaw końcową historię myślenia
    if (!think_buffer.empty()) {
      think_buffer += Loc("\r\n[Koniec myślenia. Kod wygenerowany.]",
                          "\r\n[End of thinking. Code generated.]");
      std::wstring w_think = Utf8ToUtf16(think_buffer);
      ::SetWindowTextW(g_hHistory, w_think.c_str());
    }

    // Uruchomienie trackera na podstawie zaktualizowanych linii
    auto currentPos = ::SendMessage(curScintilla, SCI_GETCURRENTPOS, 0, 0);
    int endLine = (int)::SendMessage(curScintilla, SCI_LINEFROMPOSITION,
                                     (WPARAM)currentPos, 0);
    AIManager::getInstance().startTracking(prompt, generated, startLine,
                                           endLine, currentFilePath);

    // Przywrócenie przycisku do stanu pierwotnego
    isGenerating = false;
    SetWindowTextW(g_hBtn,
                   Loc(L"Wygeneruj Kod AI", L"Generate AI Code").c_str());

    // Ustaw focus z powrotem na edytor
    SetFocus(curScintilla);
  }).detach();
}

LRESULT CALLBACK AIPanelProc(HWND hwnd, UINT msg, WPARAM wParam,
                             LPARAM lParam) {
  switch (msg) {
  case WM_CREATE: {
    // Pole historii myślenia na samej górze
    g_hHistory = CreateWindowExW(
        WS_EX_CLIENTEDGE, L"EDIT",
        Loc(L"Panel Myślenia AI gotowy.\r\n", L"AI Thinking Panel ready.\r\n")
            .c_str(),
        WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOVSCROLL | ES_MULTILINE |
            ES_READONLY,
        0, 0, 100, 50, hwnd, (HMENU)3, NULL, NULL);

    // Główne pole zapytania użytkownika
    g_hEdit = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "",
                              WS_CHILD | WS_VISIBLE | WS_BORDER |
                                  ES_AUTOVSCROLL | ES_MULTILINE | ES_WANTRETURN,
                              0, 0, 100, 100, hwnd, (HMENU)1, NULL, NULL);

    HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
    SendMessage(g_hHistory, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));
    SendMessage(g_hEdit, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));

    g_hBtn = CreateWindowExW(
        0, L"BUTTON", Loc(L"Wygeneruj Kod AI", L"Generate AI Code").c_str(),
        WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON, 0, 0, 100, 30, hwnd, (HMENU)2,
        NULL, NULL);
    SendMessage(g_hBtn, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));

    // Etykieta statusu trenowania
    g_hStatusLabel = CreateWindowExW(
        0, L"STATIC",
        Loc(L"AI połączone z chmurą...", L"AI connected to cloud...").c_str(),
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
    int historyHeight = 60; // Wysokość paska myślenia na górze

    // Okno myślenia na samej górze
    MoveWindow(g_hHistory, 5, 5, width - 10, historyHeight, TRUE);
    // Pole tekstowe zajmuje przestrzeń pod oknem myślenia
    MoveWindow(g_hEdit, 5, historyHeight + 10, width - 10,
               height - historyHeight - btnHeight - 20, TRUE);
    // Etykieta statusu po lewej na dole
    MoveWindow(g_hStatusLabel, 5, height - btnHeight - 5, width - 140,
               btnHeight, TRUE);
    // Przycisk po prawej na dole
    MoveWindow(g_hBtn, width - 130, height - btnHeight - 5, 120, btnHeight,
               TRUE);
    return 0;
  }
  case WM_COMMAND: {
    if (LOWORD(wParam) == 2) {
      ExecuteAIGeneration();
    }
    return 0;
  }
  case WM_DESTROY: {
    KillTimer(hwnd, 1);
    g_hAIPanel = NULL;
    return 0;
  }
  }
  return DefWindowProc(hwnd, msg, wParam, lParam);
}

void InitAIPanel() {
  if (g_hAIPanel != NULL)
    return;

  WNDCLASSA wc = {0};
  wc.lpfnWndProc = AIPanelProc;
  wc.hInstance = GetModuleHandle(NULL);
  wc.lpszClassName = "NppAI_PanelClass";
  wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE + 1);
  RegisterClassA(&wc);

  g_hAIPanel =
      CreateWindowExA(0, "NppAI_PanelClass", "NppAI - Chat",
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
      SetWindowTextW(g_hStatusLabel, Loc(L"AI połączone z chmurą...",
                                         L"AI connected to cloud...")
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

void sendSelectionToChat() {
  // Pobranie uchwytu Scintilli
  int which = -1;
  ::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0,
                (LPARAM)&which);
  if (which == -1)
    return;
  HWND curScintilla = (which == 0) ? nppData._scintillaMainHandle
                                   : nppData._scintillaSecondHandle;

  // Pobranie zaznaczonego tekstu
  auto selLen = ::SendMessage(curScintilla, SCI_GETSELTEXT, 0, 0);
  if (selLen <= 1) {
    return; // Nic nie zaznaczono
  }

  std::vector<char> selText(selLen);
  ::SendMessage(curScintilla, SCI_GETSELTEXT, 0, (LPARAM)selText.data());
  std::string prompt(selText.begin(),
                     selText.end() - 1); // remove null terminator

  // Pokaż panel
  if (!isPanelRegistered) {
    InitAIPanel();
  } else {
    ::SendMessage(nppData._nppHandle, NPPM_DMMSHOW, 0, (LPARAM)g_hAIPanel);
  }

  // Dodaj zaznaczony kod do pola textarea
  std::string currentText;
  int textLen = GetWindowTextLengthA(g_hEdit);
  if (textLen > 0) {
    std::vector<char> buf(textLen + 1);
    GetWindowTextA(g_hEdit, buf.data(), textLen + 1);
    currentText = std::string(buf.data()) + "\r\n";
  }

  std::string newText = currentText + "```\r\n" + prompt + "\r\n```\r\n";
  SetWindowTextA(g_hEdit, newText.c_str());

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
