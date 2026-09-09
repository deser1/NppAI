//this file is part of notepad++
//Copyright (C)2022 Don HO <don.h@free.fr>
//
//This program is free software; you can redistribute it and/or
//modify it under the terms of the GNU General Public License
//as published by the Free Software Foundation; either
//version 2 of the License, or (at your option) any later version.
//
//This program is distributed in the hope that it will be useful,
//but WITHOUT ANY WARRANTY; without even the implied warranty of
//MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//GNU General Public License for more details.
//
//You should have received a copy of the GNU General Public License
//along with this program; if not, write to the Free Software
//Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

#include "PluginDefinition.h"
#include "menuCmdID.h"
#include "AIManager.h"
#include "TelemetryManager.h"
#include "Notepad_plus_msgs.h"
#include "DockingFeature/Docking.h"
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <fstream>
#include <sstream>

extern NppData nppData;

// Okienko wprowadzania promptu (Dockable Panel)
HWND g_hAIPanel = NULL;
HWND g_hEdit = NULL;
HWND g_hBtn = NULL;
HWND g_hHistory = NULL; // Nowe pole do wyswietlania myslenia
HWND g_hStatusLabel = NULL; // Dodane: etykieta postępu trenowania
bool isPanelRegistered = false;
std::atomic<bool> isGenerating{false};

// Funkcja pomocnicza do konwersji UTF-8 na UTF-16
std::wstring Utf8ToUtf16(const std::string& utf8) {
    if (utf8.empty()) return std::wstring();
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, &utf8[0], (int)utf8.size(), NULL, 0);
    std::wstring utf16(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, &utf8[0], (int)utf8.size(), &utf16[0], size_needed);
    return utf16;
}

// Funkcja odpowiedzialna za wykonanie akcji generowania po wpisaniu tekstu
void ExecuteAIGeneration() {
    if (!g_hEdit) return;
    
    if (isGenerating) {
        // Jeśli generujemy, przerwijmy to!
        AIManager::getInstance().stopGeneration();
        SetWindowTextA(g_hBtn, "Wygeneruj Kod AI");
        isGenerating = false;
        return;
    }

    char buf[4096] = {0};
    GetWindowTextA(g_hEdit, buf, sizeof(buf));
    std::string prompt = buf;
    if (prompt.empty()) return;
    
    // Wyczyść pole po pobraniu tekstu
    SetWindowTextW(g_hEdit, L"");
    
    // Wyczyść historię myślenia
    SetWindowTextW(g_hHistory, L"Czekam na odpowiedź...\r\n");

    // Pobranie uchwytu Scintilli
    int which = -1;
    ::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0, (LPARAM)&which);
    if (which == -1) return;
    HWND curScintilla = (which == 0) ? nppData._scintillaMainHandle : nppData._scintillaSecondHandle;

    // Uruchomienie trackera
    int startLine = (int)::SendMessage(curScintilla, SCI_LINEFROMPOSITION, (WPARAM)::SendMessage(curScintilla, SCI_GETCURRENTPOS, 0, 0), 0);

    // Zmień przycisk na "Zatrzymaj"
    isGenerating = true;
    SetWindowTextA(g_hBtn, "Zatrzymaj");

    // Wygenerowanie kodu w osobnym wątku, by nie blokować interfejsu (Notepad++ brak odpowiedzi)
    std::thread([prompt, curScintilla, startLine]() {
        // Zmienne do obsługi tagu <THINK>
        std::string think_buffer = "";
        bool is_thinking = false;

        // Wygenerowanie kodu strumieniowo
        std::string generated = AIManager::getInstance().generateCode(prompt, "",
            [&is_thinking, &think_buffer, curScintilla](char c, bool isThought) {
                if (isThought) {
                    // Jesteśmy w trakcie myślenia
                    think_buffer += c;
                    // Możemy aktualizować pole historii na żywo (wymaga konwersji std::string -> LPCSTR)
                    // Ze względów wydajnościowych robimy to co kilka znaków lub na nowej linii
                    if (c == '\n' || think_buffer.length() % 20 == 0) {
                        std::wstring w_think = Utf8ToUtf16(think_buffer);
                        ::SetWindowTextW(g_hHistory, w_think.c_str());
                    }
                } else {
                    // Jesteśmy w trakcie generowania faktycznego kodu, wypisujemy go do edytora
                    std::string s(1, c);
                    ::SendMessage(curScintilla, SCI_REPLACESEL, 0, (LPARAM)s.c_str());
                }
            },
            [curScintilla](int count) {
                // Ta funkcja jest wywoływana, gdy silnik AI "cofnie" się o kilka znaków 
                for (int i = 0; i < count; i++) {
                    ::SendMessage(curScintilla, SCI_DELETEBACK, 0, 0);
                }
            }
        );

        // Ustaw końcową historię myślenia
        if (!think_buffer.empty()) {
            think_buffer += "\r\n[Koniec myślenia. Kod wygenerowany.]";
            std::wstring w_think = Utf8ToUtf16(think_buffer);
            ::SetWindowTextW(g_hHistory, w_think.c_str());
        }

        // Uruchomienie trackera na podstawie zaktualizowanych linii
        auto currentPos = ::SendMessage(curScintilla, SCI_GETCURRENTPOS, 0, 0);
        int endLine = (int)::SendMessage(curScintilla, SCI_LINEFROMPOSITION, (WPARAM)currentPos, 0);
        AIManager::getInstance().startTracking(prompt, generated, startLine, endLine);
        
        // Przywrócenie przycisku do stanu pierwotnego
        isGenerating = false;
        SetWindowTextA(g_hBtn, "Wygeneruj Kod AI");

        // Ustaw focus z powrotem na edytor
        SetFocus(curScintilla);
    }).detach();
}

LRESULT CALLBACK AIPanelProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            // Pole historii myślenia na samej górze
            g_hHistory = CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", L"Panel Myślenia AI gotowy.\r\n",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOVSCROLL | ES_MULTILINE | ES_READONLY,
                0, 0, 100, 50, hwnd, (HMENU)3, NULL, NULL);
                
            // Główne pole zapytania użytkownika
            g_hEdit = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOVSCROLL | ES_MULTILINE | ES_WANTRETURN,
                0, 0, 100, 100, hwnd, (HMENU)1, NULL, NULL);
            
            HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
            SendMessage(g_hHistory, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));
            SendMessage(g_hEdit, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));
            
            g_hBtn = CreateWindowExA(0, "BUTTON", "Wygeneruj Kod AI",
                WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
                0, 0, 100, 30, hwnd, (HMENU)2, NULL, NULL);
            SendMessage(g_hBtn, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));

            // Etykieta statusu trenowania
            g_hStatusLabel = CreateWindowExW(0, L"STATIC", L"AI gotowe do pracy",
                WS_CHILD | WS_VISIBLE,
                0, 0, 100, 30, hwnd, (HMENU)4, NULL, NULL);
            SendMessage(g_hStatusLabel, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));

            // Timer do odświeżania logów co 2 sekundy
            SetTimer(hwnd, 1, 2000, NULL);
            
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
            MoveWindow(g_hEdit, 5, historyHeight + 10, width - 10, height - historyHeight - btnHeight - 20, TRUE);
            // Etykieta statusu po lewej na dole
            MoveWindow(g_hStatusLabel, 5, height - btnHeight - 5, width - 140, btnHeight, TRUE);
            // Przycisk po prawej na dole
            MoveWindow(g_hBtn, width - 130, height - btnHeight - 5, 120, btnHeight, TRUE);
            return 0;
        }
        case WM_TIMER: {
            if (wParam == 1) {
                std::ifstream file("D:\\Projekty\\AI_Coding_Notepad++\\training_log.txt");
                if (file.is_open()) {
                    std::string line, last_line;
                    while (std::getline(file, line)) {
                        if (!line.empty()) {
                            last_line = line;
                        }
                    }
                    if (!last_line.empty() && last_line.find("Krok") != std::string::npos) {
                        // Wstawiamy poprawny polski tekst wymuszając kodowanie UTF-8 (przedrostek u8)
                        size_t pipe_pos = last_line.find("|");
                        size_t loss_pos = last_line.find("(Loss)");
                        std::string clean_line = last_line;
                        if (pipe_pos != std::string::npos && loss_pos != std::string::npos) {
                            clean_line = last_line.substr(0, pipe_pos + 1) + u8" Błąd " + last_line.substr(loss_pos);
                        }
                        
                        std::string status = u8"Trenowanie w tle: " + clean_line;
                        int len = MultiByteToWideChar(CP_UTF8, 0, status.c_str(), -1, NULL, 0);
                        if (len > 0) {
                            std::vector<wchar_t> wbuf(len);
                            MultiByteToWideChar(CP_UTF8, 0, status.c_str(), -1, wbuf.data(), len);
                            SetWindowTextW(g_hStatusLabel, wbuf.data());
                        }
                    } else if (!last_line.empty() && last_line.find("Zako") != std::string::npos) {
                        SetWindowTextW(g_hStatusLabel, L"AI gotowe do pracy (Trenowanie zakonczone)");
                    }
                }
            }
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
    if (g_hAIPanel != NULL) return;
    
    WNDCLASSA wc = {0};
    wc.lpfnWndProc = AIPanelProc;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "NppAI_PanelClass";
    wc.hbrBackground = (HBRUSH)(COLOR_BTNFACE+1);
    RegisterClassA(&wc);

    g_hAIPanel = CreateWindowExA(
        0, "NppAI_PanelClass", "NppAI - Chat",
        WS_CHILD | WS_CLIPCHILDREN | WS_CLIPSIBLINGS,
        0, 0, 400, 200, nppData._nppHandle, NULL, wc.hInstance, NULL
    );

    DockedWidgetData tbData = {0};
    tbData.hClient = g_hAIPanel;
    tbData.pszName = TEXT("NppAI - Prompt");
    tbData.dlgID = 0;
    tbData.uMask = DWS_DF_CONT_BOTTOM | DWS_ICONTAB | DWS_ICONBAR;
    tbData.pszModuleName = NPP_PLUGIN_NAME;

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
void pluginInit(HANDLE /*hModule*/)
{
}

//
// Here you can do the clean up, save the parameters (if any) for the next session
//
void pluginCleanUp()
{
}

//
// Initialization of your plugin commands
// You should fill your plugins commands here
void commandMenuInit()
{

    //--------------------------------------------//
    //-- STEP 3. CUSTOMIZE YOUR PLUGIN COMMANDS --//
    //--------------------------------------------//
    // with function :
    // setCommand(int index,                      // zero based number to indicate the order of command
    //            TCHAR *commandName,             // the command name that you want to see in plugin menu
    //            PFUNCPLUGINCMD functionPointer, // the symbol of function (function pointer) associated with this command. The body should be defined below. See Step 4.
    //            ShortcutKey *shortcut,          // optional. Define a shortcut to trigger this command
    //            bool check0nInit                // optional. Make this menu item be checked visually
    //            );
    setCommand(0, TEXT("Pokaż Panel AI"), toggleAIPanel, NULL, false);
    setCommand(1, TEXT("Zapytaj AI o zaznaczony kod"), sendSelectionToChat, NULL, false);
    setCommand(2, TEXT("Zmień status telemetrii"), toggleTelemetry, NULL, false);
}

//
// Here you can do the clean up (especially for the shortcut)
//
void commandMenuCleanUp()
{
	// Don't forget to deallocate your shortcut here
}


//
// This function help you to initialize your plugin commands
//
bool setCommand(size_t index, TCHAR *cmdName, PFUNCPLUGINCMD pFunc, ShortcutKey *sk, bool check0nInit) 
{
    if (index >= nbFunc)
        return false;

    if (!pFunc)
        return false;

    lstrcpy(funcItem[index]._itemName, cmdName);
    funcItem[index]._pFunc = pFunc;
    funcItem[index]._init2Check = check0nInit;
    funcItem[index]._pShKey = sk;

    return true;
}

//----------------------------------------------//
//-- STEP 4. DEFINE YOUR ASSOCIATED FUNCTIONS --//
//----------------------------------------------//
void generateAICode()
{
    // Nie jest już bezpośrednio wywoływana z menu. Zastąpiona przez toggleAIPanel.
}

void toggleAIPanel()
{
    if (!isPanelRegistered) {
        InitAIPanel();
    } else {
        // Jeśli już zarejestrowany, pokazujemy go
        ::SendMessage(nppData._nppHandle, NPPM_DMMSHOW, 0, (LPARAM)g_hAIPanel);
    }
}

void sendSelectionToChat()
{
    // Pobranie uchwytu Scintilli
    int which = -1;
    ::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0, (LPARAM)&which);
    if (which == -1) return;
    HWND curScintilla = (which == 0) ? nppData._scintillaMainHandle : nppData._scintillaSecondHandle;

    // Pobranie zaznaczonego tekstu
    auto selLen = ::SendMessage(curScintilla, SCI_GETSELTEXT, 0, 0);
    if (selLen <= 1) {
        return; // Nic nie zaznaczono
    }

    std::vector<char> selText(selLen);
    ::SendMessage(curScintilla, SCI_GETSELTEXT, 0, (LPARAM)selText.data());
    std::string prompt(selText.begin(), selText.end() - 1); // remove null terminator

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

void toggleTelemetry()
{
    ::MessageBox(NULL, TEXT("Telemetria i uczenie globalne: AKTYWNE\nTwój kod będzie anonimizowany przed wysłaniem."), TEXT("NppAI Ustawienia"), MB_OK);
}
