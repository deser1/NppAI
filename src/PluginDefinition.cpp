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
#include "Docking.h"
#include <string>
#include <vector>

extern NppData nppData;

// Okienko wprowadzania promptu (Dockable Panel)
HWND g_hAIPanel = NULL;
HWND g_hEdit = NULL;
HWND g_hBtn = NULL;
bool isPanelRegistered = false;

// Funkcja odpowiedzialna za wykonanie akcji generowania po wpisaniu tekstu
void ExecuteAIGeneration() {
    if (!g_hEdit) return;
    
    char buf[4096] = {0};
    GetWindowTextA(g_hEdit, buf, sizeof(buf));
    std::string prompt = buf;
    if (prompt.empty()) return;
    
    // Wyczyść pole po pobraniu tekstu
    SetWindowTextA(g_hEdit, "");

    // Pobranie uchwytu Scintilli
    int which = -1;
    ::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0, (LPARAM)&which);
    if (which == -1) return;
    HWND curScintilla = (which == 0) ? nppData._scintillaMainHandle : nppData._scintillaSecondHandle;

    // Wygenerowanie kodu
    std::string generated = AIManager::getInstance().generateCode(prompt, "");

    // Wklejenie w miejscu kursora
    ::SendMessage(curScintilla, SCI_REPLACESEL, 0, (LPARAM)generated.c_str());

    // Uruchomienie trackera
    auto currentPos = ::SendMessage(curScintilla, SCI_GETCURRENTPOS, 0, 0);
    int currentLine = ::SendMessage(curScintilla, SCI_LINEFROMPOSITION, currentPos, 0);
    AIManager::getInstance().startTracking(prompt, generated, currentLine - 3, currentLine);
    
    // Ustaw focus z powrotem na edytor
    SetFocus(curScintilla);
}

LRESULT CALLBACK AIPanelProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_CREATE: {
            g_hEdit = CreateWindowExA(WS_EX_CLIENTEDGE, "EDIT", "",
                WS_CHILD | WS_VISIBLE | WS_BORDER | ES_AUTOVSCROLL | ES_MULTILINE | ES_WANTRETURN,
                0, 0, 100, 100, hwnd, (HMENU)1, NULL, NULL);
            
            HFONT hFont = (HFONT)GetStockObject(DEFAULT_GUI_FONT);
            SendMessage(g_hEdit, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));
            
            g_hBtn = CreateWindowExA(0, "BUTTON", "Wygeneruj Kod AI",
                WS_CHILD | WS_VISIBLE | BS_DEFPUSHBUTTON,
                0, 0, 100, 30, hwnd, (HMENU)2, NULL, NULL);
            SendMessage(g_hBtn, WM_SETFONT, (WPARAM)hFont, MAKELPARAM(TRUE, 0));
            return 0;
        }
        case WM_SIZE: {
            int width = LOWORD(lParam);
            int height = HIWORD(lParam);
            int btnHeight = 30;
            // Pole tekstowe zajmuje całą przestrzeń oprócz przycisku na dole
            MoveWindow(g_hEdit, 5, 5, width - 10, height - btnHeight - 15, TRUE);
            // Przycisk po prawej na dole
            MoveWindow(g_hBtn, width - 130, height - btnHeight - 5, 120, btnHeight, TRUE);
            return 0;
        }
        case WM_COMMAND: {
            if (LOWORD(wParam) == 2) { 
                ExecuteAIGeneration();
            }
            return 0;
        }
        case WM_DESTROY: {
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
    setCommand(1, TEXT("Zmień status telemetrii"), toggleTelemetry, NULL, false);
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

void toggleTelemetry()
{
    ::MessageBox(NULL, TEXT("Telemetria i uczenie globalne: AKTYWNE\nTwój kod będzie anonimizowany przed wysłaniem."), TEXT("NppAI Ustawienia"), MB_OK);
}
