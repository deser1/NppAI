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
#include <string>
#include <vector>

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
    setCommand(0, TEXT("Wygeneruj Kod AI"), generateAICode, NULL, false);
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
    // Pobranie uchwytu Scintilli
    int which = -1;
    ::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0, (LPARAM)&which);
    if (which == -1) return;
    HWND curScintilla = (which == 0) ? nppData._scintillaMainHandle : nppData._scintillaSecondHandle;

    // Pobranie zaznaczonego tekstu (promptu użytkownika)
    auto selLen = ::SendMessage(curScintilla, SCI_GETSELTEXT, 0, 0);
    if (selLen <= 1) {
        ::MessageBox(NULL, TEXT("Zaznacz tekst, który ma być promptem dla AI."), TEXT("NppAI"), MB_OK);
        return;
    }

    std::vector<char> selText(selLen);
    ::SendMessage(curScintilla, SCI_GETSELTEXT, 0, (LPARAM)selText.data());
    std::string prompt(selText.begin(), selText.end() - 1); // remove null terminator

    // Wygenerowanie kodu z użyciem AIManager
    std::string generated = AIManager::getInstance().generateCode(prompt, "");

    // Podmiana zaznaczonego tekstu na wygenerowany kod
    ::SendMessage(curScintilla, SCI_REPLACESEL, 0, (LPARAM)generated.c_str());

    // Pobranie linii w których znajduje się wklejony kod (do trackowania)
    auto currentPos = ::SendMessage(curScintilla, SCI_GETCURRENTPOS, 0, 0);
    int currentLine = ::SendMessage(curScintilla, SCI_LINEFROMPOSITION, currentPos, 0);
    
    // Uruchomienie trackera Diff
    AIManager::getInstance().startTracking(prompt, generated, currentLine - 3, currentLine);
}

void toggleTelemetry()
{
    ::MessageBox(NULL, TEXT("Telemetria i uczenie globalne: AKTYWNE\nTwój kod będzie anonimizowany przed wysłaniem."), TEXT("NppAI Ustawienia"), MB_OK);
}
