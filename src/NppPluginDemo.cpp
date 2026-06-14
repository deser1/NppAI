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
#include "AIManager.h"
#include "Scintilla.h"

extern FuncItem funcItem[nbFunc];
extern NppData nppData;


BOOL APIENTRY DllMain(HANDLE hModule, DWORD  reasonForCall, LPVOID /*lpReserved*/)
{
	try {

		switch (reasonForCall)
		{
			case DLL_PROCESS_ATTACH:
				pluginInit(hModule);
				break;

			case DLL_PROCESS_DETACH:
				pluginCleanUp();
				break;

			case DLL_THREAD_ATTACH:
				break;

			case DLL_THREAD_DETACH:
				break;
		}
	}
	catch (...) { return FALSE; }

    return TRUE;
}


extern "C" __declspec(dllexport) void setInfo(NppData notpadPlusData)
{
	nppData = notpadPlusData;
	commandMenuInit();
}

extern "C" __declspec(dllexport) const TCHAR * getName()
{
	return NPP_PLUGIN_NAME;
}

extern "C" __declspec(dllexport) FuncItem * getFuncsArray(int *nbF)
{
	*nbF = nbFunc;
	return funcItem;
}


extern "C" __declspec(dllexport) void beNotified(SCNotification *notifyCode)
{
	switch (notifyCode->nmhdr.code) 
	{
		case NPPN_READY:
		{
			// Ustawienie czasu oczekiwania na 600ms, aby uaktywnić zdarzenie "najechaniu myszką"
			::SendMessage(nppData._scintillaMainHandle, SCI_SETMOUSEDWELLTIME, 600, 0);
			::SendMessage(nppData._scintillaSecondHandle, SCI_SETMOUSEDWELLTIME, 600, 0);
			break;
		}
		case SCN_DWELLSTART:
		{
			int pos = (int)notifyCode->position;
			if (pos != -1) {
				HWND curScintilla = (HWND)notifyCode->nmhdr.hwndFrom;
				int selStart = (int)::SendMessage(curScintilla, SCI_GETSELECTIONSTART, 0, 0);
				int selEnd = (int)::SendMessage(curScintilla, SCI_GETSELECTIONEND, 0, 0);
				// Jeśli myszka jest nad zaznaczonym tekstem
				if (selStart != selEnd && pos >= selStart && pos <= selEnd) {
					const char* plainMsg = "NppAI: Kliknij tutaj, aby wyslac zaznaczenie do Chatu";
					::SendMessage(curScintilla, SCI_CALLTIPSHOW, pos, (LPARAM)plainMsg);
				}
			}
			break;
		}
		case SCN_DWELLEND:
		{
			HWND curScintilla = (HWND)notifyCode->nmhdr.hwndFrom;
			::SendMessage(curScintilla, SCI_CALLTIPCANCEL, 0, 0);
			break;
		}
		case SCN_CALLTIPCLICK:
		{
			HWND curScintilla = (HWND)notifyCode->nmhdr.hwndFrom;
			::SendMessage(curScintilla, SCI_CALLTIPCANCEL, 0, 0);
			sendSelectionToChat();
			break;
		}
		case NPPN_SHUTDOWN:
		{
			commandMenuCleanUp();
		}
		break;

		case NPPN_FILESAVED:
		{
			// Kiedy użytkownik zapisze plik, sprawdzamy czy zmodyfikował wygenerowany kod
			int which = -1;
			::SendMessage(nppData._nppHandle, NPPM_GETCURRENTSCINTILLA, 0, (LPARAM)&which);
			if (which != -1) {
				HWND curScintilla = (which == 0) ? nppData._scintillaMainHandle : nppData._scintillaSecondHandle;
				int length = ::SendMessage(curScintilla, SCI_GETLENGTH, 0, 0);
				std::string content(length + 1, '\0');
				::SendMessage(curScintilla, SCI_GETTEXT, length + 1, (LPARAM)content.data());
				AIManager::getInstance().checkModifications(content);
			}
		}
		break;

		default:
			return;
	}
}


// Here you can process the Npp Messages 
// I will make the messages accessible little by little, according to the need of plugin development.
// Please let me know if you need to access to some messages :
// https://github.com/notepad-plus-plus/notepad-plus-plus/issues
//
extern "C" __declspec(dllexport) LRESULT messageProc(UINT /*Message*/, WPARAM /*wParam*/, LPARAM /*lParam*/)
{/*
	if (Message == WM_MOVE)
	{
		::MessageBox(NULL, "move", "", MB_OK);
	}
*/
	return TRUE;
}

#ifdef UNICODE
extern "C" __declspec(dllexport) BOOL isUnicode()
{
    return TRUE;
}
#endif //UNICODE
