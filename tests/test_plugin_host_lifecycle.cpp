#include <windows.h>
#include <iostream>
#include <string>
#include <cstring>

#include "Notepad_plus_msgs.h"
#include "PluginInterface.h"
#include "Scintilla.h"

namespace {
struct HostProbe {
    int currentScintilla = 0;
};

struct ScintillaProbe {
    int dwellTime = -1;
    int setDwellCalls = 0;
    int selectionStart = 10;
    int selectionEnd = 20;
    int callTipShowCalls = 0;
    int callTipCancelCalls = 0;
    int callTipPosition = -1;
    std::string callTipText;
    std::string selectedText = "int answer = 42;";
    int getSelectedTextCalls = 0;
};

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

LRESULT CALLBACK HostWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    if (message == NPPM_GETCURRENTSCINTILLA) {
        auto* hostProbe = reinterpret_cast<HostProbe*>(
            GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (hostProbe && lParam)
            *reinterpret_cast<int*>(lParam) = hostProbe->currentScintilla;
        return TRUE;
    }

    auto* probe = reinterpret_cast<ScintillaProbe*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
    if (message == SCI_SETMOUSEDWELLTIME) {
        if (probe) {
            probe->dwellTime = static_cast<int>(wParam);
            ++probe->setDwellCalls;
        }
        return 0;
    }
    if (message == SCI_GETSELECTIONSTART) return probe ? probe->selectionStart : 0;
    if (message == SCI_GETSELECTIONEND) return probe ? probe->selectionEnd : 0;
    if (message == SCI_GETSELTEXT) {
        if (!probe) return 0;
        ++probe->getSelectedTextCalls;
        if (!lParam) return static_cast<LRESULT>(probe->selectedText.size() + 1);
        memcpy(reinterpret_cast<void*>(lParam), probe->selectedText.c_str(),
               probe->selectedText.size() + 1);
        return static_cast<LRESULT>(probe->selectedText.size() + 1);
    }
    if (message == SCI_CALLTIPSHOW) {
        if (probe) {
            ++probe->callTipShowCalls;
            probe->callTipPosition = static_cast<int>(wParam);
            probe->callTipText = lParam ? reinterpret_cast<const char*>(lParam) : "";
        }
        return 0;
    }
    if (message == SCI_CALLTIPCANCEL) {
        if (probe) ++probe->callTipCancelCalls;
        return 0;
    }
    return DefWindowProcW(hwnd, message, wParam, lParam);
}

HWND createHostWindow(HINSTANCE instance, const wchar_t* className) {
    WNDCLASSW wc{};
    wc.lpfnWndProc = HostWindowProc;
    wc.hInstance = instance;
    wc.lpszClassName = className;
    RegisterClassW(&wc);
    return CreateWindowExW(0, className, L"NppAI integration host", WS_OVERLAPPED,
                           0, 0, 320, 200, nullptr, nullptr, instance, nullptr);
}
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) {
        std::cerr << "Usage: TestPluginHostLifecycle <NppAI.dll>\n";
        return 2;
    }

    HMODULE plugin = LoadLibraryW(argv[1]);
    if (!plugin) {
        std::cerr << "FAIL: unable to load plugin DLL, error=" << GetLastError() << "\n";
        return 1;
    }

    using SetInfoForTesting = void (__cdecl*)(NppData);
    using GetFuncsArray = FuncItem* (__cdecl*)(int*);
    using BeNotified = void (__cdecl*)(SCNotification*);
    using BuildSelectionPromptForTesting = const char* (__cdecl*)(HWND);

    const auto setInfoForTesting =
        reinterpret_cast<SetInfoForTesting>(GetProcAddress(plugin, "setInfoForTesting"));
    const auto getFuncsArray =
        reinterpret_cast<GetFuncsArray>(GetProcAddress(plugin, "getFuncsArray"));
    const auto beNotified =
        reinterpret_cast<BeNotified>(GetProcAddress(plugin, "beNotified"));
    const auto buildSelectionPromptForTesting =
        reinterpret_cast<BuildSelectionPromptForTesting>(
            GetProcAddress(plugin, "buildSelectionPromptForTesting"));

    bool ok = true;
    ok &= check(setInfoForTesting != nullptr, "test host initialization seam exists");
    ok &= check(getFuncsArray != nullptr, "getFuncsArray export exists");
    ok &= check(beNotified != nullptr, "beNotified export exists");
    ok &= check(buildSelectionPromptForTesting != nullptr,
                "selection prompt test seam exists");

    HINSTANCE instance = GetModuleHandleW(nullptr);
    HWND nppHost = createHostWindow(instance, L"NppAI_Test_NppHost");
    HWND scintillaMain = createHostWindow(instance, L"NppAI_Test_ScintillaMain");
    HWND scintillaSecond = createHostWindow(instance, L"NppAI_Test_ScintillaSecond");
    HostProbe hostProbe{};
    ScintillaProbe mainProbe{};
    ScintillaProbe secondProbe{};
    if (nppHost) SetWindowLongPtrW(nppHost, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&hostProbe));
    if (scintillaMain) SetWindowLongPtrW(scintillaMain, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&mainProbe));
    if (scintillaSecond) SetWindowLongPtrW(scintillaSecond, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(&secondProbe));
    ok &= check(nppHost && scintillaMain && scintillaSecond, "host windows created");

    if (setInfoForTesting && nppHost && scintillaMain && scintillaSecond) {
        NppData data{};
        data._nppHandle = nppHost;
        data._scintillaMainHandle = scintillaMain;
        data._scintillaSecondHandle = scintillaSecond;
        setInfoForTesting(data);
    }

    PFUNCPLUGINCMD selectionCommand = nullptr;
    if (getFuncsArray) {
        int count = 0;
        FuncItem* functions = getFuncsArray(&count);
        ok &= check(functions != nullptr, "host initialization exposes command array");
        ok &= check(count == 3, "host initialization registers three commands");
        if (functions && count == 3) {
            ok &= check(functions[0]._pFunc != nullptr, "AI panel callback registered");
            ok &= check(functions[1]._pFunc != nullptr, "selection callback registered");
            selectionCommand = functions[1]._pFunc;
            ok &= check(functions[2]._pFunc != nullptr, "telemetry callback registered");
        }
    }

    if (selectionCommand && nppHost && scintillaMain && scintillaSecond) {
        mainProbe.selectedText.clear();
        secondProbe.selectedText.clear();

        hostProbe.currentScintilla = 0;
        const int mainCallsBefore = mainProbe.getSelectedTextCalls;
        const int secondCallsBefore = secondProbe.getSelectedTextCalls;
        selectionCommand();
        ok &= check(mainProbe.getSelectedTextCalls > mainCallsBefore,
                    "selection command reads from active main Scintilla");
        ok &= check(secondProbe.getSelectedTextCalls == secondCallsBefore,
                    "selection command does not read inactive second Scintilla");

        hostProbe.currentScintilla = 1;
        const int mainCallsAfterMain = mainProbe.getSelectedTextCalls;
        const int secondCallsAfterMain = secondProbe.getSelectedTextCalls;
        selectionCommand();
        ok &= check(secondProbe.getSelectedTextCalls > secondCallsAfterMain,
                    "selection command reads from active second Scintilla");
        ok &= check(mainProbe.getSelectedTextCalls == mainCallsAfterMain,
                    "selection command does not read inactive main Scintilla");
    }

    if (buildSelectionPromptForTesting && scintillaMain) {
        mainProbe.selectedText = "int answer = 42;";
        const char* prompt = buildSelectionPromptForTesting(scintillaMain);
        ok &= check(prompt &&
                        std::string(prompt) ==
                            "```\r\nint answer = 42;\r\n```\r\n",
                    "selected editor text is converted into a fenced prompt");

        mainProbe.selectedText.clear();
        prompt = buildSelectionPromptForTesting(scintillaMain);
        ok &= check(prompt && std::string(prompt).empty(),
                    "empty editor selection produces no prompt");
    }

    if (beNotified && scintillaMain && scintillaSecond) {
        SCNotification ready{};
        ready.nmhdr.code = NPPN_READY;
        beNotified(&ready);
        ok &= check(mainProbe.setDwellCalls == 1 && mainProbe.dwellTime == 600,
                    "NPPN_READY configures 600ms dwell time on main Scintilla");
        ok &= check(secondProbe.setDwellCalls == 1 && secondProbe.dwellTime == 600,
                    "NPPN_READY configures 600ms dwell time on second Scintilla");

        SCNotification dwellStart{};
        dwellStart.nmhdr.code = SCN_DWELLSTART;
        dwellStart.nmhdr.hwndFrom = scintillaMain;
        dwellStart.position = 15;
        beNotified(&dwellStart);
        ok &= check(mainProbe.callTipShowCalls == 1,
                    "SCN_DWELLSTART shows a calltip over selected text");
        ok &= check(mainProbe.callTipPosition == 15,
                    "SCN_DWELLSTART shows the calltip at the hovered position");
        ok &= check(mainProbe.callTipText.find("NppAI:") == 0,
                    "SCN_DWELLSTART provides the NppAI calltip text");

        const int shownAfterSelectedHover = mainProbe.callTipShowCalls;
        dwellStart.position = 25;
        beNotified(&dwellStart);
        ok &= check(mainProbe.callTipShowCalls == shownAfterSelectedHover,
                    "SCN_DWELLSTART does not show a calltip outside selected text");

        mainProbe.selectionStart = 10;
        mainProbe.selectionEnd = 10;
        dwellStart.position = 10;
        beNotified(&dwellStart);
        ok &= check(mainProbe.callTipShowCalls == shownAfterSelectedHover,
                    "SCN_DWELLSTART does not show a calltip for an empty selection");

        SCNotification dwellEnd{};
        dwellEnd.nmhdr.code = SCN_DWELLEND;
        dwellEnd.nmhdr.hwndFrom = scintillaMain;
        beNotified(&dwellEnd);
        ok &= check(mainProbe.callTipCancelCalls == 1,
                    "SCN_DWELLEND cancels the active calltip");

        // Keep the selection empty so sendSelectionToChat returns before
        // creating the docked panel; this isolates the click lifecycle contract.
        mainProbe.selectionStart = 10;
        mainProbe.selectionEnd = 10;
        SCNotification callTipClick{};
        callTipClick.nmhdr.code = SCN_CALLTIPCLICK;
        callTipClick.nmhdr.hwndFrom = scintillaMain;
        beNotified(&callTipClick);
        ok &= check(mainProbe.callTipCancelCalls == 2,
                    "SCN_CALLTIPCLICK cancels the active calltip");

        SCNotification shutdown{};
        shutdown.nmhdr.code = NPPN_SHUTDOWN;
        beNotified(&shutdown);
        ok &= check(true, "ready and shutdown notifications completed");
    }

    if (scintillaSecond) DestroyWindow(scintillaSecond);
    if (scintillaMain) DestroyWindow(scintillaMain);
    if (nppHost) DestroyWindow(nppHost);
    FreeLibrary(plugin);

    if (!ok) return 1;
    std::cout << "Notepad++ plugin host lifecycle integration test passed.\n";
    return 0;
}
