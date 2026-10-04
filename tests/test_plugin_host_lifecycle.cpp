#include <windows.h>
#include <iostream>
#include <string>

#include "Notepad_plus_msgs.h"
#include "PluginInterface.h"
#include "Scintilla.h"

namespace {
struct ScintillaProbe {
    int dwellTime = -1;
    int setDwellCalls = 0;
};

bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}

LRESULT CALLBACK HostWindowProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam) {
    if (message == SCI_SETMOUSEDWELLTIME) {
        auto* probe = reinterpret_cast<ScintillaProbe*>(GetWindowLongPtrW(hwnd, GWLP_USERDATA));
        if (probe) {
            probe->dwellTime = static_cast<int>(wParam);
            ++probe->setDwellCalls;
        }
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

    const auto setInfoForTesting =
        reinterpret_cast<SetInfoForTesting>(GetProcAddress(plugin, "setInfoForTesting"));
    const auto getFuncsArray =
        reinterpret_cast<GetFuncsArray>(GetProcAddress(plugin, "getFuncsArray"));
    const auto beNotified =
        reinterpret_cast<BeNotified>(GetProcAddress(plugin, "beNotified"));

    bool ok = true;
    ok &= check(setInfoForTesting != nullptr, "test host initialization seam exists");
    ok &= check(getFuncsArray != nullptr, "getFuncsArray export exists");
    ok &= check(beNotified != nullptr, "beNotified export exists");

    HINSTANCE instance = GetModuleHandleW(nullptr);
    HWND nppHost = createHostWindow(instance, L"NppAI_Test_NppHost");
    HWND scintillaMain = createHostWindow(instance, L"NppAI_Test_ScintillaMain");
    HWND scintillaSecond = createHostWindow(instance, L"NppAI_Test_ScintillaSecond");
    ScintillaProbe mainProbe{};
    ScintillaProbe secondProbe{};
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

    if (getFuncsArray) {
        int count = 0;
        FuncItem* functions = getFuncsArray(&count);
        ok &= check(functions != nullptr, "host initialization exposes command array");
        ok &= check(count == 3, "host initialization registers three commands");
        if (functions && count == 3) {
            ok &= check(functions[0]._pFunc != nullptr, "AI panel callback registered");
            ok &= check(functions[1]._pFunc != nullptr, "selection callback registered");
            ok &= check(functions[2]._pFunc != nullptr, "telemetry callback registered");
        }
    }

    if (beNotified && scintillaMain && scintillaSecond) {
        SCNotification ready{};
        ready.nmhdr.code = NPPN_READY;
        beNotified(&ready);
        ok &= check(mainProbe.setDwellCalls == 1 && mainProbe.dwellTime == 600,
                    "NPPN_READY configures 600ms dwell time on main Scintilla");
        ok &= check(secondProbe.setDwellCalls == 1 && secondProbe.dwellTime == 600,
                    "NPPN_READY configures 600ms dwell time on second Scintilla");

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
