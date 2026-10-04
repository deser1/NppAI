#include <windows.h>
#include <iostream>
#include <string>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << "\n"; return false; }
    return true;
}

struct NppData {
    HWND nppHandle = nullptr;
    HWND scintillaMainHandle = nullptr;
    HWND scintillaSecondHandle = nullptr;
};

using PluginCommand = void (__cdecl*)();

struct ShortcutKey {
    bool isCtrl = false;
    bool isAlt = false;
    bool isShift = false;
    UCHAR key = 0;
};

struct FuncItem {
    wchar_t itemName[64] = { L'\0' };
    PluginCommand function = nullptr;
    int commandId = 0;
    bool initToCheck = false;
    ShortcutKey* shortcut = nullptr;
};
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) { std::cerr << "Usage: TestPluginDllContract <NppAI.dll>\n"; return 2; }
    HMODULE plugin = LoadLibraryW(argv[1]);
    if (!plugin) { std::cerr << "FAIL: unable to load plugin DLL, error=" << GetLastError() << "\n"; return 1; }

    using SetInfo = void (__cdecl*)(NppData);
    using SetInfoForTesting = void (__cdecl*)(NppData);
    using GetName = const wchar_t* (__cdecl*)();
    using GetFuncsArray = FuncItem* (__cdecl*)(int*);
    using BeNotified = void (__cdecl*)(void*);
    using IsUnicode = BOOL (__cdecl*)();
    using MessageProc = LRESULT (__cdecl*)(UINT, WPARAM, LPARAM);

    const auto setInfo = reinterpret_cast<SetInfo>(GetProcAddress(plugin, "setInfo"));
    const auto setInfoForTesting = reinterpret_cast<SetInfoForTesting>(GetProcAddress(plugin, "setInfoForTesting"));
    const auto getName = reinterpret_cast<GetName>(GetProcAddress(plugin, "getName"));
    const auto getFuncsArray = reinterpret_cast<GetFuncsArray>(GetProcAddress(plugin, "getFuncsArray"));
    const auto beNotified = reinterpret_cast<BeNotified>(GetProcAddress(plugin, "beNotified"));
    const auto isUnicode = reinterpret_cast<IsUnicode>(GetProcAddress(plugin, "isUnicode"));
    const auto messageProc = reinterpret_cast<MessageProc>(GetProcAddress(plugin, "messageProc"));

    bool ok = true;
    ok &= check(setInfo != nullptr, "setInfo export exists");
    ok &= check(setInfoForTesting != nullptr, "test host initialization seam exists");
    ok &= check(getName != nullptr, "getName export exists");
    ok &= check(getFuncsArray != nullptr, "getFuncsArray export exists");
    ok &= check(beNotified != nullptr, "beNotified export exists");
    ok &= check(isUnicode != nullptr, "isUnicode export exists");
    ok &= check(messageProc != nullptr, "messageProc export exists");

    if (getName) ok &= check(getName() && std::wstring(getName()) == L"NppAI", "plugin reports expected name");

    // Exercise the same host initialization path as setInfo without requiring a
    // model artifact. This lets CI validate command registration independently
    // from model deployment.
    if (setInfoForTesting)
        setInfoForTesting(NppData{});

    if (getFuncsArray) {
        int count = 0;
        FuncItem* functions = getFuncsArray(&count);
        ok &= check(functions != nullptr, "plugin exposes command array");
        ok &= check(count == 3, "plugin exposes all expected command slots");
        if (functions && count == 3) {
            ok &= check(std::wstring(functions[0].itemName) == L"Pokaż Panel AI", "AI panel command is registered");
            ok &= check(std::wstring(functions[1].itemName) == L"Zapytaj AI o zaznaczony kod", "selection command is registered");
            ok &= check(std::wstring(functions[2].itemName) == L"Zmień status telemetrii", "telemetry command is registered");
            ok &= check(functions[0].function && functions[1].function && functions[2].function, "command callbacks are registered");
        }
    }

    if (isUnicode) ok &= check(isUnicode() == TRUE, "plugin declares Unicode support");
    if (messageProc) ok &= check(messageProc(0, 0, 0) == TRUE, "messageProc smoke test");

    FreeLibrary(plugin);
    if (!ok) return 1;
    std::cout << "Notepad++ plugin DLL contract integration test passed.\n";
    return 0;
}
