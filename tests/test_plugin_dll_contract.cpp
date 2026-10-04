#include <windows.h>
#include <iostream>
#include <string>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) { std::cerr << "FAIL: " << message << "\n"; return false; }
    return true;
}
}

int wmain(int argc, wchar_t** argv) {
    if (argc != 2) { std::cerr << "Usage: TestPluginDllContract <NppAI.dll>\n"; return 2; }
    HMODULE plugin = LoadLibraryW(argv[1]);
    if (!plugin) { std::cerr << "FAIL: unable to load plugin DLL, error=" << GetLastError() << "\n"; return 1; }

    using GetName = const wchar_t* (*)();
    using GetFuncsArray = void* (*)(int*);
    using IsUnicode = BOOL (*)();
    using MessageProc = LRESULT (*)(UINT, WPARAM, LPARAM);
    const auto getName = reinterpret_cast<GetName>(GetProcAddress(plugin, "getName"));
    const auto getFuncsArray = reinterpret_cast<GetFuncsArray>(GetProcAddress(plugin, "getFuncsArray"));
    const auto isUnicode = reinterpret_cast<IsUnicode>(GetProcAddress(plugin, "isUnicode"));
    const auto messageProc = reinterpret_cast<MessageProc>(GetProcAddress(plugin, "messageProc"));

    bool ok = true;
    ok &= check(getName != nullptr, "getName export exists");
    ok &= check(getFuncsArray != nullptr, "getFuncsArray export exists");
    ok &= check(isUnicode != nullptr, "isUnicode export exists");
    ok &= check(messageProc != nullptr, "messageProc export exists");
    if (getName) ok &= check(getName() && std::wstring(getName()) == L"NppAI", "plugin reports expected name");
    if (getFuncsArray) {
        int count = 0; void* functions = getFuncsArray(&count);
        ok &= check(functions != nullptr, "plugin exposes command array");
        ok &= check(count == 3, "plugin exposes all expected commands");
    }
    if (isUnicode) ok &= check(isUnicode() == TRUE, "plugin declares Unicode support");
    if (messageProc) ok &= check(messageProc(0, 0, 0) == TRUE, "messageProc smoke test");

    FreeLibrary(plugin);
    if (!ok) return 1;
    std::cout << "Notepad++ plugin DLL contract integration test passed.\n";
    return 0;
}
