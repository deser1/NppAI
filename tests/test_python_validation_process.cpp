#include "PythonValidationProcess.h"
#include <windows.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <thread>
#include <atomic>
int wmain(int argc, wchar_t** argv) {
    if (argc != 3) return 2;
    auto directory = std::filesystem::temp_directory_path() / (L"nppai-validator-fixture " + std::to_wstring(GetCurrentProcessId()));
    std::filesystem::create_directories(directory);
    auto cases = directory / L"authored cases.json";
    std::ofstream(cases) << "[]";
    SetEnvironmentVariableW(L"NPPAI_VALIDATION_PYTHON", argv[1]);
    SetEnvironmentVariableW(L"NPPAI_TEST_CASES", cases.c_str());
    SetEnvironmentVariableW(L"NPPAI_VALIDATION_IMAGE", L"fixture");
    bool ok = true;
    try {
        PythonValidationProcess process(argv[2]);
        auto pass = process.run("authored input", false, [] { return false; });
        ok &= pass.exitCode == 0 && pass.diagnostics.find("fixture syntax: authored input") != std::string::npos;
        auto fail = process.run("fail", true, [] { return false; });
        ok &= fail.exitCode == 1 && fail.diagnostics.find("fixture tests") != std::string::npos;
        auto flood = process.run("flood", false, [] { return false; });
        ok &= flood.exitCode == 2;
        std::atomic<bool> cancel{false};
        std::thread timer([&] { Sleep(300); cancel = true; });
        auto cancelled = process.run("wait", false, [&] { return cancel.load(); });
        timer.join(); ok &= cancelled.exitCode == 3;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; ok = false; }
    std::filesystem::remove_all(directory);
    if (!ok) std::cerr << "Validator process integration failed\n";
    return ok ? 0 : 1;
}
