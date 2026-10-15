#include "PythonValidationProcess.h"
#include <windows.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

namespace {
std::wstring environment(const wchar_t* key) {
    DWORD length = GetEnvironmentVariableW(key, nullptr, 0);
    if (!length) return {};
    std::wstring result(length, L'\0');
    DWORD copied = GetEnvironmentVariableW(key, &result[0], length);
    if (!copied || copied >= length) return {};
    result.resize(copied); return result;
}
std::wstring quote(const std::wstring& value) {
    std::wstring result = L"\""; std::size_t slashes = 0;
    for (wchar_t c : value) {
        if (c == L'\\') { ++slashes; continue; }
        result.append(c == L'"' ? slashes * 2 + 1 : slashes, L'\\');
        slashes = 0; result += c;
    }
    result.append(slashes * 2, L'\\'); return result + L'"';
}
struct Handle {
    HANDLE value = nullptr;
    ~Handle() { if (value && value != INVALID_HANDLE_VALUE) CloseHandle(value); }
};
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        std::random_device random;
        for (int i = 0; i < 8; ++i) {
            path = std::filesystem::temp_directory_path() /
                (L"nppai-validation-" + std::to_wstring(random()) + L"-" + std::to_wstring(random()));
            if (std::filesystem::create_directory(path)) return;
        }
        throw std::runtime_error("Cannot create validation staging directory");
    }
    ~Temporary() { std::error_code error; std::filesystem::remove_all(path, error); }
};
void write(const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path, std::ios::binary);
    output.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!output) throw std::runtime_error("Cannot stage validation input");
}
}
PythonValidationProcess::PythonValidationProcess(const std::filesystem::path& bundledScript) {
    python_ = environment(L"NPPAI_VALIDATION_PYTHON");
    script_ = bundledScript;
    image_ = environment(L"NPPAI_VALIDATION_IMAGE");
    const std::filesystem::path cases = environment(L"NPPAI_TEST_CASES");
    if (!python_.is_absolute() || !std::filesystem::is_regular_file(python_) ||
        !std::filesystem::is_regular_file(script_) || !cases.is_absolute() ||
        !std::filesystem::is_regular_file(cases) || image_.empty())
        throw std::runtime_error("Configure Python validation: NPPAI_VALIDATION_PYTHON, NPPAI_VALIDATION_IMAGE, NPPAI_TEST_CASES");
    if (std::filesystem::file_size(cases) > 65536)
        throw std::runtime_error("Test suite exceeds 64 KiB");
    std::ifstream input(cases, std::ios::binary);
    cases_.resize(65537);
    input.read(&cases_[0], static_cast<std::streamsize>(cases_.size()));
    cases_.resize(static_cast<std::size_t>(input.gcount()));
    if (cases_.size() > 65536) throw std::runtime_error("Test suite exceeds 64 KiB");
    if (!input.eof() && input.fail()) throw std::runtime_error("Cannot read test suite");
}
AgentCommandResult PythonValidationProcess::run(const std::string& source, bool tests,
    const std::function<bool()>& cancelled) const {
    if (cancelled && cancelled()) return {3, "Validation cancelled"};
    try {
        Temporary temp;
        const auto candidate = temp.path / L"candidate.py";
        const auto cases = temp.path / L"cases.json";
        const auto cancel = temp.path / L"cancel";
        write(candidate, source); write(cases, cases_);
        std::vector<std::wstring> arguments{python_.wstring(), L"-I", L"-B", L"-X", L"utf8", script_.wstring(),
            L"--phase", tests ? L"tests" : L"syntax", L"--source", candidate.wstring(),
            L"--cases", cases.wstring(), L"--cancel-file", cancel.wstring(), L"--image", image_};
        std::wstring command;
        for (const auto& argument : arguments) { if (!command.empty()) command += L' '; command += quote(argument); }
        SECURITY_ATTRIBUTES security{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
        Handle readPipe, writePipe, nullInput, job, process, thread;
        if (!CreatePipe(&readPipe.value, &writePipe.value, &security, 0) ||
            !SetHandleInformation(readPipe.value, HANDLE_FLAG_INHERIT, 0))
            return {2, "Cannot create validator output pipe"};
        nullInput.value = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE,
            &security, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (nullInput.value == INVALID_HANDLE_VALUE) return {2, "Cannot prepare validator input"};
        SIZE_T bytes = 0;
        InitializeProcThreadAttributeList(nullptr, 1, 0, &bytes);
        std::vector<unsigned char> attributes(bytes);
        auto list = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attributes.data());
        if (!InitializeProcThreadAttributeList(list, 1, 0, &bytes))
            return {2, "Cannot prepare validator process"};
        struct AttributeCleanup { LPPROC_THREAD_ATTRIBUTE_LIST value; ~AttributeCleanup() { DeleteProcThreadAttributeList(value); } } cleanup{list};
        HANDLE inherited[]{writePipe.value, nullInput.value};
        if (!UpdateProcThreadAttribute(list, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST,
            inherited, sizeof(inherited), nullptr, nullptr)) return {2, "Cannot restrict validator handles"};
        STARTUPINFOEXW start{}; start.StartupInfo.cb = sizeof(start);
        start.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
        start.StartupInfo.hStdInput = nullInput.value;
        start.StartupInfo.hStdOutput = start.StartupInfo.hStdError = writePipe.value;
        start.lpAttributeList = list;
        job.value = CreateJobObjectW(nullptr, nullptr);
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
        limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (!job.value || !SetInformationJobObject(job.value, JobObjectExtendedLimitInformation, &limits, sizeof(limits)))
            return {2, "Cannot contain validator process tree"};
        PROCESS_INFORMATION info{};
        if (!CreateProcessW(python_.c_str(), &command[0], nullptr, nullptr, TRUE,
            CREATE_SUSPENDED | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT, nullptr,
            temp.path.c_str(), &start.StartupInfo, &info)) return {2, "Cannot start configured Python validator"};
        process.value = info.hProcess; thread.value = info.hThread;
        if (!AssignProcessToJobObject(job.value, process.value)) {
            TerminateProcess(process.value, 2); return {2, "Cannot contain validator process tree"};
        }
        ResumeThread(thread.value);
        CloseHandle(writePipe.value); writePipe.value = nullptr;
        const auto started = std::chrono::steady_clock::now();
        auto stopAt = started;
        bool stopping = false, cancelledByUser = false, overflow = false;
        std::string diagnostics;
        auto drain = [&]() {
            DWORD available = 0;
            while (PeekNamedPipe(readPipe.value, nullptr, 0, nullptr, &available, nullptr) && available) {
                char buffer[4096]; DWORD read = 0;
                if (!ReadFile(readPipe.value, buffer, (std::min)(available, DWORD(sizeof(buffer))), &read, nullptr) || !read) break;
                if (diagnostics.size() + read <= 16384) diagnostics.append(buffer, read);
                else overflow = true;
            }
        };
        while (WaitForSingleObject(process.value, 50) == WAIT_TIMEOUT) {
            drain();
            const auto now = std::chrono::steady_clock::now();
            const bool stop = (cancelled && cancelled());
            if (!stopping && (stop || overflow || now - started > std::chrono::seconds(35))) {
                cancelledByUser = stop; stopping = true; stopAt = now; write(cancel, "cancel");
            }
            if (stopping && now - stopAt > std::chrono::seconds(10)) {
                TerminateJobObject(job.value, 2); WaitForSingleObject(process.value, 5000);
                return {cancelledByUser ? 3 : 2, cancelledByUser ? "Validation cancelled" : "Validator deadline/output limit exceeded"};
            }
        }
        drain();
        if (cancelledByUser || (cancelled && cancelled())) return {3, "Validation cancelled"};
        if (overflow || stopping) return {2, "Validator deadline/output limit exceeded"};
        DWORD exit = 2; GetExitCodeProcess(process.value, &exit);
        if (exit > 3 || diagnostics.empty()) return {2, "Validator exited without a valid result"};
        return {int(exit), diagnostics.substr(0, 4096)};
    } catch (const std::exception& error) { return {2, error.what()}; }
}
