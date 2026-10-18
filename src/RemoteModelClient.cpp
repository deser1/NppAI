#include "RemoteModelClient.h"
#include "AutomaticModelRouting.h"
#include <windows.h>
#include <winhttp.h>
#include <stdexcept>
#include <memory>
#pragma comment(lib, "winhttp.lib")
namespace {
struct Close { void operator()(void* value) const { if (value) WinHttpCloseHandle(value); } };
using Handle = std::unique_ptr<void, Close>;
std::wstring wide(const std::string& text) {
    int size = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), nullptr, 0);
    if (size <= 0) throw std::runtime_error("Invalid UTF-8 backend configuration");
    std::wstring value(size, L'\0');
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), int(text.size()), &value[0], size);
    return value;
}
void require(bool ok) {
    if (!ok) throw std::runtime_error("Online model connection failed (Windows error " +
                                     std::to_string(GetLastError()) + "). Check the NppAI server and NPPAI_GENERATION_URL.");
}
}
namespace RemoteModelClient {
std::string generate(const std::string& endpoint, const std::string& key,
                     const std::string& body, int budget, const std::atomic<bool>& cancelled) {
    if (key.empty() || key.size() > 4096) throw std::runtime_error("Invalid model API key");
    for (unsigned char c : key) if (c < 33 || c > 126) throw std::runtime_error("Invalid model API key");
    auto url = wide(endpoint);
    URL_COMPONENTS parts{}; parts.dwStructSize = sizeof(parts);
    parts.dwSchemeLength = parts.dwHostNameLength = parts.dwUrlPathLength =
        parts.dwExtraInfoLength = parts.dwUserNameLength = parts.dwPasswordLength = DWORD(-1);
    require(WinHttpCrackUrl(url.c_str(), DWORD(url.size()), 0, &parts) != FALSE);
    std::wstring host(parts.lpszHostName, parts.dwHostNameLength);
    const std::wstring path(parts.lpszUrlPath, parts.dwUrlPathLength);
    const bool secure = parts.nScheme == INTERNET_SCHEME_HTTPS;
    const bool loopback = host == L"127.0.0.1" || host == L"localhost" || host == L"::1" || host == L"[::1]";
    if ((!secure && !(parts.nScheme == INTERNET_SCHEME_HTTP && loopback)) ||
        parts.dwUserNameLength || parts.dwPasswordLength || parts.dwExtraInfoLength ||
        path != L"/api/v1/generate")
        throw std::runtime_error("Online model requires HTTPS or loopback HTTP at /api/v1/generate");
    if (cancelled) throw std::runtime_error("Generation cancelled");
    Handle session(WinHttpOpen(L"NppAI/1", WINHTTP_ACCESS_TYPE_NO_PROXY,
                              WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0));
    require(bool(session));
    require(WinHttpSetTimeouts(session.get(), 5000, 5000, 30000, 30000) != FALSE);
    Handle connection(WinHttpConnect(session.get(), host.c_str(), parts.nPort, 0));
    require(bool(connection));
    Handle call(WinHttpOpenRequest(connection.get(), L"POST", path.c_str(), nullptr,
        WINHTTP_NO_REFERER, WINHTTP_DEFAULT_ACCEPT_TYPES, secure ? WINHTTP_FLAG_SECURE : 0));
    require(bool(call));
    DWORD redirects = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
    require(WinHttpSetOption(call.get(), WINHTTP_OPTION_REDIRECT_POLICY, &redirects, sizeof(redirects)) != FALSE);
    DWORD noAuth = WINHTTP_DISABLE_AUTHENTICATION;
    require(WinHttpSetOption(call.get(), WINHTTP_OPTION_DISABLE_FEATURE, &noAuth, sizeof(noAuth)) != FALSE);
    const auto headers = L"Content-Type: application/json\r\nX-API-Key: " + wide(key) + L"\r\n";
    require(WinHttpSendRequest(call.get(), headers.c_str(), DWORD(headers.size()),
        const_cast<char*>(body.data()), DWORD(body.size()), DWORD(body.size()), 0) != FALSE);
    require(WinHttpReceiveResponse(call.get(), nullptr) != FALSE);
    DWORD status = 0, size = sizeof(status);
    require(WinHttpQueryHeaders(call.get(), WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                               WINHTTP_HEADER_NAME_BY_INDEX, &status, &size, WINHTTP_NO_HEADER_INDEX) != FALSE);
    if (status != 200) throw std::runtime_error("Online model returned HTTP " + std::to_string(status));
    std::string json;
    char buffer[2048]; DWORD received = 0;
    do {
        if (cancelled) throw std::runtime_error("Generation cancelled");
        require(WinHttpReadData(call.get(), buffer, sizeof(buffer), &received) != FALSE);
        if (json.size() + received > 16384) throw std::runtime_error("Online model response too large");
        json.append(buffer, received);
    } while (received);
    if (cancelled) throw std::runtime_error("Generation cancelled");
    return AutomaticModelRouting::decodeResponse(json, budget);
}
}
