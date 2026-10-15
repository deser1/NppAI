#pragma once
#include <atomic>
#include <string>
namespace RemoteModelClient {
std::string generate(const std::string& endpoint, const std::string& apiKey,
                     const std::string& requestJson, int budget,
                     const std::atomic<bool>& cancelled);
}
