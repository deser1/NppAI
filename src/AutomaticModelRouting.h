#pragma once
#include <cstddef>
#include <string>

namespace AutomaticModelRouting {
enum class Mode { Automatic, Local, Online };
enum class Backend { Local, Online, Unavailable };
struct Decision { Backend backend; std::string reason; };
Mode parseMode(const std::string& value);
Decision select(Mode mode, bool localReady, std::size_t promptBytes,
                std::size_t localContext, bool remoteConfigured);
std::string encodeRequest(const std::string& task, const std::string& context,
                          const std::string& assistantPrefix, int maxTokens);
std::string decodeResponse(const std::string& json, int maxTokens);
}
