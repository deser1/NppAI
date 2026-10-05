#include "RepositoryContext.h"
#include <filesystem>

namespace RepositoryContext {

std::string buildQuery(const std::string& prompt, const std::string& currentFilePath) {
    if (currentFilePath.empty())
        return prompt;

    std::filesystem::path path(currentFilePath);
    const std::string filename = path.filename().string();
    if (filename.empty())
        return prompt;

    return prompt + "\ncurrent file " + filename;
}

std::string limit(const std::string& context, std::size_t maxBytes) {
    if (maxBytes == 0 || context.empty())
        return {};
    if (context.size() <= maxBytes)
        return context;

    std::size_t end = maxBytes;
    while (end > 0 &&
           (static_cast<unsigned char>(context[end]) & 0xC0) == 0x80) {
        --end;
    }
    return context.substr(0, end);
}

}
