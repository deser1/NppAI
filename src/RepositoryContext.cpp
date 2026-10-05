#include "RepositoryContext.h"

namespace RepositoryContext {

std::string buildQuery(const std::string& prompt, const std::string& currentFilePath) {
    if (currentFilePath.empty())
        return prompt;

    const std::size_t separator = currentFilePath.find_last_of("\\\/");
    const std::string filename =
        separator == std::string::npos ? currentFilePath : currentFilePath.substr(separator + 1);
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
