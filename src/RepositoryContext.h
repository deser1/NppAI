#pragma once
#include <cstddef>
#include <string>

namespace RepositoryContext {
std::string buildQuery(const std::string& prompt, const std::string& currentFilePath);
std::string limit(const std::string& context, std::size_t maxBytes = 12000);
}
