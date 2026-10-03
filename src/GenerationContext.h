#pragma once
#include <cstddef>
#include <string>

namespace GenerationContext {
std::string select(const std::string& retrievedContext,
                   const std::string& legacyContext,
                   std::size_t legacyLimit = 1000);
}
