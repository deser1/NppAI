#include "GenerationContext.h"

namespace GenerationContext {

std::string select(const std::string& retrievedContext,
                   const std::string& legacyContext,
                   std::size_t legacyLimit) {
    if (!retrievedContext.empty())
        return retrievedContext;

    if (legacyContext.size() <= legacyLimit)
        return legacyContext;

    return legacyContext.substr(legacyContext.size() - legacyLimit);
}

}
