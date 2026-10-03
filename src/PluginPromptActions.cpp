#include "PluginPromptActions.h"

namespace PluginPromptActions {

std::string appendSelection(const std::string& currentPrompt,
                            const std::string& selection) {
    std::string result = currentPrompt;
    if (!result.empty())
        result += "\r\n";

    result += "```\r\n";
    result += selection;
    result += "\r\n```\r\n";
    return result;
}

}
