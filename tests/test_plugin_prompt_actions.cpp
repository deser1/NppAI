#include "PluginPromptActions.h"
#include <iostream>
#include <string>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << "\n";
        return false;
    }
    return true;
}
}

int main() {
    bool ok = true;

    ok &= check(PluginPromptActions::appendSelection("", "int x = 1;") ==
                    "```\r\nint x = 1;\r\n```\r\n",
                "selection is fenced for an empty prompt");

    ok &= check(PluginPromptActions::appendSelection(
                    "Explain this", "int x = 1;") ==
                    "Explain this\r\n```\r\nint x = 1;\r\n```\r\n",
                "selection is appended to an existing prompt");

    const std::string multiline = "if (ready) {\n  run();\n}";
    ok &= check(PluginPromptActions::appendSelection("", multiline) ==
                    "```\r\nif (ready) {\n  run();\n}\r\n```\r\n",
                "multiline code is preserved");

    const std::string utf8 = u8"std::string tekst = \"zażółć\";";
    ok &= check(PluginPromptActions::appendSelection("Popraw:", utf8) ==
                    std::string("Popraw:\r\n```\r\n") + utf8 +
                        "\r\n```\r\n",
                "UTF-8 selection is preserved");

    if (!ok)
        return 1;
    std::cout << "Plugin prompt action integration tests passed.\n";
    return 0;
}
