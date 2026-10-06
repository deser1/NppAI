#include "WorkspaceFileReader.h"
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {
bool check(bool condition, const char* message) {
    if (!condition) std::cerr << "FAIL: " << message << "\n";
    return condition;
}
}

int main() {
    bool ok = true;
    const auto base = std::filesystem::temp_directory_path() / "nppai_workspace_reader";
    const auto workspace = base / "workspace";
    std::filesystem::remove_all(base);
    std::filesystem::create_directories(workspace / "src");

    { std::ofstream(workspace / "src" / "main.cpp") << "int main() { return 0; }"; }
    { std::ofstream(base / "secret.txt") << "outside"; }
    { std::ofstream(workspace / "large.txt") << "123456789"; }

    WorkspaceFileReader reader(workspace, 8 * 1024);
    std::string content, error;
    ok &= check(reader.read("src/main.cpp", content, error) &&
                    content == "int main() { return 0; }",
                "relative file inside workspace can be read");

    content.clear();
    error.clear();
    ok &= check(!reader.read("../secret.txt", content, error) && content.empty(),
                "parent traversal outside workspace is rejected");

    content.clear();
    error.clear();
    ok &= check(!reader.read(base / "secret.txt", content, error) && content.empty(),
                "absolute path outside workspace is rejected");

    WorkspaceFileReader limited(workspace, 4);
    content.clear();
    error.clear();
    ok &= check(!limited.read("large.txt", content, error) && content.empty(),
                "configured file-size limit is enforced");

    std::filesystem::remove_all(base);
    return ok ? 0 : 1;
}
