#include "RAGManager.h"
#include <cctype>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace {
std::string languageFor(const std::filesystem::path& path) {
    static const std::unordered_map<std::string, std::string> languages = {
        {".c","c"},{".h","cpp"},{".cc","cpp"},{".cpp","cpp"},{".cxx","cpp"},{".hpp","cpp"},
        {".py","python"},{".js","javascript"},{".ts","typescript"},{".tsx","typescript"},
        {".java","java"},{".cs","csharp"},{".go","go"},{".rs","rust"},{".php","php"},
        {".rb","ruby"},{".kt","kotlin"},{".swift","swift"},{".json","json"},{".md","markdown"},
        {".cmake","cmake"},{".yml","yaml"},{".yaml","yaml"},{".xml","xml"},{".html","html"},
        {".css","css"},{".sql","sql"},{".sh","shell"},{".ps1","powershell"}
    };
    auto ext = path.extension().string();
    for (char& c : ext) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    const auto it = languages.find(ext);
    if (it != languages.end()) return it->second;
    if (path.filename() == "CMakeLists.txt") return "cmake";
    return "";
}
bool ignoredDirectory(const std::filesystem::path& path) {
    const auto name = path.filename().string();
    return name == ".git" || name == ".github" || name == "build" || name == "out" ||
           name == "node_modules" || name == ".vs" || name == ".idea";
}
}

size_t RAGManager::indexRepository(const std::string& rootPath) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path root = fs::absolute(rootPath, ec);
    if (ec || !fs::is_directory(root, ec)) return 0;
    size_t indexed = 0;
    fs::recursive_directory_iterator it(root, fs::directory_options::skip_permission_denied, ec), end;
    for (; it != end; it.increment(ec)) {
        if (ec) { ec.clear(); continue; }
        if (it->is_directory(ec)) {
            if (ignoredDirectory(it->path())) it.disable_recursion_pending();
            continue;
        }
        if (!it->is_regular_file(ec)) continue;
        const std::string language = languageFor(it->path());
        if (language.empty()) continue;
        const auto size = it->file_size(ec);
        if (ec || size > 4ULL * 1024ULL * 1024ULL) { ec.clear(); continue; }
        std::ifstream input(it->path(), std::ios::binary);
        if (!input) continue;
        std::ostringstream content;
        content << input.rdbuf();
        std::string source = fs::relative(it->path(), root, ec).generic_string();
        if (ec) { ec.clear(); source = it->path().generic_string(); }
        updateSource(content.str(), source, language);
        ++indexed;
    }
    return indexed;
}
