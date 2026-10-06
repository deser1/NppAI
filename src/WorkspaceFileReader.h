#pragma once
#include <cstddef>
#include <filesystem>
#include <string>

class WorkspaceFileReader {
public:
    explicit WorkspaceFileReader(std::filesystem::path workspaceRoot,
                                 std::size_t maxBytes = 1024 * 1024);

    bool read(const std::filesystem::path& requestedPath, std::string& content,
              std::string& error) const;

private:
    std::filesystem::path workspaceRoot_;
    std::size_t maxBytes_;
};
