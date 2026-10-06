#include "WorkspaceFileReader.h"
#include <fstream>

namespace {
bool isWithin(const std::filesystem::path& root, const std::filesystem::path& candidate) {
    auto rootIt = root.begin();
    auto candidateIt = candidate.begin();
    for (; rootIt != root.end(); ++rootIt, ++candidateIt) {
        if (candidateIt == candidate.end() || *rootIt != *candidateIt)
            return false;
    }
    return true;
}
}

WorkspaceFileReader::WorkspaceFileReader(std::filesystem::path workspaceRoot,
                                         std::size_t maxBytes)
    : workspaceRoot_(std::filesystem::weakly_canonical(std::move(workspaceRoot))),
      maxBytes_(maxBytes) {}

bool WorkspaceFileReader::read(const std::filesystem::path& requestedPath,
                               std::string& content, std::string& error) const {
    content.clear();
    error.clear();

    if (workspaceRoot_.empty() || requestedPath.empty()) {
        error = "Workspace root and requested path must not be empty.";
        return false;
    }

    std::error_code ec;
    const auto candidate = std::filesystem::weakly_canonical(
        requestedPath.is_absolute() ? requestedPath : workspaceRoot_ / requestedPath, ec);
    if (ec || !isWithin(workspaceRoot_, candidate)) {
        error = "Requested file is outside the workspace.";
        return false;
    }

    if (!std::filesystem::is_regular_file(candidate, ec) || ec) {
        error = "Requested path is not a readable regular file.";
        return false;
    }

    const auto size = std::filesystem::file_size(candidate, ec);
    if (ec || size > maxBytes_) {
        error = "Requested file exceeds the configured read limit.";
        return false;
    }

    std::ifstream input(candidate, std::ios::binary);
    if (!input) {
        error = "Requested file could not be opened.";
        return false;
    }

    content.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    if (content.size() > maxBytes_) {
        content.clear();
        error = "Requested file exceeds the configured read limit.";
        return false;
    }
    return true;
}
