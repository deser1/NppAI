#pragma once
#include "AgentFeedbackLoop.h"
#include <filesystem>
#include <functional>

class PythonValidationProcess {
public:
    explicit PythonValidationProcess(const std::filesystem::path& bundledScript);
    AgentCommandResult run(const std::string& candidate, bool tests,
                           const std::function<bool()>& cancelled) const;
private:
    std::filesystem::path python_, script_;
    std::wstring image_;
    std::string cases_;
};
