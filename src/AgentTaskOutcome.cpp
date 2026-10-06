#include "AgentTaskOutcome.h"
#include <sstream>

namespace {
std::string escapeJson(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
        case '\\': out += "\\\\"; break;
        case '"': out += "\\\""; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                const char hex[] = "0123456789abcdef";
                out += "\\u00";
                out += hex[(static_cast<unsigned char>(ch) >> 4) & 0x0f];
                out += hex[static_cast<unsigned char>(ch) & 0x0f];
            } else {
                out += ch;
            }
        }
    }
    return out;
}

const char* statusName(AgentTaskStatus status) {
    switch (status) {
    case AgentTaskStatus::Succeeded: return "succeeded";
    case AgentTaskStatus::RolledBack: return "rolled_back";
    default: return "failed";
    }
}
}

std::string AgentTaskOutcome::toJsonLine() const {
    std::ostringstream out;
    out << "{\"task\":\"" << escapeJson(task)
        << "\",\"status\":\"" << statusName(status)
        << "\",\"attempts\":" << attempts
        << ",\"patch\":\"" << escapeJson(patch)
        << "\",\"diagnostics\":[";
    for (std::size_t i = 0; i < diagnostics.size(); ++i) {
        if (i) out << ',';
        out << "\"" << escapeJson(diagnostics[i]) << "\"";
    }
    out << "]}";
    return out.str();
}

AgentTaskOutcome AgentTaskOutcomeBuilder::fromRepairResult(
    const std::string& task,
    const StructuredPatch& patch,
    const RepairLoopResult& result,
    bool rolledBack) {
    AgentTaskOutcome outcome;
    outcome.task = task;
    outcome.status = result.succeeded ? AgentTaskStatus::Succeeded
                                      : (rolledBack ? AgentTaskStatus::RolledBack
                                                    : AgentTaskStatus::Failed);
    outcome.attempts = result.attempts;
    outcome.patch = patch.unifiedDiff();
    for (const auto& attempt : result.history)
        outcome.diagnostics.push_back(attempt.diagnostics);
    return outcome;
}
