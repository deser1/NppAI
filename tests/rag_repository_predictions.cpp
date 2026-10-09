// Generate ranked source predictions directly from the production RAGManager.
#include "RAGManager.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <vector>

namespace {
std::string jsonEscape(const std::string& value) {
    std::string out;
    for (unsigned char c : value) {
        switch (c) {
        case '"': out += "\\\""; break;
        case '\\': out += "\\\\"; break;
        case '\n': out += "\\n"; break;
        case '\r': out += "\\r"; break;
        case '\t': out += "\\t"; break;
        default:
            if (c < 0x20) return "";
            out += static_cast<char>(c);
        }
    }
    return out;
}
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: RagRepositoryPredictions <repository-root> <judgments.jsonl> <top-k>\n";
        return 2;
    }
    const int topK = std::atoi(argv[3]);
    if (topK <= 0 || topK > 100) return 2;
    std::ifstream judgments(argv[2]);
    if (!judgments) {
        std::cerr << "Unable to read judgments file\n";
        return 2;
    }
    auto& rag = RAGManager::getInstance();
    rag.clearForTesting();
    const size_t indexed = rag.indexRepository(argv[1]);
    if (!indexed) {
        std::cerr << "No repository documents indexed\n";
        return 1;
    }
    // Simple JSONL reader: query_id and query must be JSON strings on each line.
    // For full JSON escaping and validation, use a proper JSON parser in a follow-up.
    const std::regex idPattern(R"rgx("query_id"\s*:\s*"([^"\\]*)")rgx");
    const std::regex queryPattern(R"rgx("query"\s*:\s*"([^"\\]*)")rgx");
    std::string line;
    while (std::getline(judgments, line)) {
        if (line.empty()) continue;
        std::smatch id, query;
        if (!std::regex_search(line, id, idPattern) || !std::regex_search(line, query, queryPattern)) {
            std::cerr << "Invalid judgment record (query_id/query missing or escaped)\n";
            return 2;
        }
        const auto sources = rag.retrieveRankedSources(query[1].str(), topK);
        std::cout << "{\"query_id\":\"" << jsonEscape(id[1].str()) << "\",\"ranked_sources\":[";
        for (size_t i = 0; i < sources.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '"' << jsonEscape(sources[i]) << '"';
        }
        std::cout << "]}\n";
    }
    return 0;
}
