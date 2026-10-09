// Generate ranked source predictions directly from the production RAGManager.
#include "RAGManager.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <stdexcept>
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

// Decode JSON string escapes in the two required query fields.
std::string decodeJsonString(const std::string& encoded) {
    std::string result;
    auto appendUtf8 = [&](unsigned int cp) {
        if (cp <= 0x7F) result += static_cast<char>(cp);
        else if (cp <= 0x7FF) {
            result += static_cast<char>(0xC0 | (cp >> 6));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        } else if (cp <= 0xFFFF) {
            result += static_cast<char>(0xE0 | (cp >> 12));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        } else {
            result += static_cast<char>(0xF0 | (cp >> 18));
            result += static_cast<char>(0x80 | ((cp >> 12) & 0x3F));
            result += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
            result += static_cast<char>(0x80 | (cp & 0x3F));
        }
    };
    auto hex4 = [&](size_t pos) -> unsigned int {
        if (pos + 4 > encoded.size()) throw std::runtime_error("truncated Unicode escape");
        unsigned int value = 0;
        for (size_t j = 0; j < 4; ++j) {
            char ch = encoded[pos + j];
            unsigned int digit;
            if (ch >= '0' && ch <= '9') digit = ch - '0';
            else if (ch >= 'a' && ch <= 'f') digit = ch - 'a' + 10;
            else if (ch >= 'A' && ch <= 'F') digit = ch - 'A' + 10;
            else throw std::runtime_error("invalid Unicode escape");
            value = (value << 4) | digit;
        }
        return value;
    };
    for (size_t i = 0; i < encoded.size(); ++i) {
        if (encoded[i] != '\\') {
            if (static_cast<unsigned char>(encoded[i]) < 0x20)
                throw std::runtime_error("unescaped control character");
            result += encoded[i];
            continue;
        }
        if (++i >= encoded.size()) throw std::runtime_error("truncated escape");
        switch (encoded[i]) {
        case '"': result += '"'; break;
        case '\\': result += '\\'; break;
        case '/': result += '/'; break;
        case 'n': result += '\n'; break;
        case 'r': result += '\r'; break;
        case 't': result += '\t'; break;
        case 'b': result += '\b'; break;
        case 'f': result += '\f'; break;
        case 'u': {
            unsigned int cp = hex4(i + 1);
            i += 4;
            if (cp >= 0xD800 && cp <= 0xDBFF) {
                if (i + 6 >= encoded.size() || encoded[i + 1] != '\\' || encoded[i + 2] != 'u')
                    throw std::runtime_error("missing low surrogate");
                unsigned int low = hex4(i + 3);
                if (low < 0xDC00 || low > 0xDFFF)
                    throw std::runtime_error("invalid low surrogate");
                cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
                i += 6;
            } else if (cp >= 0xDC00 && cp <= 0xDFFF) {
                throw std::runtime_error("unpaired low surrogate");
            }
            appendUtf8(cp);
            break;
        }
        default: throw std::runtime_error("unsupported JSON escape");
        }
    }
    return result;
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
    const std::regex idPattern(R"rgx("query_id"\s*:\s*"((?:\\.|[^"\\])*)")rgx");
    const std::regex queryPattern(R"rgx("query"\s*:\s*"((?:\\.|[^"\\])*)")rgx");
    std::string line;
    while (std::getline(judgments, line)) {
        if (line.empty()) continue;
        std::smatch id, query;
        if (!std::regex_search(line, id, idPattern) || !std::regex_search(line, query, queryPattern)) {
            std::cerr << "Invalid judgment record (query_id/query missing or escaped)\n";
            return 2;
        }
        std::string decodedId, decodedQuery;
        try {
            decodedId = decodeJsonString(id[1].str());
            decodedQuery = decodeJsonString(query[1].str());
        } catch (const std::exception& error) {
            std::cerr << "Invalid JSON string escape: " << error.what() << "\n";
            return 2;
        }
        const auto sources = rag.retrieveRankedSources(decodedQuery, topK);
        std::cout << "{\"query_id\":\"" << jsonEscape(decodedId) << "\",\"ranked_sources\":[";
        for (size_t i = 0; i < sources.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << '"' << jsonEscape(sources[i]) << '"';
        }
        std::cout << "]}\n";
    }
    return 0;
}
