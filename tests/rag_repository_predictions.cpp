// Generate ranked source predictions directly from the production RAGManager.
#include "RAGManager.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <cctype>
#include <map>
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


namespace {
class JsonReader {
public:
    explicit JsonReader(const std::string& text) : text_(text) {}
    std::map<std::string, std::string> readJudgment() {
        std::map<std::string, std::string> fields;
        spaces();
        expect('{');
        spaces();
        if (!take('}')) {
            do {
                spaces();
                std::string key = stringValue();
                spaces();
                expect(':');
                spaces();
                if (fields.count(key)) throw std::runtime_error("duplicate JSON field");
                if (key == "query_id" || key == "query") {
                    if (peek() != '"') throw std::runtime_error("query fields must be strings");
                    fields.emplace(key, stringValue());
                } else {
                    skipValue(0);
                    fields.emplace(key, "");
                }
                spaces();
                if (take('}')) break;
                expect(',');
            } while (true);
        }
        spaces();
        if (pos_ != text_.size()) throw std::runtime_error("trailing JSON data");
        if (!fields.count("query_id") || !fields.count("query"))
            throw std::runtime_error("missing query_id or query");
        return fields;
    }
private:
    const std::string& text_;
    size_t pos_ = 0;
    void spaces() {
        while (pos_ < text_.size() && (text_[pos_] == ' ' || text_[pos_] == '\t' ||
               text_[pos_] == '\r' || text_[pos_] == '\n')) ++pos_;
    }
    char peek() const { return pos_ < text_.size() ? text_[pos_] : '\0'; }
    bool take(char c) { if (peek() == c && pos_ < text_.size()) { ++pos_; return true; } return false; }
    void expect(char c) { if (!take(c)) throw std::runtime_error("invalid JSON syntax"); }
    std::string stringValue() {
        expect('"');
        std::string encoded;
        bool escaped = false;
        while (pos_ < text_.size()) {
            char c = text_[pos_++];
            if (c == '"' && !escaped) return decodeJsonString(encoded);
            encoded += c;
            if (c == '\\' && !escaped) escaped = true;
            else escaped = false;
        }
        throw std::runtime_error("unterminated JSON string");
    }
    void literal(const char* word) {
        while (*word) { if (pos_ == text_.size() || text_[pos_++] != *word++)
            throw std::runtime_error("invalid JSON literal"); }
    }
    void number() {
        take('-');
        if (take('0')) {
            if (peek() >= '0' && peek() <= '9') throw std::runtime_error("leading zero");
        } else {
            if (peek() < '1' || peek() > '9') throw std::runtime_error("invalid number");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (take('.')) {
            if (peek() < '0' || peek() > '9') throw std::runtime_error("invalid fraction");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
        if (take('e') || take('E')) {
            if (!take('+')) take('-');
            if (peek() < '0' || peek() > '9') throw std::runtime_error("invalid exponent");
            while (peek() >= '0' && peek() <= '9') ++pos_;
        }
    }
    void skipValue(int depth) {
        if (depth > 64) throw std::runtime_error("JSON nesting limit exceeded");
        spaces();
        if (peek() == '"') { stringValue(); return; }
        if (take('{')) {
            spaces();
            if (take('}')) return;
            do {
                spaces(); stringValue(); spaces(); expect(':');
                skipValue(depth + 1); spaces();
                if (take('}')) return;
                expect(',');
            } while (true);
        }
        if (take('[')) {
            spaces();
            if (take(']')) return;
            do {
                skipValue(depth + 1); spaces();
                if (take(']')) return;
                expect(',');
            } while (true);
        }
        if (peek() == 't') { literal("true"); return; }
        if (peek() == 'f') { literal("false"); return; }
        if (peek() == 'n') { literal("null"); return; }
        number();
    }
};
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
    std::string line;
    while (std::getline(judgments, line)) {
        if (line.empty()) continue;
        std::string decodedId, decodedQuery;
        try {
            const auto fields = JsonReader(line).readJudgment();
            decodedId = fields.at("query_id");
            decodedQuery = fields.at("query");
        } catch (const std::exception& error) {
            std::cerr << "Invalid JSON string escape or judgment record: " << error.what() << "\n";
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
