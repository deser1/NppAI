#include "AutomaticModelRouting.h"
#include <map>
#include <stdexcept>

namespace AutomaticModelRouting {
Mode parseMode(const std::string& value) {
    if (value.empty() || value == "auto" || value == "hybrid") return Mode::Automatic;
    if (value == "local") return Mode::Local;
    if (value == "online") return Mode::Online;
    throw std::runtime_error("Invalid NPPAI_PLUGIN_MODE (auto/local/online)");
}
Decision select(Mode mode, bool ready, std::size_t bytes, std::size_t capacity, bool remote) {
    const bool fits = ready && capacity > 0 && bytes <= capacity;
    if (mode == Mode::Online)
        return remote ? Decision{Backend::Online, "Online: explicitly selected"}
                      : Decision{Backend::Unavailable, "Online model is not configured"};
    if (fits) return {Backend::Local, "Local: model loaded and prompt fits context"};
    if (mode == Mode::Automatic && remote)
        return {Backend::Online, ready ? "Online: prompt exceeds local context"
                                      : "Online: local model unavailable"};
    return {Backend::Unavailable, ready ? "Prompt exceeds local context; no permitted online backend"
                                      : "Local model unavailable; no permitted online backend"};
}
namespace {
std::string quote(const std::string& value) {
    std::string result = "\"";
    const char* hex = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') { result += '\\'; result += char(c); }
        else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else result += char(c);
    }
    return result + '"';
}
// Deliberately narrow reader: the version-1 response is a flat, fixed schema.
// Reject nested values, duplicates, unknown fields and trailing data.
class Reader {
    const std::string& text;
    std::size_t pos = 0;
    char peek() const { return pos < text.size() ? text[pos] : '\0'; }
    bool take(char c) { if (pos < text.size() && text[pos] == c) { ++pos; return true; } return false; }
    void expect(char c) { if (!take(c)) fail(); }
    void space() { while (peek() == ' ' || peek() == '\n' || peek() == '\r' || peek() == '\t') ++pos; }
    static void fail() { throw std::runtime_error("Invalid generation response protocol"); }
    unsigned hex4() {
        unsigned value = 0;
        for (int i = 0; i < 4; ++i) {
            if (pos == text.size()) fail();
            const char c = text[pos++];
            unsigned digit;
            if (c >= '0' && c <= '9') digit = c - '0';
            else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
            else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
            else { fail(); return 0; }
            value = value * 16 + digit;
        }
        return value;
    }
    static void utf8(std::string& value, unsigned cp) {
        if (cp < 128) value += char(cp);
        else if (cp < 2048) { value += char(0xc0 | (cp >> 6)); value += char(0x80 | (cp & 63)); }
        else if (cp < 65536) {
            value += char(0xe0 | (cp >> 12)); value += char(0x80 | ((cp >> 6) & 63)); value += char(0x80 | (cp & 63));
        } else {
            value += char(0xf0 | (cp >> 18)); value += char(0x80 | ((cp >> 12) & 63));
            value += char(0x80 | ((cp >> 6) & 63)); value += char(0x80 | (cp & 63));
        }
    }
    std::string string() {
        expect('"'); std::string value;
        while (pos < text.size()) {
            unsigned char c = text[pos++];
            if (c == '"') return value;
            if (c < 32) fail();
            if (c == '\\') {
                if (pos == text.size()) fail();
                c = text[pos++];
                switch (c) {
                case '"': case '\\': case '/': value += char(c); break;
                case 'b': value += '\b'; break;
                case 'f': value += '\f'; break;
                case 'n': value += '\n'; break;
                case 'r': value += '\r'; break;
                case 't': value += '\t'; break;
                case 'u': {
                    unsigned cp = hex4();
                    if (cp >= 0xd800 && cp <= 0xdbff) {
                        expect('\\'); expect('u'); unsigned low = hex4();
                        if (low < 0xdc00 || low > 0xdfff) fail();
                        cp = 65536 + ((cp - 0xd800) << 10) + low - 0xdc00;
                    } else if (cp >= 0xdc00 && cp <= 0xdfff) fail();
                    utf8(value, cp); break;
                }
                default: fail();
                }
            } else if (c < 128) value += char(c);
            else {
                // Validate raw UTF-8, including overlong forms and surrogates.
                const unsigned length = c >= 0xc2 && c <= 0xdf ? 2 :
                    c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
                if (!length) fail();
                unsigned cp = c & ((1u << (7 - length)) - 1);
                for (unsigned i = 1; i < length; ++i) {
                    if (pos == text.size()) fail();
                    unsigned char next = text[pos++];
                    if ((next & 0xc0) != 0x80) fail();
                    cp = (cp << 6) | (next & 63);
                }
                if (cp < (length == 2 ? 128u : length == 3 ? 2048u : 65536u) ||
                    cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff)) fail();
                utf8(value, cp);
            }
        }
        fail(); return {};
    }
    static std::size_t characters(const std::string& value) {
        std::size_t count = 0;
        for (unsigned char c : value) if ((c & 0xc0) != 0x80) ++count;
        return count;
    }
    int integer() {
        if (peek() < '0' || peek() > '9') fail();
        int value = text[pos++] - '0';
        if (!value && peek() >= '0' && peek() <= '9') fail();
        while (peek() >= '0' && peek() <= '9') {
            value = value * 10 + text[pos++] - '0';
            if (value > 512) fail();
        }
        return value;
    }
public:
    explicit Reader(const std::string& input) : text(input) {}
    std::string read(int budget) {
        if (text.size() > 16384) fail();
        std::map<std::string, bool> seen;
        std::string output, model, finish;
        int version = -1, tokens = -1;
        space(); expect('{');
        do {
            space(); auto key = string(); space(); expect(':'); space();
            if (seen[key]) fail();
            seen[key] = true;
            if (key == "protocol_version") version = integer();
            else if (key == "generated_tokens") tokens = integer();
            else if (key == "model_id") model = string();
            else if (key == "finish_reason") finish = string();
            else if (key == "output") output = string();
            else fail();
            space(); if (take('}')) break; expect(',');
        } while (true);
        space();
        if (pos != text.size() || seen.size() != 5 || version != 1 || tokens < 0 ||
            tokens > budget || model.empty() || characters(model) > 128 || characters(output) > 2048 ||
            (finish != "eos" && finish != "length") || output.find('\0') != std::string::npos) fail();
        return output;
    }
};
}
std::string encodeRequest(const std::string& task, const std::string& context,
                          const std::string& prefix, int budget) {
    if (task.empty() || task.size() > 100000 || context.size() > 100000 ||
        prefix.size() > 1024 || budget < 1 || budget > 512)
        throw std::runtime_error("Generation request exceeds protocol limits");
    return "{\"protocol_version\":1,\"prompt\":" + quote(task) + ",\"context\":" + quote(context) +
        ",\"assistant_prefix\":" + quote(prefix) + ",\"max_new_tokens\":" + std::to_string(budget) + "}";
}
std::string decodeResponse(const std::string& json, int budget) { return Reader(json).read(budget); }
}
