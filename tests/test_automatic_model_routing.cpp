#include "AutomaticModelRouting.h"
#ifdef _WIN32
#include "RemoteModelClient.h"
#endif
#include <atomic>
#include <iostream>
#include <stdexcept>
using namespace AutomaticModelRouting;
void check(bool value) { if (!value) throw std::runtime_error("test assertion failed"); }
int main(int argc, char** argv) {
    (void)argc; (void)argv;
    try {
#ifdef _WIN32
        if (argc == 3 && std::string(argv[1]) == "--http") {
            std::atomic<bool> cancelled{false};
            std::cout << RemoteModelClient::generate(argv[2], "test-key",
                encodeRequest("task", "context", "def f(x):\n    ", 3), 3, cancelled);
            return 0;
        }
#endif
        check(parseMode("") == Mode::Automatic && parseMode("hybrid") == Mode::Automatic);
        check(select(Mode::Automatic, true, 256, 256, true).backend == Backend::Local);
        check(select(Mode::Automatic, true, 257, 256, true).backend == Backend::Online);
        check(select(Mode::Automatic, false, 10, 256, true).backend == Backend::Online);
        check(select(Mode::Automatic, true, 257, 256, false).backend == Backend::Unavailable);
        check(select(Mode::Automatic, false, 0, 0, false).backend == Backend::Unavailable);
        check(select(Mode::Local, false, 10, 256, true).backend == Backend::Unavailable);
        check(select(Mode::Local, true, 257, 256, true).backend == Backend::Unavailable);
        check(select(Mode::Online, true, 10, 256, false).backend == Backend::Unavailable);
        check(select(Mode::Online, true, 10, 256, true).backend == Backend::Online);
        const auto json = encodeRequest("a\"\n", "\\ctx", "def f():\n    ", 3);
        check(json.find("a\\\"\\u000a") != std::string::npos);
        check(json.find("\"assistant_prefix\":\"def f():\\u000a    \"") != std::string::npos);
        const std::string base = "{\"protocol_version\":1,\"model_id\":\"m\",\"output\":\"a\\n\\u017c\\ud83d\\ude00\",\"generated_tokens\":3,\"finish_reason\":\"length\"}";
        check(decodeResponse(base, 3) == "a\n\xc5\xbc\xf0\x9f\x98\x80");
        const std::string bad[] = {
            "{}", base + "x", base.substr(0, base.size() - 1) + ",\"output\":\"b\"}",
            "{\"protocol_version\":true}", "{\"protocol_version\":01}",
            "{\"protocol_version\":1.0}", "{\"unknown\":{}}",
            "{\"output\":\"\\ud800\"}", "{\"output\":\"\\udc00\"}",
            "{\"output\":\"\xc0\x80\"}", "{\"output\":\"\xed\xa0\x80\"}",
            "{\"output\":\"\n\"}", std::string(16385, ' '),
            "{\"protocol_version\":1,\"model_id\":\"m\",\"output\":\"" + std::string(2049, 'a') +
                "\",\"generated_tokens\":3,\"finish_reason\":\"length\"}"
        };
        for (const auto& input : bad) {
            bool rejected = false;
            try { decodeResponse(input, 3); } catch (const std::exception&) { rejected = true; }
            check(rejected);
        }
        bool rejected = false;
        try { decodeResponse(base, 2); } catch (const std::exception&) { rejected = true; }
        check(rejected);
        std::cout << "Automatic model selection and protocol tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
