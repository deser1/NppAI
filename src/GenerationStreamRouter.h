#pragma once
#include <functional>
#include <string>

class GenerationStreamRouter {
public:
    using ThoughtUpdate = std::function<void(const std::string&)>;
    using CodeAppend = std::function<void(char)>;
    using CodeRemove = std::function<void(int)>;

    GenerationStreamRouter(ThoughtUpdate thoughtUpdate, CodeAppend codeAppend,
                           CodeRemove codeRemove);

    void onToken(char c, bool isThought);
    void onRemove(int count);
    const std::string& thoughtBuffer() const;

private:
    std::string thoughtBuffer_;
    ThoughtUpdate thoughtUpdate_;
    CodeAppend codeAppend_;
    CodeRemove codeRemove_;
};
