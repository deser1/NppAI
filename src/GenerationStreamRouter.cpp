#include "GenerationStreamRouter.h"

GenerationStreamRouter::GenerationStreamRouter(
    ThoughtUpdate thoughtUpdate, CodeAppend codeAppend, CodeRemove codeRemove)
    : thoughtUpdate_(std::move(thoughtUpdate)),
      codeAppend_(std::move(codeAppend)),
      codeRemove_(std::move(codeRemove)) {}

void GenerationStreamRouter::onToken(char c, bool isThought) {
    if (isThought) {
        thoughtBuffer_ += c;
        if (thoughtUpdate_ && (c == '\n' || thoughtBuffer_.size() % 20 == 0))
            thoughtUpdate_(thoughtBuffer_);
        return;
    }
    if (codeAppend_)
        codeAppend_(c);
}

void GenerationStreamRouter::onRemove(int count) {
    if (count > 0 && codeRemove_)
        codeRemove_(count);
}

const std::string& GenerationStreamRouter::thoughtBuffer() const {
    return thoughtBuffer_;
}
