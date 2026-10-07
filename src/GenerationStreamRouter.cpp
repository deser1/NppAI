#include "GenerationStreamRouter.h"

GenerationStreamRouter::GenerationStreamRouter(
    ThoughtUpdate thoughtUpdate, CodeAppend codeAppend, CodeRemove codeRemove,
    ProgressUpdate progressUpdate, AssistantUpdate assistantUpdate)
    : thoughtUpdate_(std::move(thoughtUpdate)),
      codeAppend_(std::move(codeAppend)),
      codeRemove_(std::move(codeRemove)),
      progressUpdate_(std::move(progressUpdate)),
      assistantUpdate_(std::move(assistantUpdate)) {}

void GenerationStreamRouter::onToken(char c, bool isThought) {
    if (isThought) {
        thoughtBuffer_ += c;
        if (thoughtUpdate_ && (c == '\n' || thoughtBuffer_.size() % 20 == 0))
            thoughtUpdate_(thoughtBuffer_);
        return;
    }
    assistantBuffer_ += c;
    if (assistantUpdate_ && (c == '\n' || assistantBuffer_.size() % 20 == 0))
        assistantUpdate_(assistantBuffer_);
    if (codeAppend_)
        codeAppend_(c);
}

void GenerationStreamRouter::onRemove(int count) {
    if (count <= 0)
        return;
    const size_t removeCount = static_cast<size_t>(count);
    if (removeCount >= assistantBuffer_.size())
        assistantBuffer_.clear();
    else
        assistantBuffer_.erase(assistantBuffer_.size() - removeCount);
    if (assistantUpdate_)
        assistantUpdate_(assistantBuffer_);
    if (codeRemove_)
        codeRemove_(count);
}

const std::string& GenerationStreamRouter::thoughtBuffer() const {
    return thoughtBuffer_;
}

const std::string& GenerationStreamRouter::assistantBuffer() const {
    return assistantBuffer_;
}

void GenerationStreamRouter::onProgress(GenerationProgressEvent::Type type,
                                        const std::string& message) {
    if (progressUpdate_)
        progressUpdate_({type, message});
}
