#pragma once
#include <functional>
#include <string>

struct GenerationProgressEvent {
    enum class Type {
        TaskStarted,
        ContextReady,
        GenerationStarted,
        GenerationCompleted,
        GenerationCancelled,
        GenerationFailed,
        FileRead,
        FileChanged,
        BuildStarted,
        BuildFailed,
        BuildPassed,
        TestStarted,
        TestFailed,
        TestPassed,
        TaskCompleted
    };

    Type type;
    std::string message;
};

class GenerationStreamRouter {
public:
    using ThoughtUpdate = std::function<void(const std::string&)>;
    using CodeAppend = std::function<void(char)>;
    using CodeRemove = std::function<void(int)>;
    using ProgressUpdate = std::function<void(const GenerationProgressEvent&)>;
    using AssistantUpdate = std::function<void(const std::string&)>;

    GenerationStreamRouter(ThoughtUpdate thoughtUpdate, CodeAppend codeAppend,
                           CodeRemove codeRemove, ProgressUpdate progressUpdate = nullptr,
                           AssistantUpdate assistantUpdate = nullptr);

    void onToken(char c, bool isThought);
    void onRemove(int count);
    const std::string& thoughtBuffer() const;
    const std::string& assistantBuffer() const;
    void onProgress(GenerationProgressEvent::Type type, const std::string& message = "");

private:
    std::string thoughtBuffer_;
    std::string assistantBuffer_;
    ThoughtUpdate thoughtUpdate_;
    CodeAppend codeAppend_;
    CodeRemove codeRemove_;
    ProgressUpdate progressUpdate_;
    AssistantUpdate assistantUpdate_;
};
