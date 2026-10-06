#include "AgentFeedbackLoop.h"

AgentFeedbackLoop::AgentFeedbackLoop(GenerationStreamRouter* progress)
    : progress_(progress) {}

AgentValidationResult AgentFeedbackLoop::validate(const Command& build,
                                                  const Command& tests) const {
    AgentValidationResult result;

    if (progress_) progress_->onProgress(GenerationProgressEvent::Type::BuildStarted,
                                        "Validating proposed workspace changes.");
    result.build = build();
    if (!result.build.succeeded()) {
        result.repairFeedback = result.build.diagnostics;
        if (progress_) progress_->onProgress(GenerationProgressEvent::Type::BuildFailed,
                                            result.build.diagnostics);
        return result;
    }
    if (progress_) progress_->onProgress(GenerationProgressEvent::Type::BuildPassed,
                                        "Build completed successfully.");

    if (progress_) progress_->onProgress(GenerationProgressEvent::Type::TestStarted,
                                        "Running tests for proposed changes.");
    result.tests = tests();
    if (!result.tests.succeeded()) {
        result.repairFeedback = result.tests.diagnostics;
        if (progress_) progress_->onProgress(GenerationProgressEvent::Type::TestFailed,
                                            result.tests.diagnostics);
        return result;
    }

    if (progress_) {
        progress_->onProgress(GenerationProgressEvent::Type::TestPassed,
                             "Tests completed successfully.");
        progress_->onProgress(GenerationProgressEvent::Type::TaskCompleted,
                             "Patch validation completed.");
    }
    result.succeeded = true;
    return result;
}
