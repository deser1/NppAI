cmake_minimum_required(VERSION 3.15)
project(NppAIPlugin VERSION 0.3.0 LANGUAGES CXX)

option(NPPAI_TEST_HOST_SEAM "Expose model-free plugin host initialization for integration tests" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
enable_testing()

if (NOT CMAKE_GENERATOR_PLATFORM)
    set(CMAKE_GENERATOR_PLATFORM x64)
endif()

find_package(OpenMP)

add_library(NppAI SHARED
    src/NppAI_Core.cpp
    src/NppAI_Core.rc
    src/PluginDefinition.cpp
    src/GenerationContext.cpp
    src/GenerationStreamRouter.cpp
    src/GenerationTracking.cpp
    src/PluginPromptActions.cpp
    src/AIManager.cpp
    src/AIPromptPipeline.cpp
    src/RAGManager.cpp
    src/RepositoryIndexer.cpp
    src/RepositoryContext.cpp
    src/WorkspaceFileReader.cpp
    src/StructuredPatch.cpp
    src/PatchReview.cpp
    src/PatchRollback.cpp
    src/AgentFeedbackLoop.cpp
    src/BoundedRepairLoop.cpp
    src/TelemetryManager.cpp
    src/TelemetryRedaction.cpp
    src/NppAIEngine.cpp
)

add_custom_command(TARGET NppAI PRE_BUILD COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/increment_version.cmake COMMENT "Incrementing plugin version...")

target_include_directories(NppAI PRIVATE src src/DockingFeature)
target_link_libraries(NppAI PRIVATE user32 gdi32 shlwapi d3d11 d3dcompiler winhttp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(NppAI PRIVATE OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    target_compile_options(NppAI PRIVATE /EHsc /W3 /WX- /utf-8)
    target_link_options(NppAI PRIVATE /OPT:REF /OPT:ICF)
    if (CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
        target_compile_options(NppAI PRIVATE /O2)
        target_link_options(NppAI PRIVATE /MACHINE:ARM64)
    elseif (CMAKE_GENERATOR_PLATFORM STREQUAL "Win32")
        target_compile_options(NppAI PRIVATE /O2)
        target_link_options(NppAI PRIVATE /MACHINE:X86)
    else()
        # Keep the binary on the x64 baseline ISA. AVX2/FMA intrinsics are
        # selected at runtime only after CPUID + OSXSAVE/XCR0 checks.
        target_compile_options(NppAI PRIVATE /O2)
        target_link_options(NppAI PRIVATE /MACHINE:X64)
    endif()
    target_compile_definitions(NppAI PRIVATE UNICODE _UNICODE)
    if(NPPAI_TEST_HOST_SEAM)
        target_compile_definitions(NppAI PRIVATE NPPAI_TESTING)
    endif()
endif()

function(add_nppai_test target source)
    add_executable(${target} ${source} src/NppAIEngine.cpp)
    target_include_directories(${target} PRIVATE src)
    target_compile_definitions(${target} PRIVATE NPPAI_TESTING)
    target_link_libraries(${target} PRIVATE d3d11 d3dcompiler)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(${target} PRIVATE OpenMP::OpenMP_CXX)
    endif()
endfunction()

add_nppai_test(TestEngine test_engine.cpp)
add_nppai_test(TestTensor tests/test_tensor.cpp)
add_nppai_test(TestModelLoader tests/test_model_loader.cpp)
add_nppai_test(TestTokenizer tests/test_tokenizer.cpp)
add_nppai_test(TestGenerationPipeline tests/test_generation_pipeline.cpp)
add_executable(TestAIPromptPipeline tests/test_ai_prompt_pipeline.cpp src/AIPromptPipeline.cpp)
target_include_directories(TestAIPromptPipeline PRIVATE src)
add_executable(TestPluginPromptActions tests/test_plugin_prompt_actions.cpp src/PluginPromptActions.cpp)
target_include_directories(TestPluginPromptActions PRIVATE src)
add_executable(TestGenerationContext tests/test_generation_context.cpp src/GenerationContext.cpp)
target_include_directories(TestGenerationContext PRIVATE src)
add_executable(TestPluginGenerationFlow
    tests/test_plugin_generation_flow.cpp
    src/PluginPromptActions.cpp
    src/GenerationContext.cpp
    src/GenerationStreamRouter.cpp
    src/AIPromptPipeline.cpp
)
target_include_directories(TestPluginGenerationFlow PRIVATE src)
add_executable(TestGenerationStreamRouter tests/test_generation_stream_router.cpp src/GenerationStreamRouter.cpp)
target_include_directories(TestGenerationStreamRouter PRIVATE src)
add_executable(TestGenerationTracking tests/test_generation_tracking.cpp src/GenerationTracking.cpp)
target_include_directories(TestGenerationTracking PRIVATE src)
add_executable(TestTelemetryRedaction tests/test_telemetry_redaction.cpp src/TelemetryRedaction.cpp)
target_include_directories(TestTelemetryRedaction PRIVATE src)
add_executable(TestRAGRetrieval tests/test_rag_retrieval.cpp src/RAGManager.cpp src/RepositoryIndexer.cpp)
target_include_directories(TestRAGRetrieval PRIVATE src)
target_compile_definitions(TestRAGRetrieval PRIVATE NPPAI_TESTING)
add_executable(TestRepositoryContext tests/test_repository_context.cpp src/RepositoryContext.cpp)
target_include_directories(TestRepositoryContext PRIVATE src)
add_executable(TestWorkspaceFileReader tests/test_workspace_file_reader.cpp src/WorkspaceFileReader.cpp)
target_include_directories(TestWorkspaceFileReader PRIVATE src)
add_executable(TestStructuredPatch tests/test_structured_patch.cpp src/StructuredPatch.cpp)
target_include_directories(TestStructuredPatch PRIVATE src)
add_executable(TestPatchReview tests/test_patch_review.cpp src/PatchReview.cpp src/StructuredPatch.cpp src/WorkspaceFileReader.cpp)
target_include_directories(TestPatchReview PRIVATE src)
add_executable(TestPatchRollback tests/test_patch_rollback.cpp src/PatchRollback.cpp src/PatchReview.cpp src/StructuredPatch.cpp src/WorkspaceFileReader.cpp)
target_include_directories(TestPatchRollback PRIVATE src)
add_executable(TestAgentFeedbackLoop tests/test_agent_feedback_loop.cpp src/AgentFeedbackLoop.cpp src/GenerationStreamRouter.cpp)
target_include_directories(TestAgentFeedbackLoop PRIVATE src)
add_executable(TestBoundedRepairLoop tests/test_bounded_repair_loop.cpp src/BoundedRepairLoop.cpp src/AgentFeedbackLoop.cpp src/GenerationStreamRouter.cpp)
target_include_directories(TestBoundedRepairLoop PRIVATE src)
add_executable(TestPluginDllContract tests/test_plugin_dll_contract.cpp)
add_dependencies(TestPluginDllContract NppAI)
add_executable(TestPluginHostLifecycle tests/test_plugin_host_lifecycle.cpp)
target_include_directories(TestPluginHostLifecycle PRIVATE src)
add_dependencies(TestPluginHostLifecycle NppAI)
if(MSVC)
    # The DLL contract compares localized command names containing Polish
    # characters. Compile the test source as UTF-8 so its expected wide
    # string literals match the UTF-8-compiled plugin.
    target_compile_options(TestPluginDllContract PRIVATE /utf-8)
    target_compile_options(TestPluginHostLifecycle PRIVATE /utf-8)
endif()

add_test(NAME TensorMath COMMAND TestTensor)
add_test(NAME ModelLoaderValidation COMMAND TestModelLoader)
add_test(NAME TokenizerFallback COMMAND TestTokenizer)
add_test(NAME GenerationPipeline COMMAND TestGenerationPipeline)
add_test(NAME AIPromptPipeline COMMAND TestAIPromptPipeline)
add_test(NAME PluginPromptActions COMMAND TestPluginPromptActions)
add_test(NAME GenerationContext COMMAND TestGenerationContext)
add_test(NAME PluginGenerationFlow COMMAND TestPluginGenerationFlow)
add_test(NAME GenerationStreamRouter COMMAND TestGenerationStreamRouter)
add_test(NAME GenerationTracking COMMAND TestGenerationTracking)
add_test(NAME TelemetryRedaction COMMAND TestTelemetryRedaction)
add_test(NAME RAGRetrieval COMMAND TestRAGRetrieval)
add_test(NAME RepositoryContext COMMAND TestRepositoryContext)
add_test(NAME WorkspaceFileReader COMMAND TestWorkspaceFileReader)
add_test(NAME StructuredPatch COMMAND TestStructuredPatch)
add_test(NAME PatchReview COMMAND TestPatchReview)
add_test(NAME PatchRollback COMMAND TestPatchRollback)
add_test(NAME AgentFeedbackLoop COMMAND TestAgentFeedbackLoop)
add_test(NAME BoundedRepairLoop COMMAND TestBoundedRepairLoop)
add_test(NAME PluginDllContract COMMAND TestPluginDllContract $<TARGET_FILE:NppAI>)
add_test(NAME PluginHostLifecycle COMMAND TestPluginHostLifecycle $<TARGET_FILE:NppAI>)

set_tests_properties(TensorMath PROPERTIES LABELS "unit" TIMEOUT 30)
set_tests_properties(ModelLoaderValidation PROPERTIES LABELS "unit;security" TIMEOUT 30)
set_tests_properties(TokenizerFallback PROPERTIES LABELS "unit" TIMEOUT 30)
set_tests_properties(GenerationPipeline PROPERTIES LABELS "integration" TIMEOUT 30)
set_tests_properties(AIPromptPipeline PROPERTIES LABELS "unit;integration" TIMEOUT 30)
set_tests_properties(PluginPromptActions PROPERTIES LABELS "unit;integration;plugin" TIMEOUT 30)
set_tests_properties(GenerationContext PROPERTIES LABELS "unit;integration;rag" TIMEOUT 30)
set_tests_properties(PluginGenerationFlow PROPERTIES LABELS "integration;plugin;rag" TIMEOUT 30)
set_tests_properties(GenerationStreamRouter PROPERTIES LABELS "unit;integration;plugin" TIMEOUT 30)
set_tests_properties(GenerationTracking PROPERTIES LABELS "unit;integration;plugin" TIMEOUT 30)
set_tests_properties(TelemetryRedaction PROPERTIES LABELS "unit;security;telemetry" TIMEOUT 30)
set_tests_properties(RAGRetrieval PROPERTIES LABELS "unit;rag" TIMEOUT 30)
set_tests_properties(RepositoryContext PROPERTIES LABELS "unit;integration;rag" TIMEOUT 30)
set_tests_properties(WorkspaceFileReader PROPERTIES LABELS "unit;integration;security;agent" TIMEOUT 30)
set_tests_properties(StructuredPatch PROPERTIES LABELS "unit;integration;agent" TIMEOUT 30)
set_tests_properties(PatchReview PROPERTIES LABELS "unit;integration;security;agent" TIMEOUT 30)
set_tests_properties(PatchRollback PROPERTIES LABELS "unit;integration;security;agent;recovery" TIMEOUT 30)
set_tests_properties(AgentFeedbackLoop PROPERTIES LABELS "unit;integration;agent" TIMEOUT 30)
set_tests_properties(BoundedRepairLoop PROPERTIES LABELS "unit;integration;agent;safety" TIMEOUT 30)
set_tests_properties(PluginDllContract PROPERTIES LABELS "integration;plugin" TIMEOUT 30)
set_tests_properties(PluginHostLifecycle PROPERTIES LABELS "integration;plugin;host" TIMEOUT 30)

option(NPPAI_BUILD_BENCHMARKS "Build NppAI microbenchmarks" OFF)

if(NPPAI_BUILD_BENCHMARKS)
    add_executable(BenchmarkTensor benchmarks/benchmark_tensor.cpp src/NppAIEngine.cpp)
    target_include_directories(BenchmarkTensor PRIVATE src)
    target_compile_definitions(BenchmarkTensor PRIVATE NPPAI_TESTING)
    target_link_libraries(BenchmarkTensor PRIVATE d3d11 d3dcompiler)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(BenchmarkTensor PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
