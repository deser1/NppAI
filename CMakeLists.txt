cmake_minimum_required(VERSION 3.15)
project(NppAIPlugin VERSION 0.3.0 LANGUAGES CXX)

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
    src/PluginPromptActions.cpp
    src/AIManager.cpp
    src/AIPromptPipeline.cpp
    src/RAGManager.cpp
    src/TelemetryManager.cpp
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
    src/AIPromptPipeline.cpp
)
target_include_directories(TestPluginGenerationFlow PRIVATE src)

add_test(NAME TensorMath COMMAND TestTensor)
add_test(NAME ModelLoaderValidation COMMAND TestModelLoader)
add_test(NAME TokenizerFallback COMMAND TestTokenizer)
add_test(NAME GenerationPipeline COMMAND TestGenerationPipeline)
add_test(NAME AIPromptPipeline COMMAND TestAIPromptPipeline)
add_test(NAME PluginPromptActions COMMAND TestPluginPromptActions)
add_test(NAME GenerationContext COMMAND TestGenerationContext)
add_test(NAME PluginGenerationFlow COMMAND TestPluginGenerationFlow)

set_tests_properties(TensorMath PROPERTIES LABELS "unit" TIMEOUT 30)
set_tests_properties(ModelLoaderValidation PROPERTIES LABELS "unit;security" TIMEOUT 30)
set_tests_properties(TokenizerFallback PROPERTIES LABELS "unit" TIMEOUT 30)
set_tests_properties(GenerationPipeline PROPERTIES LABELS "integration" TIMEOUT 30)
set_tests_properties(AIPromptPipeline PROPERTIES LABELS "unit;integration" TIMEOUT 30)
set_tests_properties(PluginPromptActions PROPERTIES LABELS "unit;integration;plugin" TIMEOUT 30)
set_tests_properties(GenerationContext PROPERTIES LABELS "unit;integration;rag" TIMEOUT 30)
set_tests_properties(PluginGenerationFlow PROPERTIES LABELS "integration;plugin;rag" TIMEOUT 30)

option(NPPAI_BUILD_BENCHMARKS "Build NppAI microbenchmarks" OFF)

if(NPPAI_BUILD_BENCHMARKS)
    add_executable(BenchmarkTensor benchmarks/benchmark_tensor.cpp src/NppAIEngine.cpp)
    target_include_directories(BenchmarkTensor PRIVATE src)
    target_link_libraries(BenchmarkTensor PRIVATE d3d11 d3dcompiler)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(BenchmarkTensor PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
