cmake_minimum_required(VERSION 3.15)
project(NppAIPlugin VERSION 1.0 LANGUAGES CXX)

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
    src/AIManager.cpp
    src/RAGManager.cpp
    src/TelemetryManager.cpp
    src/NppAIEngine.cpp
)

add_custom_command(
    TARGET NppAI PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/increment_version.cmake
    COMMENT "Incrementing plugin version..."
)

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
        target_compile_options(NppAI PRIVATE /O2 /arch:AVX2)
        target_link_options(NppAI PRIVATE /MACHINE:X64)
    endif()

    target_compile_definitions(NppAI PRIVATE UNICODE _UNICODE)
endif()

add_executable(TestEngine test_engine.cpp src/NppAIEngine.cpp)
target_include_directories(TestEngine PRIVATE src)
target_link_libraries(TestEngine PRIVATE d3d11 d3dcompiler)

if(OpenMP_CXX_FOUND)
    target_link_libraries(TestEngine PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(TestTensor tests/test_tensor.cpp src/NppAIEngine.cpp)
target_include_directories(TestTensor PRIVATE src)
target_link_libraries(TestTensor PRIVATE d3d11 d3dcompiler)

if(OpenMP_CXX_FOUND)
    target_link_libraries(TestTensor PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(TestModelLoader tests/test_model_loader.cpp src/NppAIEngine.cpp)
target_include_directories(TestModelLoader PRIVATE src)
target_link_libraries(TestModelLoader PRIVATE d3d11 d3dcompiler)

if(OpenMP_CXX_FOUND)
    target_link_libraries(TestModelLoader PRIVATE OpenMP::OpenMP_CXX)
endif()

add_test(NAME TensorMath COMMAND TestTensor)
add_test(NAME ModelLoaderValidation COMMAND TestModelLoader)

set_tests_properties(TensorMath PROPERTIES
    LABELS "unit"
    TIMEOUT 30
)

set_tests_properties(ModelLoaderValidation PROPERTIES
    LABELS "unit;security"
    TIMEOUT 30
)
