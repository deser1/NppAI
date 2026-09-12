cmake_minimum_required(VERSION 3.10)
project(NppAIPlugin VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Wykrywanie docelowej architektury (wsparcie dla x64 oraz ARM64)
if (NOT CMAKE_GENERATOR_PLATFORM)
    set(CMAKE_GENERATOR_PLATFORM x64)
endif()

# Zdefiniuj projekt jako bibliotekę współdzieloną (DLL)
add_library(NppAI SHARED
    src/NppAI_Core.cpp
    src/NppAI_Core.rc
    src/PluginDefinition.cpp
    src/AIManager.cpp
    src/RAGManager.cpp
    src/TelemetryManager.cpp
    src/NppAIEngine.cpp
)

# Automatyczne wersjonowanie (Auto-Increment Build Number) przed każdą kompilacją
add_custom_command(
    TARGET NppAI PRE_BUILD
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_SOURCE_DIR}/increment_version.cmake
    COMMENT "Zwiększanie numeru wersji wtyczki..."
)

# Cel testowy terminala
add_executable(TestEngine test_engine.cpp src/NppAIEngine.cpp)
target_link_libraries(TestEngine PRIVATE d3d11 d3dcompiler)
if(OpenMP_CXX_FOUND)
    target_link_libraries(TestEngine PRIVATE OpenMP::OpenMP_CXX)
endif()

# Nagłówki
target_include_directories(NppAI PRIVATE src src/DockingFeature)

# Połącz z bibliotekami systemowymi
target_link_libraries(NppAI PRIVATE user32 gdi32 shlwapi d3d11 d3dcompiler winhttp)

# Włącz OpenMP dla szybszego wnioskowania CPU
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(NppAI PRIVATE OpenMP::OpenMP_CXX)
endif()

# Konfiguracja kompilatora MSVC (ustawienia specyficzne dla DLL Notepad++)
if (MSVC)
    # Zoptymalizowany rozmiar dzięki Trimmingowi/LTO i statycznemu linkowaniu C++ Runtime
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE) # Włącza LTO (Link Time Optimization)
    
    # Konfiguracja flag zależnie od architektury
    if (CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
        set_target_properties(NppAI PROPERTIES
            PREFIX ""
            SUFFIX ".dll"
            COMPILE_FLAGS "/EHsc /W3 /WX- /openmp /utf-8 /O2 /GL"
            LINK_FLAGS "/MACHINE:ARM64 /LTCG /OPT:REF /OPT:ICF"
        )
    else()
        set_target_properties(NppAI PROPERTIES
            PREFIX ""
            SUFFIX ".dll"
            COMPILE_FLAGS "/EHsc /W3 /WX- /openmp /utf-8 /O2 /GL /arch:AVX2"
            LINK_FLAGS "/MACHINE:X64 /LTCG /OPT:REF /OPT:ICF"
        )
    endif()
    # UNICODE macro jest wymagane przez Notepad++
    target_compile_definitions(NppAI PRIVATE UNICODE _UNICODE)
endif()
