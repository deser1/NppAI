cmake_minimum_required(VERSION 3.10)
project(NppAIPlugin VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tylko x86_64 dla Notepad++
set(CMAKE_GENERATOR_PLATFORM x64)

# Zdefiniuj projekt jako bibliotekę współdzieloną (DLL)
add_library(NppAI SHARED
    src/NppAI_Core.cpp
    src/PluginDefinition.cpp
    src/AIManager.cpp
    src/TelemetryManager.cpp
    src/NppAIEngine.cpp
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
target_link_libraries(NppAI PRIVATE user32 gdi32 shlwapi d3d11 d3dcompiler)

# Włącz OpenMP dla szybszego wnioskowania CPU
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(NppAI PRIVATE OpenMP::OpenMP_CXX)
endif()

# Konfiguracja kompilatora MSVC (ustawienia specyficzne dla DLL Notepad++)
if (MSVC)
    set_target_properties(NppAI PROPERTIES
        PREFIX ""
        SUFFIX ".dll"
        COMPILE_FLAGS "/EHsc /W3 /WX- /openmp /utf-8"
        LINK_FLAGS "/MACHINE:X64"
    )
    # UNICODE macro jest wymagane przez Notepad++
    target_compile_definitions(NppAI PRIVATE UNICODE _UNICODE)
endif()
