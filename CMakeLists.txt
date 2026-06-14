cmake_minimum_required(VERSION 3.10)
project(NppAIPlugin VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Tylko x86_64 dla Notepad++
set(CMAKE_GENERATOR_PLATFORM x64)

# Zdefiniuj projekt jako bibliotekę współdzieloną (DLL)
add_library(NppAI SHARED
    src/NppPluginDemo.cpp
    src/PluginDefinition.cpp
    src/AIManager.cpp
    src/TelemetryManager.cpp
    src/NppAIEngine.cpp
)

# Nagłówki
target_include_directories(NppAI PRIVATE src src/DockingFeature)

# Połącz z bibliotekami systemowymi
target_link_libraries(NppAI user32 gdi32)

# Konfiguracja kompilatora MSVC (ustawienia specyficzne dla DLL Notepad++)
if (MSVC)
    set_target_properties(NppAI PROPERTIES
        PREFIX ""
        SUFFIX ".dll"
        COMPILE_FLAGS "/EHsc /W3 /WX-"
        LINK_FLAGS "/MACHINE:X64"
    )
    # UNICODE macro jest wymagane przez Notepad++
    target_compile_definitions(NppAI PRIVATE UNICODE _UNICODE)
endif()
