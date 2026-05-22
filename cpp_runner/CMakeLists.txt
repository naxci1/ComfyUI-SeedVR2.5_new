cmake_minimum_required(VERSION 3.22)
project(SeedVR2Runner VERSION 2.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt6 auto-tools
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Widgets)

# ── Sources ──────────────────────────────────────────────────────────────────
add_executable(SeedVR2Runner
    main.cpp
    ProcessEngine.h
    ProcessEngine.cpp
)

target_link_libraries(SeedVR2Runner
    PRIVATE
        Qt6::Core
        Qt6::Widgets
)

# Windows-specific: suppress the console window and embed a manifest so the
# app is DPI-aware without requiring a separate .manifest file.
if(WIN32)
    set_target_properties(SeedVR2Runner PROPERTIES
        WIN32_EXECUTABLE TRUE          # hides the console window
    )

    # Optional: embed an icon if you have one (uncomment and add icon.ico)
    # target_sources(SeedVR2Runner PRIVATE resources.rc)
endif()

# Optimised release build flags
if(MSVC)
    target_compile_options(SeedVR2Runner PRIVATE
        $<$<CONFIG:Release>:/O2 /GL /DNDEBUG>
    )
    target_link_options(SeedVR2Runner PRIVATE
        $<$<CONFIG:Release>:/LTCG>
    )
else()
    # MinGW / GCC / Clang
    target_compile_options(SeedVR2Runner PRIVATE
        $<$<CONFIG:Release>:-O2 -DNDEBUG>
    )
endif()

# ── Install / deploy helper ───────────────────────────────────────────────────
# After building, run windeployqt to bundle Qt DLLs alongside the executable:
#   windeployqt --release SeedVR2Runner.exe
