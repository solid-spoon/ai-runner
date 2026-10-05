cmake_minimum_required(VERSION 3.16)
project(airun LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|X86_64)$")
    message(FATAL_ERROR "airun requires x86_64. Detected: ${CMAKE_SYSTEM_PROCESSOR}.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(CTest)

add_library(airun_flags INTERFACE)
target_compile_features(airun_flags INTERFACE cxx_std_17)

if(MSVC)
    target_compile_options(airun_flags INTERFACE /utf-8 /W4 /EHsc /arch:AVX)
    target_compile_definitions(airun_flags INTERFACE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
else()
    target_compile_options(airun_flags INTERFACE -O3 -Wall -Wextra -mavx)
    find_package(Threads REQUIRED)
    target_link_libraries(airun_flags INTERFACE Threads::Threads)
endif()

add_executable(airun main.cpp)
target_include_directories(airun PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(airun PRIVATE airun_flags)

if(BUILD_TESTING)
    add_executable(airun_tests test_airun.cpp)
    target_include_directories(airun_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(airun_tests PRIVATE airun_flags)

    add_test(NAME airun_tests COMMAND airun_tests)
    set_tests_properties(airun_tests PROPERTIES
        TIMEOUT 60
        LABELS "unit"
    )
endif()