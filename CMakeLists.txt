cmake_minimum_required(VERSION 3.16)
project(airun LANGUAGES CXX)

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|X86_64)$")
    message(FATAL_ERROR "airun requires x86_64. Detected: ${CMAKE_SYSTEM_PROCESSOR}.")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(airun main.cpp)

target_include_directories(airun PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(MSVC)
    target_compile_options(airun PRIVATE /utf-8 /W4 /EHsc /O2 /arch:SSE2)
    target_compile_definitions(airun PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX)
else()
    target_compile_options(airun PRIVATE -O3 -Wall -Wextra -msse2 -pthread)
    find_package(Threads REQUIRED)
    target_link_libraries(airun PRIVATE Threads::Threads)
endif()

target_sources(airun PRIVATE
    st.h st_writer.h tokenizer.h matmul.h quant.h
    model.h sampler.h agent.h ui.h
)