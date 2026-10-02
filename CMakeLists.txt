cmake_minimum_required(VERSION 3.16)
project(airun LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(airun
    main.cpp
)

target_include_directories(airun PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
if(MSVC)
    target_compile_options(airun PRIVATE
        /utf-8 /W4 /EHsc /O2 /arch:SSE2
    )
    target_compile_definitions(airun PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
    )
else()
    target_compile_options(airun PRIVATE
        -O3 -Wall -Wextra -msse2 -pthread
    )
    target_link_libraries(airun PRIVATE pthread)
endif()

option(AIRUN_AVX "Enable AVX (128-bit)" OFF)
option(AIRUN_AVX2 "Enable AVX2 (256-bit)" OFF)

if(AIRUN_AVX)
    if(MSVC)
        target_compile_options(airun PRIVATE /arch:AVX)
    else()
        target_compile_options(airun PRIVATE -mavx)
    endif()
    target_compile_definitions(airun PRIVATE AIRUN_HAVE_AVX=1)
endif()

if(AIRUN_AVX2)
    if(MSVC)
        target_compile_options(airun PRIVATE /arch:AVX2)
    else()
        target_compile_options(airun PRIVATE -mavx2 -mfma)
    endif()
    target_compile_definitions(airun PRIVATE AIRUN_HAVE_AVX2=1)
endif()
target_sources(airun PRIVATE
    st.h
    tokenizer.h
    matmul.h
    model.h
    sampler.h
    agent.h
    ui.h
)
