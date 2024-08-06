cmake_minimum_required(VERSION 3.10)
project(nnv2 LANGUAGES CXX CUDA)

option(BUILD_MOCKS "build mock program to test the functions" OFF)

# Set the directories for the core, mocks, and examples
set(CORE_DIR ${CMAKE_SOURCE_DIR}/core)
set(MOCKS_DIR ${CMAKE_SOURCE_DIR}/mocks)
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)

# Collect all CUDA source and header files in the core directory
file(GLOB CORE_SOURCES ${CORE_DIR}/*.cu)
file(GLOB CORE_HEADERS ${CORE_DIR}/*.cuh)

include_directories(${CORE_DIR})

# Add the CUDA core library
add_library(core_lib SHARED ${CORE_SOURCES})
target_include_directories(core_lib PRIVATE ${CUDA_INCLUDE_DIRS})
target_compile_options(core_lib PRIVATE
                       $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

if(BUILD_MOCKS)
    # Collect all mock files in the mocks directory
    file(GLOB MOCK_SOURCES ${MOCKS_DIR}/*.cu)

    # Add test executables
    foreach(mock_source ${MOCK_SOURCES})
        get_filename_component(mock_name ${mock_source} NAME_WE)
        add_executable(${mock_name} ${mock_source})
        target_link_libraries(${mock_name} PRIVATE core_lib)
    endforeach()
endif()

# Collect all example files in the examples directory
file(GLOB EXAMPLE_SOURCES ${EXAMPLES_DIR}/*.cu)

# Add example executables
foreach(example_source ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_source} NAME_WE)
    add_executable(${example_name} ${example_source})
    target_link_libraries(${example_name} PRIVATE core_lib)
endforeach()

# Set the C++ standard to C++17 or higher
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 12)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)