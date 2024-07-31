cmake_minimum_required(VERSION 3.10)
project(nnv2 LANGUAGES CXX CUDA)

# Set the directories for the core, mocks, and examples
set(CORE_DIR ${CMAKE_SOURCE_DIR}/core)
set(MOCKS_DIR ${CMAKE_SOURCE_DIR}/mocks)
set(EXAMPLES_DIR ${CMAKE_SOURCE_DIR}/examples)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Collect all CUDA source and header files in the core directory
file(GLOB CORE_SOURCES ${CORE_DIR}/*.cu)
file(GLOB CORE_HEADERS ${CORE_DIR}/*.cuh ${CORE_DIR}/nnv2.h)

include_directories(${CORE_DIR})

# Add the CUDA core library
cuda_add_library(core_lib STATIC ${CORE_SOURCES} ${CORE_HEADERS}
                 OPTIONS --expt-extended-lambda)

# Set the CUDA compiler flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-extended-lambda")

# Collect all test files in the mocks directory
file(GLOB MOCK_SOURCES ${MOCKS_DIR}/*.cu)

# Add test executables
foreach(mock_source ${MOCK_SOURCES})
    get_filename_component(mock_name ${mock_source} NAME_WE)
    cuda_add_executable(${mock_name} ${mock_source})
    target_link_libraries(${mock_name} core_lib)
endforeach()

# Collect all example files in the examples directory
file(GLOB EXAMPLE_SOURCES ${EXAMPLES_DIR}/*.cc)

# Add example executables
foreach(example_source ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_source} NAME_WE)
    add_executable(${example_name} ${example_source})
    target_link_libraries(${example_name} core_lib cudart)
endforeach()

# Set the C++ standard to C++17 or higher
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 12)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)