cmake_minimum_required(VERSION 3.10)
project(nnv2)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# Header directories
include_directories(core)

# Enable AddressSanitizer if -DNNV2_USE_ASAN=ON
option(NNV2_USE_ASAN "Build executables with AddressSanitizer" OFF)
if(NNV2_USE_ASAN)
    add_compile_options(-fsanitize=address -fsanitize=undefined)
    add_link_options(-fsanitize=address -fsanitize=undefined)
endif()

# Find OpenBLAS
find_package(OpenBLAS REQUIRED)
if(NOT OpenBLAS_FOUND)
    message(FATAL_ERROR "OpenBLAS not found")
endif()

# Add the core source files
file(GLOB CORE_SOURCES "core/*.cc" "core/*.h")

# Create an object library for core sources
add_library(nnv2_core OBJECT ${CORE_SOURCES})

# List of example files
set(EXAMPLES
    examples/lenet5.cc
    examples/lenet5_var1.cc
    examples/2conv1fc_adam.cc
    examples/3conv2fc_rms.cc
)

# Create executable for each example
foreach(EXAMPLE ${EXAMPLES})
    get_filename_component(EXE_NAME ${EXAMPLE} NAME_WE)
    add_executable(${EXE_NAME} ${EXAMPLE} $<TARGET_OBJECTS:nnv2_core>)
    target_link_libraries(${EXE_NAME} openblas)
endforeach()

# List of mock test files
set(TESTS
    mocks/activation_test.cc
    mocks/conv_test.cc
    mocks/dataloader_test.cc
    mocks/linear_test.cc
    mocks/mathfunc_test.cc
    mocks/maxpool_test.cc
)

# Create executable for each test
foreach(TEST ${TESTS})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    add_executable(${TEST_NAME} ${TEST} $<TARGET_OBJECTS:nnv2_core>)
    target_link_libraries(${TEST_NAME} openblas)
endforeach()

# Enable testing
enable_testing()

# Add tests
foreach(TEST ${TESTS})
    get_filename_component(TEST_NAME ${TEST} NAME_WE)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endforeach()
