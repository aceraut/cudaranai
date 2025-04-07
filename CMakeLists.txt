cmake_minimum_required(VERSION 3.10)
project(nnv2 LANGUAGES CXX CUDA)

option(BUILD_TESTS "build test program to test the functions" OFF)

# Set the directories for the core, tests, and demos
set(CORE_DIR ${CMAKE_SOURCE_DIR}/core)
set(TEST_DIR ${CMAKE_SOURCE_DIR}/test)
set(DEMO_DIR ${CMAKE_SOURCE_DIR}/demo)

# Collect all CUDA source and header files in the core directory
file(GLOB CORE_SOURCES ${CORE_DIR}/*.cu)

include_directories(${CORE_DIR})

# Add the CUDA core library
add_library(core_lib SHARED ${CORE_SOURCES})
target_include_directories(core_lib PRIVATE ${CUDA_INCLUDE_DIRS})
target_compile_options(core_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-extended-lambda
        -arch=sm_75
        -gencode=arch=compute_75,code=sm_75
    >
)

if(BUILD_TESTS)
    # Collect all test files in the tests directory
    file(GLOB TEST_SOURCES ${TEST_DIR}/*.cu)

    # Add test executables
    foreach(test_source ${TEST_SOURCES})
        get_filename_component(test_name ${test_source} NAME_WE)
        add_executable(${test_name} ${test_source})
        target_link_libraries(${test_name} PRIVATE core_lib)
        target_compile_options(${test_name} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:
                -arch=sm_75
                -gencode=arch=compute_75,code=sm_75
            >
        )
    endforeach()
endif()

# Collect all demo files in the demos directory
file(GLOB DEMO_SOURCES ${DEMO_DIR}/*.cu)

# Add demo executables
foreach(demo_source ${DEMO_SOURCES})
    get_filename_component(demo_name ${demo_source} NAME_WE)
    add_executable(${demo_name} ${demo_source})
    target_link_libraries(${demo_name} PRIVATE core_lib)
    target_compile_options(${demo_name} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:
            -arch=sm_75
            -gencode=arch=compute_75,code=sm_75
        >
    )
endforeach()

# Set the C++ standard to C++17 or higher
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 12)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)