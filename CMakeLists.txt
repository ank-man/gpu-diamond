cmake_minimum_required(VERSION 3.10)
project(GPU-Diamond VERSION 1.0 LANGUAGES C CXX CUDA)

# CUDA required
find_package(CUDA REQUIRED)

# Set C and CUDA standards
set(CMAKE_C_STANDARD 99)
set(CMAKE_CUDA_STANDARD 11)

# CUDA architecture flags
set(CMAKE_CUDA_ARCHITECTURES 52 61 70 80)

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/src/basic
    ${CMAKE_SOURCE_DIR}/src/search
    ${CMAKE_SOURCE_DIR}/src/align
    ${CMAKE_SOURCE_DIR}/src/util
)

# Source files
set(C_SOURCES
    src/basic/gpu_diamond.c
)

set(CUDA_SOURCES
    src/search/gpu_diamond.cu
)

# Static library
add_library(gpudiamond STATIC ${C_SOURCES} ${CUDA_SOURCES})

target_compile_options(gpudiamond PRIVATE
    $<$<COMPILE_LANGUAGE:C>:-O2 -Wall>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
)

# Test executable
add_executable(test_gpu_diamond src/test/test_gpu_diamond.c)
target_link_libraries(test_gpu_diamond gpudiamond ${CUDA_LIBRARIES})
target_link_libraries(test_gpu_diamond cudart)

# Install targets
install(TARGETS gpudiamond test_gpu_diamond
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(FILES src/basic/gpu_diamond.h DESTINATION include)
