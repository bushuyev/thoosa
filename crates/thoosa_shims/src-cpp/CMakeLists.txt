cmake_minimum_required(VERSION 3.25)

project(thoosa_shims LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

add_library(
        thoosa_shims SHARED

        stream.cpp
        memcpy.cpp
        context.cpp
        device_stats.cpp
        errors.cpp
        inference_mode.cpp
        quantized_linear.cpp
        logging.cpp
        optimizers.cpp
        shared_arena_allocator.cpp

        cu/z_norm_op.cpp
        cu/z_norm_cuda.cu
)

target_include_directories(
        thoosa_shims
        PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_definitions(
        thoosa_shims
        PRIVATE
        _GLIBCXX_USE_CXX11_ABI=1
)

set_target_properties(
        thoosa_shims
        PROPERTIES
        CUDA_ARCHITECTURES 89
        CUDA_STANDARD 17
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON
)

target_compile_options(
        thoosa_shims
        PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

target_link_libraries(
        thoosa_shims
        PUBLIC
        ${TORCH_LIBRARIES}
        CUDA::cuda_driver
        CUDA::cudart
)

#target_link_options(
#        thoosa_shims
#        PRIVATE
#        "-Wl,--no-undefined"
#)

install(
        TARGETS thoosa_shims
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
)