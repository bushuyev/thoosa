cmake_minimum_required(VERSION 3.25)
project(thoosa_shims LANGUAGES CXX CUDA)

# Give CLion a compilation database
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_ARCHITECTURES 89)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Tell CMake where LibTorch lives (passed from build.rs)
set(Torch_DIR "${TORCH_DIR}" CACHE PATH "")

find_package(Torch REQUIRED)          # brings torch, torch_cpu, torch_cuda…
find_package(CUDAToolkit REQUIRED)

add_library(thoosa_shims SHARED stream.cpp memcpy.cpp context.cpp device_stats.cpp errors.cpp) #add file to build.rs / cargo:rerun-if-changed
target_include_directories(thoosa_shims PUBLIC ..)
target_link_libraries(thoosa_shims PUBLIC ${TORCH_LIBRARIES} CUDA::cuda_driver CUDA::cudart)


install(TARGETS thoosa_shims
        LIBRARY DESTINATION lib          # .so / .dylib
        ARCHIVE DESTINATION lib          # static libs if you ever add them
        RUNTIME DESTINATION bin)         # .dll on Windows
