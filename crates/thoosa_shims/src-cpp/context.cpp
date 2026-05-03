#include "errors.h"

#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>

#include <iostream>

extern "C" {
bool torch_cuda_hard_reset_device(int device) {
    cudaError_t e = cudaSetDevice(device);
    if (e != cudaSuccess) {
        std::cerr << "cudaSetDevice before reset failed: "
                  << cudaGetErrorString(e) << "\n";
        return false;
    }

    // Clear previous async error slot if possible.
    for (int i = 0; i < 4; ++i) {
        e = cudaGetLastError();
        if (e == cudaSuccess) {
            break;
        }
        std::cerr << "pre-reset cudaGetLastError: "
                  << cudaGetErrorName(e) << ": "
                  << cudaGetErrorString(e) << "\n";
    }

    // Optional and best-effort. May fail for fatal errors.
    try {
        at::cuda::clearCublasWorkspaces();
        c10::cuda::CUDACachingAllocator::emptyCache();
    } catch (const std::exception& ex) {
        std::cerr << "pre-reset libtorch purge failed: " << ex.what() << "\n";
    }

    e = cudaDeviceReset();
    if (e != cudaSuccess) {
        std::cerr << "cudaDeviceReset failed: "
                  << cudaGetErrorName(e) << ": "
                  << cudaGetErrorString(e) << "\n";
        return false;
    }

    // Force reinitialization.
    e = cudaSetDevice(device);
    if (e != cudaSuccess) {
        std::cerr << "cudaSetDevice after reset failed: "
                  << cudaGetErrorString(e) << "\n";
        return false;
    }

    // Raw CUDA health check.
    void* p = nullptr;
    e = cudaMalloc(&p, 4);
    if (e != cudaSuccess) {
        std::cerr << "cudaMalloc after reset failed: "
                  << cudaGetErrorString(e) << "\n";
        return false;
    }
    cudaFree(p);

    // Libtorch health check.
    // try {
    //     auto t = torch::ones({1}, torch::TensorOptions()
    //                                  .device(torch::kCUDA, device)
    //                                  .dtype(torch::kFloat32));
    //     auto cpu = t.cpu();
    //     return cpu.item<float>() == 1.0f;
    // } catch (const std::exception& ex) {
    //     std::cerr << "libtorch health check after reset failed: "
    //               << ex.what() << "\n";
    //     return false;
    // }
}


CUresult tho_cu_init(unsigned int flags) {
    return tho_record_cuda_result("cuInit", cuInit(flags));
}

CUresult tho_cu_reset(CUdevice device) {
    torch_cuda_hard_reset_device(0);
  return tho_record_cuda_result("cuDevicePrimaryCtxReset", cuDevicePrimaryCtxReset(device));
}

CUresult tho_cu_device_get(CUdevice *device, int ordinal) {
  return tho_record_cuda_result("cuDeviceGet", cuDeviceGet(device, ordinal));
}

CUresult tho_cu_device_primary_ctx_retain(CUcontext *ctx, CUdevice device) {
  return tho_record_cuda_result("cuDevicePrimaryCtxRetain", cuDevicePrimaryCtxRetain(ctx, device));
}

CUresult tho_cu_device_primary_ctx_release(CUdevice device) {
  return tho_record_cuda_result("cuDevicePrimaryCtxRelease", cuDevicePrimaryCtxRelease(device));
}

CUresult tho_cu_device_get_attribute(int *pi, int attrib, CUdevice device) {
  return tho_record_cuda_result("cuDeviceGetAttribute", cuDeviceGetAttribute(pi, static_cast<CUdevice_attribute>(attrib), device));
}

CUresult tho_cu_ctx_set_current(CUcontext ctx) {
  return tho_record_cuda_result("cuCtxSetCurrent", cuCtxSetCurrent(ctx));
}

CUresult tho_cu_ctx_get_current(CUcontext *ctx) {
  return tho_record_cuda_result("cuCtxGetCurrent", cuCtxGetCurrent(ctx));
}

CUresult tho_cu_ctx_synchronize() {
  return tho_record_cuda_result("cuCtxSynchronize", cuCtxSynchronize());
}

}
