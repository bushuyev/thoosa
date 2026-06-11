#include "errors.h"

#include <cuda.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime_api.h>
#include <iostream>

#include "logging.h"

extern "C" {

CUresult tho_cu_init(unsigned int flags) {
    tho_log(CppLogLevel::Info, "tho_cu_init");

    return tho_record_cuda_result("cuInit", cuInit(flags));
}

CUresult tho_cu_reset(CUdevice device) {
  try {
    tho_log(CppLogLevel::Info, "tho_cu_reset: entered");

    at::cuda::clearCublasWorkspaces();
    tho_log(CppLogLevel::Info, "tho_cu_reset: clearCublasWorkspaces done");

    c10::cuda::CUDACachingAllocator::emptyCache();
    tho_log(CppLogLevel::Info, "tho_cu_reset: emptyCache done");

    auto record_cuda_result = tho_record_cuda_result("cuDevicePrimaryCtxRelease_v2", cuDevicePrimaryCtxRelease_v2(device));

    tho_log(CppLogLevel::Info, "tho_cu_reset: cuDevicePrimaryCtxRelease_v2 done");

    // auto record_cuda_result = tho_record_cuda_result("cuDevicePrimaryCtxRelease_v2", cuDevicePrimaryCtxRetain(device));
    // return tho_record_cuda_result("cuDevicePrimaryCtxReset", cuDevicePrimaryCtxReset(device));
    return record_cuda_result;
    // return CUresult::CUDA_SUCCESS;//TODO add ThoShimResult and use its errors
  } catch (...) {
    return CUresult::CUDA_ERROR_UNKNOWN;
  }
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
