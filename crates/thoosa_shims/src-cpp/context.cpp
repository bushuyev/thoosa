#include "errors.h"

#include <cuda.h>

#include "logging.h"

extern "C" {

CUresult tho_cu_init(unsigned int flags) {
    tho_log(CppLogLevel::Info, "tho_cu_init");

    return tho_record_cuda_result("cuInit", cuInit(flags));
}
// Healthy per-device maintenance must not mutate LibTorch's process-wide
// allocator/workspace caches or release a primary context owned by other users.
// Fatal local errors require process relaunch and never reach this path.
CUresult tho_cu_restart(CUdevice device) {
  CUcontext context = nullptr;
  CUresult result = cuDevicePrimaryCtxRetain(&context, device);
  if (result != CUDA_SUCCESS) return tho_record_cuda_result("cuDevicePrimaryCtxRetain", result);
  CUcontext previous = nullptr;
  result = cuCtxGetCurrent(&previous);
  const bool have_previous = result == CUDA_SUCCESS;
  if (result == CUDA_SUCCESS) result = cuCtxSetCurrent(context);
  if (result == CUDA_SUCCESS) result = cuCtxSynchronize();
  const CUresult restore = have_previous ? cuCtxSetCurrent(previous) : CUDA_SUCCESS;
  const CUresult release = cuDevicePrimaryCtxRelease(device);
  if (result != CUDA_SUCCESS) return tho_record_cuda_result("tho_cu_restart", result);
  if (restore != CUDA_SUCCESS) return tho_record_cuda_result("cuCtxSetCurrent", restore);
  return tho_record_cuda_result("cuDevicePrimaryCtxRelease", release);
}

CUresult tho_cu_device_get(CUdevice *device, int ordinal) {
  return tho_record_cuda_result("cuDeviceGet", cuDeviceGet(device, ordinal));
}

CUresult tho_cu_device_primary_ctx_retain(CUcontext *ctx, CUdevice device) {
  tho_log(CppLogLevel::Info, "tho_cu_device_primary_ctx_retain: retained");

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
