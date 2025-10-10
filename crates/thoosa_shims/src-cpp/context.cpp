#include <cuda.h>

extern "C" {

CUresult tho_cu_init(unsigned int flags) {
  return cuInit(flags);
}

CUresult tho_cu_device_get(CUdevice *device, int ordinal) {
  return cuDeviceGet(device, ordinal);
}

CUresult tho_cu_device_primary_ctx_retain(CUcontext *ctx, CUdevice device) {
  return cuDevicePrimaryCtxRetain(ctx, device);
}

CUresult tho_cu_device_primary_ctx_release(CUdevice device) {
  return cuDevicePrimaryCtxRelease(device);
}

CUresult tho_cu_device_get_attribute(int *pi, int attrib, CUdevice device) {
  return cuDeviceGetAttribute(pi, static_cast<CUdevice_attribute>(attrib), device);
}

CUresult tho_cu_ctx_set_current(CUcontext ctx) {
  return cuCtxSetCurrent(ctx);
}

CUresult tho_cu_ctx_get_current(CUcontext *ctx) {
  return cuCtxGetCurrent(ctx);
}

CUresult tho_cu_ctx_synchronize() {
  return cuCtxSynchronize();
}

}
