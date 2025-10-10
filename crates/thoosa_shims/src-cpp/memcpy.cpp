#include <cuda.h>

extern "C" {
  CUresult tho_cu_memcpy_dtoh_async(
      void *dst_host,
      CUdeviceptr src_device,
      size_t byte_count,
      void *stream) {
    return cuMemcpyDtoHAsync(dst_host, src_device, byte_count,
                             reinterpret_cast<CUstream>(stream));
  }

  CUresult tho_cu_memcpy_htod_async(
      CUdeviceptr dst_device,
      const void *src_host,
      size_t byte_count,
      void *stream) {
    return cuMemcpyHtoDAsync(dst_device, src_host, byte_count,
                             reinterpret_cast<CUstream>(stream));
  }

  CUresult tho_cu_mem_host_alloc(
      void **ptr,
      size_t byte_count,
      unsigned int flags) {
    return cuMemHostAlloc(ptr, byte_count, flags);
  }

  CUresult tho_cu_mem_free_host(void *ptr) {
    return cuMemFreeHost(ptr);
  }
}
