#include <cuda.h>
#include <c10/cuda/CUDAStream.h>

extern "C" {
  CUresult tho_cu_memcpy_dtoh_async(
      void *dst_host,
      CUdeviceptr src_device,
      size_t byte_count,
      void *stream) {
    CUstream cu_stream = nullptr;
    if (stream != nullptr) {
      auto *cuda_stream = static_cast<c10::cuda::CUDAStream *>(stream);
      cu_stream = cuda_stream->stream();
    }
    return cuMemcpyDtoHAsync(dst_host, src_device, byte_count, cu_stream);
  }

  CUresult tho_cu_pointer_is_device_memory(
      int *is_device_memory,
      CUdeviceptr ptr) {
    if (is_device_memory == nullptr) {
      return CUDA_ERROR_INVALID_VALUE;
    }

    unsigned int memory_type = 0;
    CUresult result = cuPointerGetAttribute(
        &memory_type,
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        ptr);

    if (result != CUDA_SUCCESS) {
      *is_device_memory = 0;
      return result;
    }

    *is_device_memory = memory_type == CU_MEMORYTYPE_DEVICE;
    return CUDA_SUCCESS;
  }

  CUresult tho_cu_memcpy_htod_async(
      CUdeviceptr dst_device,
      const void *src_host,
      size_t byte_count,
      void *stream) {
    CUstream cu_stream = nullptr;
    if (stream != nullptr) {
      auto *cuda_stream = static_cast<c10::cuda::CUDAStream *>(stream);
      cu_stream = cuda_stream->stream();
    }
    return cuMemcpyHtoDAsync(dst_device, src_host, byte_count, cu_stream);
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
