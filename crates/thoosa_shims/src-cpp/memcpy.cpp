#include "errors.h"

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
    return tho_record_cuda_result("cuMemcpyDtoHAsync", cuMemcpyDtoHAsync(dst_host, src_device, byte_count, cu_stream));
  }

  CUresult tho_cu_pointer_is_device_memory(
      int *is_device_memory,
      CUdeviceptr ptr) {
    if (is_device_memory == nullptr) {
      return tho_set_last_error_code("cuPointerGetAttribute", CUDA_ERROR_INVALID_VALUE, "null is_device_memory");
    }

    unsigned int memory_type = 0;
    CUresult result = cuPointerGetAttribute(
        &memory_type,
        CU_POINTER_ATTRIBUTE_MEMORY_TYPE,
        ptr);

    if (result != CUDA_SUCCESS) {
      *is_device_memory = 0;
      return tho_record_cuda_result("cuPointerGetAttribute", result);
    }

    *is_device_memory = memory_type == CU_MEMORYTYPE_DEVICE;
    tho_clear_last_error();
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
    return tho_record_cuda_result("cuMemcpyHtoDAsync", cuMemcpyHtoDAsync(dst_device, src_host, byte_count, cu_stream));
  }

  CUresult tho_cu_mem_host_alloc(
      void **ptr,
      size_t byte_count,
      unsigned int flags) {
    return tho_record_cuda_result("cuMemHostAlloc", cuMemHostAlloc(ptr, byte_count, flags));
  }

  CUresult tho_cu_mem_free_host(void *ptr) {
    return tho_record_cuda_result("cuMemFreeHost", cuMemFreeHost(ptr));
  }

  CUresult tho_cu_mem_host_register(
      void *ptr,
      size_t byte_count,
      unsigned int flags) {
    return tho_record_cuda_result("cuMemHostRegister", cuMemHostRegister(ptr, byte_count, flags));
  }

  CUresult tho_cu_mem_host_unregister(void *ptr) {
    return tho_record_cuda_result("cuMemHostUnregister", cuMemHostUnregister(ptr));
  }
}
