// stream.cpp

#include "errors.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

namespace {

// Opaque holder for the real PyTorch RAII guard.
///https://github.com/pytorch/pytorch/blob/v2.7.0-rc5/c10/cuda/CUDAStream.h
///  prefer using 'CUDAStreamGuard' instead
///  * (which will switch both your current device and current stream in the way you
///  * expect, and reset it back to its original state afterward
struct ThoCudaStreamGuard {
  c10::cuda::CUDAStreamGuard guard;
  explicit ThoCudaStreamGuard(c10::cuda::CUDAStream s) : guard(s) {}
};

} // namespace

extern "C" {

void* tho_get_current_stream(int device) {
  try {
    auto s = c10::cuda::getCurrentCUDAStream(device);
    tho_clear_last_error();
    return new c10::cuda::CUDAStream(s);
  } catch (const c10::Error& e) {
    tho_set_last_error_cuda_msg("tho_get_current_stream", e.what_without_backtrace(), CUDA_ERROR_UNKNOWN);
    return nullptr;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_get_current_stream", e);
    return nullptr;
  } catch (...) {
    tho_set_last_error_code("tho_get_current_stream", CUDA_ERROR_UNKNOWN, "unknown C++ exception");
    return nullptr;
  }
}

// Match the Rust C ABI integer flag exactly.
void* tho_get_stream_from_pool(int device, int high_pri) {
  try {
    auto s = c10::cuda::getStreamFromPool(high_pri != 0, device);
    tho_clear_last_error();
    return new c10::cuda::CUDAStream(s);
  } catch (const c10::Error& e) {
    tho_set_last_error_cuda_msg("tho_get_stream_from_pool", e.what_without_backtrace(), CUDA_ERROR_UNKNOWN);
    return nullptr;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_get_stream_from_pool", e);
    return nullptr;
  } catch (...) {
    tho_set_last_error_code("tho_get_stream_from_pool", CUDA_ERROR_UNKNOWN, "unknown C++ exception");
    return nullptr;
  }
}

int tho_stream_device(void* raw_stream) {
  try {
    if (!raw_stream) {
      tho_set_last_error_code("tho_stream_device", CUDA_ERROR_INVALID_HANDLE, "null stream");
      return -1;
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    tho_clear_last_error();
    return static_cast<int>(s->device_index());
  } catch (const c10::Error& e) {
    tho_set_last_error_cuda_msg("tho_stream_device", e.what_without_backtrace(), CUDA_ERROR_UNKNOWN);
    return -1;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_stream_device", e);
    return -1;
  } catch (...) {
    tho_set_last_error_code("tho_stream_device", CUDA_ERROR_UNKNOWN, "unknown C++ exception");
    return -1;
  }
}

CUresult tho_stream_raw(void* raw_stream, void** out) {
  try {
    if (!raw_stream || !out) {
      return tho_set_last_error_code("tho_stream_raw", CUDA_ERROR_INVALID_HANDLE, "null stream");
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    tho_clear_last_error();
    // A null CUDA handle is the valid default stream, not an error.
    *out = reinterpret_cast<void*>(s->stream());
    return CUDA_SUCCESS;
  } catch (const c10::Error& e) {
    return tho_set_last_error_cuda_msg("tho_stream_raw", e.what_without_backtrace(), CUDA_ERROR_UNKNOWN);
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_stream_raw", e);
    return CUDA_ERROR_UNKNOWN;
  } catch (...) {
    return tho_set_last_error_code("tho_stream_raw", CUDA_ERROR_UNKNOWN, "unknown C++ exception");
  }
}

CUresult tho_set_current_stream(void* raw_stream) {
  try {
    if (!raw_stream) {
      return tho_set_last_error_code("tho_set_current_stream", CUDA_ERROR_INVALID_HANDLE, "null stream");
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    c10::cuda::setCurrentCUDAStream(*s);
    tho_clear_last_error();
    return CUDA_SUCCESS;
  } catch (const c10::Error& e) {
    return tho_set_last_error_cuda_msg("tho_set_current_stream", e.what_without_backtrace(), CUDA_ERROR_UNKNOWN);
  } catch (const std::exception& e) {
    return tho_set_last_error_code("tho_set_current_stream", CUDA_ERROR_UNKNOWN, e.what());
  } catch (...) {
    return tho_set_last_error_code("tho_set_current_stream", CUDA_ERROR_UNKNOWN, "unknown C++ exception");
  }
}

void* tho_stream_guard_new(void* raw_stream) {
  try {
    if (!raw_stream) {
      tho_set_last_error_code("tho_stream_guard_new", CUDA_ERROR_INVALID_HANDLE, "null stream");
      return nullptr;
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    tho_clear_last_error();
    return new ThoCudaStreamGuard(*s);
  } catch (const c10::Error& e) {
    tho_set_last_error_cuda_msg("tho_stream_guard_new", e.what_without_backtrace(), CUDA_ERROR_UNKNOWN);
    return nullptr;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_stream_guard_new", e);
    return nullptr;
  } catch (...) {
    tho_set_last_error_code("tho_stream_guard_new", CUDA_ERROR_UNKNOWN, "unknown C++ exception");
    return nullptr;
  }
}

void tho_stream_guard_delete(void* raw_guard) {
  delete static_cast<ThoCudaStreamGuard*>(raw_guard);
}

CUresult tho_stream_synchronize(void* raw_stream) {
  try {
    if (!raw_stream) {
      return tho_set_last_error_code("tho_stream_synchronize", CUDA_ERROR_INVALID_HANDLE, "null stream");
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    s->synchronize();
    tho_clear_last_error();
    return CUDA_SUCCESS;
  } catch (const c10::Error& e) {
    return tho_set_last_error_cuda_msg("tho_stream_synchronize", e.what_without_backtrace(), CUDA_ERROR_UNKNOWN);
  } catch (const std::exception& e) {
    return tho_set_last_error_code("tho_stream_synchronize", CUDA_ERROR_UNKNOWN, e.what());
  } catch (...) {
    return tho_set_last_error_code("tho_stream_synchronize", CUDA_ERROR_UNKNOWN, "unknown C++ exception");
  }
}

void tho_delete_stream(void* raw_stream) {
  delete static_cast<c10::cuda::CUDAStream*>(raw_stream);
}

} // extern "C"
