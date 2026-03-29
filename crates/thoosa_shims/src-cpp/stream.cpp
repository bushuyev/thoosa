// stream.cpp

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>

#include <exception>
#include <string>

namespace {

thread_local std::string tho_last_error;

void tho_set_last_error_msg(const char* where, const char* msg) {
  tho_last_error.clear();
  tho_last_error += where ? where : "<unknown>";
  tho_last_error += ": ";
  tho_last_error += msg ? msg : "<null>";
}

void tho_set_last_error_ex(const char* where, const std::exception& e) {
  tho_set_last_error_msg(where, e.what());
}

// Opaque holder for the real PyTorch RAII guard.
struct ThoCudaStreamGuard {
  c10::cuda::CUDAStreamGuard guard;
  explicit ThoCudaStreamGuard(c10::cuda::CUDAStream s) : guard(s) {}
};

} // namespace

extern "C" {

const char* tho_get_last_error_message() {
  return tho_last_error.c_str();
}

void* tho_get_current_stream(int device) {
  try {
    auto s = c10::cuda::getCurrentCUDAStream(device);
    return new c10::cuda::CUDAStream(s);
  } catch (const c10::Error& e) {
    tho_set_last_error_msg("tho_get_current_stream", e.what_without_backtrace());
    return nullptr;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_get_current_stream", e);
    return nullptr;
  } catch (...) {
    tho_set_last_error_msg("tho_get_current_stream", "unknown C++ exception");
    return nullptr;
  }
}

void* tho_get_stream_from_pool(int device, bool high_pri) {
  try {
    auto s = c10::cuda::getStreamFromPool(high_pri, device);
    return new c10::cuda::CUDAStream(s);
  } catch (const c10::Error& e) {
    tho_set_last_error_msg("tho_get_stream_from_pool", e.what_without_backtrace());
    return nullptr;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_get_stream_from_pool", e);
    return nullptr;
  } catch (...) {
    tho_set_last_error_msg("tho_get_stream_from_pool", "unknown C++ exception");
    return nullptr;
  }
}

int tho_stream_device(void* raw_stream) {
  try {
    if (!raw_stream) {
      tho_set_last_error_msg("tho_stream_device", "null stream");
      return -1;
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    return static_cast<int>(s->device_index());
  } catch (const c10::Error& e) {
    tho_set_last_error_msg("tho_stream_device", e.what_without_backtrace());
    return -1;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_stream_device", e);
    return -1;
  } catch (...) {
    tho_set_last_error_msg("tho_stream_device", "unknown C++ exception");
    return -1;
  }
}

void* tho_stream_raw(void* raw_stream) {
  try {
    if (!raw_stream) {
      tho_set_last_error_msg("tho_stream_raw", "null stream");
      return nullptr;
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    return reinterpret_cast<void*>(s->stream());
  } catch (const c10::Error& e) {
    tho_set_last_error_msg("tho_stream_raw", e.what_without_backtrace());
    return nullptr;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_stream_raw", e);
    return nullptr;
  } catch (...) {
    tho_set_last_error_msg("tho_stream_raw", "unknown C++ exception");
    return nullptr;
  }
}

int tho_set_current_stream(void* raw_stream) {
  try {
    if (!raw_stream) {
      tho_set_last_error_msg("tho_set_current_stream", "null stream");
      return 0;
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    c10::cuda::setCurrentCUDAStream(*s);
    return 1;
  } catch (const c10::Error& e) {
    tho_set_last_error_msg("tho_set_current_stream", e.what_without_backtrace());
    return 0;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_set_current_stream", e);
    return 0;
  } catch (...) {
    tho_set_last_error_msg("tho_set_current_stream", "unknown C++ exception");
    return 0;
  }
}

void* tho_stream_guard_new(void* raw_stream) {
  try {
    if (!raw_stream) {
      tho_set_last_error_msg("tho_stream_guard_new", "null stream");
      return nullptr;
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    return new ThoCudaStreamGuard(*s);
  } catch (const c10::Error& e) {
    tho_set_last_error_msg("tho_stream_guard_new", e.what_without_backtrace());
    return nullptr;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_stream_guard_new", e);
    return nullptr;
  } catch (...) {
    tho_set_last_error_msg("tho_stream_guard_new", "unknown C++ exception");
    return nullptr;
  }
}

void tho_stream_guard_delete(void* raw_guard) {
  delete static_cast<ThoCudaStreamGuard*>(raw_guard);
}

CUresult tho_stream_synchronize(void* raw_stream) {
  try {
    if (!raw_stream) {
      tho_set_last_error_msg("tho_stream_synchronize", "null stream");
      return CUDA_ERROR_INVALID_HANDLE;
    }
    auto* s = static_cast<c10::cuda::CUDAStream*>(raw_stream);
    s->synchronize();
    return CUDA_SUCCESS;
  } catch (const c10::Error& e) {
    tho_set_last_error_msg("tho_stream_synchronize", e.what_without_backtrace());
    return CUDA_ERROR_UNKNOWN;
  } catch (const std::exception& e) {
    tho_set_last_error_ex("tho_stream_synchronize", e);
    return CUDA_ERROR_UNKNOWN;
  } catch (...) {
    tho_set_last_error_msg("tho_stream_synchronize", "unknown C++ exception");
    return CUDA_ERROR_UNKNOWN;
  }
}

void tho_delete_stream(void* raw_stream) {
  delete static_cast<c10::cuda::CUDAStream*>(raw_stream);
}

} // extern "C"