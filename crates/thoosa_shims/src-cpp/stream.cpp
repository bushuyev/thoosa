#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

extern "C" {

  thread_local std::string tho_last_error;

  const char* tho_get_last_error_message() {
    if (tho_last_error.empty()) {
      return "";
    }
    return tho_last_error.c_str();
  }

  void *tho_get_current_stream(int device) {
    c10::cuda::CUDAStream s = c10::cuda::getCurrentCUDAStream(device);
    return new c10::cuda::CUDAStream(s);
  }

  void *tho_get_stream_from_pool(int device, bool high_pri) {
    auto s = c10::cuda::getStreamFromPool(high_pri, device);
    // Wrap the stream in a heap object so that we can move it across FFI.
    return new c10::cuda::CUDAStream(s);
  }

  void tho_set_current_stream(void *raw) {
    auto *s = static_cast<c10::cuda::CUDAStream *>(raw);
    c10::cuda::setCurrentCUDAStream(*s);
  }

  CUresult tho_stream_synchronize(void *stream) {
    // static_cast<c10::cuda::CUDAStream *>(raw)->synchronize();
    try {
      CUstream cu_stream = nullptr;
      if (stream != nullptr) {
        auto* cuda_stream = static_cast<c10::cuda::CUDAStream*>(stream);
        cu_stream = cuda_stream->stream();
      }

      // Since the rest of your shim uses Driver API, use Driver API here too.
      return cuStreamSynchronize(cu_stream);
    } catch (const c10::Error& e) {
      tho_last_error = e.what_without_backtrace();
      return CUDA_ERROR_UNKNOWN;
    } catch (const std::exception& e) {
      tho_last_error = e.what();
      return CUDA_ERROR_UNKNOWN;
    } catch (...) {
      tho_last_error = "unknown C++ exception in tho_cu_stream_synchronize";
      return CUDA_ERROR_UNKNOWN;
    }
  }

  void tho_delete_stream(void *raw) {
    delete static_cast<c10::cuda::CUDAStream *>(raw);
  }

}
