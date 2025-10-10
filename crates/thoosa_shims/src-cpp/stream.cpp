#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/torch.h>

extern "C" {

  void *tho_get_current_stream(int device) {
    c10::cuda::CUDAStream stream = c10::cuda::getCurrentCUDAStream(device);
    return stream;
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

  void tho_stream_synchronize(void *raw) {
    static_cast<c10::cuda::CUDAStream *>(raw)->synchronize();
  }

  void tho_delete_stream(void *raw) {
    delete static_cast<c10::cuda::CUDAStream *>(raw);
  }

}
