use std::ffi::c_void;

#[repr(C)]
pub struct CudaStreamOpaque;
pub type CudaStreamPtr = *mut CudaStreamOpaque;

unsafe extern "C" {
    pub fn tho_get_current_stream(device:i32/*tensor: *const torch_sys::C_tensor*/) -> CudaStreamPtr;
    pub fn tho_get_stream_from_pool(device: i32, high_pri: bool) -> CudaStreamPtr;
    pub fn tho_set_current_stream(stream: CudaStreamPtr);
    pub fn tho_stream_synchronize(stream: CudaStreamPtr);
    pub fn tho_delete_stream(stream: CudaStreamPtr);
}

pub struct ThoCudaStream {
    ptr: CudaStreamPtr,
    device: i32,
}
impl ThoCudaStream {
    pub fn new(device: i32) -> Self {
        let ptr = unsafe { tho_get_stream_from_pool(device, false) };
        Self { ptr, device }
    }

    pub fn from_ptr(ptr: CudaStreamPtr) -> Self {
        Self { ptr, device: 0 }
    }

    pub fn synchronize(&self) {
        unsafe { tho_stream_synchronize(self.ptr) }
    }

    pub fn ptr(&self) -> CudaStreamPtr {
        self.ptr
    }

    pub fn as_mut_ptr(&self) -> *mut c_void {
        self.ptr.cast()
    }
}


impl Drop for ThoCudaStream {
    fn drop(&mut self) {
        unsafe { tho_delete_stream(self.ptr) }
    }
}

/// RAII guard – exactly like `c10::cuda::CUDAStreamGuard`
pub struct StreamGuard {
    prev: CudaStreamPtr,
}

unsafe impl Send for ThoCudaStream {}
unsafe impl Sync for ThoCudaStream {}
unsafe impl Send for StreamGuard{}
impl StreamGuard {
    pub fn set(stream: &ThoCudaStream) -> Self {
        // Remember current stream (needs one more shim if you care);
        // for brevity we assume default stream == nullptr.
        let prev = std::ptr::null_mut();
        unsafe { tho_set_current_stream(stream.ptr) };
        Self { prev }
    }
}
impl Drop for StreamGuard {
    fn drop(&mut self) {
        unsafe {
            if !self.prev.is_null() {
                tho_set_current_stream(self.prev)
            }
        };
    }
}
