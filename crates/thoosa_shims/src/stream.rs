use std::ffi::c_void;
use std::fmt::Debug;
use std::sync::Arc;

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

struct Inner {
    ptr: CudaStreamPtr,
    device: i32,
}
impl Drop for Inner {
    fn drop(&mut self) {
        unsafe { tho_delete_stream(self.ptr) }
    }
}

#[derive(Clone)]
pub struct ThoCudaStream (Arc<Inner>);


impl Debug for ThoCudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThoCudaStream").finish()
    }
}
impl ThoCudaStream {
    pub fn new(device: i32) -> Self {
        let ptr = unsafe { tho_get_stream_from_pool(device, false) };
        Self(Arc::new(Inner { ptr, device }))
    }

    pub fn current(device: i32) -> Self {
        let ptr = unsafe { tho_get_current_stream(device) };
        Self(Arc::new(Inner { ptr, device }))
    }

    pub fn from_ptr(ptr: CudaStreamPtr) -> Self {
        Self(Arc::new(Inner { ptr, device: 0 }))
    }

    pub fn synchronize(&self) { unsafe { tho_stream_synchronize(self.0.ptr) } }
    pub fn ptr(&self) -> CudaStreamPtr { self.0.ptr }
    pub fn device(&self) -> i32 { self.0.device }
    pub fn as_mut_ptr(&self) -> *mut c_void { self.0.ptr.cast() }
}



/// RAII guard – exactly like `c10::cuda::CUDAStreamGuard`
pub struct ThoStreamGuard {
    prev: ThoCudaStream,
}

unsafe impl Send for ThoCudaStream {}
unsafe impl Sync for ThoCudaStream {}
unsafe impl Send for ThoStreamGuard {}
impl ThoStreamGuard {
    pub fn set(stream: &ThoCudaStream) -> Self {
        let prev = ThoCudaStream::current(stream.device());
        unsafe { tho_set_current_stream(stream.ptr()) }
        Self { prev }
    }
}

impl Drop for ThoStreamGuard {
    fn drop(&mut self) {
        unsafe { tho_set_current_stream(self.prev.ptr()) }
    }
}
