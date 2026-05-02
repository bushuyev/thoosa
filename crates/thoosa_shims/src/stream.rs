// tho_cuda_stream.rs

use std::ffi::c_void;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;
use thoosa_shared::ThoError;

use crate::cuda_result::{check_cuda_status, cuda_error_from_last_error, cuda_error_from_status};
pub use crate::cuda_result::{CuResult, CUDA_ERROR_ASSERT, CUDA_ERROR_ILLEGAL_ADDRESS, CUDA_SUCCESS};

type CudaStreamPtr = *mut c_void;
type CudaStreamGuardPtr = *mut c_void;

unsafe extern "C" {
    fn tho_get_current_stream(device: i32) -> CudaStreamPtr;
    fn tho_get_stream_from_pool(device: i32, high_pri: i32) -> CudaStreamPtr;
    fn tho_stream_device(stream: CudaStreamPtr) -> i32;
    fn tho_stream_raw(stream: CudaStreamPtr) -> *mut c_void;

    fn tho_stream_guard_new(stream: CudaStreamPtr) -> CudaStreamGuardPtr;
    fn tho_stream_guard_delete(guard: CudaStreamGuardPtr);

    fn tho_stream_synchronize(stream: CudaStreamPtr) -> CuResult;
    fn tho_delete_stream(stream: CudaStreamPtr);
}

struct Inner {
    ptr: NonNull<c_void>,
    device: i32,
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe { tho_delete_stream(self.ptr.as_ptr()) };
    }
}

#[derive(Clone)]
pub struct ThoCudaStream(Arc<Inner>);

impl Debug for ThoCudaStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThoCudaStream").finish()
    }
}

// The stream handle is just an owned wrapper around a C++ value object.
// The *guard* is the thread-affine thing, not the stream handle itself.
unsafe impl Send for ThoCudaStream {}
unsafe impl Sync for ThoCudaStream {}

impl ThoCudaStream {
    pub fn from_pool(device: i32, high_pri: bool) -> Result<Self, ThoError> {
        let ptr = unsafe { tho_get_stream_from_pool(device, if high_pri { 1 } else { 0 }) };
        let ptr = NonNull::new(ptr).ok_or_else(|| cuda_error_from_last_error("tho_get_stream_from_pool"))?;
        Ok(Self(Arc::new(Inner { ptr, device })))
    }

    pub fn current(device: i32) -> Result<Self, ThoError> {
        let ptr = unsafe { tho_get_current_stream(device) };
        let ptr = NonNull::new(ptr).ok_or_else(|| cuda_error_from_last_error("tho_get_current_stream"))?;
        Ok(Self(Arc::new(Inner { ptr, device })))
    }

    /// Takes ownership of a stream wrapper pointer allocated by the C++ shim.
    ///
    /// # Safety
    /// `ptr` must come from this shim and must be uniquely owned by the caller.
    pub unsafe fn from_owned_raw(ptr: CudaStreamPtr) -> Result<Self, ThoError> {
        let ptr = NonNull::new(ptr).ok_or_else(|| cuda_error_from_status("from_owned_raw null stream pointer", crate::cuda_result::CUDA_ERROR_INVALID_HANDLE))?;
        let device = unsafe { tho_stream_device(ptr.as_ptr()) };
        if device < 0 {
            return Err(cuda_error_from_last_error("tho_stream_device"));
        }
        Ok(Self(Arc::new(Inner { ptr, device })))
    }

    pub fn synchronize(&self) -> Result<(), ThoError> {
        let status = unsafe { tho_stream_synchronize(self.ffi_ptr()) };
        check_cuda_status(status, "tho_stream_synchronize")
    }

    /// Underlying CUDA stream handle for custom CUDA code.
    pub fn raw_cuda_stream(&self) -> Result<*mut c_void, ThoError> {
        let p = unsafe { tho_stream_raw(self.ffi_ptr()) };
        NonNull::new(p)
            .map(|p| p.as_ptr())
            .ok_or_else(|| cuda_error_from_last_error("tho_stream_raw"))
    }

    pub fn device(&self) -> i32 {
        self.0.device
    }

    pub(crate) fn ffi_ptr(&self) -> CudaStreamPtr {
        self.0.ptr.as_ptr()
    }

    pub fn ptr(&self) -> CudaStreamPtr { self.0.ptr.as_ptr() }
    pub fn as_mut_ptr(&self) -> *mut c_void { self.0.ptr.as_ptr().cast() }

}

/// RAII guard backed by real `c10::cuda::CUDAStreamGuard` on the C++ side.
///
/// Intentionally !Send and !Sync.
/// Do not hold across `.await`.
pub struct ThoStreamGuard {
    ptr: NonNull<c_void>,
    _not_send_or_sync: PhantomData<Rc<()>>,
}

impl ThoStreamGuard {
    pub fn new(stream: &ThoCudaStream) -> Result<Self, ThoError> {
        let ptr = unsafe { tho_stream_guard_new(stream.ffi_ptr()) };
        let ptr = NonNull::new(ptr).ok_or_else(|| cuda_error_from_last_error("tho_stream_guard_new"))?;
        Ok(Self {
            ptr,
            _not_send_or_sync: PhantomData,
        })
    }
}

impl Drop for ThoStreamGuard {
    fn drop(&mut self) {
        unsafe { tho_stream_guard_delete(self.ptr.as_ptr()) };
    }
}

/// Run a synchronous section with `stream` installed as current.
///
/// This closure must stay synchronous: no `.await` inside.
pub fn with_stream<F, R>(stream: &ThoCudaStream, f: F) -> Result<R, ThoError>
where
    F: FnOnce() -> R,
{
    let _guard = ThoStreamGuard::new(stream)?;
    Ok(f())
}
