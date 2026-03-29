// tho_cuda_stream.rs

use std::ffi::{c_char, c_void, CStr};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;
use thoosa_shared::ThoError;

type CudaStreamPtr = *mut c_void;
type CudaStreamGuardPtr = *mut c_void;
type CuResult = i32;

const CUDA_SUCCESS: CuResult = 0;

unsafe extern "C" {
    fn tho_get_current_stream(device: i32) -> CudaStreamPtr;
    fn tho_get_stream_from_pool(device: i32, high_pri: i32) -> CudaStreamPtr;
    fn tho_stream_device(stream: CudaStreamPtr) -> i32;
    fn tho_stream_raw(stream: CudaStreamPtr) -> *mut c_void;

    // Low-level/debug only. Prefer the guard API.
    fn tho_set_current_stream(stream: CudaStreamPtr) -> i32;

    fn tho_stream_guard_new(stream: CudaStreamPtr) -> CudaStreamGuardPtr;
    fn tho_stream_guard_delete(guard: CudaStreamGuardPtr);

    fn tho_stream_synchronize(stream: CudaStreamPtr) -> CuResult;
    fn tho_delete_stream(stream: CudaStreamPtr);

    fn tho_get_last_error_message() -> *const c_char;
}

fn last_cpp_error(prefix: &str) -> String {
    let msg = unsafe {
        let p = tho_get_last_error_message();
        if p.is_null() {
            "<no message>".to_string()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    format!("{prefix}: {msg}")
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
        let ptr = NonNull::new(ptr).ok_or_else(|| ThoError::new(last_cpp_error("tho_get_stream_from_pool failed")))?;
        Ok(Self(Arc::new(Inner { ptr, device })))
    }

    pub fn current(device: i32) -> Result<Self, ThoError> {
        let ptr = unsafe { tho_get_current_stream(device) };
        let ptr = NonNull::new(ptr).ok_or_else(|| ThoError::new(last_cpp_error("tho_get_current_stream failed")))?;
        Ok(Self(Arc::new(Inner { ptr, device })))
    }

    /// Takes ownership of a stream wrapper pointer allocated by the C++ shim.
    ///
    /// # Safety
    /// `ptr` must come from this shim and must be uniquely owned by the caller.
    pub unsafe fn from_owned_raw(ptr: CudaStreamPtr) -> Result<Self, ThoError> {
        let ptr = NonNull::new(ptr).ok_or_else(|| ThoError::new("from_owned_raw: null stream pointer"))?;
        let device = tho_stream_device(ptr.as_ptr());
        if device < 0 {
            return Err(ThoError::new(last_cpp_error("tho_stream_device failed")));
        }
        Ok(Self(Arc::new(Inner { ptr, device })))
    }

    pub fn synchronize(&self) -> Result<(), ThoError> {
        let r = unsafe { tho_stream_synchronize(self.ffi_ptr()) };
        if r == CUDA_SUCCESS {
            Ok(())
        } else {
            Err(ThoError::new(last_cpp_error("stream synchronize failed")))
        }
    }

    /// Underlying CUDA stream handle for custom CUDA code.
    pub fn raw_cuda_stream(&self) -> Result<*mut c_void, ThoError> {
        let p = unsafe { tho_stream_raw(self.ffi_ptr()) };
        NonNull::new(p)
            .map(|p| p.as_ptr())
            .ok_or_else(|| ThoError::new(last_cpp_error("tho_stream_raw failed")))
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
        let ptr = NonNull::new(ptr).ok_or_else(|| ThoError::new(last_cpp_error("tho_stream_guard_new failed")))?;
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

/// Your old "tho device index" shape:
/// 0 => CPU / no CUDA stream
/// 1 => CUDA:0
/// 2 => CUDA:1
pub fn with_pooled_cuda_stream<F, R>(
    tho_device_idx: usize,
    high_pri: bool,
    f: F,
) -> Result<(R, Option<ThoCudaStream>), ThoError>
where
    F: FnOnce(Option<&ThoCudaStream>) -> R,
{
    if tho_device_idx == 0 {
        return Ok((f(None), None));
    }

    let cuda_device = (tho_device_idx - 1) as i32;
    let stream = ThoCudaStream::from_pool(cuda_device, high_pri)?;
    let out = with_stream(&stream, || f(Some(&stream)))?;
    Ok((out, Some(stream)))
}