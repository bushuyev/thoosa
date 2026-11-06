use std::{ffi::c_void, marker::PhantomData, ptr::NonNull, rc::Rc};

unsafe extern "C" {
    fn tho_inference_mode_new(enabled: i32) -> *mut c_void;
    fn tho_inference_mode_delete(p: *mut c_void);
    fn tho_inference_mode_is_enabled() -> i32;
}

#[must_use]
pub struct InferenceModeGuard {
    ptr: NonNull<c_void>,

    // Important: PyTorch grad/inference modes are thread-local.
    // Do not allow this guard to be Send/Sync.
    _not_send_sync: PhantomData<Rc<()>>,
}

impl InferenceModeGuard {
    pub fn new(enabled: bool) -> Result<Self, thoosa_shared::ThoError> {
        let ptr = unsafe { tho_inference_mode_new(enabled as i32) };
        let ptr = NonNull::new(ptr).ok_or_else(|| crate::cuda_result::cuda_error_from_last_error("tho_inference_mode_new"))?;

        // Return native diagnostics instead of panicking at allocation failure.
        Ok(Self { ptr, _not_send_sync: PhantomData })
    }

    pub fn enabled() -> Result<Self, thoosa_shared::ThoError> {
        Self::new(true)
    }

    pub fn disabled() -> Result<Self, thoosa_shared::ThoError> {
        Self::new(false)
    }

    pub fn is_enabled() -> bool {
        unsafe { tho_inference_mode_is_enabled() != 0 }
    }
}

impl Drop for InferenceModeGuard {
    fn drop(&mut self) {
        unsafe {
            tho_inference_mode_delete(self.ptr.as_ptr());
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_inference_guard_restores_thread_state() -> Result<(), thoosa_shared::ThoError> {
        let before = InferenceModeGuard::is_enabled();
        {
            let _guard = InferenceModeGuard::new(!before)?;
            assert_eq!(InferenceModeGuard::is_enabled(), !before);
        }
        assert_eq!(InferenceModeGuard::is_enabled(), before);
        Ok(())
    }
}
