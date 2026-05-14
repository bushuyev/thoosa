use std::{
    ffi::c_void,
    marker::PhantomData,
    ptr::NonNull,
    rc::Rc,
};

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
    pub fn new(enabled: bool) -> Self {
        let ptr = unsafe { tho_inference_mode_new(enabled as i32) };
        let ptr = NonNull::new(ptr).expect("failed to create c10::InferenceMode guard");

        Self {
            ptr,
            _not_send_sync: PhantomData,
        }
    }

    pub fn enabled() -> Self {
        Self::new(true)
    }

    pub fn disabled() -> Self {
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