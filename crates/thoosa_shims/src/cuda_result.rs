use std::ffi::{c_char, CStr};

use thoosa_shared::{cuda_cause_from_error_code, ThoError};

pub type CuResult = i32;

pub const CUDA_SUCCESS: CuResult = 0;
pub const CUDA_ERROR_INVALID_HANDLE: CuResult = 400;
pub const CUDA_ERROR_ASSERT: CuResult = 710;
pub const CUDA_ERROR_ILLEGAL_ADDRESS: CuResult = 700;

unsafe extern "C" {
    fn tho_get_last_error_message() -> *const c_char;
    fn tho_get_last_error_code() -> CuResult;
}

fn last_cpp_error_message() -> Option<String> {
    let p = unsafe { tho_get_last_error_message() };
    if p.is_null() {
        return None;
    }

    let msg = unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() };
    if msg.is_empty() {
        None
    } else {
        Some(msg)
    }
}

pub fn last_cpp_error(prefix: &str) -> String {
    match last_cpp_error_message() {
        Some(msg) => format!("{prefix}: {msg}"),
        None => format!("{prefix}: <no message>"),
    }
}

pub fn cuda_error_from_status(label: &str, status: CuResult) -> ThoError {
    let msg = match last_cpp_error_message() {
        Some(detail) => format!("{label} failed with CUDA error code={status}: {detail}"),
        None => format!("{label} failed with CUDA error code={status}"),
    };
    ThoError::new_with_cause(msg, cuda_cause_from_error_code(status))
}

pub fn cuda_error_from_last_error(label: &str) -> ThoError {
    let status = unsafe { tho_get_last_error_code() };
    if status == CUDA_SUCCESS {
        ThoError::from_cuda_message(last_cpp_error(label))
    } else {
        cuda_error_from_status(label, status)
    }
}

pub fn check_cuda_status(status: CuResult, label: &str) -> Result<(), ThoError> {
    if status == CUDA_SUCCESS {
        Ok(())
    } else {
        Err(cuda_error_from_status(label, status))
    }
}
