use std::ffi::c_char;
use std::slice;
use std::str;

#[repr(C)]
#[derive(Copy, Clone)]
pub enum CppLogLevel {
    Trace = 0,
    Debug = 1,
    Info  = 2,
    Warn  = 3,
    Error = 4,
}

pub type CppLogCallback = extern "C" fn(
    level: CppLogLevel,
    msg_ptr: *const c_char,
    msg_len: usize,
);

unsafe extern "C" {
    fn tho_set_log_callback(cb: Option<CppLogCallback>);
}

extern "C" fn cpp_log_callback(
    level: CppLogLevel,
    msg_ptr: *const c_char,
    msg_len: usize,
) {
    if msg_ptr.is_null() {
        return;
    }

    let bytes = unsafe {
        slice::from_raw_parts(msg_ptr as *const u8, msg_len)
    };

    let msg = str::from_utf8(bytes).unwrap_or("<invalid utf8 from C++ shim>");

    match level {
        CppLogLevel::Trace => tracing::trace!(target: "cpp_shim", "{msg}"),
        CppLogLevel::Debug => tracing::debug!(target: "cpp_shim", "{msg}"),
        CppLogLevel::Info  => tracing::info!(target: "cpp_shim", "{msg}"),
        CppLogLevel::Warn  => tracing::warn!(target: "cpp_shim", "{msg}"),
        CppLogLevel::Error => tracing::error!(target: "cpp_shim", "{msg}"),
    }
}

pub fn init_cpp_logging() {
    unsafe {
        tho_set_log_callback(Some(cpp_log_callback));
    }
}