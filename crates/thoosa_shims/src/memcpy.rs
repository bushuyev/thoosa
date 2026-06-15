use std::ffi::c_void;

pub use crate::cuda_result::CuResult;

pub type CuDevicePtr = u64;

unsafe extern "C" {
    pub fn tho_cu_memcpy_dtoh_async(
        dst_host: *mut c_void,
        src_device: CuDevicePtr,
        byte_count: usize,
        stream: *mut c_void,
    ) -> CuResult;

    pub fn tho_cu_pointer_is_device_memory(
        is_device_memory: *mut i32,
        ptr: CuDevicePtr,
    ) -> CuResult;

    pub fn tho_cu_memcpy_htod_async(
        dst_device: CuDevicePtr,
        src_host: *const c_void,
        byte_count: usize,
        stream: *mut c_void,
    ) -> CuResult;

    pub fn tho_cu_mem_host_alloc(
        ptr: *mut *mut c_void,
        byte_count: usize,
        flags: u32,
    ) -> CuResult;

    pub fn tho_cu_mem_free_host(ptr: *mut c_void) -> CuResult;

    pub fn tho_cu_mem_host_register(
        ptr: *mut c_void,
        byte_count: usize,
        flags: u32,
    ) -> CuResult;

    pub fn tho_cu_mem_host_unregister(ptr: *mut c_void) -> CuResult;
}
