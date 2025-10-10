use std::ffi::c_void;

pub type CuDevicePtr = u64;
pub type CuResult = i32;

unsafe extern "C" {
    pub fn tho_cu_memcpy_dtoh_async(
        dst_host: *mut c_void,
        src_device: CuDevicePtr,
        byte_count: usize,
        stream: *mut c_void,
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
}
