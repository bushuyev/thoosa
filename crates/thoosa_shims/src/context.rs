use std::ffi::c_void;
use std::sync::Arc;
use log::{debug, error};
use thoosa_shared::ThoError;

use crate::cuda_result::{check_cuda_status, cuda_error_from_status, CuResult, CUDA_SUCCESS};

type CuDevice = i32;
type CuContext = *mut c_void;

const CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED: i32 = 115;

unsafe extern "C" {
    pub fn tho_cu_init(flags: u32) -> CuResult;

    pub fn tho_cu_reset(device: CuDevice) -> CuResult;
    pub fn tho_cu_device_get(device: *mut CuDevice, ordinal: i32) -> CuResult;
    pub fn tho_cu_device_primary_ctx_retain(ctx: *mut CuContext, device: CuDevice) -> CuResult;
    pub fn tho_cu_device_primary_ctx_release(device: CuDevice) -> CuResult;
    pub fn tho_cu_device_get_attribute(value: *mut i32, attrib: i32, device: CuDevice) -> CuResult;
    pub fn tho_cu_ctx_set_current(ctx: CuContext) -> CuResult;
    pub fn tho_cu_ctx_get_current(ctx: *mut CuContext) -> CuResult;
    pub fn tho_cu_ctx_synchronize() -> CuResult;
}

#[derive(Debug)]
pub struct ThoCudaContext {
    device: i32,
    cu_device: CuDevice,
    ctx: CuContext,
    has_memory_pools: bool,
}

impl ThoCudaContext {
    pub fn new(device: i32) -> Result<Self, ThoError> {
        unsafe {
            check_cuda_status(tho_cu_init(0), "cuInit")?;

            let mut cu_device: CuDevice = 0;
            check_cuda_status(tho_cu_device_get(&mut cu_device, device), "cuDeviceGet")?;

            let mut ctx: CuContext = std::ptr::null_mut();
            check_cuda_status(
                tho_cu_device_primary_ctx_retain(&mut ctx, cu_device),
                "cuDevicePrimaryCtxRetain",
            )?;

            let mut pools_supported = 0;
            check_cuda_status(
                tho_cu_device_get_attribute(
                    &mut pools_supported,
                    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED,
                    cu_device,
                ),
                "cuDeviceGetAttribute",
            )?;

            let ctx = Self {
                device,
                cu_device,
                ctx,
                has_memory_pools: pools_supported > 0,
            };

            ctx.bind_to_thread()?;
            Ok(ctx)
        }
    }

    pub fn ordinal(&self) -> usize {
        self.device as usize
    }

    pub fn has_memory_pools(&self) -> bool {
        self.has_memory_pools
    }

    pub fn bind_to_thread(&self) -> Result<(), ThoError> {
        unsafe {
            debug!("ThoCudaContext::bind_to_thread: thread={:?}", std::thread::current().id());

            let mut current: CuContext = std::ptr::null_mut();
            check_cuda_status(tho_cu_ctx_get_current(&mut current), "cuCtxGetCurrent")?;
            if current != self.ctx {
                check_cuda_status(tho_cu_ctx_set_current(self.ctx), "cuCtxSetCurrent")?;
            }
        }
        Ok(())
    }

    #[allow(dead_code)]
    ///synchronizes device   (cuCtxSynchronize), use stream.synchronize
    pub fn synchronize(&self) -> Result<(), ThoError> {
        unsafe { check_cuda_status(tho_cu_ctx_synchronize(), "cuCtxSynchronize") }
    }

    pub fn reset(&mut self) -> Result<(), ThoError> {
        unsafe {
            check_cuda_status(tho_cu_reset(self.cu_device), "cuDevicePrimaryCtxReset")?;

            let mut cu_device: CuDevice = 0;
            check_cuda_status(
                tho_cu_device_primary_ctx_retain(&mut self.ctx, cu_device),
                "cuDevicePrimaryCtxRetain",
            )
        }
    }
}

unsafe impl Send for ThoCudaContext {}
unsafe impl Sync for ThoCudaContext {}

impl Drop for ThoCudaContext {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                let status = tho_cu_device_primary_ctx_release(self.cu_device);
                if status != CUDA_SUCCESS {
                    error!("{:?}", cuda_error_from_status("cuDevicePrimaryCtxRelease", status));
                }
                self.ctx = std::ptr::null_mut();
            }
        }
    }
}
