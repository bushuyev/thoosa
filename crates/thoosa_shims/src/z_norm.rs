use anyhow::{bail, Result};
use std::ffi::{c_char, c_void, CStr};

use tch::Tensor;
use torch_sys::C_tensor;


unsafe extern "C" {
    fn thoosa_z_norm(
        x: *const c_void,
        dims: *const i64,
        dims_len: usize,
        eps: f64,
    ) -> *mut c_void;

    fn thoosa_last_error() -> *const c_char;
}


pub fn z_norm_dims_cuda(
    x: &Tensor,
    dims: &[i64],
    eps: f64,
) -> Result<Tensor> {
    let result = unsafe {
        thoosa_z_norm(
            x.as_ptr().cast::<c_void>(),
            dims.as_ptr(),
            dims.len(),
            eps,
        )
    };

    if result.is_null() {
        let message = unsafe {
            let ptr = thoosa_last_error();

            if ptr.is_null() {
                "unknown native error".to_owned()
            } else {
                CStr::from_ptr(ptr)
                    .to_string_lossy()
                    .into_owned()
            }
        };

        bail!("z_norm CUDA kernel failed: {message}");
    }

    let result =
        unsafe {
            Tensor::from_ptr(
                result.cast::<C_tensor>())
        };

    Ok(result)
}

