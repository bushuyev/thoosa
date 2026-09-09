use std::ffi::CStr;
use std::os::raw::{c_char, c_double, c_int};
use std::ptr::{null, NonNull};
use log::debug;
use tch::{nn, Device, Kind, TchError, Tensor};
use torch_sys::C_tensor;

pub type TchResult<T> = Result<T, TchError>;

#[repr(C)]
struct ThoQuantizedLinearOpaque {
    _private: [u8; 0],
}

unsafe extern "C" {
    fn tho_ql_last_error() -> *const c_char;
    fn tho_ql_clear_error();

    fn tho_ql_prepack_dynamic_i8(
        weight_f32: *const C_tensor,
        bias_f32_or_null: *const C_tensor,
        weight_scale_or_negative: c_double,
    ) -> *mut ThoQuantizedLinearOpaque;

    fn tho_ql_forward_dynamic_i8(
        ql: *const ThoQuantizedLinearOpaque,
        input_f32: *const C_tensor,
        reduce_range: c_int,
    ) -> *mut C_tensor;

    fn tho_ql_forward_relu_dynamic_i8(
        ql: *const ThoQuantizedLinearOpaque,
        input_f32: *const C_tensor,
        reduce_range: c_int,
    ) -> *mut C_tensor;

    fn tho_ql_in_features(ql: *const ThoQuantizedLinearOpaque) -> i64;
    fn tho_ql_out_features(ql: *const ThoQuantizedLinearOpaque) -> i64;
    fn tho_ql_weight_scale(ql: *const ThoQuantizedLinearOpaque) -> c_double;

    fn tho_ql_free(ql: *mut ThoQuantizedLinearOpaque);
}

#[derive(Debug)]
pub struct FastQuantizedLinear {
    raw: NonNull<ThoQuantizedLinearOpaque>,
    in_features: i64,
    out_features: i64,
    weight_scale: f64,
}

unsafe impl Send for FastQuantizedLinear {}
unsafe impl Sync for FastQuantizedLinear {}

impl Drop for FastQuantizedLinear {
    fn drop(&mut self) {
        unsafe {
            tho_ql_free(self.raw.as_ptr());
        }
    }
}

impl FastQuantizedLinear {
    pub fn from_float_linear(linear: &nn::Linear) -> TchResult<Self> {
        Self::from_float_weights(&linear.ws, linear.bs.as_ref())
    }

    pub fn from_float_weights(ws: &Tensor, bs: Option<&Tensor>) -> TchResult<Self> {
        Self::from_float_weights_with_scale(ws, bs, None)
    }

    pub fn from_float_weights_with_scale(
        ws: &Tensor,
        bs: Option<&Tensor>,
        weight_scale: Option<f64>,
    ) -> TchResult<Self> {
        debug!("FastQuantizedLinear::from_float_weights_with_scale: ws={ws:?} bs={bs:?} weight_scale={weight_scale:?}");

        ensure_cpu_float(ws, "FastQuantizedLinear weight")?;

        let (out_features, in_features) = ws.size2()?;

        if let Some(bs) = bs {
            ensure_cpu_float(bs, "FastQuantizedLinear bias")?;

            let bias_size = bs.size();
            if bias_size.as_slice() != [out_features] {
                return Err(TchError::Shape(format!(
                    "bias shape mismatch: expected [{out_features}], got {bias_size:?}"
                )));
            }
        }

        let bias_ptr = bs.map_or(null(), Tensor::as_ptr);
        let scale_arg = weight_scale.unwrap_or(-1.0);

        let raw = unsafe {
            tho_ql_clear_error();

            tho_ql_prepack_dynamic_i8(
                ws.as_ptr(),
                bias_ptr,
                scale_arg,
            )
        };

        let raw = NonNull::new(raw).ok_or_else(|| ffi_error("tho_ql_prepack_dynamic_i8"))?;

        let actual_in = unsafe { tho_ql_in_features(raw.as_ptr()) };
        let actual_out = unsafe { tho_ql_out_features(raw.as_ptr()) };
        let actual_scale = unsafe { tho_ql_weight_scale(raw.as_ptr()) };

        debug_assert_eq!(actual_in, in_features);
        debug_assert_eq!(actual_out, out_features);

        Ok(Self {
            raw,
            in_features,
            out_features,
            weight_scale: actual_scale,
        })
    }

    pub fn try_forward(&self, xs: &Tensor) -> TchResult<Tensor> {
        self.try_forward_impl(xs, false, false)
    }

    pub fn try_forward_reduce_range(&self, xs: &Tensor) -> TchResult<Tensor> {
        self.try_forward_impl(xs, false, true)
    }

    pub fn try_forward_relu(&self, xs: &Tensor) -> TchResult<Tensor> {
        self.try_forward_impl(xs, true, false)
    }

    pub fn in_features(&self) -> i64 {
        self.in_features
    }

    pub fn out_features(&self) -> i64 {
        self.out_features
    }

    pub fn weight_scale(&self) -> f64 {
        self.weight_scale
    }

    fn try_forward_impl(
        &self,
        xs: &Tensor,
        relu: bool,
        reduce_range: bool,
    ) -> TchResult<Tensor> {
        debug!("try_forward_impl: xs={xs:?}");

        ensure_cpu_float(xs, "FastQuantizedLinear input")?;
        self.check_input_shape(xs)?;

        let out_ptr = unsafe {
            tho_ql_clear_error();

            if relu {
                tho_ql_forward_relu_dynamic_i8(
                    self.raw.as_ptr(),
                    xs.as_ptr(),
                    reduce_range as c_int,
                )
            } else {
                tho_ql_forward_dynamic_i8(
                    self.raw.as_ptr(),
                    xs.as_ptr(),
                    reduce_range as c_int,
                )
            }
        };

        if out_ptr.is_null() {
            return Err(ffi_error(if relu {
                "tho_ql_forward_relu_dynamic_i8"
            } else {
                "tho_ql_forward_dynamic_i8"
            }));
        }

        Ok(unsafe { Tensor::from_ptr(out_ptr) })
    }

    fn check_input_shape(&self, xs: &Tensor) -> TchResult<()> {
        let size = xs.size();

        let Some(&last_dim) = size.last() else {
            return Err(TchError::Shape(
                "FastQuantizedLinear input must have at least one dimension".into(),
            ));
        };

        if last_dim != self.in_features {
            return Err(TchError::Shape(format!(
                "FastQuantizedLinear input last dim mismatch: expected {}, got {}; shape = {:?}",
                self.in_features, last_dim, size
            )));
        }

        Ok(())
    }
}

impl nn::Module for FastQuantizedLinear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.try_forward(xs)
            .expect("FastQuantizedLinear::forward failed")
    }
}

fn ensure_cpu_float(t: &Tensor, name: &str) -> TchResult<()> {
    if t.device() != Device::Cpu {
        return Err(TchError::Torch(format!(
            "{name} must be on CPU, got {:?}",
            t.device()
        )));
    }

    if t.kind() != Kind::Float {
        return Err(TchError::Torch(format!(
            "{name} must be Float, got {:?}",
            t.kind()
        )));
    }

    Ok(())
}

fn ffi_error(op: &str) -> TchError {
    let msg = unsafe {
        let ptr = tho_ql_last_error();

        if ptr.is_null() {
            "unknown C++ error".to_string()
        } else {
            CStr::from_ptr(ptr).to_string_lossy().into_owned()
        }
    };

    TchError::Torch(format!("{op}: {msg}"))
}
