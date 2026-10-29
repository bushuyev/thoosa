use tch::Kind::Float;
use tch::{Device, Tensor};
use crate::tch_error::tch_result;

pub fn uniform_weight(size: i64, device: Device) -> anyhow::Result<Tensor> {
    // Reject empty distributions before normalization divides by zero.
    anyhow::ensure!(size > 0, "distribution must contain at least one component");
    Ok(tch_result(tch_result(Tensor::f_ones([size], (Float, device)))?.f_div_scalar(size))?)
}
