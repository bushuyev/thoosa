use tch::Kind::Float;
use tch::{Device, Tensor};
use tch::kind::FLOAT_CPU;

pub fn uniform_weight(size: i64, device: Device) -> Tensor {
    Tensor::ones(&[size], (Float, device)) / size
}
