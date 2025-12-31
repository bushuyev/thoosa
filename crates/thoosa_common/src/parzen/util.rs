use tch::Kind::Float;
use tch::{Device, Tensor};

pub fn uniform_weight(size: i64, device: Device) -> Tensor {
    Tensor::ones(&[size], (Float, device)) / size
}
