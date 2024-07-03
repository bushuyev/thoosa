use tch::kind::FLOAT_CPU;
use tch::Tensor;

pub fn uniform_weight(size: i64) -> Tensor {
    Tensor::ones(&[size], FLOAT_CPU) / size //TODO FLOAT_CPU
}
