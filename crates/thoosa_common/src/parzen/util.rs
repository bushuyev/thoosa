use tch::Tensor;
use tch::kind::FLOAT_CPU;

pub fn uniform_weight(size: i64) -> Tensor {
    Tensor::ones(&[size], FLOAT_CPU) / size //TODO FLOAT_CPU
}
