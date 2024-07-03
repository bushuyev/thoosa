use tch::Tensor;
use crate::parzen::optimizer::{Distr, ParzenEstimator};

pub struct CategoricalParzenEstimator {

}

impl CategoricalParzenEstimator {
    pub fn new() -> Self {
        Self {}
    }
}

impl ParzenEstimator for CategoricalParzenEstimator {
    fn clc_dist(&self, samples: Tensor) -> Box<dyn Distr> {
        todo!()
    }
}