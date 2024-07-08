use tch::Tensor;
use crate::parzen::optimizer::{Distribution, ParzenEstimator, Primitives};

pub struct CategoricalParzenEstimator {

}

impl CategoricalParzenEstimator {
    pub fn new() -> Self {
        Self {}
    }
}

impl ParzenEstimator for CategoricalParzenEstimator {
    fn clc_dist(&self, samples: Tensor) -> Box<dyn Distribution> {
        Box::new(CategoricalGaussian::new())
    }
}

#[derive(Debug)]
struct CategoricalGaussian {

}

impl CategoricalGaussian {

    pub fn new() -> Self {
        Self {}
    }
}

impl Distribution for CategoricalGaussian {

    fn sample(&self, n_samples: usize) -> Vec<Primitives> {
        todo!()
    }

    fn basis_loglikelihood(&self, x: &Tensor) -> Tensor {
        todo!()
    }

    fn weights(&self) -> &Tensor {
        todo!()
    }
}