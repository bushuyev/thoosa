use crate::dims::Dims;
use thoosa_shared::traits::MonoMetrics;

#[derive(Clone, Debug)]
pub struct AStageOutputDims;

impl Dims for AStageOutputDims {
    type Metrics = MonoMetrics<0, 1>;
}

#[derive(Clone, Debug)]
pub struct BStageOutputDims;

impl Dims for BStageOutputDims {
    type Metrics = MonoMetrics<0, 1>;
}

#[derive(Clone, Debug)]
pub struct ABDims;

impl Dims for ABDims {
    type Metrics = MonoMetrics<1, 2>;
}
