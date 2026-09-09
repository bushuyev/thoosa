use std::fmt::Debug;
use tch::{Kind, Tensor};
use thoosa_shared::ThoError;
use thoosa_shared::traits::Named;
use crate::dims::Dims;
use crate::tho_tensor::ThoTensor;

pub mod data;
pub mod metrics;
pub mod output_builders;

#[cfg(test)]
pub mod tests;

pub struct ThoTensorView<'a, D: Dims, Id: Named + Debug + Clone> {
    t: Tensor,
    _owner: &'a ThoTensor<D, Id>,
}
unsafe impl<'a, D: Dims, Id: Named + Debug + Clone> Sync for ThoTensorView<'a, D, Id> {}
unsafe impl<'a, D: Dims, Id: Named + Debug + Clone> Send for ThoTensorView<'a, D, Id> {}

impl<'a, D: Dims, Id: Named + Debug + Clone> ThoTensorView<'a, D, Id> {
    pub fn select(self, dim: i64, index: i64) -> Result<Self, ThoError> {
        Ok(Self {
            t: self.t.f_select(dim, index)?,
            ..self
        })
    }
    pub fn sum(self, k: Kind) -> Result<ThoTensor<D, Id>, ThoError> {
        ThoTensor::new(self._owner.id().clone(), self.t.f_sum(k)?)
    }
}



