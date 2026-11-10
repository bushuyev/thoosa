//! Resource identity and the actual (possibly filtered or selected) tensor length
//! form one batch invariant. Source Resource::length is deliberately not used.
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use thoosa_common::dims::ChannelsData;
use thoosa_common::runner::common::cachable::{DevicesWrap, ThoDataRef};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::ThoError;
use thoosa_shared::comp_id::SynKey;

pub trait ResourceBatchTensor {
    fn sample_len(&self) -> Result<i64, ThoError>;
}

impl ResourceBatchTensor for ThoDataRef<'_, ThoTensor<ChannelsData, SynKey>> {
    fn sample_len(&self) -> Result<i64, ThoError> {
        let shape = self.as_ref().size();
        if shape.len() != 2 || shape[0] <= 0 || shape[1] < 0 {
            return Err(ThoError::new(format!("resource tensor must have shape [channels, samples], got {shape:?}")));
        }
        Ok(shape[1])
    }
}

impl ResourceBatchTensor for Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>> {
    fn sample_len(&self) -> Result<i64, ThoError> {
        let shape = self.size();
        if shape.dim_0() <= 0 || shape.dim_1() < 0 {
            return Err(ThoError::new("invalid cached resource tensor shape"));
        }
        Ok(shape.dim_1())
    }
}

/// Private parallel arrays avoid per-resource allocation. All construction and
/// mutation pairs a tensor with its resource and derives its current sample length.
#[derive(Debug, Clone)]
pub struct ResourceBatch<T, R> {
    tensors: Vec<T>,
    resources: Vec<R>,
    lengths: Vec<i64>,
}

impl<T, R> Default for ResourceBatch<T, R> {
    fn default() -> Self {
        Self {
            tensors: Vec::new(),
            resources: Vec::new(),
            lengths: Vec::new(),
        }
    }
}

impl<T: ResourceBatchTensor, R> ResourceBatch<T, R> {
    pub fn try_new(tensors: Vec<T>, resources: Vec<R>) -> Result<Self, ThoError> {
        if tensors.len() != resources.len() {
            return Err(ThoError::new(format!("resource batch has {} tensors but {} resources", tensors.len(), resources.len())));
        }
        let lengths = tensors.iter().map(ResourceBatchTensor::sample_len).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { tensors, resources, lengths })
    }

    pub fn push(&mut self, resource: R, tensor: T) -> Result<(), ThoError> {
        let length = tensor.sample_len()?;
        self.tensors.push(tensor);
        self.resources.push(resource);
        self.lengths.push(length);
        Ok(())
    }
}

impl<T, R> ResourceBatch<T, R> {
    pub fn len(&self) -> usize {
        self.resources.len()
    }
    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
    pub fn tensors(&self) -> &[T] {
        &self.tensors
    }
    pub fn resources(&self) -> &[R] {
        &self.resources
    }
    pub fn lengths(&self) -> &[i64] {
        &self.lengths
    }
    pub fn iter(&self) -> impl Iterator<Item = (&R, &T, i64)> {
        self.resources.iter().zip(&self.tensors).zip(&self.lengths).map(|((resource, tensor), length)| (resource, tensor, *length))
    }
}

pub fn checked_batch_range(range: &std::ops::Range<i64>, len: usize) -> Result<std::ops::Range<usize>, ThoError> {
    if range.start < 0 || range.end < range.start || range.end as u64 > len as u64 {
        return Err(ThoError::new(format!("invalid batch range {range:?} for length {len}")));
    }
    Ok(range.start as usize..range.end as usize)
}

/// Window counts remain attached to their source even when a short tensor emits
/// no windows. Prepared batches keep those entries to validate reuse and IPC.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ResourceWindows<R> {
    pub resource: R,
    pub samples: i64,
    pub windows: i64,
}

pub type BorrowedResourceBatch<'a, R> = ResourceBatch<ThoDataRef<'a, ThoTensor<ChannelsData, SynKey>>, R>;

impl<R: Clone> BorrowedResourceBatch<'_, R> {
    pub fn borrow_range(&self, range: std::ops::Range<usize>) -> Result<BorrowedResourceBatch<'_, R>, ThoError> {
        // Validate even internal selections before indexing; IPC selectors are untrusted.
        if range.start > range.end || range.end > self.len() {
            return Err(ThoError::new("resource batch range is out of bounds"));
        }
        ResourceBatch::try_new(self.tensors[range.clone()].iter().map(|tensor| ThoDataRef::Borrowed(tensor.as_ref())).collect(), self.resources[range].to_vec())
    }

    pub fn map_tensors<'o>(&self, mut map: impl FnMut(&ThoTensor<ChannelsData, SynKey>) -> Result<ThoDataRef<'o, ThoTensor<ChannelsData, SynKey>>, ThoError>) -> Result<BorrowedResourceBatch<'o, R>, ThoError> {
        let mapped = ResourceBatch::try_new(self.tensors.iter().map(|tensor| map(tensor.as_ref())).collect::<Result<_, _>>()?, self.resources.clone())?;
        // IPC materialization must describe the same samples as the wire handle.
        if mapped.lengths != self.lengths {
            return Err(ThoError::new("resource lengths changed during tensor conversion"));
        }
        Ok(mapped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::{Tensor, kind::FLOAT_CPU};
    use thoosa_common::tch_error::tch_result;

    #[test]
    fn test_resource_batch_rejects_mismatched_counts_and_shapes() -> Result<(), ThoError> {
        let tensor = ThoTensor::new("batch".into(), tch_result(Tensor::f_zeros([2, 7], FLOAT_CPU))?)?;
        assert!(ResourceBatch::<_, u32>::try_new(vec![ThoDataRef::Borrowed(&tensor)], vec![]).is_err());
        let invalid = ThoTensor::new("rank".into(), tch_result(Tensor::f_zeros([7], FLOAT_CPU))?)?;
        let mut batch = ResourceBatch::try_new(vec![ThoDataRef::Borrowed(&tensor)], vec![17u32])?;
        assert!(batch.push(18, ThoDataRef::Borrowed(&invalid)).is_err());
        assert_eq!(batch.resources(), &[17]);
        assert_eq!(batch.lengths(), &[7]);
        assert_eq!(batch.tensors().len(), 1);
        assert!(batch.borrow_range(0..2).is_err());
        assert!(checked_batch_range(&(-1..1), 1).is_err());
        assert!(checked_batch_range(&(1..0), 1).is_err());
        assert!(checked_batch_range(&(0..2), 1).is_err());
        assert!(batch.borrow_range(1..1)?.is_empty());
        Ok(())
    }
}
