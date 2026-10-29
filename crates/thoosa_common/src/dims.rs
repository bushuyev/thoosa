use thoosa_common::tch_error::tch_result;
use crate::tho_tensor::{KeyHolder, ThoTensor, ThoTensorInner};
use anyhow::Context as _;
use async_trait::async_trait;
use log::{debug, trace};
use std::cmp::min;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Deref, Range};
use std::sync::Arc;
use tch::{IndexOp, Tensor};
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::{Metrics, MonoMetrics, Named, RangeSelector, Select, DebugId, ThoSelector};
use thoosa_shared::{CopyToDevice, ThoError};

pub trait Dims: Send + Sync + Clone + Debug {
    type Metrics: Metrics + Debug + Clone + Send + Sync + Default + TryFrom<Vec<i64>>;
}

impl <D, Id, S> Select<S> for ThoTensor<D, Id>
where
    S:ThoSelector + 'static,
    D:Dims,
    Id:Named + Debug + Clone,
    Self:LocalSelect<Id, S, Output = ThoTensor<D, Id>>,
{
    type Output = ThoTensor<D, Id>;

    fn subview(&self, selector: S) -> Result<Self::Output, ThoError> {
        match &self.inner {
            ThoTensorInner::Val { t } => {
                Self::local_subview(self.id(), &t.lock(), selector)
            }
            ThoTensorInner::Ref { key, size, kind, selector: parent } => {
                // Children select relative to the current view, while the worker indexes the original tensor.
                let (selector, selected_size) = TensorRangesSelector::compose(parent.as_deref(), size, &selector)?;
                debug!("ThoTensor::subview: key={key:?} size={size:?} kind={kind:?}");
                Ok(Self {
                    id:self.id().clone(),
                    inner: ThoTensorInner::Ref {
                        key: key.clone(),
                        size: selected_size,
                        kind: *kind,
                        selector:Some(Box::new(selector) as Box<dyn ThoSelector>),
                    },
                    pd: PhantomData,
                }
                )
            }
        }
    }
}

/// Absolute ranges keep nested IPC views serializable without retaining a chain of selectors.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
struct TensorRangesSelector(Vec<Range<i64>>);

#[typetag::serde(name = "TensorRangesSelector")]
impl ThoSelector for TensorRangesSelector {
    fn clone_box(&self) -> Box<dyn ThoSelector> { Box::new(self.clone()) }

    fn tensor_ranges(&self) -> Result<Vec<Range<i64>>, ThoError> { Ok(self.0.clone()) }
}

impl TensorRangesSelector {
    fn compose(parent: Option<&dyn ThoSelector>, size: &[i64], child: &dyn ThoSelector) -> Result<(Self, Vec<i64>), ThoError> {
        let parent = parent.map(ThoSelector::tensor_ranges).transpose()?.unwrap_or_default();
        let child = child.tensor_ranges()?;
        if parent.len() > size.len() || child.len() > size.len() {
            return Err(ThoError::new(format!("selector rank exceeds tensor rank {}", size.len())));
        }
        let mut ranges = Vec::with_capacity(size.len());
        let mut selected_size = Vec::with_capacity(size.len());
        for (dim, &length) in size.iter().enumerate() {
            let range = child.get(dim).cloned().unwrap_or(0..length);
            // Match worker narrowing: clamp the end, reject negative, reversed, or out-of-bounds starts.
            let end = range.end.min(length);
            if range.start < 0 || range.end < range.start || range.start > end {
                return Err(ThoError::new(format!("invalid selector range {range:?} for dimension {dim} with size {length}")));
            }
            let offset = parent.get(dim).map_or(0, |range| range.start);
            let start = offset.checked_add(range.start).context("composed selector start overflow")?;
            let absolute_end = offset.checked_add(end).context("composed selector end overflow")?;
            ranges.push(start..absolute_end);
            selected_size.push(end - range.start);
        }
        Ok((Self(ranges), selected_size))
    }
}

#[derive(Clone, Debug)]
pub struct Batch<D: Dims>(D);
impl<D: Dims> Dims for Batch<D> {
    type Metrics = MonoMetrics<0, 3>;//O
}


pub type ThoBatch<D> = ThoTensor<Batch<D>, DebugId>;

#[derive(Clone, Debug, Named)]
#[name = "n/a"]
pub struct AnyDims {}
impl Dims for AnyDims {
    type Metrics = AnyMetrics;
}

#[derive(Clone, Debug, Default)]
pub struct AnyMetrics;

impl Metrics for AnyMetrics {}

impl From<Vec<i64>> for AnyMetrics {
    fn from(_value: Vec<i64>) -> Self {
        AnyMetrics
    }
}

pub trait LocalSelect<Id, S> {
    type Output;

    fn local_subview(id:&Id, t:&Tensor, selector: S) -> Result<Self::Output, ThoError>;
}

impl LocalSelect<u32, RangeSelector> for ThoTensor<AnyDims, u32> {
    type Output = ThoTensor<AnyDims, u32>;

    fn local_subview(id: &u32, t: &Tensor, selector: RangeSelector) -> Result<Self::Output, ThoError> {
        let start = i64::try_from(selector.start[0]).context("range selector start does not fit i64")?;
        let end = i64::try_from(selector.end[0]).context("range selector end does not fit i64")?;
        ThoTensor::new(*id, tch_result(t.f_i(start..end))?)
    }
}

#[derive(Clone, Debug)]
pub struct ChannelsData;

impl Dims for ChannelsData {
    type Metrics = MonoMetrics<1, 2>;//channels x seq

}
impl ChannelsData {

    pub fn sub_single(t: &Tensor, range: &Range<MonoMetrics<1, 2>>, size: &MonoMetrics<1, 2>) -> anyhow::Result<Tensor> {
        let channel_range = range.start.dim_0()..min(size.dim_0(), range.end.dim_0());
        let data_range = range.start.dim_1()..min(size.dim_1(), range.end.dim_1());

        debug!("ChannelsData2::subview: channel_range={channel_range:?} data_range={data_range:?} size={size:?}");
        trace!("ChannelsData2::subview: t=\n{}", t);

        let sub_t = t.f_i((channel_range.clone(), data_range.clone()))?;

        // assert_eq!(sub_t.size(), [])
        Ok(sub_t)
    }
}

#[derive(Clone, Debug)]
pub struct Slice; //sliced data [lice_width]
impl Dims for Slice {
    type Metrics = MonoMetrics<0, 1>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ipc::cuda_wokrer::{AllTensors, WorkerStageInput};
    use crate::tho_device::ThoDevice;
    use thoosa_shared::CudaDeviceGeneration;
    use crate::ipc::worker_msg::CudaWorkerId;

    #[test]
    fn test_nested_subviews_round_trip_ipc() -> anyhow::Result<()> {
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        // Exercise the real worker materializer on CPU so values, wire selectors, and shape are checked together.
        let tensors = AllTensors::new(CudaWorkerId { device_idx: 0, generation: CudaDeviceGeneration(0) });
        let make = || -> Result<ThoTensor<AnyDims, u32>, ThoError> {
            ThoTensor::new(1, tch_result(tch_result(Tensor::f_arange(48, (tch::Kind::Int64, tch::Device::Cpu)))?.f_reshape([12, 4]))?)
        };
        let local = make()?;
        let remote = make()?.worker_tensor_to_ref(&tensors, &ThoDevice::from(0usize))?;
        let selector = |start, end| RangeSelector([start]..[end]);
        let local_view = local.subview(selector(2, 10))?.subview(selector(3, 6))?.subview(selector(1, 2))?;
        let remote_view = remote.subview(selector(2, 10))?.subview(selector(3, 6))?.subview(selector(1, 2))?;
        assert_eq!(remote_view.size(), vec![1, 4]);
        assert_eq!(remote.size(), vec![12, 4]);
        let wire = postcard::to_allocvec(&remote_view)?;
        let restored: ThoTensor<AnyDims, u32> = postcard::from_bytes(&wire)?;
        let actual = restored.ref_to_tensor(&tensors, false)?;
        assert_eq!(actual.size(), remote_view.size());
        assert!(tch_result(actual.t_ref().f_equal(&local_view.t_ref()))?);
        let empty = remote_view.subview(selector(1, 1))?;
        assert_eq!(empty.size(), vec![0, 4]);
        assert_eq!(empty.ref_to_tensor(&tensors, false)?.size(), empty.size());
        assert!(remote_view.subview(selector(2, 3)).is_err());
        assert!(remote_view.subview(selector(1, 0)).is_err());

        // Channel/time selectors narrow the time axis and keep a channel prefix; both dimensions must match Local behavior.
        use crate::stages::freq_filter::ChannelsDataSelector;
        let make_channels = || -> Result<ThoTensor<ChannelsData, SynKey>, ThoError> {
            ThoTensor::new(thoosa_shared::mono_key!(1u32, "channels").into(), tch_result(tch_result(Tensor::f_arange(60, (tch::Kind::Int64, tch::Device::Cpu)))?.f_reshape([6, 10]))?)
        };
        let select_channels = |start: [i64; 2], end: [i64; 2]| ChannelsDataSelector(start.into()..end.into());
        let local_channels = make_channels()?.subview(select_channels([1, 2], [5, 9]))?.subview(select_channels([1, 3], [3, 5]))?;
        let remote_channels = make_channels()?.worker_tensor_to_ref(&tensors, &ThoDevice::from(0usize))?.subview(select_channels([1, 2], [5, 9]))?.subview(select_channels([1, 3], [3, 5]))?;
        let wire = postcard::to_allocvec(&remote_channels)?;
        let restored: ThoTensor<ChannelsData, SynKey> = postcard::from_bytes(&wire)?;
        let actual = restored.ref_to_tensor(&tensors, false)?;
        assert_eq!(restored.size(), vec![3, 2]);
        assert_eq!(actual.size(), restored.size());
        assert!(tch_result(actual.t_ref().f_equal(&local_channels.t_ref()))?);
        Ok(())
    }

    #[test]
    fn test_selector_composition_preserves_other_dimensions_and_clamps() -> anyhow::Result<()> {
        let parent = TensorRangesSelector(vec![2..8, 4..10]);
        let child = TensorRangesSelector(vec![1..20]);
        let (selected, size) = TensorRangesSelector::compose(Some(&parent), &[6, 6], &child)?;
        assert_eq!(selected.tensor_ranges()?, vec![3..8, 4..10]);
        assert_eq!(size, vec![5, 6]);
        let (unchanged, same_size) = TensorRangesSelector::compose(Some(&selected), &size, &TensorRangesSelector(vec![]))?;
        assert_eq!(unchanged.tensor_ranges()?, selected.tensor_ranges()?);
        assert_eq!(same_size, size);
        assert!(TensorRangesSelector::compose(None, &[6], &TensorRangesSelector(vec![0..1, 0..1])).is_err());
        Ok(())
    }
}
