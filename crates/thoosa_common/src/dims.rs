use crate::tho_tensor::ThoTensor;
use anyhow::Context as _;
use log::{debug, trace};
use std::cmp::min;
use std::fmt::Debug;
use std::ops::Range;
use tch::{IndexOp, Tensor};
use thoosa_common::tch_error::tch_result;
use thoosa_named_derive::Named;
use thoosa_shared::ThoError;
#[cfg(test)]
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::{DebugId, Metrics, MonoMetrics, RangeSelector};

pub trait Dims: Send + Sync + Clone + Debug {
    type Metrics: Metrics + Debug + Clone + Send + Sync + Default + TryFrom<Vec<i64>>;
}

#[derive(Clone, Debug)]
pub struct Batch<D: Dims>(D);
impl<D: Dims> Dims for Batch<D> {
    type Metrics = MonoMetrics<0, 3>; //O
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

    fn local_subview(id: &Id, t: &Tensor, selector: S) -> Result<Self::Output, ThoError>;
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
    type Metrics = MonoMetrics<1, 2>; //channels x seq
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
    use crate::ipc::cuda_worker::{AllTensors, WorkerStageInput};
    use crate::ipc::worker_msg::CudaWorkerId;
    use crate::tho_device::ThoDevice;
    use thoosa_shared::CudaDeviceGeneration;
    use thoosa_shared::traits::Select;

    #[test]
    fn test_nested_subviews_round_trip_ipc() -> anyhow::Result<()> {
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        // Exercise the real worker materializer on CPU so values, wire selectors, and shape are checked together.
        let tensors = AllTensors::new(CudaWorkerId { device_idx: 0, generation: CudaDeviceGeneration(0) });
        let make = || -> Result<ThoTensor<AnyDims, u32>, ThoError> { ThoTensor::new(1, tch_result(tch_result(Tensor::f_arange(48, (tch::Kind::Int64, tch::Device::Cpu)))?.f_reshape([12, 4]))?) };
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
        assert!(tch_result(actual.local()?.f_equal(&*local_view.local()?))?);
        let empty = remote_view.subview(selector(1, 1))?;
        assert_eq!(empty.size(), vec![0, 4]);
        assert_eq!(empty.ref_to_tensor(&tensors, false)?.size(), empty.size());
        assert!(remote_view.subview(selector(2, 3)).is_err());
        assert!(remote_view.subview(selector(1, 0)).is_err());

        // Channel/time selectors narrow the time axis and keep a channel prefix; both dimensions must match Local behavior.
        use crate::stages::freq_filter::ChannelsDataSelector;
        let make_channels = || -> Result<ThoTensor<ChannelsData, SynKey>, ThoError> { ThoTensor::new(thoosa_shared::mono_key!(1u32, "channels").into(), tch_result(tch_result(Tensor::f_arange(60, (tch::Kind::Int64, tch::Device::Cpu)))?.f_reshape([6, 10]))?) };
        let select_channels = |start: [i64; 2], end: [i64; 2]| ChannelsDataSelector(start.into()..end.into());
        let local_channels = make_channels()?.subview(select_channels([1, 2], [5, 9]))?.subview(select_channels([1, 3], [3, 5]))?;
        let remote_channels = make_channels()?.worker_tensor_to_ref(&tensors, &ThoDevice::from(0usize))?.subview(select_channels([1, 2], [5, 9]))?.subview(select_channels([1, 3], [3, 5]))?;
        let wire = postcard::to_allocvec(&remote_channels)?;
        let restored: ThoTensor<ChannelsData, SynKey> = postcard::from_bytes(&wire)?;
        let actual = restored.ref_to_tensor(&tensors, false)?;
        assert_eq!(restored.size(), vec![3, 2]);
        assert_eq!(actual.size(), restored.size());
        assert!(tch_result(actual.local()?.f_equal(&*local_channels.local()?))?);
        Ok(())
    }
}
