use thoosa_common::tch_error::tch_result;
use anyhow::Context as _;
use async_trait::async_trait;
use cached_derive::Cached;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::debug;
#[cfg(test)]
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
#[cfg(test)]
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use tch::{Device, IndexOp, Kind, Tensor};
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::adjusters::splits::{OutputBuilder, SplitsAdjusterContext, SplitsAdjusterContextImpl};
use thoosa_common::runner::common::cachable::{DevicesWrap};
#[cfg(test)]
use thoosa_common::runner::common::cachable::Cached;
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, NoStageAdjuster, NoopRunnableContext, OOMHandler, input_adjuster, run_ctx_adjuster};
use thoosa_named_derive::Named;
use thoosa_shared::traits::{AsyncStage, OutId, Stage, ThoSelector};
use thoosa_common::runner::sync_stage::{ErasedStageWithState, StageStateInfo, StageWithState, SyncStage};
use thoosa_shared::traits::{MaxParallelCtx, RunCtx};

use thoosa_common::dims::{AnyMetrics, ChannelsData, Dims};
use thoosa_common::transforms::filters::{AverageFilterParams, BandPassFilterParams, InputFilter, MedianFilterParams, NotchBandFilterParams, SavGolFilterParams, ZNormFilterParams};

use crate::dims::LocalSelect;
use crate::cuda_utils::tensor_mem_size;
#[cfg(test)]
use crate::ipc::serialization::json::JsonIpcCodec;
#[cfg(test)]
use crate::ipc::wraps::SyncStageIpcWrapper;
use thoosa_common::ipc::cuda_wokrer::{WorkerStageInput};
#[cfg(test)]
use thoosa_common::ipc::cuda_wokrer::AllTensors;
#[cfg(test)]
use thoosa_common::ipc::cuda_wokrer::WokerRunCtx;
use thoosa_common::ipc::proxy::{DynSyncStageWithStateEnum, ProxyStage};
use thoosa_common::ipc::worker_msg::WorkerInput;
#[cfg(test)]
use thoosa_common::ipc::wraps::WorkerWrapperWithState;
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::transforms::filters::{Padding, band_pass_filter, notch_band_rfft};
use thoosa_common::transforms::smoothing::{median, moving_average};
use thoosa_common::utils::z_norm;
use thoosa_common::ws::{WsConfigs};
#[cfg(test)]
use thoosa_common::ws::Workspace;
use thoosa_digest_id::DigestHash;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::math::sg_smoothing_coeffs;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Metrics, MonoMetrics, WithId};
use thoosa_shared::{CopyToDevice, ThoError, mono_key};
use thoosa_shared::ipc_types::{TensorKey};
use thoosa_common::ipc::tensor_info::{WorkerTensorInfoPayload};
use tokio::sync::MutexGuard;
use crate::instance_cache::InstanceCache;

pub const SYNC_FREQ_FILTER_ENDPOINT: &str = "SyncFreqFilter";

#[derive(Debug, Clone)]
pub struct BandPassDims;

impl Dims for BandPassDims {
    type Metrics = AnyMetrics;//TODO
}

#[derive(Debug, Clone, Serialize, Deserialize, DigestHash)]
#[digest(tag = "thoosa.SyncFreqFilterIpcInit:v1")]
pub struct SyncFreqFilterIpcInit {
    pub filter_configs: Vec<InputFilter>,
    pub num_channels: i64,
    pub sample_rate: f64,
    pub find_motifs_run_config_id: RunConfigId,
    pub tho_device_idx: usize
}


impl SyncFreqFilterIpcInit {

    pub fn new(filter_configs: Vec<InputFilter>, num_channels: i64, sample_rate: f64, find_motifs_run_config_id: RunConfigId, tho_device_idx: usize) -> Self {
        Self {  filter_configs, num_channels, sample_rate,find_motifs_run_config_id, tho_device_idx }
    }


    fn new_id(&self)->SynKey {//TODO add trait for init types
        SynKey::new([
            mono_key!(FREQ_FILTER, "FreqFilter"),
            mono_key!(self.blake3_hash(), format!("Hash for {:?}", self)),
        ])
    }

}

impl WorkerStageInput for SyncFreqFilterIpcInit {
}

#[typetag::serde]
impl WorkerInput for SyncFreqFilterIpcInit {}

#[derive(Debug)]
pub struct SyncFreqFilter<C>
where
    C: RunCtx + 'static,
{
    id: SynKey,
    // IPC installs this endpoint under the run-config id, not its output SynKey.
    stage_id: String,
    filters: tokio::sync::Mutex<Vec<PreparedFilter>>,
    num_channels:i64,
    sample_rate:f64,
    pd:PhantomData<C>
}


static SYNC_FREQ_FILTER_INSTANCES: InstanceCache = InstanceCache::new();

impl<C> SyncFreqFilter<C>
where
    C: RunCtx + 'static,
{

    pub async fn new(init:SyncFreqFilterIpcInit) -> Result<Arc<DynSyncStageWithStateEnum<SyncFreqFilter<C>>>, ThoError> {
        let _find_motifs_run_config_id = init.find_motifs_run_config_id;
        // Filters outlive training trials. Retain their proxy and configuration;
        // its endpoint is installed lazily in each replacement worker.
        let gate = GlobalCtx::global().cuda_activity_gate(init.tho_device_idx);
        let activity = gate.as_ref().map(thoosa_shared::CudaActivityScope::for_operation).transpose()?;
        let _generation = gate.as_ref().and_then(|gate| activity.as_ref().and_then(|scope| scope.generation(gate)));
        // let id = (TypeId::of::<Self>(), find_motifs_run_config_id, init.tho_device_idx);
        let id = init.new_id().to_string();

        SYNC_FREQ_FILTER_INSTANCES.get_or_try_insert_recreatable_with::<C, _, _, _, _>(init.tho_device_idx, GlobalCtx::global().cuda_activity_gate(init.tho_device_idx), GlobalCtx::global().is_ipc(), id.clone(), move || async move {
            let stage = if GlobalCtx::global().is_ipc() {

                // Install and invoke the endpoint with the run-config id so every filter configuration keeps its own worker-side state.

                Arc::new(DynSyncStageWithStateEnum::Proxy {
                    stage: Arc::new(ProxyStage::proxy_with_input_cloner_recreatable(
                        SYNC_FREQ_FILTER_ENDPOINT,
                        Some(id),
                        init.tho_device_idx,
                        Box::new(init.clone()),
                        Arc::new(|input: &ThoTensor<ChannelsData, SynKey>| Ok(Box::new(input.clone_ref()?) as Box<dyn WorkerInput>)),
                    ).await?),
                    init: (),
                })
            } else {
                Arc::new(DynSyncStageWithStateEnum::Local(Arc::new(Self::new_local(init)?)))
            };
            let _registration = GlobalCtx::global().register_cache((stage.stage_id(), Arc::clone(&stage) as Arc<dyn FuturesRunnerMapPurger>));

            Ok(stage)

        }).await
    }

    pub fn new_local(init:SyncFreqFilterIpcInit) ->Result<Self, ThoError>{

        let filters = Self::prepare_filters(&init.filter_configs, init.num_channels, init.sample_rate, init.tho_device_idx)?;

        Ok(Self {
            id: init.new_id(),
            stage_id: init.find_motifs_run_config_id.to_string(),
            filters: tokio::sync::Mutex::new(filters),
            num_channels:init.num_channels,
            sample_rate:init.sample_rate,
            pd: Default::default(),
        })
    }

    fn prepare_filters(filter_configs: &[InputFilter], num_channels: i64, sample_rate: f64, tho_device_idx: usize) -> Result<Vec<PreparedFilter>, ThoError> {
        // A worker uses device 1; local construction can also target CPU or another GPU.
        let device: Device = ThoDevice::from(tho_device_idx).into();
        let mut filters = Vec::with_capacity(filter_configs.len());
        for filter_config in filter_configs {
            match filter_config {
                InputFilter::NotchBandFilter(notch_config) => {
                    filters.push(PreparedFilter::NotchBand {
                        low_hz: notch_config.low_hz,
                        high_hz: notch_config.high_hz,
                    });
                }
                InputFilter::BandPassFilter(band_pass_config) => {
                    let high_hz = band_pass_config.low_hz + band_pass_config.band_hz;

                    let kernel = tch_result(tch_result(band_pass_filter(
                        band_pass_config.low_hz,
                        high_hz,
                        band_pass_config.kernel_size as i64,
                        sample_rate,
                        1e-6
                    )?.f_repeat([num_channels, 1, 1]))?.f_to_device(device))?;

                    debug!("FreqFilter::new: band_pass_config={band_pass_config:?} kernel.size={:?}", kernel.size());

                    filters.push(PreparedFilter::BandPass {
                        kernel: PreparedFilterTensor::new(kernel)?,
                        stride: band_pass_config.stride,
                        padding: band_pass_config.padding,
                    });
                }
                InputFilter::SavGolFilter(sav_gol_config) => {
                    let window_len = if sav_gol_config.window_len % 2 == 0 {
                        sav_gol_config.window_len + 1 //TODO add constraint in TPE
                    } else {
                        sav_gol_config.window_len
                    };

                    if window_len < 3  {
                        return Err(ThoError::new(format!("window_len must be odd and >= 3, got {}", window_len)));
                    }

                    if sav_gol_config.poly_order < 0 || sav_gol_config.poly_order >= window_len as i64 {
                        return Err(ThoError::new(format!("poly_order must be >= 0 and < window_len, got poly_order={} window_len={}", sav_gol_config.poly_order, window_len)));
                    }

                    let coeffs = sg_smoothing_coeffs(window_len as usize, sav_gol_config.poly_order as usize)?;
                    let weights = tch_result(tch_result(tch_result(tch_result(tch_result(Tensor::f_from_slice(&coeffs))?
                        .f_to_kind(tch::Kind::Float))?
                        .f_to_device(device))?
                        .f_view([1, 1, window_len as i64]))?
                        .f_repeat([num_channels, 1, 1]))?;

                    filters.push(PreparedFilter::SavGol {
                        weights: PreparedFilterTensor::new(weights)?,
                        window_len: window_len as i64,
                        stride: sav_gol_config.stride,
                        padding: sav_gol_config.padding,
                    });
                }
                InputFilter::MedianFilter(median_config) => {
                    let effective_window = if median_config.window % 2 == 0 {
                        median_config.window + 1
                    } else {
                        median_config.window
                    };
                    if effective_window <= 3 {
                        return Err(ThoError::new(format!("median window must be > 3 after odd normalization, got {}", median_config.window)));
                    }
                    filters.push(PreparedFilter::Median {
                        window: median_config.window,
                    });
                }
                InputFilter::AverageFilter(moving_average_config) => {
                    let effective_window = if moving_average_config.window % 2 == 0 {
                        moving_average_config.window + 1
                    } else {
                        moving_average_config.window
                    };
                    if effective_window <= 3 {
                        return Err(ThoError::new(format!("moving average window must be > 3 after odd normalization, got {}", moving_average_config.window)));
                    }
                    filters.push(PreparedFilter::MovingAverage {
                        window: moving_average_config.window,
                        step: moving_average_config.step,
                    });
                }
                InputFilter::ZNorm(_) => {
                    filters.push(PreparedFilter::ZNorm {})
                }
            }
        }

        Ok(filters)
    }

    pub fn apply_tho_tensor(&self, input_tensor: &ThoTensor<ChannelsData, SynKey>) -> Result<ThoTensor<ChannelsData, SynKey>, ThoError> {
        let filters = self.filters.try_lock().context("SyncFreqFilter::apply_tho_tensor could not lock prepared filters")?;
        self.apply_tho_tensor_with_state(filters.as_slice(), input_tensor)
    }

    fn apply_tho_tensor_with_state(&self, filters: &[PreparedFilter], input_tensor: &ThoTensor<ChannelsData, SynKey>) -> Result<ThoTensor<ChannelsData, SynKey>, ThoError> {
        debug!("FreqFilter::apply_tho_tensor: filters={filters:?}");

        let mut filtered_t = input_tensor.as_ref().shallow_clone();//.f_totype(Float)?;

        for filter in filters.iter() {
            match filter {
                PreparedFilter::NotchBand { low_hz, high_hz } => {
                    let filtered_t_out = notch_band_rfft(&filtered_t, self.sample_rate, *low_hz, *high_hz)?;

                    debug!("FreqFilter::apply_tho_tensor, NotchBand: id={} filtered_t.size={:?} filtered_t_out.size={:?}", input_tensor.id(), filtered_t.size(), filtered_t_out.size());

                    filtered_t = filtered_t_out;
                    // debug_plot!(&filtered_t, "{}_NotchBand", input_tensor.id())?;
                }
                PreparedFilter::BandPass { kernel, stride, padding } => {
                    let kernel = kernel.ready()?;
                    let input_t = tch_result(filtered_t.as_ref().f_unsqueeze(0))?;

                    let output = self.run_conv_with_padding(*stride, padding, &kernel.as_ref(), &input_t);
                    debug!("FreqFilter::apply_tho_tensor, BandPass: id={} kernel.size={:?} input.size={:?} output.size={:?}", input_tensor.id(), kernel.size(), input_t.size(), output.as_ref().map(|v| v.size()));
                    filtered_t = output?;

                    // debug_plot!(&filtered_t, "{}_BandPass", input_tensor.id())?;
                }
                PreparedFilter::SavGol { weights, window_len, stride, padding } => {
                    let weights = weights.ready()?;
                    let t = filtered_t.size()[1];
                    if t < *window_len {
                        // This input cannot satisfy this filter; callers may explicitly skip this data category.
                        return Err(ThoError::Data { kind: thoosa_shared::DataErrorKind::InputTooShort, msg: format!("time length T={} must be >= window_len={}", t, window_len) });
                    }

                    let input_t = tch_result(filtered_t.as_ref().f_unsqueeze(0))?;

                    let output = self.run_conv_with_padding(*stride, padding, &weights.as_ref(), &input_t);
                    debug!("FreqFilter::apply_tho_tensor, SavGol: id={} weights.size={:?} input.size={:?} output.size={:?}", input_tensor.id(), weights.size(), input_t.size(), output.as_ref().map(|v| v.size()));
                    filtered_t = output?;
                }
                PreparedFilter::Median { window } => {
                    let filtered_t_out = median(&filtered_t, *window)?;
                    debug!("FreqFilter::apply_tho_tensor, Median: id={} window={} filtered_t.size={:?} filtered_t_out.size={:?}", input_tensor.id(), window, filtered_t.size(), filtered_t_out.size());
                    filtered_t = filtered_t_out;
                }
                PreparedFilter::MovingAverage { window, step } => {
                    let filtered_t_out = moving_average(&filtered_t, *window, *step as i64)?;
                    debug!("FreqFilter::apply_tho_tensor, MovingAverage: id={} window={} filtered_t.size={:?} filtered_t_out.size={:?}", input_tensor.id(), window, filtered_t.size(), filtered_t_out.size());
                    filtered_t = filtered_t_out;
                }
                PreparedFilter::ZNorm {..} => {
                    debug!("FreqFilter::apply_tho_tensor, ZNorm: id={} filtered_t.size={:?}", input_tensor.id(), filtered_t.size(),);

                    filtered_t = z_norm(&filtered_t, 1)?;
                }
            }
        }

        Ok(ThoTensor::new(self.out_id(input_tensor.id()), filtered_t)?)
    }

    fn run_conv_with_padding(&self, stride: u32, padding: &Padding, kernel_t: &Tensor, input_t: &Tensor) -> Result<Tensor, ThoError> {
        let t = input_t.size()[1];
        let kernel_size = kernel_t.size()[2];
        let pad = kernel_size / 2;
        let stride_i64 = stride as i64;


        let output_3d = match padding {
            Padding::Zero(_) => tch_result(input_t.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, pad, 1, self.num_channels))?,
            Padding::Replicate(_) => {
                let left = tch_result(tch_result(input_t.f_narrow(2, 0, 1))?.f_repeat([1, 1, pad]))?;
                let right = tch_result(tch_result(input_t.f_narrow(2, t - 1, 1))?.f_repeat([1, 1, pad]))?;
                let xpad = tch_result(Tensor::f_cat(&[&left, input_t, &right], 2))?;
                tch_result(xpad.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, 0, 1, self.num_channels))?
            }
            Padding::Reflect(_) => {
                if pad >= t {
                    return Err(ThoError::new(format!("reflect padding requires pad < T, got pad={} T={}", pad, t)));
                }
                let left = tch_result(tch_result(input_t.f_narrow(2, 1, pad))?.f_flip([2]))?;
                let right = tch_result(tch_result(input_t.f_narrow(2, t - pad - 1, pad))?.f_flip([2]))?;
                let xpad = tch_result(Tensor::f_cat(&[&left, input_t, &right], 2))?;
                tch_result(xpad.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, 0, 1, self.num_channels))?
            }
        };

        Ok(tch_result(output_3d.f_view((self.num_channels, -1)))?)
    }

}

impl <C> WithId for SyncFreqFilter<C>
where
    C: RunCtx + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


// Scalar filter parameters stay in PreparedFilter while IPC transfers only the
// tensors. An empty slot means offloaded (or only partly restored), never ready.
#[derive(Debug)]
pub struct PreparedFilterTensor {
    tensor: Option<ThoTensor<BandPassDims, ()>>,
    shape: Vec<i64>,
    kind: Kind,
}

impl PreparedFilterTensor {
    fn new(tensor: Tensor) -> Result<Self, ThoError> {
        Ok(Self { shape: tensor.size(), kind: tensor.kind(), tensor: Some(ThoTensor::new((), tensor)?) })
    }

    fn ready(&self) -> Result<&ThoTensor<BandPassDims, ()>, ThoError> {
        self.tensor.as_ref().ok_or_else(|| ThoError::new("frequency filter state is offloaded or incompletely restored"))
    }
}

#[derive(Debug)]
// This is public because the public StageWithState implementation exposes the prepared filters as its associated state.
pub enum PreparedFilter {
    NotchBand { low_hz: f64, high_hz: f64 },
    BandPass {
        kernel: PreparedFilterTensor,
        stride: u32,
        padding: Padding,
    },
    SavGol {
        weights: PreparedFilterTensor,
        window_len: i64,
        stride: u32,
        padding: Padding,
    },
    Median {
        window: u32,
    },
    MovingAverage {
        window: u32,
        step: u32,
    },
    ZNorm {

    }
}

impl PreparedFilter {
    fn tensor(&self) -> Option<&PreparedFilterTensor> {
        match self {
            Self::BandPass { kernel, .. } => Some(kernel),
            Self::SavGol { weights, .. } => Some(weights),
            _ => None,
        }
    }

    fn tensor_mut(&mut self) -> Option<&mut PreparedFilterTensor> {
        match self {
            Self::BandPass { kernel, .. } => Some(kernel),
            Self::SavGol { weights, .. } => Some(weights),
            _ => None,
        }
    }
}

#[derive(Debug, Cached, Named)]
pub struct FreqFilter<C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    id: SynKey,
    _ctx: PhantomData<C>,
    filter_init: SyncFreqFilterIpcInit,
    effective_output_stride:i64,
    min_input_split_len:i64
}



impl<C> FreqFilter<C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{


    pub async fn new(filter_configs: &[InputFilter], ws_configs:WsConfigs, find_motifs_run_config_id: RunConfigId, tho_device_idx: usize) -> Result<Arc<Self>, ThoError> {
        debug!("FreqFilter::new: find_motifs_run_config_id={find_motifs_run_config_id} tho_device_idx={tho_device_idx}");
        let id:SynKey = mono_key!(find_motifs_run_config_id, "FreqFilter").into();


        let (num_channels, sample_rate)  = {
            (ws_configs.channels as i64, ws_configs.sample_rate as f64)
        };


        let effective_output_stride = Self::effective_output_stride(filter_configs)?;
        let min_input_split_len = Self::min_input_split_len(filter_configs)?;

        let filter_init = SyncFreqFilterIpcInit::new(filter_configs.to_vec(), num_channels, sample_rate, find_motifs_run_config_id, tho_device_idx);
        SyncFreqFilter::<C>::new(filter_init.clone()).await?;


        Ok(Arc::new(Self {
            //avoids caching each run_config separately when they have same filter - use filter_configs to build unique ids
            id:id.clone(),
            _ctx: PhantomData,
            filter_init,
            effective_output_stride,
            min_input_split_len
        }))

    }

    fn effective_output_stride(filter_configs: &[InputFilter]) -> Result<i64, ThoError> {
        filter_configs.iter().try_fold(1_i64, |effective_stride, filter| {
            let stage_stride = match &filter {
                InputFilter::BandPassFilter(BandPassFilterParams { stride, .. }) | InputFilter::SavGolFilter(SavGolFilterParams { stride, .. }) => *stride as i64,
                InputFilter::AverageFilter( AverageFilterParams{ step, .. }) => *step as i64,
                InputFilter::NotchBandFilter (NotchBandFilterParams{ .. }) | InputFilter::MedianFilter(MedianFilterParams{..})  => 1,
                InputFilter::ZNorm (ZNormFilterParams{..}) => {1}
            };

            effective_stride
                .checked_mul(stage_stride)
                .ok_or_else(|| ThoError::new(format!("FreqFilter::effective_output_stride overflow: effective_stride={effective_stride} stage_stride={stage_stride}")))
        })
    }

    fn min_input_split_len(filter_configs: &[InputFilter]) -> Result<i64, ThoError> {
        filter_configs.iter().rev().try_fold(1_i64, |required_len, filter| {
            match filter {
                InputFilter::NotchBandFilter (NotchBandFilterParams{ .. }) | InputFilter::MedianFilter(MedianFilterParams{..}) => Ok(required_len.max(1)),
                InputFilter::AverageFilter( AverageFilterParams{ step, .. }) =>Self::expand_required_len(required_len, *step as i64, 1),
                InputFilter::BandPassFilter(BandPassFilterParams { stride, padding, kernel_size, .. }) => {
                    let local_min = match padding {
                        Padding::Reflect(_) => (kernel_size / 2) + 1,
                        Padding::Zero(_) | Padding::Replicate(_) => 1,
                    };
                    Self::expand_required_len(required_len, *stride as i64, local_min as i64)
                }
                InputFilter::SavGolFilter(SavGolFilterParams { window_len, stride, .. }) => Self::expand_required_len(required_len, *stride as i64, *window_len as i64),
                InputFilter::ZNorm (ZNormFilterParams{..}) => { Ok(1) }
            }
        })
    }

    fn expand_required_len(required_len: i64, stride: i64, local_min: i64) -> Result<i64, ThoError> {
        if stride <= 0 {
            return Err(ThoError::new(format!("FreqFilter::expand_required_len requires stride > 0, got {stride}")));
        }

        let expanded = if required_len <= 1 {
            1
        } else {
            (required_len - 1)
                .checked_mul(stride)
                .and_then(|v| v.checked_add(1))
                .ok_or_else(|| ThoError::new(format!("FreqFilter::expand_required_len overflow: required_len={required_len} stride={stride}")))?
        };

        Ok(expanded.max(local_min).max(1))
    }
}

#[derive(Debug)]
struct FreqFilterOutputBuilder {
    output_id: SynKey,
    partials: Vec<ThoTensor<ChannelsData, SynKey>>,
}

impl FreqFilterOutputBuilder {
    fn new(output_id: SynKey) -> Self {
        Self {
            output_id,
            partials: vec![],
        }
    }
}

impl OutputBuilder for FreqFilterOutputBuilder {
    type O = ThoTensor<ChannelsData, SynKey>;

    async fn add_partial_output(&mut self, partial: Self::O) -> Result<(), ThoError> {
        // CPU transfer is part of accepting a split, so its failure is immediate.
        let partial = if partial.tho_device_idx() == 0 {
            partial
        } else {
            track_await!(partial.copy_to_device(0)).await?
        };

        self.partials.push(partial);
        Ok(())
    }

    async fn build(self) -> Result<Self::O, ThoError> {
        let mut partials = self.partials.into_iter();
        let first = partials.next().ok_or_else(|| ThoError::new("FreqFilterOutputBuilder: no partial outputs"))?;
        let mut tensors = vec![first.t()];
        tensors.extend(partials.map(|partial| partial.t()));
        let tensor_refs = tensors.iter().collect::<Vec<_>>();

        ThoTensor::new(self.output_id, tch_result(Tensor::f_cat(tensor_refs.as_slice(), 1))?)
    }
}

#[derive(Debug)]
struct FreqFilterSplitsAdjuster {
    min_split_len: MonoMetrics<1, 2>,
    alignment: i64,
}

impl FreqFilterSplitsAdjuster {
    fn new(min_split_len: MonoMetrics<1, 2>, alignment: i64) -> Self {
        Self {
            min_split_len,
            alignment: alignment.max(1),
        }
    }

    fn align_split_len(&self, split_len: &mut MonoMetrics<1, 2>) {
        let prev_len = split_len.dim::<1>();
        let min_len = self.min_split_len.dim::<1>();
        let mut aligned_len = prev_len;

        if aligned_len > min_len && self.alignment > 1 {
            aligned_len = (aligned_len / self.alignment) * self.alignment;
            if aligned_len < min_len {
                aligned_len = min_len;
            }
        }

        if aligned_len != prev_len {
            *split_len = [split_len.dim::<0>(), aligned_len].into();
        }
    }
}

#[async_trait]
impl<AC> Adjuster<AC> for FreqFilterSplitsAdjuster
where
    AC: SplitsAdjusterContext<Metrics = MonoMetrics<1, 2>> + Send + Sync,
{
    type A = ChannelsDataSelector;

    async fn adjust(
        &self,
        _adjustable: Self::A,
        adjust_context: &mut AC,
    ) -> Result<Self::A, AdjusterError<Self::A>> {
        let split_start = adjust_context.done_to();
        let mut split_end = split_start.clone();
        split_end += adjust_context.split_len().clone();
        adjust_context.input_len().cap(&mut split_end);

        debug!("FreqFilterSplitsAdjuster::adjust: split_start={split_start:?} split_end={split_end:?} input_len={:?}", adjust_context.input_len());

        Ok((split_start.clone()..split_end).into())
    }

    fn degrade(&self, adjust_context: &mut AC) -> bool {
        let prev_len = adjust_context.split_len().dim::<1>();
        *adjust_context.split_len_mut() /= 2;
        self.align_split_len(adjust_context.split_len_mut());
        let next_len = adjust_context.split_len().dim::<1>();
        let can_degrade = next_len >= self.min_split_len.dim::<1>() && next_len < prev_len;

        debug!("FreqFilterSplitsAdjuster::degrade: prev_len={prev_len} next_len={next_len} min_split_len={:?} alignment={} can_degrade={can_degrade}", self.min_split_len, self.alignment);

        can_degrade
    }
}

impl<C> Adjustable for SyncFreqFilter<C>
where
    C: RunCtx + 'static,
{
}

impl<C> OutId for SyncFreqFilter<C>
where
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for SyncFreqFilter<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData, SynKey>;
    type Out = ThoTensor<ChannelsData, SynKey>;
}

impl<C> SyncStage for SyncFreqFilter<C>
where
    C: RunCtx + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        // Expose the prepared filters to the runner so their device placement is handled before the synchronous call.
        Some(self)
    }

    fn do_run<'f>(self: Arc<Self>, input_tensor: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        self.apply_tho_tensor(input_tensor)
    }
}

impl<C> StageWithState for SyncFreqFilter<C>
where
    C: RunCtx + 'static,
{
    type S = Vec<PreparedFilter>;

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Option<tokio::sync::MappedMutexGuard<'_, Self::S>>, ThoError>> {
        async move {
            let mut filters = self.filters.lock().await;
            // Restores arrive one tensor at a time and in arbitrary order. Do
            // not expose a usable state until every tensor has been restored.
            for filter in filters.iter() {
                if let Some(tensor) = filter.tensor() { tensor.ready()?; }
            }
            for filter in filters.iter_mut() {
                if let Some(slot) = filter.tensor_mut() {
                    let tensor = slot.ready()?;
                    if tensor.tho_device_idx() != to_device_idx {
                        slot.tensor = Some(track_await!(tensor.copy_to_device(to_device_idx)).await?);
                    }
                }
            }
            Ok(Some(MutexGuard::map(filters, |v|v)))
        }.boxed()
    }

    fn get_state_info(&self, _tho_device: usize) -> BoxFuture<'_, Result<StageStateInfo, ThoError>> {
        async move {
            let filters = self.filters.lock().await;
            let mut tensors = Vec::new();
            for (index, filter) in filters.iter().enumerate() {
                if let Some(slot) = filter.tensor() {
                    let tensor = slot.ready()?;
                    // Position identifies repeated filters of the same kind and
                    // preserves their association with the retained parameters.
                    tensors.push(WorkerTensorInfoPayload::new(TensorKey::stage_state(SYNC_FREQ_FILTER_ENDPOINT, &self.stage_id, index), tensor.kind().into(), tensor.size(), tensor_mem_size(tensor.kind(), &tensor.size())));
                }
            }
            Ok(StageStateInfo::new(tensors, None))
        }.boxed()
    }

    fn get_state_tensor(&self, _tho_device: usize, tensor_id: String) -> BoxFuture<'_, Result<Tensor, ThoError>> {
        async move {
            let filters = self.filters.lock().await;
            let slot = filters.iter().enumerate().find(|(index, _)| index.to_string() == tensor_id).and_then(|(_, filter)| filter.tensor()).ok_or_else(|| ThoError::NotFound(Some(tensor_id)))?;
            // The worker's D2H request retains this alias until its copy has
            // synchronized; PurgeStage then drops the stage-owned allocation.
            Ok(tch_result(slot.ready()?.as_ref().f_alias())?)
        }.boxed()
    }

    fn set_state_tensor(&self, tho_device: usize, tensor_id: String, tensor: Tensor) -> BoxFuture<'_, Result<(), ThoError>> {
        async move {
            let mut filters = self.filters.lock().await;
            let slot = filters.iter_mut().enumerate().find(|(index, _)| index.to_string() == tensor_id).and_then(|(_, filter)| filter.tensor_mut()).ok_or_else(|| ThoError::NotFound(Some(tensor_id.clone())))?;
            if slot.tensor.is_some() {
                return Err(ThoError::new(format!("frequency filter state tensor {tensor_id} is already loaded")));
            }
            if tensor.kind() != slot.kind || tensor.size() != slot.shape || tensor.device() != Device::from(ThoDevice::from(tho_device)) {
                return Err(ThoError::new(format!("frequency filter state tensor {tensor_id} has incompatible kind, shape, or device")));
            }
            slot.tensor = Some(ThoTensor::new((), tensor)?);
            Ok(())
        }.boxed()
    }
    
    fn do_run_with_state<'f>(self: Arc<Self>, input_tensor: &Self::In<'f>, filters: &mut Self::S, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        self.apply_tho_tensor_with_state(filters.as_slice(), input_tensor)
    }
}

#[async_trait]
impl <C> FuturesRunnerMapPurger for SyncFreqFilter<C>
where
    C:RunCtx,
    Self:Send + Sync
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        let filters = self.filters.lock().await;
        device_idx != 0 && filters.iter().filter_map(PreparedFilter::tensor).any(|slot| slot.tensor.as_ref().is_some_and(|tensor| tensor.tho_device_idx() == device_idx))
    }

    async fn purge_gpu(&self, _ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, _name: &str) -> Result<(), ThoError> {
        debug!("SyncFreqFilter::purge_gpu");
        let mut filters = self.filters.lock().await;
        for filter in filters.iter_mut() {
            if let Some(slot) = filter.tensor_mut() {
                if let Some(tensor) = &slot.tensor && device_idx != 0 && tensor.tho_device_idx() == device_idx {
                    slot.tensor = Some(track_await!(tensor.copy_to_device(0)).await?);
                }
            }
        }

        Ok(())
    }

    async fn drop_gpu(&self, device_idx: usize, _name: &str) -> Result<(), ThoError> {
        // In IPC, the parent has already copied every state tensor to CPU.
        // Release worker storage, retaining only metadata needed for restoration.
        let mut filters = self.filters.lock().await;
        for filter in filters.iter_mut() {
            if let Some(slot) = filter.tensor_mut() {
                if slot.tensor.as_ref().is_some_and(|tensor| device_idx != 0 && tensor.tho_device_idx() == device_idx) {
                    slot.tensor = None;
                }
            }
        }
        Ok(())
    }

}


impl<C> OutId for FreqFilter<C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for FreqFilter<C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;
    type Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;
}


impl<C> AsyncStage for FreqFilter<C>
where
    C: MaxParallelCtx + RunCtx + Adjustable + 'static,
{


    async fn run(
        self: Arc<Self>,
        input: Self::In<'static>,
        mut ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        let out_id = self.out_id(input.id());
        let input_size = input.size().clone();
        let min_split_len = [input_size.dim::<0>(), self.min_input_split_len].into();
        // Keep the source tensor resident for filtering and every split retry.
        let mut required_inputs = HashMap::new();
        input.add_required_ids(&mut required_inputs);
        let required_ids = required_inputs.iter().map(|(key, ids)| (key.clone(), ids)).collect();
        // The async loader survives resets; resolve its endpoint for the current
        // generation on each invocation rather than retaining a stale proxy.
        let mut init = self.filter_init.clone();
        init.tho_device_idx = ctx.tho_device_idx();
        let sync_freq_filter = SyncFreqFilter::<C>::new(init).await?;

        let filtered = track_await!(OOMHandler::<'_, '_, _, NoStageAdjuster, _, ChannelsDataSelector>::new(
            vec![
                Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::MID))),
                Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("freq_filter"))),
                Adjusters::Input(input_adjuster(FreqFilterSplitsAdjuster::new(min_split_len, self.effective_output_stride))),
            ],
            None,
        )
        .run(
            &sync_freq_filter,
            Arc::clone(&input),
            ctx.branch(true),
            Contexts::new(
                SplitsAdjusterContextImpl::new(input_size, FreqFilterOutputBuilder::new(out_id)),
                NoopRunnableContext {},
                MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
            ),
        )).await?;

        debug!("FreqFilter::run: filtered={filtered:?} done");

        Ok(Arc::new(DevicesWrap::wrap(filtered)))
    }
}

impl LocalSelect<SynKey, ChannelsDataSelector> for ThoTensor<ChannelsData, SynKey> {
    type Output = ThoTensor<ChannelsData, SynKey>;

    fn local_subview(id:&SynKey, t:&Tensor, selector: ChannelsDataSelector) -> Result<Self::Output, ThoError> {
        let _size: MonoMetrics<1, 2> = t.size().try_into()?;

        let ranges = MonoMetrics::<1, 2>::map_to_ranges(&selector.0.start, &selector.0.end);
        debug!("ChannelsData2Selector::subview 1. ranges={ranges:?}, size={:?}", t.size());


        let sub_tensor = tch_result(t.f_i((ranges[0].clone(), ranges[1].clone())))?;

        Ok(ThoTensor::new(id.clone(), sub_tensor)?)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::common::cachable::SelectRef;
    use crate::ipc::cuda_process::CudaIPCActorHandle;
    use crate::ipc::cuda_wokrer::{CudaWorker, InitFn};
    use crate::ipc::worker_msg::OpResult;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::transforms::filters::ZeroPadding;
    use crate::ws::mock_ws_simp;
    use std::time::Duration;
    use std::sync::Arc;
    use tch::Tensor;
    use thoosa_common::runner::run_ctx::RunCtxImpl;
    use thoosa_common::stages::freq_filter::{SyncFreqFilter, SyncFreqFilterIpcInit};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::transforms::filters::{AverageFilterParams, InputFilter, MedianFilterParams};
    use thoosa_common::transforms::smoothing::{median, moving_average};
    use thoosa_common::utils::from;
    use thoosa_digest_id::DigestHash;
    use thoosa_shared::ThoError;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::traits::{StageHash};
    use thoosa_common::runner::sync_stage::{SyncStage};

    #[tokio::test]
    async fn test_short_filter_input_is_explicit_data_error() -> Result<(), ThoError> {
        let input = ThoTensor::new(mono_key!("short", "input").into(), tch_result(Tensor::f_ones([1, 3], (tch::Kind::Float, tch::Device::Cpu)))?)?;
        let filters = vec![InputFilter::SavGolFilter(SavGolFilterParams { window_len: 7, poly_order: 2, stride: 1, padding: Padding::Zero(ZeroPadding {}) })];
        let stage = Arc::new(SyncFreqFilter::<RunCtxImpl>::new_local(SyncFreqFilterIpcInit::new(filters, 1, 100.0, 1, 0))?);
        let error = stage.do_run(&input, &RunCtxImpl::new(0, "short input")).unwrap_err();
        assert_eq!(error.data_error_kind(), Some(thoosa_shared::DataErrorKind::InputTooShort));
        Ok(())
    }

    fn state_test_filters() -> Vec<InputFilter> {
        vec![
            InputFilter::BandPassFilter(BandPassFilterParams { low_hz: 3., band_hz: 12., kernel_size: 9, stride: 1, padding: Padding::Zero(ZeroPadding {}) }),
            InputFilter::MedianFilter(MedianFilterParams { window: 5 }),
            InputFilter::SavGolFilter(SavGolFilterParams { window_len: 7, poly_order: 2, stride: 1, padding: Padding::Zero(ZeroPadding {}) }),
            // A second band-pass verifies that tensor keys identify positions,
            // rather than collapsing repeated filter kinds into one entry.
            InputFilter::BandPassFilter(BandPassFilterParams { low_hz: 5., band_hz: 16., kernel_size: 11, stride: 2, padding: Padding::Zero(ZeroPadding {}) }),
        ]
    }

    fn state_test_input(scale: f64) -> Result<ThoTensor<ChannelsData, SynKey>, ThoError> {
        let tensor = tch_result(tch_result(tch_result(tch_result(tch_result(Tensor::f_arange(256, (Kind::Float, Device::Cpu)))?.f_mul_scalar(0.13))?.f_sin())?.f_mul_scalar(scale))?.f_view([2, 128]))?;
        ThoTensor::new(mono_key!("filter-state-input", "filter-state-input").into(), tensor)
    }

    fn state_test_reference(stage: &SyncFreqFilter<RunCtxImpl>, input: &ThoTensor<ChannelsData, SynKey>) -> Result<Tensor, ThoError> {
        // Explicit CPU convolution supplies an independent reference and avoids
        // oneDNN's raw allocator interface, unsupported by the parent's shared arena.
        let filters = stage.filters.try_lock().context("reference filter state is locked")?;
        let mut output = tch_result(input.as_ref().f_alias())?;
        for filter in filters.iter() {
            output = match filter {
                PreparedFilter::BandPass { kernel, stride, padding } | PreparedFilter::SavGol { weights: kernel, stride, padding, .. } => {
                    assert!(matches!(padding, Padding::Zero(_)));
                    let kernel = kernel.ready()?;
                    let width = kernel.size()[2];
                    tch_result(tch_result(tch_result(tch_result(output.f_constant_pad_nd([width / 2, width / 2]))?.f_unfold(1, width, *stride as i64))?.f_mul(&kernel.as_ref()))?.f_sum_dim_intlist(&[2i64][..], false, Kind::Float))?
                }
                PreparedFilter::Median { window } => median(&output, *window)?,
                _ => panic!("unexpected filter in state-transfer fixture"),
            };
        }
        Ok(output)
    }

    // Use the production stateful wrapper so metadata, D2H, worker purge,
    // per-tensor H2D, and the next stage invocation all cross the real protocol.
    fn state_test_init<'a>(_stage_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _tensors: &'a AllTensors, device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
        async move {
            let init: Box<dyn Any> = init;
            let mut init = *init.downcast::<SyncFreqFilterIpcInit>().map_err(|_| ThoError::new("unexpected filter initializer"))?;
            init.tho_device_idx = device.tho_device_idx();
            let stage = Arc::new(SyncFreqFilter::<WokerRunCtx>::new_local(init)?);
            Ok(Box::new(WorkerWrapperWithState::new(stage, PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
        }.boxed()
    }

    async fn do_test_filter_state() -> Result<(), ThoError> {
        let global = GlobalCtx::global();
        let actor = global.cuda_actor(1);
        let multiple = state_test_filters();
        let cases = [vec![], vec![InputFilter::MedianFilter(MedianFilterParams { window: 5 })], vec![multiple[0].clone()], multiple];
        for (case, configs) in cases.into_iter().enumerate() {
            let config_id = 7100 + case as RunConfigId;
            let reference = Arc::new(SyncFreqFilter::<RunCtxImpl>::new_local(SyncFreqFilterIpcInit::new(configs.clone(), 2, 100., config_id, 0))?);
            let tensor_count = reference.get_state_info(0).await?.tensors.len();
            let init = SyncFreqFilterIpcInit::new(configs, 2, 100., config_id, 1);
            let endpoint_id = init.new_id().to_string();
            let stage = SyncFreqFilter::<RunCtxImpl>::new(init).await?;
            let ctx = RunCtxImpl::new(1, "freq_filter_state");
            // Exercise both initial invocation and repeated restore cycles.
            for cycle in 0..3 {
                let input = state_test_input(cycle as f64 + 1.)?;
                let expected = state_test_reference(&reference, &input)?;
                let input = Arc::new(DevicesWrap::wrap(input));
                let output = actor.run_stage(Arc::clone(&stage), input, 1, &ctx).await?;
                let output = output.copy_to_device(0).await?;
                assert!(tch_result(output.as_ref().f_allclose(&expected, 1e-4, 1e-5, false))?, "filter output changed in case {case}, cycle {cycle}");
                // A purge for another device must preserve this stage's state.
                stage.purge_gpu(None, 2, "other device").await?;
                assert_eq!(stage.has_gpu_data(1).await, global.is_ipc() || tensor_count > 0);
                global.dump_all_to_cpu(&HashMap::new(), 1, "freq filter state test").await?;
                assert!(!stage.has_gpu_data(1).await);
                assert!(global.is_all_on_cpu(1, true));
                // The CPU dump must survive both repeated global and direct purges.
                global.dump_all_to_cpu(&HashMap::new(), 1, "already offloaded").await?;
                stage.purge_gpu(None, 1, "already offloaded").await?;

                if global.is_ipc() && tensor_count > 0 {
                    let ipc: &dyn Any = actor.as_ref();
                    let ipc = ipc.downcast_ref::<CudaIPCActorHandle>().unwrap();
                    let input = state_test_input(1.)?.copy_to_device(1).await?;
                    // Bypass proxy pre_run: a worker invocation must fail until
                    // the parent restores state, proving worker storage was dropped.
                    let response = ipc.invoke_stage(ipc.worker_id()?, SYNC_FREQ_FILTER_ENDPOINT.into(), Some(endpoint_id.clone()), Box::new(input), false).await?;
                    let OpResult::Failed(error) = response.status else { panic!("worker kept filter tensors after purge") };
                    debug!("do_test_filter_state: error={error:?}");
                    assert!(ThoError::from(error).to_string().contains("frequency filter state is offloaded"));
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_filter_state_local() -> Result<(), ThoError> {
        let global = GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let result = tokio::time::timeout(Duration::from_secs(60), do_test_filter_state()).await?;
        global.shutdown().await;
        result
    }

    #[test]
    fn test_filter_state_ipc() -> Result<(), ThoError> {
        crate::ipc::tests::run_ipc_test(module_path!(), async {
            tokio::time::timeout(Duration::from_secs(60), do_test_filter_state()).await?
        }, async |sock| {
            let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
            CudaWorker::run_worker(sock, &ws, HashMap::from([(SYNC_FREQ_FILTER_ENDPOINT.into(), Box::new(state_test_init) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
        })
    }

    #[test]
    fn test_filter_rebuilds_after_worker_reset_ipc() -> Result<(), ThoError> {
        crate::ipc::tests::run_ipc_test(module_path!(), async {
            tokio::time::timeout(Duration::from_secs(60), async {
                let global = GlobalCtx::global();
                let configs = state_test_filters();
                let init = SyncFreqFilterIpcInit::new(configs.clone(), 2, 100., 7110, 1);
                let filter = FreqFilter::<RunCtxImpl>::new(&configs, WsConfigs::new(2, 0, 100), 7110, 1).await?;
                let reference = SyncFreqFilter::<RunCtxImpl>::new_local(SyncFreqFilterIpcInit { tho_device_idx: 0, ..init.clone() })?;
                let input = state_test_input(1.)?;
                let expected = state_test_reference(&reference, &input)?;
                // As in production, register input copies so reset can discard
                // their old worker references while retaining the CPU source.
                let inputs = crate::runner::tmp_cache::TmpCached::new("filter reset input").await;
                let input_id = input.id().clone();
                inputs.insert(input);
                let input = inputs.get(&input_id).unwrap();
                let mut previous = SyncFreqFilter::<RunCtxImpl>::new(init.clone()).await?;
                let unused = SyncFreqFilter::<RunCtxImpl>::new(SyncFreqFilterIpcInit { find_motifs_run_config_id: 7111, ..init.clone() }).await?;
                for cycle in 0..3 {
                    if cycle > 0 {
                        global.cuda_actor(1).restart().await?;
                        // Unused filters remain registered without ever being
                        // rebuilt. They must not poison a later global offload.
                        assert!(!unused.has_gpu_data(1).await);
                        unused.purge_gpu(None, 1, "retired filter").await?;
                        global.dump_all_to_cpu(&HashMap::new(), 1, "before filter rebuild").await?;
                        // Exercise the retained proxy directly: new use must
                        // rebuild its kernels instead of uploading the old dump.
                        let output = global.cuda_actor(1).run_stage(Arc::clone(&previous), Arc::clone(&input), 1, &RunCtxImpl::new(1, "retained filter")).await?;
                        let output = output.copy_to_device(0).await?;
                        assert!(tch_result(output.as_ref().f_allclose(&expected, 1e-4, 1e-5, false))?, "retained filter changed after reset");
                    }
                    // Keep the async loader alive across replacement, just as
                    // RandomSlicesLoader does between training trial attempts.
                    let output = Arc::clone(&filter).run(Arc::clone(&input), RunCtxImpl::new(1, "filter reset")).await?;
                    let output = output.select_ref(0, None::<ChannelsDataSelector>).await?;
                    assert!(tch_result(output.as_ref().as_ref().f_allclose(&expected, 1e-4, 1e-5, false))?, "filter changed after reset {cycle}");
                    let current = SyncFreqFilter::<RunCtxImpl>::new(init.clone()).await?;
                    assert!(Arc::ptr_eq(&previous, &current), "recovery replaced the cached proxy instead of its endpoint");
                    assert!(Arc::ptr_eq(&current, &SyncFreqFilter::<RunCtxImpl>::new(init.clone()).await?), "same-generation filter was not shared");
                    previous = current;
                    global.dump_all_to_cpu(&HashMap::new(), 1, "filter reset test").await?;
                }
                Ok(())
            }).await?
        }, async |sock| {
            let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
            CudaWorker::run_worker(sock, &ws, HashMap::from([(SYNC_FREQ_FILTER_ENDPOINT.into(), Box::new(state_test_init) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
        })
    }

    #[test]
    fn test_cached_filter_recomputes_after_worker_reset_ipc() -> Result<(), ThoError> {
        crate::ipc::tests::run_ipc_test(module_path!(), async {
            tokio::time::timeout(Duration::from_secs(60), async {
                use crate::runner::common::cachable::CachedPurgeOutput;
                let global = GlobalCtx::global();
                let configs = state_test_filters();
                let filter = FreqFilter::<RunCtxImpl>::new(&configs, WsConfigs::new(2, 0, 100), 7112, 1).await?;
                let reference = SyncFreqFilter::<RunCtxImpl>::new_local(SyncFreqFilterIpcInit::new(configs, 2, 100., 7112, 0))?;
                let input = state_test_input(1.)?;
                let expected = state_test_reference(&reference, &input)?;
                let inputs = crate::runner::tmp_cache::TmpCached::new("cached filter reset input").await;
                let input_id = input.id().clone();
                inputs.insert(input);
                let input = inputs.get(&input_id).unwrap();
                // Keep this cache area open so cleanup cannot offload the
                // result to CPU and accidentally hide the lost-replica bug.
                let mut ctx = RunCtxImpl::new(1, "cached filter reset");
                let before = Arc::clone(&filter).run_cached(Arc::clone(&input), ctx.branch(false)).await?;
                assert!(before.is_on_device(1) && !before.is_on_device(0));
                let hit = Arc::clone(&filter).run_cached(Arc::clone(&input), ctx.branch(false)).await?;
                assert!(Arc::ptr_eq(&before, &hit), "expected a cached GPU result before reset");
                global.cuda_actor(1).restart().await?;
                assert!(!before.has_any_data(), "reset must discard the failed worker's only replica");
                // The input's CPU source survives. A fresh lookup must rerun
                // the filter, using its restored endpoint, instead of returning
                // the successful future whose output now has no replicas.
                let after = Arc::clone(&filter).run_cached(input, ctx.branch(false)).await?;
                assert!(!Arc::ptr_eq(&before, &after), "reset left an empty output in the cache");
                let output = after.select_ref(0, None::<ChannelsDataSelector>).await?;
                assert!(tch_result(output.as_ref().as_ref().f_allclose(&expected, 1e-4, 1e-5, false))?, "recomputed filter output changed after reset");
                Ok(())
            }).await?
        }, async |sock| {
            let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
            CudaWorker::run_worker(sock, &ws, HashMap::from([(SYNC_FREQ_FILTER_ENDPOINT.into(), Box::new(state_test_init) as InitFn)]), thoosa_common::tho_device::ThoDevice::from(1usize)).await
        })
    }

    #[tokio::test]
    async fn test_filter_state_restore_requires_all_tensors() -> Result<(), ThoError> {
        let stage = Arc::new(SyncFreqFilter::<RunCtxImpl>::new_local(SyncFreqFilterIpcInit::new(state_test_filters(), 2, 100., 7104, 0))?);
        let input = state_test_input(1.)?;
        let ctx = RunCtxImpl::new(0, "filter state restore");
        let expected = Arc::clone(&stage).do_run(&input, &ctx)?;
        let info = stage.get_state_info(0).await?;
        assert_eq!(info.tensors.len(), 3);
        let mut tensors = Vec::new();
        for info in info.tensors {
            let TensorKey::StageState { endpoint_name, stage_id, tensor_key } = info.tensor_key else { panic!("expected stage-state identity") };
            assert_eq!(endpoint_name, SYNC_FREQ_FILTER_ENDPOINT);
            assert_eq!(stage_id, "7104");
            let tensor = stage.get_state_tensor(0, tensor_key.clone()).await?;
            assert_eq!(tensor.size(), info.shape);
            assert_eq!(tensor_mem_size(tensor.kind(), &tensor.size()), info.tensor_byte_len);
            tensors.push((tensor_key, tensor));
        }
        // Model worker offload without requiring CUDA in this state-machine test.
        for filter in stage.filters.lock().await.iter_mut() {
            if let Some(slot) = filter.tensor_mut() { slot.tensor = None; }
        }
        assert!(StageWithState::pre_run(stage.as_ref(), 0).await.is_err());
        assert!(stage.set_state_tensor(0, "unknown".into(), tch_result(Tensor::f_zeros([1], (Kind::Float, Device::Cpu)))?).await.is_err());
        let (key, tensor) = &tensors[0];
        assert!(stage.set_state_tensor(0, key.clone(), tch_result(tensor.f_to_kind(Kind::Double))?).await.is_err());
        assert!(stage.set_state_tensor(0, key.clone(), tch_result(Tensor::f_zeros([1], (Kind::Float, Device::Cpu)))?).await.is_err());
        assert!(stage.set_state_tensor(1, key.clone(), tch_result(tensor.f_alias())?).await.is_err());
        // HashMap iteration during H2D is unordered. Restore in reverse order,
        // reject duplicates, and only permit execution after the final upload.
        while let Some((key, tensor)) = tensors.pop() {
            stage.set_state_tensor(0, key.clone(), tch_result(tensor.f_alias())?).await?;
            assert!(stage.set_state_tensor(0, key, tensor).await.is_err());
            if !tensors.is_empty() { assert!(StageWithState::pre_run(stage.as_ref(), 0).await.is_err()); }
        }
        let mut state = StageWithState::pre_run(stage.as_ref(), 0).await?.unwrap();
        let actual = Arc::clone(&stage).do_run_with_state(&input, &mut *state, &ctx)?;
        assert!(tch_result(actual.as_ref().f_allclose(&expected.as_ref(), 0., 0., false))?);
        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_apply_tho_tensor_with_median_filter() -> Result<(), ThoError> {
        let input_t = tch_result(Tensor::from_slice(&[1.0f32, 10.0, 2.0, 9.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0]).f_view([2, 5]))?;
        let input = ThoTensor::new(SynKey::new([mono_key!("median", "median")]), input_t)?;

        let output = Arc::new(SyncFreqFilter::<RunCtxImpl>::new_local(SyncFreqFilterIpcInit::new(vec![InputFilter::MedianFilter(MedianFilterParams { window: 5 })], 2, 100.0, 1, 1))?).do_run(&input, &RunCtxImpl::new(0, "median_filter"))?;
        let expected = median(&input.as_ref(), 5)?;
        let output_flat = tch_result(output.as_ref().f_view([-1]))?;
        let expected_flat = tch_result(expected.f_view([-1]))?;

        assert_eq!(from::<Vec<f32>>(&output_flat)?, from::<Vec<f32>>(&expected_flat)?);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_apply_tho_tensor_with_moving_average_filter() -> Result<(), ThoError> {

        let input_t = tch_result(Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0]).f_view([2, 5]))?;
        let input = ThoTensor::new(SynKey::new([mono_key!("moving_average", "moving_average")]), input_t)?;

        let output = Arc::new(SyncFreqFilter::<RunCtxImpl>::new_local(SyncFreqFilterIpcInit::new(vec![InputFilter::AverageFilter(AverageFilterParams { window: 4, step: 1 })], 2, 100.0, 0, 1) )?).do_run(&input, &RunCtxImpl::new(0, "moving_average_filter"))?;
        let expected = moving_average(&input.as_ref(), 4, 1)?;

        assert!(output.as_ref().allclose(&expected, 1e-5, 1e-5, false));

        Ok(())
    }



    #[test]
    fn test_hash_enum() {
        // Exercise every Rust enum field shape in the one test allowed for this change.
        #[derive(DigestHash)]
        #[digest(tag = "thoosa.tests.EnumHashProbe:v1")]
        enum EnumHashProbe {
            Unit,
            Tuple(u32),
            #[digest(name = "RenamedTuple")]
            OtherTuple(u32),
            Named { value: u32 },
        }

        let unit_hash = EnumHashProbe::Unit.blake3_hash();
        let tuple_hash = EnumHashProbe::Tuple(1).blake3_hash();
        let other_tuple_hash = EnumHashProbe::OtherTuple(1).blake3_hash();
        let named_hash = EnumHashProbe::Named { value: 1 }.blake3_hash();

        assert_ne!(unit_hash, tuple_hash);
        assert_ne!(tuple_hash, other_tuple_hash);
        assert_ne!(tuple_hash, named_hash);
    }

    #[test]
    fn test_hash(){


        let init = SyncFreqFilterIpcInit::new(
            vec![
                // Keep the smoke test non-empty so it executes the generated enum arm, not only Vec hashing.
                InputFilter::MedianFilter(MedianFilterParams { window: 5 }),
            ],
            3,
            0.63,
            3,
            4
        );


        let _d : StageHash = init.blake3_hash().into();
        assert_eq!(size_of::<StageHash>(), 32);

    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ChannelsDataSelector(pub Range<MonoMetrics<1, 2>>);

#[typetag::serde]
impl ThoSelector for ChannelsDataSelector {
    fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(self.clone())
    }

    fn tensor_ranges(&self) -> Result<Vec<Range<i64>>, ThoError> {
        Ok(MonoMetrics::<1, 2>::map_to_ranges(&self.0.start, &self.0.end).into())
    }
}

impl Adjustable for ChannelsDataSelector {}

impl From<Range<MonoMetrics<1, 2>>> for ChannelsDataSelector {
    fn from(value: Range<MonoMetrics<1, 2>>) -> Self {
        ChannelsDataSelector(value)
    }
}

pub const FREQ_FILTER: u32 = 2;
