use async_trait::async_trait;
use cached_derive::Cached;
use log::debug;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::{Device, Kind, Tensor};
use thoosa_common::debug_plot;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::adjusters::splits::{OutputBuilder, SplitsAdjusterContext, SplitsAdjusterContextImpl};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, NoopRunnableContext, OOMHandler, input_adjuster, run_ctx_adjuster};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_named_derive::Named;
use thoosa_shared::traits::RunCtx;
use thoosa_shared::traits::{AsyncStage, OutId, Stage, SyncStage};

use thoosa_common::dims::{AnyDims, AnyMetrics, ChannelsData, Dims};
use thoosa_common::transforms::filters::{AverageFilterParams, BandPassFilterParams, InputFilter, MedianFilterParams, NotchBandFilterParams, SavGolFilterParams, ZNormFilterParams};

use crate::tasks::cross_dims::ChannelsDataSelector;
use crate::tasks::stage_ids::FREQ_FILTER;
use crate::tasks::transforms::smoothing::{median, moving_average};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::transforms::filters::{Padding, band_pass_filter, notch_band_rfft};
use thoosa_common::utils::z_norm;
use thoosa_common::ws::Workspace;
use thoosa_proxy_derive::Proxy;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::math::sg_smoothing_coeffs;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Metrics, MonoMetrics, WithId};
use thoosa_shared::{CopyToDevice, ThoError, mono_key};

pub const SYNC_FREQ_FILTER_ENDPOINT: &str = "SyncFreqFilter";

#[derive(Debug)]
pub enum FreqFilterDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    Local(Arc<SyncFreqFilter<C>>),
    Proxy(Arc<ProxySyncFreqFilter>),
}

#[derive(Debug, Clone)]
struct BandPassDims;

impl Dims for BandPassDims {
    type Metrics = AnyMetrics;//TODO
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SyncFreqFilterIpcInit {
    filter_configs: Vec<InputFilter>,
    num_channels: i64,
    sample_rate: f64,
    find_motifs_run_config_id: RunConfigId
}

impl<C> From<&FreqFilter<C>> for SyncFreqFilterIpcInit
where
    C: RunCtx + Adjustable + 'static,
{
    fn from(freq_filter: &FreqFilter<C>) -> Self {
        Self {
            filter_configs: freq_filter.filter_configs.clone(),
            num_channels: freq_filter.num_channels,
            sample_rate: freq_filter.sample_rate,
            find_motifs_run_config_id: 0, //TODO
        }
    }
}

impl SyncFreqFilterIpcInit {
    pub fn new(filter_configs: Vec<InputFilter>, num_channels: i64, sample_rate: f64, find_motifs_run_config_id: RunConfigId) -> Self {
        //id: mono_key!(0, "").into(),
        Self {  filter_configs, num_channels, sample_rate,find_motifs_run_config_id }
    }
}

#[derive(Debug, Proxy)]//// fields = "id: SynKey = init.id.clone()",
#[proxy(
    common = "thoosa_common",
    name = "ProxySyncFreqFilter",
    endpoint = "SYNC_FREQ_FILTER_ENDPOINT",
    new_args = "init: SyncFreqFilterIpcInit, tho_device_idx: usize",
    init = "init.clone()",
    fields = "id: SynKey = (FREQ_FILTER, \"SyncFreqFilter\").into()",
    out_id = "&self.id + input_id",
    endpoint_init = "sync_freq_filter_endpoint_init",
    init_type = "SyncFreqFilterIpcInit",
    init_stage = "SyncFreqFilter::new_for_ipc(init)?"
)]
struct SyncFreqFilterProxyTarget {
    base: SyncFreqFilter<thoosa_common::runner::run_ctx_2::RunCtxImpl>,
}

#[derive(Debug)]
pub struct SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    filters: Vec<PreparedFilter>,
    num_channels:i64,
    sample_rate:f64,
    pd:PhantomData<C>
}

impl<C> SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{


    pub async fn new(filter_configs: &[InputFilter], num_channels:i64, sample_rate:f64, find_motifs_run_config_id: RunConfigId, tho_device_idx: usize) -> Result<FreqFilterDynSyncStage<C>, ThoError> {

        if GlobalCtx::global().is_ipc() {
            Ok(FreqFilterDynSyncStage::Proxy(Arc::new(ProxySyncFreqFilter::new(SyncFreqFilterIpcInit::new(filter_configs.to_vec(), num_channels, sample_rate, find_motifs_run_config_id), tho_device_idx).await?)))
        } else {
            Ok(FreqFilterDynSyncStage::Local(Arc::new(Self::new_local(filter_configs, num_channels, sample_rate, find_motifs_run_config_id)?)))
        }
    }

    pub fn new_local(filter_configs: &[InputFilter], num_channels:i64, sample_rate:f64, find_motifs_run_config_id: RunConfigId) ->Result<Self, ThoError>{

        let filters = SyncFreqFilter::<RunCtxImpl>::prepare_filters(filter_configs, num_channels, sample_rate)?;
        let filters_have_config = filters.iter().any(|f|{
            match f {
                PreparedFilter::ZNorm { .. } => false,
                _=> true,
            }
        });

        let filters_have_config = filters.iter().any(|f|{
            match f {
                PreparedFilter::ZNorm { .. } => false,
                _=> true,
            }
        });
        let id = if filters_have_config {
            SynKey::new([
                mono_key!(FREQ_FILTER, "FreqFilter"),
                //TODO compare filters' parameters - if there are no floats they can be used as is instead of find_motifs_run_config_id + precision for float params
                mono_key!(find_motifs_run_config_id, format!("find_motifs_run_config_id={find_motifs_run_config_id}")),
            ])
        } else {
            SynKey::new([
                mono_key!(FREQ_FILTER, "FreqFilter"),
            ])
        };


        Ok(Self {
            id,
            filters,
            num_channels,
            sample_rate,
            pd: Default::default(),
        })
    }

    fn prepare_filters(filter_configs: &[InputFilter], num_channels: i64, sample_rate: f64) -> Result<Vec<PreparedFilter>, ThoError> {
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

                    let kernel = band_pass_filter(
                        band_pass_config.low_hz,
                        high_hz,
                        band_pass_config.kernel_size as i64,
                        sample_rate,
                        1e-6
                    )?.f_repeat([num_channels, 1, 1])?.f_to_device(Cuda(0))?;//TODO device

                    debug!("FreqFilter::new: band_pass_config={band_pass_config:?} kernel.size={:?}", kernel.size());

                    filters.push(PreparedFilter::BandPass {
                        kernel: ThoTensor::new(
                            SynKey::new([
                                mono_key!(format!("{}", band_pass_config.low_hz), format!("{} low_hz", band_pass_config.low_hz)),
                                mono_key!(format!("{}", high_hz), format!("{} high_hz", high_hz) ),
                            ]),
                            kernel
                        )?,
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
                    let weights = Tensor::from_slice(&coeffs)
                        .f_to_kind(tch::Kind::Float)?
                        .f_to_device(Cuda(0))?//TODO device
                        .f_view([1, 1, window_len as i64])?
                        .f_repeat([num_channels, 1, 1])?;

                    filters.push(PreparedFilter::SavGol {
                        weights: ThoTensor::new(
                            SynKey::new([
                                mono_key!(format!("{}", sav_gol_config.window_len), "window_len"),
                                mono_key!(format!("{}", sav_gol_config.poly_order), "poly_order"),
                                mono_key!(format!("{:?}", sav_gol_config.padding), "padding"),
                            ]),
                            weights
                        )?,
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
        debug!("FreqFilter::apply_tho_tensor: self.filters={:?}", self.filters);

        let mut filtered_t = input_tensor.as_ref().shallow_clone();//.f_totype(Float)?;

        for filter in self.filters.iter() {
            match filter {
                PreparedFilter::NotchBand { low_hz, high_hz } => {
                    let filtered_t_out = notch_band_rfft(&filtered_t, self.sample_rate, *low_hz, *high_hz)?;

                    debug!("FreqFilter::apply_tho_tensor, NotchBand: id={} filtered_t.size={:?} filtered_t_out.size={:?}", input_tensor.id(), filtered_t.size(), filtered_t_out.size());

                    filtered_t = filtered_t_out;
                    // debug_plot!(&filtered_t, "{}_NotchBand", input_tensor.id())?;
                }
                PreparedFilter::BandPass { kernel, stride, padding } => {
                    let input_t = filtered_t.as_ref().f_unsqueeze(0)?;

                    let output = self.run_conv_with_padding(*stride, padding, &kernel.as_ref(), &input_t);
                    debug!("FreqFilter::apply_tho_tensor, BandPass: id={} kernel.size={:?} input.size={:?} output.size={:?}", input_tensor.id(), kernel.size(), input_t.size(), output.as_ref().map(|v| v.size()));
                    filtered_t = output?;

                    // debug_plot!(&filtered_t, "{}_BandPass", input_tensor.id())?;
                }
                PreparedFilter::SavGol { weights, window_len, stride, padding } => {
                    let t = filtered_t.size()[1];
                    if t < *window_len {
                        return Err(ThoError::new(format!("time length T={} must be >= window_len={}", t, window_len)));
                    }

                    let input_t = filtered_t.as_ref().f_unsqueeze(0)?;

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
            Padding::Zero(_) => input_t.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, pad, 1, self.num_channels)?,
            Padding::Replicate(_) => {
                let left = input_t.f_narrow(2, 0, 1)?.f_repeat([1, 1, pad])?;
                let right = input_t.f_narrow(2, t - 1, 1)?.f_repeat([1, 1, pad])?;
                let xpad = Tensor::f_cat(&[&left, input_t, &right], 2)?;
                xpad.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, 0, 1, self.num_channels)?
            }
            Padding::Reflect(_) => {
                if pad >= t {
                    return Err(ThoError::new(format!("reflect padding requires pad < T, got pad={} T={}", pad, t)));
                }
                let left = input_t.f_narrow(2, 1, pad)?.f_flip([2])?;
                let right = input_t.f_narrow(2, t - pad - 1, pad)?.f_flip([2])?;
                let xpad = Tensor::f_cat(&[&left, input_t, &right], 2)?;
                xpad.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, 0, 1, self.num_channels)?
            }
        };

        Ok(output_3d.f_view((self.num_channels, -1))?)
    }

}

impl <C> WithId for SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl SyncFreqFilter<RunCtxImpl> {
    pub(crate) fn new_for_ipc(init: SyncFreqFilterIpcInit) -> Result<Self, ThoError> {
        Ok(Self::new_local(
            &init.filter_configs,
            init.num_channels,
            init.sample_rate,
            init.find_motifs_run_config_id
        )?)
    }
}

impl<C> Adjustable for FreqFilterDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
}

impl<C> OutId for FreqFilterDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        match self {
            FreqFilterDynSyncStage::Local(local) => local.out_id(input_id),
            FreqFilterDynSyncStage::Proxy(proxy) => proxy.out_id(input_id),
        }
    }
}

impl<C> Stage for FreqFilterDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData, SynKey>;
    type Out = ThoTensor<ChannelsData, SynKey>;
}

impl<C> SyncStage for FreqFilterDynSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            FreqFilterDynSyncStage::Local(local) => Arc::clone(local).do_run(input, ctx),
            FreqFilterDynSyncStage::Proxy(proxy) => {
                let ctx = (ctx as &dyn Any).downcast_ref::<RunCtxImpl>().ok_or_else(|| ThoError::new("ProxySyncFreqFilter requires RunCtxImpl"))?;
                Arc::clone(proxy).do_run(input, ctx)
            }
        }
    }
}

#[derive(Debug)]
enum PreparedFilter {
    NotchBand { low_hz: f64, high_hz: f64 },
    BandPass {
        kernel: ThoTensor<BandPassDims, SynKey>,
        stride: u32,
        padding: Padding,
    },
    SavGol {
        weights: ThoTensor<BandPassDims, SynKey>,
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

#[derive(Debug, Cached, Named)]
pub struct FreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    filter_configs: Vec<InputFilter>,
    _ctx: PhantomData<C>,
    num_channels:i64,
    sample_rate:f64,
    sync_freq_filter: Arc<FreqFilterDynSyncStage<C>>,
    effective_output_stride:i64,
    min_input_split_len:i64
}

impl<C> FreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{


    pub async fn new(filter_configs: &[InputFilter], ws: &Arc<RwLock<Box<dyn Workspace>>>, find_motifs_run_config_id: RunConfigId, tho_device_idx: usize) -> Result<Self, ThoError> {
        let (num_channels, sample_rate)  = {
            let guard = thoosa_shared::perf_log!(thoosa_shared::sync_read_lock!(ws));
            (guard.channels() as i64, guard.sample_rate() as f64)
        };


        let effective_output_stride = Self::effective_output_stride(filter_configs)?;
        let min_input_split_len = Self::min_input_split_len(filter_configs)?;

        let sync_freq_filter= Arc::new(SyncFreqFilter::new(
            filter_configs, num_channels, sample_rate, find_motifs_run_config_id, tho_device_idx
        ).await?);

        Ok(Self {
            //avoids caching each run_config separately when they have same filter
            id: mono_key!(0, "todo").into(),
            filter_configs: filter_configs.to_vec(),
            _ctx: PhantomData,
            num_channels,
            sample_rate,
            sync_freq_filter,
            effective_output_stride,
            min_input_split_len
        })
    }



    fn run_conv_with_padding(&self, stride: u32, padding: &Padding, kernel_t: &Tensor, input_t: &Tensor) -> Result<Tensor, ThoError> {
        let t = input_t.size()[1];
        let kernel_size = kernel_t.size()[2];
        let pad = kernel_size / 2;
        let stride_i64 = stride as i64;


        let output_3d = match padding {
            Padding::Zero(_) => input_t.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, pad, 1, self.num_channels)?,
            Padding::Replicate(_) => {
                let left = input_t.f_narrow(2, 0, 1)?.f_repeat([1, 1, pad])?;
                let right = input_t.f_narrow(2, t - 1, 1)?.f_repeat([1, 1, pad])?;
                let xpad = Tensor::f_cat(&[&left, input_t, &right], 2)?;
                xpad.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, 0, 1, self.num_channels)?
            }
            Padding::Reflect(_) => {
                if pad >= t {
                    return Err(ThoError::new(format!("reflect padding requires pad < T, got pad={} T={}", pad, t)));
                }
                let left = input_t.f_narrow(2, 1, pad)?.f_flip([2])?;
                let right = input_t.f_narrow(2, t - pad - 1, pad)?.f_flip([2])?;
                let xpad = Tensor::f_cat(&[&left, input_t, &right], 2)?;
                xpad.f_conv1d(&kernel_t, None::<Tensor>, stride_i64, 0, 1, self.num_channels)?
            }
        };

        Ok(output_3d.f_view((self.num_channels, -1))?)
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
    error: Option<ThoError>,
}

impl FreqFilterOutputBuilder {
    fn new(output_id: SynKey) -> Self {
        Self {
            output_id,
            partials: vec![],
            error: None,
        }
    }
}

impl OutputBuilder for FreqFilterOutputBuilder {
    type O = ThoTensor<ChannelsData, SynKey>;

    async fn add_partial_output(&mut self, partial: Self::O) {
        if self.error.is_some() {
            return;
        }

        let partial = if partial.device() == 0 {
            partial
        } else {
            match track_await(partial.copy_to_device(0)).await {
                Ok(partial) => partial,
                Err(error) => {
                    self.error = Some(error);
                    return;
                }
            }
        };

        self.partials.push(partial);
    }

    async fn build(self) -> Result<Self::O, ThoError> {
        if let Some(error) = self.error {
            return Err(error);
        }

        let mut partials = self.partials.into_iter();
        let first = partials.next().ok_or_else(|| ThoError::new("FreqFilterOutputBuilder: no partial outputs"))?;
        let mut tensors = vec![first.t()];
        tensors.extend(partials.map(|partial| partial.t()));
        let tensor_refs = tensors.iter().collect::<Vec<_>>();

        ThoTensor::new(self.output_id, Tensor::f_cat(tensor_refs.as_slice(), 1)?)
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
    C: RunCtx + Adjustable + 'static,
{
}

impl<C> OutId for SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData, SynKey>;
    type Out = ThoTensor<ChannelsData, SynKey>;
}

impl<C> SyncStage for SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn do_run<'f>(self: Arc<Self>, input_tensor: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        self.apply_tho_tensor(input_tensor)
    }
}

impl<C> OutId for FreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for FreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;
    type Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;
}

// #[async_trait]
impl<C> AsyncStage for FreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{


    async fn run(
        self: Arc<Self>,
        input: Self::In<'static>,
        mut ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        let out_id = self.out_id(input.id());
        let input_size = input.size().clone();
        let min_split_len = [input_size.dim::<0>(), self.min_input_split_len].into();
        let required_ids: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
        // let sync_freq_filter = Arc::new(SyncFreqFilter::new(Arc::clone(&self), ctx.tho_device_idx()).await?);

        let filtered = track_await(OOMHandler::<'_, '_, _, NoTaskAdjuster, _, ChannelsDataSelector>::new(
            vec![
                Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::TWENTY))),
                Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("freq_filter"))),
                Adjusters::Input(input_adjuster(FreqFilterSplitsAdjuster::new(min_split_len, self.effective_output_stride))),
            ],
            None,
        )
        .run(
            &self.sync_freq_filter,
            input,
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


#[cfg(test)]
mod tests {
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use crate::tasks::stft::SyncStftSelector;
    use crate::tasks::transforms::freq_filter::{FreqFilter, SyncFreqFilter};
    use crate::tasks::transforms::smoothing::{median, moving_average};
    use log::debug;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::dims::ChannelsData;
    use thoosa_common::runner::common::cachable::{DevicesWrap, SelectRef};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::global_ctx::{GlobalCtx, mock_global_ctx};
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::transforms::filters::{AverageFilterParams, BandPassFilterParams, InputFilter, MedianFilterParams, Padding, SavGolFilterParams};
    use thoosa_common::utils::from;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_shared::ThoError;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::traits::{AsyncStage, RunCtx, SyncStage};

    #[tokio::test]
    async fn test_band_pass() -> Result<(), ThoError> {
        let filter_configs = vec![
            // InputFilter::NotchBandFilter(thoosa_common::transforms::filters::NotchBandFilterParams {
            //     low_hz: 58.0,
            //     high_hz: 62.0,
            // }),
            // InputFilter::SavGolFilter(SavGolFilterParams {
            //     window_len: 11,
            //     poly_order: 2,
            //     padding: SgPadding::reflect(),
            // }),
            InputFilter::BandPassFilter(BandPassFilterParams{
                low_hz: 0.0,
                band_hz: 100.0,
                kernel_size: 859,
                stride: 10,
                padding: Padding::zero(),
            }),
        ];


        run_freq_filter(filter_configs).await
    }

    #[tokio::test]
    async fn test_savitzky_golay() ->Result<(), ThoError> {

        let filter_configs = vec![

            InputFilter::SavGolFilter(SavGolFilterParams {
                window_len: 11,
                poly_order: 2,
                stride: 1,
                padding: Padding::reflect(),
            }),
        ];


        run_freq_filter(filter_configs).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_apply_tho_tensor_with_median_filter() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

        let input_t = Tensor::from_slice(&[1.0f32, 10.0, 2.0, 9.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0]).f_view([2, 5])?;
        let input = ThoTensor::new(SynKey::new([mono_key!("median", "median")]), input_t)?;

        let output = Arc::new(SyncFreqFilter::<RunCtxImpl>::new_local(&[InputFilter::MedianFilter(MedianFilterParams { window: 5 })], 2, 100.0, 1)?).do_run(&input, &RunCtxImpl::new(0, "median_filter"))?;
        let expected = median(input.as_ref(), 5)?;
        let output_flat = output.as_ref().f_view([-1])?;
        let expected_flat = expected.f_view([-1])?;

        assert_eq!(from::<Vec<f32>>(&output_flat), from::<Vec<f32>>(&expected_flat));

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_apply_tho_tensor_with_moving_average_filter() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

        let input_t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0]).f_view([2, 5])?;
        let input = ThoTensor::new(SynKey::new([mono_key!("moving_average", "moving_average")]), input_t)?;

        let output = Arc::new(SyncFreqFilter::<RunCtxImpl>::new_local(&[InputFilter::AverageFilter(AverageFilterParams { window: 4, step: 1 })], 2, 100.0, 0 )?).do_run(&input, &RunCtxImpl::new(0, "moving_average_filter"))?;
        let expected = moving_average(input.as_ref(), 4, 1)?;

        assert!(output.as_ref().allclose(&expected, 1e-5, 1e-5, false));

        Ok(())
    }


    async fn run_freq_filter(filter_configs: Vec<InputFilter>) -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 20, 100, 10000)));

        let bf = FreqFilter::new(&filter_configs, &ws, 0, 0).await?;

        let mut ctx = RunCtxImpl::new(1, "test");

        let size = [20, 12800];
        let r = Arc::new(bf).run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test", "test")]), Tensor::rand(&size, FLOAT_CPU))?)),
            ctx.branch(false)
        ).await?;

        let find_motifs_run_config = FindMotifsRunConfig::new_with_running_mean(None, 10, 100, Some(filter_configs), None);

        let adjuster = find_motifs_run_config.input_len_adjuster();


        {
            let r = r.borrow_to(ctx.tho_device_idx(), None::<SyncStftSelector>).await?;
            debug!("filter r={r:?}");
            assert_eq!(r.t_ref().size()[0], size[0]);
            assert_eq!(r.t_ref().size()[1] as u32, adjuster(size[1] as u32));
            Ok::<_, ThoError>(())
        }?;

        Ok(())
    }
}
