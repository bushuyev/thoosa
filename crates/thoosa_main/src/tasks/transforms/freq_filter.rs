use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::{Device, Kind, Tensor};
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MaxParallelAndDumpToCpuAdjustableContex, MAX_PARALLEL};
use thoosa_common::runner::adjusters::splits::{OutputBuilder, SplitsAdjusterContext, SplitsAdjusterContextImpl};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::common::OutId;
use thoosa_common::runner::common::RunCtx;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, NoopRunnableContext, OOMHandler};

use thoosa_common::dims::{AnyDims, AnyMetrics, Dims};
use thoosa_common::find_motifs_run_config::InputFilter;

use thoosa_common::transforms::filters::{band_pass_filter, notch_band_rfft, Padding};
use thoosa_shared::{mono_key, CopyToDevice, ThoError};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Metrics, MonoMetrics, WithId};
use thoosa_shared::comp_id::SynKey;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::z_norm_2;
use thoosa_common::ws::Workspace;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::math::sg_smoothing_coeffs;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsData2Selector};
use crate::tasks::stage_ids::FREQ_FILTER;
use crate::tasks::transforms::smoothing::{median, moving_average};

#[derive(Debug, Clone)]
struct BandPassDims;

impl Dims for BandPassDims {
    type Metrics = AnyMetrics;//TODO
}

#[derive(Debug)]
struct SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    freq_filter: Arc<FreqFilter<C>>,
}

impl<C> SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    //TODO remove freq_filter: Arc<FreqFilter<C>> param: move apply_tho_tensor and out_id to SyncFreqFilter and keep AsyncStage->SyncStage
    fn new(freq_filter: Arc<FreqFilter<C>>) -> Self {
        Self { freq_filter }
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
    Abs {
        // hann: ThoTensor<AnyDims, SynKey>
    }
}

#[derive(Debug, Cached, Named)]
pub struct FreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    filters: Vec<PreparedFilter>,
    _ctx: PhantomData<C>,
    num_channels:i64,
    sample_rate:f64
}

impl<C> FreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(filter_configs: &[InputFilter], ws: &Arc<RwLock<Box<dyn Workspace>>>, find_motifs_run_config_id: RunId) -> Result<Self, ThoError> {
        let (num_channels, sample_rate)  = {
            let guard = thoosa_shared::with_perf_log(
                || ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                "FreqFilter::new/ws_read",
            );
            (guard.channels() as i64, guard.sample_rate() as f64)
        };

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
                InputFilter::Abs(abs) => {
                    filters.push(PreparedFilter::Abs{
                        // hann: Tensor::f_hann_window(sample_rate as i64, (Kind::Float, Cuda(0)))?.into()//TODO view to window size
                    })
                }
            }
        }

        Ok(Self {
            id: SynKey::new([
                mono_key!(FREQ_FILTER, "FreqFilter"),
                mono_key!(find_motifs_run_config_id, format!("find_motifs_run_config_id={find_motifs_run_config_id}")),
            ]),
            filters,
            _ctx: PhantomData,
            num_channels,
            sample_rate
        })
    }

    pub fn input_device(&self) -> Device {
        self.filters
            .iter()
            .find_map(|filter| match filter {
                PreparedFilter::BandPass { kernel, .. } => Some(kernel.as_ref().device()),
                PreparedFilter::SavGol { weights, .. } => Some(weights.as_ref().device()),
                _ => None,
            })
            .unwrap_or(Cpu)
    }

    pub fn apply_tho_tensor(&self, input_tensor: &ThoTensor<ChannelsData2, SynKey>) -> Result<ThoTensor<ChannelsData2, SynKey>, ThoError> {
        let mut filtered_t = input_tensor.t_ref().totype(Float);
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
                PreparedFilter::Abs{..} => {
                    filtered_t = z_norm_2(&filtered_t, 1)?; //.f_abs()?;
                    filtered_t = filtered_t.f_fft_rfft(None, -1, "backward")?.f_abs()?;
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

    fn effective_output_stride(&self) -> Result<i64, ThoError> {
        self.filters.iter().try_fold(1_i64, |effective_stride, filter| {
            let stage_stride = match filter {
                PreparedFilter::BandPass { stride, .. } | PreparedFilter::SavGol { stride, .. } => *stride as i64,
                PreparedFilter::MovingAverage { step, .. } => *step as i64,
                PreparedFilter::NotchBand { .. } | PreparedFilter::Median { .. } => 1,
                PreparedFilter::Abs{..} => {1}
            };

            effective_stride
                .checked_mul(stage_stride)
                .ok_or_else(|| ThoError::new(format!("FreqFilter::effective_output_stride overflow: effective_stride={effective_stride} stage_stride={stage_stride}")))
        })
    }

    fn min_input_split_len(&self) -> Result<i64, ThoError> {
        self.filters.iter().rev().try_fold(1_i64, |required_len, filter| {
            match filter {
                PreparedFilter::NotchBand { .. } | PreparedFilter::Median { .. } => Ok(required_len.max(1)),
                PreparedFilter::MovingAverage { step, .. } => Self::expand_required_len(required_len, *step as i64, 1),
                PreparedFilter::BandPass { kernel, stride, padding } => {
                    let local_min = match padding {
                        Padding::Reflect(_) => (kernel.size()[2] / 2) + 1,
                        Padding::Zero(_) | Padding::Replicate(_) => 1,
                    };
                    Self::expand_required_len(required_len, *stride as i64, local_min)
                }
                PreparedFilter::SavGol { window_len, stride, .. } => Self::expand_required_len(required_len, *stride as i64, *window_len),
                PreparedFilter::Abs{..} => {Ok(1)}
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
    partials: Vec<ThoTensor<ChannelsData2, SynKey>>,
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
    type O = ThoTensor<ChannelsData2, SynKey>;

    async fn add_partial_output(&mut self, partial: Self::O) {
        if self.error.is_some() {
            return;
        }

        let partial = if partial.device() == 0 {
            partial
        } else {
            match partial.copy_to_device(0).await {
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
        let mut tensors = vec![first.t];
        tensors.extend(partials.map(|partial| partial.t));
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
    type A = ChannelsData2Selector;

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
        (*adjust_context.split_len_mut()) /= 2;
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
        self.freq_filter.out_id(input_id)
    }
}

impl<C> Stage for SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData2, SynKey>;
    type Out = ThoTensor<ChannelsData2, SynKey>;
}

impl<C> SyncStage for SyncFreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn do_run<'f>(self: Arc<Self>, input_tensor: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        self.freq_filter.apply_tho_tensor(input_tensor)
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
    type In<'a> = ThoTensor<ChannelsData2, SynKey>;
    type Out = ThoTensor<ChannelsData2, SynKey>;
}

// #[async_trait]
impl<C> AsyncStage for FreqFilter<C>
where
    C: RunCtx + Adjustable + 'static,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let out_id = self.out_id(input.id());
        let input_size = input.size().clone();
        let min_split_len = [input_size.dim::<0>(), self.min_input_split_len()?].into();
        let required_ids: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
        let sync_freq_filter = Arc::new(SyncFreqFilter::new(Arc::clone(&self)));

        let filtered = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::TWENTY))),
                Adjusters::RunContext(Box::new(DumpToCpu::new("freq_filter"))),
                Adjusters::Input(Box::new(FreqFilterSplitsAdjuster::new(min_split_len, self.effective_output_stride()?))),
            ],
            None,
        )
        .run(
            &sync_freq_filter,
            input,
            ctx.branch(Some("freq_filter"), true),
            Contexts::new(
                SplitsAdjusterContextImpl::new(input_size, FreqFilterOutputBuilder::new(out_id)),
                NoopRunnableContext {},
                MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
            ),
        )
        .await?;

        debug!("FreqFilter::run: filtered={filtered:?} done");

        Ok(Arc::new(DevicesWrap::wrap(filtered)))
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use log::debug;
    use parking_lot::RwLock;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::runnable::{AsyncStage, SyncStage};
    use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, InputFilter};
    use thoosa_common::transforms::filters::{BandPassFilterParams, MedianFilterParams, AverageFilterParams, Padding, SavGolFilterParams};
    use thoosa_shared::ThoError;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::from;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use crate::tasks::transforms::smoothing::{median, moving_average};
    use crate::tasks::transforms::freq_filter::{FreqFilter, SyncFreqFilter};
    use thoosa_common::runner::common::RunCtx;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_with_data::with_data;

    #[tokio::test]
    async fn test_band_pass() ->Result<(), ThoError> {

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

    #[test]
    fn test_apply_tho_tensor_with_median_filter() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
        let freq_filter = Arc::new(FreqFilter::<RunCtxImpl>::new(&[InputFilter::MedianFilter(MedianFilterParams { window: 5 })], &ws, 0)?);
        let input_t = Tensor::from_slice(&[1.0f32, 10.0, 2.0, 9.0, 3.0, 5.0, 4.0, 3.0, 2.0, 1.0]).f_view([2, 5])?;
        let input = ThoTensor::new(SynKey::new([mono_key!("median", "median")]), input_t)?;

        let output = Arc::new(SyncFreqFilter::new(Arc::clone(&freq_filter))).do_run(&input, &RunCtxImpl::new(0, "median_filter"))?;
        let expected = median(input.as_ref(), 5)?;
        let output_flat = output.as_ref().f_view([-1])?;
        let expected_flat = expected.f_view([-1])?;

        assert_eq!(from::<Vec<f32>>(&output_flat), from::<Vec<f32>>(&expected_flat));

        Ok(())
    }

    #[test]
    fn test_apply_tho_tensor_with_moving_average_filter() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
        let freq_filter = Arc::new(FreqFilter::<RunCtxImpl>::new(&[InputFilter::AverageFilter(AverageFilterParams { window: 4, step: 1 })], &ws, 0)?);
        let input_t = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 5.0, 4.0, 3.0, 2.0, 1.0]).f_view([2, 5])?;
        let input = ThoTensor::new(SynKey::new([mono_key!("moving_average", "moving_average")]), input_t)?;

        let output = Arc::new(SyncFreqFilter::new(Arc::clone(&freq_filter))).do_run(&input, &RunCtxImpl::new(0, "moving_average_filter"))?;
        let expected = moving_average(input.as_ref(), 4, 1)?;

        assert!(output.as_ref().allclose(&expected, 1e-5, 1e-5, false));

        Ok(())
    }


    async fn run_freq_filter(filter_configs: Vec<InputFilter>) -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        unsafe { mock_global_ctx::USE_REAL_CTX = true; }
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();
        
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 20, 100, 10000)));

        let bf = FreqFilter::new(&filter_configs, &ws, 0)?;

        let mut ctx = RunCtxImpl::new(1, "test");

        let size = [20, 12800];
        let r = Arc::new(bf).run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test", "test")]), Tensor::rand(&size, FLOAT_CPU))?)),
            ctx.branch(Some("freq_filter_test"), false)
        ).await?;

        let find_motifs_run_config = FindMotifsRunConfig::new_with_running_mean(None, 10, 100, Some(filter_configs), None);

        let adjuster = find_motifs_run_config.input_len_adjuster();


        with_data!(ctx.branch(Some("freq_filter_test"), false), r; |ctx, r|{
            debug!("filter r={r:?}");
            Ok::<_, ThoError>(())
        })?;


        // assert_eq!(adjuster(size[1] as u32), r.size().dim_1() as u32);
        //TODO asserts

        Ok(())
    }
}
