use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use cached_derive::Cached;
use log::debug;
use parking_lot::RwLock;
use rand::prelude::{IndexedRandom, SliceRandom, StdRng};
use tch::{IndexOp, Tensor};
use thoosa_common::dims::{Dims, MotifSlices};
use thoosa_common::tho_device::ThoDevice;
use crate::tasks::find_motifs_run_config::{FindMotifsRunConfig, Metrics};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_dummy_data_derive::DummyData;
use thoosa_common::ws::Workspace;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::RunId;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::cachable::{Cached, ConsumeData, DevicesWrap};
use thoosa_common::runner::common::runnable::{AsyncStage, ConsumeResult, Stage, SyncStage};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::runner::tmp_cache::TmpCached;
use thoosa_shared::traits::{Adjustable, Data, DataId, MonoMetrics, Select, WithId};
use thoosa_shared::{mono_key, CopyToDevice, TensorIdSliceIdx, ThoError};
use thoosa_common::copy_ctx;
use thoosa_store::Store;
use thoosa_with_data::with_data;
use tracing::error;
use thoosa_common::cuda_utils::with_pooled_cuda_stream;
use thoosa_common::runs::Simple;
use thoosa_common::transforms::filters::InputFilterLengthAdjuster;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists, ChannelsMotifsSlices};
use crate::tasks::motif_loader::{load_random_slices, MotifLoaderInput, MotifMeta};
use crate::tasks::slices_stats::{metric_quantiles_for_motif, metric_scalar_for_motif};
use crate::tasks::stage_ids::{MOTIF_STATS, MOTIF_STATS_SYNC_STAGE};

pub const RANDOM_STATS_GROUP: &str = "random_avg_dist"; //dummy group to load random resources, TODO - add proper method to SqlStore

#[derive(Debug, Clone)]
pub enum MotifStatsSource {
    Motif(MotifLoaderInput),
    MergedMotifs(Vec<MotifLoaderInput>),
    Random { max_slices: u32, find_motifs_run_id: RunId },
}

#[derive(Debug, Clone)]
pub struct MotifStatsInput {
    id: SynKey,
    source: MotifStatsSource,
    metrics: Vec<Metrics>,
}

impl MotifStatsInput {
    pub fn from_motif_input(
        motif_loader_input: MotifLoaderInput,
        metrics: Vec<Metrics>,
    ) -> Self {
        let metrics_sfx = metrics.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>().join(",");

        Self {
            id: &motif_loader_input.id + mono_key!(metrics_sfx.clone(), format!("metrics={metrics_sfx}")),
            source: MotifStatsSource::Motif(motif_loader_input),
            metrics,
        }
    }

    pub fn from_random(max_slices: u32, find_motifs_run_id: RunId, metrics: Vec<Metrics>) -> Self {
        let metrics_sfx = metrics.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>().join(",");
        let mut id = SynKey::new([
            mono_key!("MotifStatsInput", "stage"),
            mono_key!(find_motifs_run_id, format!("find_motifs_run_id={find_motifs_run_id}")),
            mono_key!(max_slices, format!("max_slices={max_slices}")),
            mono_key!(metrics_sfx.clone(), format!("metrics={metrics_sfx}")),
        ]);


        debug!("MotifStatsInput::from_random: id={id:?}");

        Self {
            id,
            source: MotifStatsSource::Random { max_slices, find_motifs_run_id },
            metrics,
        }
    }

    pub fn from_merged_motif_inputs(
        motif_loader_inputs: Vec<MotifLoaderInput>,
        metrics: Vec<Metrics>,
    ) -> Result<Self, ThoError> {
        if motif_loader_inputs.is_empty() {
            return Err(ThoError::new("MotifStatsInput::from_merged_motif_inputs: empty motif_loader_inputs"));
        }

        let metrics_sfx = metrics.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>().join(",");
        let id = motif_loader_inputs.iter().fold(SynKey::new([mono_key!("MotifStatsMergedInput", "stage")]), |acc, motif_loader_input| {
            &acc + motif_loader_input.id()
        });

        Ok(Self {
            id: &id + mono_key!(metrics_sfx.clone(), format!("metrics={metrics_sfx}")),
            source: MotifStatsSource::MergedMotifs(motif_loader_inputs),
            metrics,
        })
    }


    pub fn source(&self) -> &MotifStatsSource {
        &self.source
    }

    pub fn metrics(&self) -> &Vec<Metrics> {
        &self.metrics
    }

}

impl Input for MotifStatsInput {}

impl WithId for MotifStatsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for MotifStatsInput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        1usize.into()
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0
    }
}

impl Select<()> for MotifStatsInput {
    type Output = MotifStatsInput;

    fn subview(&self, _selector: ()) -> Result<Self::Output, ThoError> {
        Ok(self.clone())
    }
}

impl Adjustable for MotifStatsInput {}

#[async_trait]
impl CopyToDevice for MotifStatsInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        let source = match &self.source {
            MotifStatsSource::Motif(motif_loader_input) => MotifStatsSource::Motif(motif_loader_input.clone()),
            MotifStatsSource::MergedMotifs(motif_loader_inputs) => MotifStatsSource::MergedMotifs(motif_loader_inputs.clone()),
            MotifStatsSource::Random { max_slices, find_motifs_run_id } => MotifStatsSource::Random { max_slices: *max_slices, find_motifs_run_id: *find_motifs_run_id },
        };

        Ok(Self {
            id: self.id.clone(),
            source,
            metrics: self.metrics.clone(),
        })
    }
}

type LoadedMotif = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>;

#[derive(Debug)]
pub struct MotifStatsCalcInput<'a> {
    id: SynKey,
    data: MotifStatsCalcInputData<'a>,
    metrics: Vec<Metrics>,
    quantiles: Option<Vec<f32>>,
}

#[derive(Debug)]
enum MotifStatsCalcInputData<'a> {
    Managed(&'a LoadedMotif),
    Motifs(Vec<&'a LoadedMotif>),
}

impl<'a> MotifStatsCalcInput<'a> {
    fn from_managed(id: SynKey, motif: &'a LoadedMotif, metrics: Vec<Metrics>, quantiles: Option<Vec<f32>>) -> Self {
        Self { id, data: MotifStatsCalcInputData::Managed(motif), metrics, quantiles }
    }

    fn from_motifs(id: SynKey, motifs: Vec<&'a LoadedMotif>, metrics: Vec<Metrics>, quantiles: Option<Vec<f32>>) -> Self {
        Self { id, data: MotifStatsCalcInputData::Motifs(motifs), metrics, quantiles }
    }

    fn data(&self) -> &MotifStatsCalcInputData<'a> {
        &self.data
    }

    pub fn metrics(&self) -> &Vec<Metrics> {
        &self.metrics
    }

    pub fn quantiles(&self) -> Option<&[f32]> {
        self.quantiles.as_deref()
    }

    fn channels_size_from_tensor(size: &[i64]) -> <ThoTensor<ChannelsMotifsSlices, SynKey> as Data>::Metrics {
        match size {
            [num_slices, window] => [1, *num_slices, *window].into(),
            [channels, num_slices, window] => [*channels, *num_slices, *window].into(),
            _ => panic!("MotifStatsCalcInput::channels_size_from_tensor: expected 2D or 3D tensor, got {size:?}"),
        }
    }

    fn motifs_size(motifs: &[&LoadedMotif]) -> <ThoTensor<ChannelsMotifsSlices, SynKey> as Data>::Metrics {
        let num_slices = motifs.iter().map(|motif| motif.data().t_ref().size()[0]).sum::<i64>();
        let window = motifs.first().map(|motif| motif.data().t_ref().size()[1]).unwrap_or(0);
        [1, num_slices, window].into()
    }

}

impl Input for MotifStatsCalcInput<'_> {}

impl WithId for MotifStatsCalcInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for MotifStatsCalcInput<'_> {
    type Metrics = <ThoTensor<ChannelsMotifsSlices, SynKey> as Data>::Metrics;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        match self.data() {
            MotifStatsCalcInputData::Managed(motif) => Self::channels_size_from_tensor(motif.data().size().as_slice()),
            MotifStatsCalcInputData::Motifs(motifs) => Self::motifs_size(motifs.as_slice()),
        }
    }

    fn device(&self) -> usize {
        match self.data() {
            MotifStatsCalcInputData::Managed(motif) => motif.data().device(),
            MotifStatsCalcInputData::Motifs(motifs) => motifs.first().map(|motif| motif.data().device()).unwrap_or(0),
        }
    }

    fn mem_size(&self) -> u64 {
        match self.data() {
            MotifStatsCalcInputData::Managed(motif) => motif.data().mem_size(),
            MotifStatsCalcInputData::Motifs(motifs) => motifs.iter().map(|motif| motif.data().mem_size()).sum(),
        }
    }
}

#[derive(Debug)]
enum MotifStatsCalcInputDataWrapper {
    Managed(Arc<DevicesWrap<LoadedMotif>>),
    Motifs(Vec<Arc<DevicesWrap<LoadedMotif>>>),
}

#[derive(Debug)]
struct MotifStatsCalcInputWrapper {
    id: SynKey,
    data: MotifStatsCalcInputDataWrapper,
    metrics: Vec<Metrics>,
    quantiles: Option<Vec<f32>>,
}

impl MotifStatsCalcInputWrapper {
    fn from_managed(id: SynKey, motif: Arc<DevicesWrap<LoadedMotif>>, metrics: Vec<Metrics>, quantiles: Option<Vec<f32>>) -> Self {
        Self { id, data: MotifStatsCalcInputDataWrapper::Managed(motif), metrics, quantiles }
    }

    fn from_motifs(id: SynKey, motifs: Vec<Arc<DevicesWrap<LoadedMotif>>>, metrics: Vec<Metrics>, quantiles: Option<Vec<f32>>) -> Self {
        Self { id, data: MotifStatsCalcInputDataWrapper::Motifs(motifs), metrics, quantiles }
    }

    async fn with_calc_input<R, F, P>(&self, permits_ctx: P, do_run: F) -> Result<R, ThoError>
    where
        for<'f> F: FnOnce(&MotifStatsCalcInput<'f>, &P) -> Result<R, ThoError>,
        P: RunCtx,
    {
        let id = self.id.clone();
        let metrics = self.metrics.clone();
        let quantiles = self.quantiles.clone();

        match &self.data {
            MotifStatsCalcInputDataWrapper::Managed(motif) => {
                with_data!(permits_ctx, motif; move |ctx: P, motif: &LoadedMotif| {
                    let input = MotifStatsCalcInput::from_managed(id, motif, metrics, quantiles);
                    do_run(&input, &ctx)
                })
            }
            MotifStatsCalcInputDataWrapper::Motifs(motifs) => {
                with_data!(permits_ctx, [motifs]; move |ctx: P, motifs: &[&LoadedMotif]| {
                    let input = MotifStatsCalcInput::from_motifs(id, motifs.to_vec(), metrics, quantiles);
                    do_run(&input, &ctx)
                })
            }
        }
    }
}

impl WithId for MotifStatsCalcInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl thoosa_common::runner::common::cachable::WithDataRef for MotifStatsCalcInputWrapper {
    type O<'o> = MotifStatsCalcInput<'o>;

    fn with_data_ref<'a, R: Send + 'a, F, P>(&'a self, _to_device_idx: usize, permits_ctx: P, _trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move { self.with_calc_input(permits_ctx, do_run).await }
    }

    fn mem_size(&self) -> u64 {
        match &self.data {
            MotifStatsCalcInputDataWrapper::Managed(motif) => motif.mem_size(),
            MotifStatsCalcInputDataWrapper::Motifs(motifs) => motifs.iter().map(|motif| motif.mem_size()).sum(),
        }
    }
}

impl thoosa_common::runner::common::cachable::WithSelectDataRef<()> for MotifStatsCalcInputWrapper {
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(&'a self, _to_device_idx: usize, permits_ctx: P, _selector: (), _trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move { self.with_calc_input(permits_ctx, do_run).await }
    }
}

#[derive(Debug)]
struct MotifStatsLoadedChannelsInput {
    id: SynKey,
    channels_motifs_slices: Vec<ThoTensor<ChannelsMotifsSlices, SynKey>>,
    metrics: Vec<Metrics>,
    quantiles: Option<Vec<f32>>,
}

impl MotifStatsLoadedChannelsInput {
    fn new(
        id: SynKey,
        channels_motifs_slices: Vec<ThoTensor<ChannelsMotifsSlices, SynKey>>,
        metrics: Vec<Metrics>,
        quantiles: Option<Vec<f32>>,
    ) -> Self {
        Self { id, channels_motifs_slices, metrics, quantiles }
    }

    fn channels_motifs_slices(&self) -> &[ThoTensor<ChannelsMotifsSlices, SynKey>] {
        self.channels_motifs_slices.as_slice()
    }

    fn metrics(&self) -> &Vec<Metrics> {
        &self.metrics
    }

    fn quantiles(&self) -> Option<&[f32]> {
        self.quantiles.as_deref()
    }
}

impl Input for MotifStatsLoadedChannelsInput {}

impl WithId for MotifStatsLoadedChannelsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for MotifStatsLoadedChannelsInput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        1usize.into()
    }

    fn device(&self) -> usize {
        self.channels_motifs_slices.first().map(|tensor| tensor.device()).unwrap_or(0)
    }

    fn mem_size(&self) -> u64 {
        self.channels_motifs_slices.iter().map(|tensor| tensor.mem_size()).sum()
    }
}

#[async_trait]
impl CopyToDevice for MotifStatsLoadedChannelsInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        let mut channels_motifs_slices = Vec::with_capacity(self.channels_motifs_slices.len());
        for channels in &self.channels_motifs_slices {
            channels_motifs_slices.push(channels.copy_to_device(dst_device).await?);
        }
        Ok(Self {
            id: self.id.clone(),
            channels_motifs_slices,
            metrics: self.metrics.clone(),
            quantiles: self.quantiles.clone(),
        })
    }
}

#[derive(Debug)]
struct MotifStatsTmpCachedChannelsInput {
    id: SynKey,
    channels_motifs_slices: TmpCached<ThoTensor<ChannelsMotifsSlices, SynKey>>,
    metrics: Vec<Metrics>,
    quantiles: Option<Vec<f32>>,
}

impl MotifStatsTmpCachedChannelsInput {
    fn new(
        id: SynKey,
        channels_motifs_slices: TmpCached<ThoTensor<ChannelsMotifsSlices, SynKey>>,
        metrics: Vec<Metrics>,
        quantiles: Option<Vec<f32>>,
    ) -> Self {
        Self { id, channels_motifs_slices, metrics, quantiles }
    }
}

impl WithId for MotifStatsTmpCachedChannelsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl ConsumeData for MotifStatsTmpCachedChannelsInput {
    type O = MotifStatsLoadedChannelsInput;

    fn consume<R: Send, F, P>(
        self,
        to_device_idx: usize,
        ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl std::future::Future<Output = ConsumeResult<Self::O, R>> + Send
    where
        F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
        P: RunCtx,
    {
        async move {
            let MotifStatsTmpCachedChannelsInput {
                id,
                channels_motifs_slices,
                metrics,
                quantiles,
            } = self;

            let channels_motifs_slices = match channels_motifs_slices.take_all_recovering(to_device_idx).await {
                Ok(channels_motifs_slices) => channels_motifs_slices,
                Err(err) => {
                    return ConsumeResult::Failed {
                        input: MotifStatsLoadedChannelsInput::new(id, err.data, metrics, quantiles),
                        error: err.error,
                    };
                }
            };

            let input = MotifStatsLoadedChannelsInput::new(id, channels_motifs_slices, metrics, quantiles);

            if to_device_idx == 0 {
                do_run(input, ctx)
            } else {
                let mut r = None;
                let stream_result = with_pooled_cuda_stream(to_device_idx, |_| {
                    r = Some(do_run(input, ctx));
                    Ok(())
                }).await;
                if let Some(r) = r {
                    r
                } else {
                    stream_result.expect("with_pooled_cuda_stream failed before do_run");
                    unreachable!("with_pooled_cuda_stream returned without running do_run")
                }
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStatsTmpCachedChannelsInput::burp"))]
    fn burp(o: Self::O) -> impl std::future::Future<Output = Self> + Send {
        async move {
            let cache = TmpCached::new("tmp_motif_stats_channels_burp").await;
            for channels_motifs_slices in o.channels_motifs_slices {
                cache.insert(channels_motifs_slices);
            }
            MotifStatsTmpCachedChannelsInput::new(o.id, cache, o.metrics, o.quantiles)
        }
    }
}

fn motif_stats_stage_id() -> SynKey {
    (MOTIF_STATS_SYNC_STAGE, "MotifStatsSyncStage").into()
}

fn merged_motifs_tensor(input_id: &SynKey, motifs: &[&LoadedMotif]) -> Result<ThoTensor<ChannelsMotifsSlices, SynKey>, ThoError> {
    if motifs.is_empty() {
        return Err(ThoError::new("MotifStatsSyncStage::merged_motifs_tensor: empty motifs"));
    }

    let motifs_refs = motifs.iter().map(|motif| motif.data().t_ref()).collect::<Vec<_>>();
    ThoTensor::new(input_id.clone(), Tensor::f_cat(motifs_refs.as_slice(), 0)?)
}

fn merged_channels_tensor(
    input_id: &SynKey,
    channels_motifs_slices: &[ThoTensor<ChannelsMotifsSlices, SynKey>],
) -> Result<ThoTensor<ChannelsMotifsSlices, SynKey>, ThoError> {
    if channels_motifs_slices.is_empty() {
        return Err(ThoError::new("MotifStatsSyncStage::merged_channels_tensor: empty channels_motifs_slices"));
    }

    let channels_motifs_slices_refs = channels_motifs_slices.iter().map(|slices| slices.t_ref()).collect::<Vec<_>>();
    ThoTensor::new(input_id.clone(), Tensor::f_cat(channels_motifs_slices_refs.as_slice(), 1)?)
}

fn calc_metrics(
    out_id: SynKey,
    input_id: &SynKey,
    metrics: &[Metrics],
    quantiles: Option<&[f32]>,
    channels_motifs_slices: &Tensor,
) -> Result<MotifStatsOutput, ThoError> {
    let size = channels_motifs_slices.size();
    let (channels, num_slices, is_flat_motif) = match size.as_slice() {
        [num_slices, _window] => (1i64, *num_slices, true),
        [channels, num_slices, _window] => (*channels, *num_slices, false),
        _ => {
            return Err(ThoError::new(format!(
                "MotifStatsSyncStage::calc_metrics: expected [slices x window] or [channels x slices x window], got {:?}",
                size
            )));
        }
    };

    let mut channels_values = Vec::with_capacity(metrics.len());

    for metric in metrics {
        let mut metric_channels_values = Vec::with_capacity(channels as usize);
        for channel_idx in 0..channels as usize {
            let motif_slices = if is_flat_motif {
                channels_motifs_slices.f_i((.., ..))?
            } else {
                channels_motifs_slices.f_get(channel_idx as i64)?
            };
            let value = metric_scalar_for_motif(input_id, &motif_slices, metric)?;
            let quantiles = match quantiles {
                Some(quantiles) => metric_quantiles_for_motif(&motif_slices, metric, quantiles)?,
                None => None,
            };
            metric_channels_values.push(MetricValue::new(value, quantiles));
        }
        channels_values.push(MetricsValue::new(metric.clone(), metric_channels_values));
    }

    Ok(MotifStatsOutput::new(out_id, channels_values, num_slices as u32))
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricValue {
    value: f32,
    quantiles: Option<Vec<(f32, f32)>>,
}

impl MetricValue {
    pub fn new(value: f32, quantiles: Option<Vec<(f32, f32)>>) -> Self {
        Self { value, quantiles }
    }

    pub fn value(&self) -> f32 {
        self.value
    }

    pub fn quantiles(&self) -> Option<&Vec<(f32, f32)>> {
        self.quantiles.as_ref()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricsValue {
    metric: Metrics,
    channels_values: Vec<MetricValue>,
}

impl MetricsValue {
    pub fn new(metric: Metrics, channels_values: Vec<MetricValue>) -> Self {
        Self { metric, channels_values }
    }

    pub fn metric(&self) -> &Metrics {
        &self.metric
    }

    pub fn channels_values(&self) -> &Vec<MetricValue> {
        &self.channels_values
    }
}

#[derive(Debug, Clone)]
pub struct MotifStatsOutput {
    id: SynKey,
    channels_values: Vec<MetricsValue>,
    pub num_slices: u32,
}

impl MotifStatsOutput {
    pub fn new(id: SynKey, channels_values: Vec<MetricsValue>, num_slices: u32) -> Self {
        Self { id, channels_values, num_slices }
    }

    pub fn metrics(&self) -> Vec<Metrics> {
        self.channels_values.iter().map(|values| values.metric().clone()).collect()
    }

    fn metric_values(&self, metric: &Metrics) -> Option<&Vec<MetricValue>> {
        self.channels_values.iter().find_map(|values| if values.metric() == metric { Some(values.channels_values()) } else { None })
    }

    fn metric_values_id(&self, metric: &Metrics) -> SynKey {
        let metric_key = mono_key!(format!("{metric:?}"), format!("metric={metric:?}"));
        &self.id + &metric_key
    }

    pub fn channel_metric_value(&self, metric: &Metrics, channel: usize) -> Result<&MetricValue, ThoError> {
        let values = self.metric_values(metric).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: metric {metric:?} was not requested")))?;
        values.get(channel).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: channel index out of bounds: channel={channel} channels={}", values.len())))
    }

    pub fn channel_value(&self, metric: &Metrics, channel: usize) -> Result<f32, ThoError> {
        Ok(self.channel_metric_value(metric, channel)?.value())
    }

    pub fn channels_values(&self, metric: &Metrics) -> Result<Vec<MetricValue>, ThoError> {
        let values = self.metric_values(metric).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: metric {metric:?} was not requested")))?;
        Ok(values.clone())
    }

    pub fn channels_motifs_dists(&self, metric: &Metrics) -> Result<ThoTensor<ChannelsMotifsDists, SynKey>, ThoError> {
        let values = self.metric_values(metric).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: metric {metric:?} was not requested")))?;
        let scalar_values = values.iter().map(|value| value.value()).collect::<Vec<_>>();
        let t = Tensor::f_from_slice(scalar_values.as_slice())?.f_view([scalar_values.len() as i64, 1])?;
        ThoTensor::new(self.metric_values_id(metric), t)
    }
}

impl Data for MotifStatsOutput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        1usize.into()
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        self.channels_values
            .iter()
            .map(|values| {
                values
                    .channels_values()
                    .iter()
                    .map(|value| {
                        let quantiles_len = value.quantiles().map(|quantiles| quantiles.len()).unwrap_or(0);
                        std::mem::size_of::<MetricValue>() as u64 + (quantiles_len as u64 * std::mem::size_of::<(f32, f32)>() as u64)
                    })
                    .sum::<u64>()
            })
            .sum()
    }
}

impl WithId for MotifStatsOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for MotifStatsOutput {}

#[async_trait]
impl CopyToDevice for MotifStatsOutput {
    async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
        Ok(self.clone())
    }
}

#[derive(Debug, Named)]
struct MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    _ctx: PhantomData<C>,
}

impl<C> MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn new() -> Self {
        Self { id: motif_stats_stage_id(), _ctx: PhantomData }
    }

    fn run_input(&self, input: &MotifStatsCalcInput<'_>) -> Result<MotifStatsOutput, ThoError> {
        match input.data() {
            MotifStatsCalcInputData::Managed(motif) => calc_metrics(self.out_id(input.id()), input.id(), input.metrics(), input.quantiles(), motif.data().t_ref()),
            MotifStatsCalcInputData::Motifs(motifs) => {
                let channels_motifs_slices = merged_motifs_tensor(input.id(), motifs.as_slice())?;
                calc_metrics(self.out_id(input.id()), input.id(), input.metrics(), input.quantiles(), channels_motifs_slices.t_ref())
            }
        }
    }
}

impl<C> Default for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C> Adjustable for MotifStatsSyncStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = MotifStatsCalcInput<'a>;
    type Out = MotifStatsOutput;
}

impl<C> SyncStage for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStatsSyncStage::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("MotifStatsSyncStage::do_run: input.id={:?}", input.id);
        let output = self.run_input(input)?;
        debug!("MotifStatsSyncStage::do_run: input.id={:?} channels_values={:?}", input.id(), output.channels_values);
        Ok(output)
    }
}

#[derive(Debug, Named)]
struct MotifStatsConsumeStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    _ctx: PhantomData<C>,
}

impl<C> MotifStatsConsumeStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn new() -> Self {
        Self { id: motif_stats_stage_id(), _ctx: PhantomData }
    }

    fn run_input(&self, input: &MotifStatsLoadedChannelsInput) -> Result<MotifStatsOutput, ThoError> {
        let channels_motifs_slices = merged_channels_tensor(input.id(), input.channels_motifs_slices())?;
        calc_metrics(self.out_id(input.id()), input.id(), input.metrics(), input.quantiles(), channels_motifs_slices.t_ref())
    }
}

impl<C> Default for MotifStatsConsumeStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<C> Adjustable for MotifStatsConsumeStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for MotifStatsConsumeStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for MotifStatsConsumeStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = MotifStatsLoadedChannelsInput;
    type Out = MotifStatsOutput;
}

impl<C> SyncStage for MotifStatsConsumeStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStatsConsumeStage::consume"))]
    fn consume<'a>(self: Arc<Self>, input: Self::In<'a>, _ctx: Self::Ctx) -> ConsumeResult<Self::In<'a>, Self::Out> {
        match self.run_input(&input) {
            Ok(output) => {
                debug!("MotifStatsConsumeStage::consume: input.id={:?} channels_values={:?}", input.id(), output.channels_values);
                ConsumeResult::Ok(output)
            }
            Err(error) => ConsumeResult::Failed { input, error },
        }
    }
}

#[derive(Cached, Named)]
pub struct MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    id: SynKey,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pub motif_loader: Arc<ML>,
    clc_stats_run: Arc<MotifStatsSyncStage<C>>,
    clc_stats_consume: Arc<MotifStatsConsumeStage<C>>,
    _ctx: PhantomData<C>,
}

impl<ML, L, C> Debug for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MotifStats").finish()
    }
}

impl<ML, L, C> MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    pub fn new(
        store: Arc<Box<dyn Store>>,
        loader: Arc<L>,
        ws: Arc<RwLock<Box<dyn Workspace>>>,
        motif_loader: Arc<ML>,
    ) -> Self {
        Self {
            id: (MOTIF_STATS, "MotifStats").into(),
            store,
            loader,
            ws,
            motif_loader,
            clc_stats_run: Arc::new(MotifStatsSyncStage::default()),
            clc_stats_consume: Arc::new(MotifStatsConsumeStage::default()),
            _ctx: PhantomData,
        }
    }
}

impl<ML, L, C> OutId for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<ML, L, C> Stage for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    type Ctx = C;
    type In<'a> = MotifStatsInput;
    type Out = MotifStatsOutput;
}

impl<ML, L, C> AsyncStage for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStats::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<MotifStatsInput>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let input_id = input.id().clone();
        let (source, metrics, quantiles) = with_data!(ctx.branch(Some("motif_stats_input"), false), input; |_ctx, input: &MotifStatsInput| {
            Ok::<_, ThoError>((input.source().clone(), input.metrics().clone(), None))
        })?;
        let stats = match source {
            MotifStatsSource::Motif(motif_loader_input) => {
                debug!("MotifStats::run: motif_loader_input={motif_loader_input:?}");
                let motif = Arc::clone(&self.motif_loader)
                    .run_cached(
                        Arc::new(DevicesWrap::wrap(motif_loader_input)),
                        ctx.branch(Some("motif_stats_motif_loader"), false),
                    )
                    .await?;

                debug!("MotifStats::run: motif={motif:?}");
                let clc_input = Arc::new(MotifStatsCalcInputWrapper::from_managed(input_id.clone(), motif, metrics.clone(), quantiles.clone()));
                let required_ids: HashMap<&'static str, &std::collections::HashSet<&dyn thoosa_shared::traits::CachableIdTr>> = HashMap::new();
                let oom_handler: OOMHandler<'_, '_, _, _, _, _, ()> = OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                        Adjusters::RunContext(Box::new(DumpToCpu::new("clc_stats"))),
                    ],
                    None,
                );
                oom_handler.run(
                    &self.clc_stats_run,
                    clc_input,
                    ctx.branch(Some("motif_stats_sync"), false),
                    Contexts::new(
                        NoopInputContext(None),
                        NoopRunnableContext {},
                        MaxParallelAndDumpToCpuAdjustableContex::new(
                            MaxParallelAdjustContextImpl::new(),
                            DumpToCpuContextImpl::new(&required_ids),
                        ),
                    ),
                )
                .await?
            }
            MotifStatsSource::MergedMotifs(motif_loader_inputs) => {
                if motif_loader_inputs.is_empty() {
                    return Err(ThoError::new("MotifStats::run: merged motif inputs are empty"));
                }

                let mut expected_channel = None;
                let mut expected_window = None;
                let mut motifs = Vec::with_capacity(motif_loader_inputs.len());

                for (idx, motif_loader_input) in motif_loader_inputs.into_iter().enumerate() {
                    let motif = Arc::clone(&self.motif_loader)
                        .run_cached(
                            Arc::new(DevicesWrap::wrap(motif_loader_input)),
                            ctx.branch(Some("motif_stats_merged_motif_loader"), false),
                        )
                        .await?;

                    let meta = motif.extra().await?;
                    debug!("MotifStats::run: merged motif idx={idx} channel={} window={} size={:?}", meta.channel, meta.window, motif.size());

                    match expected_channel {
                        Some(channel) if channel != meta.channel => {
                            return Err(ThoError::new(format!("MotifStats::run: merged motifs must have the same channel, got {channel} and {}", meta.channel)));
                        }
                        None => expected_channel = Some(meta.channel),
                        _ => {}
                    }
                    match expected_window {
                        Some(window) if window != meta.window => {
                            return Err(ThoError::new(format!("MotifStats::run: merged motifs must have the same window, got {window} and {}", meta.window)));
                        }
                        None => expected_window = Some(meta.window),
                        _ => {}
                    }

                    motifs.push(motif);
                }

                let clc_input = Arc::new(MotifStatsCalcInputWrapper::from_motifs(input_id.clone(), motifs, metrics.clone(), quantiles.clone()));
                let required_ids: HashMap<&'static str, &std::collections::HashSet<&dyn thoosa_shared::traits::CachableIdTr>> = HashMap::new();
                let oom_handler: OOMHandler<'_, '_, _, _, _, _, ()> = OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                        Adjusters::RunContext(Box::new(DumpToCpu::new("clc_stats"))),
                    ],
                    None,
                );
                oom_handler.run(
                    &self.clc_stats_run,
                    clc_input,
                    ctx.branch(Some("motif_stats_sync"), false),
                    Contexts::new(
                        NoopInputContext(None),
                        NoopRunnableContext {},
                        MaxParallelAndDumpToCpuAdjustableContex::new(
                            MaxParallelAdjustContextImpl::new(),
                            DumpToCpuContextImpl::new(&required_ids),
                        ),
                    ),
                )
                .await?
            }
            MotifStatsSource::Random { max_slices, find_motifs_run_id } => {
                debug!("MotifStats::run: MotifStatsSource::Random  max_slices={max_slices:?} find_motifs_run_id={find_motifs_run_id}");

                if max_slices < 2 {
                    return Err(ThoError::new(format!("MotifStats::run: max_slices must be >= 2, got {max_slices}")));
                }


                let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
                    find_motifs_run_id,
                    &self.store.list_run_config_values(find_motifs_run_id).await?,
                )?;
                let input_len_adjuster = find_motifs_run_config.input_len_adjuster();
                let resources = self.store.list_random_resources(None).await?;
                let (slices, resources_map) = random_non_overlapping_slices(&resources, max_slices, find_motifs_run_config.window, input_len_adjuster)?;

                let (_window, channels_slices) = load_random_slices(
                    find_motifs_run_id,
                    resources_map,
                    slices,
                    Arc::clone(&self.store),
                    Arc::clone(&self.loader),
                    ctx.branch(Some("motif_stats_random_load_slices"), false),
                    &self.ws,
                    "motif_stats_random",
                )
                .await?;

                debug!("MotifStats::run: MotifStatsSource::Random  channels_slices={channels_slices:?}");
                let tmp_cache_name = channels_slices.registered_name();
                let clc_input = MotifStatsTmpCachedChannelsInput::new(input_id.clone(), channels_slices, metrics.clone(), quantiles.clone());
                let required_tmp_cache_ids = HashSet::<&dyn thoosa_shared::traits::CachableIdTr>::new();
                let required_ids = HashMap::from([(tmp_cache_name, &required_tmp_cache_ids)]);
                let oom_handler: OOMHandler<'_, '_, _, _, _, _, ()> = OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                        Adjusters::RunContext(Box::new(DumpToCpu::new("clc_stats"))),
                    ],
                    None,
                );
                oom_handler.consume(
                    &self.clc_stats_consume,
                    clc_input,
                    ctx.branch(Some("motif_stats_sync"), false),
                    Contexts::new(
                        NoopInputContext(None),
                        NoopRunnableContext {},
                        MaxParallelAndDumpToCpuAdjustableContex::new(
                            MaxParallelAdjustContextImpl::new(),
                            DumpToCpuContextImpl::new(&required_ids),
                        ),
                    ),
                )
                .await?
            }
        };

        debug!("MotifStats::run: done input_id={input_id:?} stats={:?}", stats);

        Ok(Arc::new(DevicesWrap::wrap(stats)))
    }
}

impl<ML, L, C> Adjustable for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="random_non_overlapping_slices"))]
pub fn random_non_overlapping_slices(
    resources: &[Resource],
    number_of_slices: u32,
    slice_width: u32,
    input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
) -> Result<(Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError> {

    debug!("random_non_overlapping_slices: resources={resources:?} number_of_slices={number_of_slices:?} slice_width={slice_width:?}");

    assert_ne!(number_of_slices, 0, "number_of_slices is 0");
    assert_ne!(slice_width, 0, "slice_width is 0");

    #[derive(Clone)]
    struct ResState {
        res: Resource,
        starts: Vec<u32>, // available slice starts (aligned, non-overlapping by construction)
        picked: usize,
    }

    let mut rng:StdRng = rand::make_rng();

    let mut states: Vec<ResState> = resources
        .iter()
        .filter_map(|r| {
            let resource_len = input_len_adjuster(r.length());

            if resource_len < slice_width {
                return None;
            }
            let bins = resource_len / slice_width; // number of non-overlapping aligned slices
            if bins == 0 {
                return None;
            }
            let mut starts: Vec<u32> = (0..bins).map(|i| i * slice_width).collect();
            starts.shuffle(&mut rng);
            Some(ResState {
                res: r.clone(),
                starts,
                picked: 0,
            })
        })
        .collect();

    if states.is_empty() {

        return Err(ThoError::new(format!("random_non_overlapping_slices: no suitable resource for {number_of_slices} of {slice_width} width")));
    }

    let available_total: usize = states.iter().map(|s| s.starts.len()).sum();
    if available_total < number_of_slices as usize {
        error!("random_non_overlapping_slices: no suitable resource for {number_of_slices} of {slice_width} width, slice_width={slice_width} resources={resources:?}");
        return Err(ThoError::new(format!("random_non_overlapping_slices: not enough resources ({available_total}) available for {number_of_slices} random slices")));
        // warn!("random_non_overlapping_slices: not enough resources ({available_total}) available for {number_of_slices} random slices");
    }

    let mut chosen: Vec<TensorIdSliceIdx> = Vec::with_capacity(number_of_slices as usize);
    let mut used_resources: HashMap<ResourceId, Resource> = HashMap::new();

    while chosen.len() < number_of_slices as usize {
        // Find the minimum picked-count among resources that still have available slices.
        let mut min_picked: Option<usize> = None;
        for s in &states {
            if s.starts.is_empty() {
                continue;
            }
            min_picked = Some(match min_picked {
                None => s.picked,
                Some(m) => m.min(s.picked),
            });
        }


        let min_p = min_picked.expect("no slices");

        // Candidates = those with min_picked and remaining starts.
        let mut candidates: Vec<usize> = Vec::new();
        for (i, s) in states.iter().enumerate() {
            if !s.starts.is_empty() && s.picked == min_p {
                candidates.push(i);
            }
        }

        let pick_i = *candidates.choose(&mut rng).expect("cant choose slices");

        let s = &mut states[pick_i];
        let start = s.starts.pop().expect("no more slices");
        s.picked += 1;

        chosen.push(TensorIdSliceIdx(s.res.id(), start));
        used_resources.entry(s.res.id()).or_insert(s.res.clone());
    }

    Ok((chosen, used_resources))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::motif_loader::mock_motif_loader;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::{kind::FLOAT_CPU, Tensor};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::cachable::WithDataRef;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_store::{MockStoreBuilder, RunConfigRow};
    use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_random_stats_basic() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 1000)));
        let resources = vec![
            Resource::new(1, 1, "resource_a", 10),
            Resource::new(2, 1, "resource_b", 10),
            Resource::new(3, 1, "resource_c", 10),
            Resource::new(4, 1, "resource_d", 10),
        ];

        let run_id = 42;
        let motif_width = 6;
        let store = Arc::new(
            MockStoreBuilder::new()
                .with_list_random_resources([(None, Ok(resources.clone()))])
                .with_run_config_descr([(
                    run_id,
                    Ok(RunConfigRow {
                        id: run_id,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find motifs".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(run_id), 1, motif_width, None, None)
                        .to_properties_iter(None)
                        .collect()),
                )
                .build(),
        );

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resources[0].clone(), Tensor::full(30, 0., FLOAT_CPU).view((3, -1))),
                    (resources[1].clone(), Tensor::full(30, 1., FLOAT_CPU).view((3, -1))),
                    (resources[2].clone(), Tensor::full(30, 2., FLOAT_CPU).view((3, -1))),
                    (resources[3].clone(), Tensor::full(30, 3., FLOAT_CPU).view((3, -1))),
                ])
                .finish(),
        );

        // let stage = Arc::new(RandomStats::new(Arc::clone(&store), loader, &ws));
        let stage = Arc::new(MotifStats::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            Arc::clone(&ws),
            Arc::new(MotifLoader::new(Arc::clone(&store), Arc::clone(&loader), &ws)),
        ));

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(1)));
        let metric = Metrics::default();
        let input = Arc::new(DevicesWrap::wrap(MotifStatsInput::from_random(4, run_id, vec![metric.clone()])));

        let output = Arc::clone(&stage)
            .run(
                input,
                thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx))
                    .branch(Some("random_stats"), false),
            )
            .await?;

        let mut ctx_for_data = thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx))
            .branch(Some("random_stats_data"), false);
        let size = output
            .with_data_ref(0, ctx_for_data, None, |stats: &MotifStatsOutput, _ctx| {
                let t = stats.channels_motifs_dists(&metric)?;
                debug!("t=\n{}", t.as_ref());
                assert!(Tensor::from_slice(&[4.0825f32, 4.0825, 4.0825]).view((3, 1)).allclose(t.as_ref(), 0.02, 0.02, false));

                Ok::<_, ThoError>(t.t_ref().size())
            })
            .await?;

        assert_eq!(size, vec![3, 1]);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_merged_motifs_stats_source() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let store = Arc::new(MockStoreBuilder::new().build());
        let loader = Arc::new(mock_tensor_loader::build().finish());
        let motif_input_0 = MotifLoaderInput::new((1u32, "motif_id").into(), Some(vec![100]));
        let motif_input_1 = MotifLoaderInput::new((2u32, "motif_id").into(), Some(vec![200]));
        let motif_loader = Arc::new(
            mock_motif_loader::build()
                .with_run_cached([
                    (
                        motif_input_0.clone(),
                        (
                            MotifMeta { channel: 0, find_motifs_run_id: 10, window: 3 },
                            Tensor::from_slice(&[0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0]).view((2, 3)),
                        ),
                    ),
                    (
                        motif_input_1.clone(),
                        (
                            MotifMeta { channel: 0, find_motifs_run_id: 10, window: 3 },
                            Tensor::from_slice(&[0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0]).view((2, 3)),
                        ),
                    ),
                ])
                .finish(),
        );
        let stage = Arc::new(MotifStats::new(Arc::clone(&store), loader, Arc::clone(&ws), motif_loader));
        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));
        let metric = Metrics::default();
        let input = Arc::new(DevicesWrap::wrap(MotifStatsInput::from_merged_motif_inputs(
            vec![motif_input_0, motif_input_1],
            vec![metric.clone()],
        )?));

        let output = Arc::clone(&stage)
            .run(
                input,
                thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx))
                .branch(Some("motif_stats_merged"), false),
            )
            .await?;

        let value = output
            .with_data_ref(
                0,
                thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx))
                .branch(Some("motif_stats_merged_data"), false),
                None,
                |stats: &MotifStatsOutput, _ctx| stats.channel_value(&metric, 0),
            )
            .await?;

        assert!((value - 6.6666665).abs() < 1e-4, "unexpected merged motif metric: {value}");

        Ok(())
    }

    #[test]
    fn test_random_non_overlapping_slices_uses_adjusted_length() -> Result<(), ThoError> {
        let resources = vec![
            Resource::new(1, 1, "resource_a", 10),
            Resource::new(2, 1, "resource_b", 10),
        ];
        let motif_width = 10;
        let input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync> = Arc::new(|len| len);

        let (slices, resources_map) = random_non_overlapping_slices(&resources, 2, motif_width, input_len_adjuster)?;
        assert_eq!(slices.len(), 2);
        assert_eq!(resources_map.len(), 2);
        assert!(slices.iter().all(|slice| slice.1 == 0));

        Ok(())
    }

    #[test]
    fn test_random_non_overlapping_slices_filters_resources_by_adjusted_length() {
        let resources = vec![Resource::new(1, 1, "resource_a", 10)];
        let motif_width = 10;
        let input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync> = Arc::new(|len| len - 1);

        let err = random_non_overlapping_slices(&resources, 2, motif_width, input_len_adjuster).expect_err("expected adjusted-length filtering error");

        assert!(format!("{err:?}").contains("no suitable resource for 2 of 10 width"));
    }
}
