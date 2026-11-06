use thoosa_common::runner::sync_stage::ConsumingStage;
use crate::motifs_store::MotifsStore;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;

use thoosa_common::tch_error::tch_result;

use crate::dims::{ChannelsMotifsDists, ChannelsMotifsSlices, MotifSlices};
use crate::stages::find_motifs_run_config::{FindMotifsRunConfig, Metrics};
use crate::stages::motif_loader::{MotifLoaderInput, MotifMeta, load_random_slices};
use crate::stages::slices_stats::{metric_quantiles_for_motif, metric_scalar_for_motif};
use crate::stages::stage_ids::{MOTIF_STATS, MOTIF_STATS_SYNC_STAGE};
use async_trait::async_trait;
use cached_derive::Cached;
use futures_util::future::BoxFuture;
use log::debug;
use parking_lot::RwLock;
use rand::prelude::{IndexedRandom, SliceRandom, StdRng};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tch::{IndexOp, Tensor};
use thoosa_common::data_with_extra::DataWithExtra;
use thoosa_common::dims::ChannelsData;
use thoosa_common::instance_cache::InstanceCache;
use thoosa_common::ipc::cuda_wokrer::{AllTensors, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use thoosa_common::ipc::serialization::json::JsonIpcCodec;
use thoosa_common::ipc::worker_msg::WorkerInput;
use thoosa_common::ipc::wraps::SyncStageIpcWrapper;
use thoosa_common::ipc::wraps::WorkerWrapper;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, CachedPurgeOutput, ConsumeData, DevicesWrap, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{Adjusters, NoInputAdjuster, NoStageAdjuster, OOMHandler, WithRunCtxContext, run_ctx_adjuster};
use thoosa_common::runner::sync_stage::ConsumeResult;
use thoosa_common::runner::sync_stage::SyncStage;
use thoosa_common::runner::tmp_cache::TmpCached;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_named_derive::Named;
use thoosa_properties_map::{FromHashmap, ToHashmap};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::ipc_types::WorkerOutput;
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_shared::traits::{Adjustable, AsyncStage, Data, MonoMetrics, OutId, RunCtx, Stage, WithId};
use thoosa_shared::{CopyToDevice, TensorIdSliceIdx, ThoError, mono_key};
use tracing::error;

pub const RANDOM_STATS_GROUP: &str = "random_avg_dist"; //dummy group to load random resources, TODO - add proper method to SqlStore

#[derive(Debug, Clone)]
pub enum MotifStatsSource {
    Motif(MotifLoaderInput),
    MergedMotifs(Vec<MotifLoaderInput>),
    Random { max_slices: u32, find_motifs_run_config_id: RunConfigId, resources: Option<Vec<Resource>> },
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

    pub fn from_random(max_slices: u32, find_motifs_run_config_id: RunConfigId, metrics: Vec<Metrics>) -> Self {
        Self::from_random_resources(max_slices, find_motifs_run_config_id, None, metrics)
    }

    pub fn from_random_resources(max_slices: u32, find_motifs_run_config_id: RunConfigId, resources: Option<Vec<Resource>>, metrics: Vec<Metrics>) -> Self {
        let metrics_sfx = metrics.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>().join(",");
        let resources_sfx = resources
            .as_ref()
            .map(|resources| {
                let mut ids = resources.iter().map(Resource::id).collect::<Vec<_>>();
                ids.sort_unstable();
                ids.iter().map(|id| id.to_string()).collect::<Vec<_>>().join(",")
            })
            .unwrap_or_else(|| "random".to_string());
        let id = SynKey::new([
            mono_key!("MotifStatsInput", "stage"),
            mono_key!(find_motifs_run_config_id, format!("find_motifs_run_config_id={find_motifs_run_config_id}")),
            mono_key!(max_slices, format!("max_slices={max_slices}")),
            mono_key!(resources_sfx.clone(), format!("resources={resources_sfx}")),
            mono_key!(metrics_sfx.clone(), format!("metrics={metrics_sfx}")),
        ]);

        debug!("MotifStatsInput::from_random_resources: id={id:?}");

        Self {
            id,
            source: MotifStatsSource::Random { max_slices, find_motifs_run_config_id, resources },
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


    fn size(&self) -> Self::Metrics {
        1usize.into()
    }

    fn tho_device_idx(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0
    }
}


impl Adjustable for MotifStatsInput {}

type LoadedMotif = DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>;

#[derive(Debug)]
pub struct MotifStatsCalcInput<'a> {
    id: SynKey,
    data: MotifStatsCalcInputData<'a>,
    metrics: Vec<Metrics>,
    quantiles: Option<Vec<f32>>,
}

#[derive(Debug)]
enum MotifStatsCalcInputData<'a> {
    Managed(ThoDataRef<'a, LoadedMotif>),
    Motifs(Vec<ThoDataRef<'a, LoadedMotif>>),
}

impl<'a> MotifStatsCalcInput<'a> {
    fn from_data_refs(id: SynKey, data: MotifStatsCalcInputData<'a>, metrics: Vec<Metrics>, quantiles: Option<Vec<f32>>) -> Self {
        Self { id, data, metrics, quantiles }
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

    fn motifs_size(motifs: &[ThoDataRef<'_, LoadedMotif>]) -> <ThoTensor<ChannelsMotifsSlices, SynKey> as Data>::Metrics {
        let num_slices = motifs.iter().map(|motif| motif.as_ref().data().size()[0]).sum::<i64>();
        let window = motifs.first().map(|motif| motif.as_ref().data().size()[1]).unwrap_or(0);
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


    fn size(&self) -> Self::Metrics {
        match self.data() {
            MotifStatsCalcInputData::Managed(motif) => Self::channels_size_from_tensor(motif.as_ref().data().size().as_slice()),
            MotifStatsCalcInputData::Motifs(motifs) => Self::motifs_size(motifs.as_slice()),
        }
    }

    fn tho_device_idx(&self) -> usize {
        match self.data() {
            MotifStatsCalcInputData::Managed(motif) => motif.as_ref().data().tho_device_idx(),
            MotifStatsCalcInputData::Motifs(motifs) => motifs.first().map(|motif| motif.as_ref().data().tho_device_idx()).unwrap_or(0),
        }
    }

    fn mem_size(&self) -> u64 {
        match self.data() {
            MotifStatsCalcInputData::Managed(motif) => motif.as_ref().data().mem_size(),
            MotifStatsCalcInputData::Motifs(motifs) => motifs.iter().map(|motif| motif.as_ref().data().mem_size()).sum(),
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

}

impl BorrowRef for MotifStatsCalcInputWrapper {
    fn protect(&self, device: usize) -> thoosa_common::runner::residency::ResidencyLease { match &self.data { MotifStatsCalcInputDataWrapper::Managed(value) => value.protect(device), MotifStatsCalcInputDataWrapper::Motifs(values) => thoosa_common::runner::residency::ResidencyLease::combine(values.iter().map(|value| value.protect(device))) } }

    type Out<'a> = MotifStatsCalcInput<'a> where Self: 'a;

    fn borrow_ref<'a>(&'a self, to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let data = match &self.data {
                MotifStatsCalcInputDataWrapper::Managed(motif) => {
                    let motif = t!(motif.load_from_any_device_(to_device_idx)).await?;
                    MotifStatsCalcInputData::Managed(ThoDataRef::ReadLocked(motif))
                }
                MotifStatsCalcInputDataWrapper::Motifs(motifs) => {
                    let mut loaded = Vec::with_capacity(motifs.len());
                    for motif in motifs {
                        loaded.push(ThoDataRef::ReadLocked(t!(motif.load_from_any_device_(to_device_idx)).await?));
                    }
                    MotifStatsCalcInputData::Motifs(loaded)
                }
            };
            Ok(MotifStatsCalcInput::from_data_refs(self.id.clone(), data, self.metrics.clone(), self.quantiles.clone()))
        }
    }
}

#[derive(Debug)]
pub struct MotifStatsLoadedChannelsInput {
    id: SynKey,
    channels_motifs_slices: Vec<ThoTensor<ChannelsMotifsSlices, SynKey>>,
    metrics: Vec<Metrics>,
    quantiles: Option<Vec<f32>>,
    // Preserve the cache identity when failed consumption reconstructs its inputs.
    cache_name: String,
}

impl MotifStatsLoadedChannelsInput {
    fn new(
        id: SynKey,
        mut channels_motifs_slices: Vec<ThoTensor<ChannelsMotifsSlices, SynKey>>,
        metrics: Vec<Metrics>,
        quantiles: Option<Vec<f32>>,
        cache_name: String,
    ) -> Self {
        // MotifStatsLoadedChannelsInput is to be consumed
        for t in &mut channels_motifs_slices {
            t.no_drop();
        }
        Self { id, channels_motifs_slices, metrics, quantiles, cache_name }
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


    fn size(&self) -> Self::Metrics {
        1usize.into()
    }

    fn tho_device_idx(&self) -> usize {
        self.channels_motifs_slices.first().map(|tensor| tensor.tho_device_idx()).unwrap_or(0)
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
            channels_motifs_slices.push(t!(channels.copy_to_device(dst_device)).await?);
        }
        Ok(Self {
            id: self.id.clone(),
            channels_motifs_slices,
            metrics: self.metrics.clone(),
            quantiles: self.quantiles.clone(),
            cache_name: self.cache_name.clone(),
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

impl ConsumeData for MotifStatsTmpCachedChannelsInput {
    fn protect(&self, device: usize) -> thoosa_common::runner::residency::ResidencyLease { self.channels_motifs_slices.protect(device) }

    type O = MotifStatsLoadedChannelsInput;

    fn consume<R: Send, F, P>(
        self,
        to_device_idx: usize,
        run_ctx: P,
        do_run: F,
    ) -> impl Future<Output=ConsumeResult<Self::O, R>> + Send
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

            let cache_name = channels_motifs_slices.name.clone();
            let channels_motifs_slices = match t!(channels_motifs_slices.take_all_recovering(to_device_idx)).await {
                Ok(channels_motifs_slices) => channels_motifs_slices,
                Err(err) => {
                    return ConsumeResult::Failed {
                        input: MotifStatsLoadedChannelsInput::new(id, err.data, metrics, quantiles, cache_name),
                        error: err.error,
                    };
                }
            };

            let input = MotifStatsLoadedChannelsInput::new(id, channels_motifs_slices, metrics, quantiles, cache_name);

            do_run(input, run_ctx)
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStatsTmpCachedChannelsInput::burp"))]
    fn burp(o: Self::O) -> impl std::future::Future<Output = Self> + Send {
        async move {
            let cache = TmpCached::restore(o.cache_name, o.channels_motifs_slices);
            MotifStatsTmpCachedChannelsInput::new(o.id, cache, o.metrics, o.quantiles)
        }
    }
}

fn merged_motifs_tensor(input_id: &SynKey, motifs: &[&LoadedMotif]) -> Result<ThoTensor<ChannelsMotifsSlices, SynKey>, ThoError> {
    if motifs.is_empty() {
        return Err(ThoError::new("MotifStatsSyncStage::merged_motifs_tensor: empty motifs"));
    }

    let motifs_refs = motifs.iter().map(|motif| motif.data().local()).collect::<Result<Vec<_>, _>>()?;
    ThoTensor::new(input_id.clone(), tch_result(Tensor::f_cat(motifs_refs.as_slice(), 0))?)
}

fn merged_channels_tensor(
    input_id: &SynKey,
    channels_motifs_slices: &[ThoTensor<ChannelsMotifsSlices, SynKey>],
) -> Result<ThoTensor<ChannelsMotifsSlices, SynKey>, ThoError> {
    if channels_motifs_slices.is_empty() {
        return Err(ThoError::new("MotifStatsSyncStage::merged_channels_tensor: empty channels_motifs_slices"));
    }

    let invalid_shapes = channels_motifs_slices
        .iter()
        .filter_map(|slices| (slices.size().len() != 3).then(|| (slices.id().clone(), slices.size())))
        .collect::<Vec<_>>();
    if !invalid_shapes.is_empty() {
        return Err(ThoError::new(format!("MotifStatsSyncStage::merged_channels_tensor: expected [channels x slices x window], got invalid_shapes={invalid_shapes:?} input_id={input_id:?}")));
    }

    let channels_motifs_slices_refs = channels_motifs_slices.iter().map(|slices| slices.local()).collect::<Result<Vec<_>, _>>()?;
    ThoTensor::new(input_id.clone(), tch_result(Tensor::f_cat(channels_motifs_slices_refs.as_slice(), 1))?)
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
                tch_result(channels_motifs_slices.f_i((.., ..)))?
            } else {
                tch_result(channels_motifs_slices.f_get(channel_idx as i64))?
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

    debug!("motif_stats::calc_metrics: input_id={input_id:?} channels={channels} channels_values={channels_values:?}");

    Ok(MotifStatsOutput::new(out_id, channels_values, num_slices as u32))
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MetricsValue {
    #[serde(with = "metric_serde")]
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifStatsOutput {
    id: SynKey,
    channels_values: Vec<MetricsValue>,
    pub num_slices: u32,
}

mod metric_serde {
    use super::*;

    pub fn serialize<S>(metric: &Metrics, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        metric_to_ipc(metric).serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Metrics, D::Error>
    where
        D: Deserializer<'de>,
    {
        let properties = HashMap::deserialize(deserializer)?;
        metric_from_ipc(properties).map_err(serde::de::Error::custom)
    }
}

#[derive(Debug, Serialize, Deserialize)]
enum MotifStatsIpcVal {
    Real(f64),
    Int(i64),
    UInt(u32),
    Str(String),
    Bool(bool),
}

fn metric_to_ipc(metric: &Metrics) -> HashMap<String, MotifStatsIpcVal> {
    metric.to_properties_iter(None).map(|(key, value)| {
        let value = match value {
            thoosa_shared::val::Val::Real(value) => MotifStatsIpcVal::Real(value),
            thoosa_shared::val::Val::Int(value) => MotifStatsIpcVal::Int(value),
            thoosa_shared::val::Val::UInt(value) => MotifStatsIpcVal::UInt(value),
            thoosa_shared::val::Val::Str(value) => MotifStatsIpcVal::Str(value),
            thoosa_shared::val::Val::Bool(value) => MotifStatsIpcVal::Bool(value),
        };
        (key, value)
    }).collect()
}

fn metric_from_ipc(properties: HashMap<String, MotifStatsIpcVal>) -> Result<Metrics, ThoError> {
    let properties = properties.into_iter().map(|(key, value)| {
        let value = match value {
            MotifStatsIpcVal::Real(value) => thoosa_shared::val::Val::Real(value),
            MotifStatsIpcVal::Int(value) => thoosa_shared::val::Val::Int(value),
            MotifStatsIpcVal::UInt(value) => thoosa_shared::val::Val::UInt(value),
            MotifStatsIpcVal::Str(value) => thoosa_shared::val::Val::Str(value),
            MotifStatsIpcVal::Bool(value) => thoosa_shared::val::Val::Bool(value),
        };
        (key, value)
    }).collect();
    Ok(Metrics::from_properties_map(None, &properties)?)
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
        let t = tch_result(tch_result(Tensor::f_from_slice(scalar_values.as_slice()))?.f_view([scalar_values.len() as i64, 1]))?;
        ThoTensor::new(self.metric_values_id(metric), t)
    }
}

impl Data for MotifStatsOutput {
    type Metrics = MonoMetrics<0, 1>;


    fn size(&self) -> Self::Metrics {
        1usize.into()
    }

    fn tho_device_idx(&self) -> usize {
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
//TODO add generic PlainWrapper implementing CachedPurgeOutput
impl CachedPurgeOutput for MotifStatsOutput {
    fn has_any_data(&self) -> bool { true }

    fn dump_to_cpu<'a>(&'a self, _device_idx: usize) -> futures_util::future::BoxFuture<'a, Result<(), ThoError>> {
        Box::pin(async { Ok(()) })
    }

    fn cleanup_device<'a>(&'a self, _device_idx: usize) -> futures_util::future::BoxFuture<'a, Result<(), ThoError>> {
        Box::pin(async { Ok(()) })
    }

    fn drop_device<'a>(&'a self, _device_idx: usize) -> futures_util::future::BoxFuture<'a, Result<(), ThoError>> {
        Box::pin(async { Ok(()) })
    }

    fn is_on_device(&self, device_idx: usize) -> bool {
        device_idx == 0
    }

    fn is_on_cpu_before(&self, _purge_until: u64) -> bool {
        true
    }

    fn mem_size(&self) -> u64 {
        0
    }
}


pub const MOTIF_STATS_SYNC_ENDPOINT: &str = "MotifStatsSyncStage";

#[derive(Debug, Serialize)]
struct MotifStatsIpcMotifRef<'a> {
    tensor: &'a ThoTensor<MotifSlices, SynKey>,
    meta: &'a MotifMeta,
}

#[derive(Debug, Serialize)]
enum MotifStatsIpcDataRef<'a> {
    Managed(MotifStatsIpcMotifRef<'a>),
    Motifs(Vec<MotifStatsIpcMotifRef<'a>>),
}

#[derive(Debug, Serialize)]
struct MotifStatsSyncStageIpcInputRef<'a> {
    id: &'a SynKey,
    data: MotifStatsIpcDataRef<'a>,
    metrics: Vec<HashMap<String, MotifStatsIpcVal>>,
    quantiles: &'a Option<Vec<f32>>,
}

impl<'a> MotifStatsSyncStageIpcInputRef<'a> {
    fn new(input: &'a MotifStatsCalcInput<'a>) -> Self {
        let data = match input.data() {
            MotifStatsCalcInputData::Managed(motif) => {
                let motif = motif.as_ref();
                MotifStatsIpcDataRef::Managed(MotifStatsIpcMotifRef { tensor: motif.data(), meta: motif.extra() })
            }
            MotifStatsCalcInputData::Motifs(motifs) => MotifStatsIpcDataRef::Motifs(
                motifs.iter().map(|motif| {
                    let motif = motif.as_ref();
                    MotifStatsIpcMotifRef { tensor: motif.data(), meta: motif.extra() }
                }).collect(),
            ),
        };
        let metrics = input.metrics.iter().map(metric_to_ipc).collect();
        Self { id: &input.id, data, metrics, quantiles: &input.quantiles }
    }
}

#[derive(Debug, Deserialize)]
struct MotifStatsIpcMotif {
    tensor: ThoTensor<MotifSlices, SynKey>,
    meta: MotifMeta,
}

#[derive(Debug, Deserialize)]
enum MotifStatsIpcData {
    Managed(MotifStatsIpcMotif),
    Motifs(Vec<MotifStatsIpcMotif>),
}

#[derive(Debug, Deserialize)]
struct MotifStatsSyncStageIpcInput {
    id: SynKey,
    data: MotifStatsIpcData,
    metrics: Vec<HashMap<String, MotifStatsIpcVal>>,
    quantiles: Option<Vec<f32>>,
}

impl Serialize for MotifStatsCalcInput<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        MotifStatsSyncStageIpcInputRef::new(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MotifStatsCalcInput<'static> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let input = MotifStatsSyncStageIpcInput::deserialize(deserializer)?;
        let metrics = input.metrics.into_iter().map(metric_from_ipc).collect::<Result<Vec<_>, _>>().map_err(serde::de::Error::custom)?;
        // The worker owns the deserialized tensors, so borrowed main-process inputs become owned data refs.
        let data = match input.data {
            MotifStatsIpcData::Managed(motif) => MotifStatsCalcInputData::Managed(ThoDataRef::Owned(DataWithExtra::new(motif.tensor, motif.meta))),
            MotifStatsIpcData::Motifs(motifs) => MotifStatsCalcInputData::Motifs(motifs.into_iter().map(|motif| ThoDataRef::Owned(DataWithExtra::new(motif.tensor, motif.meta))).collect()),
        };
        Ok(Self::from_data_refs(input.id, data, metrics, input.quantiles))
    }
}

#[typetag::serde(name = "MotifStatsCalcInput")]
impl WorkerInput for MotifStatsCalcInput<'static> {}


impl WorkerStageInput for MotifStatsCalcInput<'_> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        let clone_motif = |motif: &LoadedMotif| -> Result<LoadedMotif, ThoError> { Ok(DataWithExtra::new(motif.data().clone_ref()?, motif.extra().clone())) };
        // Remove input lifetimes before typetag serializes the request.
        let data = match self.data() {
            MotifStatsCalcInputData::Managed(motif) => MotifStatsCalcInputData::Managed(ThoDataRef::Owned(clone_motif(motif.as_ref())?)),
            MotifStatsCalcInputData::Motifs(motifs) => MotifStatsCalcInputData::Motifs(motifs.iter().map(|motif| Ok(ThoDataRef::Owned(clone_motif(motif.as_ref())?))).collect::<Result<Vec<_>, ThoError>>()?),
        };
        Ok(Box::new(MotifStatsCalcInput::from_data_refs(self.id.clone(), data, self.metrics.clone(), self.quantiles.clone())))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        let materialize = |motif: &LoadedMotif| -> Result<LoadedMotif, ThoError> {
            Ok(DataWithExtra::new(WorkerStageInput::ref_to_tensor(motif.data(), all_cuda_tensors, remove)?, motif.extra().clone()))
        };
        let data = match self.data() {
            MotifStatsCalcInputData::Managed(motif) => MotifStatsCalcInputData::Managed(ThoDataRef::Owned(materialize(motif.as_ref())?)),
            MotifStatsCalcInputData::Motifs(motifs) => MotifStatsCalcInputData::Motifs(motifs.iter().map(|motif| Ok(ThoDataRef::Owned(materialize(motif.as_ref())?))).collect::<Result<Vec<_>, ThoError>>()?),
        };
        Ok(Self::from_data_refs(self.id.clone(), data, self.metrics.clone(), self.quantiles.clone()))
    }
}

impl WorkerStageOutput for MotifStatsOutput {
    fn encode_from_worker(self, _all_cuda_tensors: &AllTensors, _tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(self)
    }
}


static MOTIF_STATS_SYNC_STAGE_INSTANCES: InstanceCache = InstanceCache::new();

#[derive(Debug, Named)]
pub struct MotifStatsSyncStage<C>
where
    C: RunCtx + 'static,
{
    id: SynKey,//for OutId to have different Ids for motifs and their stats
    _ctx: PhantomData<C>,
}

impl<C> MotifStatsSyncStage<C>
where
    C: RunCtx + 'static,
{
    async fn new(tho_device_idx: usize) -> Result<Arc<DynSyncStageEnum<MotifStatsSyncStage<C>>>, ThoError> {
        MOTIF_STATS_SYNC_STAGE_INSTANCES.get_or_try_insert_recreatable_with::<C, _, _, _, _>(tho_device_idx, GlobalCtx::global().cuda_activity_gate(tho_device_idx), GlobalCtx::global().is_ipc(), 0, move || async move {
            if GlobalCtx::global().is_ipc() {

                Ok(Arc::new(DynSyncStageEnum::Proxy { stage: Arc::new(ProxyStage::proxy_recreatable(MOTIF_STATS_SYNC_ENDPOINT, Some(0.to_string()), tho_device_idx, Box::new(())).await?), init: () }))
            } else {
                Ok(Arc::new(DynSyncStageEnum::Local(Arc::new(MotifStatsSyncStage::<C>::new_local()))))
            }

        }).await
    }

    fn new_local() -> Self {
        Self { id: (MOTIF_STATS_SYNC_STAGE, "MotifStatsSyncStage").into(), _ctx: PhantomData }
    }
}


impl<C> Adjustable for MotifStatsSyncStage<C> where C: RunCtx + Adjustable + 'static {}


impl<C> OutId for MotifStatsSyncStage<C>
where
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for MotifStatsSyncStage<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = MotifStatsCalcInput<'a>;
    type Out = MotifStatsOutput;
}

impl<C> SyncStage for MotifStatsSyncStage<C>
where
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStatsSyncStage::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("MotifStatsSyncStage::do_run: input.id={:?}", input.id);
        let output = (match input.data() {
            MotifStatsCalcInputData::Managed(motif) => {
                calc_metrics(self.out_id(input.id()), input.id(), input.metrics(), input.quantiles(), &*motif.as_ref().data().local()?)
            },
            MotifStatsCalcInputData::Motifs(motifs) => {
                let motifs = motifs.iter().map(ThoDataRef::as_ref).collect::<Vec<_>>();
                let channels_motifs_slices = merged_motifs_tensor(input.id(), motifs.as_slice())?;
                calc_metrics(self.out_id(input.id()), input.id(), input.metrics(), input.quantiles(), &*channels_motifs_slices.local()?)
            }
        })?;
        debug!("MotifStatsSyncStage::do_run: input.id={:?} channels_values={:?}", input.id(), output.channels_values);
        Ok(output)
    }
}

#[typetag::serde]
impl WorkerOutput for MotifStatsOutput{}

pub fn motif_stats_sync_stage_endpoint_init<'a>(_stage_id:Option<String>, _init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    use futures_util::FutureExt;
    async move {
        // WorkerStageInput owns decoded motifs while the generic shared wrapper runs the calculation.
        Ok(Box::new(WorkerWrapper::<JsonIpcCodec, _, _, _>::new(Arc::new(MotifStatsSyncStage::<WokerRunCtx>::new_local()), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

pub const MOTIF_STATS_CONSUME_ENDPOINT: &str = "MotifStatsConsumeStage";

#[derive(Debug, Serialize)]
struct MotifStatsConsumeStageIpcInputRef<'a> {
    id: &'a SynKey,
    channels_motifs_slices: &'a [ThoTensor<ChannelsMotifsSlices, SynKey>],
    metrics: Vec<HashMap<String, MotifStatsIpcVal>>,
    quantiles: &'a Option<Vec<f32>>,
    cache_name: &'a str,
}

impl<'a> MotifStatsConsumeStageIpcInputRef<'a> {
    fn new(input: &'a MotifStatsLoadedChannelsInput) -> Self {
        Self {
            id: &input.id,
            channels_motifs_slices: input.channels_motifs_slices.as_slice(),
            metrics: input.metrics.iter().map(metric_to_ipc).collect(),
            quantiles: &input.quantiles,
            cache_name: &input.cache_name,
        }
    }
}

#[derive(Debug, Deserialize)]
struct MotifStatsConsumeStageIpcInput {
    id: SynKey,
    channels_motifs_slices: Vec<ThoTensor<ChannelsMotifsSlices, SynKey>>,
    metrics: Vec<HashMap<String, MotifStatsIpcVal>>,
    quantiles: Option<Vec<f32>>,
    cache_name: String,
}

impl Serialize for MotifStatsLoadedChannelsInput {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        MotifStatsConsumeStageIpcInputRef::new(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for MotifStatsLoadedChannelsInput {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let input = MotifStatsConsumeStageIpcInput::deserialize(deserializer)?;
        let metrics = input.metrics.into_iter().map(metric_from_ipc).collect::<Result<Vec<_>, _>>().map_err(serde::de::Error::custom)?;
        Ok(Self::new(input.id, input.channels_motifs_slices, metrics, input.quantiles, input.cache_name))
    }
}

#[typetag::serde(name = "MotifStatsLoadedChannelsInput")]
impl WorkerInput for MotifStatsLoadedChannelsInput {}

impl MotifStatsLoadedChannelsInput {
}

impl WorkerStageInput for MotifStatsLoadedChannelsInput {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        // Clone only worker tensor references; the metrics remain ordinary owned metadata.
        let tensors = self.channels_motifs_slices.iter().map(ThoTensor::clone_ref).collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(Self::new(self.id.clone(), tensors, self.metrics.clone(), self.quantiles.clone(), self.cache_name.clone())))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        let tensors = self.channels_motifs_slices.iter().map(|tensor| WorkerStageInput::ref_to_tensor(tensor, all_cuda_tensors, remove)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(self.id.clone(), tensors, self.metrics.clone(), self.quantiles.clone(), self.cache_name.clone()))
    }

}


#[derive(Debug, Named)]
pub struct MotifStatsConsumeStage<C>
where
    C: RunCtx + 'static,
{
    _ctx: PhantomData<C>,
}

static MOTIF_STATS_CONSUME_STAGE_INSTANCES: InstanceCache = InstanceCache::new();

impl<C> MotifStatsConsumeStage<C>
where
    C: RunCtx + 'static,
{
    async fn new(tho_device_idx: usize) -> Result<Arc<DynSyncStageEnum<MotifStatsConsumeStage<C>>>, ThoError> {
        MOTIF_STATS_CONSUME_STAGE_INSTANCES.get_or_try_insert_recreatable_with::<C, _, _, _, _>(tho_device_idx, GlobalCtx::global().cuda_activity_gate(tho_device_idx), GlobalCtx::global().is_ipc(), 0, move || async move {
            if GlobalCtx::global().is_ipc() {

                // Use shared consuming dispatch so tensor references are deleted only after success.
                Ok(Arc::new(DynSyncStageEnum::Proxy {
                    stage: Arc::new(ProxyStage::proxy_recreatable(MOTIF_STATS_CONSUME_ENDPOINT, Some(0.to_string()), tho_device_idx, Box::new(())).await?),
                    init: ()
                }))
            } else {
                Ok(Arc::new(DynSyncStageEnum::Local(Arc::new(MotifStatsConsumeStage::<C>::new_local()))))
            }
        }).await

    }

    fn new_local() -> Self {
        Self {  _ctx: PhantomData }
    }

    fn run_input(&self, input: &MotifStatsLoadedChannelsInput) -> Result<MotifStatsOutput, ThoError> {
        let channels_motifs_slices = merged_channels_tensor(input.id(), input.channels_motifs_slices())?;
        calc_metrics(self.out_id(input.id()), input.id(), input.metrics(), input.quantiles(), &*channels_motifs_slices.local()?)
    }
}

pub fn motif_stats_consume_stage_endpoint_init<'a>(_stage_id:Option<String>, _init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    use futures_util::FutureExt;
    async move {
        // The generic wrapper now owns both consume dispatch and successful tensor-key cleanup.
        Ok(Box::new(WorkerWrapper::<JsonIpcCodec, _, _, _>::new_consuming(Arc::new(MotifStatsConsumeStage::<WokerRunCtx>::new_local()), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}



impl<C> OutId for MotifStatsConsumeStage<C>
where
    C: RunCtx  + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl<C> Stage for MotifStatsConsumeStage<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = MotifStatsLoadedChannelsInput;
    type Out = MotifStatsOutput;
}


impl<C> ConsumingStage for MotifStatsConsumeStage<C>
where
    C: RunCtx  + 'static,
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
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<LoadedMotif>>, Ctx = C> + Cached + OutId + 'static,
{
    id: SynKey,
    store: Arc<dyn MotifsStore>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    pub motif_loader: Arc<ML>,
    _ctx: PhantomData<C>,
}

impl<ML, L, C> Debug for MotifStats<ML, L, C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<LoadedMotif>>, Ctx = C> + Cached + OutId + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MotifStats").finish()
    }
}

impl<ML, L, C> MotifStats<ML, L, C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<LoadedMotif>>, Ctx = C> + Cached + OutId + 'static,
{
    pub fn new(
        store: Arc<dyn MotifsStore>,
        loader: Arc<L>,
        ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
        motif_loader: Arc<ML>,
    ) -> Self {
        Self {
            id: (MOTIF_STATS, "MotifStats").into(),
            store,
            loader,
            ws,
            motif_loader,
            _ctx: PhantomData,
        }
    }
}

impl<ML, L, C> OutId for MotifStats<ML, L, C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<LoadedMotif>>, Ctx = C> + Cached + OutId + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<ML, L, C> Stage for MotifStats<ML, L, C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<LoadedMotif>>, Ctx = C> + Cached + OutId + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<MotifStatsInput>;
    type Out = Arc<MotifStatsOutput>;
}

impl<ML, L, C> AsyncStage for MotifStats<ML, L, C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<LoadedMotif>>, Ctx = C> + Cached + OutId + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStats::run"))]
    async fn run(self: Arc<Self>, input: Arc<MotifStatsInput>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let input_id = input.id().clone();
        let (source, metrics, quantiles) = (input.source().clone(), input.metrics().clone(), None);
        let stats = match source {
            MotifStatsSource::Motif(motif_loader_input) => {
                debug!("MotifStats::run: motif_loader_input={motif_loader_input:?}");
                let motif = t!(Arc::clone(&self.motif_loader)
                    .run_cached(
                        Arc::new(motif_loader_input),
                        ctx.branch(false),
                    )).await?;

                debug!("MotifStats::run: motif={motif:?}");
                let clc_stats_run = MotifStatsSyncStage::<C>::new(ctx.tho_device_idx()).await?;
                // Statistics retries need the loaded motif, not the output statistics ID.

                let clc_input = Arc::new(MotifStatsCalcInputWrapper::from_managed(input_id.clone(), Arc::clone(&motif), metrics.clone(), quantiles.clone()));
                let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()> = OOMHandler::new(
                    vec![
                        Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
                        Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("clc_stats"))),
                    ],
                    None,
                );
                t!(oom_handler.run(
                    &clc_stats_run,
                    clc_input,
                    ctx.branch(false),
                    WithRunCtxContext::new(
                        MaxParallelAndDumpToCpuAdjustableContex::new(
                            MaxParallelAdjustContextImpl::new(),
                            DumpToCpuContextImpl::new(),
                        ),
                    ),
                )).await?
            }
            MotifStatsSource::MergedMotifs(motif_loader_inputs) => {
                if motif_loader_inputs.is_empty() {
                    return Err(ThoError::new("MotifStats::run: merged motif inputs are empty"));
                }

                let mut expected_channel = None;
                let mut expected_window = None;
                let mut motifs = Vec::with_capacity(motif_loader_inputs.len());

                for (idx, motif_loader_input) in motif_loader_inputs.into_iter().enumerate() {
                    let motif = t!(Arc::clone(&self.motif_loader)
                        .run_cached(
                            Arc::new(motif_loader_input),
                            ctx.branch(false),
                        )).await?;

                    let meta = t!(motif.extra()).await?;
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

                let clc_stats_run = MotifStatsSyncStage::<C>::new(ctx.tho_device_idx()).await?;
                // Merged statistics read every loaded motif during each attempt.

                let clc_input = Arc::new(MotifStatsCalcInputWrapper::from_motifs(input_id.clone(), motifs.clone(), metrics.clone(), quantiles.clone()));
                let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()> = OOMHandler::new(
                    vec![
                        Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
                        Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("clc_stats"))),
                    ],
                    None,
                );
                t!(oom_handler.run(
                    &clc_stats_run,
                    clc_input,
                    ctx.branch(false),
                    WithRunCtxContext::new(
                        MaxParallelAndDumpToCpuAdjustableContex::new(
                            MaxParallelAdjustContextImpl::new(),
                            DumpToCpuContextImpl::new(),
                        ),
                    ),
                )).await?
            }
            MotifStatsSource::Random { max_slices, find_motifs_run_config_id, resources } => {
                debug!("MotifStats::run: MotifStatsSource::Random  max_slices={max_slices:?} find_motifs_run_config_id={find_motifs_run_config_id} resources={resources:?}");

                if max_slices < 2 {
                    return Err(ThoError::new(format!("MotifStats::run: max_slices must be >= 2, got {max_slices}")));
                }


                let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
                    find_motifs_run_config_id,
                    &t!(self.store.list_run_config_values(find_motifs_run_config_id)).await?,
                )?;
                let input_len_adjuster = find_motifs_run_config.input_len_adjuster();
                let resources = match resources {
                    Some(resources) => resources,
                    None => t!(self.store.list_random_resources(None)).await?,
                };
                let (slices, resources_map) = random_non_overlapping_slices(&resources, max_slices, find_motifs_run_config.window, input_len_adjuster)?;

                let (_window, channels_slices) = t!(load_random_slices(
                    find_motifs_run_config_id,
                    resources_map,
                    slices,
                    Arc::clone(&self.store),
                    Arc::clone(&self.loader),
                    ctx.branch(false),
                    &self.ws,
                    "motif_stats_random",
                )).await?;

                debug!("MotifStats::run: MotifStatsSource::Random  channels_slices={channels_slices:?}");
                let clc_stats_consume = MotifStatsConsumeStage::<C>::new(ctx.tho_device_idx()).await?;


                let clc_input = MotifStatsTmpCachedChannelsInput::new(input_id.clone(), channels_slices, metrics.clone(), quantiles.clone());


                let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()> = OOMHandler::new(
                    vec![
                        Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
                        Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("clc_stats"))),
                    ],
                    None,
                );
                t!(oom_handler.consume(
                    &clc_stats_consume,
                    clc_input,
                    ctx.branch(false),
                    WithRunCtxContext::new(
                        MaxParallelAndDumpToCpuAdjustableContex::new(
                            MaxParallelAdjustContextImpl::new(),
                            DumpToCpuContextImpl::new(),
                        ),
                    ),
                )).await?
            }
        };

        debug!("MotifStats::run: done input_id={input_id:?} stats={:?}", stats);

        Ok(Arc::new(stats))
    }
}

impl<ML, L, C> Adjustable for MotifStats<ML, L, C>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<LoadedMotif>>, Ctx = C> + Cached + OutId + 'static,
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
    use crate::motifs_store::MockMotifsStoreBuilder;
    use crate::stages::motif_loader::mock_motif_loader;
    use crate::stages::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
    use crate::stages::tensor_loader::mock_tensor_loader;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::{Tensor, kind::FLOAT_CPU};
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::traits::AsyncStage;
    use thoosa_store::RunConfigRow;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_random_stats_basic() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let ws = Arc::new(RwLock::new(crate::motifs_ws::mock_motifs_ws_simp(None, 3, 100, 1000)));
        let resources = vec![
            Resource::new(1, 1, "resource_a", 10),
            Resource::new(2, 1, "resource_b", 10),
            Resource::new(3, 1, "resource_c", 10),
            Resource::new(4, 1, "resource_d", 10),
        ];

        let run_config_id = 42;
        let motif_width = 6;
        let store =             MockMotifsStoreBuilder::new()
            .with_list_random_resources([(None, Ok(resources.clone()))])
            .with_run_config_descr([(
                run_config_id,
                Ok(RunConfigRow {
                    id: run_config_id,
                    r#type: "FindMotifsRunConfig".to_string(),
                    descr: "test find motifs".to_string(),
                    run_on: "2026-01-01 00:00:00".to_string(),
                }),
            )])
            .with_list_run_config_values([(
                run_config_id,
                Ok(FindMotifsRunConfig::new_with_running_mean(Some(run_config_id), 1, motif_width, None, None)
                    .to_properties_iter(None)
                    .collect()),
            )])
            .build();

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
            MotifLoader::new(Arc::clone(&store), Arc::clone(&loader), &ws).await?,
        ));

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(1)));
        let metric = Metrics::default();
        let input = Arc::new(MotifStatsInput::from_random(4, run_config_id, vec![metric.clone()]));

        let output = Arc::clone(&stage)
            .run(
                input,
                thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx))
                    .branch(false),
            )
            .await?;

        let t = output.channels_motifs_dists(&metric)?;
        debug!("t=\n{}", t);
        assert!(Tensor::from_slice(&[4.0825f32, 4.0825, 4.0825]).view((3, 1)).allclose(&*t.local()?, 0.02, 0.02, false));
        let size = t.size();

        assert_eq!(size, vec![3, 1]);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_merged_motifs_stats_source() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let ws = Arc::new(RwLock::new(crate::motifs_ws::mock_motifs_ws_simp(None, 1, 100, 1000)));
        let store = MockMotifsStoreBuilder::new().build();
        let loader = Arc::new(mock_tensor_loader::build().finish());
        let motif_input_0 = MotifLoaderInput::new((1u32, "motif_id").into(), Some(vec![100]));
        let motif_input_1 = MotifLoaderInput::new((2u32, "motif_id").into(), Some(vec![200]));
        let motif_loader = Arc::new(
            mock_motif_loader::build()
                .with_run_cached([
                    (
                        motif_input_0.clone(),
                        (
                            MotifMeta { channel: 0, find_motifs_run_config_id: 10, window: 3 },
                            Tensor::from_slice(&[0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0]).view((2, 3)),
                        ),
                    ),
                    (
                        motif_input_1.clone(),
                        (
                            MotifMeta { channel: 0, find_motifs_run_config_id: 10, window: 3 },
                            Tensor::from_slice(&[0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0]).view((2, 3)),
                        ),
                    ),
                ])
                .finish(),
        );
        let stage = Arc::new(MotifStats::new(Arc::clone(&store), loader, Arc::clone(&ws), motif_loader));
        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));
        let metric = Metrics::default();
        let input = Arc::new(MotifStatsInput::from_merged_motif_inputs(
            vec![motif_input_0, motif_input_1],
            vec![metric.clone()],
        )?);

        let output = Arc::clone(&stage)
            .run(
                input,
                thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(ctx))
                .branch(false),
            )
            .await?;

        let value = output.channel_value(&metric, 0)?;

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
