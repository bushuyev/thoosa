use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use cached_derive::Cached;
use log::debug;
use parking_lot::RwLock;
use tch::Tensor;
use thoosa_common::dims::MotifSlices;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, SelectMotifsRunConfig};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_shared::runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::runner::permits::{PermitStep, MAX_PERMITS};
use thoosa_shared::traits::{Adjustable, Data, DataId, MonoMetrics, Select, WithId};
use thoosa_shared::{mono_key, CopyToDevice, ThoError};
use thoosa_store::Store;
use thoosa_with_data::with_data;

use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists, ChannelsMotifsSlices};
use crate::tasks::motif_loader::{load_motif_slices_, MotifLoaderInput, MotifMeta};
use crate::tasks::random_slices_stats::random_non_overlapping_slices;
use crate::tasks::slices_stats::{metric_scalar_for_motif, Aggr, Metrics};
use crate::tasks::stage_ids::{MOTIF_STATS, MOTIF_STATS_SYNC_STAGE};

const DEFAULT_Q_LO: f64 = 0.05;
const DEFAULT_Q_HI: f64 = 0.95;
const DEFAULT_SCALE_Q: f64 = 0.50;
const DEFAULT_EPS: f64 = 1e-8;

#[derive(Debug, Clone)]
pub enum MotifStatsSource {
    Motif(MotifLoaderInput),
    Random { max_slices: u32, find_motifs_run_id: RunId },
    ChannelsMotifsSlices(Arc<ThoTensor<ChannelsMotifsSlices, SynKey>>),
}

#[derive(Debug, Clone)]
pub struct MotifStatsInput {
    id: SynKey,
    source: MotifStatsSource,
    metrics: Vec<Metrics>,
    aggr: Aggr,
}

impl MotifStatsInput {
    pub fn from_motif_input(
        motif_loader_input: MotifLoaderInput,
        metrics: Vec<Metrics>,
        aggr: Aggr,
        select_motifs_run_config: Option<&SelectMotifsRunConfig>,
    ) -> Self {
        let metrics_sfx = metrics.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>().join(",");
        let aggr_sfx = format!("{aggr:?}");
        let id = match select_motifs_run_config.and_then(|cfg| cfg.id) {
            Some(select_motifs_run_id) => &motif_loader_input.id + mono_key!(select_motifs_run_id, format!("select_motifs_run_id={select_motifs_run_id}")),
            None => &(&motif_loader_input.id + mono_key!(metrics_sfx.clone(), format!("metrics={metrics_sfx}"))) + mono_key!(aggr_sfx.clone(), format!("aggr={aggr_sfx}")),
        };
        Self {
            id,
            source: MotifStatsSource::Motif(motif_loader_input),
            metrics,
            aggr,
        }
    }

    pub fn from_random(max_slices: u32, find_motifs_run_id: RunId, metrics: Vec<Metrics>, aggr: Aggr) -> Self {
        let metrics_sfx = metrics.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>().join(",");
        Self {
            id: SynKey::new([
                mono_key!("MotifStatsInput", "stage"),
                mono_key!(find_motifs_run_id, format!("find_motifs_run_id={find_motifs_run_id}")),
                mono_key!(max_slices, format!("max_slices={max_slices}")),
                mono_key!(metrics_sfx.clone(), format!("metrics={metrics_sfx}")),
                mono_key!(format!("{aggr:?}"), format!("aggr={aggr:?}")),
            ]),
            source: MotifStatsSource::Random { max_slices, find_motifs_run_id },
            metrics,
            aggr,
        }
    }


    pub fn source(&self) -> &MotifStatsSource {
        &self.source
    }

    pub fn metrics(&self) -> &Vec<Metrics> {
        &self.metrics
    }

    pub fn aggr(&self) -> Aggr {
        self.aggr
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
        match &self.source {
            MotifStatsSource::Motif(_) | MotifStatsSource::Random { .. } => 0,
            MotifStatsSource::ChannelsMotifsSlices(channels_motifs_slices) => channels_motifs_slices.mem_size(),
        }
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
            MotifStatsSource::Random { max_slices, find_motifs_run_id } => MotifStatsSource::Random { max_slices: *max_slices, find_motifs_run_id: *find_motifs_run_id },
            MotifStatsSource::ChannelsMotifsSlices(t) => MotifStatsSource::ChannelsMotifsSlices(Arc::new(t.copy_to_device(dst_device).await?)),
        };

        Ok(Self {
            id: self.id.clone(),
            source,
            metrics: self.metrics.clone(),
            aggr: self.aggr,
        })
    }
}

#[derive(Debug)]
pub struct MotifStatsCalcInput {
    id: SynKey,
    channels_motifs_slices: Arc<ThoTensor<ChannelsMotifsSlices, SynKey>>,
    metrics: Vec<Metrics>,
    aggr: Aggr,
}

impl MotifStatsCalcInput {
    pub fn new(
        id: SynKey,
        channels_motifs_slices: Arc<ThoTensor<ChannelsMotifsSlices, SynKey>>,
        metrics: Vec<Metrics>,
        aggr: Aggr,
    ) -> Self {
        Self { id, channels_motifs_slices, metrics, aggr }
    }

    pub fn channels_motifs_slices(&self) -> &Arc<ThoTensor<ChannelsMotifsSlices, SynKey>> {
        &self.channels_motifs_slices
    }

    pub fn metrics(&self) -> &Vec<Metrics> {
        &self.metrics
    }

    pub fn aggr(&self) -> Aggr {
        self.aggr
    }
}

impl Input for MotifStatsCalcInput {}

impl WithId for MotifStatsCalcInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for MotifStatsCalcInput {
    type Metrics = <ThoTensor<ChannelsMotifsSlices, SynKey> as Data>::Metrics;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        <ThoTensor<ChannelsMotifsSlices, SynKey> as Data>::size(self.channels_motifs_slices())
    }

    fn device(&self) -> usize {
        self.channels_motifs_slices.device()
    }

    fn mem_size(&self) -> u64 {
        self.channels_motifs_slices.mem_size()
    }
}

impl Select<()> for MotifStatsCalcInput {
    type Output = MotifStatsCalcInput;

    fn subview(&self, _selector: ()) -> Result<Self::Output, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            channels_motifs_slices: Arc::new(ThoTensor::new(self.channels_motifs_slices.id().clone(), self.channels_motifs_slices.t_ref().shallow_clone())?),
            metrics: self.metrics.clone(),
            aggr: self.aggr,
        })
    }
}

impl Adjustable for MotifStatsCalcInput {}

#[async_trait]
impl CopyToDevice for MotifStatsCalcInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            channels_motifs_slices: Arc::new(self.channels_motifs_slices.copy_to_device(dst_device).await?),
            metrics: self.metrics.clone(),
            aggr: self.aggr,
        })
    }
}

#[derive(Debug)]
pub struct MotifStatsOutput {
    id: SynKey,
    values: Vec<(Metrics, ThoTensor<ChannelsMotifsDists, SynKey>)>,
}

impl MotifStatsOutput {
    pub fn new(id: SynKey, values: Vec<(Metrics, ThoTensor<ChannelsMotifsDists, SynKey>)>) -> Self {
        Self { id, values }
    }

    pub fn metrics(&self) -> Vec<Metrics> {
        self.values.iter().map(|(metric, _)| *metric).collect()
    }

    fn metric_values(&self, metric: Metrics) -> Option<&ThoTensor<ChannelsMotifsDists, SynKey>> {
        self.values.iter().find_map(|(cur, values)| if *cur == metric { Some(values) } else { None })
    }

    pub fn channel_value(&self, metric: Metrics, channel: usize) -> Result<f32, ThoError> {
        let values = self.metric_values(metric).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: metric {metric:?} was not requested")))?;
        let channels = values.t_ref().size().first().copied().unwrap_or(0).max(0) as usize;
        if channel >= channels {
            return Err(ThoError::new(format!("MotifStatsOutput: channel index out of bounds: channel={channel} channels={channels}")));
        }
        Ok(values.t_ref().f_double_value(&[channel as i64, 0])? as f32)
    }

    pub fn channels_values(&self, metric: Metrics) -> Result<Vec<f32>, ThoError> {
        let values = self.metric_values(metric).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: metric {metric:?} was not requested")))?;
        let channels = values.t_ref().size().first().copied().unwrap_or(0).max(0) as i64;
        (0..channels).map(|channel| Ok(values.t_ref().f_double_value(&[channel, 0])? as f32)).collect()
    }

    pub fn channels_motifs_dists(&self, metric: Metrics) -> Result<ThoTensor<ChannelsMotifsDists, SynKey>, ThoError> {
        let values = self.metric_values(metric).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: metric {metric:?} was not requested")))?;
        ThoTensor::new(values.id().clone(), values.t_ref().shallow_clone())
    }
}

impl WithId for MotifStatsOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for MotifStatsOutput {}

impl Data for MotifStatsOutput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        self.values.len().into()
    }

    fn device(&self) -> usize {
        self.values.first().map(|(_, t)| t.device()).unwrap_or(0)
    }

    fn mem_size(&self) -> u64 {
        self.values.iter().map(|(_, t)| t.mem_size()).sum()
    }
}

#[async_trait]
impl CopyToDevice for MotifStatsOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        let mut values = Vec::with_capacity(self.values.len());
        for (metric, tensor) in &self.values {
            values.push((*metric, tensor.copy_to_device(dst_device).await?));
        }
        Ok(Self {
            id: self.id.clone(),
            values,
        })
    }
}

#[derive(Debug, Named)]
pub struct MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    q_lo: f64,
    q_hi: f64,
    scale_q: f64,
    eps: f64,
    _ctx: PhantomData<C>,
}

impl<C> MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(q_lo: f64, q_hi: f64, scale_q: f64, eps: f64) -> Self {
        Self {
            id: (MOTIF_STATS_SYNC_STAGE, "MotifStatsSyncStage").into(),
            q_lo,
            q_hi,
            scale_q,
            eps,
            _ctx: PhantomData,
        }
    }
}

impl<C> Default for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn default() -> Self {
        Self::new(DEFAULT_Q_LO, DEFAULT_Q_HI, DEFAULT_SCALE_Q, DEFAULT_EPS)
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
    type In<'a> = MotifStatsCalcInput;
    type Out = MotifStatsOutput;
}

impl<C> SyncStage for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStatsSyncStage::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let channels_motifs_slices = input.channels_motifs_slices().t_ref();
        let channels = channels_motifs_slices
            .size()
            .first()
            .copied()
            .ok_or_else(|| ThoError::new(format!("MotifStatsSyncStage::do_run: expected [channels, slices, window], got {:?}", channels_motifs_slices.size())))? as usize;

        let mut common_stats_cache: Option<Vec<thoosa_motifs::common_stats::Stats>> = None;
        let mut values = Vec::with_capacity(input.metrics().len());

        for metric in input.metrics() {
            let channel_values = if metric.is_common_stats_metric() {
                if common_stats_cache.is_none() {
                    common_stats_cache = Some(thoosa_motifs::common_stats::estimate_channel_stats_from_windows(
                        channels_motifs_slices,
                        self.q_lo,
                        self.q_hi,
                        self.scale_q,
                        self.eps,
                    )?);
                }
                let common_stats = common_stats_cache.as_ref().expect("set above");
                common_stats.iter().map(|stats| metric.common_stats_value(stats)).collect::<Result<Vec<_>, ThoError>>()?
            } else {
                let mut channel_values = Vec::with_capacity(channels);
                for channel_idx in 0..channels {
                    let motif_slices = channels_motifs_slices.f_get(channel_idx as i64)?;
                    channel_values.push(metric_scalar_for_motif(&motif_slices, *metric, input.aggr())?);
                }
                channel_values
            };
            let t = Tensor::f_from_slice(channel_values.as_slice())?.f_view([channels as i64, 1])?;
            values.push((
                *metric,
                ThoTensor::new(
                    SynKey::new([
                        mono_key!("motif_stats", "motif_stats"),
                        mono_key!(format!("{:?}", input.id()), format!("input_id={:?}", input.id())),
                        mono_key!(format!("{metric:?}"), format!("metric={metric:?}")),
                    ]),
                    t,
                )?,
            ));
        }

        Ok(MotifStatsOutput::new(self.out_id(input.id()), values))
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
    clc_stats: Arc<MotifStatsSyncStage<C>>,
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
            clc_stats: Arc::new(MotifStatsSyncStage::default()),
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
        let cache_area = input.cache_areas().clone();
        let input_id = input.id().clone();
        let (source, metrics, aggr) = with_data!(ctx.branch(Some("motif_stats_input"), false, false), input; |_ctx, input: &MotifStatsInput| {
            Ok::<_, ThoError>((input.source().clone(), input.metrics().clone(), input.aggr()))
        })?;

        let channels_motifs_slices = match source {
            MotifStatsSource::Motif(motif_loader_input) => {
                let motif = Arc::clone(&self.motif_loader)
                    .run_cached(
                        Arc::new(DevicesWrap::wrap(motif_loader_input, cache_area.clone())),
                        ctx.branch(Some("motif_stats_motif_loader"), false, false),
                    )
                    .await?;

                with_data!(ctx.branch(Some("motif_stats_motif_to_channels"), false, false), motif; |_ctx, motif: &thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>| {
                    let motif_2d = motif.data().t_ref().shallow_clone();
                    let channels_3d = motif_2d.f_unsqueeze(0)?;
                    Ok::<_, ThoError>(Arc::new(ThoTensor::new(motif.data().id().clone(), channels_3d)?))
                })?
            }
            MotifStatsSource::Random { max_slices, find_motifs_run_id } => {
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

                let (_window, channels_slices) = load_motif_slices_(
                    DataId(0),
                    find_motifs_run_id,
                    resources_map,
                    slices,
                    Arc::clone(&self.store),
                    Arc::clone(&self.loader),
                    ctx.branch(Some("motif_stats_random_load_slices"), false, false),
                    &self.ws,
                    "motif_stats_random",
                )
                .await?;

                Arc::new(ThoTensor::new(channels_slices.id().clone(), channels_slices.t_ref().shallow_clone())?)
            }
            MotifStatsSource::ChannelsMotifsSlices(t) => t,
        };

        let clc_input = Arc::new(DevicesWrap::wrap(MotifStatsCalcInput::new(input_id.clone(), channels_motifs_slices, metrics, aggr), cache_area.clone()));
        let required_ids: HashMap<&'static str, &std::collections::HashSet<&dyn thoosa_shared::traits::CachableIdTr>> = HashMap::new();
        let stats = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(PermitsAdjuster::new(MAX_PERMITS, *PermitStep::FIFTY))),
                Adjusters::RunContext(Box::new(DumpToCpu::new())),
            ],
            None,
        )
        .run(
            &self.clc_stats,
            clc_input,
            ctx.branch(Some("motif_stats_sync"), false, false),
            Contexts::new(
                NoopInputContext(None),
                NoopRunnableContext {},
                PermitsAndDumpToCpuAdjustableContex::new(PermitsAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
            ),
        )
        .await?;

        debug!("MotifStats::run: done input_id={input_id:?} stats.id={:?}", stats.id());

        Ok(Arc::new(DevicesWrap::wrap(stats, cache_area)))
    }
}

impl<ML, L, C> Adjustable for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{}
