use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use cached_derive::Cached;
use log::debug;
use parking_lot::RwLock;
use thoosa_common::dims::AnyDims;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, StatsKeys};
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
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, DataId, MonoMetrics, Select, WithId};
use thoosa_shared::{mono_key, CopyToDevice, ThoError};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use thoosa_motifs::common_stats::{estimate_channel_stats_from_windows, Stats};

use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::motif_loader::load_motif_slices_;
use crate::tasks::random_slices_stats::random_non_overlapping_slices;

const DEFAULT_Q_LO: f64 = 0.05;
const DEFAULT_Q_HI: f64 = 0.95;
const DEFAULT_SCALE_Q: f64 = 0.50;
const DEFAULT_EPS: f64 = 1e-8;

#[derive(Debug, Clone)]
pub struct CommonStatsInput {
    find_motifs_run_id: RunId,
    id: StatsKeys,
    max_slices: u32,
}

impl CommonStatsInput {
    pub fn new(find_motifs_run_id: RunId, max_slices: u32, id: StatsKeys) -> Self {
        Self { find_motifs_run_id, id, max_slices }
    }

    pub fn max_slices(&self) -> u32 {
        self.max_slices
    }

    pub fn find_motifs_run_id(&self) -> RunId {
        self.find_motifs_run_id
    }
}

impl Input for CommonStatsInput {}

impl WithId for CommonStatsInput {
    type ID = StatsKeys;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for CommonStatsInput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        1usize.into()
    }

    fn device(&self) -> usize {
        0
    }
}

#[derive(Debug, Clone)]
pub struct CommonStatsOutput {
    id: StatsKeys,
    stats: Vec<Stats>,
}

impl CommonStatsOutput {
    pub fn new(id: StatsKeys, stats: Vec<Stats>) -> Self {
        Self { id, stats }
    }

    pub fn stats(&self) -> &Vec<Stats> {
        &self.stats
    }
}

impl WithId for CommonStatsOutput {
    type ID = StatsKeys;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for CommonStatsOutput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.stats.len().into()
    }

    fn device(&self) -> usize {
        0
    }
}

impl Output for CommonStatsOutput {}

#[async_trait]
impl CopyToDevice for CommonStatsOutput {
    async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
        Ok(self.clone())
    }
}

#[derive(Debug)]
struct CommonStatsCalcInput {
    id: StatsKeys,
    channels_slices: ThoTensor<AnyDims, DataId>,
}

impl CommonStatsCalcInput {
    fn new(id: StatsKeys, channels_slices: ThoTensor<AnyDims, DataId>) -> Self {
        Self { id, channels_slices }
    }

    fn channels_slices(&self) -> &ThoTensor<AnyDims, DataId> {
        &self.channels_slices
    }
}

impl WithId for CommonStatsCalcInput {
    type ID = StatsKeys;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Input for CommonStatsCalcInput {}

impl Data for CommonStatsCalcInput {
    type Metrics = <ThoTensor<AnyDims, DataId> as Data>::Metrics;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        <ThoTensor<AnyDims, DataId> as Data>::size(self.channels_slices())
    }

    fn device(&self) -> usize {
        self.channels_slices().device()
    }

    fn mem_size(&self) -> u64 {
        self.channels_slices().mem_size()
    }
}

#[async_trait]
impl CopyToDevice for CommonStatsCalcInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(Self { id: self.id.clone(), channels_slices: self.channels_slices.copy_to_device(dst_device).await? })
    }
}

impl Select<()> for CommonStatsCalcInput {
    type Output = CommonStatsCalcInput;

    fn subview(&self, _selector: ()) -> Result<Self::Output, ThoError> {
        Ok(CommonStatsCalcInput::new(self.id.clone(), ThoTensor::new(self.channels_slices.id().clone(), self.channels_slices.t_ref().shallow_clone())?))
    }
}

impl Adjustable for CommonStatsCalcInput {}

#[derive(Debug, Named)]
struct CommonStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    q_lo: f64,
    q_hi: f64,
    scale_q: f64,
    eps: f64,
    pd: PhantomData<C>,
}

impl<C> CommonStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn new(q_lo: f64, q_hi: f64, scale_q: f64, eps: f64) -> Self {
        Self {
            id: SynKey::new([mono_key!("CommonStatsSyncStage", "stage")]),
            q_lo,
            q_hi,
            scale_q,
            eps,
            pd: PhantomData,
        }
    }
}

impl<C> Adjustable for CommonStatsSyncStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for CommonStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl<C> Stage for CommonStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = CommonStatsCalcInput;
    type Out = CommonStatsOutput;
}

impl<C> SyncStage for CommonStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CommonStatsSyncStage::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let stats = estimate_channel_stats_from_windows(input.channels_slices().t_ref(), self.q_lo, self.q_hi, self.scale_q, self.eps)?;
        Ok(CommonStatsOutput::new(self.out_id(input.id()), stats))
    }
}

#[derive(Debug, Cached, Named)]
pub struct CommonStats<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    clc_common_stats: Arc<CommonStatsSyncStage<C>>,
    pd: PhantomData<C>,
}

impl<L, C> CommonStats<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(store: Arc<Box<dyn Store>>, loader: Arc<L>, ws: Arc<RwLock<Box<dyn Workspace>>>) -> Self {
        Self {
            store,
            loader,
            ws,
            clc_common_stats: Arc::new(CommonStatsSyncStage::<C>::new(DEFAULT_Q_LO, DEFAULT_Q_HI, DEFAULT_SCALE_Q, DEFAULT_EPS)),
            pd: PhantomData,
        }
    }
}

impl<L, C> OutId for CommonStats<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl<L, C> Stage for CommonStats<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type In<'a> = CommonStatsInput;
    type Out = CommonStatsOutput;
    type Ctx = C;
}

impl<L, C> AsyncStage for CommonStats<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CommonStats::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<CommonStatsInput>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let cache_area = ctx.cache_area();
        let input_id = input.id().clone();
        let (max_slices, find_motifs_run_id) = with_data!(ctx.branch(Some("common_stats_input"), false, false), input; |_ctx, input: &CommonStatsInput| {
            Ok::<_, ThoError>((input.max_slices(), input.find_motifs_run_id()))
        })?;

        if max_slices < 2 {
            return Err(ThoError::new(format!("CommonStats: max_slices must be >= 2, got {max_slices}")));
        }

        let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(find_motifs_run_id, &self.store.list_run_config_values(find_motifs_run_id).await?)?;
        let input_len_adjuster = find_motifs_run_config.input_len_adjuster();
        let resources = self.store.list_random_resources(None).await?;
        let (slices, resources_map) = random_non_overlapping_slices(&resources, max_slices, find_motifs_run_config.window, input_len_adjuster)?;

        debug!("CommonStats::run: find_motifs_run_id={find_motifs_run_id} slices={slices:?} resources_map={resources_map:?}");

        let (_window, channels_slices) = load_motif_slices_(
            DataId(0),
            find_motifs_run_id,
            resources_map,
            slices,
            Arc::clone(&self.store),
            Arc::clone(&self.loader),
            ctx.branch(Some("common_stats_load_slices"), false, false),
            &self.ws,
            "common_stats",
        )
        .await?;

        let slice_count = channels_slices.t_ref().size()[1];
        if slice_count < 2 {
            return Err(ThoError::new(format!("CommonStats: not enough slices, got {slice_count}")));
        }

        let common_stats_input = Arc::new(DevicesWrap::wrap(CommonStatsCalcInput::new(input_id, channels_slices), cache_area.into()));
        let required_ids: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
        let common_stats = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(PermitsAdjuster::new(MAX_PERMITS, *PermitStep::TWENTY))),
                Adjusters::RunContext(Box::new(DumpToCpu::new())),
            ],
            None,
        )
        .run(
            &self.clc_common_stats,
            common_stats_input,
            ctx.branch(Some("common_stats_sync"), false, false),
            Contexts::new(
                NoopInputContext(None),
                NoopRunnableContext {},
                PermitsAndDumpToCpuAdjustableContex::new(PermitsAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
            ),
        )
        .await?;

        Ok(Arc::new(DevicesWrap::wrap(common_stats, cache_area.into())))
    }
}
