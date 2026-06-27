use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use log::{debug, error, warn};
use parking_lot::RwLock;
use async_trait::async_trait;
use tch::Kind::Float;
use tch::Tensor;
use tokio::task::JoinHandle;
use tracing::Instrument;
use thoosa_common::dims::MotifSlices;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContext};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, InputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::runs::{SavableResult, Simple};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::{load_run_config, pad_to};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourceId, ResourcesConfig};
use thoosa_shared::{async_perf_log, async_write_lock, collect_futures_with_progress, mono_key, perf_log, sync_read_lock, FuturesOrdering, ThoError};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::rayon_pool::rayon_pool;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_store::{IdSlice, Store};
use thoosa_with_data::with_data;
use thoosa_common::dims::ChannelsData;
use crate::tasks::cross_dims::{ChannelsData2Selector, ChannelsMotifsDists};
use crate::tasks::cross_loader::{RangeOrSlice, ResourcesConfigInput};
use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
use crate::tasks::find_motifs::config_to_stage;
use crate::tasks::find_motifs_run_config::{Aggr, ExtendMotifsRunConfig, FindMotifsRunConfig, MaxSlices, MaxSlicesAbs, MaxSlicesByResources, Metrics};
use crate::tasks::motif_loader::{MotifLoaderInput, MotifMeta};
use crate::tasks::motif_stats::{MotifStats, MotifStatsInput, MotifStatsOutput};
use crate::tasks::resources_motifs_clc_dist::{ClcResourcesMotifsDist, ClcResourcesMotifsDistInput, ClcResourcesMotifsDistOutput};
use crate::tasks::slices_iter::{ResourceSlices, SlicesIter};
use crate::tasks::stacked_moitfs_loader::{StackedMotifsInput, StackedMotifsLoader, StackedMotifsOutput, StackedMotifsOutputId};
use crate::tasks::stage_ids::EXTEND_MOTIFS;
use thoosa_shared::tracked_await::track_await;

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct ExtendMotifsInput {
    id: SynKey,
    resources_input: ResourcesConfigInput<Simple>,
}

impl ExtendMotifsInput {


    pub fn new(resources_input: ResourcesConfigInput<Simple>) -> Self {
        Self {
            id: SynKey::new([mono_key!("ExtendMotifsInput", "stage")]),
            resources_input,
        }
    }

    pub(crate) fn resources_input(&self) -> &ResourcesConfigInput<Simple> {
        &self.resources_input
    }

}

impl Input for ExtendMotifsInput {}

impl WithId for ExtendMotifsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Clone, DummyData)]
#[dummy_data(clone)]
pub struct ExtendMotifsSavableResult {
    dummy_id: RunConfigId,
}

impl ExtendMotifsSavableResult {
    fn new() -> Self {
        Self { dummy_id: 0 }
    }
}

impl Output for ExtendMotifsSavableResult {}


impl WithId for ExtendMotifsSavableResult {
    type ID = RunConfigId;

    fn id(&self) -> &Self::ID {
        &self.dummy_id
    }
}

#[async_trait]
impl SavableResult for ExtendMotifsSavableResult {
    type R = u32;

    async fn save(
        &mut self,
        _store: &Arc<Box<dyn Store>>,
        _ws: &Arc<RwLock<Box<dyn Workspace>>>,
        _p: Self::P<'_>,
    ) -> Result<u32, ThoError> {
        Ok(0)
    }
}

pub struct ExtendMotifsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    run_configs: Vec<ExtendMotifsRunConfig>,
    stacked_motifs_loader:Arc<StackedMotifsLoader<ML, C>>,
    find_motifs_run_config_ids:Vec<RunConfigId>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    loader:Arc<L>,
    clc_resources_motifs_dist:Arc<ClcResourcesMotifsDist<C>>,
    motif_stats:Arc<MotifStats<ML, L, C>>,
}

impl<L, ML, C> Debug for ExtendMotifsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExtendMotifsStage").field("id", &self.id).field("run_configs", &self.run_configs).finish()
    }
}

impl<L, ML, C> WithId for ExtendMotifsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<L, ML, C> ExtendMotifsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub async fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<L>,
        motif_loader: Arc<ML>,
        run_configs: Vec<ExtendMotifsRunConfig>,
        find_motifs_run_config_ids: Vec<RunConfigId>,
        max_batch: i64,
        fail_if_size_bigger_than: Option<usize>,
    ) -> Result<Self, ThoError> {
        let channels = perf_log!(sync_read_lock!(ws)).channels() as usize;

        let motif_stats = Arc::new(MotifStats::new(
            Arc::clone(store),
            Arc::clone(&loader),
            Arc::clone(ws),
            Arc::clone(&motif_loader),
        ));


        Ok(Self {
            id: SynKey::new([mono_key!(EXTEND_MOTIFS, "extend_stage")]),
            run_configs,
            ws: Arc::clone(&ws),
            store: Arc::clone(&store),
            find_motifs_run_config_ids,
            loader:Arc::clone(&loader),
            stacked_motifs_loader: Arc::new(StackedMotifsLoader::new(Arc::clone(ws), Arc::clone(&motif_loader), Arc::clone(store))),
            clc_resources_motifs_dist: Arc::new(ClcResourcesMotifsDist::<C>::new(
                channels,
                #[cfg(test)]
                fail_if_size_bigger_than
            )),
            motif_stats
        })
    }

    #[inline]
    fn slices_selector_aggr_to_stats(aggr: Aggr) -> Aggr {
        match aggr {
            Aggr::Mean(mean_aggr) => Aggr::Mean(mean_aggr),
            Aggr::Min(min_aggr) => Aggr::Min(min_aggr),
            Aggr::Max(max_aggr) => Aggr::Max(max_aggr),
            Aggr::Quantile(quantile_aggr) => Aggr::Quantile(quantile_aggr),
        }
    }

    #[inline]
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SelectMotifsStage::load_stacked_motifs"))]
    async fn load_stacked_motifs(store:&Arc<Box<dyn Store>>, stacked_motifs_loader: &Arc<StackedMotifsLoader<ML, C>>, mut ctx_for_loader: C, find_motifs_run_config_ids: &Vec<RunConfigId>) -> Result<Vec<(Arc<FindMotifsRunConfig>, Arc<DevicesWrap<StackedMotifsOutput>>)>, ThoError> {
        let mut find_motifs_run_configs = Vec::with_capacity(find_motifs_run_config_ids.len());


        for find_motifs_run_config_id in find_motifs_run_config_ids {
            find_motifs_run_configs.push(Arc::new(track_await(load_run_config(store, *find_motifs_run_config_id)).await?));
        }
        let find_motifs_run_configs_len = find_motifs_run_configs.len();

        //for each FindMotifsRunConfigs X SelectMotifsRunConfig
        let futures = find_motifs_run_configs
            .into_iter()
            .map(|find_motifs_run_config| {
                let stacked_motifs_loader = Arc::clone(&stacked_motifs_loader);
                let ctx_for_run = ctx_for_loader.branch(false);

                async move {
                    let stacked_motifs_input = Arc::new(StackedMotifsInput::new(Arc::clone(&find_motifs_run_config))?);

                    match track_await(stacked_motifs_loader.run_cached(stacked_motifs_input, ctx_for_run)).await {
                        Ok(motif_data) => Ok::<_, ThoError>(Some((find_motifs_run_config, motif_data))),
                        Err(e) => {
                            //TODO switch to CPU if OOM ctx.to_cpu...
                            error!("Can't load select_motifs_run_config: e={e:?} select_motifs_run_config={find_motifs_run_config:?}");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let run_configs__stacked_motifs = track_await(collect_futures_with_progress(
            futures,
            find_motifs_run_configs_len,
            FuturesOrdering::Ordered,
            "SelectMotifsStage::load_stacked_motifs",
        )).await
            //gpt TODO!! collect_futures_with_progress drops errors and None results; missing find_motifs runs are silently skipped.
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        Ok(run_configs__stacked_motifs)
    }


    async fn clc_stacked_motifs_statistics(
        load_and_clc_dist: Arc<MotifStats<ML, L, C>>,
        motif_data: Arc<DevicesWrap<StackedMotifsOutput>>,
        slices_selector_metric: Metrics,
        mut ctx: C,
    ) -> Result<Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>, ThoError> {
        // TODO: move this into a dedicated stage and cache the resulting motifs stats separately from stacked motifs.
        let (channels_motif_ids, find_motifs_run_config, max_motif_slices) = with_data!(ctx.branch(false), motif_data;
            |_ctx, motif_data:&StackedMotifsOutput|{
                let channels_motif_ids = motif_data.motifs.extra().0.clone();
                let find_motifs_run_config = Arc::clone(&motif_data.motifs.extra().1);
                let max_motif_slices = motif_data.motifs.data().t_ref().size()[1] as usize;
                Ok::<_, ThoError>((channels_motif_ids, find_motifs_run_config, max_motif_slices))
        })?;

        let device_idx = ctx.tho_device_idx();
        let motif_ids = channels_motif_ids
            .iter()
            .flat_map(|ids| ids.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();
        let motif_ids_len = motif_ids.len();

        let futures = motif_ids
            .into_iter()
            .map(|motif_id| {
                let load_and_clc_dist = Arc::clone(&load_and_clc_dist);
                let slices_selector_metric = slices_selector_metric.clone();
                let mut ctx_for_motif = ctx.branch(false);
                async move {
                    let result = track_await(async {
                        let motif_loader_input = MotifLoaderInput::find_motif_slices_only(motif_id.clone());
                        let clc_input = Arc::new(MotifStatsInput::from_motif_input(motif_loader_input, vec![slices_selector_metric.clone()]));
                        let motif_stats = track_await(load_and_clc_dist.run(clc_input, ctx_for_motif.branch(false))).await?;
                        let motif_stat = motif_stats.channel_value(&slices_selector_metric, 0)?;
                        Ok::<_, ThoError>((motif_id, motif_stat))
                    }).await;
                    Ok::<_, ThoError>(result)
                }
            });

        let motif_stats = track_await(collect_futures_with_progress(
            futures,
            motif_ids_len,
            FuturesOrdering::Unordered,
            "SelectMotifsStage::clc_stacked_motifs_statistics",
        )).await
            .into_iter()
            .collect::<Result<Vec<_>, ThoError>>()?
            .into_iter()
            .collect::<HashMap<_, _>>();

        let channels_motifs_stats_tensors = channels_motif_ids
            .iter()
            .enumerate()
            .map(|(channel, channel_motif_ids)| {
                if channel_motif_ids.is_empty() {
                    return Ok(Tensor::f_zeros([max_motif_slices as i64], (Float, ThoDevice::from(device_idx).into()))?);
                }
                let channel_stats = channel_motif_ids
                    .iter()
                    .map(|motif_id| {
                        motif_stats
                            .get(motif_id)
                            .copied()
                            .ok_or_else(|| ThoError::new(format!("no motif stat found for motif_id={motif_id} channel={channel}")))
                    })
                    .collect::<Result<Vec<_>, ThoError>>()?;
                let channel_stats_t = Tensor::f_from_slice(channel_stats.as_slice())?.f_to_device(ThoDevice::from(device_idx).into())?;
                if channel_stats_t.size()[0] < max_motif_slices as i64 {
                    //same padding as in StackedMotifsLoader::run
                    Ok(pad_to(
                        &channel_stats_t, &[max_motif_slices as i64],
                        //this is to avoid selecting resource slices by MaskMaker::Le in ClcResourcesMotifsDist::do_run when cross-calculated with zero-padded motif slices giving 0 distance which would below any motif statistic; use f32::MIN because motifs_stats tensors are Float.
                        f32::MIN as f64
                    )?)
                } else {
                    Ok(channel_stats_t)
                }
            })
            .collect::<Result<Vec<_>, ThoError>>()?;

        let channels_motifs_stats_t = Tensor::f_stack(&channels_motifs_stats_tensors, 0)?;
        let find_motifs_run_config_id = find_motifs_run_config.id.ok_or_else(|| ThoError::new("SelectMotifsStage::clc_stacked_motifs_statistics: no id in find_motifs_run_config"))?;
        let motifs_stats = ThoTensor::new(
            SynKey::new([
                mono_key!("motifs_stats", "motifs_stats"),
                mono_key!(find_motifs_run_config_id, format!("find_motifs_run_config_id_{find_motifs_run_config_id}")),
                mono_key!(format!("{slices_selector_metric:?}"), "slices_selector_metric"),
            ]),
            channels_motifs_stats_t,
        )?;

        Ok(Arc::new(DevicesWrap::wrap(motifs_stats)))
    }


    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SelectMotifsStage::extend_motifs_for_run_config"))]
    async fn extend_motifs_for_run_config(
        &self,
        extend_motifs_run_config: &ExtendMotifsRunConfig,
        resources: Arc<Vec<Resource>>,
        num_channels: usize,
        total_size: usize,
        mut ctx: C,
    ) -> Result<Vec<HashMap<SynKey, HashSet<IdSlice>>>, ThoError> {

        let slices_selector_metric = Metrics::from_similarity_metric(
            extend_motifs_run_config.extend_config.slices_selector_metric,
            Self::slices_selector_aggr_to_stats(extend_motifs_run_config.extend_config.slices_selector_aggr),
        );
        // This stays inside the run-config loop.
        // Stacked motifs are cached by find_motifs_run_config_id, while stats depend on metric/aggr (including floats in aggr).
        // Stats must be computed separately so motifs and statistics can be cached independently.
        let run_configs__stacked_motifs = track_await(Self::load_stacked_motifs(
            &self.store,
            &self.stacked_motifs_loader,
            ctx.branch(true),
            &self.find_motifs_run_config_ids,
        )).await?;

        let channels_motifs_added_slices: Arc<Vec<tokio::sync::RwLock<HashMap<SynKey, HashSet<IdSlice>>>>> =
            Arc::new((0..num_channels).map(|_| tokio::sync::RwLock::new(HashMap::new())).collect());

        let run_configs__stacked_motifs_dbg = Arc::new(run_configs__stacked_motifs.clone());
        let run_configs__stacked_motifs_len = run_configs__stacked_motifs.len();
        let futures = run_configs__stacked_motifs
            .into_iter()
            .map(|(find_motifs_run_config, motif_data)| {
                let resources = Arc::clone(&resources);
                let find_motifs_run_config = Arc::clone(&find_motifs_run_config);
                let channels_motifs_added_slices = Arc::clone(&channels_motifs_added_slices);
                let clc_resources_motifs_dist = Arc::clone(&self.clc_resources_motifs_dist);
                let loader = Arc::clone(&self.loader);
                let ws = Arc::clone(&self.ws);
                let extend_motifs_run_config = extend_motifs_run_config.clone();
                let mut ctx_to_move = ctx.branch(true);
                let clc_dist = Arc::clone(&self.motif_stats);
                let slices_selector_metric = slices_selector_metric.clone();
                let run_configs__stacked_motifs_dbg = Arc::clone(&run_configs__stacked_motifs_dbg);

                async move {
                    match track_await(async {
                        let motifs_stats = track_await(Self::clc_stacked_motifs_statistics(
                            clc_dist,
                            Arc::clone(&motif_data),
                            slices_selector_metric.clone(),
                            ctx_to_move.branch(false),
                        )).await?;
                        track_await(Self::extend_with_one_find_run(
                            loader,
                            ws,
                            clc_resources_motifs_dist,
                            extend_motifs_run_config,
                            find_motifs_run_config,
                            motif_data,
                            motifs_stats,
                            resources,
                            channels_motifs_added_slices,
                            100000, //TODO to FixedExtenMotifsCOnfig
                            total_size,
                            ctx_to_move.branch(false),
                        )).await
                    }).await {
                        Ok(v) => Ok::<_, ThoError>(Some(v)),
                        Err(e) => {
                            error!("SelectMotifsStage::run: got error={e:?} while selecting slices for {run_configs__stacked_motifs_dbg:?}:");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let _ = track_await(collect_futures_with_progress(
            futures,
            run_configs__stacked_motifs_len,
            FuturesOrdering::Unordered,
            "SelectMotifsStage::extend_motifs_for_run_config",
        )).await;
        //gpt TODO!! per-run errors are swallowed; channels_motifs_added_slices may be incomplete with no signal to caller.

        if Arc::strong_count(&channels_motifs_added_slices) != 1 {
            return Err(ThoError::new(format!("can't unwrap motif_added_slices for {:?}", extend_motifs_run_config)))
        }

        Ok(Arc::into_inner(channels_motifs_added_slices).expect("can't unwrap motif_added_slices").into_iter().map(|lock| lock.into_inner()).collect())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SelectMotifsStage::extend_with_one_find_run"))]
    async fn extend_with_one_find_run(
        loader: Arc<L>,
        ws: Arc<RwLock<Box<dyn Workspace>>>,
        clc_resources_motifs_dist: Arc<ClcResourcesMotifsDist<C>>,
        extend_motifs_run_config: ExtendMotifsRunConfig,
        find_motifs_run_config: Arc<FindMotifsRunConfig>,
        motif_data: Arc<DevicesWrap<StackedMotifsOutput>>,
        motifs_stats: Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>,
        resources: Arc<Vec<Resource>>,
        channels_motifs_added_slices: Arc<Vec<tokio::sync::RwLock<HashMap<SynKey, HashSet<IdSlice>>>>>,
        max_batch: i64,//TODO rename max chunk size
        total_size: usize,
        mut ctx: C,
    ) -> Result<(), ThoError> {
        let num_resources = resources.len();
        let mut t_1_batch: Vec<_> = vec![];
        let mut t_1_batch_resources: Vec<_> = vec![];
        let mut t_1_batch_dim_0_size = 0;

        for (i_0, resource_0) in resources.iter().enumerate() {
            debug!("SelectMotifsStage::run: on i_0={i_0} resource_0={resource_0:?} num_resources={num_resources}");
            let configured_loader = config_to_stage(&loader, &find_motifs_run_config, &ws)?;

            let t_0 = track_await(Arc::clone(&configured_loader)
                .run_cached(Arc::new(resource_0.clone()), ctx.branch(false))).await?;

            t_1_batch_dim_0_size += t_0.size().dim_1();
            t_1_batch.push(t_0);
            t_1_batch_resources.push(resource_0.clone());

            if t_1_batch_dim_0_size < max_batch && i_0 < num_resources - 1 {
                continue;
            }

            let t_1_batch_len = t_1_batch.len();
            let batch_1_slice_lengths = t_1_batch.iter().map(|t| t.size().dim_1()).collect::<Vec<_>>();

            debug!("SelectMotifsStage::run: 0. find_motifs_run_config={find_motifs_run_config:?}");
            debug!("SelectMotifsStage::run: t_0_batch={t_1_batch:?} \n motif_data={motif_data:?}");

            let resource_ids = t_1_batch_resources.iter().map(|resource| resource.id()).collect::<Vec<_>>();
            let clc_resources_input = Arc::new(SelectMotifsInputSelectorWrap::new(
                SynKey::new([
                    mono_key!("SelectMotifsInputSelectorWrap", "stage"),
                    mono_key!(format!("{resource_ids:?}"), "resource_ids"),
                ]),
                t_1_batch,
                resource_ids,
                Arc::clone(&motif_data),
                Arc::clone(&motifs_stats),
            ));

            debug!("SelectMotifsStage::run: clc_resources_input={clc_resources_input:?}");
            debug!("SelectMotifsStage::run: 1. find_motifs_run_config={find_motifs_run_config:?}");
            let required_ids = HashMap::new();
            let channels = perf_log!(sync_read_lock!(ws))
                .channels();

            let ClcResourcesMotifsDistOutput {
                find_motifs_run_config_id,
                channel_masks_dists_ids,
                resources_iter,
                ..
            } = track_await(OOMHandler::new(
                vec![
                    Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                    Adjusters::RunContext(Box::new(DumpToCpu::new("clc_resources_motifs_dist"))),
                    Adjusters::Input(Box::new(SelectMotifsAdjuster::new(find_motifs_run_config.window as i64))),
                ],
                None,
            )
                .run(
                    &clc_resources_motifs_dist,
                    Arc::clone(&clc_resources_input),
                    ctx.branch(false),
                    Contexts::new(
                        SelectMotifsSplitterContext::new(
                            t_1_batch_len as i64,
                            batch_1_slice_lengths,
                            track_await(SelectMotifsClcOutputBuilder::new(
                                StackedMotifsOutputId::new(find_motifs_run_config.id.expect("find_motifs_run_config"), vec![]),
                                channels,
                                find_motifs_run_config.step as usize,
                            )).await,
                        ),
                        NoopRunnableContext {},
                        MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
                    ),
                )).await?;
            let max_slices = extend_motifs_run_config.extend_max_slices.as_ref().map(|max_slices| {
                match max_slices {
                    MaxSlices::ByResourcesCount( MaxSlicesByResources {k}) => {
                        *k * num_resources as u32
                    }
                    MaxSlices::Abs(MaxSlicesAbs{count}) => {
                        *count
                    }
                }
            }).unwrap_or(u32::MAX) as usize;

            debug!("SelectMotifsStage::run: channel_masks_dists_ids={channel_masks_dists_ids:?} find_motifs_run_config_id={find_motifs_run_config_id}");
            let resources_slices = Arc::new(resources_iter.clone().collect::<Vec<_>>());
            let channel_masks_dists_ids_len = channel_masks_dists_ids.len();
            let futures = channel_masks_dists_ids
                .into_iter()
                .enumerate()
                .map(|(channel, channel_mask_dists_ids)| {
                    let resources_slices = Arc::clone(&resources_slices);
                    let channels_motifs_added_slices = Arc::clone(&channels_motifs_added_slices);
                    async move {
                        let result = track_await(async {
                            if let Some(mask_dists_ids) = channel_mask_dists_ids {
                                debug!("SelectMotifsStage::run: channel={channel} mask_dists_ids={:?}", mask_dists_ids.1);
                                let mask_dists = track_await(mask_dists_ids.0.to_cpu()).await?;
                                let mask_size = mask_dists.mask_size_hint()?;
                                assert_eq!(mask_size.len(), 3, "SelectMotifsStage::run: expected 3D mask, got {mask_size:?}");
                                assert_eq!(mask_size[0], 1, "SelectMotifsStage::run: expected single-channel mask, got {mask_size:?}");
                                assert_eq!(mask_size[1] as usize, mask_dists_ids.1.len(), "SelectMotifsStage::run: motif/mask size mismatch mask={mask_size:?} motifs={}", mask_dists_ids.1.len());
                                assert_eq!(mask_size[2] as usize, resources_slices.len(), "SelectMotifsStage::run: resources/mask size mismatch mask={mask_size:?} resources={}", resources_slices.len());

                                let motif_count = mask_dists_ids.1.len();
                                let resources_slices_cloned = Arc::clone(&resources_slices);
                                let matches_by_motif = track_await(tokio::task::spawn_blocking(move || {
                                    tracing::info_span!(
                                        "extend_motifs::group_mask_matches",
                                        kind = "spawn_blocking",
                                        file = file!(),
                                        line = line!(),
                                        task = format!("group mask matches for channel={channel} motif_count={motif_count}"),
                                    ).in_scope(|| {
                                        use rayon::prelude::*;
                                        rayon_pool().install(|| {
                                            let mut match_indices = mask_dists.into_indices_iter()?;
                                            let mut matches = match_indices.par_bridge().map(|(mask_channel, motif_idx, slice_idx, _)| {
                                                (mask_channel as usize, motif_idx as usize, slice_idx as usize)
                                            }).collect::<Vec<_>>();
                                            matches.par_sort_unstable_by(|l, r| l.1.cmp(&r.1).then(l.2.cmp(&r.2)));

                                            let mut grouped_matches = vec![Vec::<IdSlice>::new(); motif_count];
                                            for (mask_channel, motif_idx, slice_idx) in matches {
                                                assert_eq!(mask_channel, 0, "SelectMotifsStage::run: expected single-channel mask index, got {mask_channel}");
                                                assert!(motif_idx < motif_count, "SelectMotifsStage::run: motif index out of bounds motif_idx={motif_idx} motif_count={motif_count}");
                                                assert!(slice_idx < resources_slices_cloned.len(), "SelectMotifsStage::run: resource slice index out of bounds slice_idx={slice_idx} resources_len={}", resources_slices_cloned.len());
                                                let slice = resources_slices_cloned[slice_idx];
                                                grouped_matches[motif_idx].push(IdSlice(0, slice.0, slice.1));
                                            }

                                            Ok::<_, ThoError>(grouped_matches)
                                        })
                                    })
                                })).await??;


                                let mut motif_added_slices_guard = track_await(channels_motifs_added_slices[channel].write()).await;

                                for (motif_id, matched_slices) in mask_dists_ids.1.into_iter().zip(matches_by_motif.into_iter()) {

                                    let slices = motif_added_slices_guard.entry(motif_id.clone()).or_insert(HashSet::new());
                                    let mut added_slices = 0;
                                    let mut skipped_slices = 0;
                                    for slice in matched_slices {
                                        if slices.len() > max_slices {
                                            continue;
                                        }
                                        if slices.insert(slice) {
                                            added_slices += 1;
                                        } else {
                                            skipped_slices += 1;
                                        }
                                    }

                                    debug!("SelectMotifsStage::run: motif_id={motif_id} skipped_slices={skipped_slices} added_slices={added_slices}");
                                }
                            }
                            Ok::<_, ThoError>(())
                        }).await;
                        Ok::<_, ThoError>(result)
                    }
                });

            track_await(collect_futures_with_progress(
                futures,
                channel_masks_dists_ids_len,
                FuturesOrdering::Unordered,
                "SelectMotifsStage::extend_with_one_find_run_channel_masks",
            )).await
                .into_iter()
                .collect::<Result<Vec<_>, ThoError>>()?;

            debug!("SelectMotifsStage::run: 2. find_motifs_run_config={find_motifs_run_config:?}");


            t_1_batch_dim_0_size = 0;
            t_1_batch = vec![];
            t_1_batch_resources.clear();
        }

        debug!("SelectMotifsStage::run: processed find_motifs_run_config={find_motifs_run_config:?} extend_motifs_run_config={extend_motifs_run_config:?}");
        Ok(())
    }

}

impl<L, ML, C> Stage for ExtendMotifsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<ExtendMotifsInput>;
    type Out = Arc<ExtendMotifsSavableResult>;
}

impl<L, ML, C> AsyncStage for ExtendMotifsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ExtendMotifsStage::run"))]
    async fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("ExtendMotifsStage::run: input={input:?}");

        let resources_config = input.resources_input().resources_config().clone();
        let resources = Arc::new(track_await(resources_config.to_resources(&self.store)).await?);
        if resources.is_empty() {
            warn!("ExtendMotifsStage::run: no labels resources");
            return Err(ThoError::new(format!("no resources found in {input:?}")))
        }
        let num_channels = perf_log!(sync_read_lock!(self.ws)).channels() as usize;
        let total_size = resources.iter().map(|resource| resource.length() as usize).sum::<usize>();
        let run_configs_len = self.run_configs.len();
        let run_configs = self.run_configs.clone();
        let futures = run_configs.into_iter().map(|extend_motifs_run_config| {
            let self_for_fut = Arc::clone(&self);
            let resources_for_fut = Arc::clone(&resources);
            let mut ctx_for_fut = ctx.branch(false);
            async move {
                let channels_motifs_added_slices = track_await(self_for_fut.extend_motifs_for_run_config(
                    &extend_motifs_run_config,
                    resources_for_fut,
                    num_channels,
                    total_size,
                    ctx_for_fut.branch(false),
                )).await?;
                debug!("ExtendMotifsStage::run: extend_motifs_for_run_config done");
                let extend_motifs_run_config_id = extend_motifs_run_config.id.expect("no extend_motifs_run_config_id");
                for (channel, motif_slices) in channels_motifs_added_slices.into_iter().enumerate() {
                    for (motif_id, slices) in motif_slices {
                        debug!("ExtendMotifsStage::run: motif_id={motif_id} slices.len={} channel={channel}", slices.len());
                        track_await(self_for_fut.store.add_slices_to_motif(extend_motifs_run_config_id, motif_id.vals[0].to_data_id().0, channel as i64, slices)).await?
                    }
                }
                Ok::<_, ThoError>(())
            }
        });
        let _ = track_await(collect_futures_with_progress(futures, run_configs_len, FuturesOrdering::Unordered, "ExtendMotifsStage::run")).await;
        Ok(Arc::new(ExtendMotifsSavableResult::new()))
    }
}

#[derive(Debug)]
struct SelectMotifsInputSelectorWrap {
    id: SynKey,
    batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
    resource_ids: Vec<ResourceId>,
    resource_offsets: Vec<u32>,
    stacked_motifs: Arc<DevicesWrap<StackedMotifsOutput>>,
    motifs_stats: Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>,
}

impl SelectMotifsInputSelectorWrap {
    pub fn new(
        id: SynKey,
        batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
        resource_ids: Vec<ResourceId>,
        stacked_motifs: Arc<DevicesWrap<StackedMotifsOutput>>,
        motifs_stats: Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>,
    ) -> Self {
        let resource_offsets = vec![0; resource_ids.len()];
        Self {
            id,
            batch,
            resource_ids,
            resource_offsets,
            stacked_motifs,
            motifs_stats,
        }
    }
}

impl Data for SelectMotifsInputSelectorWrap {
    type Metrics = i64;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.batch.len() as i64
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        self.batch.iter().map(|batch| batch.mem_size()).sum::<u64>()
            + self.stacked_motifs.mem_size()
            + self.motifs_stats.mem_size()
    }
}

impl WithId for SelectMotifsInputSelectorWrap {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl WithDataRef for SelectMotifsInputSelectorWrap {
    type O<'o> = ClcResourcesMotifsDistInput<'o>;

    fn with_data_ref<'a, R: Send + 'a, F, P>(&'a self, _to_device_idx: usize, permits_ctx: P, _trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        F: for <'f> FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move {
            let t_1_batch = self.batch.clone();
            let stacked_motifs = Arc::clone(&self.stacked_motifs);
            let motifs_stats = Arc::clone(&self.motifs_stats);
            with_data!(permits_ctx, [t_1_batch], stacked_motifs, motifs_stats;
                |
                 ctx,
                 batch_v:&[&ThoTensor<ChannelsData, SynKey>],
                 stacked_motifs:&StackedMotifsOutput,
                 motifs_stats:&ThoTensor<ChannelsMotifsDists, SynKey>
                |{

                    do_run(&ClcResourcesMotifsDistInput::new_with_offsets(
                        batch_v,
                        self.resource_ids.clone(),
                        self.resource_offsets.clone(),
                        stacked_motifs,
                        motifs_stats,
                    ), &ctx)

            })
        }
    }
}

impl WithSelectDataRef<SelectMotifsInputSelector> for SelectMotifsInputSelectorWrap {
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(&'a self, _to_device_idx: usize, mut ctx: P, selector: SelectMotifsInputSelector, _trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        P: RunCtx + 'a,
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a
    {
        async move {

            debug!("SelectMotifsInputSelectorWrap::with_selected_data_ref: selector={selector:?}");

            let (t_1_batch, resource_ids, resource_offsets) = match selector.batch_selector {
                RangeOrSlice::Full => (self.batch.clone(), self.resource_ids.clone(), self.resource_offsets.clone()),
                RangeOrSlice::Range(range) => {
                    let range = range.start as usize..range.end as usize;
                    (
                        self.batch[range.clone()].to_vec(),
                        self.resource_ids[range.clone()].to_vec(),
                        self.resource_offsets[range].to_vec(),
                    )
                }
                RangeOrSlice::Slice((i, range)) => {
                    let t = self.batch.get(i).ok_or_else(|| ThoError::new(format!("SelectMotifsInputSelectorWrap: invalid slice idx {i}")))?;
                    let channels = t.size().dim_0();
                    let selector = ChannelsData2Selector([0, range.start].into()..[channels, range.end].into());
                    let selected = Arc::new(DevicesWrap::wrap(track_await(t.with_selected_data(ctx.tho_device_idx(), ctx.branch(false), selector, |data, _ctx| {
                        Ok(data)
                    })).await?));
                    let offset = self.resource_offsets.get(i).copied().unwrap_or(0);
                    let range_offset = u32::try_from(range.start).map_err(|_| ThoError::new(format!("invalid batch slice start: {}", range.start)))?;
                    (vec![selected], vec![self.resource_ids[i]], vec![offset + range_offset])
                }
            };

            let stacked_motifs_output = Arc::clone(&self.stacked_motifs);
            let motifs_stats = Arc::clone(&self.motifs_stats);

            with_data!(ctx, [t_1_batch], stacked_motifs_output, motifs_stats;
                |
                 ctx,
                 batch_v:&[&ThoTensor<ChannelsData, SynKey>],
                 stacked_motifs_output:&StackedMotifsOutput,
                 motifs_stats:&ThoTensor<ChannelsMotifsDists, SynKey>
                |{
                    let inpt = ClcResourcesMotifsDistInput::new_with_offsets(
                        batch_v,
                        resource_ids,
                        resource_offsets,
                        stacked_motifs_output,
                        motifs_stats,
                    );

                    do_run(&inpt, &ctx)
            })
        }
    }
}


#[derive(Debug)]
struct SelectMotifsSavableResultBuilder {
    id:StackedMotifsOutputId,
    output:Vec<JoinHandle<Result<ClcResourcesMotifsDistOutput, ThoError>>>,//TODO TempCache
    channels:u32,
    step:usize
}

impl SelectMotifsSavableResultBuilder {
    pub async fn new(id:StackedMotifsOutputId, channels:u32, step:usize) -> Self {

        Self {
            id,
            output: vec![],
            channels,
            step
        }
    }
}

impl OutputBuilder for SelectMotifsSavableResultBuilder {
    type O = ClcResourcesMotifsDistOutput;

    async fn add_partial_output(&mut self, o: Self::O) {
        debug!("SelectMotifsSavableResultBuilder::add_partial_output: o={o:?}");

        self.output.push (tokio::spawn(async {
            track_await(o.to_cpu()).await
        }.instrument(tracing::info_span!(
            "extend_motifs::partial_output_to_cpu",
            kind = "spawn",
            file = file!(),
            line = line!(),
            task = "move partial extended motif output to CPU",
        ))));
    }

    async fn build(self) -> Result<Self::O, ThoError> {

        let mut resource_ids_sizes: Vec<ResourceSlices> = vec![];


        let output:Vec<ClcResourcesMotifsDistOutput> = track_await(futures::future::try_join_all(self.output)).await?
            .into_iter()
            .collect::<Result<Vec<_>, ThoError>>()?;

        // partial output contains masks for
        // motif_c0_0, motif_c0_1... motif_c0_m.
        // motif_c1_0, motif_c1_1... motif_c1_n
        // x
        // unfolded resource_partN_0, unfolded resource_partN_1, ...
        //
        //if a chanel has no motifs then all partial outputs will contain None and corresponding vec in channel_masks will be empty.

        let channel_masks_motif_ids = output.into_iter().try_fold(
            (
                (0..self.channels).map(|c|vec![]).collect::<Vec<Vec<Tensor>>>(),
                (0..self.channels).map(|c|vec![]).collect::<Vec<Vec<SynKey>>>()
            ),

            |mut channel_masks, partial_output| -> Result<_, ThoError> {

                debug!("SelectMotifsSavableResultBuilder::build: partial_output={partial_output:?}");

                assert_eq!(partial_output.channel_masks_dists_ids.len() as u32, self.channels);

                for (c, mask_op) in partial_output.channel_masks_dists_ids.into_iter().enumerate(){
                    debug!("SelectMotifsSavableResultBuilder::build: c={c} mask_op={mask_op:?}");

                    if let Some(mask) = mask_op {
                        let mask_t = mask.0.unpack()?.0;

                        debug!("SelectMotifsSavableResultBuilder::build: c={c} mask_t=\n{mask_t}");

                        channel_masks.0[c].push(mask_t);
                        channel_masks.1[c] = mask.1;//TODO collect dists, needed for dumping plots only
                    }
                };

                resource_ids_sizes.extend(partial_output.resources_iter.resource_ids_sizes);

                Ok(channel_masks)
            })?;

        let channel_masks_dists_ids = channel_masks_motif_ids.0.into_iter().zip(channel_masks_motif_ids.1).map(|channel_masks_motif_ids|{
            debug!("SelectMotifsSavableResultBuilder::build: channel_masks_motif_ids={channel_masks_motif_ids:?}");

            if channel_masks_motif_ids.0.is_empty() {
                Ok(None)
            } else {
                Ok(Some(
                    (
                        DistsMask::new(
                            Tensor::f_concat(&channel_masks_motif_ids.0, 2)?, //1 dim - resources
                            None,
                            DistsMaskType::AsIs,
                        )?,
                        channel_masks_motif_ids.1
                    ),
                ))
            }

        })
            .collect::<Result<Vec<_>, ThoError>>()?;

        Ok(ClcResourcesMotifsDistOutput::new(
            self.id.find_motifs_run_config_id,
            channel_masks_dists_ids,
            SlicesIter::new_with_offsets(self.step, resource_ids_sizes)
        ))
    }
}

#[derive(Debug)]
struct SelectMotifsClcOutputBuilder {
    inner: SelectMotifsSavableResultBuilder,
}

impl SelectMotifsClcOutputBuilder {
    pub async fn new(id: StackedMotifsOutputId, channels: u32, step: usize) -> Self {
        Self {
            inner: track_await(SelectMotifsSavableResultBuilder::new(id, channels, step)).await,
        }
    }
}

impl OutputBuilder for SelectMotifsClcOutputBuilder {
    type O = ClcResourcesMotifsDistOutput;

    async fn add_partial_output(&mut self, o: Self::O) {
        track_await(self.inner.add_partial_output(o)).await;
    }

    async fn build(self) -> Result<Self::O, ThoError> {
        track_await(self.inner.build()).await
    }
}

#[derive(Debug, Clone, Default)]
struct SelectMotifsInputSelector {
    batch_selector: RangeOrSlice,
}

impl Adjustable for SelectMotifsInputSelector {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Batch1SplitMode {
    Range,
    Slice,
}

#[derive(Debug)]
struct SliceAdjusterContext {
    input_len: i64,
    split_len: i64,
    done_to: i64,
}

impl SliceAdjusterContext {
    fn new() -> Self {
        Self {
            input_len: 0,
            split_len: 0,
            done_to: 0,
        }
    }

    fn reset(&mut self, input_len: i64) {
        self.input_len = input_len;
        self.done_to = 0;
        if self.split_len == 0 {
            self.split_len = input_len;
        }
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }
}

impl SplitsAdjusterContext for SliceAdjusterContext {
    type Metrics = i64;

    fn input_len(&self) -> &Self::Metrics {
        &self.input_len
    }

    fn split_len(&self) -> &Self::Metrics {
        &self.split_len
    }

    fn done_to(&self) -> &Self::Metrics {
        &self.done_to
    }

    fn split_len_mut(&mut self) -> &mut Self::Metrics {
        &mut self.split_len
    }
}

#[derive(Debug)]
struct SelectMotifsSplitterContext<'a, ST>
where
    ST: Stage<Out=ClcResourcesMotifsDistOutput> + Adjustable + 'a,
    ST::In<'a>: Data<Metrics = i64>,
{
    input_len: i64,
    split_len: i64,
    done_to: i64,
    output_builder: SelectMotifsClcOutputBuilder,
    batch_1_mode: Batch1SplitMode,
    slice_ctx: SliceAdjusterContext,
    slice_resource_idx: usize,
    batch_1_slice_lengths: Vec<i64>,
    pd: PhantomData<&'a ST>,
}

impl <'a, ST> SelectMotifsSplitterContext<'a, ST>
where
    ST: Stage<Out=ClcResourcesMotifsDistOutput> + Adjustable + 'a,
    ST::In<'a>: Data<Metrics = i64>,
{
    pub fn new(batch_1_len: <<ST as Stage>::In<'a> as Data>::Metrics, batch_1_slice_lengths: Vec<i64>, output_builder: SelectMotifsClcOutputBuilder) -> Self {
        debug!("SelectMotifsSplitterContext::new: batch_1_len={batch_1_len:?}");

        Self {
            input_len: batch_1_len,
            split_len: batch_1_len,
            done_to: 0,
            output_builder,
            batch_1_mode: Batch1SplitMode::Range,
            slice_ctx: SliceAdjusterContext::new(),
            slice_resource_idx: 0,
            batch_1_slice_lengths,
            pd: Default::default(),
        }
    }

    fn start_slice_mode(&mut self) -> bool {
        if self.batch_1_mode == Batch1SplitMode::Slice {
            return true;
        }

        self.batch_1_mode = Batch1SplitMode::Slice;
        self.slice_resource_idx = self.done_to as usize;

        let input_len = self
            .batch_1_slice_lengths
            .get(self.slice_resource_idx)
            .copied()
            .unwrap_or(0);
        self.slice_ctx.reset(input_len);

        input_len > 0
    }

    fn advance_slice_resource(&mut self) {
        self.done_to += 1;
        if self.done_to >= self.input_len {
            return;
        }

        self.slice_resource_idx = self.done_to as usize;
        let input_len = self
            .batch_1_slice_lengths
            .get(self.slice_resource_idx)
            .copied()
            .unwrap_or(0);
        self.slice_ctx.reset(input_len);
    }
}

impl <'a, ST> SplitsAdjusterContext for SelectMotifsSplitterContext<'a, ST>
where
    ST: Stage<Out=ClcResourcesMotifsDistOutput> + Adjustable + 'a,
    ST::In<'a>: Data<Metrics = i64>,
{
    type Metrics = <<ST as Stage>::In<'a> as Data>::Metrics;

    fn input_len(&self) -> &Self::Metrics {
        &self.input_len
    }

    fn split_len(&self) -> &Self::Metrics {
        &self.split_len
    }

    fn done_to(&self) -> &Self::Metrics {
        &self.done_to
    }

    fn split_len_mut(&mut self) -> &mut Self::Metrics {
        &mut self.split_len
    }
}

impl <'a, 'z, ST> InputContext<'a, ST> for SelectMotifsSplitterContext<'z, ST>
where
    ST: Stage<Out=ClcResourcesMotifsDistOutput> + Adjustable + 'a,
    ST::In<'z>: Data<Metrics = i64>,
{
    async fn add_partial_output(&mut self, out: ST::Out) {
        debug!("SelectMotifsSplitterContext::add_partial_output: out={out:?}");
        track_await(self.output_builder.add_partial_output(out)).await;
        match self.batch_1_mode {
            Batch1SplitMode::Range => {
                self.done_to += self.split_len;
            }
            Batch1SplitMode::Slice => {
                self.slice_ctx.done_to += self.slice_ctx.split_len;
                if self.slice_ctx.all_done() {
                    self.advance_slice_resource();
                }
            }
        }
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }

    async fn build_final_output(self) -> Result<ST::Out, ThoError> {
        track_await(self.output_builder.build()).await
    }
}

#[derive(Debug)]
struct SelectMotifsAdjuster {
    batch_1_adjuster: SplitsAdjuster<Range<i64>, i64>,
    batch_1_slice_adjuster: SplitsAdjuster<Range<i64>, i64>,
}

impl SelectMotifsAdjuster {
    pub fn new(min_tensor_split_len: i64) -> Self {
        Self {
            batch_1_adjuster: SplitsAdjuster::new(1),
            batch_1_slice_adjuster: SplitsAdjuster::new(min_tensor_split_len),
        }
    }
}

#[async_trait]
impl <'a, ST> Adjuster<SelectMotifsSplitterContext<'a, ST>> for SelectMotifsAdjuster
where
    ST: Stage<Out=ClcResourcesMotifsDistOutput> + Adjustable + 'a,
    ST::In<'a>: Data<Metrics = i64>,
{
    type A = SelectMotifsInputSelector;

    async fn adjust(&self, mut adjustable: Self::A, adjust_context: &mut SelectMotifsSplitterContext<'a, ST>) -> Result<Self::A, AdjusterError<Self::A>> {
        match adjust_context.batch_1_mode {
            Batch1SplitMode::Range => {
                let result = track_await(self.batch_1_adjuster.adjust(0i64..0, adjust_context)).await;
                debug!("SelectMotifsAdjuster::adjust: adjust_context={adjust_context:?} result={result:?}");
                match result {
                    Ok(range) => {
                        adjustable.batch_selector = RangeOrSlice::Range(range);
                        Ok(adjustable)
                    }
                    Err(_) => Err(AdjusterError::AdjustFailed(adjustable)),
                }
            }
            Batch1SplitMode::Slice => {
                let result = track_await(self.batch_1_slice_adjuster.adjust(0i64..0, &mut adjust_context.slice_ctx)).await;
                debug!("SelectMotifsAdjuster::adjust: slice adjust_context={adjust_context:?} result={result:?}");
                match result {
                    Ok(range) => {
                        adjustable.batch_selector = RangeOrSlice::Slice((adjust_context.slice_resource_idx, range));
                        Ok(adjustable)
                    }
                    Err(_) => Err(AdjusterError::AdjustFailed(adjustable)),
                }
            }
        }
    }

    fn degrade(&self, adjust_context: &mut SelectMotifsSplitterContext<'a, ST>) -> bool {
        match adjust_context.batch_1_mode {
            Batch1SplitMode::Range => {
                if self.batch_1_adjuster.degrade(adjust_context) {
                    return true;
                }
                if !adjust_context.start_slice_mode() {
                    return false;
                }
                self.batch_1_slice_adjuster.degrade(&mut adjust_context.slice_ctx)
            }
            Batch1SplitMode::Slice => self.batch_1_slice_adjuster.degrade(&mut adjust_context.slice_ctx),
        }
    }
}



#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use log::debug;
    use parking_lot::RwLock;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::Tensor;
    use thoosa_common::runner::adjusters::splits::OutputBuilder;
    use thoosa_common::runner::common::cachable::{DevicesWrap, WithDataRef};
    use thoosa_common::runner::common::{mock_run_ctx, MockRunCtx, RunCtx};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::to_vec_f32;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
    use crate::tasks::extend_motifs::{ExtendMotifsStage, SelectMotifsSavableResultBuilder};
    use crate::tasks::find_motifs_run_config::{FindMotifsRunConfig, Metrics};
    use crate::tasks::motif_loader::{mock_motif_loader, MotifLoaderInput, MotifMeta};
    use crate::tasks::motif_loader::mock_motif_loader::MockMotifLoader;
    use crate::tasks::motif_stats::MotifStats;
    use crate::tasks::resources_motifs_clc_dist::ClcResourcesMotifsDistOutput;
    use crate::tasks::slices_iter::SlicesIter;
    use crate::tasks::stacked_moitfs_loader::{StackedMotifsOutput, StackedMotifsOutputId};
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::tensor_loader::mock_tensor_loader::MockTensorLoader;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_clc_stacked_motifs_statistics_duplicates_and_pads_with_min() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

        let find_motifs_run_config_id = 7u32;
        let find_motifs_run_config = Arc::new(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 1, 3, None, None));

        let channels_motif_ids = vec![
            vec![(11u32, "motif_id").into(), (11u32, "motif_id").into(), (12u32, "motif_id").into()],
            vec![(13u32, "motif_id").into()],
        ];
        let motifs_t = Tensor::zeros([2, 4, 3], FLOAT_CPU);
        let motifs = ThoTensor::new(SynKey::new([mono_key!("motifs", "test_motifs")]), motifs_t)?;
        let stacked_motifs_output = StackedMotifsOutput::new(
            SynKey::new([mono_key!("StackedMotifsOutput", "test_stacked_motifs_output")]),
            DataWithExtra::new(motifs, (channels_motif_ids, Arc::clone(&find_motifs_run_config), vec![])),
        );
        let motif_data = Arc::new(DevicesWrap::wrap(stacked_motifs_output));

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (
                MotifLoaderInput::find_motif_slices_only((11u32, "motif_id").into()),
                (
                    MotifMeta { channel: 0, find_motifs_run_config_id, window: 3 },
                    Tensor::from_slice(&[0.0f32, 0.0, 0.0, 3.0, 0.0, 0.0]).view([2, 3]),
                ),
            ),
            (
                MotifLoaderInput::find_motif_slices_only((12u32, "motif_id").into()),
                (
                    MotifMeta { channel: 0, find_motifs_run_config_id, window: 3 },
                    Tensor::from_slice(&[0.0f32, 0.0, 0.0, 4.0, 0.0, 0.0]).view([2, 3]),
                ),
            ),
            (
                MotifLoaderInput::find_motif_slices_only((13u32, "motif_id").into()),
                (
                    MotifMeta { channel: 1, find_motifs_run_config_id, window: 3 },
                    Tensor::from_slice(&[0.0f32, 0.0, 0.0, 5.0, 0.0, 0.0]).view([2, 3]),
                ),
            ),
        ]).finish());
        let store = Arc::new(MockStoreBuilder::new().build());
        let loader = Arc::new(mock_tensor_loader::build().finish());
        let clc_dist = Arc::new(MotifStats::new(store, loader, Arc::clone(&ws), motif_loader));

        let mut ctx = mock_run_ctx::default(1);
        let metric = Metrics::default();
        let motifs_stats = ExtendMotifsStage::<MockTensorLoader, MockMotifLoader, MockRunCtx>::clc_stacked_motifs_statistics(
            clc_dist,
            motif_data,
            metric,
            ctx.branch(false),
        )
            .await?;

        let mut ctx_for_data = ctx.branch(false);
        let motifs_stats_t: Tensor = motifs_stats
            .with_data_ref(0, ctx_for_data, None, |motifs_stats: &ThoTensor<_, _>, _ctx| Ok(motifs_stats.t_ref().shallow_clone()))
            .await?;
        assert_eq!(motifs_stats_t.size(), vec![2, 4]);

        let values = to_vec_f32(&motifs_stats_t);
        assert!((values[0] - 3.0).abs() < 1e-6);
        assert!((values[1] - 3.0).abs() < 1e-6);
        assert!((values[2] - 4.0).abs() < 1e-6);
        assert!((values[4] - 5.0).abs() < 1e-6);
        for idx in [3usize, 5, 6, 7] {
            assert!(values[idx].is_infinite() || values[idx] < -1.0e30, "expected padded value to be very negative at idx={idx}, got {}", values[idx]);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_output_builder() -> Result<(), ThoError>  {
        GlobalCtx::init_devices(2);
        let num_channels = 3;
        let mut output_builder = Box::new(SelectMotifsSavableResultBuilder::new(StackedMotifsOutputId::new(0, vec![]), num_channels as u32, 50).await);

        let mask_0 = Tensor::rand(&[3, 5, 7], FLOAT_CUDA).lt(0.5);
        let expected_mask_0 = mask_0.f_to_device(tch::Device::Cpu)?;
        let mask_0_size = mask_0.size();
        output_builder.add_partial_output(
            ClcResourcesMotifsDistOutput::new(
                0,
                vec![
                    Some(
                        (
                            DistsMask::new(
                                //3 channels, 5 motifs, 7 resource slices
                                mask_0,
                                None,
                                DistsMaskType::MaskPacked(mask_0_size, None),
                            )?,
                            vec![
                                (0u32, "motif_id").into()
                            ]
                        )
                    ),
                    None,
                    None
                ],
                SlicesIter::new(100, vec![
                    (0u32, 100),
                    (1u32, 100)
                ])
            )
        ).await;

        let mask_1 = Tensor::rand(&[3, 5, 7], FLOAT_CUDA).lt(0.5);
        let expected_mask_1 = mask_1.f_to_device(tch::Device::Cpu)?;
        let mask_1_size = mask_1.size();
        output_builder.add_partial_output(
            ClcResourcesMotifsDistOutput::new(
                0,
                vec![
                    Some(
                        (
                            DistsMask::new(
                                //3 channels, 5 motifs, 7 resource slices
                                mask_1,
                                None,
                                DistsMaskType::MaskPacked(mask_1_size, None),
                            )?,
                            vec![
                                (0u32, "motif_id").into()
                            ]
                        )
                    ),
                    None,
                    None
                ],
                SlicesIter::new(100, vec![
                    (0u32, 100),
                    (1u32, 100)
                ])
            )
        ).await;

        let ClcResourcesMotifsDistOutput {
            channel_masks_dists_ids,
            ..
        } = output_builder.build().await?;

        assert_eq!(channel_masks_dists_ids.len(), num_channels);
        let mut channel_masks_dists_ids = channel_masks_dists_ids.into_iter();
        let expected_mask = Tensor::f_concat(&[expected_mask_0, expected_mask_1], 2)?;
        let channel_0 = channel_masks_dists_ids.next().expect("channel 0").expect("channel 0 mask");
        let (mask, _) = channel_0.0.unpack()?;
        debug!("mask=\t{mask}");
        assert_eq!(mask.size(), vec![3, 5, 14]);
        assert_eq!(mask, expected_mask);
        assert!(channel_masks_dists_ids.next().expect("channel 1").is_none());
        assert!(channel_masks_dists_ids.next().expect("channel 2").is_none());


        Ok(())
    }

}
