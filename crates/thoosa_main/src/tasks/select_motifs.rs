use std::collections::{HashMap, HashSet};
use std::f64;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use async_trait::async_trait;
use log::{debug, error, info, trace, warn};
use parking_lot::RwLock;
use tch::Device::Cpu;
use tch::Kind::{Float, Uint8};
use tch::{IndexOp, Kind, Tensor};
use tch::kind::FLOAT_CUDA;
use tokio::io::AsyncWriteExt;
use tokio::sync::OwnedRwLockWriteGuard;
use tokio::task::{JoinHandle, JoinSet};
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContext};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, InputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::runner::tmp_cache::TmpCached;
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::cuda_utils::pack_bits_bool;
use thoosa_common::dims::{ChannelsData, MotifSlices};
use thoosa_common::tho_device::ThoDevice;
use crate::tasks::find_motifs_run_config::{Aggr as RunConfigAggr, ExtendMotifsRunConfig, FindMotifsRunConfig, MaxSlices, MaxSlicesAbs, MaxSlicesByResources, Metrics, MotifCriterionConstraint, SelectMotifsRunConfig, SelectMotifsRunResult, SelectMotifsSavableResult};
use thoosa_common::runs::{Labeled, SavableResult, Simple};
use thoosa_common::utils::{load_run_config, pad_to};
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::FixedFindMotifsConfig;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourceId, ResourcesConfig};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{async_perf_log, async_write_lock, collect_futures_with_progress, mono_key, perf_log, sync_read_lock, StreamBuffering, ThoError};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{hump, Adjustable, Data, DataId, MonoMetrics, WithId};
use thoosa_shared::comp_id::SynKey;
use thoosa_store::{IdSlice, Store};
use thoosa_with_data::with_data;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use crate::cli::Commands::SelectMotifs;
use crate::tasks::slices_stats::Aggr;
use crate::tasks::stacked_moitfs_loader::{StackedMotifsInput, StackedMotifsLoader, StackedMotifsOutput, StackedMotifsOutputId};
use crate::tasks::cross_batch_clc::{unfold_and_cat, CrossInput, UnfoldParams};
use crate::tasks::cross_dims::{ChannelsData2Selector, ChannelsMotifsDists, ChannelsMotifsSlices};
use crate::tasks::cross_loader::{RangeOrSlice, ResourcesConfigInput};
use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
use crate::tasks::motif_stats::{MotifStats, MotifStatsInput, MotifStatsOutput};
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_shared::rayon_pool::rayon_pool;
use crate::buffer_limits::LOAD_STACKED_MOTIFS_LIMIT;
use crate::tasks::find_motifs::config_to_stage;
use crate::tasks::tensor_loader::RawTensorLoader;
use crate::tasks::resources_motifs_clc_dist::{ClcResourcesMotifsDist, ClcResourcesMotifsDistInput, ClcResourcesMotifsDistOutput};

pub use crate::tasks::resources_motifs_clc_dist::{fold_dists, fold_mask};
use crate::tasks::slices_iter::{ResourceSlices, SlicesIter};
use crate::tasks::stage_ids::SELECT_MOTIFS;
use tracing::Instrument;
use thoosa_shared::tracked_await::track_await;
//The overall logic in Find and Select Motifs Stages is as following:
// Find Motifs - either
//  * accumulate distance statistics into threshold distance while iterating through slices
//  * precalculate threshold distance statistics from randomly selected slices
//
//  build motifs from slices distance between which is less than threshold distance
//
//  Select Motifs
//    1. if ExtendMotifsRunConfigRangeOrValue is run before SelectMotifsStage
//       for each select_motif_config
//        multiple find_motifs_run_config_ids -> select_motif_run_config_id
//
//       calculate threshold distance from motifs found during Find Motifs stage
//       optionally add slices to motifs which is less than threshold distance.
//
//       distance statistic so far is only Euclidean distance
//    2. When invoked directly (SelectMotifsOutputMode::SelectMotifsStage)
//          filter (select) motifs found in FindMotifs step by  MotifCriteria
//       In FindAndSelectMotifsStage, selected motifs are scored for TPE optimization
//
//

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SelectMotifsOutputMode {
    Standalone,
    WithConstraintMetrics,
}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct SelectMotifsInput {
    id: SynKey,
    resources_input: ResourcesConfigInput<Simple>,
    output_mode: SelectMotifsOutputMode,
}

impl SelectMotifsInput {
    pub fn standalone(resources_input: ResourcesConfigInput<Simple>) -> Self {
        Self::new(resources_input, SelectMotifsOutputMode::Standalone)
    }

    pub fn with_constraint_metrics(resources_input: ResourcesConfigInput<Simple>) -> Self {
        Self::new(resources_input, SelectMotifsOutputMode::WithConstraintMetrics)
    }

    fn new(resources_input: ResourcesConfigInput<Simple>, output_mode: SelectMotifsOutputMode) -> Self {
        Self {
            id: SynKey::new([mono_key!("SelectMotifsInput", "stage")]),
            resources_input,
            output_mode,
        }
    }

    pub(crate) fn resources_input(&self) -> &ResourcesConfigInput<Simple> {
        &self.resources_input
    }

    pub(crate) fn output_mode(&self) -> SelectMotifsOutputMode {
        self.output_mode
    }
}

impl Input for SelectMotifsInput {}

impl WithId for SelectMotifsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


#[derive(Debug, Clone)]
struct SelectedMotifEvaluation {
    motif_id: u32,
    criterion_scores: Vec<(&'static str, f64)>,
    weighted_criterion_scores: Vec<(&'static str, f64)>,
}

impl SelectedMotifEvaluation {
    fn score_sum(&self) -> f64 {
        if self.weighted_criterion_scores.is_empty() {
            1.0
        } else {
            self.weighted_criterion_scores.iter().map(|(_, score)| *score).sum()
        }
    }

    fn to_savable_score(&self) -> (u32, f64) {
        (self.motif_id, self.score_sum())
    }
}

pub fn clc_constraint_metrics<'a, I>(select_motifs_run_config: &SelectMotifsRunConfig, selected_motif_criterion_scores: I) -> HashMap<String, f32>
where
    I: IntoIterator<Item = &'a [(&'static str, f64)]>,
{
    let selected_motif_criterion_scores = selected_motif_criterion_scores.into_iter().collect::<Vec<_>>();
    debug!("clc_constraint_metrics: selected_motif_criterion_scores={selected_motif_criterion_scores:?}");

    let constraints = select_motifs_run_config.motif_criteria.as_slice();
    let mut metrics = Metrics::optimizer_constraints(constraints);

    if constraints.is_empty() {
        return metrics;
    }

    if selected_motif_criterion_scores.is_empty() {
        for constraint in constraints {
            metrics.insert(constraint.key().expect("select_motifs constraint key").to_string(), constraint.min);
        }
        return metrics;
    }

    let mut min_scores = constraints
        .iter()
        .map(|constraint| (constraint.key().expect("select_motifs constraint key"), f64::INFINITY))
        .collect::<HashMap<_, _>>();

    for criterion_scores in selected_motif_criterion_scores {
        for (constraint_key, score) in criterion_scores {
            let min_score = min_scores.get_mut(constraint_key).expect("missing criterion score accumulator");
            *min_score = (*min_score).min(*score);
        }
    }

    for constraint in constraints {
        let key = constraint.key().expect("select_motifs constraint key");
        metrics.insert(key.to_string(), (constraint.min as f64 - min_scores[key]) as f32);
    }

    debug!("clc_constraint_metrics: metrics={metrics:?}");
    metrics
}

// Searches for matches between motifs from previous FindMotifsStage runs in the input resources,
// then selects motifs by motif-quality criteria.
// Stacked motifs are loaded in StackedMotifsLoader, while motif statistics are calculated in SelectMotifsStage
// for the requested metric/aggr and then used to build a threshold mask in ClcResourcesMotifsDist matches.
// Finally, each motif is scored by configured criteria and kept only if every configured criterion
// satisfies its own minimum threshold.
pub struct SelectMotifsStage<L, ML, C>//TODO rename to SelectMotifs
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    id: SynKey,
    store: Arc<Box<dyn Store>>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    loader:Arc<L>,

    run_configs:Vec<SelectMotifsRunConfig>,//TODO map to Vec<Arc in new
    find_motifs_run_config_ids:Vec<RunConfigId>,

    motif_stats:Arc<MotifStats<ML, L, C>>,

    find_motif_run_config_ids_motif_ids: HashMap<RunConfigId, Vec<u32>>
}

impl <L, ML, C> Debug for SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectMotifsStage")
            .field("id", self.id())
            .field("run_configs", &self.run_configs)
            .field("find_motifs_run_config_ids", &self.find_motifs_run_config_ids)
            .finish()
    }
}

impl  <L, ML, C> WithId for SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


impl <L, ML, C> SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    pub async fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader:Arc<L>,
        motif_loader:Arc<ML>,
        run_configs:Vec<SelectMotifsRunConfig>,
        find_motifs_run_config_ids:Vec<RunConfigId>,
        // #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Result<Self, ThoError> {

        let motif_stats = Arc::new(MotifStats::new(
            Arc::clone(store),
            Arc::clone(&loader),
            Arc::clone(ws),
            Arc::clone(&motif_loader),
        ));

        //gpt Snapshot motif ids for each find_motifs_run_config_id; used later for criteria filtering.
        let mut find_motif_run_config_ids_motif_ids = HashMap::new();
        for find_motifs_run_config_id in &find_motifs_run_config_ids {
            find_motif_run_config_ids_motif_ids.insert(*find_motifs_run_config_id, track_await(store.list_motifs(*find_motifs_run_config_id)).instrument(tracing::debug_span!("new/1", file = file!(), line = line!())).await?);
        }



        Ok(Self {
            id: SynKey::new([mono_key!(SELECT_MOTIFS, "stage")]),
            store: Arc::clone(store),
            ws: Arc::clone(ws),
            loader,
            run_configs,
            find_motifs_run_config_ids,

            motif_stats,
            find_motif_run_config_ids_motif_ids,
        })
    }

    fn clc_constraint_metrics(select_motifs_run_config: &SelectMotifsRunConfig, selected_motifs: &[SelectedMotifEvaluation]) -> HashMap<String, f32> {
        clc_constraint_metrics(select_motifs_run_config, selected_motifs.iter().map(|selected_motif| selected_motif.criterion_scores.as_slice()))
    }

    async fn filter_motifs_by_criteria(
        &self,
        select_motifs_run_config: &SelectMotifsRunConfig,
        mut ctx: C,
    ) -> Result<Vec<SelectedMotifEvaluation>, ThoError> {

        debug!("SelectMotifsStage::run: 0. self.find_motif_run_config_ids_motif_ids={:?}", self.find_motif_run_config_ids_motif_ids);

        let criteria_constraints = select_motifs_run_config.motif_criteria.clone();
        if criteria_constraints.is_empty() {
            warn!("SelectMotifsStage::run: no motif criteria configured");
            return Ok(self.find_motif_run_config_ids_motif_ids.values().flat_map(|motifs| motifs.iter()).map(|v| SelectedMotifEvaluation {
                motif_id: *v,
                criterion_scores: vec![],
                weighted_criterion_scores: vec![],
            }).collect::<Vec<_>>());
        }

        let motif_ids = self.find_motif_run_config_ids_motif_ids.values().flat_map(|motifs| motifs.iter()).collect::<Vec<_>>();

        let select_motifs_run_config_id = select_motifs_run_config.id.expect("no select_motifs_run_config");
        let criteria_metrics = criteria_constraints.iter().map(|constraint| constraint.metric.clone()).collect::<Vec<_>>();
        info!("SelectMotifsStage::run: 1. motif_ids.len={} criteria_metrics={criteria_metrics:?}", motif_ids.len());
        let motif_stats = Arc::clone(&self.motif_stats);

        let motifs_len = motif_ids.len();

        let futures = motif_ids.into_iter().map(|motif_id|{
            let motif_stats = Arc::clone(&motif_stats);
            let criteria_constraints = criteria_constraints.clone();
            let criteria_metrics = criteria_metrics.clone();
            let mut ctx = ctx.branch(false);

            async move {
                let motif_stats_output = track_await(Arc::clone(&motif_stats)
                    .run_cached(
                        Arc::new(DevicesWrap::wrap(MotifStatsInput::from_motif_input(
                            MotifLoaderInput::new((*motif_id, "motif_id").into(), Some(vec![select_motifs_run_config_id])),
                            criteria_metrics.clone(),
                        ))),
                        ctx.branch(false),
                    ))
                    .instrument(tracing::debug_span!("filter_motifs_by_criteria/1", file = file!(), line = line!())).await?;

                with_data!(ctx, motif_stats_output; |_ctx, motif_stats_output: &MotifStatsOutput| {
                    let criterion_scores = criteria_constraints
                        .iter()
                        .map(|constraint| {
                            Ok::<_, ThoError>((
                                constraint.key()?,
                                motif_stats_output.channel_value(&constraint.metric, 0)? as f64,
                            ))
                        })
                        .collect::<Result<Vec<_>, ThoError>>()?;
                    let satisfies_constraints = criteria_constraints.iter().all(|constraint| {
                        criterion_scores
                            .iter()
                            .find(|(constraint_key, _)| *constraint_key == constraint.key().expect("select_motifs constraint key"))
                            .map(|(key, score)| {
                                let r = *score >= constraint.min as f64;
                                debug!("SelectMotifsStage::run: motif_id={motif_id} key={key} score={score} constraint.min={} r={r}", constraint.min);

                                r
                            })
                            .unwrap_or(false)
                    });
                    debug!("SelectMotifsStage::run: motif_id={motif_id} satisfies_constraints={satisfies_constraints} criterion_scores={criterion_scores:?} criteria_constraints={criteria_constraints:?}");
                    let k = hump(motif_stats_output.num_slices as f64, 20., 0.1, 5.0);//TODO config
                    let weighted_criterion_scores = criterion_scores.iter().map(|(key, score)| (*key, score*k)).collect::<Vec<_>>();
                    Ok::<Option<SelectedMotifEvaluation>, ThoError>(satisfies_constraints.then_some(SelectedMotifEvaluation {
                        motif_id: *motif_id,
                        criterion_scores,
                        weighted_criterion_scores,
                    }))
                })

            }
        });

        let run_results = track_await(collect_futures_with_progress(
            futures,
            motifs_len,
            crate::buffer_limits::SELECT_MOTIFS_STAGE_RUN_CONFIGS,
            StreamBuffering::Unordered,
            "filter_motifs_by_criteria",
        )).instrument(tracing::debug_span!("filter_motifs_by_criteria/2", file = file!(), line = line!())).await
        .into_iter().filter_map(|v| v).collect::<Vec<SelectedMotifEvaluation>>();


        debug!("SelectMotifsStage::run: run_config={select_motifs_run_config:?} filtered extended_motif_ids.len={}", run_results.len());

        Ok(run_results)
    }

}


impl <L,ML, C> Stage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static

{
    type Ctx = C;
    type In<'a> = SelectMotifsInput;
    type Out = SelectMotifsSavableResult;
}


impl <L, ML, C> AsyncStage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SelectMotifsStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        debug!("SelectMotifsStage::run: input={input:?}");

        let (resources_config, output_mode) = with_data!(ctx.branch(false), input; |_ctx, input: &SelectMotifsInput| {
            Ok::<_, ThoError>((input.resources_input().resources_config().clone(), input.output_mode()))
        })?;
        let resources = track_await(resources_config.to_resources(&self.store)).instrument(tracing::debug_span!("run/1", file = file!(), line = line!())).await?;

        debug!("SelectMotifsStage::run: resources={resources:?}");
        if resources.is_empty() {
            warn!("SelectMotifsStage::run: no labels resources");
            return Err(ThoError::new(format!(
                "no resources found in {input:?}"
            )))
        }


        let run_configs_len = self.run_configs.len();

        debug!("SelectMotifsStage::run: self.run_configs.len={}", self.run_configs.len());
        let eta_guard = CollectEtaInvocationGuard::new("SelectMotifsStage::run", run_configs_len, 1);


        let futures = self.run_configs.iter().map(|select_motifs_run_config|{
            let mut ctx_for_futs = ctx.branch(false);
            let self_for_futs = Arc::clone(&self);

            async move {
                let selected_motifs = track_await(self_for_futs
                    // TODO(perf): criteria filtering reloads metrics for each select config; keep a per-run motif metric cache when only thresholds differ.
                    .filter_motifs_by_criteria(select_motifs_run_config, ctx_for_futs.branch(false)))
                    .instrument(tracing::debug_span!("run/2", file = file!(), line = line!())).await?;

                let constraint_metrics = match output_mode {
                    SelectMotifsOutputMode::Standalone => None,
                    SelectMotifsOutputMode::WithConstraintMetrics => Some(Self::clc_constraint_metrics(select_motifs_run_config, selected_motifs.as_slice())),
                };
                let selected_motif_ids_scores = selected_motifs.iter().map(SelectedMotifEvaluation::to_savable_score).collect::<Vec<_>>();



                Ok::<_, ThoError>(SelectMotifsRunResult::new(select_motifs_run_config.clone(), selected_motif_ids_scores, constraint_metrics))
            }
        });

        let run_results = track_await(collect_futures_with_progress(
            futures,
            run_configs_len,
            crate::buffer_limits::SELECT_MOTIFS_STAGE_RUN_CONFIGS,
            StreamBuffering::Unordered,
            "SelectMotifsStage::run",
        )).instrument(tracing::debug_span!("run/3", file = file!(), line = line!())).await;


        debug!("SelectMotifsStage::run: run_results={run_results:?}");

        Ok(Arc::new(DevicesWrap::wrap(SelectMotifsSavableResult::new(run_results))))
    }
}


#[cfg(test)]
pub mod tests {
    use std::collections::{HashMap, HashSet};
    use std::path::Path;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use dashmap::DashMap;
    use log::{debug, LevelFilter};
    use log::Level::Debug;
    use log::LevelFilter::{Info, Trace};
    use parking_lot::RwLock;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{IndexOp, Kind, Tensor};
    use tch::Device::Cuda;
    use tch::Kind::Bool;
    use tokio::fs;
    use thoosa_common::runner::adjusters::splits::OutputBuilder;
    use thoosa_common::runner::common::cachable::{DevicesWrap, WithDataRef};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::{mock_run_ctx, MockRunCtx, RunCtx};
    use thoosa_common::runner::common::DropIf::IfNotMax;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::adjusters::max_parallel::MAX_PARALLEL;
    use crate::tasks::find_motifs_run_config::{CoherenceRunConfig, ExtendMotifsRunConfig, FindMotifsRunConfig, InlierRatioRunConfig, MotifCriterionConstraint, NormThicknessRunConfig, SelectMotifsRunConfig};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runs::{Labeled, SavableResult, Simple};
    use thoosa_common::tch_test_utils;
    use thoosa_common::utils::{draw_function_plot, reset_print_options, to_vec_f32, z_norm};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_motifs::criteria::mock_criteria;
    use thoosa_common::tch_test_utils::generate_motifs_with_avg_distance;
    use thoosa_common::tch_test_utils::Interval;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{cdt_for_dir, TensorIdSliceIdx, ThoError};
    use thoosa_shared::ResourceType::NotInGroup;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::traits::DataId;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_store::{IdSlice, MockStore, MockStoreBuilder, RunConfigRow, Store};
    use thoosa_common::tho_tensor::ThoTensor;
    use crate::tasks::find_motifs_run_config::SelectMotifsSavableResult;
    use thoosa_common::dims::ChannelsData;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
    use crate::tasks::motif_stats::MotifStats;
    use crate::tasks::motif_loader::{mock_motif_loader, MotifLoaderInput, MotifMeta};
    use crate::tasks::motif_loader::mock_motif_loader::MockMotifLoader;
    use crate::tasks::resources_motifs_clc_dist::ClcResourcesMotifsDistOutput;
    use crate::tasks::slices_stats::{Aggr, Metrics};
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use crate::tasks::extend_motifs::{ExtendMotifsInput, ExtendMotifsStage};
    use crate::tasks::select_motifs::{fold_dists, fold_mask, SelectMotifsInput, SelectMotifsStage};
    use crate::tasks::slices_iter::SlicesIter;
    use crate::tasks::stacked_moitfs_loader::{StackedMotifsOutput, StackedMotifsOutputId};
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::tensor_loader::mock_tensor_loader::MockTensorLoader;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    fn standalone_select_input(resources_group: &str) -> Arc<DevicesWrap<SelectMotifsInput>> {
        Arc::new(DevicesWrap::wrap(SelectMotifsInput::standalone(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))))))
    }

    fn select_input_with_constraint_metrics(resources_group: &str) -> Arc<DevicesWrap<SelectMotifsInput>> {
        Arc::new(DevicesWrap::wrap(SelectMotifsInput::with_constraint_metrics(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))))))
    }

    fn find_motifs_run_config_descr(run_config_id: RunConfigId) -> (RunConfigId, Result<RunConfigRow, ThoError>) {
        (
            run_config_id,
            Ok(RunConfigRow {
                id: run_config_id,
                r#type: "FindMotifsRunConfig".to_string(),
                descr: "test find motifs".to_string(),
                run_on: "2026-01-01 00:00:00".to_string(),
            }),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_1_resource_1_label_1_channel_1_motif() -> Result<(), ThoError> {
        //TODO this doesn't work
        // let dash_map = Box::leak(Box::new(DashMap::new()));
        // MockMotifLoader::cache_context().expect().times(0..).return_const(&*dash_map);


        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        // log::set_max_level(LevelFilter::Debug);
        // reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);



        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 10000);

        let find_motifs_run_config_id = 1;
        let select_motifs_run_config_id = 2;

        let channel_0_motifs_ids = vec![1];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[vec![(30, 100, MotifKind::Pulse)]], 100, 10.0f32, true, true)?;
        let resources_channels_intervals = [(&resource_a, &[channel_0_motifs_ids.as_slice()][..], &resource_a_intervals)];

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_config_id)])
                .with_list_run_config_values([(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_list_motifs([
                    (find_motifs_run_config_id as RunConfigId, Ok(channel_0_motifs_ids.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone()])),
                ])
                .with_add_slices_to_motif(resources_add_slices_to_motif_expectations(select_motifs_run_config_id, &channel_0_motifs_ids, &resources_channels_intervals))//<-- assert
                .build());

        debug!("test_single_label_one_motif: resource_a_t={resource_a_t:?} resource_a_intervals={resource_a_intervals:?}");

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t.shallow_clone()),
                ])
                .finish(),
        );

        let motif_t = tch_test_utils::motifs_tensor(100, 3, MotifKind::Pulse, 0.3, 0.4);


        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(1)), (MotifMeta { channel: 0, find_motifs_run_config_id: find_motifs_run_config_id, window: 100 }, motif_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(1, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_config_id: find_motifs_run_config_id, window: 100 }, motif_t)),
        ]).finish());

        let extend_motifs_stage: Arc<ExtendMotifsStage<_, _, MockRunCtx>> = Arc::new(ExtendMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![ExtendMotifsRunConfig::new(Some(select_motifs_run_config_id))],
            vec![find_motifs_run_config_id],
            1000,
            None,
        ).await?);

        let _result = extend_motifs_stage.run(Arc::new(DevicesWrap::wrap(ExtendMotifsInput::new(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into())))))), mock_run_ctx::default(1)).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_2_resources_1_label_1_channel_1_motif() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(Info);
        reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 800);
        let resource_b: Resource = Resource::new(2, 1, "resource_b", 600);

        let find_motifs_run_config_id = 1;
        let select_motifs_run_config_id = 2;

        let channel_0_motifs_ids = vec![1];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[vec![(3, 100, MotifKind::Pulse)]], 100, 10.0f32, true, true)?;
        let (resource_b_t, resource_b_intervals) = tch_test_utils::resource_tensor(resource_b.length() as usize, &[vec![(2, 100, MotifKind::Pulse)]], 200, 10.0f32, true, true)?;
        let resources_channels_intervals = [(&resource_a, &[channel_0_motifs_ids.as_slice()][..], &resource_a_intervals), (&resource_b, &[channel_0_motifs_ids.as_slice()][..],  &resource_b_intervals)];

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_config_id)])
                .with_list_run_config_values([(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_list_motifs([
                    (find_motifs_run_config_id as RunConfigId, Ok(channel_0_motifs_ids.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone(), resource_b.clone()])),
                ])
                .with_add_slices_to_motif(resources_add_slices_to_motif_expectations(select_motifs_run_config_id, &channel_0_motifs_ids, &resources_channels_intervals))//<-- assert
                .build());


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t.shallow_clone()),
                    (resource_b.clone(), resource_b_t.shallow_clone()),
                ])
                .finish(),
        );

        let motif_t = tch_test_utils::motifs_tensor(100, 3, MotifKind::Pulse, 0.3, 0.4);


        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(1)), (MotifMeta { channel: 0, find_motifs_run_config_id: find_motifs_run_config_id, window: 100 }, motif_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(1, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_config_id: find_motifs_run_config_id, window: 100 }, motif_t)),
        ]).finish());


        let extend_motifs_stage: Arc<ExtendMotifsStage<_, _, MockRunCtx>> = Arc::new(ExtendMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![ExtendMotifsRunConfig::new(Some(select_motifs_run_config_id))],
            vec![find_motifs_run_config_id],
            1000,
            None
        ).await?);

        let _result = extend_motifs_stage.run(Arc::new(DevicesWrap::wrap(ExtendMotifsInput::new(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into())))))), mock_run_ctx::default(0)).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_1_resource_1_label_1_channel_2_motifs() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(LevelFilter::Info);
        reset_print_options();
        GlobalCtx::init_devices(2);
        unsafe { mock_global_ctx::USE_REAL_CTX = true; }
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 1000);

        let find_motifs_run_config_id = 1;
        let select_motifs_run_config_id = 2;

        let motifs_ids = vec![1, 2];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[vec![(1, 100, MotifKind::Pulse), (1, 100, MotifKind::Triangle)]], 300, 10.0f32, true, true)?;

        let resources_channels_intervals = [(&resource_a, &[motifs_ids.as_slice()][..], &resource_a_intervals)];

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_config_id)])
                .with_list_run_config_values([(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_list_motifs([
                    (find_motifs_run_config_id as RunConfigId, Ok(motifs_ids.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone()])),
                ])
                .with_add_slices_to_motif(resources_add_slices_to_motif_expectations(select_motifs_run_config_id, &motifs_ids, &resources_channels_intervals))//<-- assert
                .build());


        debug!("test_single_label_one_motif: resource_a_t={resource_a_t:?} resource_a_intervals={resource_a_intervals:?}");

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t.shallow_clone()),
                ])
                .finish(),
        );

        let motif_0_t = tch_test_utils::motifs_tensor(100, 3, MotifKind::Pulse, 0.3, 0.4);
        let motif_1_t = tch_test_utils::motifs_tensor(100, 2, MotifKind::Triangle, 0.3, 0.4);

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(motifs_ids[0])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_0_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(1, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_0_t)),
            (MotifLoaderInput::all_slices(DataId(motifs_ids[1])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_1_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(motifs_ids[1], "motif").into()), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_1_t)),
        ]).finish());


        let extend_motifs_stage: Arc<ExtendMotifsStage<_, _, MockRunCtx>> = Arc::new(ExtendMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![ExtendMotifsRunConfig::new(Some(select_motifs_run_config_id))],
            vec![find_motifs_run_config_id],
            1000,
            None
        ).await?);

        let _result = extend_motifs_stage.run(Arc::new(DevicesWrap::wrap(ExtendMotifsInput::new(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into())))))), mock_run_ctx::default(0)).await?;

        Ok(())
    }

    fn resources_add_slices_to_motif_expectations(run_config_id: RunConfigId, channel_0_motifs_ids: &[u32], resources_channel_intervals: &[(&Resource, &[&[u32]], &Vec<Vec<Vec<Interval>>>)]) -> Vec<(RunConfigId, u32, i64, HashSet<IdSlice>)> {

        let mut motif_hash = HashMap::new();
        for (resource, channel_0_motifs_ids, channel_intervals) in resources_channel_intervals {

            for (c, motifs_intervals) in channel_intervals.iter().enumerate() {

                for (m_idx, intervals) in motifs_intervals.iter().enumerate() {

                    let motif_id = channel_0_motifs_ids[c][m_idx]; //.get(m_idx).expect(&format!("no motif for index {m_idx}"));
                    let mut motif_intervals = motif_hash.entry((motif_id.clone(), c)).or_insert_with(HashSet::new);

                    for interval in intervals {
                        motif_intervals.insert(IdSlice(0, resource.id(), interval.start as u32));
                    }
                }
            }
        }

        motif_hash.into_iter().map(|(motif_id_channel, v)| (run_config_id, motif_id_channel.0, motif_id_channel.1 as i64, v)).collect::<Vec<_>>()

    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_3_resources_2_labels_3_channels_3_motifs_cuda() -> Result<(), ThoError> {
        log::set_max_level(LevelFilter::Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 10000)));

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 900);
        let resource_b: Resource = Resource::new(2, 1, "resource_b", 800);
        let resource_c: Resource = Resource::new(3, 1, "resource_c", 600);

        let find_motifs_run_config_id = 1;
        let select_motifs_run_config_id = 2;

        let channel_0_motif_0 = 1;
        let channel_0_motif_1 = 2;
        let channel_1_motif_0 = 3;

        let channel_0_motifs_ids = vec![channel_0_motif_0, channel_0_motif_1];
        let channel_1_motifs_ids = vec![channel_1_motif_0];
        let channel_2_motifs_ids:Vec<u32> = vec![];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[
                    vec![(3, 100, MotifKind::Pulse)],
                    vec![(2, 100, MotifKind::Sin)],
                    vec![]
                ], 100, 10.0f32,
                                                                                   true,
                                                                                   true)?;
        let (resource_b_t, resource_b_intervals) = tch_test_utils::resource_tensor(resource_b.length() as usize, &[
                    vec![(2, 100, MotifKind::Pulse)],
                    vec![(1, 100, MotifKind::Sin)],
                    vec![]
                ], 100, 10.0f32,
                                                                                   true,
                                                                                   true)?;
        let (resource_c_t, resource_c_intervals) = tch_test_utils::resource_tensor(resource_c.length() as usize, &[
                    vec![(2, 100, MotifKind::Triangle)],
                    vec![(1, 100, MotifKind::Sin)],
                    vec![]
                ], 100, 10.0f32,
                                                                                   true,
                                                                                   true)?;

        let resource_ab_channel_0_motifs = [channel_0_motif_0];
        let resource_c_channel_0_motifs = [channel_0_motif_1];
        let resource_abc_channel_1_motifs = [channel_1_motif_0];
        let resources_channels_intervals = [
            (&resource_a, &[&resource_ab_channel_0_motifs[..], &resource_abc_channel_1_motifs[..], &[][..]][..], &resource_a_intervals),
            (&resource_b, &[&resource_ab_channel_0_motifs[..], &resource_abc_channel_1_motifs[..], &[][..]][..], &resource_b_intervals),
            (&resource_c, &[&resource_c_channel_0_motifs[..], &resource_abc_channel_1_motifs[..], &[][..]][..], &resource_c_intervals),
        ];

        debug!("test_3_resources_2_labels_3_channels_3_motifs: resources_channels_intervals={resources_channels_intervals:?}");

        let expected_add_slices_to_motif_calls = resources_add_slices_to_motif_expectations(select_motifs_run_config_id, &channel_0_motifs_ids, &resources_channels_intervals);
        debug!("test_3_resources_2_labels_3_channels_3_motifs: expected_add_slices_to_motif_calls={expected_add_slices_to_motif_calls:?}");

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_config_id)])
                .with_list_run_config_values([(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_list_motifs([
                    (find_motifs_run_config_id as RunConfigId, Ok(vec![
                        channel_0_motif_0,
                        channel_0_motif_1,
                        channel_1_motif_0
                    ]))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone(), resource_b.clone(), resource_c.clone()])),
                ])
                .with_add_slices_to_motif(expected_add_slices_to_motif_calls)//<-- assert
                .build());


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), resource_a_t),
                    (resource_b.clone(), resource_b_t),
                    (resource_c.clone(), resource_c_t),
                ])
                .finish(),
        );

        let motif_0_0_t = tch_test_utils::motifs_tensor(100, 3, MotifKind::Pulse, 0.3, 0.4);
        let motif_0_1_t = tch_test_utils::motifs_tensor(100, 2, MotifKind::Triangle, 0.3, 0.4);
        let motif_1_0_t = tch_test_utils::motifs_tensor(100, 4, MotifKind::Sin, 0.3, 0.4);
        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(channel_0_motif_0)), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_0_0_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(channel_0_motif_0, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_0_0_t)),

            (MotifLoaderInput::all_slices(DataId(channel_0_motif_1)), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_0_1_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(channel_0_motif_1, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_0_1_t)),

            (MotifLoaderInput::all_slices(DataId(channel_1_motif_0)), (MotifMeta { channel: 1, find_motifs_run_config_id, window: 100 }, motif_1_0_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(channel_1_motif_0, "motif").into()), (MotifMeta { channel: 1, find_motifs_run_config_id, window: 100 }, motif_1_0_t)),
        ]).finish());


        let extend_motifs_stage: Arc<ExtendMotifsStage<_, _, MockRunCtx>> = Arc::new(ExtendMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![ExtendMotifsRunConfig::new(Some(select_motifs_run_config_id))],
            vec![find_motifs_run_config_id],
            10000, //TODO fails with max_batch 1000
            Some(801) //TODO check if ran through OOMHandler
        ).await?);

        let _result = extend_motifs_stage.run(Arc::new(DevicesWrap::wrap(ExtendMotifsInput::new(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into())))))), mock_run_ctx::default(1)).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_filter_motifs_by_criteria_filters_by_slices_and_score() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let find_motifs_run_config_id: RunConfigId = 11;
        let select_motifs_run_config_id: RunConfigId = 22;

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let store = Arc::new(MockStoreBuilder::new().with_list_motifs([(find_motifs_run_config_id, Ok(vec![1, 2, 3, 4]))]).build());
        let loader = Arc::new(mock_tensor_loader::build().finish());

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::new((1u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros(&[9, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((2u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros(&[10, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((3u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros(&[101, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((4u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros(&[100, 4], (Kind::Float, tch::Device::Cpu)))),
        ]).finish());

        let stage = SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_config_id))],
            vec![find_motifs_run_config_id],
            None,
        ).await?;

        let _criteria_mocks = mock_criteria::build()
            .with_norm_thickness(|x, _eps| {
                let score = if x.size()[0] == 10 { 0.3 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .with_inlier_ratio(|x, _k, _eps| {
                let score = if x.size()[0] == 10 { 0.3 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .with_coherence(|x, _iters, _eps| {
                let score = if x.size()[0] == 10 { 0.3 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .install();

        let mut select_motifs_run_config = SelectMotifsRunConfig::new(Some(select_motifs_run_config_id));
        select_motifs_run_config.motif_criteria = vec![
            MotifCriterionConstraint { metric: Metrics::NormThickness(NormThicknessRunConfig { eps: 1e-8 }), min: Metrics::DEFAULT_CRITERION_MIN },
            MotifCriterionConstraint { metric: Metrics::InlierRatio(InlierRatioRunConfig { k: 3.0, eps: 1e-8 }), min: Metrics::DEFAULT_CRITERION_MIN },
            MotifCriterionConstraint { metric: Metrics::Coherence(CoherenceRunConfig { k: 3.0, eps: 1e-8 }), min: Metrics::DEFAULT_CRITERION_MIN },
        ];

        let filtered = stage.filter_motifs_by_criteria(&select_motifs_run_config, mock_run_ctx::default(0)).await?;

        assert_eq!(filtered.len(), 3, "filtered={filtered:?}");

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_filter_motifs_by_norm_thickness() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let find_motifs_run_config_id: RunConfigId = 11;
        let select_motifs_run_config_id: RunConfigId = 22;

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let store = Arc::new(MockStoreBuilder::new().with_list_motifs([(find_motifs_run_config_id, Ok(vec![1, 2, 3, 4]))]).build());
        let loader = Arc::new(mock_tensor_loader::build().finish());

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::new((1u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros(&[9, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((2u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros(&[10, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((3u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros(&[101, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((4u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros(&[100, 4], (Kind::Float, tch::Device::Cpu)))),
        ]).finish());

        let stage = SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_config_id))],
            vec![find_motifs_run_config_id],
            None,
        ).await?;

        let _criteria_mocks = mock_criteria::build()
            .with_norm_thickness(|x, _eps| {
                let score = if x.size()[0] <= 10 { 0.3 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .install();

        let mut select_motifs_run_config = SelectMotifsRunConfig::new(Some(select_motifs_run_config_id));
        select_motifs_run_config.motif_criteria = vec![
            MotifCriterionConstraint { metric: Metrics::NormThickness(NormThicknessRunConfig { eps: 1e-8 }), min: Metrics::DEFAULT_CRITERION_MIN },
        ];

        let filtered = stage.filter_motifs_by_criteria(&select_motifs_run_config, mock_run_ctx::default(0)).await?;
        debug!("filtered={filtered:?}");

        assert_eq!(filtered.len(), 2, "filtered={filtered:?}");

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_filter_motifs_requires_each_criterion_to_meet_min() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let find_motifs_run_config_id: RunConfigId = 41;
        let select_motifs_run_config_id: RunConfigId = 42;

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let store = Arc::new(MockStoreBuilder::new().with_list_motifs([(find_motifs_run_config_id, Ok(vec![1, 2]))]).build());
        let loader = Arc::new(mock_tensor_loader::build().finish());

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::new((1u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros([4, 4], FLOAT_CPU))),
            (MotifLoaderInput::new((2u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::zeros([5, 4], FLOAT_CPU))),
        ]).finish());

        let stage = SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_config_id))],
            vec![find_motifs_run_config_id],
            None,
        ).await?;

        let _criteria_mocks = mock_criteria::build()
            .with_norm_thickness(|x, _eps| {
                let score = if x.size()[0] == 4 { 0.95 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .with_inlier_ratio(|x, _k, _eps| {
                let score = if x.size()[0] == 4 { 0.3 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .install();

        let mut select_motifs_run_config = SelectMotifsRunConfig::new(Some(select_motifs_run_config_id));
        select_motifs_run_config.motif_criteria = vec![
            MotifCriterionConstraint { metric: Metrics::NormThickness(NormThicknessRunConfig { eps: 1e-8 }), min: 0.5 },
            MotifCriterionConstraint { metric: Metrics::InlierRatio(InlierRatioRunConfig { k: 3.0, eps: 1e-8 }), min: 0.5 },
        ];

        let filtered = stage.filter_motifs_by_criteria(&select_motifs_run_config, mock_run_ctx::default(0)).await?;

        assert_eq!(filtered.iter().map(|motif| motif.motif_id).collect::<Vec<_>>(), vec![2], "filtered={filtered:?}");

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_select_motifs_run_reuses_criterion_scores_for_constraint_metrics() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let find_motifs_run_config_id: RunConfigId = 31;
        let select_motifs_run_config_id: RunConfigId = 32;
        let resources_group = "constraint_metrics_group";
        let norm_thickness_calls = Arc::new(AtomicUsize::new(0));
        let inlier_ratio_calls = Arc::new(AtomicUsize::new(0));

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let store = Arc::new(
            MockStoreBuilder::new()
                .with_list_motifs([(find_motifs_run_config_id, Ok(vec![101, 102]))])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![Resource::new(1, 1, "resource_a", 1000)])),
                ])
                .build(),
        );
        let loader = Arc::new(mock_tensor_loader::build().finish());
        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (
                MotifLoaderInput::new((101u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])),
                (
                    MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 },
                    Tensor::zeros([2, 4], FLOAT_CPU),
                ),
            ),
            (
                MotifLoaderInput::new((102u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])),
                (
                    MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 },
                    Tensor::zeros([4, 4], FLOAT_CPU),
                ),
            ),
        ]).finish());

        let norm_thickness_calls_clone = Arc::clone(&norm_thickness_calls);
        let inlier_ratio_calls_clone = Arc::clone(&inlier_ratio_calls);
        let _criteria_mocks = mock_criteria::build()
            .with_norm_thickness(move |x, _eps| {
                norm_thickness_calls_clone.fetch_add(1, Ordering::SeqCst);
                let score = if x.size()[0] == 2 { 0.8 } else { 0.9 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .with_inlier_ratio(move |x, _k, _eps| {
                inlier_ratio_calls_clone.fetch_add(1, Ordering::SeqCst);
                let score = if x.size()[0] == 2 { 0.7 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .install();

        let mut select_motifs_run_config = SelectMotifsRunConfig::new(Some(select_motifs_run_config_id));
        select_motifs_run_config.motif_criteria = vec![
            MotifCriterionConstraint { metric: Metrics::NormThickness(NormThicknessRunConfig { eps: 1e-8 }), min: 0.7 },
            MotifCriterionConstraint { metric: Metrics::InlierRatio(InlierRatioRunConfig { k: 3.0, eps: 1e-8 }), min: 0.65 },
        ];

        let stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![select_motifs_run_config],
            vec![find_motifs_run_config_id],
            None,
        ).await?);

        let result: Arc<DevicesWrap<SelectMotifsSavableResult>> = stage
            .run(select_input_with_constraint_metrics(resources_group), mock_run_ctx::default(0))
            .await?;
        let savable: SelectMotifsSavableResult = Arc::into_inner(result)
            .expect("test_select_motifs_run_reuses_criterion_scores_for_constraint_metrics: output still shared")
            .take_data(0)
            .await?;
        let metrics = savable
            .constraint_metrics()
            .cloned()
            .expect("missing constraint metrics");

        assert_eq!(metrics.len(), Metrics::CONSTRAINT_KEYS.len() + Metrics::EXTRA_CONSTRAINT_KEYS.len());
        assert!((metrics[Metrics::NORM_THICKNESS_MIN_KEY] + 0.10).abs() < 1e-6, "metrics={metrics:?}");
        assert!((metrics[Metrics::INLIER_RATIO_MIN_KEY] + 0.05).abs() < 1e-6, "metrics={metrics:?}");
        assert_eq!(metrics[Metrics::COHERENCE_MIN_KEY], 0.0, "metrics={metrics:?}");
        assert_eq!(norm_thickness_calls.load(Ordering::SeqCst), 2, "metrics={metrics:?}");
        assert_eq!(inlier_ratio_calls.load(Ordering::SeqCst), 2, "metrics={metrics:?}");

        Ok(())
    }

    #[test]
    fn resource_tensor_supports_multiple_kinds_per_channel() -> Result<(), ThoError> {
        let (tensor, intervals) = tch_test_utils::resource_tensor(
            1000,
            &[vec![(1, 50, MotifKind::Pulse), (1, 60, MotifKind::Sin)], vec![(2, 80, MotifKind::Triangle)]], 300, 10.0f32,
            true,
            true,
        )?;

        assert_eq!(tensor.size()[0], 2i64);
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].len(), 2);
        assert_eq!(intervals[1].len(), 1);
        assert_eq!(intervals[0][0].len(), 1);
        assert_eq!(intervals[0][1].len(), 1);
        assert_eq!(intervals[1][0].len(), 2);
        let first_pulse = intervals[0][0].last().unwrap();
        let first_sin = intervals[0][1].first().unwrap();
        assert!(first_pulse.end <= first_sin.start);

        Ok(())
    }

    // #[test]
    fn resource_tensor_offsets_channels_when_space_allows() -> Result<(), ThoError> {
        let (_, intervals) = tch_test_utils::resource_tensor(
            1200,
            &[vec![(1, 100, MotifKind::Pulse)], vec![(1, 120, MotifKind::Triangle)]], 300, 10.0f32,
            true,
            true,
        )?;

        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[0].len(), 1);
        assert_eq!(intervals[1].len(), 1);
        assert_eq!(intervals[0][0].len(), 1);
        assert_eq!(intervals[1][0].len(), 1);
        let first_channel_interval = &intervals[0][0][0];
        let second_channel_interval = &intervals[1][0][0];
        assert!(first_channel_interval.end <= second_channel_interval.start);

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_uses_existing_slices_when_extend_config_is_none() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);
        // reset_print_options();

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let resources_group = "group";
        let resource_a = Resource::new(1, 1, "resource_a", 1000);

        let find_motifs_run_config_id = 1;
        let select_motifs_run_config_id = 2;

        let mut store_mock = MockStore::new();
        let resource_for_resources_query = resource_a.clone();
        store_mock.expect_list_resources_not_in_group().times(1).returning(move |group, _limit| {
            if group == resources_group {
                Ok(vec![resource_for_resources_query.clone()])
            } else {
                Err(ThoError::new(format!("unexpected resources group: {group}")))
            }
        });
        store_mock.expect_list_motifs().times(1).returning(move |run_config_id| {
            if run_config_id != find_motifs_run_config_id {
                return Err(ThoError::new(format!("unexpected find_motifs_run_config_id: {run_config_id}")));
            }
            Ok(vec![11, 12])
        });
        store_mock.expect_list_run_config_values().times(0);
        store_mock.expect_add_slices_to_motif().times(0);

        let store: Arc<Box<dyn Store>> = Arc::new(Box::new(store_mock) as Box<dyn Store>);

        let loader = Arc::new(mock_tensor_loader::build().with_out_id().finish());

        let motif_11_t = Tensor::stack(
            &[Tensor::zeros([100], FLOAT_CPU), Tensor::ones([100], FLOAT_CPU)],
            0,
        );
        let motif_12_t = Tensor::stack(
            &[Tensor::zeros([100], FLOAT_CPU), Tensor::ones([100], FLOAT_CPU).f_mul_scalar(2.0)?],
            0,
        );

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(11)), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_11_t)),
            (MotifLoaderInput::all_slices(DataId(12)), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_12_t)),
        ]).finish());

        let mut select_config = SelectMotifsRunConfig::new(Some(select_motifs_run_config_id));

        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![select_config],
            vec![find_motifs_run_config_id],
            None,
        ).await?);

        let result:Arc<DevicesWrap<SelectMotifsSavableResult>> = select_motifs_stage
            .run(
                standalone_select_input(resources_group),
                mock_run_ctx::default(0),
            )
            .await?;

        let mut ctx_for_result = mock_run_ctx::default(0);
        let result_debug:String = result
            .with_data_ref(0, ctx_for_result, None, |v, _ctx| Ok(format!("{v:?}")))
            .await?;
        assert!(result_debug.contains("(11,"));
        assert!(result_debug.contains("(12,"));

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_filters_motifs_by_criteria_min() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);
        reset_print_options();

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let resources_group = "group";
        let resource_a = Resource::new(1, 1, "resource_a", 1000);

        let find_motifs_run_config_id = 1;
        let select_motifs_run_config_id = 2;

        let mut store_mock = MockStore::new();
        let resource_for_resources_query = resource_a.clone();
        store_mock.expect_list_resources_not_in_group().times(1).returning(move |group, _limit| {
            if group == resources_group {
                Ok(vec![resource_for_resources_query.clone()])
            } else {
                Err(ThoError::new(format!("unexpected resources group: {group}")))
            }
        });
        store_mock.expect_list_motifs().times(1).returning(move |run_config_id| {
            if run_config_id != find_motifs_run_config_id {
                return Err(ThoError::new(format!("unexpected find_motifs_run_config_id: {run_config_id}")));
            }
            Ok(vec![21, 22])
        });
        store_mock.expect_list_run_config_values().times(0);
        store_mock.expect_add_slices_to_motif().times(0);

        let store: Arc<Box<dyn Store>> = Arc::new(Box::new(store_mock) as Box<dyn Store>);
        let loader = Arc::new(mock_tensor_loader::build().with_out_id().finish());

        let motif_21_t = Tensor::from_slice(&[
            0.0f32, 1.0, 0.0,
            0.0f32, 1.0, 0.0,
            0.0,    1.0, 0.0,
        ])
            .view([3, 3]);
        let motif_22_t = Tensor::from_slice(&[
            0.0f32, 0.0, 0.0,
            0.0f32, 0.0, 0.0,
            10.0,  10.0, 10.0,
        ])
            .view([3, 3]);

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::new((21u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_21_t)),
            (MotifLoaderInput::new((22u32, "motif_id").into(), Some(vec![select_motifs_run_config_id])), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, motif_22_t)),
        ]).finish());

        let _criteria_mocks = mock_criteria::build()
            .with_norm_thickness(|x, _eps| {
                let score = if x.double_value(&[2, 0]) > 1.0 { 0.3 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .install();

        let mut select_config = SelectMotifsRunConfig::new(Some(select_motifs_run_config_id));
        select_config.motif_criteria = vec![MotifCriterionConstraint {
            metric: Metrics::NormThickness(NormThicknessRunConfig { eps: 1e-8 }),
            min: Metrics::DEFAULT_CRITERION_MIN,
        }];

        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![select_config],
            vec![find_motifs_run_config_id],
            None,
        ).await?);

        let result:Arc<DevicesWrap<SelectMotifsSavableResult>> = select_motifs_stage
            .run(
                standalone_select_input(resources_group),
                mock_run_ctx::default(0),
            )
            .await?;

        let mut ctx_for_result = mock_run_ctx::default(0);
        let result_debug:String = result
            .with_data_ref(0, ctx_for_result, None, |v, _ctx| Ok(format!("{v:?}")))
            .await?;

        debug!("result_debug={result_debug}");

        assert!(result_debug.contains("(21,"));
        assert!(!result_debug.contains("(22,"));

        Ok(())
    }

    #[test]
    fn test_fold_mask(){
        let mask = Tensor::from_slice(&[
            false, true,   false, false, false, //1
            false, true,   false, false, false, //1
            false, false,  false, false, false, //1

            false, false,  true,  false, false, //2
            false, false,  false, false, false, //2
        ] ).view((1, 5, -1));
        debug!("mask: \n{mask}");


        let ids = vec![1, 1, 1, 2, 2];

        let folded_mask = fold_mask(mask, &ids, 1).unwrap();

        assert_eq!(
            Tensor::from_slice(&[
                false, true,   false, false, false, //1
                false, false,  true,  false, false, //2
            ] ).view((1, 2, -1)),
            folded_mask,
        );

    }

    #[test]
    fn test_fold_mask_two_dims(){
        let mask_0 = Tensor::from_slice(&[
            false, true,   false, false, false, //1
            false, true,   false, false, false, //1
            false, false,  false, false, false, //1

            false, false,  true,  false, false, //2
            false, false,  false, false, false, //2
        ] ).view((1, 5, -1));
        let mask_1 = Tensor::from_slice(&[
            false, false,  true,  false, false, //1
            false, false,  true,  false, false, //1
            false, false,  false, false, false, //1

            false, false,  false,  false, false, //2
            false, true,  false, false, false, //2
        ] ).view((1, 5, -1));
        let mask = Tensor::cat(&[mask_0, mask_1], 0);
        debug!("mask: \n{mask}");


        let ids = vec![1, 1, 1, 2, 2];

        let folded_mask = fold_mask(mask, &ids, 1).unwrap();

        assert_eq!(
            Tensor::from_slice(&[
                false, true,   false, false, false, //1
                false, false,  true,  false, false, //2
            ] ).view((1, 2, -1)),
            folded_mask.narrow(0, 0, 1),
        );

        assert_eq!(
            Tensor::from_slice(&[
                false, false,  true, false, false, //1
                false, true,   false,  false, false, //2
            ] ).view((1, 2, -1)),
            folded_mask.narrow(0, 1, 1),
        );

    }



    #[test]
    fn test_fold_mask_all_false() {
        let mask = Tensor::zeros([1, 4, 3], (Kind::Bool, tch::Device::Cpu));
        let ids = vec![7, 7, 8, 8];
        let folded = fold_mask(mask, &ids, 1).unwrap();
        let expected = Tensor::zeros([1, 2, 3], (Kind::Bool, tch::Device::Cpu));
        assert_eq!(folded, expected);
    }

    #[test]
    fn test_fold_dists_basic_avg_rows() {
        // shape: [1, 5, 3], fold along dim=1 with ids = [1,1,1,2,2]
        // Group 1 (rows 0..2) avg -> [3,3,3]
        // Group 2 (rows 3..4) avg -> [5,5,10]
        let x = Tensor::from_slice(&[
            // row 0
            1.0f32, 2.0, 3.0,
            // row 1
            3.0, 2.0, 1.0,
            // row 2
            5.0, 5.0, 5.0,
            // row 3
            10.0, 0.0, 10.0,
            // row 4
            0.0, 10.0, 10.0,
        ])
            .view([1, 5, 3]);

        let ids = vec![1, 1, 1, 2, 2];

        let folded = fold_dists(x, &ids, 1).unwrap();
        let expected = Tensor::from_slice(&[
            // group 1 avg:
            3.0f32, 3.0, 3.0,
            // group 2 avg:
            5.0, 5.0, 10.0,
        ])
            .view([1, 2, 3]);

        assert_eq!(folded, expected);
    }

    #[test]
    fn test_fold_dists_last_dim_avg_cols() {
        // Fold along the last dimension (dim=2). shape: [2, 2, 4]
        // ids = [10,10,20,20] -> two groups of contiguous columns
        let x = Tensor::from_slice(&[
            // batch 0
            // row 0
            1.0f32, 3.0, 5.0, 7.0,
            // row 1
            2.0, 4.0, 6.0, 8.0,
            // batch 1
            // row 0
            10.0, 0.0, 10.0, 0.0,
            // row 1
            0.0, 10.0, 20.0, 30.0,
        ]).view([2, 2, 4]);

        let ids = vec![10, 10, 20, 20];

        let folded = fold_dists(x, &ids, 2).unwrap();

        // For batch 0:
        // row0: [1,3,5,7] -> A=(1+3)/2=2, B=(5+7)/2=6
        // row1: [2,4,6,8] -> A=3, B=7
        //
        // For batch 1:
        // row0: [10,0,10,0] -> A=5, B=5
        // row1: [0,10,20,30] -> A=5, B=25
        let expected = Tensor::from_slice(&[
            // batch 0
            2.0f32, 6.0, // row 0 -> [A,B]
            3.0, 7.0,    // row 1 -> [A,B]
            // batch 1
            5.0, 5.0,    // row 0
            5.0, 25.0,   // row 1
        ])
            .view([2, 2, 2]);

        assert_eq!(folded, expected);
    }

    #[test]
    fn test_fold_mask_bool_or_contiguous() {
        // sanity check for boolean folding using same machinery
        let mask = Tensor::from_slice(&[
            false, true,   false, false, false, //1
            false, true,   false, false, false, //1
            false, false,  false, false, false, //1
            false, false,  true,  false, false, //2
            false, false,  false, false, false, //2
        ])
            .view([1, 5, 5]);

        let ids = vec![1, 1, 1, 2, 2];

        let folded_mask = fold_mask(mask, &ids, 1).unwrap();

        let expected = Tensor::from_slice(&[
            false, true,  false, false, false, // group 1 (OR of first 3)
            false, false, true,  false, false, // group 2 (OR of last 2)
        ])
            .view([1, 2, 5]);

        assert_eq!(folded_mask, expected);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn heavy_test_oom_cuda() -> Result<(), ThoError> {
        log::set_max_level(LevelFilter::Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 10000)));
        let run_config_id = 1;


        let resource_len = 100_000;
        let resources_label_0 = (0..100).map(|i|Resource::new(i, 1, format!("resource_l0_{i}"), resource_len)).collect::<Vec<_>>();
        let resources_label_1 = (0..100).map(|i|Resource::new(i, 1, format!("resource_l1_{i}"), resource_len)).collect::<Vec<_>>();
        let resources_x = (0..100).map(|i|Resource::new(i, 1, format!("resource_x_{i}"), resource_len)).collect::<Vec<_>>();

        let find_motifs_run_config_id = 1;
        let select_motifs_run_config_id = 1;

        let motif_ids = (0..10000u32).collect::<Vec<_>>();


        let resources_group = "group";


        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(run_config_id)])
                .with_list_run_config_values([(
                    run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(run_config_id), 10, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_list_motifs([
                    (find_motifs_run_config_id as RunConfigId, Ok(motif_ids.clone()))
                ])
                .with_list_group_labeled_resources_sets([
                    (resources_group, Ok(vec![
                        ("Lable_0".to_string(), "resource_set_for_label_0".to_string()),
                        ("Lable_1".to_string(), "resource_set_for_label_1".to_string())
                    ]))
                ])
                .with_list_resource_set_resources_by_descr([
                    ("resource_set_for_label_0", Ok(resources_label_0.clone())),
                    ("resource_set_for_label_1", Ok(resources_label_1.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(resources_x)),
                ])

                .with_add_slices_to_motif_at_least_once()//fails because builder is not implemented
                .build());


        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(
                    resources_label_0.iter().chain(resources_label_1.iter()).map(|resouce|{
                        (resouce.clone(),
                            tch_test_utils::resource_tensor(resource_len as usize, &[
                                                            vec![(3, 100, MotifKind::Pulse)],
                                                            vec![(3, 100, MotifKind::Sin)],
                                                            vec![]
                                                        ], 100, 10.0f32,
                                                            true,
                                                            true).unwrap().0
                        )
                    })
                )
                .finish(),
        );

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached(
            motif_ids.iter().map(|m|((*m).into(), (MotifMeta { channel: 0, find_motifs_run_config_id, window: 100 }, tch_test_utils::motifs_tensor(100, 3, MotifKind::Pulse, 0.3, 0.4))))
        ).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_config_id))],
            vec![find_motifs_run_config_id], //slices
            None
        ).await?);

        let ctx =  mock_run_ctx::default(1);

        let result = select_motifs_stage.run(standalone_select_input(resources_group), ctx).await?;

        Ok(())
    }


}
