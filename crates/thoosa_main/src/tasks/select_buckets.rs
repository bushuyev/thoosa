#![allow(non_camel_case_types)]

use thoosa_common::dims::ChannelsData;
use crate::tasks::find_motifs_run_config::{EnergyMetric, FindMotifsRunConfig, Metrics, MotifCriterionConstraint, SelectMotifsRunConfig, SelectMotifsRunResult, SelectMotifsSavableResult};
use crate::tasks::motif_loader::{MotifLoaderInput, MotifMeta};
use crate::tasks::motif_stats::{MotifStats, MotifStatsInput, MotifStatsOutput};
use crate::tasks::select_motifs::{clc_constraint_metrics, SelectMotifsInput, SelectMotifsOutputMode};
use log::{debug, info};
use parking_lot::RwLock;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_common::dims::MotifSlices;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::load_run_config;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, WithId};
use thoosa_shared::{collect_futures_with_progress, mono_key, FuturesOrdering, ThoError};
use thoosa_store::Store;
use crate::tasks::stage_ids::SELECT_BUCKETS;
use thoosa_shared::tracked_await::track_await;


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, DummyData)]
#[allow(non_camel_case_types)]
pub enum SelectionMode {
    Filter(FilterSelectionMode),
    Best(BestSelectionMode),
}

impl SelectionMode {
    fn selected_evaluations(&self, mut evaluations: Vec<BucketEvaluation>) -> Vec<BucketEvaluation> {
        match self {
            Self::Filter(v) => evaluations.into_iter().filter(|evaluation| evaluation.score > v.threshold as f64).collect(),
            Self::Best(v) => {
                evaluations.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal).then_with(|| a.motif_id.cmp(&b.motif_id)));
                evaluations.truncate(v.limit);
                evaluations
            }
        }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, DummyData)]
pub struct FilterSelectionMode {
    #[in_map]
    #[run_config]
    pub threshold: f32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, DummyData)]
pub struct BestSelectionMode {
    #[in_map]
    #[run_config]
    pub limit: usize,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, DummyData)]
#[run_config_grid]
pub struct SelectBucketsRunConfig {
    #[in_map]
    pub id: Option<RunConfigId>,

    #[in_map]
    #[run_config]
    pub selection_mode: SelectionMode,
}

impl SelectBucketsRunConfig {
    pub fn new(id: Option<RunConfigId>, energy_k: f64) -> Self {
        Self { id, selection_mode: SelectionMode::Filter(FilterSelectionMode { threshold: energy_k as f32 }) }
    }

    fn to_select_motifs_run_config(&self) -> SelectMotifsRunConfig {
        SelectMotifsRunConfig::new_with_fields(self.id, vec![])
    }

    fn constraint_metrics_run_config(&self) -> SelectMotifsRunConfig {
        let criteria = match &self.selection_mode {
            SelectionMode::Filter(v) => vec![MotifCriterionConstraint { metric: energy_metric(), min: v.threshold }],
            SelectionMode::Best(_) => vec![],
        };
        SelectMotifsRunConfig::new_with_fields(self.id, criteria)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BucketEnergy {
    mean_energy: f64,
    num_slices: u32,
}

#[derive(Debug, Clone)]
struct BucketEvaluation {
    motif_id: u32,
    energy: f64,
    score: f64,
    criterion_scores: Vec<(&'static str, f64)>,
}

impl BucketEvaluation {
    fn to_savable_score(&self) -> (u32, f64) {
        (self.motif_id, self.score)
    }
}

fn energy_metric() -> Metrics {
    Metrics::Energy(EnergyMetric::default())
}

pub struct SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    run_configs: Vec<SelectBucketsRunConfig>,
    sync_select_buckets_stage: Arc<SyncSelectBucketsStage<C>>,
    pd: PhantomData<(L, ML, C)>,
}

impl<L, ML, C> Debug for SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectBucketsStage")
            .field("id", &self.id)
            .field("run_configs", &self.run_configs)
            .field("sync_select_buckets_stage", &self.sync_select_buckets_stage)
            .finish()
    }
}

impl<L, ML, C> WithId for SelectBucketsStage<L, ML, C>
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

impl<L, ML, C> SelectBucketsStage<L, ML, C>
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
        run_configs: Vec<SelectBucketsRunConfig>,
        find_motifs_run_config_ids: Vec<RunConfigId>,
        select_motifs_input: Arc<SelectMotifsInput>,
        mut init_ctx: C,
    ) -> Result<Self, ThoError> {
        let motif_stats = Arc::new(MotifStats::new(
            Arc::clone(store),
            Arc::clone(&loader),
            Arc::clone(ws),
            Arc::clone(&motif_loader),
        ));
        let (resources, output_mode) = track_await(Self::load_resources_for_selection(
            &select_motifs_input,
            store,
            init_ctx.branch(false),
        )).await?;
        if resources.is_empty() {
            return Err(ThoError::new(format!("SelectBucketsStage::new: no resources found in {select_motifs_input:?}")));
        }

        let mut bucket_mean_energies = HashMap::new();
        let mut reference_mean_energies = HashMap::new();
        for find_motifs_run_config_id in &find_motifs_run_config_ids {
            let motif_ids = track_await(store.list_motifs(*find_motifs_run_config_id)).await?;
            let energies = track_await(Self::preload_bucket_mean_energies(
                &motif_stats,
                *find_motifs_run_config_id,
                motif_ids,
                init_ctx.branch(true),
            )).await?;
            let find_motifs_run_config: FindMotifsRunConfig = track_await(load_run_config(store, *find_motifs_run_config_id)).await?;
            let reference_energy = track_await(Self::reference_energy_for_find_run(
                &motif_stats,
                resources.as_ref(),
                *find_motifs_run_config_id,
                &find_motifs_run_config,
                energies.len(),
                init_ctx.branch(false),
            )).await?;
            debug!("SelectBucketsStage::new: find_motifs_run_config_id={find_motifs_run_config_id} reference_energy={reference_energy} energies_len={}", energies.len());
            bucket_mean_energies.insert(*find_motifs_run_config_id, energies);
            reference_mean_energies.insert(*find_motifs_run_config_id, reference_energy);
        }

        Ok(Self {
            id: SynKey::new([mono_key!(SELECT_BUCKETS, "SelectBucketsStage")]),
            run_configs,
            sync_select_buckets_stage: Arc::new(SyncSelectBucketsStage::new(
                find_motifs_run_config_ids,
                bucket_mean_energies,
                reference_mean_energies,
                output_mode,
            )),
            pd: PhantomData,
        })
    }

    async fn preload_bucket_mean_energies(
        motif_stats: &Arc<MotifStats<ML, L, C>>,
        find_motifs_run_config_id: RunConfigId,
        motif_ids: Vec<u32>,
        mut ctx: C,
    ) -> Result<HashMap<u32, BucketEnergy>, ThoError> {
        let energy_metric = energy_metric();
        let motif_ids_len = motif_ids.len();
        let futures = motif_ids.into_iter().map(|motif_id| {
            let motif_stats = Arc::clone(motif_stats);
            let energy_metric = energy_metric.clone();
            let mut ctx = ctx.branch(true);

            async move {
                let stats = track_await(Arc::clone(&motif_stats)
                    .run(
                        Arc::new(MotifStatsInput::from_motif_input(
                            MotifLoaderInput::find_motif_slices_only((motif_id, "motif_id").into()),
                            vec![energy_metric.clone()],
                        )),
                        ctx.branch(false),
                    )).await?;

                let energy = BucketEnergy {
                    mean_energy: stats.channel_value(&energy_metric, 0)? as f64,
                    num_slices: stats.num_slices,
                };

                debug!("SelectBucketsStage::preload_bucket_mean_energies: find_motifs_run_config_id={find_motifs_run_config_id} motif_id={motif_id} energy={energy:?}");
                Ok::<_, ThoError>((motif_id, energy))
            }
        });

        let energies = track_await(collect_futures_with_progress(
            futures,
            motif_ids_len,
            FuturesOrdering::Unordered,
            "SelectBucketsStage::preload_bucket_mean_energies",
        )).await;
        if energies.len() != motif_ids_len {
            return Err(ThoError::new(format!("SelectBucketsStage::preload_bucket_mean_energies: preloaded {} of {motif_ids_len} motifs", energies.len())));
        }

        Ok(energies.into_iter().collect())
    }

    async fn load_resources_for_selection(
        input: &Arc<SelectMotifsInput>,
        store: &Arc<Box<dyn Store>>,
        mut ctx: C,
    ) -> Result<(Arc<Vec<Resource>>, SelectMotifsOutputMode), ThoError> {
        let (resources_config, output_mode) = (input.resources_input().resources_config().clone(), input.output_mode());

        Ok((Arc::new(track_await(resources_config.to_resources(store)).await?.into_iter().collect::<Vec<_>>()), output_mode))
    }

    fn reference_random_slices(resources: &[Resource], find_motifs_run_config: &FindMotifsRunConfig, motif_count: usize) -> Result<u32, ThoError> {
        if find_motifs_run_config.window == 0 {
            return Err(ThoError::new("SelectBucketsStage::reference_random_slices: find motif window must be > 0"));
        }

        let input_len_adjuster = find_motifs_run_config.input_len_adjuster();
        let available = resources
            .iter()
            .map(|resource| input_len_adjuster(resource.length()) / find_motifs_run_config.window)
            .sum::<u32>();
        if available < 2 {
            return Err(ThoError::new(format!("SelectBucketsStage::reference_random_slices: only {available} slices available for window {}", find_motifs_run_config.window)));
        }

        let desired = u32::try_from(motif_count.max(2)).unwrap_or(u32::MAX);
        Ok(desired.min(available))
    }

    async fn reference_energy_for_find_run(
        motif_stats: &Arc<MotifStats<ML, L, C>>,
        resources: &[Resource],
        find_motifs_run_config_id: RunConfigId,
        find_motifs_run_config: &FindMotifsRunConfig,
        motif_count: usize,
        mut ctx: C,
    ) -> Result<f64, ThoError> {
        let max_slices = Self::reference_random_slices(resources, find_motifs_run_config, motif_count)?;
        let energy_metric = energy_metric();
        let stats = track_await(Arc::clone(motif_stats)
            .run_cached(
                Arc::new(MotifStatsInput::from_random_resources(
                    max_slices,
                    find_motifs_run_config_id,
                    Some(resources.to_vec()),
                    vec![energy_metric.clone()],
                )),
                ctx.branch(false),
            )).await?;

        let values = stats.channels_values(&energy_metric)?;
        if values.is_empty() {
            return Err(ThoError::new("SelectBucketsStage::reference_energy_for_find_run: no channel energy values"));
        }
        Ok(values.iter().map(|value| value.value() as f64).sum::<f64>() / values.len() as f64)
    }

}

#[derive(Debug)]
struct SelectBucketsSyncInput<'a> {
    select_config: &'a SelectBucketsRunConfig,
}

impl<'a> SelectBucketsSyncInput<'a> {
    fn new(select_config: &'a SelectBucketsRunConfig) -> Self {
        Self { select_config }
    }
}

impl WithId for SelectBucketsSyncInput<'_> {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Data for SelectBucketsSyncInput<'_> {
    type Metrics = ();
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        std::mem::size_of_val(self.select_config) as u64
    }
}

impl Input for SelectBucketsSyncInput<'_> {}

#[derive(Debug)]
struct SelectBucketsSyncInputWrapper {
    select_config: SelectBucketsRunConfig,
}

impl SelectBucketsSyncInputWrapper {
    fn new(select_config: SelectBucketsRunConfig) -> Self {
        Self { select_config }
    }
}

impl WithId for SelectBucketsSyncInputWrapper {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl WithDataRef for SelectBucketsSyncInputWrapper {
    type O<'o> = SelectBucketsSyncInput<'o>;

    fn with_data_ref<'a, R: Send + 'a, F, P>(&'a self, _to_device_idx: usize, permits_ctx: P, _trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move {
            let input = SelectBucketsSyncInput::new(&self.select_config);
            do_run(&input, &permits_ctx)
        }
    }
}

impl WithSelectDataRef<()> for SelectBucketsSyncInputWrapper {
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, permits_ctx: P, _selector: (), trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move {
            track_await(self.with_data_ref(to_device_idx, permits_ctx, trace_ctx, do_run)).await
        }
    }
}

#[derive(Debug)]
struct SyncSelectBucketsStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    find_motifs_run_config_ids: Vec<RunConfigId>,
    bucket_mean_energies: HashMap<RunConfigId, HashMap<u32, BucketEnergy>>,
    reference_mean_energies: HashMap<RunConfigId, f64>,
    output_mode: SelectMotifsOutputMode,
    pd: PhantomData<C>,
}

impl<C> SyncSelectBucketsStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn new(
        find_motifs_run_config_ids: Vec<RunConfigId>,
        bucket_mean_energies: HashMap<RunConfigId, HashMap<u32, BucketEnergy>>,
        reference_mean_energies: HashMap<RunConfigId, f64>,
        output_mode: SelectMotifsOutputMode,
    ) -> Self {
        Self { find_motifs_run_config_ids, bucket_mean_energies, reference_mean_energies, output_mode, pd: PhantomData }
    }

    fn evaluations_for_find_run(&self, find_motifs_run_config_id: RunConfigId) -> Result<Vec<BucketEvaluation>, ThoError> {
        let bucket_energies = self
            .bucket_mean_energies
            .get(&find_motifs_run_config_id)
            .ok_or_else(|| ThoError::new(format!("SyncSelectBucketsStage::evaluations_for_find_run: missing bucket energies for find_motifs_run_config_id={find_motifs_run_config_id}")))?;
        let reference_energy = *self
            .reference_mean_energies
            .get(&find_motifs_run_config_id)
            .ok_or_else(|| ThoError::new(format!("SyncSelectBucketsStage::evaluations_for_find_run: missing reference energy for find_motifs_run_config_id={find_motifs_run_config_id}")))?;
        debug!("SyncSelectBucketsStage::evaluations_for_find_run: find_motifs_run_config_id={find_motifs_run_config_id} reference_energy={reference_energy} bucket_energies_len={}", bucket_energies.len());

        Ok(bucket_energies
            .iter()
            .map(|(motif_id, energy)| {
                let score = if reference_energy == 0.0 { f64::INFINITY } else { energy.mean_energy / reference_energy };
                BucketEvaluation {
                    motif_id: *motif_id,
                    energy: energy.mean_energy,
                    score,
                    criterion_scores: vec![(Metrics::ENERGY_MIN_KEY, score)],
                }
            })
            .collect())
    }

    fn select_for_run_config(&self, select_config: &SelectBucketsRunConfig, ctx: &C) -> Result<SelectMotifsRunResult, ThoError> {
        let mut evaluations = Vec::new();

        for find_motifs_run_config_id in &self.find_motifs_run_config_ids {
            let mut find_evaluations = self.evaluations_for_find_run(*find_motifs_run_config_id)?;
            evaluations.append(&mut find_evaluations);
        }

        let selected_evaluations = select_config.selection_mode.selected_evaluations(evaluations);
        let selected = selected_evaluations.iter().map(BucketEvaluation::to_savable_score).collect::<Vec<_>>();
        info!("SyncSelectBucketsStage::select_for_run_config: select_config={select_config:?} selected={selected:?} ctx_id={:?}", ctx.id());
        for evaluation in &selected_evaluations {
            debug!("SyncSelectBucketsStage::select_for_run_config: selected motif_id={} energy={} score={}", evaluation.motif_id, evaluation.energy, evaluation.score);
        }

        let constraint_metrics = match self.output_mode {
            SelectMotifsOutputMode::Standalone => None,
            SelectMotifsOutputMode::WithConstraintMetrics => {
                let constraint_run_config = select_config.constraint_metrics_run_config();
                Some(clc_constraint_metrics(&constraint_run_config, selected_evaluations.iter().map(|evaluation| evaluation.criterion_scores.as_slice())))
            }
        };

        Ok(SelectMotifsRunResult::new(
            select_config.to_select_motifs_run_config(),
            selected,
            constraint_metrics,
        ))
    }
}

impl<C> Adjustable for SyncSelectBucketsStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for SyncSelectBucketsStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, _input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        ()
    }
}

impl<C> Stage for SyncSelectBucketsStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = SelectBucketsSyncInput<'a>;
    type Out = SelectMotifsRunResult;
}

impl<C> SyncStage for SyncSelectBucketsStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "SyncSelectBucketsStage::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        self.select_for_run_config(input.select_config, ctx)
    }
}

impl<L, ML, C> Stage for SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<SelectMotifsInput>;
    type Out = Arc<SelectMotifsSavableResult>;
}

impl<L, ML, C> AsyncStage for SelectBucketsStage<L, ML, C>
where
    L: for<'i> AsyncStage<In<'i> = Arc<Resource>, Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = Arc<MotifLoaderInput>, Out = Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>>, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "SelectBucketsStage::run"))]
    async fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("SelectBucketsStage::run: input={input:?}");

        let run_configs_len = self.run_configs.len();
        let futures = self.run_configs.iter().map(|select_config| {
            let sync_stage = Arc::clone(&self.sync_select_buckets_stage);
            let mut ctx = ctx.branch(false);
            let select_config = select_config.clone();
            async move {
                let required_ids: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
                let oom_handler: OOMHandler<'_, '_, _, _, _, _, ()> = OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                        Adjusters::RunContext(Box::new(DumpToCpu::new("select_buckets"))),
                    ],
                    None,
                );

                track_await(oom_handler
                    .run(
                        &sync_stage,
                        Arc::new(SelectBucketsSyncInputWrapper::new(select_config)),
                        ctx.branch(false),
                        Contexts::new(
                            NoopInputContext(None),
                            NoopRunnableContext {},
                            MaxParallelAndDumpToCpuAdjustableContex::new(
                                MaxParallelAdjustContextImpl::new(),
                                DumpToCpuContextImpl::new(&required_ids),
                            ),
                        ),
                    )).await
            }
        });

        let run_results = track_await(collect_futures_with_progress(
            futures,
            run_configs_len,
            FuturesOrdering::Unordered,
            "SelectBucketsStage::run",
        )).await;
        if run_results.len() != run_configs_len {
            return Err(ThoError::new(format!("SelectBucketsStage::run: completed {} of {run_configs_len} run configs", run_results.len())));
        }

        info!("SelectBucketsStage::run: selected motifs for {} run configs", run_results.len());
        Ok(Arc::new(SelectMotifsSavableResult::new(run_results)))
    }
}

#[cfg(test)]
mod tests {
    use super::{BestSelectionMode, BucketEvaluation, SelectBucketsRunConfig, SelectBucketsStage, SelectionMode};
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use crate::tasks::motif_loader::{mock_motif_loader, MotifLoaderInput, MotifMeta};
    use crate::tasks::select_motifs::SelectMotifsInput;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::{Kind, Tensor};
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runs::Simple;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{RunConfig, RunConfigId};
    use thoosa_shared::ResourceType::NotInGroup;
    use thoosa_shared::ThoError;
    use thoosa_store::{MockStoreBuilder, RunConfigRow};

    #[test]
    fn test_selection_mode_best_selects_highest_scores() {
        let selected = SelectionMode::Best(BestSelectionMode { limit: 2 }).selected_evaluations(vec![
            BucketEvaluation { motif_id: 1, energy: 1.0, score: 0.5, criterion_scores: vec![] },
            BucketEvaluation { motif_id: 2, energy: 2.0, score: 2.0, criterion_scores: vec![] },
            BucketEvaluation { motif_id: 3, energy: 3.0, score: 1.5, criterion_scores: vec![] },
        ]);

        assert_eq!(selected.iter().map(|evaluation| evaluation.motif_id).collect::<Vec<_>>(), vec![2, 3]);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_select_buckets_stage_selects_motifs_over_reference_energy() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let resource_a = Resource::new(1, 1, "resource_a", 4);
        let resource_b = Resource::new(2, 1, "resource_b", 4);
        let resources_group = "group";
        let find_motifs_run_config_id = 11;
        let select_buckets_run_config_id = 22;
        let low_motif_id = 101;
        let high_motif_id = 102;

        let store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_config_id)])
                .with_list_run_config_values([(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 1, 4, None, None).to_properties_iter(None).collect()),
                )])
                .with_list_motifs([(find_motifs_run_config_id, Ok(vec![low_motif_id, high_motif_id]))])
                .with_list_resources_not_in_group([(resources_group, Ok(vec![resource_a.clone(), resource_b.clone()]))])
                .build(),
        );

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resource_a.clone(), Tensor::f_ones([1, 4], (Kind::Float, tch::Device::Cpu))?),
                    (resource_b.clone(), Tensor::f_full([1, 4], 2.0, (Kind::Float, tch::Device::Cpu))?),
                ])
                .finish(),
        );

        let motif_loader = Arc::new(
            mock_motif_loader::build()
                .with_run_cached([
                    (
                        MotifLoaderInput::find_motif_slices_only((low_motif_id, "motif_id").into()),
                        (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::f_full([1, 4], 2.0, (Kind::Float, tch::Device::Cpu))?),
                    ),
                    (
                        MotifLoaderInput::find_motif_slices_only((high_motif_id, "motif_id").into()),
                        (MotifMeta { channel: 0, find_motifs_run_config_id, window: 4 }, Tensor::f_full([1, 4], 3.0, (Kind::Float, tch::Device::Cpu))?),
                    ),
                ])
                .finish(),
        );

        let input = Arc::new(SelectMotifsInput::standalone(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into())))));

        let stage = Arc::new(
            SelectBucketsStage::new(
                &ws,
                &store,
                loader,
                motif_loader,
                vec![SelectBucketsRunConfig::new(Some(select_buckets_run_config_id), 2.0)],
                vec![find_motifs_run_config_id],
                Arc::clone(&input),
                mock_run_ctx::default(0),
            )
            .await?,
        );
        let result = stage.run(input, mock_run_ctx::default(0)).await?;
        let result = Arc::into_inner(result).expect("stage output should not be shared");

        assert_eq!(result.selected_motif_ids_scores(), &[(high_motif_id, 3.6)]);

        Ok(())
    }

    fn find_motifs_run_config_descr(run_config_id: RunConfigId) -> (RunConfigId, Result<RunConfigRow, ThoError>) {
        (
            run_config_id,
            Ok(RunConfigRow {
                id: run_config_id,
                r#type: FindMotifsRunConfig::type_name().to_string(),
                descr: "test find motifs".to_string(),
                run_on: "2026-01-01 00:00:00".to_string(),
            }),
        )
    }
}
