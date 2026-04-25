use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use clap::Args;
use log::debug;
use parking_lot::RwLock;
use thoosa_common::find_motifs_run_config::{Aggr as FindMotifsAggr, FindMotifsRunConfig, MeanAggr, Metrics, SlicesFilter, StatsKeys};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::group_slices;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_shared::{mono_key, ThoError};
use thoosa_store::{IdSlice, Store};
use thoosa_with_data::with_data;

use crate::tasks::motif_loader::MotifLoaderInput;
use crate::tasks::motif_stats::{MotifStatsInput, MotifStatsOutput};
use crate::tasks::slices_stats::Aggr;
use crate::tasks::stage_ids::MERGE_SELECTED_MOTIFS;

#[derive(Debug, Args, Clone)]
pub struct MergeSelectedMotifsArgs {
    #[arg(long, value_delimiter = ',')]
    pub select_motifs_run_ids: Vec<RunId>,
}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct MergeSelectedMotifsInput {
    id: SynKey,
    run_ids: Vec<RunId>,
}

impl MergeSelectedMotifsInput {
    pub fn new(run_ids: Vec<RunId>) -> Self {
        Self {
            id: SynKey::new([mono_key!("MergeSelectedMotifsInput", "stage")]),
            run_ids,
        }
    }

    pub fn run_ids(&self) -> &Vec<RunId> {
        &self.run_ids
    }
}

impl Input for MergeSelectedMotifsInput {}

impl WithId for MergeSelectedMotifsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Clone, DummyData)]
#[dummy_data(no_clone)]
pub struct MergeSelectedMotifsOutput {
    id: RunId,
    merged_groups_by_channel: Vec<usize>,
}

impl MergeSelectedMotifsOutput {
    pub fn new(id: RunId, merged_groups_by_channel: Vec<usize>) -> Self {
        Self { id, merged_groups_by_channel }
    }

    pub fn merged_groups_by_channel(&self) -> &Vec<usize> {
        &self.merged_groups_by_channel
    }
}

impl WithId for MergeSelectedMotifsOutput {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for MergeSelectedMotifsOutput {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct SelectedMotifKey {
    select_run_id: RunId,
    motif_id: u32,
}

impl SelectedMotifKey {
    fn new(select_run_id: RunId, motif_id: u32) -> Self {
        Self { select_run_id, motif_id }
    }

    fn motif_loader_input(&self) -> MotifLoaderInput {
        MotifLoaderInput::new((self.motif_id, "motif_id").into(), Some(vec![self.select_run_id]))
    }
}

#[derive(Debug, Clone, PartialEq)]
struct MotifComparisonConfig {
    stats_keys: StatsKeys,
    metric: Metrics,
}

impl MotifComparisonConfig {
    fn from_find_motifs_run_config(run_config: &FindMotifsRunConfig) -> Self {
        let aggr = match run_config.slices_filter {
            SlicesFilter::Accumulate(ref v) => find_motifs_aggr_to_stats(v.aggr),
            SlicesFilter::Precalculate(ref v) => find_motifs_aggr_to_stats(v.aggr),
            // Exact thresholds do not carry an aggregation parameter, so use unscaled mean compactness for merge comparisons.
            SlicesFilter::ExactS(_) => Aggr::Mean(MeanAggr { k: 1.0 }),
        };

        Self {
            stats_keys: run_config.to_stats_keys(),
            metric: Metrics::from_similarity_metric(run_config.similarity_metric, aggr),
        }
    }

    fn is_compatible_with(&self, other: &Self) -> bool {
        self.stats_keys == other.stats_keys && self.metric == other.metric
    }
}

#[derive(Debug, Clone)]
struct SelectedMotifCandidate {
    key: SelectedMotifKey,
    channel: usize,
    comparison: MotifComparisonConfig,
}

fn find_motifs_aggr_to_stats(aggr: FindMotifsAggr) -> Aggr {
    match aggr {
        FindMotifsAggr::Mean(v) => Aggr::Mean(v),
        FindMotifsAggr::Min(v) => Aggr::Min(v),
        FindMotifsAggr::Max(v) => Aggr::Max(v),
        FindMotifsAggr::Quantile(v) => Aggr::Quantile(v),
    }
}

#[derive(Named)]
pub struct MergeSelectedMotifs<MS, C>
where
    MS: for<'i> AsyncStage<In<'i> = MotifStatsInput, Out = MotifStatsOutput, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    store: Arc<Box<dyn Store>>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    motif_stats: Arc<MS>,
    _ctx: PhantomData<C>,
}

impl<MS, C> Debug for MergeSelectedMotifs<MS, C>
where
    MS: for<'i> AsyncStage<In<'i> = MotifStatsInput, Out = MotifStatsOutput, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MergeSelectedMotifs").field("id", &self.id).finish()
    }
}

impl<MS, C> MergeSelectedMotifs<MS, C>
where
    MS: for<'i> AsyncStage<In<'i> = MotifStatsInput, Out = MotifStatsOutput, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(store: &Arc<Box<dyn Store>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, motif_stats: Arc<MS>) -> Self {
        Self {
            id: (MERGE_SELECTED_MOTIFS, "MergeSelectedMotifs").into(),
            store: Arc::clone(store),
            ws: Arc::clone(ws),
            motif_stats,
            _ctx: PhantomData,
        }
    }

    async fn load_candidates_by_channel(&self, run_ids: &[RunId]) -> Result<Vec<Vec<SelectedMotifCandidate>>, ThoError> {
        let channels = thoosa_shared::with_perf_log(
            || self.ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "MergeSelectedMotifs/load_candidates/read_ws",
        )
        .channels();
        let mut candidates_by_channel = vec![Vec::new(); channels as usize];
        let mut run_config_cache = HashMap::<RunId, FindMotifsRunConfig>::new();

        for select_run_id in run_ids {
            let selected_motifs = self.store.list_selected_motifs(*select_run_id, channels).await?;
            for (channel_idx, channel_motifs) in selected_motifs.into_iter().enumerate() {
                for (motif_id, _score) in channel_motifs {
                    let (motif_channel, find_motifs_run_id) = self.store.get_motif_meta(motif_id).await?;
                    if motif_channel != channel_idx {
                        return Err(ThoError::new(format!("MergeSelectedMotifs::run: motif_id={motif_id} listed under channel={channel_idx}, but store returned channel={motif_channel}")));
                    }

                    if !run_config_cache.contains_key(&find_motifs_run_id) {
                        let run_config = FindMotifsRunConfig::from_properties_map_with_id(
                            find_motifs_run_id,
                            &self.store.list_run_config_values(find_motifs_run_id).await?,
                        )?;
                        run_config_cache.insert(find_motifs_run_id, run_config);
                    }

                    let find_motifs_run_config = run_config_cache.get(&find_motifs_run_id).expect("set above");
                    candidates_by_channel[channel_idx].push(SelectedMotifCandidate {
                        key: SelectedMotifKey::new(*select_run_id, motif_id),
                        channel: channel_idx,
                        comparison: MotifComparisonConfig::from_find_motifs_run_config(find_motifs_run_config),
                    });
                }
            }
        }

        Ok(candidates_by_channel)
    }

    async fn run_metric(&self, motif_stats_input: MotifStatsInput, metric: Metrics, mut ctx: C) -> Result<f32, ThoError> {
        let stats = Arc::clone(&self.motif_stats)
            .run_cached(
                Arc::new(DevicesWrap::wrap(motif_stats_input)),
                ctx.branch(Some("merge_selected_motifs_motif_stats"), false),
            )
            .await?;

        with_data!(ctx.branch(Some("merge_selected_motifs_metric_value"), false), stats; |_ctx, stats: &MotifStatsOutput| {
            stats.channel_value(&metric, 0)
        })
    }

    async fn metric_for_candidate(&self, candidate: &SelectedMotifCandidate, ctx: C) -> Result<f32, ThoError> {
        self.run_metric(
            MotifStatsInput::from_motif_input(candidate.key.motif_loader_input(), vec![candidate.comparison.metric.clone()]),
            candidate.comparison.metric.clone(),
            ctx,
        )
        .await
    }

    async fn metric_for_pair(&self, left: &SelectedMotifCandidate, right: &SelectedMotifCandidate, ctx: C) -> Result<f32, ThoError> {
        self.run_metric(
            MotifStatsInput::from_merged_motif_inputs(
                vec![left.key.motif_loader_input(), right.key.motif_loader_input()],
                vec![left.comparison.metric.clone()],
            )?,
            left.comparison.metric.clone(),
            ctx,
        )
        .await
    }

    async fn collect_group_slices(&self, channel: usize, motif_keys: &HashSet<SelectedMotifKey>) -> Result<HashSet<IdSlice>, ThoError> {
        let mut merged_slices = HashSet::new();
        for motif_key in motif_keys {
            let (motif_channel, _find_run_id, slices, _resources_map) = self
                .store
                .list_motif_slices(motif_key.motif_id, Some(vec![motif_key.select_run_id]), None, None)
                .await?;

            if motif_channel != channel as i64 {
                return Err(ThoError::new(format!("MergeSelectedMotifs::run: motif_id={} expected channel={} but list_motif_slices returned channel={motif_channel}", motif_key.motif_id, channel)));
            }

            merged_slices.extend(slices.into_iter().map(|slice| IdSlice(0, slice.0, slice.1)));
        }

        Ok(merged_slices)
    }
}

impl<MS, C> Stage for MergeSelectedMotifs<MS, C>
where
    MS: for<'i> AsyncStage<In<'i> = MotifStatsInput, Out = MotifStatsOutput, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = MergeSelectedMotifsInput;
    type Out = MergeSelectedMotifsOutput;
}

impl<MS, C> AsyncStage for MergeSelectedMotifs<MS, C>
where
    MS: for<'i> AsyncStage<In<'i> = MotifStatsInput, Out = MotifStatsOutput, Ctx = C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MergeSelectedMotifs::run"))]
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> impl Future<Output = Result<Arc<DevicesWrap<Self::Out>>, ThoError>> + Send
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        async move {
            let mut run_ids = with_data!(ctx.branch(Some("merge_selected_motifs_input"), false), input; |_ctx, input: &MergeSelectedMotifsInput| {
                Ok::<_, ThoError>(input.run_ids().clone())
            })?;
            if run_ids.is_empty() {
                return Err(ThoError::new("MergeSelectedMotifs::run: run_ids is empty"));
            }
            run_ids.sort();
            run_ids.dedup();

            let merge_run_id = self
                .store
                .save_run_config(Some(run_ids.as_slice()), "MergeMotifs", "MergeMotifs", HashMap::new())
                .await?;

            let candidates_by_channel = self.load_candidates_by_channel(run_ids.as_slice()).await?;
            let mut merged_groups_by_channel = Vec::with_capacity(candidates_by_channel.len());
            let mut motif_ids = vec![];
            for (channel, candidates) in candidates_by_channel.into_iter().enumerate() {
                let mut similar_pairs = Vec::new();
                let mut self_metric_cache = HashMap::<SelectedMotifKey, f32>::new();

                for left_idx in 0..candidates.len() {
                    for right_idx in (left_idx + 1)..candidates.len() {
                        let left = &candidates[left_idx];
                        let right = &candidates[right_idx];
                        debug!("MergeSelectedMotifs::run: channel={channel} comparing left={:?} right={:?}", left.key, right.key);

                        if left.channel != channel || right.channel != channel {
                            return Err(ThoError::new(format!("MergeSelectedMotifs::run: candidate channel mismatch, channel={channel} left_channel={} right_channel={}", left.channel, right.channel)));
                        }
                        if !left.comparison.is_compatible_with(&right.comparison) {
                            debug!("MergeSelectedMotifs::run: skipping incompatible motifs left={:?} right={:?}", left.key, right.key);
                            continue;
                        }

                        let left_metric = if let Some(metric) = self_metric_cache.get(&left.key) {
                            *metric
                        } else {
                            let metric = self.metric_for_candidate(left, ctx.branch(Some("merge_selected_motifs_self_left"), false)).await?;
                            self_metric_cache.insert(left.key.clone(), metric);
                            metric
                        };
                        let right_metric = if let Some(metric) = self_metric_cache.get(&right.key) {
                            *metric
                        } else {
                            let metric = self.metric_for_candidate(right, ctx.branch(Some("merge_selected_motifs_self_right"), false)).await?;
                            self_metric_cache.insert(right.key.clone(), metric);
                            metric
                        };
                        let merged_metric = self.metric_for_pair(left, right, ctx.branch(Some("merge_selected_motifs_pair"), false)).await?;

                        debug!("MergeSelectedMotifs::run: channel={channel} left_metric={left_metric:.6} right_metric={right_metric:.6} merged_metric={merged_metric:.6}");
                        if merged_metric <= left_metric * 1.2 && merged_metric <= right_metric * 1.2 {//TODO param
                            similar_pairs.push((left.key.clone(), right.key.clone()));
                        }
                    }
                }

                let motif_groups = group_slices(similar_pairs.into_iter())
                    .into_iter()
                    .filter(|(_, motifs)| motifs.len() > 1)
                    .collect::<HashMap<_, _>>();
                merged_groups_by_channel.push(motif_groups.len());

                if motif_groups.is_empty() {
                    continue;
                }

                let mut slice_groups = HashMap::with_capacity(motif_groups.len());
                for (group_id, motif_keys) in motif_groups {
                    let merged_slices = self.collect_group_slices(channel, &motif_keys).await?;
                    if !merged_slices.is_empty() {
                        slice_groups.insert(group_id, merged_slices);
                    }
                }

                if !slice_groups.is_empty() {
                    motif_ids.extend(
                        self.store
                            .save_groups_as_motifs(merge_run_id, channel as i64, slice_groups)
                            .await?
                            .into_iter()
                            .map(|motif_id| (motif_id, 1.0)),
                    );
                }
            }

            motif_ids.sort_by_key(|(motif_id, _)| *motif_id);
            let select_motifs_run_id = self.store.save_selected_motifs(merge_run_id, motif_ids).await?;

            Ok(Arc::new(DevicesWrap::wrap(MergeSelectedMotifsOutput::new(select_motifs_run_id, merged_groups_by_channel))))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tch::Tensor;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::cachable::WithDataRef;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_store::{CheckGroups, MockStoreBuilder};
    use thoosa_shared::TensorIdSliceIdx;

    use crate::tasks::motif_loader::{mock_motif_loader, MotifLoaderInput, MotifMeta};
    use crate::tasks::motif_stats::MotifStats;
    use crate::tasks::tensor_loader::mock_tensor_loader;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_merge_selected_motifs_saves_merged_groups() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        unsafe { mock_global_ctx::USE_REAL_CTX = true; }
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let find_run_id = 10;
        let select_run_id_0 = 100;
        let select_run_id_1 = 200;
        let merge_run_id = 300;
        let store = Arc::new(
            MockStoreBuilder::new()
                .with_save_run_config(
                    Some(vec![select_run_id_0, select_run_id_1]),
                    "MergeMotifs",
                    "MergeMotifs",
                    "run_ids",
                    HashMap::new(),
                    Ok(merge_run_id),
                )
                .with_list_selected_motifs([
                    (select_run_id_0, Ok(vec![vec![(1u32, 1.0f64)]])),
                    (select_run_id_1, Ok(vec![vec![(2u32, 1.0f64)]])),
                ])
                .with_get_motif_meta([
                    (1u32, Ok((0usize, find_run_id))),
                    (2u32, Ok((0usize, find_run_id))),
                ])
                .with_list_run_config_values(
                    find_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_run_id), 1, 3, None, Some(1.0)).to_properties_iter(None).collect()),
                )
                .with_list_motif_slices(1u32, Some(vec![select_run_id_0]), None, None, Ok((0i64, find_run_id, vec![TensorIdSliceIdx(1, 10), TensorIdSliceIdx(1, 20)], HashMap::new())))
                .with_list_motif_slices(2u32, Some(vec![select_run_id_1]), None, None, Ok((0i64, find_run_id, vec![TensorIdSliceIdx(2, 30), TensorIdSliceIdx(2, 40)], HashMap::new())))
                .with_save_groups_as_motifs(vec![CheckGroups::Lengths(vec![4])])
                .with_save_selected_motifs([(merge_run_id, vec![(1u32, 1.0f64)], Ok(merge_run_id))])
                .build(),
        );
        let motif_input_0 = MotifLoaderInput::new((1u32, "motif_id").into(), Some(vec![select_run_id_0]));
        let motif_input_1 = MotifLoaderInput::new((2u32, "motif_id").into(), Some(vec![select_run_id_1]));
        let motif_loader = Arc::new(
            mock_motif_loader::build()
                .with_run_cached([
                    (
                        motif_input_0.clone(),
                        (
                            MotifMeta { channel: 0, find_motifs_run_id: find_run_id, window: 3 },
                            Tensor::from_slice(&[0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0]).view((2, 3)),
                        ),
                    ),
                    (
                        motif_input_1.clone(),
                        (
                            MotifMeta { channel: 0, find_motifs_run_id: find_run_id, window: 3 },
                            Tensor::from_slice(&[0.0f32, 0.0, 0.0, 10.0, 0.0, 0.0]).view((2, 3)),
                        ),
                    ),
                ])
                .finish(),
        );
        let motif_stats = Arc::new(MotifStats::new(
            Arc::clone(&store),
            Arc::new(mock_tensor_loader::build().finish()),
            Arc::clone(&ws),
            motif_loader,
        ));
        let stage = Arc::new(MergeSelectedMotifs::new(&store, &ws, motif_stats));
        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));

        let output = stage
            .run(
                Arc::new(DevicesWrap::wrap(MergeSelectedMotifsInput::new(vec![select_run_id_0, select_run_id_1]))),
                thoosa_shared::with_perf_log(
                    || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                    "MergeSelectedMotifs::test/write_ctx",
                )
                .branch(Some("merge_selected_motifs"), false),
            )
            .await?;

        let (run_id, merged_groups_by_channel) = output
            .with_data_ref(
                0,
                thoosa_shared::with_perf_log(
                    || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                    "MergeSelectedMotifs::test/read_ctx",
                )
                .branch(Some("merge_selected_motifs_output"), false),
                None,
                |output: &MergeSelectedMotifsOutput, _ctx| Ok::<_, ThoError>((*output.id(), output.merged_groups_by_channel().clone())),
            )
            .await?;

        assert_eq!(run_id, merge_run_id);
        assert_eq!(merged_groups_by_channel, vec![1]);

        Ok(())
    }
}
