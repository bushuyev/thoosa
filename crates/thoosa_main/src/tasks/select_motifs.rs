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
use polars_arrow::io::iterator::StreamingIterator;
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
use thoosa_common::dims::MotifSlices;
use thoosa_common::tho_device::ThoDevice;
use crate::tasks::find_motifs_run_config::{Aggr as RunConfigAggr, ExtendMotifsConfig, FindMotifsRunConfig, MaxSlices, MaxSlicesAbs, MaxSlicesByResources, Metrics, MotifCriterionConstraint, SelectMotifsRunConfig, SelectMotifsRunResult, SelectMotifsSavableResult};
use thoosa_common::runs::{Labeled, Simple};
use thoosa_common::utils::{load_run_config, pad_to};
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::FixedFindMotifsConfig;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourceId, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
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
use crate::tasks::cross_dims::{ChannelsData2, ChannelsData2Selector, ChannelsMotifsDists, ChannelsMotifsSlices};
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
//The overall logic in Find and Select Motifs Stages is as following:
// Find Motifs - either
//  * accumulate distance statistics into threshold distance while iterating through slices
//  * precalculate threshold distance statistics from randomly selected slices
//
//  build motifs from slices distance between which is less than threshold distance
//
//  Select Motifs
//    1. if SelectMotifsRunConfigRangeOrValue::extend_config is set
//       for each select_motif_config
//        multiple find_motifs_run_ids -> select_motif_run_id
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

// Searches for matches between motifs from previous FindMotifsStage runs in the input resources,
// then selects motifs by motif-quality criteria.
// Stacked motifs are loaded in StackedMotifsLoader, while motif statistics are calculated in SelectMotifsStage
// for the requested metric/aggr and then used to build a threshold mask in ClcResourcesMotifsDist matches.
// Finally, each motif is scored by configured criteria and kept only if every configured criterion
// satisfies its own minimum threshold.
pub struct SelectMotifsStage<L, ML, C>//TODO rename to SelectMotifs
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    id: SynKey,
    store: Arc<Box<dyn Store>>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    loader:Arc<L>,
    stacked_motifs_loader:Arc<StackedMotifsLoader<ML, C>>,
    run_configs:Vec<SelectMotifsRunConfig>,//TODO map to Vec<Arc in new
    find_motifs_run_ids:Vec<RunId>,
    clc_resources_motifs_dist:Arc<ClcResourcesMotifsDist<C>>,
    max_batch:i64,//TODO to SelectMotifsSFixedConfig,
    motif_stats:Arc<MotifStats<ML, L, C>>,

    find_motif_run_ids_motif_ids: HashMap<RunId, Vec<u32>>
}

impl <L, ML, C> Debug for SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectMotifsStage")
            .field("id", self.id())
            .field("run_configs", &self.run_configs)
            .field("find_motifs_run_ids", &self.find_motifs_run_ids)
            .finish()
    }
}

impl  <L, ML, C> WithId for SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
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
           o.to_cpu().await
       }));
    }

    async fn build(self) -> Result<Self::O, ThoError> {

        let mut resource_ids_sizes: Vec<ResourceSlices> = vec![];


        let output:Vec<ClcResourcesMotifsDistOutput> = futures::future::try_join_all(self.output).await?
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
            self.id.find_motifs_run_id,
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
            inner: SelectMotifsSavableResultBuilder::new(id, channels, step).await,
        }
    }
}

impl OutputBuilder for SelectMotifsClcOutputBuilder {
    type O = ClcResourcesMotifsDistOutput;

    async fn add_partial_output(&mut self, o: Self::O) {
        self.inner.add_partial_output(o).await;
    }

    async fn build(self) -> Result<Self::O, ThoError> {
        self.inner.build().await
    }
}

#[derive(Debug, Clone, Default)]
struct SelectMotifsInputSelector {
    batch_selector: RangeOrSlice,
}

impl Adjustable for SelectMotifsInputSelector {}

#[derive(Debug)]
struct SelectMotifsInputSelectorWrap {
    id: SynKey,
    batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>,
    resource_ids: Vec<ResourceId>,
    resource_offsets: Vec<u32>,
    stacked_motifs: Arc<DevicesWrap<StackedMotifsOutput>>,
    motifs_stats: Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>,
}

impl SelectMotifsInputSelectorWrap {
    pub fn new(
        id: SynKey,
        batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>,
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
                 batch_v:&[&ThoTensor<ChannelsData2, SynKey>],
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
                    let selected = Arc::new(DevicesWrap::wrap(t.with_selected_data(ctx.device_idx(), ctx.branch(Some("select_motifs_input_selected_slice"), false), selector, |data, _ctx| {
                                        Ok(data)
                                    }).await?));
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
                 batch_v:&[&ThoTensor<ChannelsData2, SynKey>],
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
        self.output_builder.add_partial_output(out).await;
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
        self.output_builder.build().await
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
                let result = self.batch_1_adjuster.adjust(0i64..0, adjust_context).await;
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
                let result = self.batch_1_slice_adjuster.adjust(0i64..0, &mut adjust_context.slice_ctx).await;
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


impl <L, ML, C> SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    pub async fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader:Arc<L>,
        motif_loader:Arc<ML>,
        run_configs:Vec<SelectMotifsRunConfig>,
        find_motifs_run_ids:Vec<RunId>,
        max_batch: i64,
        // #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Result<Self, ThoError> {

        let clc_dist = Arc::new(MotifStats::new(
            Arc::clone(store),
            Arc::clone(&loader),
            Arc::clone(ws),
            Arc::clone(&motif_loader),
        ));

        //gpt Snapshot motif ids for each find_motifs_run_id; used later for criteria filtering.
        let mut find_motif_run_ids_motif_ids = HashMap::new();
        for find_motifs_run_id in &find_motifs_run_ids {
            find_motif_run_ids_motif_ids.insert(*find_motifs_run_id, store.list_motifs(*find_motifs_run_id).await?);
        }

        let channels = perf_log!(sync_read_lock!(ws))
        .channels() as usize;

        Ok(Self {
            id: SynKey::new([mono_key!(SELECT_MOTIFS, "stage")]),
            store: Arc::clone(store),
            ws: Arc::clone(ws),
            loader,
            stacked_motifs_loader: Arc::new(StackedMotifsLoader::new(Arc::clone(ws), Arc::clone(&motif_loader), Arc::clone(store))),
            run_configs,
            find_motifs_run_ids,
            clc_resources_motifs_dist: Arc::new(ClcResourcesMotifsDist::<C>::new(
                channels,
                #[cfg(test)]
                 fail_if_size_bigger_than
            )),
            max_batch,
            motif_stats: clc_dist,
            find_motif_run_ids_motif_ids,
        })
    }

    #[inline]
    fn slices_selector_aggr_to_stats(aggr: RunConfigAggr) -> Aggr {
        match aggr {
            RunConfigAggr::Mean(mean_aggr) => Aggr::Mean(mean_aggr),
            RunConfigAggr::Min(min_aggr) => Aggr::Min(min_aggr),
            RunConfigAggr::Max(max_aggr) => Aggr::Max(max_aggr),
            RunConfigAggr::Quantile(quantile_aggr) => Aggr::Quantile(quantile_aggr),
        }
    }

    #[inline]
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SelectMotifsStage::load_stacked_motifs"))]
    async fn load_stacked_motifs(store:&Arc<Box<dyn Store>>, stacked_motifs_loader: &Arc<StackedMotifsLoader<ML, C>>, mut ctx_for_loader: C, find_motifs_run_ids: &Vec<RunId>) -> Result<Vec<(Arc<FindMotifsRunConfig>, Arc<DevicesWrap<StackedMotifsOutput>>)>, ThoError> {
        let mut find_motifs_run_configs = Vec::with_capacity(find_motifs_run_ids.len());


        for find_motifs_run_id in find_motifs_run_ids {
            find_motifs_run_configs.push(Arc::new(load_run_config(store, *find_motifs_run_id).await?));
        }
            let find_motifs_run_configs_len = find_motifs_run_configs.len();

            //for each FindMotifsRunConfigs X SelectMotifsRunConfig
            let futures = find_motifs_run_configs
                .into_iter()
                .map(|find_motifs_run_config| {
                    let stacked_motifs_loader = Arc::clone(&stacked_motifs_loader);
                    let ctx_for_run = ctx_for_loader.branch(Some("select_motifs_stacked_motifs"), false);

                    async move {
                        let stacked_motifs_input = Arc::new(DevicesWrap::wrap(StackedMotifsInput::new(Arc::clone(&find_motifs_run_config))?));

                        match stacked_motifs_loader.run_cached(stacked_motifs_input, ctx_for_run).await {
                            Ok(motif_data) => Ok::<_, ThoError>(Some((find_motifs_run_config, motif_data))),
                            Err(e) => {
                                //TODO switch to CPU if OOM ctx.to_cpu...
                                error!("Can't load select_motifs_run_config: e={e:?} select_motifs_run_config={find_motifs_run_config:?}");
                                Ok::<_, ThoError>(None)
                            }
                        }
                    }
                });

            let run_configs__stacked_motifs = collect_futures_with_progress(
                futures,
                find_motifs_run_configs_len,
                LOAD_STACKED_MOTIFS_LIMIT,
                StreamBuffering::Ordered,
                "SelectMotifsStage::load_stacked_motifs",
            )
            .await
            //gpt TODO!! collect_futures_with_progress drops errors and None results; missing find_motifs runs are silently skipped.
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        Ok(run_configs__stacked_motifs)
    }

    async fn load_resources_for_selection(&self, input: &Arc<DevicesWrap<SelectMotifsInput>>, mut ctx: C) -> Result<(Arc<Vec<Resource>>, SelectMotifsOutputMode), ThoError> {
        let (resources_config, output_mode) = with_data!(ctx, input; |_ctx, input: &SelectMotifsInput| {
            Ok::<_, ThoError>((input.resources_input().resources_config().clone(), input.output_mode()))
        })?;
        Ok((Arc::new(resources_config.to_resources(&self.store).await?.into_iter().collect::<Vec<_>>()), output_mode))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SelectMotifsStage::extend_motifs_for_run_config"))]
    async fn extend_motifs_for_run_config(
        &self,
        select_motifs_run_config: &SelectMotifsRunConfig,
        extend_config: &ExtendMotifsConfig,
        resources: Arc<Vec<Resource>>,
        num_channels: usize,
        total_size: usize,
        mut ctx: C,
    ) -> Result<Vec<HashMap<SynKey, HashSet<IdSlice>>>, ThoError> {
        let slices_selector_metric = Metrics::from_similarity_metric(
            extend_config.slices_selector_metric,
            Self::slices_selector_aggr_to_stats(extend_config.slices_selector_aggr),
        );
        // This stays inside the run-config loop.
        // Stacked motifs are cached by find_motifs_run_id, while stats depend on metric/aggr (including floats in aggr).
        // Stats must be computed separately so motifs and statistics can be cached independently.
        let run_configs__stacked_motifs = Self::load_stacked_motifs(
            &self.store,
            &self.stacked_motifs_loader,
            ctx.branch("load_stacked_motifs".into(), true),
            &self.find_motifs_run_ids,
        )
        .await?;

        let channels_motifs_added_slices: Arc<Vec<tokio::sync::RwLock<HashMap<SynKey, HashSet<IdSlice>>>>> =
            Arc::new((0..num_channels).map(|_| tokio::sync::RwLock::new(HashMap::new())).collect());

        let run_configs__stacked_motifs_dbg = &run_configs__stacked_motifs;
        let run_configs__stacked_motifs_len = run_configs__stacked_motifs.len();
        let futures = run_configs__stacked_motifs
            .iter()
            .cloned()
            .map(|(find_motifs_run_config, motif_data)| {
                let resources = Arc::clone(&resources);
                let find_motifs_run_config = Arc::clone(&find_motifs_run_config);
                let channels_motifs_added_slices = Arc::clone(&channels_motifs_added_slices);
                let clc_resources_motifs_dist = Arc::clone(&self.clc_resources_motifs_dist);
                let loader = Arc::clone(&self.loader);
                let ws = Arc::clone(&self.ws);
                let select_motifs_run_config = select_motifs_run_config.clone();
                let mut ctx_to_move = ctx.branch(None, true);
                let clc_dist = Arc::clone(&self.motif_stats);
                let max_batch = self.max_batch;
                let slices_selector_metric = slices_selector_metric.clone();
                let run_configs__stacked_motifs_dbg = run_configs__stacked_motifs_dbg;

                async move {
                    match (async {
                        let motifs_stats = Self::clc_stacked_motifs_statistics(
                            clc_dist,
                            Arc::clone(&motif_data),
                            slices_selector_metric.clone(),
                            ctx_to_move.branch("clc_stacked_motifs_statistics".into(), false),
                        )
                        .await?;
                        Self::extend_with_one_find_run(
                            loader,
                            ws,
                            clc_resources_motifs_dist,
                            select_motifs_run_config,
                            find_motifs_run_config,
                            motif_data,
                            motifs_stats,
                            resources,
                            channels_motifs_added_slices,
                            max_batch,
                            total_size,
                            ctx_to_move.branch("extend_with_one_find_run".into(), false),
                        )
                        .await
                    })
                    .await {
                        Ok(v) => Ok::<_, ThoError>(Some(v)),
                        Err(e) => {
                            error!("SelectMotifsStage::run: got error={e:?} while selecting slices for {run_configs__stacked_motifs_dbg:?}:");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let _ = collect_futures_with_progress(
            futures,
            run_configs__stacked_motifs_len,
            crate::buffer_limits::SELECT_MOTIFS_STAGE_FIND_MOTIFS_BUFFER_UNORDERED_LIMIT,
            StreamBuffering::Unordered,
            "SelectMotifsStage::extend_motifs_for_run_config",
        )
        .await;
        //gpt TODO!! per-run errors are swallowed; channels_motifs_added_slices may be incomplete with no signal to caller.

        if Arc::strong_count(&channels_motifs_added_slices) != 1 {
            return Err(ThoError::new(format!("can't unwrap motif_added_slices for {:?}", select_motifs_run_config)))
        }

        Ok(Arc::into_inner(channels_motifs_added_slices).expect("can't unwrap motif_added_slices").into_iter().map(|lock| lock.into_inner()).collect())
    }

    async fn clc_stacked_motifs_statistics(
        load_and_clc_dist: Arc<MotifStats<ML, L, C>>,
        motif_data: Arc<DevicesWrap<StackedMotifsOutput>>,
        slices_selector_metric: Metrics,
        mut ctx: C,
    ) -> Result<Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>, ThoError> {
        // TODO: move this into a dedicated stage and cache the resulting motifs stats separately from stacked motifs.
        let (channels_motif_ids, find_motifs_run_config, max_motif_slices) = with_data!(ctx.branch(Some("select_motifs_stacked_motifs_meta"), false), motif_data;
            |_ctx, motif_data:&StackedMotifsOutput|{
                let channels_motif_ids = motif_data.motifs.extra().0.clone();
                let find_motifs_run_config = Arc::clone(&motif_data.motifs.extra().1);
                let max_motif_slices = motif_data.motifs.data().t_ref().size()[1] as usize;
                Ok::<_, ThoError>((channels_motif_ids, find_motifs_run_config, max_motif_slices))
        })?;

        let device_idx = ctx.device_idx();
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
                let mut ctx_for_motif = ctx.branch(Some(format!("select_motifs_clc_stacked_motifs_statistics_motif_{motif_id}").as_str()), false);
                async move {
                    let result = (async {
                        let motif_loader_input = MotifLoaderInput::find_motif_slices_only(motif_id.clone());
                        let clc_input = Arc::new(DevicesWrap::wrap(MotifStatsInput::from_motif_input(motif_loader_input, vec![slices_selector_metric.clone()])));
                        let motif_stat: f32 = load_and_clc_dist
                            .run(clc_input, ctx_for_motif.branch(Some("select_motifs_clc_stacked_motifs_statistics_clc"), false))
                            .await?
                            .with_data_ref(
                                0,
                                ctx_for_motif.branch(Some("select_motifs_clc_stacked_motifs_statistics_clc_with_data_ref"), false),
                                None,
                                |stats: &MotifStatsOutput, _ctx| stats.channel_value(&slices_selector_metric, 0),
                            )
                            .await?;
                        Ok::<_, ThoError>((motif_id, motif_stat))
                    })
                    .await;
                    Ok::<_, ThoError>(result)
                }
            });

        let motif_stats = collect_futures_with_progress(
            futures,
            motif_ids_len,
            crate::buffer_limits::STACKED_MOTIFS_LOADER_RUN,
            StreamBuffering::Unordered,
            "SelectMotifsStage::clc_stacked_motifs_statistics",
        )
        .await
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
        let find_motifs_run_id = find_motifs_run_config.id.ok_or_else(|| ThoError::new("SelectMotifsStage::clc_stacked_motifs_statistics: no id in find_motifs_run_config"))?;
        let motifs_stats = ThoTensor::new(
            SynKey::new([
                mono_key!("motifs_stats", "motifs_stats"),
                mono_key!(find_motifs_run_id, format!("find_motifs_run_id_{find_motifs_run_id}")),
                mono_key!(format!("{slices_selector_metric:?}"), "slices_selector_metric"),
            ]),
            channels_motifs_stats_t,
        )?;

        Ok(Arc::new(DevicesWrap::wrap(motifs_stats)))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SelectMotifsStage::extend_with_one_find_run"))]
    async fn extend_with_one_find_run(
        loader: Arc<L>,
        ws: Arc<RwLock<Box<dyn Workspace>>>,
        clc_resources_motifs_dist: Arc<ClcResourcesMotifsDist<C>>,
        select_motifs_run_config: SelectMotifsRunConfig,
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

            let t_0 = Arc::clone(&configured_loader)
                .run_cached(Arc::new(DevicesWrap::wrap(resource_0.clone())), ctx.branch(Some("select_motifs_loader"), false))
                .await?;

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
                find_motifs_run_id,
                channel_masks_dists_ids,
                resources_iter,
                ..
            } = OOMHandler::new(
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
                ctx.branch(Some("select_motifs_clc_resources"), false),
                Contexts::new(
                    SelectMotifsSplitterContext::new(
                        t_1_batch_len as i64,
                        batch_1_slice_lengths,
                        SelectMotifsClcOutputBuilder::new(
                            StackedMotifsOutputId::new(find_motifs_run_config.id.expect("find_motifs_run_config"), vec![]),
                            channels,
                            find_motifs_run_config.step as usize,
                        )
                        .await,
                    ),
                    NoopRunnableContext {},
                    MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
                ),
            )
            .await?;
            let max_slices = select_motifs_run_config.extend_max_slices.as_ref().map(|max_slices| {
                match max_slices {
                    MaxSlices::ByResourcesCount( MaxSlicesByResources {k}) => {
                        *k * num_resources as u32
                    }
                    MaxSlices::Abs(MaxSlicesAbs{count}) => {
                        *count
                    }
                }
            }).unwrap_or(u32::MAX) as usize;

            debug!("SelectMotifsStage::run: channel_masks_dists_ids={channel_masks_dists_ids:?} find_motifs_run_id={find_motifs_run_id}");
            let resources_slices = Arc::new(resources_iter.clone().collect::<Vec<_>>());
            let channel_masks_dists_ids_len = channel_masks_dists_ids.len();
            let futures = channel_masks_dists_ids
                .into_iter()
                .enumerate()
                .map(|(channel, channel_mask_dists_ids)| {
                    let resources_slices = Arc::clone(&resources_slices);
                    let channels_motifs_added_slices = Arc::clone(&channels_motifs_added_slices);
                    async move {
                        let result = (async {
                            if let Some(mask_dists_ids) = channel_mask_dists_ids {
                                debug!("SelectMotifsStage::run: channel={channel} mask_dists_ids={:?}", mask_dists_ids.1);
                                let mask_dists = mask_dists_ids.0.to_cpu().await?;
                                let mask_size = mask_dists.mask_size_hint()?;
                                assert_eq!(mask_size.len(), 3, "SelectMotifsStage::run: expected 3D mask, got {mask_size:?}");
                                assert_eq!(mask_size[0], 1, "SelectMotifsStage::run: expected single-channel mask, got {mask_size:?}");
                                assert_eq!(mask_size[1] as usize, mask_dists_ids.1.len(), "SelectMotifsStage::run: motif/mask size mismatch mask={mask_size:?} motifs={}", mask_dists_ids.1.len());
                                assert_eq!(mask_size[2] as usize, resources_slices.len(), "SelectMotifsStage::run: resources/mask size mismatch mask={mask_size:?} resources={}", resources_slices.len());

                                let motif_count = mask_dists_ids.1.len();
                                let resources_slices_cloned = Arc::clone(&resources_slices);
                                let matches_by_motif = tokio::task::spawn_blocking(move || {
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
                                }).await??;


                                let mut motif_added_slices_guard = async_perf_log!(async_write_lock!(channels_motifs_added_slices[channel]));

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

            collect_futures_with_progress(
                futures,
                channel_masks_dists_ids_len,
                crate::buffer_limits::SELECT_MOTIFS_STAGE_FIND_MOTIFS_BUFFER_UNORDERED_LIMIT,
                StreamBuffering::Unordered,
                "SelectMotifsStage::extend_with_one_find_run_channel_masks",
            )
            .await
            .into_iter()
            .collect::<Result<Vec<_>, ThoError>>()?;

            debug!("SelectMotifsStage::run: 2. find_motifs_run_config={find_motifs_run_config:?}");
            

            t_1_batch_dim_0_size = 0;
            t_1_batch = vec![];
            t_1_batch_resources.clear();
        }

        debug!("SelectMotifsStage::run: processed find_motifs_run_config={find_motifs_run_config:?} select_motifs_run_config={select_motifs_run_config:?}");
        Ok(())
    }


    fn clc_constraint_metrics(select_motifs_run_config: &SelectMotifsRunConfig, selected_motifs: &[SelectedMotifEvaluation]) -> HashMap<String, f32> {
        debug!("SelectMotifsStage::clc_constraint_metrics: selected_motifs={selected_motifs:?}");

        let constraints = select_motifs_run_config.motif_criteria.as_slice();
        let mut metrics = Metrics::optimizer_constraints(constraints);

        if constraints.is_empty() {
            return metrics;
        }

        if selected_motifs.is_empty() {
            for constraint in constraints {
                metrics.insert(constraint.key().expect("select_motifs constraint key").to_string(), constraint.min);
            }
            return metrics;
        }

        let mut min_scores = constraints
            .iter()
            .map(|constraint| (constraint.key().expect("select_motifs constraint key"), f64::INFINITY))
            .collect::<HashMap<_, _>>();

        for selected_motif in selected_motifs {
            for (constraint_key, score) in &selected_motif.criterion_scores {
                let min_score = min_scores.get_mut(constraint_key).expect("missing criterion score accumulator");
                *min_score = (*min_score).min(*score);
            }
        }

        for constraint in constraints {
            let key = constraint.key().expect("select_motifs constraint key");
            metrics.insert(key.to_string(), (constraint.min as f64 - min_scores[key]) as f32);
        }

        debug!("SelectMotifsStage::clc_constraint_metrics: metrics={metrics:?}");
        metrics
    }

    async fn filter_motifs_by_criteria(
        &self,
        select_motifs_run_config: &SelectMotifsRunConfig,
        mut ctx: C,
    ) -> Result<Vec<SelectedMotifEvaluation>, ThoError> {

        debug!("SelectMotifsStage::run: 0. self.find_motif_run_ids_motif_ids={:?}", self.find_motif_run_ids_motif_ids);

        let criteria_constraints = select_motifs_run_config.motif_criteria.clone();
        if criteria_constraints.is_empty() {
            warn!("SelectMotifsStage::run: no motif criteria configured");
            return Ok(self.find_motif_run_ids_motif_ids.values().flat_map(|motifs| motifs.iter()).map(|v| SelectedMotifEvaluation {
                motif_id: *v,
                criterion_scores: vec![],
                weighted_criterion_scores: vec![],
            }).collect::<Vec<_>>());
        }

        let motif_ids = self.find_motif_run_ids_motif_ids.values().flat_map(|motifs| motifs.iter()).collect::<Vec<_>>();

        let select_motifs_run_id = select_motifs_run_config.id.expect("no select_motifs_run_config");
        let criteria_metrics = criteria_constraints.iter().map(|constraint| constraint.metric.clone()).collect::<Vec<_>>();
        info!("SelectMotifsStage::run: 1. motif_ids.len={} criteria_metrics={criteria_metrics:?}", motif_ids.len());
        let motif_stats = Arc::clone(&self.motif_stats);

        let motifs_len = motif_ids.len();

        let futures = motif_ids.into_iter().map(|motif_id|{
            let motif_stats = Arc::clone(&motif_stats);
            let criteria_constraints = criteria_constraints.clone();
            let criteria_metrics = criteria_metrics.clone();
            let mut ctx = ctx.branch(Some("select_motifs_filter_motif_criteria"), false);

            async move {
                let motif_stats_output = Arc::clone(&motif_stats)
                    .run_cached(
                        Arc::new(DevicesWrap::wrap(MotifStatsInput::from_motif_input(
                            MotifLoaderInput::new((*motif_id, "motif_id").into(), Some(vec![select_motifs_run_id])),
                            criteria_metrics.clone(),
                        ))),
                        ctx.branch(Some("select_motifs_filter_motif_stats"), false),
                    )
                    .await?;

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

        let run_results = collect_futures_with_progress(
            futures,
            motifs_len,
            crate::buffer_limits::SELECT_MOTIFS_STAGE_RUN_CONFIGS,
            StreamBuffering::Unordered,
            "filter_motifs_by_criteria",
        ).await
        .into_iter().filter_map(|v| v).collect::<Vec<SelectedMotifEvaluation>>();


        debug!("SelectMotifsStage::run: run_config={select_motifs_run_config:?} filtered extended_motif_ids.len={}", run_results.len());

        Ok(run_results)
    }

}


impl <L,ML, C> Stage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static

{
    type Ctx = C;
    type In<'a> = SelectMotifsInput;
    type Out = SelectMotifsSavableResult;
}


impl <L, ML, C> AsyncStage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SelectMotifsStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        debug!("SelectMotifsStage::run: input={input:?}");

        let (resources, output_mode) = self.load_resources_for_selection(&input, ctx.branch("load_resources_for_selection".into(), false)).await?;

        debug!("SelectMotifsStage::run: resources={resources:?}");
        if resources.is_empty() {
            warn!("SelectMotifsStage::run: no labels resources");
            return Err(ThoError::new(format!(
                "no resources found in {input:?}"
            )))
        }

        let num_channels = perf_log!(sync_read_lock!(self.ws))
        .channels() as usize;
        let total_size = resources.iter().map(|resource| resource.length() as usize).sum::<usize>();

        let run_configs_len = self.run_configs.len();

        debug!("SelectMotifsStage::run: self.run_configs.len={}", self.run_configs.len());
        let eta_guard = CollectEtaInvocationGuard::new("SelectMotifsStage::run", run_configs_len, 1);


        let futures = self.run_configs.iter().map(|select_motifs_run_config|{
            let mut ctx_for_futs = ctx.branch("SelectMotifsStage::run, run_configs_futures_cycle".into(), false);
            let resources_for_futs = Arc::clone(&resources);
            let self_for_futs = Arc::clone(&self);

            async move {
                if let Some(extend_config) = select_motifs_run_config.extend_config.as_ref() {

                    // TODO(perf): extension recomputes resource/motif distance masks per select config; cache intermediate masks by find_motifs_run_id and resources set before filtering.
                    let channels_motifs_added_slices = self_for_futs.extend_motifs_for_run_config(
                        select_motifs_run_config,
                        extend_config,
                        resources_for_futs,
                        num_channels,
                        total_size,
                        ctx_for_futs.branch("extend_motifs_for_run_config".into(), false),
                    ).await?;

                    debug!("SelectMotifsStage::run: extend_motifs_for_run_config done");

                    let select_motifs_run_config_id = select_motifs_run_config.id.expect("no select_motifs_run_config_id");
                    for (channel, motif_slices) in channels_motifs_added_slices.into_iter().enumerate() {
                        for (motif_id, slices) in motif_slices {
                            debug!("SelectMotifsStage::run: motif_id={motif_id} slices.len={} channel={channel}", slices.len());

                            self_for_futs.store.add_slices_to_motif(select_motifs_run_config_id, motif_id.vals[0].to_data_id().0, channel as i64, slices).await?
                        }
                    }
                }

                let selected_motifs = self_for_futs
                    // TODO(perf): criteria filtering reloads metrics for each select config; keep a per-run motif metric cache when only thresholds differ.
                    .filter_motifs_by_criteria(select_motifs_run_config, ctx_for_futs.branch("filter_motifs_by_criteria".into(), false))
                    .await?;

                let constraint_metrics = match output_mode {
                    SelectMotifsOutputMode::Standalone => None,
                    SelectMotifsOutputMode::WithConstraintMetrics => Some(Self::clc_constraint_metrics(select_motifs_run_config, selected_motifs.as_slice())),
                };
                let selected_motif_ids_scores = selected_motifs.iter().map(SelectedMotifEvaluation::to_savable_score).collect::<Vec<_>>();



                Ok::<_, ThoError>(SelectMotifsRunResult::new(select_motifs_run_config.clone(), selected_motif_ids_scores, constraint_metrics))
            }
        });

        let run_results = collect_futures_with_progress(
            futures,
            run_configs_len,
            crate::buffer_limits::SELECT_MOTIFS_STAGE_RUN_CONFIGS,
            StreamBuffering::Unordered,
            "SelectMotifsStage::run",
        ).await;


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
    use crate::tasks::find_motifs_run_config::{CoherenceRunConfig, FindMotifsRunConfig, InlierRatioRunConfig, MotifCriterionConstraint, NormThicknessRunConfig, SelectMotifsRunConfig};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runs::{Labeled, Simple};
    use thoosa_common::tch_test_utils;
    use thoosa_common::utils::{draw_function_plot, reset_print_options, to_vec_f32, z_norm, z_norm_2};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_motifs::criteria::mock_criteria;
    use thoosa_common::tch_test_utils::generate_motifs_with_avg_distance;
    use thoosa_common::tch_test_utils::Interval;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{cdt_for_dir, TensorIdSliceIdx, ThoError};
    use thoosa_shared::ResourceType::NotInGroup;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::traits::DataId;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_store::{IdSlice, MockStore, MockStoreBuilder, RunConfigRow, Store};
    use thoosa_common::tho_tensor::ThoTensor;
    use crate::tasks::find_motifs_run_config::SelectMotifsSavableResult;
    use crate::tasks::cross_dims::ChannelsData2;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
    use crate::tasks::motif_stats::MotifStats;
    use crate::tasks::motif_loader::{mock_motif_loader, MotifLoaderInput, MotifMeta};
    use crate::tasks::motif_loader::mock_motif_loader::MockMotifLoader;
    use crate::tasks::resources_motifs_clc_dist::ClcResourcesMotifsDistOutput;
    use crate::tasks::slices_stats::{Aggr, Metrics};
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use crate::tasks::select_motifs::{fold_dists, fold_mask, SelectMotifsInput, SelectMotifsSavableResultBuilder, SelectMotifsStage};
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

    fn find_motifs_run_config_descr(run_id: RunId) -> (RunId, Result<RunConfigRow, ThoError>) {
        (
            run_id,
            Ok(RunConfigRow {
                id: run_id,
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

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

        let channel_0_motifs_ids = vec![1];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[vec![(30, 100, MotifKind::Pulse)]], 100, 10.0f32, true, true)?;
        let resources_channels_intervals = [(&resource_a, &[channel_0_motifs_ids.as_slice()][..], &resource_a_intervals)];

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_id)])
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(channel_0_motifs_ids.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone()])),
                ])
                .with_add_slices_to_motif(resources_add_slices_to_motif_expectations(select_motifs_run_id, &channel_0_motifs_ids, &resources_channels_intervals))//<-- assert
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

        let motif_t = motifs_tensor(100, 3, MotifKind::Pulse);


        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(1)), (MotifMeta { channel: 0, find_motifs_run_id: find_motifs_run_id, window: 100 }, motif_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(1, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_id: find_motifs_run_id, window: 100 }, motif_t)),
        ]).finish());

        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_id))],
            vec![find_motifs_run_id],
            1000,
            None,
        ).await?);

        let result = select_motifs_stage.run(standalone_select_input(resources_group), mock_run_ctx::default(1)).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_2_resources_1_label_1_channel_1_motif() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(LevelFilter::Info);
        reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 800);
        let resource_b: Resource = Resource::new(2, 1, "resource_b", 600);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

        let channel_0_motifs_ids = vec![1];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[vec![(3, 100, MotifKind::Pulse)]], 100, 10.0f32, true, true)?;
        let (resource_b_t, resource_b_intervals) = tch_test_utils::resource_tensor(resource_b.length() as usize, &[vec![(2, 100, MotifKind::Pulse)]], 200, 10.0f32, true, true)?;
        let resources_channels_intervals = [(&resource_a, &[channel_0_motifs_ids.as_slice()][..], &resource_a_intervals), (&resource_b, &[channel_0_motifs_ids.as_slice()][..],  &resource_b_intervals)];

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_id)])
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(channel_0_motifs_ids.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone(), resource_b.clone()])),
                ])
                .with_add_slices_to_motif(resources_add_slices_to_motif_expectations(select_motifs_run_id, &channel_0_motifs_ids, &resources_channels_intervals))//<-- assert
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

        let motif_t = motifs_tensor(100, 3, MotifKind::Pulse);


        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(1)), (MotifMeta { channel: 0, find_motifs_run_id: find_motifs_run_id, window: 100 }, motif_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(1, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_id: find_motifs_run_id, window: 100 }, motif_t)),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_id))],
            vec![find_motifs_run_id],
            1000,
            None
        ).await?);

        let result = select_motifs_stage.run(standalone_select_input(resources_group), mock_run_ctx::default(0)).await?;

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

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

        let motifs_ids = vec![1, 2];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[vec![(1, 100, MotifKind::Pulse), (1, 100, MotifKind::Triangle)]], 300, 10.0f32, true, true)?;

        let resources_channels_intervals = [(&resource_a, &[motifs_ids.as_slice()][..], &resource_a_intervals)];

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_id)])
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(motifs_ids.clone()))
                ])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![resource_a.clone()])),
                ])
                .with_add_slices_to_motif(resources_add_slices_to_motif_expectations(select_motifs_run_id, &motifs_ids, &resources_channels_intervals))//<-- assert
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

        let motif_0_t = motifs_tensor(100, 3, MotifKind::Pulse);
        let motif_1_t = motifs_tensor(100, 2, MotifKind::Triangle);

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(motifs_ids[0])), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(1, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_t)),
            (MotifLoaderInput::all_slices(DataId(motifs_ids[1])), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_1_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(motifs_ids[1], "motif").into()), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_1_t)),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_id))],
            vec![find_motifs_run_id],
            1000,
            None
        ).await?);


        let result = select_motifs_stage.run(standalone_select_input(resources_group), mock_run_ctx::default(0)).await?;

        Ok(())
    }

    fn resources_add_slices_to_motif_expectations(run_id: RunId, channel_0_motifs_ids: &[u32], resources_channel_intervals: &[(&Resource, &[&[u32]], &Vec<Vec<Vec<Interval>>>)]) -> Vec<(RunId, u32, i64, HashSet<IdSlice>)> {

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

        motif_hash.into_iter().map(|(motif_id_channel, v)| (run_id, motif_id_channel.0, motif_id_channel.1 as i64, v)).collect::<Vec<_>>()

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

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

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

        let expected_add_slices_to_motif_calls = resources_add_slices_to_motif_expectations(select_motifs_run_id, &channel_0_motifs_ids, &resources_channels_intervals);
        debug!("test_3_resources_2_labels_3_channels_3_motifs: expected_add_slices_to_motif_calls={expected_add_slices_to_motif_calls:?}");

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(find_motifs_run_id)])
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(vec![
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

        let motif_0_0_t = motifs_tensor(100, 3, MotifKind::Pulse);
        let motif_0_1_t = motifs_tensor(100, 2, MotifKind::Triangle);
        let motif_1_0_t = motifs_tensor(100, 4, MotifKind::Sin);
        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(channel_0_motif_0)), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_0_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(channel_0_motif_0, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_0_t)),

            (MotifLoaderInput::all_slices(DataId(channel_0_motif_1)), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_1_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(channel_0_motif_1, "motif").into()), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_1_t)),

            (MotifLoaderInput::all_slices(DataId(channel_1_motif_0)), (MotifMeta { channel: 1, find_motifs_run_id, window: 100 }, motif_1_0_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(mono_key!(channel_1_motif_0, "motif").into()), (MotifMeta { channel: 1, find_motifs_run_id, window: 100 }, motif_1_0_t)),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_id))],
            vec![find_motifs_run_id],
            10000, //TODO fails with max_batch 1000
            Some(801) //TODO check if ran through OOMHandler
        ).await?);


        let result = select_motifs_stage.run(standalone_select_input(resources_group), mock_run_ctx::default(1)).await?;

        Ok(())
    }

    ///found motif's slices of same kind
    //TODO make slices a bit different
    pub fn motifs_tensor(motif_len:usize, num_slices: usize, motif_kind: MotifKind) -> Tensor {


        let t = Tensor::stack(
            generate_motifs_with_avg_distance(
                gen_motif(motif_len, &motif_kind).as_slice(), num_slices, 0.3, 0.4, None, 10.0
            ).0.into_iter().map(|motif| Tensor::from_slice(motif.as_slice())).collect::<Vec<_>>().as_slice(), 0
        );

        debug!("motifs_tensor: t.size={:?}", t.size());

        t
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_filter_motifs_by_criteria_filters_by_slices_and_score() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let find_motifs_run_id: RunId = 11;
        let select_motifs_run_id: RunId = 22;

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let store = Arc::new(MockStoreBuilder::new().with_list_motifs([(find_motifs_run_id, Ok(vec![1, 2, 3, 4]))]).build());
        let loader = Arc::new(mock_tensor_loader::build().finish());

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::new((1u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros(&[9, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((2u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros(&[10, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((3u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros(&[101, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((4u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros(&[100, 4], (Kind::Float, tch::Device::Cpu)))),
        ]).finish());

        let stage = SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_id))],
            vec![find_motifs_run_id],
            1024,
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

        let mut select_motifs_run_config = SelectMotifsRunConfig::new(Some(select_motifs_run_id));
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

        let find_motifs_run_id: RunId = 11;
        let select_motifs_run_id: RunId = 22;

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let store = Arc::new(MockStoreBuilder::new().with_list_motifs([(find_motifs_run_id, Ok(vec![1, 2, 3, 4]))]).build());
        let loader = Arc::new(mock_tensor_loader::build().finish());

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::new((1u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros(&[9, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((2u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros(&[10, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((3u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros(&[101, 4], (Kind::Float, tch::Device::Cpu)))),
            (MotifLoaderInput::new((4u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros(&[100, 4], (Kind::Float, tch::Device::Cpu)))),
        ]).finish());

        let stage = SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_id))],
            vec![find_motifs_run_id],
            1024,
            None,
        ).await?;

        let _criteria_mocks = mock_criteria::build()
            .with_norm_thickness(|x, _eps| {
                let score = if x.size()[0] <= 10 { 0.3 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .install();

        let mut select_motifs_run_config = SelectMotifsRunConfig::new(Some(select_motifs_run_id));
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

        let find_motifs_run_id: RunId = 41;
        let select_motifs_run_id: RunId = 42;

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let store = Arc::new(MockStoreBuilder::new().with_list_motifs([(find_motifs_run_id, Ok(vec![1, 2]))]).build());
        let loader = Arc::new(mock_tensor_loader::build().finish());

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::new((1u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros([4, 4], FLOAT_CPU))),
            (MotifLoaderInput::new((2u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 4 }, Tensor::zeros([5, 4], FLOAT_CPU))),
        ]).finish());

        let stage = SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_id))],
            vec![find_motifs_run_id],
            1024,
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

        let mut select_motifs_run_config = SelectMotifsRunConfig::new(Some(select_motifs_run_id));
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

        let find_motifs_run_id: RunId = 31;
        let select_motifs_run_id: RunId = 32;
        let resources_group = "constraint_metrics_group";
        let norm_thickness_calls = Arc::new(AtomicUsize::new(0));
        let inlier_ratio_calls = Arc::new(AtomicUsize::new(0));

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let store = Arc::new(
            MockStoreBuilder::new()
                .with_list_motifs([(find_motifs_run_id, Ok(vec![101, 102]))])
                .with_list_resources_not_in_group([
                    (resources_group, Ok(vec![Resource::new(1, 1, "resource_a", 1000)])),
                ])
                .build(),
        );
        let loader = Arc::new(mock_tensor_loader::build().finish());
        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (
                MotifLoaderInput::new((101u32, "motif_id").into(), Some(vec![select_motifs_run_id])),
                (
                    MotifMeta { channel: 0, find_motifs_run_id, window: 4 },
                    Tensor::zeros([2, 4], FLOAT_CPU),
                ),
            ),
            (
                MotifLoaderInput::new((102u32, "motif_id").into(), Some(vec![select_motifs_run_id])),
                (
                    MotifMeta { channel: 0, find_motifs_run_id, window: 4 },
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

        let mut select_motifs_run_config = SelectMotifsRunConfig::new(Some(select_motifs_run_id));
        select_motifs_run_config.extend_config = None;
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
            vec![find_motifs_run_id],
            1024,
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
    async fn test_clc_stacked_motifs_statistics_duplicates_and_pads_with_min() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

        let find_motifs_run_id = 7u32;
        let find_motifs_run_config = Arc::new(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), 1, 3, None, None));

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
                    MotifMeta { channel: 0, find_motifs_run_id, window: 3 },
                    Tensor::from_slice(&[0.0f32, 0.0, 0.0, 3.0, 0.0, 0.0]).view([2, 3]),
                ),
            ),
            (
                MotifLoaderInput::find_motif_slices_only((12u32, "motif_id").into()),
                (
                    MotifMeta { channel: 0, find_motifs_run_id, window: 3 },
                    Tensor::from_slice(&[0.0f32, 0.0, 0.0, 4.0, 0.0, 0.0]).view([2, 3]),
                ),
            ),
            (
                MotifLoaderInput::find_motif_slices_only((13u32, "motif_id").into()),
                (
                    MotifMeta { channel: 1, find_motifs_run_id, window: 3 },
                    Tensor::from_slice(&[0.0f32, 0.0, 0.0, 5.0, 0.0, 0.0]).view([2, 3]),
                ),
            ),
        ]).finish());
        let store = Arc::new(MockStoreBuilder::new().build());
        let loader = Arc::new(mock_tensor_loader::build().finish());
        let clc_dist = Arc::new(MotifStats::new(store, loader, Arc::clone(&ws), motif_loader));

        let mut ctx = mock_run_ctx::default(1);
        let metric = Metrics::default();
        let motifs_stats = SelectMotifsStage::<MockTensorLoader, MockMotifLoader, MockRunCtx>::clc_stacked_motifs_statistics(
            clc_dist,
            motif_data,
            metric,
            ctx.branch("test_clc_stacked_motifs_statistics".into(), false),
        )
        .await?;

        let mut ctx_for_data = ctx.branch(Some("test_clc_stacked_motifs_statistics_data"), false);
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_uses_existing_slices_when_extend_config_is_none() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);
        // reset_print_options();

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));
        let resources_group = "group";
        let resource_a = Resource::new(1, 1, "resource_a", 1000);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

        let mut store_mock = MockStore::new();
        let resource_for_resources_query = resource_a.clone();
        store_mock.expect_list_resources_not_in_group().times(1).returning(move |group, _limit| {
            if group == resources_group {
                Ok(vec![resource_for_resources_query.clone()])
            } else {
                Err(ThoError::new(format!("unexpected resources group: {group}")))
            }
        });
        store_mock.expect_list_motifs().times(1).returning(move |run_id| {
            if run_id != find_motifs_run_id {
                return Err(ThoError::new(format!("unexpected find_motifs_run_id: {run_id}")));
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
            (MotifLoaderInput::all_slices(DataId(11)), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_11_t)),
            (MotifLoaderInput::all_slices(DataId(12)), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_12_t)),
        ]).finish());

        let mut select_config = SelectMotifsRunConfig::new(Some(select_motifs_run_id));
        select_config.extend_config = None;

        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![select_config],
            vec![find_motifs_run_id],
            1000,
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

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

        let mut store_mock = MockStore::new();
        let resource_for_resources_query = resource_a.clone();
        store_mock.expect_list_resources_not_in_group().times(1).returning(move |group, _limit| {
            if group == resources_group {
                Ok(vec![resource_for_resources_query.clone()])
            } else {
                Err(ThoError::new(format!("unexpected resources group: {group}")))
            }
        });
        store_mock.expect_list_motifs().times(1).returning(move |run_id| {
            if run_id != find_motifs_run_id {
                return Err(ThoError::new(format!("unexpected find_motifs_run_id: {run_id}")));
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
            (MotifLoaderInput::new((21u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_21_t)),
            (MotifLoaderInput::new((22u32, "motif_id").into(), Some(vec![select_motifs_run_id])), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_22_t)),
        ]).finish());

        let _criteria_mocks = mock_criteria::build()
            .with_norm_thickness(|x, _eps| {
                let score = if x.double_value(&[2, 0]) > 1.0 { 0.3 } else { 0.8 };
                Ok(Tensor::from(score).to_kind(x.kind()).to_device(x.device()))
            })
            .install();

        let mut select_config = SelectMotifsRunConfig::new(Some(select_motifs_run_id));
        select_config.extend_config = None;
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
            vec![find_motifs_run_id],
            1000,
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
        let run_id = 1;


        let resource_len = 100_000;
        let resources_label_0 = (0..100).map(|i|Resource::new(i, 1, format!("resource_l0_{i}"), resource_len)).collect::<Vec<_>>();
        let resources_label_1 = (0..100).map(|i|Resource::new(i, 1, format!("resource_l1_{i}"), resource_len)).collect::<Vec<_>>();
        let resources_x = (0..100).map(|i|Resource::new(i, 1, format!("resource_x_{i}"), resource_len)).collect::<Vec<_>>();

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 1;

        let motif_ids = (0..10000u32).collect::<Vec<_>>();


        let resources_group = "group";


        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([find_motifs_run_config_descr(run_id)])
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(run_id), 10, 100, None, None).to_properties_iter(None).collect()),
                )
                .with_list_motifs([
                    (find_motifs_run_id as RunId, Ok(motif_ids.clone()))
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
            motif_ids.iter().map(|m|((*m).into(), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motifs_tensor(100, 3, MotifKind::Pulse))))
        ).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new(Some(select_motifs_run_id))],
            vec![find_motifs_run_id],
            30_000_000, //slices
            None
        ).await?);

        let ctx =  mock_run_ctx::default(1);

        let result = select_motifs_stage.run(standalone_select_input(resources_group), ctx).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_output_builder() -> Result<(), ThoError>  {
        GlobalCtx::init_devices(2);
        let num_channels = 3;
        let mut output_builder = Box::new(SelectMotifsSavableResultBuilder::new(StackedMotifsOutputId::new(0, vec![]), num_channels as u32, 50).await);

        let mask_0 = Tensor::rand(&[3, 5, 7], FLOAT_CUDA).lt(0.5);
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

        for masks_dists_ids in channel_masks_dists_ids {
            match masks_dists_ids {
                None => {}
                Some(v) => {
                    let (mask, _) = v.0.unpack()?;
                    debug!("mask=\t{mask}");//TODO check mask - should be concat mask_0  mask_1
                }
            }
        }


        Ok(())
    }

}
