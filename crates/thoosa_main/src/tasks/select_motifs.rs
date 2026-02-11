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
use rayon::iter::ParallelBridge;
use tch::Device::Cpu;
use tch::Kind::{Float, Uint8};
use tch::{IndexOp, Kind, Tensor};
use tch::kind::FLOAT_CUDA;
use tokio::io::AsyncWriteExt;
use tokio::sync::OwnedRwLockWriteGuard;
use tokio::task::{JoinHandle, JoinSet};
use thoosa_shared::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_shared::runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use thoosa_shared::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContext};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, InputContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::runner::tmp_cache::TmpCached;
use thoosa_common::bool_iter::BoolTensorIterator;
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::cuda_utils::pack_bits_bool;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::find_motifs_run_config::{Aggr as RunConfigAggr, ExtendMotifsConfig, FindMotifsRunConfig, ComparisonMethod as RunConfigMetrics, MotifCriterion, MotifSelector, SelectMotifsRunConfig};
use thoosa_common::runs::{Labeled, Simple};
use thoosa_common::utils::pad_to;
use thoosa_common::ws::Workspace;
use thoosa_motifs::criteria::{coherence, inlier_ratio, norm_thickness};
use thoosa_motifs::find::{FixedFindMotifsConfig, SelectMotifsSavableResult};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::{Resource, ResourceId, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Adjustable, Data, DataId, MonoMetrics, WithId};
use thoosa_shared::comp_id::SynKey;
use thoosa_store::{IdSlice, Store};
use thoosa_with_data::with_data;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::runner::permits::{PermitStep, MAX_PERMITS};
use crate::cli::Commands::SelectMotifs;
use crate::tasks::slices_stats::{Aggr, ScalarStatsValue, Metrics, SlicesStats};
use crate::tasks::stacked_moitfs_loader::{StackedMotifsInput, StackedMotifsLoader, StackedMotifsOutput, StackedMotifsOutputId};
use crate::tasks::cross_batch_clc::{unfold_and_cat, CrossInput, UnfoldParams};
use crate::tasks::cross_dims::{ChannelsData2, ChannelsData2Selector, ChannelsMotifsDists, ChannelsMotifsSlices, MotifSlices};
use crate::tasks::cross_loader::{RangeOrSlice, ResourcesConfigInput};
use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
use crate::tasks::load_motif_and_clc_dist::{DistOutputBuilder, LoadMotifAndClcDistanceInput, LoadMotifAndClcDistanceStage, MotifSlicesSelector};
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
use crate::tasks::find_motifs::config_to_stage;
use crate::tasks::tensor_loader::RawTensorLoader;
use crate::tasks::resources_motifs_clc_dist::{ClcResourcesMotifsDist, ClcResourcesMotifsDistInput, ClcResourcesMotifsDistOutput};

pub use crate::tasks::resources_motifs_clc_dist::{fold_dists, fold_mask};
use crate::tasks::slices_iter::{ResourceSlices, SlicesIter};


//The overall logic in Find and Select Motifs Stages is as following:
// Find Motifs - either
//  * accumulate distance statistics into threshold distance while iterating through slices
//  * precalculate threshold distance statistics from randomly selected slices
//
//  build motifs from slices distance between which is less than threshold distance
//
//  Select Motifs
//   for each select_motif_config
//    multiple find_motifs_run_ids -> select_motif_run_id
//
//   calculate threshold distance from motifs found during Find Motifs stage
//   optionally add slices to motifs which is less than threshold distance.
//
//   distance statistic so far is only Euclidean distance

// Searches for matches between motifs from previous FindMotifsStage runs in the input resources,
// then selects motifs by motif-quality criteria.
// Stacked motifs are loaded in StackedMotifsLoader, while motif statistics are calculated in SelectMotifsStage
// for the requested metric/aggr and then used to build a threshold mask in ClcResourcesMotifsDist matches.
// Finally, each motif is scored by configured criteria, cumulative average is tracked,
// and a motif is dropped as soon as its average criterion score goes below threshold.
pub struct SelectMotifsStage<L, ML, C>//TODO rename to SelectMotifs
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    id: SynKey,
    store: Arc<RwLock<Box<dyn Store>>>,
    ws:Arc<RwLock<Box<dyn Workspace>>>,
    loader:Arc<L>,
    stacked_motifs_loader:Arc<StackedMotifsLoader<ML, C>>,
    run_configs:Vec<SelectMotifsRunConfig>,//TODO map to Vec<Arc in new
    find_motifs_run_ids:Vec<RunId>,
    clc_resources_motifs_dist:Arc<ClcResourcesMotifsDist<C>>,
    max_batch:i64,//TODO to SelectMotifsSFixedConfig,
    clc_dist:Arc<LoadMotifAndClcDistanceStage<ML, SlicesStats<C>, C>>,

    find_motif_run_ids_motif_ids: HashMap<RunId, Vec<u32>>
}

impl <L, ML, C> Debug for SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
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
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
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
           Ok(o.to_cpu().await)
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
                (0..self.channels).map(|c|vec![]).collect::<Vec<Vec<DataId>>>()
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
                            None,
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
}

impl WithId for SelectMotifsInputSelectorWrap {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl WithDataRef for SelectMotifsInputSelectorWrap {
    type O<'o> = ClcResourcesMotifsDistInput<'o>;

    async fn with_data_ref<R, F, P>(&self, _to_device_idx: usize, permits_ctx: &mut P, _trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        F: for <'f> FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError>,
        P: RunCtx,
    {
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
                ), ctx)

        })
    }

    fn cache_area(&self) -> &CacheArea {
        self.stacked_motifs.cache_areas()
    }
}

impl WithSelectDataRef<SelectMotifsInputSelector> for SelectMotifsInputSelectorWrap {
    async fn with_selected_data_ref<'o, R, F, P>(&self, _to_device_idx: usize, ctx: &mut P, selector: SelectMotifsInputSelector, _trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        P: RunCtx,
        for <'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError>
    {

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
                let selected = Arc::new(DevicesWrap::wrap(t.with_selected_data(ctx.device_idx(), ctx, selector, |data, _ctx|{
                    Ok(data)
                }).await?, CacheArea::default()));
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

                do_run(&inpt, ctx)
        })
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
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{

    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader:Arc<L>,
        motif_loader:Arc<ML>,
        run_configs:Vec<SelectMotifsRunConfig>,
        find_motifs_run_ids:Vec<RunId>,
        max_batch: i64,
        #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Self {



        let clc_dist = Arc::new(LoadMotifAndClcDistanceStage::new(
            Arc::clone(&motif_loader),
            Arc::new(SlicesStats::<C>::new())
        ));

        //gpt Snapshot motif ids for each find_motifs_run_id; used later for criteria filtering.
        let find_motif_run_ids_motif_ids = find_motifs_run_ids
            .iter()
            .map(|find_motifs_run_id| {
                store
                    .read()
                    .list_motifs(*find_motifs_run_id)
                    .map(|motif_ids| (*find_motifs_run_id, motif_ids))
            })
            .collect::<Result<HashMap<_, _>, ThoError>>()
            //gpt TODO!! panics on missing motifs; consider propagating a typed error instead.
            .expect(&format!("can't load motifs for find_motifs_run_ids={find_motifs_run_ids:?}"));

        let channels = ws.read().channels() as usize;


        Self {
            id: SynKey::new([mono_key!("SelectMotifsStage", "stage")]),
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
            clc_dist,
            find_motif_run_ids_motif_ids,
        }
    }

    #[inline]
    fn slices_selector_metric_to_stats(metric: RunConfigMetrics) -> Metrics {
        match metric {
            RunConfigMetrics::EucDist(_) => Metrics::EucDist,
            RunConfigMetrics::PearsonCorr(_) => Metrics::PearsonCorr,
        }
    }

    #[inline]
    fn slices_selector_aggr_to_stats(aggr: RunConfigAggr) -> Aggr {
        match aggr {
            RunConfigAggr::Mean(mean_aggr) => Aggr::Mean(mean_aggr.k),
            RunConfigAggr::Min(min_aggr) => Aggr::Min(min_aggr.k),
            RunConfigAggr::Max(max_aggr) => Aggr::Max(max_aggr.k),
            RunConfigAggr::Quantile(quantile_aggr) => Aggr::Quantile(quantile_aggr.q),
        }
    }

    #[inline]
    async fn load_stacked_motifs(store:&Arc<RwLock<Box<dyn Store>>>, stacked_motifs_loader: &Arc<StackedMotifsLoader<ML, C>>, ctx_for_loader: &mut C, find_motifs_run_ids: &Vec<RunId>) -> Result<Vec<(Arc<FindMotifsRunConfig>, Arc<DevicesWrap<StackedMotifsOutput>>)>, ThoError> {
        let find_motifs_run_configs = find_motifs_run_ids
            .iter()
            .map(|find_motifs_run_id| -> Result<_, ThoError> {
                Ok(Arc::new(FindMotifsRunConfig::from_properties_map_with_id(
                    *find_motifs_run_id,
                    &store.read().list_run_config_values(*find_motifs_run_id)?,
                )?))
            })
            .collect::<Result<Vec<_>, ThoError>>()?;

            //for each FindMotifsRunConfigs X SelectMotifsRunConfig
            let futures = find_motifs_run_configs
                .into_iter()
                .map(|find_motifs_run_config| {
                    let stacked_motifs_loader = Arc::clone(&stacked_motifs_loader);
                    let ctx_for_run = ctx_for_loader.branch(Some("select_motifs_stacked_motifs"), false, false);

                    async move {
                        let stacked_motifs_input = Arc::new(DevicesWrap::wrap(
                            StackedMotifsInput::new(Arc::clone(&find_motifs_run_config)),
                            CacheArea::default()
                        ));

                        match stacked_motifs_loader.run_cached(stacked_motifs_input, ctx_for_run).await {
                            Ok(motif_data) => Ok::<_, ThoError>(Some((find_motifs_run_config, motif_data))),
                            Err(e) => {
                                //TODO switch to CPU if OOM ctx.to_cpu...
                                error!("Can't load select_motifs_run_config: e={e:?} select_motifs_run_config={find_motifs_run_config:?}");
                                Ok::<_, ThoError>(None)
                            }
                        }
                    }
                })
                .collect::<Vec<_>>();

            let run_configs__stacked_motifs = collect_futures_with_progress(
                futures,
                1,
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

    async fn load_resources_for_selection(&self, input: &Arc<DevicesWrap<ResourcesConfigInput<Simple>>>, ctx: &mut C) -> Result<Arc<Vec<Resource>>, ThoError> {
        with_data!(ctx.branch(Some("select_motifs_resources"), false, false), input; |_ctx, input: &ResourcesConfigInput<Simple>|{
            Ok::<Arc<Vec<Resource>>, ThoError>(Arc::new(input
                .resources_config()
                .to_resources(&self.store)?
                .into_iter()
                .collect::<Vec<_>>()))
        })
    }

    async fn extend_motifs_for_run_config(
        &self,
        select_motifs_run_config: &SelectMotifsRunConfig,
        extend_config: &ExtendMotifsConfig,
        resources: Arc<Vec<Resource>>,
        num_channels: usize,
        total_size: usize,
        ctx: &mut C,
    ) -> Result<Vec<HashMap<DataId, HashSet<IdSlice>>>, ThoError> {
        let slices_selector_metric = Self::slices_selector_metric_to_stats(extend_config.slices_selector_metric);
        let slices_selector_aggr = Self::slices_selector_aggr_to_stats(extend_config.slices_selector_aggr);
        // This stays inside the run-config loop.
        // Stacked motifs are cached by find_motifs_run_id, while stats depend on metric/aggr (including floats in aggr).
        // Stats must be computed separately so motifs and statistics can be cached independently.
        let run_configs__stacked_motifs = Self::load_stacked_motifs(
            &self.store,
            &self.stacked_motifs_loader,
            ctx,
            &self.find_motifs_run_ids,
        )
        .await?;

        let channels_motifs_added_slices: Arc<Vec<tokio::sync::RwLock<HashMap<DataId, HashSet<IdSlice>>>>> =
            Arc::new((0..num_channels).map(|_| tokio::sync::RwLock::new(HashMap::new())).collect());

        let run_configs__stacked_motifs_dbg = &run_configs__stacked_motifs;
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
                let mut ctx_for_find_run = ctx.branch(None, true, false);
                let clc_dist = Arc::clone(&self.clc_dist);
                let max_batch = self.max_batch;
                let slices_selector_metric = slices_selector_metric;
                let slices_selector_aggr = slices_selector_aggr;
                let run_configs__stacked_motifs_dbg = run_configs__stacked_motifs_dbg;

                async move {
                    match (async {
                        let motifs_stats = Self::clc_stacked_motifs_statistics(
                            clc_dist,
                            Arc::clone(&motif_data),
                            slices_selector_metric,
                            slices_selector_aggr,
                            &mut ctx_for_find_run,
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
                            ctx_for_find_run,
                        )
                        .await
                    })
                    .await {
                        Ok(v) => Ok::<_, ThoError>(Some(v)),
                        Err(e) => {
                            error!("SelectMotifsStage::run: failed to select slices for {run_configs__stacked_motifs_dbg:?}: e={e:?}");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            })
            .collect::<Vec<_>>();

        let _ = collect_futures_with_progress(
            futures,
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
        clc_dist: Arc<LoadMotifAndClcDistanceStage<ML, SlicesStats<C>, C>>,
        motif_data: Arc<DevicesWrap<StackedMotifsOutput>>,
        slices_selector_metric: Metrics,
        slices_selector_aggr: Aggr,
        ctx: &mut C,
    ) -> Result<Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>, ThoError> {
        // TODO: move this into a dedicated stage and cache the resulting motifs stats separately from stacked motifs.
        let (channels_motif_ids, find_motifs_run_config, max_motif_slices) = with_data!(ctx.branch(Some("select_motifs_stacked_motifs_meta"), false, false), motif_data;
            |_ctx, motif_data:&StackedMotifsOutput|{
                let channels_motif_ids = motif_data.motifs.extra().0.clone();
                let find_motifs_run_config = Arc::clone(&motif_data.motifs.extra().1);
                let max_motif_slices = motif_data.motifs.data().t_ref().size()[1] as usize;
                Ok::<_, ThoError>((channels_motif_ids, find_motifs_run_config, max_motif_slices))
        })?;

        let cache_area = motif_data.cache_areas().clone();
        let device_idx = ctx.device_idx();
        let motif_ids = channels_motif_ids
            .iter()
            .flat_map(|ids| ids.iter().cloned())
            .collect::<HashSet<_>>()
            .into_iter()
            .collect::<Vec<_>>();

        let futures = motif_ids
            .into_iter()
            .map(|motif_id| {
                let clc_dist = Arc::clone(&clc_dist);
                let mut ctx_for_motif = ctx.branch(Some("select_motifs_clc_stacked_motifs_statistics_motif"), false, false);
                let cache_area = cache_area.clone();
                async move {
                    let result = (async {
                        let motif_loader_input = MotifLoaderInput::find_motif_slices_only(motif_id.clone());
                        let clc_input = Arc::new(DevicesWrap::wrap(
                            LoadMotifAndClcDistanceInput::new(motif_loader_input, slices_selector_metric, slices_selector_aggr),
                            cache_area,
                        ));
                        let motif_stat = clc_dist
                            .run(clc_input, ctx_for_motif.branch(Some("select_motifs_clc_stacked_motifs_statistics_clc"), false, false))
                            .await?
                            .with_data_ref(0, &mut ctx_for_motif, None, |dist_val: &ScalarStatsValue, _ctx| Ok(dist_val.value()))
                            .await?;
                        Ok::<_, ThoError>((motif_id, motif_stat))
                    })
                    .await;
                    Ok::<_, ThoError>(result)
                }
            })
            .collect::<Vec<_>>();

        let motif_stats = collect_futures_with_progress(
            futures,
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
                    return Ok(Tensor::zeros(&[max_motif_slices as i64], (Float, ThoDevice::from(device_idx).into())));
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
                let channel_stats_t = Tensor::from_slice(channel_stats.as_slice()).to_device(ThoDevice::from(device_idx).into());
                if channel_stats_t.size()[0] < max_motif_slices as i64 {
                    //same padding as in StackedMotifsLoader::run
                    Ok(pad_to(
                        &channel_stats_t, &[max_motif_slices as i64],
                        //this is to avoid selecting resource slices by MaskMaker::Le in ClcResourcesMotifsDist::do_run when cross-calculated with zero-padded motif slices giving 0 distance which would below any motif statistic; use f32::MIN because motifs_stats tensors are Float.
                        f32::MIN as f64
                    ).expect("pad to failed"))
                } else {
                    Ok(channel_stats_t)
                }
            })
            .collect::<Result<Vec<_>, ThoError>>()?;

        let channels_motifs_stats_t = Tensor::stack(&channels_motifs_stats_tensors, 0);
        let find_motifs_run_id = find_motifs_run_config.id.expect("no id in find_motifs_run_config");
        let motifs_stats = ThoTensor::new(
            SynKey::new([
                mono_key!("motifs_stats", "motifs_stats"),
                mono_key!(find_motifs_run_id, format!("find_motifs_run_id_{find_motifs_run_id}")),
                mono_key!(format!("{slices_selector_metric:?}"), "slices_selector_metric"),
                mono_key!(format!("{slices_selector_aggr:?}"), "slices_selector_aggr"),
            ]),
            channels_motifs_stats_t,
        )?;

        Ok(Arc::new(DevicesWrap::wrap(motifs_stats, cache_area)))
    }

    async fn extend_with_one_find_run(
        loader: Arc<L>,
        ws: Arc<RwLock<Box<dyn Workspace>>>,
        clc_resources_motifs_dist: Arc<ClcResourcesMotifsDist<C>>,
        select_motifs_run_config: SelectMotifsRunConfig,
        find_motifs_run_config: Arc<FindMotifsRunConfig>,
        motif_data: Arc<DevicesWrap<StackedMotifsOutput>>,
        motifs_stats: Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>,
        resources: Arc<Vec<Resource>>,
        channels_motifs_added_slices: Arc<Vec<tokio::sync::RwLock<HashMap<DataId, HashSet<IdSlice>>>>>,
        max_batch: i64,
        total_size: usize,
        mut ctx: C,
    ) -> Result<(), ThoError> {
        let num_resources = resources.len();
        let mut total_processed_size = 0;
        let mut t_1_batch: Vec<_> = vec![];
        let mut t_1_batch_resources: Vec<_> = vec![];
        let mut t_1_batch_dim_0_size = 0;

        for (i_0, resource_0) in resources.iter().enumerate() {
            debug!("SelectMotifsStage::run: on i_0={i_0} resource_0={resource_0:?} num_resources={num_resources}");
            let configured_loader = config_to_stage(&loader, &find_motifs_run_config, &ws)?;

            let t_0 = Arc::clone(&configured_loader)
                .run_cached(Arc::new(DevicesWrap::wrap(resource_0.clone(), CacheArea::default())), ctx.branch(Some("select_motifs_loader"), false, false))
                .await?;

            t_1_batch_dim_0_size += t_0.size().dim_1();
            total_processed_size += t_0.size().dim_1();
            t_1_batch.push(t_0);
            t_1_batch_resources.push(resource_0.clone());

            if t_1_batch_dim_0_size < max_batch && i_0 < num_resources - 1 {
                continue;
            }

            debug!("running batch {} length done {}%", t_1_batch.len(), ((total_processed_size as f32 / total_size as f32) * 100.0) as u32);
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
            let channels = ws.read().channels() as u32;

            let ClcResourcesMotifsDistOutput {
                find_motifs_run_id,
                channel_masks_dists_ids,
                resources_iter,
                ..
            } = OOMHandler::new(
                vec![
                    Adjusters::RunContext(Box::new(PermitsAdjuster::new(MAX_PERMITS, *PermitStep::TWENTY))),
                    Adjusters::RunContext(Box::new(DumpToCpu::new())),
                    Adjusters::Input(Box::new(SelectMotifsAdjuster::new(find_motifs_run_config.window as i64))),
                ],
                None,
            )
            .run(
                &clc_resources_motifs_dist,
                Arc::clone(&clc_resources_input),
                ctx.branch(Some("select_motifs_clc_resources"), false, false),
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
                    PermitsAndDumpToCpuAdjustableContex::new(PermitsAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
                ),
            )
            .await?;

            debug!("SelectMotifsStage::run: channel_masks_dists_ids={channel_masks_dists_ids:?} find_motifs_run_id={find_motifs_run_id}");
            for channel_mask_dists_ids in channel_masks_dists_ids.into_iter().enumerate() {
                if let (channel, Some(mask_dists_ids)) = channel_mask_dists_ids {
                    debug!("SelectMotifsStage::run: channel={channel} mask_dists_ids={:?}", mask_dists_ids.1);
                    let mut motif_added_slices_guard = channels_motifs_added_slices[channel].write().await;
                    let (cpu_mask, _) = mask_dists_ids.0.to_cpu().await?.unpack()?;

                    trace!("SelectMotifsStage::run: channel={channel} cpu_mask=\n{cpu_mask}");
                    let mut mask_in_channel = BoolTensorIterator::from(&cpu_mask);

                    for motif_id in mask_dists_ids.1 {
                        let resources_iter = resources_iter.clone();
                        let slices = motif_added_slices_guard.entry(motif_id.clone()).or_insert(HashSet::new());
                        let mut added_slices = 0;
                        let mut skipped_slices = 0;

                        for slice_1 in resources_iter {
                            if let Some(mask) = mask_in_channel.next() {
                                debug!("SelectMotifsStage::run: motif_id={motif_id:?} channel={channel} slice_1={slice_1:?} mask={mask}");
                                if mask {
                                    slices.insert(IdSlice(0, slice_1.0, slice_1.1));
                                    added_slices += 1;
                                } else {
                                    skipped_slices += 1;
                                }
                            } else {
                                warn!("SelectMotifsStage::run: no mask for motif_id={motif_id:?} channel={channel} slice_1={slice_1:?} ");
                            }
                        }

                        debug!("SelectMotifsStage::run: motif_id={motif_id} skipped_slices={skipped_slices} added_slices={added_slices}")
                    }
                }
            }

            debug!("SelectMotifsStage::run: 2. find_motifs_run_config={find_motifs_run_config:?}");
            print!(".");

            t_1_batch_dim_0_size = 0;
            t_1_batch = vec![];
            t_1_batch_resources.clear();
        }

        debug!("SelectMotifsStage::run: processed find_motifs_run_config={find_motifs_run_config:?} select_motifs_run_config={select_motifs_run_config:?}");
        Ok(())
    }

    fn clc_criterion_score(motif_slices: &Tensor, criterion: &MotifCriterion) -> Result<f64, ThoError> {
        let score_t = match criterion {
            MotifCriterion::NormThickness(v) => norm_thickness(motif_slices, v.eps)?,
            MotifCriterion::InlierRatio(v) => inlier_ratio(motif_slices, v.k, v.eps)?,
            MotifCriterion::Coherence(v) => coherence(motif_slices, v.k.round().max(1.0) as i64, v.eps)?,
        };
        let raw_score = score_t.f_double_value(&[])?;
        if raw_score.is_finite() {
            Ok(raw_score.clamp(0.0, 1.0))
        } else {
            warn!("SelectMotifsStage::run: got non-finite criterion score={raw_score}, treating it as 0.0");
            Ok(0.0)
        }
    }

    async fn filter_motifs_by_criteria(
        &self,
        select_motifs_run_config: &SelectMotifsRunConfig,
        // motif_ids: Vec<(u32, f64)>,
        ctx: &mut C,
    ) -> Result<Vec<(u32, f64)>, ThoError> {
        assert!((0.0..=1.0).contains(&select_motifs_run_config.threshold), "SelectMotifsRunConfig::threshold must be within 0..1, got {}", select_motifs_run_config.threshold);
        debug!("SelectMotifsStage::run: 0. self.find_motif_run_ids_motif_ids={:?}", self.find_motif_run_ids_motif_ids);



        let motif_criteria: Vec<MotifCriterion> = Vec::from(&select_motifs_run_config.motif_criteria);
        if motif_criteria.is_empty() {
            debug!("SelectMotifsStage::run: no motif criteria configured");
            return Ok(self.find_motif_run_ids_motif_ids.values().flat_map(|motifs| motifs.iter()).map(|v| (*v, 1.0)).collect::<Vec<_>>());
        }

        let extended_motif_ids = self.find_motif_run_ids_motif_ids.values().flat_map(|motifs| motifs.iter()).collect::<Vec<_>>();
        debug!("SelectMotifsStage::run: 1. extended_motif_ids.len={}", extended_motif_ids.len());

        let threshold = select_motifs_run_config.threshold as f64;
        let mut extended_motif_ids_filtered = Vec::with_capacity(extended_motif_ids.len());

        for motif_id in extended_motif_ids {
            //gpt TODO!! MotifLoaderInput::all_slices ignores select_motifs_run_id; criteria see slices from all select runs, not just this run_config.
            let motif_data = Arc::clone(&self.clc_dist.motif_loader)
                .run_cached(
                    Arc::new(DevicesWrap::wrap(MotifLoaderInput::all_slices(DataId(*motif_id)), CacheArea::default())),
                    ctx.branch(Some("select_motifs_filter_motif_loader"), false, false),
                )
                .await?;

            let kee_score = with_data!(ctx.branch(Some("select_motifs_filter_motif_criteria"), false, false), motif_data; |_ctx, motif_data: &DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>| {
                let motif_slices = motif_data.data().t_ref();
                if motif_slices.size()[0] < 3 {
                    return Ok::<(bool, f64), ThoError>((false, 0.0));
                }
                let mut score_sum = 0.0f64;
                let mut avg_score = 1.0;
                for (idx, criterion) in motif_criteria.iter().enumerate() {
                    let criterion_score = Self::clc_criterion_score(motif_slices, criterion)?;
                    score_sum += criterion_score;
                    avg_score = score_sum / (idx + 1) as f64;
                    debug!("SelectMotifsStage::run: motif_id={motif_id} criterion={criterion:?} criterion_score={criterion_score:.6} avg_score={avg_score:.6} threshold={threshold:.6}");
                    if avg_score < threshold {
                        return Ok::<(bool, f64), ThoError>((false, avg_score));
                    }
                }

                Ok::<(bool, f64), ThoError>((true, avg_score))
            })?;

            if kee_score.0 {
                extended_motif_ids_filtered.push((*motif_id, kee_score.1));
            }
        }

        debug!("SelectMotifsStage::run: run_config={select_motifs_run_config:?} filtered extended_motif_ids.len={}", extended_motif_ids_filtered.len());
        Ok(extended_motif_ids_filtered)
    }

}


impl <L,ML, C> Stage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static

{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<Simple>;
    type Out = SelectMotifsSavableResult;
}


impl <L, ML, C> AsyncStage  for  SelectMotifsStage<L, ML, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + 'static,
    ML: for <'i> AsyncStage<In<'i>=MotifLoaderInput, Out=DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + Adjustable + 'static
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SelectMotifsStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let resources = self.load_resources_for_selection(&input, &mut ctx).await?;

        debug!("SelectMotifsStage::run: resources={resources:?}");
        if resources.is_empty() {
            warn!("SelectMotifsStage::run: no labels resources");
            return Err(ThoError::new(format!(
                "no resources found in {input:?}"
            )))
        }

        let num_channels = self.ws.read().channels() as usize;
        let total_size = resources.iter().map(|resource| resource.length() as usize).sum::<usize>();

        let mut run_config_extended_motif_ids: Vec<(SelectMotifsRunConfig, Vec<(u32, f64)>)> = vec![];
        debug!("SelectMotifsStage::run: self.run_configs.len={}", self.run_configs.len());
        for select_motifs_run_config in self.run_configs.iter() {

            if let Some(extend_config) = select_motifs_run_config.extend_config.as_ref() {
                let channels_motifs_added_slices = self.extend_motifs_for_run_config(
                    select_motifs_run_config,
                    extend_config,
                    Arc::clone(&resources),
                    num_channels,
                    total_size,
                    &mut ctx,
                ).await?;

                let select_motifs_run_config_id = select_motifs_run_config.id.expect("no select_motifs_run_config_id");
                for (channel, motif_slices) in channels_motifs_added_slices.into_iter().enumerate() {
                    for (motif_id, slices) in motif_slices {
                        debug!("SelectMotifsStage::run: motif_id={motif_id} slices.len={} channel={channel}", slices.len());
                        self.store.write().add_slices_to_motif(select_motifs_run_config_id, *motif_id, channel as i64, slices)?;
                    }
                }
            }

            let extended_motif_ids_filtered = self
                .filter_motifs_by_criteria(select_motifs_run_config, &mut ctx)
                .await?;

            run_config_extended_motif_ids.push((select_motifs_run_config.clone(), extended_motif_ids_filtered));

        }


        debug!("SelectMotifsStage::run: extended_motif_ids={run_config_extended_motif_ids:?}");

        Ok(Arc::new(DevicesWrap::wrap(SelectMotifsSavableResult::new(run_config_extended_motif_ids), CacheArea::default())))
    }
}


#[cfg(test)]
pub mod tests {
    use std::collections::{HashMap, HashSet};
    use std::path::Path;
    use std::sync::Arc;
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
    use thoosa_shared::runner::adjusters::splits::OutputBuilder;
    use thoosa_shared::runner::common::cachable::{DevicesWrap, WithDataRef};
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::{mock_run_ctx, MockRunCtx, RunCtx};
    use thoosa_shared::runner::common::DropIf::IfNotMax;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_shared::runner::permits::MAX_PERMITS;
    use thoosa_common::find_motifs_run_config::{ByAvgEucDist, FindMotifsRunConfig, MotifSelector, NormThicknessRunConfig, SelectMotifsRunConfig};
    use thoosa_common::runs::{Labeled, Simple};
    use thoosa_common::tch_test_utils;
    use thoosa_common::utils::{draw_function_plot, reset_print_options, to_vec_f32, z_norm, z_norm_2};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_motifs::distance::clc_dist_d2_and_mask;
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
    use thoosa_store::{IdSlice, MockStore, MockStoreBuilder, Store};
    use thoosa_common::tho_tensor::ThoTensor;
    use crate::tasks::cross_dims::ChannelsData2;
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
    use crate::tasks::load_motif_and_clc_dist::LoadMotifAndClcDistanceStage;
    use crate::tasks::motif_loader::{mock_motif_loader, MotifLoaderInput, MotifMeta};
    use crate::tasks::motif_loader::mock_motif_loader::MockMotifLoader;
    use crate::tasks::resources_motifs_clc_dist::ClcResourcesMotifsDistOutput;
    use crate::tasks::slices_stats::{Aggr, Metrics, SlicesStats};
    use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use crate::tasks::select_motifs::{fold_dists, fold_mask, SelectMotifsSavableResultBuilder, SelectMotifsStage};
    use crate::tasks::slices_iter::SlicesIter;
    use crate::tasks::stacked_moitfs_loader::{StackedMotifsOutput, StackedMotifsOutputId};
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use crate::tasks::tensor_loader::mock_tensor_loader::MockTensorLoader;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_1_resource_1_label_1_channel_1_motif() -> Result<(), ThoError> {
        //TODO this doesn't work
        // let dash_map = Box::leak(Box::new(DashMap::new()));
        // MockMotifLoader::cache_context().expect().times(0..).return_const(&*dash_map);


        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(LevelFilter::Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);



        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 10000)));

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 10000);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

        let channel_0_motifs_ids = vec![1];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[vec![(30, 100, MotifKind::Pulse)]], 100, 10.0f32)?;
        let resources_channels_intervals = [(&resource_a, &[channel_0_motifs_ids.as_slice()][..], &resource_a_intervals)];

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
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
                .build()));

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
            (MotifLoaderInput::find_motif_slices_only(DataId(1)), (MotifMeta { channel: 0, find_motifs_run_id: find_motifs_run_id, window: 100 }, motif_t)),
        ]).finish());

        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByAvgEucDist(ByAvgEucDist { k: 0.5 }))],
            vec![find_motifs_run_id],
            1000,
            None,
        ));

        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))), CacheArea::default())), mock_run_ctx::default(1),
        ).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_2_resources_1_label_1_channel_1_motif() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(LevelFilter::Info);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 800);
        let resource_b: Resource = Resource::new(2, 1, "resource_b", 600);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

        let channel_0_motifs_ids = vec![1];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[vec![(3, 100, MotifKind::Pulse)]], 100, 10.0f32)?;
        let (resource_b_t, resource_b_intervals) = tch_test_utils::resource_tensor(resource_b.length() as usize, &[vec![(2, 100, MotifKind::Pulse)]], 200, 10.0f32)?;
        let resources_channels_intervals = [(&resource_a, &[channel_0_motifs_ids.as_slice()][..], &resource_a_intervals), (&resource_b, &[channel_0_motifs_ids.as_slice()][..],  &resource_b_intervals)];

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
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
                .build()));


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
            (MotifLoaderInput::find_motif_slices_only(DataId(1)), (MotifMeta { channel: 0, find_motifs_run_id: find_motifs_run_id, window: 100 }, motif_t)),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByAvgEucDist(ByAvgEucDist { k: 0.5 }))],
            vec![find_motifs_run_id],
            1000,
            None
        ));

        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))), CacheArea::default())), mock_run_ctx::default(0),
        ).await?;

        Ok(())
    }

    fn build_mocks(){

    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_1_resource_1_label_1_channel_2_motifs() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/test_single_label_one_motif");//.join(cdt_for_dir());
        fs::create_dir_all(test_dir).await?;

        log::set_max_level(LevelFilter::Info);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000)));

        let resource_a: Resource = Resource::new(1, 1, "resource_a", 1000);

        let find_motifs_run_id = 1;
        let select_motifs_run_id = 2;

        let motifs_ids = vec![1, 2];

        let resources_group = "group";

        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor(resource_a.length() as usize, &[vec![(1, 100, MotifKind::Pulse), (1, 100, MotifKind::Triangle)]], 300, 10.0f32)?;

        let resources_channels_intervals = [(&resource_a, &[motifs_ids.as_slice()][..], &resource_a_intervals)];

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
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
                .build()));


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
            (MotifLoaderInput::find_motif_slices_only(DataId(motifs_ids[0])), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_t)),
            (MotifLoaderInput::all_slices(DataId(motifs_ids[1])), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_1_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(DataId(motifs_ids[1])), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_1_t)),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByAvgEucDist(ByAvgEucDist { k: 0.5 }))],
            vec![find_motifs_run_id],
            1000,
            None
        ));


        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))), CacheArea::default())), mock_run_ctx::default(0),
        ).await?;

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
        ], 100, 10.0f32)?;
        let (resource_b_t, resource_b_intervals) = tch_test_utils::resource_tensor(resource_b.length() as usize, &[
            vec![(2, 100, MotifKind::Pulse)],
            vec![(1, 100, MotifKind::Sin)],
            vec![]
        ], 100, 10.0f32)?;
        let (resource_c_t, resource_c_intervals) = tch_test_utils::resource_tensor(resource_c.length() as usize, &[
            vec![(2, 100, MotifKind::Triangle)],
            vec![(1, 100, MotifKind::Sin)],
            vec![]
        ], 100, 10.0f32)?;

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

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
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
                .build()));


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
            (MotifLoaderInput::find_motif_slices_only(DataId(channel_0_motif_0)), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_0_t)),

            (MotifLoaderInput::all_slices(DataId(channel_0_motif_1)), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_1_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(DataId(channel_0_motif_1)), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_0_1_t)),

            (MotifLoaderInput::all_slices(DataId(channel_1_motif_0)), (MotifMeta { channel: 1, find_motifs_run_id, window: 100 }, motif_1_0_t.shallow_clone())),
            (MotifLoaderInput::find_motif_slices_only(DataId(channel_1_motif_0)), (MotifMeta { channel: 1, find_motifs_run_id, window: 100 }, motif_1_0_t)),
        ]).finish());


        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByAvgEucDist(ByAvgEucDist { k: 0.5 }))],
            vec![find_motifs_run_id],
            10000, //TODO fails with max_batch 1000
            Some(801) //TODO check if ran through OOMHandler
        ));


        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))), CacheArea::default())), mock_run_ctx::default(1),
        ).await?;

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

    #[test]
    fn resource_tensor_supports_multiple_kinds_per_channel() -> Result<(), ThoError> {
        let (tensor, intervals) = tch_test_utils::resource_tensor(
            1000,
            &[vec![(1, 50, MotifKind::Pulse), (1, 60, MotifKind::Sin)], vec![(2, 80, MotifKind::Triangle)]], 300, 10.0f32,
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

    #[test]
    fn resource_tensor_offsets_channels_when_space_allows() -> Result<(), ThoError> {
        let (_, intervals) = tch_test_utils::resource_tensor(
            1000,
            &[vec![(1, 100, MotifKind::Pulse)], vec![(1, 120, MotifKind::Triangle)]], 300, 10.0f32,
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
        GlobalCtx::init_devices(1);

        let find_motifs_run_id = 7u32;
        let find_motifs_run_config = Arc::new(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), 1, 3, None, None));

        let channels_motif_ids = vec![
            vec![DataId(11), DataId(11), DataId(12)],
            vec![DataId(13)],
        ];
        let motifs_t = Tensor::zeros([2, 4, 3], FLOAT_CPU);
        let motifs = ThoTensor::new(SynKey::new([mono_key!("motifs", "test_motifs")]), motifs_t)?;
        let stacked_motifs_output = StackedMotifsOutput::new(
            SynKey::new([mono_key!("StackedMotifsOutput", "test_stacked_motifs_output")]),
            DataWithExtra::new(motifs, (channels_motif_ids, Arc::clone(&find_motifs_run_config), vec![])),
        );
        let motif_data = Arc::new(DevicesWrap::wrap(stacked_motifs_output, CacheArea::default()));

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (
                MotifLoaderInput::find_motif_slices_only(DataId(11)),
                (
                    MotifMeta { channel: 0, find_motifs_run_id, window: 3 },
                    Tensor::from_slice(&[0.0f32, 0.0, 0.0, 3.0, 0.0, 0.0]).view([2, 3]),
                ),
            ),
            (
                MotifLoaderInput::find_motif_slices_only(DataId(12)),
                (
                    MotifMeta { channel: 0, find_motifs_run_id, window: 3 },
                    Tensor::from_slice(&[0.0f32, 0.0, 0.0, 4.0, 0.0, 0.0]).view([2, 3]),
                ),
            ),
            (
                MotifLoaderInput::find_motif_slices_only(DataId(13)),
                (
                    MotifMeta { channel: 1, find_motifs_run_id, window: 3 },
                    Tensor::from_slice(&[0.0f32, 0.0, 0.0, 5.0, 0.0, 0.0]).view([2, 3]),
                ),
            ),
        ]).finish());
        let clc_dist = Arc::new(LoadMotifAndClcDistanceStage::new(motif_loader, Arc::new(SlicesStats::<MockRunCtx>::new())));

        let mut ctx = mock_run_ctx::default(0);
        let motifs_stats = SelectMotifsStage::<MockTensorLoader, MockMotifLoader, MockRunCtx>::clc_stacked_motifs_statistics(
            clc_dist,
            motif_data,
            Metrics::EucDist,
            Aggr::Mean(1.0),
            &mut ctx,
        )
        .await?;

        let mut ctx_for_data = ctx.branch(Some("test_clc_stacked_motifs_statistics_data"), false, false);
        let motifs_stats_t: Tensor = motifs_stats
            .with_data_ref(0, &mut ctx_for_data, None, |motifs_stats: &ThoTensor<_, _>, _ctx| Ok(motifs_stats.t_ref().shallow_clone()))
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
        GlobalCtx::init_devices(1);
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
            Ok(vec![11, 12])
        });
        store_mock.expect_list_run_config_values().times(0);
        store_mock.expect_add_slices_to_motif().times(0);

        let store: Arc<RwLock<Box<dyn Store>>> = Arc::new(RwLock::new(Box::new(store_mock)));

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

        let mut select_config = SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByAvgEucDist(ByAvgEucDist { k: 0.5 }));
        select_config.extend_config = None;
        select_config.threshold = 0.0;

        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![select_config],
            vec![find_motifs_run_id],
            1000,
            None,
        ));

        let result = select_motifs_stage
            .run(
                Arc::new(DevicesWrap::wrap(
                    ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))),
                    CacheArea::default(),
                )),
                mock_run_ctx::default(0),
            )
            .await?;

        let mut ctx_for_result = mock_run_ctx::default(0);
        let result_debug = result
            .with_data_ref(0, &mut ctx_for_result, None, |v, _ctx| Ok(format!("{v:?}")))
            .await?;
        assert!(result_debug.contains("(11,"));
        assert!(result_debug.contains("(12,"));

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_run_filters_motifs_by_criteria_threshold() -> Result<(), ThoError> {
        GlobalCtx::init_devices(1);
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

        let store: Arc<RwLock<Box<dyn Store>>> = Arc::new(RwLock::new(Box::new(store_mock)));
        let loader = Arc::new(mock_tensor_loader::build().with_out_id().finish());

        let motif_21_t = Tensor::from_slice(&[
            0.0f32, 1.0, 0.0,
            0.0,    1.0, 0.0,
        ])
            .view([2, 3]);
        let motif_22_t = Tensor::from_slice(&[
            0.0f32, 0.0, 0.0,
            10.0,  10.0, 10.0,
        ])
            .view([2, 3]);

        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(21)), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_21_t)),
            (MotifLoaderInput::all_slices(DataId(22)), (MotifMeta { channel: 0, find_motifs_run_id, window: 100 }, motif_22_t)),
        ]).finish());

        let mut select_config = SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByAvgEucDist(ByAvgEucDist { k: 0.5 }));
        select_config.extend_config = None;
        select_config.motif_criteria.norm_thickness = Some(NormThicknessRunConfig { eps: 1e-8 });
        select_config.threshold = 0.5;

        let select_motifs_stage: Arc<SelectMotifsStage<_, _, MockRunCtx>> = Arc::new(SelectMotifsStage::new(
            &ws,
            &store,
            loader,
            motif_loader,
            vec![select_config],
            vec![find_motifs_run_id],
            1000,
            None,
        ));

        let result = select_motifs_stage
            .run(
                Arc::new(DevicesWrap::wrap(
                    ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))),
                    CacheArea::default(),
                )),
                mock_run_ctx::default(0),
            )
            .await?;

        let mut ctx_for_result = mock_run_ctx::default(0);
        let result_debug = result
            .with_data_ref(0, &mut ctx_for_result, None, |v, _ctx| Ok(format!("{v:?}")))
            .await?;

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

        let folded_mask = fold_mask(mask, &ids, 1);

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

        let folded_mask = fold_mask(mask, &ids, 1);

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
        let folded = fold_mask(mask, &ids, 1);
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

        let folded = fold_dists(x, &ids, 1);
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

        let folded = fold_dists(x, &ids, 2);

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

        let folded_mask = fold_mask(mask, &ids, 1);

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


        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
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
                .build()));


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
                            ], 100, 10.0f32).unwrap().0
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
            vec![SelectMotifsRunConfig::new_(Some(select_motifs_run_id), MotifSelector::ByAvgEucDist(ByAvgEucDist { k: 0.5 }))],
            vec![find_motifs_run_id],
            30_000_000, //slices
            None
        ));

        let ctx =  mock_run_ctx::default(1);

        let result = select_motifs_stage.run(
            Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(NotInGroup(resources_group.into()))), CacheArea::default())), ctx,
        ).await?;

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
                                None,
                            )?,
                            vec![
                                0.into()
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
                                None,
                            )?,
                            vec![
                                0.into()
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
