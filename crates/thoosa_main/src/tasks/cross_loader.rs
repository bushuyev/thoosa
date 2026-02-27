use crate::tasks::cross_batch_clc::{CrossClcResultMerge, CrossInput, CrossInputBatch1Selector, CrossUnfoldClcOutput};
#[cfg(test)]
use crate::tasks::cross_batch_clc::CrossUnfoldClc;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsData2Selector};
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
use crate::tasks::tensor_loader::RawTensorLoader;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, DevicesWrapCopyMode, WithDataRef, WithSelectDataRef};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use futures_util::future::BoxFuture;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, info, warn};
use parking_lot::RwLock;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{AddAssign, Range};
use std::pin::Pin;
use std::sync::Arc;
use std::vec;
use futures_util::FutureExt;
use tch::{IndexOp, Tensor};
use tch::Device::Cpu;
use thoosa_common::dims::{AnyDims, AnyMetrics, AnySub, ChannelsData, ChannelsSlicesData, Dims};
use thoosa_common::runs::Simple;
use thoosa_motifs::find::FindMotifsSavableResult;
use thoosa_shared::{simple_id, CopyToDevice, ThoError};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_shared::runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use thoosa_shared::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContext, SplitsAdjusterContextImpl};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::resource::{Resource, ResourceStore, ResourcesConfig};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, InputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::runner::permits::{PermitStep, MAX_PERMITS};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Named, WithId};
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_shared::mono_key;
use thoosa_store::Store;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::torch_runner::output_builders::SingleDimTensorOutputBuilder;
use thoosa_shared::traits::MonoMetrics;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::run_configs::RunId;
use thoosa_with_data::{async_with_data, with_data};
use crate::tasks::dists_tracker::DistsMask;
use crate::tasks::slices_iter::SlicesIter;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrossUnfoldClcOutputMergeMode {
    GpuMerge,
    CpuMerge,
}

#[derive(Debug)]
pub struct CrossUnfoldClcOutputBuilder<R: Debug> {
    partials: Vec<CrossUnfoldClcOutput<R>>,
    merge_mode: CrossUnfoldClcOutputMergeMode,
    error: Option<ThoError>,
}

impl<R: Debug> CrossUnfoldClcOutputBuilder<R> {
    fn new(merge_mode: CrossUnfoldClcOutputMergeMode) -> Self {
        Self {
            partials: Vec::new(),
            merge_mode,
            error: None,
        }
    }
}

impl<R> OutputBuilder for CrossUnfoldClcOutputBuilder<R>
where
    R: CrossClcResultMerge + Debug + 'static,
{
    type O = CrossUnfoldClcOutput<R>;

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CrossUnfoldClcOutputBuilder::add_partial_output"))]
    async fn add_partial_output(&mut self, o: Self::O) {
        debug!("CrossUnfoldClcOutputBuilder::add_partial_output: o={o:?}");
        if self.error.is_some() {
            return;
        }

        match self.merge_mode {
            CrossUnfoldClcOutputMergeMode::GpuMerge => {
                self.partials.push(o);
            }
            CrossUnfoldClcOutputMergeMode::CpuMerge => {
                let iter_0 = o.iter_0();
                let iter_1 = o.iter_1();
                let clc_result = o.clc_result();
                // Copy packed masks to CPU before any merge to avoid GPU OOM.
                match clc_result.to_cpu().await {
                    Ok(clc_result) => {
                        self.partials
                            .push(CrossUnfoldClcOutput::new(clc_result, iter_0, iter_1, mono_key!("CrossUnfoldClcOutput", "stage").into()));
                    }
                    Err(err) => {
                        self.error = Some(err);
                    }
                }
            }
        }
    }

    async fn build(self) -> Result<Self::O, ThoError> {
        if let Some(err) = self.error {
            return Err(err);
        }

        let mut iter = self.partials.into_iter();
        let first = iter
            .next()
            .ok_or_else(|| ThoError::new("CrossUnfoldClcOutputBuilder: no partial outputs"))?;
        let iter_0 = first.iter_0();
        let iter_1 = first.iter_1();

        let iter_0_step = iter_0.step;
        let iter_1_step = iter_1.step;
        let iter_0_resources = iter_0.resource_ids_sizes;
        let mut iter_1_resources = iter_1.resource_ids_sizes;

        let mut clc_results = Vec::new();

        for partial in iter {
            let iter_1 = partial.iter_1();
            iter_1_resources.extend(iter_1.resource_ids_sizes);
            clc_results.push(partial.clc_result());
        }

        let clc_result = first.clc_result().merge_batch_1_many(clc_results)?;

        Ok(CrossUnfoldClcOutput::new(
            clc_result,
            SlicesIter::new_with_offsets(iter_0_step, iter_0_resources),
            SlicesIter::new_with_offsets(iter_1_step, iter_1_resources),
            mono_key!("CrossUnfoldClcOutput", "stage").into(),//TODO
        ))
    }
}


pub struct CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for<'i> Cached<
        In<'i> = Resource,
        Out = ThoTensor<ChannelsData2, SynKey>,
        Ctx=C
    > + OutId + 'static,

    R: Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<
        In<'i> = CrossInput<'i, Arc<Resource>>,
        Out = CrossUnfoldClcOutput<R>,
        Ctx=C
    > + Adjustable + OutId + Send + Sync + 'static,

    RSV: for<'i> SyncStage<
        In<'i> = CrossUnfoldClcOutput<R>,
        Out = (),
        Ctx=C
    > + Stage<Ctx=C> + Send + Sync + 'static,
{
    find_motifs_run_config_id: RunId,
    configured_loader: Arc<L>,
    store: Arc<Box<dyn Store>>,
    cross_unfold_clc: Arc<CLC>,
    receiver_factory: Box<dyn Fn(&[Arc<Resource>]) -> Arc<RSV> + Send + Sync>,
    chunk_size: usize,
    on_batch_done: Box<dyn Fn(R) -> BoxFuture<'static, Result<R, ThoError>> + Sync + Send>,
    on_all_done: Box<dyn Fn(&Arc<RSV>) -> BoxFuture<'static, Result<(), ThoError>> + Sync + Send>,
    output_merge_mode: CrossUnfoldClcOutputMergeMode,
    step:u32,
    pd: PhantomData<(ResC, R, C)>,
}

impl<ResC, L, CLC, RSV, R, C> Debug for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx=C> + OutId + 'static,
    R: Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Adjustable + OutId + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossLoader").finish()
    }
}

impl<ResC, L, CLC, RSV, R, C> CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx=C> + OutId + 'static,
    R: Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Adjustable + OutId + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
{
    pub fn new<ReceiverFactory, OnAllDone, OnBatchDone>(
        find_motifs_run_config_id: RunId, //fod logging
        configured_loader: Arc<L>,
        store: &Arc<Box<dyn Store>>,
        cross_unfold_clc: Arc<CLC>,
        receiver_factory: ReceiverFactory,
        chunk_size: usize,
        on_batch_done:OnBatchDone,
        on_all_done: OnAllDone,
        output_merge_mode: CrossUnfoldClcOutputMergeMode,
        step: u32,
    ) -> Self
    where
        ReceiverFactory: Fn(&[Arc<Resource>]) -> Arc<RSV> + Sync + Send + 'static,
        OnBatchDone: Fn(R) -> BoxFuture<'static, Result<R, ThoError>> + Sync + Send + 'static,
        OnAllDone: Fn(&Arc<RSV>) -> BoxFuture<'static, Result<(), ThoError>> + Sync + Send + 'static,
    {
        Self {
            find_motifs_run_config_id,
            configured_loader,
            store: Arc::clone(store),
            cross_unfold_clc,
            receiver_factory: Box::new(receiver_factory),
            chunk_size,
            on_batch_done: Box::new(on_batch_done),
            on_all_done: Box::new(on_all_done),
            output_merge_mode,
            pd: Default::default(),
            step
        }
    }
}

//TODO move to lib
#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct ResourcesConfigInput<RC: ResourcesConfig + Debug + Send + Sync + Clone> {
    resources_config: RC,
}

impl<RC: ResourcesConfig + Debug + Send + Sync + Clone> ResourcesConfigInput<RC> {
    pub(crate) async fn check(self, store: &Arc<Box<dyn Store>>) -> Result<Self, ThoError> {
        self.resources_config.check(store).await?;

        Ok(self)
    }
}

impl<RC: ResourcesConfig + Debug + Send + Sync + Clone> ResourcesConfigInput<RC> {
    pub fn new(resources_config: RC) -> Self {
        Self {
            resources_config,
        }
    }


    pub fn resources_config(&self) -> &RC {
        &self.resources_config
    }
}

impl<ResC: ResourcesConfig + Debug + Send + Sync + Clone> WithId for ResourcesConfigInput<ResC> {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl<RC: ResourcesConfig + Debug + Send + Sync + Clone> Input for ResourcesConfigInput<RC> {}


impl<ResC, L, CLC, RSV, R, C> Stage for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx=C> + OutId + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Adjustable + OutId + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
{
    type Ctx=C;

    type In<'a> = ResourcesConfigInput<ResC>
    where
        Self: 'a;
    type Out = ();
}

// #[async_trait]
impl<ResC, L, CLC, RSV, R, C> AsyncStage for CrossLoader<ResC, L, CLC, RSV, R, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'a> Self::In<'a>: Data + WithId + Send + Sync,
    ResourcesConfigInput<ResC>: Data<M=CpuOnly<ResourcesConfigInput<ResC>>> + WithId + Send + Sync,
    CpuOnly<ResourcesConfigInput<ResC>>: DevicesWrapCopyMode<ResourcesConfigInput<ResC>>,
    Self::Out: Data + WithId + Send + Sync,
    <Self::Out as Data>::M: DevicesWrapCopyMode<Self::Out>,
    ResC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx=C> + OutId + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Adjustable + OutId + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CrossLoader::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        // let chunk_size = self.chunk_size;
        let max_batch = i64::try_from(self.chunk_size).unwrap_or(i64::MAX);
        let device_idx = ctx.device_idx();
        let output_merge_mode = self.output_merge_mode;

        let resources_config = input
            .with_data_ref(0, ctx.branch(Some("cross_loader_resources"), false, false), None, |input: &ResourcesConfigInput<ResC>, _ctx| {
                Ok(input.resources_config.clone())
            })
            .await?;
        let resources = resources_config
            .to_resources(&self.store)
            .await?
            .into_iter()
            .filter(|r| r.length() > self.step)
            .map(Arc::new)
            .collect::<Vec<_>>();

        debug!("CrossLoader::run:  max_batch={max_batch} resources={resources:?}");

        let receiver = (self.receiver_factory)(resources.as_slice());

        let mut t_0_batch: Vec<_> = vec![];
        let mut t_0_batch_resources: Vec<_> = vec![];
        let mut t_0_batch_offsets: Vec<u32> = vec![];
        let mut t_0_batch_len = 0;

        let mut skip_0_i = 0;

        for (i_0, resource_0) in resources.iter().enumerate() {
            debug!("CrossLoader::run: find_motifs_run_config_id={} on i_0={i_0} resource_0={resource_0:?}", self.find_motifs_run_config_id);

            let t_0 = Arc::clone(&self.configured_loader)
                .run_cached(Arc::new(DevicesWrap::wrap(resource_0.as_ref().clone(), CacheArea::default())), ctx.branch(Some("cross_loader_t_0"), false, false))
                .await?;

            t_0_batch_len += t_0.size().dim_1();
            t_0_batch.push(t_0);
            t_0_batch_resources.push(Arc::clone(resource_0));
            t_0_batch_offsets.push(0);

            if t_0_batch_len < max_batch && i_0 < resources.len() - 1 {
                continue;
            }

            let mut t_1_batch: Vec<_> = vec![];
            let mut t_1_batch_resources: Vec<_> = vec![];
            let mut t_1_batch_offsets: Vec<u32> = vec![];
            let mut t_1_batch_len = 0;

            for (i_1, resource_1) in resources
                .iter()
                .enumerate()
                .skip(skip_0_i)//skipping lower triangle
            {
                let t_1 = Arc::clone(&self.configured_loader)
                    .run_cached(Arc::new(DevicesWrap::wrap(resource_1.as_ref().clone(), CacheArea::default())), ctx.branch(Some("cross_loader_t_1"), false, false))
                    .await?;

                t_1_batch_len += t_1.size().dim_1();
                t_1_batch.push(t_1);
                t_1_batch_resources.push(Arc::clone(resource_1));
                t_1_batch_offsets.push(0);

                if t_1_batch_len < max_batch && i_1 < resources.len() - 1 {
                    continue;
                }

                debug!("CrossLoader::run: find_motifs_run_config_id={} t_0_batch={t_0_batch:?} t_1_batch={t_1_batch:?}", self.find_motifs_run_config_id);
                let batches_concated = [t_0_batch.clone(), t_1_batch.clone()].concat();
                let cross_unfold_clc = Arc::clone(&self.cross_unfold_clc);


                let t_0_batch_resources_cloned = t_0_batch_resources.clone();
                let t_0_batch_resources_cloned_1 = t_0_batch_resources.clone();
                let t_1_batch_resources_cloned = t_1_batch_resources.clone();
                let t_1_batch_resources_cloned_1 = t_1_batch_resources.clone();

                let batch_1_slice_lengths = t_1_batch
                    .iter()
                    .map(|t| t.size().dim_1())
                    .collect::<Vec<_>>();

                let input = CrossInputSelectorWrap::new(
                    mono_key!(self.find_motifs_run_config_id, "find_motifs_run_config_id").into(),//TODO
                    t_0_batch.clone(),
                    t_1_batch.clone(),
                    t_0_batch_resources_cloned.clone(),
                    t_1_batch_resources_cloned.clone(),
                    CacheArea::default()
                );
                let required_ids: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
                match OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(PermitsAdjuster::new(
                            MAX_PERMITS,
                            *PermitStep::TWENTY,
                        ))),
                        Adjusters::RunContext(Box::new(DumpToCpu::new())),
                        Adjusters::Input(Box::new(CrossLoaderAdjuster::new())),

                    ],
                    None,
                ).run(
                        &cross_unfold_clc,
                        Arc::new(input),
                        ctx.branch("batch_clc_ctx".into(), false, false),
                        Contexts::new(
                            CrossSplitterContext::new(t_1_batch_resources_cloned.len() as i64, batch_1_slice_lengths),
                            NoopRunnableContext {},
                            PermitsAndDumpToCpuAdjustableContex::new(
                                PermitsAdjustContextImpl::new(),
                                DumpToCpuContextImpl::new(&required_ids),
                            ),
                        ),
                ).await {
                    Ok( mut cross_clc_output  ) => {
                        let cross_clc_output = cross_clc_output.replace_clc_result(&self.on_batch_done).await?;

                        let receiver_arc = Arc::clone(&receiver);

                        receiver_arc.consume(cross_clc_output, ctx.branch(Some("cross_loader_receiver"), false, false)).await?;

                    }
                    Err(e) => {
                        warn!("CrossLoader::run: got error {e:?}, find_motifs_run_config_id={}", self.find_motifs_run_config_id);
                    }
                };

                t_1_batch = vec![];
                t_1_batch_resources = vec![];
                t_1_batch_offsets = vec![];
                t_1_batch_len = 0;
            }

            skip_0_i = i_0;
            t_0_batch.clear();
            t_0_batch_resources.clear();
            t_0_batch_offsets.clear();
            t_0_batch_len = 0;


        }

        (self.on_all_done)(&receiver).await?;

        Ok(Arc::new(DevicesWrap::wrap((), CacheArea::default())))
    }
}

#[derive(Debug)]
struct CrossInputSelectorWrap {
    id: SynKey,
    batch_0:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>,
    batch_1:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>,
    resources_0: Vec<Arc<Resource>>,
    resources_1: Vec<Arc<Resource>>,
    cache_area: CacheArea,
}


impl Data for CrossInputSelectorWrap {
    type Metrics = i64;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.resources_1.len() as i64
    }

    fn device(&self) -> usize {
        0
    }
}


impl WithId for CrossInputSelectorWrap {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl CrossInputSelectorWrap {
    pub fn new(
        id:SynKey,
        batch_0: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>,
        batch_1: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>,
        resources_0: Vec<Arc<Resource>>,
        resources_1: Vec<Arc<Resource>>,
        cache_area: CacheArea,
    ) -> Self {
        Self {id, batch_0, batch_1, resources_0, resources_1,  cache_area }
    }
}

#[derive(Debug)]
struct CrossLoaderAdjuster {

    batch_1_adjuster: SplitsAdjuster<Range<i64>, i64>,
    batch_1_slice_adjuster: SplitsAdjuster<Range<i64>, i64>,
}

impl CrossLoaderAdjuster {
    pub fn new() -> Self {
        Self {
            batch_1_adjuster: SplitsAdjuster::new(1),
            batch_1_slice_adjuster: SplitsAdjuster::new(1),
        }
    }
}

#[async_trait]
impl <'a, ST, R> Adjuster<CrossSplitterContext<'a, ST, R>> for CrossLoaderAdjuster
where
    ST:Stage<Out=CrossUnfoldClcOutput<R>> + Adjustable + 'a,
    ST::In<'a>: Data<Metrics = i64>,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
{
    type A = CrossInputSelector;

    async fn adjust(&self, mut adjustable: Self::A, adjust_context: &mut CrossSplitterContext<'a, ST, R>) -> Result<Self::A, AdjusterError<Self::A>> {

        match adjust_context.batch_1_mode {
            Batch1SplitMode::Range => {
                let result = self.batch_1_adjuster.adjust(0i64..0, adjust_context).await;
                debug!("CrossLoaderAdjuster::adjust: adjust_context={adjust_context:?} result={result:?}");
                match result {
                    Ok(range) => {
                        adjustable.batch_1_selector = RangeOrSlice::Range(range);

                        Ok(adjustable)
                    }
                    Err(_) => {
                        Err(AdjusterError::AdjustFailed(adjustable))
                    }
                }
            }
            Batch1SplitMode::Slice => {
                let result = self.batch_1_slice_adjuster.adjust(0i64..0, &mut adjust_context.slice_ctx).await;
                debug!("CrossLoaderAdjuster::adjust: slice adjust_context={adjust_context:?} result={result:?}");
                match result {
                    Ok(range) => {
                        adjustable.batch_1_selector = RangeOrSlice::Slice((adjust_context.slice_resource_idx, range));

                        Ok(adjustable)
                    }
                    Err(_) => Err(AdjusterError::AdjustFailed(adjustable)),
                }
            }
        }

    }

    fn degrade(&self, adjust_context: &mut CrossSplitterContext<'a, ST, R>) -> bool {
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

#[derive(Debug)]
pub enum CrossSplitterContextState<'a, R>
where
    R:Stage<Out=CrossUnfoldClcOutput<DistsMask>> + Adjustable + 'a,

{
    Batch0(SplitsAdjusterContextImpl<'a, R, CrossUnfoldClcOutputBuilder<DistsMask>>),
    Batch1(SplitsAdjusterContextImpl<'a, R, CrossUnfoldClcOutputBuilder<DistsMask>>),
    Slice(SplitsAdjusterContextImpl<'a, R, CrossUnfoldClcOutputBuilder<DistsMask>>)
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
struct CrossSplitterContext<'a, ST, R>
where
    ST:Stage<Out=CrossUnfoldClcOutput<R>> + Adjustable + 'a,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
    ST::In<'a>: Data<Metrics = i64>,
{
    input_len: i64,
    split_len: i64,
    done_to: i64,
    output_builder: CrossUnfoldClcOutputBuilder<R>,
    batch_1_mode: Batch1SplitMode,
    slice_ctx: SliceAdjusterContext,
    slice_resource_idx: usize,
    batch_1_slice_lengths: Vec<i64>,
    pd:PhantomData<&'a ST>

}


impl <'a, ST, R> CrossSplitterContext<'a, ST, R>
where
    ST:Stage<Out=CrossUnfoldClcOutput<R>> + Adjustable + 'a,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
    ST::In<'a>: Data<Metrics = i64>,

{
    pub fn new(batch_1_len:<<ST as Stage>::In<'a> as Data>::Metrics, batch_1_slice_lengths: Vec<i64>) -> Self {
        debug!("CrossSplitterContext::new: batch_1_len={batch_1_len:?}");

        Self {
            input_len: batch_1_len,
            split_len: batch_1_len,
            done_to: 0,
            output_builder: CrossUnfoldClcOutputBuilder::new(CrossUnfoldClcOutputMergeMode::CpuMerge),
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

impl <'a, ST, R> SplitsAdjusterContext for CrossSplitterContext<'a, ST, R>
where
    ST:Stage<Out=CrossUnfoldClcOutput<R>> + Adjustable + 'a,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
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

impl <'a, 'z, ST, R> InputContext<'a, ST> for CrossSplitterContext<'z, ST, R>
where
    ST: Stage<Out = CrossUnfoldClcOutput<R>> + Adjustable + 'a,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
    ST::In<'z>: Data<Metrics = i64>,

{
    async fn add_partial_output(&mut self, out: ST::Out) {
        debug!("CrossSplitterContext::add_partial_output: out={out:?}");
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

#[derive(Debug, Clone)]
pub enum RangeOrSlice {
    Full,
    Range(Range<i64>), //range of tensors
    Slice((usize, Range<i64>))//slice of single tensor
}

impl Default for RangeOrSlice {
    fn default() -> Self {
        RangeOrSlice::Full
    }
}

impl RangeOrSlice {
    pub async fn select<R>(&self, batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>, mut run_ctx: R) -> Result<Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>, ThoError>
    where
        R:RunCtx
    {
        debug!("RangeOrSlice::select: batch={batch:?} self={self:?}");

        Ok(match self {
            RangeOrSlice::Range(range) => {
                batch[range.start as usize..range.end as usize].to_vec()
            }
            RangeOrSlice::Slice((i, range)) => {
                let channels = batch[*i].size().dim_0();
                let selector = ChannelsData2Selector([0, range.start].into()..[channels, range.end].into());
                vec![
                    Arc::new(DevicesWrap::wrap(batch[*i].with_selected_data(run_ctx.device_idx(), run_ctx.branch(Some("cross_loader_range_or_slice_select"), false, false), selector, |data, run_ctx|{
                        Ok(data)
                    }).await?, CacheArea::default()))
                ]
            }
            RangeOrSlice::Full => {
                batch
            }
        })
    }
}


#[derive(Debug, Clone, Default)]
pub struct CrossInputSelector {
    batch_0_selector: RangeOrSlice,
    batch_1_selector: RangeOrSlice,

}

impl Adjustable for CrossInputSelector {}


impl WithDataRef for CrossInputSelectorWrap
{
    type O<'o> = CrossInput<'o, Arc<Resource>>;

    async fn with_data_ref<Rt, F, P>(
        &self,
        _to_device_idx: usize,
        permits_ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<Rt, ThoError>
    where
            for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<Rt, ThoError> + Send,
            P: RunCtx,
    {
        let t_0_batch= self.batch_0.clone();
        let t_1_batch= self.batch_1.clone();
        with_data!(permits_ctx, [t_0_batch], [t_1_batch];
            |
             ctx,
             batch_0:&[&ThoTensor<ChannelsData2, SynKey>],
             batch_1:&[&ThoTensor<ChannelsData2, SynKey>],
            |{

                do_run(&CrossInput::new(
                    [batch_0,batch_1].concat().as_slice(),
                    self.resources_0.as_slice(),
                    self.resources_1.as_slice(),
                    self.id.clone()
                ), &ctx)

        })
    }

    fn cache_area(&self) -> &CacheArea {
        &self.cache_area
    }
}


impl WithSelectDataRef<CrossInputSelector> for CrossInputSelectorWrap {
    async fn with_selected_data_ref<'o, Rt, F, C>(
        &self,
        _to_device_idx: usize,
        mut run_ctx: C,
        selector: CrossInputSelector,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<Rt, ThoError>
    where
            for<'f> F: FnOnce(&Self::O<'f>, &C) -> Result<Rt, ThoError>,
            C: RunCtx,
    {

        debug!("WithSelectDataRef<CrossInputSelector> for CrossInput2::with_data: selector.batch_0_selector={:?} selector.batch_1_selector={:?}", selector.batch_0_selector, selector.batch_1_selector);

        let t_0_batch = selector.batch_0_selector.select(self.batch_0.clone(), run_ctx.branch(Some("cross_loader_batch_0_select"), false, false)).await?;
        let t_1_batch = selector.batch_1_selector.select(self.batch_1.clone(), run_ctx.branch(Some("cross_loader_batch_1_select"), false, false)).await?;

        //TODO move to RangeOrSlice::select
        let resources_0 = match &selector.batch_0_selector {
            RangeOrSlice::Full => self.resources_0.as_slice(),
            RangeOrSlice::Range(range) => {
                let range = range.start as usize..range.end as usize;
                &self.resources_0[range.clone()]
            }
            RangeOrSlice::Slice((i, range)) => {
                std::slice::from_ref(&self.resources_0[*i])
            }
        };

        let resources_1 = match &selector.batch_1_selector {
            RangeOrSlice::Full => self.resources_1.as_slice(),
            RangeOrSlice::Range(range) => {
                let range = range.start as usize..range.end as usize;
                &self.resources_1[range.clone()]
            }
            RangeOrSlice::Slice((i, range)) => {
                std::slice::from_ref(&self.resources_1[*i])
            }
        };

        with_data!(run_ctx, [t_0_batch], [t_1_batch];
            |
             ctx,
             batch_0:&[&ThoTensor<ChannelsData2, SynKey>],
             batch_1:&[&ThoTensor<ChannelsData2, SynKey>],
            |{

                debug!("WithSelectDataRef<CrossInputSelector> for CrossInput2::with_data: batch_0={batch_0:?} batch_1={batch_1:?}");

                do_run(&CrossInput::new(
                    [batch_0,batch_1].concat().as_slice(),
                    resources_0,
                    resources_1,
                    self.id.clone()
                ), &ctx)

        })
    }
}

#[cfg(test)]
pub(crate) mod mock_cross_loader_clc {
    use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
    use thoosa_shared::runner::common::{OutId, RunCtx};
    use crate::tasks::cross_batch_clc::{CrossInput, CrossUnfoldClcOutput};
    use std::fmt::Debug;
    use std::marker::PhantomData;
    use std::sync::Arc;
    use thoosa_shared::ThoError;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{Adjustable, WithId};

    type RunBatchFn<Out, C> = dyn for<'a> Fn(
        &CrossInput<'a, Arc<Resource>>,
        &C,
    ) -> Result<CrossUnfoldClcOutput<Out>, ThoError>
    + Send
    + Sync
    + 'static;

    #[derive(Clone)]
    pub struct MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        run_batch_fn: Arc<RunBatchFn<Out, C>>,
        _ctx: PhantomData<C>,
    }

    impl<Out, C> std::fmt::Debug for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockCrossLoaderClc").finish()
        }
    }

    impl<Out, C> Adjustable for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {}

    impl<Out, C> OutId for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
            input_id.clone()
        }
    }

    impl<Out, C> Stage for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        type Ctx = C;
        type In<'a> = CrossInput<'a, Arc<Resource>>;
        type Out = CrossUnfoldClcOutput<Out>;
    }

    impl<Out, C> SyncStage for MockCrossLoaderClc<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        fn do_run<'s>(
            self: Arc<Self>,
            input: &Self::In<'s>,
            ctx: &Self::Ctx,
        ) -> Result<Self::Out, ThoError>
        where
            Self: Sized,
        {
            let f = Arc::clone(&self.run_batch_fn);
            f(input, ctx)
        }
    }

    pub struct MockCrossLoaderClcBuilder<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        run_batch_fn: Option<Arc<RunBatchFn<Out, C>>>,
        _ctx: PhantomData<C>,
    }

    impl<Out, C> MockCrossLoaderClcBuilder<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        pub fn with_run_batch_fn<F>(mut self, f: F) -> Self
        where
            F: for<'a> Fn(
                &CrossInput<'a, Arc<Resource>>,
                &C,
            ) -> Result<CrossUnfoldClcOutput<Out>, ThoError>
            + Send
            + Sync
            + 'static,
        {
            self.run_batch_fn = Some(Arc::new(f));
            self
        }

        pub fn finish(self) -> MockCrossLoaderClc<Out, C> {
            let run_batch_fn = self
                .run_batch_fn
                .expect("MockCrossLoaderClcBuilder requires run_batch_fn to be set");
            MockCrossLoaderClc {
                run_batch_fn,
                _ctx: PhantomData,
            }
        }
    }

    pub fn build<Out, C>() -> MockCrossLoaderClcBuilder<Out, C>
    where
        Out: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        MockCrossLoaderClcBuilder {
            run_batch_fn: None,
            _ctx: PhantomData,
        }
    }
}

#[cfg(test)]
#[allow(refining_impl_trait)]
pub(crate) mod mock_cross_loader_receiver {
    use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
    use thoosa_shared::runner::common::RunCtx;
    use mockall::mock;
    use std::fmt::Debug;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use thoosa_shared::ThoError;
    use thoosa_shared::applicables_runner::{Input, Output};

    mock! {
        #[derive(Debug)]
        pub CrossLoaderReceiver<In, C>
        where
            In: Input + Output + Send + Debug + 'static,
            C: RunCtx + 'static,
        {
        }

        impl<In, C> Stage for CrossLoaderReceiver<In, C>
        where
            In: Input + Output + Send + Debug + 'static,
            C: RunCtx + 'static,
        {
            type Ctx = C;
            type In<'a> = In where Self: 'a;
            type Out = ();
        }

        impl<In, C> SyncStage for CrossLoaderReceiver<In, C>
        where
            In: Input + Output + Send + Debug + 'static,
            C: RunCtx + 'static,
        {
            fn consume<'s>(
                self: Arc<Self>,
                out: <Self as Stage>::In<'static>,
                ctx: <Self as Stage>::Ctx,
            ) -> Pin<Box<dyn Future<Output = Result<<Self as Stage>::Out, ThoError>> + Send + 's>>;
        }
    }

    pub struct MockCrossLoaderReceiverBuilder<In, C>
    where
        In: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        mock: MockCrossLoaderReceiver<In, C>,
        consume_configured: bool,
    }

    impl<In, C> MockCrossLoaderReceiverBuilder<In, C>
    where
        In: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        pub fn with_receive_fn<F>(mut self, f: F) -> Self
        where
            F: Fn(In, C) -> Result<(), ThoError> + Send + Sync + 'static,
        {
            let receive_fn = Arc::new(f);
            self.mock
                .expect_consume()
                .returning(move |out, ctx| {
                    let receive_fn = Arc::clone(&receive_fn);
                    Box::pin(async move { receive_fn(out, ctx) })
                });
            self.consume_configured = true;
            self
        }

        pub fn finish(self) -> MockCrossLoaderReceiver<In, C> {
            assert!(
                self.consume_configured,
                "MockCrossLoaderReceiverBuilder requires receive_fn to be set"
            );
            self.mock
        }
    }

    pub fn build<In, C>() -> MockCrossLoaderReceiverBuilder<In, C>
    where
        In: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        MockCrossLoaderReceiverBuilder {
            mock: MockCrossLoaderReceiver::new(),
            consume_configured: false,
        }
    }
}


#[cfg(test)]
mod tests {
    use super::mock_cross_loader_clc;
    use super::mock_cross_loader_receiver;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use super::*;
    use crate::tasks::cross_batch_clc::{CrossUnfoldClc, CrossUnfoldClcOutput, UnfoldParams};
    use crate::tasks::cross_loader::{
        CrossLoader, CrossUnfoldClcOutputMergeMode, ResourcesConfigInput,
    };
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType, DistsTracker};
    use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use crate::tasks::slices_iter::SlicesIter;
    use thoosa_shared::runner::common::{mock_run_ctx, MockRunCtx};
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use futures_util::FutureExt;
    use log::{debug, set_max_level};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, Instant};
    use log::LevelFilter::{Debug, Info};
    use tch::{Device, Kind, Tensor};
    use tch::Device::Cuda;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::Kind::{Float, Uint8};
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::ws::FsWorkspace;
    use thoosa_motifs::find::FixedFindMotifsConfig;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::{Resource, ResourceStats};
    use thoosa_shared::{cdt_for_dir, ResourceType, ThoError};
    use thoosa_shared::mono_key;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_store::fs_store::ChunkedSlicesReader;
    use thoosa_store::{MockStoreBuilder, SlicesMatch, SqlStore};
    use tokio::task::yield_now;
    use tokio::time::sleep;
    use tracing::subscriber::NoSubscriber;
    use thoosa_common::copy_ctx::CopyCtx;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_shared::Cause::SomethingElse;
    use thoosa_shared::perf_logging::set_perf_log_level;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Observation<ID> {
        tensor_ids: Vec<ID>,
        batch_0_ids: Vec<u32>,
        batch_1_ids: Vec<u32>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct SplitObservation {
        batch_0_ids: Vec<u32>,
        batch_1_ids: Vec<u32>,
        tensor_lens: Vec<i64>,
    }

    #[derive(Debug, Clone, PartialEq, Eq, DummyData)]
    struct TestBatchOutput {
        id: SynKey,
        call_index: usize,
    }

    impl TestBatchOutput {
        fn new(call_index: usize) -> Self {
            Self {
                id: SynKey::new([mono_key!("batch_output", "batch_output")]),
                call_index,
            }
        }
    }

    impl WithId for TestBatchOutput {
        type ID = SynKey;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl Input for TestBatchOutput {}
    impl Output for TestBatchOutput {}

    impl CrossClcResultMerge for TestBatchOutput {
        fn to_cpu(self) -> BoxFuture<'static, Result<Self, ThoError>> {
            async move { Ok(self) }.boxed()
        }

        fn merge_batch_1(self, _other: Self) -> Result<Self, ThoError> {
            Ok(self)
        }
    }

    #[derive(Debug, Clone)]
    struct CpuMergeOutput {
        id: SynKey,
        to_cpu_calls: Arc<AtomicUsize>,
    }

    impl CpuMergeOutput {
        fn new(to_cpu_calls: Arc<AtomicUsize>, call_index: usize) -> Self {
            Self {
                id: SynKey::new([mono_key!("cpu_merge_output", format!("call_{call_index}"))]),
                to_cpu_calls,
            }
        }
    }

    impl WithId for CpuMergeOutput {
        type ID = SynKey;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl CrossClcResultMerge for CpuMergeOutput {
        fn to_cpu(self) -> BoxFuture<'static, Result<Self, ThoError>> {
            async move {
                self.to_cpu_calls.fetch_add(1, Ordering::SeqCst);
                Ok(self)
            }
            .boxed()
        }

        fn merge_batch_1(self, _other: Self) -> Result<Self, ThoError> {
            Ok(self)
        }
    }

    fn slice_sum(base: f32, slice_idx: u32) -> f32 {
        2.0 * base + (2.0 * slice_idx as f32 + 1.0) / 10.0
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_loader_adjuster() -> Result<(), ThoError> {
        let ctx = GlobalCtx::init_devices(2);

        let ca = CacheArea::new(1);
        let batch_0_len = 2;
        let batch_1_len = 2;

        let resources_0 = (0..batch_0_len).map(|i| Arc::new(Resource::new(i, 1, format!("aaa_{}", i), 100))).collect::<Vec<_>>();
        let batch_0_tensors = (0..batch_0_len).map(|i| {
            let i = i as i64;
            Tensor::arange_start(i*1000, (i+1)*1000, FLOAT_CPU).view((1, -1))
        }).collect::<Vec<_>>();
        let batch_0 = (0..batch_0_len).map(|i| Ok(Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData2, SynKey>::new("".into(), batch_0_tensors[i as usize].shallow_clone())?, ca.clone())))).collect::<Result<Vec<_>, ThoError>>()?;

        let resources_1 = (batch_0_len..batch_0_len +batch_1_len).map(|i| Arc::new(Resource::new(i, 1, format!("aaa_{}", i), 100))).collect::<Vec<_>>();
        let batch_1_tensors = (batch_0_len..batch_0_len +batch_1_len).map(|i| {
            let i = i as i64;
            Tensor::arange_start(i*1000, (i+1)*1000, FLOAT_CPU).view((1, -1))
        }).collect::<Vec<_>>();

        let batch_1 = (batch_0_len..batch_0_len +batch_1_len).map(|i| Ok(Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData2, SynKey>::new("".into(), batch_1_tensors[(i -batch_0_len)as usize].shallow_clone())?, ca.clone())))).collect::<Result<Vec<_>, ThoError>>()?;


        let input = CrossInputSelectorWrap::new(
            "".into(),
            batch_0,
            batch_1,
            resources_0.clone(),
            resources_1.clone(),
            ca.clone()
        );
        let mut cross_loader_adjuster = CrossLoaderAdjuster::new();

        let mut selector = CrossInputSelector::default();

        let mut run_ctx = RunCtxWithPermits::new(1, "test");
        let r = input.with_selected_data_ref(1, run_ctx.branch(Some("cross_loader_test_adjuster_full"), false, false), selector.clone(), None, |data, ctx|{
            assert_eq!(data.tensors().len(), 4);

            assert!(batch_0_tensors[0].allclose(&data.tensors()[0].t.to_device(Cpu), 0.02, 0.02, false));
            assert!(batch_0_tensors[1].allclose(&data.tensors()[1].t.to_device(Cpu), 0.02, 0.02, false));
            assert!(batch_1_tensors[0].allclose(&data.tensors()[2].t.to_device(Cpu), 0.02, 0.02, false));
            assert!(batch_1_tensors[1].allclose(&data.tensors()[3].t.to_device(Cpu), 0.02, 0.02, false));

            debug!("data={data:?}");
            Ok(())
        }).await?;


        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_loader_adjuster_degrades_to_slice() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);

        type AdjusterStage = mock_cross_loader_clc::MockCrossLoaderClc<TestBatchOutput, RunCtxWithPermits>;

        let ca = CacheArea::new(1);
        let resources_0 = vec![Arc::new(Resource::new(0, 1, "r0", 4))];
        let resources_1 = vec![Arc::new(Resource::new(1, 1, "r1", 8))];

        let batch_0_tensors = vec![Tensor::arange_start(0, 4, FLOAT_CPU).view((1, -1))];
        let batch_1_tensors = vec![Tensor::arange_start(0, 8, FLOAT_CPU).view((1, -1))];

        let batch_0 = vec![
            Arc::new(DevicesWrap::wrap(
                ThoTensor::<ChannelsData2, SynKey>::new("".into(), batch_0_tensors[0].shallow_clone())?,
                ca.clone(),
            )),
        ];
        let batch_1 = vec![
            Arc::new(DevicesWrap::wrap(
                ThoTensor::<ChannelsData2, SynKey>::new("".into(), batch_1_tensors[0].shallow_clone())?,
                ca.clone(),
            )),
        ];

        let input = CrossInputSelectorWrap::new(
            "".into(),
            batch_0,
            batch_1,
            resources_0.clone(),
            resources_1.clone(),
            ca.clone(),
        );

        let mut context = CrossSplitterContext::<AdjusterStage, TestBatchOutput>::new(1, vec![batch_1_tensors[0].size()[1]]);
        let cross_loader_adjuster = CrossLoaderAdjuster::new();
        let mut selector = CrossInputSelector::default();

        assert!(cross_loader_adjuster.degrade(&mut context));
        assert_eq!(context.batch_1_mode, Batch1SplitMode::Slice);

        selector = cross_loader_adjuster
            .adjust(selector, &mut context)
            .await
            .map_err(|e| ThoError::new(format!("adjust failed: {e:?}")))?;
        assert!(matches!(&selector.batch_1_selector, RangeOrSlice::Slice((0, range)) if range.start == 0 && range.end == 4));

        let mut run_ctx = RunCtxWithPermits::new(1, "test_adjuster_slice");
        input
            .with_selected_data_ref(1, run_ctx.branch(Some("cross_loader_test_adjuster_degrades_0"), false, false), selector.clone(), None, |data, _ctx| {
                assert_eq!(data.tensors()[1].t.size()[1], 4);
                Ok(())
            })
            .await?;

        let dummy_output = CrossUnfoldClcOutput::new(TestBatchOutput::new(0), SlicesIter::new(1, vec![]), SlicesIter::new(1, vec![]), mono_key!("CrossUnfoldClcOutput", "stage").into());
        context.add_partial_output(dummy_output).await;

        selector = cross_loader_adjuster
            .adjust(selector, &mut context)
            .await
            .map_err(|e| ThoError::new(format!("adjust failed: {e:?}")))?;
        assert!(matches!(&selector.batch_1_selector, RangeOrSlice::Slice((0, range)) if range.start == 4 && range.end == 8));

        input
            .with_selected_data_ref(1, run_ctx.branch(Some("cross_loader_test_adjuster_degrades_1"), false, false), selector.clone(), None, |data, _ctx| {
                assert_eq!(data.tensors()[1].t.size()[1], 4);
                Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_loader_adjuster_slice_advances_resources() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);

        type AdjusterStage = mock_cross_loader_clc::MockCrossLoaderClc<TestBatchOutput, RunCtxWithPermits>;

        let ca = CacheArea::new(1);
        let resources_0 = vec![Arc::new(Resource::new(0, 1, "r0", 4))];
        let resources_1 = vec![
            Arc::new(Resource::new(1, 1, "r1", 6)),
            Arc::new(Resource::new(2, 1, "r2", 4)),
        ];

        let batch_0_tensors = vec![Tensor::arange_start(0, 4, FLOAT_CPU).view((1, -1))];
        let batch_1_tensors = vec![
            Tensor::arange_start(0, 6, FLOAT_CPU).view((1, -1)),
            Tensor::arange_start(0, 4, FLOAT_CPU).view((1, -1)),
        ];

        let batch_0 = vec![
            Arc::new(DevicesWrap::wrap(
                ThoTensor::<ChannelsData2, SynKey>::new("".into(), batch_0_tensors[0].shallow_clone())?,
                ca.clone(),
            )),
        ];
        let batch_1 = vec![
            Arc::new(DevicesWrap::wrap(
                ThoTensor::<ChannelsData2, SynKey>::new("".into(), batch_1_tensors[0].shallow_clone())?,
                ca.clone(),
            )),
            Arc::new(DevicesWrap::wrap(
                ThoTensor::<ChannelsData2, SynKey>::new("".into(), batch_1_tensors[1].shallow_clone())?,
                ca.clone(),
            )),
        ];

        let input = CrossInputSelectorWrap::new(
            "".into(),
            batch_0,
            batch_1,
            resources_0.clone(),
            resources_1.clone(),
            ca.clone(),
        );

        let batch_1_lengths = batch_1_tensors.iter().map(|t| t.size()[1]).collect::<Vec<_>>();
        let mut context = CrossSplitterContext::<AdjusterStage, TestBatchOutput>::new(2, batch_1_lengths);
        let cross_loader_adjuster = CrossLoaderAdjuster::new();
        let mut selector = CrossInputSelector::default();

        assert!(cross_loader_adjuster.degrade(&mut context));
        assert!(cross_loader_adjuster.degrade(&mut context));

        selector = cross_loader_adjuster
            .adjust(selector, &mut context)
            .await
            .map_err(|e| ThoError::new(format!("adjust failed: {e:?}")))?;
        assert!(matches!(&selector.batch_1_selector, RangeOrSlice::Slice((0, range)) if range.start == 0 && range.end == 3));

        let mut run_ctx = RunCtxWithPermits::new(1, "test_adjuster_slice_resources");
        input
            .with_selected_data_ref(1, run_ctx.branch(Some("cross_loader_test_adjuster_resources_0"), false, false), selector.clone(), None, |data, _ctx| {
                assert_eq!(data.batch_1()[0].id(), resources_1[0].id());
                assert_eq!(data.tensors()[1].t.size()[1], 3);
                Ok(())
            })
            .await?;

        let dummy_output_0 = CrossUnfoldClcOutput::new(TestBatchOutput::new(0), SlicesIter::new(1, vec![]), SlicesIter::new(1, vec![]), mono_key!("CrossUnfoldClcOutput", "stage").into());
        context.add_partial_output(dummy_output_0).await;

        selector = cross_loader_adjuster
            .adjust(selector, &mut context)
            .await
            .map_err(|e| ThoError::new(format!("adjust failed: {e:?}")))?;
        assert!(matches!(&selector.batch_1_selector, RangeOrSlice::Slice((0, range)) if range.start == 3 && range.end == 6));

        input
            .with_selected_data_ref(1, run_ctx.branch(Some("cross_loader_test_adjuster_resources_1"), false, false), selector.clone(), None, |data, _ctx| {
                assert_eq!(data.batch_1()[0].id(), resources_1[0].id());
                assert_eq!(data.tensors()[1].t.size()[1], 3);
                Ok(())
            })
            .await?;

        let dummy_output_1 = CrossUnfoldClcOutput::new(TestBatchOutput::new(1), SlicesIter::new(1, vec![]), SlicesIter::new(1, vec![]), mono_key!("CrossUnfoldClcOutput", "stage").into());
        context.add_partial_output(dummy_output_1).await;

        selector = cross_loader_adjuster
            .adjust(selector, &mut context)
            .await
            .map_err(|e| ThoError::new(format!("adjust failed: {e:?}")))?;
        assert!(matches!(&selector.batch_1_selector, RangeOrSlice::Slice((1, range)) if range.start == 0 && range.end == 3));

        input
            .with_selected_data_ref(1, run_ctx.branch(Some("cross_loader_test_adjuster_resources_2"), false, false), selector, None, |data, _ctx| {
                assert_eq!(data.batch_1()[0].id(), resources_1[1].id());
                assert_eq!(data.tensors()[1].t.size()[1], 3);
                Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_clc_with_indexes_mask_without_oom() -> Result<(), ThoError> {
        test_cross_clc(None, 20, |(mask_size, _dist_size)|DistsMaskType::MaskIndexes(mask_size)).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_clc_with_indexes_mask_with_oom() -> Result<(), ThoError> {
        test_cross_clc(Some(20), 40, |(mask_size, _dist_size)|DistsMaskType::MaskIndexes(mask_size)).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_clc_with_packed_mask_without_oom() -> Result<(), ThoError> {
        test_cross_clc(None, 20, |(mask_size, dist_size)|DistsMaskType::MaskPacked(mask_size, dist_size)).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_clc_with_packed_mask_with_oom() -> Result<(), ThoError> {
        test_cross_clc(Some(20), 40, |(mask_size, dist_size)|DistsMaskType::MaskPacked(mask_size, dist_size)).await
    }

    async fn test_cross_clc<F>(oom_limit: Option<i64>, chunk_size: usize, mask_type:F) -> Result<(), ThoError>
    where
        F:Fn((Vec<i64>, Option<Vec<i64>>))->DistsMaskType + Send + Sync + 'static
    {
        GlobalCtx::init_devices(2);

        let tensor_len = 10usize;
        let unfold_size = 2i64;
        let unfold_step = 1i64;

        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 1, 1000, 32000)?));
        let store_path = thoosa_shared::with_perf_log(
            || ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout").store_path().to_path_buf(),
            "CrossLoader::test_cross_clc/ws_read_store_path",
        );
        let store = Arc::new(SqlStore::from_path(&store_path).await?);

        let resource_keys: Vec<String> = vec!["111", "222", "333", "444", "555", "666"]
            .into_iter()
            .map(|v| v.to_string())
            .collect();


        let mut ctx = RunCtxWithPermits::new(1, "test_cross_clc");

        let resource_stats = resource_keys
            .iter()
            .map(|key| (key.clone(), (ResourceStats::new(vec![1, tensor_len as u32]), vec![])))
            .collect::<HashMap<_, _>>();

        let import_id = store.save_import("test resources".to_string()).await?;
        store.save_resources_labels(&resource_stats, import_id).await?;

        let run_id = store.save_run_config(
            None,
            &"find_motifs",
            "test_cross_clc",
            "",
            HashMap::new(),
        ).await?;
        let run_config = FindMotifsRunConfig::new_with_running_mean(Some(run_id), 1, 2, None, None);
        let fixed_config = FixedFindMotifsConfig::new(2, false);

        let resource_key_refs = resource_keys.iter().map(|v| v.as_str()).collect::<Vec<_>>();
        let resources = store.list_resources_by_file_stems(&resource_key_refs).await?;
        let resources_to_move = resources.clone();
        let oom_triggered = Arc::new(AtomicBool::new(false));
        let oom_triggered_cloned = Arc::clone(&oom_triggered);

        let loader = Arc::new(RawTensorLoader::<RunCtxWithPermits>::new(
            move |key, _| {
                let i = resources_to_move
                    .iter()
                    .position(|v| v.eq(key))
                    .expect(format!("no {key:?}").as_str())
                    + 1;
                let v = (0..tensor_len)
                    .map(|v| i as f32 + v as f32 / 10.0)
                    .collect::<Vec<f32>>();
                debug!("executor: key={key:?} i={i} v={v:?}");

                Ok(ThoTensor::new(
                    SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]),
                    Tensor::from_slice(v.as_slice()).view((1, tensor_len as i64)),
                )?)
            },
            ThoDevice::from(Device::Cpu),
        ));


        let batch_clc = Arc::new(CrossUnfoldClc::new(
            move |t_0, t_1, _input, _ctx, _| {

                if let Some(limit) = oom_limit {
                    let batch_1_len = t_1.size()[1];
                    if batch_1_len > limit {
                        debug!("test_cross_clc, batch_clc, triggering OOM: batch_1_len={batch_1_len} limit={limit}");
                        oom_triggered_cloned.store(true, Ordering::SeqCst);
                        return Err(ThoError::oom());
                    }
                }
                debug!("test_cross_cls, batch_clc, no OOM: t_0={:?}, t_1={:?}", t_0.size(), t_1.size());
                let sum_0 = t_0.f_sum_dim_intlist(2, false, Kind::Float)?;
                let sum_1 = t_1.f_sum_dim_intlist(2, false, Kind::Float)?;
                let dists = sum_0.unsqueeze(2) + sum_1.unsqueeze(1);
                let mask = (sum_0.unsqueeze(2) - sum_1.unsqueeze(1)).lt(0.0);
                let mask_size = mask.size();
                let dists_size = Some(dists.size());
                Ok(DistsMask::new(mask, Some(dists), mask_type((mask_size, dists_size)), None)?)
            },
            (),
            UnfoldParams::new(1, unfold_size, unfold_step),
        ));

        let cross_loader = Arc::new(CrossLoader::new(
            1,
            loader,
            &store,
            batch_clc,
            {
                let ws = Arc::clone(&ws);
                let store = Arc::clone(&store);
                let run_config = run_config.clone();
                let fixed_config = fixed_config.clone();
                move |_resources: &[Arc<Resource>]| {
                    // Integration-style test to exercise CrossLoader + DistsTracker end-to-end.
                    Arc::new(DistsTracker::<RunCtxWithPermits>::new(
                        &ws,
                        &store,
                        &run_config,
                        &fixed_config,
                        None,
                    ))
                }
            },
            chunk_size, //TODO config
            |batch| async { batch.to_cpu().await }.boxed(),
            |_receiver| async {
                Ok(())
            }.boxed(),
            CrossUnfoldClcOutputMergeMode::GpuMerge,
            0,
        ));

        let cache_area = ctx.cache_area();
        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(Simple::new(ResourceType::List(resource_keys))),
            cache_area.into(),
        ));

        let output:Arc<DevicesWrap<()>> = cross_loader
            .run(input, ctx.branch(Some("cross_loader_run"), false, false))
            .await?;

        assert_eq!(output.id(), &());
        if oom_limit.is_some() {
            assert!(oom_triggered.load(Ordering::SeqCst));
        }

        let slices_per_resource = (tensor_len as i64 - unfold_size + 1) as u32;
        let mut expected = HashMap::new();

        for (i_0, resource_0) in resources.iter().enumerate() {
            let base_0 = (i_0 + 1) as f32;
            for (i_1, resource_1) in resources.iter().enumerate().skip(i_0 + 1) {
                let base_1 = (i_1 + 1) as f32;
                for slice_0 in 0..slices_per_resource {
                    for slice_1 in 0..slices_per_resource {
                        let sum_0 = slice_sum(base_0, slice_0);
                        let sum_1 = slice_sum(base_1, slice_1);
                        if sum_0 < sum_1 {
                            expected.insert(
                                (resource_0.id(), slice_0, resource_1.id(), slice_1),
                                sum_0 + sum_1,
                            );
                        }
                    }
                }
            }
        }

        let expected_len = expected.len();
        assert!(expected_len > 0);

        let slices_path = thoosa_shared::with_perf_log(
            || ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout").tmp_path().join(format!("{run_id}_0_slices.zstd")),
            "CrossLoader::test_cross_clc/ws_read_tmp_path",
        );
        let reader = ChunkedSlicesReader::<SlicesMatch>::new(&slices_path)
            .map_err(|e| ThoError::from_error(format!("read slices file {slices_path:?}"), e))?;

        let mut remaining = expected;
        let mut actual_count = 0usize;
        for chunk in reader {
            let chunk = chunk.map_err(|e| ThoError::from_error("read slices chunk", e))?;
            for entry in chunk {
                debug!("entry={entry:?}");
                actual_count += 1;
                let key = (entry.0.0, entry.0.1, entry.1.0, entry.1.1);
                let expected_dist = remaining
                    .remove(&key)
                    .unwrap_or_else(|| panic!("unexpected slice match {key:?}"));
                let delta = (entry.2 - expected_dist).abs();
                assert!(
                    delta < 1e-4,
                    "dist mismatch for {key:?}: got {}, expected {}",
                    entry.2,
                    expected_dist
                );
            }
        }

        assert_eq!(actual_count, expected_len);
        assert!(remaining.is_empty());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cross_loader() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);

        let resource_a: Resource = (1, 1, "resource_a", 2).into();
        let resource_b: Resource = (2, 1, "resource_b", 2).into();
        let resources = vec![resource_a.clone(), resource_b.clone()];



        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_run_cached([
                    (resource_a.clone(), Tensor::from_slice(&[1.0, 2.0]).view((1,2))),
                    (resource_b.clone(), Tensor::from_slice(&[3.0, 4.0]).view((1,2))),
                ])
                .finish(),
        );

        let observations = Arc::new(Mutex::new(Vec::<Observation<SynKey>>::new()));
        let produced_outputs = Arc::new(Mutex::new(Vec::<TestBatchOutput>::new()));
        let counter = Arc::new(AtomicUsize::new(0));

        let batch_observations = Arc::clone(&observations);
        let batch_outputs = Arc::clone(&produced_outputs);
        let batch_counter = Arc::clone(&counter);

        let batch_clc = Arc::new(
            mock_cross_loader_clc::build::<TestBatchOutput, MockRunCtx>()
                .with_run_batch_fn(move |input, _ctx| {
                    let observation = Observation {
                        tensor_ids: input.tensors().iter().map(|t| t.id().clone()).collect::<Vec<_>>(),
                        batch_0_ids: input.batch_0().iter().map(|r| r.as_ref().id()).collect(),
                        batch_1_ids: input.batch_1().iter().map(|r| r.as_ref().id()).collect(),
                    };

                    batch_observations.lock().unwrap().push(observation);

                    let call_index = batch_counter.fetch_add(1, Ordering::SeqCst);
                    let output = TestBatchOutput::new(call_index);
                    batch_outputs.lock().unwrap().push(output.clone());

                    Ok(CrossUnfoldClcOutput::new(
                        output,
                        SlicesIter::new(1, vec![]),
                        SlicesIter::new(1, vec![]), mono_key!("CrossUnfoldClcOutput", "stage").into(),
                    ))
                })
                .finish(),
        );

        let received_outputs = Arc::new(Mutex::new(Vec::<TestBatchOutput>::new()));
        let receiver_observed = Arc::clone(&received_outputs);

        let mut ctx = mock_run_ctx::default(0);

        let mut mocked_store = Arc::new(
            MockStoreBuilder::new()
                .with_list_resources_by_file_stems(
                    vec!["resource_a".to_string(), "resource_b".to_string()],
                    Ok(resources.clone()),
                )
                .build() as Box<dyn Store>,
        );

        let cross_loader = Arc::new(CrossLoader::new(
            1,
            loader,
            &mocked_store,
            batch_clc,
            {
                let receiver_observed = Arc::clone(&receiver_observed);
                move |_resources: &[Arc<Resource>]| {
                    Arc::new(
                        mock_cross_loader_receiver::build::<CrossUnfoldClcOutput<TestBatchOutput>, MockRunCtx>()
                            .with_receive_fn({
                                let receiver_observed = Arc::clone(&receiver_observed);
                                move |out, _ctx| {
                                    let result = out.clc_result();
                                    receiver_observed.lock().unwrap().push(result);
                                    Ok(())
                                }
                            })
                            .finish(),
                    )
                }
            },
            2,
            |batch|{
                async {Ok(batch)}.boxed()
            },
            |p| {
                async {
                    // cross_loader.receiver;
                    Ok(())
                }
                .boxed()
            },
            CrossUnfoldClcOutputMergeMode::GpuMerge,
            0,
        ));

        let cache_area = ctx.cache_area();
        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(Simple::new(ResourceType::List(vec![
                "resource_a".to_string(),
                "resource_b".to_string(),
            ]))),
            cache_area.into(),
        ));

        let _ = cross_loader
            .run(input, ctx.branch(Some("cross_loader_run"), false, false))
            .await?;

        yield_now().await;
        sleep(Duration::from_millis(10)).await;

        let observed = observations.lock().unwrap().clone();

        let expected = vec![
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_a", "resource_a")]),
                    SynKey::new([mono_key!("resource_a", "resource_a")]),
                ],
                batch_0_ids: vec![resources[0].id()],
                batch_1_ids: vec![resources[0].id()],
            },
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_a", "resource_a")]),
                    SynKey::new([mono_key!("resource_b", "resource_b")]),
                ],
                batch_0_ids: vec![resources[0].id()],
                batch_1_ids: vec![resources[1].id()],
            },
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_b", "resource_b")]),
                    SynKey::new([mono_key!("resource_a", "resource_a")]),
                ],
                batch_0_ids: vec![resources[1].id()],
                batch_1_ids: vec![resources[0].id()],
            },
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_b", "resource_b")]),
                    SynKey::new([mono_key!("resource_b", "resource_b")]),
                ],
                batch_0_ids: vec![resources[1].id()],
                batch_1_ids: vec![resources[1].id()],
            },
        ];

        assert_eq!(observed, expected);

        let produced = produced_outputs
            .lock()
            .expect("produced_outputs mutex poisoned")
            .clone();
        let received = received_outputs
            .lock()
            .expect("received_outputs mutex poisoned")
            .clone();

        let expected_outputs: Vec<TestBatchOutput> = (0..expected.len())
            .map(TestBatchOutput::new)
            .collect();

        assert_eq!(produced, expected_outputs);
        assert_eq!(produced, received);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cross_loader_splits_large_tensor() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);

        let tensor_len = 8usize;
        let chunk_size = 3usize;
        let max_len = (chunk_size as i64) * 2;
        let slice_len = (tensor_len as i64) / 2;
        let second_slice_len = (tensor_len as i64) - slice_len;

        let resource_a: Resource = (1, 1, "resource_a", tensor_len as u32).into();
        let resources = vec![resource_a.clone()];

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_run_cached([(
                    resource_a.clone(),
                    Tensor::arange(tensor_len as i64, (Kind::Float, Device::Cpu)).view((1, tensor_len as i64)),
                )])
                .finish(),
        );

        let observations = Arc::new(Mutex::new(Vec::<SplitObservation>::new()));
        let observed_batches = Arc::clone(&observations);
        let oom_triggered = Arc::new(AtomicBool::new(false));
        let oom_triggered_cloned = Arc::clone(&oom_triggered);

        let batch_clc = Arc::new(
            mock_cross_loader_clc::build::<TestBatchOutput, MockRunCtx>()
                .with_run_batch_fn(move |input, _ctx| {
                    let tensor_lens = input
                        .tensors()
                        .iter()
                        .map(|t| t.t_ref().size()[1])
                        .collect::<Vec<_>>();
                    let batch_1_len = *tensor_lens.last().unwrap_or(&0);
                    if batch_1_len > max_len {
                        oom_triggered_cloned.store(true, Ordering::SeqCst);
                        return Err(ThoError::oom());
                    }

                    let observation = SplitObservation {
                        batch_0_ids: input.batch_0().iter().map(|r| r.as_ref().id()).collect(),
                        batch_1_ids: input.batch_1().iter().map(|r| r.as_ref().id()).collect(),
                        tensor_lens,
                    };

                    observed_batches.lock().unwrap().push(observation);

                    Ok(CrossUnfoldClcOutput::new(
                        TestBatchOutput::new(0),
                        SlicesIter::new(1, vec![]),
                        SlicesIter::new(1, vec![]), mono_key!("CrossUnfoldClcOutput", "stage").into(),
                    ))
                })
                .finish(),
        );

        let receiver = Arc::new(
            mock_cross_loader_receiver::build::<CrossUnfoldClcOutput<TestBatchOutput>, MockRunCtx>()
                .with_receive_fn(|_out, _ctx| Ok(()))
                .finish(),
        );

        let mut ctx = mock_run_ctx::default(1);

        let mut mocked_store = Arc::new(
            MockStoreBuilder::new()
                .with_list_resources_by_file_stems(
                    vec!["resource_a".to_string()],
                    Ok(resources.clone()),
                )
                .build() as Box<dyn Store>,
        );

        let cross_loader = Arc::new(CrossLoader::new(
            1,
            loader,
            &mocked_store,
            batch_clc,
            move |_resources: &[Arc<Resource>]| Arc::clone(&receiver),
            chunk_size,
            |batch| async { Ok(batch) }.boxed(),
            |_receiver| async {
                Ok(())
            }.boxed(),
            CrossUnfoldClcOutputMergeMode::GpuMerge,
            1,
        ));

        let cache_area = ctx.cache_area();
        let input = Arc::new(DevicesWrap::wrap(
            ResourcesConfigInput::new(Simple::new(ResourceType::List(vec![
                "resource_a".to_string(),
            ]))),
            cache_area.into(),
        ));

        let _ = cross_loader.run(input, ctx.branch(Some("cross_loader_run"), false, false)).await?;

        let observed = observations.lock().expect("observations.lock").clone();
        assert!(oom_triggered.load(Ordering::SeqCst));
        assert_eq!(observed.len(), 2);
        assert_eq!(
            observed,
            vec![
                SplitObservation {
                    batch_0_ids: vec![resource_a.id()],
                    batch_1_ids: vec![resource_a.id()],
                    tensor_lens: vec![tensor_len as i64, slice_len],
                },
                SplitObservation {
                    batch_0_ids: vec![resource_a.id()],
                    batch_1_ids: vec![resource_a.id()],
                    tensor_lens: vec![tensor_len as i64, second_slice_len],
                },
            ]
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_output_builder_cpu_merge_calls_to_cpu() -> Result<(), ThoError> {
        let to_cpu_calls = Arc::new(AtomicUsize::new(0));
        let output_0 = CrossUnfoldClcOutput::new(
            CpuMergeOutput::new(Arc::clone(&to_cpu_calls), 0),
            SlicesIter::new(1, vec![]),
            SlicesIter::new(1, vec![]), mono_key!("CrossUnfoldClcOutput", "stage").into(),
        );
        let output_1 = CrossUnfoldClcOutput::new(
            CpuMergeOutput::new(Arc::clone(&to_cpu_calls), 1),
            SlicesIter::new(1, vec![]),
            SlicesIter::new(1, vec![]), mono_key!("CrossUnfoldClcOutput", "stage").into(),
        );

        let mut builder = CrossUnfoldClcOutputBuilder::new(CrossUnfoldClcOutputMergeMode::CpuMerge);
        builder.add_partial_output(output_0).await;
        builder.add_partial_output(output_1).await;

        let _ = builder.build().await?;

        assert_eq!(to_cpu_calls.load(Ordering::SeqCst), 2);

        Ok(())
    }

    #[tokio::test]
    async fn test_output_builder_gpu_merge_skips_to_cpu() -> Result<(), ThoError> {
        let to_cpu_calls = Arc::new(AtomicUsize::new(0));
        let output_0 = CrossUnfoldClcOutput::new(
            CpuMergeOutput::new(Arc::clone(&to_cpu_calls), 0),
            SlicesIter::new(1, vec![]),
            SlicesIter::new(1, vec![]), mono_key!("CrossUnfoldClcOutput", "stage").into(),
        );
        let output_1 = CrossUnfoldClcOutput::new(
            CpuMergeOutput::new(Arc::clone(&to_cpu_calls), 1),
            SlicesIter::new(1, vec![]),
            SlicesIter::new(1, vec![]), mono_key!("CrossUnfoldClcOutput", "stage").into(),
        );

        let mut builder = CrossUnfoldClcOutputBuilder::new(CrossUnfoldClcOutputMergeMode::GpuMerge);
        builder.add_partial_output(output_0).await;
        builder.add_partial_output(output_1).await;

        let _ = builder.build().await?;

        assert_eq!(to_cpu_calls.load(Ordering::SeqCst), 0);

        Ok(())
    }

}
