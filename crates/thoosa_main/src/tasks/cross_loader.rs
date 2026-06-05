use crate::tasks::cross_batch_clc::{CrossClcResultMerge, CrossInput, CrossInputBatch1Selector, CrossUnfoldClcOutput};
#[cfg(test)]
use crate::tasks::cross_batch_clc::{CrossBatchFn, CrossUnfoldClc};
use crate::tasks::cross_dims::ChannelsData2Selector;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use crate::tasks::tensor_loader::RawTensorLoader;
use async_trait::async_trait;
use std::collections::{HashMap, HashSet};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, DevicesWrapCopyMode, WithDataRef, WithSelectDataRef};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::runner::common::{OutId, RunCtx};
use futures_util::future::BoxFuture;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, info, warn};
use parking_lot::RwLock;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{AddAssign, Range};
use std::sync::Arc;
use std::vec;
use futures_util::FutureExt;
use mockall_double::double;
use tch::{IndexOp, Tensor};
use tch::Device::Cpu;
use thoosa_common::dims::{AnyDims, AnyMetrics, ChannelsData, Dims};
use thoosa_common::runs::Simple;
use thoosa_motifs::find::FindMotifsSavableResult;
use thoosa_shared::{simple_id, CopyToDevice, ThoError};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContext, SplitsAdjusterContextImpl};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_shared::resource::{Resource, ResourceStore, ResourcesConfig};
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, InputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Named, WithId};
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_shared::mono_key;
use thoosa_store::Store;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::torch_runner::output_builders::SingleDimTensorOutputBuilder;
use thoosa_shared::traits::MonoMetrics;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_with_data::with_data;
use crate::tasks::dists_tracker::DistsMask;
use crate::tasks::slices_iter::SlicesIter;
use tracing::Instrument;

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
                match clc_result.to_cpu().instrument(tracing::debug_span!("add_partial_output/1", file = file!(), line = line!())).await {
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
        Out = ThoTensor<ChannelsData, SynKey>,
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
    pub(crate) find_motifs_run_config_id: RunConfigId,
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
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx=C> + OutId + 'static,
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
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx=C> + OutId + 'static,
    R: CrossClcResultMerge + Debug + Send + Sync + 'static,
    CLC: for<'i> SyncStage<In<'i> = CrossInput<'i, Arc<Resource>>, Out = CrossUnfoldClcOutput<R>, Ctx=C, > + Adjustable + OutId + Send + Sync + 'static,
    RSV: for<'i> SyncStage<In<'i> = CrossUnfoldClcOutput<R>, Ctx=C> + Stage<Ctx=C, Out = ()> + Send + Sync + 'static,
{
    pub fn new<ReceiverFactory, OnAllDone, OnBatchDone>(
        find_motifs_run_config_id: RunConfigId, //fod logging
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

    async fn run_loaded_batch(
        &self,
        t_0_batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
        t_0_batch_resources: Vec<Arc<Resource>>,
        t_1_batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
        t_1_batch_resources: Vec<Arc<Resource>>,
        mut ctx: C,
        receiver: Arc<RSV>,
    ) -> Result<(), ThoError> {
        debug!("CrossLoader::run_loaded_batch: find_motifs_run_config_id={} t_0_batch={t_0_batch:?} t_1_batch={t_1_batch:?}", self.find_motifs_run_config_id);
        let cross_unfold_clc = Arc::clone(&self.cross_unfold_clc);
        let batch_1_slice_lengths = t_1_batch
            .iter()
            .map(|t| t.size().dim_1())
            .collect::<Vec<_>>();

        let input = CrossInputSelectorWrap::new(
            mono_key!(self.find_motifs_run_config_id, "find_motifs_run_config_id").into(),
            t_0_batch,
            t_1_batch,
            t_0_batch_resources,
            t_1_batch_resources.clone(),
            CacheArea::default()
        );
        let required_ids: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
        match OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(
                    MAX_PARALLEL,
                    *MaxParallelStep::FIFTY,
                ))),
                Adjusters::RunContext(Box::new(DumpToCpu::new("cross_unfold_clc"))),
                Adjusters::Input(Box::new(CrossLoaderAdjuster::new())),
            ],
            None,
        ).run(
            &cross_unfold_clc,
            Arc::new(input),
            ctx.branch(false),
            Contexts::new(
                CrossSplitterContext::new(t_1_batch_resources.len() as i64, batch_1_slice_lengths),
                NoopRunnableContext {},
                MaxParallelAndDumpToCpuAdjustableContex::new(
                    MaxParallelAdjustContextImpl::new(),
                    DumpToCpuContextImpl::new(&required_ids),
                ),
            ),
        ).instrument(tracing::debug_span!("run_loaded_batch/1", file = file!(), line = line!())).await {
            Ok(mut cross_clc_output) => {
                let cross_clc_output = cross_clc_output.replace_clc_result(&self.on_batch_done).instrument(tracing::debug_span!("run_loaded_batch/2", file = file!(), line = line!())).await?;
                let receiver_ctx = ctx.branch(false);

                tokio::task::spawn_blocking(move || {
                    tracing::info_span!(
                        "cross_loader::consume_cross_output",
                        kind = "spawn_blocking",
                        file = file!(),
                        line = line!(),
                        task = "consume cross calculation output",
                    ).in_scope(|| {
                        receiver.consume(cross_clc_output, receiver_ctx);
                        Ok::<_, ThoError>(())
                    })
                }).instrument(tracing::debug_span!("run_loaded_batch/3", file = file!(), line = line!())).await??;
            }
            Err(e) => {
                warn!("CrossLoader::run: got error  find_motifs_run_config_id={} {e:?}", self.find_motifs_run_config_id);
            }
        };

        Ok(())
    }

    async fn run_inner_batches(
        &self,
        resources: &[Arc<Resource>],
        t_0_batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
        t_0_batch_resources: Vec<Arc<Resource>>,
        outer_batch_last_i: usize,
        skip_0_i: usize,
        max_batch: i64,
        receiver: Arc<RSV>,
        mut ctx: C,
    ) -> Result<(), ThoError> {
        let mut t_1_batch: Vec<_> = vec![];
        let mut t_1_batch_resources: Vec<_> = vec![];
        let mut t_1_batch_len = 0;
        let cross_loader_inner = CollectEtaInvocationGuard::new(&format!("CrossLoader::run, inner: run_config_id={} i_0={outer_batch_last_i}", self.find_motifs_run_config_id), resources.len(), 1);

        for (i_1, resource_1) in resources.iter().enumerate().skip(skip_0_i) {
            let t_1 = Arc::clone(&self.configured_loader)
                .run_cached(Arc::new(DevicesWrap::wrap(resource_1.as_ref().clone())), ctx.branch(false))
                .instrument(tracing::debug_span!("run_inner_batches/1", file = file!(), line = line!())).await?;
            let t_1_len = t_1.size().dim_1();

            if t_1_len > max_batch {
                if !t_1_batch.is_empty() {
                    self.run_loaded_batch(
                        t_0_batch.clone(),
                        t_0_batch_resources.clone(),
                        std::mem::take(&mut t_1_batch),
                        std::mem::take(&mut t_1_batch_resources),
                        ctx.branch(false),
                        Arc::clone(&receiver),
                    ).instrument(tracing::debug_span!("run_inner_batches/2", file = file!(), line = line!())).await?;
                    t_1_batch_len = 0;
                    collect_eta_record_completion(cross_loader_inner.id());
                }

                self.run_loaded_batch(
                    t_0_batch.clone(),
                    t_0_batch_resources.clone(),
                    vec![t_1],
                    vec![Arc::clone(resource_1)],
                    ctx.branch(false),
                    Arc::clone(&receiver),
                ).instrument(tracing::debug_span!("run_inner_batches/3", file = file!(), line = line!())).await?;
                collect_eta_record_completion(cross_loader_inner.id());
                continue;
            }

            t_1_batch_len += t_1_len;
            t_1_batch.push(t_1);
            t_1_batch_resources.push(Arc::clone(resource_1));

            if t_1_batch_len < max_batch && i_1 < resources.len() - 1 {
                continue;
            }

            self.run_loaded_batch(
                t_0_batch.clone(),
                t_0_batch_resources.clone(),
                std::mem::take(&mut t_1_batch),
                std::mem::take(&mut t_1_batch_resources),
                ctx.branch(false),
                Arc::clone(&receiver),
            ).instrument(tracing::debug_span!("run_inner_batches/4", file = file!(), line = line!())).await?;
            t_1_batch_len = 0;
            collect_eta_record_completion(cross_loader_inner.id());
        }

        Ok(())
    }
}

//TODO move to lib
#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct ResourcesConfigInput<RC: ResourcesConfig + Debug + Send + Sync + Clone> {
    resources_config: RC,
}

impl<RC: ResourcesConfig + Debug + Send + Sync + Clone> ResourcesConfigInput<RC> {
    pub async fn check(self, store: &Arc<Box<dyn Store>>) -> Result<Self, ThoError> {
        self.resources_config.check(store).instrument(tracing::debug_span!("check/1", file = file!(), line = line!())).await?;

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
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx=C> + OutId + 'static,
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
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData, SynKey>, Ctx=C> + OutId + 'static,
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

        let resources_config = input
            .with_data_ref(0, ctx.branch(false), None, |input: &ResourcesConfigInput<ResC>, _ctx| {
                Ok(input.resources_config.clone())
            })
            .instrument(tracing::debug_span!("run/1", file = file!(), line = line!())).await?;
        let resources = resources_config
            .to_resources(&self.store)
            .instrument(tracing::debug_span!("run/2", file = file!(), line = line!())).await?
            .into_iter()
            .filter(|r| r.length() > self.step)
            .map(Arc::new)
            .collect::<Vec<_>>();

        debug!("CrossLoader::run:  max_batch={max_batch} resources={resources:?}");

        let receiver = (self.receiver_factory)(resources.as_slice());

        let mut t_0_batch: Vec<_> = vec![];
        let mut t_0_batch_resources: Vec<_> = vec![];
        let mut t_0_batch_len = 0;
        let mut t_0_batch_last_i: Option<usize> = None;

        let mut skip_0_i = 0;

        let cross_loader_outer = CollectEtaInvocationGuard::new(&format!("CrossLoader::run, outer: run_config_id={}", self.find_motifs_run_config_id), resources.len(), 1);
        for (i_0, resource_0) in resources.iter().enumerate() {
            debug!("CrossLoader::run: find_motifs_run_config_id={} on i_0={i_0} resource_0={resource_0:?}", self.find_motifs_run_config_id);

            let t_0 = Arc::clone(&self.configured_loader)
                .run_cached(Arc::new(DevicesWrap::wrap(resource_0.as_ref().clone())), ctx.branch(false))
                .instrument(tracing::debug_span!("run/3", file = file!(), line = line!())).await?;
            let t_0_len = t_0.size().dim_1();

            if t_0_len > max_batch {
                if !t_0_batch.is_empty() {
                    let pending_last_i = t_0_batch_last_i.expect("CrossLoader::run: missing t_0_batch_last_i for pending batch");
                    self.run_inner_batches(
                        resources.as_slice(),
                        std::mem::take(&mut t_0_batch),
                        std::mem::take(&mut t_0_batch_resources),
                        pending_last_i,
                        skip_0_i,
                        max_batch,
                        Arc::clone(&receiver),
                        ctx.branch(false),
                    ).instrument(tracing::debug_span!("run/4", file = file!(), line = line!())).await?;
                    skip_0_i = pending_last_i;
                    t_0_batch_len = 0;
                    t_0_batch_last_i = None;
                    collect_eta_record_completion(cross_loader_outer.id());
                }

                self.run_inner_batches(
                    resources.as_slice(),
                    vec![t_0],
                    vec![Arc::clone(resource_0)],
                    i_0,
                    skip_0_i,
                    max_batch,
                    Arc::clone(&receiver),
                    ctx.branch(false),
                ).instrument(tracing::debug_span!("run/5", file = file!(), line = line!())).await?;
                skip_0_i = i_0;
                collect_eta_record_completion(cross_loader_outer.id());
                continue;
            }

            t_0_batch_len += t_0_len;
            t_0_batch.push(t_0);
            t_0_batch_resources.push(Arc::clone(resource_0));
            t_0_batch_last_i = Some(i_0);

            if t_0_batch_len < max_batch && i_0 < resources.len() - 1 {
                continue;
            }

            self.run_inner_batches(
                resources.as_slice(),
                std::mem::take(&mut t_0_batch),
                std::mem::take(&mut t_0_batch_resources),
                i_0,
                skip_0_i,
                max_batch,
                Arc::clone(&receiver),
                ctx.branch(false),
            ).instrument(tracing::debug_span!("run/6", file = file!(), line = line!())).await?;

            skip_0_i = i_0;
            t_0_batch_len = 0;
            t_0_batch_last_i = None;

            collect_eta_record_completion(cross_loader_outer.id());
        }

        (self.on_all_done)(&receiver).instrument(tracing::debug_span!("run/7", file = file!(), line = line!())).await?;

        Ok(Arc::new(DevicesWrap::wrap(())))
    }
}

#[derive(Debug)]
struct CrossInputSelectorWrap {
    id: SynKey,
    batch_0:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
    batch_1:Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
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

    fn mem_size(&self) -> u64 {
        self.batch_0.iter().map(|batch| batch.mem_size()).sum::<u64>()
            + self.batch_1.iter().map(|batch| batch.mem_size()).sum::<u64>()
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
        batch_0: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
        batch_1: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
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
                let result = self.batch_1_adjuster.adjust(0i64..0, adjust_context).instrument(tracing::debug_span!("adjust/1", file = file!(), line = line!())).await;
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
                let result = self.batch_1_slice_adjuster.adjust(0i64..0, &mut adjust_context.slice_ctx).instrument(tracing::debug_span!("adjust/2", file = file!(), line = line!())).await;
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
        self.output_builder.add_partial_output(out).instrument(tracing::debug_span!("add_partial_output/1", file = file!(), line = line!())).await;
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
        self.output_builder.build().instrument(tracing::debug_span!("build_final_output/1", file = file!(), line = line!())).await
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
    pub async fn select<R>(&self, batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, mut run_ctx: R) -> Result<Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>, ThoError>
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
                    Arc::new(DevicesWrap::wrap(batch[*i].with_selected_data(run_ctx.tho_device_idx(), run_ctx.branch(false), selector, |data, run_ctx|{
                        Ok(data)
                    }).instrument(tracing::debug_span!("select/1", file = file!(), line = line!())).await?))
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

    fn with_data_ref<'a, Rt: Send + 'a, F, P>(
        &'a self,
        _to_device_idx: usize,
        permits_ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl std::future::Future<Output = Result<Rt, ThoError>> + Send + 'a
    where
            for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<Rt, ThoError> + Send + 'a,
            P: RunCtx + 'a,
    {
        async move {
            let t_0_batch= self.batch_0.clone();
            let t_1_batch= self.batch_1.clone();
            with_data!(permits_ctx, [t_0_batch], [t_1_batch];
                |
                 ctx,
                 batch_0:&[&ThoTensor<ChannelsData, SynKey>],
                 batch_1:&[&ThoTensor<ChannelsData, SynKey>],
                |{

                    do_run(&CrossInput::new(
                        [batch_0,batch_1].concat().as_slice(),
                        self.resources_0.as_slice(),
                        self.resources_1.as_slice(),
                        self.id.clone()
                    ), &ctx)

            })
        }
    }

}


impl WithSelectDataRef<CrossInputSelector> for CrossInputSelectorWrap {
    fn with_selected_data_ref<'a, 'o, Rt: Send + 'a, F, C>(
        &'a self,
        _to_device_idx: usize,
        mut run_ctx: C,
        selector: CrossInputSelector,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl std::future::Future<Output = Result<Rt, ThoError>> + Send + 'a
    where
            for<'f> F: FnOnce(&Self::O<'f>, &C) -> Result<Rt, ThoError> + Send + 'a,
            C: RunCtx + 'a,
    {
        async move {

            debug!("WithSelectDataRef<CrossInputSelector> for CrossInput2::with_data: selector.batch_0_selector={:?} selector.batch_1_selector={:?}", selector.batch_0_selector, selector.batch_1_selector);

            let t_0_batch = selector.batch_0_selector.select(self.batch_0.clone(), run_ctx.branch(false)).instrument(tracing::debug_span!("with_selected_data_ref/1", file = file!(), line = line!())).await?;
            let t_1_batch = selector.batch_1_selector.select(self.batch_1.clone(), run_ctx.branch(false)).instrument(tracing::debug_span!("with_selected_data_ref/2", file = file!(), line = line!())).await?;

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
                 batch_0:&[&ThoTensor<ChannelsData, SynKey>],
                 batch_1:&[&ThoTensor<ChannelsData, SynKey>],
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
}

#[cfg(test)]
pub(crate) mod mock_cross_loader_clc {
    use thoosa_common::runner::common::runnable::{Stage, SyncStage};
    use thoosa_common::runner::common::{OutId, RunCtx};
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
    use mockall::mock;
    use std::fmt::Debug;
    use std::sync::Arc;
    use thoosa_common::runner::common::runnable::{ConsumeResult, Stage, SyncStage};
    use thoosa_common::runner::common::RunCtx;
    use thoosa_shared::applicables_runner::{Input, Output};

    trait CrossLoaderReceiverMockApi<In, C>
    where
        In: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        fn consume_owned(&self, out: In, ctx: C) -> ConsumeResult<In, ()>;
    }

    mock! {
        #[derive(Debug)]
        pub CrossLoaderReceiver<In, C>
        where
            In: Input + Output + Send + Debug + 'static,
            C: RunCtx + 'static,
        {}

        impl<In, C> CrossLoaderReceiverMockApi<In, C> for CrossLoaderReceiver<In, C>
        where
            In: Input + Output + Send + Debug + 'static,
            C: RunCtx + 'static,
        {
            fn consume_owned(&self, out: In, ctx: C) -> ConsumeResult<In, ()>;
        }
    }

    impl<In, C> Stage for MockCrossLoaderReceiver<In, C>
    where
        In: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        type Ctx = C;
        type In<'a> = In where Self: 'a;
        type Out = ();
    }

    impl<In, C> SyncStage for MockCrossLoaderReceiver<In, C>
    where
        In: Input + Output + Send + Debug + 'static,
        C: RunCtx + 'static,
    {
        fn consume<'f>(
            self: Arc<Self>,
            out: <Self as Stage>::In<'f>,
            ctx: <Self as Stage>::Ctx,
        ) -> ConsumeResult<<Self as Stage>::In<'f>, <Self as Stage>::Out> {
            self.consume_owned(out, ctx)
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
            F: Fn(In, C) -> ConsumeResult<In, ()> + Send + Sync + 'static,
        {
            let receive_fn = Arc::new(f);
            self.mock
                .expect_consume_owned()
                .returning(move |out, ctx| Arc::clone(&receive_fn)(out, ctx));
            self.consume_configured = true;
            self
        }

        pub fn finish(self) -> MockCrossLoaderReceiver<In, C> {
            assert!(self.consume_configured, "MockCrossLoaderReceiverBuilder requires receive_fn to be set");
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
    use thoosa_common::runner::common::runnable::ConsumeResult;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use super::*;
    use crate::tasks::cross_batch_clc::{CrossUnfoldClc, CrossUnfoldClcOutput, UnfoldParams};
    use crate::tasks::cross_loader::{
        CrossLoader, CrossUnfoldClcOutputMergeMode, ResourcesConfigInput,
    };
    use crate::tasks::dists_tracker::{DistsMask, DistsMaskType, DistsTracker};
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use crate::tasks::slices_iter::SlicesIter;
    use thoosa_common::runner::common::{mock_run_ctx, MockRunCtx};
    use thoosa_common::runner::cache_areas::CacheArea;
    // use thoosa_common::runner::global_ctx::GlobalCtx;
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
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
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
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_shared::Cause::SomethingElse;
    use thoosa_shared::perf_logging::set_perf_log_level;
    use thoosa_shared::run_configs::RUN_ID;

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
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let ca = CacheArea::new(1);
        let batch_0_len = 2;
        let batch_1_len = 2;

        let resources_0 = (0..batch_0_len).map(|i| Arc::new(Resource::new(i, 1, format!("aaa_{}", i), 100))).collect::<Vec<_>>();
        let batch_0_tensors = (0..batch_0_len).map(|i| {
            let i = i as i64;
            Tensor::arange_start(i*1000, (i+1)*1000, FLOAT_CPU).view((1, -1))
        }).collect::<Vec<_>>();
        let batch_0 = (0..batch_0_len).map(|i| Ok(Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData, SynKey>::new("".into(), batch_0_tensors[i as usize].shallow_clone())?)))).collect::<Result<Vec<_>, ThoError>>()?;

        let resources_1 = (batch_0_len..batch_0_len +batch_1_len).map(|i| Arc::new(Resource::new(i, 1, format!("aaa_{}", i), 100))).collect::<Vec<_>>();
        let batch_1_tensors = (batch_0_len..batch_0_len +batch_1_len).map(|i| {
            let i = i as i64;
            Tensor::arange_start(i*1000, (i+1)*1000, FLOAT_CPU).view((1, -1))
        }).collect::<Vec<_>>();

        let batch_1 = (batch_0_len..batch_0_len +batch_1_len).map(|i| Ok(Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData, SynKey>::new("".into(), batch_1_tensors[(i -batch_0_len)as usize].shallow_clone())?)))).collect::<Result<Vec<_>, ThoError>>()?;


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

        let mut run_ctx = RunCtxImpl::new(1, "test");
        let r = input.with_selected_data_ref(1, run_ctx.branch(false), selector.clone(), None, |data, ctx|{
            assert_eq!(data.tensors().len(), 4);

            assert!(Tensor::arange_start(0, 1000, FLOAT_CPU).view((1, -1)).allclose(&data.tensors()[0].t.to_device(Cpu), 0.02, 0.02, false));
            assert!(Tensor::arange_start(1000, 2000, FLOAT_CPU).view((1, -1)).allclose(&data.tensors()[1].t.to_device(Cpu), 0.02, 0.02, false));
            assert!(Tensor::arange_start(2000, 3000, FLOAT_CPU).view((1, -1)).allclose(&data.tensors()[2].t.to_device(Cpu), 0.02, 0.02, false));
            assert!(Tensor::arange_start(3000, 4000, FLOAT_CPU).view((1, -1)).allclose(&data.tensors()[3].t.to_device(Cpu), 0.02, 0.02, false));

            debug!("data={data:?}");
            Ok(())
        }).await?;


        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_loader_adjuster_degrades_to_slice() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        type AdjusterStage = mock_cross_loader_clc::MockCrossLoaderClc<TestBatchOutput, RunCtxImpl>;

        let ca = CacheArea::new(1);
        let resources_0 = vec![Arc::new(Resource::new(0, 1, "r0", 4))];
        let resources_1 = vec![Arc::new(Resource::new(1, 1, "r1", 8))];

        let batch_0_tensors = vec![Tensor::arange_start(0, 4, FLOAT_CPU).view((1, -1))];
        let batch_1_tensors = vec![Tensor::arange_start(0, 8, FLOAT_CPU).view((1, -1))];

        let batch_0 = vec![
            Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData, SynKey>::new("".into(), batch_0_tensors[0].shallow_clone())?)),
        ];
        let batch_1 = vec![
            Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData, SynKey>::new("".into(), batch_1_tensors[0].shallow_clone())?)),
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

        let mut run_ctx = RunCtxImpl::new(1, "test_adjuster_slice");
        input
            .with_selected_data_ref(1, run_ctx.branch(false), selector.clone(), None, |data, _ctx| {
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
            .with_selected_data_ref(1, run_ctx.branch(false), selector.clone(), None, |data, _ctx| {
                assert_eq!(data.tensors()[1].t.size()[1], 4);
                Ok(())
            })
            .await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cross_loader_adjuster_slice_advances_resources() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        type AdjusterStage = mock_cross_loader_clc::MockCrossLoaderClc<TestBatchOutput, RunCtxImpl>;

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
            Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData, SynKey>::new("".into(), batch_0_tensors[0].shallow_clone())?)),
        ];
        let batch_1 = vec![
            Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData, SynKey>::new("".into(), batch_1_tensors[0].shallow_clone())?)),
            Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData, SynKey>::new("".into(), batch_1_tensors[1].shallow_clone())?)),
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

        let mut run_ctx = RunCtxImpl::new(1, "test_adjuster_slice_resources");
        input
            .with_selected_data_ref(1, run_ctx.branch(false), selector.clone(), None, |data, _ctx| {
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
            .with_selected_data_ref(1, run_ctx.branch(false), selector.clone(), None, |data, _ctx| {
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
            .with_selected_data_ref(1, run_ctx.branch(false), selector, None, |data, _ctx| {
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
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);

        let tensor_len = 10usize;
        let unfold_size = 2i64;
        let unfold_step = 1i64;

        let test_dir = Path::new("/tmp/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 1, 1000, 32000)?));
        let store_path = thoosa_shared::perf_log!(thoosa_shared::sync_read_lock!(ws).store_path().to_path_buf());
        let store = Arc::new(SqlStore::from_path(&store_path).await?);
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let resource_keys: Vec<String> = vec!["111", "222", "333", "444", "555", "666"]
            .into_iter()
            .map(|v| v.to_string())
            .collect();


        let mut ctx = RunCtxImpl::new(1, "test_cross_clc");

        let resource_stats = resource_keys
            .iter()
            .map(|key| (key.clone(), (ResourceStats::new(vec![1, tensor_len as u32]), vec![])))
            .collect::<HashMap<_, _>>();

        let import_id = store.save_import("test resources".to_string()).await?;
        store.save_resources_labels(&resource_stats, import_id).await?;

        let run_config_id = store.save_run_config(
            None,
            &"find_motifs",
            "test_cross_clc",
            HashMap::new(),
        ).await?;
        let run_config = FindMotifsRunConfig::new_with_running_mean(Some(run_config_id), 1, 2, None, None);
        let fixed_config = FixedFindMotifsConfig::new(2, false);

        let resource_key_refs = resource_keys.iter().map(|v| v.as_str()).collect::<Vec<_>>();
        let resources = store.list_resources_by_file_stems(&resource_key_refs).await?;
        let resources_to_move = resources.clone();
        let oom_triggered = Arc::new(AtomicBool::new(false));
        let oom_triggered_cloned = Arc::clone(&oom_triggered);

        let loader = Arc::new(RawTensorLoader::<RunCtxImpl>::new(
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


        let batch_clc_fn: Box<dyn CrossBatchFn<DistsMask, (), RunCtxImpl>> = Box::new(
            move |t_0: Tensor, t_1: Tensor, _input: &CrossInput<Arc<Resource>>, _ctx: &(), _: &RunCtxImpl| {

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
                Ok(DistsMask::new(mask, Some(dists), mask_type((mask_size, dists_size)))?)
            },
        );
        let batch_clc = Arc::new(CrossUnfoldClc::new(
            batch_clc_fn,
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
                    Arc::new(DistsTracker::<RunCtxImpl>::new(
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

        let input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(ResourceType::List(resource_keys)))));

        let output:Arc<DevicesWrap<()>> = cross_loader
            .run(input, ctx.branch(false))
            .await?;

        assert_eq!(output.id(), &());
        if oom_limit.is_some() {
            assert!(oom_triggered.load(Ordering::SeqCst));
        }

        let slices_per_resource = (tensor_len as i64 - unfold_size + 1) as u32;
        let mut expected = HashMap::new();

        for (i_0, resource_0) in resources.iter().enumerate() {
            let base_0 = (i_0 + 1) as f32;
            for (i_1, resource_1) in resources.iter().enumerate().skip(i_0) {
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
        debug!("test_cross_clc: expected={expected:?}");

        let expected_len = expected.len();
        assert!(expected_len > 0);

        let slices_path = thoosa_shared::perf_log!(thoosa_shared::sync_read_lock!(ws).tmp_path().join(format!("{run_config_id}_0_slices.zstd")));
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
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

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
                                    ConsumeResult::Ok(())
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

        let input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(ResourceType::List(vec![
            "resource_a".to_string(),
            "resource_b".to_string(),
        ])))));

        let _ = cross_loader
            .run(input, ctx.branch(false))
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
    async fn test_cross_loader_flushes_before_large_resource() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let resource_small: Resource = (1, 1, "resource_small", 2).into();
        let resource_large: Resource = (2, 1, "resource_large", 8).into();
        let resources = vec![resource_small.clone(), resource_large.clone()];

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_run_cached([
                    (resource_small.clone(), Tensor::from_slice(&[1.0, 2.0]).view((1, 2))),
                    (resource_large.clone(), Tensor::arange(8, (Kind::Float, Device::Cpu)).view((1, 8))),
                ])
                .finish(),
        );

        let observations = Arc::new(Mutex::new(Vec::<Observation<SynKey>>::new()));
        let batch_observations = Arc::clone(&observations);

        let batch_clc = Arc::new(
            mock_cross_loader_clc::build::<TestBatchOutput, MockRunCtx>()
                .with_run_batch_fn(move |input, _ctx| {
                    batch_observations.lock().unwrap().push(Observation {
                        tensor_ids: input.tensors().iter().map(|t| t.id().clone()).collect::<Vec<_>>(),
                        batch_0_ids: input.batch_0().iter().map(|r| r.as_ref().id()).collect(),
                        batch_1_ids: input.batch_1().iter().map(|r| r.as_ref().id()).collect(),
                    });

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
                .with_receive_fn(|_out, _ctx| ConsumeResult::Ok(()))
                .finish(),
        );

        let mut ctx = mock_run_ctx::default(1);

        let mut mocked_store = Arc::new(
            MockStoreBuilder::new()
                .with_list_resources_by_file_stems(
                    vec!["resource_small".to_string(), "resource_large".to_string()],
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
            3,
            |batch| async { Ok(batch) }.boxed(),
            |_receiver| async {
                Ok(())
            }.boxed(),
            CrossUnfoldClcOutputMergeMode::GpuMerge,
            0,
        ));

        let input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(ResourceType::List(vec![
            "resource_small".to_string(),
            "resource_large".to_string(),
        ])))));

        let _ = cross_loader.run(input, ctx.branch(false)).await?;

        let observed = observations.lock().unwrap().clone();
        let expected = vec![
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_small", "resource_small")]),
                    SynKey::new([mono_key!("resource_small", "resource_small")]),
                ],
                batch_0_ids: vec![resource_small.id()],
                batch_1_ids: vec![resource_small.id()],
            },
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_small", "resource_small")]),
                    SynKey::new([mono_key!("resource_large", "resource_large")]),
                ],
                batch_0_ids: vec![resource_small.id()],
                batch_1_ids: vec![resource_large.id()],
            },
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_large", "resource_large")]),
                    SynKey::new([mono_key!("resource_small", "resource_small")]),
                ],
                batch_0_ids: vec![resource_large.id()],
                batch_1_ids: vec![resource_small.id()],
            },
            Observation {
                tensor_ids: vec![
                    SynKey::new([mono_key!("resource_large", "resource_large")]),
                    SynKey::new([mono_key!("resource_large", "resource_large")]),
                ],
                batch_0_ids: vec![resource_large.id()],
                batch_1_ids: vec![resource_large.id()],
            },
        ];

        assert_eq!(observed, expected);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_cross_loader_splits_large_tensor() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

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
                .with_receive_fn(|_out, _ctx| ConsumeResult::Ok(()))
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

        let input = Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(ResourceType::List(vec![
            "resource_a".to_string(),
        ])))));

        let _ = cross_loader.run(input, ctx.branch(false)).await?;

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
