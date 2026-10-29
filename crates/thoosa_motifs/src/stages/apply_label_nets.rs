use crate::dims::{ChannelsMotifsData, LabelProbability, MotifProbability};
use crate::motif_tho_tensor::MotifThoTensor;
use crate::motifs_store::MotifsStore;
use crate::train_nets::{FinalNetParams, LabelNetParams, MotifNetParams};
use async_trait::async_trait;
use cached_derive::Cached;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::{debug, info, trace, warn};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use tch::Device::Cuda;
use tch::Kind::Float;
use tch::{IndexOp, Tensor};
use thoosa_common::data_with_extra::DataWithExtra;
use thoosa_common::dims::{Batch, Slice};
use thoosa_common::nets::batches::BatchLengths;
use thoosa_common::nets::common_nets::ThoModuleWithStore;
use thoosa_common::nets::nodes::ThoModule;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::common::DropIf;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, ConsumeData, DevicesWrap, SelectRef, ThoDataRef};
use thoosa_common::runner::global_ctx::CacheKey;
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{Adjusters, NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoStageAdjuster, OOMHandler, SyncStageInit, WithRunCtxContext, run_ctx_adjuster};
use thoosa_common::runner::sync_stage::ConsumeResult;
use thoosa_common::runner::sync_stage::{ErasedStageWithState, StageWithState, SyncStage};
use thoosa_common::runner::tmp_cache::TmpCached;
use thoosa_common::tch_error::{tch_error, tch_result};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::comp_id::{SynKey, hashmap_as_vec};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::traits::{Adjustable, AsyncStage, CachableIdTr, Data, DebugId, MaxParallelCtx, MonoMetrics, OutId, RunCtx, Stage, WithId};
use thoosa_shared::{CopyToDevice, FuturesOrdering, ThoError, collect_futures_with_progress, mono_key, perf_log, sync_read_lock};
use thoosa_store::{Store, TrainedNet, TrainedNetsBy};
use tokio::sync::MutexGuard;

use crate::stages::apply_motif_nets::{APPLY_MOTIF_NET_ENDPOINT, ApplyMotifNets, STACK_APPLIED_MOTIF_NETS_ENDPOINT, StackAppliedMotifNets, StackAppliedMotifNetsInput, SyncApplyMotifNet};
use crate::stages::load_motif_net::MotifNetMeta;
use crate::stages::stage_ids::APPLY_LABEL_NET;
use crate::stages::stft::SyncStftSelector;
use thoosa_common::dims::ChannelsData;
use thoosa_common::instance_cache::InstanceCache;
use thoosa_common::ipc::cuda_wokrer::{AllTensors, WokerRunCtx, WorkerStageInput};
use thoosa_common::ipc::proxy::{DynSyncStageEnum, DynSyncStageWithStateEnum, ProxyStage};
use thoosa_common::ipc::serialization::json::{IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::worker_msg::WorkerInput;
use thoosa_common::ipc::wraps::SyncStageIpcWrapper;
use thoosa_common::ipc::wraps::{WorkerWrapper, WorkerWrapperWithState};
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_shared::tracked_await::track_await;

pub const APPLY_LABEL_NET_ENDPOINT: &str = "APPLY_LABEL_NET_ENDPOINT";

fn ordered_label_features(discovered: impl IntoIterator<Item = (RunConfigId, String)>, requested: &[RunConfigId]) -> anyhow::Result<Vec<(RunConfigId, String)>> {
    let mut labels = HashMap::new();
    for (id, label) in discovered {
        anyhow::ensure!(labels.insert(id, label).is_none(), "ambiguous label model for run {id}");
    }
    let mut seen = HashSet::new();
    requested.iter().map(|&id| {
        anyhow::ensure!(seen.insert(id), "duplicate label feature run {id}");
        let label = labels.remove(&id).ok_or_else(|| anyhow::anyhow!("missing label model for feature run {id}"))?;
        Ok((id, label))
    }).collect()
}


pub type MotifNetsOutput = DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>;

#[derive(Debug)]
struct ApplyNets<C, AMN>
where
    C: RunCtx + Adjustable + 'static,
    AMN: AsyncStage<Ctx = C, Out = Arc<DevicesWrap<MotifNetsOutput>>>, // + thoosa_common::runner::common::runnable::WrappedRunInputStage,
    for<'a> AMN: Stage<In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
{
    apply_motif_nets: Arc<AMN>,
    apply_label_net: Arc<DynSyncStageWithStateEnum<ApplyLabelNet<C>,ApplyLabelNetInit>>
}

/// Applies label nets and returns per-model probabilities. Output is a flat list of label-model
/// scores (Model1ForLabelA, Model2ForLabelA, ...), not a per-label aggregate.
/// This can be used both for applying models for various labels or many different models for same label (for final model or inference)
#[derive(Debug, Cached, Named)]
pub struct ApplyLabelNets<C, AMN>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    AMN: AsyncStage<Ctx = C, Out = Arc<DevicesWrap<MotifNetsOutput>>>,
    for<'a> AMN: Stage<In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
{
    id: SynKey,
    apply_nets: Vec<ApplyNets<C, AMN>>,
    stack_applied_label_nets: Arc<DynSyncStageEnum<StackAppliedLabelNets<C>>>,
    pd: PhantomData<C>,
}


impl<C> ApplyLabelNets<C, ApplyMotifNets<C>>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    /// Reconstruct feature positions from the schema stored alongside the final model's weights.
    pub async fn from_final_net_params<R>(
        ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
        store: &Arc<dyn MotifsStore>,
        params: &FinalNetParams,
        stage_ctx: R,
        motif_slices_step: u32,
    ) -> Result<Self, ThoError>
    where
        R: RunCtx + MaxParallelCtx + Adjustable + 'static,
    {
        Self::new_with_apply_motif_nets(ws, store, params.label_feature_schema()?.to_vec(), stage_ctx, motif_slices_step).await
    }

    pub async fn new_with_apply_motif_nets<R>(
        ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
        store: &Arc<dyn MotifsStore>,
        train_label_nets_run_config_ids: Vec<RunConfigId>,
        stage_ctx: R,
        motif_slices_step: u32,
    ) -> Result<Self, ThoError>
    where
        R: RunCtx + MaxParallelCtx + Adjustable + 'static,
    {
        let motif_ws = Arc::clone(ws);
        let motif_store = Arc::clone(store);

        track_await!(Self::new_with_motif_nets_builder(
            ws,
            store,
            train_label_nets_run_config_ids,
            stage_ctx,
            move |stage_ctx, motif_nets| {
                let ws = Arc::clone(&motif_ws);
                let store = Arc::clone(&motif_store);

                async move {
                    track_await!(ApplyMotifNets::from_trained_nets(
                        &ws,
                        &store,
                        stage_ctx,
                        motif_slices_step,
                        None,
                        motif_nets,
                        false,
                    )).await
                }
            },
        )).await
    }
}

impl<C, AMN> ApplyLabelNets<C, AMN>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    AMN: AsyncStage<Ctx = C, Out = Arc<DevicesWrap<MotifNetsOutput>>>,
    for<'a> AMN: Stage<In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
{
    pub async fn new_with_motif_nets_builder<R, F, Fut>(
        ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
        store: &Arc<dyn MotifsStore>,
        train_label_nets_run_config_ids: Vec<RunConfigId>,
        mut stage_ctx: R,
        mut build_motif_nets: F,
    ) -> Result<Self, ThoError>
    where
        R: RunCtx + MaxParallelCtx + Adjustable + 'static,
        F: FnMut(R, Vec<TrainedNet>) -> Fut,
        Fut: Future<Output = Result<AMN, ThoError>>,
    {
        let tho_device_idx = stage_ctx.tho_device_idx();

        // Discovery reads model files and metadata under a synchronous workspace lock.
        let discovery_ws = Arc::clone(ws);
        let requested_ids = train_label_nets_run_config_ids.clone();
        let features = tokio::task::spawn_blocking(move || -> Result<_, ThoError> {
            let all_label_nets = perf_log!(sync_read_lock!(discovery_ws)).load_label_nets(Some(requested_ids.as_slice()), &ThoDevice::from(tho_device_idx).into(), false, false)?;
            let discovered = all_label_nets.into_iter().flat_map(|(label, nets)| nets.into_iter().map(move |(id, _, _)| (id, label.clone())));
            Ok(ordered_label_features(discovered, &requested_ids)?)
        }).await??;

        let mut apply_nets = vec![];
        let mut train_label_net_run_config_ids = vec![];

        // Requested IDs define feature positions; discovery only resolves their labels.
        for (train_label_net_run_config_id, label) in features {
            let label_nets = track_await!(store.list_trained_nets(TrainedNetsBy::RunConfigIds(vec![train_label_net_run_config_id]), None, None)).await?;

            debug!("ApplyLabelNets::new: train_label_net_run_config_id={train_label_net_run_config_id} label_nets={label_nets:?}");

            if label_nets.len() != 1 { return Err(ThoError::new(format!("expected one label model for run {train_label_net_run_config_id}, found {}", label_nets.len()))); }
            let label_net = &label_nets[0];

            let motif_net_run_config_ids = track_await!(store.list_input_nets(label_net.id)).await?;

            info!("ApplyLabelNets::new: label_net_id={}  motif_net_ids={motif_net_run_config_ids:?}", label_net.id);

            if motif_net_run_config_ids.is_empty() { return Err(ThoError::new(format!("label model run {train_label_net_run_config_id} has no input motif models"))); }

            let motif_nets = track_await!(store.list_trained_nets(TrainedNetsBy::RunConfigIds(motif_net_run_config_ids), None, None)).await?;

            debug!("ApplyLabelNets::new: train_label_net_run_config_id={train_label_net_run_config_id} motif_nets={motif_nets:?}");

            let apply_label_net_init = ApplyLabelNetInit::new(
                label_net.id,
                label.clone(),
                train_label_net_run_config_id,
                label_net.net_path.clone(),
                false,
            );
            train_label_net_run_config_ids.push(train_label_net_run_config_id);
            apply_nets.push(ApplyNets {
                apply_motif_nets: Arc::new(track_await!(build_motif_nets(stage_ctx.branch(false), motif_nets)).await?),
                apply_label_net: track_await!(ApplyLabelNet::new(apply_label_net_init, stage_ctx.branch(true), ws)).await?,
            });
        }


        Ok(Self {
            id: train_label_net_run_config_ids.into_iter().map(|run_config_id|mono_key!(run_config_id, format!("run_config_id={run_config_id}"))).collect::<Vec<_>>().into(),
            apply_nets,
            // Reuse the endpoint for every resource; ordering and output identity belong to each request.
            stack_applied_label_nets: StackAppliedLabelNets::<C>::new(tho_device_idx.into()).await?,
            pd: PhantomData,
        })
    }

    pub fn num_nets(&self) -> usize {
        self.apply_nets.len()
    }
}

impl<C, AMN> Stage for ApplyLabelNets<C, AMN>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    AMN: AsyncStage<Ctx = C, Out = Arc<DevicesWrap<MotifNetsOutput>>>,
    for<'a> AMN: Stage<In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
{
    type Ctx = C;
    type In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;
    type Out = Arc<DevicesWrap<ThoTensor<LabelProbability, SynKey>>>;
}

impl<C, AMN> AsyncStage for ApplyLabelNets<C, AMN>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    AMN: AsyncStage<Ctx = C, Out = Arc<DevicesWrap<MotifNetsOutput>>>, // + thoosa_common::runner::common::runnable::WrappedRunInputStage,
    for<'a> AMN: Stage<In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
{
    // type RunInput = Arc<DevicesWrap<Self::In<'static>>>;
    // type RunOutput = Arc<DevicesWrap<Self::Out>>;

    // fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
    //     thoosa_shared::traits::WithId::id(input)
    // }
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyLabelNets::run"))]
    async fn run(
        self: Arc<Self>,
        input: Self::In<'static>,
        mut ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        debug!("ApplyLabelNets::run: input={input:?}");
        let input_id = input.id().clone();


        debug!("ApplyLabelNets::run: input={input:?} self.apply_nets.len={} label_nets={:?}", self.apply_nets.len(), self.apply_nets);

        let tmp_cache = track_await!(TmpCached::new("tmp_stack_applied_label_nets")).await;
        let tmp_cache_key = tmp_cache.cache_key().clone();
        let tmp_cache: Arc<Mutex<TmpCached<ThoTensor<Batch<LabelProbability>, SynKey>>>> = Arc::new(Mutex::new(tmp_cache));
        let label_output_order = Arc::new(Mutex::new(vec![None; self.apply_nets.len()]));

        //labels order is maintained in StackAppliedLabelNets::consume
        let futures = self.apply_nets
            .iter()
            .enumerate()
            .map(|(apply_net_idx, apply_net)| {
                let input: Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>> = Arc::clone(&input);
                let apply_motif_nets = Arc::clone(&apply_net.apply_motif_nets);
                let apply_label_net = Arc::clone(&apply_net.apply_label_net);
                let mut ctx = ctx.branch(false);
                let tmp_cache = Arc::clone(&tmp_cache);
                let label_output_order = Arc::clone(&label_output_order);
                async move {

                    #[cfg(test)]{
                        {
                            let input = SelectRef::select_ref(&input, ctx.tho_device_idx(), None::<SyncStftSelector>).await?;
                            trace!("ApplyLabelNets::run: input=\n{}", input.as_ref().as_ref());
                            Ok::<(), ThoError>(())
                        }?;
                    }

                    //First apply motif nets to the raw input to produce motif probabilities used by label nets.
                    let motif_probs: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>> = apply_motif_nets
                        .run(input, ctx.branch(false))
                        .await?;

                    debug!("ApplyLabelNets::run: motif_probs={motif_probs:?}");

                    #[cfg(test)]{
                        {
                            let probs = SelectRef::select_ref(&motif_probs, ctx.tho_device_idx(), None::<SyncStftSelector>).await?;
                            trace!("ApplyLabelNets::run: probs=\n{}", probs.as_ref().data().as_ref());
                            Ok::<(), ThoError>(())
                        }?;
                    }

                    //Each label net runs independently; ordering is preserved by buffer mode below - (relevant for final net training, if final inference done per label it doesn't matter).
                    // Preserve the actual cache owners when the motif stage returns cached probabilities.
                    let mut required_inputs = HashMap::new();
                    motif_probs.add_required_ids(&mut required_inputs);
                    let required_ids = required_inputs.iter().map(|(key, ids)| (key.clone(), ids)).collect();
                    let input = Arc::new(ApplyLabelNetInputWrapper::new(Arc::clone(&motif_probs)));

                    let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()> = OOMHandler::new(vec![
                        Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
                        Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("apply_label_net_init"))),
                    ], None);
                    let result = track_await!(oom_handler.run(
                        &apply_label_net,
                        input,
                        ctx.branch(false),
                         WithRunCtxContext::new(MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids))),)
                    ).await?.into_inner();

                    debug!("ApplyLabelNets::run: result={result:?}");

                    #[cfg(test)]{
                        {
                            trace!("ApplyLabelNets::run: result=\n{}", result.as_ref());
                            Ok::<(), ThoError>(())
                        }?;
                    }

                    label_output_order.lock()[apply_net_idx] = Some(result.id().clone());
                    tmp_cache.lock().insert(result);

                    Ok::<_, ThoError>(())
                }
            });

        let _label_outputs = collect_futures_with_progress(
            futures,
            self.apply_nets.len(),
            FuturesOrdering::Ordered,
            thoosa_shared::CollectionErrorPolicy::Propagate,
            "ApplyLabelNets::run",
        )
        .await?
        .into_iter()
        .collect::<Vec<_>>();

        debug!("ApplyLabelNets::run: label_outputs stacked label_output_order={label_output_order:?} count={}", self.apply_nets.len());

        let label_output_order = Arc::into_inner(label_output_order)
            .expect("ApplyLabelNets::run: label_output_order still shared")
            .into_inner()
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ThoError::new("ApplyLabelNets::run: missing label output id"))?;
        // Keep every label probability resident while the consuming stack stage retries.
        let required_tmp_cache_ids = label_output_order.iter().map(|id| id as &dyn CachableIdTr).collect();
        let order = label_output_order.iter().cloned().enumerate().map(|(idx, id)| (id, idx)).collect();
        let stack_input = StackAppliedLabelNetsInputWrapper::new(self.out_id(&input_id), order, Arc::into_inner(tmp_cache).expect("ApplyLabelNets::run: tmp_cache still shared").into_inner());
        let required_ids = HashMap::from([(tmp_cache_key, &required_tmp_cache_ids)]);
        let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()> = OOMHandler::new(
            vec![
                Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
                Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("stack_apply_label_nets"))),
            ],
            None,
        );
        let label_outputs = track_await!(oom_handler.consume(
            &self.stack_applied_label_nets,
            stack_input,
            ctx.branch(true),
            WithRunCtxContext::new(
                MaxParallelAndDumpToCpuAdjustableContex::new(
                    MaxParallelAdjustContextImpl::new(),
                    DumpToCpuContextImpl::new(&required_ids),
                ),
            ),
        )).await?.into_inner();



        Ok(Arc::new(DevicesWrap::wrap(label_outputs)))
    }
}

impl<C, AMN> OutId for ApplyLabelNets<C, AMN>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    AMN: AsyncStage<Ctx = C, Out = Arc<DevicesWrap<MotifNetsOutput>>>, // + thoosa_common::runner::common::runnable::WrappedRunInputStage,
    for<'a> AMN: Stage<In<'a> = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StackAppliedLabelNetsInput {
    // Full aggregate identity and resource-specific ordering must travel together on retries and over IPC.
    id: SynKey,
    #[serde(with = "hashmap_as_vec")]
    label_output_order: HashMap<SynKey, usize>,
    device_idx: usize,
    label_outputs: Vec<ThoTensor<Batch<LabelProbability>, SynKey>>,
    // Preserve the temporary cache identity through local and IPC consume retries.
    cache_name: String,
}

impl StackAppliedLabelNetsInput {
    fn new(id: SynKey, label_output_order: HashMap<SynKey, usize>, mut label_outputs: Vec<ThoTensor<Batch<LabelProbability>, SynKey>>, device_idx: usize, cache_name: String) -> Self {
        for t in &mut label_outputs {
            t.no_drop();
        }
        Self { id, label_output_order, device_idx, label_outputs, cache_name }
    }
}

impl WorkerStageInput for StackAppliedLabelNetsInput {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        // Preserve stack ordering while cloning only worker tensor references.
        let outputs = self.label_outputs.iter().map(ThoTensor::clone_ref).collect::<Result<Vec<_>, _>>()?;
        Ok(Box::new(Self::new(self.id.clone(), self.label_output_order.clone(), outputs, self.device_idx, self.cache_name.clone())))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        let outputs = self.label_outputs.iter().map(|tensor| WorkerStageInput::ref_to_tensor(tensor, all_cuda_tensors, remove)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self::new(self.id.clone(), self.label_output_order.clone(), outputs, self.device_idx, self.cache_name.clone()))
    }

}

#[typetag::serde(name = "StackAppliedLabelNetsInput")]
impl WorkerInput for StackAppliedLabelNetsInput {}

impl WithId for StackAppliedLabelNetsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for StackAppliedLabelNetsInput {
    type Metrics = MonoMetrics<0, 1>;


    fn size(&self) -> Self::Metrics {
        self.label_outputs.len().into()
    }

    fn tho_device_idx(&self) -> usize {
        self.device_idx
    }

    fn mem_size(&self) -> u64 {
        self.label_outputs.iter().map(|label_output| label_output.mem_size()).sum()
    }
}

impl Input for StackAppliedLabelNetsInput {}
impl Adjustable for StackAppliedLabelNetsInput {}

#[derive(Debug)]
struct StackAppliedLabelNetsInputWrapper {
    id: SynKey,
    label_output_order: HashMap<SynKey, usize>,
    tmp_cache: TmpCached<ThoTensor<Batch<LabelProbability>, SynKey>>,
}

impl StackAppliedLabelNetsInputWrapper {
    fn new(id: SynKey, label_output_order: HashMap<SynKey, usize>, tmp_cache: TmpCached<ThoTensor<Batch<LabelProbability>, SynKey>>) -> Self {
        Self { id, label_output_order, tmp_cache }
    }
}

impl ConsumeData for StackAppliedLabelNetsInputWrapper {
    type O = StackAppliedLabelNetsInput;

    fn consume<R: Send, F, P>(
        self,
        to_device_idx: usize,
        run_ctx: P,
        do_run: F,
    ) -> impl Future<Output = ConsumeResult<Self::O, R>> + Send
    where
        F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
        P: RunCtx,
    {
        async move {
            let StackAppliedLabelNetsInputWrapper { id, label_output_order, tmp_cache } = self;

            let cache_name = tmp_cache.name.clone();
            match track_await!(tmp_cache.take_all_recovering(to_device_idx)).await {
                Ok(label_outputs) => do_run(StackAppliedLabelNetsInput::new(id, label_output_order, label_outputs, to_device_idx, cache_name), run_ctx),
                Err(err) => ConsumeResult::Failed { input: StackAppliedLabelNetsInput::new(id, label_output_order, err.data, to_device_idx, cache_name), error: err.error },
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="StackAppliedLabelNetsInputWrapper::burp"))]
    fn burp(o: Self::O) -> impl Future<Output = Self> + Send {
        async move {
            let cache = TmpCached::restore(o.cache_name, o.label_outputs);
            StackAppliedLabelNetsInputWrapper::new(o.id, o.label_output_order, cache)
        }
    }
}

pub const STACK_APPLIED_LABEL_NETS_ENDPOINT: &str = "STACK_APPLIED_LABEL_NETS_ENDPOINT";

// Installing a stateless stack endpoint needs no resource or label-set metadata.
#[derive(Debug, Serialize, Deserialize)]
pub struct StackAppliedLabelNetsInit;

#[typetag::serde(name = "StackAppliedLabelNetsInit")]
impl WorkerInput for StackAppliedLabelNetsInit {}

#[derive(Debug, Named)]
pub struct StackAppliedLabelNets<C>
where
    C: RunCtx + 'static,
{
    pd: PhantomData<C>,
}

static STACK_APPLIED_LABEL_NETS_INSTANCES: InstanceCache = InstanceCache::new();

impl<C> StackAppliedLabelNets<C>
where
    C: RunCtx + 'static,
{
    async fn new(device: ThoDevice) -> Result<Arc<DynSyncStageEnum<Self>>, ThoError> {
        // The shared cache serializes installation and isolates worker generations and devices.
        STACK_APPLIED_LABEL_NETS_INSTANCES.get_or_try_insert_recreatable_with::<C, _, _, _, _>(device.tho_device_idx(), GlobalCtx::global().cuda_activity_gate(device.tho_device_idx()), GlobalCtx::global().is_ipc(), (), move || async move {
            if GlobalCtx::global().is_ipc() {
                // Different context-type cache entries must not overwrite one another in the same worker.
                static STAGE_ID: AtomicU32 = AtomicU32::new(0);
                let stage_id = STAGE_ID.fetch_add(1, Relaxed);
                Ok(Arc::new(DynSyncStageEnum::Proxy {
                    stage: Arc::new(ProxyStage::proxy_recreatable(STACK_APPLIED_LABEL_NETS_ENDPOINT, Some(stage_id.to_string()), device.tho_device_idx(), Box::new(StackAppliedLabelNetsInit)).await?),
                    init: (),
                }))
            } else {
                Ok(Arc::new(DynSyncStageEnum::Local(Arc::new(Self::new_local().await?))))
            }
        }).await
    }

    pub async fn new_local() -> Result<Self, ThoError> {
        Ok(Self { pd: PhantomData })
    }
}

pub fn stack_applied_label_nets_init<'a>(stage_id:Option<String>, init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let Ok(init) = (init as Box<dyn std::any::Any>).downcast::<StackAppliedLabelNetsInit>() else {
            return Err(ThoError::new("StackAppliedLabelNets endpoint received invalid initializer"));
        };
        Ok(Box::new(WorkerWrapper::new(Arc::new(StackAppliedLabelNets::<WokerRunCtx>::new_local().await?), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}


impl<C> Adjustable for StackAppliedLabelNets<C>
where
    C: RunCtx + 'static,
{
}

impl<C> OutId for StackAppliedLabelNets<C>
where
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        input_id.clone()
    }
}

impl<C> Stage for StackAppliedLabelNets<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = StackAppliedLabelNetsInput;
    type Out = MotifThoTensor<LabelProbability>;
}

impl<C> SyncStage for StackAppliedLabelNets<C>
where
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="StackAppliedLabelNets::consume"))]
    fn consume<'f>(self: Arc<Self>, mut input: Self::In<'f>, _ctx: Self::Ctx) -> ConsumeResult<Self::In<'f>, Self::Out> {
        let input_id = input.id().clone();
        let mut label_outputs = std::mem::take(&mut input.label_outputs);
        label_outputs.sort_by_key(|label_output| input.label_output_order.get(label_output.id()).copied().unwrap_or(usize::MAX));

        let out_id = self.out_id(&input_id);
        let tensor = if label_outputs.is_empty() {
            match Tensor::f_empty([0], (Float, ThoDevice::from(input.device_idx).into())) {
                Ok(tensor) => tensor,
                Err(error) => {
                    input.label_outputs = label_outputs;
                    return ConsumeResult::Failed { input, error: tch_error(error) };
                }
            }
        } else {
            let stacked = label_outputs.iter().map(|label_output| {
                trace!("StackAppliedLabelNets::consume: label_output.size=\n{}", label_output.t_ref());
                // Ok(label_output.t_ref().f_sigmoid()?)
                Ok(label_output.t_ref())
            }).collect::<Result<Vec<_>, _>>();

            match stacked.and_then(|stacked| Tensor::f_stack(stacked.as_slice(), 0)) {
                Ok(tensor) => tensor,
                Err(error) => {
                    input.label_outputs = label_outputs;
                    return ConsumeResult::Failed { input, error: tch_error(error) };
                }
            }
        };

        match ThoTensor::new(out_id, tensor) {
            Ok(tensor) => ConsumeResult::Ok(tensor.into()),
            Err(error) => {
                input.label_outputs = label_outputs;
                ConsumeResult::Failed { input, error }
            }
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct ApplyLabelNetInput<'a> {
    id: SynKey,
    input: ThoDataRef<'a, DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>,
}

impl Data for ApplyLabelNetInput<'_> {
    type Metrics = <DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>> as Data>::Metrics;


    fn size(&self) -> Self::Metrics {
        self.input.as_ref().size()
    }

    fn tho_device_idx(&self) -> usize {
        self.input.as_ref().tho_device_idx()
    }

    fn mem_size(&self) -> u64 {
        self.input.as_ref().mem_size()
    }
}

impl WithId for ApplyLabelNetInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Input for ApplyLabelNetInput<'_> {}

impl WorkerStageInput for ApplyLabelNetInput<'_> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        let input = self.input.as_ref();
        // Own the metadata and cloned tensor reference before erasing the borrowed lifetime.
        Ok(Box::new(ApplyLabelNetInput { id: self.id.clone(), input: ThoDataRef::Owned(DataWithExtra::new(input.data().clone_ref()?, input.extra().clone())) }))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        let input = self.input.as_ref();
        Ok(Self { id: self.id.clone(), input: ThoDataRef::Owned(DataWithExtra::new(WorkerStageInput::ref_to_tensor(input.data(), all_cuda_tensors, remove)?, input.extra().clone())) })
    }
}

#[typetag::serde(name = "ApplyLabelNetInput")]
impl WorkerInput for ApplyLabelNetInput<'static> {}

#[derive(Debug)]
struct ApplyLabelNetInputWrapper {
    id: SynKey,
    input: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>,
}

impl ApplyLabelNetInputWrapper {
    fn new(input: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>>) -> Self {
        Self { id: input.id().clone(), input }
    }
}

impl BorrowRef for ApplyLabelNetInputWrapper {
    type Out<'a> = ApplyLabelNetInput<'a> where Self: 'a;

    fn borrow_ref<'a>(&'a self, to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let input = track_await!(BorrowRef::borrow_ref(&self.input, to_device_idx)).await?;
            Ok(ApplyLabelNetInput { id: self.id.clone(), input })
        }
    }

}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApplyLabelNetInit {
    trained_net_id: u32,
    label: String,
    train_label_net_run_config_id: RunConfigId,
    label_net_path:PathBuf,
    quantize:bool
}

impl ApplyLabelNetInit {
    pub fn new(trained_net_id: u32, label: String, train_label_net_run_config_id: RunConfigId, label_net_path:PathBuf, quantize:bool) -> Self {
        Self { trained_net_id, label, train_label_net_run_config_id, label_net_path, quantize }
    }

    pub(super) fn inference_id(&self) -> SynKey {
        SynKey::new([
            mono_key!(APPLY_LABEL_NET, "ApplyLabelNet"),
            mono_key!(self.trained_net_id, format!("apply label net {}", self.trained_net_id)),
            mono_key!(self.quantize, "quantize")
        ])
    }

}

impl WorkerStageInput for ApplyLabelNetInit {
}

#[typetag::serde(name = "ApplyLabelNetInit")]
impl WorkerInput for ApplyLabelNetInit {}

#[async_trait]
impl<C, RC> SyncStageInit<RC, DynSyncStageWithStateEnum<ApplyLabelNet<C>, ApplyLabelNetInit>, dyn crate::motifs_ws::MotifsWorkspace> for ApplyLabelNetInit
where
    C: RunCtx + 'static,
    RC: RunCtx + Send + Sync + 'static,
{
    async fn init(self, run_context: RC, ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>) -> Result<Arc<DynSyncStageWithStateEnum<ApplyLabelNet<C>, ApplyLabelNetInit>>, ThoError> {
        let device: ThoDevice = run_context.tho_device_idx().into();
        let id = self.inference_id();
        if GlobalCtx::global().is_ipc() {
            let (_run_config_id, _params, _) = perf_log!(sync_read_lock!(ws)).load_label_net(&self.label_net_path, &device, self.quantize, false)?;

            // Keep a cloned initializer as proxy metadata because worker installation consumes its serialized copy.
            Ok(Arc::new(DynSyncStageWithStateEnum::Proxy {

                stage: Arc::new(ProxyStage::proxy_recreatable(APPLY_LABEL_NET_ENDPOINT, Some(id.to_string()), device.tho_device_idx(), Box::new(self.clone())).await?),
                init: self,
            }))
        } else {
            let (_run_config_id, _params, net) = perf_log!(sync_read_lock!(ws)).load_label_net(&self.label_net_path, &device, self.quantize, true)?;
            let net = net.ok_or_else(|| ThoError::new(format!("workspace did not load label net from {:?}", self.label_net_path)))?;

            Ok(Arc::new(DynSyncStageWithStateEnum::Local(Arc::new(ApplyLabelNet::new_local(
                id,
                self.label,
                self.train_label_net_run_config_id,
                net,
            ).await?))))
        }
    }
}


#[derive(Debug)]
struct ApplyLabelNet<C>
where
    C: RunCtx + 'static,
{
    id: SynKey,
    label: String,
    train_label_net_run_config_id: RunConfigId,
    // Inference state is acquired once per sync run instead of being copied in every input.
    net: tokio::sync::Mutex<ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>>,
    pd: PhantomData<C>,
}

#[async_trait]
impl <C> FuturesRunnerMapPurger for ApplyLabelNet<C>
where
    C: RunCtx + 'static
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        self.net.lock().await.has_gpu_data(device_idx)
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        let mut guard = self.net.lock().await;
        if guard.has_gpu_data(device_idx) {
            *guard = guard.copy_to_device(0).await?;
        }
        Ok(())
    }

    async fn drop_gpu(&self, device_idx: usize, _name: &str) -> Result<(), ThoError> {
        self.net.lock().await.drop_gpu(device_idx)
    }
}

impl<C> WithId for ApplyLabelNet<C>
where
    C: RunCtx + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


// Loading a trained label net repeatedly must reuse both its module state and installed endpoint.
static APPLY_LABEL_NET_INSTANCES: InstanceCache = InstanceCache::new();

impl<C> ApplyLabelNet<C>
where
    C: RunCtx + 'static,
{
    async fn new<RC>(
        sync_init: ApplyLabelNetInit,
        run_context: RC,
        ws: &Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>,
    ) -> Result<Arc<DynSyncStageWithStateEnum<ApplyLabelNet<C>, ApplyLabelNetInit>>, ThoError>
    where
        RC: RunCtx + MaxParallelCtx + Adjustable + Send + Sync + 'static,
    {
        let ws = Arc::clone(ws);
        APPLY_LABEL_NET_INSTANCES.get_or_try_insert_recreatable_with::<C, _, _, _, _>(run_context.tho_device_idx(), GlobalCtx::global().cuda_activity_gate(run_context.tho_device_idx()), GlobalCtx::global().is_ipc(), sync_init.trained_net_id, move || async move {
            // Construction has no input tensors that must remain on this device, so an OOM retry may dump every cached tensor.
            let required_ids: HashMap<CacheKey, &HashSet<&dyn CachableIdTr>> = HashMap::new();
            let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()> = OOMHandler::new(
                vec![
                    Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
                    Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("apply_label_net_init"))),
                ],
                None,
            );
            let stage = oom_handler.new_stage(
                sync_init,
                run_context,
                WithRunCtxContext::new(MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids))),
                &ws,
            ).await?;

            ////ApplyLabelNet is needed for final net training - no need to unregister
            GlobalCtx::global().register_cache((stage.stage_id(), Arc::clone(&stage) as Arc<dyn FuturesRunnerMapPurger>));

            Ok(stage)
        }).await
    }

    async fn new_local(
        id:SynKey,
        label: String,
        train_label_net_run_config_id: RunConfigId,
        label_net: ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>,
    ) -> Result<Self, ThoError> {
        Ok(Self {
            id,
            label,
            train_label_net_run_config_id,
            net: tokio::sync::Mutex::new(label_net),
            pd: PhantomData,
        })
    }
}

pub fn apply_label_net_endpoint_init<'a>(stage_id:Option<String>, init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let Ok(init) = (init as Box<dyn std::any::Any>).downcast::<ApplyLabelNetInit>() else {
            return Err(ThoError::new("ApplyLabelNet endpoint received invalid initializer"));
        };
        let init = *init;
        debug!("apply_net_stage_2_endpoint_init: init={init:?}");

        let (run_config_id, params, net) = perf_log!(sync_read_lock!(ws)).load_label_net( //TODO in IPC mode this is needed only to get motif_width from  params. Add util to load params without module
               &init.label_net_path,
               &tho_device,
               init.quantize,
                true
        )?;

        let stage: ApplyLabelNet<WokerRunCtx> = ApplyLabelNet::new_local(
           init.inference_id(),
           init.label,
           init.train_label_net_run_config_id,
           net.expect("net"),
        ).await?;

        // The common stateful wrapper performs state preparation and converts worker outputs back to references.
        Ok(Box::new(WorkerWrapperWithState::new(Arc::new(stage), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

impl<C> Adjustable for ApplyLabelNet<C>
where
    C: RunCtx,
{
}

impl<C> Stage for ApplyLabelNet<C>
where
    C: RunCtx,
{
    type Ctx = C;
    type In<'a> = ApplyLabelNetInput<'a>;
    type Out = MotifThoTensor<Batch<LabelProbability>>;//TODO it's Batch<LabelProbability>
}

impl<C> SyncStage for ApplyLabelNet<C>
where
    C: RunCtx + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        Some(self)
    }
}

impl<C> StageWithState for ApplyLabelNet<C>
where
    C: RunCtx + 'static,
{
    type S = ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>;

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Option<tokio::sync::MappedMutexGuard<'_, Self::S>>, ThoError>> {
        async move {
            let mut net = self.net.lock().await;
            net.ensure_ready()?;
            if net.tho_device_idx() != to_device_idx {
                *net = net.copy_to_device(to_device_idx).await?;
            }
            Ok(Some(MutexGuard::map(net, |v|v)))
        }.boxed()
    }

    // The module owns the transfer schema and releases all parameter aliases.
    fn get_state_info(&self, _device: usize) -> BoxFuture<'_, Result<thoosa_common::runner::sync_stage::StageStateInfo, ThoError>> {
        async move { self.net.lock().await.state_info() }.boxed()
    }

    fn get_state_tensor(&self, _device: usize, tensor_id: String) -> BoxFuture<'_, Result<Tensor, ThoError>> {
        async move { self.net.lock().await.state_tensor(&tensor_id) }.boxed()
    }

    fn set_state_tensor(&self, device: usize, tensor_id: String, tensor: Tensor) -> BoxFuture<'_, Result<(), ThoError>> {
        async move { self.net.lock().await.set_state_tensor(&tensor_id, tensor, device) }.boxed()
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyLabelNet::do_run"))]
    fn do_run_with_state<'a>(self: Arc<Self>, input: &Self::In<'a>, net: &mut Self::S, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        trace!("ApplyLabelNet::do_run_with_state: id={:?} input=\n{}", self.id, input.input.as_ref().data().t_ref());
        debug!("ApplyLabelNet::do_run_with_state: id={:?} input={:?}", self.id, input);
        let out_id = self.out_id(input.id());
        let input = input.input.as_ref();
        let num_motifs = input.data().size()[0] as usize;
        let batch_input: ThoTensor<ChannelsMotifsData, DebugId> = input.data().t_ref().unsqueeze(0).into();//TODO unsqueeze is for dummy batch dimension, batching should be done in ApplyLabelNets

        // Label nets receive one length per chunk, so use the longest motif row to keep every real score while shorter rows remain zero-padded.
        let mut lengths = BatchLengths::new(vec![*input.extra.as_ref().iter().max().unwrap_or(&0)]);//there is always single element until batching is implemented

        debug!("ApplyLabelNet::do_run_with_state: id={:?} batch_input={batch_input:?} lengths={lengths:?} ", self.id);

        let out = net.tho_forward(&batch_input, Some(&mut lengths), false)?;

        debug!("ApplyLabelNet::do_run_with_state: id={:?} out={}", self.id, out.t_ref());

        ThoTensor::new(out_id, out.t()).map(Into::into)
    }
}

impl<C> OutId for ApplyLabelNet<C>
where
    C: RunCtx,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

pub fn sigmoid(v: f64) -> f64 {//TODO move to util
    1.0 / (1.0 + (-v).exp())
}


pub fn logit(x: f64) -> Result<f64, ThoError> {
    if !(0.0..1.0).contains(&x) {
        return Err(ThoError::new("logit is defined only for 0 < x < 1"));
    }

    Ok((x / (1.0 - x)).ln())
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_label_feature_schema_rejects_missing_duplicate_and_ambiguous_models() -> anyhow::Result<()> {
        use super::ordered_label_features;
        let discovered = || vec![(2, "B".to_string()), (1, "A".to_string())];
        assert_eq!(ordered_label_features(discovered(), &[1, 2])?, vec![(1, "A".into()), (2, "B".into())]);
        assert_eq!(ordered_label_features(discovered(), &[2, 1])?, discovered());
        assert!(ordered_label_features(discovered(), &[1, 3]).is_err());
        assert!(ordered_label_features(discovered(), &[1, 1]).is_err());
        assert!(ordered_label_features(vec![(1, "A".into()), (1, "B".into())], &[1]).is_err());

        let mut params = crate::train_nets::FinalNetParams::new(2, 1, 1, 1);
        assert!(params.label_feature_schema().is_err());
        assert!(params.set_label_feature_schema(vec![1]).is_err());
        assert!(params.set_label_feature_schema(vec![1, 1]).is_err());
        params.set_label_feature_schema(vec![2, 1])?;
        let restored: crate::train_nets::FinalNetParams = serde_json5::from_str(&serde_json5::to_string(&params)?)?;
        assert_eq!(restored.label_feature_schema()?, &[2, 1]);
        let legacy: crate::train_nets::FinalNetParams = serde_json5::from_str("{in_features: 2, mid_features: 1, out_features: 1, num_layers: 1}")?;
        assert!(legacy.label_feature_schema().is_err());
        Ok(())
    }
    use crate::dims::{ChannelsMotifsData, LabelProbability, MotifProbability};
    use crate::motifs_store::MockMotifsStoreBuilder;
    use crate::stages::apply_label_nets::{ApplyLabelNet, ApplyLabelNetInput, ApplyLabelNets, logit, sigmoid};
    use crate::stages::apply_motif_nets::ApplyMotifNets;
    use crate::stages::find_motifs_run_config::FindMotifsRunConfig;
    use crate::stages::stft::SyncStftSelector;
    use crate::train_nets::{FinalNetParams, LabelNetParams, LabelNetTypeParams, MotifCosine2DNetArgs, MotifNetParams, MotifNetTypeArgs, TrainMotifNetRunConfig};
    use log::LevelFilter::{Trace, Warn};
    use log::{debug, set_max_level, trace};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tch::Kind::Float;
    use tch::nn::{self, VarStore};
    use tch::{Device, Kind, Tensor};
    use thoosa_common::data_with_extra::DataWithExtra;
    use thoosa_common::dims::ChannelsData;
    use thoosa_common::dims::{Batch, Slice};
    use thoosa_common::nets::batches::BatchLengths;
    use thoosa_common::nets::common_nets::{CommonTrainingRunConfig, LinLenNetParams, ThoModuleWithStore};
    use thoosa_common::nets::nodes::{ThoClosureModule, ThoModule};
    use thoosa_common::runner::common::cachable::{DevicesWrap, SelectRef, ThoDataRef};
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::global_ctx::{GlobalCtx, mock_global_ctx};
    use thoosa_common::runner::sync_stage::StageWithState;
    use thoosa_common::tch_error::tch_result;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::{assert_close, to_vec_f32};
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::traits::{AsyncStage, OutId, RunCtx, WithId};
    use thoosa_shared::{ThoError, mono_key};
    use thoosa_store::{RunConfigRow, TrainedNet};

    fn const_motif_net(
        motif_width: u32,
        step: u32,
        k: f64,
        device: Device,
    ) -> (
        Box<dyn Fn() -> MotifNetParams + Sync + Send + 'static>,
        Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send + 'static>,
    ) {
        (
            Box::new(move || MotifNetParams::new(motif_width, 1, 1, step, 32, 10)),
            Box::new(move || {
                let k = k;
                let builder = Arc::new(move |_path: &nn::Path| {
                    let module = ThoClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        let batch = t.size()[0];
                        // let out = Tensor::f_ones(&[batch, 1], (Kind::Float, t.device()))?
                        let out = t.f_mean_dim(1,  true, Float )?.f_mul_scalar_(k)?;;

                        debug!("const_motif_net: k={k} t=\n{t} \nout={out}");
                        Ok(out)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<Batch<Slice>, MotifProbability>>)
                });
                let mut vs = VarStore::new(device);
                let module = builder(&vs.root()).expect("const_motif_net builder");
                ThoModuleWithStore::with_factory(
                    SynKey::new([mono_key!(format!("const_motif_net_{k}"), "const_motif_net")]),
                    module,
                    vs,
                    Some(builder),
                )
            }),
        )
    }

    fn sum_label_net(
        motif_nets: u32,
        slices_len: u32,
        k: f64,
        device: Device,
    ) -> (
        Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>,
        Box<dyn Fn() -> ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>> + Sync + Send + 'static>,
    ) {
        (
            Box::new(move || {
                LabelNetParams::new(
                    motif_nets,
                    LabelNetTypeParams::LinNetParams(LinLenNetParams::new(slices_len, 1)),
                )
            }),
            Box::new(move || {
                let k = k;
                let builder = Arc::new(move |_path: &nn::Path| {
                    let module = ThoClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        let out = sum_label_t(k, t)?;
                        Ok(out)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<ChannelsMotifsData, Batch<LabelProbability>>>)
                });
                let mut vs = VarStore::new(device);
                let module = builder(&vs.root()).expect("sum_label_net builder");
                ThoModuleWithStore::with_factory(
                    SynKey::new([mono_key!(format!("sum_label_net_{k}"), "sum_label_net")]),
                    module,
                    vs,
                    Some(builder),
                )
            }),
        )
    }

    fn sum_label_t(k: f64, t: &Tensor) -> Result<Tensor, ThoError> {
        let sum = tch_result(t.f_sum_dim_intlist(&[1, 2][..], false, None))?;
        let mut sum = tch_result(sum.f_unsqueeze(1))?;
        let out = tch_result(sum.f_mul_scalar_(k))?;
        Ok(out)
    }

    fn weighted_label_net(
        motif_nets: u32,
        slices_len: u32,
        weights: Vec<f64>,
        device: Device,
    ) -> (
        Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>,
        Box<dyn Fn() -> ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>> + Sync + Send + 'static>,
    ) {
        (
            Box::new(move || {
                LabelNetParams::new(
                    motif_nets,
                    LabelNetTypeParams::LinNetParams(LinLenNetParams::new(slices_len, 1)),
                )
            }),
            Box::new(move || {
                let weights = weights.clone();
                let key_weights = weights.clone();
                let builder = Arc::new(move |_path: &nn::Path| {
                    let weights = weights.clone();
                    let module = ThoClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        Ok(weighted_label_t(&weights, t)?)
                    })));
                    Ok(Box::new(module) as Box<dyn ThoModule<ChannelsMotifsData, Batch<LabelProbability>>>)
                });
                let mut vs = VarStore::new(device);
                let module = builder(&vs.root()).expect("weighted_label_net builder");
                ThoModuleWithStore::with_factory(
                    SynKey::new([mono_key!(format!("weighted_label_net_{key_weights:?}"), "weighted_label_net")]),
                    module,
                    vs,
                    Some(builder),
                )
            }),
        )
    }

    fn weighted_label_t(weights: &[f64], t: &Tensor) -> Result<Tensor, ThoError> {
        let weights = weights.iter().map(|v| *v as f32).collect::<Vec<_>>();
        let weights = tch_result(tch_result(tch_result(Tensor::from_slice(&weights)
            .f_to_device(t.device()))?
            .f_totype(t.kind()))?
            .f_view([1, weights.len() as i64, 1]))?;
        let weighted = tch_result(t.f_mul(&weights))?;
        let sum = tch_result(weighted.f_sum_dim_intlist(&[1, 2][..], false, None))?;
        Ok(tch_result(sum.f_unsqueeze(1))?)
    }

    #[derive(Clone, Debug)]
    struct TestMotifNetSpec {
        train_run_config_id: RunConfigId,
        trained_net_id: u32,
        motif_id: u32,
        channel: u32,
        value: f64,
        loss: f32,
        path: PathBuf,
    }

    impl TestMotifNetSpec {
        fn new(train_run_config_id: RunConfigId, trained_net_id: u32, motif_id: u32, channel: u32, value: f64, loss: f32) -> Self {
            Self {
                train_run_config_id,
                trained_net_id,
                motif_id,
                channel,
                value,
                loss,
                path: PathBuf::from(format!("{channel}/{motif_id}_{train_run_config_id}")),
            }
        }
    }

    #[derive(Clone, Debug)]
    struct TestLabelNetSpec {
        label: String,
        train_run_config_id: RunConfigId,
        trained_net_id: u32,
        motif_run_config_ids: Vec<RunConfigId>,
        weights: Vec<f64>,
        loss: f32,
        path: PathBuf,
    }

    impl TestLabelNetSpec {
        fn new(label: &str, train_run_config_id: RunConfigId, trained_net_id: u32, motif_run_config_ids: Vec<RunConfigId>, weights: Vec<f64>, loss: f32) -> Self {
            Self {
                label: label.to_string(),
                train_run_config_id,
                trained_net_id,
                motif_run_config_ids,
                weights,
                loss,
                path: PathBuf::from(format!("{}/{train_run_config_id}", label.to_ascii_lowercase())),
            }
        }
    }

    fn motif_nets_for_run_config_ids<'a>(motifs: &'a [TestMotifNetSpec], run_config_ids: &[RunConfigId]) -> Vec<&'a TestMotifNetSpec> {
        let mut selected = motifs
            .iter()
            .filter(|motif| run_config_ids.contains(&motif.train_run_config_id))
            .collect::<Vec<_>>();
        selected.sort_by(|a, b| a.loss.partial_cmp(&b.loss).expect("test motif losses must be comparable").then(a.trained_net_id.cmp(&b.trained_net_id)));
        selected
    }

    fn slices_len(motif_width: u32, step: u32) -> u32 {
        ((motif_width * 3 - motif_width) / step) + 1
    }


    fn expected_label_value(label: &TestLabelNetSpec, motifs: &[TestMotifNetSpec], motif_width: u32, step: u32) -> f32 {
        let motif_nets = motif_nets_for_run_config_ids(motifs, &label.motif_run_config_ids);
        assert_eq!(label.weights.len(), motif_nets.len(), "weights must match selected motif nets for {label:?}");
        let per_slice = motif_nets
            .iter()
            .zip(label.weights.iter())
            .map(|(motif, weight)| {

                let motif_p = motif.value * weight;
                debug!("expected_label_value: motif={motif:?} motif_p={motif_p}");
                motif_p

            })
            .sum::<f64>();
        let slices = slices_len(motif_width, step) as f64;



        let r = (slices * per_slice) as f32;

        debug!("expected_label_value: per_slice={per_slice} slices={slices} r={r}");

        // r
        // sigmoid(r as f64) as f32
        r
    }

    fn assert_values_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_close(*actual as f64, *expected as f64, 0.0001);
        }
    }




    async fn run_apply_label_case(
        labels: &[TestLabelNetSpec],
        motifs: &[TestMotifNetSpec],
        motif_width: u32,
        step: u32,
        channels: u32,
    ) -> Result<(Vec<i64>, Vec<f32>), ThoError> {
        GlobalCtx::init_devices(1)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let find_motifs_run_config_id = 300;
        let device = Device::Cpu;
        let slices_len = slices_len(motif_width, step);

        let motif_ws_entries = motifs
            .iter()
            .map(|motif| {
                let (params, net) = const_motif_net(motif_width, step, motif.value, device);
                (motif.path.clone(), params, net)
            })
            .collect::<Vec<_>>();

        let mut label_ws_entries: HashMap<String, Vec<_>> = HashMap::new();
        let mut single_label_ws_entries = Vec::new();
        for label in labels.iter().rev() {
            let motif_nets = motif_nets_for_run_config_ids(motifs, &label.motif_run_config_ids);
            let (params, net) = weighted_label_net(motif_nets.len() as u32, slices_len, label.weights.clone(), device);
            debug!("run_apply_label_case: label.label={:?} label.train_run_config_id={}", label.label, label.train_run_config_id);
            label_ws_entries.entry(label.label.clone()).or_default().push((label.train_run_config_id, params, net));
            // Register the exact checkpoint path used by ApplyLabelNet::new so each trained net receives its own mock module.
            let (params, net) = weighted_label_net(motif_nets.len() as u32, slices_len, label.weights.clone(), device);
            single_label_ws_entries.push((label.path.clone(), label.train_run_config_id, params, net));
        }


        let ws = Arc::new(RwLock::new(
            crate::motifs_ws::MotifsWsMockBuilder::new(None, channels, 100, motif_width)
                .with_load_motif_net(motif_ws_entries)
                .with_load_label_nets(label_ws_entries)
                .with_load_label_net(single_label_ws_entries)
                .build(),
        ));

        let mut list_trained_nets = labels
            .iter()
            .map(|label| {
                (
                    vec![label.train_run_config_id],
                    None,
                    None,
                    Ok(vec![TrainedNet { id: label.trained_net_id, run_config_id: label.train_run_config_id, net_path: label.path.clone(), loss: label.loss }]),
                )
            })
            .collect::<Vec<_>>();
        let mut seen_motif_run_sets = Vec::<Vec<RunConfigId>>::new();
        for label in labels {
            if seen_motif_run_sets.iter().any(|run_config_ids| *run_config_ids == label.motif_run_config_ids) {
                continue;
            }
            let trained_nets = motif_nets_for_run_config_ids(motifs, &label.motif_run_config_ids)
                .into_iter()
                .map(|motif| TrainedNet { id: motif.trained_net_id, run_config_id: motif.train_run_config_id, net_path: motif.path.clone(), loss: motif.loss })
                .collect::<Vec<_>>();
            list_trained_nets.push((label.motif_run_config_ids.clone(), None, None, Ok(trained_nets)));
            seen_motif_run_sets.push(label.motif_run_config_ids.clone());
        }
        let mut run_config_values = HashMap::from([(
            find_motifs_run_config_id,
            Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), step, 100, None, None).to_properties_iter(None).collect()),
        )]);
        for motif in motifs {
            run_config_values.insert(
                motif.train_run_config_id,
                Ok(TrainMotifNetRunConfig::new(
                    Some(motif.train_run_config_id),
                    CommonTrainingRunConfig::default(),
                    MotifNetTypeArgs::Cosine2DNet(MotifCosine2DNetArgs{
                        a: 10.0,
                        b: 0.0,
                        init_mode: Default::default(),
                        stft_config: Default::default(),
                    }),
                ).to_properties_iter(None).collect()),
            );
        }

        let store = MockMotifsStoreBuilder::new()
                .with_run_config_descr([(
                    find_motifs_run_config_id,
                    Ok(RunConfigRow {
                        id: find_motifs_run_config_id,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find motifs".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_get_motif_meta(
                    motifs
                        .iter()
                        .map(|motif| (motif.motif_id, Ok((motif.channel as usize, find_motifs_run_config_id))))
                        .collect::<HashMap<_, _>>(),
                )
                .with_list_run_config_values(run_config_values)
                .with_list_trained_nets(list_trained_nets)
                .with_list_input_nets(
                    labels
                        .iter()
                        .map(|label| (label.trained_net_id, Ok(label.motif_run_config_ids.clone())))
                        .collect::<HashMap<_, _>>(),
                )
                .build();

        debug!("run_apply_label_case: labels={labels:?}");
        let mut params = FinalNetParams::new(labels.len() as u32, 1, 1, 1);
        params.set_label_feature_schema(labels.iter().map(|label| label.train_run_config_id).collect())?;
        let params: FinalNetParams = serde_json5::from_str(&serde_json5::to_string(&params)?)?;
        let apply_label_nets = Arc::new(ApplyLabelNets::<mock_run_ctx::MockRunCtx, ApplyMotifNets<mock_run_ctx::MockRunCtx>>::from_final_net_params(
            &ws,
            &store,
            &params,
            mock_run_ctx::default(0),
            step,
        ).await?);
        assert_eq!(labels.len(), apply_label_nets.num_nets());

        // Reuse the real pipeline across concurrent and later resources to catch stale stack metadata.
        let mut run_ctx = mock_run_ctx::default(0);
        let stack = Arc::clone(&apply_label_nets.stack_applied_label_nets);
        let make_input = |resource: u32| -> Result<_, ThoError> {
            Ok(Arc::new(DevicesWrap::wrap(ThoTensor::<ChannelsData, SynKey>::new(
                SynKey::new([mono_key!(resource, "resource")]),
                tch_result(Tensor::f_ones([channels as i64, motif_width as i64 * 3], (Kind::Float, device)))?,
            )?)))
        };
        let first = make_input(1u32)?;
        let second = make_input(2u32)?;
        let third = make_input(3u32)?;
        let expected_ids = [first.id(), second.id(), third.id()].map(|id| apply_label_nets.out_id(id));
        let (out, other) = tokio::try_join!(
            apply_label_nets.clone().run(first, run_ctx.branch(false)),
            apply_label_nets.clone().run(second, run_ctx.branch(false)),
        )?;
        let later = apply_label_nets.clone().run(third, run_ctx.branch(false)).await?;
        assert!(Arc::ptr_eq(&stack, &apply_label_nets.stack_applied_label_nets), "resource runs must retain the same stacking stage");
        for (output, expected_id) in [&out, &other, &later].into_iter().zip(expected_ids) {
            assert_eq!(output.id(), &expected_id, "each resource must keep its aggregate output identity");
        }
        let out = SelectRef::select_ref(&out, 0, None::<SyncStftSelector>).await?;
        for other in [&other, &later] {
            let other = SelectRef::select_ref(other, 0, None::<SyncStftSelector>).await?;
            assert!(tch_result(out.as_ref().t_ref().f_allclose(&other.as_ref().t_ref(), 1e-6, 1e-6, false))?, "per-resource output ids must not change label order");
        }
        let out = tch_result(out.as_ref().t_ref().f_squeeze_dim(-1))?;
        Ok((out.size(), to_vec_f32(&out)?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_orders_and_values() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

        let motif_width = 4;
        let step = 1;
        let motifs = vec![
            TestMotifNetSpec::new(10, 1000, 1, 0, logit(0.1)?, 0.1),
            TestMotifNetSpec::new(20, 2000, 2, 0, logit(0.2)?, 0.2),
            // TestMotifNetSpec::new(30, 3000, 3, 0, 2.0, 0.2),
        ];
        let labels = vec![
            TestLabelNetSpec::new("A", 100, 10000, vec![10, 20/*, 30*/], vec![0.1, 0.1/*, 0.1*/], 0.1),
            // TestLabelNetSpec::new("B", 200, 20000, vec![20], vec![0.2, 0.2], 0.1),
        ];

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 3).await?;
        let expected = labels
            .iter()
            .map(|label| expected_label_value(label, &motifs, motif_width, step))
            .collect::<Vec<_>>();

        debug!("test_apply_label_nets_orders_and_values: values={values:?} expected={expected:?}");
        assert_eq!(vec![1, 1], size);
        assert_values_close(&values, &expected);

        Ok(())
    }



    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_handles_three_motif_nets_for_one_label() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

        let motif_width = 4;
        let step = 1;
        let motifs = vec![
            TestMotifNetSpec::new(10, 1000, 0, 0, 1.0, 0.1),
            TestMotifNetSpec::new(20, 1001, 1, 1, 2.0, 0.2),
            TestMotifNetSpec::new(30, 1002, 2, 2, 3.0, 0.3),
        ];
        let labels = vec![
            TestLabelNetSpec::new("A", 100, 2000, vec![10, 20, 30], vec![1.0, 10.0, 100.0], 0.1),
        ];

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 3).await?;
        let expected = vec![expected_label_value(&labels[0], &motifs, motif_width, step)];

        assert_eq!(vec![1, 1], size);
        assert_values_close(&values, &expected);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_handles_more_labels_and_label_nets() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

        let motif_width = 4;
        let step = 1;
        let motifs = vec![
            TestMotifNetSpec::new(10, 1000, 0, 0, 1.0, 0.1),
            TestMotifNetSpec::new(20, 1001, 1, 1, 2.0, 0.2),
            TestMotifNetSpec::new(30, 1002, 2, 2, 3.0, 0.3),
        ];
        let labels = vec![
            TestLabelNetSpec::new("A", 100, 2000, vec![10, 20, 30], vec![1.0, 10.0, 100.0], 0.1),
            TestLabelNetSpec::new("A", 101, 2001, vec![20, 30], vec![2.0, 20.0], 0.2),
            TestLabelNetSpec::new("B", 200, 3000, vec![10], vec![3.0], 0.3),
            TestLabelNetSpec::new("C", 300, 4000, vec![30], vec![4.0], 0.4),
        ];

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 3).await?;
        let expected = labels
            .iter()
            .map(|label| expected_label_value(label, &motifs, motif_width, step))
            .collect::<Vec<_>>();

        assert_eq!(vec![4, 1], size);
        assert_values_close(&values, &expected);

        Ok(())
    }
}

#[cfg(test)]
#[path = "apply_net_state_tests.rs"]
mod state_tests;

#[cfg(test)]
#[path = "stack_applied_label_nets_tests.rs"]
mod stack_tests;
