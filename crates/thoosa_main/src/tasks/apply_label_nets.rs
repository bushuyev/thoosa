use cached_derive::Cached;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::{debug, trace, warn};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use serde::{Deserialize, Serialize};
use tch::Kind::Float;
use tch::{IndexOp, Tensor};
use tch::Device::Cuda;
use thoosa_common::batches::BatchLengths;
use thoosa_common::cuda_utils::split_padded_motif_scores;
use thoosa_common::dims::{Batch, ChannelsMotifsData, LabelProbability, MotifProbability, Slice};
use thoosa_common::my_tch::ThoModule;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::common::DropIf;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, ConsumeData, DevicesWrap, SelectRef, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::oom_handler::{Adjusters, NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler, WithRunCtxContext, run_ctx_adjuster};
use thoosa_common::runner::tmp_cache::TmpCached;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{LabelNetParams, MotifNetParams, ThoModuleWithStore};
use thoosa_common::ws::Workspace;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::comp_id::{SynKey, hashmap_as_vec};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::traits::ConsumeResult;
use thoosa_shared::traits::{Adjustable, AsyncStage, CachableIdTr, Data, ErasedStageWithState, MaxParallelCtx, MonoMetrics, OutId, RunCtx, SimpleId, Stage, StageWithState, SyncStage, WithId};
use thoosa_shared::{CopyToDevice, FuturesOrdering, ThoError, collect_futures_with_progress, mono_key, perf_log, sync_read_lock};
use thoosa_store::{Store, TrainedNet};

use crate::tasks::apply_cosine2d_motif_nets::ApplyCosine2DMotifNets;
use crate::tasks::apply_motif_nets::{ApplyMotifNet, ApplyMotifNets, APPLY_MOTIF_NET_ENDPOINT, STACK_APPLIED_MOTIF_NETS_ENDPOINT, StackAppliedMotifNets, StackAppliedMotifNetsInput};
use crate::tasks::stft::SyncStftSelector;
use thoosa_common::dims::ChannelsData;
use thoosa_common::ipc::codec::{install_ipc_endpoint, IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::cuda_wokrer::{WokerRunCtx, WorkerStageInput};
use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
use thoosa_common::ipc::wraps::{DynSyncStageEnum, DynSyncStageWithStateEnum, ProxyStage, WorkerWrapper, WorkerWrapperWithState};
use thoosa_shared::tracked_await::track_await;
use crate::tasks::load_motif_net::MotifNetMeta;


pub const APPLY_LABEL_NET_ENDPOINT: &str = "APPLY_LABEL_NET_ENDPOINT";

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
    batch_in_chunks: usize,
    pd: PhantomData<C>,
}

impl<C> ApplyLabelNets<C, ApplyCosine2DMotifNets<C>>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    pub async fn new<R>(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        train_label_nets_run_config_ids: Vec<RunConfigId>,
        stage_ctx: R,
        motif_slices_step: u32,
        batch_in_chunks: usize, run_stft: bool,
    ) -> Result<Self, ThoError>
    where
        R: RunCtx + MaxParallelCtx + Adjustable + 'static,
    {
        let motif_ws = Arc::clone(ws);
        let motif_store = Arc::clone(store);

        track_await(Self::new_with_motif_nets_builder(
            ws,
            store,
            train_label_nets_run_config_ids,
            stage_ctx,
            batch_in_chunks,
            move |stage_ctx, motif_nets| {
                let ws = Arc::clone(&motif_ws);
                let store = Arc::clone(&motif_store);

                async move {
                    track_await(ApplyCosine2DMotifNets::from_trained_nets(
                        &ws,
                        &store,
                        stage_ctx,
                        motif_slices_step,
                        None,
                        motif_nets,
                        false,
                        run_stft,
                    )).await
                }
            },
        )).await
    }
}

impl<C> ApplyLabelNets<C, ApplyMotifNets<C>>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
{
    pub async fn new_with_apply_motif_nets<R>(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        train_label_nets_run_config_ids: Vec<RunConfigId>,
        stage_ctx: R,
        motif_slices_step: u32,
        batch_in_chunks: usize,
    ) -> Result<Self, ThoError>
    where
        R: RunCtx + MaxParallelCtx + Adjustable + 'static,
    {
        let motif_ws = Arc::clone(ws);
        let motif_store = Arc::clone(store);

        track_await(Self::new_with_motif_nets_builder(
            ws,
            store,
            train_label_nets_run_config_ids,
            stage_ctx,
            batch_in_chunks,
            move |stage_ctx, motif_nets| {
                let ws = Arc::clone(&motif_ws);
                let store = Arc::clone(&motif_store);

                async move {
                    track_await(ApplyMotifNets::from_trained_nets(
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
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        train_label_nets_run_config_ids: Vec<RunConfigId>,
        mut stage_ctx: R,
        batch_in_chunks: usize,
        mut build_motif_nets: F,
    ) -> Result<Self, ThoError>
    where
        R: RunCtx + Adjustable + 'static,
        F: FnMut(R, Vec<TrainedNet>) -> Fut,
        Fut: Future<Output = Result<AMN, ThoError>>,
    {
        let tho_device_idx = stage_ctx.tho_device_idx();


        let all_label_nets = perf_log!(sync_read_lock!(ws)).load_label_nets(Some(train_label_nets_run_config_ids.as_slice()), &ThoDevice::from(tho_device_idx).into(), false)?;

        let mut apply_nets = vec![];

        debug!("ApplyLabelNets::new: all_label_nets={all_label_nets:?}");

        for (label, nets) in all_label_nets {

            debug!("ApplyLabelNets::new: label={label:?} nets.len={} nets={nets:?}", nets.len());

            for (train_label_net_run_config_id, label_net_params, net) in nets {
                let label_nets = track_await(store.list_trained_nets(vec![train_label_net_run_config_id], None, None)).await?;

                debug!("ApplyLabelNets::new: train_label_net_run_config_id={train_label_net_run_config_id} label_nets={label_nets:?}");

                assert_eq!(label_nets.len(), 1); //for now there is always single net for run_config_id
                let label_net = &label_nets[0];

                let motif_net_run_config_ids = track_await(store.list_input_nets(label_net.id)).await?;

                debug!("ApplyLabelNets::new: label_net_id={}  motif_net_ids={motif_net_run_config_ids:?}", label_net.id);

                if motif_net_run_config_ids.is_empty() {
                    warn!("no motifs for train_label_net_run_config_id={train_label_net_run_config_id}");
                    continue//TODO, change to assert, just for tests
                }

                let motif_nets = track_await(store.list_trained_nets(motif_net_run_config_ids, None, None)).await?;

                debug!("ApplyLabelNets::new: train_label_net_run_config_id={train_label_net_run_config_id} motif_nets={motif_nets:?}");

                apply_nets.push(ApplyNets {
                    apply_motif_nets: Arc::new(track_await(build_motif_nets(stage_ctx.branch(false), motif_nets)).await?),
                    apply_label_net: Arc::new(track_await(ApplyLabelNet::new(label.clone(), train_label_net_run_config_id, label_net, 1.into(), batch_in_chunks, ws, false)).await?),
                });
            }
        }


        Ok(Self {
            id: mono_key!(0, "todo").into(),
            apply_nets,
            batch_in_chunks,
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

        #[cfg(debug_assertions)]
        let input_debug = input.clone();

        debug!("ApplyLabelNets::run: self.apply_nets.len={} label_nets={:?}", self.apply_nets.len(), self.apply_nets);

        let tmp_cache = track_await(TmpCached::new("tmp_stack_applied_label_nets")).await;
        let tmp_cache_name = tmp_cache.registered_name();
        let tmp_cache: Arc<Mutex<TmpCached<ThoTensor<LabelProbability, SynKey>>>> = Arc::new(Mutex::new(tmp_cache));
        let label_output_order = Arc::new(Mutex::new(vec![None; self.apply_nets.len()]));

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
                            let input = SelectRef::borrow_to(&input, ctx.tho_device_idx(), None::<SyncStftSelector>).await?;
                            trace!("ApplyLabelNets::run: input=\n{}", input.as_ref().as_ref());
                            Ok::<(), ThoError>(())
                        }?;
                    }

                    //gpt First apply motif nets to the raw input to produce motif probabilities used by label nets.
                    let motif_probs: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>>> = apply_motif_nets
                        .run(input, ctx.branch(false))
                        .await?;

                    debug!("ApplyLabelNets::run: motif_probs={motif_probs:?}");

                    #[cfg(test)]{
                        {
                            let probs = SelectRef::borrow_to(&motif_probs, ctx.tho_device_idx(), None::<SyncStftSelector>).await?;
                            trace!("ApplyLabelNets::run: probs=\n{}", probs.as_ref().data().as_ref());
                            Ok::<(), ThoError>(())
                        }?;
                    }

                    //gpt Each label net runs independently; ordering is preserved by buffer mode below - (relevant for final net training, if final inference done per label it doesn't matter).
                    let input = Arc::new(ApplyLabelNetInputWrapper::new(motif_probs));
                    let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()> = OOMHandler::new(vec![], None);
                    let result: ThoTensor<LabelProbability, SynKey> = track_await(oom_handler.run(&apply_label_net, input, ctx.branch(false), NoContexts)).await?;

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
            "ApplyLabelNets::run",
        )
        .await
        .into_iter()
        .collect::<Vec<_>>();

        debug!("ApplyLabelNets::run: label_outputs stacked label_output_order={label_output_order:?} count={}", self.apply_nets.len());

        let label_output_order = Arc::into_inner(label_output_order)
            .expect("ApplyLabelNets::run: label_output_order still shared")
            .into_inner()
            .into_iter()
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| ThoError::new("ApplyLabelNets::run: missing label output id"))?;
        let stack_applied_label_nets = Arc::new(StackAppliedLabelNets::<C>::new(self.id.clone(), label_output_order, 1.into()).await?);
        let stack_input = StackAppliedLabelNetsInputWrapper::new(input_id.clone(), Arc::into_inner(tmp_cache).expect("ApplyLabelNets::run: tmp_cache still shared").into_inner());
        let required_tmp_cache_ids = HashSet::<&dyn CachableIdTr>::new();
        let required_ids = HashMap::from([(tmp_cache_name, &required_tmp_cache_ids)]);
        let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, _, ()> = OOMHandler::new(
            vec![
                Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("stack_apply_label_nets"))),
            ],
            None,
        );
        let label_outputs = track_await(oom_handler.consume(
            &stack_applied_label_nets,
            stack_input,
            ctx.branch(true),
            WithRunCtxContext::new(
                MaxParallelAndDumpToCpuAdjustableContex::new(
                    MaxParallelAdjustContextImpl::new(),
                    DumpToCpuContextImpl::new(&required_ids),
                ),
            ),
        )).await?;

        #[cfg(debug_assertions)]
        debug!("ApplyLabelNets::run: done, input={input_debug:?}");

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
    id: SynKey,
    device_idx: usize,
    label_outputs: Vec<ThoTensor<LabelProbability, SynKey>>,
}

impl StackAppliedLabelNetsInput {
    fn new(id: SynKey, label_outputs: Vec<ThoTensor<LabelProbability, SynKey>>, device_idx: usize) -> Self {
        Self { id, device_idx, label_outputs }
    }
}

impl WorkerStageInput for StackAppliedLabelNetsInput {
    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        let input = C::decode::<StackAppliedLabelNetsInput>(input)?;
        Ok(StackAppliedLabelNetsInput{
            id: input.id,
            device_idx: input.device_idx,
            label_outputs: input.label_outputs.into_iter().map(|v|ThoTensor::<LabelProbability, SynKey>::ref_to_tensor(all_cuda_tensors, v)).collect::<Result<_, ThoError>>()?,
        })
    }
}

impl WithId for StackAppliedLabelNetsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for StackAppliedLabelNetsInput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.label_outputs.len().into()
    }

    fn device(&self) -> usize {
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
    tmp_cache: TmpCached<ThoTensor<LabelProbability, SynKey>>,
}

impl StackAppliedLabelNetsInputWrapper {
    fn new(id: SynKey, tmp_cache: TmpCached<ThoTensor<LabelProbability, SynKey>>) -> Self {
        Self { id, tmp_cache }
    }
}

impl WithId for StackAppliedLabelNetsInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl ConsumeData for StackAppliedLabelNetsInputWrapper {
    type O = StackAppliedLabelNetsInput;

    fn consume<R: Send, F, P>(
        self,
        to_device_idx: usize,
        ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl Future<Output = ConsumeResult<Self::O, R>> + Send
    where
        F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
        P: RunCtx,
    {
        async move {
            let StackAppliedLabelNetsInputWrapper { id, tmp_cache } = self;

            match track_await(tmp_cache.take_all_recovering(to_device_idx)).await {
                Ok(label_outputs) => do_run(StackAppliedLabelNetsInput::new(id, label_outputs, to_device_idx), ctx),
                Err(err) => ConsumeResult::Failed { input: StackAppliedLabelNetsInput::new(id, err.data, to_device_idx), error: err.error },
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="StackAppliedLabelNetsInputWrapper::burp"))]
    fn burp(o: Self::O) -> impl Future<Output = Self> + Send {
        async move {
            let cache = track_await(TmpCached::new("tmp_stack_applied_label_nets_burp")).await;
            for label_output in o.label_outputs {
                cache.insert(label_output);
            }
            StackAppliedLabelNetsInputWrapper::new(o.id, cache)
        }
    }
}

pub const STACK_APPLIED_LABEL_NETS_ENDPOINT: &str = "STACK_APPLIED_LABEL_NETS_ENDPOINT";

#[derive(Debug, Serialize, Deserialize)]
pub struct StackAppliedLabelNetsInit {
    id: SynKey,

    #[serde(with = "hashmap_as_vec")]
    label_output_order: HashMap<SynKey, usize>,
}

impl StackAppliedLabelNetsInit {
    pub fn new(id: SynKey, label_output_order: HashMap<SynKey, usize>) -> Self {
        Self { id, label_output_order }
    }
}

#[derive(Debug, Named)]
pub struct StackAppliedLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    label_output_order: HashMap<SynKey, usize>,
    pd: PhantomData<C>,
}

impl<C> StackAppliedLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    async fn new(id: SynKey, label_output_order: Vec<SynKey>, device: ThoDevice) -> Result<DynSyncStageEnum<StackAppliedLabelNets<C>>, ThoError> {
        let order = label_output_order.into_iter().enumerate().map(|(idx, id)| (id, idx)).collect();

        if GlobalCtx::global().is_ipc() {
            static STAGE_ID:AtomicU32 = AtomicU32::new(0);//TODO
            let stage_id = STAGE_ID.fetch_add(1, Relaxed);
            install_ipc_endpoint::<JsonIpcCodec, _>(GlobalCtx::global().cuda_actor(device.tho_device_idx()).as_ref(), STACK_APPLIED_LABEL_NETS_ENDPOINT, stage_id, &StackAppliedLabelNetsInit::new(id, order)).await?;
            Ok(DynSyncStageEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy(STACK_APPLIED_LABEL_NETS_ENDPOINT, stage_id, device.tho_device_idx())),
            })
        } else {
            Ok(DynSyncStageEnum::Local(Arc::new(Self::new_local(StackAppliedLabelNetsInit::new(id, order)).await?)))
        }
    }

    pub async fn new_local(init:StackAppliedLabelNetsInit) -> Result<Self, ThoError>{
        Ok(Self {
            id:init.id,
            label_output_order:init.label_output_order,
            pd: PhantomData,
        })

    }

}

pub fn stack_applied_label_nets(init_msg: Vec<u8>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init = JsonIpcCodec::decode(&init_msg)?;
        Ok(Box::new(WorkerWrapper::new(Arc::new(StackAppliedLabelNets::<WokerRunCtx>::new_local(init).await?), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}


impl<C> Adjustable for StackAppliedLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
}

impl<C> OutId for StackAppliedLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for StackAppliedLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = StackAppliedLabelNetsInput;
    type Out = ThoTensor<LabelProbability, SynKey>;
}

impl<C> SyncStage for StackAppliedLabelNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="StackAppliedLabelNets::consume"))]
    fn consume<'f>(self: Arc<Self>, mut input: Self::In<'f>, _ctx: Self::Ctx) -> ConsumeResult<Self::In<'f>, Self::Out> {
        let input_id = input.id().clone();
        let mut label_outputs = std::mem::take(&mut input.label_outputs);
        label_outputs.sort_by_key(|label_output| self.label_output_order.get(label_output.id()).copied().unwrap_or(usize::MAX));

        let out_id = self.out_id(&input_id);
        let tensor = if label_outputs.is_empty() {
            match Tensor::f_empty([0], (Float, ThoDevice::from(input.device_idx).into())) {
                Ok(tensor) => tensor,
                Err(error) => {
                    input.label_outputs = label_outputs;
                    return ConsumeResult::Failed { input, error: error.into() };
                }
            }
        } else {
            let stacked = label_outputs.iter().map(|label_output| {
                debug!("StackAppliedLabelNets::consume: label_output.size={:?}", label_output.t_ref().size());
                label_output.t_ref().f_sigmoid()
            }).collect::<Result<Vec<_>, _>>();
            match stacked.and_then(|stacked| Tensor::f_stack(stacked.as_slice(), 0)) {
                Ok(tensor) => tensor,
                Err(error) => {
                    input.label_outputs = label_outputs;
                    return ConsumeResult::Failed { input, error: error.into() };
                }
            }
        };

        match ThoTensor::new(out_id, tensor) {
            Ok(tensor) => ConsumeResult::Ok(tensor),
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
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.input.as_ref().size()
    }

    fn device(&self) -> usize {
        self.input.as_ref().device()
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
    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        let input = C::decode::<Self>(input)?;
        let ThoDataRef::Owned(input_ref) = input.input else {
            panic!("unexpected ref: input.batch={:?}", input.input);
        };

        let DataWithExtra{data, extra} = input_ref;

        Ok(Self{
            id: input.id,
            input: ThoDataRef::Owned(DataWithExtra::new(
                ThoTensor::ref_to_tensor(all_cuda_tensors, data)?,
                Arc::into_inner(extra).unwrap()//TODO
            ))
        })
    }
}

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

impl WithId for ApplyLabelNetInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl BorrowRef for ApplyLabelNetInputWrapper {
    type Out<'a> = ApplyLabelNetInput<'a> where Self: 'a;

    fn borrow_to<'a>(&'a self, to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let input = track_await(BorrowRef::borrow_to(&self.input, to_device_idx)).await?;
            Ok(ApplyLabelNetInput { id: self.id.clone(), input })
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ApplyLabelNetInit {
    label: String,
    train_label_net_run_config_id: RunConfigId,
    label_net_path:PathBuf,
    batch_in_chunks: usize,
    quantize:bool
}

impl ApplyLabelNetInit {
    pub fn new(label: String, train_label_net_run_config_id: RunConfigId, label_net_path:PathBuf, batch_in_chunks: usize, quantize:bool) -> Self {
        Self { label, train_label_net_run_config_id, label_net_path, batch_in_chunks, quantize }
    }
}

impl WorkerStageInput for ApplyLabelNetInit {
    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        C::decode(input)
    }
}


#[derive(Debug)]
struct ApplyLabelNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    label: String,
    train_label_net_run_config_id: RunConfigId,
    // Inference state is acquired once per sync run instead of being copied in every input.
    net: tokio::sync::Mutex<ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>>,
    pd: PhantomData<C>,
    batch_in_chunks: usize
}

impl<C> ApplyLabelNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    async fn new(
        label: String,
        train_label_net_run_config_id: RunConfigId,
        trained_net:&TrainedNet,
        device: ThoDevice,
        batch_in_chunks: usize,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        quantize: bool,
    ) -> Result<DynSyncStageWithStateEnum<ApplyLabelNet<C>, ApplyLabelNetInit>, ThoError> {
        let (run_config_id, params, net) = perf_log!(sync_read_lock!(ws)).load_label_net(
               &trained_net.net_path,
               &device,
               quantize
        )?;

        let init = ApplyLabelNetInit::new(
            label.clone(),
            train_label_net_run_config_id,
            trained_net.net_path.clone(),
            batch_in_chunks,
            quantize
        );

        if GlobalCtx::global().is_ipc() {
            static STAGE_ID:AtomicU32 = AtomicU32::new(0);//TODO
            let stage_id = STAGE_ID.fetch_add(1, Relaxed);
            thoosa_common::ipc::codec::install_ipc_endpoint::<JsonIpcCodec, ApplyLabelNetInit>(GlobalCtx::global().cuda_actor(device.tho_device_idx()).as_ref(), APPLY_LABEL_NET_ENDPOINT, stage_id, &init).await?;
            Ok(DynSyncStageWithStateEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy(APPLY_LABEL_NET_ENDPOINT, stage_id, device.tho_device_idx())),
                init,
            })
        } else {
            Ok(
                DynSyncStageWithStateEnum::Local(Arc::new(ApplyLabelNet::new_local(
                    label,
                    train_label_net_run_config_id,
                    net,
                    batch_in_chunks
                ).await?))
            )
        }
    }

    async fn new_local(
        label: String,
        train_label_net_run_config_id: RunConfigId,
        net: ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>,
        batch_in_chunks: usize,
    ) -> Result<Self, ThoError> {
        Ok(Self {
            id: mono_key!(label.clone(), "ApplyLabelNet") + net.id(),
            label,
            train_label_net_run_config_id,
            net: tokio::sync::Mutex::new(net),
            batch_in_chunks,
            pd: PhantomData,
        })
    }
}

pub fn apply_label_net_endpoint_init(init_msg: Vec<u8>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init = ApplyLabelNetInit::decode_for_worker::<JsonIpcCodec>(&init_msg, all_cuda_tensors)?;
        debug!("apply_net_stage_2_endpoint_init: init={init:?}");

        let (run_config_id, params, net) = perf_log!(sync_read_lock!(ws)).load_label_net( //TODO in IPC mode this is needed only to get motif_width from  params. Add util to load params without module
               &init.label_net_path,
               &Cuda(0),//TODO
               init.quantize
        )?;

        let stage: ApplyLabelNet<WokerRunCtx> = ApplyLabelNet::new_local(
           init.label,
           init.train_label_net_run_config_id,
           net,
           init.batch_in_chunks
        ).await?;

        // The common stateful wrapper performs state preparation and converts worker outputs back to references.
        Ok(Box::new(WorkerWrapperWithState::new(Arc::new(stage), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

impl<C> Adjustable for ApplyLabelNet<C>
where
    C: RunCtx + Adjustable,
{
}

impl<C> Stage for ApplyLabelNet<C>
where
    C: RunCtx + Adjustable,
{
    type Ctx = C;
    type In<'a> = ApplyLabelNetInput<'a>;
    type Out = ThoTensor<LabelProbability, SynKey>;//TODO it's Batch<LabelProbability>
}

impl<C> SyncStage for ApplyLabelNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        Some(self)
    }
}

impl<C> StageWithState for ApplyLabelNet<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type S = ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>;

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<tokio::sync::MutexGuard<'_, Self::S>, ThoError>> {
        async move {
            let mut net = self.net.lock().await;
            if net.device() != to_device_idx {
                *net = net.copy_to_device(to_device_idx).await?;
            }
            Ok(net)
        }.boxed()
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyLabelNet::do_run"))]
    fn do_run_with_state<'a>(self: Arc<Self>, input: &Self::In<'a>, net: tokio::sync::MutexGuard<'_, Self::S>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("ApplyLabelNet::run: id={:?} input={input:?}", self.id);
        let out_id = self.out_id(input.id());
        let input = input.input.as_ref();
        let batch_input_parts = split_padded_motif_scores(input.data().t_ref(), input.extra(), self.batch_in_chunks as i64, 0.0)?;
        let batch_input = Tensor::f_stack(batch_input_parts.0.as_slice(), 1)?.f_transpose(0, 1)?;
        let batch_input: ThoTensor<ChannelsMotifsData, SimpleId> = batch_input.into();
        let mut lengths = BatchLengths::new(batch_input_parts.1.iter().map(|value| value[0]).collect());
        let out = net.tho_forward(&batch_input, Some(&mut lengths), false)?;
        ThoTensor::new(out_id, out.t())
    }
}

impl<C> OutId for ApplyLabelNet<C>
where
    C: RunCtx + Adjustable,
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
    use crate::tasks::apply_label_nets::{ApplyLabelNets, logit, sigmoid};
    use crate::tasks::apply_motif_nets::ApplyMotifNets;
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use crate::tasks::stft::SyncStftSelector;
    use log::LevelFilter::{Trace, Warn};
    use log::{debug, set_max_level, trace};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use tch::Kind::Float;
    use tch::nn::{self, VarStore};
    use tch::{Device, Kind, Tensor};
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::dims::ChannelsData;
    use thoosa_common::dims::{Batch, ChannelsMotifsData, LabelProbability, MotifProbability, Slice};
    use thoosa_common::my_tch::{MyClosureModule, ThoModule};
    use thoosa_common::runner::common::cachable::{DevicesWrap, SelectRef};
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::global_ctx::{GlobalCtx, mock_global_ctx};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{CommonTrainingRunConfig, LabelNetParams, LabelNetTypeParams, LinLenNetParams, MotifCosine2DNetArgs, MotifNetParams, MotifNetTypeArgs, ThoModuleWithStore, TrainMotifNetRunConfig};
    use thoosa_common::utils::{assert_close, to_vec_f32};
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::traits::AsyncStage;
    use thoosa_shared::traits::RunCtx;
    use thoosa_shared::{ThoError, mono_key};
    use thoosa_store::{MockStoreBuilder, RunConfigRow, TrainedNet};

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
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
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
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
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
        let sum = t.f_sum_dim_intlist(&[1, 2][..], false, None)?;
        let mut sum = sum.f_unsqueeze(1)?;
        let out = sum.f_mul_scalar_(k)?;
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
                    let module = MyClosureModule::new(Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        weighted_label_t(&weights, t)
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
        let weights = Tensor::from_slice(&weights)
            .f_to_device(t.device())?
            .f_totype(t.kind())?
            .f_view([1, weights.len() as i64, 1])?;
        let weighted = t.f_mul(&weights)?;
        let sum = weighted.f_sum_dim_intlist(&[1, 2][..], false, None)?;
        Ok(sum.f_unsqueeze(1)?)
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

                let motif_p = sigmoid(motif.value) * weight;
                debug!("expected_label_value: motif={motif:?} motif_p={motif_p}");
                motif_p

            })
            .sum::<f64>();
        let slices = slices_len(motif_width, step) as f64;



        let r = (slices * per_slice) as f32;

        debug!("expected_label_value: per_slice={per_slice} slices={slices} r={r}");

        sigmoid(r as f64) as f32
    }

    fn assert_values_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_close(*actual as f64, *expected as f64, 0.0001);
        }
    }

    fn sorted_values(mut values: Vec<f32>) -> Vec<f32> {
        values.sort_by(|a, b| a.partial_cmp(b).expect("test values must be comparable"));
        values
    }

    async fn run_apply_label_case(
        labels: &[TestLabelNetSpec],
        motifs: &[TestMotifNetSpec],
        motif_width: u32,
        step: u32,
        channels: u32,
        batch_chunks: usize,
    ) -> Result<(Vec<i64>, Vec<f32>), ThoError> {
        GlobalCtx::init_devices(1);
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
        for label in labels {
            let motif_nets = motif_nets_for_run_config_ids(motifs, &label.motif_run_config_ids);
            let (params, net) = weighted_label_net(motif_nets.len() as u32, slices_len, label.weights.clone(), device);
                debug!("run_apply_label_case: label.label={:?} label.train_run_config_id={}", label.label, label.train_run_config_id);
            label_ws_entries.entry(label.label.clone()).or_default().push((label.train_run_config_id, params, net));
        }


        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, channels, 100, motif_width)
                .with_load_motif_net(motif_ws_entries)
                .with_load_label_nets(label_ws_entries)
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

        let store = Arc::new(
            MockStoreBuilder::new()
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
                .build(),
        );

        debug!("run_apply_label_case: labels={labels:?}");
        let apply_label_nets = Arc::new(ApplyLabelNets::<mock_run_ctx::MockRunCtx, ApplyMotifNets<mock_run_ctx::MockRunCtx>>::new_with_apply_motif_nets(
            &ws,
            &store,
            labels.iter().map(|label| label.train_run_config_id).collect(),
            mock_run_ctx::default(0),
            step,
            batch_chunks,
        ).await?);
        assert_eq!(labels.len(), apply_label_nets.num_nets());

        let input = ThoTensor::<ChannelsData, SynKey>::new(
            SynKey::new([mono_key!("input", "input")]),
            Tensor::f_ones(&[channels as i64, (motif_width as i64) * 3], (Kind::Float, device))?,
        )?;

        let mut run_ctx = mock_run_ctx::default(0);
        let out = apply_label_nets.run(Arc::new(DevicesWrap::wrap(input)), run_ctx.branch(false)).await?;


        {
            let out = SelectRef::borrow_to(&out, run_ctx.tho_device_idx(), None::<SyncStftSelector>).await?;
            debug!("t=\n{}", out.t_ref());
            let out = out.as_ref().as_ref().f_squeeze_dim(-1)?;
            Ok::<_, ThoError>((out.size(), to_vec_f32(&out.totype(Float))))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_orders_and_values() -> Result<(), ThoError> {
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

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 3, 1).await?;
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
    async fn test_apply_label_nets_orders_and_values_two_chunks() -> Result<(), ThoError> {
        set_max_level(Trace);
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

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 1, 2).await?;
        let expected = labels
            .iter()
            .map(|label| expected_label_value(label, &motifs, motif_width, step))
            .collect::<Vec<_>>();

        debug!("test_apply_label_nets_orders_and_values: values={values:?} expected={expected:?}");
        /*
        motif probabilities are split in padded chunks of equal size:
        [
          [//chunk 0 - sum  (label net) 0.12
               [0.1000, 0.1000, 0.1000, 0.1000, 0.0000],
               [0.2000, 0.2000, 0.2000, 0.2000, 0.0000]
          ],

          [//chunk 1 - sum 0.15
              [0.1000, 0.1000, 0.1000, 0.1000, 0.1000],
              [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]
          ]
        ]
       */
        assert_eq!(vec![1, 2], size);
        assert_values_close(&values, &[sigmoid(0.12) as f32, sigmoid(0.15) as f32]);

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_handles_three_motif_nets_for_one_label() -> Result<(), ThoError> {
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

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 3, 1).await?;
        let expected = vec![expected_label_value(&labels[0], &motifs, motif_width, step)];

        assert_eq!(vec![1, 1], size);
        assert_values_close(&values, &expected);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_apply_label_nets_handles_more_labels_and_label_nets() -> Result<(), ThoError> {
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

        let (size, values) = run_apply_label_case(&labels, &motifs, motif_width, step, 3, 1).await?;
        let expected = labels
            .iter()
            .map(|label| expected_label_value(label, &motifs, motif_width, step))
            .collect::<Vec<_>>();

        assert_eq!(vec![4, 1], size);
        assert_values_close(&sorted_values(values), &sorted_values(expected));

        Ok(())
    }
}
