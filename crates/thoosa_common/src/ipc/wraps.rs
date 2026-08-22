use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use log::{debug, error};
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tch::Tensor;
use tokio::sync::MutexGuard;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Adjustable, CachableIdTr, ConsumeResult, ErasedStageWithState, MaxParallelCtx, RunCtx, Stage, StageWithState, SyncStage};
use crate::ipc::codec::{IpcCodec, IpcSyncStageRunInput, JsonIpcCodec};
use crate::ipc::cuda_process::{encode_endpoint_payload_bytes, CudaIPCActorHandle};
use crate::ipc::cuda_wokrer::{WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::json_ser::{ErasedWithState, SyncStageIpcConsumeResponse, SyncStageIpcResponse, SyncStageIpcWrapper, SyncStageIpcWrapperWithState};
use crate::ipc::worker_msg::{CudaWorkerMsgIn, Operation, WorkerInput, WorkerOutput};
use crate::runner::common::cachable::ThoDataRef;
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::purge::FuturesRunnerMapPurger;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;

#[derive(Debug)]
pub struct ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    pd:PhantomData<St>,
    endpoint_name:&'static str,
    // Each proxy keeps the id used at installation so invocations reach its own worker-side state.
    stage_id: u32,
    // Drop has no run context, so retain the worker index that owns the installed endpoint.
    tho_device_idx: usize,

}

impl <St> Stage for ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}



impl <St>SyncStage for ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ProxyStage::do_run"))]
    fn do_run<'a>(self: std::sync::Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, thoosa_shared::ThoError>
    where
        Self: Sized,
    {

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                // let endpoint_name = crate::runner::test::proxy_tests::TEST_ENDPOINT;
                let actor = thoosa_common::runner::global_ctx::GlobalCtx::global().cuda_actor(thoosa_shared::traits::RunCtx::tho_device_idx(ctx)).as_ref();
                let input = thoosa_common::ipc::codec::JsonIpcCodec::encode(input)?;
                let input = IpcSyncStageRunInput::new(input);
                let input = JsonIpcCodec::encode(&input)?;
                tracing::debug!("invoke_ipc_sync_stage: endpoint_name={} stage_id={} 1. input={input:?}", self.endpoint_name, self.stage_id);


                let response = actor.invoke_endpoint_bytes(
                    self.stage_id,
                    encode_endpoint_payload_bytes(self.endpoint_name, &input).into_bytes()
                ).await?;

                let r = thoosa_common::ipc::codec::JsonIpcCodec::decode(&response);
                tracing::debug!("invoke_ipc_sync_stage: endpoint_name={} stage_id={} response={r:?}", self.endpoint_name, self.stage_id);

                r
            })
        })
    }


    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ProxyStage::consume"))]
    fn consume<'a>(self: Arc<Self>, input: Self::In<'a>, ctx: Self::Ctx) -> ConsumeResult<Self::In<'a>, Self::Out>
    where
        Self: Sized,
    {

        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {


                let encoded = JsonIpcCodec::encode(&input)?;
                let request = IpcSyncStageRunInput::new(encoded);
                let encoded = JsonIpcCodec::encode(&request)?;
                let actor = GlobalCtx::global().cuda_actor(ctx.tho_device_idx()).as_ref();
                let response = actor.consume_endpoint_bytes(
                    self.stage_id,
                    encode_endpoint_payload_bytes(self.endpoint_name, &encoded).into_bytes(),
                ).await?;
                JsonIpcCodec::decode(&response)
            })
        });
        match result {
            Ok(output) => ConsumeResult::Ok(output),
            Err(error) => ConsumeResult::Failed { input, error },
        }
    }
}

impl <St> ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    pub fn proxy(endpoint_name:&'static str, stage_id: u32, tho_device_idx: usize) -> Self {
        Self {
            endpoint_name,
            stage_id,
            tho_device_idx,
            pd:PhantomData
        }
    }

}

impl <St> Drop for ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    fn drop(&mut self) {
        let endpoint_name = self.endpoint_name;
        let stage_id = self.stage_id;
        let tho_device_idx = self.tho_device_idx;

        debug!("ProxyStage::drop: endpoint_name={endpoint_name} stage_id={stage_id} tho_device_idx={tho_device_idx}");

        tokio::spawn(async move {
            if let Err(error) = GlobalCtx::global().cuda_actor(tho_device_idx).drop_endpoint(stage_id, endpoint_name.to_string()).await {
                error!("ProxyStage::drop: failed to drop endpoint_name={endpoint_name} stage_id={stage_id} tho_device_idx={tho_device_idx}: {error:?}");
            }
        });
    }
}


///
#[derive(Debug)]
pub enum DynSyncStageEnum<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    Local(Arc<St>),
    Proxy {
        stage: Arc<ProxyStage<St>>,
    }

}

impl <St> Stage for DynSyncStageEnum<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}

impl<St> Adjustable for DynSyncStageEnum<St>
where
    St: SyncStage + Adjustable,
    for<'a> <St as Stage>::In<'a>: WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned,
{
    // Preserve the local stage's adjustment marker across local/proxy dispatch.
}

impl <St> SyncStage for DynSyncStageEnum<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(input, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, &ctx),//TODO TestRunCtx
        }
    }


    fn consume<'a>(self: Arc<Self>, input: Self::In<'a>, ctx: Self::Ctx) -> ConsumeResult<Self::In<'a>, Self::Out>
    where
        Self: Sized,
    {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).consume(input, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).consume(input, ctx),
        }
    }


}

impl <St> StageWithState for DynSyncStageEnum<St>
where
    St:SyncStage+StageWithState+Send+Sync,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned,
{
    type S = <St as StageWithState>::S;


    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
        match self {
            Self::Local(stage) => StageWithState::pre_run(stage.as_ref(), to_device_idx),
            Self::Proxy { .. } => async {
                Err(ThoError::new("proxy ApplyStage2 does not have local state"))
            }.boxed(),
        }
    }

    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, state: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => StageWithState::do_run_with_state(Arc::clone(stage), input, state, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, ctx),
        }
    }
}

///
#[derive(Debug)]
pub enum DynSyncStageWithStateEnum<St, Init=()>
where
    St:SyncStage+StageWithState,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    Local(Arc<St>),
    Proxy {
        stage: Arc<ProxyStage<St>>,
        init: Init
    }
}

impl <St, Init> Stage for DynSyncStageWithStateEnum<St, Init>
where
    St:SyncStage+StageWithState,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}

impl <St, Init> Adjustable for DynSyncStageWithStateEnum<St, Init>
where
    St: SyncStage + StageWithState + Adjustable,
    for <'a> <St as Stage>::In<'a>: WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    // Preserve the wrapped stage's adjustment capability across local/proxy dispatch.
}

impl <St, Init> SyncStage for DynSyncStageWithStateEnum<St, Init>
where
    St: SyncStage + StageWithState + Send + Sync,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(input, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, &ctx),//TODO TestRunCtx
        }
    }

    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        // Advertise typed state execution without exposing ThoTrainModule to the common runner.
        match self {
            Self::Local(_) => Some(self),
            Self::Proxy { .. } => None,
        }
    }
}


///
impl <St, Init> StageWithState for DynSyncStageWithStateEnum<St, Init>
where
    St:SyncStage+StageWithState+Send+Sync,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    type S = <St as StageWithState>::S;


    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
        match self {
            Self::Local(stage) => StageWithState::pre_run(stage.as_ref(), to_device_idx),
            Self::Proxy { .. } => async {
                Err(ThoError::new("proxy ApplyStage2 does not have local state"))
            }.boxed(),
        }
    }

    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, state: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => StageWithState::do_run_with_state(Arc::clone(stage), input, state, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, ctx),
        }
    }
}
///
///
#[async_trait]
impl <St, Init> FuturesRunnerMapPurger for DynSyncStageWithStateEnum<St, Init>
where
    St:SyncStage+StageWithState + FuturesRunnerMapPurger,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        match self {
            DynSyncStageWithStateEnum::Local(stage) => {
                stage.has_gpu_data(device_idx).await
            }
            DynSyncStageWithStateEnum::Proxy { .. } => {
                true
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            DynSyncStageWithStateEnum::Local(local) => {
                local.purge_gpu(ids_to_keep, device_idx, name).await
            }
            DynSyncStageWithStateEnum::Proxy { stage,  .. } => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        thoosa_common::ipc::codec::purge_ipc_sync_stage::<thoosa_common::ipc::codec::JsonIpcCodec>(
                            GlobalCtx::global().cuda_actor(device_idx).as_ref(),
                            stage.endpoint_name,
                            stage.stage_id,
                        ).await
                    })
                })?;
                Ok(())
            }
        }
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            DynSyncStageWithStateEnum::Local(local) => {
                local.drop_gpu(device_idx, name).await
            }
            DynSyncStageWithStateEnum::Proxy { .. } => {
                Ok(())
            }
        }
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        match self {
            DynSyncStageWithStateEnum::Local(local) => {
                local.purge_cpu(purge_until).await
            }
            DynSyncStageWithStateEnum::Proxy { .. } => {
                0
            }
        }
    }
}
///

pub trait WorkerWrapperCtx: RunCtx {
    fn from_worker_ctx(ctx: &WokerRunCtx) -> Self;
}

impl WorkerWrapperCtx for WokerRunCtx {
    fn from_worker_ctx(ctx: &WokerRunCtx) -> Self {
        WokerRunCtx::new(ctx.tho_device_idx(), ctx.max_other_parallel())
    }
}

impl WorkerWrapperCtx for crate::runner::run_ctx_2::RunCtxImpl {
    fn from_worker_ctx(ctx: &WokerRunCtx) -> Self {
        let mut stage_ctx = Self::new(ctx.tho_device_idx(), "WorkerWrapper");
        stage_ctx.set_max_other_parallel(ctx.max_other_parallel());
        stage_ctx
    }
}


pub type InpF<I>= Box<dyn for<'a> Fn(&'a dyn WorkerInput)->Box<&I> + Sync + Send>;

pub struct WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    base:Arc<S>,
    pd:PhantomData<Co>,
    inpf:InpF<I>
}

impl <Co, S, I, O> Debug for WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,

{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerWrapper")
            .field("base", &self.base)
            .finish()
    }
}

impl <Co, S, I, O> WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    pub fn new(base: Arc<S>, pd: PhantomData<Co>) -> Self {
        Self {
            base, pd,
            inpf: Box::new(|i| {
                let a: &dyn WorkerInput = i;
                let a: &dyn Any = a;

                Box::new(a.downcast_ref::<I>().expect(&format!("downcast_ref failed for input={i:?}")))
            })
        }
    }

    pub fn new_i(base: Arc<S>, pd: PhantomData<Co>, inpf:InpF<I>) -> Self {
        Self { base, pd, inpf }
    }
}
impl <Co, S, I, O>SyncStageIpcWrapper for WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput+ 'static,
{
    fn do_run(&self, _input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Vec<u8>, ThoError> {
        debug!("SimpWrapper::do_run: 0. input={_input:?}");
        let (input, max_other_parallel) = JsonIpcCodec::decode::<IpcSyncStageRunInput>(_input)?.into_parts();
        let input = WorkerStageInput::decode_for_worker::<JsonIpcCodec>(&input, all_cuda_tensors)?;
        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx(), max_other_parallel.unwrap_or_else(|| run_ctx.max_other_parallel()));
        let stage_ctx = S::Ctx::from_worker_ctx(&worker_ctx);

        debug!("SimpWrapper::do_run: 2. input={input:?}");
        let out = Arc::clone(&self.base).do_run(&input, &stage_ctx)?;


        // Convert nested worker outputs before encoding the complete stage output.
        Co::encode(&out.encode_from_worker(all_cuda_tensors, _tho_device)?)

    }

    fn consume_response(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<SyncStageIpcConsumeResponse, ThoError> {
        let (input, max_other_parallel) = JsonIpcCodec::decode::<IpcSyncStageRunInput>(input)?.into_parts();
        let (input, consumed_tensor_keys) = I::decode_for_worker_consuming::<JsonIpcCodec>(&input, all_cuda_tensors)?;
        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx(), max_other_parallel.unwrap_or_else(|| run_ctx.max_other_parallel()));
        let stage_ctx = S::Ctx::from_worker_ctx(&worker_ctx);
        match Arc::clone(&self.base).consume(input, stage_ctx) {
            ConsumeResult::Ok(output) => Ok(SyncStageIpcConsumeResponse::encoded(Co::encode(&output.encode_from_worker(all_cuda_tensors, tho_device)?)?, consumed_tensor_keys)),
            ConsumeResult::Failed { error, .. } => Err(error),
        }
    }

    fn consume_2(&self, input: Box<dyn WorkerInput>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
        debug!("SyncStageIpcWrapper::do_run_2: input={input:?}");

        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx(), 0);//TODO fixit
        let stage_ctx = S::Ctx::from_worker_ctx(&worker_ctx);


        let a = (self.inpf)(input.as_ref());
        let out = Arc::clone(&self.base).do_run(&a.ref_to_tensor(all_cuda_tensors, true)?, &stage_ctx)?;

        debug!("SyncStageIpcWrapper::do_run_2: out={out:?}");

        Ok(Box::new(out.encode_from_worker(all_cuda_tensors, tho_device)?))
    }



    fn do_run_2(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>

    {
        debug!("SyncStageIpcWrapper::do_run_2: input={input:?}");

        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx(), 0);//TODO fixit
        let stage_ctx = S::Ctx::from_worker_ctx(&worker_ctx);

        let a = (self.inpf)(input.as_ref());
        let out = Arc::clone(&self.base).do_run(&a.ref_to_tensor(all_cuda_tensors, false)?, &stage_ctx)?;

        debug!("SyncStageIpcWrapper::do_run_2: out={out:?}");

        Ok(Box::new(out.encode_from_worker(all_cuda_tensors, tho_device)?))
    }

}
///
#[derive(Debug)]
pub struct WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: SyncStage + StageWithState + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    base:Arc<St>,
    pd:PhantomData<Co>
}

impl <Co, St, I, O> WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: SyncStage + StageWithState + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    pub fn new(base: Arc<St>, pd: PhantomData<Co>) -> Self {
        Self { base, pd }
    }
}

impl <Co, St, I, O>SyncStageIpcWrapper for WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
    St: SyncStage + StageWithState+ Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
        // Keep this endpoint's concrete training-module state behind the erased worker capability.
        Some(self as &dyn ErasedWithState)
    }

    fn purge(&self, tho_device: ThoDevice) -> BoxFuture<Result<(), ThoError>>
    where
        Self: Sync
    {
        // Purge uses the application's CPU-first device index, not tch::Device.
        // self.base.purge_gpu(None, tho_device.tho_device_idx(), "zzz") TODO

        async {Ok(())}.boxed()
    }
}

impl <Co, St, I, O> SyncStageIpcWrapperWithState for WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: SyncStage + StageWithState + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput +WorkerOutput + 'static,
{
    type S = <St as StageWithState>::S;

    fn pre_run<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<MutexGuard<'a, Self::S>, ThoError>> {
        StageWithState::pre_run(self.base.as_ref(), tho_device.tho_device_idx())
    }

    fn do_run_with_state(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, state: MutexGuard<'_, Self::S>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<SyncStageIpcResponse, ThoError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {

                let (input, max_other_parallel) = JsonIpcCodec::decode::<IpcSyncStageRunInput>(input)?.into_parts();
                let input = WorkerStageInput::decode_for_worker::<JsonIpcCodec>(&input, all_cuda_tensors)?;
                let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx(), max_other_parallel.unwrap_or_else(|| run_ctx.max_other_parallel()));
                let stage_ctx = St::Ctx::from_worker_ctx(&worker_ctx);

                debug!("TrainTestStep2IpcWrapper::do_run_with_state: input={:?}", input);
                let stage = Arc::clone(&self.base);

                let output = StageWithState::do_run_with_state(Arc::clone(&stage), &input, state, &stage_ctx)?;
                debug!("TrainTestStep2IpcWrapper::do_run_with_state: output={:?}", output);

                Co::encode(&output.encode_from_worker(all_cuda_tensors, tho_device)?)
            })
        }).map(SyncStageIpcResponse::Encoded)
    }

    fn do_run_with_state_2<'a>(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, state: MutexGuard<'_, Self::S>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
        debug!("SyncStageIpcWrapper::do_run_2: input={input:?}");

        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx(), 0);//TODO fixit
        let stage_ctx = St::Ctx::from_worker_ctx(&worker_ctx);

        let a: &dyn WorkerInput = input.as_ref();
        let a: &dyn Any = a;

        let a = a.downcast_ref::<I>().expect(&format!("downcast_ref failed for input={input:?}"));
        let out = Arc::clone(&self.base).do_run_with_state(&a.ref_to_tensor(all_cuda_tensors, false)?, state, &stage_ctx)?;

        debug!("SyncStageIpcWrapper::do_run_2: out={out:?}");

        Ok(Box::new(out.encode_from_worker(all_cuda_tensors, tho_device)?))
    }
}