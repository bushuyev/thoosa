use crate::copy_ctx::CudaActorHandle;
use crate::ipc::cuda_process::CudaIPCActorHandle;
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::ipc::serialization::json::{IpcCodec, JsonIpcCodec};
use crate::ipc::worker_msg::{CudaWorkerId, CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation, WorkerInput, WorkerOutput};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::purge::FuturesRunnerMapPurger;
use crate::tho_tensor::ThoTensor;
use async_trait::async_trait;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::{debug, error};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::any::Any;
use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::hash::Hasher;
use std::marker::PhantomData;
use std::string::ToString;
use std::sync::{Arc, LazyLock};
use parking_lot::RwLock;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Adjustable, CachableIdTr, ConsumeResult, ErasedStageWithState, OutId, RunCtx, Stage, StageStageInfo, StageWithState, SyncStage, WithId};
use tokio::sync::MutexGuard;

pub struct ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static
{
    pd:PhantomData<St>,
    endpoint_name:&'static str,
    // Each proxy keeps the id used at installation so invocations reach its own worker-side state.
    stage_id: Option<String>,
    // Drop has no run context, so retain the worker index that owns the installed endpoint.
    tho_device_idx: usize,
    worker_id: CudaWorkerId,
    // Generic inputs can supply the concrete typetag conversion that a blanket WorkerStageInput implementation cannot express.
    clone_refs: Option<Arc<dyn for<'a> Fn(&<St as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> + Send + Sync>>
}

impl<St> Debug for ProxyStage<St>
where
    St: SyncStage,
    for<'a> <St as Stage>::In<'a>: WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProxyStage2")
            .field("endpoint_name", &self.endpoint_name)
            .field("stage_id", &self.stage_id)
            .field("tho_device_idx", &self.tho_device_idx)
            .finish_non_exhaustive()
    }
}

impl <St> Stage for ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned  + 'static
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}



impl <St>SyncStage for ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ProxyStage2::do_run"))]
    fn do_run<'a>(self: std::sync::Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, thoosa_shared::ThoError>
    where
        Self: Sized,
    {

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {

                debug!("ProxyStage2::do_run: input={input:?}");

                let actor:&dyn Any = GlobalCtx::global().cuda_actor(ctx.tho_device_idx()).as_ref();
                let ipc_actor:&CudaIPCActorHandle = actor.downcast_ref().unwrap();


                let rs = ipc_actor.invoke_stage(
                    self.worker_id,
                    self.endpoint_name.to_string(),
                    self.stage_id.clone(),
                    self.clone_refs(input)?, //clones refs
                    false,
                ).await?;

                debug!("ProxyStage2::do_run: rs={rs:?}");


                Self::handle_rs(rs)


            })
        })
    }


    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ProxyStage2::consume"))]
    fn consume<'a>(self: Arc<Self>, input: Self::In<'a>, ctx: Self::Ctx) -> ConsumeResult<Self::In<'a>, Self::Out>
    where
        Self: Sized,
    {

        let result = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async {


                let actor:&dyn Any = GlobalCtx::global().cuda_actor(ctx.tho_device_idx()).as_ref();
                let ipc_actor:&CudaIPCActorHandle = actor.downcast_ref().unwrap();


                let rs = ipc_actor.invoke_stage(
                    self.worker_id,
                    self.endpoint_name.to_string(),
                    self.stage_id.clone(),
                    self.clone_refs(&input)?, //clones refs
                    true,
                ).await?;

                debug!("ProxyStage2::consume: rs={rs:?}");
                Self::handle_rs(rs)
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
    pub async fn proxy(
        endpoint_name:&'static str,
        stage_id: Option<String>,
        tho_device_idx: usize,
        init_data: Box<dyn WorkerInput>
    ) -> Result<Self, ThoError> {
        Self::proxy_inner(endpoint_name, stage_id.clone(), tho_device_idx, init_data, None).await
    }

    pub async fn proxy_with_input_cloner(
        endpoint_name: &'static str,
        stage_id: Option<String>,
        tho_device_idx: usize,
        init_data: Box<dyn WorkerInput>,
        clone_refs: Arc<dyn for<'a> Fn(&<St as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> + Send + Sync>,
    ) -> Result<Self, ThoError> {
        // Direct generic inputs such as ThoTensor still need to enter the concrete WorkerInput registry.
        Self::proxy_inner(endpoint_name, stage_id.clone(), tho_device_idx, init_data, Some(clone_refs)).await
    }

    async fn proxy_inner(
        endpoint_name: &'static str,
        stage_id: Option<String>,
        tho_device_idx: usize,
        init_data: Box<dyn WorkerInput>,
        clone_refs: Option<Arc<dyn for<'a> Fn(&<St as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> + Send + Sync>>,
    ) -> Result<Self, ThoError> {

        let actor:&dyn Any = GlobalCtx::global().cuda_actor(tho_device_idx).as_ref();
        let ipc_actor:&CudaIPCActorHandle = actor.downcast_ref().unwrap();

        let worker_id = ipc_actor.worker_id()?;
        let rs = ipc_actor.install_stage(
            worker_id,
            endpoint_name.to_string(),
            stage_id.clone(),
            init_data,
        ).await?;

        match rs {
            CudaWorkerMsgOut { status, .. } => {
                match status {
                    OpResult::Ok(_) => {
                        Ok(Self {
                            endpoint_name,
                            stage_id,
                            tho_device_idx,
                            worker_id,
                            pd:PhantomData,
                            clone_refs,
                        })

                    }
                    OpResult::Failed(error) => {
                        Err(error.into())
                    }
                }
            }
        }
    }

    fn clone_refs<'a>(&self, input: &<St as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> {
        match &self.clone_refs {
            Some(clone_refs) => clone_refs(input),
            None => input.clone_refs(),
        }
    }

    fn handle_rs(rs: CudaWorkerMsgOut) -> Result<<St as Stage>::Out, ThoError> {
        match rs {
            CudaWorkerMsgOut { status, .. } => {
                match status {
                    OpResult::Ok(r) => {
                        match r {
                            None => {
                                Err(ThoError::new("no response".to_string()))
                            }
                            Some(v) => {
                                let r: Box<dyn Any> = v as Box<dyn Any>;
                                Ok(*r.downcast::<St::Out>().unwrap())
                            }
                        }
                    }
                    OpResult::Failed(error) => {
                        Err(error.into())
                    }
                }
            }
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
        let stage_id = self.stage_id.clone();
        let tho_device_idx = self.tho_device_idx;
        let worker_id = self.worker_id;

        debug!("ProxyStage2::drop: endpoint_name={endpoint_name} stage_id={stage_id:?} tho_device_idx={tho_device_idx}");

        let actor:&dyn Any = GlobalCtx::global().cuda_actor(tho_device_idx).as_ref();
        let ipc_actor:&CudaIPCActorHandle = actor.downcast_ref().unwrap();


        tokio::spawn(async move {

            debug!("ProxyStage2::drop: endpoint_name={endpoint_name} stage_id={stage_id:?} tho_device_idx={tho_device_idx} spawned");

            if let Err(error) = ipc_actor.drop_endpoint(stage_id.clone(), endpoint_name.to_string(), worker_id).await {
                if !GlobalCtx::is_shutting_down() {
                    error!("ProxyStage2::drop: failed to drop endpoint_name={endpoint_name} stage_id={stage_id:?} tho_device_idx={tho_device_idx}: {error:?}");
                }
            }
            Ok::<_, ThoError>(())
        });
    }
}


///
#[derive(Debug)]
pub enum DynSyncStageEnum<St, Init=()>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static
{
    Local(Arc<St>),
    Proxy {
        stage: Arc<ProxyStage<St>>,
        init: Init
    }

}

impl <St, Init> Stage for DynSyncStageEnum<St, Init>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}

impl<St, Init> Adjustable for DynSyncStageEnum<St, Init>
where
    St: SyncStage + Adjustable,
    for<'a> <St as Stage>::In<'a>: WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    // Preserve the local stage's adjustment marker across local/proxy dispatch.
}

impl <St, Init> SyncStage for DynSyncStageEnum<St, Init>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
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


impl <St, Init> StageWithState for DynSyncStageEnum<St, Init>
where
    St:SyncStage+StageWithState+Send+Sync,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
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


#[async_trait]
impl <St, Init> FuturesRunnerMapPurger for DynSyncStageEnum<St, Init>
where
    St:SyncStage+StageWithState + FuturesRunnerMapPurger,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        match self {
            DynSyncStageEnum::Local(stage) => {
                stage.has_gpu_data(device_idx).await
            }
            DynSyncStageEnum::Proxy { stage, .. } => {
                stage.tho_device_idx == device_idx
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            DynSyncStageEnum::Local(local) => {
                local.purge_gpu(ids_to_keep, device_idx, name).await
            }
            DynSyncStageEnum::Proxy { stage,  .. } => {
                if stage.tho_device_idx != device_idx { return Ok(()); }
                crate::ipc::serialization::json::purge_ipc_sync_stage(
                    GlobalCtx::global().cuda_actor(device_idx).as_ref(),
                    stage.endpoint_name,
                    stage.stage_id.clone(),
                    stage.worker_id,
                ).await
            }
        }
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            DynSyncStageEnum::Local(local) => {
                local.drop_gpu(device_idx, name).await
            }
            DynSyncStageEnum::Proxy { .. } => {
                Ok(())
            }
        }
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        match self {
            DynSyncStageEnum::Local(local) => {
                local.purge_cpu(purge_until).await
            }
            DynSyncStageEnum::Proxy { .. } => {
                0
            }
        }
    }
}

#[derive(Debug)]
pub enum DynSyncStageWithStateEnum<St, Init = ()>
where
    St: SyncStage + StageWithState,
    for<'a> <St as Stage>::In<'a>: WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned + 'static,
    Init: Debug + Send + Sync + 'static,
{
    Local(Arc<St>),
    Proxy {
        stage: Arc<ProxyStage<St>>,
        init: Init,//if async stages need to know parameters with which sync stage was initialized (motif_id, out_id etc)
        state: RwLock<Option<Box<St::S>>>//TODO remove Option - the state should always be available?
    },
}

impl <St, Init> Stage for DynSyncStageWithStateEnum<St, Init>
where
    St:SyncStage+StageWithState,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}

impl <St, Init> Adjustable for DynSyncStageWithStateEnum<St, Init>
where
    St: SyncStage + StageWithState + Adjustable,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    // Preserve the wrapped stage's adjustment capability across local/proxy dispatch.
}

impl <St, Init> SyncStage for DynSyncStageWithStateEnum<St, Init>
where
    St: SyncStage + StageWithState + Send + Sync,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
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
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
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
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static,
    St::S: 'static
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        match self {
            DynSyncStageWithStateEnum::Local(stage) => {
                stage.has_gpu_data(device_idx).await
            }
            DynSyncStageWithStateEnum::Proxy { stage, .. } => {
                stage.tho_device_idx == device_idx
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            DynSyncStageWithStateEnum::Local(local) => {
                local.purge_gpu(ids_to_keep, device_idx, name).await
            }
            DynSyncStageWithStateEnum::Proxy { stage, init, state, } => {
                if stage.tho_device_idx != device_idx {//not our device is being purged
                    return Ok(());
                }


                let state_out = GlobalCtx::global().cuda_actor(device_idx).purge_stage_state(
                    stage.stage_id.clone(),
                    stage.endpoint_name.to_string(),
                    stage.worker_id,
                ).await?;

                if let OpResult::Ok(Some(state_box)) = state_out.status {
                    if let Ok(state_rs) = (state_box as Box<dyn Any>).downcast::<St::S>() {
                        *state.write() = Some(state_rs);
                    }
                }

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


static EMPTY_STR: LazyLock<String> = LazyLock::new(|| String::from(""));


impl <St, Init> WithId for DynSyncStageWithStateEnum<St, Init>
where
    St:SyncStage+StageWithState + WithId,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static,
    <St as WithId>::ID:AsRef<String>

{
    type ID = String;

    //to use same id for proxies and local stages with state when registering in GlobalCtx::global().register_cache
    fn id(&self) -> &String {
        match self {
            DynSyncStageWithStateEnum::Local(stage) => {
                stage.id().as_ref()
            }
            DynSyncStageWithStateEnum::Proxy { stage, .. } => {
                stage.stage_id.as_ref().unwrap_or_else(|| {
                    &EMPTY_STR
                })
            }
        }
    }
}
