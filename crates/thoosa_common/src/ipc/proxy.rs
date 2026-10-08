use crate::copy_ctx::CudaActorHandle;
use crate::ipc::cuda_process::{CudaIPCActor, CudaIPCActorHandle};
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::ipc::serialization::json::{IpcCodec, JsonIpcCodec};
use crate::ipc::worker_msg::{CudaWorkerId, CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation, WorkerInput};
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
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::hash::Hasher;
use std::marker::PhantomData;
use std::mem;
use std::string::ToString;
use std::sync::{Arc, LazyLock};
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering::Relaxed;
use parking_lot::RwLock;
use tch::Tensor;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Adjustable, CachableIdTr, ConsumeResult, ErasedStageWithState, OutId, RunCtx, Stage, StageStateInfo, StageWithState, SyncStage, WithId};
use tokio::sync::{MappedMutexGuard, Mutex, MutexGuard, RwLockReadGuard};
use thoosa_shared::ipc_types::{H2DTensorKey, WorkerOutput};

#[derive(Debug)]
pub enum StageState<S> {
    BeingLoaded{ tensors:HashMap<String, Tensor>, keys_to_load:Vec<String>},
    Ready(S),
    OffLoaded(Vec<String>)
}

impl <S> StageState<S>
where
    S:From<HashMap<String, Tensor>>
{
    pub async fn set_state_tensor(tensor_id: String, t: Tensor, mut guard: MutexGuard<'_, StageState<S>>) ->  Result<(), ThoError>

    {
        *guard = match  &mut *guard {
            StageState::Ready(state) => {
                panic!("unexpected state");
            },
            StageState::OffLoaded(keys) => {
                assert!(!keys.is_empty(), "no keys");

                debug!("TestStageB::set_state_tensor: StageState::OffLoaded keys={keys:?}");

                let keys_len = keys.len();
                let mut keys = mem::take(keys);
                keys.retain(|v| !v.eq(&tensor_id));

                if keys_len - 1 != keys.len() {
                    return Err(ThoError::NotFound(Some(tensor_id)));
                }

                debug!("TestStageB::set_state_tensor: StageState::OffLoaded keys_len={keys_len:?} keys.len={}", keys.len());

                if keys.len() == 0 { //state contains single tensor
                    StageState::Ready(HashMap::from([(tensor_id, t)]).into())
                } else {
                    StageState::BeingLoaded { tensors: HashMap::from([(tensor_id, t)]), keys_to_load: keys }
                }
            },
            StageState::BeingLoaded { tensors, keys_to_load: keys } => {
                assert!(!keys.is_empty(), "no keys");

                debug!("TestStageB::set_state_tensor: StageState::OffLoaded keys={keys:?}");

                let keys_len = keys.len();
                let mut keys = mem::take(keys);
                keys.retain(|v| !v.eq(&tensor_id));

                if keys_len - 1 != keys.len() {
                    return Err(ThoError::NotFound(Some(tensor_id)));
                }

                debug!("TestStageB::set_state_tensor: StageState::OffLoaded keys_len={keys_len:?} keys.len={}", keys.len());
                tensors.insert(tensor_id, t);
                if keys.len() == 0 {
                    StageState::Ready(mem::take(tensors).into())
                } else {
                    StageState::BeingLoaded { tensors: mem::take(tensors), keys_to_load: keys }
                }
            }
        };

        Ok(())
    }

}



impl <S> AsRef<S> for  StageState<S> {
    fn as_ref(&self) -> &S {
        if let StageState::Ready(v) = self {
            v
        } else {
            panic!("StageState::as_ref: no state")
        }
    }
}

impl <S> AsMut<S> for  StageState<S> {
    fn as_mut(&mut self) -> &mut S {

        if let StageState::Ready(v) = self {
            v
        } else {
            panic!("StageState::as_mut: no state")
        }
    }
}

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
    clone_refs: Option<Arc<dyn for<'a> Fn(&<St as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> + Send + Sync>>,

    state: Mutex<Option<Box<StageStateDump>>>,
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
            .field("state", &self.state)
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
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {

        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {
                let actor:&dyn Any = GlobalCtx::global().cuda_actor(ctx.tho_device_idx()).as_ref();
                let ipc_actor:&CudaIPCActorHandle = actor.downcast_ref().unwrap();


                debug!("ProxyStage2::do_run: input={input:?}");

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

impl <St>StageWithState for ProxyStage<St>
where
    St:SyncStage + StageWithState,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static
{
    type S = Option<Box<StageStateDump>>;

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Option<MappedMutexGuard<'_, Self::S>>, ThoError>> {
        debug!("ProxyStage::pre_run: to_device_idx={to_device_idx}");
        async {
            let guard = self.state.lock().await;
            Ok(Some(MutexGuard::map(guard, |v|v)))

        }.boxed()
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
                            state: Default::default(),
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

    async fn purge_gpu(self: Arc<Self>, _ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, tho_device_idx: usize, _name: &str) -> Result<(), ThoError>
    where St: Send + Sync + 'static
    {
        if self.tho_device_idx != tho_device_idx { return Ok(()); }
        let actor = GlobalCtx::global().cuda_actor(tho_device_idx);
        let actor = (actor.as_ref() as &dyn Any).downcast_ref::<CudaIPCActorHandle>().expect("no CudaIPCActor");
        let activity = actor.activity_for(self.worker_id)?;
        // Cancellation must not lose a completed D2H dump after the worker drops
        // its storage. Retain both the proxy and its device lease until commit.
        tokio::spawn(activity.run(async move { self.purge_gpu_inner(tho_device_idx).await })).await?
    }

    async fn purge_gpu_inner(&self, tho_device_idx: usize) -> Result<(), ThoError> {
        debug!("FuturesRunnerMapPurger>::purge_gpu: stage={:?}  device_idx={tho_device_idx}", self);

        if self.tho_device_idx != tho_device_idx {//not our device is being purged
            return Ok(());
        }

        // Keep the CPU dump across repeated purges and serialize offload with
        // restoration. The worker no longer owns its tensors after the first dump.
        let mut state = self.state.lock().await;
        if let Some(state) = state.as_mut() {
            if state.restore_started {
                let actor = GlobalCtx::global().cuda_actor(tho_device_idx);
                let actor = (actor.as_ref() as &dyn Any).downcast_ref::<CudaIPCActorHandle>().expect("no CudaIPCActor");
                actor.discard_stage_state(self.worker_id, self.endpoint_name.to_string(), self.stage_id.clone()).await?;
                state.restore_started = false;
            }
            return Ok(());
        }

        let actor = GlobalCtx::global().cuda_actor(tho_device_idx);
        let state_out = actor.purge_stage_state(
            self.stage_id.clone(),
            self.endpoint_name.to_string(),
            self.worker_id,
        ).await?;
        debug!("FuturesRunnerMapPurger>::purge_gpu: state_out={state_out:?}");



        *state = Some(Box::new(state_out));
        debug!("FuturesRunnerMapPurger>::purge_gpu: stage={:?}  device_idx={tho_device_idx} done", self);

        Ok(())
    }

    async fn load_gpu(self: Arc<Self>, tho_device_idx: usize) -> Result<(), ThoError>
    where St: Send + Sync + 'static
    {
        let actor = GlobalCtx::global().cuda_actor(tho_device_idx);
        let actor = (actor.as_ref() as &dyn Any).downcast_ref::<CudaIPCActorHandle>().expect("no CudaIPCActor");
        let activity = actor.activity_for(self.worker_id)?;
        // Await the whole transaction even if its caller is cancelled, so a
        // subsequent purge cannot race an outstanding tensor upload.
        tokio::spawn(activity.run(async move { self.load_gpu_inner(tho_device_idx).await })).await?
    }

    async fn load_gpu_inner(&self, tho_device_idx: usize) -> Result<(), ThoError> {
        let actor = GlobalCtx::global().cuda_actor(tho_device_idx);
        let actor = (actor.as_ref() as &dyn Any).downcast_ref::<CudaIPCActorHandle>().expect("no CudaIPCActor");
        let mut guard = self.state.lock().await;
        if let Some(state) = guard.as_mut() {
            if state.restore_started {
                actor.discard_stage_state(self.worker_id, self.endpoint_name.to_string(), self.stage_id.clone()).await?;
            }
            state.restore_started = true;
            for (key, tensor) in &state.tensors {
                let key = H2DTensorKey::StageState {
                    endpoint_name: self.endpoint_name.to_string(),
                    stage_id: self.stage_id.as_ref().ok_or_else(|| ThoError::new("stateful IPC stage has no stage id"))?.clone(),
                    tensor_id: key.clone(),
                };
                // Keep the CPU allocation until every upload and reconstruction
                // succeeds. OOM recovery can then discard partial GPU state and retry.
                // Release the synchronous tensor guard before awaiting the upload.
                let upload = ThoTensor::new((), tensor.as_ref().f_alias()?)?;
                actor.copy_stage_state(upload, tho_device_idx.into(), Some(key)).await?;
            }
            *guard = None;
        }
        Ok(())
    }
}



impl <St> Drop for ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    fn drop(&mut self) {
        debug!("ProxyStage2::drop: endpoint_name={:?}", self.endpoint_name);

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


#[async_trait]
impl <St, Init> FuturesRunnerMapPurger for DynSyncStageEnum<St, Init>//TODO stages without State don't have GPU data, remove
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
        state: RwLock<Option<Box<StageStateDump>>>//TODO remove - state was moved to ProxyStage
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
    St: SyncStage + StageWithState + Send + Sync + 'static,
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
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, &ctx),
        }
    }

    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        // Advertise typed state execution without exposing ThoTrainModule to the common runner.
        match self {
            Self::Local(_) => Some(self),
            Self::Proxy { stage,.. } => Some(self),
        }
    }
}

///
impl <St, Init> StageWithState for DynSyncStageWithStateEnum<St, Init>
where
    St:SyncStage+StageWithState+Send+Sync + 'static,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static
{
    type S = <St as StageWithState>::S;


    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Option<tokio::sync::MappedMutexGuard<'_, Self::S>>, ThoError>> {
        match self {
            Self::Local(stage) => {
                //TODO add semaphore with single permit
                StageWithState::pre_run(stage.as_ref(), to_device_idx)
            },
            Self::Proxy { stage, init,.. } => {
                let stage = Arc::clone(&stage);
                async move {
                    debug!("SyncStage>::pre_run: stage={stage:?}");

                    stage.load_gpu(to_device_idx).await?;

                    Ok(None)

                }.boxed()
            },
        }
    }

    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, state: &mut Self::S, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => StageWithState::do_run_with_state(Arc::clone(stage), input, state, ctx),
            Self::Proxy { stage, .. } => {
                Arc::clone(stage).do_run(input, ctx)
            },
        }
    }
}
///
///
///
#[derive(Debug)]
pub struct StageStateDump {
    scalars:Option<Box<dyn WorkerOutput>>,
    tensors:HashMap<String, ThoTensor>,
    restore_started: bool,
}

impl StageStateDump {
    pub fn new(scalars: Option<Box<dyn WorkerOutput>>, tensors: HashMap<String, ThoTensor>) -> Self {
        Self { scalars, tensors, restore_started: false }
    }
}

#[async_trait]
impl <St, Init> FuturesRunnerMapPurger for DynSyncStageWithStateEnum<St, Init>
where
    St:SyncStage+StageWithState + FuturesRunnerMapPurger + 'static,
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
                stage.tho_device_idx == device_idx && stage.state.lock().await.as_ref().is_none_or(|state| state.restore_started)
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, tho_device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            DynSyncStageWithStateEnum::Local(local) => {
                local.purge_gpu(ids_to_keep, tho_device_idx, name).await
            }
            DynSyncStageWithStateEnum::Proxy { stage, init, state, } => {
               Arc::clone(stage).purge_gpu(ids_to_keep, tho_device_idx, name).await?;
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
