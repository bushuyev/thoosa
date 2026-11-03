use std::{borrow::Borrow, future::Future};
use crate::runner::execution::{StageExecution, ConsumingExecution, ExecutionContext};
use thoosa_common::tch_error::tch_result;
use crate::ipc::cuda_process::CudaIPCActorHandle;
use crate::ipc::endpoints::{EndpointRegistration, StageRecovery};
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::ipc::worker_msg::{CudaWorkerId, CudaWorkerMsgOut, OpResult, WorkerInput};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::purge::CachePurger;
use crate::tho_tensor::ThoTensor;
use async_trait::async_trait;
use anyhow::Context as _;
use log::{debug, warn};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::mem;
use std::string::ToString;
use std::sync::{Arc, LazyLock};
use std::sync::atomic::Ordering::Relaxed;
use tch::Tensor;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Adjustable, CachableIdTr, RunCtx, Stage, WithId};
use thoosa_common::runner::sync_stage::{ConsumeResult, ConsumingStage, StageWithState};
use thoosa_common::ipc::state_transfer::StageStateInfo;
use tokio::sync::{Mutex, MutexGuard};
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
            StageState::Ready(_state) => {
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
    St:Stage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static
{
    // Validate the IPC backend once; operations and cleanup retain its owner handle.
    actor: Arc<CudaIPCActorHandle>,
    pd:PhantomData<St>,
    endpoint_name:&'static str,
    // Each proxy keeps the id used at installation so invocations reach its own worker-side state.
    stage_id: Option<String>,
    // Drop has no run context, so retain the worker index that owns the installed endpoint.
    tho_device_idx: usize,
    worker_id: CudaWorkerId,
    registration: Option<Arc<EndpointRegistration>>,
    state_generation: std::sync::atomic::AtomicU64,
    // Generic inputs can supply the concrete typetag conversion that a blanket WorkerStageInput implementation cannot express.
    clone_refs: Option<Arc<dyn for<'a> Fn(&<St as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> + Send + Sync>>,

    // Invocations share admission; offload owns it exclusively until CPU commit.
    execution: tokio::sync::RwLock<()>,
    state: Mutex<Option<Box<StageStateDump>>>,
}

impl<St> Debug for ProxyStage<St>
where
    St: Stage,
    for<'a> <St as Stage>::In<'a>: WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProxyStage2")
            .field("endpoint_name", &self.endpoint_name)
            .field("stage_id", &self.stage_id)
            .field("tho_device_idx", &self.tho_device_idx)
            // Debug output must not lock and format the tensor-bearing dump.
            .field("state_generation", &self.state_generation.load(Relaxed))
            .finish_non_exhaustive()
    }
}

impl <St> Stage for ProxyStage<St>
where
    St:Stage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned  + 'static
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}

impl<St> StageExecution for ProxyStage<St>
where
    St: StageExecution + Send + Sync + 'static,
    for<'a> St::In<'a>: WorkerStageInput + Serialize,
    St::Out: WorkerStageOutput + DeserializeOwned + 'static,
{
    fn execute<'a, 'i: 'a, I, IFut>(self: Arc<Self>, input: IFut, _execution: &'a ExecutionContext<'_>, ctx: &'a Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<Self::Out, ThoError>>
    where Self: 'a, Self::In<'i>: Sync, Self::Out: Send, I: Borrow<Self::In<'i>> + Send + 'a, IFut: Future<Output = Result<I, ThoError>> + Send + 'a {
        Box::pin(async move {
            let input = input.await?;
            let refs = self.clone_refs(input.borrow())?;
            Self::handle_rs(self.invoke(refs, ctx.tho_device_idx(), false).await?)
        })
    }
}

impl<St> ConsumingExecution for ProxyStage<St>
where
    St: ConsumingStage + Send + Sync + 'static,
    for<'a> St::In<'a>: WorkerStageInput + Serialize,
    St::Out: WorkerStageOutput + DeserializeOwned + 'static,
{
    fn execute_consume<'a, 'i: 'a>(self: Arc<Self>, input: Self::In<'i>, _execution: &'a ExecutionContext<'_>, ctx: Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<ConsumeResult<Self::In<'i>, Self::Out>, ThoError>>
    where Self: 'a, Self::In<'i>: Send + 'a, Self::Out: Send, Self::Ctx: 'a {
        Box::pin(async move {
            let refs = self.clone_refs(&input);
            let result = async move {
                Self::handle_rs(self.invoke(refs?, ctx.tho_device_idx(), true).await?)
            }.await;
            Ok(match result {
                Ok(output) => ConsumeResult::Ok(output),
                Err(error) => ConsumeResult::Failed { input, error },
            })
        })
    }
}






impl <St> ProxyStage<St>
where
    St:Stage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    pub async fn proxy(
        endpoint_name:&'static str,
        stage_id: Option<String>,
        tho_device_idx: usize,
        init_data: Box<dyn WorkerInput>
    ) -> Result<Self, ThoError> {
        Self::proxy_inner(endpoint_name, stage_id.clone(), tho_device_idx, init_data, None, StageRecovery::OwnerManaged).await
    }

    /// The initializer must be independent of retired worker tensors and mutable
    /// progress. Keep model artifacts immutable for the lifetime of this proxy.
    pub async fn proxy_recreatable(endpoint_name: &'static str, stage_id: Option<String>, tho_device_idx: usize, init_data: Box<dyn WorkerInput>) -> Result<Self, ThoError> {
        Self::proxy_inner(endpoint_name, stage_id, tho_device_idx, init_data, None, StageRecovery::RecreateFromInit).await
    }

    pub async fn proxy_with_input_cloner_recreatable(
        endpoint_name: &'static str,
        stage_id: Option<String>,
        tho_device_idx: usize,
        init_data: Box<dyn WorkerInput>,
        clone_refs: Arc<dyn for<'a> Fn(&<St as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> + Send + Sync>,
    ) -> Result<Self, ThoError> {
        Self::proxy_inner(endpoint_name, stage_id, tho_device_idx, init_data, Some(clone_refs), StageRecovery::RecreateFromInit).await
    }

    pub async fn proxy_with_input_cloner(
        endpoint_name: &'static str,
        stage_id: Option<String>,
        tho_device_idx: usize,
        init_data: Box<dyn WorkerInput>,
        clone_refs: Arc<dyn for<'a> Fn(&<St as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> + Send + Sync>,
    ) -> Result<Self, ThoError> {
        // Direct generic inputs such as ThoTensor still need to enter the concrete WorkerInput registry.
        Self::proxy_inner(endpoint_name, stage_id.clone(), tho_device_idx, init_data, Some(clone_refs), StageRecovery::OwnerManaged).await
    }

    async fn proxy_inner(
        endpoint_name: &'static str,
        stage_id: Option<String>,
        tho_device_idx: usize,
        init_data: Box<dyn WorkerInput>,
        clone_refs: Option<Arc<dyn for<'a> Fn(&<St as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> + Send + Sync>>,
        recovery: StageRecovery,
    ) -> Result<Self, ThoError> {

        let ipc_actor = GlobalCtx::global().execution(tho_device_idx).ipc_actor()?;
        Self::install_on(ipc_actor, endpoint_name, stage_id, tho_device_idx, init_data, clone_refs, recovery).await
    }

    // Backend installation passes the actor explicitly; compatibility constructors
    // resolve the installed runtime only once, before entering this boundary.
    pub(crate) async fn install_on(
        ipc_actor: Arc<CudaIPCActorHandle>,
        endpoint_name: &'static str,
        stage_id: Option<String>,
        tho_device_idx: usize,
        init_data: Box<dyn WorkerInput>,
        clone_refs: Option<crate::runner::stage_install::InputCloner<St>>,
        recovery: StageRecovery,
    ) -> Result<Self, ThoError> {
        let worker_id = ipc_actor.worker_id()?;
        let registration = if recovery == StageRecovery::RecreateFromInit {
            let registration = ipc_actor.register_stage(worker_id, endpoint_name.to_string(), stage_id.clone(), init_data).await?;
            ipc_actor.ensure_stage(Arc::clone(&registration), worker_id).await?;
            Some(registration)
        } else {
            let response = ipc_actor.install_stage(worker_id, endpoint_name.to_string(), stage_id.clone(), init_data).await?;
            if let OpResult::Failed(error) = response.status { return Err(error.into()); }
            None
        };
        Ok(Self { actor: ipc_actor, endpoint_name, stage_id, tho_device_idx, worker_id, registration, state_generation: std::sync::atomic::AtomicU64::new(worker_id.generation.0), pd: PhantomData, clone_refs, execution: Default::default(), state: Default::default() })
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
                        warn!("handle_rs: error={error:?}");
                        Err(error.into())
                    }
                }
            }
        }
    }

    async fn invoke(self: Arc<Self>, input: Box<dyn WorkerInput>, device: usize, consume: bool) -> Result<CudaWorkerMsgOut, ThoError>
    where St: Send + Sync + 'static,
    {
        if device != self.tho_device_idx { return Err(ThoError::new("IPC proxy invoked on another device")); }
        let actor = self.actor.clone();
        // Bind a new operation, never an already-admitted old operation, to the
        // replacement. Tensor references still undergo their own identity checks.
        let worker_id = if self.registration.is_some() { actor.worker_id()? } else { self.worker_id };
        let activity = actor.activity_for(worker_id)?;
        // The caller may now cancel at any await. An owned task retains the
        // proxy, activity, and endpoint lease through restoration and reply, so
        // cancellation cannot admit an export while accepted work is still live.
        tokio::spawn(activity.run(async move {
            // Invocations share the endpoint lease. The worker never needs the
            // parent lock to reply; offload takes it exclusively through CPU commit.
            let _execution = self.execution.read().await;
            let mut state = self.state.lock().await;
            if let Some(registration) = &self.registration {
                if self.state_generation.load(Relaxed) != worker_id.generation.0 {
                    // A freshly reconstructed immutable module supersedes the
                    // old offload transaction and its generation-bound keys.
                    *state = None;
                    self.state_generation.store(worker_id.generation.0, Relaxed);
                }
                actor.ensure_stage(Arc::clone(registration), worker_id).await?;
            }
            self.load_gpu_inner(device, worker_id, &mut state).await?;
            drop(state);
            Ok::<_, ThoError>(actor.invoke_stage(worker_id, self.endpoint_name.to_string(), self.stage_id.clone(), input, consume).await?)
        })).await.context("IPC invocation task failed")?
    }

    fn is_current_worker(&self) -> bool {
        // Registries can retain unused proxies after reset. The retired worker
        // owns no live GPU state to offload; invocation still checks its identity.
        self.installed_worker().is_some_and(|worker| self.actor.is_current_worker(worker))
    }

    fn installed_worker(&self) -> Option<CudaWorkerId> {
        self.registration.as_ref().map_or(Some(self.worker_id), |registration| registration.installed_worker())
    }

    async fn purge_gpu(self: Arc<Self>, _ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, tho_device_idx: usize, _name: &str) -> Result<(), ThoError>
    where St: Send + Sync + 'static
    {
        if self.tho_device_idx != tho_device_idx || !self.is_current_worker() { return Ok(()); }
        let actor = self.actor.clone();
        let worker_id = self.installed_worker().ok_or_else(|| ThoError::new("endpoint has not been installed"))?;
        let activity = actor.activity_for(worker_id)?;
        // Cancellation must not lose a completed D2H dump after the worker drops
        // its storage. Retain both the proxy and its device lease until commit.
        tokio::spawn(activity.run(async move { self.purge_gpu_inner(tho_device_idx, worker_id).await })).await?
    }

    async fn purge_gpu_inner(&self, tho_device_idx: usize, worker_id: CudaWorkerId) -> Result<(), ThoError> {
        debug!("FuturesRunnerMapPurger>::purge_gpu: stage={:?}  device_idx={tho_device_idx}", self);

        if self.tho_device_idx != tho_device_idx {//not our device is being purged
            return Ok(());
        }

        // Keep the CPU dump across repeated purges and serialize offload with
        // restoration. The worker no longer owns its tensors after the first dump.
        let _execution = self.execution.write().await;
        let mut state = self.state.lock().await;
        if let Some(state) = state.as_mut() {
            if state.restore_started {
                let actor = self.actor.clone();
                actor.discard_stage_state(worker_id, self.endpoint_name.to_string(), self.stage_id.clone()).await?;
                state.restore_started = false;
            }
            return Ok(());
        }

        let actor = &self.actor;
        let state_out = actor.purge_stage_state(
            self.stage_id.clone(),
            self.endpoint_name.to_string(),
            worker_id,
        ).await?;
        debug!("FuturesRunnerMapPurger>::purge_gpu: tensor_count={} has_scalars={}", state_out.tensors.len(), state_out.scalars.is_some());



        *state = Some(Box::new(state_out));
        // Publish the CPU backup before authorizing destruction of worker state.
        // On a failed commit, keep the backup and discard any remaining GPU state
        // before a restore or another purge.
        let dump = state.as_mut().expect("CPU backup just installed");
        dump.restore_started = true;
        if let Some(export) = dump.export.take() {
            export.commit().await?;
        }
        dump.restore_started = false;
        debug!("FuturesRunnerMapPurger>::purge_gpu: stage={:?}  device_idx={tho_device_idx} done", self);

        Ok(())
    }

    async fn load_gpu_inner(&self, tho_device_idx: usize, worker_id: CudaWorkerId, guard: &mut Option<Box<StageStateDump>>) -> Result<(), ThoError> {
        let actor = self.actor.clone();
        if let Some(state) = guard.as_mut() {
            if state.restore_started {
                actor.discard_stage_state(worker_id, self.endpoint_name.to_string(), self.stage_id.clone()).await?;
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
                let upload = ThoTensor::new((), tch_result(tensor.as_ref().f_alias())?)?;
                actor.copy_stage_state(upload, tho_device_idx.into(), Some(key)).await?;
            }
            *guard = None;
        }
        Ok(())
    }
}



impl <St> Drop for ProxyStage<St>
where
    St:Stage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    fn drop(&mut self) {
        if self.registration.is_some() { return; }
        self.actor.release_endpoint(self.stage_id.clone(), self.endpoint_name.to_string(), self.worker_id);
    }
}


///
#[derive(Debug)]
pub enum DynSyncStageEnum<St, Init=()>
where
    St:Stage,
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
    St:Stage,
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
    St: Stage + Adjustable,
    for<'a> <St as Stage>::In<'a>: WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    // Preserve the local stage's adjustment marker across local/proxy dispatch.
}



impl <St, Init> StageExecution for DynSyncStageEnum<St, Init>
where
    St: StageExecution + Send + Sync + 'static,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static
{
    fn execute<'a, 'i: 'a, I, IFut>(self: Arc<Self>, input: IFut, execution: &'a ExecutionContext<'_>, ctx: &'a Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<Self::Out, ThoError>>
    where Self: 'a, Self::In<'i>: Sync, Self::Out: Send, I: Borrow<Self::In<'i>> + Send + 'a, IFut: Future<Output = Result<I, ThoError>> + Send + 'a {
        Box::pin(async move {
            match self.as_ref() {
                Self::Local(stage) => Arc::clone(stage).execute(input, execution, ctx).await,
                Self::Proxy { stage, .. } => Arc::clone(stage).execute(input, execution, ctx).await,
            }
        })
    }
}

impl <St, Init> ConsumingExecution for DynSyncStageEnum<St, Init>
where
    St: ConsumingStage + Send + Sync + 'static,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static
{
    fn execute_consume<'a, 'i: 'a>(self: Arc<Self>, input: Self::In<'i>, execution: &'a ExecutionContext<'_>, ctx: Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<ConsumeResult<Self::In<'i>, Self::Out>, ThoError>>
    where Self: 'a, Self::In<'i>: Send + 'a, Self::Out: Send, Self::Ctx: 'a {
        Box::pin(async move {
            match self.as_ref() {
                Self::Local(stage) => Arc::clone(stage).execute_consume(input, execution, ctx).await,
                Self::Proxy { stage, .. } => Arc::clone(stage).execute_consume(input, execution, ctx).await,
            }
        })
    }
}



#[derive(Debug)]
pub enum DynSyncStageWithStateEnum<St, Init = ()>
where
    St: Stage + StageWithState,
    for<'a> <St as Stage>::In<'a>: WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned + 'static,
    Init: Debug + Send + Sync + 'static,
{
    Local(Arc<St>),
    Proxy {
        stage: Arc<ProxyStage<St>>,
        init: Init,//if async stages need to know parameters with which sync stage was initialized (motif_id, out_id etc)
    },
}

impl <St, Init> Stage for DynSyncStageWithStateEnum<St, Init>
where
    St:Stage+StageWithState,
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
    St: Stage + StageWithState + Adjustable,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out: WorkerStageOutput + DeserializeOwned,
    Init:Debug + Send + Sync + 'static
{
    // Preserve the wrapped stage's adjustment capability across local/proxy dispatch.
}



impl <St, Init> StageExecution for DynSyncStageWithStateEnum<St, Init>
where
    St: StageExecution + StageWithState + Send + Sync + 'static,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static
{
    fn execute<'a, 'i: 'a, I, IFut>(self: Arc<Self>, input: IFut, execution: &'a ExecutionContext<'_>, ctx: &'a Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<Self::Out, ThoError>>
    where Self: 'a, Self::In<'i>: Sync, Self::Out: Send, I: Borrow<Self::In<'i>> + Send + 'a, IFut: Future<Output = Result<I, ThoError>> + Send + 'a {
        Box::pin(async move {
            match self.as_ref() {
                Self::Local(stage) => Arc::clone(stage).execute(input, execution, ctx).await,
                Self::Proxy { stage, .. } => Arc::clone(stage).execute(input, execution, ctx).await,
            }
        })
    }
}


///

///
///
///
// Dropping an uncommitted dump closes decision and makes the actor abort the
// worker transaction. The actor owns activity until that cleanup has replied.
#[derive(Debug)]
pub(crate) struct PreparedStageStatePurge {
    pub decision: tokio::sync::oneshot::Sender<bool>,
    pub completed: tokio::sync::oneshot::Receiver<Result<(), ThoError>>,
}

impl PreparedStageStatePurge {
    // A checkpoint releases the export freeze without evicting live GPU state.
    pub(crate) async fn keep_live(self) -> Result<(), ThoError> {
        let _ = self.decision.send(false);
        self.completed.await.context("stage export completion channel closed")?
    }

    pub(crate) async fn commit(self) -> Result<(), ThoError> {
        let _ = self.decision.send(true);
        self.completed.await.context("stage export completion channel closed")?
    }
}

#[derive(Debug)]
pub struct StageStateDump {
    scalars:Option<Box<dyn WorkerOutput>>,
    tensors:HashMap<String, ThoTensor>,
    restore_started: bool,
    pub(crate) export: Option<PreparedStageStatePurge>,
}

impl StageStateDump {
    pub fn new(scalars: Option<Box<dyn WorkerOutput>>, tensors: HashMap<String, ThoTensor>) -> Self {
        Self { scalars, tensors, restore_started: false, export: None }
    }
}

#[async_trait]
impl <St, Init> CachePurger for DynSyncStageWithStateEnum<St, Init>
where
    St:Stage+StageWithState + CachePurger + 'static,
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
                stage.tho_device_idx == device_idx && stage.is_current_worker() && stage.state.lock().await.as_ref().is_none_or(|state| state.restore_started)
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, tho_device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            DynSyncStageWithStateEnum::Local(local) => {
                local.purge_gpu(ids_to_keep, tho_device_idx, name).await
            }
            DynSyncStageWithStateEnum::Proxy { stage, init: _init, } => {
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


impl <St, Init> DynSyncStageWithStateEnum<St, Init>
where
    St:Stage+StageWithState + WithId,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned + 'static,
    Init:Debug + Send + Sync + 'static,
    <St as WithId>::ID:Display

{

    //to use same id for proxies and local stages with state when registering in GlobalCtx::global().register_cache
    pub fn stage_id(&self) -> String {
        match self {
            DynSyncStageWithStateEnum::Local(stage) => {
                stage.id().to_string()
            }
            DynSyncStageWithStateEnum::Proxy { stage, .. } => {
                stage.stage_id.as_ref().unwrap_or_else(|| {
                    &EMPTY_STR
                }).clone()
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod async_dispatch_tests {
    use super::*;
    use std::time::Duration;

    pub async fn assert_cancelled_invocation_holds_export_lease<St>(proxy: Arc<ProxyStage<St>>, input: Box<dyn WorkerInput>) -> Result<(), ThoError>
    where
        St: Stage + Send + Sync + 'static,
        for<'a> St::In<'a>: WorkerStageInput + Serialize,
        St::Out: WorkerStageOutput + DeserializeOwned + 'static,
    {
        // Hold restoration to suspend an invocation after it acquires execution
        // admission. Cancellation must not release that lease to an offloader.
        let state = proxy.state.lock().await;
        let invocation = tokio::spawn(Arc::clone(&proxy).invoke(input, proxy.tho_device_idx, false));
        tokio::time::timeout(Duration::from_secs(5), async {
            while proxy.execution.try_write().is_ok() { tokio::task::yield_now().await; }
        }).await?;
        invocation.abort();
        assert!(invocation.await.unwrap_err().is_cancelled());
        assert!(proxy.execution.try_write().is_err(), "cancelled caller must not release accepted invocation admission");
        let offload = Arc::clone(&proxy).purge_gpu(None, proxy.tho_device_idx, "cancelled invocation");
        tokio::pin!(offload);
        assert!(futures::poll!(&mut offload).is_pending());
        drop(state);
        tokio::time::timeout(Duration::from_secs(10), offload).await??;
        assert!(proxy.state.lock().await.is_some(), "offload must commit its CPU backup after the invocation completes");
        Ok(())
    }
}

#[path = "state_checkpoint.rs"]
mod state_checkpoint;
pub(crate) use state_checkpoint::{CpuStateCheckpoint, checkpoint_operation, copy_state_tensor};
