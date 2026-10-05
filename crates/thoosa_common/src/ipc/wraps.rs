use crate::ipc::cuda_process::CudaIPCActorHandle;
use crate::ipc::cuda_wokrer::{AllTensors, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::serialization::json::{IpcCodec, JsonIpcCodec};
use crate::ipc::worker_msg::{CudaWorkerMsgIn, Operation, WorkerInput};
use crate::runner::common::cachable::ThoDataRef;
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::purge::FuturesRunnerMapPurger;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt};
use futures_util::future::BoxFuture;
use log::{debug, error};
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use std::any::Any;
use std::borrow::Borrow;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use tch::Tensor;
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{Adjustable, CachableIdTr, ConsumeResult, ErasedStageWithState, RunCtx, Stage, StageStateInfo, StageWithState, SyncStage};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tokio::sync::{MappedMutexGuard, MutexGuard};
use thoosa_shared::ipc_types::WorkerOutput;

///

pub trait WorkerWrapperCtx: RunCtx {
    fn from_worker_ctx(ctx: &WokerRunCtx) -> Self;
}

impl WorkerWrapperCtx for WokerRunCtx {
    fn from_worker_ctx(ctx: &WokerRunCtx) -> Self {
        WokerRunCtx::new(ctx.tho_device_idx())
    }
}

impl WorkerWrapperCtx for crate::runner::run_ctx::RunCtxImpl {
    fn from_worker_ctx(ctx: &WokerRunCtx) -> Self {
        let mut stage_ctx = Self::new(ctx.tho_device_idx(), "WorkerWrapper");

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

    fn consume(&self, input: Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
        debug!("SyncStageIpcWrapper::consume: input={input:?}");

        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx());//TODO fixit
        let stage_ctx = S::Ctx::from_worker_ctx(&worker_ctx);


        let a = (self.inpf)(input.as_ref());
        let worker_input = a.ref_to_tensor(all_cuda_tensors, false)?;
        match Arc::clone(&self.base).consume(worker_input, stage_ctx) {
            ConsumeResult::Ok(out) => {
                // Commit removal only after the stage accepts the input so a failed consume can be retried.
                drop(a.ref_to_tensor(all_cuda_tensors, true)?);
                debug!("SyncStageIpcWrapper::consume: out={out:?}");
                Ok(Box::new(out.encode_from_worker(all_cuda_tensors, tho_device)?))
            }
            ConsumeResult::Failed { error, .. } => Err(error),
        }
    }



    fn do_run(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>

    {
        debug!("SyncStageIpcWrapper::do_run: input={input:?}");

        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx());
        let stage_ctx = S::Ctx::from_worker_ctx(&worker_ctx);

        let a = (self.inpf)(input.as_ref());
        let out = Arc::clone(&self.base).do_run(&a.ref_to_tensor(all_cuda_tensors, false)?, &stage_ctx)?;

        debug!("SyncStageIpcWrapper::do_run: out={out:?}");

        Ok(Box::new(out.encode_from_worker(all_cuda_tensors, tho_device)?))
    }

}
///
// A downstream crate can register its own wire type and lend the shared stage its
// original input, without implementing WorkerInput for a foreign generic type.
fn borrowed_worker_input<I: 'static, W: Borrow<I> + 'static>(input: &dyn WorkerInput) -> Result<&I, ThoError> {
    let input: &dyn Any = input;
    input.downcast_ref::<W>().map(Borrow::borrow).ok_or_else(|| ThoError::new(format!("worker expected input type {}", std::any::type_name::<W>())))
}

#[derive(Debug)]
pub struct WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: SyncStage + StageWithState + FuturesRunnerMapPurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    base:Arc<St>,
    pd:PhantomData<Co>,
    input_ref: for<'a> fn(&'a dyn WorkerInput) -> Result<&'a I, ThoError>,
}

impl <Co, St, I, O> WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: SyncStage + StageWithState + FuturesRunnerMapPurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    pub fn new(base: Arc<St>, pd: PhantomData<Co>) -> Self {
        Self { base, pd, input_ref: borrowed_worker_input::<I, I> }
    }

    /// Accept a crate-owned WorkerInput wrapper that borrows the shared stage input.
    /// The proxy's input cloner must send W; only I materializes tensor references.
    pub fn new_with_input<W>(base: Arc<St>, pd: PhantomData<Co>) -> Self
    where
        W: WorkerInput + Borrow<I>,
    {
        Self { base, pd, input_ref: borrowed_worker_input::<I, W> }
    }
}

impl <Co, St, I, O>SyncStageIpcWrapper for WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
    St: SyncStage + StageWithState + FuturesRunnerMapPurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
        // Keep this endpoint's concrete training-module state behind the erased worker capability.
        Some(self as &dyn ErasedWithState)
    }

    fn purge(&self, tho_device: ThoDevice) -> BoxFuture<'_, Result<(), ThoError>>
    where
        Self: Sync
    {
        debug!("SyncStageIpcWrapper>::purge: self.base={:?}", self.base);

        let base = Arc::clone(&self.base);
        async move {

            base.drop_gpu(tho_device.tho_device_idx(), "worker").await?;

            Ok(())
        }.boxed()
    }
}

impl <Co, St, I, O> SyncStageIpcWrapperWithState for WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: SyncStage + StageWithState + FuturesRunnerMapPurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput +WorkerOutput + 'static,
{
    type S = <St as StageWithState>::S;

    fn pre_run<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<tokio::sync::MappedMutexGuard<'a, Self::S>, ThoError>> {
        async {
            Ok(StageWithState::pre_run(self.base.as_ref(), tho_device.tho_device_idx()).await?.expect("no state 2"))//TODO error?
        }.boxed()
    }

    fn get_state_info<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<StageStateInfo, ThoError>> {
        StageWithState::get_state_info(self.base.as_ref(), tho_device.tho_device_idx())
    }

    fn get_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id: String) -> BoxFuture<'a, Result<Tensor, ThoError>> {
        StageWithState::get_state_tensor(self.base.as_ref(), tho_device.tho_device_idx(), tensor_id)
    }

    fn set_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id: String, t:Tensor) -> BoxFuture<'a, Result<(), ThoError>> {
        StageWithState::set_state_tensor(self.base.as_ref(), tho_device.tho_device_idx(), tensor_id, t)
    }

    fn do_run_with_state<'a>(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, state: tokio::sync::MappedMutexGuard<'_, Self::S>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
        debug!("SyncStageIpcWrapper::do_run_with_state: input={input:?}");

        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx());//TODO fixit
        let stage_ctx = St::Ctx::from_worker_ctx(&worker_ctx);

        let a = (self.input_ref)(input.as_ref())?;
        let out = Arc::clone(&self.base).do_run_with_state(&a.ref_to_tensor(all_cuda_tensors, false)?, state, &stage_ctx)?;

        debug!("SyncStageIpcWrapper::do_run_with_state: out={out:?}");

        Ok(Box::new(out.encode_from_worker(all_cuda_tensors, tho_device)?))
    }
}

///object-safe trait for running stages in CudaWorker
pub trait SyncStageIpcWrapper: Send + Sync + Debug {
    fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
        None
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcWrapper::do_run_response"))]
    fn do_run(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
        unimplemented!()
    }
    fn consume(&self, input: Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>{
        unimplemented!()
    }
    
    fn purge(&self, tho_device: ThoDevice) -> BoxFuture<'_, Result<(), ThoError>>//move to SyncStageIpcWrapperWithState
    {
        async move {
            Ok(())
        }.boxed()
    }
}

pub trait SyncStageIpcWrapperWithState: SyncStageIpcWrapper {
    type S: Debug + Send + Sync;

    fn pre_run<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<tokio::sync::MappedMutexGuard<'a, Self::S>, ThoError>>;

    fn get_state_info<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<StageStateInfo, ThoError>>;

    fn get_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id:String) -> BoxFuture<'a, Result<Tensor, ThoError>>;
    fn set_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id:String, t:Tensor) -> BoxFuture<'a, Result<(), ThoError>>;

    fn do_run_with_state<'a>(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, state: tokio::sync::MappedMutexGuard<'_, Self::S>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>;
}

pub trait ErasedWithState: Sync + Debug {

    fn do_run_with_state<'a>(&'a self, input: &'a Box<dyn WorkerInput>, all_cuda_tensors: &'a AllTensors, tho_device: &'a ThoDevice, stream: &'a ThoCudaStream, run_ctx: &'a WokerRunCtx) -> BoxFuture<'a, Result<Box<dyn WorkerOutput>, ThoError>>;
    fn get_state_info<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<StageStateInfo, ThoError>>;
    fn get_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id:String) -> BoxFuture<'a, Result<Tensor, ThoError>>;
    fn set_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id:String, t:Tensor) -> BoxFuture<'a, Result<(), ThoError>>;
}

impl<T> ErasedWithState for T
where
    T: SyncStageIpcWrapperWithState + 'static,
{

    fn do_run_with_state<'a>(&'a self, input: &'a Box<dyn WorkerInput>, all_cuda_tensors: &'a AllTensors, tho_device: &'a ThoDevice, stream: &'a ThoCudaStream, run_ctx: &'a WokerRunCtx) -> BoxFuture<'a, Result<Box<dyn WorkerOutput>, ThoError>> {
        async move {
            let state = SyncStageIpcWrapperWithState::pre_run(self, tho_device).await?;
            // Create the thread-affine stream guard after pre_run so it is never held across an await.
            let _guard = ThoStreamGuard::new(stream)?;
            let output = SyncStageIpcWrapperWithState::do_run_with_state(self, input, all_cuda_tensors, state, tho_device, run_ctx)?;
            // Match stateless worker execution by completing queued CUDA work before another stream can consume the returned tensor reference.
            stream.synchronize()?;
            Ok(output)
        }.boxed()
    }

    fn get_state_info<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<StageStateInfo, ThoError>> {
        async move {
            SyncStageIpcWrapperWithState::get_state_info(self, tho_device).await
        }.boxed()
    }

    fn get_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id: String) -> BoxFuture<'a, Result<Tensor, ThoError>> {
        async move {
            SyncStageIpcWrapperWithState::get_state_tensor(self, tho_device, tensor_id).await
        }.boxed()
    }

    fn set_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id: String, t:Tensor) -> BoxFuture<'a, Result<(), ThoError>> {
        async move {
            SyncStageIpcWrapperWithState::set_state_tensor(self, tho_device, tensor_id, t).await
        }.boxed()
    }
}

