//! Synchronous stage contracts live with the tensor runtime because checkpoint
//! transfer owns LibTorch tensors; shared asynchronous stages stay tensor-independent.
use futures_util::{FutureExt, future::BoxFuture};
use serde::{Deserialize, Serialize};
use std::{fmt::Debug, sync::Arc};
use tch::Tensor;
use thoosa_shared::{ThoError, ipc_types::WorkerOutput, traits::Stage};
use crate::ipc::tensor_info::WorkerTensorInfoPayload;

pub trait SyncStage: Stage + Debug {

    // Remote execution waits for I/O and must bypass the local synchronous
    // stream boundary. Ordinary tensor stages keep the default local dispatch.
    fn as_async(&self) -> Option<&dyn AsyncStageDispatch<Self>>
    where Self: Sized,
    {
        None
    }

    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>>
    where
        Self: Sized,
    {
        None
    }

    fn do_run<'f>(self: Arc<Self>, _input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {//TODO remove _ctx - it's used for logging and in tests
        unimplemented!()
    }

    fn consume<'f>(
        self: Arc<Self>,
        _input: Self::In<'f>,
        _ctx: Self::Ctx,
    ) -> ConsumeResult<Self::In<'f>, Self::Out> {
        unimplemented!()
    }
}

// This capability keeps the input/output contract of a SyncStage while allowing
// actor helpers to await remote (IPC Mode) execution without blocking a runtime thread.
pub trait AsyncStageDispatch<S: SyncStage>: Sync {
    fn run<'a, 'i: 'a>(&self, stage: Arc<S>, input: &'a S::In<'i>, ctx: &'a S::Ctx) -> BoxFuture<'a, Result<S::Out, ThoError>>
    where S: 'a;

    fn consume<'a, 'i: 'a>(&self, stage: Arc<S>, input: S::In<'i>, ctx: S::Ctx) -> BoxFuture<'a, ConsumeResult<S::In<'i>, S::Out>>
    where S: 'a, S::In<'i>: Send + 'a;
}

pub trait StageWithState: SyncStage + Sync {//TODO rename to StatefulStage
    // State is accessed exclusively through a mutex guard, so it need only be Send.
    type S: Debug + Send;//State

    ///returns Option::None in IPC mode when it's used from ProxyStage where instead of copying from CPU to GPU it uploads StageState to the worker
    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Option<tokio::sync::MappedMutexGuard<'_, Self::S>>, ThoError>>;
    fn get_state_info(&self, tho_device: usize) -> BoxFuture<'_, Result<StageStateInfo, ThoError>> {
        async {
            Ok(StageStateInfo::empty())
        }.boxed()
    }

    // Install checkpoint metadata and discard live state before accepting its tensors.
    fn reset_state(&self, _info: StageStateInfo) -> BoxFuture<'_, Result<(), ThoError>> {
        Box::pin(async { Err(ThoError::new("stage does not support checkpoint restoration")) })
    }

    fn get_state_tensor(&self, tho_device: usize, tensor_id:String) -> BoxFuture<'_, Result<Tensor, ThoError>> {
        async {
            todo!()
        }.boxed()
    }

    fn set_state_tensor(&self, tho_device: usize, tensor_id:String, t:Tensor) -> BoxFuture<'_, Result<(), ThoError>> {
        async {
            todo!()
        }.boxed()
    }


    // The execution wrapper owns the guard until GPU synchronization completes;
    // stage implementations may borrow the state but must not release its lock.
    fn do_run_with_state<'f>(self: Arc<Self>, _input: &Self::In<'f>, _state: &mut Self::S, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        unimplemented!()
    }
}

pub trait ErasedStageState<S>: Send
where
    S: SyncStage,
{
    fn do_run_with_state<'f>(&mut self, stage: Arc<S>, input: &S::In<'f>, ctx: &S::Ctx) -> Result<S::Out, ThoError>;
}

#[derive(Debug)]
pub enum ConsumeResult<I, O> {
    Failed{input:I, error:ThoError},
    Ok(O)
}

///for keeping stages with different state types in the CudaWorker's installed stages map
pub trait ErasedStageWithState<S>
where
    S: SyncStage,
{
    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Option<Box<dyn ErasedStageState<S> + '_>>, ThoError>>;
}

impl<S> ErasedStageWithState<S> for S
where
    S: StageWithState,
{
    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Option<Box<dyn ErasedStageState<S> + '_>>, ThoError>> {
        async move {
            let state = StageWithState::pre_run(self, to_device_idx).await?;
            // Keep the concrete guard type inside the erased value while it spans later async preparation.
            Ok(state.map(|state|Box::new(StageStateGuard::<S> { state }) as Box<dyn ErasedStageState<S>>))
        }.boxed()
    }
}


struct StageStateGuard<'a, S>
where
    S: StageWithState,
{
    state: tokio::sync::MappedMutexGuard<'a, S::S>,
}

impl<S> ErasedStageState<S> for StageStateGuard<'_, S>
where
    S: StageWithState,
{
    fn do_run_with_state<'f>(&mut self, stage: Arc<S>, input: &S::In<'f>, ctx: &S::Ctx) -> Result<S::Out, ThoError> {
        // Keep ownership of the erased guard in the caller through synchronization.
        StageWithState::do_run_with_state(stage, input, &mut *self.state, ctx)
    }
}



#[derive(Debug, Serialize, Deserialize)]
pub struct StageStateInfo {
    pub scalars:Option<Box<dyn WorkerOutput>>,
    pub tensors:Vec<WorkerTensorInfoPayload>
}

impl StageStateInfo {
    pub fn new(tensors: Vec<WorkerTensorInfoPayload>, scalars: Option<Box<dyn WorkerOutput>>) -> Self {
        Self { scalars, tensors }
    }

    pub fn empty() -> Self {
        Self { scalars: None, tensors:Vec::new() }
    }


}

#[typetag::serde]
impl WorkerOutput for StageStateInfo {

}
