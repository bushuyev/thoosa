pub mod tho_tensor;

use crate::ipc::codec::{IpcCodec, IpcSyncStageRunInput, JsonIpcCodec};
use crate::ipc::cuda_wokrer::{WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use thoosa_shared::traits::MaxParallelCtx;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use std::collections::HashMap;
use std::fmt::Debug;
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tch::Tensor;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Stage, SyncStage};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tokio::sync::MutexGuard;
use tracing::debug;
use crate::ipc::worker_msg::{WorkerInput, WorkerOutput};

pub struct SyncStageIpcStreamResponse {
    write: Option<Box<dyn FnOnce(&mut dyn Write) -> Result<(), ThoError> + Send>>,
}

impl Debug for SyncStageIpcStreamResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncStageIpcStreamResponse").finish()
    }
}

impl SyncStageIpcStreamResponse {
    pub fn new(write: impl FnOnce(&mut dyn Write) -> Result<(), ThoError> + Send + 'static) -> Self {
        Self { write: Some(Box::new(write)) }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcStreamResponse::write_to"))]
    pub(crate) fn write_to(mut self, output: &mut dyn Write) -> Result<(), ThoError> {
        self.write.take().ok_or_else(|| ThoError::new("IPC stream response writer already consumed"))?(output)
    }
}

#[derive(Debug)]
pub enum SyncStageIpcResponse {
    Encoded(Vec<u8>),
    // Stream(SyncStageIpcStreamResponse),
}

#[derive(Debug)]
pub struct SyncStageIpcConsumeResponse {
    pub response: SyncStageIpcResponse,
    pub consumed_tensor_keys: Vec<u64>,
}

impl SyncStageIpcConsumeResponse {
    pub fn encoded(response: Vec<u8>, consumed_tensor_keys: Vec<u64>) -> Self {
        Self { response: SyncStageIpcResponse::Encoded(response), consumed_tensor_keys }
    }
}

pub trait SyncStageIpcWrapper: Send + Sync + Debug {
    fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
        None
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcWrapper::do_run_response"))]
    fn do_run(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
        todo!()
    }
    fn consume(&self, input: Box<dyn WorkerInput>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>{
        todo!()
    }


    fn purge(&self, tho_device: ThoDevice) -> BoxFuture<Result<(), ThoError>>
    {
        async move {
            Ok(())
        }.boxed()
    }
}

pub trait SyncStageIpcWrapperWithState: SyncStageIpcWrapper {
    type S: Debug + Send + Sync;

    fn pre_run<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<MutexGuard<'a, Self::S>, ThoError>>;


    fn do_run_with_state_2<'a>(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, state: MutexGuard<'_, Self::S>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>;
}

pub trait ErasedWithState: Sync + Debug {

    fn do_run_with_state_2<'a>(&'a self, input: &'a Box<dyn WorkerInput>, all_cuda_tensors: &'a Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &'a ThoDevice, stream: &'a ThoCudaStream, run_ctx: &'a WokerRunCtx) -> BoxFuture<'a, Result<Box<dyn WorkerOutput>, ThoError>>;
}

impl<T> ErasedWithState for T
where
    T: SyncStageIpcWrapperWithState + 'static,
{

    fn do_run_with_state_2<'a>(&'a self, input: &'a Box<dyn WorkerInput>, all_cuda_tensors: &'a Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &'a ThoDevice, stream: &'a ThoCudaStream, run_ctx: &'a WokerRunCtx) -> BoxFuture<'a, Result<Box<dyn WorkerOutput>, ThoError>> {
        async move {
            let state = SyncStageIpcWrapperWithState::pre_run(self, tho_device).await?;
            // Create the thread-affine stream guard after pre_run so it is never held across an await.
            let _guard = ThoStreamGuard::new(stream)?;
            SyncStageIpcWrapperWithState::do_run_with_state_2(self, input, all_cuda_tensors, state, tho_device, run_ctx)
        }.boxed()
    }
}


pub trait ToFromBytes {
    fn to_bytes<T>(value:&T) -> Result<Vec<u8>, ThoError>
    where
        T: Serialize + ?Sized;
    fn from_bytes<'a, T>(s: &'a [u8]) -> Result<T, ThoError>
    where
        T: Deserialize<'a>;
}

pub struct PostcardToFromBytes;

impl ToFromBytes for PostcardToFromBytes {

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="PostcardToFromBytes::to_bytes"))]
    fn to_bytes<T>(value: &T) -> Result<Vec<u8>, ThoError>
    where
        T: Serialize + ?Sized{
        postcard::to_allocvec(&value).map_err(|e|ThoError::new(e.to_string()))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="PostcardToFromBytes::from_bytes"))]
    fn from_bytes<'a, T>(s: &'a [u8]) -> Result<T, ThoError>
    where
        T: Deserialize<'a>
    {
        postcard::from_bytes::<T>(s).map_err(|e|ThoError::new(e.to_string()))
    }
}
