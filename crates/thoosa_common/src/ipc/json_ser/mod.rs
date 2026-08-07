pub mod tho_tensor;

use crate::ipc::codec::{IpcCodec, IpcSyncStageRunInput, JsonIpcCodec, encode_sync_stage_run_input};
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
use tch::Tensor;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{Stage, SyncStage};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tokio::sync::MutexGuard;
use tracing::debug;

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

pub trait SyncStageIpcWrapper:Sync + Debug {
    fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
        None
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcWrapper::do_run"))]
    fn do_run(&self, _input: &[u8], _all_cuda_tensors:&Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice, _run_ctx: &WokerRunCtx) ->Result<Vec<u8>, ThoError> {
        Err(ThoError::new("IPC endpoint does not support do_run"))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcWrapper::do_run_response"))]
    fn do_run_response(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<SyncStageIpcResponse, ThoError> {
        self.do_run(input, all_cuda_tensors, tho_device, &run_ctx).map(SyncStageIpcResponse::Encoded)
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcWrapper::consume_response"))]
    fn consume_response(&self, _input: &[u8], _all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<SyncStageIpcConsumeResponse, ThoError> {
        Err(ThoError::new("IPC endpoint does not support consume"))
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

    fn do_run_with_state(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, state: MutexGuard<'_, Self::S>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<SyncStageIpcResponse, ThoError>;
}

pub trait ErasedWithState: Sync + Debug {
    fn do_run_with_state<'a>(&'a self, input: &'a [u8], all_cuda_tensors: &'a Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &'a ThoDevice, stream: &'a ThoCudaStream, run_ctx: &'a WokerRunCtx) -> BoxFuture<'a, Result<SyncStageIpcResponse, ThoError>>;
}

impl<T> ErasedWithState for T
where
    T: SyncStageIpcWrapperWithState + 'static,
{
    fn do_run_with_state<'a>(&'a self, input: &'a [u8], all_cuda_tensors: &'a Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &'a ThoDevice, stream: &'a ThoCudaStream, run_ctx: &'a WokerRunCtx) -> BoxFuture<'a, Result<SyncStageIpcResponse, ThoError>> {
        async move {
            let state = SyncStageIpcWrapperWithState::pre_run(self, tho_device).await?;
            // Create the thread-affine stream guard after pre_run so it is never held across an await.
            let _guard = ThoStreamGuard::new(stream)?;
            SyncStageIpcWrapperWithState::do_run_with_state(self, input, all_cuda_tensors, state, tho_device, run_ctx)
        }.boxed()
    }
}

#[derive(Debug)]
pub struct IpcSyncStageWrapper<C, S, I, O> {
    sync_stage: Arc<S>,
    endpoint_name: &'static str,
    pd: PhantomData<fn(C, I) -> O>,
}

impl<C, S, I, O> IpcSyncStageWrapper<C, S, I, O> {
    pub fn new(endpoint_name: &'static str, sync_stage: S) -> Self {
        Self {
            sync_stage: Arc::new(sync_stage),
            endpoint_name,
            pd: PhantomData,
        }
    }
}

pub type JsonSyncStageWrapper<S, I, O> = IpcSyncStageWrapper<JsonIpcCodec, S, I, O>;


impl<C, S, I, O> SyncStageIpcWrapper for IpcSyncStageWrapper<C, S, I, O>
where
    C: IpcCodec,
    // SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
    // must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = RunCtxImpl>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    fn do_run(&self, _input: &[u8], _all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice, _run_ctx: &WokerRunCtx) -> Result<Vec<u8>, ThoError> {
        debug!("IpcSyncStageWrapper::do_run: input.len={}", _input.len());

        let input = C::decode::<IpcSyncStageRunInput>(_input)?;
        let input = I::decode_for_worker::<C>(&input.input(), _all_cuda_tensors)?;
        debug!("IpcSyncStageWrapper::do_run: input={input:?}");

        let mut ctx = RunCtxImpl::new(1, self.endpoint_name);//TODO add CudaWOrkerRunCtx without max_other_parallel

        let out = Arc::clone(&self.sync_stage).do_run(&input, &ctx)?;
        debug!("IpcSyncStageWrapper::do_run: out={out:?}");
        out.encode_from_worker::<C>(_all_cuda_tensors, _tho_device)
    }

    fn consume_response(&self, _input: &[u8], _all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice, _ctx: &WokerRunCtx) -> Result<SyncStageIpcConsumeResponse, ThoError> {

        let (input, consumed_tensor_keys) = I::decode_for_worker_consuming::<C>(&C::decode::<IpcSyncStageRunInput>(_input)?.input(), _all_cuda_tensors)?;
        let mut ctx = RunCtxImpl::new(_tho_device.tho_device_idx(), self.endpoint_name);

        match Arc::clone(&self.sync_stage).consume(input, ctx) {
            thoosa_shared::traits::ConsumeResult::Ok(output) => {
                let response = output.encode_from_worker::<C>(_all_cuda_tensors, _tho_device)?;
                Ok(SyncStageIpcConsumeResponse::encoded(response, consumed_tensor_keys))
            }
            thoosa_shared::traits::ConsumeResult::Failed { error, .. } => Err(error),
        }
    }
}

pub fn make_ipc_sync_stage_wrapper<C, S, I, O>(endpoint_name: &'static str, sync_stage: S) -> Box<dyn SyncStageIpcWrapper>
where
    C: IpcCodec,
    S: SyncStage + Debug + Send + Sync + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = RunCtxImpl>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    Box::new(IpcSyncStageWrapper::<C, S, I, O>::new(endpoint_name, sync_stage))
}

pub fn make_json_sync_stage_wrapper<S, I, O>(endpoint_name: &'static str, sync_stage: S) -> Box<dyn SyncStageIpcWrapper>
where
    S: SyncStage + Debug + Send + Sync + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = RunCtxImpl>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    make_ipc_sync_stage_wrapper::<JsonIpcCodec, S, I, O>(endpoint_name, sync_stage)
}
