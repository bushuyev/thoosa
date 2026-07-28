pub mod tho_tensor;

use std::collections::HashMap;
use std::fmt::Debug;
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;
use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use tokio::sync::{Mutex, MutexGuard};
use tch::Tensor;
use tracing::debug;
use thoosa_shared::ThoError;
use crate::ipc::codec::{encode_sync_stage_run_input, IpcCodec, IpcSyncStageRunInput, JsonIpcCodec};
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::runner::common::MaxParallelCtx;
use crate::runner::common::runnable::{Stage, SyncStage};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;

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

pub trait SyncStageIpcWrapper:Debug {
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcWrapper::do_run"))]
    fn do_run(&self, _input: &[u8], _all_cuda_tensors:&Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice) ->Result<Vec<u8>, ThoError> {
        Err(ThoError::new("IPC endpoint does not support do_run"))
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcWrapper::do_run_response"))]
    fn do_run_response(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<SyncStageIpcResponse, ThoError> {
        self.do_run(input, all_cuda_tensors, tho_device).map(SyncStageIpcResponse::Encoded)
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcWrapper::consume_response"))]
    fn consume_response(&self, _input: &[u8], _all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice) -> Result<SyncStageIpcConsumeResponse, ThoError> {
        Err(ThoError::new("IPC endpoint does not support consume"))
    }

    fn purge(&self, input: Vec<u8>, all_cuda_tensors: Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: ThoDevice) -> BoxFuture<Result<(), ThoError>>
    where Self: Sync
    {
        async move {
            Ok(())
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

pub fn encode_json_sync_stage_run_input(input: String, ctx: &RunCtxImpl) -> Result<String, ThoError> {
    let input = encode_sync_stage_run_input::<JsonIpcCodec>(input.as_bytes(), ctx)?;
    String::from_utf8(input).map_err(|e| ThoError::new(format!("json sync stage run input is not utf8: {e}")))
}

impl<C, S, I, O> SyncStageIpcWrapper for IpcSyncStageWrapper<C, S, I, O>
where
    C: IpcCodec,
    S: SyncStage + Debug + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = RunCtxImpl>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        debug!("IpcSyncStageWrapper::do_run: input.len={}", input.len());

        let (input, max_other_parallel) = match C::decode::<IpcSyncStageRunInput>(input) {
            Ok(input) => input.into_parts(),
            Err(e) => {
                debug!("IpcSyncStageWrapper::do_run: e={e:?}");

                (input.to_vec(), None)
            },
        };
        let input = I::decode_for_worker::<C>(&input, all_cuda_tensors)?;
        debug!("IpcSyncStageWrapper::do_run: input={input:?}");

        let mut ctx = RunCtxImpl::new(1, self.endpoint_name);
        if let Some(max_other_parallel) = max_other_parallel {
            ctx.set_max_other_parallel(max_other_parallel);
        }
        let out = Arc::clone(&self.sync_stage).do_run(&input, &ctx)?;
        debug!("IpcSyncStageWrapper::do_run: out={out:?}");
        out.encode_from_worker::<C>(all_cuda_tensors, tho_device)
    }

    fn consume_response(&self, _input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<SyncStageIpcConsumeResponse, ThoError> {
        let (input, max_other_parallel) = match C::decode::<IpcSyncStageRunInput>(_input) {
            Ok(input) => input.into_parts(),
            Err(_) => (_input.to_vec(), None),
        };
        let (input, consumed_tensor_keys) = I::decode_for_worker_consuming::<C>(&input, all_cuda_tensors)?;
        let mut ctx = RunCtxImpl::new(tho_device.tho_device_idx(), self.endpoint_name);
        if let Some(max_other_parallel) = max_other_parallel {
            ctx.set_max_other_parallel(max_other_parallel);
        }
        match Arc::clone(&self.sync_stage).consume(input, ctx) {
            crate::runner::common::runnable::ConsumeResult::Ok(output) => {
                let response = output.encode_from_worker::<C>(all_cuda_tensors, tho_device)?;
                Ok(SyncStageIpcConsumeResponse::encoded(response, consumed_tensor_keys))
            }
            crate::runner::common::runnable::ConsumeResult::Failed { error, .. } => Err(error),
        }
    }
}

pub fn make_ipc_sync_stage_wrapper<C, S, I, O>(endpoint_name: &'static str, sync_stage: S) -> Box<dyn SyncStageIpcWrapper>
where
    C: IpcCodec,
    S: SyncStage + Debug + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = RunCtxImpl>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    Box::new(IpcSyncStageWrapper::<C, S, I, O>::new(endpoint_name, sync_stage))
}

pub fn make_json_sync_stage_wrapper<S, I, O>(endpoint_name: &'static str, sync_stage: S) -> Box<dyn SyncStageIpcWrapper>
where
    S: SyncStage + Debug + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = RunCtxImpl>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    make_ipc_sync_stage_wrapper::<JsonIpcCodec, S, I, O>(endpoint_name, sync_stage)
}
