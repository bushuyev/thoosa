pub mod tho_tensor;

use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::Mutex;
use tch::Tensor;
use tracing::debug;
use thoosa_shared::ThoError;
use crate::ipc::codec::{encode_sync_stage_run_input, IpcCodec, IpcSyncStageRunInput, JsonIpcCodec};
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::runner::common::MaxParallelCtx;
use crate::runner::common::runnable::{Stage, SyncStage};
use crate::runner::run_ctx_2::RunCtxImpl;

pub trait SyncStageIpcWrapper:Debug {
    fn do_run(&self, _input: &[u8], all_cuda_tensors:&Arc<Mutex<HashMap<u64, Tensor>>>) ->Result<Vec<u8>, ThoError>;
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
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Vec<u8>, ThoError> {
        let (input, max_other_parallel) = match C::decode::<IpcSyncStageRunInput>(input) {
            Ok(input) => input.into_parts(),
            Err(_) => (input.to_vec(), None),
        };
        let input = I::decode_for_worker::<C>(&input, all_cuda_tensors)?;
        let mut ctx = RunCtxImpl::new(1, self.endpoint_name);
        if let Some(max_other_parallel) = max_other_parallel {
            ctx.set_max_other_parallel(max_other_parallel);
        }
        let out = Arc::clone(&self.sync_stage).do_run(&input, &ctx)?;
        debug!("IpcSyncStageWrapper::do_run: out={out:?}");
        out.encode_from_worker::<C>(all_cuda_tensors)
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
