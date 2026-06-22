use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use tokio::sync::Mutex;
use tch::Tensor;
use thoosa_shared::ThoError;
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::runner::common::runnable::{Stage, SyncStage};
use crate::runner::run_ctx_2::RunCtxImpl;

pub trait SyncStageJsonWrapper:Debug {
    fn do_run(&self, _input:String, all_cuda_tensors:&Arc<Mutex<HashMap<u64, Tensor>>>) ->Result<String, ThoError>;
}

#[derive(Debug)]
pub(crate) struct JsonSyncStageWrapper<S, I, O> {
    sync_stage: Arc<S>,
    endpoint_name: &'static str,
    pd: PhantomData<fn(I) -> O>,
}

impl<S, I, O> JsonSyncStageWrapper<S, I, O> {
    pub fn new(endpoint_name: &'static str, sync_stage: S) -> Self {
        Self {
            sync_stage: Arc::new(sync_stage),
            endpoint_name,
            pd: PhantomData,
        }
    }
}

impl<S, I, O> SyncStageJsonWrapper for JsonSyncStageWrapper<S, I, O>
where
    S: SyncStage + Debug + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = RunCtxImpl>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    fn do_run(&self, input: String, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<String, ThoError> {
        let input = I::decode_for_worker(input, all_cuda_tensors)?;
        let ctx = RunCtxImpl::new(1, self.endpoint_name);
        let out = Arc::clone(&self.sync_stage).do_run(&input, &ctx)?;
        out.encode_from_worker(all_cuda_tensors)
    }
}

pub(crate) fn make_json_sync_stage_wrapper<S, I, O>(endpoint_name: &'static str, sync_stage: S) -> Box<dyn SyncStageJsonWrapper>
where
    S: SyncStage + Debug + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = RunCtxImpl>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    Box::new(JsonSyncStageWrapper::<S, I, O>::new(endpoint_name, sync_stage))
}