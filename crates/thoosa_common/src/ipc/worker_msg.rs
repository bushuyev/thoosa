use std::any::Any;
use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use thoosa_shared::ThoError;
use crate::ipc::cuda_wokrer::WorkerStageOutput;
use crate::ipc::ipc_cuda_utils::IpcTensorKind;
use crate::runner::common::cachable::ThoDataRef;

#[typetag::serde(tag = "type")]
pub trait WorkerInput:Any + Send + Sync + Debug{

}

#[typetag::serde(name = "void")]
impl WorkerInput for (){

}
#[typetag::serde(tag = "type")]
pub trait WorkerOutput:Any + Send + Sync + Debug{

}

#[derive(Debug, Serialize, Deserialize)]
pub struct DebugOutput {
    debug_data_snap:Vec<String>
}

impl DebugOutput {
    pub fn new(debug_data_snap: &Vec<String>) -> Self {
        Self { debug_data_snap:debug_data_snap.clone() }
    }

    pub fn debug_data_snap(&self) -> &Vec<String> {
        &self.debug_data_snap
    }
}

#[typetag::serde]
impl WorkerOutput for DebugOutput {

}


#[derive(Debug, Serialize, Deserialize)]
pub enum Operation {
    DescribeRef {
        tensor_key: u64
    },
    H2dCopy {
        kind: IpcTensorKind,
        shape: Vec<i64>,
        tensor_byte_len: usize,
        // The parent owns the shared allocator, so the worker addresses the live CPU tensor by its offset in the arena mapping.
        arena_offset: usize,
    },
    D2hCopy {
        tensor_key: u64,
        kind: IpcTensorKind,
        shape: Vec<i64>,
        tensor_byte_len: usize,
        // D2H writes into a CPU tensor preallocated by the parent because only the parent may allocate and free arena storage.
        arena_offset: usize,
    },
    DropThoTensor { tensor_key: u64 },
    InstallStage { endpoint_name: String, stage_id: u32, init_data: Box<dyn WorkerInput> },
    InvokeStage { endpoint_name: String, stage_id: u32, input: Box<dyn WorkerInput>, consume: bool },
    PurgeStage { endpoint_name: String, stage_id: u32 },
    DropStage { endpoint_name: String, stage_id: u32 },
    Debug,
    Shutdown {dummy: String},//without dummy empty message deserialized as Shutdown?
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaWorkerMsgIn {
    pub request_id:u64,
    pub operation: Operation
}

impl CudaWorkerMsgIn {
    pub fn new(request_id: u64, operation:Operation) -> Self {
        Self {
            request_id,
            operation
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum OpResult {
    Ok(Option<Box<dyn WorkerOutput>>),
    // Preserve retryability separately because not every worker OOM can be handled by an input/run-context adjuster.
    Failed{msg:String, is_oom:bool, is_retryable_oom: bool, needs_reset:bool}
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaWorkerMsgOut {
    pub request_id:u64,
    pub status:OpResult

}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WorkerCopyDonePayload {
    pub(crate) transfer_id: u64,
    pub(crate) tensor_ref: u64,
}

#[typetag::serde]
impl WorkerOutput for WorkerCopyDonePayload {}

impl OpResult {
    pub fn failed(msg:String)-> Self {
        OpResult::Failed {msg, is_oom:false, is_retryable_oom: false, needs_reset:false}
    }
}

impl From<ThoError> for OpResult {
    fn from(e: ThoError) -> Self {
        let is_oom = e.is_oom();
        let is_retryable_oom = e.is_retryable_oom();
        let needs_reset = e.needs_cuda_ctx_reset();
        OpResult::Failed {
            msg: e.msg,
            is_oom,
            is_retryable_oom,
            needs_reset,
        }
    }
}
