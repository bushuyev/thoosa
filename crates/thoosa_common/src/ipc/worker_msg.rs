use std::any::Any;
use std::fmt::Debug;
use serde::{Deserialize, Serialize};
use crate::ipc::cuda_wokrer::WorkerStageOutput;
use crate::runner::common::cachable::ThoDataRef;

#[typetag::serde(tag = "type")]
pub trait WorkerInput:Any + Send + Sync + Debug{

}

#[typetag::serde(tag = "type")]
pub trait WorkerOutput:Any + Send + Sync + Debug{

}


#[derive(Debug, Serialize, Deserialize)]
pub enum Operation {
    DescribeRef {
        tensor_key:u64
    },
    H2dStart,
    InstallStage{ endpoint_name:String, stage_id: u32, init_data:Box<dyn WorkerInput>},
    InvokeStage{ endpoint_name:String, stage_id: u32, input:Box<dyn WorkerInput>, consume:bool}
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
    Failed(String)
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaWorkerMsgOut {
    pub request_id:u64,
    pub status:OpResult

}