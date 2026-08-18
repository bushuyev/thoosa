use std::any::Any;
use std::fmt::Debug;
use serde::{Deserialize, Serialize};

#[typetag::serde(tag = "type")]
pub trait WorkerInput:Any + Send + Sync + Debug{
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Operation {
    InstallStage{key:String, init_data:Box<dyn WorkerInput>},
    InvokeStage
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaWorkerMsg {
    pub request_id:u64,
    pub operation:Operation
}

impl CudaWorkerMsg {
    pub fn new(request_id: u64, operation:Operation) -> Self {
        Self {
            request_id,
            operation
        }
    }
}