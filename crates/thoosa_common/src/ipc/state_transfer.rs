//! Backend state transport is independent of kernel execution. Only stages
//! installed with this capability can export/import worker tensors.
use futures_util::future::BoxFuture;
use serde::{Deserialize, Serialize};
use tch::Tensor;
use thoosa_shared::{ThoError, ipc_types::WorkerOutput};
use crate::ipc::tensor_info::WorkerTensorInfoPayload;

pub trait StateTransfer: Send + Sync + std::fmt::Debug {
    fn get_state_info(&self, device: usize) -> BoxFuture<'_, Result<StageStateInfo, ThoError>>;
    fn get_state_tensor(&self, device: usize, tensor_id: String) -> BoxFuture<'_, Result<Tensor, ThoError>>;
    fn set_state_tensor(&self, device: usize, tensor_id: String, tensor: Tensor) -> BoxFuture<'_, Result<(), ThoError>>;
}

// Checkpoint reset is a stronger capability than ordinary residency transfers.
pub trait CheckpointRestore: StateTransfer + std::fmt::Debug {
    fn reset_state(&self, info: StageStateInfo) -> BoxFuture<'_, Result<(), ThoError>>;
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
