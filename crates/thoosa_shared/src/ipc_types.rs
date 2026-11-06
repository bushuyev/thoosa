use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Debug;

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum TensorKey {
    Data(u64), //all_cuda_tensors
    StageState { endpoint_name: String, stage_id: String, tensor_key: String },
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum H2DTensorKey {
    New, //all_cuda_tensors
    StageState { endpoint_name: String, stage_id: String, tensor_id: String },
}

impl From<u64> for TensorKey {
    fn from(value: u64) -> Self {
        TensorKey::Data(value)
    }
}

impl TensorKey {
    pub fn stage_state<S0: ToString, S1: ToString, S2: ToString>(endpoint_name: S0, stage_id: S1, tensor_key: S2) -> Self {
        TensorKey::StageState {
            endpoint_name: endpoint_name.to_string(),
            stage_id: stage_id.to_string(),
            tensor_key: tensor_key.to_string(),
        }
    }

    pub fn to_tensor_key(&self) -> String {
        match self {
            TensorKey::Data(v) => v.to_string(),
            TensorKey::StageState { tensor_key, .. } => tensor_key.clone(),
        }
    }
}

#[typetag::serde(tag = "type")]
pub trait WorkerOutput: Any + Send + Sync + Debug {}
