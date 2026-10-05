use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Debug;
use tch::Kind;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum IpcTensorKind {
    Uint8,
    Int8,
    Int16,
    Int32,
    Int64,
    Half,
    Float,
    Double,
    ComplexHalf,
    ComplexFloat,
    ComplexDouble,
    Bool,
    QInt8,
    QUInt8,
    QInt32,
    BFloat16,
    Float8e5m2,
    Float8e4m3fn,
    Float8e5m2fnuz,
    Float8e4m3fnuz,
}

impl From<Kind> for IpcTensorKind {
    fn from(value: Kind) -> Self {
        match value {
            Kind::Uint8 => Self::Uint8,
            Kind::Int8 => Self::Int8,
            Kind::Int16 => Self::Int16,
            Kind::Int => Self::Int32,
            Kind::Int64 => Self::Int64,
            Kind::Half => Self::Half,
            Kind::Float => Self::Float,
            Kind::Double => Self::Double,
            Kind::ComplexHalf => Self::ComplexHalf,
            Kind::ComplexFloat => Self::ComplexFloat,
            Kind::ComplexDouble => Self::ComplexDouble,
            Kind::Bool => Self::Bool,
            Kind::QInt8 => Self::QInt8,
            Kind::QUInt8 => Self::QUInt8,
            Kind::QInt32 => Self::QInt32,
            Kind::BFloat16 => Self::BFloat16,
            Kind::Float8e5m2 => Self::Float8e5m2,
            Kind::Float8e4m3fn => Self::Float8e4m3fn,
            Kind::Float8e5m2fnuz => Self::Float8e5m2fnuz,
            Kind::Float8e4m3fnuz => Self::Float8e4m3fnuz,
        }
    }
}

impl From<IpcTensorKind> for Kind {
    fn from(value: IpcTensorKind) -> Self {
        match value {
            IpcTensorKind::Uint8 => Kind::Uint8,
            IpcTensorKind::Int8 => Kind::Int8,
            IpcTensorKind::Int16 => Kind::Int16,
            IpcTensorKind::Int32 => Kind::Int,
            IpcTensorKind::Int64 => Kind::Int64,
            IpcTensorKind::Half => Kind::Half,
            IpcTensorKind::Float => Kind::Float,
            IpcTensorKind::Double => Kind::Double,
            IpcTensorKind::ComplexHalf => Kind::ComplexHalf,
            IpcTensorKind::ComplexFloat => Kind::ComplexFloat,
            IpcTensorKind::ComplexDouble => Kind::ComplexDouble,
            IpcTensorKind::Bool => Kind::Bool,
            IpcTensorKind::QInt8 => Kind::QInt8,
            IpcTensorKind::QUInt8 => Kind::QUInt8,
            IpcTensorKind::QInt32 => Kind::QInt32,
            IpcTensorKind::BFloat16 => Kind::BFloat16,
            IpcTensorKind::Float8e5m2 => Kind::Float8e5m2,
            IpcTensorKind::Float8e4m3fn => Kind::Float8e4m3fn,
            IpcTensorKind::Float8e5m2fnuz => Kind::Float8e5m2fnuz,
            IpcTensorKind::Float8e4m3fnuz => Kind::Float8e4m3fnuz,
        }
    }
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum TensorKey {
    Data(u64),//all_cuda_tensors
    StageState{endpoint_name:String, stage_id:String, tensor_id:String}
}

#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize)]
pub enum H2DTensorKey {
    New,//all_cuda_tensors
    StageState{endpoint_name:String, stage_id:String, tensor_id:String}
}

impl From<u64> for TensorKey {
    fn from(value: u64) -> Self {
        TensorKey::Data(value)
    }
}

impl TensorKey {
    pub fn stage_state<S0:ToString, S1:ToString, S2:ToString>(endpoint_name:S0, stage_id:S1, tensor_id:S2) -> Self {
        TensorKey::StageState {
            endpoint_name:endpoint_name.to_string(),
            stage_id:stage_id.to_string(),
            tensor_id:tensor_id.to_string()
        }
    }
}



#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerTensorInfoPayload {
    pub tensor_key: TensorKey,
    pub kind: IpcTensorKind,
    pub shape: Vec<i64>,
    pub tensor_byte_len: usize,
}

impl WorkerTensorInfoPayload {
    pub fn new(tensor_key: TensorKey, kind: IpcTensorKind, shape: Vec<i64>, tensor_byte_len: usize) -> Self {
        Self { tensor_key, kind, shape, tensor_byte_len }
    }
}

#[typetag::serde]
impl WorkerOutput for WorkerTensorInfoPayload {

}

#[typetag::serde(tag = "type")]
pub trait WorkerOutput:Any + Send + Sync + Debug{

}