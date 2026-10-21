use crate::ipc::cuda_wokrer::WorkerStageOutput;
use crate::runner::common::cachable::ThoDataRef;
use log::warn;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::fmt::Debug;
use thoosa_shared::ipc_types::{H2DTensorKey, IpcTensorKind, TensorKey, WorkerOutput, WorkerTensorInfoPayload};
use thoosa_shared::traits::StageStateInfo;
use thoosa_shared::{CudaAction, SyncStageRunnerError, ThoError};

/// Numeric tensor and endpoint IDs are only meaningful in their originating
/// worker. A replacement process may allocate exactly the same IDs again.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CudaWorkerId {
    pub device_idx: usize,
    pub generation: thoosa_shared::CudaDeviceGeneration,
}

impl CudaWorkerId {
    pub fn check(self, expected: Self) -> Result<(), ThoError> {
        if self != expected {
            return Err(ThoError::ContextReset.context(format!("CUDA worker identity {self:?} does not match {expected:?}")));
        }
        Ok(())
    }
}

#[typetag::serde(tag = "type")]
pub trait WorkerInput: Any + Send + Sync + Debug {

}

#[typetag::serde(name = "void")]
impl WorkerInput for (){

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


// A batch has one export lease and one reply, while retaining each tensor's
// exact shape/type and its independently owned parent allocation.
#[derive(Debug, Serialize, Deserialize)]
pub struct StageStatePurgeDestination {
    pub info: WorkerTensorInfoPayload,
    pub arena_offset: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum Operation {
    DescribeRef {
        tensor_key: u64
    },
    H2dCopy {
        tensor_key: H2DTensorKey,
        kind: IpcTensorKind,
        shape: Vec<i64>,
        tensor_byte_len: usize,
        // The parent owns the shared allocator, so the worker addresses the live CPU tensor by its offset in the arena mapping.
        arena_offset: usize,
        dst_key: Option<H2DTensorKey>
    },
    D2hCopy {
        // State reads must belong to the export that froze this endpoint.
        export_id: Option<u64>,
        tensor_key: TensorKey,
        kind: IpcTensorKind,
        shape: Vec<i64>,
        tensor_byte_len: usize,
        // D2H writes into a CPU tensor preallocated by the parent because only the parent may allocate and free arena storage.
        arena_offset: usize,
    },
    DropThoTensor { tensor_key: u64 },
    InstallStage { endpoint_name: String, stage_id: Option<String>, init_data: Box<dyn WorkerInput> },
    InvokeStage { endpoint_name: String, stage_id: Option<String>, input: Box<dyn WorkerInput>, consume: bool },
    GetStageStateInfo { endpoint_name: String, stage_id: Option<String> },
    PurgeStage { endpoint_name: String, stage_id: Option<String> },//moves CudaWorker's StageWithState's state into thoosa_common::ipc::proxy::ProxyStage::state
    // Begin uses its request ID as the transaction identity; finish has its own request ID.
    BeginStageStatePurge { endpoint_name: String, stage_id: Option<String> },
    ExportStageTensors { endpoint_name: String, stage_id: Option<String>, export_id: u64, destinations: Vec<StageStatePurgeDestination> },
    FinishStageStatePurge { endpoint_name: String, stage_id: Option<String>, export_id: u64, commit: bool },
    ResetStageState { endpoint_name: String, stage_id: Option<String>, info: StageStateInfo },
    DropStage { endpoint_name: String, stage_id: Option<String> },
    Debug,
    Shutdown {dummy: String},//without dummy empty message deserialized as Shutdown?
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaWorkerMsgIn {
    pub worker_id: CudaWorkerId,
    pub request_id:u64,
    pub operation: Operation
}

impl CudaWorkerMsgIn {
    pub fn new(worker_id: CudaWorkerId, request_id: u64, operation:Operation) -> Self {
        Self {
            worker_id,
            request_id,
            operation
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub enum OpResult {
    Ok(Option<Box<dyn WorkerOutput>>),
    Failed(WorkerError),
}

// Serialize recovery semantics and diagnostics explicitly; anyhow itself is process-local.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WorkerError {
    kind: WorkerErrorKind,
    message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum WorkerErrorKind {//TODO remove?
    Cuda(CudaAction),
    AdjusterFailed,//TODO it can't happen in Worker
    NotFound,//TODO add what
    ContextReset,
    SubmitErrorClosed,
    SubmitErrorCancelled,
    Other,
    // Keep terminal recovery distinct from the restartable errors it contains.
    CudaRecoveryFailed { original: Box<WorkerError>, recovery: Option<Box<WorkerError>> },
    // Preserve both the OOM cause and the prohibition on replaying a partial update.
    TrainingSessionFailed { original: Box<WorkerError>, recovery: Option<Box<WorkerError>> },
}

impl WorkerErrorKind {
    fn from_error(error: &ThoError) -> Self {
        match error.what() {
            ThoError::Cuda { action, .. } => Self::Cuda(*action),
            ThoError::CudaRecoveryFailed(failure) => Self::CudaRecoveryFailed {
                original: Box::new(failure.original.clone().into()),
                recovery: failure.recovery.clone().map(|error| Box::new(error.into())),
            },
            ThoError::TrainingSessionFailed(failure) => Self::TrainingSessionFailed {
                original: Box::new(failure.original.clone().into()),
                recovery: failure.recovery.clone().map(|error| Box::new(error.into())),
            },
            ThoError::AdjusterFailed => Self::AdjusterFailed,
            ThoError::NotFound(what) => {
                warn!("WorkerErrorKind:NotFound: what={what:?}");
                Self::NotFound//TODO add what
            },//TODO add
            ThoError::ContextReset => Self::ContextReset,
            ThoError::SyncStageRunner(SyncStageRunnerError::SubmitErrorClosed) => Self::SubmitErrorClosed,
            ThoError::SyncStageRunner(SyncStageRunnerError::SubmitErrorCancelled) => Self::SubmitErrorCancelled,
            // Job IDs remain in the diagnostic; the parent needs the underlying recovery action.
            ThoError::SyncStageRunner(SyncStageRunnerError::JobError { inner, .. }) => Self::from_error(inner),
            ThoError::Anyhow(_) => Self::Other,
        }
    }
}

impl From<ThoError> for WorkerError {
    fn from(error: ThoError) -> Self {
        Self { kind: WorkerErrorKind::from_error(&error), message: format!("{error:#}") }
    }
}

impl WorkerError {
    pub(crate) fn requires_worker_exit(&self) -> bool {
        match &self.kind {
            WorkerErrorKind::TrainingSessionFailed { original, recovery } => original.requires_worker_exit() || recovery.as_ref().is_some_and(|error| error.requires_worker_exit()),
            _ => matches!(self.kind, WorkerErrorKind::Cuda(CudaAction::NeedsRestart) | WorkerErrorKind::CudaRecoveryFailed { .. }),
        }
    }
}

impl From<WorkerError> for ThoError {
    fn from(error: WorkerError) -> Self {
        warn!("<thoosa_shared::ThoError as core::convert::From>::from: error={error:?}");

        let WorkerError { kind, message } = error;
        match kind {
            WorkerErrorKind::Cuda(action) => Self::Cuda { action, msg: message },
            WorkerErrorKind::AdjusterFailed => Self::AdjusterFailed.context(message),
            WorkerErrorKind::NotFound => Self::NotFound(Some("worker".to_string())).context(message),
            WorkerErrorKind::ContextReset => Self::ContextReset.context(message),
            WorkerErrorKind::SubmitErrorClosed => Self::from(SyncStageRunnerError::SubmitErrorClosed).context(message),
            WorkerErrorKind::SubmitErrorCancelled => Self::from(SyncStageRunnerError::SubmitErrorCancelled).context(message),
            WorkerErrorKind::Other => Self::new(message),
            WorkerErrorKind::TrainingSessionFailed { original, recovery } => Self::TrainingSessionFailed(std::sync::Arc::new(thoosa_shared::TrainingSessionFailure {
                original: (*original).into(),
                recovery: recovery.map(|error| (*error).into()),
            })).context(message),
            WorkerErrorKind::CudaRecoveryFailed { original, recovery } => Self::CudaRecoveryFailed(std::sync::Arc::new(thoosa_shared::CudaRecoveryFailure {
                original: (*original).into(),
                recovery: recovery.map(|error| (*error).into()),
            })).context(message),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CudaWorkerMsgOut {
    pub worker_id: CudaWorkerId,
    pub request_id:u64,
    pub status:OpResult
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WorkerCopyDonePayload {
    pub(crate) transfer_id: u64,
    pub(crate) tensor_ref: TensorKey,
}

#[typetag::serde]
impl WorkerOutput for WorkerCopyDonePayload {}

impl OpResult {
    pub fn failed(msg:String)-> Self {
        ThoError::new(msg).into()
    }
}

impl From<ThoError> for OpResult {
    fn from(e: ThoError) -> Self {
        OpResult::Failed(e.into())
    }
}
