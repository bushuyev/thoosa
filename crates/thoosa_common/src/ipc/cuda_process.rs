use std::collections::HashMap;
use std::collections::HashSet;
use std::any::Any;
use std::error::Error;
use std::ffi::c_void;
use std::fmt::Debug;
use std::io::ErrorKind;
use std::mem;
use std::os::fd::{AsFd, OwnedFd};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::future::BoxFuture;
use log::{debug, error};
use serde::{Deserialize, Serialize};
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::Tensor;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex, RwLock};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tub::{Guard, Pool};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::{mono_key, Cause, ThoError};
use thoosa_shared::traits::{Data, Named};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::cuda_result::check_cuda_status;
use thoosa_shims::memcpy::{tho_cu_memcpy_dtoh_async, tho_cu_memcpy_htod_async};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use crate::copy_ctx::{CopyCtx, CudaActorHandle, ExecSyncStageActionResult};
use crate::cuda_utils::{copy_chunks, tensor_mem_size};
use crate::dims::Dims;
use crate::ipc::ipc_cuda_utils::{recv_fd, send_fd, CuMemHostRegistrationGuard, IpcProcessRole, IpcTensorKind, IpcTensorTransferMetadata, IpcTransferDirection, SharedMemfdBuffer};
use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster};
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::common::RunCtx;
use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::utils::tensor_digest;
#[cfg(test)]
use crate::torch_runner::tests::load_and_clc::clc::ClcStage;
#[cfg(test)]
use crate::torch_runner::tests::load_and_clc::loader::LoadedTensor;

pub(crate) const IPC_SOCK_ENV: &str = "THOOSA_WORKER_IPC_SOCK";
const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;
pub(crate) const SHARED_MEMFD_PAYLOAD_PREFIX: &str = "shared-memfd:";
const COPY_DONE_PAYLOAD_PREFIX: &str = "copy-done:";
const COPY_ERROR_PAYLOAD_PREFIX: &str = "copy-error:";
pub(crate) const DESCRIBE_REF_PAYLOAD_PREFIX: &str = "describe-ref:";
pub(super) const TENSOR_INFO_PAYLOAD_PREFIX: &str = "tensor-info:";
pub(crate) const H2D_START_PAYLOAD_PREFIX: &str = "h2d-start:";
pub(crate) const D2H_START_PAYLOAD_PREFIX: &str = "d2h-start:";
pub(crate) const INSTALL_ENDPOINT_PREFIX: &str = "install_endpoint:";
pub(crate) const INVOKE_ENDPOINT_PREFIX: &str = "invoke_endpoint:";
const WORKER_COPY_STREAMS: usize = 16;
pub(crate) const H2D_START_OK_PAYLOAD: &str = "h2d-start-ok";
pub(crate) const DEFAULT_ENDPOINT_NAME: &str = "SyncClcTask";
pub(crate) const ENDPOINT_OK_PAYLOAD: &str = "endpoint-ok";
pub(crate) const DROP_THO_TENSOR: &str = "drop_tho_tensor:";

pub(crate) fn encode_endpoint_payload(endpoint_name: &str, payload: &str) -> String {
    format!("{endpoint_name}\t{payload}")
}

pub(crate) fn split_endpoint_payload(payload: &str) -> Result<(&str, &str), ThoError> {
    payload.split_once('\t').map(|v|Ok(v)).unwrap_or_else(||Err(ThoError::new(format!("can't parse payload: {payload}"))))
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkerCopyDonePayload {
    transfer_id: u64,
    tensor_digest_hex: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkerCopyErrorPayload {
    msg: String,
    is_oom: bool,
    is_retryable_oom: bool,
    needs_cuda_ctx_reset: bool,
}

impl WorkerCopyErrorPayload {
    fn from_error(error: &ThoError) -> Self {
        Self {
            msg: error.to_string(),
            is_oom: error.is_oom(),
            is_retryable_oom: error.is_retryable_oom(),
            needs_cuda_ctx_reset: error.needs_cuda_ctx_reset(),
        }
    }

    fn into_error(self) -> ThoError {
        ThoError::new_with_cause(
            "CudaWoker happened in the CudaWorker",
            Cause::CudaWorker {
                msg: self.msg,
                is_oom: self.is_oom,
                is_retryable_oom: self.is_retryable_oom,
                needs_cuda_ctx_reset: self.needs_cuda_ctx_reset,
            },
        )
    }

    fn from_legacy_message(msg: &str) -> Self {
        let inferred = ThoError::new(msg.to_string());
        Self {
            msg: msg.to_string(),
            is_oom: inferred.is_oom(),
            is_retryable_oom: inferred.is_retryable_oom(),
            needs_cuda_ctx_reset: inferred.needs_cuda_ctx_reset(),
        }
    }
}

pub(crate) fn encode_worker_copy_error(error: &ThoError) -> String {
    let payload = WorkerCopyErrorPayload::from_error(error);
    match serde_json::to_string(&payload) {
        Ok(json) => format!("{COPY_ERROR_PAYLOAD_PREFIX}{json}"),
        Err(e) => format!("{COPY_ERROR_PAYLOAD_PREFIX}failed to encode worker error: {e}; original error: {error}"),
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WorkerTensorInfoPayload {
    tensor_key: u64,
    kind: IpcTensorKind,
    shape: Vec<i64>,
    tensor_byte_len: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WorkerH2dStartPayload {
    pub(crate) transfer_id: u64,
    pub(crate) worker_id: u64,
    pub(crate) kind: IpcTensorKind,
    pub(crate) shape: Vec<i64>,
    pub(crate) tensor_byte_len: usize,
    pub(crate) chunk_count: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct WorkerD2hStartPayload {
    pub(crate) transfer_id: u64,
    pub(crate) tensor_key: u64,
}

pub(crate) struct WorkerCudaResources {
    _context: Arc<ThoCudaContext>,
    stream_pool: Arc<Pool<ThoCudaStream>>,
}

impl WorkerCudaResources {
    fn new(cuda_device: usize) -> Result<Self, ThoError> {
        let context = Arc::new(ThoCudaContext::new(cuda_device as i32)?);
        context.bind_to_thread()?;
        let streams = (0..WORKER_COPY_STREAMS)
            .map(|_| ThoCudaStream::from_pool(cuda_device as i32, true))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            _context: context,
            stream_pool: Arc::new(Pool::from_iter(streams)),
        })
    }
}

pub(crate) struct WorkerH2dChunk {
    pub(crate) metadata: IpcTensorTransferMetadata,
    pub(crate) buffer: SharedMemfdBuffer,
}

pub(crate) struct PendingWorkerH2dTransfer {
    request_id: u64,
    metadata: Option<IpcTensorTransferMetadata>,
    destination: Option<Tensor>,
    received_bytes: usize,
    seen_chunks: HashSet<usize>,
    chunks: Vec<WorkerH2dChunk>,
}

impl PendingWorkerH2dTransfer {
    pub(crate) fn new(request_id: u64, chunk: WorkerH2dChunk) -> Result<Self, ThoError> {
        let mut transfer = Self {
            request_id,
            metadata: None,
            destination: None,
            received_bytes: 0,
            seen_chunks: HashSet::new(),
            chunks: Vec::new(),
        };
        transfer.push(chunk)?;
        Ok(transfer)
    }

    pub(crate) fn new_preflight(request_id: u64, destination: Tensor) -> Self {
        Self {
            request_id,
            metadata: None,
            destination: Some(destination),
            received_bytes: 0,
            seen_chunks: HashSet::new(),
            chunks: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, chunk: WorkerH2dChunk) -> Result<(), ThoError> {
        if self.request_id != chunk.metadata.transfer_id {
            return Err(ThoError::new(format!("H2D request id={} does not match transfer_id={}", self.request_id, chunk.metadata.transfer_id)));
        }
        if let Some(metadata) = &self.metadata {
            if metadata.kind != chunk.metadata.kind || metadata.shape != chunk.metadata.shape || metadata.tensor_byte_len != chunk.metadata.tensor_byte_len || metadata.chunk_count != chunk.metadata.chunk_count {
                return Err(ThoError::new(format!("inconsistent H2D chunk metadata for transfer {}", metadata.transfer_id)));
            }
        } else {
            self.metadata = Some(chunk.metadata.clone());
        }
        if !self.seen_chunks.insert(chunk.metadata.chunk_index) {
            return Err(ThoError::new(format!("duplicate H2D chunk index {} for transfer {}", chunk.metadata.chunk_index, chunk.metadata.transfer_id)));
        }
        self.received_bytes = self.received_bytes.checked_add(chunk.metadata.byte_len).ok_or_else(|| ThoError::new("H2D received byte count overflow"))?;
        if self.received_bytes > chunk.metadata.tensor_byte_len {
            return Err(ThoError::new(format!("H2D transfer {} received {} bytes but tensor has {} bytes", chunk.metadata.transfer_id, self.received_bytes, chunk.metadata.tensor_byte_len)));
        }
        self.chunks.push(chunk);
        Ok(())
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.metadata.as_ref().is_some_and(|metadata| self.received_bytes == metadata.tensor_byte_len && self.chunks.len() == metadata.chunk_count)
    }
}

pub(crate) struct WorkerD2hChunk {
    pub(crate) metadata: IpcTensorTransferMetadata,
    pub(crate) buffer: SharedMemfdBuffer,
}

pub(crate) struct PendingWorkerD2hTransfer {
    request_id: u64,
    tensor_key: u64,
    metadata: Option<IpcTensorTransferMetadata>,
    received_bytes: usize,
    seen_chunks: HashSet<usize>,
    chunks: Vec<WorkerD2hChunk>,
}

impl PendingWorkerD2hTransfer {
    pub(crate) fn new(request_id: u64, tensor_key: u64) -> Self {
        Self {
            request_id,
            tensor_key,
            metadata: None,
            received_bytes: 0,
            seen_chunks: HashSet::new(),
            chunks: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, chunk: WorkerD2hChunk) -> Result<(), ThoError> {
        if self.request_id != chunk.metadata.transfer_id {
            return Err(ThoError::new(format!("D2H request id={} does not match transfer_id={}", self.request_id, chunk.metadata.transfer_id)));
        }
        if let Some(metadata) = &self.metadata {
            if metadata.kind != chunk.metadata.kind || metadata.shape != chunk.metadata.shape || metadata.tensor_byte_len != chunk.metadata.tensor_byte_len || metadata.chunk_count != chunk.metadata.chunk_count {
                return Err(ThoError::new(format!("inconsistent D2H chunk metadata for transfer {}", metadata.transfer_id)));
            }
        } else {
            self.metadata = Some(chunk.metadata.clone());
        }
        if !self.seen_chunks.insert(chunk.metadata.chunk_index) {
            return Err(ThoError::new(format!("duplicate D2H chunk index {} for transfer {}", chunk.metadata.chunk_index, chunk.metadata.transfer_id)));
        }
        self.received_bytes = self.received_bytes.checked_add(chunk.metadata.byte_len).ok_or_else(|| ThoError::new("D2H received byte count overflow"))?;
        if self.received_bytes > chunk.metadata.tensor_byte_len {
            return Err(ThoError::new(format!("D2H transfer {} received {} bytes but tensor has {} bytes", chunk.metadata.transfer_id, self.received_bytes, chunk.metadata.tensor_byte_len)));
        }
        self.chunks.push(chunk);
        Ok(())
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.metadata.as_ref().is_some_and(|metadata| self.received_bytes == metadata.tensor_byte_len && self.chunks.len() == metadata.chunk_count)
    }
}

pub(crate) struct WorkerReply {
    pub(crate) id: u64,
    pub(crate) payload: OutgoingRequestPayload,
}


pub(crate) async fn worker_response_writer_loop<W>(
    mut writer: W,
    mut reply_rx: mpsc::Receiver<WorkerReply>,
) -> Result<(), ThoError>
where
    W: AsyncWrite + Unpin,
{
    while let Some(reply) = reply_rx.recv().await {
        let frame = encode_wire_msg(reply.id, &reply.payload)?;
        if let Err(error) = write_text_frame(&mut writer, &frame).await {
            if Error::source(&error)
                .and_then(|source| source.downcast_ref::<std::io::Error>())
                .is_some_and(|source| matches!(source.kind(), ErrorKind::BrokenPipe | ErrorKind::UnexpectedEof))
            {
                debug!("worker_response_writer_loop: parent IPC stream closed");
                break;
            }
            return Err(error);
        }
    }

    Ok(())
}

pub(crate) async fn describe_worker_tensor(
    all_cuda_tensors: Arc<Mutex<HashMap<u64, Tensor>>>,
    tensor_key: u64,
) -> Result<WorkerTensorInfoPayload, ThoError> {
    let tensors = all_cuda_tensors.lock().await;
    let tensor = tensors
        .get(&tensor_key)
        .ok_or_else(|| ThoError::new(format!("worker tensor key={tensor_key} not found")))?;
    if !tensor.is_contiguous() {
        return Err(ThoError::new(format!("worker tensor key={tensor_key} is not contiguous")));
    }
    let shape = tensor.size();
    Ok(WorkerTensorInfoPayload {
        tensor_key,
        kind: tensor.kind().into(),
        tensor_byte_len: tensor_mem_size(tensor.kind(), &shape),
        shape,
    })
}

fn worker_role_cuda_device(role: &IpcProcessRole, context: &str) -> Result<usize, ThoError> {
    let IpcProcessRole::Worker { worker_id } = role else {
        return Err(ThoError::new(format!("{context} must be worker, got {role:?}")));
    };
    let tho_device_idx = usize::try_from(*worker_id)
        .map_err(|_| ThoError::new(format!("worker id {worker_id} does not fit usize")))?;
    tho_device_idx
        .checked_sub(1)
        .ok_or_else(|| ThoError::new("worker id 0 cannot map to CUDA device"))
}

fn worker_cuda_device(metadata: &IpcTensorTransferMetadata) -> Result<usize, ThoError> {
    worker_role_cuda_device(&metadata.consumer, "H2D consumer")
}

fn worker_d2h_cuda_device(metadata: &IpcTensorTransferMetadata) -> Result<usize, ThoError> {
    worker_role_cuda_device(&metadata.producer, "D2H producer")
}

pub(crate) fn validate_worker_h2d_chunk(chunk: &WorkerH2dChunk) -> Result<usize, ThoError> {
    if !matches!(chunk.metadata.direction, IpcTransferDirection::HostToDevice) {
        return Err(ThoError::new(format!("worker H2D copy got direction {:?}", chunk.metadata.direction)));
    }
    if !matches!(chunk.metadata.producer, IpcProcessRole::Parent) {
        return Err(ThoError::new(format!("worker H2D copy got producer {:?}", chunk.metadata.producer)));
    }
    chunk.metadata.validate_for_buffer(&chunk.buffer)?;
    worker_cuda_device(&chunk.metadata)
}

pub(crate) fn validate_worker_d2h_chunk(chunk: &WorkerD2hChunk) -> Result<usize, ThoError> {
    if !matches!(chunk.metadata.direction, IpcTransferDirection::DeviceToHost) {
        return Err(ThoError::new(format!("worker D2H copy got direction {:?}", chunk.metadata.direction)));
    }
    if !matches!(chunk.metadata.producer, IpcProcessRole::Worker { .. }) {
        return Err(ThoError::new(format!("worker D2H copy got producer {:?}", chunk.metadata.producer)));
    }
    if !matches!(chunk.metadata.consumer, IpcProcessRole::Parent) {
        return Err(ThoError::new(format!("worker D2H copy got consumer {:?}", chunk.metadata.consumer)));
    }
    chunk.metadata.validate_for_buffer(&chunk.buffer)?;
    worker_d2h_cuda_device(&chunk.metadata)
}

pub(crate) fn get_or_create_worker_cuda_resources(
    resources: &mut HashMap<usize, Arc<WorkerCudaResources>>,
    cuda_device: usize,
) -> Result<Arc<WorkerCudaResources>, ThoError> {
    if let Some(resources) = resources.get(&cuda_device) {
        return Ok(Arc::clone(resources));
    }
    let created = Arc::new(WorkerCudaResources::new(cuda_device)?);
    resources.insert(cuda_device, Arc::clone(&created));
    Ok(created)
}

pub(crate) fn prepare_worker_h2d_transfer(
    start: WorkerH2dStartPayload,
    resources: &mut HashMap<usize, Arc<WorkerCudaResources>>,
) -> Result<PendingWorkerH2dTransfer, ThoError> {
    let cuda_device = worker_role_cuda_device(&IpcProcessRole::Worker { worker_id: start.worker_id }, "H2D preflight worker")?;
    if start.chunk_count == 0 {
        return Err(ThoError::new("H2D preflight chunk_count must be non-zero"));
    }
    let expected = tensor_mem_size(start.kind.into(), &start.shape);
    if expected != start.tensor_byte_len {
        return Err(ThoError::new(format!("H2D preflight tensor_byte_len={} but dtype/shape require {expected}", start.tensor_byte_len)));
    }
    if expected == 0 {
        return Err(ThoError::new("worker IPC H2D preflight does not support zero-byte tensors"));
    }

    let _resources = get_or_create_worker_cuda_resources(resources, cuda_device)?;
    let destination = Tensor::f_empty(start.shape.as_slice(), (start.kind.into(), Cuda(cuda_device)))?;

    Ok(PendingWorkerH2dTransfer::new_preflight(start.transfer_id, destination))
}

fn copy_worker_h2d_transfer_to_cuda(
    mut transfer: PendingWorkerH2dTransfer,
    stream: Guard<ThoCudaStream>,
) -> Result<(u64, Tensor, String), ThoError> {
    let metadata = transfer.metadata.as_ref().ok_or_else(|| ThoError::new(format!("H2D transfer {} has no metadata", transfer.request_id)))?.clone();
    let cuda_device = worker_cuda_device(&metadata)?;
    if stream.device() != cuda_device as i32 {
        return Err(ThoError::new(format!("worker copy stream device={} does not match cuda_device={cuda_device}", stream.device())));
    }

    transfer.chunks.sort_by_key(|chunk| chunk.metadata.tensor_offset);
    let mut expected_offset = 0usize;
    for chunk in &transfer.chunks {
        if chunk.metadata.tensor_offset != expected_offset {
            return Err(ThoError::new(format!("H2D transfer {} has gap or overlap at tensor offset {}; got chunk offset {}", metadata.transfer_id, expected_offset, chunk.metadata.tensor_offset)));
        }
        expected_offset = expected_offset.checked_add(chunk.metadata.byte_len).ok_or_else(|| ThoError::new("H2D expected tensor offset overflow"))?;
    }
    if expected_offset != metadata.tensor_byte_len {
        return Err(ThoError::new(format!("H2D transfer {} copied {expected_offset} bytes but tensor has {} bytes", metadata.transfer_id, metadata.tensor_byte_len)));
    }

    let _thread_context = ThoCudaContext::new(stream.device())?;
    let _stream_guard = ThoStreamGuard::new(&stream)?;
    let destination = match transfer.destination.take() {
        Some(destination) => {
            if destination.kind() != metadata.tch_kind() || destination.size() != metadata.shape {
                return Err(ThoError::new(format!("H2D preallocated destination shape/kind mismatch for transfer {}", metadata.transfer_id)));
            }
            destination
        }
        None => Tensor::f_empty(metadata.shape.as_slice(), (metadata.tch_kind(), Cuda(cuda_device)))?,
    };
    let dst_base = destination.data_ptr() as u64;

    for chunk in &transfer.chunks {
        let source = unsafe {
            // SAFETY: metadata.validate_for_buffer was run before insertion and sort does not mutate buffer metadata.
            chunk.buffer.as_ptr().add(chunk.metadata.offset)
        };
        let destination_offset = dst_base
            .checked_add(chunk.metadata.tensor_offset as u64)
            .ok_or_else(|| ThoError::new("H2D destination pointer overflow"))?;
        let _registration = CuMemHostRegistrationGuard::new(&chunk.buffer)?;
        let result = unsafe {
            // SAFETY: destination_offset points into the CUDA tensor byte range checked by metadata tensor offsets, source points into the registered live mmap, and the stream context is current on this OS thread.
            tho_cu_memcpy_htod_async(destination_offset, source.cast::<c_void>(), chunk.metadata.byte_len, stream.as_mut_ptr())
        };
        check_cuda_status(result, "cuMemcpyHtoDAsync(worker memfd chunk)")?;

    }
    stream.synchronize()?;

    let copied_cpu = destination.f_to_device(Cpu)?;
    let copied_digest = tensor_digest(&copied_cpu)?;
    debug!("copy_worker_h2d_transfer_to_cuda: copied tensor digest={copied_digest:?}");
    Ok((metadata.transfer_id, destination, copied_digest.hex))
}

pub(crate) fn spawn_worker_h2d_copy_task(
    transfer: PendingWorkerH2dTransfer,
    resources: Arc<WorkerCudaResources>,
    all_cuda_tensors: Arc<Mutex<HashMap<u64, Tensor>>>,
    reply_tx: mpsc::Sender<WorkerReply>,
) {
    tokio::spawn(async move {
        let request_id = transfer.request_id;
        let stream = resources.stream_pool.acquire().await;
        let copy_result = tokio::task::spawn_blocking(move || copy_worker_h2d_transfer_to_cuda(transfer, stream))
            .await
            .map_err(|e| ThoError::new(format!("worker H2D copy task join error: {e}")))
            .and_then(|result| result);

        let payload = match copy_result {
            Ok((transfer_id, tensor, tensor_digest_hex)) => {
                //TODO should tensors have their own id generator?
                all_cuda_tensors.lock().await.insert(transfer_id, tensor);
                let done = WorkerCopyDonePayload { transfer_id, tensor_digest_hex };
                match serde_json::to_string(&done) {
                    Ok(done_json) => OutgoingRequestPayload::Msg(format!("{COPY_DONE_PAYLOAD_PREFIX}{done_json}")),
                    Err(e) => OutgoingRequestPayload::Msg(encode_worker_copy_error(&ThoError::new(format!("failed to encode worker copy result for request {request_id}: {e}")))),
                }
            }
            Err(e) => OutgoingRequestPayload::Msg(encode_worker_copy_error(&e)),
        };

        let _ = reply_tx.send(WorkerReply { id: request_id, payload }).await;
    });
}

fn copy_worker_d2h_transfer_to_memfd(
    mut transfer: PendingWorkerD2hTransfer,
    stream: Guard<ThoCudaStream>,
    source: Tensor,
) -> Result<(u64, String), ThoError> {
    let metadata = transfer.metadata.as_ref().ok_or_else(|| ThoError::new(format!("D2H transfer {} has no metadata", transfer.request_id)))?.clone();
    let cuda_device = worker_d2h_cuda_device(&metadata)?;
    if stream.device() != cuda_device as i32 {
        return Err(ThoError::new(format!("worker D2H stream device={} does not match cuda_device={cuda_device}", stream.device())));
    }
    if source.kind() != metadata.tch_kind() {
        return Err(ThoError::new(format!("worker tensor key={} kind {:?} does not match D2H metadata {:?}", transfer.tensor_key, source.kind(), metadata.kind)));
    }
    if source.size() != metadata.shape {
        return Err(ThoError::new(format!("worker tensor key={} shape {:?} does not match D2H metadata {:?}", transfer.tensor_key, source.size(), metadata.shape)));
    }
    if !source.is_contiguous() {
        return Err(ThoError::new(format!("worker tensor key={} is not contiguous", transfer.tensor_key)));
    }
    let Cuda(source_cuda_device) = source.device() else {
        return Err(ThoError::new(format!("worker tensor key={} is not on CUDA: {:?}", transfer.tensor_key, source.device())));
    };
    if source_cuda_device != cuda_device {
        return Err(ThoError::new(format!("worker tensor key={} device={source_cuda_device} does not match D2H metadata cuda_device={cuda_device}", transfer.tensor_key)));
    }

    transfer.chunks.sort_by_key(|chunk| chunk.metadata.tensor_offset);
    let mut expected_offset = 0usize;
    for chunk in &transfer.chunks {
        if chunk.metadata.tensor_offset != expected_offset {
            return Err(ThoError::new(format!("D2H transfer {} has gap or overlap at tensor offset {}; got chunk offset {}", metadata.transfer_id, expected_offset, chunk.metadata.tensor_offset)));
        }
        expected_offset = expected_offset.checked_add(chunk.metadata.byte_len).ok_or_else(|| ThoError::new("D2H expected tensor offset overflow"))?;
    }
    if expected_offset != metadata.tensor_byte_len {
        return Err(ThoError::new(format!("D2H transfer {} copied {expected_offset} bytes but tensor has {} bytes", metadata.transfer_id, metadata.tensor_byte_len)));
    }

    let _thread_context = ThoCudaContext::new(stream.device())?;
    let _stream_guard = ThoStreamGuard::new(&stream)?;
    let src_base = source.data_ptr() as u64;
    for chunk in &mut transfer.chunks {
        let source_offset = src_base
            .checked_add(chunk.metadata.tensor_offset as u64)
            .ok_or_else(|| ThoError::new("D2H source pointer overflow"))?;
        let destination = unsafe {
            // SAFETY: validate_worker_d2h_chunk checked offset + byte_len against the live mmap before insertion.
            chunk.buffer.as_mut_ptr().add(chunk.metadata.offset)
        };
        let _registration = CuMemHostRegistrationGuard::new(&chunk.buffer)?;
        let result = unsafe {
            // SAFETY: source_offset points into the local CUDA tensor byte range checked by metadata tensor offsets, destination points into the registered live mmap, and the stream context is current on this OS thread.
            tho_cu_memcpy_dtoh_async(destination.cast::<c_void>(), source_offset, chunk.metadata.byte_len, stream.as_mut_ptr())
        };
        check_cuda_status(result, "cuMemcpyDtoHAsync(worker memfd chunk)")?;
    }
    stream.synchronize()?;

    let source_cpu = source.f_to_device(Cpu)?;
    let source_digest = tensor_digest(&source_cpu)?;
    debug!("copy_worker_d2h_transfer_to_memfd: source tensor digest={source_digest:?}");
    Ok((metadata.transfer_id, source_digest.hex))
}

pub(crate) fn spawn_worker_d2h_copy_task(
    transfer: PendingWorkerD2hTransfer,
    resources: Arc<WorkerCudaResources>,
    all_cuda_tensors: Arc<Mutex<HashMap<u64, Tensor>>>,
    reply_tx: mpsc::Sender<WorkerReply>,
) {
    tokio::spawn(async move {
        let request_id = transfer.request_id;
        let tensor_key = transfer.tensor_key;
        let source = {
            let tensors = all_cuda_tensors.lock().await;
            tensors.get(&tensor_key).map(Tensor::shallow_clone)
        };
        let copy_result = match source {
            Some(source) => {
                let stream = resources.stream_pool.acquire().await;
                tokio::task::spawn_blocking(move || copy_worker_d2h_transfer_to_memfd(transfer, stream, source))
                    .await
                    .map_err(|e| ThoError::new(format!("worker D2H copy task join error: {e}")))
                    .and_then(|result| result)
            }
            None => Err(ThoError::new(format!("worker tensor key={tensor_key} not found"))),
        };

        let payload = match copy_result {
            Ok((transfer_id, tensor_digest_hex)) => {
                let done = WorkerCopyDonePayload { transfer_id, tensor_digest_hex };
                match serde_json::to_string(&done) {
                    Ok(done_json) => OutgoingRequestPayload::Msg(format!("{COPY_DONE_PAYLOAD_PREFIX}{done_json}")),
                    Err(e) => OutgoingRequestPayload::Msg(encode_worker_copy_error(&ThoError::new(format!("failed to encode worker D2H result for request {request_id}: {e}")))),
                }
            }
            Err(e) => OutgoingRequestPayload::Msg(encode_worker_copy_error(&e)),
        };

        let _ = reply_tx.send(WorkerReply { id: request_id, payload }).await;
    });
}


pub(crate) async fn handle_worker_request(payload: String) -> String {
    debug!("handle_worker_request: payload={payload:?}");

    match payload.as_str() {
        "ping" => "pong".to_string(),

        "status" => "worker-status: alive".to_string(),

        msg if msg.starts_with("copy: ") => {
            let text = &msg["copy: ".len()..];

            format!("copy: {text} done")
        }


        other => {
            format!("unknown-command {other:?}")
        }
    }
}


pub enum CudaIPCActorMessage {
    Copy{ src:ThoTensor, dst_device:ThoDevice, respond_to: oneshot::Sender<Result<ThoTensor, ThoError>>},
    SyncStream {
        stream: Guard<ThoCudaStream>,
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    GetExecStream {
        respond_to: oneshot::Sender<Result<Guard<ThoCudaStream>, ThoError>>
    },
    DropThoTensor {
        key:u64,
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    Reset {
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    Shutdown {
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    InstallEndpoint {
        init_msg: String, respond_to: oneshot::Sender<Result<(), ThoError>>
    },

    InvokeEndpoint {
        input: String, respond_to: oneshot::Sender<Result<String, ThoError>>
    },
}

struct OutgoingRequest {
    id: u64,
    payload: OutgoingRequestPayload,
}

pub enum OutgoingRequestPayload {
    Msg(String),//action:data
    SharedMemfdBuffer{
        meta: IpcTensorTransferMetadata,
        fd:OwnedFd,
    },
}

#[derive(Debug)]
pub struct CudaIPCActor {
    receiver: mpsc::Receiver<CudaIPCActorMessage>,

    spawned_state: SpawnedState,
    next_id: Arc<AtomicU64>,
    copy_ctx:Arc<RwLock<CopyCtx>>,
    buf_size: usize,
    tho_device_idx: usize,
    cuda_worker_args: Arc<Vec<String>>
}

#[derive(Debug)]
struct SpawnedState {
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    tx:Sender<OutgoingRequest>,
    child:Child,
    writer_handle:JoinHandle<()>,
    reader_handle:JoinHandle<()>,
    cancel: CancellationToken,

}

impl SpawnedState {

    async fn shutdown(mut self, reason: impl Into<String>) {
        let reason = reason.into();

        // Tell reader/writer loops to stop.
        self.cancel.cancel();

        // All in-flight requests belong to the old child.
        // fail_all_pending(&self.pending, reason).await;

        // Close this sender. This only closes the rx if there are no other tx clones.
        drop(self.tx);

        // Make sure loops do not stay stuck in read/write forever.
        self.reader_handle.abort();
        self.writer_handle.abort();

        let _ = self.reader_handle.await;
        let _ = self.writer_handle.await;

        match self.child.try_wait() {
            Ok(Some(status)) => {
                debug!("worker already exited with status={status}");
            }

            Ok(None) => {
                if let Err(e) = self.child.kill().await {
                    error!("failed to kill worker process: {e}");
                }
            }

            Err(e) => {
                error!("failed to query worker process status: {e}");

                if let Err(e) = self.child.kill().await {
                    error!("failed to kill worker process after try_wait error: {e}");
                }
            }
        }
    }
}

impl CudaIPCActor {
    pub async fn new(tho_device_idx:usize, cuda_worker_args: Vec<&str>, receiver: mpsc::Receiver<CudaIPCActorMessage>) -> Result<Self, ThoError> {
        let cuda_worker_args = Arc::new(cuda_worker_args.into_iter().map(|s| s.to_string()).collect());

        let spawned_state = Self::spawn_worker_and_loops(&cuda_worker_args).await?;


        let copy_ctx = CopyCtx::make_instance_for_cuda_device(tho_device_idx - 1, 0.2, 16);
        let buf_size = copy_ctx.buf_size;

        Ok(Self {
            receiver,
            spawned_state,
            next_id: Arc::new(AtomicU64::new(1)),
            copy_ctx: Arc::new(RwLock::new(copy_ctx)),
            buf_size,
            tho_device_idx,
            cuda_worker_args,
        })
    }

    async fn spawn_worker_and_loops(cuda_worker_args: &Vec<String>) -> Result<SpawnedState, ThoError> {
        debug!("CudaIPCActor::spawn_worker_and_loops, cuda_worker_args={cuda_worker_args:?}");

        let (mut child, stream) = Self::spawn_worker(cuda_worker_args).await?;

        let (reader, writer) = stream.into_split();

        let (tx, rx): (Sender<OutgoingRequest>, Receiver<OutgoingRequest>) = mpsc::channel::<OutgoingRequest>(1024);

        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>> = Arc::new(Mutex::new(HashMap::new()));
        let cancel = CancellationToken::new();
        let writer_handle = tokio::spawn(writer_loop(writer, rx, Arc::clone(&pending), cancel.child_token()));
        let reader_handle = tokio::spawn(reader_loop(reader, Arc::clone(&pending), cancel.child_token()));

        debug!("CudaIPCActor::spawn_worker_and_loops, ok");

        Ok( SpawnedState {
            child,
            tx,
            pending,
            writer_handle,
            reader_handle,
            cancel
        })
    }

    async fn spawn_worker(args: &Vec<String>) -> Result<(Child, UnixStream), ThoError> {
        let exe = std::env::current_exe()?;

        debug!("CudaIPCActor::spawn_worker, exe={exe:?}");

        let dir = tempfile::tempdir()?;
        let sock_path = dir.path().join("worker.sock");

        debug!("CudaIPCActor::spawn_worker, dir={dir:?} sock_path={sock_path:?}");

        let listener = UnixListener::bind(&sock_path)?;

        debug!("CudaIPCActor::spawn_worker, listener={listener:?}");

        let mut command = &mut Command::new(exe);

        for arg in args {
            command = command.arg(arg);
        }

        debug!("CudaIPCActor::spawn_worker, command={command:?}");

        let mut child = command

            .env(IPC_SOCK_ENV, &sock_path)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        let accepted = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            listener.accept(),
        )
            .await;

        debug!("CudaIPCActor::spawn_worker: accepted={accepted:?}");

        let (mut stream, _addr) = match accepted {
            Ok(Ok(pair)) => pair,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => match child.try_wait()? {
                Some(status) => {
                    return Err(ThoError::new(format!("worker exited before connecting: status={status}")));
                }
                None => {
                    let _ = child.kill().await;
                    return Err(ThoError::new("worker did not connect within timeout"));
                }
            },
        };

        debug!("CudaIPCActor::spawn_worker: _addr={_addr:?}");

        let ready = read_text_frame(&mut stream).await?;
        debug!("CudaIPCActor::spawn_worker: main received: {ready:?}");
        assert_eq!(ready, "worker-ready");

        // The socket file can be removed after accept. The connected UnixStream
        // remains valid.
        drop(dir);

        Ok((child, stream))
    }

    fn handle_ref_d2h_copy(
        &mut self,
        tensor_key: u64,
        dst_device: ThoDevice,
        respond_to: oneshot::Sender<Result<ThoTensor, ThoError>>,
    ) -> Result<(), ThoError> {
        if dst_device.ordinal() != 0 {
            return Err(ThoError::new(format!("worker tensor ref copy currently supports only CPU destination, got dst_device={dst_device:?}")));
        }

        let describe_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let pending = Arc::clone(&self.spawned_state.pending);
        let self_tx = self.spawned_state.tx.clone();
        let copy_ctx = Arc::clone(&self.copy_ctx);
        let buf_size = self.buf_size;
        let worker_id = self.tho_device_idx as u64;

        tokio::spawn(async move {
            let result = async move {
                let info_reply = send_request_and_wait(
                    self_tx.clone(),
                    Arc::clone(&pending),
                    OutgoingRequest {
                        id: describe_id,
                        payload: OutgoingRequestPayload::Msg(format!("{DESCRIBE_REF_PAYLOAD_PREFIX}{tensor_key}")),
                    },
                ).await?;
                let info = parse_worker_tensor_info_payload(&info_reply)?;
                if info.tensor_key != tensor_key {
                    return Err(ThoError::new(format!("worker described tensor key={} but requested key={tensor_key}", info.tensor_key)));
                }

                let chunks = copy_chunks(info.tensor_byte_len, buf_size, "worker IPC D2H copy")?;
                let chunk_count = chunks.len();
                let pool = {
                    let copy_ctx = copy_ctx.read().await;
                    Arc::clone(copy_ctx.ipc_bufs_pool.as_ref().ok_or_else(|| ThoError::new("missing ipc buffer pool"))?)
                };

                let (reply_tx, reply_rx) = oneshot::channel();
                {
                    let mut pending = pending.lock().await;
                    let old = pending.insert(transfer_id, reply_tx);
                    debug_assert!(old.is_none());
                }

                let final_result = async {

                    let start_json = serde_json::to_string(&WorkerD2hStartPayload { transfer_id, tensor_key })
                        .map_err(|e| ThoError::new(format!("failed to encode D2H start payload: {e}")))?;

                    if self_tx.send(OutgoingRequest { id: transfer_id, payload: OutgoingRequestPayload::Msg(format!("{D2H_START_PAYLOAD_PREFIX}{start_json}")) }).await.is_err() {
                        return Err(ThoError::new("worker IPC writer task is closed"));
                    }

                    let mut held_chunks = Vec::with_capacity(chunk_count);
                    for chunk in chunks {
                        let buf_stream = pool.acquire().await;
                        if chunk.len > buf_stream.0.len() {
                            return Err(ThoError::new(format!("D2H chunk len={} exceeds shared buffer len={}", chunk.len, buf_stream.0.len())));
                        }
                        let metadata = IpcTensorTransferMetadata {
                            transfer_id,
                            buffer_id: buf_stream.0.buffer_id(),
                            byte_len: chunk.len,
                            tensor_byte_len: info.tensor_byte_len,
                            tensor_offset: chunk.offset,
                            chunk_index: chunk.index,
                            chunk_count,
                            mapped_len: buf_stream.0.len(),
                            offset: 0,
                            kind: info.kind,
                            shape: info.shape.clone(),
                            direction: IpcTransferDirection::DeviceToHost,
                            producer: IpcProcessRole::Worker { worker_id },
                            consumer: IpcProcessRole::Parent,
                        };
                        metadata.validate_for_buffer(&buf_stream.0)?;
                        debug!("CudaIPCActor::Copy D2H: sending shared memfd chunk to worker: chunk={chunk:?} metadata={metadata:?} local_mmap_ptr={:?}", buf_stream.0.as_ptr());

                        let fd = buf_stream.0.try_clone_fd()?;
                        if self_tx.send(OutgoingRequest { id: transfer_id, payload: OutgoingRequestPayload::SharedMemfdBuffer { meta: metadata.clone(), fd } }).await.is_err() {
                            return Err(ThoError::new("worker IPC writer task is closed"));
                        }
                        held_chunks.push(ParentD2hChunk { metadata, buffer: buf_stream });
                    }

                    let reply = reply_rx
                        .await
                        .map_err(|_| ThoError::new(format!("response channel for D2H request {transfer_id} was dropped")))??;
                    Ok::<_, ThoError>((reply, held_chunks))
                }.await;

                if final_result.is_err() {
                    remove_pending_request(&pending, transfer_id).await;
                }
                let (reply, held_chunks) = final_result?;
                let done = parse_worker_copy_done_payload(&reply)?;
                if done.transfer_id != transfer_id {
                    return Err(ThoError::new(format!("worker D2H response transfer_id={} but expected {transfer_id}", done.transfer_id)));
                }
                let cpu_tensor = assemble_parent_d2h_chunks(held_chunks, &info)?;
                let cpu_digest = tensor_digest(&cpu_tensor)?;
                if cpu_digest.hex != done.tensor_digest_hex {
                    return Err(ThoError::new(format!("worker D2H digest mismatch for transfer {transfer_id}: worker={} parent={}", done.tensor_digest_hex, cpu_digest.hex)));
                }
                debug!("CudaIPCActor::Copy D2H: worker and parent tensor digests matched: transfer_id={transfer_id} digest={}", cpu_digest.hex);

                ThoTensor::new((), cpu_tensor)
            }.await;

            let _ = respond_to.send(result);
        });

        Ok(())
    }

    async fn handle_message(&mut self, msg: CudaIPCActorMessage) -> Result<(), ThoError>{
        match msg {
            CudaIPCActorMessage::Copy { src, dst_device, respond_to } => {
                if let Some(tensor_key) = src.ref_key() {
                    return self.handle_ref_d2h_copy(tensor_key, dst_device, respond_to);
                }

                let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let preflight_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let pending = Arc::clone(&self.spawned_state.pending);
                let self_tx = self.spawned_state.tx.clone();
                let copy_ctx = Arc::clone(&self.copy_ctx);
                let buf_size = self.buf_size;
                let dst_worker_id = dst_device.ordinal() as u64;

                let tensor_mem_size = tensor_mem_size(src.t_ref().kind(), &src.t_ref().size());
                debug!("CudaIPCActor::handle_message: src_t.size={:?} tensor_mem_size={tensor_mem_size} buf_size={buf_size}",src.t_ref().size());
                if tensor_mem_size == 0 {
                    return Err(ThoError::new("worker IPC H2D copy does not support zero-byte tensors"));
                }

                // The parent side of H2D only stages CPU bytes into memfd-backed chunks; the worker owns the later CUDA copy.
                let chunk_groups = copy_chunks(tensor_mem_size, buf_size, "copy_h2d_from_ptr")?.into_iter().map(|chunk| vec![chunk]).collect::<Vec<_>>();
                let chunk_count = chunk_groups.iter().map(Vec::len).sum::<usize>();
                let src_kind = src.t_ref().kind();
                let src_shape = src.t_ref().size();
                let src_ptr: usize = src.t_ref().data_ptr() as usize;

                tokio::spawn(async move {
                    let result = async move {
                        let _src_keepalive = src;

                        let preflight_json = serde_json::to_string(&WorkerH2dStartPayload {
                            transfer_id,
                            worker_id: dst_worker_id,
                            kind: src_kind.into(),
                            shape: src_shape.clone(),
                            tensor_byte_len: tensor_mem_size,
                            chunk_count,
                        }).map_err(|e| ThoError::new(format!("failed to encode H2D start payload: {e}")))?;

                        let preflight_reply = send_request_and_wait(
                            self_tx.clone(),
                            Arc::clone(&pending),
                            OutgoingRequest {
                                id: preflight_id,
                                payload: OutgoingRequestPayload::Msg(format!("{H2D_START_PAYLOAD_PREFIX}{preflight_json}")),
                            },
                        ).await?;
                        parse_worker_h2d_start_payload(&preflight_reply)?;

                        let source_digest = tensor_digest(_src_keepalive.t_ref())?;
                        debug!("CudaIPCActor::handle_message, CudaIPCActor::Copy: original tensor digest={source_digest:?}");
                        let source_digest_hex = source_digest.hex.clone();

                        {
                            let mut pending = pending.lock().await;
                            let old = pending.insert(transfer_id, reply_tx);
                            debug_assert!(old.is_none());
                        }

                        let final_result = async {
                            let mut held_buffers = Vec::new();
                            for chunk_group in chunk_groups {
                                for chunk in chunk_group {
                                    let mut buf_stream = copy_ctx.read().await.ipc_bufs_pool.as_ref().unwrap().acquire().await;
                                    let src_off_ptr = src_ptr.checked_add(chunk.offset).ok_or_else(|| ThoError::new("source tensor pointer offset overflow"))?;
                                    if chunk.len > buf_stream.0.len() {
                                        return Err(ThoError::new(format!("copy chunk len={} exceeds shared buffer len={}", chunk.len, buf_stream.0.len())));
                                    }
                                    unsafe {
                                        // SAFETY: src_ptr is the data pointer of _src_keepalive, chunk comes from copy_chunks over the tensor byte size, and the destination mmap range was checked against the buffer length above.
                                        std::ptr::copy_nonoverlapping(src_off_ptr as *const u8, buf_stream.0.as_mut_ptr(), chunk.len);
                                    }
                                    let metadata = IpcTensorTransferMetadata {
                                        transfer_id,
                                        buffer_id: buf_stream.0.buffer_id(),
                                        byte_len: chunk.len,
                                        tensor_byte_len: tensor_mem_size,
                                        tensor_offset: chunk.offset,
                                        chunk_index: chunk.index,
                                        chunk_count,
                                        mapped_len: buf_stream.0.len(),
                                        offset: 0,
                                        kind: src_kind.into(),
                                        shape: src_shape.clone(),
                                        direction: IpcTransferDirection::HostToDevice,
                                        producer: IpcProcessRole::Parent,
                                        consumer: IpcProcessRole::Worker { worker_id: dst_worker_id },
                                    };
                                    debug!("CudaIPCActor::handle_message, CudaIPCActor::Copy: copied tensor chunk to shared memfd: chunk={chunk:?} metadata={metadata:?} local_mmap_ptr={:?}", buf_stream.0.as_ptr());

                                    let fd = buf_stream.0.try_clone_fd()?;
                                    if self_tx.send(OutgoingRequest { id: transfer_id, payload: OutgoingRequestPayload::SharedMemfdBuffer { meta: metadata, fd } }).await.is_err() {
                                        return Err(ThoError::new("worker IPC writer task is closed"));
                                    }
                                    held_buffers.push(buf_stream);
                                }
                            }

                            let id = reply_rx
                                .await
                                .map_err(|_| ThoError::new(format!("response channel for request {transfer_id} was dropped")))??;

                            let done = parse_worker_copy_done_payload(&id)?;
                            if done.tensor_digest_hex != source_digest_hex {
                                return Err(ThoError::new(format!("worker copy digest mismatch for transfer {}: source={} copied={}", done.transfer_id, source_digest_hex, done.tensor_digest_hex)));
                            }
                            debug!("CudaIPCActor::handle_message, CudaIPCActor::Copy: worker copied tensor digest matched: transfer_id={} digest={}", done.transfer_id, done.tensor_digest_hex);

                            Ok::<_, ThoError>(ThoTensor::new_ref((), transfer_id/*it's used as key for all_cuda_tensorsat crates/thoosa_common/src/torch_runner/cuda_process.rs:370 */, vec![], tch::Kind::Float))
                        }.await;

                        if final_result.is_err() {
                            remove_pending_request(&pending, transfer_id).await;
                        }

                        final_result
                    }.await;

                    let _ = respond_to.send(result);
                });

            }
            CudaIPCActorMessage::DropThoTensor { key, respond_to } => {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let pending = Arc::clone(&self.spawned_state.pending);
                let self_tx = self.spawned_state.tx.clone();

                tokio::spawn(async move {
                    {
                        let mut pending = pending.lock().await;
                        let old = pending.insert(id, reply_tx);
                        debug_assert!(old.is_none());
                    }

                    if self_tx.send(OutgoingRequest { id, payload: OutgoingRequestPayload::Msg(encode_endpoint_payload(DROP_THO_TENSOR, &key.to_string())) }).await.is_err() {
                        let mut pending = pending.lock().await;
                        pending.remove(&id);
                        let _ = respond_to.send(Err(ThoError::new("worker IPC writer task is closed")));
                        return;
                    }

                    let reply = reply_rx
                        .await
                        .map_err(|_| ThoError::new(format!("response channel for drop request {id} was dropped")))
                        .and_then(|reply| reply);

                    match reply {
                        Ok(reply) if reply == "Ok" => {
                            let _ = respond_to.send(Ok(()));
                        }
                        Ok(reply) => {
                            let _ = respond_to.send(Err(ThoError::new(format!("unexpected drop response: {reply:?}"))));
                        }
                        Err(error) => {
                            let _ = respond_to.send(Err(error));
                        }
                    }
                });
            }
            CudaIPCActorMessage::SyncStream { stream, respond_to } => {
                tokio::spawn(async move {
                    let result = tokio::task::spawn_blocking(move || stream.synchronize())
                        .await
                        .map_err(|e| ThoError::new(format!("IPC exec stream sync task join error: {e}")))
                        .and_then(|result| result);
                    let _ = respond_to.send(result);
                });
            }
            CudaIPCActorMessage::GetExecStream { respond_to } => {
                let copy_ctx = Arc::clone(&self.copy_ctx);
                tokio::spawn(async move {
                    let stream = copy_ctx.read().await.acquire_exec_stream().await;
                    let _ = respond_to.send(Ok(stream));
                });
            }
            CudaIPCActorMessage::Reset { respond_to } => {
                debug!("CudaIPCActor::handle_message, CudaIPCActorMessage::Reset");
                let self_arc = Arc::clone(&self.cuda_worker_args);


                let mut old_pawned_state = mem::replace(&mut self.spawned_state, Self::spawn_worker_and_loops(&self_arc).await?);
                old_pawned_state.shutdown("ups").await;

                debug!("CudaIPCActor::handle_message, CudaIPCActorMessage::Reset done");

                let _ = respond_to.send(Ok(()));
            }
            CudaIPCActorMessage::Shutdown { respond_to } => {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let pending = Arc::clone(&self.spawned_state.pending);
                let self_tx = self.spawned_state.tx.clone();

                tokio::spawn(async move {
                    {
                        let mut pending = pending.lock().await;
                        let old = pending.insert(id, reply_tx);
                        debug_assert!(old.is_none());
                    }

                    if self_tx.send(OutgoingRequest { id, payload: OutgoingRequestPayload::Msg("shutdown".to_string()) }).await.is_err() {
                        let mut pending = pending.lock().await;
                        pending.remove(&id);
                        let _ = respond_to.send(Err(ThoError::new("worker IPC writer task is closed")));
                        return;
                    }

                    let reply = reply_rx
                        .await
                        .map_err(|_| ThoError::new(format!("response channel for shutdown request {id} was dropped")))
                        .and_then(|reply| reply);

                    match reply {
                        Ok(reply) if reply == "shutdown-ok" => {
                            let _ = respond_to.send(Ok(()));
                        }
                        Ok(reply) => {
                            let _ = respond_to.send(Err(ThoError::new(format!("unexpected shutdown response: {reply:?}"))));
                        }
                        Err(error) => {
                            let _ = respond_to.send(Err(error));
                        }
                    }
                });
            }
            CudaIPCActorMessage::InstallEndpoint { init_msg, respond_to } => {
                let self_tx = self.spawned_state.tx.clone();
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.spawned_state.pending);

                tokio::spawn(async move {

                    let response = send_request_and_wait(
                        self_tx,
                        pending,
                        OutgoingRequest {
                            id: next_id,
                            payload: OutgoingRequestPayload::Msg(format!("{INSTALL_ENDPOINT_PREFIX}{init_msg}")),
                        },
                    ).await.and_then(|response| {
                        if let Some(error) = response.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
                            Err(parse_worker_copy_error_payload(error))
                        } else if response == ENDPOINT_OK_PAYLOAD {
                            Ok(())
                        } else {
                            Err(ThoError::new(format!("unexpected endpoint install response: {response:?}")))
                        }
                    });
                    let _ = respond_to.send(response);

                    Ok::<_, ThoError>(())
                });
            }
            CudaIPCActorMessage::InvokeEndpoint { input, respond_to } => {
                let self_tx = self.spawned_state.tx.clone();
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.spawned_state.pending);

                tokio::spawn(async move {

                    let response = send_request_and_wait(
                        self_tx,
                        pending,
                        OutgoingRequest {
                            id: next_id,
                            payload: OutgoingRequestPayload::Msg(format!("{INVOKE_ENDPOINT_PREFIX}{input}")),
                        },
                    ).await.and_then(|response| {
                        if let Some(error) = response.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
                            Err(parse_worker_copy_error_payload(error))
                        } else {
                            Ok(response)
                        }
                    });
                    let _ = respond_to.send(response);

                    Ok::<_, ThoError>(())
                });
            }

        }

        Ok(())
    }

    async fn run(mut actor: CudaIPCActor) -> Result<(), ThoError> {
        while let Some(msg) = actor.receiver.recv().await {
            actor.handle_message(msg).await?;
        }

        Ok(())
    }
}

async fn writer_loop<W>(
    mut writer: W,
    mut rx: mpsc::Receiver<OutgoingRequest>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    cancel: CancellationToken,
) where
    W: AsyncWrite + Unpin + AsRef<UnixStream>,
{
    loop {
        let req = tokio::select! {
            _ = cancel.cancelled() => {
                break;
            }

            maybe_req = rx.recv() => {
                match maybe_req {
                    Some(req) => req,
                    None => break,
                }
            }
        };

        let frame = match encode_wire_msg(req.id, &req.payload) {
            Ok(frame) => frame,
            Err(e) => {
                error!("worker IPC writer failed to encode request {}: {e}", req.id);
                fail_one_pending(&pending, req.id, format!("failed to encode request {}: {e}", req.id)).await;
                continue;
            }
        };

        if let Err(e) = write_text_frame(&mut writer, &frame).await {
            error!("worker IPC writer failed: {e}");

            fail_one_pending(
                &pending,
                req.id,
                format!("failed to write request {}: {e}", req.id),
            )
                .await;

            fail_all_pending(&pending, format!("worker IPC writer stopped: {e}")).await;
            break;
        }

        if let OutgoingRequestPayload::SharedMemfdBuffer { fd, .. } = &req.payload {
            if let Err(e) = send_fd(writer.as_ref(), fd.as_fd()).await {
                error!("worker IPC fd send failed: {e}");
                fail_one_pending(&pending, req.id, format!("failed to send request {} fd: {e}", req.id)).await;
                fail_all_pending(&pending, format!("worker IPC writer stopped while sending fd: {e}")).await;
                break;
            }
        }
    }

    debug!("worker IPC writer loop finished");
}

async fn reader_loop<R>(
    mut reader: R,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    cancel: CancellationToken,
) where
    R: AsyncRead + Unpin,
{
    loop {
        let frame = tokio::select! {
            _ = cancel.cancelled() => {
                break;
            }

            result = read_text_frame(&mut reader) => {
                match result {
                    Ok(frame) => frame,
                    Err(e) => {
                        debug!("worker IPC reader stopped: {e}");

                        fail_all_pending(
                            &pending,
                            format!("worker IPC reader stopped: {e}"),
                        )
                        .await;

                        break;
                    }
                }
            }
        };


        let (id, payload) = match parse_wire_msg(&frame) {
            Ok(parsed) => parsed,
            Err(e) => {
                error!("bad worker response frame {frame:?}: {e}");
                continue;
            }
        };

        let reply_tx = {
            let mut pending = pending.lock().await;
            pending.remove(&id)
        };

        match reply_tx {
            Some(reply_tx) => {
                let _ = reply_tx.send(Ok(payload));
            }
            None => {
                error!("received response for unknown request id={id}, payload={payload:?}");
            }
        }
    }

    debug!("worker IPC reader loop finished");
}

pub(crate) fn parse_wire_msg(frame: &str) -> Result<(u64, String), ThoError> {
    let Some((id, payload)) = frame.split_once('\t') else {
        return Err(ThoError::new(format!("bad wire frame without request id separator: {frame:?}")));
    };

    let id = id
        .parse::<u64>()
        .map_err(|e| ThoError::new(format!("bad request id {id:?}: {e}")))?;

    Ok((id, payload.to_string()))
}

fn parse_worker_copy_done_payload(payload: &str) -> Result<WorkerCopyDonePayload, ThoError> {
    if let Some(error) = payload.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
        return Err(parse_worker_copy_error_payload(error));
    }
    let done_json = payload.strip_prefix(COPY_DONE_PAYLOAD_PREFIX)
        .ok_or_else(|| ThoError::new(format!("worker copy response did not start with {COPY_DONE_PAYLOAD_PREFIX:?}: {payload:?}")))?;
    serde_json::from_str(done_json)
        .map_err(|e| ThoError::new(format!("failed to parse worker copy response: {e}")))
}

fn parse_worker_h2d_start_payload(payload: &str) -> Result<(), ThoError> {
    if let Some(error) = payload.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
        return Err(parse_worker_copy_error_payload(error));
    }
    if payload == H2D_START_OK_PAYLOAD {
        Ok(())
    } else {
        Err(ThoError::new(format!("worker H2D preflight response was not {H2D_START_OK_PAYLOAD:?}: {payload:?}")))
    }
}

fn parse_worker_tensor_info_payload(payload: &str) -> Result<WorkerTensorInfoPayload, ThoError> {
    if let Some(error) = payload.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
        return Err(parse_worker_copy_error_payload(error));
    }
    let info_json = payload.strip_prefix(TENSOR_INFO_PAYLOAD_PREFIX)
        .ok_or_else(|| ThoError::new(format!("worker tensor info response did not start with {TENSOR_INFO_PAYLOAD_PREFIX:?}: {payload:?}")))?;
    serde_json::from_str(info_json)
        .map_err(|e| ThoError::new(format!("failed to parse worker tensor info response: {e}")))
}

fn parse_worker_copy_error_payload(payload: &str) -> ThoError {
    match serde_json::from_str::<WorkerCopyErrorPayload>(payload) {
        Ok(error) => error.into_error(),
        Err(_) => WorkerCopyErrorPayload::from_legacy_message(payload).into_error(),
    }
}

struct ParentD2hChunk {
    metadata: IpcTensorTransferMetadata,
    buffer: Guard<(SharedMemfdBuffer, Arc<ThoCudaStream>)>,
}

fn assemble_parent_d2h_chunks(
    mut chunks: Vec<ParentD2hChunk>,
    info: &WorkerTensorInfoPayload,
) -> Result<Tensor, ThoError> {
    chunks.sort_by_key(|chunk| chunk.metadata.tensor_offset);
    let mut expected_offset = 0usize;
    for chunk in &chunks {
        chunk.metadata.validate_for_buffer(&chunk.buffer.0)?;
        if chunk.metadata.kind != info.kind || chunk.metadata.shape != info.shape || chunk.metadata.tensor_byte_len != info.tensor_byte_len {
            return Err(ThoError::new(format!("parent D2H chunk metadata does not match worker tensor info for transfer {}", chunk.metadata.transfer_id)));
        }
        if chunk.metadata.tensor_offset != expected_offset {
            return Err(ThoError::new(format!("parent D2H chunks have gap or overlap at tensor offset {}; got chunk offset {}", expected_offset, chunk.metadata.tensor_offset)));
        }
        expected_offset = expected_offset.checked_add(chunk.metadata.byte_len).ok_or_else(|| ThoError::new("parent D2H expected tensor offset overflow"))?;
    }
    if expected_offset != info.tensor_byte_len {
        return Err(ThoError::new(format!("parent D2H received {expected_offset} bytes but tensor has {} bytes", info.tensor_byte_len)));
    }

    let destination = Tensor::f_empty(info.shape.as_slice(), (info.kind.into(), Cpu))?;
    let dst_base = destination.data_ptr() as usize;
    for chunk in &chunks {
        let destination = dst_base
            .checked_add(chunk.metadata.tensor_offset)
            .ok_or_else(|| ThoError::new("parent D2H destination pointer overflow"))?;
        let source = unsafe {
            // SAFETY: validate_for_buffer checked offset + byte_len against the live mmap and the buffer is held until this copy finishes.
            chunk.buffer.0.as_ptr().add(chunk.metadata.offset)
        };
        unsafe {
            // SAFETY: source points into the validated live mmap chunk, destination points into the owned CPU tensor byte range checked by tensor offsets, and the copy completes before pooled buffers are released.
            std::ptr::copy_nonoverlapping(source, destination as *mut u8, chunk.metadata.byte_len);
        }
    }
    Ok(destination)
}

async fn send_request_and_wait(
    tx: mpsc::Sender<OutgoingRequest>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    request: OutgoingRequest,
) -> Result<String, ThoError> {
    let id = request.id;
    let (reply_tx, reply_rx) = oneshot::channel();
    {
        let mut pending = pending.lock().await;
        let old = pending.insert(id, reply_tx);
        debug_assert!(old.is_none());
    }
    if tx.send(request).await.is_err() {
        let mut pending = pending.lock().await;
        pending.remove(&id);
        return Err(ThoError::new("worker IPC writer task is closed"));
    }
    reply_rx
        .await
        .map_err(|_| ThoError::new(format!("response channel for request {id} was dropped")))?
}

async fn remove_pending_request(
    pending: &Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    id: u64,
) {
    let mut pending = pending.lock().await;
    pending.remove(&id);
}


async fn fail_one_pending(
    pending: &Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    id: u64,
    msg: String,
) {
    let reply_tx = {
        let mut pending = pending.lock().await;
        pending.remove(&id)
    };

    if let Some(reply_tx) = reply_tx {
        let _ = reply_tx.send(Err(ThoError::new(msg)));
    }
}

async fn fail_all_pending(
    pending: &Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    msg: String,
) {
    let pending_items = {
        let mut pending = pending.lock().await;
        pending.drain().collect::<Vec<_>>()
    };

    for (_id, reply_tx) in pending_items {
        let _ = reply_tx.send(Err(ThoError::new(msg.clone())));
    }
}

fn encode_wire_msg(id: u64, payload: &OutgoingRequestPayload) -> Result<String, ThoError> {

    let encoded_payload = match payload {
        OutgoingRequestPayload::Msg(msg) => {
            msg.clone()
        }
        OutgoingRequestPayload::SharedMemfdBuffer{ meta, .. } => {
            let metadata_json = serde_json::to_string(meta)
                .map_err(|e| ThoError::new(format!("failed to encode shared memfd metadata: {e}")))?;
            format!("{SHARED_MEMFD_PAYLOAD_PREFIX}{metadata_json}")
        }
    };

    Ok(format!("{id}\t{encoded_payload}"))
}


#[derive(Debug, Clone)]
pub struct CudaIPCActorHandle {
    sender: mpsc::Sender<CudaIPCActorMessage>,
}


impl CudaIPCActorHandle {
    pub async fn new(tho_device_id:usize, cuda_worker_args: Vec<&str>) -> Result<Self, ThoError> {
        debug!("CudaIPCActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let actor = CudaIPCActor::new(tho_device_id, cuda_worker_args, receiver).await?;
        tokio::spawn(CudaIPCActor::run(actor));

        Ok(Self {
            sender,
        })
    }

    async fn sync_stream(&self, stream: Guard<ThoCudaStream>) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SyncStream {
                stream,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }

    async fn acquire_exec_stream(&self) -> Result<Guard<ThoCudaStream>, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::GetExecStream {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }

}

#[async_trait]
impl CudaActorHandle for CudaIPCActorHandle {
    async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) -> Result<ThoTensor, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Copy {
                src,
                dst_device,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    async fn reset(&self) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Reset {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO

    }

    async fn shutdown(&self) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Shutdown {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }

    fn exec_sync_stage_boxed<'a>(
        &'a self,
        action: Box<dyn FnOnce(Guard<ThoCudaStream>) -> BoxFuture<'a, Result<ExecSyncStageActionResult<Box<dyn Any + Send>>, ThoError>> + Send + 'a>,
    ) -> BoxFuture<'a, Result<Box<dyn Any + Send>, ThoError>> {
        Box::pin(async move {
            let stream = self.acquire_exec_stream().await?;
            match action(stream).await? {
                ExecSyncStageActionResult::Sync { stream, out } => {
                    self.sync_stream(stream).await?;
                    Ok(out)
                }
                ExecSyncStageActionResult::NoSync(out) => Ok(out),
            }
        })
    }

    async fn install_endpoint(&self, init_msg: String) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::InstallEndpoint {
                init_msg,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    async fn invoke_endpoint(&self, input: String) -> Result<String, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::InvokeEndpoint {
                input,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO

    }

    async fn drop_tho_tensor(&self, t:ThoTensor) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::DropThoTensor {
                key: t.ref_key().ok_or_else(||ThoError::new(format!("Not a ref tensor: {t:?}")))?,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }
}

pub(crate) async fn write_text_frame<W>(
    writer: &mut W,
    text: &str,
) -> Result<(), ThoError>
where
    W: AsyncWrite + Unpin,
{
    write_frame(writer, text.as_bytes()).await?;
    Ok(())
}

pub(crate) async fn read_text_frame<R>(
    reader: &mut R,
) -> Result<String, ThoError>
where
    R: AsyncRead + Unpin,
{
    let payload = read_frame(reader).await?;

    String::from_utf8(payload)
        .map_err(|e| ThoError::new(format!("received non-utf8 frame: {e}")))
}

async fn write_frame<W>(
    writer: &mut W,
    payload: &[u8],
) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    if payload.len() > u32::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("frame too large: {} bytes", payload.len()),
        ));
    }

    let len = payload.len() as u32;

    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(payload).await?;
    writer.flush().await?;

    Ok(())
}

async fn read_frame<R>(
    reader: &mut R,
) -> std::io::Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut len_buf = [0u8; 4];

    reader.read_exact(&mut len_buf).await?;

    let len = u32::from_le_bytes(len_buf) as usize;

    if len > MAX_FRAME_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame too large: {len} bytes"),
        ));
    }

    let mut payload = vec![0u8; len];

    reader.read_exact(&mut payload).await?;

    Ok(payload)
}


#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::process::Stdio;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Instant;
    use log::{debug, info};
    use rand::random_range;
    use tch::{Device, Kind, Tensor};
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{UnixListener, UnixStream};
    use tokio::process::{Child, Command};
    use tokio::sync::Mutex;
    use tokio::task::JoinSet;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::{mono_key, Cause, ThoError};
    use thoosa_shared::tracked_await::track_await;
    use crate::copy_ctx::CudaActorHandle;
    use crate::cuda_utils::{cuda_mem, fail_with_unrecoverable_error, speed_gb_per_s};
    use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster};
    use crate::runner::common::cachable::DevicesWrap;
    use crate::runner::common::RunCtx;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
    use crate::runner::run_ctx_2::RunCtxImpl;
    use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, OOMRunnerConfig};
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::ipc::cuda_process::{encode_endpoint_payload, read_text_frame, write_text_frame, CudaIPCActor, IPC_SOCK_ENV};
    use crate::ipc::cuda_wokrer::CudaWorker;
    use crate::ipc::json_ser::SyncStageJsonWrapper;
    use crate::ipc::tests::init_ipc_test;
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    use crate::torch_runner::tests::load_and_clc::loader::LoadedTensor;
    use crate::utils::{lin_for_size, tensor_digest};



    macro_rules! pure_module_path {
        () => {{
            let p = module_path!();

            match p.split_once("::") {
                Some((_crate_name, rest)) => rest,
                None => "",
            }
        }};
    }

    async fn spawn_test_worker(test_name: &str) -> Result<(Child, UnixStream), ThoError> {

        let module = pure_module_path!();

        let test_name = if module.is_empty() {
            test_name.to_string()
        } else {
            format!("{module}::{test_name}")
        };
        let args:Vec<String> = vec![test_name, "--exact".to_string(), "--nocapture".to_string()];

        CudaIPCActor::spawn_worker(&args).await
    }


    async fn run_ipc_copy_perf(cuda_actor: &'static Box<dyn CudaActorHandle>, threads: u8, runs: u8) -> Result<(f64, f64), ThoError> {
        let (free_mem, _total_mem) = cuda_mem();
        let init_type = Float;
        let dim_1 = 4u64;
        let retained_copies = u64::from(threads).max(1) * u64::from(runs).max(1);
        let total_size = (free_mem as f64 * 0.2) as u64 / init_type.elt_size_in_bytes() as u64 / retained_copies;
        let input_n = (total_size / dim_1).max(1);

        let d2h_time = Arc::new(AtomicU64::new(0));
        let h2d_time = Arc::new(AtomicU64::new(0));
        let mut tasks = JoinSet::new();

        for thread_i in 0..threads {
            tasks.spawn(run_ipc_copy_perf_thread(
                thread_i,
                input_n as i64,
                dim_1 as i64,
                runs,
                init_type,
                Arc::clone(&d2h_time),
                Arc::clone(&h2d_time),
                cuda_actor,
            ));
        }

        while let Some(result) = tasks.join_next().await {
            result.map_err(|e| ThoError::new(format!("IPC copy perf task join error: {e}")))??;
        }

        let total_copied_mem_size = input_n * dim_1 * init_type.elt_size_in_bytes() as u64 * runs as u64 * threads as u64;

        Ok((
            speed_gb_per_s(total_copied_mem_size, d2h_time.load(Relaxed)),
            speed_gb_per_s(total_copied_mem_size, h2d_time.load(Relaxed)),
        ))
    }

    async fn run_ipc_copy_perf_thread(
        _thread_i: u8,
        input_n: i64,
        dim_1: i64,
        runs: u8,
        init_type: tch::Kind,
        d2h_time: Arc<AtomicU64>,
        h2d_time: Arc<AtomicU64>,
        cuda_actor: &'static Box<dyn CudaActorHandle>,
    ) -> Result<(), ThoError> {
        let dim_0 = random_range(0.5 * input_n as f64..1.5 * input_n as f64).max(1.0) as i64;
        let mut src_t = Tensor::rand([dim_0, dim_1], FLOAT_CPU);

        for _ in 0..runs {
            let inst_0 = Instant::now();
            let cuda_ref = track_await(cuda_actor.copy(src_t.into(), 1.into())).await?;
            h2d_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            let inst_0 = Instant::now();
            let cpu_tensor = track_await(cuda_actor.copy(cuda_ref, 0.into())).await?;
            d2h_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            let tensor = cpu_tensor.t();
            if tensor.kind() != init_type {
                return Err(ThoError::new(format!("IPC copy perf expected kind {init_type:?}, got {:?}", tensor.kind())));
            }
            src_t = tensor;
        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_copy_ipc_cuda_worker_propagates_oom() {
        debug!("test_copy_ipc_cuda_worker_propagates_oom, IPC_SOCK_ENV={:?}",std::env::var_os(IPC_SOCK_ENV));

        if std::env::var_os(IPC_SOCK_ENV).is_some() {
            CudaWorker::run_worker(HashMap::new()).await.unwrap()
        } else {
            run_ipc_copy_oom_main("test_copy_ipc_cuda_worker_propagates_oom").await.unwrap();
        }

        async fn run_ipc_copy_oom_main(test_name: &str) -> Result<(), ThoError> {

            init_ipc_test(test_name, pure_module_path!()).await?;

            let tho_device_idx = 1;
            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let (free_mem, _total_mem) = cuda_mem();
            let oversize_bytes = free_mem.saturating_add(256 * 1024 * 1024);
            let elements = oversize_bytes / Float.elt_size_in_bytes() + 1;
            let dim_1 = 4usize;
            let dim_0 = elements.div_ceil(dim_1);
            let too_large_tensor = Tensor::f_empty([dim_0 as i64, dim_1 as i64], FLOAT_CPU)?;

            let copy_result = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                track_await(cuda_actor.copy(too_large_tensor.into(), tho_device_idx.into())),
            ).await.map_err(|_| ThoError::new("timed out waiting for worker IPC oversized H2D copy to fail"))?;

            let shutdown_result = cuda_actor.shutdown().await;

            let error = match copy_result {
                Ok(_) => return Err(ThoError::new("worker IPC copy unexpectedly succeeded for oversized tensor")),
                Err(error) => error,
            };

            debug!("run_ipc_copy_oom: error={error:?}");

            match error.what() {
                thoosa_shared::Cause::CudaWorker { is_oom, is_retryable_oom, needs_cuda_ctx_reset, .. } => {
                    assert!(*is_oom, "worker OOM flag was not propagated: {error:?}");
                    assert!(*is_retryable_oom, "worker retryable OOM flag was not propagated: {error:?}");
                    assert!(!*needs_cuda_ctx_reset, "plain worker OOM should not request CUDA context reset: {error:?}");
                }
                other => return Err(ThoError::new(format!("expected Cause::CudaWoker, got {other:?}: {error:?}"))),
            }
            assert!(error.is_oom(), "ThoError::is_oom should delegate to Cause::CudaWoker: {error:?}");
            assert!(error.is_retryable_oom(), "ThoError::is_retryable_oom should delegate to Cause::CudaWoker: {error:?}");

            shutdown_result?;

            Ok(())
        }

    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_copy_ipc_unrecoverable_cuda_error() -> Result<(), ThoError> {
        debug!("test_copy_ipc_cuda_worker_propagates_oom, IPC_SOCK_ENV={:?}",std::env::var_os(IPC_SOCK_ENV));

        #[derive(Debug)]
        struct _SyncStageJsonWrapper {

        }

        impl SyncStageJsonWrapper for _SyncStageJsonWrapper {
            fn do_run(&self, _input: String, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<String, ThoError> {
                debug!("test_copy_ipc_cuda_worker_propagates_oom: do_run={_input:?}");
                if _input.eq("fail") {
                    fail_with_unrecoverable_error();
                } else {

                }


                Ok("ok".to_string())
            }
        }

        if std::env::var_os(IPC_SOCK_ENV).is_some() {
            let r= CudaWorker::run_worker(HashMap::from([
                (
                    "test_copy_ipc_unrecoverable_cuda_error".to_string(),
                    Box::new(|s|{
                        debug!("test_copy_ipc_cuda_worker_propagates_oom: installing _SyncStageJsonWrapper");
                        Ok(Box::new(_SyncStageJsonWrapper{}) as Box<dyn SyncStageJsonWrapper>)
                    }) as Box<dyn Fn(String) -> Result<Box<dyn SyncStageJsonWrapper>, ThoError>>,
                )
            ])).await;

            debug!("test_copy_ipc_cuda_worker_propagates_oom: run_worker r={r:?}");

        } else {
            init_ipc_test("test_copy_ipc_unrecoverable_cuda_error", pure_module_path!()).await?;

            let tho_device_idx = 1;
            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);

            let result = async {
                GlobalCtx::global().cuda_actor(tho_device_idx).install_endpoint(encode_endpoint_payload("test_copy_ipc_unrecoverable_cuda_error", "xxx")).await?;
                let r = GlobalCtx::global().cuda_actor(tho_device_idx).invoke_endpoint(encode_endpoint_payload("test_copy_ipc_unrecoverable_cuda_error", "fail")).await;

                debug!("test_copy_ipc_cuda_worker_propagates_oom: r={r:?}");

                if let Err(e) = &r {
                    debug!("test_copy_ipc_cuda_worker_propagates_oom: needs_cuda_ctx_reset={}", e.needs_cuda_ctx_reset());
                    assert!(e.needs_cuda_ctx_reset());
                } else {
                    panic!("no error");
                }

                GlobalCtx::global().cuda_actor(tho_device_idx).reset().await?;


                Ok::<(), ThoError>(())
            }.await?;

            debug!("test_copy_ipc_cuda_worker_propagates_oom: result={result:?}");
        }

        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 64)]
    async fn heavy_copy_ipc_cuda_worker_once() -> Result<(), ThoError>{
        debug!("test_copy_ipc_cuda_worker_once, IPC_SOCK_ENV={:?}",std::env::var_os(IPC_SOCK_ENV));

        if std::env::var_os(IPC_SOCK_ENV).is_some() {
            CudaWorker::run_worker(HashMap::new()).await
        } else {
            init_ipc_test("test_copy_ipc_cuda_worker", pure_module_path!()).await?;

            let tho_device_idx = 1;
            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let (d2h, h2d)  = run_ipc_copy_perf(cuda_actor, 3, 3).await?;

            info!("\n IPC worker copy d2h speed: {d2h:#2} GB/s, h2d speed {h2d:#2} GB/s");

            cuda_actor.shutdown().await?;

            Ok(())
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_copy_ipc_cuda_worker() -> Result<(), ThoError> {
        debug!("test_cuda_process, IPC_SOCK_ENV={:?}",std::env::var_os(IPC_SOCK_ENV));

        if std::env::var_os(IPC_SOCK_ENV).is_some() {
            CudaWorker::run_worker(HashMap::new()).await

        } else {
            init_ipc_test("test_copy_ipc_cuda_worker", pure_module_path!()).await?;

            let tho_device_idx = 1;

            let main_process_cpu_tensor = Tensor::rand(&[1000, 10000], FLOAT_CPU);
            let expected_digest = tensor_digest(&main_process_cpu_tensor)?;


            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let cuda_ref = track_await(cuda_actor.copy(
                    main_process_cpu_tensor.into(),
                    tho_device_idx.into(),//Cuda
                )).await?;

            debug!("run_main: cuda_ref={cuda_ref:?}");

            let reff = cuda_ref.ref_key().unwrap();

            let cpu_tensor = track_await(cuda_actor.copy(
                    cuda_ref,
                    0.into(),//Cpu
                )).await?;

            debug!("run_main: cpu_tensor={cpu_tensor:?}");

            let actual_digest = tensor_digest(cpu_tensor.t_ref())?;
            if actual_digest.hex != expected_digest.hex {
                return Err(ThoError::new(format!("worker CUDA round-trip digest mismatch: expected={} actual={}", expected_digest.hex, actual_digest.hex)));
            }

            cuda_actor.drop_tho_tensor(ThoTensor::new_ref((), reff, vec![], tch::Kind::Float)).await?;

            let r = track_await(cuda_actor
                .copy(
                    ThoTensor::new_ref((), reff, vec![], tch::Kind::Float),
                    0.into(), //Cpu
                )).await;

            let Err(ThoError { msg, what:Cause::CudaWorker { msg:what_msg, is_oom, is_retryable_oom, needs_cuda_ctx_reset },.. }) = r else {
                panic!("Got r={r:?}, expected CudaWorker error");
            } ;

            assert_eq!("ThoError: Unknown error, cause=worker tensor key=1 not found", what_msg);


            cuda_actor.shutdown().await?;

            Ok(())
        }
    }

}
