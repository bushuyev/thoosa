use crate::copy_ctx::{CopyCtx, CudaActorHandle, HandleCtx};
use crate::cuda_utils::{make_chunks, tensor_mem_size};
use crate::dims::Dims;
use crate::ipc::ipc_cuda_utils::{CuMemHostRegistrationGuard, IpcProcessRole, IpcTensorKind, IpcTensorTransferMetadata, IpcTransferDirection, SharedMemfdBuffer, recv_fd, send_fd};
use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster};
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::utils::tensor_digest;
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::future::BoxFuture;
use log::{debug, error, info, trace, warn};
use rayon::iter::ParallelBridge;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::backtrace::Backtrace;
use std::cell::RefCell;
use std::collections::HashMap;
use std::collections::HashSet;
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
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::{Device, Tensor};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::rayon_pool::rayon_pool;
use thoosa_shared::traits::RunCtx;
use thoosa_shared::traits::{Data, Named};
use thoosa_shared::{Cause, ThoError, mono_key};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::cuda_result::check_cuda_status;
use thoosa_shims::memcpy::{tho_cu_memcpy_dtoh_async, tho_cu_memcpy_htod_async};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncSeekExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{Mutex, RwLock, mpsc, oneshot};
use tokio::task::{JoinHandle, spawn_blocking};
use tokio_util::sync::CancellationToken;
use tracing::Instrument;
use tracing::field::debug;
use tub::{Guard, Pool};
use crate::ipc::cuda_wokrer::insert_worker_tensor;
use crate::ipc::worker_msg::{CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation, WorkerInput, WorkerOutput};

pub const CUDA_VISIBLE_DEVICES: &str = "CUDA_VISIBLE_DEVICES";//it's set to '' for main process to make sure it doesn't use GPU
pub const IPC_SOCK_ENV: &str = "THOOSA_WORKER_IPC_SOCK";
pub  const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;
pub(crate) const SHARED_MEMFD_PAYLOAD_PREFIX: &str = "shared-memfd:";
const COPY_ERROR_PAYLOAD_PREFIX: &str = "copy-error:";
pub(crate) const DESCRIBE_REF_PAYLOAD_PREFIX: &str = "describe-ref:";
pub(super) const TENSOR_INFO_PAYLOAD_PREFIX: &str = "tensor-info:";
pub(crate) const H2D_START_PAYLOAD_PREFIX: &str = "h2d-start:";
pub(crate) const D2H_START_PAYLOAD_PREFIX: &str = "d2h-start:";
pub(crate) const INSTALL_ENDPOINT_PREFIX: &str = "install_endpoint:";
pub(crate) const INVOKE_ENDPOINT_PREFIX: &str = "invoke_endpoint:";
pub(crate) const PURGE_ENDPOINT_PREFIX: &str = "purge_endpoint:";
pub(crate) const DROP_ENDPOINT_PREFIX: &str = "drop_endpoint:";
pub(crate) const CONSUME_ENDPOINT_PREFIX: &str = "consume_endpoint:";
const WORKER_COPY_STREAMS: usize = 32;
// Keep IPC staging bounded independently of GPU VRAM. Sixteen 512 MiB sparse memfd mappings retain enough capacity for the existing oversized-copy OOM test while avoiding a pool sized as a fraction of total VRAM.
const IPC_COPY_BUFFERS: usize = 16;
const IPC_COPY_BUFFER_SIZE: usize = 512 * 1024 * 1024;
pub(crate) const H2D_START_OK_PAYLOAD: &str = "h2d-start-ok";
pub(crate) const DEFAULT_ENDPOINT_NAME: &str = "SyncClcTask";
pub(crate) const ENDPOINT_OK_PAYLOAD: &str = "endpoint-ok";
pub(crate) const DROP_THO_TENSOR: &str = "drop_tho_tensor:";
const ENDPOINT_BYTES_PAYLOAD_PREFIX: &str = "endpoint-bytes:";
const BINARY_RESPONSE_CHUNK_PREFIX: &[u8] = b"binary-response-chunk:";
const BINARY_RESPONSE_END_PAYLOAD: &[u8] = b"binary-response-end";
const BINARY_RESPONSE_CHUNK_LEN: usize = 10 * 1024 * 1024;

pub(crate) fn encode_endpoint_payload(endpoint_name: &str, payload: &str) -> String {
    format!("{endpoint_name}\t{payload}")
}

// Keep the stage id separate from the endpoint payload so the worker can still select the initializer by its static endpoint name.
fn encode_stage_payload(stage_id: u32, payload: &str) -> String {
    format!("{stage_id}\t{payload}")
}

pub(crate) fn split_stage_payload(payload: &str) -> Result<(u32, &str), ThoError> {
    let (stage_id, payload) = payload.split_once('\t').ok_or_else(|| ThoError::new(format!("can't parse stage payload: {payload}")))?;
    Ok((stage_id.parse::<u32>()?, payload))
}

// Installed stages share an initializer name but must have distinct invocation keys.
pub(crate) fn endpoint_stage_key(endpoint_name: &str, stage_id: u32) -> String {
    format!("{endpoint_name}{stage_id}")
}

pub(crate) fn encode_endpoint_payload_bytes(endpoint_name: &str, payload: &[u8]) -> String {
    format!("{endpoint_name}\t{ENDPOINT_BYTES_PAYLOAD_PREFIX}{}", bytes_to_hex(payload))
}

pub(crate) fn encode_endpoint_response_bytes(payload: &[u8]) -> String {
    format!("{ENDPOINT_BYTES_PAYLOAD_PREFIX}{}", bytes_to_hex(payload))
}

pub(crate) fn split_endpoint_payload(payload: &str) -> Result<(&str, &str), ThoError> {
    payload.split_once('\t').map(|v|Ok(v)).unwrap_or_else(||Err(ThoError::new(format!("can't parse payload: {payload}"))))
}

pub(crate) fn split_endpoint_payload_bytes(payload: &str) -> Result<(&str, Vec<u8>), ThoError> {
    let (endpoint_name, payload) = split_endpoint_payload(payload)?;
    Ok((endpoint_name, decode_endpoint_bytes_payload(payload)?))
}

pub(crate) fn decode_endpoint_response_bytes(payload: &str) -> Result<Vec<u8>, ThoError> {
    decode_endpoint_bytes_payload(payload)
}

fn decode_endpoint_bytes_payload(payload: &str) -> Result<Vec<u8>, ThoError> {
    match payload.strip_prefix(ENDPOINT_BYTES_PAYLOAD_PREFIX) {
        Some(hex) => hex_to_bytes(hex),
        None => Ok(payload.as_bytes().to_vec()),
    }
}

fn bytes_to_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, ThoError> {
    if hex.len() % 2 != 0 {
        return Err(ThoError::new(format!("endpoint byte payload has odd hex length: {}", hex.len())));
    }
    let bytes = hex.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in bytes.chunks_exact(2) {
        let hi = hex_value(chunk[0])?;
        let lo = hex_value(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_value(byte: u8) -> Result<u8, ThoError> {
    match byte {
        b'0'..=b'9' => Ok(byte - b'0'),
        b'a'..=b'f' => Ok(byte - b'a' + 10),
        b'A'..=b'F' => Ok(byte - b'A' + 10),
        _ => Err(ThoError::new(format!("endpoint byte payload contains non-hex byte: {byte}"))),
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct WorkerCopyDonePayload {
    transfer_id: u64,
    tensor_ref:u64,
    tensor_digest_hex: Option<String>,
}

#[typetag::serde]
impl WorkerOutput for WorkerCopyDonePayload {}

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

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="encode_worker_copy_error"))]
pub(crate) fn encode_worker_copy_error(error: &ThoError) -> String {
    let payload = WorkerCopyErrorPayload::from_error(error);
    match serde_json::to_string(&payload) {
        Ok(json) => format!("{COPY_ERROR_PAYLOAD_PREFIX}{json}"),
        Err(e) => format!("{COPY_ERROR_PAYLOAD_PREFIX}failed to encode worker error: {e}; original error: {error}"),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct WorkerTensorInfoPayload {
    tensor_key: u64,
    kind: IpcTensorKind,
    shape: Vec<i64>,
    tensor_byte_len: usize,
}

impl WorkerTensorInfoPayload {
    pub fn new(tensor_key: u64, kind: IpcTensorKind, shape: Vec<i64>, tensor_byte_len: usize) -> Self {
        Self { tensor_key, kind, shape, tensor_byte_len }
    }
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
    stream_pool: Arc<Pool<ThoCudaStream>>,
}

#[typetag::serde]
impl WorkerOutput for WorkerTensorInfoPayload {

}


fn validate_ipc_chunk_capacity(chunk_count: usize, label: &str) -> Result<(), ThoError> {
    if chunk_count > IPC_COPY_BUFFERS {
        // H2D retains every buffer until the worker finishes the complete tensor, so waiting for more buffers here would deadlock instead of applying backpressure.
        return Err(ThoError::new(format!("{label} requires {chunk_count} chunks but the bounded IPC pool has {IPC_COPY_BUFFERS} buffers")));
    }
    Ok(())
}

impl WorkerCudaResources {
    fn new(cuda_device: usize) -> Result<Self, ThoError> {
        let streams = (0..WORKER_COPY_STREAMS)
            .map(|_| ThoCudaStream::from_pool(cuda_device as i32, true))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
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
    tensor_ref: u64,
    metadata: Option<IpcTensorTransferMetadata>,
    received_bytes: usize,
    seen_chunks: HashSet<usize>,
    chunks: Vec<WorkerD2hChunk>,
}

impl PendingWorkerD2hTransfer {
    pub(crate) fn new(request_id: u64, tensor_key: u64) -> Self {
        Self {
            request_id,
            tensor_ref: tensor_key,
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
        // D2H chunks are independent transfers so the parent can consume and release each shared buffer as soon as its worker copy completes.
        self.metadata.as_ref().is_some_and(|metadata| self.chunks.len() == metadata.chunk_count)
    }
}

pub(crate) struct WorkerReply {
    pub(crate) id: u64,
    pub(crate) payload: OutgoingRequestPayload,
    pub msg:Option<CudaWorkerMsgOut>//TODO replace WorkerReply with CudaWorkerMsgOut
}

pub(crate) struct IpcChunkWriter {
    chunks: mpsc::Sender<Result<Vec<u8>, ThoError>>,
}

impl IpcChunkWriter {
    pub(crate) fn new(chunks: mpsc::Sender<Result<Vec<u8>, ThoError>>) -> Self {
        Self { chunks }
    }
}

impl std::io::Write for IpcChunkWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        for chunk in buf.chunks(BINARY_RESPONSE_CHUNK_LEN) {
            self.chunks.blocking_send(Ok(chunk.to_vec()))
                .map_err(|_| std::io::Error::new(ErrorKind::BrokenPipe, "worker binary response writer is closed"))?;
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}


pub(crate) async fn worker_response_writer_loop<W>(
    mut writer: W,
    mut reply_rx: mpsc::Receiver<WorkerReply>,
) -> Result<(), ThoError>
where
    W: AsyncWrite + Unpin,
{
    while let Some(reply) = reply_rx.recv().await {
        if let Some(msg) = reply.msg {
            debug!("worker_response_writer_loop: msg={msg:?}");

            write_frame(&mut writer, &postcard::to_allocvec(&msg).unwrap()/*TODO move to codec*/, "worker").await?;
            debug!("worker_response_writer_loop: msg out ok");
            continue
        }
        let write_result = match reply.payload {
            // OutgoingRequestPayload::BinaryStream { chunks, completed } => {
            //     let result = write_binary_stream_response(&mut writer, reply.id, chunks).await;
            //     let completion = result.as_ref().map(|_| ()).map_err(|error| ThoError::new(error.to_string()));
            //     let _ = completed.send(completion);
            //     result
            // }
            payload => {
                let frame = encode_wire_frame(reply.id, &payload)?;
                write_frame(&mut writer, &frame, "worker").await
            }
        };
        if let Err(error) = write_result {
            if matches!(error.kind(), ErrorKind::BrokenPipe | ErrorKind::UnexpectedEof) {
                debug!("worker_response_writer_loop: parent IPC stream closed");
                break;
            }
            return Err(error.into());
        }
    }

    Ok(())
}

async fn write_binary_stream_response<W>(writer: &mut W, id: u64, mut chunks: mpsc::Receiver<Result<Vec<u8>, ThoError>>) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    while let Some(chunk) = chunks.recv().await {
        match chunk {
            Ok(chunk) => {
                let frame = encode_binary_response_frame(id, BINARY_RESPONSE_CHUNK_PREFIX, &chunk);
                write_frame(writer, &frame, "worker binary response chunk").await?;
            }
            Err(error) => {
                let frame = format!("{id}\t{}", encode_worker_copy_error(&error));
                write_frame(writer, frame.as_bytes(), "worker binary response error").await?;
                return Ok(());
            }
        }
    }

    let frame = encode_binary_response_frame(id, BINARY_RESPONSE_END_PAYLOAD, &[]);
    write_frame(writer, &frame, "worker binary response end").await
}

fn encode_binary_response_frame(id: u64, prefix: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut frame = format!("{id}\t").into_bytes();
    frame.extend_from_slice(prefix);
    frame.extend_from_slice(payload);
    frame
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="describe_worker_tensor"))]
pub(crate) async fn describe_worker_tensor(
    all_cuda_tensors: Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>,
    tensor_key: u64,
) -> Result<WorkerTensorInfoPayload, ThoError> {
    let tensors = all_cuda_tensors.lock();
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

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="worker_role_cuda_device"))]
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

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="get_or_create_worker_cuda_resources"))]
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

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="copy_worker_h2d_transfer_to_cuda"))]
async fn copy_worker_h2d_transfer_to_cuda(
    transfer: PendingWorkerH2dTransfer,
    resources: Arc<WorkerCudaResources>,
    with_digest: bool,
) -> Result<(u64, Tensor, Option<String>), ThoError> {
    let stream = resources.stream_pool.acquire().await;
    // CUDA host registration and stream synchronization are blocking driver calls. Run the whole transfer on a blocking-pool thread so the worker's async socket and endpoint tasks remain schedulable.
    spawn_blocking(move || copy_worker_h2d_transfer_to_cuda_blocking(transfer, stream, with_digest)).await.map_err(|error| ThoError::new(format!("worker H2D blocking task failed: {error}")))?
}

fn copy_worker_h2d_transfer_to_cuda_blocking(
    mut transfer: PendingWorkerH2dTransfer,
    stream: Guard<ThoCudaStream>,
    with_digest: bool,
) -> Result<(u64, Tensor, Option<String>), ThoError> {
    let metadata = transfer.metadata.as_ref().ok_or_else(|| ThoError::new(format!("H2D transfer {} has no metadata", transfer.request_id)))?.clone();
    let cuda_device = worker_cuda_device(&metadata)?;
    // bind_worker_copy_cuda_context(cuda_device)?;
    let _stream_guard = ThoStreamGuard::new(&stream)?;

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
        let _registration = CuMemHostRegistrationGuard::new(&chunk.buffer, chunk.metadata.offset, chunk.metadata.byte_len)?;
        let result = unsafe {
            // SAFETY: destination_offset points into the CUDA tensor byte range checked by metadata tensor offsets, source points into the registered live mmap, and the stream context is current on this OS thread.
            tho_cu_memcpy_htod_async(destination_offset, source.cast::<c_void>(), chunk.metadata.byte_len, stream.as_mut_ptr())
        };
        check_cuda_status(result, "cuMemcpyHtoDAsync(worker memfd chunk)")?;
        stream.synchronize()?;
    }


    let copied_cpu = destination.f_to_device(Cpu)?;
    let copied_digest = with_digest.then(||tensor_digest(&copied_cpu)).transpose()?;
    debug!("copy_worker_h2d_transfer_to_cuda: copied tensor digest={copied_digest:?}");
    Ok((metadata.transfer_id, destination, copied_digest.map(|d|d.hex)))
}

pub(crate) fn spawn_worker_h2d_copy_task(
    transfer: PendingWorkerH2dTransfer,
    resources: Arc<WorkerCudaResources>,
    all_cuda_tensors: Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>,
    reply_tx: mpsc::Sender<WorkerReply>,
    with_digest: bool,
) {
    tokio::spawn(async move {
        let request_id = transfer.request_id;
        let copy_result = copy_worker_h2d_transfer_to_cuda(transfer, resources, with_digest).await;

        // Copy completion uses the same postcard response envelope as the other worker operations.
        let status = match copy_result {
            Ok((transfer_id, tensor, tensor_digest_hex)) => {
                //TODO should tensors have their own id generator?
                // all_cuda_tensors.lock().insert(transfer_id, tensor);
                match insert_worker_tensor(&all_cuda_tensors, tensor) {
                    Ok(tensor_ref) => OpResult::Ok(Some(Box::new(WorkerCopyDonePayload { transfer_id, tensor_ref, tensor_digest_hex }) as Box<dyn WorkerOutput>)),
                    Err(e) => ThoError::new(format!("can't insert worker tensor {request_id}: {e}")).into(),
                }
            }
            Err(e) => e.into(),
        };

        let _ = reply_tx.send(WorkerReply {
            id: request_id,
            payload: OutgoingRequestPayload::Msg("dummy".to_string()),
            msg: Some(CudaWorkerMsgOut { request_id, status }),
        }).await;
    });
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="copy_worker_d2h_transfer_to_memfd"))]
async fn copy_worker_d2h_transfer_to_memfd(
    transfer: PendingWorkerD2hTransfer,
    resources: Arc<WorkerCudaResources>,
    source: Tensor,
    with_digest: bool,
) -> Result<(u64, Option<String>), ThoError> {
    let stream = resources.stream_pool.acquire().await;
    // Keep blocking CUDA registration and synchronization out of the async worker runtime for the same reason as H2D copies.
    spawn_blocking(move || copy_worker_d2h_transfer_to_memfd_blocking(transfer, stream, source, with_digest)).await.map_err(|error| ThoError::new(format!("worker D2H blocking task failed: {error}")))?
}

fn copy_worker_d2h_transfer_to_memfd_blocking(
    mut transfer: PendingWorkerD2hTransfer,
    stream: Guard<ThoCudaStream>,
    source: Tensor,
    with_digest: bool,
) -> Result<(u64, Option<String>), ThoError> {
    let metadata = transfer.metadata.as_ref().ok_or_else(|| ThoError::new(format!("D2H transfer {} has no metadata", transfer.request_id)))?.clone();
    let cuda_device = worker_d2h_cuda_device(&metadata)?;
    // bind_worker_copy_cuda_context(cuda_device)?;
    let _stream_guard = ThoStreamGuard::new(&stream)?;
    if source.kind() != metadata.tch_kind() {
        return Err(ThoError::new(format!("worker tensor key={} kind {:?} does not match D2H metadata {:?}", transfer.tensor_ref, source.kind(), metadata.kind)));
    }
    if source.size() != metadata.shape {
        return Err(ThoError::new(format!("worker tensor key={} shape {:?} does not match D2H metadata {:?}", transfer.tensor_ref, source.size(), metadata.shape)));
    }
    if !source.is_contiguous() {
        return Err(ThoError::new(format!("worker tensor key={} is not contiguous", transfer.tensor_ref)));
    }
    let Cuda(source_cuda_device) = source.device() else {
        return Err(ThoError::new(format!("worker tensor key={} is not on CUDA: {:?}", transfer.tensor_ref, source.device())));
    };
    if source_cuda_device != cuda_device {
        return Err(ThoError::new(format!("worker tensor key={} device={source_cuda_device} does not match D2H metadata cuda_device={cuda_device}", transfer.tensor_ref)));
    }

    transfer.chunks.sort_by_key(|chunk| chunk.metadata.tensor_offset);
    // A D2H transfer may cover only one range of the tensor; parallel parent chunk tasks no longer retain every range in one transfer.
    let mut expected_offset = transfer.chunks.first().ok_or_else(|| ThoError::new(format!("D2H transfer {} has no chunks", metadata.transfer_id)))?.metadata.tensor_offset;
    let first_offset = expected_offset;
    for chunk in &transfer.chunks {
        if chunk.metadata.tensor_offset != expected_offset {
            return Err(ThoError::new(format!("D2H transfer {} has gap or overlap at tensor offset {}; got chunk offset {}", metadata.transfer_id, expected_offset, chunk.metadata.tensor_offset)));
        }
        expected_offset = expected_offset.checked_add(chunk.metadata.byte_len).ok_or_else(|| ThoError::new("D2H expected tensor offset overflow"))?;
    }
    if expected_offset > metadata.tensor_byte_len || expected_offset - first_offset != transfer.received_bytes {
        return Err(ThoError::new(format!("D2H transfer {} range [{first_offset}, {expected_offset}) is inconsistent with tensor byte length {} and received byte count {}", metadata.transfer_id, metadata.tensor_byte_len, transfer.received_bytes)));
    }


    let src_base = source.data_ptr() as u64;
    for chunk in &mut transfer.chunks {
        let source_offset = src_base
            .checked_add(chunk.metadata.tensor_offset as u64)
            .ok_or_else(|| ThoError::new("D2H source pointer overflow"))?;

        let destination = unsafe {
            // SAFETY: validate_worker_d2h_chunk checked offset + byte_len against the live mmap before insertion.
            chunk.buffer.as_mut_ptr().add(chunk.metadata.offset)
        };

        let _registration = CuMemHostRegistrationGuard::new(&chunk.buffer, chunk.metadata.offset, chunk.metadata.byte_len)?;
        let result = unsafe {
            // SAFETY: source_offset points into the local CUDA tensor byte range checked by metadata tensor offsets, destination points into the registered live mmap, and the stream context is current on this OS thread.
            tho_cu_memcpy_dtoh_async(destination.cast::<c_void>(), source_offset, chunk.metadata.byte_len, stream.as_mut_ptr())
        };
        check_cuda_status(result, "cuMemcpyDtoHAsync(worker memfd chunk)")?;
        stream.synchronize()?;
    }


    // Independent chunk transfers must not each copy the full source tensor merely to skip a disabled diagnostic digest.
    let source_digest = if with_digest {
        let source_cpu = source.f_to_device(Cpu)?;
        Some(tensor_digest(&source_cpu)?)
    } else {
        None
    };

    debug!("copy_worker_d2h_transfer_to_memfd: source tensor digest={source_digest:?}");
    Ok((metadata.transfer_id, source_digest.map(|d|d.hex)))
}

pub(crate) fn spawn_worker_d2h_copy_task(
    transfer: PendingWorkerD2hTransfer,
    resources: Arc<WorkerCudaResources>,
    all_cuda_tensors: Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>,
    reply_tx: mpsc::Sender<WorkerReply>,
    with_digest: bool,
) {
    tokio::spawn(async move {
        let request_id = transfer.request_id;
        let tensor_ref = transfer.tensor_ref;
        let source = {
            let tensors = all_cuda_tensors.lock();
            tensors.get(&tensor_ref).map(Tensor::shallow_clone)
        };
        let copy_result = match source {
            Some(source) => {

                copy_worker_d2h_transfer_to_memfd(transfer, resources, source, with_digest).await
            }
            None => Err(ThoError::new(format!("worker tensor key={tensor_ref} not found"))),
        };

        // Copy completion uses a typed WorkerOutput so the parent does not parse JSON prefixes.
        let status = match copy_result {
            Ok((transfer_id, tensor_digest_hex)) => OpResult::Ok(Some(Box::new(WorkerCopyDonePayload { transfer_id, tensor_ref, tensor_digest_hex }) as Box<dyn WorkerOutput>)),
            Err(e) => e.into(),
        };

        let _ = reply_tx.send(WorkerReply {
            id: request_id,
            payload: OutgoingRequestPayload::Msg("dummy".to_string()),
            msg: Some(CudaWorkerMsgOut { request_id, status }),
        }).await;
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
    ConsumeEndpoint {
        input: String, respond_to: oneshot::Sender<Result<String, ThoError>>
    },
    InvokeEndpointToFile {//TODO remove
        input: String,
        output: std::fs::File,
        respond_to: oneshot::Sender<Result<std::fs::File, ThoError>>,
    },
    PurgeEndpoint {//
        stage_id: u32, endpoint_name:String,
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    DropEndpoint {
        stage_id: u32, endpoint_name: String,
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },

    SendMsg {
        input: CudaWorkerMsgIn,
        respond_to: oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>
    },
}

struct OutgoingRequest {
    id: u64,
    payload: OutgoingRequestPayload,
    msg:Option<CudaWorkerMsgIn>//TODO replace OutgoingRequest with CudaWorkerMsg
}


pub enum OutgoingRequestPayload {
    Msg(String),//action:data

    SharedMemfdBuffer{
        meta: IpcTensorTransferMetadata,
        fd:OwnedFd,
    },
}

#[derive(Debug)]
enum IncomingResponsePayload {
    Msg(String),
    // BinaryChunk(Vec<u8>),
    // BinaryEnd,
}

impl IncomingResponsePayload {
    fn into_msg(self) -> Result<String, ThoError> {
        match self {
            Self::Msg(response) => Ok(response),
            // Self::BinaryChunk(_) | Self::BinaryEnd => Err(ThoError::new("unexpected streaming worker response")),
        }
    }
}

#[derive(Debug)]
enum PendingResponse {
    Buffered(oneshot::Sender<Result<IncomingResponsePayload, ThoError>>),
    File {
        output: tokio::fs::File,
        respond_to: oneshot::Sender<Result<std::fs::File, ThoError>>,
        error: Option<ThoError>,
    },
    Msg(oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>)
}

type PendingResponses = Arc<Mutex<HashMap<u64, PendingResponse>>>;

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
    pending: PendingResponses,
    tx:Sender<OutgoingRequest>,
    child:Child,
    writer_handle:JoinHandle<()>,
    reader_handle:JoinHandle<()>,
    cancel: CancellationToken,
    device_info:DeviceInfo
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
    pub async fn new(tho_device_idx:usize, cuda_worker_args: Vec<String>, receiver: mpsc::Receiver<CudaIPCActorMessage>) -> Result<Self, ThoError> {
        let cuda_worker_args = Arc::new(cuda_worker_args);

        let spawned_state = Self::spawn_worker_and_loops(tho_device_idx, &cuda_worker_args).await?;


        debug!("CudaIPCActor::new: worker device_info={:?} ipc_copy_buffers={IPC_COPY_BUFFERS} ipc_copy_buffer_size={IPC_COPY_BUFFER_SIZE}", spawned_state.device_info);
        let copy_ctx = CopyCtx::make_ipc_instance_for_cuda_device(tho_device_idx - 1, IPC_COPY_BUFFERS, IPC_COPY_BUFFER_SIZE);
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

    async fn spawn_worker_and_loops(tho_device_idx: usize, cuda_worker_args: &Vec<String>) -> Result<SpawnedState, ThoError> {
        debug!("CudaIPCActor::spawn_worker_and_loops, cuda_worker_args={cuda_worker_args:?}");

        let (mut child, stream, device_info) = Self::spawn_worker_for_cuda_device(tho_device_idx - 1, cuda_worker_args).await?;

        let (reader, writer) = stream.into_split();

        let (tx, rx): (Sender<OutgoingRequest>, Receiver<OutgoingRequest>) = mpsc::channel::<OutgoingRequest>(1024);

        let pending: PendingResponses = Arc::new(Mutex::new(HashMap::new()));
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
            cancel,
            device_info
        })
    }

    async fn spawn_worker_for_cuda_device(cuda_device: usize, args: &Vec<String>) -> Result<(Child, UnixStream, DeviceInfo), ThoError> {
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
            .env_remove(CUDA_VISIBLE_DEVICES)
            .env(IPC_SOCK_ENV, &sock_path)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        let accepted = tokio::time::timeout(
            Duration::from_secs(5),
            listener.accept(),
        ).await;

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

        let device_info:DeviceInfo = serde_json5::from_str(&read_text_frame(&mut stream, "CudaIPCActor").await?)?;
        debug!("CudaIPCActor::spawn_worker: main received: {device_info:?}");


        // The socket file can be removed after accept. The connected UnixStream
        // remains valid.
        drop(dir);

        Ok((child, stream, device_info))
    }


    async fn handle_message(&mut self, msg: CudaIPCActorMessage) -> Result<(), ThoError>{
        match msg {
            CudaIPCActorMessage::Copy { src, dst_device, respond_to } => {

                return if let Some(tensor_key) = src.ref_key() {
                    assert_eq!(*dst_device, Cpu, "If source is tensor ref and thus live on device it can be copied only to CPU");
                     self.handle_ref_d2h_copy(tensor_key, dst_device, respond_to).await
                } else {
                    debug!("handle_message: src={src:?} dst_device={dst_device:?}");
                    assert_ne!(*dst_device, Cpu, "If source is concrete tensor it should be on CPU and can be copied only to GPU");

                    self.handle_h2d_copy(src, dst_device, respond_to)
                }

            }

            CudaIPCActorMessage::DropThoTensor { key, respond_to } => {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let pending = Arc::clone(&self.spawned_state.pending);
                let self_tx = self.spawned_state.tx.clone();

                debug!("handle_message: CudaIPCActorMessage::DropThoTensor id={id:?} key={key}");

                tokio::spawn(async move {
                    {
                        let mut pending = pending.lock().await;
                        let old = pending.insert(id, PendingResponse::Msg(reply_tx));
                        debug_assert!(old.is_none());
                    }

                    // if self_tx.send(OutgoingRequest { id, payload: OutgoingRequestPayload::Msg(encode_endpoint_payload(DROP_THO_TENSOR, &key.to_string())), msg: None }).await.is_err() {
                    if self_tx.send(
                        OutgoingRequest {
                        id,
                        payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                        msg: Some(CudaWorkerMsgIn {
                            request_id: id,
                            operation: Operation::DropThoTensor {tensor_key: key},
                        }),
                    }).await.is_err() {
                        let mut pending = pending.lock().await;
                        pending.remove(&id);
                        let _ = respond_to.send(Err(ThoError::new("worker IPC writer task is closed")));
                        return;
                    }


                    Self::handle_reply(respond_to, id, reply_rx).await;

                });
            }
            CudaIPCActorMessage::Reset { respond_to } => {
                debug!("CudaIPCActor::handle_message, CudaIPCActorMessage::Reset");
                let self_arc = Arc::clone(&self.cuda_worker_args);


                let mut old_pawned_state = mem::replace(&mut self.spawned_state, Self::spawn_worker_and_loops(self.tho_device_idx, &self_arc).await?);
                // old_pawned_state.shutdown("ups").await;

                debug!("CudaIPCActor::handle_message, CudaIPCActorMessage::Reset done");

                let _ = respond_to.send(Ok(()));
            }
            CudaIPCActorMessage::Shutdown { respond_to } => {
                // self.spawned_state.child.kill().await?;
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let pending = Arc::clone(&self.spawned_state.pending);
                let self_tx = self.spawned_state.tx.clone();



                {
                    let mut pending = pending.lock().await;
                    let old = pending.insert(next_id, PendingResponse::Msg(reply_tx));
                    debug_assert!(old.is_none());
                }

                debug!("CudaIPCActorMessage::Shutdown: next_id={next_id}");

                // if self_tx.send(OutgoingRequest { id, payload: OutgoingRequestPayload::Msg("shutdown".to_string()), msg: None }).await.is_err() {
                //     let mut pending = pending.lock().await;
                //     pending.remove(&id);
                //     let _ = respond_to.send(Err(ThoError::new("worker IPC writer task is closed")));
                //     return;
                // }
                //

                if self_tx.send(
                    OutgoingRequest {
                        id: next_id,
                        payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                        msg: Some(CudaWorkerMsgIn {
                            request_id: next_id,
                            operation: Operation::Shutdown{ dummy: "dummy".to_string() },
                        }),
                    }).await.is_err() {

                    let mut pending = pending.lock().await;
                    pending.remove(&next_id);
                    let _ = respond_to.send(Err(ThoError::new("worker IPC writer task is closed")));
                    // panic!("can't shutdown");
                    return Err(ThoError::new("can't shutdown"))
                }

                // let reply = reply_rx
                //     .await
                //     .map_err(|_| ThoError::new(format!("response channel for shutdown request {id} was dropped")))
                //     .and_then(|reply| reply)
                //     /*.and_then(IncomingResponsePayload::into_msg)*/;

                Self::handle_reply(respond_to, next_id, reply_rx).await;

                debug!("CudaIPCActorMessage::Shutdown: final");

                let _ = self.spawned_state.child.kill().await;
                self.spawned_state.reader_handle.abort();
                self.spawned_state.writer_handle.abort();
                // match reply {
                //     Ok(reply) if reply == "shutdown-ok" => {
                //         let _ = respond_to.send(Ok(()));
                //     }
                //     Ok(reply) => {
                //         let _ = respond_to.send(Err(ThoError::new(format!("unexpected shutdown response: {reply:?}"))));
                //     }
                //     Err(error) => {
                //         let _ = respond_to.send(Err(error));
                //     }
                // }

            }
            CudaIPCActorMessage::InstallEndpoint { init_msg, respond_to } => {//TODO remove
                // let self_tx = self.spawned_state.tx.clone();
                // let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                // let pending = Arc::clone(&self.spawned_state.pending);
                //
                // tokio::spawn(async move {
                //
                //     let response = send_request_and_wait(
                //         self_tx,
                //         pending,
                //         OutgoingRequest {
                //             id: next_id,
                //             payload: OutgoingRequestPayload::Msg(format!("{INSTALL_ENDPOINT_PREFIX}{init_msg}")),
                //             msg: None,
                //         },
                //     ).await.and_then(|response| {
                //         if let Some(error) = response.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
                //             Err(parse_worker_copy_error_payload(error))
                //         } else if response == ENDPOINT_OK_PAYLOAD {
                //             Ok(())
                //         } else {
                //             Err(ThoError::new(format!("unexpected endpoint install response: {response:?}")))
                //         }
                //     });
                //     let _ = respond_to.send(response);
                //
                //     Ok::<_, ThoError>(())
                // });
                todo!()
            }
            CudaIPCActorMessage::SendMsg { mut input, respond_to } => {
                let self_tx = self.spawned_state.tx.clone();
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.spawned_state.pending);

                input.request_id = next_id;

                tokio::spawn(async move {

                    let response = send_request_and_wait_msg(
                        next_id,
                        self_tx,
                        pending,
                        OutgoingRequest {
                            id: next_id,
                            payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                            msg: Some(input),
                        },
                    ).await;
                    let _ = respond_to.send(response);

                    Ok::<_, ThoError>(())
                });
            }

            CudaIPCActorMessage::InvokeEndpoint { input, respond_to } => {//TODO remove

                // let self_tx = self.spawned_state.tx.clone();
                // let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                // let pending = Arc::clone(&self.spawned_state.pending);
                //
                // _= tokio::spawn(async move {
                //
                //     let response = send_request_and_wait(
                //         self_tx,
                //         pending,
                //         OutgoingRequest {
                //             id: next_id,
                //             payload: OutgoingRequestPayload::Msg(format!("{INVOKE_ENDPOINT_PREFIX}{input}")),
                //             msg: None,
                //         },
                //     ).await.and_then(|response| {
                //         if let Some(error) = response.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
                //             Err(parse_worker_copy_error_payload(error))
                //         } else {
                //             Ok(response)
                //         }
                //     });
                //     let _ = respond_to.send(response);
                //     Ok::<_, ThoError>(())
                // });
                todo!()
            }
            CudaIPCActorMessage::ConsumeEndpoint { input, respond_to } => {
                // let self_tx = self.spawned_state.tx.clone();
                // let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                // let pending = Arc::clone(&self.spawned_state.pending);
                //
                // _= tokio::spawn(async move {
                //     let response = send_request_and_wait(
                //         self_tx,
                //         pending,
                //         OutgoingRequest {
                //             id: next_id,
                //             payload: OutgoingRequestPayload::Msg(format!("{CONSUME_ENDPOINT_PREFIX}{input}")),
                //             msg: None,
                //         },
                //     ).await.and_then(|response| {
                //         if let Some(error) = response.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
                //             Err(parse_worker_copy_error_payload(error))
                //         } else {
                //             Ok(response)
                //         }
                //     });
                //     let _ = respond_to.send(response);
                //
                //     Ok::<_, ThoError>(())
                // });
                todo!()
            }
            CudaIPCActorMessage::InvokeEndpointToFile { input, output, respond_to } => {
                let self_tx = self.spawned_state.tx.clone();
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.spawned_state.pending);
                {
                    let mut pending = pending.lock().await;
                    let old = pending.insert(next_id, PendingResponse::File { output: tokio::fs::File::from_std(output), respond_to, error: None });
                    debug_assert!(old.is_none());
                }
                if self_tx.send(OutgoingRequest { id: next_id, payload: OutgoingRequestPayload::Msg(format!("{INVOKE_ENDPOINT_PREFIX}{input}")), msg: None }).await.is_err() {
                    fail_one_pending(&pending, next_id, format!("failed to send streaming endpoint request {next_id}")).await;
                }
            }



            CudaIPCActorMessage::PurgeEndpoint { stage_id, endpoint_name, respond_to } => {
                debug!("CudaIPCActor::handle_message/CudaIPCActorMessage::PurgeEndpoint: endpoint_name={endpoint_name} stage_id={stage_id}");
                let self_tx = self.spawned_state.tx.clone();
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.spawned_state.pending);

                _= tokio::spawn(async move {

                    // let response = send_request_and_wait(
                    //     self_tx,
                    //     pending,
                    //     OutgoingRequest {
                    //         id: next_id,
                    //         payload: OutgoingRequestPayload::Msg(format!("{PURGE_ENDPOINT_PREFIX}{input}")),
                    //         msg: None,
                    //     },
                    // ).await.and_then(|response| {
                    //     if let Some(error) = response.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
                    //         Err(parse_worker_copy_error_payload(error))
                    //     } else {
                    //         Ok(response)
                    //     }
                    // })?;
                    let response = send_request_and_wait_msg(
                        next_id,
                        self_tx,
                        pending,
                        OutgoingRequest {
                            id: next_id,
                            payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                            msg: Some(CudaWorkerMsgIn {
                                request_id: next_id,
                                operation: Operation::PurgeStage {
                                    endpoint_name,
                                    stage_id,
                                },
                            }),
                        },
                    ).await?;//TODO handle error

                    debug!("CudaIPCActor::handle_message/CudaIPCActorMessage::PurgeEndpoint: response={response:?}");

                    let _ = respond_to.send(Ok(()));
                    Ok::<_, ThoError>(())
                });
            }
            CudaIPCActorMessage::DropEndpoint { stage_id, endpoint_name, respond_to } => {
                debug!("CudaIPCActor::handle_message/CudaIPCActorMessage::DropEndpoint: stage_id={stage_id} endpoint_name={endpoint_name}");
                let self_tx = self.spawned_state.tx.clone();
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.spawned_state.pending);

                // Endpoint destruction is acknowledged so callers know the worker processed the removal before reusing the key.
                tokio::spawn(async move {
                    let response = send_request_and_wait_msg(
                        next_id,
                        self_tx,
                        pending,
                        OutgoingRequest {
                            id: next_id,
                            payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                            msg: Some(CudaWorkerMsgIn {
                                request_id: next_id,
                                operation: Operation::DropStage {
                                    endpoint_name,
                                    stage_id,
                                },
                            }),
                        },
                    ).await?;
                    let _ = respond_to.send(Ok(()));
                    Ok::<_, ThoError>(())
                });
            }

        }

        Ok(())
    }

    async fn handle_reply(respond_to: oneshot::Sender<Result<(), ThoError>>, id: u64, reply_rx: tokio::sync::oneshot::Receiver<Result<CudaWorkerMsgOut, ThoError>>) {
        match reply_rx
            .await
            .map_err(|_| ThoError::new(format!("response channel for drop request {id} was dropped"))) {
            Ok(Ok(reply)) => {
                match reply.status {
                    OpResult::Ok(_) => {
                        debug!("handle_message: CudaIPCActorMessage::DropThoTensor id={id:?} send ok");

                        let _ = respond_to.send(Ok(()));
                    }
                    OpResult::Failed { msg, is_oom, needs_reset } => {
                        debug!("handle_message: CudaIPCActorMessage::DropThoTensor id={id:?} send err {msg:?}");

                        let _ = respond_to.send(Err(ThoError::from_worker(msg, is_oom, needs_reset)));
                    }
                }
            }
            e => {
                debug!("handle_message: CudaIPCActorMessage::DropThoTensor id={id:?} send err {e:?}");

                let _ = respond_to.send(Err(ThoError::new(format!("unexpected drop response: {e:?}"))));
            }
        }
    }

    fn copy_result(response: CudaWorkerMsgOut) -> Result<WorkerCopyDonePayload, ThoError> {
        match response.status {
            OpResult::Ok(Some(output)) => {
                // The copy protocol has exactly one typed success payload; reject unrelated worker outputs explicitly.
                let output: Box<dyn Any> = output;
                output.downcast::<WorkerCopyDonePayload>()
                    .map(|output| *output)
                    .map_err(|_| ThoError::new(format!("worker copy request {} returned an unexpected output type", response.request_id)))
            }
            OpResult::Ok(None) => Err(ThoError::new(format!("worker copy request {} returned no output", response.request_id))),
            OpResult::Failed { msg, is_oom, needs_reset } => Err(ThoError::from_worker(msg, is_oom, needs_reset)),
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaIPActor::handle_ref_d2h_copy"))]
    async fn handle_ref_d2h_copy(
        &mut self,
        tensor_key: u64,
        dst_device: ThoDevice,
        respond_to: oneshot::Sender<Result<ThoTensor, ThoError>>,
    ) -> Result<(), ThoError> {
        if dst_device.tho_device_idx() != 0 {
            return Err(ThoError::new(format!("worker tensor ref copy currently supports only CPU destination, got dst_device={dst_device:?}")));
        }

        let describe_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let pending = Arc::clone(&self.spawned_state.pending);
        let self_tx = self.spawned_state.tx.clone();
        let copy_ctx = Arc::clone(&self.copy_ctx);
        let buf_size = self.buf_size;
        let worker_id = self.tho_device_idx as u64;

        debug!("CudaIPCActor::handle_ref_d2h_copy: describe_id={describe_id} buf_size={buf_size}");

        let setup_result = async {
            let response = match send_request_and_wait_msg(
                describe_id,
                self_tx.clone(),
                Arc::clone(&pending),
                OutgoingRequest {
                    id: describe_id,
                    payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                    msg: Some(
                        CudaWorkerMsgIn {
                            request_id: describe_id,
                            operation: Operation::DescribeRef { tensor_key },
                        }
                    ),
                },
            ).await {
                Ok(v) => v,
                Err(e) => {
                    error!("CudaIPCActor::handle_ref_d2h_copy: got e={e:?}");
                    return Err(e);
                }
            };

            debug!("handle_ref_d2h_copy: response={response:?}");

            let CudaWorkerMsgOut{status: OpResult::Ok(Some(info)), ..} = response else {
                let msg = format!("worker tensor tensor_key={tensor_key} not found");
                return Err(ThoError::from_worker(msg, false, false));
            };
            let info : Box<dyn Any> = info;
            let info = info.downcast::<WorkerTensorInfoPayload>().unwrap();//TODO unwrap
            debug!("handle_ref_d2h_copy: info={info:?}");


            if info.tensor_key != tensor_key {
                return Err(ThoError::new(format!("worker described tensor key={} but requested key={tensor_key}", info.tensor_key)));
            }

            let chunks = make_chunks(info.tensor_byte_len, buf_size, "worker IPC D2H copy")?;

            debug!("CudaIPCActor::handle_ref_d2h_copy: describe_id={describe_id} chunks={chunks:?}");

            let pool = {
                let copy_ctx = copy_ctx.read().await;
                Arc::clone(copy_ctx.ipc_bufs_pool.as_ref().ok_or_else(|| ThoError::new("missing ipc buffer pool"))?)
            };
            // The first completed chunk allocates this tensor; later chunk tasks copy into disjoint byte ranges before the coordinator returns it.
            let destination = Arc::new(Mutex::new(None::<Tensor>));
            let mut chunk_tasks = Vec::with_capacity(chunks.len());

            debug!("CudaIPCActor::handle_ref_d2h_copy: describe_id={describe_id} got pool");

            for chunk in chunks {
                let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&pending);
                let self_tx = self_tx.clone();
                let pool = Arc::clone(&pool);
                let info = info.clone();
                let destination = Arc::clone(&destination);

                // Each chunk has its own acknowledged worker transfer so CUDA copies can use separate streams and pooled buffers can be released independently.
                let chunk_task = tokio::spawn(async move {
                    let result = async {
                        let buf = pool.acquire().await;
                        if chunk.len > buf.len() {
                            return Err(ThoError::new(format!("D2H chunk len={} exceeds shared buffer len={}", chunk.len, buf.len())));
                        }

                        let (reply_tx, reply_rx) = oneshot::channel();
                        {
                            let mut pending = pending.lock().await;
                            // The final copy acknowledgement is a postcard CudaWorkerMsgOut.
                            let old = pending.insert(transfer_id, PendingResponse::Msg(reply_tx));
                            debug_assert!(old.is_none());
                        }

                        // let start_json = serde_json::to_string(&WorkerD2hStartPayload { transfer_id, tensor_key })
                        //     .map_err(|e| ThoError::new(format!("failed to encode D2H start payload: {e}")))?;
                        // if self_tx.send(OutgoingRequest { id: transfer_id, payload: OutgoingRequestPayload::Msg(format!("{D2H_START_PAYLOAD_PREFIX}{start_json}")), msg: None }).await.is_err() {
                        //     return Err(ThoError::new("worker IPC writer task is closed"));
                        // }

                        // let d2h_response = send_request_and_wait_msg(
                        //     transfer_id,
                        //     self_tx.clone(),
                        //     Arc::clone(&pending),
                        //     OutgoingRequest {
                        //         id: transfer_id,
                        //         payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                        //         msg: Some(
                        //             CudaWorkerMsgIn {
                        //                 request_id: transfer_id,
                        //                 operation: Operation::D2HStart { tensor_key },
                        //             }
                        //         ),
                        //     },
                        // ).await?;
                        self_tx.send(
                            OutgoingRequest {
                                id: transfer_id,
                                payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                                msg: Some(
                                    CudaWorkerMsgIn {
                                        request_id: transfer_id,
                                        operation: Operation::D2HStart { tensor_key },
                                    }
                                ),
                            },
                        ).await.map_err(|e|ThoError::new(e.to_string()))?;

                        // debug!("CudaIPCActor::handle_ref_d2h_copy: d2h_response={d2h_response:?}");


                        let metadata = IpcTensorTransferMetadata {
                            transfer_id,
                            buffer_id: buf.buffer_id(),
                            byte_len: chunk.len,
                            tensor_byte_len: info.tensor_byte_len,
                            tensor_offset: chunk.offset,
                            chunk_index: 0,
                            chunk_count: 1,
                            mapped_len: buf.len(),
                            offset: 0,
                            kind: info.kind,
                            shape: info.shape.clone(),
                            direction: IpcTransferDirection::DeviceToHost,
                            producer: IpcProcessRole::Worker { worker_id },
                            consumer: IpcProcessRole::Parent,
                        };
                        metadata.validate_for_buffer(&buf)?;
                        debug!("CudaIPCActor::Copy D2H: sending shared memfd chunk to worker: chunk={chunk:?} metadata={metadata:?} local_mmap_ptr={:?}", buf.as_ptr());

                        let fd = buf.try_clone_fd()?;
                        // if self_tx.send(OutgoingRequest { id: transfer_id, payload: OutgoingRequestPayload::SharedMemfdBuffer { meta: metadata.clone(), fd }, msg: None }).await.is_err() {
                        //     return Err(ThoError::new("worker IPC writer task is closed"));
                        // }

                        self_tx.send(
                            OutgoingRequest {
                                id: transfer_id,
                                payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                                msg: Some(
                                    CudaWorkerMsgIn {
                                        request_id: transfer_id,
                                        operation: Operation::SharedMemfd { metadata:metadata.clone(), fd:Some(fd) },
                                    }
                                ),
                            },
                        ).await.map_err(|e|ThoError::new(e.to_string()))?;

                        let reply = reply_rx
                            .await
                            .map_err(|_| ThoError::new(format!("response channel for D2H request {transfer_id} was dropped")))??;

                        let done = Self::copy_result(reply)?;
                        if done.transfer_id != transfer_id {
                            return Err(ThoError::new(format!("worker D2H response transfer_id={} but expected {transfer_id}", done.transfer_id)));
                        }

                        let destination_ptr = {
                            let mut destination = destination.lock().await;
                            if destination.is_none() {
                                // Delay the full CPU allocation until a worker has actually completed the first chunk.
                                *destination = Some(Tensor::f_empty(info.shape.as_slice(), (info.kind.into(), Cpu))?);
                            }
                            destination.as_ref().ok_or_else(|| ThoError::new("parent D2H destination was not allocated"))?.data_ptr() as usize
                        };

                        let result = spawn_blocking(move || {
                            copy_parent_d2h_chunk(metadata, buf, &info, destination_ptr)
                        }).await??;

                        Ok::<_, ThoError>(done.tensor_digest_hex)
                    }.await;

                    if result.is_err() {
                        remove_pending_request(&pending, transfer_id).await;
                    }
                    result
                });
                chunk_tasks.push((transfer_id, chunk_task));
            }

            Ok::<_, ThoError>((info, destination, chunk_tasks))
        }.await;

        let (info, destination, chunk_tasks) = match setup_result {
            Ok(setup) => setup,
            Err(error) => {
                let _ = respond_to.send(Err(error));
                return Ok(());
            }
        };

        tokio::spawn(async move {
            let result = async {
                let mut first_error = None;
                let mut worker_digest_hex = None;
                for (transfer_id, chunk_task) in chunk_tasks {
                    match chunk_task.await {
                        Ok(Ok(Some(digest))) => {
                            if worker_digest_hex.as_ref().is_some_and(|expected| expected != &digest) && first_error.is_none() {
                                first_error = Some(ThoError::new(format!("worker D2H chunks returned inconsistent tensor digests for transfer {transfer_id}")));
                            } else {
                                worker_digest_hex = Some(digest);
                            }
                        }
                        Ok(Ok(None)) => {}
                        Ok(Err(error)) => {
                            if first_error.is_none() {
                                first_error = Some(error);
                            }
                        }
                        Err(error) => {
                            remove_pending_request(&pending, transfer_id).await;
                            if first_error.is_none() {
                                first_error = Some(ThoError::new(format!("parent D2H chunk task for transfer {transfer_id} failed: {error}")));
                            }
                        }
                    }
                }
                if let Some(error) = first_error {
                    return Err(error);
                }

                let cpu_tensor = {
                    let mut destination = destination.lock().await;
                    match destination.take() {
                        Some(destination) => destination,
                        None => Tensor::f_empty(info.shape.as_slice(), (info.kind.into(), Cpu))?,
                    }
                };
                if let Some(hex) = worker_digest_hex {
                    let cpu_digest = tensor_digest(&cpu_tensor)?;
                    debug!("CudaIPCActor::Copy D2H: worker and parent tensor digests matched: digest={}", cpu_digest.hex);
                    if cpu_digest.hex.ne(&hex) {
                        return Err(ThoError::new(format!("worker D2H digest mismatch: worker={} parent={}", hex, cpu_digest.hex)));
                    }
                }

                ThoTensor::new((), cpu_tensor)
            }.await;

            debug!("CudaIPCActor::Copy D2H: done");

            let _ = respond_to.send(result);
        });

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaIPActor::handle_h2d_copy"))]
    fn handle_h2d_copy(&mut self, src: ThoTensor, dst_device: ThoDevice, respond_to: oneshot::Sender<Result<ThoTensor, ThoError>>) -> Result<(), ThoError> {
        let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let preflight_id = self.next_id.fetch_add(1, Ordering::Relaxed);

        let (reply_tx, reply_rx) = oneshot::channel();
        let pending = Arc::clone(&self.spawned_state.pending);
        let self_tx = self.spawned_state.tx.clone();
        let copy_ctx = Arc::clone(&self.copy_ctx);
        let buf_size = self.buf_size;
        let dst_worker_id = dst_device.tho_device_idx() as u64;

        let tensor_mem_size = tensor_mem_size(src.t_ref().kind(), &src.t_ref().size());
        debug!("CudaIPCActor::handle_h2d_copy: src_t.size={:?} tensor_mem_size={tensor_mem_size} buf_size={buf_size}",src.t_ref().size());
        if tensor_mem_size == 0 {
            return Err(ThoError::new("worker IPC H2D copy does not support zero-byte tensors"));
        }

        // The parent side of H2D only stages CPU bytes into memfd-backed chunks; the worker owns the later CUDA copy.
        let chunks = make_chunks(tensor_mem_size, buf_size, "copy_h2d_from_ptr")?;
        let chunk_count = chunks.len();
        validate_ipc_chunk_capacity(chunk_count, "worker IPC H2D copy")?;
        let src_kind = src.t_ref().kind();
        let src_shape = src.t_ref().size();
        let src_ptr: usize = src.t_ref().data_ptr() as usize;

        tokio::spawn(async move {
            let result = async move {

                // let preflight_json = serde_json::to_string(&WorkerH2dStartPayload {
                //     transfer_id,
                //     worker_id: dst_worker_id,//TODO use device
                //     kind: src_kind.into(),
                //     shape: src_shape.clone(),
                //     tensor_byte_len: tensor_mem_size,
                //     chunk_count,
                // }).map_err(|e| ThoError::new(format!("failed to encode H2D start payload: {e}")))?;
                //
                // let preflight_reply = send_request_and_wait(
                //     self_tx.clone(),
                //     Arc::clone(&pending),
                //     OutgoingRequest {
                //         id: preflight_id,
                //         payload: OutgoingRequestPayload::Msg(format!("{H2D_START_PAYLOAD_PREFIX}{preflight_json}")),
                //         msg: None,
                //     },
                // ).await?;
                // parse_worker_h2d_start_preflight_reply(&preflight_reply)?;


                let response = send_request_and_wait_msg(
                    transfer_id,
                    self_tx.clone(),
                    Arc::clone(&pending),
                    OutgoingRequest {
                        id: transfer_id,
                        payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                        msg: Some(
                            CudaWorkerMsgIn {
                                request_id: transfer_id,
                                operation: Operation::H2dStart {
                                    worker_id: dst_worker_id,
                                    kind: src_kind.into(),
                                    shape: src_shape.clone(),
                                    tensor_byte_len: tensor_mem_size,
                                    chunk_count,
                                },
                            }
                        ),
                    },
                ).await?;

                debug!("CudaIPCActor::handle_h2d_copy: response={response:?}");

                {
                    let mut pending = pending.lock().await;
                    // H2D preflight has completed; wait for the typed final copy acknowledgement.
                    let old = pending.insert(transfer_id, PendingResponse::Msg(reply_tx));
                    debug_assert!(old.is_none());
                }

                let final_result = async {
                    let mut held_buffers = Vec::new();//TODO remove from held_buffers?
                    for chunk in chunks {

                        let mut buf = copy_ctx.read().await.ipc_bufs_pool.as_ref().unwrap().acquire().await;
                        let src_off_ptr = src_ptr.checked_add(chunk.offset).ok_or_else(|| ThoError::new("source tensor pointer offset overflow"))?;
                        if chunk.len > buf.len() {
                            return Err(ThoError::new(format!("copy chunk len={} exceeds shared buffer len={}", chunk.len, buf.len())));
                        }
                        unsafe {
                            // SAFETY: src_ptr is the data pointer of src, chunk comes from copy_chunks over the tensor byte size, and the destination mmap range was checked against the buffer length above.
                            std::ptr::copy_nonoverlapping(src_off_ptr as *const u8, buf.as_mut_ptr(), chunk.len);
                        }
                        let metadata = IpcTensorTransferMetadata {
                            transfer_id,
                            buffer_id: buf.buffer_id(),
                            byte_len: chunk.len,
                            tensor_byte_len: tensor_mem_size,
                            tensor_offset: chunk.offset,
                            chunk_index: chunk.index,
                            chunk_count,
                            mapped_len: buf.len(),
                            offset: 0,//TODO
                            kind: src_kind.into(),
                            shape: src_shape.clone(),
                            direction: IpcTransferDirection::HostToDevice,
                            producer: IpcProcessRole::Parent,
                            consumer: IpcProcessRole::Worker { worker_id: dst_worker_id },
                        };
                        debug!("CudaIPCActor::handle_h2d_copy:, CudaIPCActor::Copy: copied tensor chunk to shared memfd: chunk={chunk:?} metadata={metadata:?} local_mmap_ptr={:?}", buf.as_ptr());

                        let fd = buf.try_clone_fd()?;
                        // if self_tx.send(OutgoingRequest { id: transfer_id, payload: OutgoingRequestPayload::SharedMemfdBuffer { meta: metadata, fd }, msg: None }).await.is_err() {
                        //     return Err(ThoError::new("worker IPC writer task is closed"));
                        // }
                        self_tx.send(
                            OutgoingRequest {
                                id: transfer_id,
                                payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                                msg: Some(
                                    CudaWorkerMsgIn {
                                        request_id: transfer_id,
                                        operation: Operation::SharedMemfd { metadata:metadata.clone(), fd:Some(fd) },
                                    }
                                ),
                            },
                        ).await.map_err(|e|ThoError::new(e.to_string()))?;
                        held_buffers.push(buf);

                    }

                    let reply = reply_rx
                        .await
                        .map_err(|_| ThoError::new(format!("response channel for request {transfer_id} was dropped")))??;

                    let done = Self::copy_result(reply)?;

                    if let Some(hex) = done.tensor_digest_hex.as_ref() {
                        let source_digest = tensor_digest(src.t_ref())?;
                        debug!("CudaIPCActor::handle_h2d_copy:  original tensor digest={source_digest:?}");
                        let source_digest_hex = source_digest.hex.clone();
                        if  source_digest_hex.ne(hex) {
                            return Err(ThoError::new(format!("worker copy digest mismatch for transfer {}: source={} copied={}", done.transfer_id, source_digest_hex, hex)));
                        }
                        debug!("CudaIPCActor::handle_h2d_copy: worker copied tensor digest matched: transfer_id={} digest={}", done.transfer_id, hex);
                    }

                    Ok::<_, ThoError>(ThoTensor::new_ref((), done.tensor_ref, src_shape, src_kind, None, dst_device.tho_device_idx()))
                }.await;

                if final_result.is_err() {
                    remove_pending_request(&pending, transfer_id).await;
                }

                final_result
            }.await;

            let _ = respond_to.send(result);
        });

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
    pending: PendingResponses,
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

        if let Some(msg) = req.msg {
            debug!("writer_loop: msg={msg:?}");

            if let Err(e) = write_frame(&mut writer, &postcard::to_allocvec(&msg).unwrap()/*TODO move to codec*/, "CudaWorkerMsg").await {
                error!("worker IPC writer failed: {e}");

                fail_one_pending(
                    &pending,
                    req.id,
                    format!("failed to write request {}: {e}", req.id),
                ).await;

                fail_all_pending(&pending, format!("worker IPC writer stopped: {e}")).await;
                break;
            }
            if let Operation::SharedMemfd {fd:Some(fd), ..} = msg.operation {
                debug!("writer_loop: fd={fd:?}");
                if let Err(e) = send_fd(writer.as_ref(), fd.as_fd()).await.map_err(|e|ThoError::new(e.to_string())) {
                    error!("worker IPC fd send failed: {e}");
                    fail_one_pending(&pending, req.id, format!("failed to send request {} fd: {e}", req.id)).await;
                    fail_all_pending(&pending, format!("worker IPC writer stopped while sending fd: {e}")).await;
                    break;
                }
            }

            continue
        }

        let frame = match encode_wire_frame(req.id, &req.payload) {
            Ok(frame) => frame,
            Err(e) => {
                error!("worker IPC writer failed to encode request {}: {e}", req.id);
                fail_one_pending(&pending, req.id, format!("failed to encode request {}: {e}", req.id)).await;
                continue;
            }
        };

        if let Err(e) = write_frame(&mut writer, &frame, "CudaIPCActor").await {
            error!("worker IPC writer failed: {e}");

            fail_one_pending(
                &pending,
                req.id,
                format!("failed to write request {}: {e}", req.id),
            ).await;

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
    pending: PendingResponses,
    cancel: CancellationToken,
) where
    R: AsyncRead + Unpin,
{
    loop {
        let frame = tokio::select! {
            _ = cancel.cancelled() => {
                debug!("worker IPC reader cancelled");

                break;
            }

            result = read_frame(&mut reader, "CudaIPCActor", MAX_FRAME_LEN) => {
                match result {
                    Ok(frame) => frame,
                    Err(e) => {
                        debug!("worker IPC reader stopped: {e}");

                        fail_all_pending(&pending, format!("worker IPC reader stopped: {e}"),).await;

                        break;
                    }
                }
            }
        };

        match postcard::from_bytes::<CudaWorkerMsgOut>(&frame) {
            Ok(msg) => {
                debug!("reader_loop: msg={msg:?}");
                let pending_response = {
                    let mut pending = pending.lock().await;
                    let waiter = match pending.remove(&msg.request_id) {
                        None => {
                            error!("no waiter for {}", msg.request_id);
                            continue
                        }
                        Some(v) => {
                            debug!("reader_loop: pending response={v:?} for msg.request_id={}", msg.request_id);
                            v
                        }
                    };
                    if let PendingResponse::Msg(msg_waiter) = waiter {
                        msg_waiter.send(Ok(msg)).expect("failed to send todo");
                        debug!("reader_loop: send ok");
                    } else {
                        error!("Unexpected waiter for msg: waiter={waiter:?}")
                    }
                };
                continue
            }
            Err(e) => {
                error!("bad worker response frame: {e}");
            }
        }

    }

    debug!("worker IPC reader loop finished");
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="parse_wire_msg"))]
pub(crate) fn parse_wire_msg(frame: &str) -> Result<(u64, String), ThoError> {
    let Some((id, payload)) = frame.split_once('\t') else {
        return Err(ThoError::new(format!("bad wire frame without request id separator: {frame:?}")));
    };

    let id = id
        .parse::<u64>()
        .map_err(|e| ThoError::new(format!("bad request id {id:?}: {e}")))?;

    Ok((id, payload.to_string()))
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="parse_wire_response"))]
fn parse_wire_response(frame: Vec<u8>) -> Result<(u64, IncomingResponsePayload), ThoError> {

    let separator = frame.iter().position(|byte| *byte == b'\t')
        .ok_or_else(|| ThoError::new("bad wire response without request id separator"))?;

    let id = std::str::from_utf8(&frame[..separator])
        .map_err(|e| ThoError::new(format!("bad utf8 request id in wire response: {e}")))?
        .parse::<u64>()
        .map_err(|e| ThoError::new(format!("bad request id in wire response: {e}")))?;

    let payload = &frame[separator + 1..];

    // if let Some(payload) = payload.strip_prefix(BINARY_RESPONSE_CHUNK_PREFIX) {
    //     Ok((id, IncomingResponsePayload::BinaryChunk(payload.to_vec())))
    // } else if payload == BINARY_RESPONSE_END_PAYLOAD {
    //     Ok((id, IncomingResponsePayload::BinaryEnd))
    // } else {
        let payload = String::from_utf8(payload.to_vec())
            .map_err(|e| ThoError::new(format!("received non-utf8 message response: {e}")))?;
        Ok((id, IncomingResponsePayload::Msg(payload)))
    // }
}


#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="parse_worker_h2d_start_preflight_reply"))]
fn parse_worker_h2d_start_preflight_reply(payload: &str) -> Result<(), ThoError> {
    if let Some(error) = payload.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
        return Err(parse_worker_copy_error_payload(error));
    }
    if payload == H2D_START_OK_PAYLOAD {
        Ok(())
    } else {
        Err(ThoError::new(format!("worker H2D preflight response was not {H2D_START_OK_PAYLOAD:?}: {payload:?}")))
    }
}


#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="parse_worker_tensor_info_payload"))]
fn parse_worker_tensor_info_payload(payload: &str) -> Result<WorkerTensorInfoPayload, ThoError> {
    if let Some(error) = payload.strip_prefix(COPY_ERROR_PAYLOAD_PREFIX) {
        return Err(parse_worker_copy_error_payload(error));
    }
    let info_json = payload.strip_prefix(TENSOR_INFO_PAYLOAD_PREFIX)
        .ok_or_else(|| ThoError::new(format!("worker tensor info response did not start with {TENSOR_INFO_PAYLOAD_PREFIX:?}: {payload:?}")))?;

    serde_json::from_str(info_json)
        .map_err(|e| ThoError::new(format!("failed to parse worker tensor info response: {e}")))
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="parse_worker_copy_error_payload"))]
fn parse_worker_copy_error_payload(payload: &str) -> ThoError {
    debug!("parse_worker_copy_error_payload: payload={payload}");

    match serde_json::from_str::<WorkerCopyErrorPayload>(payload) {
        Ok(error) => error.into_error(),
        Err(_) => WorkerCopyErrorPayload::from_legacy_message(payload).into_error(),
    }
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="copy_parent_d2h_chunk"))]
fn copy_parent_d2h_chunk(
    metadata: IpcTensorTransferMetadata,
    buffer: Guard<SharedMemfdBuffer>,
    info: &WorkerTensorInfoPayload,
    destination_ptr: usize,
) -> Result<(), ThoError> {
    metadata.validate_for_buffer(&buffer)?;
    if metadata.kind != info.kind || metadata.shape != info.shape || metadata.tensor_byte_len != info.tensor_byte_len {
        return Err(ThoError::new(format!("parent D2H chunk metadata does not match worker tensor info for transfer {}", metadata.transfer_id)));
    }
    let destination = destination_ptr
        .checked_add(metadata.tensor_offset)
        .ok_or_else(|| ThoError::new("parent D2H destination pointer overflow"))?;
    let source = unsafe {
        // SAFETY: validate_for_buffer checked offset + byte_len against the live mmap and the buffer is held until this copy finishes.
        buffer.as_ptr().add(metadata.offset)
    };
    unsafe {
        // SAFETY: source points into the validated live mmap chunk, destination points into the owned CPU tensor byte range checked by metadata, and parallel chunk tasks write disjoint generated ranges.
        std::ptr::copy_nonoverlapping(source, destination as *mut u8, metadata.byte_len);
    }
    Ok(())
}

async fn send_request_and_wait(
    tx: mpsc::Sender<OutgoingRequest>,
    pending: PendingResponses,
    request: OutgoingRequest,
) -> Result<String, ThoError> {
    match send_request_and_wait_payload(tx, pending, request).await? {
        IncomingResponsePayload::Msg(response) => Ok(response),
    }
}

async fn send_request_and_wait_payload(
    tx: mpsc::Sender<OutgoingRequest>,
    pending: PendingResponses,
    request: OutgoingRequest,
) -> Result<IncomingResponsePayload, ThoError> {
    let id = request.id;
    let (reply_tx, reply_rx) = oneshot::channel();
    {
        let mut pending = pending.lock().await;
        let old = pending.insert(id, PendingResponse::Buffered(reply_tx));
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

async fn send_request_and_wait_msg<R>(
    id:u64,
    tx: mpsc::Sender<R>,
    pending: PendingResponses,
    request: R,
) -> Result<CudaWorkerMsgOut, ThoError> {
    let (reply_tx, reply_rx) = oneshot::channel();
    {
        let mut pending = pending.lock().await;
        let old = pending.insert(id, PendingResponse::Msg(reply_tx));
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
    pending: &PendingResponses,
    id: u64,
) {
    let mut pending = pending.lock().await;
    pending.remove(&id);
}


async fn fail_one_pending(
    pending: &PendingResponses,
    id: u64,
    msg: String,
) {
    let reply_tx = {
        let mut pending = pending.lock().await;
        pending.remove(&id)
    };


    if let Some(pending_response) = reply_tx {
        match pending_response {
            PendingResponse::Buffered(reply_tx) => {
                let _ = reply_tx.send(Err(ThoError::new(msg)));
            }
            PendingResponse::File { respond_to, .. } => {
                let _ = respond_to.send(Err(ThoError::new(msg)));
            }
            PendingResponse::Msg(respond_to) => {
                let _ = respond_to.send(Err(ThoError::new(msg)));
            }
        }
    }
}

async fn fail_all_pending(
    pending: &PendingResponses,
    msg: String,
) {
    let pending_items = {
        let mut pending = pending.lock().await;
        pending.drain().collect::<Vec<_>>()
    };

    for (_id, pending_response) in pending_items {
        match pending_response {
            PendingResponse::Buffered(reply_tx) => {
                let _ = reply_tx.send(Err(ThoError::new(msg.clone())));
            }
            PendingResponse::File { respond_to, .. } => {
                let _ = respond_to.send(Err(ThoError::new(msg.clone())));
            }
            PendingResponse::Msg(respond_to) => {
                let _ = respond_to.send(Err(ThoError::new(msg.clone())));
            }
        }
    }
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="encode_wire_frame"))]
fn encode_wire_frame(id: u64, payload: &OutgoingRequestPayload) -> Result<Vec<u8>, ThoError> {
    let mut frame = format!("{id}\t").into_bytes();
    match payload {
        OutgoingRequestPayload::Msg(msg) => frame.extend_from_slice(msg.as_bytes()),

        OutgoingRequestPayload::SharedMemfdBuffer{ meta, .. } => {
            let metadata_json = serde_json::to_string(meta)
                .map_err(|e| ThoError::new(format!("failed to encode shared memfd metadata: {e}")))?;

            frame.extend_from_slice(format!("{SHARED_MEMFD_PAYLOAD_PREFIX}{metadata_json}").as_bytes());
        }
    }

    Ok(frame)
}


#[derive(Debug, Clone)]
pub struct CudaIPCActorHandle {
    sender: mpsc::Sender<CudaIPCActorMessage>,
}

///executes cuda actions in a separate process started from CudaWorker::run_worker
impl CudaIPCActorHandle {
    pub async fn new(tho_device_id:usize, cuda_worker_args: Vec<String>) -> Result<Self, ThoError> {
        debug!("CudaIPCActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let actor = CudaIPCActor::new(tho_device_id, cuda_worker_args, receiver).await?;
        tokio::spawn(CudaIPCActor::run(actor));

        Ok(Self {
            sender,
        })
    }

    #[deprecated]//use install_stage_2
    async fn install_endpoint_raw(&self, init_msg: String) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::InstallEndpoint {
                init_msg,
                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::install_endpoint_raw",
            agg_perf="CudaIPCActorHandle::install_endpoint_raw",
        )).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }

    pub async fn invoke_stage(&self, endpoint_name: String, stage_id: u32, input: Box<dyn WorkerInput>, consume:bool) -> Result<CudaWorkerMsgOut, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                input: CudaWorkerMsgIn {
                    request_id: 0,
                    operation: Operation::InvokeStage {
                        endpoint_name,
                        stage_id,
                        input,//clones refs
                        consume,
                    },
                },
                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::send_msg",
            agg_perf="CudaIPCActorHandle::send_msg",
        )).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }

    pub async fn install_stage_2(&self, endpoint_name: String, stage_id: u32, init_data: Box<dyn WorkerInput>) -> Result<CudaWorkerMsgOut, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                input: CudaWorkerMsgIn {
                    request_id: 0,
                    operation: Operation::InstallStage {
                        endpoint_name,
                        stage_id,
                        init_data,
                    },
                },
                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::send_msg",
            agg_perf="CudaIPCActorHandle::send_msg",
        )).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }


    #[deprecated]//use invoke_stage
    async fn invoke_endpoint_raw(&self, input: String) -> Result<String, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::InvokeEndpoint {
                input,
                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::invoke_endpoint_raw",
            agg_perf="CudaIPCActorHandle::invoke_endpoint_raw",
        )).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }

    #[deprecated]//use invoke_stage
    async fn consume_endpoint_raw(&self, input: String) -> Result<String, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::ConsumeEndpoint {
                input,
                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::consume_endpoint_raw",
            agg_perf="CudaIPCActorHandle::consume_endpoint_raw",
        )).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }

}

struct IPCHanldeCTx {

}
#[async_trait]
impl HandleCtx for IPCHanldeCTx {

}

#[async_trait]
impl CudaActorHandle for CudaIPCActorHandle {


    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaIPCActorHandle::copy"))]
    async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) -> Result<ThoTensor, ThoError> {
        debug!("CudaIPCActorHandle::copy: src={src:?} dst_device={dst_device:?}");
        assert_ne!(src.tho_device(), dst_device, "should not copy to same device");
        match &dst_device.into() {
            Cpu => {
                assert_ne!(src.tho_device(), Cpu.into(), "src device should not be CPU when copying to CPU");
                assert!(src.is_ref(), "In IPC mode, src must be a reference when copying to CPU.")
            }
            Cuda(_)=>{
                assert_eq!(src.tho_device(), Cpu.into(), "src device should be CPU when copying to GPU");
                assert!(!src.is_ref(), "IPC mode, src must be value when copying to GPU")
            }
            _ => {
                panic!("not supported")
            }
        }

        trace!("CudaIPCActorHandle::copy: src={src:?} dst_device={dst_device:?} backtrace={}", Backtrace::capture().to_string());
        debug!("CudaIPCActorHandle::copy: src={src:?} dst_device={dst_device:?}");

        if src.size().iter().all(|x| *x == 0) {
            return Ok(Tensor::empty(src.size(), (src.kind(), dst_device.into())).into());
        }

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Copy {
                src,
                dst_device,
                respond_to,
            }
        ).await.map_err(|e|{
            let msg = format!("e={e:?}");
            error!("CudaIPCActorHandle::copy: msg={msg:?}");

            ThoError::new(msg)
        })?;//TODO error


        let r = recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?;
        debug!("CudaIPCActorHandle::copy:  get result: r={r:?}");
        r
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
        debug!("CudaIPCActorHandle::shutdown");

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Shutdown {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        debug!("CudaIPCActorHandle::shutdown: recv={recv:?}");

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?

    }

    async fn install_endpoint(&self, stage_id: u32, init_msg: String) -> Result<(), ThoError> {
        self.install_endpoint_raw(encode_stage_payload(stage_id, &init_msg)).await
    }

    async fn invoke_endpoint(&self, stage_id: u32, input: String) -> Result<String, ThoError> {
        let response = self.invoke_endpoint_raw(encode_stage_payload(stage_id, &input)).await?;
        String::from_utf8(decode_endpoint_response_bytes(&response)?).map_err(|e| ThoError::new(format!("endpoint response is not utf8: {e}")))
    }



    async fn install_endpoint_bytes(&self, stage_id: u32, init_msg: Vec<u8>) -> Result<(), ThoError> {
        self.install_endpoint_raw(
            encode_stage_payload(stage_id, &String::from_utf8(init_msg).map_err(|e| ThoError::new(format!("endpoint init payload is not utf8: {e}")))?)
        ).await
    }

    async fn invoke_endpoint_bytes(&self, stage_id: u32, input: Vec<u8>) -> Result<Vec<u8>, ThoError> {
        let input = String::from_utf8(input).map_err(|e| ThoError::new(format!("endpoint invoke payload is not utf8: {e}")))?;
        let response = self.invoke_endpoint_raw(
            encode_stage_payload(stage_id, &input)
        ).await?;
        decode_endpoint_response_bytes(&response)
    }

    async fn consume_endpoint_bytes(&self, stage_id: u32, input: Vec<u8>) -> Result<Vec<u8>, ThoError> {
        let input = String::from_utf8(input).map_err(|e| ThoError::new(format!("endpoint consume payload is not utf8: {e}")))?;
        let response = self.consume_endpoint_raw(
            encode_stage_payload(stage_id, &input)
        ).await?;
        decode_endpoint_response_bytes(&response)
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


    async fn init_handle_ctx(&self) -> Result<Box<dyn crate::copy_ctx::HandleCtx>, ThoError> {
        Ok(Box::new(IPCHanldeCTx{}))
    }

    async fn purge_endpoint_bytes(&self, stage_id: u32, endpoint_name: String) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::PurgeEndpoint {
                stage_id,
                endpoint_name,
                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::purge_endpoint_bytes",
            agg_perf="CudaIPCActorHandle::purge_endpoint_bytes",
        )).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }

    async fn drop_endpoint(&self, stage_id: u32, endpoint_name: String) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::DropEndpoint {
                stage_id,
                endpoint_name,
                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::drop_endpoint",
            agg_perf="CudaIPCActorHandle::drop_endpoint",
        )).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }
}

pub(crate) async fn write_text_frame<W>(
    writer: &mut W,
    text: &str, descr: &str,
) -> Result<(), ThoError>
where
    W: AsyncWrite + Unpin,
{
    write_frame(writer, text.as_bytes(), descr).await?;
    Ok(())
}

pub(crate) async fn read_text_frame<R>(
    reader: &mut R, descr: &str,
) -> Result<String, ThoError>
where
    R: AsyncRead + Unpin,
{
    let payload = read_frame(reader, descr, MAX_FRAME_LEN).await?;

    match postcard::from_bytes::<CudaWorkerMsgIn>(&payload) {
        Ok(v) => {
            debug!("postcard: ok={v:?}");
            return Ok("Ok".to_string())
        },
        Err(e)=>{
            debug!("postcard: e={e:?}");
        }
    };


    String::from_utf8(payload)
        .map_err(|e| ThoError::new(format!("received non-utf8 frame: {e}")))
}

async fn write_frame<W>(
    writer: &mut W,
    payload: &[u8],
    descr: &str,
) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    debug!("write_frame: descr={descr} payload.len={}", payload.len());

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

    debug!("write_frame: descr={descr} done");

    Ok(())
}

pub async fn read_frame<R>(
    reader: &mut R,
    descr: &str,
    max_len: usize,
) -> std::io::Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    debug!("read_frame: descr={descr}");
    let mut len_buf = [0u8; 4];


    match reader.read_exact(&mut len_buf).await {
        Ok(v) => Ok(v),
        Err(e) =>  {
            if !GlobalCtx::is_shutting_down(){
                error!("read_frame/read_exact error: descr={descr} len_buf {e}");
            }

            Err(e)
        }
    }?;

    let len = u32::from_le_bytes(len_buf) as usize;

    if len > max_len {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("read_frame: decr={descr} frame too large: {len} bytes"),
        ));
    }

    let mut payload = vec![0u8; len];

    match reader.read_exact(&mut payload).await {
        Ok(v) => Ok(v),
        Err(e) =>  {
            if !GlobalCtx::is_shutting_down() {
                error!("read_frame/read_exact error: payload, len={len} {e}");
            }
            Err(e)
        }
    }?;

    debug!("read_frame: descr={descr} got payload.len={}", payload.len());

    Ok(payload)
}

#[derive(Debug, Serialize, Deserialize, Clone, Copy)]
pub(crate) struct DeviceInfo {
    pub(crate) free_mem:usize,
    pub(crate) total_mem:usize,
}

#[cfg(test)]
mod tests {
    use std::any::Any;
    use crate::copy_ctx::CudaActorHandle;
    use crate::cuda_utils::{cuda_mem, fail_with_unrecoverable_error, speed_gb_per_s};
    use crate::ipc::cuda_process::{encode_endpoint_payload, read_text_frame, write_text_frame, CudaIPCActor, CudaIPCActorHandle, DeviceInfo, IPC_SOCK_ENV};
    use crate::ipc::cuda_wokrer::{CudaWorker, InitFn, WokerRunCtx};
    use crate::ipc::json_ser::SyncStageIpcWrapper;
    use crate::ipc::tests::{init_ipc_test, run_ipc_test};
    use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster};
    use crate::runner::common::cachable::DevicesWrap;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
    use crate::runner::run_ctx_2::RunCtxImpl;
    // use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, OOMRunnerConfig};
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    // use crate::torch_runner::tests::load_and_clc::loader::LoadedTensor;
    use crate::utils::{lin_for_size, tensor_digest};
    use crate::ws::{mock_ws_simp, Workspace};
    use log::{debug, info};
    use parking_lot::RwLock;
    use rand::random_range;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::process::Stdio;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Instant;
    use futures_util::future::BoxFuture;
    use mockall::PredicateBoxExt;
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{Device, Kind, Tensor};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::perf_logging::{PERF, format_perf_map, init_tracing};
    use thoosa_shared::tracked_await::track_await;
    use thoosa_shared::traits::RunCtx;
    use thoosa_shared::{Cause, ThoError, mono_key};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{UnixListener, UnixStream};
    use tokio::process::{Child, Command};
    use tokio::sync::Mutex;
    use tokio::task::JoinSet;
    use futures_util::FutureExt;

    macro_rules! pure_module_path {
        () => {{
            let p = module_path!();

            match p.split_once("::") {
                Some((_crate_name, rest)) => rest,
                None => "",
            }
        }};
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

    #[test]
    fn test_copy_ipc_cuda_worker_propagates_oom() -> Result<(), ThoError>{
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(async {
            init_ipc_test("test_copy_ipc_cuda_worker_propagates_oom", pure_module_path!()).await?;

            let tho_device_idx = 1;

            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let (free_mem, _total_mem) = cuda_mem();
            let elements = free_mem / Float.elt_size_in_bytes() + 1;
            let dim_1 = 4usize;// 10_000; //
            let dim_0 = elements.div_ceil(dim_1);//10_000; //
            debug!("test_copy_ipc_cuda_worker_propagates_oom, dim_0={dim_0} dim_1={dim_1} ");

            let _rand = Tensor::f_rand([dim_0 as i64, dim_1 as i64-1], FLOAT_CUDA)?;
            debug!("test_copy_ipc_cuda_worker_propagates_oom, _rand={_rand:?}");


            let too_large_tensor = Tensor::f_rand([dim_0 as i64, 1], FLOAT_CPU)?;

            let copy_result = tokio::time::timeout(
                std::time::Duration::from_secs(600),
                track_await(cuda_actor.copy(too_large_tensor.into(), tho_device_idx.into())),
            ).await.map_err(|_| ThoError::new("timed out waiting for worker IPC oversized H2D copy to fail"))?;



            let error = match copy_result {
                Ok(_) => return Err(ThoError::new("worker IPC copy unexpectedly succeeded for oversized tensor")),
                Err(error) => error,
            };

            debug!("run_ipc_copy_oom: error={error:?}");

            match error.what() {
                Cause::CudaWorker { is_oom, is_retryable_oom, needs_cuda_ctx_reset, .. } => {
                    assert!(*is_oom, "worker OOM flag was not propagated: {error:?}");
                    assert!(*is_retryable_oom, "worker retryable OOM flag was not propagated: {error:?}");
                    assert!(!*needs_cuda_ctx_reset, "plain worker OOM should not request CUDA context reset: {error:?}");
                }
                other => return Err(ThoError::new(format!("expected Cause::CudaWoker, got {other:?}: {error:?}"))),
            }
            assert!(error.is_oom(), "ThoError::is_oom should delegate to Cause::CudaWoker: {error:?}");
            assert!(error.is_retryable_oom(), "ThoError::is_retryable_oom should delegate to Cause::CudaWoker: {error:?}");


            debug!("All ok");

            Ok(())
        },
       async |sock_path| {
            let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
            CudaWorker::run_worker(sock_path, &ws,  HashMap::new(), ThoDevice::from(Cuda(0))).await
        })
    }



    #[test]
    fn heavy_copy_ipc_cuda_worker_once() -> Result<(), ThoError>{
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(async {
            init_ipc_test("heavy_copy_ipc_cuda_worker_once", pure_module_path!()).await?;

            let tho_device_idx = 1;
            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let (d2h, h2d)  = run_ipc_copy_perf(cuda_actor, 3, 3).await?;

            info!("\n IPC worker copy d2h speed: {d2h:#2} GB/s, h2d speed {h2d:#2} GB/s");


            Ok(())
        }, async |sock_path| {
            let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
            CudaWorker::run_worker(sock_path, &ws, HashMap::new(), ThoDevice::from(Cuda(0))).await
        })
    }

    #[test]
    fn test_copy_ipc_cuda_worker_ok() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(async {
            init_ipc_test("test_copy_ipc_cuda_worker_ok", pure_module_path!()).await?;

            let tho_device_idx = 1;

            let main_process_cpu_tensor = Tensor::rand(&[5_000, 10_000], FLOAT_CPU);
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

            cuda_actor.drop_tho_tensor(ThoTensor::new_ref((), reff, vec![], tch::Kind::Float, None, 1)).await?;

            debug!("run_main: drop ok");

            let r = track_await(cuda_actor.copy(
                ThoTensor::new_ref((), reff, vec![1000, 10000], tch::Kind::Float, None, 1),
                0.into(), //Cpu
            )).await;

            debug!("run_main: copy r={r:?}");

            let Err(ThoError { msg, what:Cause::CudaWorker { msg:what_msg, is_oom, is_retryable_oom, needs_cuda_ctx_reset },.. }) = r else {
                panic!("Got r={r:?}, expected CudaWorker error");
            } ;

            debug!("run_main: what_msg={what_msg:?}");
            debug!("run_main: msg={msg:?}");


            assert_eq!("Worker", what_msg);//TODO err code
            assert_eq!("worker tensor tensor_key=1 not found", msg);//TODO err code


            info!("main: \n{}", format_perf_map(&PERF));

            Ok(())
        }, async |sock_path| {
            let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
            CudaWorker::run_worker(sock_path, &ws, HashMap::new(), ThoDevice::from(Cuda(0))).await
        })
    }

}
