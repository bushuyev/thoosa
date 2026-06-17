use std::collections::HashMap;
use std::collections::HashSet;
use std::ffi::c_void;
use std::fmt::Debug;
use std::os::fd::{AsFd, OwnedFd};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use dashmap::DashMap;
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
use tokio::task::JoinHandle;
use tub::{Guard, Pool};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::traits::{Data, Named};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::cuda_result::check_cuda_status;
use thoosa_shims::memcpy::{tho_cu_memcpy_dtoh_async, tho_cu_memcpy_htod_async};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use crate::copy_ctx::{CopyCtx, CudaActorMessage};
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
use crate::torch_runner::tests::{ClcTask, LoadedTensor};

pub(crate) const IPC_SOCK_ENV: &str = "THOOSA_WORKER_IPC_SOCK";
const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;
pub(crate) const SHARED_MEMFD_PAYLOAD_PREFIX: &str = "shared-memfd:";
const COPY_DONE_PAYLOAD_PREFIX: &str = "copy-done:";
pub(crate) const DESCRIBE_REF_PAYLOAD_PREFIX: &str = "describe-ref:";
pub(super) const TENSOR_INFO_PAYLOAD_PREFIX: &str = "tensor-info:";
pub(crate) const D2H_START_PAYLOAD_PREFIX: &str = "d2h-start:";
const WORKER_COPY_STREAMS: usize = 16;

#[derive(Debug, Serialize, Deserialize)]
struct WorkerCopyDonePayload {
    transfer_id: u64,
    tensor_digest_hex: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(super) struct WorkerTensorInfoPayload {
    tensor_key: u64,
    kind: IpcTensorKind,
    shape: Vec<i64>,
    tensor_byte_len: usize,
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
    metadata: IpcTensorTransferMetadata,
    received_bytes: usize,
    seen_chunks: HashSet<usize>,
    chunks: Vec<WorkerH2dChunk>,
}

impl PendingWorkerH2dTransfer {
    pub(crate) fn new(request_id: u64, chunk: WorkerH2dChunk) -> Result<Self, ThoError> {
        let mut seen_chunks = HashSet::new();
        if !seen_chunks.insert(chunk.metadata.chunk_index) {
            return Err(ThoError::new(format!("duplicate H2D chunk index {}", chunk.metadata.chunk_index)));
        }
        Ok(Self {
            request_id,
            metadata: chunk.metadata.clone(),
            received_bytes: chunk.metadata.byte_len,
            seen_chunks,
            chunks: vec![chunk],
        })
    }

    pub(crate) fn push(&mut self, chunk: WorkerH2dChunk) -> Result<(), ThoError> {
        if self.request_id != chunk.metadata.transfer_id {
            return Err(ThoError::new(format!("H2D request id={} does not match transfer_id={}", self.request_id, chunk.metadata.transfer_id)));
        }
        if self.metadata.kind != chunk.metadata.kind || self.metadata.shape != chunk.metadata.shape || self.metadata.tensor_byte_len != chunk.metadata.tensor_byte_len || self.metadata.chunk_count != chunk.metadata.chunk_count {
            return Err(ThoError::new(format!("inconsistent H2D chunk metadata for transfer {}", self.metadata.transfer_id)));
        }
        if !self.seen_chunks.insert(chunk.metadata.chunk_index) {
            return Err(ThoError::new(format!("duplicate H2D chunk index {} for transfer {}", chunk.metadata.chunk_index, self.metadata.transfer_id)));
        }
        self.received_bytes = self.received_bytes.checked_add(chunk.metadata.byte_len).ok_or_else(|| ThoError::new("H2D received byte count overflow"))?;
        if self.received_bytes > self.metadata.tensor_byte_len {
            return Err(ThoError::new(format!("H2D transfer {} received {} bytes but tensor has {} bytes", self.metadata.transfer_id, self.received_bytes, self.metadata.tensor_byte_len)));
        }
        self.chunks.push(chunk);
        Ok(())
    }

    pub(crate) fn is_complete(&self) -> bool {
        self.received_bytes == self.metadata.tensor_byte_len && self.chunks.len() == self.metadata.chunk_count
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
        write_text_frame(&mut writer, &frame).await?;
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

fn copy_worker_h2d_transfer_to_cuda(
    mut transfer: PendingWorkerH2dTransfer,
    stream: Guard<ThoCudaStream>,
) -> Result<(u64, Tensor, String), ThoError> {
    let cuda_device = worker_cuda_device(&transfer.metadata)?;
    if stream.device() != cuda_device as i32 {
        return Err(ThoError::new(format!("worker copy stream device={} does not match cuda_device={cuda_device}", stream.device())));
    }

    transfer.chunks.sort_by_key(|chunk| chunk.metadata.tensor_offset);
    let mut expected_offset = 0usize;
    for chunk in &transfer.chunks {
        if chunk.metadata.tensor_offset != expected_offset {
            return Err(ThoError::new(format!("H2D transfer {} has gap or overlap at tensor offset {}; got chunk offset {}", transfer.metadata.transfer_id, expected_offset, chunk.metadata.tensor_offset)));
        }
        expected_offset = expected_offset.checked_add(chunk.metadata.byte_len).ok_or_else(|| ThoError::new("H2D expected tensor offset overflow"))?;
    }
    if expected_offset != transfer.metadata.tensor_byte_len {
        return Err(ThoError::new(format!("H2D transfer {} copied {expected_offset} bytes but tensor has {} bytes", transfer.metadata.transfer_id, transfer.metadata.tensor_byte_len)));
    }

    let _thread_context = ThoCudaContext::new(stream.device())?;
    let _stream_guard = ThoStreamGuard::new(&stream)?;
    let destination = Tensor::f_empty(transfer.metadata.shape.as_slice(), (transfer.metadata.tch_kind(), Cuda(cuda_device)))?;
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
    Ok((transfer.metadata.transfer_id, destination, copied_digest.hex))
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
                    Err(e) => OutgoingRequestPayload::Msg(format!("copy-error: failed to encode worker copy result for request {request_id}: {e}")),
                }
            }
            Err(e) => OutgoingRequestPayload::Msg(format!("copy-error: {e}")),
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
                    Err(e) => OutgoingRequestPayload::Msg(format!("copy-error: failed to encode worker D2H result for request {request_id}: {e}")),
                }
            }
            Err(e) => OutgoingRequestPayload::Msg(format!("copy-error: {e}")),
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
        stream:Guard<ThoCudaStream>,
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    GetExecStream {
        respond_to: oneshot::Sender<Result<Guard<ThoCudaStream>, ThoError>>
    },
    Reset {
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    Shutdown {
        respond_to: oneshot::Sender<Result<(), ThoError>>
    }

}

struct OutgoingRequest {
    id: u64,
    payload: OutgoingRequestPayload,
}

pub enum OutgoingRequestPayload {
    Msg(String),
    SharedMemfdBuffer{
        meta: IpcTensorTransferMetadata,
        fd:OwnedFd,
    },
}

#[derive(Debug)]
pub struct CudaIPCActor {
    tx: mpsc::Sender<OutgoingRequest>,
    receiver: mpsc::Receiver<CudaIPCActorMessage>,
    child:Child,

    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    next_id: Arc<AtomicU64>,
    copy_ctx:Arc<RwLock<CopyCtx>>,
    buf_size: usize,
    tho_device_idx: usize,
}

impl CudaIPCActor {
    pub async fn new(tho_device_idx:usize, cuda_worker_args: Vec<&str>, receiver: mpsc::Receiver<CudaIPCActorMessage>) -> Result<Self, ThoError> {

        let (mut child, stream) = Self::spawn_worker(cuda_worker_args).await?;

        let (reader, writer) = stream.into_split();

        let (tx, rx) = mpsc::channel::<OutgoingRequest>(1024);

        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        tokio::spawn(writer_loop(writer, rx, Arc::clone(&pending)));
        tokio::spawn(reader_loop(reader, Arc::clone(&pending)));


        let copy_ctx = CopyCtx::make_instance_for_cuda_device(tho_device_idx - 1, 0.2, 16);
        let buf_size = copy_ctx.buf_size;
        Ok(Self {
            tx,
            receiver,
            child,
            pending,
            next_id: Arc::new(AtomicU64::new(1)),
            copy_ctx: Arc::new(RwLock::new(copy_ctx)),
            buf_size,
            tho_device_idx,
        })
    }

    async fn spawn_worker(args: Vec<&str>) -> Result<(Child, UnixStream), ThoError> {
        let exe = std::env::current_exe()?;

        debug!("spawn_worker, exe={exe:?}");

        let mut command = &mut Command::new(exe);

        for arg in args {
            command = command.arg(arg);
        }

        let dir = tempfile::tempdir()?;
        let sock_path = dir.path().join("worker.sock");

        debug!("spawn_worker, dir={dir:?} sock_path={sock_path:?}");

        let mut child = command

            .env(IPC_SOCK_ENV, &sock_path)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        let listener = UnixListener::bind(&sock_path)?;

        debug!("spawn_worker, listener={listener:?}");

        let accepted = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            listener.accept(),
        )
            .await;

        debug!("spawn_worker, accepted={accepted:?}");

        let (mut stream, _addr) = match accepted {
            Ok(Ok(pair)) => pair,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => match child.try_wait()? {
                Some(status) => {
                    return Err(ThoError::new(format!(
                        "worker exited before connecting: status={status}"
                    )));
                }
                None => {
                    let _ = child.kill().await;
                    return Err(ThoError::new("worker did not connect within timeout"));
                }
            },
        };

        debug!("spawn_worker, _addr={_addr:?}");

        let ready = read_text_frame(&mut stream).await?;
        debug!("main received: {ready:?}");
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
        let pending = Arc::clone(&self.pending);
        let self_tx = self.tx.clone();
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

    fn handle_message(&mut self, msg: CudaIPCActorMessage) -> Result<(), ThoError>{
        match msg {
            CudaIPCActorMessage::Copy { src, dst_device, respond_to } => {
                if let Some(tensor_key) = src.ref_key() {
                    return self.handle_ref_d2h_copy(tensor_key, dst_device, respond_to);
                }

                let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let pending = Arc::clone(&self.pending);
                let self_tx = self.tx.clone();
                let copy_ctx = Arc::clone(&self.copy_ctx);
                let buf_size = self.buf_size;
                let dst_worker_id = dst_device.ordinal() as u64;

                let source_digest = tensor_digest(src.t_ref())?;
                debug!("CudaIPCActor::Copy: original tensor digest={source_digest:?}");
                let source_digest_hex = source_digest.hex.clone();
                let tensor_mem_size = tensor_mem_size(src.t_ref().kind(), &src.t_ref().size());
                debug!("copy_h2d_from_ptr: src_t.size={:?} tensor_mem_size={tensor_mem_size} buf_size={buf_size}",src.t_ref().size());
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
                    let _src_keepalive = src;

                    {

                        let mut pending = pending.lock().await;

                        let old = pending.insert(transfer_id, reply_tx);
                        debug_assert!(old.is_none());
                    }

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
                            debug!("CudaIPCActor::Copy: copied tensor chunk to shared memfd: chunk={chunk:?} metadata={metadata:?} local_mmap_ptr={:?}", buf_stream.0.as_ptr());

                            let fd = buf_stream.0.try_clone_fd()?;
                            let send_result = self_tx.send(OutgoingRequest { id: transfer_id, payload: OutgoingRequestPayload::SharedMemfdBuffer { meta: metadata, fd } }).await;

                            if send_result.is_err() {
                                let mut pending = pending.lock().await;
                                pending.remove(&transfer_id);
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
                    debug!("CudaIPCActor::Copy: worker copied tensor digest matched: transfer_id={} digest={}", done.transfer_id, done.tensor_digest_hex);

                    respond_to.send(Ok(ThoTensor::new_ref((), transfer_id/*it's used as key for all_cuda_tensorsat crates/thoosa_common/src/torch_runner/cuda_process.rs:370 */)))
                        .map_err(|e| ThoError::new(format!("response channel for request {id} got {e:?}")))?;

                    Ok::<_, ThoError>(())
                });


            }
            CudaIPCActorMessage::SyncStream { .. } => {}
            CudaIPCActorMessage::GetExecStream { .. } => {}
            CudaIPCActorMessage::Reset { .. } => {}
            CudaIPCActorMessage::Shutdown { respond_to } => {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let pending = Arc::clone(&self.pending);
                let self_tx = self.tx.clone();

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
        }

        Ok(())
    }

    async fn run(mut actor: CudaIPCActor) -> Result<(), ThoError> {
        while let Some(msg) = actor.receiver.recv().await {
            actor.handle_message(msg)?;
        }

        Ok(())
    }
}

async fn writer_loop<W>(
    mut writer: W,
    mut rx: mpsc::Receiver<OutgoingRequest>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
) where
    W: AsyncWrite + Unpin + AsRef<UnixStream>,
{
    while let Some(req) = rx.recv().await {
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
) where
    R: AsyncRead + Unpin,
{
    loop {
        let frame = match read_text_frame(&mut reader).await {
            Ok(frame) => frame,
            Err(e) => {
                debug!("worker IPC reader stopped: {e}");

                fail_all_pending(&pending, format!("worker IPC reader stopped: {e}")).await;
                break;
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
        return Err(ThoError::new(format!(
            "bad wire frame without request id separator: {frame:?}"
        )));
    };

    let id = id
        .parse::<u64>()
        .map_err(|e| ThoError::new(format!("bad request id {id:?}: {e}")))?;

    Ok((id, payload.to_string()))
}

fn parse_worker_copy_done_payload(payload: &str) -> Result<WorkerCopyDonePayload, ThoError> {
    if let Some(error) = payload.strip_prefix("copy-error:") {
        return Err(ThoError::new(format!("worker copy failed: {error}")));
    }
    let done_json = payload.strip_prefix(COPY_DONE_PAYLOAD_PREFIX)
        .ok_or_else(|| ThoError::new(format!("worker copy response did not start with {COPY_DONE_PAYLOAD_PREFIX:?}: {payload:?}")))?;
    serde_json::from_str(done_json)
        .map_err(|e| ThoError::new(format!("failed to parse worker copy response: {e}")))
}

fn parse_worker_tensor_info_payload(payload: &str) -> Result<WorkerTensorInfoPayload, ThoError> {
    if let Some(error) = payload.strip_prefix("copy-error:") {
        return Err(ThoError::new(format!("worker describe failed: {error}")));
    }
    let info_json = payload.strip_prefix(TENSOR_INFO_PAYLOAD_PREFIX)
        .ok_or_else(|| ThoError::new(format!("worker tensor info response did not start with {TENSOR_INFO_PAYLOAD_PREFIX:?}: {payload:?}")))?;
    serde_json::from_str(info_json)
        .map_err(|e| ThoError::new(format!("failed to parse worker tensor info response: {e}")))
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
        debug!("CudaActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let actor = CudaIPCActor::new(tho_device_id, cuda_worker_args, receiver).await?;
        tokio::spawn(CudaIPCActor::run(actor));

        Ok(Self {
            sender,
        })
    }

    pub async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) ->Result<ThoTensor, ThoError> {
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

    pub async fn sync_stream(&self, stream:Guard<ThoCudaStream>) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SyncStream {
                stream,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    pub async fn reset(&self) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Reset {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    pub async fn shutdown(&self) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Shutdown {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?
    }

    //TODO these are 'exec' streams, not for copy
    pub async fn acquire_exec_stream(&self) -> Result<Guard<ThoCudaStream>, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::GetExecStream {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
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
    use std::path::PathBuf;
    use std::process::Stdio;
    use std::sync::Arc;
    use log::debug;
    use tch::{Device, Tensor};
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{UnixListener, UnixStream};
    use tokio::process::{Child, Command};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_shared::tracked_await::track_await;
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
    use crate::ipc::cuda_process::{read_text_frame, write_text_frame, CudaIPCActor, IPC_SOCK_ENV};
    use crate::ipc::cuda_wokrer::CudaWorker;
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    use crate::torch_runner::tests::{ClcTask, ClcTaskOutput, LoadAndClcTask, LoadAndClcTaskId, LoadedTensor};
    use crate::utils::tensor_digest;



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
        let args:Vec<&str> = vec![&test_name, "--exact", "--nocapture"];

        CudaIPCActor::spawn_worker(args).await
    }


    async fn run_main(test_name: &str) -> Result<(), ThoError> {
        debug!("run_main: test_name={test_name:?}");
        let module = pure_module_path!();

        let test_name = if module.is_empty() {
            test_name.to_string()
        } else {
            format!("{module}::{test_name}")
        };
        let args:Vec<&str> = vec![test_name.clone().leak(), "--exact", "--nocapture"];

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices_2(2, args).await?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        // let (mut child, mut stream) = spawn_test_worker(test_name).await?;
        debug!("run_main: test_name={test_name:?}, spawn_worker ok");

        let tho_device_idx = 1;

        let main_process_cpu_tensor = Tensor::rand(&[1000, 10000], FLOAT_CPU);
        let expected_digest = tensor_digest(&main_process_cpu_tensor)?;


        let cuda_actor = GlobalCtx::global().cuda_ipc_actor(tho_device_idx);
        let cuda_ref = track_await(cuda_actor
            .copy(
                main_process_cpu_tensor.into(),
                tho_device_idx.into(),//Cuda
            )).await?;

        debug!("run_main: cuda_ref={cuda_ref:?}");

        let cpu_tensor = track_await(cuda_actor
            .copy(
                cuda_ref,
                0.into(),//Cpu
            )).await?;

        debug!("run_main: cpu_tensor={cpu_tensor:?}");

        let actual_digest = tensor_digest(cpu_tensor.t_ref())?;
        if actual_digest.hex != expected_digest.hex {
            return Err(ThoError::new(format!("worker CUDA round-trip digest mismatch: expected={} actual={}", expected_digest.hex, actual_digest.hex)));
        }

        cuda_actor.shutdown().await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_copy_ipc_cuda_worker() {
        debug!("test_cuda_process, IPC_SOCK_ENV={:?}",std::env::var_os(IPC_SOCK_ENV));

        if std::env::var_os(IPC_SOCK_ENV).is_some() {
            // CudaProcess::new().unwrap();//will be called from run_cli //crates/thoosa_main/src/cli.rs:339, i.e. if app is started as "worker"
            CudaWorker::run_worker().await.unwrap()
        } else {
            run_main("test_copy_to_cuda_worker").await.unwrap();
        }
    }

}
