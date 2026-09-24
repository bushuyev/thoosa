use crate::copy_ctx::{CudaActorHandle, HandleCtx};
use crate::cuda_utils::tensor_mem_size;
use crate::ipc::ipc_cuda_utils::{IpcTensorKind, send_fd};
use crate::ipc::serialization::{PostcardToFromBytes, ToFromBytes};
use crate::ipc::worker_msg::{CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation, WorkerCopyDonePayload, WorkerInput, WorkerOutput};
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use anyhow::Context as _;
use async_trait::async_trait;
use log::{debug, error, info, trace};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::backtrace::Backtrace;
use std::collections::HashMap;
use std::fmt::Debug;
use std::os::fd::{AsFd, OwnedFd};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tch::Device::{Cpu, Cuda};
use tch::{Device, Tensor};
use thoosa_shared::ThoError;
use thoosa_shims::arena::{shared_arena_capacity, shared_arena_offset_of};
use thoosa_shims::context::ThoCudaContext;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{Mutex, mpsc, oneshot};
use tokio::task::{JoinHandle, spawn_blocking};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;
use tracing::field::debug;

pub const CUDA_VISIBLE_DEVICES: &str = "CUDA_VISIBLE_DEVICES";//it's set to '' for main process to make sure it doesn't use GPU
pub const IPC_SOCK_ENV: &str = "THOOSA_WORKER_IPC_SOCK";
pub  const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;
// Shutdown normally completes immediately, but keep a bounded kill fallback for a worker stuck after acknowledging the request.
const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);

pub(crate) fn encode_endpoint_payload(endpoint_name: &str, payload: &str) -> String {
    format!("{endpoint_name}\t{payload}")
}



// Installed stages share an initializer name but must have distinct invocation keys.
pub(crate) fn endpoint_stage_key(endpoint_name: &str, stage_id: u32) -> String {
    format!("{endpoint_name}{stage_id}")
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

#[typetag::serde]
impl WorkerOutput for WorkerTensorInfoPayload {

}

pub(crate) async fn worker_response_writer_loop<W>(
    mut writer: W,
    mut reply_rx: mpsc::Receiver<CudaWorkerMsgOut>,
) -> Result<(), ThoError>
where
    W: AsyncWrite + Unpin,
{
    while let Some(msg) = reply_rx.recv().await {
        trace!("worker_response_writer_loop: msg={msg:?}");
        write_frame(&mut writer, &PostcardToFromBytes::to_bytes(&msg)?/*TODO move to codec*/, "worker").await?;
        trace!("worker_response_writer_loop: msg out ok");
    }
    Ok(())
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
    Copy{ src:ThoTensor, dst_device:ThoDevice, activity: thoosa_shared::CudaActivityScope, respond_to: oneshot::Sender<Result<ThoTensor, ThoError>>},
    DropThoTensor {
        key:u64,
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    Restart {
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    Shutdown {
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    SendMsg {
        input: CudaWorkerMsgIn,
        respond_to: oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>
    },
}

// Typed postcard replies are routed directly by request id; no string response envelope remains.
type PendingResponses = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>>>>;

#[derive(Debug)]
pub struct CudaIPCActor {
    receiver: mpsc::Receiver<CudaIPCActorMessage>,

    spawned_state: SpawnedState,
    next_id: Arc<AtomicU64>,
    tho_device_idx: usize,
    cuda_worker_args: Arc<Vec<String>>,
    // Keep an actor-owned descriptor so a reset can map the same parent allocator into the replacement worker.
    arena_fd: OwnedFd,
}

#[derive(Debug)]
struct SpawnedState {
    pending: PendingResponses,
    tx:Sender<CudaWorkerMsgIn>,
    child:Child,
    writer_handle:JoinHandle<()>,
    reader_handle:JoinHandle<()>,
    cancel: CancellationToken,
    device_info:DeviceInfo
}

impl SpawnedState {

    async fn wait_for_graceful_exit(&mut self) -> Result<(), ThoError> {
        let wait_result = tokio::time::timeout(WORKER_SHUTDOWN_TIMEOUT, self.child.wait()).await;

        // The child has either closed its socket or exceeded the shutdown deadline, so its I/O loops no longer serve useful work.
        self.cancel.cancel();
        self.reader_handle.abort();
        self.writer_handle.abort();

        match wait_result {
            Ok(Ok(status)) if status.success() => {
                debug!("CUDA worker exited cleanly after shutdown");
                Ok(())
            }
            Ok(Ok(status)) => Err(ThoError::new(format!("CUDA worker exited with status {status} after shutdown"))),
            Ok(Err(error)) => Err(error.into()),
            Err(_) => {
                error!("CUDA worker did not exit within {WORKER_SHUTDOWN_TIMEOUT:?} after shutdown; killing it");
                self.child.kill().await.context("kill CUDA worker after shutdown timeout")?;
                Err(ThoError::new(format!("CUDA worker did not exit within {WORKER_SHUTDOWN_TIMEOUT:?} after shutdown")))
            }
        }
    }

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
    pub async fn new(tho_device_idx:usize, cuda_worker_args: Vec<String>, receiver: mpsc::Receiver<CudaIPCActorMessage>, arena_fd: OwnedFd) -> Result<Self, ThoError> {
        let cuda_worker_args = Arc::new(cuda_worker_args);

        let spawned_state = Self::spawn_worker_and_loops(tho_device_idx, &cuda_worker_args, arena_fd.as_fd()).await?;


        debug!("CudaIPCActor::new: worker device_info={:?}", spawned_state.device_info);

        Ok(Self {
            receiver,
            spawned_state,
            next_id: Arc::new(AtomicU64::new(1)),
            tho_device_idx,
            cuda_worker_args,
            arena_fd,
        })
    }

    async fn spawn_worker_and_loops(tho_device_idx: usize, cuda_worker_args: &Vec<String>, arena_fd: std::os::fd::BorrowedFd<'_>) -> Result<SpawnedState, ThoError> {
        debug!("CudaIPCActor::spawn_worker_and_loops, cuda_worker_args={cuda_worker_args:?}");

        let (mut child, stream, device_info) = Self::spawn_worker_for_cuda_device(tho_device_idx - 1, cuda_worker_args, arena_fd).await?;

        let (reader, writer) = stream.into_split();

        let (tx, rx): (Sender<CudaWorkerMsgIn>, Receiver<CudaWorkerMsgIn>) = mpsc::channel::<CudaWorkerMsgIn>(1024);

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

    async fn spawn_worker_for_cuda_device(cuda_device: usize, args: &Vec<String>, arena_fd: std::os::fd::BorrowedFd<'_>) -> Result<(Child, UnixStream, DeviceInfo), ThoError> {
        let exe = std::env::current_exe()?;

        debug!("CudaIPCActor::spawn_worker, exe={exe:?}");

        let dir = tempfile::tempdir()?;
        let sock_path = dir.path().join(format!("worker_{cuda_device}.sock"));

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

        let device_info:DeviceInfo = PostcardToFromBytes::from_bytes(&read_frame(&mut stream, "DeviceInfo/main", MAX_FRAME_LEN).await?)?;
        debug!("CudaIPCActor::spawn_worker: main received: {device_info:?}");

        // The arena descriptor is the only descriptor sent per worker; all tensor copies subsequently use offsets in this mapping.
        send_fd(&stream, arena_fd).await?;

        // The socket file can be removed after accept. The connected UnixStream
        // remains valid.
        drop(dir);

        Ok((child, stream, device_info))
    }


    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaIPActor::handle_message"))]
    async fn handle_message(&mut self, msg: CudaIPCActorMessage) -> Result<(), ThoError>{
        match msg {
            CudaIPCActorMessage::Copy { src, dst_device, activity, respond_to } => {

                return if let Some(tensor_key) = src.ref_key() {
                    assert_eq!(*dst_device, Cpu, "If source is tensor ref and thus live on device it can be copied only to CPU");
                     self.handle_ref_d2h_copy(tensor_key, dst_device, activity, respond_to).await
                } else {
                    debug!("handle_message: src={src:?} dst_device={dst_device:?}");
                    assert_ne!(*dst_device, Cpu, "If source is concrete tensor it should be on CPU and can be copied only to GPU");

                    self.handle_h2d_copy(src, dst_device, activity, respond_to)
                }

            }

            CudaIPCActorMessage::DropThoTensor { key, respond_to } => {
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let pending = Arc::clone(&self.spawned_state.pending);
                let self_tx = self.spawned_state.tx.clone();

                debug!("handle_message: CudaIPCActorMessage::DropThoTensor id={id:?} key={key:?}");

                tokio::spawn(async move {
                    {
                        let mut pending = pending.lock().await;
                        let old = pending.insert(id, reply_tx);
                        debug_assert!(old.is_none());
                    }

                    if self_tx.send(
                        CudaWorkerMsgIn {
                            request_id: id,
                            operation: Operation::DropThoTensor {tensor_key: key},
                        },
                    ).await.is_err() {
                        let mut pending = pending.lock().await;
                        pending.remove(&id);
                        let _ = respond_to.send(Err(ThoError::new("worker IPC writer task is closed")));
                        return;
                    }


                    let _ = respond_to.send(Self::await_unit_reply(id, reply_rx).await);

                });
            }
            CudaIPCActorMessage::Restart { respond_to } => {
                info!("CudaIPCActor::handle_message, CudaIPCActorMessage::Reset");

                let replacement = match Self::spawn_worker_and_loops(self.tho_device_idx, &self.cuda_worker_args, self.arena_fd.as_fd()).await {
                    Ok(replacement) => replacement,
                    Err(error) => {
                        let _ = respond_to.send(Err(error));
                        return Ok(());
                    }
                };
                let old_spawned_state = std::mem::replace(&mut self.spawned_state, replacement);
                old_spawned_state.shutdown("CUDA worker reset").await;
                debug!("CudaIPCActor::handle_message, CudaIPCActorMessage::Reset done");
                let _ = respond_to.send(Ok(()));
            }
            CudaIPCActorMessage::Shutdown { respond_to } => {
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let pending = Arc::clone(&self.spawned_state.pending);
                let self_tx = self.spawned_state.tx.clone();
                {
                    let mut pending = pending.lock().await;
                    let old = pending.insert(next_id, reply_tx);
                    debug_assert!(old.is_none());
                }

                debug!("CudaIPCActorMessage::Shutdown: next_id={next_id}");

                if self_tx.send(
                    CudaWorkerMsgIn {
                            request_id: next_id,
                            operation: Operation::Shutdown{ dummy: "dummy".to_string() },
                    },
                ).await.is_err() {

                    let mut pending = pending.lock().await;
                    pending.remove(&next_id);
                    let _ = respond_to.send(Err(ThoError::new("worker IPC writer task is closed")));
                    return Err(ThoError::new("can't shutdown"))
                }

                let reply_result = Self::await_unit_reply(next_id, reply_rx).await;
                // The acknowledgement only means the worker drained its requests. Waiting for process exit also preserves its final logs and timing output.
                let exit_result = self.spawned_state.wait_for_graceful_exit().await;
                let result = reply_result.and(exit_result);
                let _ = respond_to.send(result);

                debug!("CudaIPCActorMessage::Shutdown: final");
            }
            CudaIPCActorMessage::SendMsg { mut input, respond_to } => {
                let self_tx = self.spawned_state.tx.clone();
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.spawned_state.pending);

                input.request_id = next_id;

                tokio::spawn(async move {

                    let response = send_request_and_wait_msg(
                        self_tx,
                        pending,
                        input,
                    ).await;
                    let _ = respond_to.send(response);

                    Ok::<_, ThoError>(())
                });
            }


        }

        Ok(())
    }

    async fn await_unit_reply(id: u64, reply_rx: tokio::sync::oneshot::Receiver<Result<CudaWorkerMsgOut, ThoError>>) -> Result<(), ThoError> {
        let reply = reply_rx.await.with_context(|| format!("response channel for request {id} was dropped"))??;
        match reply.status {
            OpResult::Ok(_) => {
                debug!("await_unit_reply: id={id:?} ok");
                Ok(())
            }
            OpResult::Failed(error) => {
                debug!("await_unit_reply: id={id:?} error={error:?}");
                Err(error.into())
            }
        }
    }

    fn copy_result(response: CudaWorkerMsgOut) -> Result<WorkerCopyDonePayload, ThoError> {
        match response.status {
            OpResult::Ok(Some(output)) => {
                // The copy protocol has exactly one typed success payload; reject unrelated worker outputs explicitly.
                let output: Box<dyn Any> = output;
                let Ok(output) = output.downcast::<WorkerCopyDonePayload>() else {
                    return Err(ThoError::new(format!("worker copy request {} returned an unexpected output type", response.request_id)));
                };
                Ok(*output)
            }
            OpResult::Ok(None) => Err(ThoError::new(format!("worker copy request {} returned no output", response.request_id))),
            OpResult::Failed(error) => Err(error.into()),
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaIPActor::handle_ref_d2h_copy"))]
    async fn handle_ref_d2h_copy(
        &mut self,
        tensor_key: u64,
        dst_device: ThoDevice,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<ThoTensor, ThoError>>,
    ) -> Result<(), ThoError> {
        if dst_device.tho_device_idx() != 0 {
            return Err(ThoError::new(format!("worker tensor ref copy currently supports only CPU destination, got dst_device={dst_device:?}")));
        }

        let describe_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let pending = Arc::clone(&self.spawned_state.pending);
        let self_tx = self.spawned_state.tx.clone();
        debug!("CudaIPCActor::handle_ref_d2h_copy: describe_id={describe_id}");

        let response = send_request_and_wait_msg(
            self_tx.clone(),
            Arc::clone(&pending),
            CudaWorkerMsgIn {
                request_id: describe_id,
                operation: Operation::DescribeRef { tensor_key },
            },
        ).await;
        let info = match response.and_then(worker_tensor_info) {
            Ok(info) => info,
            Err(error) => {
                let _ = respond_to.send(Err(error));
                return Ok(());
            }
        };
        if info.tensor_key != tensor_key {
            let _ = respond_to.send(Err(ThoError::new(format!("worker described tensor key={} but requested key={tensor_key}", info.tensor_key))));
            return Ok(());
        }
        let expected_byte_len = tensor_mem_size(info.kind.into(), &info.shape);
        if expected_byte_len != info.tensor_byte_len {
            let _ = respond_to.send(Err(ThoError::new(format!("worker described tensor key={tensor_key} with byte length {} but kind/shape require {expected_byte_len}", info.tensor_byte_len))));
            return Ok(());
        }

        // Allocate the final tensor before dispatch so the worker only writes into storage owned and eventually freed by the parent allocator.
        let destination = match Tensor::f_empty(info.shape.as_slice(), (info.kind.into(), Cpu)) {
            Ok(destination) => destination,
            Err(error) => {
                let _ = respond_to.send(Err(error.into()));
                return Ok(());
            }
        };
        let arena_offset = match parent_shared_arena_offset(&destination, info.tensor_byte_len, "D2H destination") {
            Ok(arena_offset) => arena_offset,
            Err(error) => {
                let _ = respond_to.send(Err(error));
                return Ok(());
            }
        };
        let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);

        tokio::spawn(async move {
            // The caller may have been cancelled. Keep reset blocked until the
            // worker acknowledges that it finished accessing shared storage.
            let _activity = activity;
            let result = async {
                let response = send_request_and_wait_msg(
                    self_tx,
                    pending,
                    CudaWorkerMsgIn {
                        request_id: transfer_id,
                        operation: Operation::D2hCopy {
                            tensor_key,
                            kind: info.kind,
                            shape: info.shape.clone(),
                            tensor_byte_len: info.tensor_byte_len,
                            arena_offset,
                        },
                    },
                ).await?;
                let done = Self::copy_result(response)?;
                if done.transfer_id != transfer_id || done.tensor_ref != tensor_key {
                    return Err(ThoError::new(format!("worker D2H response was transfer_id={} tensor_ref={}, expected transfer_id={transfer_id} tensor_ref={tensor_key}", done.transfer_id, done.tensor_ref)));
                }
                ThoTensor::new((), destination)
            }.await;
            debug!("CudaIPCActor::Copy D2H: done, result={result:?}");
            let _ = respond_to.send(result);
        });

        Ok(())
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaIPActor::handle_h2d_copy"))]
    fn handle_h2d_copy(&mut self, src: ThoTensor, dst_device: ThoDevice, activity: thoosa_shared::CudaActivityScope, respond_to: oneshot::Sender<Result<ThoTensor, ThoError>>) -> Result<(), ThoError> {
        let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let pending = Arc::clone(&self.spawned_state.pending);
        let self_tx = self.spawned_state.tx.clone();
        let dst_worker_id = dst_device.tho_device_idx() as u64;
        let src_kind = src.kind();
        let src_shape = src.size();
        let tensor_byte_len = tensor_mem_size(src_kind, &src_shape);
        let src_tensor = src.t();

        debug!("CudaIPCActor::handle_h2d_copy: src_shape={src_shape:?} tensor_byte_len={tensor_byte_len}");

        tokio::spawn(async move {
            // Transfer ownership follows the request, not the awaiting caller.
            let _activity = activity;
            let result = async {
                // Older CPU tensors may predate allocator installation; copy only those tensors once into arena-owned storage before sharing their offset.
                let (shared_source, arena_offset) = spawn_blocking(move || ensure_parent_shared_arena_tensor(src_tensor, tensor_byte_len))
                    .await
                    .context("join parent H2D arena staging task")??;
                let response = send_request_and_wait_msg(
                    self_tx,
                    pending,
                    CudaWorkerMsgIn {
                        request_id: transfer_id,
                        operation: Operation::H2dCopy {
                            kind: src_kind.into(),
                            shape: src_shape.clone(),
                            tensor_byte_len,
                            arena_offset,
                        },
                    },
                ).await?;
                debug!("CudaIPCActor::handle_h2d_copy: response={response:?}");
                let done = Self::copy_result(response)?;
                if done.transfer_id != transfer_id {
                    return Err(ThoError::new(format!("worker H2D response transfer_id={} but expected {transfer_id}", done.transfer_id)));
                }
                // The worker has completed its read before acknowledging, so the parent may now release the arena allocation.
                drop(shared_source);
                Ok(ThoTensor::new_ref((), done.tensor_ref, src_shape, src_kind, None, dst_device.tho_device_idx()))
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

fn worker_tensor_info(response: CudaWorkerMsgOut) -> Result<WorkerTensorInfoPayload, ThoError> {
    let request_id = response.request_id;
    match response.status {
        OpResult::Ok(Some(output)) => {
            let output: Box<dyn Any> = output;
            let Ok(output) = output.downcast::<WorkerTensorInfoPayload>() else {
                return Err(ThoError::new(format!("worker describe request {request_id} returned an unexpected output type")));
            };
            Ok(*output)
        }
        OpResult::Ok(None) => Err(ThoError::new(format!("worker describe request {request_id} returned no tensor metadata"))),
        OpResult::Failed(error) => Err(error.into()),
    }
}

fn parent_shared_arena_offset(tensor: &Tensor, tensor_byte_len: usize, label: &str) -> Result<usize, ThoError> {
    if tensor_byte_len == 0 {
        // LibTorch represents zero-byte allocations with a null data pointer, so no actual arena address is required.
        return Ok(0);
    }
    let raw_offset = unsafe { shared_arena_offset_of(tensor.data_ptr()) };
    if raw_offset < 0 {
        return Err(ThoError::new(format!("{label} is not allocated in the parent shared arena")));
    }
    let arena_offset = usize::try_from(raw_offset).with_context(|| format!("{label} arena offset {raw_offset} does not fit usize"))?;
    let arena_end = arena_offset.checked_add(tensor_byte_len).ok_or_else(|| ThoError::new(format!("{label} arena range overflow")))?;
    let arena_capacity = unsafe { shared_arena_capacity() };
    if arena_end > arena_capacity {
        return Err(ThoError::new(format!("{label} arena range [{arena_offset}, {arena_end}) exceeds capacity {arena_capacity}")));
    }
    Ok(arena_offset)
}

fn ensure_parent_shared_arena_tensor(source: Tensor, tensor_byte_len: usize) -> Result<(Tensor, usize), ThoError> {
    if source.device() != Cpu {
        return Err(ThoError::new(format!("IPC H2D source must be on CPU, got {:?}", source.device())));
    }
    if !source.is_contiguous() {
        return Err(ThoError::new("IPC H2D source must be contiguous"));
    }
    if let Ok(arena_offset) = parent_shared_arena_offset(&source, tensor_byte_len, "H2D source") {
        return Ok((source, arena_offset));
    }

    // Tensors created before IPC initialization still use LibTorch's old CPU allocator; migrate them without reintroducing per-transfer memfds.
    let mut shared_source = Tensor::f_empty(source.size().as_slice(), (source.kind(), Cpu))?;
    shared_source.f_copy_(&source)?;
    let arena_offset = parent_shared_arena_offset(&shared_source, tensor_byte_len, "staged H2D source")?;
    Ok((shared_source, arena_offset))
}

async fn writer_loop<W>(
    mut writer: W,
    mut rx: mpsc::Receiver<CudaWorkerMsgIn>,
    pending: PendingResponses,
    cancel: CancellationToken,
) where
    W: AsyncWrite + Unpin,
{
    loop {

        let msg = tokio::select! {
            _ = cancel.cancelled() => {
                break;
            }

            maybe_msg = rx.recv() => {
                match maybe_msg {
                    Some(msg) => msg,
                    None => break,
                }
            }
        };

        let request_id = msg.request_id;
        debug!("writer_loop: msg={msg:?}");
        let frame = match PostcardToFromBytes::to_bytes(&msg) {
            Ok(frame) => frame,
            Err(e) => {
                error!("worker IPC writer failed to encode request {request_id}: {e}");
                fail_one_pending(&pending, request_id, format!("failed to encode request {request_id}: {e}")).await;
                continue;
            }
        };

        if let Err(e) = write_frame(&mut writer, &frame, "CudaWorkerMsg").await {
            error!("worker IPC writer failed: {e}");
            fail_one_pending(&pending, request_id, format!("failed to write request {request_id}: {e}")).await;
            fail_all_pending(&pending, format!("worker IPC writer stopped: {e}")).await;
            break;
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

        match PostcardToFromBytes::from_bytes::<CudaWorkerMsgOut>(&frame) {
            Ok(msg) => {
                debug!("reader_loop: msg={msg:?}");
                {
                    let mut pending = pending.lock().await;
                    let msg_waiter = match pending.remove(&msg.request_id) {
                        None => {
                            error!("no waiter for {}", msg.request_id);
                            continue
                        }
                        Some(v) => {
                            trace!("reader_loop: pending response={v:?} for msg.request_id={}", msg.request_id);
                            v
                        }
                    };
                    if msg_waiter.send(Ok(msg)).is_ok() {
                        debug!("reader_loop: send ok");
                    } else {
                        debug!("reader_loop: response receiver was dropped");
                    }
                }
                continue
            }
            Err(e) => {
                error!("bad worker response frame: {e}");
            }
        }

    }

    debug!("worker IPC reader loop finished");
}

async fn send_request_and_wait_msg(
    tx: mpsc::Sender<CudaWorkerMsgIn>,
    pending: PendingResponses,
    request: CudaWorkerMsgIn,
) -> Result<CudaWorkerMsgOut, ThoError> {
    let id = request.request_id;
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
        .with_context(|| format!("response channel for request {id} was dropped"))?
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


    if let Some(respond_to) = reply_tx {
        let _ = respond_to.send(Err(ThoError::new(msg)));
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

    for (_id, respond_to) in pending_items {
        let _ = respond_to.send(Err(ThoError::new(msg.clone())));
    }
}


#[derive(Debug, Clone)]
pub struct CudaIPCActorHandle {
    sender: mpsc::Sender<CudaIPCActorMessage>,
}

///executes cuda actions in a separate process started from CudaWorker::run_worker
impl CudaIPCActorHandle {
    pub async fn new(tho_device_id:usize, cuda_worker_args: Vec<String>, arena_fd: OwnedFd) -> Result<Self, ThoError> {
        debug!("CudaIPCActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let actor = CudaIPCActor::new(tho_device_id, cuda_worker_args, receiver, arena_fd).await?;
        tokio::spawn(CudaIPCActor::run(actor));

        Ok(Self {
            sender,
        })
    }

    pub async fn invoke_stage(&self, endpoint_name: String, stage_id: u32, input: Box<dyn WorkerInput>, consume:bool) -> anyhow::Result<CudaWorkerMsgOut> {
        debug!("CudaIPCActorHandle::invoke_stage: endpoint_name={endpoint_name} stage_id={stage_id} input={input:?}");
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                input: CudaWorkerMsgIn {
                    request_id: 0,
                    operation: Operation::InvokeStage {
                        endpoint_name,
                        stage_id,
                        input,//cloned refs
                        consume,
                    },
                },
                respond_to,
            }
        )
        .instrument(tracing::debug_span!("CudaIPCActorHandle::invoke_stage",agg_perf="CudaIPCActorHandle::send_msg",)).await
        .context("CUDA actor channel closed")?;

        Ok(recv.await.context("CUDA actor response channel closed")??)
    }

    pub async fn install_stage(&self, endpoint_name: String, stage_id: u32, init_data: Box<dyn WorkerInput>) -> anyhow::Result<CudaWorkerMsgOut> {
        debug!("CudaIPCActorHandle::invoke_stage: install_stage_2={endpoint_name} stage_id={stage_id} init_data={init_data:?}");

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
        )
        .instrument(tracing::debug_span!("CudaIPCActorHandle::send_msg",agg_perf="CudaIPCActorHandle::install_stage_2",)).await
        .context("CUDA actor channel closed")?;

        Ok(recv.await.context("CUDA actor response channel closed")??)
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
        let activity = thoosa_shared::CudaActivityScope::for_operation()?;
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

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Copy {
                src,
                dst_device,
                activity,
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        let r = recv.await.context("CUDA actor response channel closed")?;
        debug!("CudaIPCActorHandle::copy:  get result: r={r:?}");
        r
    }

    async fn restart(&self) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Restart {
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        recv.await.context("CUDA actor response channel closed")?

    }

    async fn shutdown(&self) -> Result<(), ThoError> {
        debug!("CudaIPCActorHandle::shutdown");

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Shutdown {
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        debug!("CudaIPCActorHandle::shutdown: recv={recv:?}");

        recv.await.context("CUDA actor response channel closed")?

    }

    async fn drop_tho_tensor(&self, key:u64) -> Result<(), ThoError> {
        debug!("CudaIPCActorHandle::drop_tho_tensor: key={key}");
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::DropThoTensor {
                key,
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        recv.await.context("CUDA actor response channel closed")?
    }


    async fn init_handle_ctx(&self) -> Result<Box<dyn crate::copy_ctx::HandleCtx>, ThoError> {
        Ok(Box::new(IPCHanldeCTx{}))
    }

    async fn purge_stage_state(&self, stage_id: u32, endpoint_name: String) -> Result<CudaWorkerMsgOut, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                input: CudaWorkerMsgIn {
                    request_id: 0,
                    operation: Operation::PurgeStage {
                        endpoint_name,
                        stage_id,
                    },
                },
                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::purge_endpoint_bytes",
            agg_perf="CudaIPCActorHandle::purge_endpoint_bytes",
        )).await.context("CUDA actor channel closed")?;

        recv.await.context("CUDA actor response channel closed")?
    }

    async fn drop_endpoint(&self, stage_id: u32, endpoint_name: String) -> Result<CudaWorkerMsgOut, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                input: CudaWorkerMsgIn {
                    request_id: 0,
                    operation: Operation::DropStage {
                        endpoint_name,
                        stage_id,
                    },
                },
                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::drop_endpoint",
            agg_perf="CudaIPCActorHandle::drop_endpoint",
        )).await.context("CUDA actor channel closed")?;

        recv.await.context("CUDA actor response channel closed")?
    }

    async fn debug_worker(&self) -> Result<CudaWorkerMsgOut, ThoError> {
        debug!("CudaIPCActorHandle::debug_worker");

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                input: CudaWorkerMsgIn {
                    request_id: 0,
                    operation: Operation::Debug,
                },
                respond_to,
            }
        )
            .instrument(tracing::debug_span!("CudaIPCActorHandle::send_msg",agg_perf="CudaIPCActorHandle::install_stage_2",)).await
            .context("CUDA actor channel closed")?;

        recv.await.context("CUDA actor response channel closed")?
    }

}


#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="write_frame"))]
pub async fn write_frame<W>(
    writer: &mut W,
    payload: &[u8],
    descr: &str,
) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    trace!("write_frame: descr={descr} payload.len={}", payload.len());

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

    trace!("write_frame: descr={descr} done");

    Ok(())
}


#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="read_frame"))]
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
    use anyhow::Context as _;
    use crate::copy_ctx::CudaActorHandle;
    use crate::cuda_utils::{cuda_mem, fail_with_unrecoverable_error, speed_gb_per_s};
    use crate::ipc::cuda_process::{CudaIPCActor, CudaIPCActorHandle, DeviceInfo, IPC_SOCK_ENV, encode_endpoint_payload};
    use crate::ipc::cuda_wokrer::CudaWorker;
    use crate::ipc::tests::run_ipc_test;
    use crate::ipc::wraps::SyncStageIpcWrapper;
    use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster};
    use crate::runner::common::cachable::DevicesWrap;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
    use crate::runner::run_ctx::RunCtxImpl;
    // use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, OOMRunnerConfig};
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    // use crate::torch_runner::tests::load_and_clc::loader::LoadedTensor;
    use crate::utils::{lin_for_size, tensor_digest};
    use crate::ws::{Workspace, mock_ws_simp};
    use futures_util::FutureExt;
    use futures_util::future::BoxFuture;
    use log::{debug, info};
    use mockall::PredicateBoxExt;
    use parking_lot::RwLock;
    use rand::random_range;
    use std::any::Any;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::process::Stdio;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Instant;
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{Device, Kind, Tensor};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::perf_logging::{PERF, format_perf_map, init_tracing};
    use thoosa_shared::tracked_await::track_await;
    use thoosa_shared::traits::RunCtx;
    use thoosa_shared::{ThoError, mono_key};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{UnixListener, UnixStream};
    use tokio::process::{Child, Command};
    use tokio::sync::Mutex;
    use tokio::task::JoinSet;

    fn round_trip_worker_error(error: ThoError) -> ThoError {
        use crate::ipc::serialization::{PostcardToFromBytes, ToFromBytes};
        use crate::ipc::worker_msg::CudaWorkerMsgOut;

        // Exercise the worker's reply encoding and the parent's actual copy error consumer.
        let reply = CudaWorkerMsgOut { request_id: 42, status: error.into() };
        let bytes = PostcardToFromBytes::to_bytes(&reply).unwrap();
        let decoded = PostcardToFromBytes::from_bytes(&bytes).unwrap();
        CudaIPCActor::copy_result(decoded).expect_err("worker error must fail the copy")
    }

    #[test]
    fn test_worker_error_round_trip_preserves_cuda_recovery() {
        use thoosa_shared::{CudaAction, SyncStageRunnerError};

        for action in [CudaAction::HandleOOM, CudaAction::NeedsRestart, CudaAction::Other] {
            let error = ThoError::Cuda { action, msg: "device failure detail".into() }.context("forward model");
            let job = ThoError::from(SyncStageRunnerError::JobError { job_id: 27, inner: Box::new(error) });
            let received = round_trip_worker_error(job.context("worker request 42"));
            assert!(matches!(&received, ThoError::Cuda { action: received_action, .. } if *received_action == action));
            assert_eq!(received.is_oom(), action == CudaAction::HandleOOM);
            assert_eq!(received.is_retryable_oom(), action == CudaAction::HandleOOM);
            assert_eq!(received.needs_cuda_ctx_reset(), action == CudaAction::NeedsRestart);
            let message = received.to_string();
            assert!(message.contains("worker request 42") && message.contains("job 27") && message.contains("forward model") && message.contains("device failure detail"));
        }
    }

    #[test]
    fn test_worker_error_round_trip_preserves_non_cuda_semantics() {
        use thoosa_shared::SyncStageRunnerError;

        for error in [
            ThoError::AdjusterFailed,
            ThoError::ContextReset,
            ThoError::NotFound,
            SyncStageRunnerError::SubmitErrorClosed.into(),
            SyncStageRunnerError::SubmitErrorCancelled.into(),
        ] {
            let received = round_trip_worker_error(error.clone().context("worker operation"));
            assert_eq!(std::mem::discriminant(received.what()), std::mem::discriminant(error.what()));
            assert_eq!((received.is_oom(), received.is_retryable_oom(), received.needs_cuda_ctx_reset()), (error.is_oom(), error.is_retryable_oom(), error.needs_cuda_ctx_reset()));
            assert_eq!(received.what().to_string(), error.to_string());
        }

        let received = round_trip_worker_error(ThoError::from(std::io::Error::other("CUDA out of memory")).context("read model"));
        assert!(matches!(received, ThoError::Anyhow(_)));
        assert!(!received.is_oom() && !received.is_retryable_oom() && !received.needs_cuda_ctx_reset());
        assert!(received.to_string().contains("read model") && received.to_string().contains("CUDA out of memory"));
    }

    #[test]
    fn test_worker_error_round_trip_preserves_terminal_recovery_failure() {
        use thoosa_shared::{CudaAction, CudaRecoveryFailure};

        // Terminal errors must retain diagnostics without becoming retryable again
        // when they pass through the same wire path as ordinary CUDA failures.
        for recovery in [None, Some(ThoError::oom().context("replacement allocation"))] {
            let has_recovery = recovery.is_some();
            let original = ThoError::Cuda { action: CudaAction::NeedsRestart, msg: "illegal address".into() }.context("original stage");
            let error = ThoError::CudaRecoveryFailed(Arc::new(CudaRecoveryFailure { original, recovery }));
            let received = round_trip_worker_error(error.context("worker recovery"));
            assert!(received.is_cuda_recovery_failed());
            assert!(!received.is_oom() && !received.is_retryable_oom() && !received.needs_cuda_ctx_reset());
            let ThoError::CudaRecoveryFailed(failure) = received.what() else { panic!("terminal error was lost") };
            assert!(failure.original.needs_cuda_ctx_reset());
            assert_eq!(failure.recovery.is_some(), has_recovery);
            assert!(received.to_string().contains("original stage") && received.to_string().contains("illegal address"));
            if let Some(recovery) = &failure.recovery {
                assert!(recovery.is_oom());
                assert!(recovery.to_string().contains("replacement allocation"));
            }
        }
    }

    #[tokio::test]
    async fn test_cancelled_copy_caller_keeps_activity_with_request() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        let handle = CudaIPCActorHandle { sender };
        // A CPU tensor exercises real copy admission and message ownership
        // without starting a CUDA worker or allocating GPU storage.
        let src = ThoTensor::new((), Tensor::f_from_slice(&[1f32]).unwrap()).unwrap();
        let mut copy = Box::pin(handle.copy(src, Device::Cuda(0).into()));
        assert!(futures::poll!(copy.as_mut()).is_pending());
        let request = receiver.recv().await.unwrap();
        drop(copy);
        let pause = thoosa_shared::cuda_activity_gate().request_pause(ThoError::ContextReset).unwrap();
        let mut idle = Box::pin(thoosa_shared::cuda_activity_gate().wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending());
        // The request retains the scope even though its caller has gone away.
        // Discarding it before dispatch is equivalent to completing that activity.
        drop(request);
        tokio::time::timeout(std::time::Duration::from_secs(2), idle).await.unwrap();
        thoosa_shared::cuda_activity_gate().finish_pause(pause, &[]);
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
            result.context("join IPC copy perf task")??;
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
    fn test_copy_cuda_worker_propagates_oom_ipc() -> Result<(), ThoError>{
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
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
            ).await.context("timed out waiting for worker IPC oversized H2D copy to fail")?;

            debug!("test_copy_ipc_cuda_worker_propagates_oom: copy_result={copy_result:?}");

            let error = match copy_result {
                Ok(_) => return Err(ThoError::new("worker IPC copy unexpectedly succeeded for oversized tensor")),
                Err(error) => error,
            };

            debug!("run_ipc_copy_oom: error={error:?}");

            assert!(matches!(error, ThoError::Cuda { action: thoosa_shared::CudaAction::HandleOOM, .. }), "worker OOM action was not propagated: {error:?}");
            assert!(error.is_oom(), "worker OOM classification was lost: {error:?}");
            assert!(error.is_retryable_oom(), "worker OOM retryability was lost: {error:?}");
            assert!(!error.needs_cuda_ctx_reset(), "ordinary worker OOM must not reset CUDA: {error:?}");


            debug!("All ok");

                Ok(())
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::new(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }



    #[test]
    fn heavy_copy_ipc_cuda_worker_once() -> Result<(), ThoError>{
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                let tho_device_idx = 1;
                let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
                let (d2h, h2d)  = run_ipc_copy_perf(cuda_actor, 3, 3).await?;

                info!("\n IPC worker copy d2h speed: {d2h:#2} GB/s, h2d speed {h2d:#2} GB/s");


                Ok(())
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::new(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }

    #[test]
    fn test_copy_cuda_worker_ok_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
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
                cuda_ref.shallow_clone(),//otherwise cuda_ref gets dropped and deleted from worker
                0.into(),//Cpu
            )).await?;

            debug!("run_main: cpu_tensor={cpu_tensor:?}");

            let actual_digest = tensor_digest(cpu_tensor.t_ref())?;
            if actual_digest.hex != expected_digest.hex {
                return Err(ThoError::new(format!("worker CUDA round-trip digest mismatch: expected={} actual={}", expected_digest.hex, actual_digest.hex)));
            }

            cuda_actor.drop_tho_tensor(reff).await?;

            debug!("run_main: drop ok");

            let r = track_await(cuda_actor.copy(
                ThoTensor::new_ref((), reff, vec![1000, 10000], tch::Kind::Float, None, 1),
                0.into(), //Cpu
            )).await;

            debug!("run_main: copy r={r:?}");

            let Err(error) = r else {
                panic!("Got r={r:?}, expected CudaWorker error");
            } ;
            assert!(matches!(error, ThoError::Anyhow(_)));
            assert_eq!("worker tensor tensor_key=1000000 not found", error.to_string());


            info!("main: \n{}", format_perf_map(&PERF));

                Ok(())
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::new(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }

}
