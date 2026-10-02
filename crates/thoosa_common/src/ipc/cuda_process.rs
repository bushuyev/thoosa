use super::pending::{KeepAlive, PendingRequests};
use crate::copy_ctx::{CudaActorHandle, HandleCtx};
use crate::cuda_utils::tensor_mem_size;
use crate::ipc::ipc_cuda_utils::send_fd;
use crate::ipc::serialization::{PostcardToFromBytes, ToFromBytes};
use crate::ipc::worker_msg::{CudaWorkerId, CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation, WorkerCopyDonePayload, WorkerInput};
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
use std::fmt::{Debug, Display};
use std::os::fd::{AsFd, OwnedFd};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tch::Device::{Cpu, Cuda};
use tch::{Device, Tensor};
use thoosa_shared::CudaActivityPauseTicket;
use thoosa_shared::ThoError;
use thoosa_shared::ipc_types::{IpcTensorKind, TensorKey};
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
use thoosa_shared::ipc_types::WorkerTensorInfoPayload;
use thoosa_shared::traits::StageStateInfo;
use crate::ipc::proxy::StageStateDump;

pub const CUDA_VISIBLE_DEVICES: &str = "CUDA_VISIBLE_DEVICES";//it's set to '' for main process to make sure it doesn't use GPU
pub const WORKER_ID_ENV: &str = "THOOSA_CUDA_WORKER_ID";
pub const IPC_SOCK_ENV: &str = "THOOSA_WORKER_IPC_SOCK";
pub  const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;
// Shutdown normally completes immediately, but keep a bounded kill fallback for a worker stuck after acknowledging the request.
const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const WORKER_READY_TIMEOUT: Duration = Duration::from_secs(30);
const WORKER_REAP_TIMEOUT: Duration = Duration::from_secs(10);

pub(crate) fn encode_endpoint_payload(endpoint_name: &str, payload: &str) -> String {
    format!("{endpoint_name}\t{payload}")
}



// Installed stages share an initializer name but must have distinct invocation keys.
pub(crate) fn endpoint_stage_key<StageId>(endpoint_name: String, stage_id: Option<&StageId>) -> String
where
    StageId:Display
{
    match stage_id {
        None => endpoint_name,
        Some(stage_id) => {
            format!("{endpoint_name}{stage_id}")
        }
    }
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
        worker_id: CudaWorkerId,
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    PurgeStage {
        input: CudaWorkerMsgIn,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<StageStateDump, ThoError>>
    },

    SendMsg {
        input: CudaWorkerMsgIn,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>
    },
}

// Recovery and shutdown never compete with ordinary work for queue capacity.
#[derive(Debug)]
enum BackendControl {
    Retire { authority: CudaActivityPauseTicket, respond_to: oneshot::Sender<Result<(), ThoError>> },
    Restart { authority: CudaActivityPauseTicket, respond_to: oneshot::Sender<Result<(), ThoError>> },
    Shutdown { respond_to: oneshot::Sender<Result<(), ThoError>> },
}

type PendingResponses = Arc<PendingRequests>;

#[derive(Debug)]
pub struct CudaIPCActor {
    gate: Arc<thoosa_shared::CudaActivityGate>,
    receiver: mpsc::Receiver<CudaIPCActorMessage>,

    control: mpsc::UnboundedReceiver<BackendControl>,
    spawned_state: Option<SpawnedState>,
    retirement_error: Option<ThoError>,
    next_id: Arc<AtomicU64>,
    tho_device_idx: usize,
    cuda_worker_args: Arc<Vec<String>>,
    // Keep an actor-owned descriptor so a reset can map the same parent allocator into the replacement worker.
    arena_fd: OwnedFd,
}

#[derive(Debug)]
struct SpawnedState {
    worker_id: CudaWorkerId,
    pending: PendingResponses,
    tx:Sender<CudaWorkerMsgIn>,
    child:Child,
    writer_handle:JoinHandle<()>,
    reader_handle:JoinHandle<()>,
    cancel: CancellationToken,
    device_info:DeviceInfo
}

// Killing is not completion: shared arena allocations remain live until wait()
// proves the process can no longer access them.
async fn kill_and_reap(child: &mut Child) -> Result<(), ThoError> {
    if child.try_wait()?.is_none() {
        child.start_kill().context("terminate CUDA worker")?;
        timeout(WORKER_REAP_TIMEOUT, child.wait()).await.context("timed out reaping CUDA worker")?.context("reap CUDA worker")?;
    }
    Ok(())
}

impl SpawnedState {
    async fn stop_io(&mut self) {
        self.cancel.cancel();
        self.reader_handle.abort();
        self.writer_handle.abort();
        let _ = (&mut self.reader_handle).await;
        let _ = (&mut self.writer_handle).await;
    }

    async fn retire(mut self, reason: ThoError) -> Result<(), ThoError> {
        self.pending.close(reason.clone());
        self.stop_io().await;
        kill_and_reap(&mut self.child).await?;
        self.pending.exited(reason);
        Ok(())
    }

    async fn shutdown(self, request_id: u64) -> Result<(), ThoError> {
        self.shutdown_with_timeout(request_id, WORKER_SHUTDOWN_TIMEOUT).await
    }

    async fn shutdown_with_timeout(mut self, request_id: u64, deadline: Duration) -> Result<(), ThoError> {
        self.pending.close(ThoError::new("CUDA worker is shutting down"));
        // Bound the whole exchange, including a full writer queue and a worker
        // which never acknowledges, rather than only the final child wait.
        let graceful = timeout(deadline, async {
            let reply = send_request(self.tx.clone(), Arc::clone(&self.pending), CudaWorkerMsgIn::new(self.worker_id, request_id, Operation::Shutdown { dummy: "shutdown".into() }), None, true).await?;
            unit_reply(reply)?;
            let status = self.child.wait().await?;
            if !status.success() { return Err(ThoError::new(format!("CUDA worker exited with {status}"))); }
            Ok::<_, ThoError>(())
        }).await;
        self.stop_io().await;
        kill_and_reap(&mut self.child).await?;
        self.pending.exited(ThoError::new("CUDA worker exited during shutdown"));
        graceful.context("CUDA worker shutdown timed out")?
    }
}

impl Drop for SpawnedState {
    fn drop(&mut self) {
        self.pending.close(ThoError::ContextReset);
        self.cancel.cancel();
        self.reader_handle.abort();
        self.writer_handle.abort();
        // Normal paths have already reaped. On task/runtime cancellation the
        // Child's kill_on_drop terminates it; the registry quarantines buffers
        // unless exit can be confirmed here.
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            self.pending.exited(ThoError::ContextReset);
        } else {
            self.pending.quarantine(ThoError::ContextReset.context("CUDA worker retirement interrupted before confirmed exit"));
        }
    }
}

fn unit_reply(response: CudaWorkerMsgOut) -> Result<(), ThoError> {
    match response.status { OpResult::Ok(_) => Ok(()), OpResult::Failed(error) => Err(error.into()) }
}

impl CudaIPCActor {
    async fn new(gate: Arc<thoosa_shared::CudaActivityGate>, cuda_worker_args: Vec<String>, receiver: mpsc::Receiver<CudaIPCActorMessage>, control: mpsc::UnboundedReceiver<BackendControl>, arena_fd: OwnedFd) -> Result<Self, ThoError> {
        let tho_device_idx = gate.device_idx();
        let cuda_worker_args = Arc::new(cuda_worker_args);

        let spawned_state = Self::spawn_worker_and_loops(CudaWorkerId { device_idx: tho_device_idx, generation: gate.device_generation() }, &cuda_worker_args, arena_fd.as_fd()).await?;


        debug!("CudaIPCActor::new: worker device_info={:?}", spawned_state.device_info);

        Ok(Self {
            gate,
            receiver,
            spawned_state: Some(spawned_state),
            retirement_error: None,
            control,
            next_id: Arc::new(AtomicU64::new(1)),
            tho_device_idx,
            cuda_worker_args,
            arena_fd,
        })
    }

    async fn spawn_worker_and_loops(worker_id: CudaWorkerId, cuda_worker_args: &Vec<String>, arena_fd: std::os::fd::BorrowedFd<'_>) -> Result<SpawnedState, ThoError> {
        debug!("CudaIPCActor::spawn_worker_and_loops, cuda_worker_args={cuda_worker_args:?}");

        let (child, stream, device_info) = Self::spawn_worker_for_cuda_device(worker_id, cuda_worker_args, arena_fd).await?;

        let (reader, writer) = stream.into_split();

        let (tx, rx): (Sender<CudaWorkerMsgIn>, Receiver<CudaWorkerMsgIn>) = mpsc::channel::<CudaWorkerMsgIn>(1024);

        let pending = Arc::new(PendingRequests::new(worker_id));
        let cancel = CancellationToken::new();
        let writer_handle = tokio::spawn(writer_loop(writer, rx, Arc::clone(&pending), cancel.child_token()));
        let reader_handle = tokio::spawn(reader_loop(reader, Arc::clone(&pending), cancel.child_token()));

        debug!("CudaIPCActor::spawn_worker_and_loops, ok");

        Ok( SpawnedState {
            worker_id,
            child,
            tx,
            pending,
            writer_handle,
            reader_handle,
            cancel,
            device_info
        })
    }

    async fn spawn_worker_for_cuda_device(worker_id: CudaWorkerId, args: &Vec<String>, arena_fd: std::os::fd::BorrowedFd<'_>) -> Result<(Child, UnixStream, DeviceInfo), ThoError> {
        let cuda_device = worker_id.device_idx - 1;
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
            // Each child sees its assigned physical GPU as CUDA ordinal zero.
            .env(CUDA_VISIBLE_DEVICES, cuda_device.to_string())
            .kill_on_drop(true)
            .env(IPC_SOCK_ENV, &sock_path)
            .env(WORKER_ID_ENV, serde_json::to_string(&worker_id)?)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        let ready = timeout(WORKER_READY_TIMEOUT, async {
            let (mut stream, _) = listener.accept().await?;
            // Send the arena first. Readiness is acknowledged only after all
            // stream/copy pools and the shared mapping have been initialized.
            send_fd(&stream, arena_fd).await?;
            let device_info: DeviceInfo = PostcardToFromBytes::from_bytes(&read_frame(&mut stream, "DeviceInfo/main", MAX_FRAME_LEN).await?)?;
            device_info.worker_id.check(worker_id)?;
            Ok::<_, ThoError>((stream, device_info))
        }).await;
        let (stream, device_info) = match ready {
            Ok(Ok(ready)) => ready,
            result => {
                kill_and_reap(&mut child).await?;
                return Err(match result { Ok(Err(error)) => error, Err(error) => ThoError::from(error).context("CUDA worker readiness timed out"), Ok(Ok(_)) => unreachable!() });
            }
        };

        Ok((child, stream, device_info))
    }


    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaIPActor::handle_message"))]
    async fn handle_message(&mut self, msg: CudaIPCActorMessage) -> Result<(), ThoError>{
        if self.spawned_state.is_none() {
            Self::reject(msg, ThoError::ContextReset);
            return Ok(());
        }
        match msg {
            CudaIPCActorMessage::Copy { src, dst_device, activity, respond_to } => {
                if let Err(error) = activity.check_operation(&self.gate) {
                    let _ = respond_to.send(Err(error));
                    return Ok(());
                }

                return if src.ref_key().is_some() {
                    if let Err(error) = src.ref_worker_id().expect("tensor worker").check(self.worker().worker_id) {
                        let _ = respond_to.send(Err(error));
                        return Ok(());
                    }
                    assert_eq!(*dst_device, Cpu, "If source is tensor ref and thus live on device it can be copied only to CPU");
                     self.handle_ref_d2h_copy(src, dst_device, activity, respond_to)
                } else {
                    debug!("handle_message: src={src:?} dst_device={dst_device:?}");
                    assert_ne!(*dst_device, Cpu, "If source is concrete tensor it should be on CPU and can be copied only to GPU");

                    self.handle_h2d_copy(src, dst_device, activity, respond_to)
                }

            }

            CudaIPCActorMessage::DropThoTensor { key, worker_id, respond_to } => {
                // A delayed destructor must never delete a replacement worker's key.
                if worker_id != self.worker().worker_id {
                    let _ = respond_to.send(Ok(()));
                    return Ok(());
                }
                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.worker().pending);
                let tx = self.worker().tx.clone();
                tokio::spawn(async move {
                    let result = send_request_and_wait_msg(tx, pending, CudaWorkerMsgIn::new(worker_id, id, Operation::DropThoTensor { tensor_key: key })).await.and_then(unit_reply);
                    let _ = respond_to.send(result);
                });
            }

            CudaIPCActorMessage::PurgeStage {mut input, activity, respond_to} => {
                let self_tx = self.worker().tx.clone();
                let pending = Arc::clone(&self.worker().pending);
                let worker_id = self.worker().worker_id;
                let next_id = Arc::clone(&self.next_id);
                let tx = self.worker().tx.clone();
                tokio::spawn(async move {
                    let Operation::GetStageStateInfo { endpoint_name, stage_id } = &input.operation else {
                        panic!("not Operation::GetStageStateInfo");
                    };
                    let endpoint_name= endpoint_name.clone();
                    let stage_id= stage_id.clone();
                    let response = send_request_and_wait_msg(
                        self_tx.clone(),
                        Arc::clone(&pending),
                        input,
                    ).await?;

                    if let OpResult::Ok(Some(v)) = response.status {
                        debug!("handle_message: Operation::PurgeStage: v={v:?}");
                        let mut tensors:HashMap<String, ThoTensor> = HashMap::new();
                        let Ok(info) = (v as Box<dyn Any>).downcast::<StageStateInfo>() else {
                            panic!("");//TODO return err
                        };

                        for (key, tensor_info) in info.tensors {
                            let t = Self::do_ref_d2h_copy(
                                worker_id,
                                next_id.fetch_add(1, Ordering::Relaxed),
                                Arc::clone(&pending),
                                tx.clone(),
                                tensor_info
                            ).await?;

                            tensors.insert(key, t);
                        }

                        debug!("handle_message: Operation::PurgeStage: tensors={tensors:?}");
                        let _ = respond_to.send(Ok(StageStateDump::new(info.scalars, tensors)));
                    }


                    let response = send_request_and_wait_msg(
                        self_tx,
                        Arc::clone(&pending),
                        CudaWorkerMsgIn {
                            worker_id,
                            request_id: 0,
                            operation: Operation::PurgeStage {
                                endpoint_name,
                                stage_id,
                            },
                        },
                    ).await?;

                    Ok::<_, ThoError>(())
                });
            }
            CudaIPCActorMessage::SendMsg { mut input, activity, respond_to } => {

                if let Err(error) = input.worker_id.check(self.worker().worker_id).and_then(|_| activity.check_operation(&self.gate)) {

                    let response = if matches!(input.operation, Operation::DropStage { .. } | Operation::GetStageStateInfo { .. }) {
                        Ok(CudaWorkerMsgOut { worker_id: input.worker_id, request_id: input.request_id, status: OpResult::Ok(None) })
                    } else {
                        Err(error)
                    };
                    let _ = respond_to.send(response);
                    return Ok(());
                }
                let self_tx = self.worker().tx.clone();
                let next_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.worker().pending);
                let gate = Arc::clone(&self.gate);
                let generation = input.worker_id.generation;

                input.request_id = next_id;
                let worker_id = self.worker().worker_id;
                let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let pending = Arc::clone(&self.worker().pending);
                let tx = self.worker().tx.clone();
                tokio::spawn(activity.run(async move {

                    match input.operation {

                        _=> {
                            let response = send_request_and_wait_msg(
                                self_tx,
                                pending,
                                input,
                            ).await;
                            // Reporting belongs to the admitted request, even if its
                            // proxy/initializer caller was cancelled before the reply.
                            let response = response.map(|mut response| {
                                if let OpResult::Failed(error) = response.status {
                                    let error = crate::copy_ctx::report_cuda_copy_result::<()>(Err(error.into()), &gate, generation).unwrap_err();
                                    response.status = error.into();
                                }
                                response
                            });
                            let _ = respond_to.send(response);

                            Ok::<_, ThoError>(())
                        }

                    }

                }));
            }


        }

        Ok(())
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

    fn handle_ref_d2h_copy(&mut self, source: ThoTensor, dst_device: ThoDevice, activity: thoosa_shared::CudaActivityScope, respond_to: oneshot::Sender<Result<ThoTensor, ThoError>>) -> Result<(), ThoError> {
        if dst_device.tho_device_idx() != 0 { return Err(ThoError::new("worker reference copy requires a CPU destination")); }
        let tensor_key = source.ref_key().expect("worker reference");
        let gate = Arc::clone(&self.gate);
        let generation = activity.generation(&gate).expect("copy device lease");
        let worker_id = self.worker().worker_id;
        let describe_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let pending = Arc::clone(&self.worker().pending);
        let tx = self.worker().tx.clone();
        tokio::spawn(activity.run(async move {
            let _source = source;
            let result = async {
                let info = worker_tensor_info(send_request_and_wait_msg(tx.clone(), Arc::clone(&pending), CudaWorkerMsgIn::new(worker_id, describe_id, Operation::DescribeRef { tensor_key })).await?)?;
                if info.tensor_key != tensor_key.into() || tensor_mem_size(info.kind.into(), &info.shape) != info.tensor_byte_len {
                    return Err(ThoError::new("worker returned inconsistent tensor metadata"));
                }

                Self::do_ref_d2h_copy(worker_id, transfer_id, pending, tx, info).await
            }.await;
            let _ = respond_to.send(crate::copy_ctx::report_cuda_copy_result(result, &gate, generation));
        }));
        Ok(())
    }

    async fn do_ref_d2h_copy(worker_id: CudaWorkerId, transfer_id: u64, pending: Arc<PendingRequests>, tx: Sender<CudaWorkerMsgIn>, info: WorkerTensorInfoPayload) -> Result<ThoTensor, ThoError> {


        let destination = Tensor::f_empty(info.shape.as_slice(), (info.kind.into(), Cpu))?;
        let arena_offset = parent_shared_arena_offset(&destination, info.tensor_byte_len, "D2H destination")?;
        let destination = Arc::new(parking_lot::Mutex::new(destination));
        // The registry retains this allocation even if this task aborts
        // or transport fails while the worker is still writing to it.
        let response = send_request(tx, pending, CudaWorkerMsgIn::new(worker_id, transfer_id, Operation::D2hCopy {
            tensor_key:info.tensor_key.clone(),
            kind: info.kind,
            shape: info.shape,
            tensor_byte_len: info.tensor_byte_len,
            arena_offset,
        }), Some(Box::new(Arc::clone(&destination))), false).await?;
        let done = Self::copy_result(response)?;

        if done.transfer_id != transfer_id || done.tensor_ref != info.tensor_key {
            return Err(ThoError::new("worker D2H reply has an unexpected transfer identity"));
        }
        let destination = Arc::try_unwrap(destination).map_err(|_| ThoError::new("completed D2H storage is still leased"))?.into_inner();
        ThoTensor::new((), destination)
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaIPActor::handle_h2d_copy"))]
    fn handle_h2d_copy(&mut self, src: ThoTensor, dst_device: ThoDevice, activity: thoosa_shared::CudaActivityScope, respond_to: oneshot::Sender<Result<ThoTensor, ThoError>>) -> Result<(), ThoError> {
        if dst_device.tho_device_idx() != self.tho_device_idx {
            let _ = respond_to.send(Err(ThoError::new("H2D destination belongs to another CUDA actor")));
            return Ok(());
        }
        let gate = Arc::clone(&self.gate);
        let generation = activity.generation(&gate).expect("copy device lease");
        let worker_id = self.worker().worker_id;
        let transfer_id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let pending = Arc::clone(&self.worker().pending);
        let self_tx = self.worker().tx.clone();
        let src_kind = src.kind();
        let src_shape = src.size();
        let tensor_byte_len = tensor_mem_size(src_kind, &src_shape);
        let src_tensor = src.t();

        debug!("CudaIPCActor::handle_h2d_copy: src_shape={src_shape:?} tensor_byte_len={tensor_byte_len}");

        tokio::spawn(activity.run(async move {
            // Transfer ownership follows the request, not the awaiting caller.
            let result = async {
                // Older CPU tensors may predate allocator installation; copy only those tensors once into arena-owned storage before sharing their offset.
                let (shared_source, arena_offset) = spawn_blocking(move || ensure_parent_shared_arena_tensor(src_tensor, tensor_byte_len))
                    .await
                    .context("join parent H2D arena staging task")??;
                let response = send_request(
                    self_tx,
                    pending,
                    CudaWorkerMsgIn {
                        worker_id,
                        request_id: transfer_id,
                        operation: Operation::H2dCopy {
                            kind: src_kind.into(),
                            shape: src_shape.clone(),
                            tensor_byte_len,
                            arena_offset,
                        },
                    },
                    Some(Box::new(shared_source)),
                    false,
                ).await?;
                debug!("CudaIPCActor::handle_h2d_copy: response={response:?}");
                let done = Self::copy_result(response)?;
                if done.transfer_id != transfer_id {
                    return Err(ThoError::new(format!("worker H2D response transfer_id={} but expected {transfer_id}", done.transfer_id)));
                }
                let TensorKey::Data(tensor_ref ) = done.tensor_ref else {
                    return Err(ThoError::new(format!("worker H2D got done.tensor_ref={:?} expecting data", done.tensor_ref)));
                };
                // The worker has completed its read before acknowledging, so the parent may now release the arena allocation.
                Ok(ThoTensor::new_ref((), tensor_ref, src_shape, src_kind, None, worker_id))
            }.await;
            let _ = respond_to.send(crate::copy_ctx::report_cuda_copy_result(result, &gate, generation));
        }));

        Ok(())
    }

    fn worker(&self) -> &SpawnedState { self.spawned_state.as_ref().expect("admitted work requires a live worker") }

    fn reject(msg: CudaIPCActorMessage, error: ThoError) {
        match msg {
            CudaIPCActorMessage::Copy { respond_to, .. } => { let _ = respond_to.send(Err(error)); }
            CudaIPCActorMessage::SendMsg { respond_to, .. } => { let _ = respond_to.send(Err(error)); }
            CudaIPCActorMessage::DropThoTensor { respond_to, .. } => { let _ = respond_to.send(Ok(())); }
            CudaIPCActorMessage::PurgeStage {respond_to, .. } => { let _ = respond_to.send(Err(error)); }
        }
    }

    async fn retire(&mut self, reason: ThoError) -> Result<(), ThoError> {
        if let Some(error) = &self.retirement_error { return Err(error.clone()); }
        if let Some(worker) = self.spawned_state.take() {
            if let Err(error) = worker.retire(reason).await {
                self.retirement_error = Some(error.clone());
                return Err(error);
            }
        }
        Ok(())
    }

    async fn control(&mut self, control: BackendControl) -> bool {
        match control {
            BackendControl::Retire { authority, respond_to } => {
                let result = match self.gate.check_recovery_authority(authority, false) {
                    Ok(()) => self.retire(ThoError::ContextReset).await,
                    Err(error) => Err(error),
                };
                let _ = respond_to.send(result);
            }
            BackendControl::Restart { authority, respond_to } => {
                let result = async {
                    self.gate.check_recovery_authority(authority, true)?;
                    self.retire(ThoError::ContextReset).await?;
                    // A fresh primary context requires the old process to have
                    // exited before its replacement is even spawned.
                    let worker_id = CudaWorkerId { device_idx: self.tho_device_idx, generation: thoosa_shared::CudaDeviceGeneration(self.gate.device_generation().0.checked_add(1).expect("worker generation overflow")) };
                    let replacement = Self::spawn_worker_and_loops(worker_id, &self.cuda_worker_args, self.arena_fd.as_fd()).await?;
                    if let Err(error) = self.gate.check_recovery_authority(authority, true) {
                        replacement.retire(error.clone()).await?;
                        return Err(error);
                    }
                    self.spawned_state = Some(replacement);
                    Ok(())
                }.await;
                let _ = respond_to.send(result);
            }
            BackendControl::Shutdown { respond_to } => {
                self.receiver.close();
                while let Ok(message) = self.receiver.try_recv() { Self::reject(message, ThoError::new("CUDA actor is shutting down")); }
                let result = match self.spawned_state.take() {
                    Some(worker) => worker.shutdown(self.next_id.fetch_add(1, Ordering::Relaxed)).await,
                    None => Ok(()),
                };
                let _ = respond_to.send(result);
                return true;
            }
        }
        false
    }

    async fn run(mut actor: CudaIPCActor) -> Result<(), ThoError> {
        loop {
            let failed = actor.spawned_state.as_ref().map(|worker| worker.pending.failed.clone());
            tokio::select! {
                biased;
                Some(control) = actor.control.recv() => { if actor.control(control).await { return Ok(()); } }
                _ = async { match failed { Some(failed) => failed.cancelled().await, None => std::future::pending().await } } => {
                    let worker = actor.worker();
                    let generation = worker.worker_id.generation;
                    let error = worker.pending.failure();
                    // Restart's acknowledgment precedes coordinator publication.
                    // If the ready child dies in that interval, report against
                    // its published generation instead of discarding it as future.
                    if generation.0 > actor.gate.device_generation().0 {
                        if let Err(failure) = actor.gate.wait_if_paused().await {
                            actor.retire(failure).await?;
                            continue;
                        }
                    }
                    // Transport loss is reported even if all callers cancelled.
                    let cause = if error.needs_cuda_ctx_reset() {
                        error.clone()
                    } else {
                        ThoError::Cuda { action: thoosa_shared::CudaAction::NeedsRestart, msg: format!("CUDA worker transport failed: {error}") }
                    };
                    if GlobalCtx::is_installed() {
                        let _ = crate::copy_ctx::report_cuda_copy_result::<()>(Err(cause), &actor.gate, generation);
                    } else {
                        // A worker can die while GlobalCtx is still constructing
                        // other devices. Close this gate without using the singleton.
                        actor.gate.fail_process(cause);
                    }
                    actor.retire(error).await?;
                }
                message = actor.receiver.recv() => match message {
                    Some(message) => actor.handle_message(message).await?,
                    None => { actor.retire(ThoError::new("CUDA actor closed")).await?; return Ok(()); }
                }
            }
        }
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

struct TransportTaskGuard<'a> {
    pending: &'a PendingRequests,
    cancel: &'a CancellationToken,
}

impl Drop for TransportTaskGuard<'_> {
    fn drop(&mut self) {
        // A panic or unexpected task exit must wake the actor's recovery path,
        // just like an I/O error. Intentional retirement already owns cleanup.
        if !self.cancel.is_cancelled() {
            self.pending.transport_failed(ThoError::new("worker transport task stopped"));
        }
    }
}

async fn writer_loop<W>(
    mut writer: W,
    mut rx: mpsc::Receiver<CudaWorkerMsgIn>,
    pending: PendingResponses,
    cancel: CancellationToken,
) where
    W: AsyncWrite + Unpin,
{
    let _guard = TransportTaskGuard { pending: &pending, cancel: &cancel };
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
                pending.finish_one(request_id, Err(ThoError::new(format!("failed to encode request {request_id}: {e}"))));
                continue;
            }
        };

        if let Err(e) = write_frame(&mut writer, &frame, "CudaWorkerMsg").await {
            error!("worker IPC writer failed: {e}");
            pending.transport_failed(ThoError::new(format!("worker IPC writer stopped: {e}")));
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
    let _guard = TransportTaskGuard { pending: &pending, cancel: &cancel };
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

                        pending.transport_failed(ThoError::new(format!("worker IPC reader stopped: {e}")));

                        break;
                    }
                }
            }
        };

        match PostcardToFromBytes::from_bytes::<CudaWorkerMsgOut>(&frame) {
            Ok(msg) => {
                debug!("reader_loop: msg={msg:?}");
                if let Err(error) = pending.complete(msg) {
                    pending.transport_failed(error);
                    break;
                }
            }
            Err(e) => {
                pending.transport_failed(ThoError::new(format!("bad worker response frame: {e}")));
                break;
            }
        }

    }

    debug!("worker IPC reader loop finished");
}

//TODO inline
async fn send_request_and_wait_msg(tx: mpsc::Sender<CudaWorkerMsgIn>, pending: PendingResponses, request: CudaWorkerMsgIn) -> Result<CudaWorkerMsgOut, ThoError> {
    send_request(tx, pending, request, None, false).await
}

async fn send_request(tx: mpsc::Sender<CudaWorkerMsgIn>, pending: PendingResponses, request: CudaWorkerMsgIn, keepalive: Option<KeepAlive>, shutdown: bool) -> Result<CudaWorkerMsgOut, ThoError> {
    let id = request.request_id;
    let (reply_tx, reply_rx) = oneshot::channel();
    let permit = tx.reserve().await.context("worker IPC writer task is closed")?;
    pending.enqueue(permit, request, reply_tx, keepalive, shutdown)?;
    reply_rx.await.with_context(|| format!("response channel for request {id} was dropped"))?
}

#[derive(Debug, Clone)]
pub struct CudaIPCActorHandle {
    gate: Arc<thoosa_shared::CudaActivityGate>,
    tho_device_idx: usize,
    sender: mpsc::Sender<CudaIPCActorMessage>,
    control: mpsc::UnboundedSender<BackendControl>,
}

///executes cuda actions in a separate process started from CudaWorker::run_worker
impl CudaIPCActorHandle {
    pub fn worker_id(&self) -> Result<CudaWorkerId, ThoError> {
        let activity = thoosa_shared::CudaActivityScope::for_operation(&self.gate)?;
        Ok(CudaWorkerId { device_idx: self.tho_device_idx, generation: activity.generation(&self.gate).expect("worker lease") })
    }

    fn activity_for(&self, worker_id: CudaWorkerId) -> Result<thoosa_shared::CudaActivityScope, ThoError> {
        let activity = thoosa_shared::CudaActivityScope::for_operation(&self.gate)?;
        worker_id.check(CudaWorkerId { device_idx: self.tho_device_idx, generation: activity.generation(&self.gate).expect("worker lease") })?;
        Ok(activity)
    }
    pub async fn new(gate: Arc<thoosa_shared::CudaActivityGate>, cuda_worker_args: Vec<String>, arena_fd: OwnedFd) -> Result<Self, ThoError> {
        let tho_device_id = gate.device_idx();
        debug!("CudaIPCActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let (control, controls) = mpsc::unbounded_channel();
        let actor = CudaIPCActor::new(Arc::clone(&gate), cuda_worker_args, receiver, controls, arena_fd).await?;
        tokio::spawn(CudaIPCActor::run(actor));

        Ok(Self {
            tho_device_idx: tho_device_id,
            gate,
            sender,
            control,
        })
    }

    pub async fn invoke_stage(&self, worker_id: CudaWorkerId, endpoint_name: String, stage_id: Option<String>, input: Box<dyn WorkerInput>, consume:bool) -> anyhow::Result<CudaWorkerMsgOut> {
        debug!("CudaIPCActorHandle::invoke_stage: endpoint_name={endpoint_name} stage_id={stage_id:?} input={input:?}");
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                activity: self.activity_for(worker_id)?,
                input: CudaWorkerMsgIn {
                    worker_id,
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

    pub async fn install_stage(&self, worker_id: CudaWorkerId, endpoint_name: String, stage_id: Option<String>, init_data: Box<dyn WorkerInput>) -> anyhow::Result<CudaWorkerMsgOut> {
        debug!("CudaIPCActorHandle::invoke_stage: install_stage_2={endpoint_name} stage_id={stage_id:?} init_data={init_data:?}");

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                activity: self.activity_for(worker_id)?,
                input: CudaWorkerMsgIn {
                    worker_id,
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
        let activity = thoosa_shared::CudaActivityScope::for_operation(&self.gate)?;
        if let Some(worker_id) = src.ref_worker_id() {
            worker_id.check(CudaWorkerId { device_idx: self.tho_device_idx, generation: activity.generation(&self.gate).expect("copy lease") })?;
        }
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

    #[cfg(test)]
    async fn restart(&self) -> Result<(), ThoError> {
        crate::runner::global_ctx::GlobalCtx::global().restart_cuda_device(self.tho_device_idx).await
    }

    async fn restart_backend(&self, authority: CudaActivityPauseTicket) -> Result<(), ThoError> {
        self.gate.check_recovery_authority(authority, true)?;
        let (respond_to, recv) = oneshot::channel();
        self.control.send(BackendControl::Restart { authority, respond_to }).context("CUDA actor control channel closed")?;
        recv.await.context("CUDA restart response channel closed")?
    }

    async fn retire_backend(&self, authority: CudaActivityPauseTicket) -> Result<(), ThoError> {
        self.gate.check_recovery_authority(authority, false)?;
        let (respond_to, recv) = oneshot::channel();
        self.control.send(BackendControl::Retire { authority, respond_to }).context("CUDA actor control channel closed")?;
        recv.await.context("CUDA retirement response channel closed")?
    }

    async fn shutdown(&self) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.control.send(BackendControl::Shutdown { respond_to }).context("CUDA actor control channel closed")?;
        recv.await.context("CUDA shutdown response channel closed")?
    }

    async fn drop_tho_tensor(&self, key:u64, worker_id: CudaWorkerId) -> Result<(), ThoError> {
        if worker_id.device_idx != self.tho_device_idx || worker_id.generation != self.gate.device_generation() {
            return Ok(());
        }
        debug!("CudaIPCActorHandle::drop_tho_tensor: key={key}");
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::DropThoTensor {
                key,
                worker_id,
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        recv.await.context("CUDA actor response channel closed")?
    }


    async fn init_handle_ctx(&self) -> Result<Box<dyn crate::copy_ctx::HandleCtx>, ThoError> {
        Ok(Box::new(IPCHanldeCTx{}))
    }


    async fn purge_stage_state(&self, stage_id: Option<String>, endpoint_name: String, worker_id: CudaWorkerId) -> Result<StageStateDump, ThoError> {
        debug!("CudaIPCActorHandle::purge_stage_state: stage_id={stage_id:?} endpoint_name={endpoint_name:?} worker_id={worker_id:?} self.gate={:?}", self.gate);
        if worker_id.device_idx != self.tho_device_idx || worker_id.generation != self.gate.device_generation() || self.gate.is_paused() {
            return Err(ThoError::AdjusterFailed);//TODO add ThoError variant
        }
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::PurgeStage {
                activity: self.activity_for(worker_id)?,
                input: CudaWorkerMsgIn {
                    worker_id,
                    request_id: 0,
                    operation: Operation::GetStageStateInfo {
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

    async fn drop_endpoint(&self, stage_id: Option<String>, endpoint_name: String, worker_id: CudaWorkerId) -> Result<CudaWorkerMsgOut, ThoError> {
        // Retired endpoint destructors are expected after cache invalidation.
        // They must neither wait for recovery nor reach the replacement worker.
        if worker_id.device_idx != self.tho_device_idx || worker_id.generation != self.gate.device_generation() || self.gate.is_paused() {
            return Ok(CudaWorkerMsgOut { worker_id, request_id: 0, status: OpResult::Ok(None) });
        }
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                activity: self.activity_for(worker_id)?,
                input: CudaWorkerMsgIn {
                    worker_id,
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
        let worker_id = self.worker_id()?;
        debug!("CudaIPCActorHandle::debug_worker");

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SendMsg {
                activity: self.activity_for(worker_id)?,
                input: CudaWorkerMsgIn {
                    worker_id,
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
    pub worker_id: CudaWorkerId,
    pub(crate) free_mem:usize,
    pub(crate) total_mem:usize,
}

#[cfg(test)]
mod tests {
    use super::{BackendControl, PendingRequests};
    use crate::copy_ctx::CudaActorHandle;
    use crate::cuda_utils::{cuda_mem, fail_with_unrecoverable_error, speed_gb_per_s};
    use crate::ipc::cuda_process::{CudaIPCActor, CudaIPCActorHandle, CudaIPCActorMessage, DeviceInfo, IPC_SOCK_ENV, encode_endpoint_payload};
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
    use anyhow::Context as _;
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
    use tokio::sync::mpsc;
    use tokio::task::JoinSet;

    fn round_trip_worker_error(error: ThoError) -> ThoError {
        use crate::ipc::serialization::{PostcardToFromBytes, ToFromBytes};
        use crate::ipc::worker_msg::CudaWorkerMsgOut;

        // Exercise the worker's reply encoding and the parent's actual copy error consumer.
        let reply = CudaWorkerMsgOut { worker_id: crate::ipc::worker_msg::CudaWorkerId { device_idx: 2, generation: thoosa_shared::CudaDeviceGeneration(7) }, request_id: 42, status: error.into() };
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
            ThoError::NotFound(Some("worker".to_string())),
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
    async fn test_recovery_control_bypasses_full_work_queue_and_checks_authority() {
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let (sender, _receiver) = mpsc::channel(1);
        let (control, mut controls) = mpsc::unbounded_channel();
        let handle = CudaIPCActorHandle { gate: Arc::clone(&gate), tho_device_idx: 1, sender, control };
        let (respond_to, _) = tokio::sync::oneshot::channel();
        let worker_id = handle.worker_id().unwrap();
        handle.sender.send(CudaIPCActorMessage::DropThoTensor { key: 42, worker_id, respond_to }).await.unwrap();
        let wrong = thoosa_shared::CudaActivityGate::new(2).request_pause(ThoError::ContextReset).unwrap();
        assert!(handle.restart_backend(wrong).await.is_err());
        assert!(handle.retire_backend(wrong).await.is_err());
        assert!(controls.try_recv().is_err());
        let authority = gate.request_pause(ThoError::ContextReset).unwrap();
        let mut retirement = Box::pin(handle.retire_backend(authority));
        assert!(futures::poll!(retirement.as_mut()).is_pending());
        let BackendControl::Retire { authority: received, respond_to } = controls.try_recv().unwrap() else { panic!("expected retirement control") };
        assert_eq!(authority, received);
        respond_to.send(Ok(())).unwrap();
        retirement.await.unwrap();
    }

    #[tokio::test]
    async fn test_worker_exit_precedes_pending_storage_release_and_shutdown_is_bounded() {
        use super::*;
        struct ExitCheckedStorage(u32);
        impl Drop for ExitCheckedStorage {
            fn drop(&mut self) {
                // Assert at the allocation's destruction, not only after the
                // retirement future returns: releasing before kill must fail.
                assert!(!std::path::Path::new(&format!("/proc/{}", self.0)).exists(), "storage released while worker can still access it");
            }
        }
        for graceful in [false, true] {
            let child = Command::new("sleep").arg("60").kill_on_drop(true).spawn().unwrap();
            let pid = child.id().unwrap();
            let worker_id = CudaWorkerId { device_idx: 1, generation: thoosa_shared::CudaDeviceGeneration(0) };
            let pending = Arc::new(PendingRequests::new(worker_id));
            let (tx, mut rx) = mpsc::channel(1);
            let storage = Arc::new(ExitCheckedStorage(pid));
            let weak = Arc::downgrade(&storage);
            let (reply, mut response) = oneshot::channel();
            pending.enqueue(tx.reserve().await.unwrap(), CudaWorkerMsgIn::new(worker_id, 1, Operation::Debug), reply, Some(Box::new(storage)), false).unwrap();
            rx.recv().await.unwrap();
            let worker = SpawnedState {
                worker_id, pending: Arc::clone(&pending), tx, child,
                writer_handle: tokio::spawn(async move { let _rx = rx; std::future::pending::<()>().await; }),
                reader_handle: tokio::spawn(std::future::pending()),
                cancel: CancellationToken::new(), device_info: DeviceInfo { worker_id, free_mem: 0, total_mem: 0 },
            };
            pending.close(ThoError::ContextReset);
            assert!(response.try_recv().is_err());
            assert!(weak.upgrade().is_some());
            if graceful {
                assert!(timeout(Duration::from_secs(2), worker.shutdown_with_timeout(2, Duration::from_millis(10))).await.unwrap().is_err());
            } else {
                timeout(Duration::from_secs(2), worker.retire(ThoError::ContextReset)).await.unwrap().unwrap();
            }
            assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists(), "pending buffers cannot be released before child exit and reaping");
            assert!(response.await.unwrap().is_err());
            assert!(weak.upgrade().is_none());
        }
    }

    #[tokio::test]
    async fn test_retired_ipc_handles_cannot_reach_replacement_worker() {
        use crate::ipc::worker_msg::CudaWorkerId;
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
        let handle = CudaIPCActorHandle { gate: Arc::clone(&gate), tho_device_idx: 1, sender, control: mpsc::unbounded_channel().0 };
        let retired = handle.worker_id().unwrap();
        let old_tensor = || ThoTensor::new_ref((), 42, vec![1], Kind::Float, None, retired);
        let ticket = gate.request_pause(ThoError::ContextReset).unwrap();
        gate.finish_pause(ticket);

        assert!(matches!(handle.copy(old_tensor(), 0usize.into()).await.unwrap_err().what(), ThoError::ContextReset));
        assert!(matches!(ThoError::from(handle.invoke_stage(retired, "stage".into(), Some(42.to_string()), Box::new(()), false).await.unwrap_err()).what(), ThoError::ContextReset));
        assert!(handle.install_stage(retired, "stage".into(), Some(42.to_string()), Box::new(())).await.is_err());
        handle.drop_tho_tensor(42, retired).await.unwrap();
        handle.drop_endpoint(Some(42.to_string()), "stage".into(), retired).await.unwrap();
        handle.purge_stage_state(Some(42.to_string()), "stage".into(), retired).await.unwrap();
        assert!(receiver.try_recv().is_err(), "old IDs must never be rebound to the new worker");

        let fresh: CudaWorkerId = handle.worker_id().unwrap();
        let mut invoke = Box::pin(handle.invoke_stage(fresh, "stage".into(), Some(42.to_string()), Box::new(()), false));
        assert!(futures::poll!(invoke.as_mut()).is_pending());
        let message = receiver.recv().await.unwrap();
        drop(invoke);
        let ticket = gate.request_pause(ThoError::ContextReset).unwrap();
        let mut idle = Box::pin(gate.wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending(), "cancelled invocation must keep its request leased");
        let CudaIPCActorMessage::SendMsg { input, .. } = &message else { panic!("expected stage request") };
        assert_eq!(input.worker_id, fresh);
        drop(message);
        idle.await;
        gate.finish_pause(ticket);
    }

    #[tokio::test]
    async fn test_ipc_reply_identity_is_checked_after_wire_routing() {
        use crate::ipc::serialization::{PostcardToFromBytes, ToFromBytes};
        use crate::ipc::worker_msg::{CudaWorkerId, CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation};
        use thoosa_shared::CudaDeviceGeneration;
        for wrong_worker in [CudaWorkerId { device_idx: 1, generation: CudaDeviceGeneration(0) }, CudaWorkerId { device_idx: 2, generation: CudaDeviceGeneration(1) }] {
            let current = CudaWorkerId { device_idx: 1, generation: CudaDeviceGeneration(1) };
            let pending = Arc::new(PendingRequests::new(current));
            let cancel = tokio_util::sync::CancellationToken::new();
            let (reader, mut writer) = tokio::io::duplex(1024);
            let reader_task = tokio::spawn(super::reader_loop(reader, Arc::clone(&pending), cancel.clone()));
            let (sender, mut requests) = tokio::sync::mpsc::channel(1);
            let request = CudaWorkerMsgIn::new(current, 42, Operation::Debug);
            let mut response = Box::pin(super::send_request_and_wait_msg(sender, Arc::clone(&pending), request));
            assert!(futures::poll!(response.as_mut()).is_pending());
            assert_eq!(requests.recv().await.unwrap().worker_id, current);
            let reply = CudaWorkerMsgOut { worker_id: wrong_worker, request_id: 42, status: OpResult::Ok(None) };
            super::write_frame(&mut writer, &PostcardToFromBytes::to_bytes(&reply).unwrap(), "test").await.unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), pending.failed.cancelled()).await.unwrap();
            assert!(futures::poll!(response.as_mut()).is_pending(), "a malformed reply cannot prove worker access ended");
            pending.exited(ThoError::ContextReset);
            let error = tokio::time::timeout(std::time::Duration::from_secs(2), response).await.unwrap().unwrap_err();
            assert!(matches!(error.what(), ThoError::ContextReset));
            cancel.cancel();
            reader_task.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_ipc_copy_admission_is_device_local() {
        let failed_gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let healthy_gate = Arc::new(thoosa_shared::CudaActivityGate::new(2));
        let (failed_tx, mut failed_rx) = tokio::sync::mpsc::channel(1);
        let (healthy_tx, mut healthy_rx) = tokio::sync::mpsc::channel(1);
        let failed = CudaIPCActorHandle { gate: Arc::clone(&failed_gate), tho_device_idx: 1, sender: failed_tx, control: mpsc::unbounded_channel().0 };
        let healthy = CudaIPCActorHandle { gate: healthy_gate, tho_device_idx: 2, sender: healthy_tx, control: mpsc::unbounded_channel().0 };
        let pause = failed_gate.request_pause(ThoError::ContextReset).unwrap();
        for terminal in [false, true] {
            if terminal { failed_gate.fail_pause(pause, None); }
            let src = || ThoTensor::new((), Tensor::f_from_slice(&[1f32]).unwrap()).unwrap();
            let error = failed.copy(src(), 1usize.into()).await.unwrap_err();
            assert_eq!(error.is_cuda_recovery_failed(), terminal);
            assert!(failed_rx.try_recv().is_err());
            // Copy admission is in the real parent handle; the worker need not
            // start to prove an unrelated recovery cannot block publication.
            let mut copy = Box::pin(healthy.copy(src(), 2usize.into()));
            assert!(futures::poll!(copy.as_mut()).is_pending());
            assert!(matches!(healthy_rx.try_recv().unwrap(), CudaIPCActorMessage::Copy { .. }));
        }
    }

    #[tokio::test]
    async fn test_cancelled_copy_caller_keeps_activity_with_request() {
        let (sender, mut receiver) = tokio::sync::mpsc::channel(1);
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let handle = CudaIPCActorHandle { sender, tho_device_idx: 1, gate: Arc::clone(&gate), control: mpsc::unbounded_channel().0 };
        // A CPU tensor exercises real copy admission and message ownership
        // without starting a CUDA worker or allocating GPU storage.
        let src = ThoTensor::new((), Tensor::f_from_slice(&[1f32]).unwrap()).unwrap();
        let mut copy = Box::pin(handle.copy(src, Device::Cuda(0).into()));
        assert!(futures::poll!(copy.as_mut()).is_pending());
        let request = receiver.recv().await.unwrap();
        drop(copy);
        let pause = gate.request_pause(ThoError::ContextReset).unwrap();
        let mut idle = Box::pin(gate.wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending());
        // The request retains the scope even though its caller has gone away.
        // Discarding it before dispatch is equivalent to completing that activity.
        drop(request);
        tokio::time::timeout(std::time::Duration::from_secs(2), idle).await.unwrap();
        gate.finish_pause(pause);
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

            cuda_actor.drop_tho_tensor(reff, cuda_ref.ref_worker_id().unwrap()).await?;

            debug!("run_main: drop ok");

            let r = track_await(cuda_actor.copy(
                ThoTensor::new_ref((), reff, vec![1000, 10000], tch::Kind::Float, None, cuda_ref.ref_worker_id().unwrap()),
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
