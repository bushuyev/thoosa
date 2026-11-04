use thoosa_common::tch_error::tch_result;
use super::pending::{KeepAlive, PendingRequests};
use super::endpoints::{EndpointEvent, EndpointKey, EndpointRegistration, EndpointRegistry};
use crate::copy_ctx::{CudaActorHandle};
use crate::cuda_utils::tensor_mem_size;
use crate::ipc::ipc_cuda_utils::send_fd;
use crate::ipc::serialization::{PostcardToFromBytes, ToFromBytes};
use crate::ipc::worker_msg::{CudaWorkerId, CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation, WorkerCopyDonePayload, WorkerInput};
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use anyhow::Context as _;
use async_trait::async_trait;
use log::{debug, error, info, trace, warn};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::backtrace::Backtrace;
#[cfg(test)]
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::os::fd::{AsFd, OwnedFd};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tch::Device::{Cpu, Cuda};
use tch::Tensor;
use thoosa_shared::CudaActivityPauseTicket;
use thoosa_shared::ThoError;
use thoosa_shared::ipc_types::{H2DTensorKey, TensorKey};
use thoosa_shims::arena::{shared_arena_capacity, shared_arena_offset_of};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::mpsc::{Receiver, Sender};
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, spawn_blocking};
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::Instrument;
use thoosa_common::ipc::tensor_info::WorkerTensorInfoPayload;
use thoosa_common::ipc::state_transfer::StageStateInfo;
use crate::ipc::proxy::StageStateDump;

pub const CUDA_VISIBLE_DEVICES: &str = "CUDA_VISIBLE_DEVICES";//it's set to '' for main process to make sure it doesn't use GPU
pub const WORKER_ID_ENV: &str = "THOOSA_CUDA_WORKER_ID";
pub const IPC_SOCK_ENV: &str = "THOOSA_WORKER_IPC_SOCK";
pub  const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;
// Shutdown normally completes immediately, but keep a bounded kill fallback for a worker stuck after acknowledging the request.
const WORKER_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(30);
const WORKER_READY_TIMEOUT: Duration = Duration::from_secs(30);
const WORKER_REAP_TIMEOUT: Duration = Duration::from_secs(10);




// Installed stages share an initializer name but must have distinct invocation keys.
pub(crate) fn endpoint_stage_key<StageId>(endpoint_name: String, stage_id: Option<&StageId>) -> String
where
    StageId:Display
{
    match stage_id {
        None => format!("{}:{endpoint_name}:none", endpoint_name.len()),
        Some(stage_id) => {
            // The endpoint and optional instance id form a pair, not a concatenated name.
            format!("{}:{endpoint_name}:some:{stage_id}", endpoint_name.len())
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




// State uploads acknowledge installation; they do not allocate a Data key
// whose lifetime can be represented by a worker tensor reference.
#[derive(Debug)]
pub enum IpcCopyResult {
    Tensor(ThoTensor),
    StateInstalled,
}

// Actor messages describe handle operations; only the actor builds the worker protocol.
#[derive(Debug)]
enum CudaIPCActorMessage {
    Copy{
        src:ThoTensor,
        dst_device:ThoDevice,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<IpcCopyResult, ThoError>>,
        dst_key: Option<H2DTensorKey>//TODO it's needed only for IPC d2h
    },

    PurgeStage {
        worker_id: CudaWorkerId,
        endpoint_name: String,
        stage_id: Option<String>,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<StageStateDump, ThoError>>
    },

    InvokeStage {
        worker_id: CudaWorkerId,
        endpoint_name: String,
        stage_id: Option<String>,
        input: Box<dyn WorkerInput>,
        consume: bool,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>,
    },
    InstallStage {
        worker_id: CudaWorkerId,
        endpoint_name: String,
        stage_id: Option<String>,
        init_data: Box<dyn WorkerInput>,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>,
    },
    RegisterStage {
        worker_id: CudaWorkerId,
        key: EndpointKey,
        init_data: Box<dyn WorkerInput>,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<Arc<EndpointRegistration>, ThoError>>,
    },
    EnsureStage {
        worker_id: CudaWorkerId,
        registration: Arc<EndpointRegistration>,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<(), ThoError>>,
    },
    DiscardStageState {
        worker_id: CudaWorkerId,
        endpoint_name: String,
        stage_id: Option<String>,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>,
    },
    ResetStageState {
        worker_id: CudaWorkerId,
        endpoint_name: String,
        stage_id: Option<String>,
        info: StageStateInfo,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>,
    },
    DebugWorker {
        worker_id: CudaWorkerId,
        activity: thoosa_shared::CudaActivityScope,
        respond_to: oneshot::Sender<Result<CudaWorkerMsgOut, ThoError>>,
    },
}

// Recovery and shutdown never compete with ordinary work for queue capacity.
#[derive(Debug)]
enum BackendControl {
    ReleaseEndpoint { stage_id: Option<String>, endpoint: String, worker_id: CudaWorkerId },
    ReleaseTensor { key: u64, worker_id: CudaWorkerId },
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
    endpoints: EndpointRegistry,
    endpoint_events: mpsc::UnboundedReceiver<EndpointEvent>,
    endpoint_events_tx: mpsc::UnboundedSender<EndpointEvent>,
    spawned_state: Option<SpawnedState>,
    retirement_error: Option<ThoError>,
    next_id: Arc<AtomicU64>,
    tho_device_idx: usize,
    cuda_worker_args: Arc<Vec<String>>,
    // Keep an actor-owned descriptor so a reset can map the same parent allocator into the replacement worker.
    arena_fd: OwnedFd,
    release: Option<TensorReleaseHandle>,
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
    debug!("cuda_process::kill_and_reap: child={child:?}");

    if child.try_wait()?.is_none() {
        debug!("cuda_process::kill_and_reap: start_kill");

        child.start_kill().context("terminate CUDA worker")?;
        timeout(WORKER_REAP_TIMEOUT, child.wait()).await.context("timed out reaping CUDA worker")?.context("reap CUDA worker")?;

        debug!("cuda_process::kill_and_reap: start_kill done");
    }
    debug!("cuda_process::kill_and_reap: done");
    Ok(())
}

impl SpawnedState {
    async fn stop_io(&mut self) {
        debug!("SpawnedState::stop_io");

        self.cancel.cancel();

        debug!("SpawnedState::stop_io: cancel ok");

        self.reader_handle.abort();
        self.writer_handle.abort();

        let _ = (&mut self.reader_handle).await;
        debug!("SpawnedState::stop_io: reader_handle ok");

        let _ = (&mut self.writer_handle).await;
        debug!("SpawnedState::stop_io: writer_handle ok");
    }

    async fn retire(mut self, reason: ThoError) -> Result<(), ThoError> {
        debug!("SpawnedState::retire: reason={reason:?}");
        self.pending.close(reason.clone());
        self.stop_io().await;
        debug!("SpawnedState::retire: stop_io done");

        kill_and_reap(&mut self.child).await?;
        self.pending.exited(reason);

        debug!("SpawnedState::retire: done");

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
            debug!("shutdown_with_timeout: reply={reply:?}");
            unit_reply(reply)?;
            // The acknowledgement precedes process exit; keep shared buffers
            // alive until the worker has actually stopped accessing them.
            let status = self.child.wait().await?;
            debug!("shutdown_with_timeout: status={status:?}");
            if !status.success() { return Err(ThoError::new(format!("CUDA worker exited with {status}"))); }
            Ok::<_, ThoError>(())
        }).await;
        self.stop_io().await;
        // A failed or timed-out graceful shutdown still has to reap the child.
        kill_and_reap(&mut self.child).await?;
        self.pending.exited(ThoError::new("CUDA worker exited during shutdown"));
        graceful.context("CUDA worker shutdown timed out")?
    }
}

impl Drop for SpawnedState {
    fn drop(&mut self) {
        //TODO this can be skipped when shutting down
        self.pending.close(ThoError::CudaBeingRecovered);
        self.cancel.cancel();
        self.reader_handle.abort();
        self.writer_handle.abort();
        // Normal paths have already reaped. On task/runtime cancellation the
        // Child's kill_on_drop terminates it; the registry quarantines buffers
        // unless exit can be confirmed here.
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            self.pending.exited(ThoError::CudaBeingRecovered);
        } else {
            self.pending.quarantine(ThoError::CudaBeingRecovered.context("CUDA worker retirement interrupted before confirmed exit"));
        }
    }
}

fn unit_reply(response: CudaWorkerMsgOut) -> Result<(), ThoError> {
    match response.status { OpResult::Ok(_) => Ok(()), OpResult::Failed(error) => Err(error.into()) }
}

impl CudaIPCActor {
    async fn new(gate: Arc<thoosa_shared::CudaActivityGate>, cuda_worker_args: Vec<String>, receiver: mpsc::Receiver<CudaIPCActorMessage>, control: mpsc::UnboundedReceiver<BackendControl>, arena_fd: OwnedFd, release: Option<TensorReleaseHandle>) -> Result<Self, ThoError> {
        let tho_device_idx = gate.device_idx();
        let cuda_worker_args = Arc::new(cuda_worker_args);

        let spawned_state = Self::spawn_worker_and_loops(CudaWorkerId { device_idx: tho_device_idx, generation: gate.device_generation() }, &cuda_worker_args, arena_fd.as_fd(), release.clone()).await?;


        debug!("CudaIPCActor::new: worker device_info={:?}", spawned_state.device_info);

        let (endpoint_events_tx, endpoint_events) = mpsc::unbounded_channel();
        Ok(Self {
            gate,
            receiver,
            endpoints: EndpointRegistry::new(endpoint_events_tx.clone()),
            endpoint_events,
            endpoint_events_tx,
            spawned_state: Some(spawned_state),
            retirement_error: None,
            release: release.clone(),
            control,
            next_id: Arc::new(AtomicU64::new(1)),
            tho_device_idx,
            cuda_worker_args,
            arena_fd,
        })
    }

    async fn spawn_worker_and_loops(worker_id: CudaWorkerId, cuda_worker_args: &Vec<String>, arena_fd: std::os::fd::BorrowedFd<'_>, release: Option<TensorReleaseHandle>) -> Result<SpawnedState, ThoError> {
        info!("CudaIPCActor::spawn_worker_and_loops: cuda_worker_args={cuda_worker_args:?}");

        let (child, stream, device_info) = Self::spawn_worker_for_cuda_device(worker_id, cuda_worker_args, arena_fd).await?;

        info!("CudaIPCActor::spawn_worker_and_loops: spawn_worker_for_cuda_device ok");

        let (reader, writer) = stream.into_split();

        let (tx, rx): (Sender<CudaWorkerMsgIn>, Receiver<CudaWorkerMsgIn>) = mpsc::channel::<CudaWorkerMsgIn>(2048);

        let pending = Arc::new(PendingRequests::new(worker_id));
        let cancel = CancellationToken::new();
        let writer_handle = tokio::spawn(writer_loop(writer, rx, Arc::clone(&pending), cancel.child_token()));
        let reader_handle = tokio::spawn(reader_loop(reader, Arc::clone(&pending), cancel.child_token(), release));

        info!("CudaIPCActor::spawn_worker_and_loops: ok");

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

        debug!("CudaIPCActor::spawn_worker_for_cuda_device: exe={exe:?}");

        let dir = tempfile::tempdir()?;
        let sock_path = dir.path().join(format!("worker_{cuda_device}.sock"));

        debug!("CudaIPCActor::spawn_worker_for_cuda_device: dir={dir:?} sock_path={sock_path:?}");

        let listener = UnixListener::bind(&sock_path)?;

        debug!("CudaIPCActor::spawn_worker_for_cuda_device: listener={listener:?}");

        let mut command = &mut Command::new(exe);

        for arg in args {
            command = command.arg(arg);
        }

        debug!("CudaIPCActor::spawn_worker_for_cuda_device command={command:?}");

        let worker_id_str = serde_json::to_string(&worker_id)?;
        debug!("CudaIPCActor::spawn_worker_for_cuda_device: worker_id_str={worker_id_str}");

        let mut child = command
            // Each child sees its assigned physical GPU as CUDA ordinal zero.
            .env(CUDA_VISIBLE_DEVICES, cuda_device.to_string())
            .kill_on_drop(true)
            .env(IPC_SOCK_ENV, &sock_path)
            .env(WORKER_ID_ENV, worker_id_str)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        debug!("CudaIPCActor::spawn_worker_for_cuda_device: child={child:?}");

        let ready = timeout(WORKER_READY_TIMEOUT, async {
            debug!("CudaIPCActor::spawn_worker_for_cuda_device: before accept");

            let accept_r = listener.accept().await;

            debug!("CudaIPCActor::spawn_worker_for_cuda_device: accept_r={accept_r:?}");

            let (mut stream, _) = accept_r?;
            // Send the arena first. Readiness is acknowledged only after all
            // stream/copy pools and the shared mapping have been initialized.
            debug!("CudaIPCActor::spawn_worker_for_cuda_device: before send_fd");
            send_fd(&stream, arena_fd).await?;
            let device_info: DeviceInfo = PostcardToFromBytes::from_bytes(&read_frame(&mut stream, "DeviceInfo/main", MAX_FRAME_LEN).await?)?;

            debug!("CudaIPCActor::spawn_worker_for_cuda_device: device_info={device_info:?}");

            device_info.worker_id.check(worker_id)?;
            Ok::<_, ThoError>((stream, device_info))
        }).await;


        debug!("CudaIPCActor::spawn_worker_for_cuda_device: ready={ready:?}");

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
        debug!("CudaIPCActor::handle_message: msg={msg:?}");

        // Retirement can consume the worker before a queued message is handled.
        // Resolve its response without borrowing a missing transport.
        if self.spawned_state.is_none() {
            Self::reject(msg, ThoError::CudaBeingRecovered);
            return Ok(());
        }
        let (worker_id, activity) = match &msg {
            CudaIPCActorMessage::Copy { src, activity, .. } => (src.ref_worker_id(), activity),
            CudaIPCActorMessage::PurgeStage { worker_id, activity, .. }
            | CudaIPCActorMessage::InvokeStage { worker_id, activity, .. }
            | CudaIPCActorMessage::InstallStage { worker_id, activity, .. }
            | CudaIPCActorMessage::RegisterStage { worker_id, activity, .. }
            | CudaIPCActorMessage::EnsureStage { worker_id, activity, .. }
            | CudaIPCActorMessage::DiscardStageState { worker_id, activity, .. }
            | CudaIPCActorMessage::ResetStageState { worker_id, activity, .. }
            | CudaIPCActorMessage::DebugWorker { worker_id, activity, .. } => (Some(*worker_id), activity),
        };
        // Recheck after queueing: a request or tensor key from a retired
        // worker must never be interpreted by its replacement. CPU uploads
        // have no worker reference, so their activity supplies the generation.
        let identity = worker_id.map_or(Ok(()), |worker_id| worker_id.check(self.worker().worker_id));
        // The lease must belong to this gate and generation, with valid
        // recovery authority if applicable. Already admitted work may drain
        // while paused, so checking is_paused here would reject valid work.
        if let Err(error) = identity.and_then(|_| activity.check_operation(&self.gate)) {
            Self::reject(msg, error);
            return Ok(());
        }

        let (worker_id, operation, activity, respond_to) = match msg {
            CudaIPCActorMessage::RegisterStage { key, init_data, respond_to, .. } => {
                let result = self.endpoints.register(self.tho_device_idx, key, init_data);
                let _ = respond_to.send(result);
                return Ok(());
            }
            CudaIPCActorMessage::EnsureStage { worker_id, registration, activity, respond_to } => {
                if !self.endpoints.contains(&registration) {
                    let _ = respond_to.send(Err(ThoError::new("endpoint registration was retired")));
                    return Ok(());
                }
                let tx = self.worker().tx.clone();
                let pending = Arc::clone(&self.worker().pending);
                let next_id = Arc::clone(&self.next_id);
                // Own installation through its acknowledgement, even if its caller
                // is cancelled. Commit before releasing activity or publishing success.
                tokio::spawn(activity.run(async move {
                    let result = async {
                        let mut installed = registration.slot.lock().await;
                        if *installed == Some((registration.token, worker_id)) { return Ok(()); }
                        if let Some((_, previous)) = *installed {
                            if previous == worker_id {
                                let operation = Operation::DropStage { endpoint_name: registration.key.endpoint_name.clone(), stage_id: registration.key.stage_id.clone() };
                                unit_reply(send_request_and_wait_msg(tx.clone(), Arc::clone(&pending), CudaWorkerMsgIn::new(worker_id, next_id.fetch_add(1, Ordering::Relaxed), operation)).await?)?;
                            }
                            *installed = None;
                        }
                        let operation = Operation::InstallStage { endpoint_name: registration.key.endpoint_name.clone(), stage_id: registration.key.stage_id.clone(), init_data: registration.init_data()? };
                        unit_reply(send_request_and_wait_msg(tx, pending, CudaWorkerMsgIn::new(worker_id, next_id.fetch_add(1, Ordering::Relaxed), operation)).await?)?;
                        *installed = Some((registration.token, worker_id));
                        registration.installed(worker_id);
                        Ok::<_, ThoError>(())
                    }.await;
                    let _ = respond_to.send(result);
                }));
                return Ok(());
            }
            CudaIPCActorMessage::Copy { src, dst_device, activity, respond_to, dst_key } => {
                return if src.ref_key().is_some() {
                    // Worker references can only be downloaded; concrete CPU
                    // storage can only be uploaded by this IPC copy path.
                    assert_eq!(*dst_device, Cpu, "If source is tensor ref and thus live on device it can be copied only to CPU");
                    self.handle_ref_d2h_copy(src, dst_device, activity, respond_to)
                } else {
                    debug!("CudaIPCActor::handle_message: src={src:?} dst_device={dst_device:?}");
                    assert_ne!(*dst_device, Cpu, "If source is concrete tensor it should be on CPU and can be copied only to GPU");
                    self.handle_h2d_copy(src, dst_device, activity, respond_to, dst_key)
                }
            }
            CudaIPCActorMessage::PurgeStage { worker_id, endpoint_name, stage_id, activity, respond_to } => {
                let pending = Arc::clone(&self.worker().pending);
                let next_id = Arc::clone(&self.next_id);
                let tx = self.worker().tx.clone();
                let gate = Arc::clone(&self.gate);
                tokio::spawn(activity.run(async move {
                    Self::export_stage(worker_id, endpoint_name, stage_id, next_id, pending, tx, gate, respond_to).await;
                }));
                return Ok(());
            }
            CudaIPCActorMessage::InvokeStage { worker_id, endpoint_name, stage_id, input, consume, activity, respond_to } =>
                (worker_id, Operation::InvokeStage { endpoint_name, stage_id, input, consume }, activity, respond_to),
            CudaIPCActorMessage::InstallStage { worker_id, endpoint_name, stage_id, init_data, activity, respond_to } =>
                (worker_id, Operation::InstallStage { endpoint_name, stage_id, init_data }, activity, respond_to),
            CudaIPCActorMessage::DiscardStageState { worker_id, endpoint_name, stage_id, activity, respond_to } =>
                (worker_id, Operation::PurgeStage { endpoint_name, stage_id }, activity, respond_to),
            CudaIPCActorMessage::ResetStageState { worker_id, endpoint_name, stage_id, info, activity, respond_to } =>
                (worker_id, Operation::ResetStageState { endpoint_name, stage_id, info }, activity, respond_to),
            CudaIPCActorMessage::DebugWorker { worker_id, activity, respond_to } =>
                (worker_id, Operation::Debug, activity, respond_to),
        };
        // All single-reply operations share request IDs, transport ownership and
        // activity lifetime, including when the caller cancels after enqueueing.
        let input = CudaWorkerMsgIn::new(worker_id, self.next_id.fetch_add(1, Ordering::Relaxed), operation);
        let pending = Arc::clone(&self.worker().pending);
        let tx = self.worker().tx.clone();
        tokio::spawn(activity.run(async move {
            // Stage failures belong to SyncStageRunner::submit_observed.
            // Forward the original reply so the admitted job observes it
            // before initiating recovery, even if its submitter cancelled.
            let response = send_request_and_wait_msg(tx, pending, input).await;
            let _ = respond_to.send(response);
        }));
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

    fn handle_ref_d2h_copy(&mut self, source: ThoTensor, dst_device: ThoDevice, activity: thoosa_shared::CudaActivityScope, respond_to: oneshot::Sender<Result<IpcCopyResult, ThoError>>) -> Result<(), ThoError> {
        // A worker reference must be materialized in parent CPU storage.
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

                Self::do_ref_d2h_copy(worker_id, transfer_id, pending, tx, info, None).await.map(IpcCopyResult::Tensor)
            }.await;
            let _ = respond_to.send(crate::copy_ctx::report_cuda_copy_result(result, &gate, generation));
        }));
        Ok(())
    }

    // The actor owns the transaction even when its caller disappears. Preparing
    // only freezes/copies; the proxy authorizes purge after storing the CPU dump.
    async fn export_stage(worker_id: CudaWorkerId, endpoint_name: String, stage_id: Option<String>, next_id: Arc<AtomicU64>, pending: Arc<PendingRequests>, tx: Sender<CudaWorkerMsgIn>, gate: Arc<thoosa_shared::CudaActivityGate>, respond_to: oneshot::Sender<Result<StageStateDump, ThoError>>) {
        use futures_util::FutureExt;
        let export_id = next_id.fetch_add(1, Ordering::Relaxed);
        let mut begun = false;
        let prepare = async {
            let response = send_request_and_wait_msg(tx.clone(), Arc::clone(&pending), CudaWorkerMsgIn::new(worker_id, export_id, Operation::BeginStageStatePurge { endpoint_name: endpoint_name.clone(), stage_id: stage_id.clone() })).await?;
            if matches!(response.status, OpResult::Ok(_)) { begun = true; }
            let info = match response.status {
                OpResult::Failed(error) => return Err(error.into()),
                OpResult::Ok(Some(value)) => (value as Box<dyn Any>).downcast::<StageStateInfo>().map_err(|_| ThoError::new("invalid stage export description"))?,
                OpResult::Ok(None) => return Err(ThoError::new("missing stage export description")),
            };
            let mut tensors = Vec::with_capacity(info.tensors.len());
            let mut destinations = Vec::with_capacity(info.tensors.len());
            for tensor_info in info.tensors {
                let key = tensor_info.tensor_key.to_tensor_key();
                let tensor = tch_result(Tensor::f_empty(tensor_info.shape.as_slice(), (tensor_info.kind.into(), Cpu)))?;
                let arena_offset = parent_shared_arena_offset(&tensor, tensor_info.tensor_byte_len, "state D2H destination")?;
                destinations.push(crate::ipc::worker_msg::StageStatePurgeDestination { info: tensor_info, arena_offset });
                tensors.push((key, tensor));
            }
            // One request takes the worker export mutex once and coalesces its
            // tensors into reusable pinned staging buffers. The pending registry
            // retains ALL destinations until reply or confirmed worker exit.
            let tensors = Arc::new(parking_lot::Mutex::new(tensors));
            if !destinations.is_empty() {
                let request = CudaWorkerMsgIn::new(worker_id, next_id.fetch_add(1, Ordering::Relaxed), Operation::ExportStageTensors {
                    endpoint_name: endpoint_name.clone(), stage_id: stage_id.clone(), export_id, destinations,
                });
                unit_reply(send_request(tx.clone(), Arc::clone(&pending), request, Some(Box::new(Arc::clone(&tensors))), false).await?)?;
            }
            let tensors = Arc::try_unwrap(tensors).map_err(|_| ThoError::new("completed state D2H storage is still leased"))?.into_inner();
            let tensors = tensors.into_iter().map(|(key, tensor)| Ok((key, ThoTensor::new((), tensor)?))).collect::<Result<_, ThoError>>()?;
            Ok(StageStateDump::new(info.scalars, tensors))
        };
        // Catch polling panics as well as ordinary errors so every accepted actor
        // operation resolves its sender and attempts to release the worker lease.
        let prepared = std::panic::AssertUnwindSafe(prepare).catch_unwind().await.unwrap_or_else(|_| Err(ThoError::new("stage export preparation panicked")));
        let finish = |commit| send_request_and_wait_msg(tx.clone(), Arc::clone(&pending), CudaWorkerMsgIn::new(worker_id, next_id.fetch_add(1, Ordering::Relaxed), Operation::FinishStageStatePurge { endpoint_name: endpoint_name.clone(), stage_id: stage_id.clone(), export_id, commit }));
        match prepared {
            Err(error) => {
                if begun {
                    let cleanup = finish(false).await.and_then(unit_reply);
                    // Request errors do not break transport. The worker releases
                    // its transaction lease even when finishing fails.
                    if let Err(cleanup_error) = cleanup {
                        let error = crate::copy_ctx::finish_cuda_error(error, Err(cleanup_error));
                        let _ = respond_to.send(crate::copy_ctx::report_cuda_copy_result(Err(error), &gate, worker_id.generation));
                        return;
                    }
                }
                let _ = respond_to.send(crate::copy_ctx::report_cuda_copy_result(Err(error), &gate, worker_id.generation));
            }
            Ok(mut dump) => {
                let (decision, decided) = oneshot::channel();
                let (completed, completion) = oneshot::channel();
                dump.export = Some(crate::ipc::proxy::PreparedStageStatePurge { decision, completed: completion });
                // A dropped response or dump closes decided, selecting abort.
                let _ = respond_to.send(Ok(dump));
                let commit = tokio::select! {
                    decision = decided => decision.unwrap_or(false),
                    _ = pending.failed.cancelled() => {
                        let _ = completed.send(Err(pending.failure()));
                        return;
                    }
                };
                let result = finish(commit).await.and_then(unit_reply);
                let result = crate::copy_ctx::report_cuda_copy_result(result, &gate, worker_id.generation);
                let _ = completed.send(result);
            }
        }
    }

    async fn do_ref_d2h_copy(worker_id: CudaWorkerId, transfer_id: u64, pending: Arc<PendingRequests>, tx: Sender<CudaWorkerMsgIn>, info: WorkerTensorInfoPayload, export_id: Option<u64>) -> Result<ThoTensor, ThoError> {


        let destination = tch_result(Tensor::f_empty(info.shape.as_slice(), (info.kind.into(), Cpu)))?;
        let arena_offset = parent_shared_arena_offset(&destination, info.tensor_byte_len, "D2H destination")?;
        let destination = Arc::new(parking_lot::Mutex::new(destination));
        // The registry retains this allocation even if this task aborts
        // or transport fails while the worker is still writing to it.
        let response = send_request(tx, pending, CudaWorkerMsgIn::new(worker_id, transfer_id, Operation::D2hCopy {
            export_id,
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
    fn handle_h2d_copy(&mut self, src: ThoTensor, dst_device: ThoDevice, activity: thoosa_shared::CudaActivityScope, respond_to: oneshot::Sender<Result<IpcCopyResult, ThoError>>, dst_key: Option<H2DTensorKey>) -> Result<(), ThoError> {
        // Uploading through another device's actor would assign the tensor to
        // the wrong worker even though the activity lease itself is valid.
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
        let src_tensor = src.into_local()?;

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
                            tensor_key: H2DTensorKey::New,
                            kind: src_kind.into(),
                            shape: src_shape.clone(),
                            tensor_byte_len,
                            arena_offset,
                            dst_key: dst_key.clone(),
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
                match (dst_key, done.tensor_ref) {
                    (Some(H2DTensorKey::StageState { endpoint_name, stage_id, tensor_id }), TensorKey::StageState { endpoint_name: returned_endpoint, stage_id: returned_stage, tensor_key })
                        if endpoint_name == returned_endpoint && stage_id == returned_stage && tensor_id == tensor_key => Ok(IpcCopyResult::StateInstalled),
                    (None | Some(H2DTensorKey::New), TensorKey::Data(tensor_ref)) => {
                        // Only ordinary data copies own a worker tensor reference.
                        Ok(IpcCopyResult::Tensor(ThoTensor::new_ref((), tensor_ref, src_shape, src_kind, None, worker_id)))
                    }
                    (expected, actual) => Err(ThoError::new(format!("worker H2D returned {actual:?} for destination {expected:?}"))),
                }
            }.await;
            let _ = respond_to.send(crate::copy_ctx::report_cuda_copy_result(result, &gate, generation));
        }));

        Ok(())
    }

    fn worker(&self) -> &SpawnedState { self.spawned_state.as_ref().expect("admitted work requires a live worker") }

    fn reject(msg: CudaIPCActorMessage, error: ThoError) {
        match msg {
            CudaIPCActorMessage::Copy { respond_to, .. } => { let _ = respond_to.send(Err(error)); }
            CudaIPCActorMessage::RegisterStage { respond_to, .. } => { let _ = respond_to.send(Err(error)); }
            CudaIPCActorMessage::EnsureStage { respond_to, .. } => { let _ = respond_to.send(Err(error)); }
            CudaIPCActorMessage::InvokeStage { respond_to, .. }
            | CudaIPCActorMessage::InstallStage { respond_to, .. }
            | CudaIPCActorMessage::DiscardStageState { respond_to, .. }
            | CudaIPCActorMessage::ResetStageState { respond_to, .. }
            | CudaIPCActorMessage::DebugWorker { respond_to, .. } => { let _ = respond_to.send(Err(error)); }
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
        debug!("CudaIPCActor::control: control={control:?}");
        match control {
            BackendControl::ReleaseEndpoint { stage_id, endpoint, worker_id } => {
                if let Some(worker) = &self.spawned_state {
                    if worker.worker_id == worker_id {
                        let request = CudaWorkerMsgIn::new(worker_id, self.next_id.fetch_add(1, Ordering::Relaxed), Operation::DropStage { endpoint_name: endpoint, stage_id });
                        let result = async {
                            let permit = worker.tx.reserve().await.context("worker writer closed")?;
                            if self.gate.is_paused() { return Ok(()); }//restart is in progress
                            let (reply, _response) = oneshot::channel();
                            worker.pending.enqueue(permit, request, reply, None, false)
                        }.await;
                        if let Err(error) = result { warn!("endpoint release failed: {error:#}"); }
                    }
                }
            }
            BackendControl::ReleaseTensor { key, worker_id } => {
                // Recheck after queueing: recovery may have started since Drop.
                // Cache invalidation precedes retirement, so a worker can still
                // be present even though its tensors will be reclaimed at exit.
                if self.gate.is_paused() { return false; }
                if let Some(worker) = &self.spawned_state {
                    if worker.worker_id == worker_id {
                        let request = CudaWorkerMsgIn::new(worker_id, self.next_id.fetch_add(1, Ordering::Relaxed), Operation::DropThoTensor { tensor_key: key });
                        // Queue before later shutdown, but do not wait for CUDA:
                        // a hung worker must still be interruptible by retirement.
                        let result = async {
                            let permit = worker.tx.reserve().await.context("worker writer closed")?;
                            // The writer may have been full when recovery began.
                            if self.gate.is_paused() { return Ok(()); }//restart is in progress
                            let (reply, _response) = oneshot::channel();
                            worker.pending.enqueue(permit, request, reply, None, false)
                        }.await;
                        if let Err(error) = result { warn!("tensor release failed: {error:#}"); }
                    }
                }
            }
            BackendControl::Retire { authority, respond_to } => {//invoked from CudaRecoveryCoordinator::run
                // A queued recovery command may outlive its coordinator cycle.
                let result = match self.gate.check_recovery_authority(authority, false) {
                    Ok(()) => self.retire(ThoError::CudaBeingRecovered).await,
                    Err(error) => Err(error),
                };
                let _ = respond_to.send(result);
            }
            BackendControl::Restart { authority, respond_to } => {//invoked from CudaRecoveryCoordinator::run
                let result = async {
                    // Recheck queued authority and drained activity before replacing CUDA state.
                    self.gate.check_recovery_authority(authority, true)?;
                    debug!("CudaIPCActor::control, BackendControl::Restart: check_recovery_authority ok");
                    // Fatal CUDA failures may prevent graceful shutdown. Reap
                    // the failed worker and propagate retirement errors before
                    // creating another process for the same device.
                    // The coordinator sends Retire only after a drain timeout.
                    // Normally this is the first retirement; after a successful
                    // early retirement, spawned_state is None and this is a no-op.
                    self.retire(ThoError::CudaBeingRecovered).await?;

                    // A fresh primary context requires the old process to have
                    // exited before its replacement is even spawned.
                    let worker_id = CudaWorkerId { device_idx: self.tho_device_idx, generation: thoosa_shared::CudaDeviceGeneration(self.gate.device_generation().0.checked_add(1).expect("worker generation overflow")) };
                    let replacement = Self::spawn_worker_and_loops(worker_id, &self.cuda_worker_args, self.arena_fd.as_fd(), self.release.clone()).await?;
                    // Spawning awaits readiness; do not publish a replacement
                    // if recovery authority was revoked during that wait.
                    if let Err(error) = self.gate.check_recovery_authority(authority, true) {
                        warn!("CudaIPCActor::control: gate.check_recovery_authority error={error:?}");

                        replacement.retire(error.clone()).await?;
                        return Err(error);
                    }
                    self.spawned_state = Some(replacement);
                    Ok(())
                }.await;

                info!("CudaIPCActor::control: BackendControl::Restart done");
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

    fn endpoint_event(&mut self, event: EndpointEvent) {
        match event {
            EndpointEvent::Forget { key, token } => self.endpoints.forget(&key, token),
            EndpointEvent::Release { key, token, slot } => {
                let worker = self.spawned_state.as_ref().map(|worker| (worker.worker_id, worker.tx.clone(), Arc::clone(&worker.pending)));
                let next_id = Arc::clone(&self.next_id);
                let events = self.endpoint_events_tx.clone();
                // Cleanup has no CUDA admission dependency. A slot lock orders
                // deletion with reuse of the same logical key without blocking recovery.
                tokio::spawn(async move {
                    let result = async {
                        let mut installed = slot.lock().await;
                        if let Some((owner, previous)) = *installed {
                            if owner == token {
                                if let Some((worker_id, tx, pending)) = worker {
                                    if previous == worker_id {
                                        let operation = Operation::DropStage { endpoint_name: key.endpoint_name.clone(), stage_id: key.stage_id.clone() };
                                        unit_reply(send_request_and_wait_msg(tx, pending, CudaWorkerMsgIn::new(worker_id, next_id.fetch_add(1, Ordering::Relaxed), operation)).await?)?;
                                    }
                                }
                                *installed = None;
                            }
                        }
                        Ok::<_, ThoError>(())
                    }.await;
                    if result.is_ok() {
                        let _ = events.send(EndpointEvent::Forget { key, token });
                    }
                    // On transport failure keep the slot until reuse or actor
                    // shutdown; a new owner must not overtake uncertain deletion.
                });
            }
        }
    }

    async fn run(mut actor: CudaIPCActor) -> Result<(), ThoError> {
        loop {
            let failed = actor.spawned_state.as_ref().map(|worker| worker.pending.failed.clone());

            tokio::select! {
                biased;
                Some(control) = actor.control.recv() => {
                    debug!("CudaIPCActor::run: control={control:?}");

                    if actor.control(control).await {
                        return Ok(());
                    }

                    debug!("CudaIPCActor::run: control not ok");
                }
                Some(event) = actor.endpoint_events.recv() => actor.endpoint_event(event),
                _v = async {
                    match failed {
                        Some(failed) => {
                            debug!("CudaIPCActor::run: failed={failed:?}");
                            failed
                        }.cancelled().await,
                        None => std::future::pending().await
                    }
                } => {
                    debug!("CudaIPCActor::run: _v={_v:?}");

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
                    // A broken transport cannot serve or safely release pending
                    // requests. Replacing this IPC worker is the last resort,
                    // independent of the original error's CUDA classification.
                    let cause = error.clone().context("CUDA worker transport is unavailable");
                    debug!("CudaIPCActor::run: cause={cause:?} is_installed={}", GlobalCtx::is_installed());

                    if GlobalCtx::is_installed() {
                        let _ = GlobalCtx::global().get_runner(actor.gate.device_idx()).request_cuda_reset_for_generation(generation, cause);
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
    warn!("ensure_parent_shared_arena_tensor: got tensor created out of arena: source={source:?}");
    let mut shared_source = tch_result(Tensor::f_empty(source.size().as_slice(), (source.kind(), Cpu)))?;
    tch_result(shared_source.f_copy_(&source))?;
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

        // Async proxy callers can disappear while the worker runs. Register
        // ownership before publishing the frame so even an immediate reply
        // cannot race retention of its tensor references and initializer data.
        if pending.retain_until_reply(request_id, Box::new(msg)).is_err() { break; }

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
    release: Option<TensorReleaseHandle>,
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
                        warn!("worker IPC reader stopped: {e}");

                        pending.transport_failed(ThoError::new(format!("worker IPC reader stopped: {e}")));

                        break;
                    }
                }
            }
        };

        match match &release { Some(owner) => owner.decoding(|| PostcardToFromBytes::from_bytes::<CudaWorkerMsgOut>(&frame)), None => PostcardToFromBytes::from_bytes::<CudaWorkerMsgOut>(&frame) } {
            Ok(msg) => {
                debug!("reader_loop: msg={msg:?}");
                if let Err(error) = pending.complete(msg) {
                    warn!("reader_loop: error={error:?}");
                    pending.transport_failed(error);
                    break;
                }
            }
            Err(e) => {
                warn!("reader_loop: e={e:?}");

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

/// A narrow release capability retained by remote resources. Drop only queues
/// work for a running worker; recovery reclaims its tensors by retiring it.
/// The owning actor drains ordinary releases before its queued shutdown command.
#[derive(Debug, Clone)]
pub struct TensorReleaseHandle { control: mpsc::UnboundedSender<BackendControl>, gate: Arc<thoosa_shared::CudaActivityGate>, pub(crate) owner_id: u64 }
impl TensorReleaseHandle {
    pub(crate) fn release(&self, key: u64, worker_id: CudaWorkerId) {
        // Recovery pauses this gate before dropping cached tensors. Avoid even
        // queueing their releases, and discard late drops from retired workers.
        if self.gate.is_paused() || worker_id.device_idx != self.gate.device_idx() || worker_id.generation != self.gate.device_generation() { return; }
        let _ = self.control.send(BackendControl::ReleaseTensor { key, worker_id });
    }
    pub(crate) fn decoding<T>(&self, f: impl FnOnce() -> T) -> T {
        struct Restore(Option<TensorReleaseHandle>);
        impl Drop for Restore { fn drop(&mut self) { DECODING_RELEASE.with(|slot| *slot.borrow_mut() = self.0.take()); } }
        let previous = DECODING_RELEASE.with(|slot| slot.replace(Some(self.clone())));
        let _restore = Restore(previous);
        f()
    }
    pub(crate) fn current() -> Option<Self> { DECODING_RELEASE.with(|slot| slot.borrow().clone()) }
}
thread_local! { static DECODING_RELEASE: std::cell::RefCell<Option<TensorReleaseHandle>> = const { std::cell::RefCell::new(None) }; }

#[derive(Debug, Clone)]
pub struct CudaIPCActorHandle {
    gate: Arc<thoosa_shared::CudaActivityGate>,
    tho_device_idx: usize,
    sender: mpsc::Sender<CudaIPCActorMessage>,
    control: mpsc::UnboundedSender<BackendControl>,
    release_id: u64,
}

///executes cuda actions in a separate process started from CudaWorker::run_worker
impl CudaIPCActorHandle {
    pub(crate) async fn purge_stage_state(&self, stage_id: Option<String>, endpoint_name: String, worker_id: CudaWorkerId) -> Result<StageStateDump, ThoError> {
        debug!("CudaIPCActorHandle::purge_stage_state: stage_id={stage_id:?} endpoint_name={endpoint_name:?} worker_id={worker_id:?} self.gate.device_generation={:?}, self.gate.is_paused={:?}", self.gate.device_generation(), self.gate.is_paused());
        // A purge must export live state; a retired or paused worker cannot
        // supply a usable backup. Keep the adjuster failure distinct from a
        // destructor no-op, then let activity_for close the admission race.
        if !self.is_current_worker(worker_id) || self.gate.is_paused() {
            return Err(ThoError::CudaBeingRecovered);
        }
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::PurgeStage {
                worker_id,
                endpoint_name,
                stage_id,
                activity: self.activity_for(worker_id)?,

                respond_to,
            }
        ).instrument(tracing::debug_span!(
            "CudaIPCActorHandle::purge_endpoint_bytes",
            agg_perf="CudaIPCActorHandle::purge_endpoint_bytes",
        )).await.context("CUDA actor channel closed")?;

        recv.await.context("CUDA actor response channel closed")?
    }

    pub(crate) async fn retire_backend(&self, authority: CudaActivityPauseTicket) -> Result<(), ThoError> {
        // Retirement needs the current recovery ticket but may precede draining:
        // killing a failed worker can be necessary to release pending activity.
        self.gate.check_recovery_authority(authority, false)?;
        let (respond_to, recv) = oneshot::channel();
        self.control.send(BackendControl::Retire { authority, respond_to }).context("CUDA actor control channel closed")?;
        recv.await.context("CUDA retirement response channel closed")?
    }

    pub(crate) fn release_endpoint(&self, stage_id: Option<String>, endpoint: String, worker_id: CudaWorkerId) {
        // Cleanup and shutdown share FIFO submission; Drop needs no Tokio context.
        let _ = self.control.send(BackendControl::ReleaseEndpoint { stage_id, endpoint, worker_id });
    }

    pub(crate) fn activity_gate(&self) -> Arc<thoosa_shared::CudaActivityGate> { Arc::clone(&self.gate) }
    pub fn tensor_releaser(&self) -> TensorReleaseHandle { TensorReleaseHandle { control: self.control.clone(), gate: Arc::clone(&self.gate), owner_id: self.release_id } }
    pub(crate) async fn register_stage(&self, worker_id: CudaWorkerId, endpoint_name: String, stage_id: Option<String>, init_data: Box<dyn WorkerInput>) -> Result<Arc<EndpointRegistration>, ThoError> {
        let (respond_to, response) = oneshot::channel();
        self.sender.send(CudaIPCActorMessage::RegisterStage { worker_id, key: EndpointKey { endpoint_name, stage_id }, init_data, activity: self.activity_for(worker_id)?, respond_to }).await.context("CUDA actor channel closed")?;
        response.await.context("CUDA registration response channel closed")?
    }

    pub(crate) async fn ensure_stage(&self, registration: Arc<EndpointRegistration>, worker_id: CudaWorkerId) -> Result<(), ThoError> {
        let (respond_to, response) = oneshot::channel();
        self.sender.send(CudaIPCActorMessage::EnsureStage { registration, worker_id, activity: self.activity_for(worker_id)?, respond_to }).await.context("CUDA actor channel closed")?;
        response.await.context("CUDA installation response channel closed")?
    }

    pub fn worker_id(&self) -> Result<CudaWorkerId, ThoError> {
        // Obtain the generation through admission so callers cannot acquire a
        // usable worker identity during failed recovery or without a valid scope.
        let activity = thoosa_shared::CudaActivityScope::for_operation(&self.gate)?;
        Ok(CudaWorkerId { device_idx: self.tho_device_idx, generation: activity.generation(&self.gate).expect("worker lease") })
    }

    pub(crate) fn activity_for(&self, worker_id: CudaWorkerId) -> Result<thoosa_shared::CudaActivityScope, ThoError> {
        // Admit before awaiting queue capacity. The message then owns this lease,
        // keeping recovery aware of the work even if the caller is cancelled.
        let activity = thoosa_shared::CudaActivityScope::for_operation(&self.gate)?;
        // Bind stage/tensor identities to the admitted device and generation;
        // silently substituting the current worker could address unrelated keys.
        worker_id.check(CudaWorkerId { device_idx: self.tho_device_idx, generation: activity.generation(&self.gate).expect("worker lease") })?;
        Ok(activity)
    }
    fn copy_activity(&self, src: &ThoTensor, dst_device: ThoDevice) -> Result<thoosa_shared::CudaActivityScope, ThoError> {
        // Reference downloads need the source worker's lease; CPU uploads only
        // need admission to this device. Both copy entry points use these rules.
        let activity = match src.ref_worker_id() {
            Some(worker_id) => self.activity_for(worker_id)?,
            None => thoosa_shared::CudaActivityScope::for_operation(&self.gate)?,
        };
        // IPC supports CPU values uploaded to CUDA and worker references
        // downloaded to CPU. Reject impossible directions before queueing them.
        assert_ne!(src.tho_device(), dst_device, "should not copy to same device");
        match &dst_device.into() {
            Cpu => {
                assert_ne!(src.tho_device(), thoosa_common::tho_device::ThoDevice::from(0usize), "src device should not be CPU when copying to CPU");
                assert!(src.is_ref(), "In IPC mode, src must be a reference when copying to CPU.")
            }
            Cuda(_)=>{
                assert_eq!(src.tho_device(), thoosa_common::tho_device::ThoDevice::from(0usize), "src device should be CPU when copying to GPU");
                assert!(!src.is_ref(), "IPC mode, src must be value when copying to GPU")
            }
            _ => {
                panic!("not supported")
            }
        }
        Ok(activity)
    }

    pub(crate) fn is_current_worker(&self, worker_id: CudaWorkerId) -> bool {
        // Cleanup targets belong to one device and generation. This snapshot is
        // only an early filter; the actor checks again after receiving the work.
        worker_id.device_idx == self.tho_device_idx && worker_id.generation == self.gate.device_generation()
    }

    pub async fn new(gate: Arc<thoosa_shared::CudaActivityGate>, cuda_worker_args: Vec<String>, arena_fd: OwnedFd) -> Result<Self, ThoError> {
        let tho_device_id = gate.device_idx();
        debug!("CudaIPCActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let (control, controls) = mpsc::unbounded_channel();
        static NEXT_RELEASE_OWNER: AtomicU64 = AtomicU64::new(1);
        let release_id = NEXT_RELEASE_OWNER.fetch_add(1, Ordering::Relaxed);
        let actor = CudaIPCActor::new(Arc::clone(&gate), cuda_worker_args, receiver, controls, arena_fd, Some(TensorReleaseHandle { control: control.clone(), gate: Arc::clone(&gate), owner_id: release_id })).await?;
        tokio::spawn(CudaIPCActor::run(actor));

        Ok(Self {
            tho_device_idx: tho_device_id,
            gate,
            sender,
            control,
            release_id,
        })
    }

    pub async fn invoke_stage(&self, worker_id: CudaWorkerId, endpoint_name: String, stage_id: Option<String>, input: Box<dyn WorkerInput>, consume:bool) -> anyhow::Result<CudaWorkerMsgOut> {
        debug!("CudaIPCActorHandle::invoke_stage: endpoint_name={endpoint_name} stage_id={stage_id:?} input={input:?}");
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::InvokeStage {
                activity: self.activity_for(worker_id)?,
                worker_id,
                endpoint_name,
                stage_id,
                input,
                consume,
                respond_to,
            }
        )
        .instrument(tracing::debug_span!("CudaIPCActorHandle::invoke_stage",agg_perf="CudaIPCActorHandle::send_msg",)).await
        .context("invoke_stage: CUDA actor channel closed")?;

        Ok(recv.await.context("invoke_stage: CUDA actor response channel closed")??)
    }

    pub async fn install_stage(&self, worker_id: CudaWorkerId, endpoint_name: String, stage_id: Option<String>, init_data: Box<dyn WorkerInput>) -> anyhow::Result<CudaWorkerMsgOut> {
        debug!("CudaIPCActorHandle::invoke_stage: install_stage_2={endpoint_name} stage_id={stage_id:?} init_data={init_data:?}");

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::InstallStage {
                activity: self.activity_for(worker_id)?,
                worker_id,
                endpoint_name,
                stage_id,
                init_data,
                respond_to,
            }
        )
        .instrument(tracing::debug_span!("CudaIPCActorHandle::send_msg",agg_perf="CudaIPCActorHandle::install_stage_2",)).await
        .context("CUDA actor channel closed")?;

        Ok(recv.await.context("CUDA actor response channel closed")??)
    }


    /// Release a failed/partial restore without copying it over the intact CPU backup.
    pub(crate) async fn discard_stage_state(&self, worker_id: CudaWorkerId, endpoint_name: String, stage_id: Option<String>) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(CudaIPCActorMessage::DiscardStageState {
            activity: self.activity_for(worker_id)?,
            worker_id,
            endpoint_name,
            stage_id,
            respond_to,
        }).await.context("CUDA actor channel closed")?;
        unit_reply(recv.await.context("CUDA actor response channel closed")??)
    }

    pub(crate) async fn reset_stage_state(&self, worker_id: CudaWorkerId, endpoint_name: String, stage_id: Option<String>, info: StageStateInfo) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(CudaIPCActorMessage::ResetStageState {
            activity: self.activity_for(worker_id)?,
            worker_id,
            endpoint_name,
            stage_id,
            info,
            respond_to,
        }).await.context("CUDA actor channel closed")?;
        unit_reply(recv.await.context("CUDA actor response channel closed")??)
    }

    pub async fn copy_stage_state(&self, src:ThoTensor, dst_device:ThoDevice, dst_key: Option<H2DTensorKey>) -> Result<(), ThoError> {
        let activity = self.copy_activity(&src, dst_device)?;

        trace!("CudaIPCActorHandle::copy_stage_state: src={src:?} dst_device={dst_device:?} backtrace={}", Backtrace::capture().to_string());
        debug!("CudaIPCActorHandle::copy_stage_state: src={src:?} dst_device={dst_device:?}");

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Copy {
                src,
                dst_device,
                activity,
                respond_to,
                dst_key
            }
        ).await.context("CUDA actor channel closed")?;

        let r = recv.await.context("CUDA actor response channel closed")?;
        debug!("CudaIPCActorHandle::copy_stage_state:  get result: r={r:?}");
        match r? {
            IpcCopyResult::StateInstalled => Ok(()),
            IpcCopyResult::Tensor(_) => Err(ThoError::new("state upload returned a data tensor")),
        }
    }

    pub async fn shutdown(&self) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.control.send(BackendControl::Shutdown { respond_to }).context("CUDA actor control channel closed")?;
        recv.await.context("CUDA shutdown response channel closed")?
    }

}

#[async_trait]
impl CudaActorHandle for CudaIPCActorHandle {

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaIPCActorHandle::copy"))]
    async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) -> Result<ThoTensor, ThoError> {
        let activity = self.copy_activity(&src, dst_device)?;

        trace!("CudaIPCActorHandle::copy: src={src:?} dst_device={dst_device:?} backtrace={}", Backtrace::capture().to_string());
        debug!("CudaIPCActorHandle::copy: src={src:?} dst_device={dst_device:?}");

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Copy {
                src,
                dst_device,
                activity,
                respond_to,
                dst_key: None,
            }
        ).await.context("CUDA actor channel closed")?;

        let r = recv.await.context("CUDA actor response channel closed")?;
        debug!("CudaIPCActorHandle::copy:  get result: r={r:?}");
        match r? {
            IpcCopyResult::Tensor(mut tensor) => { tensor.set_release_handle(self.tensor_releaser()); Ok(tensor) },
            IpcCopyResult::StateInstalled => Err(ThoError::new("data copy returned a state acknowledgment")),
        }
    }

    #[cfg(test)]
    async fn restart(&self) -> Result<(), ThoError> {
        GlobalCtx::global().restart_cuda_device(self.tho_device_idx).await
    }

    async fn restart_backend(&self, authority: CudaActivityPauseTicket) -> Result<(), ThoError> {
        // Only this gate's current recovery coordinator may replace a worker,
        // and restart requires all ordinary activity to have drained.
        self.gate.check_recovery_authority(authority, true)?;
        debug!("CudaIPCActorHandle::restart_backend: check_recovery_authority ok");

        let (respond_to, recv) = oneshot::channel();
        let r = self.control.send(BackendControl::Restart { authority, respond_to }).context("CUDA actor control channel closed");

        debug!("CudaIPCActorHandle::restart_backend: Restart send r={r:?}");

        recv.await.context("CUDA restart response channel closed")?
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
    use crate::tch_error::tch_result;
    use super::{BackendControl, PendingRequests};
    use crate::copy_ctx::CudaActorHandle;
    use crate::cuda_utils::{cuda_mem, speed_gb_per_s};
    use crate::ipc::cuda_process::{CudaIPCActor, CudaIPCActorHandle, CudaIPCActorMessage};
    use crate::ipc::cuda_wokrer::CudaWorker;
    use crate::ipc::tests::run_ipc_test;
    use crate::runner::global_ctx::GlobalCtx;
    // use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, OOMRunnerConfig};
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    // use crate::torch_runner::tests::load_and_clc::loader::LoadedTensor;
    use crate::utils::tensor_digest;
    use crate::ws::mock_ws_simp;
    use anyhow::Context as _;
    use log::{debug, info};
    use parking_lot::RwLock;
    use rand::random_range;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::AtomicU64;
    use std::sync::atomic::Ordering::Relaxed;
    use std::time::Instant;
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::{Kind, Tensor};
    use thoosa_shared::perf_logging::{PERF, format_perf_map};
    use thoosa_shared::tracked_await::track_await;
    use thoosa_shared::ThoError;
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
    fn test_remote_release_retains_owner_and_runs_once_ipc() {
        use super::TensorReleaseHandle;
        let (control, mut commands) = mpsc::unbounded_channel();
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let worker = crate::ipc::worker_msg::CudaWorkerId { device_idx: 1, generation: gate.device_generation() };
        let owner = TensorReleaseHandle { control, gate, owner_id: 999999 };
        let tensor: ThoTensor = owner.decoding(|| ThoTensor::new_ref((), 123, vec![1], Kind::Float, None, worker));
        let clone = tensor.clone_ref().unwrap();
        drop(tensor);
        assert!(commands.try_recv().is_err());
        std::thread::spawn(move || drop(clone)).join().unwrap();
        assert!(matches!(commands.try_recv().unwrap(), BackendControl::ReleaseTensor { key: 123, worker_id } if worker_id == worker));
        assert!(commands.try_recv().is_err());
    }

    #[test]
    fn test_remote_release_skips_recovering_and_retired_workers_ipc() {
        use super::TensorReleaseHandle;
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let (control, mut commands) = mpsc::unbounded_channel();
        let owner = TensorReleaseHandle { control, gate: Arc::clone(&gate), owner_id: 999998 };
        let worker = crate::ipc::worker_msg::CudaWorkerId { device_idx: 1, generation: gate.device_generation() };
        let tensor = |key, worker| -> ThoTensor { owner.decoding(|| ThoTensor::new_ref((), key, vec![1], Kind::Float, None, worker)) };
        let cached = tensor(123, worker);
        let retained = tensor(124, worker);

        // Cache invalidation runs while the old worker generation is still
        // current. Its last references must disappear without queueing cleanup.
        let ticket = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        drop(cached);
        assert!(commands.try_recv().is_err(), "recovering worker releases must not be queued");
        gate.finish_pause(ticket);
        drop(retained);
        assert!(commands.try_recv().is_err(), "retired worker releases must not be queued");

        // The same release owner survives replacement, so fresh tensors must
        // resume normal cleanup rather than retaining a permanent suppression flag.
        let fresh = crate::ipc::worker_msg::CudaWorkerId { generation: gate.device_generation(), ..worker };
        drop(tensor(125, fresh));
        assert!(matches!(commands.try_recv().unwrap(), BackendControl::ReleaseTensor { key: 125, worker_id } if worker_id == fresh));
        let ticket = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        gate.fail_pause(ticket, Some(ThoError::new("replacement failed")));
        drop(tensor(126, fresh));
        assert!(commands.try_recv().is_err(), "failed recovery must keep cleanup suppressed");
    }

    #[test]
    fn test_worker_error_round_trip_preserves_cuda_recovery() {
        use thoosa_shared::{CudaAction, SyncStageRunnerError};

        for action in [CudaAction::HandleOOM, CudaAction::NeedsRestart, CudaAction::Other] {
            // Use the constructor so injected CUDA errors capture their backtrace.
            let error = thoosa_shared::tho_error::cuda_error(action, "device failure detail".into()).context("forward model");
            let job = ThoError::from(SyncStageRunnerError::JobError { job_id: 27, inner: Box::new(error) });
            let received = round_trip_worker_error(job.context("worker request 42"));
            assert!(matches!(&received, ThoError::Cuda { action: received_action, .. } if *received_action == action));
            assert_eq!(received.is_oom(), action == CudaAction::HandleOOM);
            assert_eq!(received.is_retryable_oom(), action == CudaAction::HandleOOM);
            assert_eq!(received.needs_cuda_recovery(), action == CudaAction::NeedsRestart);
            let message = received.to_string();
            assert!(message.contains("worker request 42") && message.contains("job 27") && message.contains("forward model") && message.contains("device failure detail"));
        }
    }

    #[test]
    fn test_worker_error_round_trip_preserves_libtorch_recovery() {
        // Exercise both direct tensor-result conversion and the shared anyhow boundary.
        for (message, oom, reset) in [
            ("CUDA out of memory", true, false),
            ("CUDA error: an illegal memory access was encountered", false, true),
        ] {
            let direct = crate::tch_error::tch_error(tch::TchError::Torch(message.into()));
            let contextual = ThoError::from(anyhow::Error::new(tch::TchError::Torch(message.into())).context("forward model"));
            for error in [direct, contextual] {
                let received = round_trip_worker_error(error.context("worker request"));
                assert_eq!((received.is_oom(), received.is_retryable_oom(), received.needs_cuda_recovery()), (oom, oom, reset));
                assert!(received.to_string().contains(message));
            }
        }
    }

    #[test]
    fn test_worker_error_round_trip_preserves_non_cuda_semantics() {
        use thoosa_shared::SyncStageRunnerError;

        for error in [
            ThoError::AdjusterFailed,
            ThoError::CudaBeingRecovered,
            ThoError::NotFound(Some("worker".to_string())),
            SyncStageRunnerError::SubmitErrorClosed.into(),
            SyncStageRunnerError::SubmitErrorCancelled.into(),
        ] {
            let received = round_trip_worker_error(error.clone().context("worker operation"));
            assert_eq!(std::mem::discriminant(received.what()), std::mem::discriminant(error.what()));
            assert_eq!((received.is_oom(), received.is_retryable_oom(), received.needs_cuda_recovery()), (error.is_oom(), error.is_retryable_oom(), error.needs_cuda_recovery()));
            assert_eq!(received.what().to_string(), error.to_string());
        }

        let received = round_trip_worker_error(ThoError::from(std::io::Error::other("CUDA out of memory")).context("read model"));
        assert!(matches!(received, ThoError::Anyhow(_)));
        assert!(received.is_oom() && received.is_retryable_oom() && !received.needs_cuda_recovery());
        assert!(received.to_string().contains("read model") && received.to_string().contains("CUDA out of memory"));
    }

    #[test]
    fn test_worker_error_round_trip_preserves_terminal_training_oom() {
        for recovery in [None, Some(ThoError::oom().context("restore buffers"))] {
            let error = ThoError::TrainingSessionFailed(Arc::new(thoosa_shared::TrainingSessionFailure { original: ThoError::oom().context("optimizer update"), recovery }));
            // Exercise the actual worker reply codec, preserving diagnostics and retry policy.
            let received = round_trip_worker_error(error.context("training failed"));
            assert!(received.is_oom());
            assert!(!received.is_retryable_oom());
            assert!(!received.needs_cuda_recovery());
            assert!(format!("{received:#}").contains("out of memory"));
            let ThoError::TrainingSessionFailed(failure) = received.what() else { panic!("training failure classification was lost") };
            assert!(failure.original.is_oom());
        }
    }

    #[test]
    fn test_worker_error_round_trip_preserves_terminal_recovery_failure() {
        use thoosa_shared::{CudaAction, CudaRecoveryFailure};

        // Terminal errors must retain diagnostics without becoming retryable again
        // when they pass through the same wire path as ordinary CUDA failures.
        for recovery in [None, Some(ThoError::oom().context("replacement allocation"))] {
            let has_recovery = recovery.is_some();
            let original = thoosa_shared::tho_error::cuda_error(CudaAction::NeedsRestart, "illegal address".into()).context("original stage");
            let error = ThoError::CudaRecoveryFailed(Arc::new(CudaRecoveryFailure { original, recovery }));
            let received = round_trip_worker_error(error.context("worker recovery"));
            assert!(received.is_cuda_recovery_failed());
            assert!(!received.is_oom() && !received.is_retryable_oom() && !received.needs_cuda_recovery());
            let ThoError::CudaRecoveryFailed(failure) = received.what() else { panic!("terminal error was lost") };
            assert!(failure.original.needs_cuda_recovery());
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
        let handle = CudaIPCActorHandle { release_id: 0, gate: Arc::clone(&gate), tho_device_idx: 1, sender, control };
        let (respond_to, _) = tokio::sync::oneshot::channel();
        let worker_id = handle.worker_id().unwrap();
        // Fill the ordinary work queue; recovery must still use its control channel.
        handle.sender.send(CudaIPCActorMessage::DebugWorker { worker_id, activity: handle.activity_for(worker_id).unwrap(), respond_to }).await.unwrap();
        let wrong = thoosa_shared::CudaActivityGate::new(2).request_pause(ThoError::CudaBeingRecovered).unwrap();
        assert!(handle.restart_backend(wrong).await.is_err());
        assert!(handle.retire_backend(wrong).await.is_err());
        assert!(controls.try_recv().is_err());
        let authority = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
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
            pending.close(ThoError::CudaBeingRecovered);
            assert!(response.try_recv().is_err());
            assert!(weak.upgrade().is_some());
            if graceful {
                assert!(timeout(Duration::from_secs(2), worker.shutdown_with_timeout(2, Duration::from_millis(10))).await.unwrap().is_err());
            } else {
                timeout(Duration::from_secs(2), worker.retire(ThoError::CudaBeingRecovered)).await.unwrap().unwrap();
            }
            assert!(!std::path::Path::new(&format!("/proc/{pid}")).exists(), "pending buffers cannot be released before child exit and reaping");
            assert!(response.await.unwrap().is_err());
            assert!(weak.upgrade().is_none());
        }
    }

    #[tokio::test]
    async fn test_retired_handles_cannot_reach_replacement_worker_ipc() -> Result<(), ThoError>{
        use crate::ipc::worker_msg::CudaWorkerId;
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);

        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let (sender, mut receiver) = tokio::sync::mpsc::channel(8);
        let (control, mut controls) = mpsc::unbounded_channel();
        let handle = CudaIPCActorHandle { release_id: 0, gate: Arc::clone(&gate), tho_device_idx: 1, sender, control };
        let retired = handle.worker_id()?;

        let old_tensor = || ThoTensor::new_ref((), 42, vec![1], Kind::Float, None, retired);
        let ticket = gate.request_pause(ThoError::CudaBeingRecovered)?;
        // Explicit cleanup must also avoid IPC before the old worker is retired.
        handle.tensor_releaser().release(42, retired);
        assert!(receiver.try_recv().is_err(), "recovering worker cleanup must not be queued");
        assert!(controls.try_recv().is_err(), "recovering tensor cleanup must not enter the control queue");
        gate.finish_pause(ticket);

        assert!(matches!(handle.copy(old_tensor(), 0usize.into()).await.unwrap_err().what(), ThoError::CudaBeingRecovered));
        assert!(matches!(ThoError::from(handle.invoke_stage(retired, "stage".into(), Some(42.to_string()), Box::new(()), false).await.unwrap_err()).what(), ThoError::CudaBeingRecovered));
        assert!(handle.install_stage(retired, "stage".into(), Some(42.to_string()), Box::new(())).await.is_err());


        handle.tensor_releaser().release(42, retired);
        assert!(controls.try_recv().is_err(), "retired tensor cleanup must not enter the control queue");
        // Endpoint cleanup is queued with its original identity for the actor to check.
        handle.release_endpoint(Some(42.to_string()), "stage".into(), retired);
        let BackendControl::ReleaseEndpoint { worker_id, .. } = controls.try_recv().unwrap() else { panic!("expected endpoint release") };
        assert_eq!(worker_id, retired);

        assert!(receiver.try_recv().is_err(), "old IDs must never be rebound to the new worker");

        let fresh: CudaWorkerId = handle.worker_id()?;
        let mut invoke = Box::pin(handle.invoke_stage(fresh, "stage".into(), Some(42.to_string()), Box::new(()), false));
        assert!(futures::poll!(invoke.as_mut()).is_pending());
        let message = receiver.recv().await.unwrap();
        drop(invoke);
        let ticket = gate.request_pause(ThoError::CudaBeingRecovered)?;
        let mut idle = Box::pin(gate.wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending(), "cancelled invocation must keep its request leased");
        let CudaIPCActorMessage::InvokeStage { worker_id, .. } = &message else { panic!("expected stage request") };
        assert_eq!(*worker_id, fresh);
        drop(message);
        idle.await;
        gate.finish_pause(ticket);

        Ok(())
    }

    #[tokio::test]
    async fn test_reply_identity_is_checked_after_wire_routing_ipc() {
        use crate::ipc::serialization::{PostcardToFromBytes, ToFromBytes};
        use crate::ipc::worker_msg::{CudaWorkerId, CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation};
        use thoosa_shared::CudaDeviceGeneration;
        for wrong_worker in [CudaWorkerId { device_idx: 1, generation: CudaDeviceGeneration(0) }, CudaWorkerId { device_idx: 2, generation: CudaDeviceGeneration(1) }] {
            let current = CudaWorkerId { device_idx: 1, generation: CudaDeviceGeneration(1) };
            let pending = Arc::new(PendingRequests::new(current));
            let cancel = tokio_util::sync::CancellationToken::new();
            let (reader, mut writer) = tokio::io::duplex(1024);
            let reader_task = tokio::spawn(super::reader_loop(reader, Arc::clone(&pending), cancel.clone(), None));
            let (sender, mut requests) = tokio::sync::mpsc::channel(1);
            let request = CudaWorkerMsgIn::new(current, 42, Operation::Debug);
            let mut response = Box::pin(super::send_request_and_wait_msg(sender, Arc::clone(&pending), request));
            assert!(futures::poll!(response.as_mut()).is_pending());
            assert_eq!(requests.recv().await.unwrap().worker_id, current);
            let reply = CudaWorkerMsgOut { worker_id: wrong_worker, request_id: 42, status: OpResult::Ok(None) };
            super::write_frame(&mut writer, &PostcardToFromBytes::to_bytes(&reply).unwrap(), "test").await.unwrap();
            tokio::time::timeout(std::time::Duration::from_secs(2), pending.failed.cancelled()).await.unwrap();
            assert!(futures::poll!(response.as_mut()).is_pending(), "a malformed reply cannot prove worker access ended");
            pending.exited(ThoError::CudaBeingRecovered);
            let error = tokio::time::timeout(std::time::Duration::from_secs(2), response).await.unwrap().unwrap_err();
            assert!(matches!(error.what(), ThoError::CudaBeingRecovered));
            cancel.cancel();
            reader_task.await.unwrap();
        }
    }

    #[tokio::test]
    async fn test_copy_admission_is_device_local_ipc() {
        let failed_gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let healthy_gate = Arc::new(thoosa_shared::CudaActivityGate::new(2));
        let (failed_tx, mut failed_rx) = tokio::sync::mpsc::channel(1);
        let (healthy_tx, mut healthy_rx) = tokio::sync::mpsc::channel(1);
        let failed = CudaIPCActorHandle { release_id: 0, gate: Arc::clone(&failed_gate), tho_device_idx: 1, sender: failed_tx, control: mpsc::unbounded_channel().0 };
        let healthy = CudaIPCActorHandle { release_id: 0, gate: healthy_gate, tho_device_idx: 2, sender: healthy_tx, control: mpsc::unbounded_channel().0 };
        let pause = failed_gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
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
        let handle = CudaIPCActorHandle { release_id: 0, sender, tho_device_idx: 1, gate: Arc::clone(&gate), control: mpsc::unbounded_channel().0 };
        // A CPU tensor exercises real copy admission and message ownership
        // without starting a CUDA worker or allocating GPU storage.
        let src = ThoTensor::new((), Tensor::f_from_slice(&[1f32]).unwrap()).unwrap();
        let mut copy = Box::pin(handle.copy(src, thoosa_common::tho_device::ThoDevice::from(1usize)));
        assert!(futures::poll!(copy.as_mut()).is_pending());
        let request = receiver.recv().await.unwrap();
        drop(copy);
        let pause = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        let mut idle = Box::pin(gate.wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending());
        // The request retains the scope even though its caller has gone away.
        // Discarding it before dispatch is equivalent to completing that activity.
        drop(request);
        tokio::time::timeout(std::time::Duration::from_secs(2), idle).await.unwrap();
        gate.finish_pause(pause);
    }

    async fn run_ipc_copy_perf(cuda_actor: &Arc<dyn CudaActorHandle>, threads: u8, runs: u8) -> Result<(f64, f64), ThoError> {
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
                Arc::clone(&cuda_actor),
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
        cuda_actor: Arc<dyn CudaActorHandle>,
    ) -> Result<(), ThoError> {
        let dim_0 = random_range(0.5 * input_n as f64..1.5 * input_n as f64).max(1.0) as i64;
        let mut src_t = Tensor::rand([dim_0, dim_1], FLOAT_CPU);

        for _ in 0..runs {
            let inst_0 = Instant::now();
            let cuda_ref = track_await!(cuda_actor.copy(src_t.into(), 1.into())).await?;
            h2d_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            let inst_0 = Instant::now();
            let cpu_tensor = track_await!(cuda_actor.copy(cuda_ref, 0.into())).await?;
            d2h_time.fetch_add(inst_0.elapsed().as_millis() as u64, Relaxed);

            let tensor = cpu_tensor.into_local()?;
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

            let _rand = tch_result(Tensor::f_rand([dim_0 as i64, dim_1 as i64-1], FLOAT_CUDA))?;
            debug!("test_copy_ipc_cuda_worker_propagates_oom, _rand={_rand:?}");


            let too_large_tensor = tch_result(Tensor::f_rand([dim_0 as i64, 1], FLOAT_CPU))?;

            let copy_result = tokio::time::timeout(
                std::time::Duration::from_secs(600),
                track_await!(cuda_actor.copy(too_large_tensor.into(), tho_device_idx.into())),
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
            assert!(!error.needs_cuda_recovery(), "ordinary worker OOM must not reset CUDA: {error:?}");


            debug!("All ok");

                Ok(())
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::new(),
                    ThoDevice::from(1usize),
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
                let (d2h, h2d)  = run_ipc_copy_perf(&cuda_actor, 3, 3).await?;

                info!("\n IPC worker copy d2h speed: {d2h:#2} GB/s, h2d speed {h2d:#2} GB/s");


                Ok(())
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::new(),
                    ThoDevice::from(1usize),
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
            let expected_digest = tch_result(tensor_digest(&main_process_cpu_tensor))?;


            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let cuda_ref = track_await!(cuda_actor.copy(
                main_process_cpu_tensor.into(),
                tho_device_idx.into(),//Cuda
            )).await?;

            debug!("run_main: cuda_ref={cuda_ref:?}");

            let reff = cuda_ref.ref_key().unwrap();

            let cpu_tensor = track_await!(cuda_actor.copy(
                cuda_ref.clone_ref()?,//otherwise cuda_ref gets dropped and deleted from worker
                0.into(),//Cpu
            )).await?;

            debug!("run_main: cpu_tensor={cpu_tensor:?}");

            let actual_digest = tch_result(tensor_digest(&*cpu_tensor.local()?))?;
            if actual_digest.hex != expected_digest.hex {
                return Err(ThoError::new(format!("worker CUDA round-trip digest mismatch: expected={} actual={}", expected_digest.hex, actual_digest.hex)));
            }

            // Release uses the control queue, which the actor drains before ordinary work.
            let ipc = GlobalCtx::global().execution(1).ipc_actor()?;
            ipc.tensor_releaser().release(reff, cuda_ref.ref_worker_id().unwrap());
            GlobalCtx::global().execution(tho_device_idx).ipc_actor()?.debug_worker().await?;

            debug!("run_main: drop ok");

            let r = track_await!(cuda_actor.copy(
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
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::new(),
                    ThoDevice::from(1usize),
                ).await
            },
        )
    }

}

#[cfg(test)]
mod export_tests {
    use super::*;
    use futures_util::FutureExt;

    fn reply(pending: &PendingRequests, request: CudaWorkerMsgIn, status: OpResult) {
        pending.complete(CudaWorkerMsgOut { worker_id: request.worker_id, request_id: request.request_id, status }).unwrap();
    }

    #[tokio::test]
    async fn test_export_batches_all_tensors_and_finishes_after_success_failure_or_cancellation() {
        for outcome in 0..3 {
            let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
            let worker = CudaWorkerId { device_idx: 1, generation: gate.device_generation() };
            let pending = Arc::new(PendingRequests::new(worker));
            let (tx, mut requests) = mpsc::channel(8);
            let (send, receive) = oneshot::channel();
            let activity = thoosa_shared::CudaActivityScope::for_operation(&gate).unwrap();
            let task = tokio::spawn(activity.run(CudaIPCActor::export_stage(worker, "batch".into(), Some("1".into()), Arc::new(AtomicU64::new(1)), Arc::clone(&pending), tx, Arc::clone(&gate), send)));
            let begin = requests.recv().await.unwrap();
            let export_id = begin.request_id;
            // Zero-byte tensors exercise the real actor protocol without
            // installing a process-global allocator or requiring CUDA.
            let tensors = (0..8).map(|index| WorkerTensorInfoPayload::new(TensorKey::stage_state("batch", "1", index.to_string()), tch::Kind::Float.into(), vec![0, index], 0)).collect();
            reply(&pending, begin, OpResult::Ok(Some(Box::new(StageStateInfo::new(tensors, None)))));
            let batch = requests.recv().await.unwrap();
            let Operation::ExportStageTensors { export_id: batch_export_id, destinations, .. } = &batch.operation else { panic!("expected one batched export") };
            assert_eq!(*batch_export_id, export_id);
            assert_eq!(destinations.len(), 8);
            assert!(requests.try_recv().is_err(), "there must be no per-tensor requests");
            let mut receive = if outcome == 2 { drop(receive); None } else { Some(receive) };
            reply(&pending, batch, if outcome == 1 { ThoError::oom().into() } else { OpResult::Ok(None) });
            let keep_live = if outcome == 0 {
                let mut dump = receive.take().unwrap().await.unwrap().unwrap();
                assert!(requests.try_recv().is_err(), "publication must precede export completion");
                Some(tokio::spawn(dump.export.take().unwrap().keep_live()))
            } else {
                // Failure must resolve the caller only after abort releases the freeze.
                if let Some(receive) = receive.as_mut() { assert!(receive.try_recv().is_err()); }
                None
            };
            let finish = requests.recv().await.unwrap();
            assert!(matches!(finish.operation, Operation::FinishStageStatePurge { commit: false, export_id: id, .. } if id == export_id));
            let pause = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
            assert!(gate.wait_until_idle().now_or_never().is_none(), "accepted export must retain activity until finish replies");
            reply(&pending, finish, OpResult::Ok(None));
            if let Some(keep_live) = keep_live { keep_live.await.unwrap().unwrap(); }
            if let Some(receive) = receive { assert!(receive.await.unwrap().unwrap_err().is_oom()); }
            task.await.unwrap();
            gate.wait_until_idle().await;
            gate.finish_pause(pause);
            assert!(!pending.failed.is_cancelled(), "a recoverable batch failure must keep transport usable");
        }
    }

    #[tokio::test]
    async fn test_concurrent_exports_have_unique_requests_and_wait_for_backup_commit() {
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let worker = CudaWorkerId { device_idx: 1, generation: gate.device_generation() };
        let pending = Arc::new(PendingRequests::new(worker));
        let next_id = Arc::new(AtomicU64::new(1));
        let (tx, mut requests) = mpsc::channel(8);
        let mut results = Vec::new();
        let mut tasks = Vec::new();
        for name in ["a", "b"] {
            let (send, receive) = oneshot::channel();
            let activity = thoosa_shared::CudaActivityScope::for_operation(&gate).unwrap();
            tasks.push(tokio::spawn(activity.run(CudaIPCActor::export_stage(worker, name.into(), Some("1".into()), Arc::clone(&next_id), Arc::clone(&pending), tx.clone(), Arc::clone(&gate), send))));
            results.push(receive);
        }
        let mut ids = std::collections::HashSet::new();
        let mut exports = HashMap::new();
        for _ in 0..2 {
            let request = requests.recv().await.unwrap();
            let Operation::BeginStageStatePurge { endpoint_name, .. } = &request.operation else { panic!("expected begin") };
            exports.insert(endpoint_name.clone(), request.request_id);
            assert!(ids.insert(request.request_id));
            reply(&pending, request, OpResult::Ok(Some(Box::new(StageStateInfo::empty()))));
        }
        let mut commits = Vec::new();
        for result in results {
            let mut backup = result.await.unwrap().unwrap();
            assert!(requests.try_recv().is_err(), "worker state must survive until the CPU backup is committed");
            commits.push(backup.export.take().unwrap().commit());
        }
        let commits = tokio::spawn(futures_util::future::join_all(commits));
        for _ in 0..2 {
            let request = requests.recv().await.unwrap();
            let Operation::FinishStageStatePurge { endpoint_name, export_id, commit: true, .. } = &request.operation else { panic!("expected commit") };
            assert_eq!(*export_id, exports[endpoint_name]);
            assert!(ids.insert(request.request_id), "every wire message needs a fresh request ID");
            reply(&pending, request, OpResult::Ok(None));
        }
        assert!(commits.await.unwrap().into_iter().all(|result| result.is_ok()));
        for task in tasks { task.await.unwrap(); }
        assert_eq!(ids.len(), 4);
    }

    #[tokio::test]
    async fn test_cancelled_export_aborts_before_releasing_activity() {
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let worker = CudaWorkerId { device_idx: 1, generation: gate.device_generation() };
        let pending = Arc::new(PendingRequests::new(worker));
        let (tx, mut requests) = mpsc::channel(8);
        let (send, receive) = oneshot::channel();
        let activity = thoosa_shared::CudaActivityScope::for_operation(&gate).unwrap();
        let task = tokio::spawn(activity.run(CudaIPCActor::export_stage(worker, "a".into(), Some("1".into()), Arc::new(AtomicU64::new(1)), Arc::clone(&pending), tx, Arc::clone(&gate), send)));
        let request = requests.recv().await.unwrap();
        drop(receive);
        reply(&pending, request, OpResult::Ok(Some(Box::new(StageStateInfo::empty()))));
        let abort = requests.recv().await.unwrap();
        assert!(matches!(abort.operation, Operation::FinishStageStatePurge { commit: false, .. }));
        let ticket = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        assert!(gate.wait_until_idle().now_or_never().is_none(), "the transaction must retain activity through the abort reply");
        reply(&pending, abort, OpResult::Ok(None));
        task.await.unwrap();
        gate.wait_until_idle().await;
        gate.finish_pause(ticket);
    }

    #[tokio::test]
    async fn test_export_description_error_replies_without_starting_an_abort() {
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let worker = CudaWorkerId { device_idx: 1, generation: gate.device_generation() };
        let pending = Arc::new(PendingRequests::new(worker));
        let (tx, mut requests) = mpsc::channel(8);
        let (send, receive) = oneshot::channel();
        let task = tokio::spawn(CudaIPCActor::export_stage(worker, "missing".into(), None, Arc::new(AtomicU64::new(1)), Arc::clone(&pending), tx, Arc::clone(&gate), send));
        let error = thoosa_shared::tho_error::cuda_error(thoosa_shared::CudaAction::HandleOOM, "description failed".into());
        reply(&pending, requests.recv().await.unwrap(), error.into());
        assert!(receive.await.unwrap().unwrap_err().is_oom());
        task.await.unwrap();
        assert!(requests.try_recv().is_err(), "failed begin has no transaction to abort");
        assert!(!pending.failed.is_cancelled(), "ordinary begin errors must not retire a healthy worker");
    }

    #[tokio::test]
    async fn test_export_commit_failure_is_typed_and_keeps_transport() {
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let worker = CudaWorkerId { device_idx: 1, generation: gate.device_generation() };
        let pending = Arc::new(PendingRequests::new(worker));
        let (tx, mut requests) = mpsc::channel(8);
        let (send, receive) = oneshot::channel();
        let task = tokio::spawn(CudaIPCActor::export_stage(worker, "a".into(), Some("1".into()), Arc::new(AtomicU64::new(1)), Arc::clone(&pending), tx, Arc::clone(&gate), send));
        reply(&pending, requests.recv().await.unwrap(), OpResult::Ok(Some(Box::new(StageStateInfo::empty()))));
        let mut dump = receive.await.unwrap().unwrap();
        let commit = tokio::spawn(dump.export.take().unwrap().commit());
        let request = requests.recv().await.unwrap();
        let error = thoosa_shared::tho_error::cuda_error(thoosa_shared::CudaAction::HandleOOM, "purge failed".into());
        reply(&pending, request, error.into());
        assert!(commit.await.unwrap().unwrap_err().is_oom(), "the final worker error must reach the offload caller");
        assert!(!pending.failed.is_cancelled(), "failed commit must not retire an otherwise healthy worker");
        assert_eq!(gate.device_generation(), worker.generation);
        task.await.unwrap();
    }
}

impl CudaIPCActorHandle {
    pub async fn debug_worker(&self) -> Result<CudaWorkerMsgOut, ThoError> {
        let worker_id = self.worker_id()?;
        debug!("CudaIPCActorHandle::debug_worker");

        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::DebugWorker {
                activity: self.activity_for(worker_id)?,
                worker_id,
                respond_to,
            }
        )
            .instrument(tracing::debug_span!("CudaIPCActorHandle::send_msg",agg_perf="CudaIPCActorHandle::install_stage_2",)).await
            .context("CUDA actor channel closed")?;

        recv.await.context("CUDA actor response channel closed")?
    }
}
