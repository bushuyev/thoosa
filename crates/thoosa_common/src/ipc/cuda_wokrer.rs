use crate::cuda_utils::{cuda_mem_for_device, tensor_mem_size};
use crate::data_with_extra::DataWithExtra;
use crate::dims::Dims;
use crate::ipc::codec::IpcCodec;
use crate::ipc::cuda_process;
use crate::ipc::cuda_process::{CONSUME_ENDPOINT_PREFIX, D2H_START_PAYLOAD_PREFIX, DESCRIBE_REF_PAYLOAD_PREFIX, DROP_ENDPOINT_PREFIX, DROP_THO_TENSOR, DeviceInfo, ENDPOINT_OK_PAYLOAD, H2D_START_OK_PAYLOAD, H2D_START_PAYLOAD_PREFIX, INSTALL_ENDPOINT_PREFIX, INVOKE_ENDPOINT_PREFIX, IPC_SOCK_ENV, IpcChunkWriter, OutgoingRequestPayload, PURGE_ENDPOINT_PREFIX, PendingWorkerD2hTransfer, PendingWorkerH2dTransfer, SHARED_MEMFD_PAYLOAD_PREFIX, TENSOR_INFO_PAYLOAD_PREFIX, WorkerCudaResources, WorkerD2hChunk, WorkerD2hStartPayload, WorkerH2dChunk, WorkerH2dStartPayload, WorkerReply, WorkerTensorInfoPayload};
use crate::ipc::ipc_cuda_utils::{IpcTensorTransferMetadata, IpcTransferDirection, SharedMemfdBuffer, recv_fd};
use crate::ipc::json_ser::{ErasedWithState, PostcardToFromBytes, SyncStageIpcResponse, SyncStageIpcWrapper, ToFromBytes};
use crate::ipc::proxy_tests::TestOutputA;
use crate::ipc::worker_msg::{CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation, WorkerInput, WorkerOutput};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::{ThoTensor, ThoTensorInner};
use crate::ws::{FsWorkspace, Workspace, WsConfigs};
use clap::Parser;
use dashmap::DashMap;
use futures_util::future::BoxFuture;
use log::{debug, error, info, trace, LevelFilter};
use parking_lot::{Mutex, RwLock};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Serializer;
use std::collections::HashMap;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Debug;
use std::io::ErrorKind;
use std::os::fd::OwnedFd;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use tch::{Device, Tensor};
use thoosa_shared::perf_logging::init_tracing;
use thoosa_shared::traits::{Adjustable, Data, MaxParallelCtx, Named, RunCtx, WithId};
use thoosa_shared::{CopyToDevice, ThoError, panic_to_err, panic_to_err_async};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};
use tokio::task::{JoinHandle, JoinSet};
use tracing::field::debug;
use tub::Pool;

const WORKER_EXEC_STREAMS: usize = 32;
static NEXT_WORKER_TENSOR_KEY: AtomicU64 = AtomicU64::new(1);

//1 cuda device per worker
pub struct CudaWorker;

#[deprecated]//use InitFn2
pub type InitFn = Box<dyn Fn(Vec<u8>, Arc<RwLock<Box<dyn Workspace>>>, &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<Result<Box<dyn SyncStageIpcWrapper>, ThoError>>  >;
pub type InitFn2 = Box<dyn Fn(Box<dyn WorkerInput>, Arc<RwLock<Box<dyn Workspace>>>, &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<Result<Box<dyn SyncStageIpcWrapper>, ThoError>> + Send + Sync>;

type InstalledEndpoints = Arc<DashMap<String, Option<Arc<dyn SyncStageIpcWrapper>>>>;

#[derive(Clone)]
struct WorkerRequestContext {
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    endpoint_inits: Arc<HashMap<String, InitFn2>>,
    exec_stream_pool: Arc<Pool<ThoCudaStream>>,
    all_cuda_tensors: Arc<Mutex<HashMap<u64, Tensor>>>,
    pending_h2d_transfers: Arc<Mutex<HashMap<u64, PendingWorkerH2dTransfer>>>,
    pending_d2h_transfers: Arc<Mutex<HashMap<u64, PendingWorkerD2hTransfer>>>,
    cuda_resources: Arc<Mutex<HashMap<usize, Arc<WorkerCudaResources>>>>,
    installed_endpoints: InstalledEndpoints,
    reply_tx: mpsc::Sender<WorkerReply>,
    tho_device: ThoDevice,
    with_digest: bool,
}

enum WorkerReaderAction {
    Ready,
    ReceiveFd(oneshot::Sender<std::io::Result<OwnedFd>>),
}

enum WorkerRequestControl {
    Continue,
    Shutdown { request_id: u64 },
}

pub trait Init {
    fn init() -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>;
}


#[derive(Parser, Clone, Debug)]
#[command(version, about, long_about = None)]
pub struct WorkerCli {
    #[arg(long, default_value = ".")]
    pub ws_path: String,

    #[arg(long, default_value = "INFO")]
    pub term_logger_level: LevelFilter,

    #[arg(long, default_value = "INFO")]
    pub write_logger_level: LevelFilter,

    #[arg(long, default_value_t = 0)]
    pub cuda_device:usize
}

impl CudaWorker {

    pub fn run_if_sock(cdt:&str, app_configs:&WsConfigs, init_map: HashMap<String, InitFn2>) -> Result<(), ThoError>{

        let ipc_sock = std::env::var_os(IPC_SOCK_ENV);
        println!("CudaWorker::run_if_sock: ipc_sock={ipc_sock:?}");

        if let Some(sock_path) = ipc_sock {



            let worker_cli = WorkerCli::parse();
            let ws_root:PathBuf = worker_cli.ws_path.into();



            let log_filename = ws_root.join("log")
                .join(format!("thoosa_worker_{}.log", cdt).to_owned());

            let _logging_guards = init_tracing(
                worker_cli.term_logger_level,
                false,
                worker_cli.write_logger_level,
                false,
                log_filename,
                "info",
                None,
                Some(([127, 0, 0, 1], 6667).into()),
            )?;

            debug!("CudaWorker::run_if_sock: sock_path={sock_path:?}");
            let exec_context = ThoCudaContext::new(0i32)?;//TODO device
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .on_thread_start(move || {

                    if let Err(err) = exec_context.bind_to_thread() {
                        panic!("Failed to create context: {err:?}");
                    }

                })
                .thread_name_fn(|| {
                    static TOKIO_THREAD_ID: AtomicUsize = AtomicUsize::new(0);
                    let id = TOKIO_THREAD_ID.fetch_add(1, Ordering::Relaxed);
                    format!("cuda-worker-{id}")
                })
                .build()?;
            rt.block_on(async move {
                let ws = Arc::new(RwLock::new(FsWorkspace::new(
                    ws_root.clone(),
                    app_configs.clone(),
                )?));

                // Application workers use only postcard/typetag endpoint initializers after the proxy migration.
                CudaWorker::run_worker2(sock_path, &ws, HashMap::new(), init_map, ThoDevice::from(Device::Cuda(0))).await//TODO Cuda(0)
            })?;

            std::process::exit(0);
        }

        Ok(())
    }

    pub async fn run_worker(
        sock_path:OsString,
        ws:&Arc<RwLock<Box<dyn Workspace>>>,
        endpoint_inits: HashMap<String, InitFn>,
        tho_device: ThoDevice
    ) -> Result<(), ThoError> {
        Self::run_worker2(sock_path, ws, endpoint_inits, HashMap::new(), tho_device).await
    }

    pub async fn run_worker2(
        sock_path:OsString,
        ws:&Arc<RwLock<Box<dyn Workspace>>>,
        _endpoint_inits: HashMap<String, InitFn>,
        endpoint_inits2: HashMap<String, InitFn2>,
        tho_device: ThoDevice
    ) -> Result<(), ThoError> {
        // Stateful capability is erased per endpoint so one worker can host unrelated state types.
        GlobalCtx::init_devices(2);//TODO config

        let with_digest = false;//TODO config

        let cuda_device = tho_device.gpu_ordinal();

        let exec_stream_pool = Arc::new(Pool::from_iter(
            (0..WORKER_EXEC_STREAMS)
                .map(|_| ThoCudaStream::from_pool(cuda_device as i32, false))
                .collect::<Result<Vec<_>, _>>()?,
        ));

        info!("run_worker: sock_path={sock_path:?} cuda_device={cuda_device} endpoint_inits.keys={:?}", endpoint_inits2.keys().collect::<Vec<_>>());

        let stream = UnixStream::connect(PathBuf::from(sock_path)).await?;

        debug!("run_worker: worker connected");

        let (mut reader, mut writer) = stream.into_split();

        let (free_mem, total_mem) = cuda_mem_for_device(cuda_device);

        cuda_process::write_frame(
            &mut writer,
            &PostcardToFromBytes::to_bytes(&DeviceInfo {
                free_mem,
                total_mem
            })?,
            "DeviceInfo/worker"
        ).await?;

        let (reply_tx, reply_rx) = mpsc::channel::<WorkerReply>(1024);

        let writer_task = tokio::spawn(cuda_process::worker_response_writer_loop(writer, reply_rx));
        let all_cuda_tensors = Arc::new(parking_lot::Mutex::new(HashMap::new()));//TODO to RwLock?
        let request_context = WorkerRequestContext {
            ws: Arc::clone(ws),
            endpoint_inits: Arc::new(endpoint_inits2),
            exec_stream_pool,
            all_cuda_tensors,
            pending_h2d_transfers: Arc::new(Mutex::new(HashMap::new())),
            pending_d2h_transfers: Arc::new(Mutex::new(HashMap::new())),
            cuda_resources: Arc::new(Mutex::new(HashMap::new())),
            installed_endpoints: Arc::new(DashMap::new()),
            reply_tx: reply_tx.clone(),
            tho_device,
            with_digest,
        };
        let mut request_tasks = JoinSet::new();

        // Decode in the spawned task, but synchronize ancillary descriptor receipt before reading the next frame.
        let shutdown_request_id = loop {
            let completed = tokio::select! {
                biased;
                joined = request_tasks.join_next(), if !request_tasks.is_empty() => {
                    let joined = joined.ok_or_else(|| ThoError::new("worker request JoinSet ended unexpectedly"))?;
                    let control = joined.map_err(|error| ThoError::new(format!("worker request task failed: {error}")))??;
                    Some(control)
                }
                payload = cuda_process::read_frame(&mut reader, "run_worker", cuda_process::MAX_FRAME_LEN) => {
                    let payload = payload?;
                    let (reader_action_tx, reader_action_rx) = oneshot::channel();
                    let task_context = request_context.clone();
                    request_tasks.spawn(async move { Self::process_worker_payload(payload, reader_action_tx, task_context).await });

                    match reader_action_rx.await.map_err(|_| ThoError::new("worker request task stopped before releasing the socket reader"))? {
                        WorkerReaderAction::Ready => {}
                        WorkerReaderAction::ReceiveFd(fd_tx) => {
                            let fd_result = recv_fd(reader.as_ref()).await;
                            let fd_error = fd_result.as_ref().err().map(ToString::to_string);
                            if fd_tx.send(fd_result).is_err() {
                                return Err(ThoError::new("worker request task stopped before receiving its shared-memory descriptor"));
                            }
                            if let Some(error) = fd_error {
                                return Err(ThoError::new(format!("failed to receive worker shared-memory descriptor: {error}")));
                            }
                        }
                    }
                    None
                }
            };

            if let Some(WorkerRequestControl::Shutdown { request_id }) = completed {
                break request_id;
            }
        };

        // A shutdown response means every request accepted before shutdown has finished, not merely that shutdown was decoded.
        while let Some(joined) = request_tasks.join_next().await {
            let control = joined.map_err(|error| ThoError::new(format!("worker request task failed during shutdown: {error}")))??;
            if let WorkerRequestControl::Shutdown { request_id } = control {
                return Err(ThoError::new(format!("worker received duplicate shutdown request {request_id}")));
            }
        }

        info!("worker finished");
        reply_tx
            .send(WorkerReply {
                id: shutdown_request_id,
                payload: OutgoingRequestPayload::Msg("dummy".to_string()),
                msg: Some(CudaWorkerMsgOut { request_id: shutdown_request_id, status: OpResult::Ok(None) }),
            })
            .await
            .map_err(|_| ThoError::new("worker response writer is closed"))?;

        // Close the response channel and wait for the writer so the shutdown acknowledgement is on the socket before returning.
        drop(request_context);
        drop(reply_tx);
        writer_task.await.map_err(|error| ThoError::new(format!("worker response writer task failed: {error}")))??;
        debug!("run_worker: exiting run_worker2");
        Ok(())
    }

    async fn process_worker_payload(payload: Vec<u8>, reader_action_tx: oneshot::Sender<WorkerReaderAction>, context: WorkerRequestContext) -> Result<WorkerRequestControl, ThoError> {
        let mut request = match PostcardToFromBytes::from_bytes::<CudaWorkerMsgIn>(&payload) {
            Ok(request) => request,
            Err(error) => {
                let _ = reader_action_tx.send(WorkerReaderAction::Ready);
                error!("postcard: e={error:?}");
                return Ok(WorkerRequestControl::Continue);
            }
        };
        debug!("run_worker: postcard: ok={request:?}");

        if let Operation::SharedMemfd { fd, .. } = &mut request.operation {
            let (fd_tx, fd_rx) = oneshot::channel();
            reader_action_tx.send(WorkerReaderAction::ReceiveFd(fd_tx)).map_err(|_| ThoError::new("worker socket reader stopped before receiving shared-memory descriptor"))?;
            *fd = Some(fd_rx.await.map_err(|_| ThoError::new("worker socket reader dropped shared-memory descriptor response"))??);
        } else {
            reader_action_tx.send(WorkerReaderAction::Ready).map_err(|_| ThoError::new("worker socket reader stopped before request dispatch"))?;
        }

        Self::process_worker_request(request, context).await
    }

    async fn process_worker_request(request: CudaWorkerMsgIn, context: WorkerRequestContext) -> Result<WorkerRequestControl, ThoError> {
        let request_id = request.request_id;
        let op_r: Result<WorkerRequestControl, ThoError> = (async || match request.operation {
            Operation::DescribeRef { tensor_key } => {
                let status = {
                    let tensors = context.all_cuda_tensors.lock();
                    match tensors.get(&tensor_key) {
                        None => OpResult::failed(format!("worker tensor tensor_key={tensor_key} not found")),
                        Some(tensor) => OpResult::Ok(Some(Box::new(WorkerTensorInfoPayload::new(tensor_key, tensor.kind().into(), tensor.size(), tensor_mem_size(tensor.kind(), &tensor.size()))) as Box<dyn WorkerOutput>)),
                    }
                };
                context.reply_tx.send(WorkerReply { id: request_id, payload: OutgoingRequestPayload::Msg("dummy".to_string()), msg: Some(CudaWorkerMsgOut { request_id, status }) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                debug!("run_worker: DescribeRef response {request_id} sent");
                Ok(WorkerRequestControl::Continue)
            }
            Operation::H2dStart { worker_id, kind, shape, tensor_byte_len, chunk_count } => {
                debug!("run_worker: Operation::H2dStart worker_id={worker_id:?} kind={kind:?} shape={shape:?} tensor_byte_len={tensor_byte_len:?} chunk_count={chunk_count}");
                {
                    let mut cuda_resources = context.cuda_resources.lock();
                    panic_to_err!(cuda_process::prepare_worker_h2d_transfer(WorkerH2dStartPayload { transfer_id: 0, worker_id, kind, shape, tensor_byte_len, chunk_count }, &mut cuda_resources))?;
                }
                context.reply_tx.send(WorkerReply { id: request_id, payload: OutgoingRequestPayload::Msg("dummy".to_string()), msg: Some(CudaWorkerMsgOut { request_id, status: OpResult::Ok(None) }) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                debug!("run_worker: Operation::H2dStart done");
                Ok(WorkerRequestControl::Continue)
            }
            Operation::D2HStart { tensor_key } => {
                debug!("run_worker: Operation::D2HStart tensor_key={tensor_key}");
                if !context.all_cuda_tensors.lock().contains_key(&tensor_key) {
                    return Err(ThoError::new(format!("No tensor with tensor_key={tensor_key}")));
                }
                let old = context.pending_d2h_transfers.lock().insert(request_id, PendingWorkerD2hTransfer::new(request_id, tensor_key));
                if old.is_some() {
                    return Err(ThoError::new(format!("duplicate D2H transfer start for transfer_id={request_id}")));
                }
                debug!("run_worker: Operation::D2HStart done");
                Ok(WorkerRequestControl::Continue)
            }
            Operation::SharedMemfd { metadata, fd } => {
                let fd = fd.ok_or_else(|| ThoError::new(format!("shared-memory request {request_id} did not receive a descriptor")))?;
                let buffer = SharedMemfdBuffer::from_received_fd(fd, metadata.mapped_len)?;
                match metadata.direction {
                    IpcTransferDirection::HostToDevice => {
                        debug!("run_worker: IpcTransferDirection::HostToDevice");
                        let chunk = WorkerH2dChunk { metadata, buffer };
                        let cuda_device = cuda_process::validate_worker_h2d_chunk(&chunk)?;
                        let preview_len = chunk.metadata.byte_len.min(16);
                        let preview = unsafe {
                            // SAFETY: validate_worker_h2d_chunk checked offset + byte_len against the live mmap, and preview_len is bounded by byte_len.
                            std::slice::from_raw_parts(chunk.buffer.as_ptr().add(chunk.metadata.offset), preview_len)
                        };
                        debug!("run_worker: received shared memfd H2D tensor chunk: metadata={:?} local_mmap_ptr={:?} preview={preview:?}", chunk.metadata, chunk.buffer.as_ptr());
                        let transfer_id = chunk.metadata.transfer_id;
                        let completed = {
                            let mut pending = context.pending_h2d_transfers.lock();
                            match pending.get_mut(&transfer_id) {
                                Some(transfer) => transfer.push(chunk)?,
                                None => {
                                    pending.insert(transfer_id, PendingWorkerH2dTransfer::new(request_id, chunk)?);
                                }
                            }
                            if pending.get(&transfer_id).is_some_and(PendingWorkerH2dTransfer::is_complete) {
                                Some(pending.remove(&transfer_id).ok_or_else(|| ThoError::new(format!("complete H2D transfer {transfer_id} disappeared")))?)
                            } else {
                                None
                            }
                        };
                        if let Some(transfer) = completed {
                            let resources = cuda_process::get_or_create_worker_cuda_resources(&mut context.cuda_resources.lock(), cuda_device)?;
                            cuda_process::spawn_worker_h2d_copy_task(transfer, resources, Arc::clone(&context.all_cuda_tensors), context.reply_tx.clone(), context.with_digest);
                        }
                    }
                    IpcTransferDirection::DeviceToHost => {
                        debug!("run_worker: IpcTransferDirection::DeviceToHost");
                        let chunk = WorkerD2hChunk { metadata, buffer };
                        let cuda_device = cuda_process::validate_worker_d2h_chunk(&chunk)?;
                        debug!("run_worker: received shared memfd D2H tensor chunk: metadata={:?} local_mmap_ptr={:?}", chunk.metadata, chunk.buffer.as_ptr());
                        let transfer_id = chunk.metadata.transfer_id;
                        let completed = {
                            let mut pending = context.pending_d2h_transfers.lock();
                            pending.get_mut(&transfer_id).ok_or_else(|| ThoError::new(format!("received D2H chunk before start for transfer_id={transfer_id}")))?.push(chunk)?;
                            if pending.get(&transfer_id).is_some_and(PendingWorkerD2hTransfer::is_complete) {
                                Some(pending.remove(&transfer_id).ok_or_else(|| ThoError::new(format!("complete D2H transfer {transfer_id} disappeared")))?)
                            } else {
                                None
                            }
                        };
                        if let Some(transfer) = completed {
                            let resources = cuda_process::get_or_create_worker_cuda_resources(&mut context.cuda_resources.lock(), cuda_device)?;
                            cuda_process::spawn_worker_d2h_copy_task(transfer, resources, Arc::clone(&context.all_cuda_tensors), context.reply_tx.clone(), context.with_digest);
                        }
                    }
                    IpcTransferDirection::GpuToGpuThroughHost => return Err(ThoError::new("worker IPC actor does not handle combined GPU-to-GPU chunks directly")),
                }
                Ok(WorkerRequestControl::Continue)
            }
            Operation::DropThoTensor { tensor_key } => {
                debug!("run_worker: Operation::DropThoTensor: request_id={request_id} tensor_key={tensor_key}");
                let removed = context.all_cuda_tensors.lock().remove(&tensor_key);
                debug!("run_worker: Operation::DropThoTensor: removed={removed:?}");
                let status = if removed.is_none() { OpResult::failed("Not found".to_string()) } else { OpResult::Ok(None) };
                context.reply_tx.send(WorkerReply { id: request_id, payload: OutgoingRequestPayload::Msg("dummy".to_string()), msg: Some(CudaWorkerMsgOut { request_id, status }) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                Ok(WorkerRequestControl::Continue)
            }
            Operation::InstallStage { endpoint_name: key, stage_id, init_data } => {
                let init_fn = context.endpoint_inits.get(&key).ok_or_else(|| ThoError::new(format!("No init function for key={key} stage_id={stage_id}")))?;
                let endpoint_key = cuda_process::endpoint_stage_key(&key, stage_id);
                debug!("run_worker: install endpoint key={endpoint_key} installed_endpoints keys={:?}", context.installed_endpoints.iter().map(|entry| entry.key().clone()).collect::<Vec<_>>());
                let is_duplicate = match context.installed_endpoints.entry(endpoint_key.clone()) {
                    dashmap::mapref::entry::Entry::Occupied(_) => true,
                    dashmap::mapref::entry::Entry::Vacant(entry) => {
                        // Reserve the composite key before initialization so concurrent duplicate installs cannot allocate state twice.
                        entry.insert(None);
                        false
                    }
                };
                if is_duplicate {
                    context.reply_tx.send(WorkerReply { id: request_id, payload: OutgoingRequestPayload::Msg("dummy".to_string()), msg: Some(CudaWorkerMsgOut { request_id, status: OpResult::failed(format!("endpoint {endpoint_key:?} is already installed")) }) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                    return Ok(WorkerRequestControl::Continue);
                }
                let endpoint_wrapper = match init_fn(init_data, Arc::clone(&context.ws), &context.all_cuda_tensors).await {
                    Ok(endpoint_wrapper) => endpoint_wrapper,
                    Err(error) => {
                        context.installed_endpoints.remove(&endpoint_key);
                        return Err(error);
                    }
                };
                debug!("run_worker: install endpoint key={key:?} stage_id={stage_id} endpoint_key={endpoint_key:?} endpoint_wrapper={endpoint_wrapper:?}");
                context.installed_endpoints.insert(endpoint_key, Some(Arc::from(endpoint_wrapper)));
                context.reply_tx.send(WorkerReply { id: request_id, payload: OutgoingRequestPayload::Msg("dummy".to_string()), msg: Some(CudaWorkerMsgOut { request_id, status: OpResult::Ok(None) }) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                Ok(WorkerRequestControl::Continue)
            }
            Operation::InvokeStage { endpoint_name, stage_id, input, consume } => {
                let endpoint_key = cuda_process::endpoint_stage_key(&endpoint_name, stage_id);
                debug!("run_worker: Operation::InvokeStage: endpoint_name={endpoint_name} stage_id={stage_id} input={input:?} consume={consume}");
                let endpoint = context.installed_endpoints.get(&endpoint_key).and_then(|entry| entry.value().as_ref().map(Arc::clone)).ok_or_else(|| ThoError::new(format!("no initialized endpoint for endpoint_key={endpoint_key}")))?;
                let run_ctx = WokerRunCtx::new(context.tho_device.tho_device_idx());
                let stream = context.exec_stream_pool.acquire().await;
                let result = match endpoint.as_with_state() {
                    Some(with_state) => {
                        debug!("run_worker: invoke endpoint endpoint_name={endpoint_name:?} stage_id={stage_id} endpoint_key={endpoint_key:?} as_with_state");
                        // The erased endpoint performs its own typed pre_run before entering the CUDA stream.
                        panic_to_err_async!(with_state.do_run_with_state_2(&input, &context.all_cuda_tensors, &context.tho_device, &stream, &run_ctx))
                    }
                    None => {
                        debug!("run_worker: invoke endpoint endpoint_name={endpoint_name:?} stage_id={stage_id} endpoint_key={endpoint_key:?} no state");
                        panic_to_err!({
                            let _guard = ThoStreamGuard::new(&stream)?;
                            let response = if consume { endpoint.consume_2(input, &context.all_cuda_tensors, &context.tho_device, &run_ctx)? } else { endpoint.do_run_2(&input, &context.all_cuda_tensors, &context.tho_device, &run_ctx)? };
                            stream.synchronize()?;
                            Ok::<_, ThoError>(response)
                        })
                    }
                }?;
                debug!("run_worker: invoke endpoint endpoint_name={endpoint_name:?} stage_id={stage_id} endpoint_key={endpoint_key:?} result={result:?}");
                context.reply_tx.send(WorkerReply { id: request_id, payload: OutgoingRequestPayload::Msg("dummy".to_string()), msg: Some(CudaWorkerMsgOut { request_id, status: OpResult::Ok(Some(result)) }) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                Ok(WorkerRequestControl::Continue)
            }
            Operation::PurgeStage { endpoint_name, stage_id } => {
                let endpoint_key = cuda_process::endpoint_stage_key(&endpoint_name, stage_id);
                debug!("run_worker: purge endpoint: endpoint_name={endpoint_name} stage_id={stage_id} endpoint_key={endpoint_key:?}");
                let endpoint = context.installed_endpoints.get(&endpoint_key).and_then(|entry| entry.value().as_ref().map(Arc::clone));
                if let Some(endpoint) = endpoint {
                    endpoint.purge(context.tho_device.clone()).await?;
                } else {
                    error!("no endpoint found");
                }
                context.reply_tx.send(WorkerReply { id: request_id, payload: OutgoingRequestPayload::Msg("dummy".to_string()), msg: Some(CudaWorkerMsgOut { request_id, status: OpResult::Ok(None) }) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                Ok(WorkerRequestControl::Continue)
            }
            Operation::DropStage { endpoint_name, stage_id } => {
                let endpoint_key = cuda_process::endpoint_stage_key(&endpoint_name, stage_id);
                let removed = context.installed_endpoints.remove(&endpoint_key).is_some();
                debug!("run_worker: drop endpoint: endpoint key={endpoint_key} removed={removed} installed_endpoints keys={:?}", context.installed_endpoints.iter().map(|entry| entry.key().clone()).collect::<Vec<_>>());
                context.reply_tx.send(WorkerReply { id: request_id, payload: OutgoingRequestPayload::Msg("dummy".to_string()), msg: Some(CudaWorkerMsgOut { request_id, status: OpResult::Ok(None) }) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                Ok(WorkerRequestControl::Continue)
            }
            Operation::Shutdown { dummy: _ } => {
                debug!("run_worker: worker got shutdown command request_id={request_id}");
                Ok(WorkerRequestControl::Shutdown { request_id })
            }
        })().await;

        match op_r {
            Ok(control) => Ok(control),
            Err(error) => {
                error!("run_worker: e={error:?}");
                context.reply_tx.send(WorkerReply { id: request_id, payload: OutgoingRequestPayload::Msg("dummy".to_string()), msg: Some(CudaWorkerMsgOut { request_id, status: error.into() }) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                Ok(WorkerRequestControl::Continue)
            }
        }
    }
}

pub trait WorkerStageInput: Debug + Sized + Debug {

    //for cloning a on main-process-side ThoTensor/ThoModule  containing references to the data kept on  worker side
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {//on main side
        error!("clone_refs: self={self:?}");
        todo!()
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, remove: bool) -> Result<Self, ThoError> {//on worker side
        error!("WorkerStageInput::ref_to_tensor: self={self:?}");
        todo!()
    }

    #[deprecated]//the method is to be removed after migration to postcard is done
    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError>;//on worker side

    #[deprecated]//the method is to be removed after migration to postcard is done
    fn decode_for_worker_consuming<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<(Self, Vec<u64>), ThoError> {
        Ok((Self::decode_for_worker::<C>(input, all_cuda_tensors)?, Vec::new()))
    }
}

pub trait WorkerStageOutput: Debug + Sized + Serialize {
    fn encode_from_worker(self, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Self, ThoError>;
}

impl<D, Id> WorkerStageInput for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + DeserializeOwned,
{


    fn ref_to_tensor(&self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, remove: bool) -> Result<Self, ThoError> {
        // Typed worker wrappers receive the tensor reference as a trait object and materialize it only inside the worker.
        Self::ref_to_tensor_(all_cuda_tensors, self, remove)
    }

    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        debug!("ThoTensor::decode_for_worker: input={input:?}");
        let input_ref = C::decode::<Self>(input)?;
        debug!("ThoTensor::decode_for_worker: input_ref={input_ref:?}");

        Self::ref_to_tensor(all_cuda_tensors, input_ref)
    }

    fn decode_for_worker_consuming<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<(Self, Vec<u64>), ThoError> {
        let input_ref = C::decode::<Self>(input)?;
        let input_key = input_ref.ref_key().ok_or_else(|| ThoError::new(format!("worker consuming sync stage expected ThoTensor ref input, got {input_ref:?}")))?;
        Ok((Self::ref_to_tensor(all_cuda_tensors, input_ref)?, vec![input_key]))
    }


}



impl <D,E>WorkerStageOutput for DataWithExtra<D, E>
where
    D: WorkerStageOutput + Data + CopyToDevice + WithId + Sync,
    E: Clone + Serialize + Debug + Send + Sync,
{
    fn encode_from_worker(self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        // Convert the nested worker output before the IPC wrapper serializes the complete value.
        let extra = self.extra().clone();
        Ok(DataWithExtra::new(self.data.encode_from_worker(all_cuda_tensors, tho_device)?, extra))
    }
}

impl<D, Id> ThoTensor<D, Id>
where
    D: Dims,
    Id: Clone + Debug + Named,
{
    pub fn ref_to_tensor(all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, input_ref: ThoTensor<D, Id>) -> Result<ThoTensor<D, Id>, ThoError> {
        Self::ref_to_tensor_(all_cuda_tensors, &input_ref, false)
    }
    pub fn ref_to_tensor_(all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, input_ref: &ThoTensor<D, Id>, remove: bool) -> Result<ThoTensor<D, Id>, ThoError> {//TODO make input_ref &self
        let input_key = input_ref.ref_key().ok_or_else(|| ThoError::new(format!("worker sync stage expected ThoTensor ref input, got {input_ref:?}")))?;

        let mut tensors = all_cuda_tensors.lock();

        debug!("ThoTensor::ref_to_tensor: input_key={input_key:?}");
        trace!("ThoTensor::ref_to_tensor: input_key={input_key:?} keys={:?}", tensors.keys());

        if remove {
            let tensor  = tensors
                .remove(&input_key)
                .ok_or_else(|| ThoError::new(format!("worker sync stage could not find tensor ref key={input_key}")))?;

            ThoTensor::new(input_ref.id().clone(), tensor)
        } else {
            let source = tensors
                .get(&input_key)
                .ok_or_else(|| ThoError::new(format!("worker sync stage could not find tensor ref key={input_key}")))?;

            let mut tensor = source.f_alias()?;

            if let Some(selector) = input_ref.selector() {
                let ranges = selector.tensor_ranges()?;
                if ranges.len() > tensor.dim() {
                    return Err(ThoError::new(format!("selector {selector:?} has {} ranges for a {}-dimensional tensor", ranges.len(), tensor.dim())));
                }
                for (dim, range) in ranges.into_iter().enumerate() {
                    let dim_size = tensor.size()[dim];
                    if range.start < 0 || range.end < range.start {
                        return Err(ThoError::new(format!("selector {selector:?} has invalid range {range:?} for dimension {dim}")));
                    }
                    let end = range.end.min(dim_size);
                    if range.start > end {
                        return Err(ThoError::new(format!("selector {selector:?} starts outside dimension {dim} with size {dim_size}")));
                    }
                    tensor = tensor.f_narrow(dim as i64, range.start, end - range.start)?;
                }
            }

            ThoTensor::new(input_ref.id().clone(), tensor)
        }

    }
}

impl<D, Id> WorkerStageOutput for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Serialize,
{
    fn encode_from_worker(self, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        self.worker_tensor_to_ref(all_cuda_tensors, tho_device)
    }
}

impl<D, Id> ThoTensor<D, Id>
where
    D: Dims,
    Id: Clone + Debug + Named + Serialize,
{
    pub fn worker_tensor_to_ref(self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<ThoTensor<D, Id>, ThoError> {
        let out_id = self.id().clone();
        let out_tensor = self.t();
        let size = out_tensor.size();
        let kind = out_tensor.kind();
        let out_key = insert_worker_tensor(all_cuda_tensors, out_tensor)?;

        Ok(ThoTensor::<D, Id>::new_ref(out_id, out_key, size, kind, None, tho_device.tho_device_idx()))
    }
}

pub fn insert_worker_tensor(all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tensor: Tensor) -> Result<u64, ThoError> {
    let mut tensors = all_cuda_tensors.lock();
    loop {
        let key = NEXT_WORKER_TENSOR_KEY
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |key| key.checked_add(1))
            .map_err(|_| ThoError::new("worker tensor key space exhausted"))?;

        debug!("insert_worker_tensor: key={key}");

        if let std::collections::hash_map::Entry::Vacant(entry) = tensors.entry(key) {
            entry.insert(tensor);
            return Ok(key);
        }
    }
}



#[derive(Debug)]
pub struct WokerRunCtx {
    tho_device_idx: usize,
}


impl WokerRunCtx {
    pub fn new(tho_device_idx: usize) -> Self {
        Self { tho_device_idx }
    }
}

impl Named for WokerRunCtx {}


impl WithId for WokerRunCtx{
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl RunCtx for WokerRunCtx {
    fn tho_device_idx(&self) -> usize {
        self.tho_device_idx
    }

    fn branch(&mut self, with_new_cache_area: bool) -> Self
    where
        Self: Sized
    {
        WokerRunCtx::new(self.tho_device_idx)
    }

    fn add_purge_gpu_closure(&mut self, cl: Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>) {
    }
}
