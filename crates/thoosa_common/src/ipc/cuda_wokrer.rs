use crate::cuda_utils::{copy_h2d_from_ptr, cuda_mem_for_device, tensor_mem_size};
use crate::data_with_extra::DataWithExtra;
use crate::dims::{AnyDims, Dims};
use crate::ipc::codec::IpcCodec;
use crate::ipc::cuda_process;
use crate::ipc::cuda_process::{DeviceInfo, IPC_SOCK_ENV, PendingWorkerD2hTransfer, PendingWorkerH2dTransfer, WorkerD2hChunk, WorkerH2dChunk, WorkerH2dStartPayload, WorkerTensorInfoPayload};
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
use std::fs::File;
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
use memmap2::{MmapMut, MmapOptions};
use thoosa_shared::tracked_await::track_await;
use crate::copy_ctx::CopyCtx;

const WORKER_EXEC_STREAMS: usize = 32;
static NEXT_WORKER_TENSOR_KEY: AtomicU64 = AtomicU64::new(1);

//1 cuda device per worker
pub struct CudaWorker;

pub type InitFn = Box<dyn Fn(Box<dyn WorkerInput>, Arc<RwLock<Box<dyn Workspace>>>, &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<Result<Box<dyn SyncStageIpcWrapper>, ThoError>> + Send + Sync>;

type InstalledEndpoints = Arc<DashMap<String, Option<Arc<dyn SyncStageIpcWrapper>>>>;

#[derive(Clone)]
struct WorkerRequestContext {
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    endpoint_inits: Arc<HashMap<String, InitFn>>,
    exec_stream_pool: Arc<Pool<ThoCudaStream>>,
    all_cuda_tensors: Arc<Mutex<HashMap<u64, Tensor>>>,
    pending_h2d_transfers: Arc<Mutex<HashMap<u64, PendingWorkerH2dTransfer>>>,
    // Reuse each shared-memory pool mapping and its CUDA host registration instead of paying mmap and page-pinning costs for every tensor copy.
    copy_buffers: Arc<Mutex<HashMap<u64, Arc<SharedMemfdBuffer>>>>,
    // A worker is bound to one CUDA device, so copy streams need one pool rather than a device-keyed map and lock on every chunk.
    copy_stream_pool: Arc<Pool<ThoCudaStream>>,
    installed_endpoints: InstalledEndpoints,
    reply_tx: mpsc::Sender<CudaWorkerMsgOut>,
    tho_device: ThoDevice,
    copy_ctx:Arc<tokio::sync::RwLock<CopyCtx>>
}

enum WorkerReaderAction {
    Ready,
    ReceiveFd(oneshot::Sender<std::io::Result<OwnedFd>>),
    Shutdown { request_id: u64 },
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

    pub fn run_if_sock(cdt:&str, app_configs:&WsConfigs, init_map: HashMap<String, InitFn>) -> Result<(), ThoError>{

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
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    init_map,
                    ThoDevice::from(Device::Cuda(0)),
                ).await//TODO Cuda(0)
            })?;

            std::process::exit(0);
        }

        Ok(())
    }

    pub async fn run_worker(
        sock_path:OsString,
        ws:&Arc<RwLock<Box<dyn Workspace>>>,
        endpoint_inits: HashMap<String, InitFn>,
        tho_device: ThoDevice //handles execution on single Cuda device
    ) -> Result<(), ThoError> {
        info!("run_worker: sock_path={sock_path:?} tho_device={tho_device:?} endpoint_inits.keys={:?}", endpoint_inits.keys().collect::<Vec<_>>());

        // Stateful capability is erased per endpoint so one worker can host unrelated state types.
        GlobalCtx::init_devices(2)?;//TODO config

        let cuda_device = tho_device.gpu_ordinal();

        let exec_stream_pool = Arc::new(Pool::from_iter(
            (0..WORKER_EXEC_STREAMS)
                .map(|_| ThoCudaStream::from_pool(cuda_device as i32, false))
                .collect::<Result<Vec<_>, _>>()?,
        ));
        let copy_stream_pool = Arc::new(Pool::from_iter(
            (0..cuda_process::WORKER_COPY_STREAMS)
                .map(|_| ThoCudaStream::from_pool(cuda_device as i32, true))
                .collect::<Result<Vec<_>, _>>()?,
        ));



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

        let arena_fd = recv_fd(reader.as_ref()).await?;
        debug!("run_worker: arena_fd={arena_fd:?}");
        let file = File::from(arena_fd);


        let arena_size = file.metadata()?.len() as usize;

        let mut arena = unsafe { MmapOptions::new().len(arena_size).map_mut(&file)? };
        debug!("run_worker: arena={arena:?}");
        let arena_ptr = arena.as_mut_ptr() as usize;

        let (reply_tx, reply_rx) = mpsc::channel::<CudaWorkerMsgOut>(1024);

        let writer_task = tokio::spawn(cuda_process::worker_response_writer_loop(writer, reply_rx));
        let all_cuda_tensors = Arc::new(parking_lot::Mutex::new(HashMap::new()));//TODO to RwLock?
        let request_context = WorkerRequestContext {
            ws: Arc::clone(ws),
            endpoint_inits: Arc::new(endpoint_inits),
            exec_stream_pool,
            all_cuda_tensors,
            pending_h2d_transfers: Arc::new(Mutex::new(HashMap::new())),
            copy_buffers: Arc::new(Mutex::new(HashMap::new())),
            copy_stream_pool,
            installed_endpoints: Arc::new(DashMap::new()),
            reply_tx: reply_tx.clone(),
            tho_device,

            copy_ctx: Arc::new(tokio::sync::RwLock::new(CopyCtx::make_instance_for_cuda_device(cuda_device, 0.2, 16)?))
        };
        let mut request_tasks = JoinSet::new();

        // Always finish each frame read once started: read_exact is not cancellation-safe, so selecting it against a completed request task can leave the stream cursor inside a frame.
        let shutdown_request_id = loop {
            let payload = cuda_process::read_frame(&mut reader, "run_worker", cuda_process::MAX_FRAME_LEN).await?;
            let (reader_action_tx, reader_action_rx) = oneshot::channel();
            let task_context = request_context.clone();
            request_tasks.spawn(async move {

                Self::process_worker_payload(payload, reader_action_tx, task_context, arena_ptr).await });

            let shutdown = match reader_action_rx.await.map_err(|_| ThoError::new("worker request task stopped before releasing the socket reader"))? {
                WorkerReaderAction::Ready => None,
                WorkerReaderAction::ReceiveFd(fd_tx) => {
                    let fd_result = recv_fd(reader.as_ref()).await;
                    let fd_error = fd_result.as_ref().err().map(ToString::to_string);
                    if fd_tx.send(fd_result).is_err() {
                        return Err(ThoError::new("worker request task stopped before receiving its shared-memory descriptor"));
                    }
                    if let Some(error) = fd_error {
                        return Err(ThoError::new(format!("failed to receive worker shared-memory descriptor: {error}")));
                    }
                    None
                }
                WorkerReaderAction::Shutdown { request_id } => Some(request_id),
            };

            if let Some(request_id) = shutdown {
                break request_id;
            }

            // Reap tasks only between complete frames so doing so can never cancel an in-progress socket read.
            while let Some(joined) = request_tasks.try_join_next() {
                let control = joined.map_err(|error| ThoError::new(format!("worker request task failed: {error}")))??;
                if let WorkerRequestControl::Shutdown { request_id } = control {
                    return Err(ThoError::new(format!("worker request {request_id} completed as shutdown without stopping the socket reader")));
                }
            }
        };

        // A shutdown response means every request accepted before shutdown has finished, not merely that shutdown was decoded.
        let mut shutdown_task_finished = false;
        while let Some(joined) = request_tasks.join_next().await {
            let control = joined.map_err(|error| ThoError::new(format!("worker request task failed during shutdown: {error}")))??;
            if let WorkerRequestControl::Shutdown { request_id } = control {
                if request_id != shutdown_request_id || shutdown_task_finished {
                    return Err(ThoError::new(format!("worker received unexpected shutdown request {request_id}")));
                }
                shutdown_task_finished = true;
            }
        }
        if !shutdown_task_finished {
            return Err(ThoError::new(format!("worker shutdown request {shutdown_request_id} did not finish")));
        }


        reply_tx
            .send(CudaWorkerMsgOut { request_id: shutdown_request_id, status: OpResult::Ok(None) })
            .await
            .map_err(|_| ThoError::new("worker response writer is closed"))?;

        // Close the response channel and wait for the writer so the shutdown acknowledgement is on the socket before returning.
        drop(request_context);
        drop(reply_tx);
        writer_task.await.map_err(|error| ThoError::new(format!("worker response writer task failed: {error}")))??;
        info!("worker finished");
        Ok(())
    }

    async fn process_worker_payload(payload: Vec<u8>, reader_action_tx: oneshot::Sender<WorkerReaderAction>, context: WorkerRequestContext, arena_ptr: usize) -> Result<WorkerRequestControl, ThoError> {
        let mut request = match PostcardToFromBytes::from_bytes::<CudaWorkerMsgIn>(&payload) {
            Ok(request) => request,
            Err(error) => {
                let _ = reader_action_tx.send(WorkerReaderAction::Ready);
                error!("postcard: e={error:?}");
                return Ok(WorkerRequestControl::Continue);
            }
        };
        debug!("run_worker: postcard: ok={request:?}");

        match &mut request.operation {
            Operation::SharedMemfd { fd, .. } => {
                let (fd_tx, fd_rx) = oneshot::channel();
                reader_action_tx.send(WorkerReaderAction::ReceiveFd(fd_tx)).map_err(|_| ThoError::new("worker socket reader stopped before receiving shared-memory descriptor"))?;
                *fd = Some(fd_rx.await.map_err(|_| ThoError::new("worker socket reader dropped shared-memory descriptor response"))??);
            }
            Operation::Shutdown { .. } => {
                // Stop the sole reader before it waits for another frame; the shutdown task itself is still drained with all previously accepted work.
                reader_action_tx.send(WorkerReaderAction::Shutdown { request_id: request.request_id }).map_err(|_| ThoError::new("worker socket reader stopped before shutdown dispatch"))?;
            }
            _ => {
                reader_action_tx.send(WorkerReaderAction::Ready).map_err(|_| ThoError::new("worker socket reader stopped before request dispatch"))?;
            }
        }

        Self::process_worker_request(request, context, arena_ptr).await
    }

    async fn process_worker_request(request: CudaWorkerMsgIn, context: WorkerRequestContext, arena_ptr: usize) -> Result<WorkerRequestControl, ThoError> {
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
                context.reply_tx.send(CudaWorkerMsgOut { request_id, status }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                debug!("run_worker: DescribeRef response {request_id} sent");
                Ok(WorkerRequestControl::Continue)
            }
            Operation::H2dStart { worker_id, kind, shape, tensor_byte_len, chunk_count } => {
                debug!("run_worker: Operation::H2dStart worker_id={worker_id:?} kind={kind:?} shape={shape:?} tensor_byte_len={tensor_byte_len:?} chunk_count={chunk_count} arena_ptr={arena_ptr}");
                let cuda_device = context.tho_device.gpu_ordinal();

                let tensor_ptr =
                    unsafe {
                        (arena_ptr as *mut u8)
                            .add(chunk_count)
                    };

                // debug!("run_worker: Operation::H2dStart tensor_ptr={tensor_ptr:?}");



                // let t =         unsafe {
                //     Tensor::from_blob(
                //         tensor_ptr,
                //         &shape,
                //         &[],
                //         kind.into(),
                //         Device::Cpu,
                //     )
                // };
                // debug!("run_worker: t=\n{t}");

                // let tt = ThoTensor::<AnyDims, ()>::new((), t)?;
                //
                // let cuda_t = context.copy_ctx.read().await.copy(tt.as_ref(), 1, kind.into()).await?;
                let cuda_t = context.copy_ctx.read().await.copy_from_ptr(
                    unsafe {
                        (arena_ptr as *mut u8)
                            .add(chunk_count) as usize
                    }, &shape, 1, kind.into()).await?;
                debug!("run_worker: cuda_t=\n{cuda_t}, cuda_t.device={:?}", cuda_t.device());

                // let transfer = panic_to_err!(cuda_process::prepare_worker_h2d_transfer(WorkerH2dStartPayload { transfer_id: request_id, worker_id, kind, shape, tensor_byte_len, chunk_count }, cuda_device))?;
                // // Retain preflight metadata so the ordered start is validated once; CUDA allocation remains bounded by completed staged transfers.
                // let old = context.pending_h2d_transfers.lock().insert(request_id, transfer);
                // if old.is_some() {
                //     return Err(ThoError::new(format!("duplicate H2D transfer start for transfer_id={request_id}")));
                // }
                // context.reply_tx.send(CudaWorkerMsgOut { request_id, status: OpResult::Ok(None) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                
                debug!("run_worker: Operation::H2dStart done");
                Ok(WorkerRequestControl::Continue)
            }
            Operation::SharedMemfd { metadata, tensor_key, fd } => {
                let fd = fd.ok_or_else(|| ThoError::new(format!("shared-memory request {request_id} did not receive a descriptor")))?;
                let buffer = if let Some(buffer) = context.copy_buffers.lock().get(&metadata.buffer_id).cloned() {
                    if buffer.len() != metadata.mapped_len {
                        return Err(ThoError::new(format!("shared-memory buffer {} changed mapped length from {} to {}", metadata.buffer_id, buffer.len(), metadata.mapped_len)));
                    }
                    buffer
                } else {
                    let buffer = Arc::new(SharedMemfdBuffer::from_received_fd(metadata.buffer_id, fd, metadata.mapped_len)?);
                    context.copy_buffers.lock().insert(metadata.buffer_id, Arc::clone(&buffer));
                    buffer
                };
                match metadata.direction {
                    IpcTransferDirection::HostToDevice => {
                        if tensor_key.is_some() {
                            return Err(ThoError::new(format!("H2D shared-memory request {request_id} unexpectedly contains a tensor key")));
                        }
                        debug!("run_worker: IpcTransferDirection::HostToDevice");
                        let chunk = WorkerH2dChunk { metadata, buffer };
                        let cuda_device = context.tho_device.gpu_ordinal();
                        cuda_process::validate_worker_h2d_chunk(&chunk, cuda_device)?;
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
                            cuda_process::spawn_worker_h2d_copy_task(transfer, Arc::clone(&context.copy_stream_pool), cuda_device, Arc::clone(&context.all_cuda_tensors), context.reply_tx.clone());
                        }
                    }
                    IpcTransferDirection::DeviceToHost => {
                        debug!("run_worker: IpcTransferDirection::DeviceToHost");
                        let chunk = WorkerD2hChunk { metadata, buffer };
                        let cuda_device = context.tho_device.gpu_ordinal();
                        cuda_process::validate_worker_d2h_chunk(&chunk, cuda_device)?;
                        debug!("run_worker: received shared memfd D2H tensor chunk: metadata={:?} local_mmap_ptr={:?}", chunk.metadata, chunk.buffer.as_ptr());
                        let tensor_key = tensor_key.ok_or_else(|| ThoError::new(format!("D2H shared-memory request {request_id} is missing its tensor key")))?;
                        let mut transfer = PendingWorkerD2hTransfer::new(request_id, tensor_key);
                        transfer.push(chunk)?;
                        if !transfer.is_complete() {
                            return Err(ThoError::new(format!("D2H shared-memory request {request_id} did not contain a complete chunk transfer")));
                        }
                        cuda_process::spawn_worker_d2h_copy_task(transfer, Arc::clone(&context.copy_stream_pool), cuda_device, Arc::clone(&context.all_cuda_tensors), context.reply_tx.clone());
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
                context.reply_tx.send(CudaWorkerMsgOut { request_id, status }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
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
                    context.reply_tx.send(CudaWorkerMsgOut { request_id, status: OpResult::failed(format!("endpoint {endpoint_key:?} is already installed")) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
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
                context.reply_tx.send(CudaWorkerMsgOut { request_id, status: OpResult::Ok(None) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
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
                            let response = if consume { endpoint.consume(input, &context.all_cuda_tensors, &context.tho_device, &run_ctx)? } else { endpoint.do_run(&input, &context.all_cuda_tensors, &context.tho_device, &run_ctx)? };
                            stream.synchronize()?;
                            Ok::<_, ThoError>(response)
                        })
                    }
                }?;
                debug!("run_worker: invoke endpoint endpoint_name={endpoint_name:?} stage_id={stage_id} endpoint_key={endpoint_key:?} result={result:?}");
                context.reply_tx.send(CudaWorkerMsgOut { request_id, status: OpResult::Ok(Some(result)) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
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
                context.reply_tx.send(CudaWorkerMsgOut { request_id, status: OpResult::Ok(None) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                Ok(WorkerRequestControl::Continue)
            }
            Operation::DropStage { endpoint_name, stage_id } => {
                let endpoint_key = cuda_process::endpoint_stage_key(&endpoint_name, stage_id);
                let removed = context.installed_endpoints.remove(&endpoint_key).is_some();
                debug!("run_worker: drop endpoint: endpoint key={endpoint_key} removed={removed} installed_endpoints keys={:?}", context.installed_endpoints.iter().map(|entry| entry.key().clone()).collect::<Vec<_>>());
                context.reply_tx.send(CudaWorkerMsgOut { request_id, status: OpResult::Ok(None) }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
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
                context.reply_tx.send(CudaWorkerMsgOut { request_id, status: error.into() }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
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
