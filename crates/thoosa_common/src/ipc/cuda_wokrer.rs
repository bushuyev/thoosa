use crate::copy_ctx::CopyCtx;
use crate::cuda_utils::{cuda_mem_for_device, tensor_mem_size};
use crate::data_with_extra::DataWithExtra;
use crate::dims::Dims;
use crate::ipc::cuda_process;
use crate::ipc::cuda_process::{DeviceInfo, IPC_SOCK_ENV };
use crate::ipc::ipc_cuda_utils::recv_fd;
use crate::ipc::proxy_tests::TestOutputA;
use crate::ipc::serialization::json::IpcCodec;
use crate::ipc::serialization::{PostcardToFromBytes, ToFromBytes};
use crate::ipc::worker_msg::{CudaWorkerId};
use crate::ipc::worker_msg::{CudaWorkerMsgIn, CudaWorkerMsgOut, DebugOutput, OpResult, Operation, WorkerCopyDonePayload, WorkerInput};
use crate::ipc::wraps::{ErasedWithState, SyncStageIpcWrapper, SyncStageIpcWrapperWithState, WorkerWrapperWithState};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::{ThoTensor, ThoTensorInner};
use crate::ws::{FsWorkspace, Workspace, WsConfigs};
use anyhow::Context as _;
use clap::Parser;
use dashmap::DashMap;
use futures_util::future::BoxFuture;
use log::{LevelFilter, debug, error, info, trace};
use memmap2::{MmapMut, MmapOptions};
use parking_lot::{Mutex, RwLock};
use rand::{Rng, rng};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Serializer;
use std::collections::HashMap;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Debug;
use std::fs::File;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, OnceLock};
use tch::{Device, Tensor};
use thoosa_shared::ipc_types::{H2DTensorKey, IpcTensorKind, TensorKey, WorkerOutput, WorkerTensorInfoPayload};
use thoosa_shared::perf_logging::{PERF, format_perf_map, init_tracing};
use thoosa_shared::traits::{Adjustable, Data, MaxParallelCtx, Named, RunCtx, StageStateInfo, WithId};
use thoosa_shared::{CopyToDevice, ThoError, panic_to_err, panic_to_err_async};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tokio::task::JoinSet;
use tracing::field::debug;
use tub::Pool;



const WORKER_EXEC_STREAMS: usize = 32;
static NEXT_WORKER_TENSOR_KEY: AtomicU64 = AtomicU64::new(1_000_000);//if starts from 0 it's difficult to find them in logs


pub static DEBUG_DATA:LazyLock<Arc<Mutex<Vec<String>>>> = LazyLock::new(|| Arc::new(Mutex::new(vec![])));

//1 cuda device per worker
pub struct CudaWorker;


//stage_id - None for stateless stages, init_data, ctx, all_tensors, device
pub type InitFn<W = dyn Workspace> = Box<dyn for <'a> Fn(Option<String>, Box<dyn WorkerInput>, Arc<RwLock<Box<W>>>, &'a AllTensors, &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> + Send + Sync>;

type InstalledStages = Arc<DashMap<String, Option<Arc<dyn SyncStageIpcWrapper>>>>;

struct WorkerSharedArena {
    mapping: MmapMut,
}

impl WorkerSharedArena {
    fn address(&self, arena_offset: usize, byte_len: usize, label: &str) -> Result<usize, ThoError> {
        let arena_end = arena_offset.checked_add(byte_len).ok_or_else(|| ThoError::new(format!("{label} arena range overflow")))?;
        if arena_end > self.mapping.len() {
            return Err(ThoError::new(format!("{label} arena range [{arena_offset}, {arena_end}) exceeds worker mapping length {}", self.mapping.len())));
        }
        self.mapping.as_ptr().addr().checked_add(arena_offset).ok_or_else(|| ThoError::new(format!("{label} arena address overflow")))
    }
}

#[derive(Debug)]
pub struct AllTensors {
    pub worker_id: CudaWorkerId,
    // Tensor values and their aliases are treated as read-only while stored; the write lock only protects map membership.
    tensors: RwLock<HashMap<u64, Tensor>>,
}

impl AllTensors {
    pub fn new(worker_id: CudaWorkerId) -> Self {
        Self { worker_id, tensors: RwLock::new(HashMap::new()) }
    }
}

// SAFETY: every map access is synchronized, no mutable tensor reference is exposed, and callers preserve the read-only-storage invariant when using aliases.
unsafe impl Sync for AllTensors {}
struct WorkerRequestContext<W: ?Sized> {
    ws: Arc<RwLock<Box<W>>>,
    endpoint_inits: Arc<HashMap<String, InitFn<W>>>,
    exec_stream_pool: Arc<Pool<ThoCudaStream>>,
    all_cuda_tensors: Arc<AllTensors>,
    // Retaining the mapping in every request context keeps all arena-derived pointers valid until in-flight copies finish.
    shared_arena: Arc<WorkerSharedArena>,
    // CUDA copies still use the worker's pinned-buffer implementation; only the old per-transfer IPC memfds are replaced by arena offsets.
    copy_ctx: Arc<tokio::sync::RwLock<CopyCtx>>,
    installed_stages: InstalledStages,
    reply_tx: mpsc::Sender<CudaWorkerMsgOut>,
    tho_device: ThoDevice,
}

impl<W: ?Sized> Clone for WorkerRequestContext<W> {
    fn clone(&self) -> Self {
        Self {
            ws: Arc::clone(&self.ws),
            endpoint_inits: Arc::clone(&self.endpoint_inits),
            exec_stream_pool: Arc::clone(&self.exec_stream_pool),
            all_cuda_tensors: Arc::clone(&self.all_cuda_tensors),
            shared_arena: Arc::clone(&self.shared_arena),
            copy_ctx: Arc::clone(&self.copy_ctx),
            installed_stages: Arc::clone(&self.installed_stages),
            reply_tx: self.reply_tx.clone(),
            tho_device: self.tho_device,
        }
    }
}

pub trait Init {
    fn init() -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>;
}

fn validate_arena_tensor_metadata(kind: IpcTensorKind, shape: &[i64], tensor_byte_len: usize, arena_offset: usize, shared_arena: &WorkerSharedArena, label: &str) -> Result<(), ThoError> {
    let expected_byte_len = tensor_mem_size(kind.into(), shape);
    if expected_byte_len != tensor_byte_len {
        return Err(ThoError::new(format!("{label} byte length {tensor_byte_len} does not match kind/shape size {expected_byte_len}")));
    }
    shared_arena.address(arena_offset, tensor_byte_len, label)?;
    Ok(())
}

fn prepare_d2h_source(all_cuda_tensors: &AllTensors, tensor_key: u64, kind: IpcTensorKind, shape: &[i64], cuda_device: usize) -> Result<Tensor, ThoError> {
    let tensors = all_cuda_tensors.tensors.read();
    let source = tensors.get(&tensor_key).ok_or_else(|| ThoError::new(format!("worker tensor tensor_key={tensor_key} not found")))?;
    let tensor_kind = kind.into();
    if source.kind() != tensor_kind || source.size() != shape {
        return Err(ThoError::new(format!("worker tensor key={tensor_key} metadata changed before D2H copy: kind={:?} shape={:?}", source.kind(), source.size())));
    }
    if !source.is_contiguous() {
        return Err(ThoError::new(format!("worker tensor key={tensor_key} is not contiguous")));
    }
    let Device::Cuda(source_cuda_device) = source.device() else {
        return Err(ThoError::new(format!("worker tensor key={tensor_key} is not on CUDA: {:?}", source.device())));
    };
    if source_cuda_device != cuda_device {
        return Err(ThoError::new(format!("worker tensor key={tensor_key} device={source_cuda_device} does not match worker CUDA device={cuda_device}")));
    }
    // A fallible alias keeps the CUDA storage alive after releasing the tensor map while CopyCtx awaits its chunk stages.
    Ok(source.f_alias()?)
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

    pub fn run_if_sock(cdt:&str, app_configs:&WsConfigs, init_map: HashMap<String, InitFn>) -> anyhow::Result<()>{
        Self::run_if_sock_with_workspace(cdt, app_configs, init_map, FsWorkspace::new)
    }

    /// Lets feature crates construct their extended workspace trait object inside the CUDA worker.
    pub fn run_if_sock_with_workspace<W, F>(cdt: &str, app_configs: &WsConfigs, init_map: HashMap<String, InitFn<W>>, workspace_factory: F) -> anyhow::Result<()>
    where
        W: Workspace + ?Sized + 'static,
        F: FnOnce(PathBuf, WsConfigs) -> anyhow::Result<Box<W>>,
    {

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
                let ws = Arc::new(RwLock::new(workspace_factory(ws_root.clone(), app_configs.clone())?));

                // Application workers use only postcard/typetag endpoint initializers after the proxy migration.
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    init_map,
                    ThoDevice::from(Device::Cuda(0)),
                ).await//TODO Cuda(0)
            })?;

            info!("perf: \n{}", format_perf_map(&PERF));

            std::process::exit(0);
        }

        Ok(())
    }

    pub async fn run_worker<W>(
        sock_path:OsString,
        ws:&Arc<RwLock<Box<W>>>,
        endpoint_inits: HashMap<String, InitFn<W>>,
        tho_device: ThoDevice //handles execution on single Cuda device
    ) -> anyhow::Result<()>
    where
        W: Workspace + ?Sized + 'static,
    {
        info!("run_worker: sock_path={sock_path:?} tho_device={tho_device:?} endpoint_inits.keys={:?}", endpoint_inits.keys().collect::<Vec<_>>());

        // Stateful capability is erased per endpoint so one worker can host unrelated state types.
        // GlobalCtx::init_devices(2)?;//TODO config

        // Identity comes from the parent before requests are accepted, never
        // from a tensor's numeric key or a request supplied by the caller.
        let worker_id: CudaWorkerId = serde_json::from_str(&std::env::var(cuda_process::WORKER_ID_ENV).context("missing CUDA worker identity")?)?;
        let cuda_device = tho_device.gpu_ordinal();

        let exec_stream_pool = Arc::new(Pool::from_iter(
            (0..WORKER_EXEC_STREAMS)
                .map(|_| ThoCudaStream::from_pool(cuda_device as i32, false))
                .collect::<Result<Vec<_>, _>>()?,
        ));
        let stream = UnixStream::connect(PathBuf::from(sock_path)).await?;

        debug!("run_worker: worker connected");

        let (mut reader, mut writer) = stream.into_split();

        let (free_mem, total_mem) = cuda_mem_for_device(cuda_device);

        let arena_fd = recv_fd(reader.as_ref()).await?;
        debug!("run_worker: arena_fd={arena_fd:?}");
        let file = File::from(arena_fd);


        let arena_size = file.metadata()?.len() as usize;

        let mut arena = unsafe { MmapOptions::new().len(arena_size).map_mut(&file)? };
       /* unsafe {
            assert_eq!(libc::madvise(arena.as_mut_ptr().cast(), arena.len(), libc::MADV_RANDOM), 0);
            assert_eq!(libc::madvise(arena.as_mut_ptr().cast(), arena.len(), libc::MADV_HUGEPAGE), 0);
        };*/


        // The descriptor can close after mmap; the mapping itself remains shared and is retained by the request context.
        let shared_arena = Arc::new(WorkerSharedArena { mapping: arena });
        debug!("run_worker: mapped shared arena len={arena_size}");

        let copy_ctx = Arc::new(tokio::sync::RwLock::new(CopyCtx::make_instance_for_cuda_device(cuda_device, 0.2, 32)?));
        // Readiness follows all fallible device and arena initialization.
        cuda_process::write_frame(
            &mut writer,
            &PostcardToFromBytes::to_bytes(&DeviceInfo {
                worker_id,
                free_mem,
                total_mem
            })?,
            "DeviceInfo/worker"
        ).await?;

        let (reply_tx, reply_rx) = mpsc::channel::<CudaWorkerMsgOut>(1024);

        let writer_task = tokio::spawn(cuda_process::worker_response_writer_loop(writer, reply_rx));
        let all_cuda_tensors = Arc::new(AllTensors::new(worker_id));
        let request_context = WorkerRequestContext {
            ws: Arc::clone(ws),
            endpoint_inits: Arc::new(endpoint_inits),
            exec_stream_pool,
            all_cuda_tensors,
            shared_arena,
            copy_ctx,
            installed_stages: Arc::new(DashMap::new()),
            reply_tx: reply_tx.clone(),
            tho_device,
        };
        let mut request_tasks = JoinSet::new();

        // Always finish each frame read once started: read_exact is not cancellation-safe, so selecting it against a completed request task can leave the stream cursor inside a frame.
        let shutdown_request_id = loop {
            let payload = cuda_process::read_frame(&mut reader, "run_worker", cuda_process::MAX_FRAME_LEN).await?;
            let request = match PostcardToFromBytes::from_bytes::<CudaWorkerMsgIn>(&payload) {
                Ok(request) => request,
                Err(error) => {
                    error!("postcard: e={error:?}");
                    continue;
                }
            };
            debug!("run_worker: postcard: ok={request:?}");

            if let Err(error) = request.worker_id.check(worker_id) {
                reply_tx.send(CudaWorkerMsgOut { worker_id, request_id: request.request_id, status: error.into() }).await?;
                continue;
            }

            // Shutdown is loop control rather than a request task: stop accepting frames, then drain all work accepted before it.
            if matches!(&request.operation, Operation::Shutdown { .. }) {
                let request_id = request.request_id;
                info!("run_worker: Operation::Shutdown request_id={request_id}");
                info!("run_worker: break loop on shutdown");
                break request_id;
            }

            let task_context = request_context.clone();
            request_tasks.spawn(async move { Self::process_worker_request(request, task_context).await });

            // Reap stages only between complete frames so doing so can never cancel an in-progress socket read.
            while let Some(joined) = request_tasks.try_join_next() {
                joined.context("worker request task failed")??;
            }
        };

        // A shutdown response means every request accepted before shutdown has finished, not merely that shutdown was decoded.
        while let Some(joined) = request_tasks.join_next().await {
            joined.context("worker request task failed during shutdown")??;
        }

        reply_tx
            .send(CudaWorkerMsgOut { worker_id, request_id: shutdown_request_id, status: OpResult::Ok(None) })
            .await
            .context("worker response writer is closed")?;

        // Close the response channel and wait for the writer so the shutdown acknowledgement is on the socket before returning.
        drop(request_context);
        drop(reply_tx);
        writer_task.await.context("worker response writer task failed")??;
        info!("worker finished");
        Ok(())
    }

    async fn process_worker_request<W: Workspace + ?Sized + 'static>(request: CudaWorkerMsgIn, context: WorkerRequestContext<W>) -> anyhow::Result<()> {
        let request_id = request.request_id;
        let worker_id = context.all_cuda_tensors.worker_id;
        let op_r: Result<(), ThoError> = (async || {
            request.worker_id.check(worker_id)?;
            match request.operation {
                Operation::DescribeRef { tensor_key } => {
                    let status = {
                        let tensors = context.all_cuda_tensors.tensors.read();
                        match tensors.get(&tensor_key) {
                            None => OpResult::failed(format!("worker tensor tensor_key={tensor_key} not found")),
                            Some(tensor) => OpResult::Ok(Some(Box::new(WorkerTensorInfoPayload::new(tensor_key.into(), tensor.kind().into(), tensor.size(), tensor_mem_size(tensor.kind(), &tensor.size()))) as Box<dyn WorkerOutput>)),
                        }
                    };
                    context.reply_tx.send(CudaWorkerMsgOut { worker_id, request_id, status }).await.context("worker response writer is closed")?;
                    debug!("run_worker: DescribeRef response {request_id} sent");
                    Ok(())
                }
                Operation::H2dCopy { tensor_key, kind, shape, tensor_byte_len, arena_offset, dst_key } => {
                    debug!("run_worker: Operation::H2dCopy kind={kind:?} shape={shape:?} tensor_byte_len={tensor_byte_len:?} arena_offset={arena_offset}");

                    let cuda_device = context.tho_device.gpu_ordinal();
                    validate_arena_tensor_metadata(kind, &shape, tensor_byte_len, arena_offset, &context.shared_arena, "H2D source")?;

                    let source_address = context.shared_arena.address(arena_offset, tensor_byte_len, "H2D source")?;
                    let cuda_t = context.copy_ctx.read().await.copy_from_ptr(source_address, &shape, context.tho_device.tho_device_idx(), kind.into()).await?;

                    let tensor_ref = if let Some(H2DTensorKey::StageState { endpoint_name, stage_id, tensor_id }) = dst_key {
                        let endpoint_key = cuda_process::endpoint_stage_key(endpoint_name.clone(), Some(&stage_id));

                        let endpoint: Option<Arc<dyn SyncStageIpcWrapper>> = context.installed_stages.get(&endpoint_key).and_then(|entry| entry.value().as_ref().map(Arc::clone));


                        endpoint
                            .expect(&format!("no endpoint {endpoint_name:?}"))
                            .as_with_state()
                            .unwrap()
                            .set_state_tensor(&context.tho_device, tensor_id.clone(), cuda_t).await?;

                        TensorKey::StageState {
                            endpoint_name,
                            stage_id,
                            tensor_key: tensor_id,
                        }
                    } else {
                        insert_worker_tensor(&context.all_cuda_tensors, cuda_t)?.into()
                    };

                    debug!("process_worker_request: Operation::H2dCopy: tensor_ref={tensor_ref:?} request.request_id={:?}", request.request_id);

                    context.reply_tx.send(CudaWorkerMsgOut {
                        worker_id,
                        request_id,
                        status: OpResult::Ok(Some(Box::new(WorkerCopyDonePayload { transfer_id: request_id, tensor_ref }) as Box<dyn WorkerOutput>)),
                    }).await.context("worker response writer is closed")?;
                    debug!("run_worker: Operation::H2dCopy done");
                    Ok(())
                }
                Operation::D2hCopy { tensor_key, kind, shape, tensor_byte_len, arena_offset } => {
                    debug!("run_worker: Operation::D2hCopy tensor_key={tensor_key:?} kind={kind:?} shape={shape:?} tensor_byte_len={tensor_byte_len} arena_offset={arena_offset}");
                    validate_arena_tensor_metadata(kind, &shape, tensor_byte_len, arena_offset, &context.shared_arena, "D2H destination")?;

                    let cuda_device = context.tho_device.gpu_ordinal();
                    let source = match &tensor_key {
                        TensorKey::Data(tensor_key) => {
                            prepare_d2h_source(&context.all_cuda_tensors, tensor_key.clone(), kind, &shape, cuda_device)
                        }
                        TensorKey::StageState{ endpoint_name, stage_id, tensor_key: tensor_id } => {
                            let endpoint_key = cuda_process::endpoint_stage_key(endpoint_name.clone(), Some(stage_id));

                            let endpoint: Option<Arc<dyn SyncStageIpcWrapper>> = context.installed_stages.get(&endpoint_key).and_then(|entry| entry.value().as_ref().map(Arc::clone));

                            if let Some(endpoint) = endpoint {
                                // endpoint.purge(context.tho_device.clone()).await?;
                                if let Some(with_state) = endpoint.as_with_state(){

                                    with_state.get_state_tensor(&context.tho_device, tensor_id.clone()).await
                                } else {
                                    panic!("no state for: endpoint_key={endpoint_key:}");
                                }
                            } else {
                                panic!("no endpoint found: endpoint_key={endpoint_key:}");
                            }
                        }
                    }?;
                    let source_ptr = source.data_ptr() as usize;
                    let destination_address = context.shared_arena.address(arena_offset, tensor_byte_len, "D2H destination")?;
                    context.copy_ctx.read().await.copy_to_cpu_ptr(source_ptr, &shape, kind.into(), destination_address).await?;
                    // The alias protects worker CUDA storage until every chunk has synchronized and written the parent mapping.
                    drop(source);
                    context.reply_tx.send(CudaWorkerMsgOut {
                        worker_id,
                        request_id,
                        status: OpResult::Ok(Some(Box::new(WorkerCopyDonePayload { transfer_id: request_id, tensor_ref: tensor_key }) as Box<dyn WorkerOutput>)),
                    }).await.context("worker response writer is closed")?;
                    Ok(())
                }
                Operation::DropThoTensor { tensor_key } => {
                    debug!("run_worker: Operation::DropThoTensor: request_id={request_id} tensor_key={tensor_key}");
                    Self::drop_tensor(&context, request_id, &tensor_key).await?;
                    Ok(())
                }
                Operation::InstallStage { endpoint_name, stage_id, init_data } => {
                    debug!("run_worker: install endpoint endpoint_name={endpoint_name} stage_id={stage_id:?}");

                    let init_fn = context.endpoint_inits.get(&endpoint_name).ok_or_else(|| ThoError::new(format!("No init function for endpoint_name={endpoint_name:?} stage_id={stage_id:?}")))?;
                    let endpoint_key = cuda_process::endpoint_stage_key(endpoint_name, stage_id.as_ref());
                    debug!("run_worker: install endpoint key={endpoint_key} installed_endpoints keys={:?}", context.installed_stages.iter().map(|entry| entry.key().clone()).collect::<Vec<_>>());
                    let is_duplicate = match context.installed_stages.entry(endpoint_key.clone()) {
                        dashmap::mapref::entry::Entry::Occupied(_) => true,
                        dashmap::mapref::entry::Entry::Vacant(entry) => {
                            // Reserve the composite key before initialization so concurrent duplicate installs cannot allocate state twice.
                            entry.insert(None);
                            false
                        }
                    };
                    if is_duplicate {
                        context.reply_tx.send(CudaWorkerMsgOut { worker_id, request_id, status: OpResult::failed(format!("endpoint {endpoint_key:?} is already installed")) }).await.context("worker response writer is closed")?;
                        return Ok(());
                    }
                    let endpoint_wrapper = match init_fn(stage_id, init_data, Arc::clone(&context.ws), &context.all_cuda_tensors, &context.tho_device).await {
                        Ok(endpoint_wrapper) => endpoint_wrapper,
                        Err(error) => {
                            context.installed_stages.remove(&endpoint_key);
                            return Err(error);
                        }
                    };
                    debug!("run_worker: install endpoint endpoint_key={endpoint_key:?} endpoint_wrapper={endpoint_wrapper:?}");
                    context.installed_stages.insert(endpoint_key, Some(Arc::from(endpoint_wrapper)));
                    context.reply_tx.send(CudaWorkerMsgOut { worker_id, request_id, status: OpResult::Ok(None) }).await.context("worker response writer is closed")?;
                    Ok(())
                }
                Operation::InvokeStage { endpoint_name, stage_id, input, consume } => {
                    debug!("run_worker: Operation::InvokeStage: endpoint_name={endpoint_name} stage_id={stage_id:?} input={input:?} consume={consume}");

                    let endpoint_key = cuda_process::endpoint_stage_key(endpoint_name, stage_id.as_ref());

                    let endpoint = context.installed_stages.get(&endpoint_key).and_then(|entry| entry.value().as_ref().map(Arc::clone)).ok_or_else(|| ThoError::new(format!("no initialized endpoint for endpoint_key={endpoint_key}")))?;
                    let run_ctx = WokerRunCtx::new(context.tho_device.tho_device_idx());
                    let stream = context.exec_stream_pool.acquire().await;
                    let result = match endpoint.as_with_state() {
                        Some(with_state) => {
                            debug!("run_worker: invoke endpoint endpoint_key={endpoint_key:?} as_with_state");
                            // The erased endpoint performs its own typed pre_run before entering the CUDA stream.
                            panic_to_err_async!(with_state.do_run_with_state(&input, &context.all_cuda_tensors, &context.tho_device, &stream, &run_ctx))
                        }
                        None => {
                            debug!("run_worker: invoke endpoint endpoint_key={endpoint_key:?} no state");
                            panic_to_err!({
                            let _guard = ThoStreamGuard::new(&stream)?;
                            let response = if consume {
                                endpoint.consume(input, &context.all_cuda_tensors, &context.tho_device, &run_ctx)?
                            } else {
                                endpoint.do_run(&input, &context.all_cuda_tensors, &context.tho_device, &run_ctx)?
                            };
                            stream.synchronize()?;
                            Ok::<_, ThoError>(response)
                        })
                        }
                    }?;
                    debug!("run_worker: invoke endpoint endpoint_key={endpoint_key:?} result={result:?}");
                    context.reply_tx.send(CudaWorkerMsgOut { worker_id, request_id, status: OpResult::Ok(Some(result)) }).await.context("worker response writer is closed")?;
                    Ok(())
                }
                //rename to DescribeStageState[Tensors]
                Operation::PurgeStage {endpoint_name, stage_id } => {
                    debug!("run_worker: purge endpoint: endpoint_name={endpoint_name} stage_id={stage_id:?}");

                    let stage_key = cuda_process::endpoint_stage_key(endpoint_name, stage_id.as_ref());

                    let stage: Option<Arc<dyn SyncStageIpcWrapper>> = context.installed_stages.get(&stage_key).and_then(|entry| entry.value().as_ref().map(Arc::clone));

                    if let Some(stage) = stage {
                        stage.purge(context.tho_device.clone()).await?;

                        context.reply_tx.send(CudaWorkerMsgOut { worker_id, request_id, status: OpResult::Ok(None) }).await.context("worker response writer is closed")?;

                    } else {
                        error!("no endpoint found: endpoint_key={stage_key:}");
                    }


                    Ok(())
                }
                Operation::GetStageStateInfo { endpoint_name, stage_id } => {
                    debug!("run_worker: purge endpoint: endpoint_name={endpoint_name} stage_id={stage_id:?}");

                    let endpoint_key = cuda_process::endpoint_stage_key(endpoint_name.clone(), stage_id.as_ref());

                    let endpoint: Option<Arc<dyn SyncStageIpcWrapper>> = context.installed_stages.get(&endpoint_key).and_then(|entry| entry.value().as_ref().map(Arc::clone));

                    if let Some(endpoint) = endpoint {
                        
                        if let Some(with_state) = endpoint.as_with_state(){

                            let mut state_info = with_state.get_state_info(&context.tho_device).await?;
                            // The installation owns the routing identity. Shared model
                            // state (notably TrainStep) only supplies tensor names.
                            for tensor in &mut state_info.tensors {
                                tensor.tensor_key = TensorKey::stage_state(&endpoint_name, stage_id.as_deref().unwrap_or(""), tensor.tensor_key.to_tensor_key());
                            }

                            context.reply_tx.send(
                                CudaWorkerMsgOut { worker_id, request_id, status: OpResult::Ok(Some(Box::new(state_info))) }
                            ).await.context("worker response writer is closed")?;
                        } else {
                            error!("no state for: endpoint_key={endpoint_key:}");
                        }
                    } else {
                        error!("no endpoint found: endpoint_key={endpoint_key:}");
                    }


                    Ok(())
                }
                Operation::DropStage { endpoint_name, stage_id } => {
                    let endpoint_key = cuda_process::endpoint_stage_key(endpoint_name, stage_id.as_ref());
                    let removed = context.installed_stages.remove(&endpoint_key).is_some();
                    debug!("run_worker: drop endpoint: endpoint key={endpoint_key} removed={removed} installed_endpoints keys={:?}", context.installed_stages.iter().map(|entry| entry.key().clone()).collect::<Vec<_>>());
                    context.reply_tx.send(CudaWorkerMsgOut { worker_id, request_id, status: OpResult::Ok(None) }).await.context("worker response writer is closed")?;
                    Ok(())
                }
                Operation::Shutdown { dummy: _ } => {
                    Err(ThoError::new(format!("worker shutdown request {request_id} escaped the reader loop")))
                }
                Operation::Debug => {

                    info!("run_worker: Operation::Debug context.tho_device={:?} all_cuda_tensors.len={}", context.tho_device, context.all_cuda_tensors.tensors.read().len());

                    for (key, tensor) in context.all_cuda_tensors.tensors.read().iter() {
                        info!("run_worker: key={key:?} tensor={tensor:?}");
                    }

                    let out = Box::new(DebugOutput::new(DEBUG_DATA.lock().as_ref()));
                    context.reply_tx.send(CudaWorkerMsgOut { worker_id, request_id, status: OpResult::Ok(Some(out)) }).await.context("worker response writer is closed")?;


                    Ok(())
                }
            }
        })().await;

        match op_r {
            Ok(()) => Ok(()),
            Err(error) => {
                error!("run_worker: e={error:?}");
                context.reply_tx.send(CudaWorkerMsgOut { worker_id, request_id, status: error.into() }).await.context("worker response writer is closed")?;
                Ok(())
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaWorker::drop_tensor"))]
    async fn drop_tensor<W: Workspace + ?Sized>(context: &WorkerRequestContext<W>, request_id: u64, tensor_key: &u64) -> anyhow::Result<()>{
        let removed = context.all_cuda_tensors.tensors.write().remove(&tensor_key);
        debug!("run_worker: Operation::DropThoTensor: removed={removed:?}");
        let status = if removed.is_none() { OpResult::failed("Not found".to_string()) } else { OpResult::Ok(None) };
        context.reply_tx.send(CudaWorkerMsgOut { worker_id: context.all_cuda_tensors.worker_id, request_id, status }).await.context("worker response writer is closed")
    }
}

pub trait WorkerStageInput: Debug + Sized + Debug {

    //for cloning on the main-process-side ThoTensor/ThoModule  containing references to the data kept on  worker side
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {//on main side
        error!("clone_refs: self={self:?}");
        unimplemented!()
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {//on worker side
        error!("WorkerStageInput::ref_to_tensor: self={self:?}");
        unimplemented!()
    }

}

pub trait WorkerStageOutput: Debug + Sized + Serialize {
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError>;
}

impl<D, Id> WorkerStageInput for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + DeserializeOwned,
{

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        // Typed worker wrappers receive the tensor reference as a trait object and materialize it only inside the worker.
        Self::ref_to_tensor_(all_cuda_tensors, self, remove)
    }

}

impl <D,E>WorkerStageOutput for DataWithExtra<D, E>
where
    D: WorkerStageOutput + Data + CopyToDevice + WithId + Sync,
    E: Clone + Serialize + Debug + Send + Sync,
{
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
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

    fn ref_to_tensor_(all_cuda_tensors: &AllTensors, input_ref: &ThoTensor<D, Id>, remove: bool) -> Result<ThoTensor<D, Id>, ThoError> {//TODO make input_ref &self
        input_ref.ref_worker_id().ok_or_else(|| ThoError::new("worker input must be a tensor reference"))?.check(all_cuda_tensors.worker_id)?;
        let input_key = input_ref.ref_key().ok_or_else(|| ThoError::new(format!("worker sync stage expected ThoTensor ref input, got {input_ref:?}")))?;

        debug!("ThoTensor::ref_to_tensor: input_key={input_key:?} selector={:?} remove={remove}", input_ref.selector());

        if remove {
            let mut tensors = all_cuda_tensors.tensors.write();
            debug!("ThoTensor::ref_to_tensor: input_key={input_key:?} keys={:?}", tensors.keys());
            let tensor  = tensors
                .remove(&input_key)
                .ok_or_else(|| ThoError::new(format!("worker sync stage could not find tensor ref key={input_key}")))?;

            ThoTensor::new(input_ref.id().clone(), tensor)
        } else {
            let tensors = all_cuda_tensors.tensors.read();
            debug!("ThoTensor::ref_to_tensor: input_key={input_key:?} keys={:?}", tensors.keys());
            let source = tensors
                .get(&input_key)
                .ok_or_else(|| ThoError::new(format!("worker sync stage could not find tensor ref key={input_key}")))?;

            debug!("ThoTensor::ref_to_tensor: input_key={input_key:?} source={source:?}");

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

            debug!("ThoTensor::ref_to_tensor: input_key={input_key:?} tensor={tensor:?}");

            ThoTensor::new(input_ref.id().clone(), tensor)
        }

    }
}

impl<D, Id> WorkerStageOutput for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Serialize,
{
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        self.worker_tensor_to_ref(all_cuda_tensors, tho_device)
    }
}

impl<D, Id> ThoTensor<D, Id>
where
    D: Dims,
    Id: Clone + Debug + Named + Serialize,
{
    pub fn worker_tensor_to_ref(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<ThoTensor<D, Id>, ThoError> {
        let out_id = self.id().clone();
        let out_tensor = self.t();
        let size = out_tensor.size();
        let kind = out_tensor.kind();
        let out_key = insert_worker_tensor(all_cuda_tensors, out_tensor)?;

        debug!("worker_tensor_to_ref: out_key={out_key} out_id={out_id:?}");

        Ok(ThoTensor::<D, Id>::new_ref(out_id, out_key, size, kind, None, all_cuda_tensors.worker_id))
    }
}

pub fn insert_worker_tensor(all_cuda_tensors: &AllTensors, tensor: Tensor) -> anyhow::Result<u64> {
    let mut tensors = all_cuda_tensors.tensors.write();
    loop {
        let key = match NEXT_WORKER_TENSOR_KEY.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |key| key.checked_add(1)) {
            Ok(key) => key,
            Err(_) => anyhow::bail!("worker tensor key space exhausted"),
        };


        if let std::collections::hash_map::Entry::Vacant(entry) = tensors.entry(key) {
            debug!("insert_worker_tensor: key={key:?} tensor={tensor:?}");

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

#[cfg(test)]
mod generation_tests {
    use super::*;
    use thoosa_shared::CudaDeviceGeneration;

    #[test]
    fn test_reused_tensor_key_rejects_old_generation_and_other_device() {
        let installed = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
        installed.expect().return_const(false);
        let current = CudaWorkerId { device_idx: 2, generation: CudaDeviceGeneration(3) };
        let tensors = AllTensors::new(current);
        tensors.tensors.write().insert(42, Tensor::f_from_slice(&[7f32]).unwrap());
        for worker_id in [
            CudaWorkerId { generation: CudaDeviceGeneration(2), ..current },
            CudaWorkerId { device_idx: 1, ..current },
        ] {
            let stale: ThoTensor = ThoTensor::new_ref((), 42, vec![1], tch::Kind::Float, None, worker_id);
            for consume in [false, true] {
                assert!(matches!(stale.ref_to_tensor(&tensors, consume).unwrap_err().what(), ThoError::ContextReset));
                assert!(tensors.tensors.read().contains_key(&42), "stale consume must not remove the replacement tensor");
            }
        }
        let valid: ThoTensor = ThoTensor::new_ref((), 42, vec![1], tch::Kind::Float, None, current);
        assert_eq!(valid.ref_to_tensor(&tensors, false).unwrap().t_ref().double_value(&[0]), 7.);
        valid.ref_to_tensor(&tensors, true).unwrap();
        assert!(!tensors.tensors.read().contains_key(&42));
    }
}
