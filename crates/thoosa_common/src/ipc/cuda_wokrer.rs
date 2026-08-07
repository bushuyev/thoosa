use std::collections::HashMap;
use std::error::Error;
use std::ffi::OsString;
use std::fmt::Debug;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use clap::Parser;
use dashmap::DashMap;
use futures_util::future::BoxFuture;
use log::{debug, error, LevelFilter};
use parking_lot::{Mutex, RwLock};
use thoosa_shared::{panic_to_err, panic_to_err_async, ThoError};
use tokio::net::UnixStream;
use tokio::sync::mpsc;
use tch::{Device, Tensor};
use tokio::task::JoinHandle;
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Serializer;
use tracing::field::debug;
use thoosa_shared::traits::{MaxParallelCtx, Named, RunCtx, WithId};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tub::Pool;
use thoosa_shared::perf_logging::init_tracing;
use crate::cuda_utils::cuda_mem_for_device;
use crate::dims::Dims;
use crate::ipc::codec::IpcCodec;
use crate::ipc::cuda_process;
use crate::ipc::cuda_process::{DeviceInfo, IpcChunkWriter, OutgoingRequestPayload, PendingWorkerD2hTransfer, PendingWorkerH2dTransfer, WorkerCudaResources, WorkerD2hChunk, WorkerD2hStartPayload, WorkerH2dChunk, WorkerH2dStartPayload, WorkerReply, CONSUME_ENDPOINT_PREFIX, D2H_START_PAYLOAD_PREFIX, DESCRIBE_REF_PAYLOAD_PREFIX, DROP_THO_TENSOR, ENDPOINT_OK_PAYLOAD, H2D_START_OK_PAYLOAD, H2D_START_PAYLOAD_PREFIX, INSTALL_ENDPOINT_PREFIX, INVOKE_ENDPOINT_PREFIX, IPC_SOCK_ENV, SHARED_MEMFD_PAYLOAD_PREFIX, TENSOR_INFO_PAYLOAD_PREFIX, PURGE_ENDPOINT_PREFIX};
use crate::ipc::ipc_cuda_utils::{recv_fd, IpcTensorTransferMetadata, IpcTransferDirection, SharedMemfdBuffer};
use crate::ipc::json_ser::{SyncStageIpcResponse, SyncStageIpcWrapper};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::ws::{WsConfigs, FsWorkspace, Workspace};

const WORKER_EXEC_STREAMS: usize = 32;
static NEXT_WORKER_TENSOR_KEY: AtomicU64 = AtomicU64::new(1);

//1 cuda device per worker
pub struct CudaWorker;

pub type InitFn = Box<dyn Fn(Vec<u8>, Arc<RwLock<Box<dyn Workspace>>>, &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>  >;

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

                CudaWorker::run_worker(sock_path, &ws, init_map, ThoDevice::from(Device::Cuda(0))).await//TODO Cuda(0)
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
        // Stateful capability is erased per endpoint so one worker can host unrelated state types.
        GlobalCtx::init_devices(2);//TODO config

        let with_digest = false;//TODO config

        let cuda_device = tho_device.gpu_ordinal();

        let exec_stream_pool = Arc::new(Pool::from_iter(
            (0..WORKER_EXEC_STREAMS)
                .map(|_| ThoCudaStream::from_pool(cuda_device as i32, false))
                .collect::<Result<Vec<_>, _>>()?,
        ));

        debug!("run_worker: sock_path={sock_path:?} cuda_device={cuda_device} endpoint_inits.keys={:?}", endpoint_inits.keys().collect::<Vec<_>>());

        let stream = UnixStream::connect(PathBuf::from(sock_path)).await?;

        debug!("run_worker: worker connected");

        let (mut reader, mut writer) = stream.into_split();

        let (free_mem, total_mem) = cuda_mem_for_device(cuda_device);

        cuda_process::write_text_frame(
            &mut writer,
            &serde_json5::to_string(&DeviceInfo {
                free_mem,
                total_mem
            })?,
            "run_worker"
        ).await?;

        let (reply_tx, reply_rx) = mpsc::channel::<WorkerReply>(1024);

        let writer_task = tokio::spawn(cuda_process::worker_response_writer_loop(writer, reply_rx));
        let all_cuda_tensors = Arc::new(parking_lot::Mutex::new(HashMap::new()));//TODO to RwLock?
        let mut pending_h2d_transfers = HashMap::<u64, PendingWorkerH2dTransfer>::new();
        let mut pending_d2h_transfers = HashMap::<u64, PendingWorkerD2hTransfer>::new();
        let mut cuda_resources = HashMap::<usize, Arc<WorkerCudaResources>>::new();
        let installed_endpoints = Arc::new(RwLock::new(HashMap::<String, Box<dyn SyncStageIpcWrapper>>::new()));

        loop {
            let frame = match cuda_process::read_text_frame(&mut reader, "run_worker").await {
                Ok(frame) => frame,
                Err(error) => {
                    if Error::source(&error)
                        .and_then(|source| source.downcast_ref::<std::io::Error>())
                        .is_some_and(|source| source.kind() == ErrorKind::UnexpectedEof)
                    {
                        debug!("run_worker: parent IPC stream closed");
                        break;
                    }
                    return Err(error);
                }
            };
            let (id, payload) = cuda_process::parse_wire_msg(&frame)?;

            debug!("run_worker: worker received: id={id}, payload={payload:?}");

            if payload == "shutdown" {
                reply_tx
                    .send(WorkerReply {
                        id,
                        payload: OutgoingRequestPayload::Msg("shutdown-ok".to_string()),
                    })
                    .await
                    .map_err(|_| ThoError::new("worker response writer is closed"))?;

                debug!("run_worker: worker got shutdown command");
                break;
            }

            if let Some(tensor_key) = payload.strip_prefix(DESCRIBE_REF_PAYLOAD_PREFIX) {

                let tensor_key = tensor_key
                    .parse::<u64>()
                    .map_err(|e| ThoError::new(format!("bad tensor ref key {tensor_key:?}: {e}")))?;

                let payload = match cuda_process::describe_worker_tensor(Arc::clone(&all_cuda_tensors), tensor_key).await {
                    Ok(info) => {

                        let info_json = serde_json::to_string(&info)
                            .map_err(|e| ThoError::new(format!("failed to encode worker tensor info: {e}")))?;

                        format!("{TENSOR_INFO_PAYLOAD_PREFIX}{info_json}")
                    }
                    Err(error) => cuda_process::encode_worker_copy_error(&error),
                };
                reply_tx
                    .send(WorkerReply {
                        id,
                        payload: OutgoingRequestPayload::Msg(payload),
                    })
                    .await
                    .map_err(|_| ThoError::new("worker response writer is closed"))?;
                continue;
            }

            if let Some(start_json) = payload.strip_prefix(H2D_START_PAYLOAD_PREFIX) {
                debug!("run_worker: H2D_START_PAYLOAD_PREFIX, start_json={start_json:?}");

                let payload = match serde_json::from_str::<WorkerH2dStartPayload>(start_json) {
                    Ok(start) => {
                        let transfer_id = start.transfer_id;
                        match cuda_process::prepare_worker_h2d_transfer(start, &mut cuda_resources) {
                            Ok(transfer) => {
                                if pending_h2d_transfers.insert(transfer_id, transfer).is_some() {
                                    cuda_process::encode_worker_copy_error(&ThoError::new(format!("duplicate H2D transfer start for transfer_id={transfer_id}")))
                                } else {
                                    H2D_START_OK_PAYLOAD.to_string()
                                }
                            }
                            Err(error) => cuda_process::encode_worker_copy_error(&error),
                        }
                    }
                    Err(error) => cuda_process::encode_worker_copy_error(&ThoError::new(format!("failed to parse H2D start payload: {error}"))),
                };

                debug!("run_worker: H2D_START_PAYLOAD_PREFIX, payload={payload:?}");
                reply_tx
                    .send(WorkerReply {
                        id,
                        payload: OutgoingRequestPayload::Msg(payload),
                    })
                    .await
                    .map_err(|_| ThoError::new("worker response writer is closed"))?;
                continue;
            }

            if let Some(start_json) = payload.strip_prefix(D2H_START_PAYLOAD_PREFIX) {
                debug!("run_worker: D2H_START_PAYLOAD_PREFIX, start_json={start_json:?}");

                let start: WorkerD2hStartPayload = serde_json::from_str(start_json).map_err(|e| ThoError::new(format!("failed to parse D2H start payload: {e}")))?;
                if !all_cuda_tensors.lock().contains_key(&start.tensor_key) {
                    return Err(ThoError::new(format!("No tensor with tensor_key={}", start.tensor_key)));
                }

                let old = pending_d2h_transfers.insert(start.transfer_id, PendingWorkerD2hTransfer::new(start.transfer_id, start.tensor_key));
                if old.is_some() {
                    return Err(ThoError::new(format!("duplicate D2H transfer start for transfer_id={}", start.transfer_id)));
                }
                continue;
            }

            if let Some(metadata_json) = payload.strip_prefix(SHARED_MEMFD_PAYLOAD_PREFIX) {
                debug!("run_worker: SHARED_MEMFD_PAYLOAD_PREFIX, start_json={metadata_json:?}");

                let metadata: IpcTensorTransferMetadata = serde_json::from_str(metadata_json)
                    .map_err(|e| ThoError::new(format!("failed to parse shared memfd metadata: {e}")))?;

                let fd = recv_fd(reader.as_ref()).await?;

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
                        match pending_h2d_transfers.get_mut(&transfer_id) {
                            Some(transfer) => transfer.push(chunk)?,
                            None => {
                                pending_h2d_transfers.insert(transfer_id, PendingWorkerH2dTransfer::new(id, chunk)?);
                            }
                        }
                        if pending_h2d_transfers.get(&transfer_id).is_some_and(PendingWorkerH2dTransfer::is_complete) {
                            let transfer = pending_h2d_transfers.remove(&transfer_id).expect("complete H2D transfer disappeared");
                            let resources = cuda_process::get_or_create_worker_cuda_resources(&mut cuda_resources, cuda_device)?;
                            cuda_process::spawn_worker_h2d_copy_task(transfer, resources, Arc::clone(&all_cuda_tensors), reply_tx.clone(), with_digest);
                        }
                    }
                    IpcTransferDirection::DeviceToHost => {
                        debug!("run_worker: IpcTransferDirection::DeviceToHost");

                        let chunk = WorkerD2hChunk { metadata, buffer };
                        let cuda_device = cuda_process::validate_worker_d2h_chunk(&chunk)?;
                        debug!("run_worker: received shared memfd D2H tensor chunk: metadata={:?} local_mmap_ptr={:?}", chunk.metadata, chunk.buffer.as_ptr());
                        let transfer_id = chunk.metadata.transfer_id;
                        let transfer = pending_d2h_transfers
                            .get_mut(&transfer_id)
                            .ok_or_else(|| ThoError::new(format!("received D2H chunk before start for transfer_id={transfer_id}")))?;
                        transfer.push(chunk)?;

                        if pending_d2h_transfers.get(&transfer_id).is_some_and(PendingWorkerD2hTransfer::is_complete) {
                            let transfer = pending_d2h_transfers.remove(&transfer_id).expect("complete D2H transfer disappeared");
                            let resources = cuda_process::get_or_create_worker_cuda_resources(&mut cuda_resources, cuda_device)?;
                            cuda_process::spawn_worker_d2h_copy_task(transfer, resources, Arc::clone(&all_cuda_tensors), reply_tx.clone(), with_digest);
                        }
                    }
                    IpcTransferDirection::GpuToGpuThroughHost => {
                        return Err(ThoError::new("worker IPC actor does not handle combined GPU-to-GPU chunks directly"));
                    }
                }
                continue;
            }

            if let Some(init_msg) = payload.strip_prefix(INSTALL_ENDPOINT_PREFIX) {//TODO UNINSTALL_ENDPOINT on drop
                let (endpoint_name, init_msg) = cuda_process::split_endpoint_payload_bytes(init_msg)?;
                debug!("run_worker: install endpoint endpoint_name={endpoint_name:?} init_msg_len={} endpoint_inits.contains_key={}", init_msg.len(), endpoint_inits.contains_key(endpoint_name));


                let payload = match endpoint_inits.get(endpoint_name) {

                    Some(init_endpoint) => {

                        match init_endpoint(init_msg, Arc::clone(&ws), &all_cuda_tensors) {
                            Ok(endpoint_wrapper) => {
                                debug!("run_worker: install endpoint endpoint_name={endpoint_name:?} endpoint_wrapper={endpoint_wrapper:?}");
                                installed_endpoints.write().insert(endpoint_name.to_string(), endpoint_wrapper);
                                ENDPOINT_OK_PAYLOAD.to_string()
                            }
                            Err(error) => cuda_process::encode_worker_copy_error(&error),
                        }
                    }
                    None => cuda_process::encode_worker_copy_error(&ThoError::new(format!("no endpoint init for {endpoint_name:?}"))),
                };
                reply_tx
                    .send(WorkerReply {
                        id,
                        payload: OutgoingRequestPayload::Msg(payload),
                    })
                    .await
                    .map_err(|_| ThoError::new("worker response writer is closed"))?;

                continue;
            }

            let endpoint_request = payload.strip_prefix(CONSUME_ENDPOINT_PREFIX).map(|input| (true, input)).or_else(|| payload.strip_prefix(INVOKE_ENDPOINT_PREFIX).map(|input| (false, input)));
            if let Some((consume, input)) = endpoint_request {
                let (endpoint_name, input) = cuda_process::split_endpoint_payload_bytes(input)?;
                debug!("run_worker: invoke endpoint endpoint_name={endpoint_name:?} consume={consume} input_len={}", input.len());

                let response = {
                    let stream = exec_stream_pool.acquire().await;//TODO pass pool to do_run_with_state
                    match installed_endpoints.read().get(endpoint_name) {
                        Some(endpoint) => {
                            let r = if let Some(with_state) = endpoint.as_with_state() {
                                // The erased endpoint performs its own typed pre_run before entering the CUDA stream.
                                panic_to_err_async!(with_state.do_run_with_state(&input, &all_cuda_tensors, &tho_device, &stream)).map(|response| (response, Vec::new()))
                            } else {
                                panic_to_err!({
                                    let _guard = ThoStreamGuard::new(&stream)?;
                                    let run_ctx = WokerRunCtx;
                                    let response = if consume {
                                        endpoint.consume_response(&input, &all_cuda_tensors, &tho_device, &run_ctx).map(|response| (response.response, response.consumed_tensor_keys))
                                    } else {
                                        endpoint.do_run_response(&input, &all_cuda_tensors, &tho_device, &run_ctx).map(|response| (response, Vec::new()))
                                    };
                                    let sync_result = stream.synchronize();
                                    match (response, sync_result) {
                                        (Ok(response), Ok(())) => Ok(response),
                                        (Err(error), Ok(())) => Err(error),
                                        (Ok(_), Err(error)) => Err(error),
                                        (Err(error), Err(sync_error)) if sync_error.is_oom() || sync_error.needs_cuda_ctx_reset() => Err(sync_error),
                                        (Err(error), Err(_)) => Err(error),
                                    }
                                })
                            };

                            if let  Err(e) = &r {

                                error!("run_worker: invoke endpoint endpoint_name={endpoint_name:?} needs_cuda_ctx_reset={}", e.needs_cuda_ctx_reset());
                            } else if let Ok(response) = &r {
                                let (response_is_binary, response_len) = match &response.0 {
                                    SyncStageIpcResponse::Encoded(response) => (false, response.len()),
                                    // SyncStageIpcResponse::Stream(_) => (true, 0),
                                };
                                debug!("run_worker: invoke endpoint endpoint_name={endpoint_name:?} consume={consume} response_is_binary={response_is_binary} response_len={response_len}");
                            }
                            r
                        },
                        None => Err(ThoError::new(format!("endpoint {endpoint_name:?} is not installed"))),
                    }
                };

                let payload = match response {
                    Ok((SyncStageIpcResponse::Encoded(response), consumed_tensor_keys)) => {
                        let mut tensors = all_cuda_tensors.lock();
                        for key in consumed_tensor_keys {
                            tensors.remove(&key);
                        }
                        OutgoingRequestPayload::Msg(cuda_process::encode_endpoint_response_bytes(&response))
                    }
                    // Ok((SyncStageIpcResponse::Stream(response), consumed_tensor_keys)) => {
                    //     let (chunks_tx, chunks_rx) = mpsc::channel(2);
                    //     let (completed_tx, completed_rx) = tokio::sync::oneshot::channel();
                    //     reply_tx.send(WorkerReply { id, payload: OutgoingRequestPayload::BinaryStream { chunks: chunks_rx, completed: completed_tx } }).await.map_err(|_| ThoError::new("worker response writer is closed"))?;
                    //
                    //     let stream = exec_stream_pool.acquire().await;
                    //     let producer_tx = chunks_tx.clone();
                    //     let serialization_result = tokio::task::spawn_blocking(move || {
                    //         let mut output = IpcChunkWriter::new(producer_tx);
                    //         panic_to_err!({
                    //
                    //             let _guard = ThoStreamGuard::new(&stream)?;
                    //             let response_result = response.write_to(&mut output);
                    //             let sync_result = stream.synchronize();
                    //
                    //             match (response_result, sync_result) {
                    //                 (Ok(()), Ok(())) => Ok(()),
                    //                 (Err(error), Ok(())) => Err(error),
                    //                 (Ok(()), Err(error)) => Err(error),
                    //                 (Err(error), Err(sync_error)) if sync_error.is_oom() || sync_error.needs_cuda_ctx_reset() => Err(sync_error),
                    //                 (Err(error), Err(_)) => Err(error),
                    //             }
                    //         })
                    //     }).await.unwrap_or_else(|error| Err(ThoError::new(format!("worker binary response serializer task failed: {error}"))));
                    //
                    //     let serialization_succeeded = serialization_result.is_ok();
                    //     if let Err(error) = serialization_result {
                    //         let _ = chunks_tx.send(Err(error)).await;
                    //     }
                    //     drop(chunks_tx);
                    //     completed_rx.await.map_err(|_| ThoError::new("worker binary response completion channel was dropped"))??;
                    //     if serialization_succeeded {
                    //         let mut tensors = all_cuda_tensors.lock();
                    //         for key in consumed_tensor_keys {
                    //             tensors.remove(&key);
                    //         }
                    //     }
                    //     continue;
                    // }
                    Err(error) => OutgoingRequestPayload::Msg(cuda_process::encode_worker_copy_error(&error)),
                };
                reply_tx
                    .send(WorkerReply { id, payload })
                    .await
                    .map_err(|_| ThoError::new("worker response writer is closed"))?;

                continue;
            }
            if let Some(input) = payload.strip_prefix(DROP_THO_TENSOR) {
                let (endpoint_name, input) = cuda_process::split_endpoint_payload(input)?;
                debug!("run_worker: drop ther-tensor: input={input:?}");
                let tensor_key = input.parse::<u64>()?;
                debug!("run_worker: drop ther-tensor: tensor_key={tensor_key}");

                let msg = match all_cuda_tensors.lock().remove(&tensor_key) {
                    None => "Not found",
                    Some(_) => "Ok"
                };

                reply_tx
                    .send(WorkerReply { id, payload: OutgoingRequestPayload::Msg(msg.to_string()) })
                    .await
                    .map_err(|_| ThoError::new("worker response writer is closed"))?;

                continue;
            }

            if let Some(input) = payload.strip_prefix(PURGE_ENDPOINT_PREFIX) {
                let (endpoint_name, input) = cuda_process::split_endpoint_payload(input)?;
                debug!("run_worker: purge endpoint: endpoint_name={endpoint_name} input={input:?}");
                match installed_endpoints.read().get(endpoint_name) {
                    Some(endpoint) => {
                        endpoint.purge(1.into()).await?;//TODO device
                    }
                    None => {
                        error!("no endpoint found"); //TODO handle
                    }
                };
                reply_tx
                    .send(WorkerReply { id, payload: OutgoingRequestPayload::Msg("ok".to_string()) })
                    .await
                    .map_err(|_| ThoError::new("worker response writer is closed"))?;

                continue;
            }

            let reply_tx = reply_tx.clone();

            //TODO remove
            tokio::spawn(async move {
                let reply = cuda_process::handle_worker_request(payload).await;

                debug!("run_worker: reply={reply}");

                let _ = reply_tx
                    .send(WorkerReply { id, payload: OutgoingRequestPayload::Msg(reply) })
                    .await;
            });
        }

        drop(reply_tx);

        writer_task
            .await
            .map_err(|e| ThoError::new(format!("worker writer task join error: {e}")))??;

        debug!("worker finished");

        Ok(())
    }

}

pub trait WorkerStageInput: Debug + Sized {
    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError>;

    fn decode_for_worker_consuming<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> Result<(Self, Vec<u64>), ThoError> {
        Ok((Self::decode_for_worker::<C>(input, all_cuda_tensors)?, Vec::new()))
    }
}

pub trait WorkerStageOutput: Debug + Sized {
    fn encode_from_worker<C: IpcCodec>(self, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError>;
}

impl<D, Id> WorkerStageInput for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + DeserializeOwned,
{
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

impl<D, Id> ThoTensor<D, Id>
where
    D: Dims,
    Id: Clone + Debug + Named,
{
    pub fn ref_to_tensor(all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, input_ref: ThoTensor<D, Id>) -> Result<ThoTensor<D, Id>, ThoError> {
        let input_key = input_ref.ref_key().ok_or_else(|| ThoError::new(format!("worker sync stage expected ThoTensor ref input, got {input_ref:?}")))?;

        let tensors = all_cuda_tensors.lock();

        debug!("ThoTensor::ref_to_tensor: input_key={input_key:?} keys={:?}", tensors.keys());

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

impl<D, Id> WorkerStageOutput for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Serialize,
{
    fn encode_from_worker<C: IpcCodec>(self, all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let tho_tensor_ref = self.worker_tensor_to_ref(all_cuda_tensors, &tho_device)?;

        C::encode(&tho_tensor_ref)
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
pub struct WokerRunCtx;

impl Named for WokerRunCtx {}


impl WithId for WokerRunCtx{
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl RunCtx for WokerRunCtx {
    fn tho_device_idx(&self) -> usize {
        1
    }

    fn branch(&mut self, with_new_cache_area: bool) -> Self
    where
        Self: Sized
    {
        WokerRunCtx
    }

    fn add_purge_gpu_closure(&mut self, cl: Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>) {
    }
}