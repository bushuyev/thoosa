use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::io::ErrorKind;
use std::path::PathBuf;
use std::sync::Arc;
use dashmap::DashMap;
use log::debug;
use parking_lot::RwLock;
use thoosa_shared::{panic_to_err, ThoError};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, Mutex};
use tch::Tensor;
use tokio::task::JoinHandle;
use serde::de::DeserializeOwned;
use serde::Serialize;
use thoosa_shared::traits::Named;
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tub::Pool;
use crate::dims::Dims;
use crate::ipc::cuda_process;
use crate::ipc::cuda_process::{OutgoingRequestPayload, PendingWorkerD2hTransfer, PendingWorkerH2dTransfer, WorkerCudaResources, WorkerD2hChunk, WorkerD2hStartPayload, WorkerH2dChunk, WorkerH2dStartPayload, WorkerReply, D2H_START_PAYLOAD_PREFIX, DESCRIBE_REF_PAYLOAD_PREFIX, DROP_THO_TENSOR, ENDPOINT_OK_PAYLOAD, H2D_START_OK_PAYLOAD, H2D_START_PAYLOAD_PREFIX, INSTALL_ENDPOINT_PREFIX, INVOKE_ENDPOINT_PREFIX, IPC_SOCK_ENV, SHARED_MEMFD_PAYLOAD_PREFIX, TENSOR_INFO_PAYLOAD_PREFIX};
use crate::ipc::ipc_cuda_utils::{recv_fd, IpcTensorTransferMetadata, IpcTransferDirection, SharedMemfdBuffer};
use crate::ipc::json_ser::SyncStageJsonWrapper;
use crate::runner::common::runnable::SyncStage;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;

const WORKER_EXEC_STREAMS: usize = 32;

pub struct CudaWorker {
    all_cuda_tensors:DashMap<u64, Tensor>,
    handle:JoinHandle<Result<(), ThoError>>,
}


pub trait Init {
    fn init() -> Result<Box<dyn SyncStageJsonWrapper>, ThoError>;
}

impl CudaWorker {

    pub async fn run_worker(endpoint_inits: HashMap<String, Box<dyn Fn(String) -> Result<Box<dyn SyncStageJsonWrapper>, ThoError>  >>, tho_device: ThoDevice) -> Result<(), ThoError> {

        let sock_path = std::env::var_os(IPC_SOCK_ENV)
            .ok_or_else(|| ThoError::new("missing THOOSA_WORKER_IPC_SOCK"))?;
        let cuda_device = tho_device.gpu_ordinal();
        let _exec_context = ThoCudaContext::new(cuda_device as i32)?;
        let exec_stream_pool = Arc::new(Pool::from_iter(
            (0..WORKER_EXEC_STREAMS)
                .map(|_| ThoCudaStream::from_pool(cuda_device as i32, false))
                .collect::<Result<Vec<_>, _>>()?,
        ));

        debug!("running worker: sock_path={sock_path:?} cuda_device={cuda_device} endpoint_inits.keys={:?}", endpoint_inits.keys().collect::<Vec<_>>());

        let stream = UnixStream::connect(PathBuf::from(sock_path)).await?;

        debug!("run_worker: worker connected");

        let (mut reader, mut writer) = stream.into_split();

        cuda_process::write_text_frame(&mut writer, "worker-ready").await?;

        let (reply_tx, reply_rx) = mpsc::channel::<WorkerReply>(1024);

        let writer_task = tokio::spawn(cuda_process::worker_response_writer_loop(writer, reply_rx));
        let all_cuda_tensors = Arc::new(Mutex::new(HashMap::new()));
        let mut pending_h2d_transfers = HashMap::<u64, PendingWorkerH2dTransfer>::new();
        let mut pending_d2h_transfers = HashMap::<u64, PendingWorkerD2hTransfer>::new();
        let mut cuda_resources = HashMap::<usize, Arc<WorkerCudaResources>>::new();
        let installed_endpoints = Arc::new(RwLock::new(HashMap::<String, Box<dyn SyncStageJsonWrapper>>::new()));

        loop {
            let frame = match cuda_process::read_text_frame(&mut reader).await {
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
                let start: WorkerD2hStartPayload = serde_json::from_str(start_json)
                    .map_err(|e| ThoError::new(format!("failed to parse D2H start payload: {e}")))?;
                let old = pending_d2h_transfers.insert(start.transfer_id, PendingWorkerD2hTransfer::new(start.transfer_id, start.tensor_key));
                if old.is_some() {
                    return Err(ThoError::new(format!("duplicate D2H transfer start for transfer_id={}", start.transfer_id)));
                }
                continue;
            }

            if let Some(metadata_json) = payload.strip_prefix(SHARED_MEMFD_PAYLOAD_PREFIX) {

                let metadata: IpcTensorTransferMetadata = serde_json::from_str(metadata_json)
                    .map_err(|e| ThoError::new(format!("failed to parse shared memfd metadata: {e}")))?;

                let fd = recv_fd(reader.as_ref()).await?;

                let buffer = SharedMemfdBuffer::from_received_fd(fd, metadata.mapped_len)?;
                match metadata.direction {
                    IpcTransferDirection::HostToDevice => {
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
                            cuda_process::spawn_worker_h2d_copy_task(transfer, resources, Arc::clone(&all_cuda_tensors), reply_tx.clone());
                        }
                    }
                    IpcTransferDirection::DeviceToHost => {
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
                            cuda_process::spawn_worker_d2h_copy_task(transfer, resources, Arc::clone(&all_cuda_tensors), reply_tx.clone());
                        }
                    }
                    IpcTransferDirection::GpuToGpuThroughHost => {
                        return Err(ThoError::new("worker IPC actor does not handle combined GPU-to-GPU chunks directly"));
                    }
                }
                continue;
            }

            if let Some(init_msg) = payload.strip_prefix(INSTALL_ENDPOINT_PREFIX) {
                let (endpoint_name, init_msg) = cuda_process::split_endpoint_payload(init_msg)?;
                debug!("run_worker: install endpoint endpoint_name={endpoint_name:?} init_msg={init_msg} endpoint_inits.contains_key={}", endpoint_inits.contains_key(endpoint_name));

                let payload = match endpoint_inits.get(endpoint_name) {

                    Some(init_endpoint) => {

                        match init_endpoint(init_msg.to_string()) {
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

            if let Some(input) = payload.strip_prefix(INVOKE_ENDPOINT_PREFIX) {
                let (endpoint_name, input) = cuda_process::split_endpoint_payload(input)?;
                debug!("run_worker: invoke endpoint endpoint_name={endpoint_name:?} input={input:?}");

                let response = {
                    let stream = exec_stream_pool.acquire().await;
                    match installed_endpoints.read().get(endpoint_name) {
                        Some(endpoint) => {
                            let r = panic_to_err!({
                                let _thread_context = ThoCudaContext::new(stream.device())?;
                                let _guard = ThoStreamGuard::new(&stream)?;
                                let response = endpoint.do_run(input.to_string(), &all_cuda_tensors);
                                if response.is_ok() {
                                    stream.synchronize()?;
                                }
                                response
                            });

                            if let  Err(e) = &r {

                                debug!("run_worker: invoke endpoint endpoint_name={endpoint_name:?} needs_cuda_ctx_reset={}", e.needs_cuda_ctx_reset());
                            }
                            r
                        },
                        None => Err(ThoError::new(format!("endpoint {endpoint_name:?} is not installed"))),
                    }
                };

                debug!("run_worker: invoke endpoint  response={response:?}");

                let payload = match response {
                    Ok(response) => OutgoingRequestPayload::Msg(response),
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

                let msg = match all_cuda_tensors.lock().await.remove(&tensor_key) {
                    None => "Not found",
                    Some(_) => "Ok"
                };

                reply_tx
                    .send(WorkerReply { id, payload: OutgoingRequestPayload::Msg(msg.to_string()) })
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


//TODO merge WorkerStageInput/Output, add 'encoder' GAT and move impl to json_er
pub(crate) trait WorkerStageInput: Debug + Sized {
    fn decode_for_worker(input: String, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError>;
}

pub(crate) trait WorkerStageOutput: Debug + Sized {
    fn encode_from_worker(self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<String, ThoError>;
}

impl<D, Id> WorkerStageInput for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + DeserializeOwned,
{
    fn decode_for_worker(input: String, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        let input_ref = serde_json::from_str::<Self>(&input)?;
        let input_key = input_ref.ref_key().ok_or_else(|| ThoError::new(format!("worker sync stage expected ThoTensor ref input, got {input_ref:?}")))?;
        let input_tensor = tokio::task::block_in_place(|| {
            let tensors = tokio::runtime::Handle::current().block_on(all_cuda_tensors.lock());
            Ok::<_, ThoError>(tensors
                .get(&input_key)
                .ok_or_else(|| ThoError::new(format!("worker sync stage could not find tensor ref key={input_key}")))?
                .shallow_clone())
        })?;
        ThoTensor::new(input_ref.id().clone(), input_tensor)
    }
}

impl<D, Id> WorkerStageOutput for ThoTensor<D, Id>
where
    D: Dims,
    Id: Named + Debug + Clone + Serialize,
{
    fn encode_from_worker(self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<String, ThoError> {
        let out_id = self.id().clone();
        let out_tensor = self.t();
        let out_key = insert_worker_tensor(all_cuda_tensors, out_tensor)?;
        Ok(serde_json::to_string(&ThoTensor::<D, Id>::new_ref(out_id, out_key, vec![], tch::Kind::Float))?)
    }
}

fn insert_worker_tensor(all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tensor: Tensor) -> Result<u64, ThoError> {
    tokio::task::block_in_place(|| {
        let mut tensors = tokio::runtime::Handle::current().block_on(all_cuda_tensors.lock());
        let mut key = tensors
            .keys()
            .copied()
            .max()
            .unwrap_or(0)
            .checked_add(1)
            .ok_or_else(|| ThoError::new("worker tensor key space exhausted"))?;
        while tensors.contains_key(&key) {
            key = key.checked_add(1).ok_or_else(|| ThoError::new("worker tensor key space exhausted"))?;
        }
        tensors.insert(key, tensor);
        Ok(key)
    })
}
