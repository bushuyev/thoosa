use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use std::sync::{Arc};
use dashmap::DashMap;
use log::debug;
use parking_lot::RwLock;
use thoosa_shared::ThoError;
use tokio::net::UnixStream;
use tokio::sync::{mpsc, Mutex};
use tch::Tensor;
use tokio::task::JoinHandle;
use crate::ipc::cuda_process;
use crate::ipc::cuda_process::{TENSOR_INFO_PAYLOAD_PREFIX, OutgoingRequestPayload, PendingWorkerD2hTransfer, PendingWorkerH2dTransfer, WorkerCudaResources, WorkerD2hChunk, WorkerD2hStartPayload, WorkerH2dChunk, WorkerH2dStartPayload, WorkerReply, D2H_START_PAYLOAD_PREFIX, DESCRIBE_REF_PAYLOAD_PREFIX, H2D_START_OK_PAYLOAD, H2D_START_PAYLOAD_PREFIX, IPC_SOCK_ENV, SHARED_MEMFD_PAYLOAD_PREFIX, INSTALL_ENDPOINT_PREFIX};
use crate::ipc::ipc_cuda_utils::{recv_fd, IpcTensorTransferMetadata, IpcTransferDirection, SharedMemfdBuffer};
use crate::runner::common::runnable::SyncStage;

pub struct CudaWorker {
    all_cuda_tensors:DashMap<u64, Tensor>,
    handle:JoinHandle<Result<(), ThoError>>,
}


pub trait SyncStageJsonWrapper:Debug {
    fn do_dun(&self, _input:String)->Result<String, ThoError>;
}
pub trait Init {
    fn init() -> Result<Box<dyn SyncStageJsonWrapper>, ThoError>;
}

impl CudaWorker {



    pub async fn run_worker(endpoint_inits: HashMap<String, Box<dyn Fn(String) -> Result<Box<dyn SyncStageJsonWrapper>, ThoError>  >>) -> Result<(), ThoError> {

        let sock_path = std::env::var_os(IPC_SOCK_ENV)
            .ok_or_else(|| ThoError::new("missing THOOSA_WORKER_IPC_SOCK"))?;

        debug!("running worker: sock_path={sock_path:?}");

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
        let mut installed_endpoints = Arc::new(RwLock::new(HashMap::<String, Box<dyn SyncStageJsonWrapper>>::new()));

        loop {
            let frame = cuda_process::read_text_frame(&mut reader).await?;
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
                let endpoint_wrapper = endpoint_inits.get("SyncClcTask").expect("no SyncClcTask")(init_msg.to_string()).expect("SyncClcTask");
                debug!("run_worker: install endpoint init_msg={init_msg} endpoint_wrapper={endpoint_wrapper:?}");

                installed_endpoints.write().insert("SyncClcTask".to_string(), endpoint_wrapper);//TODO key

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
