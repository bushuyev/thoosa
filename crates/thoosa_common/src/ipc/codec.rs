use std::fmt::Debug;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::debug;
use thoosa_shared::ThoError;

use crate::copy_ctx::CudaActorHandle;
use crate::ipc::cuda_process::encode_endpoint_payload_bytes;
use crate::runner::adjusters::max_parallel::MaxParallel;
use thoosa_shared::traits::MaxParallelCtx;
use crate::runner::run_ctx_2::RunCtxImpl;

pub trait IpcCodec: Debug + Send + Sync + 'static {
    fn encode<T>(value: &T) -> Result<Vec<u8>, ThoError>
    where
        T: Serialize + ?Sized;

    fn decode<T>(bytes: &[u8]) -> Result<T, ThoError>
    where
        T: DeserializeOwned;
}

#[derive(Debug)]
pub struct JsonIpcCodec;

impl IpcCodec for JsonIpcCodec {
    fn encode<T>(value: &T) -> Result<Vec<u8>, ThoError>
    where
        T: Serialize + ?Sized,
    {
        serde_json::to_vec(value).map_err(ThoError::from)
    }

    fn decode<T>(bytes: &[u8]) -> Result<T, ThoError>
    where
        T: DeserializeOwned,
    {
        serde_json::from_slice(bytes).map_err(ThoError::from)
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct IpcSyncStageRunInput {
    input: Vec<u8>,
    max_other_parallel: Option<u8>,
}

impl IpcSyncStageRunInput {
    pub fn new(input: Vec<u8>) -> Self {
        Self {
            input,
            max_other_parallel: None,
        }
    }

    pub fn with_max_other_parallel(input: Vec<u8>, max_other_parallel: u8) -> Self {
        Self { input, max_other_parallel: Some(max_other_parallel) }
    }

    pub fn input(self) -> Vec<u8> {
        self.input
    }

    pub fn into_parts(self) -> (Vec<u8>, Option<u8>) {
        (self.input, self.max_other_parallel)
    }
}


pub async fn purge_ipc_sync_stage<C>(
    actor: &dyn CudaActorHandle,
    endpoint_name: &str,
    stage_id: u32,
) -> Result<(), ThoError>
where
    C: IpcCodec,
{
    debug!("purge_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id}");


    let response = actor.purge_stage_state(
        stage_id,
        encode_endpoint_payload_bytes(endpoint_name, b"purge")
    ).await?;

    debug!("invoke_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id}");

    Ok(())
}