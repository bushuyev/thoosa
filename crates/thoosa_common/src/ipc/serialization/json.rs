use std::fmt::Debug;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::debug;
use thoosa_shared::ThoError;

use crate::copy_ctx::CudaActorHandle;
use crate::runner::adjusters::max_parallel::MaxParallel;
use thoosa_shared::traits::MaxParallelCtx;
use crate::runner::run_ctx::RunCtxImpl;

pub trait IpcCodec: Debug + Send + Sync + 'static {
    fn encode<T>(value: &T) -> anyhow::Result<Vec<u8>>
    where
        T: Serialize + ?Sized;

    fn decode<T>(bytes: &[u8]) -> anyhow::Result<T>
    where
        T: DeserializeOwned;
}

#[derive(Debug)]
pub struct JsonIpcCodec;

impl IpcCodec for JsonIpcCodec {
    fn encode<T>(value: &T) -> anyhow::Result<Vec<u8>>
    where
        T: Serialize + ?Sized,
    {
        Ok(serde_json::to_vec(value)?)
    }

    fn decode<T>(bytes: &[u8]) -> anyhow::Result<T>
    where
        T: DeserializeOwned,
    {
        Ok(serde_json::from_slice(bytes)?)
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
        endpoint_name.to_string()
    ).await?;

    debug!("invoke_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id}");

    Ok(())
}