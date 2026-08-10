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
}

impl IpcSyncStageRunInput {
    pub fn new(input: Vec<u8>) -> Self {
        Self {
            input,
        }
    }

    pub fn input(self) -> Vec<u8> {
        self.input
    }
}

pub fn encode_sync_stage_run_input<C>(input: &[u8]) -> Result<Vec<u8>, ThoError>
where
    C: IpcCodec,
{
    C::encode(&IpcSyncStageRunInput::new(input.to_vec()))
}

//invoked form Proxy macro
pub async fn install_ipc_endpoint<C, T>(
    actor: &dyn CudaActorHandle,
    endpoint_name: &str,
    stage_id: u32,
    init_msg: &T,
) -> Result<(), ThoError>
where
    C: IpcCodec,
    T: Serialize + ?Sized,
{
    let init_msg = C::encode(init_msg)?;
    actor
        .install_endpoint(stage_id, encode_endpoint_payload_bytes(endpoint_name, &init_msg))
        .await
}
//invoked form Proxy macro, don't use it, Proxy macro is going to be removed
pub async fn install_empty_ipc_endpoint(
    actor: &dyn CudaActorHandle,
    endpoint_name: &str,
    stage_id: u32,
) -> Result<(), ThoError> {
    actor
        .install_endpoint_bytes(
            stage_id,
            encode_endpoint_payload_bytes(endpoint_name, &[]).into_bytes()
        ).await
}

pub async fn invoke_ipc_endpoint<C, I, O>(
    actor: &dyn CudaActorHandle,
    endpoint_name: &str,
    stage_id: u32,
    input: &I,
) -> Result<O, ThoError>
where
    C: IpcCodec,
    I: Debug + Serialize + ?Sized,
    O: Debug + DeserializeOwned,
{
    debug!("invoke_ipc_endpoint: endpoint_name={endpoint_name} stage_id={stage_id} input={input:?}");

    let input = C::encode(input)?;

    let response = actor.invoke_endpoint_bytes(
        stage_id,
        encode_endpoint_payload_bytes(endpoint_name, &input).into_bytes()
    ).await?;

    let r = C::decode(&response);

    debug!("invoke_ipc_endpoint: endpoint_name={endpoint_name} stage_id={stage_id} response={r:?}");

    r
}

pub async fn consume_ipc_endpoint<C, I, O>(
    actor: &dyn CudaActorHandle,
    endpoint_name: &str,
    stage_id: u32,
    input: &I,
) -> Result<O, ThoError>
where
    C: IpcCodec,
    I: Debug + Serialize + ?Sized,
    O: Debug + DeserializeOwned,
{
    debug!("consume_ipc_endpoint: endpoint_name={endpoint_name} stage_id={stage_id} input={input:?}");

    let input = C::encode(input)?;
    let response = actor.consume_endpoint_bytes(stage_id, encode_endpoint_payload_bytes(endpoint_name, &input).into_bytes()).await?;
    let result = C::decode(&response);

    debug!("consume_ipc_endpoint: endpoint_name={endpoint_name} stage_id={stage_id} response={result:?}");

    result
}

pub async fn invoke_ipc_endpoint_to_file<C, I>(
    actor: &dyn CudaActorHandle,
    endpoint_name: &str,
    stage_id: u32,
    input: &I,
) -> Result<std::fs::File, ThoError>
where
    C: IpcCodec,
    I: Debug + Serialize + ?Sized,
{
    debug!("invoke_ipc_endpoint_to_file: endpoint_name={endpoint_name} stage_id={stage_id} input={input:?}");

    let input = C::encode(input)?;
    let output = tempfile::tempfile()?;
    let response = actor.invoke_endpoint_to_file(stage_id, encode_endpoint_payload_bytes(endpoint_name, &input).into_bytes(), output).await?;

    debug!("invoke_ipc_endpoint_to_file: endpoint_name={endpoint_name} stage_id={stage_id} done");

    Ok(response)
}

pub async fn invoke_ipc_sync_stage<C, I, O>(
    actor: &dyn CudaActorHandle,
    endpoint_name: &str,
    stage_id: u32,
    input: &I,
) -> Result<O, ThoError>
where
    C: IpcCodec,
    I: Debug + Serialize + ?Sized,
    O: Debug + DeserializeOwned,
{
    debug!("invoke_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id} 0. input={input:?}");

    let input = C::encode(input)?;
    debug!("invoke_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id} 1. input={input:?}");

    let input = encode_sync_stage_run_input::<C>(&input)?;

    debug!("invoke_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id} 2. input={input:?}");

    let response = actor.invoke_endpoint_bytes(
        stage_id,
        encode_endpoint_payload_bytes(endpoint_name, &input).into_bytes()
    ).await?;

    let r = C::decode(&response);
    debug!("invoke_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id} response={r:?}");

    r
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


    let response = actor.purge_endpoint_bytes(
        stage_id,
        encode_endpoint_payload_bytes(endpoint_name, b"purge")
    ).await?;

    debug!("invoke_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id}");

    Ok(())
}

pub async fn consume_ipc_sync_stage<C, I, O>(
    actor: &dyn CudaActorHandle,
    endpoint_name: &str,
    stage_id: u32,
    input: &I,
    ctx: &RunCtxImpl,
) -> Result<O, ThoError>
where
    C: IpcCodec,
    I: Debug + Serialize + ?Sized,
    O: Debug + DeserializeOwned,
{
    debug!("consume_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id} input={input:?}");

    let input = C::encode(input)?;
    let input = encode_sync_stage_run_input::<C>(&input)?;
    let response = actor.consume_endpoint_bytes(stage_id, encode_endpoint_payload_bytes(endpoint_name, &input).into_bytes()).await?;
    let result = C::decode(&response);

    debug!("consume_ipc_sync_stage: endpoint_name={endpoint_name} stage_id={stage_id} response={result:?}");

    result
}
