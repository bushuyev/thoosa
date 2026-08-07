use std::marker::PhantomData;
use std::sync::Arc;
use serde::de::DeserializeOwned;
use serde::Serialize;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::ThoError;
use thoosa_shared::traits::{RunCtx, Stage, SyncStage};
use crate::ipc::codec::{IpcCodec, JsonIpcCodec};
use crate::ipc::cuda_process::encode_endpoint_payload_bytes;
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::runner::global_ctx::GlobalCtx;

#[derive(Debug)]
pub struct ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    pd:PhantomData<St>,
    endpoint_name:&'static str
}

impl <St> Stage for ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}

impl <St>SyncStage for ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    fn do_run<'a>(self: std::sync::Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, thoosa_shared::ThoError>
    where
        Self: Sized,
    {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {

                // let endpoint_name = crate::runner::test::proxy_tests::TEST_ENDPOINT;
                let actor = thoosa_common::runner::global_ctx::GlobalCtx::global().cuda_actor(thoosa_shared::traits::RunCtx::tho_device_idx(ctx)).as_ref();
                let input = thoosa_common::ipc::codec::JsonIpcCodec::encode(input)?;
                tracing::debug!("invoke_ipc_sync_stage: endpoint_name={} 1. input={input:?}", self.endpoint_name);


                let response = actor.invoke_endpoint_bytes(
                    encode_endpoint_payload_bytes(self.endpoint_name, &input).into_bytes()
                ).await?;

                let r = thoosa_common::ipc::codec::JsonIpcCodec::decode(&response);
                tracing::debug!("invoke_ipc_sync_stage: endpoint_name={} response={r:?}", self.endpoint_name);

                r
            })
        })
    }
}

impl <St> ProxyStage<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    pub fn proxy(endpoint_name:&'static str) -> Self {
        Self {
            endpoint_name,
            pd:PhantomData
        }
    }

}

#[test]
fn test_0()
{

}