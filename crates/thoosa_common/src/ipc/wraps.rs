use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::future::BoxFuture;
use futures_util::FutureExt;
use serde::de::DeserializeOwned;
use serde::Serialize;
use tokio::sync::MutexGuard;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::ThoError;
use thoosa_shared::traits::{CachableIdTr, ErasedStageWithState, RunCtx, Stage, StageWithState, SyncStage};
use crate::ipc::codec::{IpcCodec, JsonIpcCodec};
use crate::ipc::cuda_process::encode_endpoint_payload_bytes;
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::purge::FuturesRunnerMapPurger;

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


///
#[derive(Debug)]
pub enum DynSyncStageEnum<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    Local(Arc<St>),
    Proxy {
        stage: Arc<ProxyStage<St>>,
    }
}

impl <St> Stage for DynSyncStageEnum<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}

impl <St> SyncStage for DynSyncStageEnum<St>
where
    St:SyncStage,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(input, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, &ctx),//TODO TestRunCtx
        }
    }


}
///
#[derive(Debug)]
pub enum DynSyncStageWithStateEnum<St>
where
    St:SyncStage+StageWithState,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    Local(Arc<St>),
    Proxy {
        stage: Arc<ProxyStage<St>>,
    }
}

impl <St> Stage for DynSyncStageWithStateEnum<St>
where
    St:SyncStage+StageWithState,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    type Ctx = St::Ctx;
    type In<'a> = St::In<'a>;
    type Out = St::Out;
}

impl <St> SyncStage for DynSyncStageWithStateEnum<St>
where
    St: SyncStage + StageWithState + Send + Sync,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(input, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, &ctx),//TODO TestRunCtx
        }
    }

    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        // Advertise typed state execution without exposing ThoTrainModule to the common runner.
        match self {
            Self::Local(_) => Some(self),
            Self::Proxy { .. } => None,
        }
    }
}


///
impl <St> StageWithState for DynSyncStageWithStateEnum<St>
where
    St:SyncStage+StageWithState+Send+Sync,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned
{
    type S = <St as StageWithState>::S;


    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
        match self {
            Self::Local(stage) => StageWithState::pre_run(stage.as_ref(), to_device_idx),
            Self::Proxy { .. } => async {
                Err(ThoError::new("proxy ApplyStage2 does not have local state"))
            }.boxed(),
        }
    }

    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, state: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => StageWithState::do_run_with_state(Arc::clone(stage), input, state, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, ctx),
        }
    }
}
///
///
#[async_trait]
impl <St> FuturesRunnerMapPurger for DynSyncStageWithStateEnum<St>
where
    St:SyncStage+StageWithState + FuturesRunnerMapPurger,
    for <'a> <St as Stage>::In<'a>:WorkerStageInput + Serialize,
    <St as Stage>::Out:WorkerStageOutput + DeserializeOwned

{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        match self {
            DynSyncStageWithStateEnum::Local(stage) => {
                stage.has_gpu_data(device_idx).await
            }
            DynSyncStageWithStateEnum::Proxy { .. } => {
                true
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            DynSyncStageWithStateEnum::Local(local) => {
                local.purge_gpu(ids_to_keep, device_idx, name).await
            }
            DynSyncStageWithStateEnum::Proxy { stage,  .. } => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        thoosa_common::ipc::codec::purge_ipc_sync_stage::<thoosa_common::ipc::codec::JsonIpcCodec>(
                            GlobalCtx::global().cuda_actor(device_idx).as_ref(),
                            stage.endpoint_name,
                        ).await
                    })
                })?;
                Ok(())
            }
        }
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            DynSyncStageWithStateEnum::Local(local) => {
                local.drop_gpu(device_idx, name).await
            }
            DynSyncStageWithStateEnum::Proxy { .. } => {
                Ok(())
            }
        }
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        match self {
            DynSyncStageWithStateEnum::Local(local) => {
                local.purge_cpu(purge_until).await
            }
            DynSyncStageWithStateEnum::Proxy { .. } => {
                0
            }
        }
    }
}

#[test]
fn test_0()
{

}