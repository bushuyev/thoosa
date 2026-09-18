//! Tensor-free IPC fixture for measuring the sync-stage invocation path without tensor transfer work.

use crate::ipc::cuda_wokrer::{AllTensors, CudaWorker, InitFn, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use crate::ipc::proxy_tests::{TestRunCtx, TestStageA};
use crate::ipc::serialization::json::{IpcCodec, JsonIpcCodec};
use crate::ipc::tests::{init_ipc_test, run_ipc_test};
use crate::ipc::worker_msg::{WorkerInput, WorkerOutput};
use crate::ipc::wraps::SyncStageIpcWrapper;
use crate::ipc::wraps::WorkerWrapper;
use crate::runner::common::cachable::{BorrowRef, ThoDataRef};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use crate::runner::run_ctx::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::ws::{Workspace, mock_ws_simp};
use futures_util::future::{BoxFuture, FutureExt};
use log::info;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::ffi::OsString;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Duration;
use tch::{Device, Tensor};
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::perf_logging::{PERF, format_perf_map};
use thoosa_shared::traits::{RunCtx, Stage, SyncStage};

const SIMP_ENDPOINT: &str = "SimpStage";

#[derive(Clone, Debug, Deserialize, Serialize)]
struct SimpInput {
    value: u32,
}

impl Input for SimpInput {}

impl BorrowRef for SimpInput {
    type Out<'a> = ThoDataRef<'a, Self> where Self: 'a;

    fn borrow_ref<'a>(&'a self, _to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        // Scalar-only input is device-independent, so IPC invocation can borrow it without a tensor copy.
        async move { Ok(ThoDataRef::Borrowed(self)) }
    }

}

#[typetag::serde]
impl WorkerInput for SimpInput {}

impl WorkerStageInput for SimpInput {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        // There are no worker-owned tensor references to rewrite; clone only the scalar payload.
        Ok(Box::new(self.clone()))
    }

    fn ref_to_tensor(&self, _all_cuda_tensors: &AllTensors, _remove: bool) -> Result<Self, ThoError> {
        // Keep this conversion explicit so the test covers the normal worker wrapper without touching tensor storage.
        Ok(self.clone())
    }

}

#[derive(Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SimpOutput {
    value: u32,
}

impl Output for SimpOutput {}

#[typetag::serde]
impl WorkerOutput for SimpOutput {}

impl WorkerStageOutput for SimpOutput {
    fn encode_from_worker(self, _all_cuda_tensors: &AllTensors, _tho_device: &ThoDevice) -> Result<Self, ThoError> {
        // Scalar-only output needs no conversion into a worker tensor reference.
        Ok(self)
    }
}

#[derive(Debug)]
struct SimpStage<C>
where
    C:RunCtx
{
    pd:PhantomData<C>
}

impl <C> SimpStage<C>
where
    C:RunCtx
{
    pub async fn new() -> Result<DynSyncStageEnum<SimpStage<RunCtxImpl>>, ThoError> {
        if GlobalCtx::global().is_ipc() {
            Ok(DynSyncStageEnum::Proxy {
                stage:Arc::new(ProxyStage::<SimpStage<RunCtxImpl>>::proxy(SIMP_ENDPOINT, 0, 1, Box::new(())).await?),
                init:()
            })

        } else {
            Ok(DynSyncStageEnum::Local(Arc::new(SimpStage::<RunCtxImpl>::new_local())))
        }

    }

    pub fn new_local() -> Self {
        Self { pd:PhantomData }
    }
}

impl<C> Stage for SimpStage<C>
where
    C:RunCtx
{
    type Ctx = C;
    type In<'a> = SimpInput;
    type Out = SimpOutput;
}

impl<C> SyncStage for SimpStage<C>
where
    C:RunCtx
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SimpStage::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        // Identity work isolates IPC and sync-stage dispatch overhead from tensor computation.
        // std::thread::sleep(Duration::from_secs(1));

        Ok(SimpOutput { value: input.value * 10 })
    }
}

fn simp_stage_init<'a>(init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init: Box<dyn Any> = init;
        let _init = init.downcast::<()>().map_err(|_| ThoError::new("SimpStage expected unit initialization data"))?;

        Ok(Box::new(WorkerWrapper::new(Arc::new(SimpStage::<WokerRunCtx>::new_local()), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep::do_test_simp"))]
async fn do_test_simp() -> Result<(), ThoError> {
    let cuda_actor = GlobalCtx::global().cuda_actor(1);
    let stage = Arc::new(SimpStage::<RunCtxImpl>::new().await?);

    let ctx = RunCtxImpl::new(1, "test");

    for _ in 0..100 {
        let input = SimpInput { value: 42 };
        let output = cuda_actor.run_stage(Arc::clone(&stage), input, 1, &ctx).await?;
        assert_eq!(output, SimpOutput { value: 420 });
    }


    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_simp_local() -> Result<(), ThoError> {
    GlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    do_test_simp().await?;

    info!("local perf:\n{}", format_perf_map(&PERF));

    Ok(())
}

#[test]
fn test_simp_ipc() -> Result<(), ThoError> {
    let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

    run_ipc_test(
        module_path!(),
        async {

            do_test_simp().await?;

            Ok(())
        },
        async |sock_path: OsString| {
            CudaWorker::run_worker(
                sock_path,
                &ws,
                HashMap::from([(SIMP_ENDPOINT.to_string(), Box::new(simp_stage_init) as InitFn)]),
                ThoDevice::from(Device::Cuda(0)),
            ).await
        },
    )
}
