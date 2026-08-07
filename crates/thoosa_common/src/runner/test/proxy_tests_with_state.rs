use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use futures_util::future::{BoxFuture, FutureExt, ready};
use futures_util::{StreamExt, ready};
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::{Mutex, RwLock};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use tch::Device::Cpu;
use tch::Kind::Int;
use tch::Tensor;
use tokio::sync::{MutexGuard, RwLockReadGuard};
use tracing::field::debug;
use thoosa_proxy_derive::Proxy;
use thoosa_shared::{mono_key, CopyToDevice, ThoError};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{CachableIdTr, Data, DynSyncStage, DynSyncStageBox, DynSyncStageOutId, ErasedStageState, ErasedStageWithState, MaxParallelCtx, Named, OutId, RunCtx, Stage, StageWithState, SyncStage, WithId};
use crate::dims::{Batch, Dims};
use crate::ipc::codec::{encode_sync_stage_run_input, IpcCodec, IpcSyncStageRunInput, JsonIpcCodec};
use crate::ipc::cuda_process::encode_endpoint_payload_bytes;
use crate::ipc::cuda_wokrer::{WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::json_ser::{ErasedWithState, SyncStageIpcResponse, SyncStageIpcWrapper, SyncStageIpcWrapperWithState};
use crate::ipc::wraps::ProxyStage;
use crate::runner::common::cachable::{BorrowRef, ThoDataRef};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::purge::FuturesRunnerMapPurger;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::train_nets::ThoModuleWithStore;

#[derive(Debug, Serialize, Deserialize)]
pub struct TestStageInit {
    k:i64,
    tho_device_ids:usize
}

impl TestStageInit {
    pub fn new(k: i64, tho_device_ids:usize) -> Self {
        Self { k, tho_device_ids }
    }
}

impl WorkerStageInput for TestStageInit {
    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        let mut input = JsonIpcCodec::decode::<Self>(&input)?;

        Ok(input)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestInput{
    value: ThoTensor
}
impl Input for TestInput{}

impl WithId for TestInput {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl BorrowRef for TestInput {
    type Out<'a> = ThoDataRef<'a, Self> where Self: 'a;

    fn borrow_to<'o>(&'o self, to_tho_device_idx: usize) -> impl Future<Output=Result<Self::Out<'o>, ThoError>> + Send
    where
        Self: 'o
    {
        async move {
            if self.value.device() == to_tho_device_idx {
                Ok(ThoDataRef::Borrowed(self))
            } else {
                Ok(ThoDataRef::Owned(TestInput {
                    value: track_await(self.value.copy_to_device(to_tho_device_idx)).await?,
                }))
            }
        }
    }
}

impl WorkerStageInput for TestInput {
    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        let mut input = JsonIpcCodec::decode::<Self>(&input)?;

        debug!("SimpWrapper::do_run: 1. input={input:?}");

        input.value = ThoTensor::ref_to_tensor(all_cuda_tensors, input.value)?;

        Ok(input)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestOutput{
    value: ThoTensor
}
impl Output for TestOutput{}

impl WithId for TestOutput {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}


impl WorkerStageOutput for TestOutput {
    fn encode_from_worker<C: IpcCodec>(self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        JsonIpcCodec::encode(&TestOutput{
            value: self.value.worker_tensor_to_ref(all_cuda_tensors, &tho_device)?,
        })
    }
}


#[derive(Debug)]
pub struct TestRunCtx;//corresponds to RunCtxImpl

impl Named for TestRunCtx {}

impl MaxParallelCtx for TestRunCtx{
    fn set_max_other_parallel(&mut self, permit: u8) {}

    fn max_other_parallel(&self) -> u8 { 0 }
}

impl WithId for TestRunCtx{
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl RunCtx for TestRunCtx {
    fn tho_device_idx(&self) -> usize {
        1
    }

    fn branch(&mut self, with_new_cache_area: bool) -> Self
    where
        Self: Sized
    {
        TestRunCtx
    }

    fn add_purge_gpu_closure(&mut self, cl: Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>) {
    }
}


#[derive(Debug)]
pub struct TestStage<P, C>
where
    P:Debug,
    C:RunCtx
{
    pd:PhantomData<(P, C)>,
    k:tokio::sync::Mutex<ThoTensor> //state - needs to be dumped to CPU when handling OOM in other stages
}

impl <P, C> TestStage<P, C>
where
    P:Debug,
    C:RunCtx
{
    pub fn new(init:TestStageInit) -> Self {
        Self {
            k: tokio::sync::Mutex::new(Tensor::scalar_tensor(init.k, (Int, ThoDevice::from(init.tho_device_ids).into())).into()),
            pd: PhantomData,
        }
    }
}

unsafe impl <P, C> Sync for TestStage<P, C>
where
    P:Debug,
    C:RunCtx
{}

impl <P, C> Stage for TestStage<P, C>
where
    P:Debug,
    C:RunCtx
{
    type Ctx = C;
    type In<'a> = TestInput;
    type Out = TestOutput;
}

impl <P, C> SyncStage for TestStage<P, C>
where
    P:Debug,
    C:RunCtx
{
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>>
    where
        Self: Sized
    {
        Some(self)
    }

}
impl <P, C> StageWithState for TestStage<P, C>
where
    P:Debug,
    C:RunCtx
{
    type S = ThoTensor;//can be a struct containing a tensor, module etc

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {//ensure state is present on the target device
        async move {
            let mut guard = self.k.lock().await;
            if guard.device() != to_device_idx {
                *guard = guard.copy_to_device(to_device_idx).await?;
            }
            Ok(guard)
        }.boxed()
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "TestStage::do_run_with_state"))]
    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, mut k: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {

        Ok(TestOutput{
            value: (input.value.as_ref() * k.as_ref()).into(),
        })
    }
}

#[async_trait]
impl <P, C> FuturesRunnerMapPurger for TestStage<P, C>
where
    P:Debug + Send + Sync,
    C:RunCtx
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        self.k.lock().await.tho_device().ne(&Cpu.into())
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        let mut guard = self.k.lock().await;
        *guard = guard.copy_to_device(0).await?;

        Ok(())
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        Ok(())
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
       0
    }
}


#[derive(Debug)]
pub struct ProxyTestStage;

const TEST_ENDPOINT: &str = "TestStage";


impl ProxyTestStage {

    pub async fn new<'a>(tho_device_idx: usize, init:TestStageInit) -> Result<TestDynSyncStage<u32, TestRunCtx>, ThoError>//DynSyncStageBox2 is needed to implement SyncStage required by bounds in OOMHandler::run and CudaActorHandle::run_stage

    {
        if GlobalCtx::global().is_ipc() {
            crate::ipc::codec::install_ipc_endpoint::<JsonIpcCodec, TestStageInit>(
                GlobalCtx::global().cuda_actor(tho_device_idx).as_ref(),
                TEST_ENDPOINT,
                &init
            ).await?;
            Ok(TestDynSyncStage::Proxy {
                stage: Arc::new(ProxyStage::proxy(TEST_ENDPOINT)),
            })
        } else {
            Ok(TestDynSyncStage::Local(Arc::new(TestStage::<u32, TestRunCtx>::new(init))))
        }

    }

}

#[derive(Debug)]
pub enum TestDynSyncStage<P, C>
where
    P:Debug + Send + Sync,
    C:RunCtx + 'static
{
    Local(Arc<TestStage<P, C>>),
    Proxy {
        stage: Arc<ProxyStage<TestStage<P, C>>>,
    }
}

impl <P, C> Stage for TestDynSyncStage<P, C>
where
    P:Debug + Send + Sync,
    C:RunCtx + 'static
{
    type Ctx = C;
    type In<'a> = TestInput;
    type Out = TestOutput;
}

impl <P, C> SyncStage for TestDynSyncStage<P, C>
where
    P:Debug + Send + Sync,
    C:RunCtx + 'static
{
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        // Advertise typed state execution without exposing ThoTrainModule to the common runner.
        match self {
            Self::Local(_) => Some(self),
            Self::Proxy { .. } => None,
        }
    }

    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(input, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, ctx),
        }
    }
}

impl <P, C> StageWithState for TestDynSyncStage<P, C>
where
    P:Debug + Send + Sync,
    C:RunCtx + 'static
{
    type S = <TestStage<P, C> as StageWithState>::S;

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

#[async_trait]
impl <P, C> FuturesRunnerMapPurger for TestDynSyncStage<P, C>
where
    P:Debug + Send + Sync,
    C:RunCtx + 'static
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        match self {
            TestDynSyncStage::Local(stage) => {
                stage.has_gpu_data(device_idx).await
            }
            TestDynSyncStage::Proxy { .. } => {
                true
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            TestDynSyncStage::Local(local) => {
                local.purge_gpu(ids_to_keep, device_idx, name).await
            }
            TestDynSyncStage::Proxy { stage,  .. } => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        thoosa_common::ipc::codec::purge_ipc_sync_stage::<thoosa_common::ipc::codec::JsonIpcCodec>(
                            GlobalCtx::global().cuda_actor(device_idx).as_ref(),
                            TEST_ENDPOINT,
                        ).await
                    })
                })?;
                Ok(())
            }
        }
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            TestDynSyncStage::Local(local) => {
                local.drop_gpu(device_idx, name).await
            }
            TestDynSyncStage::Proxy { .. } => {
                Ok(())
            }
        }
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        match self {
            TestDynSyncStage::Local(local) => {
                local.purge_cpu(purge_until).await
            }
            TestDynSyncStage::Proxy { .. } => {
                0
            }
        }
    }
}



pub fn test_with_state_init(init_msg: Vec<u8>, ws: Arc<parking_lot::RwLock<Box<dyn thoosa_common::ws::Workspace>>>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Box<dyn crate::ipc::json_ser::SyncStageIpcWrapper>, thoosa_shared::ThoError> {

    let init = TestStageInit::decode_for_worker::<JsonIpcCodec>(&init_msg, all_cuda_tensors)?;
    debug!("test_with_state_init: init={init:?}");

    Ok(Box::new(SimpWrapper {
        base: Arc::new(TestStage::<(), WokerRunCtx>::new(init)),
        pd: PhantomData::<JsonIpcCodec>
    }))
}


#[derive(Debug)]
struct SimpWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + StageWithState<S=ThoTensor> + Debug + Send + Sync + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = WokerRunCtx>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    base:Arc<S>,
    pd:PhantomData<(Co)>
}

impl <Co, S, I, O>SyncStageIpcWrapper for SimpWrapper<Co, S, I, O>
where
    Co: IpcCodec,
    S: SyncStage + StageWithState<S=ThoTensor>+ Debug + Send + Sync + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = WokerRunCtx>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
        // Keep this endpoint's concrete training-module state behind the erased worker capability.
        Some(self as &dyn ErasedWithState)
    }

    fn purge(&self, tho_device: ThoDevice) -> BoxFuture<Result<(), ThoError>>
    where
        Self: Sync
    {
        // Purge uses the application's CPU-first device index, not tch::Device.
        // self.base.purge_gpu(None, tho_device.tho_device_idx(), "zzz") TODO

        async {Ok(())}.boxed()
    }
}

impl <Co, S, I, O> SyncStageIpcWrapperWithState for SimpWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + StageWithState<S=ThoTensor> + Debug + Send + Sync + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = WokerRunCtx>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    type S = ThoTensor;

    fn pre_run<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<MutexGuard<'a, Self::S>, ThoError>> {
        StageWithState::pre_run(self.base.as_ref(), tho_device.tho_device_idx())
    }

    fn do_run_with_state(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, state: MutexGuard<'_, Self::S>, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<SyncStageIpcResponse, ThoError> {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {

                let input = WorkerStageInput::decode_for_worker::<JsonIpcCodec>(input, all_cuda_tensors)?;

                debug!("TrainTestStep2IpcWrapper::do_run_with_state: input={:?}", input);
                let stage = Arc::clone(&self.base);

                let output = StageWithState::do_run_with_state(Arc::clone(&stage), &input, state, run_ctx)?;
                debug!("TrainTestStep2IpcWrapper::do_run_with_state: output={:?}", output);

                output.encode_from_worker::<JsonIpcCodec>(all_cuda_tensors, tho_device)
            })
        }).map(SyncStageIpcResponse::Encoded)
    }
}


mod tests {
    use std::collections::HashMap;
    use log::debug;
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use tracing::field::debug;
    use thoosa_common::runner::test::proxy_tests_with_state::{TestInput, TestRunCtx, TestStage};
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::SyncStage;
    use crate::ipc::cuda_wokrer::{CudaWorker, InitFn};
    use crate::ipc::tests::run_ipc_test;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::purge::FuturesRunnerMapPurger;
    use crate::runner::test::proxy_tests_with_state::{test_with_state_init, TEST_ENDPOINT, TestStageInit, ProxyTestStage};
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::ws::{mock_ws_simp, FsWorkspace, WsConfigs};


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_with_state_local() -> Result<(), ThoError>{
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        do_test_with_state().await?;

        Ok(())
    }

    #[test]
    fn test_with_state_ipc() -> Result<(), ThoError> {

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

        run_ipc_test(
            async {
                let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
                let test_name = format!("{module}::test_with_state_ipc");
                GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
                let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, true);

                do_test_with_state().await?;

                Ok(())
            },

            async |sock_path| {

                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::from([
                        (
                            TEST_ENDPOINT.to_string(),
                            Box::new(test_with_state_init) as InitFn,
                        ),
                    ]),
                    ThoDevice::from(Cuda(0)),
                ).await
            }
        )
    }

    async fn do_test_with_state() -> Result<(), ThoError> {
        let cuda_actor = GlobalCtx::global().cuda_actor(1);

        let test_stage = Arc::new(ProxyTestStage::new(1, TestStageInit::new(2, 1)).await?);

        GlobalCtx::global().register_cache(("train_step", Arc::clone(&test_stage) as Arc<dyn FuturesRunnerMapPurger>));

        let cuda_tensor = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU))?, 1.into()).await?;

        debug!("test_with_state_target: cuda_tensor={cuda_tensor:?}");

        let input = TestInput {
            value: cuda_tensor,
        };
        let ctx = TestRunCtx;
        let result = cuda_actor.run_stage(Arc::clone(&test_stage), input, 1, &ctx).await?;

        debug!("result={result:?}");
        let cpu_tensor = cuda_actor.copy(result.value, 0.into()).await?;

        let cpu_t = cpu_tensor.t();
        debug!("cpu_tensor={}", cpu_t);
        assert!(Tensor::from_slice(&[2., 2., 2.]).totype(Float).allclose(&cpu_t, 0.01, 0.01, false));

        assert!(!GlobalCtx::global().is_all_on_cpu(1, true));
        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;
        // assert!(GlobalCtx::global().is_all_on_cpu(1, true)); //TODO it fails because proxies always return true from has_gpu_data

        let cuda_tensor = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU)*2)?, 1.into()).await?;

        debug!("test_with_state_target: cuda_tensor={cuda_tensor:?}");

        let input = TestInput {
            value: cuda_tensor,
        };

        let result = cuda_actor.run_stage(test_stage, input, 1, &ctx).await?;

        debug!("result={result:?}");
        let cpu_tensor = cuda_actor.copy(result.value, 0.into()).await?;

        let cpu_t = cpu_tensor.t();
        debug!("cpu_tensor={}", cpu_t);
        assert!(Tensor::from_slice(&[4., 4., 4.]).totype(Float).allclose(&cpu_t, 0.01, 0.01, false));


        Ok(())
    }
}