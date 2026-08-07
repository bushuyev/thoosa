use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use futures_util::future::{BoxFuture, FutureExt, ready};
use futures_util::{StreamExt, ready};
use std::sync::Arc;
use log::debug;
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use tch::Tensor;
use tracing::field::debug;
use thoosa_proxy_derive::Proxy;
use thoosa_shared::{CopyToDevice, ThoError};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Data, DynSyncStage, DynSyncStageBox, ErasedStageWithState, MaxParallelCtx, Named, OutId, RunCtx, Stage, SyncStage, WithId};
use crate::ipc::codec::{IpcCodec, IpcSyncStageRunInput, JsonIpcCodec};
use crate::ipc::cuda_process::encode_endpoint_payload_bytes;
use crate::ipc::cuda_wokrer::{WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::json_ser::SyncStageIpcWrapper;
use crate::runner::common::cachable::{BorrowRef, ThoDataRef};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;


#[derive(Debug, Serialize, Deserialize)]
struct TestStageInit {
    k:i64
}

impl TestStageInit {
    pub fn new(k: i64) -> Self {
        Self { k }
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
    k:i64,
    pd:PhantomData<(P, C)>
}

impl <P, C> TestStage<P, C>
where
    P:Debug,
    C:RunCtx
{
    pub fn new(init:TestStageInit) -> Self {
        Self { k: init.k, pd: PhantomData }
    }
}


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
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        Ok(TestOutput{
            value: (input.value.as_ref() * self.k).into(),
        })
    }
}


#[derive(Debug)]
pub struct ProxyTestStage;

const TEST_ENDPOINT: &str = "TestStage";


impl Stage for ProxyTestStage {
    type Ctx = <TestStage<u32, TestRunCtx> as Stage>::Ctx;
    type In<'a> = <TestStage<u32, TestRunCtx> as Stage>::In<'a>;
    type Out = <TestStage<u32, TestRunCtx> as Stage>::Out;
}

impl thoosa_shared::traits::SyncStage for ProxyTestStage {
    fn do_run<'a>(self: std::sync::Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, thoosa_shared::ThoError>
    where
        Self: Sized,
    {
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(async move {

                let endpoint_name = TEST_ENDPOINT;
                let actor = thoosa_common::runner::global_ctx::GlobalCtx::global().cuda_actor(thoosa_shared::traits::RunCtx::tho_device_idx(ctx)).as_ref();
                let input = thoosa_common::ipc::codec::JsonIpcCodec::encode(input)?;
                tracing::debug!("invoke_ipc_sync_stage: endpoint_name={endpoint_name} 1. input={input:?}");


                let response = actor.invoke_endpoint_bytes(
                    encode_endpoint_payload_bytes(endpoint_name, &input).into_bytes()
                ).await?;

                let r = thoosa_common::ipc::codec::JsonIpcCodec::decode(&response);
                tracing::debug!("invoke_ipc_sync_stage: endpoint_name={endpoint_name} response={r:?}");

                r
            })
        })
    }
}


impl ProxyTestStage {

    pub async fn new<'a>(tho_device_idx: usize, init: TestStageInit) -> Result<TestDynSyncStage<u32, TestRunCtx>, ThoError>//DynSyncStageBox2 is needed to implement SyncStage required by bounds in OOMHandler::run and CudaActorHandle::run_stage

    {
        if GlobalCtx::global().is_ipc() {
            crate::ipc::codec::install_ipc_endpoint::<JsonIpcCodec, TestStageInit>(
                GlobalCtx::global().cuda_actor(tho_device_idx).as_ref(),
                TEST_ENDPOINT,
                &init
            ).await?;
            Ok(TestDynSyncStage::Proxy {
                stage: Arc::new(ProxyTestStage::proxy()),
            })
        } else {
            Ok(TestDynSyncStage::Local(Arc::new(TestStage::<u32, TestRunCtx>::new(TestStageInit::new(2)))))
        }

    }

    pub fn proxy() -> Self {
        Self {}
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
        stage: Arc<ProxyTestStage>,
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
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx ) -> Result<Self::Out, ThoError>
    where
        Self: Sized,
    {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(input, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, &TestRunCtx),//TODO TestRunCtx
        }
    }
}


pub fn test_simp_init(init_msg: Vec<u8>, ws: Arc<parking_lot::RwLock<Box<dyn thoosa_common::ws::Workspace>>>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Box<dyn crate::ipc::json_ser::SyncStageIpcWrapper>, thoosa_shared::ThoError> {

    Ok(Box::new(SimpWrapper {
        base: Arc::new(TestStage::<(), WokerRunCtx>::new(TestStageInit::decode_for_worker::<JsonIpcCodec>(&init_msg, all_cuda_tensors)?)),
        pd: PhantomData::<JsonIpcCodec>
    }))
}


#[derive(Debug)]
struct SimpWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
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
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    for<'a> S: Stage<In<'a> = I, Out = O, Ctx = WokerRunCtx>,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    fn do_run(&self, _input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Vec<u8>, ThoError> {
        debug!("SimpWrapper::do_run: 0. input={_input:?}");
        let input = WorkerStageInput::decode_for_worker::<JsonIpcCodec>(_input, all_cuda_tensors)?;

        debug!("SimpWrapper::do_run: 2. input={input:?}");
        let out = Arc::clone(&self.base).do_run(&input, run_ctx)?;


        out.encode_from_worker::<JsonIpcCodec>(all_cuda_tensors, _tho_device)

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
    use thoosa_common::runner::test::proxy_tests::{TestInput, TestRunCtx, TestStage};
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::SyncStage;
    use crate::ipc::cuda_wokrer::{CudaWorker, InitFn};
    use crate::ipc::tests::run_ipc_test;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::test::proxy_tests::{ProxyTestStage, TEST_ENDPOINT, test_simp_init, TestOutput, TestStageInit};
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::ws::{mock_ws_simp, FsWorkspace, WsConfigs};


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_simp_local() -> Result<(), ThoError>{
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        do_test_simp().await?;

        Ok(())
    }

    #[test]
    fn test_simp_ipc() -> Result<(), ThoError>{


        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

        run_ipc_test(
            async {
                let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
                let test_name = format!("{module}::test_simp_ipc");
                GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
                let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, true);

                do_test_simp().await?;


                Ok(())
            },

            async |sock_path| {

                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::from([
                        (
                            TEST_ENDPOINT.to_string(),
                            Box::new(test_simp_init) as InitFn,
                        ),
                    ]),
                    ThoDevice::from(Cuda(0)),
                ).await
            }
        )
    }

    async fn do_test_simp() -> Result<(), ThoError> {
        let cuda_actor = GlobalCtx::global().cuda_actor(1);

        let test_stage = Arc::new(ProxyTestStage::new(1, TestStageInit::new(2)).await?);

        let cuda_tensor = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU))?, 1.into()).await?;

        debug!("test_simp_target: cuda_tensor={cuda_tensor:?}");

        let input = TestInput {
            value: cuda_tensor,
        };
        let ctx = TestRunCtx;

        let result = cuda_actor.run_stage(test_stage, input, 1, &ctx).await?;

        debug!("result={result:?}");

        let cpu_tensor = cuda_actor.copy(result.value, 0.into()).await?;

        let cpu_t = cpu_tensor.t();
        debug!("cpu_tensor={}", cpu_t);

        assert!(Tensor::from_slice(&[2., 2., 2.]).totype(Float).allclose(&cpu_t, 0.01, 0.01, false));
        Ok(())
    }
}