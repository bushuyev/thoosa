use crate::ipc::codec::{IpcCodec, IpcSyncStageRunInput, JsonIpcCodec};
use crate::ipc::cuda_process::encode_endpoint_payload_bytes;
use crate::ipc::cuda_wokrer::{WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::json_ser::SyncStageIpcWrapper;
use crate::ipc::wraps::{DynSyncStageEnum, ProxyStage, WorkerWrapper};
use crate::runner::common::cachable::{BorrowRef, ThoDataRef};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use futures_util::future::{BoxFuture, FutureExt, ready};
use futures_util::{StreamExt, ready};
use log::debug;
use parking_lot::Mutex;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::{Arc, LazyLock};
use mockall::PredicateBoxExt;
use tch::Tensor;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Data, DynSyncStage, DynSyncStageBox, ErasedStageWithState, MaxParallelCtx, Named, OutId, RunCtx, Stage, SyncStage, WithId};
use thoosa_shared::{CopyToDevice, ThoError};
use tracing::field::debug;


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
    fn encode_from_worker(self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(TestOutput { value: self.value.encode_from_worker(all_cuda_tensors, tho_device)? })
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

    pub async fn new<'a>(tho_device_idx: usize, init: TestStageInit) -> Result<DynSyncStageEnum<TestStage<u32, TestRunCtx>>, ThoError>//DynSyncStageBox2 is needed to implement SyncStage required by bounds in OOMHandler::run and CudaActorHandle::run_stage

    {
        if GlobalCtx::global().is_ipc() {
            crate::ipc::codec::install_ipc_endpoint::<JsonIpcCodec, TestStageInit>(
                GlobalCtx::global().cuda_actor(tho_device_idx).as_ref(),
                TEST_ENDPOINT,
                0,
                &init
            ).await?;
            Ok(DynSyncStageEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy(TEST_ENDPOINT, 0, tho_device_idx)),
            })
        } else {
            Ok(DynSyncStageEnum::Local(Arc::new(TestStage::<u32, TestRunCtx>::new_local(init))))
        }

    }

    pub fn new_local(init:TestStageInit) -> Self {
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


const TEST_ENDPOINT: &str = "TestStage";


pub fn test_simp_init(init_msg: Vec<u8>, ws: Arc<parking_lot::RwLock<Box<dyn thoosa_common::ws::Workspace>>>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<Result<Box<dyn crate::ipc::json_ser::SyncStageIpcWrapper>, thoosa_shared::ThoError>> {

    async move {
        Ok(Box::new(WorkerWrapper::new(
            Arc::new(TestStage::<(), WokerRunCtx>::new_local(TestStageInit::decode_for_worker::<JsonIpcCodec>(&init_msg, all_cuda_tensors)?)),
            PhantomData::<JsonIpcCodec>
        )) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}


mod tests {
    use crate::ipc::cuda_wokrer::{CudaWorker, InitFn};
    use crate::ipc::proxy_tests::{ TEST_ENDPOINT, TestOutput, TestStageInit, test_simp_init};
    use crate::ipc::proxy_tests::{TestInput, TestRunCtx, TestStage};
    use crate::ipc::tests::run_ipc_test;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::ws::{FsWorkspace, WsConfigs, mock_ws_simp};
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::SyncStage;
    use tracing::field::debug;


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

        let test_stage = Arc::new(TestStage::<u32, TestRunCtx>::new(1, TestStageInit::new(2)).await?);

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
