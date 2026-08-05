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
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{MaxParallelCtx, Named, RunCtx, Stage, SyncStage, WithId};
use crate::ipc::codec::{IpcCodec, IpcSyncStageRunInput, JsonIpcCodec};
use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::ipc::json_ser::SyncStageIpcWrapper;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;

#[derive(Debug, Serialize, Deserialize)]
pub struct TestInput{
    value: ThoTensor
}
impl Input for TestInput{}

// impl WorkerStageInput for TestInput {
//     fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
//         TestInput {
//             value: (),
//         }
//     }
// }

#[derive(Debug, Serialize, Deserialize)]
pub struct TestOutput{
    value: ThoTensor
}
impl Output for TestOutput{}


#[derive(Debug)]
pub struct TestRunCtx;

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
pub struct TestStage<P>
where
    P:Debug
{
    pd:PhantomData<P>
}

impl <P> TestStage<P>
where
    P:Debug
{
    pub fn new() -> Self {
        Self { pd: PhantomData }
    }
}

impl <P> Stage for TestStage<P>
where
    P:Debug

{
    type Ctx = TestRunCtx;
    type In<'a> = TestInput;
    type Out = TestOutput;
}

impl <P> SyncStage for TestStage<P>
where
    P:Debug
{
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        Ok(TestOutput{
            value: (input.value.as_ref() * 2).into(),
        })
    }
}




const TEST_ENDPOINT: &str = "TestStage";

#[derive(Debug, Proxy)]
#[proxy(
    name = "ProxyTestStage",
    endpoint = "TEST_ENDPOINT",
    new_args = "tho_device_idx: usize",
    endpoint_init = "test_simp_init",
    wrapper = "SimpWrapper"
)]
struct TestStageSimpTarget {
    base:Arc<TestStage<()>>
}

impl TestStageSimpTarget {
    pub fn new() -> Self {
        Self {
            base: Arc::new(TestStage::new()),
        }
    }
}

#[derive(Debug)]
struct SimpWrapper {
    stage:Arc<TestStageSimpTarget>
}
impl SyncStageIpcWrapper for SimpWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let input = JsonIpcCodec::decode::<IpcSyncStageRunInput>(input)?;
        debug!("SimpWrapper::do_run: 0. input={input:?}");

        let input = JsonIpcCodec::decode::<TestInput>(&input.input())?;

        debug!("SimpWrapper::do_run: 1. input={input:?}");

        let input = TestInput {
            value: ThoTensor::ref_to_tensor(all_cuda_tensors, input.value)?,
        };

        debug!("SimpWrapper::do_run: 2. input={input:?}");
        let mut ctx = TestRunCtx;
        let out = Arc::clone(&self.stage.base).do_run(&input, &ctx)?;

        JsonIpcCodec::encode(&TestOutput{
            value: out.value.worker_tensor_to_ref(all_cuda_tensors, &tho_device)?,
        })

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
    use crate::runner::test::proxy_tests::{ProxyTestStage, TestStageSimpTarget, TEST_ENDPOINT, test_simp_init};
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::ws::{mock_ws_simp, FsWorkspace, WsConfigs};

    #[test]
    fn test_simp_target() -> Result<(), ThoError>{
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

        run_ipc_test(
            async {
                let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
                let test_name = format!("{module}::test_simp_target");
                GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
                let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, true);
                let cuda_actor = GlobalCtx::global().cuda_actor(1);

                let test_stage = Arc::new(ProxyTestStage::new(1).await?);

                let worker_tensor  = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU))?, 1.into()).await?;

                debug!("test_simp_target: worker_tensor={worker_tensor:?}");

                let input = TestInput{
                    value: worker_tensor,
                };
                let ctx = TestRunCtx;
                let result = test_stage.do_run(&input, &ctx)?;

                debug!("result={result:?}");

                let cpu_tensor = cuda_actor.copy(result.value, 0.into()).await?;

                let cpu_t = cpu_tensor.t();
                debug!("cpu_tensor={}", cpu_t);

                assert!(Tensor::from_slice(&[2., 2., 2.]).totype(Float).allclose(&cpu_t, 0.01, 0.01, false));


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
}