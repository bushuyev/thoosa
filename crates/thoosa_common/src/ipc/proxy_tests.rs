use crate::ipc::cuda_wokrer::{AllTensors, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use crate::ipc::serialization::json::JsonIpcCodec;
use crate::ipc::worker_msg::WorkerInput;
use crate::ipc::wraps::SyncStageIpcWrapper;
use crate::ipc::wraps::WorkerWrapper;
use crate::runner::common::cachable::{BorrowRef, ThoDataRef};
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::ws::Workspace;
use futures_util::future::{BoxFuture, FutureExt};
use log::debug;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::borrow::Cow;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_common::runner::sync_stage::SyncStage;
use thoosa_common::tch_error::tch_result;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::ipc_types::WorkerOutput;
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::{MaxParallelCtx, Named, RunCtx, Stage, WithId};
use thoosa_shared::{CopyToDevice, ThoError};
use tracing::field::debug;

#[derive(Debug, Serialize, Deserialize)]
struct TestStageInitA {
    k: i64,
}

impl TestStageInitA {
    pub fn new(k: i64) -> Self {
        Self { k }
    }
}

#[typetag::serde]
impl WorkerInput for TestStageInitA {}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestInputA<'a> {
    value: ThoTensor,
    fail_with_oom: bool,
    name: Cow<'a, str>,
}

impl<'a> Input for TestInputA<'a> {}

impl<'a> WithId for TestInputA<'a> {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

#[typetag::serde]
impl WorkerInput for TestInputA<'static> {}

impl<'a> BorrowRef for TestInputA<'a> {
    fn protect(&self, _device: usize) -> thoosa_common::runner::residency::ResidencyLease {
        thoosa_common::runner::residency::ResidencyLease::default()
    }

    // Rebind the input's borrowed field to the output borrow so it matches Stage::In<'o> for every caller lifetime.
    type Out<'o>
        = ThoDataRef<'o, TestInputA<'o>>
    where
        Self: 'o;

    fn borrow_ref<'o>(&'o self, to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'o>, ThoError>> + Send
    where
        Self: 'o,
    {
        async move {
            if self.value.tho_device_idx() == to_tho_device_idx {
                Ok(ThoDataRef::Borrowed(self))
            } else {
                Ok(ThoDataRef::Owned(TestInputA {
                    value: t!(self.value.copy_to_device(to_tho_device_idx)).await?,
                    fail_with_oom: self.fail_with_oom,
                    // Own the copied field so the returned value does not borrow from this temporary operation.
                    name: Cow::Owned(self.name.to_string()),
                }))
            }
        }
    }
}

impl<'a> WorkerStageInput for TestInputA<'a> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        Ok(Box::new(TestInputA {
            value: self.value.clone_ref()?,
            fail_with_oom: self.fail_with_oom,
            // typetag requires a 'static value, so only the IPC copy owns fields that were borrowed by the caller.
            name: Cow::Owned(self.name.to_string()),
        }))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        Ok(TestInputA {
            value: self.value.ref_to_tensor(all_cuda_tensors, remove)?,
            fail_with_oom: self.fail_with_oom,
            name: self.name.clone(),
        })
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestOutputA {
    value: ThoTensor,
}
impl Output for TestOutputA {}

#[typetag::serde]
impl WorkerOutput for TestOutputA {}

impl WithId for TestOutputA {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl WorkerStageOutput for TestOutputA {
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(TestOutputA {
            value: self.value.encode_from_worker(all_cuda_tensors, tho_device)?,
        })
    }
}

#[derive(Debug)]
pub struct TestRunCtx; //corresponds to RunCtxImpl

impl Named for TestRunCtx {}

impl MaxParallelCtx for TestRunCtx {
    fn set_max_other_parallel(&mut self, _permit: u8) {}

    fn max_other_parallel(&self) -> u8 {
        0
    }
}

impl WithId for TestRunCtx {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl RunCtx for TestRunCtx {
    fn tho_device_idx(&self) -> usize {
        1
    }

    fn branch(&mut self, _with_new_cache_area: bool) -> Self
    where
        Self: Sized,
    {
        TestRunCtx
    }

    fn add_purge_gpu_closure(&mut self, _cl: Box<dyn Fn() -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>) {}
}

#[derive(Debug)]
pub struct TestStageA<P, C>
where
    P: Debug,
    C: RunCtx,
{
    k: i64,
    pd: PhantomData<(P, C)>,
}

impl<P, C> TestStageA<P, C>
where
    P: Debug,
    C: RunCtx,
{
    // The initializer is private to this test module, so the constructor is too.
    async fn new<'a>(tho_device_idx: usize, init: TestStageInitA) -> Result<DynSyncStageEnum<TestStageA<u32, TestRunCtx>>, ThoError> //DynSyncStageBox2 is needed to implement SyncStage required by bounds in OOMHandler::run and CudaActorHandle::run_stage
    {
        if GlobalCtx::global().is_ipc() {
            Ok(DynSyncStageEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy(TEST_ENDPOINT, Some(0.to_string()), tho_device_idx, Box::new(init)).await?),
                init: (),
            })
        } else {
            Ok(DynSyncStageEnum::Local(Arc::new(TestStageA::<u32, TestRunCtx>::new_local(init))))
        }
    }

    fn new_local(init: TestStageInitA) -> Self {
        Self { k: init.k, pd: PhantomData }
    }
}

impl<P, C> Stage for TestStageA<P, C>
where
    P: Debug,
    C: RunCtx,
{
    type Ctx = C;
    type In<'a> = TestInputA<'a>;
    type Out = TestOutputA;
}

impl<P, C> SyncStage for TestStageA<P, C>
where
    P: Debug,
    C: RunCtx,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "TestStage::do_run"))]
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        if input.fail_with_oom {
            return Err(ThoError::oom());
        }
        debug("TestStage::do_run: input={input:?}");
        Ok(TestOutputA {
            value: tch_result(input.value.local()?.f_mul_scalar(self.k))?.into(),
        })
    }
}

const TEST_ENDPOINT: &str = "TestStage";

pub fn test_simp_init<'a>(_stage_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    debug!("test_simp_init_2: 0");
    let init: Box<dyn Any> = init;

    let init = init.downcast::<TestStageInitA>().unwrap();

    async move {
        debug!("test_simp_init_2: 1");

        Ok(Box::new(WorkerWrapper::new(Arc::new(TestStageA::<(), WokerRunCtx>::new_local(*init)), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }
    .boxed()
}

mod tests {
    #[cfg(test)]
    use crate::ipc::cuda_wokrer::CudaWorker;
    #[cfg(test)]
    use crate::ipc::cuda_wokrer::InitFn;
    #[cfg(test)]
    use crate::ipc::proxy_tests::TEST_ENDPOINT;
    use crate::ipc::proxy_tests::TestStageInitA;
    #[cfg(test)]
    use crate::ipc::proxy_tests::test_simp_init;
    use crate::ipc::proxy_tests::{TestInputA, TestRunCtx, TestStageA};
    #[cfg(test)]
    use crate::ipc::tests::run_ipc_test;
    use crate::runner::global_ctx::GlobalCtx;
    #[cfg(test)]
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    #[cfg(test)]
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    #[cfg(test)]
    use crate::ws::mock_ws_simp;
    use log::debug;
    #[cfg(test)]
    use log::info;
    #[cfg(test)]
    use parking_lot::RwLock;
    #[cfg(test)]
    use std::collections::HashMap;
    use std::sync::Arc;
    use tch::Kind::Float;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_shared::ThoError;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_simp_tensor_local() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        do_test_tensor_simp().await?;

        Ok(())
    }

    #[test]
    fn test_simp_tensor_ipc() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));

        run_ipc_test(
            module_path!(),
            async {
                do_test_tensor_simp().await?;

                debug!("test_simp_ipc ok");

                Ok(())
            },
            async |sock_path| {
                return {
                    let r = CudaWorker::run_worker(sock_path, &ws, HashMap::from([(TEST_ENDPOINT.to_string(), Box::new(test_simp_init) as InitFn)]), ThoDevice::from(1usize)).await;

                    info!("test_simp_ipc: r={r:?}");

                    Ok(())
                };
            },
        )
    }

    async fn do_test_tensor_simp() -> Result<(), ThoError> {
        debug!("do_test_simp");
        let cuda_actor = GlobalCtx::global().cuda_actor(1);

        let test_stage = Arc::new(TestStageA::<u32, TestRunCtx>::new(1, TestStageInitA::new(2)).await?);

        debug!("do_test_simp: test_stage={test_stage:?}");

        let cuda_tensor = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU))?, 1.into()).await?;

        debug!("do_test_simp: cuda_tensor={cuda_tensor:?}");

        let input = TestInputA {
            value: cuda_tensor,
            fail_with_oom: false,
            name: "ref".into(),
        };
        let ctx = TestRunCtx;

        let result = GlobalCtx::global().execution(1).run_stage(test_stage, input, 1, &ctx).await?;

        debug!("do_test_simp: result={result:?}");

        let cpu_tensor = cuda_actor.copy(result.value, 0.into()).await?;

        let cpu_t = cpu_tensor.into_local()?;
        debug!("do_test_simp: cpu_tensor={}", cpu_t);

        assert!(Tensor::from_slice(&[2., 2., 2.]).totype(Float).allclose(&cpu_t, 0.01, 0.01, false));
        Ok(())
    }

    #[test]
    fn test_oom_error_ipc() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));

        run_ipc_test(
            module_path!(),
            async {
                let cuda_actor = GlobalCtx::global().cuda_actor(1);

                let test_stage = Arc::new(TestStageA::<u32, TestRunCtx>::new(1, TestStageInitA::new(2)).await?);

                let cuda_tensor = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU))?, 1.into()).await?;

                debug!("test_simp_target: cuda_tensor={cuda_tensor:?}");

                let input = TestInputA {
                    value: cuda_tensor,
                    fail_with_oom: true,
                    name: "ref".into(),
                };
                let ctx = TestRunCtx;

                let result = GlobalCtx::global().execution(1).run_stage(test_stage, input, 1, &ctx).await;
                debug!("result={result:?}");
                let Err(e) = result else {
                    panic!("not an error");
                };

                assert!(e.is_oom());

                Ok(())
            },
            async |sock_path| CudaWorker::run_worker(sock_path, &ws, HashMap::from([(TEST_ENDPOINT.to_string(), Box::new(test_simp_init) as InitFn)]), ThoDevice::from(1usize)).await,
        )
    }
}
