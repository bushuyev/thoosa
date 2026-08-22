use std::any::Any;
use crate::dims::{Batch, Dims};
use crate::ipc::codec::{IpcCodec, IpcSyncStageRunInput, JsonIpcCodec};
use crate::ipc::cuda_process::{encode_endpoint_payload_bytes, CudaIPCActorHandle};
use crate::ipc::cuda_wokrer::{WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::json_ser::{ErasedWithState, SyncStageIpcResponse, SyncStageIpcWrapper, SyncStageIpcWrapperWithState};
use crate::ipc::wraps::{DynSyncStageEnum, DynSyncStageWithStateEnum, ProxyStage, WorkerWrapperWithState};
use crate::runner::common::cachable::{BorrowRef, ThoDataRef};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::purge::FuturesRunnerMapPurger;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::train_nets::ThoModuleWithStore;
use async_trait::async_trait;
use futures_util::future::{BoxFuture, FutureExt, ready};
use futures_util::{StreamExt, ready};
use log::debug;
use parking_lot::{Mutex, RwLock};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Device::Cpu;
use tch::Kind::Int;
use tch::Tensor;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{CachableIdTr, Data, DynSyncStage, DynSyncStageBox, DynSyncStageOutId, ErasedStageState, ErasedStageWithState, MaxParallelCtx, Named, OutId, RunCtx, Stage, StageWithState, SyncStage, WithId};
use thoosa_shared::{CopyToDevice, ThoError, mono_key};
use tokio::sync::{MutexGuard, RwLockReadGuard};
use tracing::field::debug;
use crate::ipc::worker_msg::{CudaWorkerMsgIn, CudaWorkerMsgOut, OpResult, Operation, WorkerInput, WorkerOutput};
use crate::ipc::wraps2::{DynSyncStageEnum2, DynSyncStageWithStateEnum2, ProxyStage2};

#[derive(Debug, Serialize, Deserialize)]
pub struct TestStageInitB {
    k:i64,
    tho_device_ids:usize
}

impl TestStageInitB {
    pub fn new(k: i64, tho_device_ids:usize) -> Self {
        Self { k, tho_device_ids }
    }
}

#[typetag::serde]
impl WorkerInput for TestStageInitB{

}

impl WorkerStageInput for TestStageInitB {
    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        let mut input = JsonIpcCodec::decode::<Self>(&input)?;

        Ok(input)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestInputB {
    value: ThoTensor
}

impl Input for TestInputB {}

impl WithId for TestInputB {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl BorrowRef for TestInputB {
    type Out<'a> = ThoDataRef<'a, Self> where Self: 'a;

    fn borrow_to<'o>(&'o self, to_tho_device_idx: usize) -> impl Future<Output=Result<Self::Out<'o>, ThoError>> + Send
    where
        Self: 'o
    {
        async move {
            if self.value.device() == to_tho_device_idx {
                Ok(ThoDataRef::Borrowed(self))
            } else {
                Ok(ThoDataRef::Owned(TestInputB {
                    value: track_await(self.value.copy_to_device(to_tho_device_idx)).await?,
                }))
            }
        }
    }
}

impl WorkerStageInput for TestInputB {
    fn clone_refs(&self) -> Result<Self, ThoError> {
        Ok(TestInputB {
            value: self.value.clone_refs()?,
        })
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, remove: bool) -> Result<Self, ThoError> {
        Ok(TestInputB {
            value: ThoTensor::ref_to_tensor_(all_cuda_tensors, &self.value, remove)?,
        })
    }

    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        let mut input = JsonIpcCodec::decode::<Self>(&input)?;

        debug!("SimpWrapper::do_run: 1. input={input:?}");

        input.value = ThoTensor::ref_to_tensor(all_cuda_tensors, input.value)?;

        Ok(input)
    }
}



#[typetag::serde]
impl WorkerInput for TestInputB {

}


#[derive(Debug, Serialize, Deserialize)]
pub struct TestOutputB {
    value: ThoTensor
}
impl Output for TestOutputB {}

impl WithId for TestOutputB {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

#[typetag::serde]
impl WorkerOutput for TestOutputB {

}

impl WorkerStageOutput for TestOutputB {
    fn encode_from_worker(self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(TestOutputB { value: self.value.encode_from_worker(all_cuda_tensors, tho_device)? })
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
pub struct TestStageB<P, C>
where
    P:Debug,
    C:RunCtx
{
    pd:PhantomData<(P, C)>,
    k:tokio::sync::Mutex<ThoTensor> //state - needs to be dumped to CPU when handling OOM in other stages
}

impl <P, C> TestStageB<P, C>
where
    P:Debug,
    C:RunCtx
{
    pub async fn new(stage_id: u32, tho_device_idx: usize, init: TestStageInitB) -> Result<DynSyncStageWithStateEnum2<TestStageB<u32, TestRunCtx>>, ThoError> {
        if GlobalCtx::global().is_ipc() {
            let actor:&dyn Any = GlobalCtx::global().cuda_actor(tho_device_idx).as_ref();
            let ipc_actor:&CudaIPCActorHandle = actor.downcast_ref().unwrap();

            let rs = ipc_actor.send_msg(CudaWorkerMsgIn {
                request_id: 0,
                operation: Operation::InstallStage {
                    endpoint_name: TEST_ENDPOINT.to_string(),
                    stage_id,
                    init_data: Box::new(init),
                },
            }).await?;

            debug!("TestStage::new: rs={rs:?}");

            match rs {
                CudaWorkerMsgOut { request_id, status } => {
                    match status {
                        OpResult::Ok(_) => {
                            Ok(DynSyncStageWithStateEnum2::Proxy {
                                stage: Arc::new(ProxyStage2::proxy(TEST_ENDPOINT, stage_id, tho_device_idx)),
                                init: (),
                            })

                        }
                        OpResult::Failed{ msg, is_oom, needs_reset } => {
                            Err(ThoError::from_worker(msg, is_oom, needs_reset))
                        }
                    }
                }
            }


        } else {
            Ok(DynSyncStageWithStateEnum2::Local(Arc::new(TestStageB::<u32, TestRunCtx>::new_local(init))))
        }
    }

    pub fn new_local(init: TestStageInitB) -> Self {
        Self {
            k: tokio::sync::Mutex::new(Tensor::scalar_tensor(init.k, (Int, ThoDevice::from(init.tho_device_ids).into())).into()),
            pd: PhantomData
        }
    }

}

unsafe impl <P, C> Sync for TestStageB<P, C>
where
    P:Debug,
    C:RunCtx
{}

impl <P, C> Stage for TestStageB<P, C>
where
    P:Debug,
    C:RunCtx
{
    type Ctx = C;
    type In<'a> = TestInputB;
    type Out = TestOutputB;
}

impl <P, C> SyncStage for TestStageB<P, C>
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
impl <P, C> StageWithState for TestStageB<P, C>
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

        Ok(TestOutputB {
            value: (input.value.as_ref() * k.as_ref()).into(),
        })
    }
}

#[async_trait]
impl <P, C> FuturesRunnerMapPurger for TestStageB<P, C>
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

const TEST_ENDPOINT: &str = "TestStageB";


pub fn with_state_init(init:Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn thoosa_common::ws::Workspace>>>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> BoxFuture<Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    let init: Box<dyn Any> = init;
    let init = init.downcast::<TestStageInitB>().unwrap();
    async move {
        debug!("test_with_state_init: init={init:?}");

        Ok(Box::new(WorkerWrapperWithState::new(
            Arc::new(TestStageB::<(), WokerRunCtx>::new_local(*init)),
            PhantomData::<JsonIpcCodec>
        )) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}

mod tests {
    use std::time::Duration;
    use crate::ipc::codec::{JsonIpcCodec, install_ipc_endpoint};
    use crate::ipc::cuda_wokrer::{CudaWorker, InitFn, InitFn2};
    use crate::ipc::proxy_tests_with_state::{TestInputB, TestRunCtx, TestStageB};
    use crate::ipc::proxy_tests_with_state::{TEST_ENDPOINT, TestStageInitB, with_state_init};
    use crate::ipc::tests::run_ipc_test;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::purge::FuturesRunnerMapPurger;
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

                CudaWorker::run_worker2(
                    sock_path,
                    &ws,
                    HashMap::from([

                    ]),
                    HashMap::from([
                        (
                            TEST_ENDPOINT.to_string(),
                            Box::new(with_state_init) as InitFn2,
                        ),
                    ]),
                    ThoDevice::from(Cuda(0)),
                ).await
            }
        )
    }

    #[test]
    fn test_distinct_stateful_endpoints_ipc() -> Result<(), ThoError> {

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));

        run_ipc_test(
            async {
                let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
                let test_name = format!("{module}::test_with_state_ipc");
                GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
                let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, true);

                do_test_distinct_stateful_endpoints().await?;

                Ok(())
            },

            async |sock_path| {

                CudaWorker::run_worker2(
                    sock_path,
                    &ws,
                    HashMap::from([

                    ]),
                    HashMap::from([
                        (
                            TEST_ENDPOINT.to_string(),
                            Box::new(with_state_init) as InitFn2,
                        ),
                    ]),
                    ThoDevice::from(Cuda(0)),
                ).await
            }
        )
    }


    async fn do_test_distinct_stateful_endpoints() -> Result<(), ThoError> {
        let cuda_actor = GlobalCtx::global().cuda_actor(1);
        // Reuse one initializer name with different ids to prove both worker-side state instances remain independently invokable.
        let first_stage = Arc::new(TestStageB::<(), TestRunCtx>::new(1, 1, TestStageInitB::new(2, 1)).await?);
        let second_stage = Arc::new(TestStageB::<(), TestRunCtx>::new(2, 1, TestStageInitB::new(3, 1)).await?);

        let first_input = TestInputB { value: cuda_actor.copy(ThoTensor::new((), Tensor::f_ones([1, 3], FLOAT_CPU)?)?, 1.into()).await? };
        let first_result = cuda_actor.run_stage(Arc::clone(&first_stage), first_input, 1, &TestRunCtx).await?;
        let first_cpu = cuda_actor.copy(first_result.value, 0.into()).await?;

        debug!("do_test_distinct_stateful_endpoints: first_cpu=\n{}", first_cpu.t_ref());
        assert!(Tensor::f_from_slice(&[2.0f32, 2.0, 2.0])?.f_view([1, 3])?.f_allclose(first_cpu.t_ref(), 0.01, 0.01, false)?);

        let second_input = TestInputB { value: cuda_actor.copy(ThoTensor::new((), Tensor::f_ones([1, 3], FLOAT_CPU)?)?, 1.into()).await? };
        let second_result = cuda_actor.run_stage(Arc::clone(&second_stage), second_input, 1, &TestRunCtx).await?;
        let second_cpu = cuda_actor.copy(second_result.value, 0.into()).await?;

        debug!("do_test_distinct_stateful_endpoints: second_cpu=\n{}", second_cpu.t_ref());
        assert!(Tensor::f_from_slice(&[3.0f32, 3.0, 3.0])?.f_view([1, 3])?.f_allclose(second_cpu.t_ref(), 0.01, 0.01, false)?);


        // Dropping the last proxy reference must asynchronously release its worker key so that the same id can be installed again.
        drop(first_stage);
        tokio::time::timeout(Duration::from_secs(5), async {
             TestStageB::<(), TestRunCtx>::new(1, 1, TestStageInitB::new(2, 1)).await
        }).await??;

        // A repeated stage_id must fail without replacing the already-installed stage.
       let Err(e) = TestStageB::<(), TestRunCtx>::new( 2,1,  TestStageInitB::new(4, 1)).await else {
           panic!("should fail to install same endpoint")
       };

        println!("do_test_distinct_stateful_endpoints: e={e:?}");

        cuda_actor.drop_endpoint(1, TEST_ENDPOINT.to_string()).await?;

        Ok(())
    }

    async fn do_test_with_state() -> Result<(), ThoError> {
        let cuda_actor = GlobalCtx::global().cuda_actor(1);

        // Installation and proxy invocation must use the same id to select this stateful worker stage.
        let test_stage = Arc::new(TestStageB::<(), TestRunCtx>::new(1, 1, TestStageInitB::new(2, 1)).await?);

        GlobalCtx::global().register_cache(("train_step", Arc::clone(&test_stage) as Arc<dyn FuturesRunnerMapPurger>));

        let cuda_tensor = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU))?, 1.into()).await?;

        debug!("test_with_state_target: cuda_tensor={cuda_tensor:?}");

        let input = TestInputB {
            value: cuda_tensor,
        };
        let ctx = TestRunCtx;
        let result = cuda_actor.run_stage(Arc::clone(&test_stage), input, 1, &ctx).await?;

        debug!("test_with_state_target: result={result:?}");
        let cpu_tensor = cuda_actor.copy(result.value, 0.into()).await?;

        let cpu_t = cpu_tensor.t();
        debug!("test_with_state_target: cpu_tensor={}", cpu_t);
        assert!(Tensor::from_slice(&[2., 2., 2.]).totype(Float).allclose(&cpu_t, 0.01, 0.01, false));

        assert!(!GlobalCtx::global().is_all_on_cpu(1, true));
        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;
        // assert!(GlobalCtx::global().is_all_on_cpu(1, true)); //TODO it fails because proxies always return true from has_gpu_data

        let cuda_tensor = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU)*2)?, 1.into()).await?;

        debug!("test_with_state_target: cuda_tensor={cuda_tensor:?}");

        let input = TestInputB {
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
