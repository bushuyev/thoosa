use crate::cuda_utils::tensor_mem_size;
use crate::ipc::cuda_worker::{AllTensors, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::proxy::{DynSyncStageWithStateEnum, ProxyStage, StageState};
use crate::ipc::serialization::json::JsonIpcCodec;
use crate::ipc::worker_msg::WorkerInput;
use crate::ipc::wraps::{SyncStageIpcWrapper, WorkerWrapperWithState};
use crate::runner::common::cachable::{BorrowRef, ThoDataRef};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::purge::CachePurger;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::ws::Workspace;
use async_trait::async_trait;
use futures_util::future::{BoxFuture, FutureExt};
use log::debug;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::borrow::Borrow;
use std::collections::HashMap;
use std::fmt::Debug;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Kind::Int;
use tch::Tensor;
use thoosa_common::ipc::state_transfer::StageStateInfo;
use thoosa_common::ipc::state_transfer::StateTransfer;
use thoosa_common::ipc::tensor_info::WorkerTensorInfoPayload;
use thoosa_common::runner::execution::{ExecutionContext, StageExecution, execute_stateful};
use thoosa_common::runner::sync_stage::PrepareStage;
use thoosa_common::runner::sync_stage::StageWithState;
use thoosa_common::tch_error::tch_result;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::ipc_types::{TensorKey, WorkerOutput};
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::{MaxParallelCtx, Named, RunCtx, Stage, WithId};
use thoosa_shared::{CopyToDevice, ThoError, mono_key};
use tokio::sync::MutexGuard;

#[derive(Debug, Serialize, Deserialize)]
pub struct TestStageInitB {
    k_0: i64,
    k_1: i64,
    tho_device_idx: usize,
}

impl TestStageInitB {
    pub fn new(k_0: i64, k_1: i64, tho_device_ids: usize) -> Self {
        Self { k_0, k_1, tho_device_idx: tho_device_ids }
    }
}

#[typetag::serde]
impl WorkerInput for TestStageInitB {}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestInputB {
    value: ThoTensor,
}

impl Input for TestInputB {}

impl WithId for TestInputB {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl BorrowRef for TestInputB {
    fn protect(&self, _device: usize) -> thoosa_common::runner::residency::ResidencyLease {
        thoosa_common::runner::residency::ResidencyLease::default()
    }

    type Out<'a>
        = ThoDataRef<'a, Self>
    where
        Self: 'a;

    fn borrow_ref<'o>(&'o self, to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'o>, ThoError>> + Send
    where
        Self: 'o,
    {
        async move {
            if self.value.tho_device_idx() == to_tho_device_idx {
                Ok(ThoDataRef::Borrowed(self))
            } else {
                Ok(ThoDataRef::Owned(TestInputB {
                    value: t!(self.value.copy_to_device(to_tho_device_idx)).await?,
                }))
            }
        }
    }
}

impl WorkerStageInput for TestInputB {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        Ok(Box::new(TestInputB { value: self.value.clone_ref()? }))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        Ok(TestInputB {
            value: self.value.ref_to_tensor(all_cuda_tensors, remove)?,
        })
    }
}

#[typetag::serde]
impl WorkerInput for TestInputB {}

#[derive(Debug, Serialize, Deserialize)]
pub struct TestOutputB {
    value: ThoTensor,
}
impl Output for TestOutputB {}

impl WithId for TestOutputB {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

#[typetag::serde]
impl WorkerOutput for TestOutputB {}

impl WorkerStageOutput for TestOutputB {
    fn encode_from_worker(self, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(TestOutputB {
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
pub struct TestStageBState {
    //tensors need to be dumped to CPU and removed from State when handling OOM in other stages
    k_0: ThoTensor,
    k_1: ThoTensor,
}

impl TestStageBState {
    pub fn new(k_0: i64, k_1: i64, tho_device_idx: usize) -> Self {
        Self {
            k_0: Tensor::scalar_tensor(k_0, (Int, ThoDevice::from(tho_device_idx).into())).into(),
            k_1: Tensor::scalar_tensor(k_1, (Int, ThoDevice::from(tho_device_idx).into())).into(),
        }
    }
}

impl From<HashMap<String, Tensor>> for TestStageBState {
    fn from(mut map: HashMap<String, Tensor>) -> Self {
        assert_eq!(map.len(), 2);

        TestStageBState {
            k_0: map.remove(K_0).expect("no K_0").into(),
            k_1: map.remove(K_1).expect("no K_1").into(),
        }
    }
}

#[derive(Debug)]
pub struct TestStageB<P, C>
where
    P: Debug + Send + Sync,
    C: RunCtx,
{
    id: SynKey,
    pd: PhantomData<(P, C)>,
    state: tokio::sync::Mutex<StageState<TestStageBState>>, //updatable state (like ThoTrainModule in TrainStep
}

impl<P, C> TestStageB<P, C>
where
    P: Debug + Send + Sync,
    C: RunCtx,
{
    pub async fn new(stage_id: Option<String>, tho_device_idx: usize, init: TestStageInitB) -> Result<Arc<DynSyncStageWithStateEnum<TestStageB<u32, TestRunCtx>>>, ThoError> {
        let stage_id: String = stage_id.expect("stage_id"); //TODO should be generated inside stage based on init data. Fix when WithID is changed to return Cow
        if GlobalCtx::global().is_ipc() {
            Ok(Arc::new(DynSyncStageWithStateEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy(TEST_ENDPOINT, Some(stage_id), tho_device_idx, Box::new(init)).await?),
                init: (),
            }))
        } else {
            Ok(Arc::new(DynSyncStageWithStateEnum::Local(Arc::new(TestStageB::<u32, TestRunCtx>::new_local(stage_id, init)))))
        }
    }

    pub fn new_local(stage_id: String, init: TestStageInitB) -> Self {
        Self {
            id: mono_key!(stage_id, "TestStageB").into(),
            // Preserve both configured tensors so export tests can detect losing either one.
            state: tokio::sync::Mutex::new(StageState::Ready(TestStageBState::new(init.k_0, init.k_1, init.tho_device_idx))),
            pd: PhantomData,
        }
    }
}

const TEST_ENDPOINT: &str = "TestStageB";

pub fn with_state_init<'a>(stage_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn Workspace>>>, _all_cuda_tensors: &'a AllTensors, _tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    let init: Box<dyn Any> = init;
    let init = init.downcast::<TestStageInitB>().unwrap();
    async move {
        debug!("test_with_state_init: init={init:?}");

        Ok(Box::new(WorkerWrapperWithState::new(Arc::new(TestStageB::<(), WokerRunCtx>::new_local(stage_id.expect("no id"), *init)), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }
    .boxed()
}

// Test stages must obey the same auto-trait bounds as production stages.
impl<P, C> Stage for TestStageB<P, C>
where
    P: Debug + Send + Sync,
    C: RunCtx,
{
    type Ctx = C;
    type In<'a> = TestInputB;
    type Out = TestOutputB;
}

impl<P, C> StageExecution for TestStageB<P, C>
where
    P: Debug + Send + Sync,
    C: RunCtx,
{
    fn execute<'a, 'i: 'a, I, IFut>(self: Arc<Self>, input: IFut, execution: &'a ExecutionContext<'_>, ctx: &'a Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<Self::Out, ThoError>>
    where
        Self: 'a,
        Self::In<'i>: Sync,
        Self::Out: Send,
        I: Borrow<Self::In<'i>> + Send + 'a,
        IFut: Future<Output = Result<I, ThoError>> + Send + 'a,
    {
        Box::pin(execute_stateful(self, input, execution, ctx))
    }
}

const K_0: &'static str = "K_0";
const K_1: &'static str = "K_1";

impl<P, C> PrepareStage for TestStageB<P, C>
where
    P: Debug + Send + Sync,
    C: RunCtx,
{
    type S = TestStageBState;
    fn pre_run(&self, _to_device_idx: usize) -> BoxFuture<'_, Result<tokio::sync::MappedMutexGuard<'_, Self::S>, ThoError>> {
        //ensure state is present on the target device
        async move {
            let lock_guard: MutexGuard<StageState<TestStageBState>> = self.state.lock().await;

            debug!("TestStageB::pre_run: lock_guard={lock_guard:?}");

            Ok(MutexGuard::map(lock_guard, |state| state.as_mut()))
        }
        .boxed()
    }
}

impl<P, C> StageWithState for TestStageB<P, C>
where
    P: Debug + Send + Sync,
    C: RunCtx,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "TestStage::do_run_with_state"))]
    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, k: &mut Self::S, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("TestStageB::do_run_with_state: input.value=\n{} \nk.k_0=\n{} \nk.k_1=\n{}", input.value, k.k_0, k.k_1);
        Ok(TestOutputB {
            value: tch_result(input.value.local()?.f_mul(&tch_result(k.k_0.local()?.f_add(&*k.k_1.local()?))?))?.into(),
        })
    }
}

impl<P, C> StateTransfer for TestStageB<P, C>
where
    P: Debug + Send + Sync,
    C: RunCtx,
{
    fn get_state_info(&self, _tho_device: usize) -> BoxFuture<'_, Result<StageStateInfo, ThoError>> {
        async {
            let guard = self.state.lock().await;
            let k_0 = &guard.as_ref().k_0;
            let k_1 = &guard.as_ref().k_1;
            Ok(StageStateInfo::new(
                vec![
                    WorkerTensorInfoPayload::new(TensorKey::stage_state(TEST_ENDPOINT, &self.id, K_0), k_0.kind().into(), k_0.size(), tensor_mem_size(k_0.kind(), &k_0.size())),
                    WorkerTensorInfoPayload::new(TensorKey::stage_state(TEST_ENDPOINT, &self.id, K_1), k_1.kind().into(), k_1.size(), tensor_mem_size(k_1.kind(), &k_1.size())),
                ],
                None, //Some(Box::new(TestStageStateScalars::new(guard.counter)))
            ))
        }
        .boxed()
    }
    fn get_state_tensor(&self, _tho_device: usize, tensor_id: String) -> BoxFuture<'_, Result<Tensor, ThoError>> {
        async move {
            match &*self.state.lock().await {
                StageState::Ready(state) => match tensor_id.as_str() {
                    K_0 => Ok(tch_result(state.k_0.local()?.f_alias())?),
                    K_1 => Ok(tch_result(state.k_1.local()?.f_alias())?),
                    _ => {
                        panic!("unexpected tensor_id={tensor_id}")
                    }
                },
                _ => {
                    panic!("no state")
                }
            }
        }
        .boxed()
    }
    fn set_state_tensor(&self, _tho_device: usize, tensor_id: String, t: Tensor) -> BoxFuture<'_, Result<(), ThoError>> {
        async {
            debug!("TestStageB::set_state_tensor: tensor_id={tensor_id:?}");

            let guard = self.state.lock().await;

            StageState::set_state_tensor(tensor_id, t, guard).await?;

            Ok(())
        }
        .boxed()
    }
}

#[async_trait]
impl<P, C> CachePurger for TestStageB<P, C>
where
    P: Debug + Send + Sync,
    C: RunCtx,
{
    async fn has_gpu_data(&self, _device_idx: usize) -> bool {
        match &*self.state.lock().await {
            StageState::Ready(_) => true,
            StageState::BeingLoaded { tensors, .. } => !tensors.is_empty(),
            StageState::OffLoaded(_) => false,
        }
    }

    async fn drop_gpu(&self, _device_idx: usize, _name: &str) -> Result<(), ThoError> {
        debug!("TestStageB::drop_gpu");
        *self.state.lock().await = StageState::OffLoaded(vec![K_0.to_string(), K_1.to_string()]);
        Ok(())
    }

    async fn purge_cpu(&self, _purge_until: u64) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use crate::ipc::cuda_worker::{CudaWorker, InitFn};
    use crate::ipc::proxy::DynSyncStageWithStateEnum;
    use crate::ipc::proxy_tests_with_state::{TEST_ENDPOINT, TestStageInitB, with_state_init};
    use crate::ipc::proxy_tests_with_state::{TestInputB, TestRunCtx, TestStageB};
    use crate::ipc::tests::run_ipc_test;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::purge::CachePurger;
    use crate::tch_error::tch_result;
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::ws::mock_ws_simp;
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::time::Duration;
    use tch::Kind::Float;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::ThoError;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_with_state_local() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        do_test_with_state().await?;

        Ok(())
    }

    #[test]
    fn test_with_state_ipc() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));

        run_ipc_test(
            module_path!(),
            async {
                do_test_with_state().await?;

                Ok(())
            },
            async |sock_path| CudaWorker::run_worker(sock_path, &ws, HashMap::from([(TEST_ENDPOINT.to_string(), Box::new(with_state_init) as InitFn)]), ThoDevice::from(1usize)).await,
        )
    }

    #[test]
    fn test_state_reconstruction_and_stale_cleanup_after_replacement_ipc() -> Result<(), ThoError> {
        run_ipc_test(
            module_path!(),
            async {
                tokio::time::timeout(Duration::from_secs(45), async {
                    let global = GlobalCtx::global();
                    let actor = global.cuda_actor(1);
                    let ipc = GlobalCtx::global().execution(1).ipc_actor()?;
                    let retired = ipc.worker_id()?;

                    let old_stage = TestStageB::<(), TestRunCtx>::new(Some(7.to_string()), 1, TestStageInitB::new(2, 0, 1)).await?;
                    let old_tensor = actor.copy(tch_result(Tensor::f_ones([1, 3], FLOAT_CPU))?.into(), 1.into()).await?;
                    let old_key = old_tensor.ref_key().unwrap();
                    ipc.purge_stage_state(Some(7.to_string()), TEST_ENDPOINT.into(), retired).await?;
                    actor.restart().await?;

                    let fresh_stage = TestStageB::<(), TestRunCtx>::new(Some(7.to_string()), 1, TestStageInitB::new(3, 0, 1)).await?;
                    let fresh_tensor = actor.copy(tch_result(Tensor::f_ones([1, 3], FLOAT_CPU))?.into(), 1.into()).await?;
                    // Deliberately collide numeric identities across generations.
                    assert_eq!(fresh_tensor.ref_key(), Some(old_key));

                    let fresh_input = || TestInputB { value: fresh_tensor.clone_ref().unwrap() };
                    let error = GlobalCtx::global().execution(1).run_stage(Arc::clone(&old_stage), fresh_input(), 1, &TestRunCtx).await.unwrap_err();
                    assert!(matches!(error.what(), ThoError::CudaBeingRecovered));

                    let error = actor.copy(old_tensor.clone_ref()?, 0.into()).await.unwrap_err();
                    assert!(matches!(error.what(), ThoError::CudaBeingRecovered));

                    // Embedded stale tensor references must also be rejected by
                    // the new worker when the endpoint itself belongs to it.
                    let error = GlobalCtx::global().execution(1).run_stage(Arc::clone(&fresh_stage), TestInputB { value: old_tensor.clone_ref()? }, 1, &TestRunCtx).await.unwrap_err();
                    assert!(matches!(error.what(), ThoError::CudaBeingRecovered));
                    ipc.release_endpoint(Some(7.to_string()), TEST_ENDPOINT.into(), retired);

                    ipc.tensor_releaser().release(old_key, retired);
                    drop(old_stage);
                    drop(old_tensor);
                    // Reconstructed state and live colliding tensor survive every
                    // old cleanup path, including the deferred Drop implementations.
                    for _ in 0..2 {
                        let result = GlobalCtx::global().execution(1).run_stage(Arc::clone(&fresh_stage), fresh_input(), 1, &TestRunCtx).await?;
                        let cpu = actor.copy(result.value, 0.into()).await?;
                        assert!(tch_result(tch_result(Tensor::f_full([1, 3], 3., FLOAT_CPU))?.f_allclose(&*cpu.local()?, 0.01, 0.01, false))?);
                    }
                    assert_eq!(ipc.worker_id()?.generation.0, retired.generation.0 + 1);
                    Ok::<_, ThoError>(())
                })
                .await?
            },
            async |sock| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
                CudaWorker::run_worker(sock, &ws, HashMap::from([(TEST_ENDPOINT.to_owned(), Box::new(with_state_init) as InitFn)]), ThoDevice::from(1usize)).await
            },
        )
    }

    #[test]
    fn test_cancelled_async_invocation_preserves_endpoint_lease_ipc() -> Result<(), ThoError> {
        run_ipc_test(
            module_path!(),
            async {
                let stage = TestStageB::<(), TestRunCtx>::new(Some("cancelled".into()), 1, TestStageInitB::new(2, 3, 1)).await?;
                let DynSyncStageWithStateEnum::Proxy { stage: proxy, .. } = stage.as_ref() else { panic!("expected proxy") };
                let actor = GlobalCtx::global().cuda_actor(1);
                let value = actor.copy(ThoTensor::new((), tch_result(Tensor::f_ones([1, 3], FLOAT_CPU))?)?, 1usize.into()).await?;
                crate::ipc::proxy::async_dispatch_tests::assert_cancelled_invocation_holds_export_lease(Arc::clone(proxy), Box::new(TestInputB { value })).await?;
                // The same worker and endpoint must still restore and execute after
                // the cancelled call and its serialized offload finish.
                let value = actor.copy(ThoTensor::new((), tch_result(Tensor::f_ones([1, 3], FLOAT_CPU))?)?, 1usize.into()).await?;
                let output = GlobalCtx::global().execution(1).run_stage(stage, TestInputB { value }, 1, &TestRunCtx).await?;
                let cpu = actor.copy(output.value, 0usize.into()).await?;
                assert!(tch_result(tch_result(Tensor::f_full([1, 3], 5., FLOAT_CPU))?.f_allclose(&*cpu.local()?, 0.01, 0.01, false))?);
                Ok(())
            },
            async |sock| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
                CudaWorker::run_worker(sock, &ws, HashMap::from([(TEST_ENDPOINT.to_owned(), Box::new(with_state_init) as InitFn)]), ThoDevice::from(1usize)).await
            },
        )
    }

    #[test]
    fn test_concurrent_state_offload_and_invocation_ipc() -> Result<(), ThoError> {
        run_ipc_test(
            module_path!(),
            async {
                // Exercise real multi-tensor export, restore and invocation on two
                // endpoints; neither global OOM ordering nor a shared request ID can
                // serialize this direct-purge workload for us.
                tokio::time::timeout(Duration::from_secs(60), async {
                    let mut tasks = tokio::task::JoinSet::new();
                    for id in 1..=2 {
                        let stage = TestStageB::<(), TestRunCtx>::new(Some(id.to_string()), 1, TestStageInitB::new(id + 1, 3, 1)).await?;
                        let purge = Arc::clone(&stage);
                        tasks.spawn(async move {
                            for _ in 0..12 {
                                purge.purge_gpu(None, 1, "concurrent export test").await?;
                                tokio::task::yield_now().await;
                            }
                            Ok::<_, ThoError>(())
                        });
                        tasks.spawn(async move {
                            let actor = GlobalCtx::global().cuda_actor(1);
                            for _ in 0..12 {
                                let value = actor.copy(ThoTensor::new((), tch_result(Tensor::f_ones([1, 3], FLOAT_CPU))?)?, 1usize.into()).await?;
                                let output = GlobalCtx::global().execution(1).run_stage(Arc::clone(&stage), TestInputB { value }, 1, &TestRunCtx).await?;
                                let cpu = actor.copy(output.value, 0usize.into()).await?;
                                assert!(
                                    tch_result(tch_result(Tensor::f_full([1, 3], (id + 4) as f64, FLOAT_CPU))?.f_allclose(&*cpu.local()?, 0.01, 0.01, false))?,
                                    "offload must preserve both tensors of the endpoint state"
                                );
                            }
                            Ok::<_, ThoError>(())
                        });
                    }
                    while let Some(result) = tasks.join_next().await {
                        result??;
                    }
                    Ok::<_, ThoError>(())
                })
                .await
                .map_err(|_| ThoError::new("concurrent endpoint export timed out"))?
            },
            async |sock| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));
                CudaWorker::run_worker(sock, &ws, HashMap::from([(TEST_ENDPOINT.to_owned(), Box::new(with_state_init) as InitFn)]), ThoDevice::from(1usize)).await
            },
        )
    }

    #[test]
    fn test_distinct_stateful_endpoints_ipc() -> Result<(), ThoError> {
        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)?));

        run_ipc_test(
            module_path!(),
            async {
                do_test_distinct_stateful_endpoints().await?;

                Ok(())
            },
            async |sock_path| CudaWorker::run_worker(sock_path, &ws, HashMap::from([(TEST_ENDPOINT.to_string(), Box::new(with_state_init) as InitFn)]), ThoDevice::from(1usize)).await,
        )
    }

    async fn do_test_distinct_stateful_endpoints() -> Result<(), ThoError> {
        let cuda_actor = GlobalCtx::global().cuda_actor(1);
        // Reuse one initializer name with different ids to prove both worker-side state instances remain independently invokable.
        let first_stage = TestStageB::<(), TestRunCtx>::new(Some(1.to_string()), 1, TestStageInitB::new(2, 0, 1)).await?;
        let second_stage = TestStageB::<(), TestRunCtx>::new(Some(2.to_string()), 1, TestStageInitB::new(3, 0, 1)).await?;

        let first_input = TestInputB {
            value: cuda_actor.copy(ThoTensor::new((), tch_result(Tensor::f_ones([1, 3], FLOAT_CPU))?)?, 1.into()).await?,
        };
        let first_result = GlobalCtx::global().execution(1).run_stage(Arc::clone(&first_stage), first_input, 1, &TestRunCtx).await?;
        debug!("do_test_distinct_stateful_endpoints: first_result={first_result:?}");

        let first_cpu = first_result.value.copy_to_device(0).await?; //shallow_clone to avoid id being dropped before copy starts

        debug!("do_test_distinct_stateful_endpoints: first_cpu=\n{}", first_cpu);
        assert!(tch_result(tch_result(tch_result(Tensor::f_from_slice(&[2.0f32, 2.0, 2.0]))?.f_view([1, 3]))?.f_allclose(&*first_cpu.local()?, 0.01, 0.01, false))?);

        let second_input = TestInputB {
            value: cuda_actor.copy(ThoTensor::new((), tch_result(Tensor::f_ones([1, 3], FLOAT_CPU))?)?, 1.into()).await?,
        };
        let second_result = GlobalCtx::global().execution(1).run_stage(Arc::clone(&second_stage), second_input, 1, &TestRunCtx).await?;
        let second_cpu = cuda_actor.copy(second_result.value, 0.into()).await?;

        debug!("do_test_distinct_stateful_endpoints: second_cpu=\n{}", second_cpu);
        assert!(tch_result(tch_result(tch_result(Tensor::f_from_slice(&[3.0f32, 3.0, 3.0]))?.f_view([1, 3]))?.f_allclose(&*second_cpu.local()?, 0.01, 0.01, false))?);

        // Dropping the last proxy reference must asynchronously release its worker key so that the same id can be installed again.
        drop(first_stage);
        tokio::task::yield_now().await;
        tokio::time::sleep(Duration::from_secs(3)).await;

        TestStageB::<(), TestRunCtx>::new(Some(1.to_string()), 1, TestStageInitB::new(2, 0, 1)).await?;

        // A repeated stage_id must fail without replacing the already-installed stage.
        let Err(e) = TestStageB::<(), TestRunCtx>::new(Some(2.to_string()), 1, TestStageInitB::new(4, 0, 1)).await else {
            panic!("should fail to install same endpoint")
        };

        println!("do_test_distinct_stateful_endpoints: e={e:?}");

        // Explicit cleanup uses the same owning control channel as proxy destruction.
        let ipc = GlobalCtx::global().execution(1).ipc_actor()?;
        ipc.release_endpoint(Some(1.to_string()), TEST_ENDPOINT.to_string(), ipc.worker_id()?);

        Ok(())
    }

    //installs Stage, runs it, dumps its state to CPU, runs it again restoring state
    async fn do_test_with_state() -> Result<(), ThoError> {
        let cuda_actor = GlobalCtx::global().cuda_actor(1);

        // Installation and proxy invocation must use the same id to select this stateful worker stage.
        let test_stage = TestStageB::<(), TestRunCtx>::new(Some(1.to_string()), 1, TestStageInitB::new(2, 0, 1)).await?;

        GlobalCtx::global().register_cache(("train_step".into(), Arc::clone(&test_stage) as Arc<dyn CachePurger>));

        let cuda_tensor = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU))?, 1.into()).await?;

        debug!("test_with_state_target: 0. cuda_tensor={cuda_tensor:?}");

        let input = TestInputB { value: cuda_tensor };
        let ctx = TestRunCtx;
        let result = GlobalCtx::global().execution(1).run_stage(Arc::clone(&test_stage), input, 1, &ctx).await?;

        debug!("test_with_state_target: result={result:?}");
        let cpu_tensor = cuda_actor.copy(result.value, 0.into()).await?;

        let cpu_t = cpu_tensor.into_local()?;
        debug!("test_with_state_target: cpu_tensor={}", cpu_t);
        assert!(Tensor::from_slice(&[2., 2., 2.]).totype(Float).allclose(&cpu_t, 0.01, 0.01, false));

        assert!(!GlobalCtx::global().is_all_on_cpu_force_check(1).await);
        let dump_all_to_cpu_result = GlobalCtx::global().dump_all_to_cpu(1, "test").await;
        debug!("test_with_state_target: dump_all_to_cpu_result={dump_all_to_cpu_result:?}");
        assert!(dump_all_to_cpu_result.is_ok());

        let cuda_tensor = cuda_actor.copy(ThoTensor::new((), Tensor::ones(&[1, 3], FLOAT_CPU) * 2)?, 1.into()).await?;

        debug!("test_with_state_target: 1. cuda_tensor={cuda_tensor:?}");

        let input = TestInputB { value: cuda_tensor };

        let result = GlobalCtx::global().execution(1).run_stage(test_stage, input, 1, &ctx).await?;

        debug!("result={result:?}");
        let cpu_tensor = cuda_actor.copy(result.value, 0.into()).await?;

        let cpu_t = cpu_tensor.into_local()?;
        debug!("cpu_tensor={}", cpu_t);
        assert!(Tensor::from_slice(&[4., 4., 4.]).totype(Float).allclose(&cpu_t, 0.01, 0.01, false));

        debug!("all ok");
        Ok(())
    }
}
