use thoosa_common::tch_error::tch_result;
use crate::ipc::cuda_process::IPC_SOCK_ENV;
use crate::ipc::cuda_wokrer::CudaWorker;
use crate::ipc::wraps::SyncStageIpcWrapper;
use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelStep};
use crate::runner::common::cachable::{BorrowRef, ConsumeData, DevicesWrap};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::global_ctx::mock_global_ctx::{GlobalCtxMockGuard, MockGlobalCtxBuilder};
use crate::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoStageAdjuster, OOMHandler};
use crate::runner::run_ctx::RunCtxImpl;
use crate::runner::test::FailUntil::NaxParallel;
use crate::runner::test::{ClcOOMRunner, FailUntil, OOMRunnerConfig, run_mul_task_invocations};
// use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, FailUntil, OOMRunnerConfig};
// use crate::runner::test::FailUntil::NaxParallel;
use crate::runner::test_utils::parse_numeric_part;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::{ThoTensor, ThoTensorInner};
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
// use crate::torch_runner::tests::load_and_clc::loader::LoadedTensor;
// use crate::torch_runner::tests::load_and_clc::{mul_check, sync_clc_endpoint_init, sync_test_tensor_loader_endpoint_init, LoadAndClcStage, LoadAndClcStageId, SYNC_CLC_ENDPOINT, SYNC_TEST_TENSOR_LOADER_ENDPOINT};
// use crate::torch_runner::tests::load_and_clc::clc::{ClcStage, ClcStageOutput};
// use crate::torch_runner::tests::model_test::{run_model_test_pipeline, sync_model_test_endpoint_inits};
// use crate::torch_runner::tests::model_test::common::ModelTestId;
use crate::utils::lin_for_size;
use log::{LevelFilter, debug, info, warn};
use serde::de::Unexpected::Option;
use std::collections::HashMap;
use std::ffi::OsString;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::atomic::Ordering::Acquire;
use std::sync::atomic::{AtomicUsize, Ordering};
use tch::Device::Cuda;
use tch::Kind::Float;
use tch::{Device, Tensor};
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::perf_logging::{PERF, format_perf_map};
use thoosa_shared::test_utils::init_test_logging;
use thoosa_shared::tracked_await::track_await;
use thoosa_common::runner::sync_stage::ConsumeResult;
use thoosa_shared::traits::{Adjustable, AsyncStage, OutId, RunCtx, Stage, WithId};
use thoosa_common::runner::sync_stage::{SyncStage};
use thoosa_shared::{ThoError, mono_key, panic_to_err_async};
use thoosa_shims::context::ThoCudaContext;
use tokio::sync::Notify;
use tokio::task;
use tokio::task::{JoinSet, coop};
use tracing::field::debug;

#[cfg(test)]
#[ctor::ctor]
pub fn init() {
    thoosa_shared::test_utils::init_test_logging();
}


pub fn run_ipc_test<'f, FM, FW>(
    module_path:&str,
    main_f:FM,
    worker_f:FW
) -> Result<(), ThoError>
where
    FM:Future<Output=Result<(), ThoError>>,
    FW:AsyncFn(OsString)->anyhow::Result<()>,
{
    run_ipc_test_with_runtime(module_path, main_f, worker_f, tokio::runtime::Builder::new_multi_thread().worker_threads(32))
}

// Let dispatch regressions constrain parent runtime capacity independently of
// the CUDA worker. Existing integration tests retain their original runtime.
fn run_ipc_test_with_runtime<FM, FW>(module_path: &str, main_f: FM, worker_f: FW, main_builder: &mut tokio::runtime::Builder) -> Result<(), ThoError>
where
    FM: Future<Output = Result<(), ThoError>>,
    FW: AsyncFn(OsString) -> anyhow::Result<()>,
{
    if let Some(sock_path) = std::env::var_os(IPC_SOCK_ENV) {
        init_test_logging();

        debug!("run_ipc_test: sock_path={sock_path:?}");


        let tho_cuda_context = ThoCudaContext::new(0)?;


        let worker_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(32)
            .enable_all()
            .on_thread_start(move || {

                if let Err(err) = tho_cuda_context.bind_to_thread() {
                    panic!("Failed to create context: {err:?}"); //TODO device
                }
            })
            .thread_name_fn(|| {
                static TOKIO_THREAD_ID: AtomicUsize = AtomicUsize::new(0);
                let id = TOKIO_THREAD_ID.fetch_add(1, Ordering::Relaxed);
                format!("test-cuda-worker-{id}")
            })
            .build()?;

        debug!("run_ipc_test: worker_rt={worker_rt:?}");

        worker_rt.block_on(async {
            debug!("run_ipc_test: in block_on");
            let is_installed_context = crate::runner::global_ctx::MockGlobalCtx::is_installed_context();
            is_installed_context.expect().return_const(false);

            let r = panic_to_err_async!(worker_f(sock_path));

            info!("worker perf:\n{}", format_perf_map(&PERF));
            info!("worker done: r={r:?}");

            // Preserve the original tests' behavior by propagating worker failures through the harness.
            r
        })?;

    } else {
        let main_rt = main_builder
            .enable_all()
            .thread_name_fn(|| {
                static TOKIO_THREAD_ID: AtomicUsize = AtomicUsize::new(0);
                let id = TOKIO_THREAD_ID.fetch_add(1, Ordering::Relaxed);
                format!("test-main-{id}")
            })
            .build()?;

        main_rt.block_on(async {
            let _ctx = init_ipc_test(module_path).await?;
            let r = main_f.await;
            info!("main done: r={r:?}");

            GlobalCtx::global().shutdown().await;

            info!("main perf:\n{}", format_perf_map(&PERF));

            // Preserve the original tests' behavior by propagating main-process assertion failures.
            r
        })?;
    }

    Ok(())
}


pub async fn init_ipc_test(module:&str) -> Result<impl GlobalCtxMockGuard, ThoError> {
    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;

    let module = module.split_once("::").map(|(_, module)| module).unwrap_or("");
    let test_name = format!("{module}::{}", get_test_name());

    debug!("init_ipc_test2: module={module} test_name={test_name:?}");

    GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string(), "--no-capture".to_string()]).await?;
    let ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, true);

    debug!("init_ipc_test2: done");

    Ok(ctx)
}

fn get_test_name() -> String {
    let full_name = std::env::var("NEXTEST_TEST_NAME")
        .ok()
        .or_else(|| std::thread::current().name().map(str::to_owned))
        .unwrap_or_else(|| "<unknown>".into());

    full_name
        .rsplit("::")
        .next()
        .unwrap()
        .to_owned()
}

#[cfg(test)]
mod tests {
    use crate::tch_error::tch_result;
    use anyhow::Context as _;
    use super::run_ipc_test;
    use crate::ipc::cuda_wokrer::{AllTensors, CudaWorker, InitFn};
    use crate::ipc::worker_msg::{DebugOutput, OpResult, WorkerInput};
    use crate::ipc::wraps::SyncStageIpcWrapper;
    use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelStep};
    use crate::runner::common::cachable::{BorrowRef, ConsumeData, DevicesWrap};
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::{GlobalCtxMockGuard, MockGlobalCtxBuilder};
    use crate::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoStageAdjuster, OOMHandler};
    use crate::runner::run_ctx::RunCtxImpl;
    use crate::runner::test::FailUntil::NaxParallel;
    use crate::runner::test::{ClcOOMRunner, FailUntil, OOMRunnerConfig, run_mul_task_invocations};
    // use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, FailUntil, OOMRunnerConfig};
    // use crate::runner::test::FailUntil::NaxParallel;
    use crate::runner::test_utils::parse_numeric_part;
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::{ThoTensor, ThoTensorInner};
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    use crate::torch_runner::tests::load_and_clc::clc::{ClcStage, ClcStageOutput};
    use crate::torch_runner::tests::load_and_clc::loader::{LoadedTensor, SyncTestTensorLoader};
    use crate::torch_runner::tests::load_and_clc::{LoadAndClcStage, LoadAndClcStageId, SYNC_CLC_ENDPOINT, SYNC_TEST_TENSOR_LOADER_ENDPOINT, do_real_oom_dump_and_split, mul_check, mul_check_with_exp_f, sync_clc_endpoint_init, sync_test_tensor_loader_endpoint_init};
    use crate::torch_runner::tests::model_test::common::ModelTestId;
    use crate::torch_runner::tests::model_test::{run_model_test_pipeline, sync_model_test_endpoint_inits};
    use futures_util::FutureExt;
    use futures_util::future::{BoxFuture, join_all};
    use log::{LevelFilter, debug, info};
    use parking_lot::RwLock;
    use serde::de::Unexpected::Option;
    use std::any::Any;
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::sync::Arc;
    use std::sync::atomic::Ordering::Acquire;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use tch::{Device, Tensor};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::test_utils::init_test_logging;
    use thoosa_shared::tracked_await::track_await;
    use thoosa_common::runner::sync_stage::ConsumeResult;
    use thoosa_shared::traits::{Adjustable, AsyncStage, OutId, RunCtx, Stage, WithId};
    use thoosa_common::runner::sync_stage::{SyncStage};
    use thoosa_shared::{ThoError, mono_key, os_thread_id};
    use tokio::task::JoinSet;
    use tracing::field::debug;

    use crate::cuda_utils::cuda_mem;
    use crate::dims::AnyDims;
    use crate::utils::lin_for_size;
    use crate::ws::{Workspace, mock_ws_simp};
    use thoosa_named_derive::Named;
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::perf_logging::{PERF, format_perf_map};
    use thoosa_shims::context::ThoCudaContext;

    fn check_sum_result(mul_k: f64, size: [i64; 2], resource: &str, out: &ThoTensor<ClcStageOutput, SynKey>) ->Result<(), ThoError>{
        let self_k = parse_numeric_part(resource).unwrap_or(1.0);
        let expected = lin_for_size(self_k * mul_k, size, out.t_ref().kind(), &out.t_ref().device())?
            .sum_dim_intlist(1, false, out.t_ref().kind())
            .view((-1, 1));
        let ok = expected.allclose(&out.t_ref(), 0.1, 0.1, false);
        debug!("check_sum_result: resource={resource} expected=\n{expected} actual=\n{}", out.t_ref());
        assert!(ok);

        Ok(())
    }

    fn sync_clc_endpoint_inits() -> HashMap<String, InitFn> {
        HashMap::from([
            (
                SYNC_TEST_TENSOR_LOADER_ENDPOINT.to_string(),
                Box::new(sync_test_tensor_loader_endpoint_init) as InitFn
            ),
            (
                SYNC_CLC_ENDPOINT.to_string(),
                Box::new(sync_clc_endpoint_init) as InitFn
            )
        ])
    }

    const SYNC_CONSUME_TEST_ENDPOINT: &str = "SyncConsumeTestStage";

    #[derive(Debug, Named)]
    pub struct SyncConsumeTestStage {
        id: SynKey,
    }

    impl SyncConsumeTestStage {
        fn new() -> Self {
            Self { id: mono_key!("SyncConsumeTestStage", "stage").into() }
        }

        fn consume_input(&self, input: &ThoTensor<LoadedTensor, SynKey>) -> Result<ThoTensor<LoadedTensor, SynKey>, ThoError> {
            if tch_result(input.t_ref().f_double_value(&[0, 0]))? < 0.0 {
                return Err(ThoError::new("consume test failure"));
            }
            ThoTensor::new(self.out_id(input.id()), tch_result(input.t_ref().f_mul_scalar(2.0))?)
        }
    }

    impl Adjustable for SyncConsumeTestStage {}

    impl OutId for SyncConsumeTestStage {
        fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
            &self.id + input_id
        }
    }

    impl Stage for SyncConsumeTestStage {
        type Ctx = RunCtxImpl;
        type In<'a> = ThoTensor<LoadedTensor, SynKey>;
        type Out = ThoTensor<LoadedTensor, SynKey>;
    }

    impl SyncStage for SyncConsumeTestStage {
        fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _: &Self::Ctx) -> Result<Self::Out, ThoError> {
            self.consume_input(input)
        }

        fn consume<'a>(self: Arc<Self>, input: Self::In<'a>, _ctx: Self::Ctx) -> ConsumeResult<Self::In<'a>, Self::Out> {
            match self.consume_input(&input) {
                Ok(output) => ConsumeResult::Ok(output),
                Err(error) => ConsumeResult::Failed { input, error },
            }
        }
    }

    async fn new_sync_consume_test_stage(tho_device_idx: usize) -> Result<crate::ipc::proxy::DynSyncStageEnum<SyncConsumeTestStage>, ThoError> {
        // ProxyStage2 carries typed consume requests and releases worker inputs only after success.
        Ok(crate::ipc::proxy::DynSyncStageEnum::Proxy {
            stage: Arc::new(crate::ipc::proxy::ProxyStage::proxy_with_input_cloner(
                SYNC_CONSUME_TEST_ENDPOINT,
                Some(0.to_string()),
                tho_device_idx,
                Box::new(()),
                Arc::new(|input: &ThoTensor<LoadedTensor, SynKey>| Ok(Box::new(input.clone_ref()?) as Box<dyn WorkerInput>)),
            ).await?),
            init: (),
        })
    }

    fn sync_consume_test_endpoint_init<'a>(tage_id: std::option::Option<String>, init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
        use futures_util::FutureExt;
        async move {
            // This stage has no initializer fields, but still uses the typed install message.
            Ok(Box::new(crate::ipc::wraps::WorkerWrapper::<crate::ipc::serialization::json::JsonIpcCodec, _, _, _>::new(Arc::new(SyncConsumeTestStage::new()), std::marker::PhantomData)) as Box<dyn SyncStageIpcWrapper>)
        }.boxed()
    }

    #[derive(Debug)]
    struct ConsumeTestInput {
        input: ThoTensor<LoadedTensor, SynKey>,
    }

    impl ConsumeTestInput {
        fn new(input: ThoTensor<LoadedTensor, SynKey>) -> Self {
            Self { input }
        }
    }

    impl ConsumeData for ConsumeTestInput {
        type O = ThoTensor<LoadedTensor, SynKey>;

        fn consume<R: Send, F, P>(self, to_device_idx: usize, run_ctx: P, do_run: F) -> impl Future<Output=ConsumeResult<Self::O, R>> + Send
        where
            F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
            P: RunCtx,
        {
            async move { do_run(self.input, run_ctx) }
        }

        fn burp(input: Self::O) -> impl Future<Output=Self> + Send {
            async move { Self::new(input) }
        }
    }

    fn sync_consume_endpoint_inits() -> HashMap<String, InitFn> {
        HashMap::from([(
            SYNC_CONSUME_TEST_ENDPOINT.to_string(),
            Box::new(sync_consume_test_endpoint_init) as InitFn,
        )])
    }

    #[test]
    fn test_async_actor_dispatch_with_no_spare_runtime_threads_ipc() -> Result<(), ThoError> {
        super::run_ipc_test_with_runtime(
            module_path!(),
            async {
                let stage = Arc::new(new_sync_consume_test_stage(1).await?);
                // H2D arena staging legitimately uses spawn_blocking. Finish
                // input copies before exhausting that pool to isolate invocation.
                let actor = GlobalCtx::global().cuda_actor(1);
                let mut inputs = Vec::new();
                for _ in 0..12 {
                    let source: ThoTensor = tch_result(tch_result(Tensor::f_from_slice(&[1.0f32, 2.0]))?.f_view([1, 2]))?.into();
                    let reference = actor.copy(source, 1usize.into()).await?;
                    inputs.push(ThoTensor::<LoadedTensor, SynKey>::new_ref(mono_key!("dispatch", "input").into(), reference.ref_key().unwrap(), reference.size(), reference.kind(), None, reference.ref_worker_id().unwrap()));
                }
                // Occupy the only blocking thread too. Every actor invocation,
                // reader, and reply must progress on this single async thread.
                let (release, blocked) = std::sync::mpsc::channel::<()>();
                let (started, running) = tokio::sync::oneshot::channel();
                let blocker = tokio::task::spawn_blocking(move || {
                    let _ = started.send(());
                    let _ = blocked.recv_timeout(std::time::Duration::from_secs(30));
                });
                running.await.context("blocking-thread reservation failed")?;
                let result = tokio::time::timeout(std::time::Duration::from_secs(20), async {
                    let mut jobs = JoinSet::new();
                    for (index, input) in inputs.into_iter().enumerate() {
                        let stage = Arc::clone(&stage);
                        jobs.spawn(async move {
                            let actor = GlobalCtx::global().cuda_actor(1);
                            let output = GlobalCtx::global().get_runner(1).submit(12, async move {
                                let ctx = RunCtxImpl::new(1, "async dispatch");
                                match index % 3 {
                                    0 => actor.run_stage(stage, Arc::new(DevicesWrap::wrap(input)), 1, &ctx).await,
                                    1 => actor.run_stage_with_select(stage, Arc::new(DevicesWrap::wrap(input)), None, 1, &ctx).await,
                                    _ => match actor.consume_stage(stage, ConsumeTestInput::new(input), 1, ctx).await? {
                                        ConsumeResult::Ok(output) => Ok(output),
                                        ConsumeResult::Failed { error, .. } => Err(error),
                                    },
                                }
                            }, None).await?;
                            let output = actor.copy(output.strip_types(), 0usize.into()).await?;
                            assert_eq!(tch_result(output.t_ref().f_double_value(&[0, 0]))?, 2.0);
                            assert_eq!(tch_result(output.t_ref().f_double_value(&[0, 1]))?, 4.0);
                            Ok::<_, ThoError>(())
                        });
                    }
                    while let Some(result) = jobs.join_next().await { result??; }
                    Ok::<_, ThoError>(())
                }).await;
                drop(release);
                blocker.await?;
                result.context("IPC dispatch stalled without spare runtime threads")?
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(sock_path, &ws, sync_consume_endpoint_inits(), ThoDevice::from(Cuda(0))).await
            },
            tokio::runtime::Builder::new_current_thread().max_blocking_threads(1),
        )
    }

    #[test]
    fn test_consume_stage_releases_inputs_only_on_success_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {

                let cuda_actor = GlobalCtx::global().cuda_actor(1);
                let stage = Arc::new(new_sync_consume_test_stage(1).await?);
                let oom_handler = OOMHandler::<NoInputAdjuster, NoStageAdjuster, NoRunCtxAdjuster, ()>::new(vec![], None);

                let input_tensor:ThoTensor = tch_result(tch_result(Tensor::f_from_slice(&[1.0f32, 2.0]))?.f_view([1, 2]))?.into();
                let input_ref = cuda_actor.copy(input_tensor.shallow_clone(), ThoDevice::from(1)).await?;
                let input_worker_id = input_ref.ref_worker_id().unwrap();
                let input_key = input_ref.ref_key().ok_or_else(|| ThoError::new("consume test expected worker input ref"))?;
                let input:ThoTensor<LoadedTensor, SynKey> = ThoTensor::new_ref(mono_key!("consume-success", "input").into(), input_key, input_ref.size(), input_ref.kind(), None, input_worker_id);

                let output = oom_handler.consume(&stage, ConsumeTestInput::new(input), RunCtxImpl::new(1, "consume-success"), NoContexts).await?;
                let output_worker_id = output.ref_worker_id().unwrap();
                let output_key = output.ref_key().ok_or_else(|| ThoError::new("consume test expected worker output ref"))?;
                let new_ref = ThoTensor::new_ref((), output_key, output.size(), output.kind(), None, output_worker_id);
                let output_cpu = cuda_actor.copy(new_ref, ThoDevice::from(0)).await?;

                assert!(tch_result(output_cpu.t_ref().f_allclose(&tch_result(tch_result(Tensor::f_from_slice(&[2.0f32, 4.0]))?.f_view([1, 2]))?, 0.0, 0.0, false))?);

                let consumed_input = cuda_actor.copy(ThoTensor::new_ref((), input_key, vec![1, 2], Float, None, input_worker_id), ThoDevice::from(0)).await;
                assert!(consumed_input.is_err(), "successful consume must remove the worker input tensor");

                let failed_tensor:ThoTensor = tch_result(tch_result(Tensor::f_from_slice(&[-1.0f32, 2.0]))?.f_view([1, 2]))?.into();
                let failed_ref = cuda_actor.copy(failed_tensor, ThoDevice::from(1)).await?;
                let failed_worker_id = failed_ref.ref_worker_id().unwrap();
                let failed_key = failed_ref.ref_key().ok_or_else(|| ThoError::new("consume test expected failed worker input ref"))?;
                let failed_input:ThoTensor<LoadedTensor, SynKey> = ThoTensor::new_ref(mono_key!("consume-failure", "input").into(), failed_key, failed_ref.size(), failed_ref.kind(), None, failed_worker_id);
                let failed_result = oom_handler.consume(&stage, ConsumeTestInput::new(failed_input.shallow_clone()), RunCtxImpl::new(1, "consume-failure"), NoContexts).await;
                assert!(failed_result.is_err());
                let new_ref = ThoTensor::new_ref((), failed_key, vec![1, 2], Float, None, failed_worker_id);
                let preserved_input = cuda_actor.copy(new_ref, ThoDevice::from(0)).await?;
                debug!("preserved_input=\n{}", preserved_input.t_ref());
                assert!(tch_result(preserved_input.t_ref().f_allclose(&tch_result(tch_result(Tensor::f_from_slice(&[-1.0f32, 2.0]))?.f_view([1, 2]))?, 0.0, 0.0, false))?);

                Ok(())
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_consume_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }


    async fn run_load_and_clc_ipc_cuda_worker(test_name: &str, resources: &[&'static str], use_matmul: bool) -> Result<(), ThoError> {
        debug!("run_load_and_clc_ipc_cuda_worker: test_name={test_name} resources={resources:?}");

        let tho_device_idx = 1;
        let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
        let size = [3, 10];
        const MUL_K: f64 = 2.;
        let clc_dim = if use_matmul { size[0] } else { size[1] };
        // All resource stages use the same CLC operation, so share one proxy and its typed worker endpoint identity.
        let clc_task = Arc::new(ClcStage::new(SynKey::new([mono_key!(0, "sync_task_0")]), MUL_K, vec![], clc_dim, Float, Device::Cuda(0).into(), use_matmul).await);

        // cuda_actor.install_endpoint(serde_json::to_string(&SyncClcTaskInit::new(mono_key!(0, "sync_task_0").into(), MUL_K, clc_dim, vec![], false))?).await?;

        let mut tasks = JoinSet::new();
        let counter = Arc::new(AtomicUsize::new(0));
        for resource in resources {
            let resource = *resource;
            debug!("run_load_and_clc_ipc_cuda_worker: resource={resource:?}");

            let cuda_actor = cuda_actor;
            let counter = Arc::clone(&counter);
            let clc_task = Arc::clone(&clc_task);
            tasks.spawn(async move {
                debug!("run_load_and_clc_ipc_cuda_worker: spawned resource={resource:?}");

                let input_id = SynKey::new([mono_key!(resource, "resource key")]);
                let self_k = parse_numeric_part(resource).unwrap_or(1.0);
                let cpu_tensor = lin_for_size(self_k, size, Float, &Device::Cpu)?;

                let worker_ref: ThoTensor<LoadedTensor, SynKey> = track_await!(cuda_actor.copy(cpu_tensor.into(), tho_device_idx.into())).await?.to_types(mono_key!(0, "todo").into())?;
                debug!("run_load_and_clc_ipc_cuda_worker: worker_ref={worker_ref:?}");

                let loaded = Arc::new(DevicesWrap::wrap(worker_ref));
                //further logging is duplicated somehow
                counter.fetch_add(1, Acquire);
                debug!("run_load_and_clc_ipc_cuda_worker: loaded={loaded:?} counter={}", counter.load(Acquire));

                let mut ctx = RunCtxImpl::new(tho_device_idx, "ipc_load_and_clc");
                let output_ref = Arc::clone(&clc_task).run(Arc::clone(&loaded), ctx.branch(false)).await?;
                let output_ref = BorrowRef::borrow_ref(&output_ref, ctx.tho_device_idx()).await?;
                let out_key = output_ref.ref_key().ok_or_else(|| ThoError::new(format!("ProxySyncClcTask expected output ref from worker, got {output_ref:?}")))?;


                debug!("run_load_and_clc_ipc_cuda_worker: out_key={out_key:?} output_ref.id={:?}", output_ref.id());

                let tensor = GlobalCtx::global().cuda_actor(1/*TODO take ctx*/)
                    .copy(ThoTensor::new_ref((), out_key, output_ref.size(), output_ref.kind(), None, output_ref.ref_worker_id().unwrap()), 0.into()).await?;
                let cpu_out: ThoTensor<ClcStageOutput, SynKey> = ThoTensor::new(output_ref.id().clone(), tensor.t())?;


                assert_eq!(cpu_out.id(), &clc_task.out_id(loaded.id()));
                if use_matmul {
                    // This test branch validates the returned tensor; surface validation failures.
                    check_sum_result(MUL_K, size, resource, &cpu_out)?;
                } else {
                    mul_check(MUL_K, size.into(), resource, &cpu_out, 0.1, 0.1);
                }


                Ok::<(), ThoError>(())
            });
        }


        let result = async {
            while let Some(joined) = tasks.join_next().await {
                joined.context("join IPC load/mul task")??;
            }

            Ok::<(), ThoError>(())
        }.await;

        debug!("run_load_and_clc_ipc_cuda_worker: counter={:?}", counter.load(Acquire));

        // run_ipc_test2 owns actor shutdown; returning the task result avoids a second shutdown request to a stopped actor.
        result
    }

    async fn run_model_test_ipc_cuda_worker<F, Fut>(_test_name: &str, run: F) -> Result<(), ThoError>
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output=Result<(), ThoError>>,
    {
        // run_ipc_test2 owns actor shutdown and propagates this pipeline result after cleanup.
        run().await
    }

    //no max_parallel tests in IPC mode since concurrency is handled in the main process

    #[test]
    fn test_load_and_clc_no_oom_tensor_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                run_load_and_clc_ipc_cuda_worker("test_load_and_clc_ipc_cuda_worker_no_oom_tensor", &["resource_1", "resource_2", "resource_3"], false).await
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_clc_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }


    #[test]
    fn test_load_and_clc_sum_no_oom_tensor_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                run_load_and_clc_ipc_cuda_worker("test_load_and_clc_ipc_cuda_worker_sum_no_oom_tensor", &["resource_1", "resource_2", "resource_3"], true).await
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_clc_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }



    #[test]
    fn test_simp_copy() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                let tho_device_idx = 1;
                let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);

                let smoke = lin_for_size(1.0, [3, 10], Float, &Device::Cpu)?;
                let expected_smoke = smoke.copy();
                let worker_ref = track_await!(cuda_actor.copy(smoke.into(), tho_device_idx.into())).await?;
                assert!(worker_ref.ref_key().is_some(), "expected IPC H2D copy to return worker tensor ref, got {worker_ref:?}");

                let cpu_copy = track_await!(cuda_actor.copy(worker_ref, 0.into())).await?;
                assert!(expected_smoke.allclose(&cpu_copy.t_ref(), 0.1, 0.1, false), "IPC worker smoke copy changed tensor values");

                Ok(())
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_clc_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }


    #[test]
    fn test_load_and_clc_oom_splits_tensor_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                let tho_device_idx = 1;
                let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
                async {
                    const MUL_K: f64 = 2.;
                    let size = [3, 140];
                    let load_and_clc = Arc::new(LoadAndClcStage::new(
                        LoadAndClcStageId("ipc_load_and_clc_oom_splits_id"),
                        MUL_K,
                        vec![FailUntil::MemLimit(40)],
                        size.into(),
                        ClcOOMRunner::new(
                            OOMRunnerConfig::Splits {},
                            Box::new(move |id| SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                                id,
                                1,
                                &size,
                                (Float, Device::Cpu),
                                BuildAs::Sum,
                            ).expect("single dim output builder").copy_partial_outputs_to_device(0)),
                        ),
                        Float,
                        Device::Cuda(0).into(),
                        true,
                    ).await);

                    run_mul_task_invocations(
                        Arc::clone(&load_and_clc),
                        &["resource_1"],
                        tho_device_idx,
                        0,
                        |_device_idx, _cache_area, name| RunCtxImpl::new(1, name),
                        move |res, out| {
                            crate::torch_runner::tests::load_and_clc::mul_check_with_exp_f(MUL_K, res, out.as_ref(), 0.01, 0.01, |k, t| {
                                Ok(lin_for_size(k, size, t.kind(), &t.device())? //same as in SyncTestTensorLoader::do_run
                                    .sum_dim_intlist(1, false, t.kind())
                                    .view((-1, 1)))
                            })
                        },
                        false,
                    ).await?;
                    Ok::<(), ThoError>(())
                }.await
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_clc_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }

    #[test]
    fn test_load_and_clc_oom_dump_to_cpu_and_splits_tensor_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                let resources = (0..1)//3
                    .map(|i| &*format!("resource_{i}").leak())
                    .collect::<Vec<_>>();

                let tho_device_idx = 1;
                let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
                async {
                    const MUL_K: f64 = 2.;
                    let size = [3, 100];
                    let load_and_clc = Arc::new(LoadAndClcStage::new(
                        LoadAndClcStageId("ipc_load_and_clc_oom_max_parallel_and_splits_id"),
                        MUL_K,
                        vec![FailUntil::DumpedToCpu, FailUntil::MemLimit(12)],
                        size.into(),
                        ClcOOMRunner::new(
                            OOMRunnerConfig::DumpedToCpuAndSplits,
                            Box::new(move |id| SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                                id,
                                0,
                                &size,
                                (Float, Device::Cpu),
                                BuildAs::Sum,
                            ).expect("single dim output builder").copy_partial_outputs_to_device(0)),
                        ),
                        Float,
                        Device::Cuda(0).into(),
                        true,
                    ).await);

                    run_mul_task_invocations(
                        Arc::clone(&load_and_clc),
                        resources.as_slice(),
                        tho_device_idx,
                        0,
                        |_device_idx, _cache_area, name| RunCtxImpl::new(1, name),
                        move |res, out| {
                            mul_check_with_exp_f(MUL_K, res, out.as_ref(), 0.01, 0.01, |k, t| {
                                Ok(lin_for_size(k, size, t.kind(), &t.device())? //same as in SyncTestTensorLoader::do_run
                                    .sum_dim_intlist(1, false, t.kind())
                                    .view((-1, 1)))
                            });
                        },
                        false,
                    ).await?;

                    let msg = cuda_actor.debug_worker().await?;
                    debug!("test_load_and_clc_oom_dump_to_cpu_and_splits_tensor_ipc: msg={msg:?}");

                    let OpResult::Ok(Some(debug_data)) = msg.status else {
                        panic!("no debug data");
                    };

                    let Ok(debug_data) = (debug_data as Box<dyn Any>).downcast::<DebugOutput>() else {
                        panic!("Can't downcast to DebugOutput");
                    };

                    //checking that worker stage receives data splits
                    let expected_debug_data = [
                            "SyncClcStage::do_run_with_state: input.size=[3, 100]",//original call without adjuster
                            "SyncClcStage::do_run_with_state: input.size=[3, 100]",//call after dump_all_to_cpu with full size input
                            "SyncClcStage::do_run_with_state: input.size=[3, 50]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 25]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 12]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 12]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 12]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 12]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 12]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 12]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 12]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 12]",
                            "SyncClcStage::do_run_with_state: input.size=[3, 4]",
                    ];
                    let debug_data_snap = debug_data.debug_data_snap();
                    assert_eq!(expected_debug_data.len(), debug_data_snap.len());


                    for (expected, actual) in expected_debug_data.into_iter().zip(debug_data_snap.into_iter()) {
                        assert_eq!(expected, actual);
                    }



                    Ok::<(), ThoError>(())
                }.await
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_clc_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }


    #[test]
    fn test_load_and_clc_oom_dump_to_cpu_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                let tho_device_idx = 1;
                let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
                async {
                    const MUL_K: f64 = 2.;
                    let size = [3, 100];
                    let load_and_clc = Arc::new(LoadAndClcStage::new(
                        LoadAndClcStageId("ipc_load_and_clc_dump_to_cpu_id"),
                        MUL_K,
                        vec![FailUntil::DumpedToCpu],
                        size.into(),
                        ClcOOMRunner::new(
                            OOMRunnerConfig::DumpedToCpu {},
                            Box::new(move |id| SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(id, 1, &size, (Float, Device::Cpu), BuildAs::Append).expect("single dim output builder").copy_partial_outputs_to_device(0)),
                        ),
                        Float,
                        Device::Cuda(0).into(),
                        false,
                    ).await);
                    run_mul_task_invocations(
                        Arc::clone(&load_and_clc),
                        &["resource_1", "resource_2", "resource_3", "resource_4", "resource_5"],
                        tho_device_idx,
                        0,
                        |_device_idx, _cache_area, name| RunCtxImpl::new(1, name),
                        move |res, out| mul_check(MUL_K, size.into(), res, out, 0.1, 0.1),
                        false,
                    ).await?;
                    assert!(GlobalCtx::global().is_all_on_cpu(tho_device_idx, false), "DumpToCpu OOM recovery should mark the IPC test device as all-on-CPU");
                    Ok::<(), ThoError>(())
                }.await
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_clc_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }

    #[test]
    fn test_ab_loader_and_classifier_cuda_worker_cpu_ipc() -> Result<(), ThoError> {
        log::set_max_level(LevelFilter::Info);
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                run_model_test_ipc_cuda_worker("test_ab_loader_and_classifier_ipc_cuda_worker_cpu", || async {
                    run_model_test_pipeline(
                        1,
                        0,
                        vec![("file_0", 1.0), ("file_1", 0.0)],
                        ModelTestId("ipc_files_0_1"),
                        ModelTestId("ipc_lin_cls"),
                        2,
                        200,
                        vec![FailUntil::Never],
                        0.8,
                    ).await
                }).await
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_model_test_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }

    #[test]
    fn test_ab_loader_and_classifier_cuda_worker_cpu_many_files_ipc() -> Result<(), ThoError> {
        log::set_max_level(LevelFilter::Info);
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                run_model_test_ipc_cuda_worker("test_ab_loader_and_classifier_ipc_cuda_worker_cpu_many_files", || async {
                    run_model_test_pipeline(
                        1,
                        0,
                        vec![
                            ("file_0", 1.0),
                            ("file_1", 0.0),
                            ("file_2", 1.0),
                            ("file_3", 0.0),
                        ],
                        ModelTestId("ipc_files_0_1_2_3"),
                        ModelTestId("ipc_lin_cls_many"),
                        4,
                        100,
                        vec![FailUntil::Never],
                        0.8,
                    ).await
                }).await
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_model_test_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }

    #[test]
    fn heavy_test_real_oom_dump_and_split_ipc() -> Result<(), ThoError> {
        // log::set_max_level(LevelFilter::Debug);
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                do_real_oom_dump_and_split().await
            },
            async |sock_path| {
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    sync_clc_endpoint_inits(),
                    ThoDevice::from(Cuda(0)),
                ).await
            },
        )
    }

    mod composite_input {
        use crate::ipc::cuda_wokrer::{CudaWorker, InitFn};
        use crate::runner::global_ctx::GlobalCtx;
        use crate::tho_device::ThoDevice;
        use crate::torch_runner::tests::composite_input::ab_clc::{ABClcStage, ProxyABClcSyncStage, SYNC_AB_CLC_ENDPOINT, sync_ab_clc_endpoint_init};
        use crate::torch_runner::tests::composite_input::{ab_clc_sync_with_selector, run_ab_clc_with_stage};
        use crate::ws::mock_ws_simp;
        use log::debug;
        use parking_lot::RwLock;
        use std::collections::HashMap;
        use std::sync::Arc;
        use tch::Device::Cuda;
        use thoosa_shared::ThoError;

        fn sync_ab_clc_endpoint_inits() -> HashMap<String, InitFn> {
            HashMap::from([
                (
                    SYNC_AB_CLC_ENDPOINT.to_string(),
                    Box::new(sync_ab_clc_endpoint_init) as InitFn,
                )
            ])
        }

        //IPC version of crates/thoosa_common/src/torch_runner/tests/composite_input/mod.rs:27
        #[test]
        fn test_ipc_ab_clc() -> Result<(), ThoError> {
            // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
            super::run_ipc_test(
                module_path!(),
                async {
                    let sync_stage = Arc::new(ProxyABClcSyncStage::new(1).await?);
                    let sync_stage_calls = Arc::clone(&sync_stage);
                    let ab_stage = Arc::new(ABClcStage::new_with_sync(sync_stage));

                    run_ab_clc_with_stage(ab_stage, move || sync_stage_calls.num_calls()).await
                },
                async |sock_path| {
                    let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                    CudaWorker::run_worker(
                        sock_path,
                        &ws,
                        sync_ab_clc_endpoint_inits(),
                        ThoDevice::from(Cuda(0)),
                    ).await
                },
            )
        }

        #[test]
        fn test_ipc_ab_clc_sync_with_selector() -> Result<(), ThoError> {
            // Use this test's exact name so the harness starts its worker branch, not the sibling test.
            super::run_ipc_test(
                module_path!(),
                async {
                    let sync_stage = Arc::new(ProxyABClcSyncStage::new(1).await?);
                    let sync_stage_calls = Arc::clone(&sync_stage);
                    let ab_stage = Arc::new(ABClcStage::new_with_sync(sync_stage));

                    ab_clc_sync_with_selector(ab_stage).await
                },
                async |sock_path| {
                    let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                    CudaWorker::run_worker(
                        sock_path,
                        &ws,
                        sync_ab_clc_endpoint_inits(),
                        ThoDevice::from(Cuda(0)),
                    ).await
                },
            )
        }

    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn heavy_concurrent_copy_libtorch() ->Result<(), ThoError>{
        heavy_test_concurrent_copy(
            "heavy_concurrent_copy_direct",
            Arc::new(move |n|{
                async move {

                    let t_cpu = Tensor::rand(&[n, DIM_1], FLOAT_CPU);
                    let t_cuda = t_cpu.to_device(Cuda(0));
                    let t_cpu = t_cuda.to_device(Cpu);

                    Ok::<(), ThoError>(())
                }.boxed()
            })
        ).await?;
        info!("\n{}", format_perf_map(&PERF));

        Ok(())
    }



    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn heavy_concurrent_copy_local() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        // log::set_max_level(log::LevelFilter::Debug   );
        let global_ctx = GlobalCtx::init_devices(2)?;
        let result = heavy_test_concurrent_copy("heavy_concurrent_copy_local", Arc::new(|n|{
            async move {
                let t_cpu = ThoTensor::new((), Tensor::rand(&[n, DIM_1], FLOAT_CPU))?;
                let t_cuda = GlobalCtx::global().cuda_actor(1).copy(t_cpu, 1.into()).await?;
                let t_cpu = GlobalCtx::global().cuda_actor(1).copy(t_cuda, 0.into()).await?;

                Ok::<(), ThoError>(())
            }.boxed()
        })).await;
        global_ctx.shutdown().await;

        info!("\n{}", format_perf_map(&PERF));
        result
    }


    #[test]
    fn heavy_concurrent_copy_ipc() -> Result<(), ThoError> {

        run_ipc_test(
            module_path!(),
            async {
                heavy_test_concurrent_copy(
                    "heavy_concurrent_copy_ipc",

                   Arc::new( |n| {
                       async move {
                           let t_cpu = ThoTensor::new((), Tensor::rand(&[n, DIM_1], FLOAT_CPU))?;
                           let t_cuda = GlobalCtx::global().cuda_actor(1).copy(t_cpu, 1.into()).await?;
                           let t_cpu = GlobalCtx::global().cuda_actor(1).copy(t_cuda.shallow_clone(), 0.into()).await?;
                           drop(t_cuda);
                           // GlobalCtx::global().cuda_actor(1).drop_tho_tensor(t_cuda.ref_key().ok_or_else(||ThoError::new(format!("Not a ref tensor: {t:?}")))?).await?;

                           Ok::<(), ThoError>(())
                       }.boxed()
                   })
                ).await
            },
            async |sock_path| {
                // set_max_level(Debug);
                let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
                CudaWorker::run_worker(
                    sock_path,
                    &ws,
                    HashMap::new(),
                    ThoDevice::from(Device::Cuda(0)),
                ).await
            },
        )
    }

    const DIM_1:i64 = 10000;

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="heavy_test_concurrent_copy"))]
    async fn heavy_test_concurrent_copy<F>(descr:&str, test_f:Arc<F>)->Result<(), ThoError>
    where
        F:Fn(i64)->BoxFuture<'static, Result<(), ThoError>> + Send + Sync + 'static
    {
        let (free_mem, total_mem) = cuda_mem();
        info!("heavy_test_concurrent_copy: free_mem={free_mem}");

        let elts = free_mem / Float.elt_size_in_bytes() / DIM_1 as usize;

        for dim_0 in  [10, 100, 1000, 10_000, 100_000, 300_000] {
            let num_tensors = elts / dim_0 as usize;
            info!("heavy_test_concurrent_copy: num_tensors={num_tensors}");
            let t = Instant::now();
            let test_f = Arc::clone(&test_f);

            let mut set = JoinSet::new();
            let mut iter = 0..num_tensors;
            let mut out = Vec::new();

            // Fill initial batch.
            for _ in 0..16 {
                let Some(item) = iter.next() else {
                    break;
                };

                set.spawn(test_f(dim_0));
            }

            while let Some(result) = set.join_next().await {
                let value = result
                    .expect("task panicked")?;

                out.push(value);

                // One slot became free: spawn one more.
                if let Some(item) = iter.next() {
                    set.spawn(test_f(dim_0));
                }
            }

            info!("n={dim_0} done in {} mls", t.elapsed().as_millis());
        }



        Ok(())
    }




}
