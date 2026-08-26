use crate::ipc::cuda_process::IPC_SOCK_ENV;
use crate::ipc::cuda_wokrer::CudaWorker;
use crate::ipc::json_ser::SyncStageIpcWrapper;
use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelStep};
use crate::runner::common::cachable::{BorrowRef, ConsumeData, DevicesWrap};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::global_ctx::mock_global_ctx::{GlobalCtxMockGuard, MockGlobalCtxBuilder};
use crate::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::runner::test::FailUntil::NaxParallel;
use crate::runner::test::{ClcOOMRunner, FailUntil, OOMRunnerConfig, run_mul_task_invocations};
// use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, FailUntil, OOMRunnerConfig};
// use crate::runner::test::FailUntil::NaxParallel;
use crate::runner::test_utils::parse_numeric_part;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::{ThoTensor, ThoTensorInner};
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
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
use tokio::task;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::test_utils::init_test_logging;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::ConsumeResult;
use thoosa_shared::traits::{Adjustable, AsyncStage, OutId, RunCtx, Stage, SyncStage, WithId};
use thoosa_shared::{mono_key, panic_to_err_async, ThoError};
use tokio::task::{coop, JoinSet};
use tracing::field::debug;
// use crate::torch_runner::tests::load_and_clc::loader::LoadedTensor;
// use crate::torch_runner::tests::load_and_clc::{mul_check, sync_clc_endpoint_init, sync_test_tensor_loader_endpoint_init, LoadAndClcStage, LoadAndClcStageId, SYNC_CLC_ENDPOINT, SYNC_TEST_TENSOR_LOADER_ENDPOINT};
// use crate::torch_runner::tests::load_and_clc::clc::{ClcStage, ClcStageOutput};
// use crate::torch_runner::tests::model_test::{run_model_test_pipeline, sync_model_test_endpoint_inits};
// use crate::torch_runner::tests::model_test::common::ModelTestId;
use crate::utils::lin_for_size;
use thoosa_named_derive::Named;
use thoosa_shared::perf_logging::{PERF, format_perf_map};
use thoosa_shims::context::ThoCudaContext;

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
    FW:AsyncFn(OsString)->Result<(), ThoError>,
{
    if let Some(sock_path) = std::env::var_os(IPC_SOCK_ENV) {
        init_test_logging();

        debug!("run_ipc_test2: sock_path={sock_path:?}");


        let tho_cuda_context = ThoCudaContext::new(0)?;

        let worker_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(16)
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

        worker_rt.block_on(async {
            let r = panic_to_err_async!(worker_f(sock_path));

            info!("worker perf:\n{}", format_perf_map(&PERF));
            info!("worker done: r={r:?}");

            // Preserve the original tests' behavior by propagating worker failures through the harness.
            r
        })?;

    } else {
        let main_rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(16)
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
            Ok::<_, ThoError>(())
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
    use super::run_ipc_test;
    use crate::ipc::cuda_wokrer::{CudaWorker, InitFn};
    use crate::ipc::json_ser::SyncStageIpcWrapper;
    use crate::ipc::worker_msg::WorkerInput;
    use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelStep};
    use crate::runner::common::cachable::{BorrowRef, ConsumeData, DevicesWrap};
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::{GlobalCtxMockGuard, MockGlobalCtxBuilder};
    use crate::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
    use crate::runner::run_ctx_2::RunCtxImpl;
    use crate::runner::test::FailUntil::NaxParallel;
    use crate::runner::test::{ClcOOMRunner, FailUntil, OOMRunnerConfig, run_mul_task_invocations};
    // use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, FailUntil, OOMRunnerConfig};
    // use crate::runner::test::FailUntil::NaxParallel;
    use crate::runner::test_utils::parse_numeric_part;
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::{ThoTensor, ThoTensorInner};
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    use futures_util::future::join_all;
    use log::{LevelFilter, debug, info};
    use parking_lot::RwLock;
    use serde::de::Unexpected::Option;
    use std::collections::HashMap;
    use std::ffi::OsString;
    use std::sync::Arc;
    use std::sync::atomic::Ordering::Acquire;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Instant;
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use tch::{Device, Tensor};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::test_utils::init_test_logging;
    use thoosa_shared::tracked_await::track_await;
    use thoosa_shared::traits::ConsumeResult;
    use thoosa_shared::traits::{Adjustable, AsyncStage, OutId, RunCtx, Stage, SyncStage, WithId};
    use thoosa_shared::{ThoError, mono_key, os_thread_id};
    use tokio::task::JoinSet;

    use crate::torch_runner::tests::load_and_clc::clc::{ClcStage, ClcStageOutput};
    use crate::torch_runner::tests::load_and_clc::loader::{LoadedTensor, SyncTestTensorLoader};
    use crate::torch_runner::tests::load_and_clc::{LoadAndClcStage, LoadAndClcStageId, SYNC_CLC_ENDPOINT, SYNC_TEST_TENSOR_LOADER_ENDPOINT, mul_check, mul_check_with_exp_f, sync_clc_endpoint_init, sync_test_tensor_loader_endpoint_init};
    use crate::torch_runner::tests::model_test::common::ModelTestId;
    use crate::torch_runner::tests::model_test::{run_model_test_pipeline, sync_model_test_endpoint_inits};
   
    use crate::utils::lin_for_size;
    use crate::ws::mock_ws_simp;
    use thoosa_named_derive::Named;
    use thoosa_shared::perf_logging::{PERF, format_perf_map};
    use thoosa_shims::context::ThoCudaContext;

    fn check_sum_result(mul_k: f64, size: [i64; 2], resource: &str, out: &ThoTensor<ClcStageOutput, SynKey>) {
        let self_k = parse_numeric_part(resource).unwrap_or(1.0);
        let expected = lin_for_size(self_k * mul_k, size, out.t_ref().kind(), &out.t_ref().device())
            .sum_dim_intlist(1, false, out.t_ref().kind())
            .view((-1, 1));
        let ok = expected.allclose(out.t_ref(), 0.1, 0.1, false);
        debug!("check_sum_result: resource={resource} expected=\n{expected} actual=\n{}", out.t_ref());
        assert!(ok);
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
            if input.t_ref().f_double_value(&[0, 0])? < 0.0 {
                return Err(ThoError::new("consume test failure"));
            }
            ThoTensor::new(self.out_id(input.id()), input.t_ref().f_mul_scalar(2.0)?)
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
                0,
                tho_device_idx,
                Box::new(()),
                Arc::new(|input: &ThoTensor<LoadedTensor, SynKey>| Ok(Box::new(input.clone_ref()?) as Box<dyn WorkerInput>)),
            ).await?),
            init: (),
        })
    }

    fn sync_consume_test_endpoint_init(_init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::ws::Workspace>>>, _all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>) -> futures_util::future::BoxFuture<'_, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
        use futures_util::FutureExt;
        async move {
            // This stage has no initializer fields, but still uses the typed install message.
            Ok(Box::new(crate::ipc::wraps::WorkerWrapper::<crate::ipc::codec::JsonIpcCodec, _, _, _>::new(Arc::new(SyncConsumeTestStage::new()), std::marker::PhantomData)) as Box<dyn SyncStageIpcWrapper>)
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

        fn consume<R: Send, F, P>(self, _to_device_idx: usize, ctx: P, _trace_ctx: std::option::Option<&'static str>, do_run: F) -> impl std::future::Future<Output=ConsumeResult<Self::O, R>> + Send
        where
            F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
            P: RunCtx,
        {
            async move { do_run(self.input, ctx) }
        }

        fn burp(input: Self::O) -> impl std::future::Future<Output=Self> + Send {
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
    fn test_consume_stage_releases_inputs_only_on_success_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {

                let cuda_actor = GlobalCtx::global().cuda_actor(1);
                let stage = Arc::new(new_sync_consume_test_stage(1).await?);
                let oom_handler = OOMHandler::<NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()>::new(vec![], None);

                let input_tensor = Tensor::f_from_slice(&[1.0f32, 2.0])?.f_view([1, 2])?;
                let input_ref = cuda_actor.copy(input_tensor.into(), ThoDevice::from(1)).await?;
                let input_key = input_ref.ref_key().ok_or_else(|| ThoError::new("consume test expected worker input ref"))?;
                let input = ThoTensor::new_ref(mono_key!("consume-success", "input").into(), input_key, input_ref.size(), input_ref.kind(), None, 1);
                let output = oom_handler.consume(&stage, ConsumeTestInput::new(input), RunCtxImpl::new(1, "consume-success"), NoContexts).await?;
                let output_key = output.ref_key().ok_or_else(|| ThoError::new("consume test expected worker output ref"))?;
                let output_cpu = cuda_actor.copy(ThoTensor::new_ref((), output_key, output.size(), output.kind(), None, 1), ThoDevice::from(0)).await?;
                assert!(output_cpu.t_ref().f_allclose(&Tensor::f_from_slice(&[2.0f32, 4.0])?.f_view([1, 2])?, 0.0, 0.0, false)?);

                let consumed_input = cuda_actor.copy(ThoTensor::new_ref((), input_key, vec![1, 2], Float, None, 1), ThoDevice::from(0)).await;
                assert!(consumed_input.is_err(), "successful consume must remove the worker input tensor");

                let failed_tensor = Tensor::f_from_slice(&[-1.0f32, 2.0])?.f_view([1, 2])?;
                let failed_ref = cuda_actor.copy(failed_tensor.into(), ThoDevice::from(1)).await?;
                let failed_key = failed_ref.ref_key().ok_or_else(|| ThoError::new("consume test expected failed worker input ref"))?;
                let failed_input = ThoTensor::new_ref(mono_key!("consume-failure", "input").into(), failed_key, failed_ref.size(), failed_ref.kind(), None, 1);
                let failed_result = oom_handler.consume(&stage, ConsumeTestInput::new(failed_input), RunCtxImpl::new(1, "consume-failure"), NoContexts).await;
                assert!(failed_result.is_err());
                let preserved_input = cuda_actor.copy(ThoTensor::new_ref((), failed_key, vec![1, 2], Float, None, 1), ThoDevice::from(0)).await?;
                assert!(preserved_input.t_ref().f_allclose(&Tensor::f_from_slice(&[-1.0f32, 2.0])?.f_view([1, 2])?, 0.0, 0.0, false)?);

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

    impl From<ThoTensor> for ThoTensor<LoadedTensor, SynKey> {
        fn from(t: ThoTensor) -> Self {
            let ThoTensor { id, inner, .. } = t;
            match inner {
                ThoTensorInner::Val { .. } => {
                    todo!()
                }
                ThoTensorInner::Ref { key, size, kind, .. } => {
                    ThoTensor::new_ref(mono_key!(0, "todo").into(), key, vec![], Float, None, 1)
                }
            }
        }
    }

    async fn run_load_and_clc_ipc_cuda_worker(test_name: &str, resources: &[&'static str], use_matmul: bool) -> Result<(), ThoError> {
        debug!("run_load_and_clc_ipc_cuda_worker: test_name={test_name} resources={resources:?}");

        let tho_device_idx = 1;
        let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
        let size = [3, 10];
        const MUL_K: f64 = 2.;
        let clc_dim = if use_matmul { size[0] } else { size[1] };
        // All resource tasks use the same CLC operation, so share one proxy and its typed worker endpoint identity.
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
                let cpu_tensor = lin_for_size(self_k, size, Float, &Device::Cpu);

                let worker_ref: ThoTensor<LoadedTensor, SynKey> = track_await(cuda_actor.copy(cpu_tensor.into(), tho_device_idx.into())).await?.into();
                debug!("run_load_and_clc_ipc_cuda_worker: worker_ref={worker_ref:?}");

                let loaded = Arc::new(DevicesWrap::wrap(worker_ref));
                //further logging is duplicated somehow
                counter.fetch_add(1, Acquire);
                debug!("run_load_and_clc_ipc_cuda_worker: loaded={loaded:?} counter={}", counter.load(Acquire));

                let mut ctx = RunCtxImpl::new(tho_device_idx, "ipc_load_and_clc");
                let output_ref = Arc::clone(&clc_task).run(Arc::clone(&loaded), ctx.branch(false)).await?;
                let output_ref = BorrowRef::borrow_to(&output_ref, ctx.tho_device_idx()).await?;
                let out_key = output_ref.ref_key().ok_or_else(|| ThoError::new(format!("ProxySyncClcTask expected output ref from worker, got {output_ref:?}")))?;


                debug!("run_load_and_clc_ipc_cuda_worker: out_key={out_key:?} output_ref.id={:?}", output_ref.id());

                let tensor = GlobalCtx::global().cuda_actor(1/*TODO take ctx*/)
                    .copy(ThoTensor::new_ref((), out_key, output_ref.size(), output_ref.kind(), None, 1), 0.into()).await?;
                let cpu_out: ThoTensor<ClcStageOutput, SynKey> = ThoTensor::new(output_ref.id().clone(), tensor.t())?;


                assert_eq!(cpu_out.id(), &clc_task.out_id(loaded.id()));
                if use_matmul {
                    check_sum_result(MUL_K, size, resource, &cpu_out);
                } else {
                    mul_check(MUL_K, size.into(), resource, &cpu_out, 0.1, 0.1);
                }


                Ok::<(), ThoError>(())
            });
        }


        let result = async {
            while let Some(joined) = tasks.join_next().await {
                joined.map_err(|e| ThoError::new(format!("IPC load/mul task join failed: {e:?}")))??;
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


    #[test]
    fn test_load_and_clc_cuda_worker_no_oom_tensor_ipc() -> Result<(), ThoError> {
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
    fn test_load_and_clc_cuda_worker_sum_no_oom_tensor_ipc() -> Result<(), ThoError> {
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
    fn test_load_and_clc_cuda_worker_oom_till_min_max_parallel_tensor_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
            let tho_device_idx = 1;
            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let result = async {
                const MUL_K: f64 = 2.;
                let size = [3, 10];
                let load_and_clc = Arc::new(LoadAndClcStage::new(
                    LoadAndClcStageId("ipc_load_and_clc_oom_till_min_max_parallel_id"),
                    MUL_K,
                    vec![NaxParallel(20)],
                    size.into(),
                    ClcOOMRunner::new(
                        OOMRunnerConfig::MaxParallel { max: 20, step: 10 },
                        Box::new(move |id| SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(id, 1, &size, (Float, Device::Cpu), BuildAs::Append).expect("single dim output builder").copy_partial_outputs_to_device(0)),
                    ),
                    Float,
                    Device::Cuda(0).into(),
                    false,
                ).await);
                run_mul_task_invocations(
                    Arc::clone(&load_and_clc),
                    &["resource_1", "resource_2", "resource_3"],
                    tho_device_idx,
                    0,
                    |_device_idx, _cache_area, name| RunCtxImpl::new(1, name),
                    move |res, out| mul_check(MUL_K, size.into(), res, out, 0.1, 0.1),
                    false,
                ).await?;
                Ok::<(), ThoError>(())
            }.await;

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
    fn test_load_and_clc_cuda_worker_oom_compete_for_max_parallel_tensor_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
            let tho_device_idx = 1;
            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let result = async {
                const MUL_K: f64 = 2.;
                let size = [3, 10];
                let load_and_clc = Arc::new(LoadAndClcStage::new(
                    LoadAndClcStageId("ipc_load_and_clc_oom_compete_for_max_parallel_id"),
                    MUL_K,
                    vec![NaxParallel(MAX_PARALLEL)],
                    size.into(),
                    ClcOOMRunner::new(
                        OOMRunnerConfig::MaxParallel { max: MAX_PARALLEL, step: *MaxParallelStep::TEN },
                        Box::new(move |id| SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(id, 1, &size, (Float, Device::Cpu), BuildAs::Append).expect("single dim output builder").copy_partial_outputs_to_device(0)),
                    ),
                    Float,
                    Device::Cuda(0).into(),
                    false,
                ).await);
                run_mul_task_invocations(
                    Arc::clone(&load_and_clc),
                    &["resource_1", "resource_2", "resource_3"],
                    tho_device_idx,
                    0,
                    |_device_idx, _cache_area, name| RunCtxImpl::new(1, name),
                    move |res, out| mul_check(MUL_K, size.into(), res, out, 0.1, 0.1),
                    false,
                ).await?;
                Ok::<(), ThoError>(())
            }.await;

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
    fn test_simp_copy() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
            let tho_device_idx = 1;
            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);

            let smoke = lin_for_size(1.0, [3000, 30000], Float, &Device::Cpu);
            let expected_smoke = smoke.copy();
            let worker_ref = track_await(cuda_actor.copy(smoke.into(), tho_device_idx.into())).await?;
            assert!(worker_ref.ref_key().is_some(), "expected IPC H2D copy to return worker tensor ref, got {worker_ref:?}");

            let cpu_copy = track_await(cuda_actor.copy(worker_ref, 0.into())).await?;
            assert!(expected_smoke.allclose(cpu_copy.t_ref(), 0.1, 0.1, false), "IPC worker smoke copy changed tensor values");

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
    fn test_load_and_clc_cuda_worker_oom_splits_tensor_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
            let tho_device_idx = 1;
            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let result = async {
                const MUL_K: f64 = 2.;
                let size = [3, 40];
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
                            BuildAs::Sum
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
                            lin_for_size(k, size, t.kind(), &t.device()) //same as in SyncTestTensorLoader::do_run
                                .sum_dim_intlist(1, false, t.kind())
                                .view((-1, 1))
                        })
                    },
                    false,
                ).await?;
                Ok::<(), ThoError>(())
            }.await;

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
    fn test_load_and_clc_cuda_worker_oom_max_parallel_and_splits_tensor_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                let resources = (0..3)
                    .map(|i| &*format!("resource_{i}").leak())
                    .collect::<Vec<_>>();

                let tho_device_idx = 1;
                let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
                let result = async {
                    const MUL_K: f64 = 2.;
                    let size = [3, 100];
                    let load_and_clc = Arc::new(LoadAndClcStage::new(
                        LoadAndClcStageId("ipc_load_and_clc_oom_max_parallel_and_splits_id"),
                        MUL_K,
                        vec![NaxParallel(MAX_PARALLEL), FailUntil::MemLimit(12)],
                        size.into(),
                        ClcOOMRunner::new(
                            OOMRunnerConfig::MaxParallelAndSplits { max: MAX_PARALLEL, step: *MaxParallelStep::TEN },
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
                                lin_for_size(k, size, t.kind(), &t.device()) //same as in SyncTestTensorLoader::do_run
                                    .sum_dim_intlist(1, false, t.kind())
                                    .view((-1, 1))
                            });
                        },
                        false,
                    ).await?;
                    Ok::<(), ThoError>(())
                }.await;

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
    fn test_load_and_clc_cuda_worker_oom_dump_to_cpu_ipc() -> Result<(), ThoError> {
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
            let tho_device_idx = 1;
            let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
            let result = async {
                const MUL_K: f64 = 2.;
                let size = [3, 100];
                let load_and_clc = Arc::new(LoadAndClcStage::new(
                    LoadAndClcStageId("ipc_load_and_clc_dump_to_cpu_id"),
                    MUL_K,
                    vec![NaxParallel(MAX_PARALLEL), FailUntil::DumpedToCpu],
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
            }.await;

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
    fn test_classifier_cuda_worker_oom_max_parallel_escalates_ipc() -> Result<(), ThoError> {
        log::set_max_level(LevelFilter::Debug);
        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(
            module_path!(),
            async {
                let files: Vec<(&'static str, f64)> = (0..16)
                    .map(|i| {
                        (
                            Box::leak(format!("file_{}", i).into_boxed_str()) as &'static str,
                            if i % 2 == 0 { 1.0 } else { 0.0 },
                        )
                    })
                    .collect();

                run_model_test_ipc_cuda_worker("test_classifier_ipc_cuda_worker_oom_max_parallel_escalates", || async move {
                    run_model_test_pipeline(
                        1,
                        0,
                        files,
                        ModelTestId("ipc_files_oom_permits"),
                        ModelTestId("ipc_lin_cls_oom"),
                        16,
                        50,
                        vec![NaxParallel(MAX_PARALLEL), FailUntil::MemLimit(128)],
                        0.5,
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
    async fn heavy_concurrent_copy_local() -> Result<(), ThoError> {//heavy_
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        // log::set_max_level(log::LevelFilter::Debug   );
        let global_ctx = GlobalCtx::init_devices(2);
        let result = heavy_test_concurrent_copy("heavy_test_run_all").await;
        global_ctx.shutdown().await;

        info!("\n{}", format_perf_map(&PERF));
        result
    }


    #[test]
    fn heavy_concurrent_copy_ipc() -> Result<(), ThoError> {//heavy_

        run_ipc_test(
            module_path!(),
            async {
                heavy_test_concurrent_copy("heavy_test_run_all_ipc").await
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

    async fn heavy_test_concurrent_copy(descr:&str)->Result<(), ThoError>{


        for n in [10, 100, 1000, 10000] {
            let t = Instant::now();
            let x = (0..100).map(|i| {
                async move {
                    let t_cpu_0 = ThoTensor::new((), Tensor::rand(&[n, 10000], FLOAT_CPU))?;
                    let t_cpu_0_t = t_cpu_0.as_ref().as_ref().shallow_clone();
                    let t_cuda = GlobalCtx::global().cuda_actor(1).copy(t_cpu_0, 1.into()).await?;
                    let t_cpu_1 = GlobalCtx::global().cuda_actor(1).copy(t_cuda, 0.into()).await?;
                    debug!("i={i} os_thread_id={}",os_thread_id());

                    assert!(t_cpu_0_t.allclose(t_cpu_1.as_ref(), 0.01, 0.01, false));

                    Ok::<_, ThoError>(())
                }
            }).collect::<JoinSet<_>>().join_all().await;

            info!("n={n} done in {} mls", t.elapsed().as_millis());
        }

        Ok(())
    }

}
