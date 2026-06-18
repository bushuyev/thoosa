use std::collections::HashMap;
use std::sync::Arc;
use log::debug;
use tch::Device;
use tch::Kind::Float;
use tokio::task::JoinSet;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::traits::WithId;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::{mono_key, ThoError};
use thoosa_with_data::with_data;
use crate::ipc::cuda_process::IPC_SOCK_ENV;
use crate::ipc::cuda_wokrer::{CudaWorker, SyncStageJsonWrapper};
use crate::runner::adjusters::max_parallel::MAX_PARALLEL;
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::common::{OutId, RunCtx};
use crate::runner::common::runnable::{AsyncStage, SyncStage};
use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, FailUntil, OOMRunnerConfig};
use crate::runner::test::FailUntil::NaxParallel;
use crate::runner::test_utils::parse_numeric_part;
use crate::tho_tensor::{ThoTensor, ThoTensorInner};
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use crate::torch_runner::tests::{ClcTask, ClcTaskOutput, LoadAndClcTask, LoadAndClcTaskId, LoadedTensor, mul_check, SyncClcTask, SyncClcTaskJsonWrapper, SyncClcTaskInit};
use crate::utils::lin_for_size;

macro_rules! pure_module_path {
    () => {{
        let p = module_path!();

        match p.split_once("::") {
            Some((_crate_name, rest)) => rest,
            None => "",
        }
    }};
}

#[ctor::ctor]
fn init() {
    thoosa_shared::test_utils::init_test_logging();
}

fn test_worker_args(test_name: &str) -> Vec<&'static str> {
    let module = pure_module_path!();
    let test_name = if module.is_empty() {
        test_name.to_string()
    } else {
        format!("{module}::{test_name}")
    };

    vec![test_name.leak(), "--exact", "--nocapture"]
}

async fn init_ipc_test(test_name: &str) -> Result<impl Drop, ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }

    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices_ipc(2, test_worker_args(test_name)).await?;
    let ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

    Ok(ctx)
}

fn check_sum_result(mul_k: f64, size: [i64; 2], resource: &str, out: &ThoTensor<crate::torch_runner::tests::ClcTaskOutput, SynKey>) {
    let self_k = parse_numeric_part(resource).unwrap_or(1.0);
    let expected = lin_for_size(self_k * mul_k, size, out.t_ref().kind(), &out.t_ref().device())
        .sum_dim_intlist(1, false, out.t_ref().kind())
        .view((-1, 1));
    let ok = expected.allclose(out.t_ref(), 0.1, 0.1, false);
    debug!("check_sum_result: resource={resource} expected=\n{expected} actual=\n{}", out.t_ref());
    assert!(ok);
}

impl From<ThoTensor> for ThoTensor<LoadedTensor, SynKey> {
    fn from(t: ThoTensor) -> Self {
        let ThoTensor { id, inner, .. } = t;
        match inner {
            ThoTensorInner::Val { .. } => {
                todo!()
            }
            ThoTensorInner::Ref { key } => {
                ThoTensor::new_ref(mono_key!(0, "todo").into(), key)
            }
        }
    }
}

async fn run_load_and_mul_ipc_cuda_worker(test_name: &str, resources: &[&'static str], use_matmul: bool) -> Result<(), ThoError> {
    let _ctx = init_ipc_test(test_name).await?;

    let tho_device_idx = 1;
    let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);
    let size = [3, 10];
    const MUL_K: f64 = 2.;
    let clc_dim = if use_matmul { size[0] } else { size[1] };

    // cuda_actor.install_endpoint(serde_json::to_string(&SyncClcTaskInit::new(mono_key!(0, "sync_task_0").into(), MUL_K, clc_dim, vec![], false))?).await?;

    let mut tasks = JoinSet::new();
    for resource in resources {
        let resource = *resource;
        let cuda_actor = cuda_actor;
        tasks.spawn(async move {
            let input_id = SynKey::new([mono_key!(resource, "resource key")]);
            let self_k = parse_numeric_part(resource).unwrap_or(1.0);
            let cpu_tensor = lin_for_size(self_k, size, Float, &Device::Cpu);

            let worker_ref: ThoTensor<LoadedTensor, SynKey> = track_await(cuda_actor.copy(cpu_tensor.into(), tho_device_idx.into())).await?.into();

            assert!(worker_ref.ref_key().is_some(), "expected IPC H2D copy to return worker tensor ref, got {worker_ref:?}");

            // let cpu_copy = track_await(cuda_actor.copy(worker_ref, 0.into())).await?;
            let loaded = Arc::new(DevicesWrap::wrap(worker_ref));

            debug!("run_load_and_mul_ipc_cuda_worker: loaded={loaded:?}");


            let clc_task = Arc::new(ClcTask::new_proxy(SynKey::new([mono_key!(0, "sync_task_0")]), MUL_K, vec![], clc_dim, Float, Device::Cpu, use_matmul).await?);
            let mut ctx = RunCtxImpl::new(tho_device_idx, "ipc_load_and_mul");
            let out = Arc::clone(&clc_task).run(Arc::clone(&loaded), ctx.branch(false)).await?;

            with_data!(ctx, out; |_ctx, out:&ThoTensor::<ClcTaskOutput, SynKey>|{
                assert_eq!(out.id(), &clc_task.out_id(loaded.id()));
                if use_matmul {
                    check_sum_result(MUL_K, size, resource, out);
                } else {
                    mul_check(MUL_K, size.into(), resource, out, 0.1, 0.1);
                }
                Ok(())
            })?;


            Ok::<(), ThoError>(())
        });
    }

    let result = async {
        while let Some(joined) = tasks.join_next().await {
            joined.map_err(|e| ThoError::new(format!("IPC load/mul task join failed: {e:?}")))??;
        }

        Ok::<(), ThoError>(())
    }.await;
    let shutdown_result = cuda_actor.shutdown().await;

    match (result, shutdown_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
    }
}

async fn run_load_and_mul_ipc_cuda_worker_oom_dump_to_cpu(test_name: &str) -> Result<(), ThoError> {
    let _ctx = init_ipc_test(test_name).await?;

    let tho_device_idx = 1;
    let cuda_actor = GlobalCtx::global().cuda_actor(tho_device_idx);

    let result = async {
        let smoke = lin_for_size(1.0, [3, 10], Float, &Device::Cpu);
        let worker_ref = track_await(cuda_actor.copy(smoke.shallow_clone().into(), tho_device_idx.into())).await?;
        assert!(worker_ref.ref_key().is_some(), "expected IPC H2D copy to return worker tensor ref, got {worker_ref:?}");
        let cpu_copy = track_await(cuda_actor.copy(worker_ref, 0.into())).await?;
        assert!(smoke.allclose(cpu_copy.t_ref(), 0.1, 0.1, false), "IPC worker smoke copy changed tensor values");

        const MUL_K: f64 = 2.;
        let size = [3, 100];
        let load_and_mul = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("ipc_load_and_mul_dump_to_cpu_id"),
            MUL_K,
            vec![NaxParallel(MAX_PARALLEL), FailUntil::DumpedToCpu],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::DumpedToCpu {},
                Box::new(move |id| {
                    SingleDimTensorOutputBuilder::<SynKey, ClcTaskOutput>::new(
                        id,
                        1,
                        &size,
                        (Float, Device::Cuda(0)),
                        BuildAs::Append,
                    ).expect("single dim output builder")
                }),
            ),
            Float,
            Device::Cuda(0),
            false,
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mul),
            &[
                "resource_1",
                "resource_2",
                "resource_3",
                "resource_4",
                "resource_5",
            ],
            tho_device_idx,
            |_device_idx, _cache_area, name| RunCtxImpl::new(1, name),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
        .await?;

        assert!(GlobalCtx::global().is_all_on_cpu(tho_device_idx), "DumpToCpu OOM recovery should mark the IPC test device as all-on-CPU");

        Ok::<(), ThoError>(())
    }.await;
    let shutdown_result = cuda_actor.shutdown().await;

    match (result, shutdown_result) {
        (Ok(()), Ok(())) => Ok(()),
        (Err(error), _) => Err(error),
        (Ok(()), Err(error)) => Err(error),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_load_and_mul_ipc_cuda_worker_no_oom_tensor() {
    debug!("test_load_and_mul_ipc_cuda_worker_no_oom_tensor, IPC_SOCK_ENV={:?}", std::env::var_os(IPC_SOCK_ENV));

    if std::env::var_os(IPC_SOCK_ENV).is_some() {
        CudaWorker::run_worker(HashMap::new()).await.unwrap()
    } else {
        run_load_and_mul_ipc_cuda_worker("test_load_and_mul_ipc_cuda_worker_no_oom_tensor", &["resource_1", "resource_2", "resource_3"], false).await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn test_load_and_mul_ipc_cuda_worker_oom_dump_to_cpu() {
    debug!("test_load_and_mul_ipc_cuda_worker_oom_dump_to_cpu, IPC_SOCK_ENV={:?}", std::env::var_os(IPC_SOCK_ENV));

    if std::env::var_os(IPC_SOCK_ENV).is_some() {
        CudaWorker::run_worker(HashMap::new()).await.unwrap()
    } else {
        run_load_and_mul_ipc_cuda_worker_oom_dump_to_cpu("test_load_and_mul_ipc_cuda_worker_oom_dump_to_cpu").await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn test_load_and_mul_ipc_cuda_worker_sum_no_oom_tensor() {
    debug!("test_load_and_mul_ipc_cuda_worker_sum_no_oom_tensor, IPC_SOCK_ENV={:?}", std::env::var_os(IPC_SOCK_ENV));

    if std::env::var_os(IPC_SOCK_ENV).is_some() {
        CudaWorker::run_worker(HashMap::from([
            (
                "SyncClcTask".to_string(),
                Box::new(|init_msg|{
                    Ok(Box::new(SyncClcTaskJsonWrapper::new(init_msg)) as Box<dyn SyncStageJsonWrapper>)
                }) as Box<dyn Fn(String) -> Result<Box<dyn SyncStageJsonWrapper>, ThoError>>,
            )
        ])).await.unwrap()
    } else {
        run_load_and_mul_ipc_cuda_worker("test_load_and_mul_ipc_cuda_worker_sum_no_oom_tensor", &["resource_1", "resource_2", "resource_3"], true).await.unwrap();
    }
}
