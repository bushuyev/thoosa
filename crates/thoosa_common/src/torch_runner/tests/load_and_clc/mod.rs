use crate as thoosa_common;
use crate::cuda_utils::cuda_mem;
use crate::dims::{AnyDims, Dims};
use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallel, MaxParallelStep};
use crate::runner::adjusters::splits::SplitsAdjusterContextImpl;
use crate::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, ThoDataRef};
use crate::runner::common::*;
use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use crate::runner::global_ctx::{GlobalCtx, mock_global_ctx};
use crate::runner::oom_handler::{Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use crate::runner::run_ctx::RunCtxImpl;
use crate::runner::test::{ClcOOMRunner, FailUntil, OOMRunnerConfig, run_mul_task_invocations};
use crate::runner::test_utils::parse_numeric_part;
use crate::runner::tmp_cache::TmpCached;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use crate::utils::lin_for_size;
use async_trait::async_trait;
pub(crate) use clc::sync_clc_endpoint_init;
use clc::{ClcDynSyncStage, ClcStage, ClcStageOutput, SyncClcStage, SyncClcStageInit};
use dashmap::DashMap;
use futures_util::StreamExt;
use futures_util::future::{BoxFuture, Shared};
pub(crate) use loader::sync_test_tensor_loader_endpoint_init;
use loader::{LoadedRange, LoadedTensor, LoaderInput, SyncTestTensorLoader, SyncTestTensorLoaderInit, TestTensorLoader};
use log::debug;
use num_traits::ToPrimitive;
use parking_lot::RwLock;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Add, Deref};
use std::ops::Mul;
use std::sync::Arc;
use std::time::Duration;
use tch::Device::{Cpu, Cuda};
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tch::{Device, IndexOp, Kind, Scalar, Tensor};
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Adjustable, AsyncStage, CachableIdTr, Data, DynSyncStageBox, MonoMetrics, Named, OutId, RunCtx, Select, DebugId, Stage, SyncStage, WithId};
use thoosa_shared::ThoError;
use thoosa_shared::{CopyToDevice, mono_key, debug_id};
use tokio::sync::Mutex;
use tracing_subscriber::fmt::format::FmtSpan;
use tracing_subscriber::{EnvFilter, fmt};

pub mod loader;
pub mod clc;

#[ctor::ctor]
fn init() {
    thoosa_shared::test_utils::init_test_logging();
}

// impl Adjustable for ThoTensor2<LoadedTensor, SimpleId> {}


pub(crate) const SYNC_CLC_ENDPOINT: &str = "SyncClcTask";
pub(crate) const SYNC_TEST_TENSOR_LOADER_ENDPOINT: &str = "SyncTestTensorLoader";



fn kind_to_worker_name(kind: Kind) -> &'static str {
    match kind {
        Kind::Float => "Float",
        Kind::Double => "Double",
        Kind::Half => "Half",
        Kind::BFloat16 => "BFloat16",
        Kind::Int => "Int",
        Kind::Int64 => "Int64",
        Kind::Bool => "Bool",
        _ => "Unsupported",
    }
}

fn kind_from_worker_name(kind: &str) -> Result<Kind, ThoError> {
    match kind {
        "Float" => Ok(Kind::Float),
        "Double" => Ok(Kind::Double),
        "Half" => Ok(Kind::Half),
        "BFloat16" => Ok(Kind::BFloat16),
        "Int" => Ok(Kind::Int),
        "Int64" => Ok(Kind::Int64),
        "Bool" => Ok(Kind::Bool),
        _ => Err(ThoError::new(format!("unsupported worker tensor loader kind {kind:?}"))),
    }
}

//////////////////////////////

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub(crate) struct LoadAndClcStageId(pub(crate) &'static str);

#[derive(Debug, Named)]
pub(crate) struct LoadAndClcStage {
    #[allow(dead_code)]
    id: LoadAndClcStageId,

    loader: Arc<TestTensorLoader>,
    clc_task: Arc<ClcDynSyncStage>,

    clc_oom_runner: ClcOOMRunner<ClcDynSyncStage, LoadedRange, SingleDimTensorOutputBuilder<SynKey, ClcStageOutput>>,
}

impl Stage for LoadAndClcStage {
    type In<'a> = LoaderInput;
    type Out = Arc<DevicesWrap<ThoTensor<ClcStageOutput, SynKey>>>;
    type Ctx = RunCtxImpl;
}


impl AsyncStage for LoadAndClcStage {

    async fn run(
        self: Arc<Self>,
        input: Self::In<'static>,
        mut ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        let ctx_for_loader = ctx.branch(false);
        let loaded = Arc::clone(&self.loader)
            .run_cached(input, ctx_for_loader)
            .await?;

        let loaded_id = loaded.id().clone();
        let output_id = self.clc_task.out_id(&loaded_id.clone());

        debug!("LoadAndMulTask::run: loaded_id={loaded_id:?} output_id={output_id:?}");

        let loaded_id_name = "LoadedTensorId";
        let output_id_name = "MulTaskOutputId";

        debug!("LoadAndMulTask::run: loaded_id_name={loaded_id_name:?} output_id={:?}" ,output_id);

        let set_loaded = [&loaded_id as &dyn CachableIdTr]
            .into_iter()
            .collect::<HashSet<&dyn CachableIdTr>>();
        let set_mul = [&output_id as &dyn CachableIdTr]
            .into_iter()
            .collect::<HashSet<&dyn CachableIdTr>>();

        let required_ids = HashMap::from([
            (loaded_id_name, &set_loaded),
            (output_id_name, &set_mul),
        ]);

        let size = loaded.size().clone();
        let result = self
            .clc_oom_runner
            .run(&self.clc_task, loaded, ctx.branch(false), &required_ids, size)
            .await?;

        let device_idx = ctx.tho_device_idx();

        Ok(Arc::new(DevicesWrap::wrap(result)))
    }
}

#[async_trait]
impl Cached for LoadAndClcStage {//for test only, normally should be implemented with Derive(Cached)
    async fn cache() -> &'static DashMap<
        <<Self as Stage>::Out as WithId>::ID,
        Shared<BoxFuture<'static, Result<Self::Out, ThoError>>>,
    >
    where
        Self: Sized,
    {
        <ClcStage as Cached>::cache().await
    }
}

impl OutId for LoadAndClcStage {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        self.clc_task.out_id(&self.loader.out_id(input_id))
    }
}

impl LoadAndClcStage {
    pub async fn new<K>(
        id: LoadAndClcStageId,
        k: K,
        fail_until: Vec<FailUntil>,
        dim: MonoMetrics<1, 2>,
        oom_runner: ClcOOMRunner<ClcDynSyncStage, LoadedRange, SingleDimTensorOutputBuilder<SynKey, ClcStageOutput>>,
        kind: Kind,
        device: ThoDevice,
        use_matmul: bool,
    ) -> Self
    where
        K: Into<Scalar> + ToPrimitive + Copy,
    {
        let dim_1 = if use_matmul {dim.dim_0()} else {dim.dim_1()};

        Self {
            id,
            loader: Arc::new(TestTensorLoader::new(dim, kind, 1).await),
            clc_task: Arc::new(SyncClcStage::new(
                SynKey::new([mono_key!(0, "mul_task_0")]),
                k,
                fail_until,
                dim_1,
                kind,
                device,
                use_matmul,
            ).await),
            clc_oom_runner: oom_runner,
        }
    }
}
//////////////////////////////

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_load_tensor_cuda_to_cuda() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }

    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;

    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    let loader = Arc::new(TestTensorLoader::new([10, 3].into(), Float, 1).await);

    do_test_loaded(1, 1, &loader).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_load_tensor_cpu_to_cuda() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }

    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;

    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    let loader = Arc::new(TestTensorLoader::new([10, 3].into(), Float, 1).await);

    do_test_loaded(0, 1, &loader).await
}

async fn do_test_loaded(
    loader_ctx_device: usize,
    load_to_device: usize,
    loader: &Arc<TestTensorLoader>,
) -> Result<(), ThoError> {
    let mut ctx_device_0 = RunCtxImpl::new(
        loader_ctx_device,
        "device_1_ctx",
    );

    // ctx_device_0.write().add_permit(GlobalCtx::global().acquire_many(loader_ctx_device, 1, Arc::new(simple_id!("zzz"))).await?);



    let loaded_tensor_0_0 = Arc::clone(&loader).run_cached(
        "resource_1".into(),
        ctx_device_0.branch(false),
    ).await?;

    debug!("loaded_tensor_0_0={loaded_tensor_0_0:?}");

    let tensor_data_0_0=BorrowRef::borrow_ref(&loaded_tensor_0_0, load_to_device).await?;

    debug!("tensor_data={}", (*tensor_data_0_0).t_ref());

    assert!(
        Tensor::from_slice(&[
            0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16.,
            17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29.
        ])
            .totype(Float)
            .to_device(Device::Cuda(0))
            .view((10, 3))
            .allclose(&(*tensor_data_0_0).t_ref(), 0.01, 0.01, false)
    );



    let _loaded_tensor_0_1 = Arc::clone(&loader)
        .run_cached(
            "resource_1".into(),
            ctx_device_0.branch(false),
        )
        .await?;

    let tensor_data_0_1 = BorrowRef::borrow_ref(&loaded_tensor_0_0, load_to_device).await?;

    // let tensor_data_0_1 = loaded_tensor_0_1.load_from_any_device(load_to_device, None).await?;

    assert_eq!(tensor_data_0_0.t_ref(), tensor_data_0_1.as_ref().t_ref());

    // let tensor_data_0_0 = loaded_tensor_0_0.load_from_any_device(load_to_device, None).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn test_load_and_clc_no_oom_tensor() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }
    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    let device_idx = 1;
    const MUL_K: f64 = 2.;

    let size = [3, 10];
    let load_and_mut = Arc::new(LoadAndClcStage::new(
        LoadAndClcStageId("load_and_clc_id"),
        MUL_K,
        vec![],
        size.into(),
        ClcOOMRunner::new(
            OOMRunnerConfig::MaxParallel { max: 20, step: 10 },
            Box::new(move |id| {
                SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                    id,
                    1,
                    &size,
                    (Float, Device::Cuda(0)),
                    BuildAs::Append,
                ).expect("single dim output builder")
            }),
        ),
        Float,
        Device::Cuda(0).into(),
        false,
    ).await);

    run_mul_task_invocations(
        Arc::clone(&load_and_mut),
        &["resource_1", "resource_2", "resource_3"],
        device_idx,
        device_idx,
        |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
        move |res, out| {
            mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
        },
        true,
    )
        .await?;

    Ok(())
}

pub(crate) fn mul_check(
    mul_k: f64,
    size: MonoMetrics<1, 2>,
    res: &str,
    result: &ThoTensor<ClcStageOutput, SynKey>,
    rtol: f64,
    atol: f64,
) {
    mul_check_with_exp_f(mul_k, res, result, rtol, atol, |k, t| {
        Ok(lin_for_size(k, (&size).into(), t.kind(), &t.device())?)
    })
}

pub fn mul_check_with_exp_f<F>(
    mul_k: f64,
    res: &str,
    result: &ThoTensor<ClcStageOutput, SynKey>,
    rtol: f64,
    atol: f64,
    exp_f: F,
) where
    F: Fn(f64, &Tensor) -> Result<Tensor, ThoError>,
{
    let self_k = parse_numeric_part(res).unwrap_or(1.0);

    let expected = exp_f(self_k * mul_k, &result.t_ref()).unwrap();

    debug!("mul_check: res={res}  result_t.size()={:?} \n\n",result.size());

    let ok = expected.allclose(&result.t_ref(), rtol, atol, false);
    // if !ok {
    debug!("mul_check: res={res} expected=\n{expected} result_t=\n{} \n\n",result.t_ref());
    // }
    assert!(ok);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn test_load_and_clc_oom_till_min_max_parallel_tensor() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }
    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    let device_idx = 1;
    const MUL_K: f64 = 2.;

    let size = [3, 10];
    let load_and_mut = Arc::new(LoadAndClcStage::new(
        LoadAndClcStageId("load_and_clc_id"),
        MUL_K,
        vec![FailUntil::NaxParallel(0)],
        size.into(),
        ClcOOMRunner::new(
            OOMRunnerConfig::MaxParallel { max: 20, step: 10 },
            Box::new(move |id| {
                SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                    id,
                    1,
                    &size,
                    (Float, Device::Cuda(0)),
                    BuildAs::Append,
                ).expect("single dim output builder")
            }),
        ),
        Float,
        Device::Cuda(0).into(),
        false,
    ).await);

    run_mul_task_invocations(
        Arc::clone(&load_and_mut),
        &["resource_1", "resource_2", "resource_3"],
        device_idx,
        device_idx,
        |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
        move |res, out| {
            mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
        },
        true,
    ).await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn test_load_and_clc_oom_compete_for_max_parallel_tensor() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }

    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    let device_idx = 1;

    const MUL_K: f64 = 2.;
    let size = [3, 10];
    let load_and_mut = Arc::new(LoadAndClcStage::new(
        LoadAndClcStageId("load_and_clc_id"),
        MUL_K,
        vec![FailUntil::NaxParallel(0)],
        size.into(),
        ClcOOMRunner::new(
            OOMRunnerConfig::MaxParallel {
                max: MAX_PARALLEL,
                step: *MaxParallelStep::FAST,
            },
            Box::new(move |id| {
                SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                    id,
                    1,
                    &size,
                    (Float, Device::Cuda(0)),
                    BuildAs::Append,
                ).expect("single dim output builder")
            }),
        ),
        Float,
        Device::Cuda(0).into(),
        false,
    ).await);

    run_mul_task_invocations(
        Arc::clone(&load_and_mut),
        &["resource_1", "resource_2", "resource_3"],
        device_idx,
        device_idx,
        |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
        move |res, out| {
            mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
        },
        true,
    )
        .await?;

    Ok::<_, ThoError>(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn test_load_and_clc_oom_splits_tensor() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }

    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    let device_idx = 1;

    const MUL_K: f64 = 2.;
    let size = [3, 40];
    let load_and_mut = Arc::new(LoadAndClcStage::new(
        LoadAndClcStageId("load_and_clc_id"),
        MUL_K,
        vec![FailUntil::MemLimit(12)], //fails until input tensor size (40) is bigger then 12
        size.into(),
        ClcOOMRunner::new(
            OOMRunnerConfig::Splits {},
            Box::new(move |id| {
                SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                    id,
                    0,
                    &size,
                    (Float, Device::Cuda(0)),
                    BuildAs::Sum,
                ).expect("single dim output builder")
            }),
        ),
        Float,
        Device::Cuda(0).into(),
        true,
    ).await);

    run_mul_task_invocations(
        Arc::clone(&load_and_mut),
        &[
            "resource_1",
            "resource_2",
            "resource_3",
            "resource_4",
            "resource_5",
        ], //,
        device_idx,
        device_idx,
        |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
        move |res, out| {
            // mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);

            mul_check_with_exp_f(MUL_K, res, out.as_ref(), 0.01, 0.01, |k, t| {
               Ok(lin_for_size(k, size, t.kind(), &t.device())? //same as in SyncTestTensorLoader::do_run
                    .sum_dim_intlist(1, false, t.kind())
                    .view((-1, 1)))
            })
        },
        true,
    ).await?;

    Ok::<_, ThoError>(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn test_load_and_clc_oom_max_parallel_and_splits_tensor() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }

    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    let device_idx = 1;
    const MUL_K: f64 = 2.;
    let size = [3, 100];
    let load_and_mut = Arc::new(LoadAndClcStage::new(
        LoadAndClcStageId("load_and_clc_id"),
        MUL_K,
        vec![FailUntil::NaxParallel(MAX_PARALLEL), FailUntil::MemLimit(12)],
        size.into(),
        ClcOOMRunner::new(
            OOMRunnerConfig::MaxParallelAndSplits {
                max: MAX_PARALLEL,
                step: *MaxParallelStep::SLOW,
            },
            Box::new(move |id| {
                SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                    id,
                    0,
                    &size,
                    (Float, Device::Cuda(0)),
                    BuildAs::Sum,
                ).expect("single dim output builder")
            }),
        ),
        Float,
        Device::Cuda(0).into(),
        true,
    ).await);

    run_mul_task_invocations(
        Arc::clone(&load_and_mut),
        (0..3)
            .map(|i| &*format!("resource_{i}").leak())
            .collect::<Vec<_>>()
            .as_slice(), //&["resource_1", "resource_2", "resource_3"],
        device_idx,
        device_idx,
        |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
        move |res, out| {
            mul_check_with_exp_f(MUL_K, res, out.as_ref(), 0.01, 0.01, |k, t| {
                Ok(lin_for_size(k, size, t.kind(), &t.device())? //same as in SyncTestTensorLoader::do_run
                    .sum_dim_intlist(1, false, t.kind())
                    .view((-1, 1)))
            });
        },
        true,
    )
        .await?;

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn test_load_and_clc_oom_dump_to_cpu() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }

    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    let device_idx = 1;

    const MUL_K: f64 = 2.;
    let size = [3, 100];
    let load_and_mut = Arc::new(LoadAndClcStage::new(
        LoadAndClcStageId("load_and_clc_id"),
        MUL_K,
        vec![FailUntil::NaxParallel(MAX_PARALLEL), FailUntil::DumpedToCpu],
        size.into(),
        ClcOOMRunner::new(
            OOMRunnerConfig::DumpedToCpu {},
            Box::new(move |id| {
                SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                    id,
                    1,
                    &[3, 100],
                    (Float, Device::Cuda(0)),
                    BuildAs::Append,
                ).expect("single dim output builder")
            }),
        ),
        Float,
        Device::Cuda(0).into(),
        false,
    ).await);

    run_mul_task_invocations(
        Arc::clone(&load_and_mut),
        (0..5)
            .map(|i| &*format!("resource_{i}").leak())
            .collect::<Vec<_>>()
            .as_slice(), //&["resource_1","resource_2",  "resource_3"], //,
        device_idx,
        device_idx,
        |_device_idx, _cache_area, name| RunCtxImpl::new(1, name),
        move |res, out| {
            mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
        },
        true,
    )
        .await?;

    Ok::<_, ThoError>(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 16)]
async fn heavy_test_purge_gpu_with_lock() -> Result<(), ThoError> {
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }

    let _global_ctx = GlobalCtx::init_devices(2)?;

    const MUL_K: f64 = 2.;
    let size = [3, 100000];

    let load_and_mut = Arc::new(LoadAndClcStage::new(
        LoadAndClcStageId("load_and_clc_id"),
        MUL_K,
        vec![],
        size.clone().into(),
        ClcOOMRunner::new(
            OOMRunnerConfig::DumpedToCpu {},
            Box::new(move |id| {
                SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                    id,
                    1,
                    &size,
                    (Float, Device::Cuda(0)),
                    BuildAs::Append,
                ).expect("single dim output builder")
            }),
        ),
        Float,
        Device::Cuda(0).into(),
        false,
    ).await);

    let ctx_device = Arc::new(RwLock::new(RunCtxImpl::new(1, "test")));

    for res in (0..10).map(|i| &*format!("resource_{i}").leak()) {

        Arc::clone(&load_and_mut)
            .run_cached(
                res.into(),
                ctx_device.write().branch(false),
            )
            .await?;
    }

    for i in 0..10 {
        let cached_loaded = <TestTensorLoader as Cached>::cache()
            .await
            // .get(&SimpleId::from(format!("loaded_resource_{i}")));
            .get(
                &SynKey::new([mono_key!(i, "resource key")])
            );
        let reff = cached_loaded.expect(&format!("no entry for resource_{i}"));
        debug!(
            "i={i} 0 cached_loaded: key={:?} val={:?}",
            reff.key(),
            reff.value().clone().await?
        );

        assert!(!reff.value().clone().await?.is_all_on_cpu().await);

        let cached_multed = <ClcStage as Cached>::cache()
            .await
            .get(
                // &SimpleId::from(format!("mult_mul_task_0_inp_loaded_resource_{i}"))
                &SynKey::new([mono_key!("MulTask", "MulTask"), mono_key!(i, "resource key")])
            );
        let reff = cached_multed.expect(&format!("no entry for esource_{i}"));
        debug!(
            "i={i} 0 cached_multed: key={:?} val={:?}",
            reff.key(),
            reff.value().clone().await?
        );

        assert!(!reff.value().clone().await?.is_all_on_cpu().await);
    }

    let mut loaded_tensors_ids = HashSet::new();
    let loaded_tensor_3 =  SynKey::new([mono_key!("resource_3", "resource_3")]);
    loaded_tensors_ids.insert(&loaded_tensor_3 as &dyn CachableIdTr);

    let mut multed_tensors_ids = HashSet::new();
    let mult_mul_task_0_inp_loaded_resource_3 = debug_id!("mult_mul_task_0_inp_loaded_resource_3");
    multed_tensors_ids.insert(&mult_mul_task_0_inp_loaded_resource_3 as &dyn CachableIdTr);

    let mut required_ids = HashMap::new();
    required_ids.insert("LoadedTensorId", &loaded_tensors_ids);
    required_ids.insert("MulTaskOutputId", &multed_tensors_ids);

    _global_ctx.dump_all_to_cpu(&required_ids, 1, "test").await?;

    for i in 0..10 {
        let cached_loaded = <TestTensorLoader as Cached>::cache()
            .await
            .get(
                // &SimpleId::from(format!("loaded_resource_{i}"))
                &SynKey::new([mono_key!(i, "resource key")])
            );
        let reff = cached_loaded.expect(&format!("no entry for loaded_resource_{i}"));
        debug!(
            "i={i} 1 cached_loaded: key={:?} val={:?}",
            reff.key(),
            reff.value().clone().await?
        );

        if i == 3 {
            assert!(!reff.value().clone().await?.is_all_on_cpu().await);
        } else {
            assert!(reff.value().clone().await?.is_all_on_cpu().await);
        }

        let cached_multed = <ClcStage as Cached>::cache()
            .await
            .get(
                // &SimpleId::from(format!("mult_mul_task_0_inp_loaded_resource_{i}"))
                &SynKey::new([mono_key!("MulTask", "MulTask"), mono_key!(i, "resource key")])
            );
        let reff = cached_multed.expect(&format!("no entry for loaded_resource_{i}"));
        debug!(
            "i={i} 1 cached_multed: key={:?} val={:?}",
            reff.key(),
            reff.value().clone().await?
        );

        if i == 3 {
            assert!(!reff.value().clone().await?.is_all_on_cpu().await);
        } else {
            assert!(reff.value().clone().await?.is_all_on_cpu().await);
        }
    }

    Ok(())
}

pub async fn do_real_oom_dump_and_split() -> Result<(), ThoError> {
    let (free_mem, _total_mem) = cuda_mem();
    let init_type = Kind::Int64;

    let total_size = (free_mem as f64 * 0.7) as i64 / init_type.elt_size_in_bytes() as i64; //123456; //25000; //123456; //
    let dim_1 = 3i64;
    let input_n = total_size / dim_1;

    debug!("test_real_oom_dump_and_split: input_n={input_n} dim_1={dim_1}");

    const MUL_K: i64 = 1;
    let size = [dim_1, input_n];

    let load_and_mut = Arc::new(LoadAndClcStage::new(
        LoadAndClcStageId("load_and_clc_id"),
        MUL_K,
        vec![],
        size.clone().into(),
        ClcOOMRunner::new(
            OOMRunnerConfig::DumpedToCpu {},
            Box::new(move |id| {
                SingleDimTensorOutputBuilder::<SynKey, ClcStageOutput>::new(
                    id,
                    1,
                    &[dim_1, 1],
                    (init_type, Cpu),
                    BuildAs::Sum,
                ).expect("single dim output builder")
            }),
        ),
        Kind::Int64,
        Cuda(0).into(),
        true,
    ).await);

    let multed = Arc::clone(&load_and_mut)
        .run_cached(
            "resource_1".into(),
            RunCtxImpl::new(1, "test"),
        )
        .await?;

    debug!("test_real_oom_dump_and_split: multed={multed:?}");

    let ThoDataRef::ReadLocked(r) = BorrowRef::borrow_ref(&multed, 1).await? else {
        panic!("test_real_oom_dump_and_split: unexpected result multed={multed:?}");
    };

    let cpu_r = r.copy_to_device(0).await?;


    mul_check_with_exp_f(MUL_K as f64, "resource_1", &cpu_r, 0.01, 0.01, |k, t| {
        Ok(lin_for_size(k, size, t.kind(), &Cpu)?
            .sum_dim_intlist(1, false, t.kind())
            .view((-1, 1)))
    });

    debug!("Ok");



    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn heavy_test_real_oom_dump_and_split_local() -> Result<(), ThoError> {
    debug!("test_real_oom_dump_and_split");
    unsafe {
        let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
    }

    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    do_real_oom_dump_and_split().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_with_tho_tensor() -> Result<(), ThoError> {
    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2)?;
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

    let tmp_cache: TmpCached<ThoTensor<AnyDims, DebugId>> = TmpCached::new("test").await;


    tokio::spawn(async move {
        let tensor_0 = Tensor::from_slice(&[0., 1., 2.]);
        let tensor = Tensor::from_slice(&[0., 1., 2.]);
        debug!("device={:?}", tensor.device());

        tmp_cache.insert(ThoTensor::<AnyDims, DebugId>::new(debug_id!("test"), tensor)?);
        let wrap = tmp_cache.get(&debug_id!("test")).unwrap();

        let v = &wrap.load_from_any_device_(1).await?;

        //ThoTensor is sync so &ThoTensor is Send =
        async {
            GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;
            //[WARN] should be logged - can't drop self.id=simple_id!("test"), locked
            Ok::<_, ThoError>(())
        }.await?;

        //copies from cpu
        // let v = &wrap.load_from_any_device_(1).await?;
        debug!("tensor={tensor_0:?}, v={v:?}");
        //TODO asserts

        Ok::<(), ThoError>(())


    }).await??;



    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 8)]
async fn test_with_data() -> Result<(), ThoError> {
    async fn load_/*from_any_device_*/(i: i32) -> i32 { // load steps invoked while borrowing cached data
        i * 10
    }

    fn make_batch_stream(
        hold_permit_while_unpolled: bool,
    ) -> (
        futures_util::stream::BoxStream<'static, i32>,
        Arc<tokio::sync::RwLock<()>>,
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<tokio::sync::Notify>,
    ) {
        let full_access_to_gpu = Arc::new(tokio::sync::RwLock::new(()));//emulates Arbiter - write lock - 100 max_parallel, read lock 1 permit
        let full_access_to_gpu_for_batch = Arc::clone(&full_access_to_gpu);
        let started = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let started_for_batch = Arc::clone(&started);
        let resume = Arc::new(tokio::sync::Notify::new());
        let resume_for_batch = Arc::clone(&resume);

        let batch = futures::stream::iter(0..3).map(move |i| {
            let full_access_to_gpu_for_batch = Arc::clone(&full_access_to_gpu_for_batch);
            let started = Arc::clone(&started_for_batch);
            let resume = Arc::clone(&resume_for_batch);

            async move {
                let guard = full_access_to_gpu_for_batch.read().await;//1 permit saved into ctx if there are no max_parallel yet
                started.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                let data = load_(i).await;

                debug!("with_data: {i} started={}", started.load(std::sync::atomic::Ordering::SeqCst));

                if i == 0 {
                    while started.load(std::sync::atomic::Ordering::SeqCst) < 3 {
                        tokio::task::yield_now().await;
                    }
                    drop(guard);
                    data
                } else {
                    if !hold_permit_while_unpolled {
                        drop(guard);
                    }
                    resume.notified().await;
                    data
                }
            }
        })
            .buffer_unordered(3)
            .boxed();

        (batch, full_access_to_gpu, started, resume)
    }

    let (mut blocked_batch, full_access_to_gpu, started, resume) = make_batch_stream(true);
    let i = tokio::time::timeout(Duration::from_millis(100), blocked_batch.next())
        .await
        .expect("first batch should arrive")
        .expect("buffer_unordered should produce one ready batch");
    assert_eq!(i, 0);
    assert_eq!(started.load(std::sync::atomic::Ordering::SeqCst), 3);

    // train cycle at crates/thoosa_main/src/stages/train_net.rs:273.
    // Buffered batch futures 1 and 2 are no longer being polled, but still hold read guards.
    let got_all_gpu = tokio::time::timeout(Duration::from_millis(100), async move {
        let _guard = full_access_to_gpu.write().await; // oom_handler.run at crates/thoosa_main/src/stages/train_net.rs:278 when it's trying to get 100% GPU
        debug!("got all gpu");
    }).await;
    assert!(got_all_gpu.is_err(), "writer should starve while buffered batch futures keep read guards alive");

    resume.notify_waiters();
    let mut remaining = vec![];
    while remaining.len() < 2 {
        remaining.push(tokio::time::timeout(Duration::from_millis(100), blocked_batch.next()).await.expect("blocked batch should resume once polled again").expect("stream should still have items"));
    }
    remaining.sort();
    assert_eq!(remaining, vec![10, 20]);

    let (mut releasable_batch, full_access_to_gpu, started, resume) = make_batch_stream(false);
    let i = tokio::time::timeout(Duration::from_millis(100), releasable_batch.next())
        .await
        .expect("first batch should arrive")
        .expect("buffer_unordered should produce one ready batch");
    assert_eq!(i, 0);
    assert_eq!(started.load(std::sync::atomic::Ordering::SeqCst), 3);

    let got_all_gpu = tokio::time::timeout(Duration::from_millis(100), async move {
        let _guard = full_access_to_gpu.write().await;
        debug!("got all gpu after pending futures released their max_parallel");
    }).await;
    assert!(got_all_gpu.is_ok(), "writer should succeed once unpolled futures stop holding permits");

    resume.notify_waiters();
    let mut remaining = vec![];
    while remaining.len() < 2 {
        remaining.push(tokio::time::timeout(Duration::from_millis(100), releasable_batch.next()).await.expect("releasable batch should resume once polled again").expect("stream should still have items"));
    }
    remaining.sort();
    assert_eq!(remaining, vec![10, 20]);

    Ok(())

}
