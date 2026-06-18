use std::fmt::Debug;
use tch::{Kind, Tensor};
use thoosa_shared::ThoError;
use thoosa_shared::traits::Named;
use crate::dims::Dims;
use crate::tho_tensor::ThoTensor;

pub mod data;
pub mod metrics;
#[cfg(test)]
mod model_test;
pub mod output_builders;

pub struct ThoTensorView<'a, D: Dims, Id: Named + Debug + Clone> {
    t: Tensor,
    _owner: &'a ThoTensor<D, Id>,
}
unsafe impl<'a, D: Dims, Id: Named + Debug + Clone> Sync for ThoTensorView<'a, D, Id> {}
unsafe impl<'a, D: Dims, Id: Named + Debug + Clone> Send for ThoTensorView<'a, D, Id> {}

impl<'a, D: Dims, Id: Named + Debug + Clone> ThoTensorView<'a, D, Id> {
    pub fn select(self, dim: i64, index: i64) -> Result<Self, ThoError> {
        Ok(Self {
            t: self.t.f_select(dim, index)?,
            ..self
        })
    }
    pub fn sum(self, k: Kind) -> Result<ThoTensor<D, Id>, ThoError> {
        ThoTensor::new(self._owner.id().clone(), self.t.f_sum(k)?)
    }
}



#[cfg(test)]
#[allow(unused_imports, unused_variables, dead_code)]
pub(crate) mod tests {

    use dashmap::DashMap;
    use crate::runner::common::*;
    use futures_util::future::{BoxFuture, Shared};
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::{HashMap, HashSet};
    use std::ops::Mul;
    use std::sync::Arc;
    use async_trait::async_trait;
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use tch::{Device, IndexOp, Kind, Scalar, Tensor};
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, MonoMetrics, Named, Select, SimpleId, WithId};
    use thoosa_shared::{mono_key, simple_id, CopyToDevice};
    use thoosa_shared::{Cause, ThoError};
    use tracing_subscriber::fmt::format::FmtSpan;
    use tracing_subscriber::{fmt, EnvFilter};
    use thoosa_shared::comp_id::{MonoKey, SynKey};
    use thoosa_shared::copy_mode::CpuOnly;
    use crate::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use crate::runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use crate::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use crate::runner::run_ctx_2::RunCtxImpl;
    use thoosa_named_derive::Named;
    use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, FailUntil, OOMRunnerConfig};
    use crate::runner::test::FailUntil::NaxParallel;
    use crate::runner::test_utils::parse_numeric_part;
    use crate::cuda_utils::cuda_mem;
    use crate::dims::{AnyDims, Dims};
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    use crate::utils::lin_for_size;
    use std::ops::Add;
    use std::time::Duration;
    use futures_util::StreamExt;
    use crate::runner::tmp_cache::TmpCached;
    use crate as thoosa_common;
    use crate::runner::adjusters::max_parallel::{MaxParallel, MaxParallelStep, MAX_PARALLEL};
    use crate::runner::adjusters::splits::SplitsAdjusterContextImpl;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::oom_handler::{Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    //////////////////LOADER
    #[derive(Debug, Clone, Eq, PartialEq, Hash, Named)]
    pub struct LoaderInputId(&'static str);

    #[derive(Debug, Clone)]
    pub struct LoaderInput {
        id: LoaderInputId,
    }
    impl Input for LoaderInput {  }

    impl WithId for LoaderInput {
        type ID = LoaderInputId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl From<&'static str> for LoaderInput {
        fn from(value: &'static str) -> Self {
            LoaderInput {
                id: LoaderInputId(value),
            }
        }
    }

    #[async_trait]
    impl CopyToDevice for LoaderInput {
        async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
            Ok(self.clone())
        }
    }

    #[async_trait]
    impl Data for LoaderInput {
        type Metrics = ();
        type M = CpuOnly<Self>;

        fn size(&self) -> Self::Metrics {
            ()
        }

        fn device(&self) -> usize {
            0
        }

        fn mem_size(&self) -> u64 {
            0
        }
    }

    #[derive(Debug, Clone)]
    pub struct LoadedTensor;
    impl Dims for LoadedTensor {
        type Metrics = MonoMetrics<1, 2>;

    }
    impl LoadedTensor {}

    // impl Adjustable for ThoTensor2<LoadedTensor, SimpleId> {}

    #[derive(Debug, Clone, Default)]
    pub(crate) struct  LoadedRange {
        range:std::ops::Range<MonoMetrics<1, 2>>
    }
    impl Adjustable for LoadedRange {}

    impl From<std::ops::Range<MonoMetrics<1, 2>>> for LoadedRange {
        fn from(value: std::ops::Range<MonoMetrics<1, 2>>) -> Self {
            Self { range: value }
        }
    }

    impl Select<LoadedRange> for ThoTensor<LoadedTensor, SynKey> {
        type Output = ThoTensor<LoadedTensor, SynKey>;

        fn subview(&self, selector: LoadedRange) -> Result<Self::Output, ThoError> {
            let ranges = MonoMetrics::<1,2>::map_to_ranges(&selector.range.start, &selector.range.end)?;//TODO unwrap

            let sub_tensor = self.t_ref().i((ranges[0].clone(), ranges[1].clone()));

            let new_id = self.id().clone(); //SimpleId::from(format!("{}:{ranges:?}", self.id().0));

            debug!("Select for ThoTensor2<LoadedTensor, SimpleId>: ranges={ranges:?} sub_tensor={sub_tensor:?}");

            ThoTensor::new(new_id, sub_tensor)
        }
    }


    #[derive(Debug, cached_derive::Cached, Named)]
    #[cache_name = "LoadedTensorId"]
    struct TestTensorLoader {
        size: MonoMetrics<1, 2>,
        kind: Kind,
    }

    impl TestTensorLoader {
        pub fn new(size: MonoMetrics<1, 2>, kind: Kind) -> Self {
            Self { size, kind }
        }
    }

    impl OutId for TestTensorLoader {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            debug!("TestTensorLoader::out_id: input_id={input_id:?}");
            SynKey::new([mono_key!(input_id.0, "resource key")])
        }
    }

    impl Stage for TestTensorLoader {
        type In<'a> = LoaderInput;
        type Out = ThoTensor<LoadedTensor, SynKey>;
        type Ctx = RunCtxImpl;
    }

    // #[async_trait]
    impl AsyncStage for TestTensorLoader {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            ctx: Self::Ctx,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
            let k = parse_numeric_part(WithId::id(input.as_ref()).0).unwrap_or(1.0);

            let device_idx = ctx.tho_device_idx();


            let tensor = ThoTensor::new(
                self.out_id(WithId::id(input.as_ref())),
                lin_for_size(
                    k,
                    (&self.size).into(),
                    self.kind,
                    &ThoDevice::from(device_idx),
                ),
            )?;

            Ok(Arc::new(DevicesWrap::wrap(tensor)))
        }

    }

    // //////////////////////////////
    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct MulK;

    impl Dims for MulK {
        type Metrics = MonoMetrics<1, 2>;
    }

    #[derive(Debug, cached_derive::Cached, Named)]
    #[cache_name = "MulTaskOutputId"]
    pub(crate) struct ClcTask
    {
        pub(crate) sync_clc: Arc<SyncClcTask>,
    }

    #[derive(Debug)]
    pub(crate) struct SyncClcTask {
        id: SynKey,
        k: ThoTensor<MulK, SimpleId>,
        fail_until: Vec<FailUntil>,
        use_matmul: bool,
    }

    impl SyncClcTask {
        pub fn new(id: SynKey,k: impl Into<Scalar>, dim: i64, fail_until: Vec<FailUntil>,kind: Kind, device: Device, use_matmul: bool) -> Self {
            let k =
                if use_matmul {
                    ThoTensor::new(
                        simple_id!("mul_k"),
                        Tensor::ones([dim, 1], (kind, device))
                            .f_mul_scalar_(k)
                            .unwrap(),
                    ).unwrap()
                } else {
                    ThoTensor::new(
                        simple_id!("mul_k"),
                        Tensor::ones([dim], (kind, device))
                            .f_mul_scalar_(k)
                            .unwrap(),
                    ).unwrap()
                };

            debug!("ClcTask::new: use_matmul={use_matmul} i.size={:?}", k.size());

            Self { id, k, fail_until, use_matmul }
        }
    }

    impl ClcTask
    {
        pub fn new(
            id: SynKey,
            k: impl Into<Scalar>,
            fail_until: Vec<FailUntil>,
            dim: i64,
            kind: Kind,
            device: Device,
            use_matmul: bool,
        ) -> Self {

            Self {
                sync_clc: Arc::new(SyncClcTask::new(id, k, dim, fail_until, kind, device, use_matmul)),
            }
        }
    }

    impl Adjustable for ClcTask {}
    impl Adjustable for SyncClcTask {}



    impl OutId for ClcTask {
        fn out_id(&self, input_id: &SynKey) -> SynKey {

            self.sync_clc.out_id(input_id)
        }
    }

    impl OutId for SyncClcTask {
        fn out_id(&self, input_id: &SynKey) -> SynKey {

            &self.id + input_id
        }
    }

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    pub struct ClcTaskOutput;
    impl Dims for ClcTaskOutput {
        type Metrics = MonoMetrics<1, 2>;
    }

    impl LoadedTensor {
        fn clc_hadamard(
            id: SynKey,
            loaded: &ThoTensor<LoadedTensor, SynKey>,
            k: &ThoTensor<MulK, SimpleId>,
        ) -> Result<ThoTensor<ClcTaskOutput, SynKey>, ThoError> {

            ThoTensor::new( id,loaded.t_ref().multiply(&k.as_ref()))

        }

        fn clc_sum(
            id: SynKey,
            loaded: &ThoTensor<LoadedTensor, SynKey>,
            k: &ThoTensor<MulK, SimpleId>,
        ) -> Result<ThoTensor<ClcTaskOutput, SynKey>, ThoError> {
            debug!("LoadedTensor::clc_sum: loaded={:?} k.t={:?}",loaded.t_ref(),k.t_ref());

            let multed = loaded.t_ref().f_multiply(&k.as_ref());

            debug!("LoadedTensor::clc_sum: multed={multed:?}");

            let sum = multed?
                .f_sum_dim_intlist(1, false, loaded.t_ref().kind());

            debug!("LoadedTensor::clc_sum: sum={sum:?}");

            let tensor = sum?
                .view((-1, 1));
            ThoTensor::new(
                id,
                tensor
            )
        }
    }

    impl Stage for ClcTask {
        type In<'a> = ThoTensor<LoadedTensor, SynKey>;
        type Out = ThoTensor<ClcTaskOutput, SynKey>;
        type Ctx = RunCtxImpl;
    }

    impl Stage for SyncClcTask {
        type In<'a> = ThoTensor<LoadedTensor, SynKey>;
        type Out = ThoTensor<ClcTaskOutput, SynKey>;
        type Ctx = RunCtxImpl;
    }

    impl SyncStage for SyncClcTask {
        fn do_run<'a>(
            self: Arc<Self>,
            input: &Self::In<'a>,
            ctx: &Self::Ctx,
        ) -> Result<Self::Out, ThoError>
        where
            Self: Sized,
        {
            // tokio::time::sleep(tokio::time::Duration::from_millis(random_range(0..10))).await;
            debug!("MulTask::run: loaded.id={:?}", input.id());

            let deive_idx = ctx.tho_device_idx();

            for fail_until in &self.fail_until {
                match fail_until {
                    FailUntil::MemLimit(mem_limit) => {
                        let _device_idx = deive_idx;
                        // debug!("MulTask::do_run: ctx={ctx:?} input.len={}", input.data.0.as_ref().unwrap().size()[0]);

                        if input.size()[0] > *mem_limit as i64 {
                            debug!(
                                "MulTask::do_run: MyError::OutOfMemory on MemLimit mem_limit={mem_limit}"
                            );
                            return Err(ThoError::oom());
                        }
                    }
                    FailUntil::NaxParallel(max_other_parallel_threshold) => {
                        let max_other_parallel = ctx.max_other_parallel();
                        debug!("MulTask::do_run: ctx={ctx:?} max_other_parallel={max_other_parallel} max_other_parallel_threshold={max_other_parallel_threshold}");

                        if max_other_parallel > *max_other_parallel_threshold {
                            return Err(ThoError::oom());
                        }
                    }

                    FailUntil::DumpedToCpu => {
                        let device_idx = ctx.tho_device_idx();
                        let is_all_on_cpu =
                            GlobalCtx::global().is_all_on_cpu(device_idx);
                        debug!(
                            "MulTask::do_run: ctx.name={:?} device_idx={} is_all_on_cpu={is_all_on_cpu}",
                            "ctx.name()",
                            device_idx
                        );

                        if !is_all_on_cpu {
                            debug!("MulTask::do_run: MyError::OutOfMemory on DumpedToCpu is_all_on_cpu={is_all_on_cpu}");
                            return Err(ThoError::oom());
                        }
                    }
                    FailUntil::Never => {}
                }
            }

            if self.use_matmul {
                LoadedTensor::clc_sum(self.out_id(input.id()), input, &self.k)
            } else {
                LoadedTensor::clc_hadamard(self.out_id(input.id()), input, &self.k)
            }
        }
    }

    #[derive(Debug)]
    struct ProxySyncClcTask {

    }

    impl Stage for ProxySyncClcTask {
        type In<'a> = ThoTensor<LoadedTensor, SynKey>;
        type Out = ThoTensor<ClcTaskOutput, SynKey>;
        type Ctx = RunCtxImpl;
    }

    impl SyncStage for ProxySyncClcTask {

    }

    // #[async_trait]
    impl AsyncStage for ClcTask {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            ctx: Self::Ctx,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
            let device_idx = ctx.tho_device_idx();

            debug!("AsyncStage for ClcTask::run: device_idx={device_idx} input={input:?}");

            let output = OOMHandler::new(vec![], None).run(
                &self.sync_clc,
                input,
                ctx,
                Contexts::new(
                    NoopInputContext(None),
                    NoopRunnableContext {},
                    NoopRunCtxContext {},
                )
            ).await?;


            Ok(Arc::new(DevicesWrap::wrap(output)))
        }

    }

    //////////////////////////////

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    pub(crate) struct LoadAndClcTaskId(pub(crate) &'static str);

    #[derive(Debug, Named)]
    pub(crate) struct LoadAndClcTask {
        #[allow(dead_code)]
        id: LoadAndClcTaskId,

        loader: Arc<TestTensorLoader>,
        clc_task: Arc<ClcTask>,
        clc_oom_runner: ClcOOMRunner<SyncClcTask, LoadedRange, SingleDimTensorOutputBuilder<SynKey, ClcTaskOutput>>,
    }

    impl Stage for LoadAndClcTask {
        type In<'a> = LoaderInput;
        type Out = ThoTensor<ClcTaskOutput, SynKey>;
        type Ctx = RunCtxImpl;
    }

    // #[async_trait]
    impl AsyncStage for LoadAndClcTask {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            mut ctx: Self::Ctx,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
            let ctx_for_loader = ctx.branch(false);
            let loaded = Arc::clone(&self.loader)
                .run_cached(Arc::clone(&input), ctx_for_loader)
                .await?;

            let loaded_id = loaded.id().clone();
            let output_id = self.clc_task.out_id(&loaded_id.clone());

            debug!("LoadAndMulTask::run: loaded_id={loaded_id:?} output_id={output_id:?}");

            let loaded_id_name = "LoadedTensorId"; //loaded.id().id_name().clone();
            let output_id_name = "MulTaskOutputId"; //loaded.id().id_name().clone();

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

            let result = self
                .clc_oom_runner
                .run(&self.clc_task.sync_clc, loaded, ctx.branch(false), &required_ids)
                .await?;

            let device_idx = ctx.tho_device_idx();

            Ok(Arc::new(DevicesWrap::wrap(result)))
        }
    }

    #[async_trait]
    impl Cached for LoadAndClcTask {
        async fn cache() -> &'static DashMap<
            <<Self as Stage>::Out as WithId>::ID,
            Shared<BoxFuture<'static, Result<Arc<DevicesWrap<Self::Out>>, ThoError>>>,
        >
        where
            Self: Sized,
        {
            <ClcTask as Cached>::cache().await
        }
    }

    impl OutId for LoadAndClcTask {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            self.clc_task.out_id(&self.loader.out_id(input_id))
        }
    }

    impl LoadAndClcTask {
        pub fn new(
            id: LoadAndClcTaskId,
            k: impl Into<Scalar>,
            fail_until: Vec<FailUntil>,
            dim: MonoMetrics<1, 2>,
            oom_runner: ClcOOMRunner<SyncClcTask, LoadedRange, SingleDimTensorOutputBuilder<SynKey, ClcTaskOutput>>,
            kind: Kind,
            device: Device,
            use_matmul: bool,
        ) -> Self {
            let dim_1 = if use_matmul {dim.dim_0()} else {dim.dim_1()};
            Self {
                id,
                loader: Arc::new(TestTensorLoader::new(dim, kind)),
                clc_task: Arc::new(ClcTask::new(
                    SynKey::new([mono_key!(0, "mul_task_0")]),
                    k,
                    fail_until,
                    dim_1,
                    kind,
                    device,
                    use_matmul,
                )),
                clc_oom_runner: oom_runner,
            }
        }
    }
    // //////////////////////////////

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_load_tensor_cuda_to_cuda() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);

        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let loader = Arc::new(TestTensorLoader::new([10, 3].into(), Float));

        do_test_loaded(1, 1, &loader).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_load_tensor_cpu_to_cuda() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);

        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let loader = Arc::new(TestTensorLoader::new([10, 3].into(), Float));

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
            Arc::new(DevicesWrap::wrap("resource_1".into())),
            ctx_device_0.branch(false),
        ).await?;

        debug!("loaded_tensor_0_0={loaded_tensor_0_0:?}");

        let tensor_data_0_0 = loaded_tensor_0_0
            .with_data_ref(load_to_device, ctx_device_0.branch(false), None, |tensor_data_0_0: &ThoTensor<LoadedTensor, SynKey>, _ctx| {
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

                    Ok(tensor_data_0_0.t_ref().shallow_clone())
                })
            .await?;

        let _loaded_tensor_0_1 = Arc::clone(&loader)
            .run_cached(
                Arc::clone(&Arc::new(DevicesWrap::wrap("resource_1".into()))),
                ctx_device_0.branch(false),
            )
            .await?;

        loaded_tensor_0_0
            .with_data_ref(load_to_device,
                           ctx_device_0.branch(false), None, move|tensor_data_0_1: &ThoTensor<LoadedTensor, SynKey>, _ctx| {
                    // let tensor_data_0_1 = loaded_tensor_0_1.load_from_any_device(load_to_device, None).await?;

                    assert_eq!(&tensor_data_0_0, tensor_data_0_1.t_ref());
                    Ok(())
                })
            .await?;
        // let tensor_data_0_0 = loaded_tensor_0_0.load_from_any_device(load_to_device, None).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_no_oom_tensor() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let device_idx = 1;
        const MUL_K: f64 = 2.;

        let size = [3, 10];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel { max: 20, step: 10 },
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
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
            .await?;

        Ok(())
    }

    pub(crate) fn mul_check(
        mul_k: f64,
        size: MonoMetrics<1, 2>,
        res: &str,
        result: &ThoTensor<ClcTaskOutput, SynKey>,
        rtol: f64,
        atol: f64,
    ) {
        mul_check_with_exp_f(mul_k, res, result, rtol, atol, |k, t| {
            lin_for_size(k, (&size).into(), t.kind(), &t.device())
        })
    }

    fn mul_check_with_exp_f<F>(
        mul_k: f64,
        res: &str,
        result: &ThoTensor<ClcTaskOutput, SynKey>,
        rtol: f64,
        atol: f64,
        exp_f: F,
    ) where
        F: Fn(f64, &Tensor) -> Tensor,
    {
        let self_k = parse_numeric_part(res).unwrap_or(1.0);

        let expected = exp_f(self_k * mul_k, &result.t_ref());

        debug!("mul_check: res={res}  result_t.size()={:?} \n\n",result.size());

        let ok = expected.allclose(&result.t_ref(), rtol, atol, false);
        // if !ok {
        debug!("mul_check: res={res} expected=\n{expected} result_t=\n{} \n\n",result.t_ref());
        // }
        assert!(ok);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_till_min_max_parallel_tensor() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let device_idx = 1;
        const MUL_K: f64 = 2.;

        let size = [3, 10];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::NaxParallel(20)],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel { max: 20, step: 10 },
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
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        ).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_compete_for_max_parallel_tensor() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let device_idx = 1;

        const MUL_K: f64 = 2.;
        let size = [3, 10];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::NaxParallel(MAX_PARALLEL)],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel {
                    max: MAX_PARALLEL,
                    step: *MaxParallelStep::TEN,
                },
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
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
            .await?;

        Ok::<_, ThoError>(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_splits_tensor() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let device_idx = 1;

        const MUL_K: f64 = 2.;
        let size = [3, 10];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::MemLimit(12)],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::Splits {},
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
            Arc::clone(&load_and_mut),
            &[
                "resource_1",
                "resource_2",
                "resource_3",
                "resource_4",
                "resource_5",
            ], //,
            device_idx,
            |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        ).await?;

        Ok::<_, ThoError>(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_max_parallel_and_splits_tensor() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let device_idx = 1;
        const MUL_K: f64 = 2.;
        let size = [3, 100];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::NaxParallel(MAX_PARALLEL), FailUntil::MemLimit(12)],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallelAndSplits {
                    max: MAX_PARALLEL,
                    step: *MaxParallelStep::TEN,
                },
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
            Arc::clone(&load_and_mut),
            (0..3)
                .map(|i| &*format!("resource_{i}").leak())
                .collect::<Vec<_>>()
                .as_slice(), //&["resource_1", "resource_2", "resource_3"],
            device_idx,
            |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
            .await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_dump_to_cpu() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let device_idx = 1;

        const MUL_K: f64 = 2.;
        let size = [3, 100];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![NaxParallel(MAX_PARALLEL), FailUntil::DumpedToCpu],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::DumpedToCpu {},
                Box::new(move |id| {
                    SingleDimTensorOutputBuilder::<SynKey, ClcTaskOutput>::new(
                        id,
                        1,
                        &[3, 100],
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
            Arc::clone(&load_and_mut),
            (0..5)
                .map(|i| &*format!("resource_{i}").leak())
                .collect::<Vec<_>>()
                .as_slice(), //&["resource_1","resource_2",  "resource_3"], //,
            device_idx,
            |_device_idx, _cache_area, name| RunCtxImpl::new(1, name),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
            .await?;

        Ok::<_, ThoError>(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn heavy_test_purge_gpu_with_lock() -> Result<(), ThoError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let _global_ctx = GlobalCtx::init_devices(2);

        const MUL_K: f64 = 2.;
        let size = [3, 100000];

        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![],
            size.clone().into(),
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

        let ctx_device = Arc::new(RwLock::new(RunCtxImpl::new(1, "test")));

        for res in (0..10).map(|i| &*format!("resource_{i}").leak()) {

            Arc::clone(&load_and_mut)
                .run_cached(
                    Arc::new(DevicesWrap::wrap(res.into())),
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

            let cached_multed = <ClcTask as Cached>::cache()
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
        let mult_mul_task_0_inp_loaded_resource_3 = simple_id!("mult_mul_task_0_inp_loaded_resource_3");
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

            let cached_multed = <ClcTask as Cached>::cache()
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn heavy_test_real_oom_dump_and_split() -> Result<(), ThoError> {
        debug!("test_real_oom_dump_and_split");
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let (free_mem, _total_mem) = cuda_mem();
        let init_type = Kind::Int64;

        let total_size = (free_mem as f64 * 0.7) as i64 / init_type.elt_size_in_bytes() as i64; //123456; //25000; //123456; //
        let dim_1 = 3i64;
        let input_n = total_size / dim_1;

        debug!("test_real_oom_dump_and_split: input_n={input_n} dim_1={dim_1}");

        const MUL_K: i64 = 1;
        let size = [dim_1, input_n];

        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![],
            size.clone().into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::DumpedToCpu {},
                Box::new(move |id| {
                    SingleDimTensorOutputBuilder::<SynKey, ClcTaskOutput>::new(
                        id,
                        1,
                        &[dim_1, 1],
                        (init_type, Device::Cuda(0)),
                        BuildAs::Sum,
                    ).expect("single dim output builder")
                }),
            ),
            Kind::Int64,
            Device::Cuda(0),
            true,
        ));

        let ctx_device = Arc::new(RwLock::new(RunCtxImpl::new(1, "test")));



        let multed = Arc::clone(&load_and_mut)
            .run_cached(
                Arc::new(DevicesWrap::wrap("resource_1".into())),
                ctx_device.write().branch(false),
            )
            .await?;

        debug!("multed={multed:?}");

        multed
            .with_data_ref(1,
                           ctx_device.write().branch(false), None, |r: &ThoTensor<ClcTaskOutput, SynKey>, _ctx| {
                    mul_check_with_exp_f(MUL_K as f64, "resource_1", r, 0.01, 0.01, |k, t| {
                        lin_for_size(k, size, t.kind(), &t.device())
                            .sum_dim_intlist(1, false, t.kind())
                            .view((-1, 1))
                    });

                    debug!("Ok");

                    Ok(())
                })
            .await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_with_tho_tensor() -> Result<(), ThoError> {


        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);
        
        let tmp_cache: TmpCached<ThoTensor<AnyDims, SimpleId>> = TmpCached::new("test").await;


        tokio::spawn(async move {
            let tensor_0 = Tensor::from_slice(&[0., 1., 2.]);
            let tensor = Tensor::from_slice(&[0., 1., 2.]);
            debug!("device={:?}", tensor.device());

            tmp_cache.insert(ThoTensor::<AnyDims, SimpleId>::new(simple_id!("test"), tensor)?);
            let wrap = tmp_cache.get(&simple_id!("test")).unwrap();

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
        async fn load_/*from_any_device_*/(i: i32) -> i32 { // load_steps invoked from with_data! crates/thoosa_macros/thoosa_with_data/src/lib.rs:143
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
                    let guard = full_access_to_gpu_for_batch.read().await;//1 permit saved into ctx if there are no max_parallel yet at crates/thoosa_macros/thoosa_with_data/src/lib.rs:127
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

        // train cycle at crates/thoosa_main/src/tasks/train_net.rs:273.
        // Buffered batch futures 1 and 2 are no longer being polled, but still hold read guards.
        let got_all_gpu = tokio::time::timeout(Duration::from_millis(100), async move {
            let _guard = full_access_to_gpu.write().await; // oom_handler.run at crates/thoosa_main/src/tasks/train_net.rs:278 when it's trying to get 100% GPU
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



}
