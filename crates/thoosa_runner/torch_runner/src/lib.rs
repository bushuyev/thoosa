#![feature(lazy_get)]

extern crate core;

pub mod data;
pub mod metrics;
#[cfg(test)]
mod model_test;
mod output_builders;

#[allow(unused_imports)]
use caching_runner::adjusters::splits::OutputBuilder;
use caching_runner::common::{DropIf, PermitsCtx};
use caching_runner::common::cachable::WithDataRef;
use caching_runner::permits::{MAX_PERMITS, NumPermits, RunnerPerm};
// test-only imports live under cfg(test)
use thoosa_common::tho_tensor::ThoTensor;
use log::debug;
use std::fmt::Debug;
use tch::{Kind, Tensor};
use thoosa_common::dims::Dims;
use thoosa_shared::traits::Named;

pub struct ThoTensorView<'a, D: Dims, Id: Named + Clone> {
    t: Tensor,
    _owner: &'a ThoTensor<D, Id>,
}
unsafe impl<'a, D: Dims, Id: Named + Clone> Sync for ThoTensorView<'a, D, Id> {}
unsafe impl<'a, D: Dims, Id: Named + Clone> Send for ThoTensorView<'a, D, Id> {}

impl<'a, D: Dims, Id: Named + Clone> ThoTensorView<'a, D, Id> {
    pub fn select(self, dim: i64, index: i64) -> Self {
        Self {
            t: self.t.select(dim, index),
            ..self
        }
    }
    pub fn sum(self, k: Kind) -> ThoTensor<D, Id> {
        ThoTensor::new(self._owner.id().clone(), self.t.sum(k))
    }
}

#[derive(Debug)]
pub struct TorchPermsCtx {
    name: &'static str,
    permits: Vec<RunnerPerm>,
}

impl TorchPermsCtx {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            permits: Vec::new(),
        }
    }
}

impl Named for TorchPermsCtx {
    fn name(&self) -> &'static str {
        self.name
    }
}

impl PermitsCtx for TorchPermsCtx {
    fn add_permit(&mut self, permit: RunnerPerm) {
        self.permits.push(permit);
    }

    fn drop_permits(&mut self, drop_if: DropIf) {
        let acquired_permits: NumPermits = self.acquired_permits();
        debug!(
            "TorchRunCtx: acquired_permits={acquired_permits} ctx.name={} drop_if={drop_if:?}",
            self.name
        );

        match drop_if {
            DropIf::Always => {
                debug!("TorchRunCtx: ctx.name={} dropping permits", self.name);
                self.permits.clear();
            }
            DropIf::IfNotMax => {
                if acquired_permits < MAX_PERMITS {
                    debug!("TorchRunCtx: ctx.name={} dropping permits", self.name);
                    self.permits.clear();
                }
            }
        }
    }

    fn acquired_permits(&self) -> NumPermits {
        self.permits.iter().map(|p| p.num_permits()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::output_builders::SingleDimTensorOutputBuilder;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_shared::traits::{Data, MonoMetrics, Select};
    use crate::output_builders::BuildAs;
    use async_trait::async_trait;
    use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use caching_runner::common::{DropIf, OutId, PermitsCtx, RunCtx};
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_shared::traits::Adjustable;
    use caching_runner::permits::{MAX_PERMITS, NumPermits, PermitStep, RunnerPerm};
    use caching_runner::test_utils::parse_numeric_part;
    use caching_runner_test::FailUntil::MinPermits;
    use caching_runner_test::{ClcOOMRunner, FailUntil, OOMRunnerConfig, run_mul_task_invocations};
    use dashmap::DashMap;
    use ambassador::Delegate;
    use caching_runner::common::ambassador_impl_PermitsCtx;
    use crate::TorchPermsCtx;
    use futures_util::future::{BoxFuture, Shared};
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::{HashMap, HashSet};
    use std::mem;
    use std::sync::Arc;
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use tch::{Device, IndexOp, Kind, Scalar, Tensor};
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::cuda_utils::cuda_mem;
    use thoosa_common::dims::Dims;
    use thoosa_common::utils::lin_for_size;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::traits::{CachableIdTr, Named, SimpleId, WithId};
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::{MyError, ThoosaError};
    use tracing_subscriber::fmt::format::FmtSpan;
    use tracing_subscriber::{EnvFilter, fmt};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[derive(Debug, Delegate)]
    #[delegate(PermitsCtx, target = "perms_ctx")]
    pub struct TorchRunCtx {
        device_idx: usize,
        loaded_tensors_cached: Vec<SimpleId>,
        multed_tensors_cached: Vec<SimpleId>,
        cache_area: u64,
        name: &'static str,
        perms_ctx: TorchPermsCtx,
    }

    impl TorchRunCtx {
        pub fn new(device_idx: usize, cache_area: u64, name: &'static str) -> Self {
            Self {
                device_idx,
                loaded_tensors_cached: vec![],
                multed_tensors_cached: vec![],
                cache_area,
                name,
                perms_ctx: TorchPermsCtx::new(name),
            }
        }
    }

    impl Adjustable for TorchRunCtx {}

    impl RunCtx for TorchRunCtx {
        fn device_idx(&self) -> usize {
            self.device_idx
        }

        fn purge_gpu_caches(ctx: Arc<RwLock<Self>>) -> BoxFuture<'static, Result<(), ThoosaError>>
        where
            Self: Sized,
        {
            Box::pin(async move {
                let (device_idx, ctx_name) = {
                    let guard = ctx.read();
                    (guard.device_idx, guard.name)
                };
                assert_ne!(device_idx, 0);

                debug!("TorchRunCtx::purge_gpu_caches: 0 ctx_name={ctx_name} device_idx={device_idx} ");

                let (cache_area, loaded_tensors_cached, mul_output_cached) = {
                    let mut guard = ctx.write();

                    (
                        guard.cache_area,
                        mem::replace(&mut guard.loaded_tensors_cached, Vec::new()),
                        mem::replace(&mut guard.multed_tensors_cached, Vec::new()),
                    )
                };
                debug!(
                    "purge_gpu_caches: 1 ctx_name={ctx_name}device_idx={device_idx} loaded_tensors_cached={loaded_tensors_cached:?} mul_output_cached={mul_output_cached:?}"
                );

                tokio::try_join!(
                    TestTensorLoader::purge_gpu_cache_area(
                        loaded_tensors_cached,
                        cache_area,
                        device_idx
                    ),
                    ClcTask::purge_gpu_cache_area(mul_output_cached, cache_area, device_idx)
                )?;

                debug!("purge_gpu_caches: 2 ctx_name={ctx_name} device_idx={device_idx}");

                Ok(())
            })
        }

        fn cache_area(&self) -> u64 {
            self.cache_area
        }
    }

    impl Named for TorchRunCtx {
        fn name(&self) -> &'static str {
            self.name
        }
    }

    //////////////////LOADER
    #[derive(Debug, Clone, Eq, PartialEq, Hash)]
    pub struct LoaderInputId(&'static str);

    impl Named for LoaderInputId {
        fn name(&self) -> &'static str {
            "LoaderInputId"
        }
    }

    #[derive(Debug, Clone)]
    pub struct LoaderInput {
        id: LoaderInputId,
    }
    impl Input for LoaderInput {}

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
        async fn copy_to_device(&self, _dst_device: usize) -> Self {
            self.clone()
        }
    }

    #[async_trait]
    impl Data for LoaderInput {
        type Metrics = ();

        fn size(&self) -> Self::Metrics {
            ()
        }

        fn device(&self) -> usize {
            0
        }
    }

    fn new_loader_input_cache(
        value: impl Into<LoaderInput>,
        device_idx: usize,
        cache_area: u64,
    ) -> Arc<DevicesWrap<LoaderInput>> {
        Arc::new(DevicesWrap::wrap(
            value.into(),
            cache_area,
        ))
    }

    #[derive(Debug, Eq, PartialEq, Hash, Clone)]
    #[allow(dead_code)]
    struct LoadedTensorId(&'static str);

    impl Named for LoadedTensorId {
        fn name(&self) -> &'static str {
            "LoadedTensorId"
        }
    }

#[derive(Debug, Clone)]
struct LoadedTensor;
impl Dims for LoadedTensor {
    type Metrics = MonoMetrics<1, 2>;

}
impl LoadedTensor {}

// impl Adjustable for ThoTensor2<LoadedTensor, SimpleId> {}

#[derive(Debug, Clone, Default)]
struct  LoadedRange {
    range:std::ops::Range<MonoMetrics<1, 2>>
}
impl Adjustable for LoadedRange {}

impl Select<LoadedRange> for ThoTensor<LoadedTensor, SimpleId> {
    type Output = ThoTensor<LoadedTensor, SimpleId>;

    fn subview(&self, selector: LoadedRange) -> Result<Self::Output, ThoosaError> {
        let ranges = MonoMetrics::<1,2>::map_to_ranges(&selector.range.start, &selector.range.end)?;//TODO unwrap

        let sub_tensor = self.t_ref().i((ranges[0].clone(), ranges[1].clone()));

        let new_id = SimpleId::from(format!("{}:{ranges:?}", self.id().0));

        debug!("Select for ThoTensor2<LoadedTensor, SimpleId>: ranges={ranges:?} sub_tensor={sub_tensor:?}");

        Ok(ThoTensor::new(new_id, sub_tensor))
    }
}


    #[derive(Debug, cached_derive::Cached)]
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
            SimpleId::from(format!("loaded_{}", input_id.0))
        }
    }

    impl Stage for TestTensorLoader {
        type In<'a> = LoaderInput;
        type Out = ThoTensor<LoadedTensor, SimpleId>;
        type Ctx = TorchRunCtx;
    }

    // #[async_trait]
    impl AsyncStage for TestTensorLoader {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
            let k = parse_numeric_part(WithId::id(input.as_ref()).0).unwrap_or(1.0);

            let (device_idx, cache_area) = {
                let guard = ctx.read();
                (guard.device_idx(), guard.cache_area())
            };

            let num_devices = GlobalCtx::global().device_count();

            let tensor = ThoTensor::new(
                self.out_id(WithId::id(input.as_ref())),
                lin_for_size(
                    k,
                    (&self.size).into(),
                    self.kind,
                    &ThoDevice::from(device_idx),
                ),
            );

            Ok(Arc::new(DevicesWrap::wrap(
                tensor,
                cache_area,
            )))
        }

        fn on_cached<'a>(
            &self,
            input: &impl WithId<ID = <<Self as Stage>::In<'a> as WithId>::ID>,
            ctx: &Arc<RwLock<Self::Ctx>>,
        ) where
            Self: 'a,
        {
            ctx.write()
                .loaded_tensors_cached
                .push(self.out_id(input.id()));
        }
    }

    // //////////////////////////////
    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct MulK;

    impl Dims for MulK {
        type Metrics = MonoMetrics<1, 2>;
    }

    #[derive(Debug, cached_derive::Cached)]
    #[cache_name = "MulTaskOutputId"]
    struct ClcTask {
        id: SimpleId,
        k: ThoTensor<MulK, SimpleId>,
        fail_until: Vec<FailUntil>,
        use_matmul: bool,
    }

    impl ClcTask {
        pub fn new(
            id: SimpleId,
            k: impl Into<Scalar>,
            fail_until: Vec<FailUntil>,
            dim: i64,
            kind: Kind,
            device: Device,
            use_matmul: bool,
        ) -> Self {
            Self {
                id,
                k: ThoTensor::new(
                    SimpleId::from("mul_k"),
                    Tensor::ones([dim], (kind, device))
                        .f_mul_scalar_(k)
                        .unwrap(),
                ),
                fail_until,
                use_matmul,
            }
        }
    }

    impl Adjustable for ClcTask {}



    impl OutId for ClcTask {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            SimpleId::from(format!("mult_{}_inp_{}", self.id, input_id))
        }
    }

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct ClcTaskOutput;
    impl Dims for ClcTaskOutput {
        type Metrics = MonoMetrics<1, 2>;
    }

    impl LoadedTensor {
        fn clc_hadamard(
            id: SimpleId,
            loaded: &ThoTensor<LoadedTensor, SimpleId>,
            k: &ThoTensor<MulK, SimpleId>,
        ) -> Result<ThoTensor<ClcTaskOutput, SimpleId>, ThoosaError> {
            loaded.run_inner_with_params(
                id,
                &|t: &Tensor, params: &[&Tensor]| Ok(t.multiply(&params[0])),
                &[k],
            )
        }

        fn clc_sum(
            id: SimpleId,
            loaded: &ThoTensor<LoadedTensor, SimpleId>,
            k: &ThoTensor<MulK, SimpleId>,
        ) -> Result<ThoTensor<ClcTaskOutput, SimpleId>, ThoosaError> {
            debug!(
                "LoadedTensor::clc_sum: loaded.t.size={:?} loaded=\n{}",
                loaded.size(),
                loaded
            );

            loaded.run_inner_with_params(
                id,
                &|t: &Tensor, params: &[&Tensor]| {
                    Ok(t.f_multiply(&params[0])?
                        .f_sum_dim_intlist(0, false, t.kind())?
                        .view((-1, 1)))
                },
                &[k],
            )
        }
    }

    impl Stage for ClcTask {
        type In<'a> = ThoTensor<LoadedTensor, SimpleId>;
        type Out = ThoTensor<ClcTaskOutput, SimpleId>;
        type Ctx = TorchRunCtx;
    }

    impl SyncStage for ClcTask {
        fn do_run<'a>(
            self: Arc<Self>,
            input: &Self::In<'a>,
            ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Self::Out, ThoosaError>
        where
            Self: Sized,
        {
            // tokio::time::sleep(tokio::time::Duration::from_millis(random_range(0..10))).await;
            debug!("MulTask::run: loaded.id={:?}", input.id());

            let deive_idx = ctx.read().device_idx;

            for fail_until in &self.fail_until {
                match fail_until {
                    FailUntil::MemLimit(mem_limit) => {
                        let _device_idx = deive_idx;
                        // debug!("MulTask::do_run: ctx={ctx:?} input.len={}", input.data.0.as_ref().unwrap().size()[0]);

                        if input.size()[0] > *mem_limit as i64 {
                            debug!(
                                "MulTask::do_run: MyError::OutOfMemory on MemLimit mem_limit={mem_limit}"
                            );
                            return Err(MyError::OutOfMemory.into());
                        }
                    }
                    FailUntil::MinPermits(min_permits) => {
                        let acquired_permits = ctx.read().acquired_permits();
                        debug!(
                            "MulTask::do_run: ctx={ctx:?} acquired_permits={acquired_permits} min_permits={min_permits}"
                        );

                        if acquired_permits < *min_permits {
                            debug!(
                                "MulTask::do_run: MyError::OutOfMemory on MinPermits min_permits={min_permits}"
                            );
                            return Err(MyError::OutOfMemory.into());
                        }
                    }

                    FailUntil::DumpedToCpu => {
                        let is_all_on_cpu =
                            GlobalCtx::global().is_all_on_cpu(ctx.read().device_idx());
                        debug!(
                            "MulTask::do_run: ctx.name={:?} device_idx={} is_all_on_cpu={is_all_on_cpu}",
                            ctx.read().name(),
                            ctx.read().device_idx()
                        );

                        if !is_all_on_cpu {
                            debug!(
                                "MulTask::do_run: MyError::OutOfMemory on DumpedToCpu is_all_on_cpu={is_all_on_cpu}"
                            );
                            return Err(MyError::OutOfMemory.into());
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

    // #[async_trait]
    impl AsyncStage for ClcTask {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
            let (device_idx, cache_area) = {
                let guard = ctx.read();
                (guard.device_idx(), guard.cache_area())
            };
            let num_devices = GlobalCtx::global().device_count();

            let stage = Arc::clone(&self);
            let ctx_for_run = Arc::clone(&ctx);

            let output = input
                .with_data_ref(device_idx, None, move |tensor: &ThoTensor<LoadedTensor, SimpleId>| {
                    Arc::clone(&stage).do_run(tensor, Arc::clone(&ctx_for_run))
                })
                .await?;

            Ok(Arc::new(DevicesWrap::wrap(
                output,
                cache_area,
            )))
        }

        fn on_cached<'a>(
            &self,
            input: &impl WithId<ID = <<Self as Stage>::In<'a> as WithId>::ID>,
            ctx: &Arc<RwLock<Self::Ctx>>,
        ) where
            Self: 'a,
        {
            ctx.write()
                .multed_tensors_cached
                .push(self.out_id(input.id()));
        }
    }

    //////////////////////////////

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct LoadAndClcTaskId(&'static str);

    #[derive(Debug)]
    struct LoadAndClcTask {
        #[allow(dead_code)]
        id: LoadAndClcTaskId,

        loader: Arc<TestTensorLoader>,
        clc_task: Arc<ClcTask>,
        clc_oom_runner: ClcOOMRunner<ClcTask, LoadedRange>,
    }

    impl Stage for LoadAndClcTask {
        type In<'a> = LoaderInput;
        type Out = ThoTensor<ClcTaskOutput, SimpleId>;
        type Ctx = TorchRunCtx;
    }

    // #[async_trait]
    impl AsyncStage for LoadAndClcTask {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
            let loaded = Arc::clone(&self.loader)
                .run_cached(Arc::clone(&input), Arc::clone(&ctx))
                .await?;

            let loaded_id = loaded.id().clone();
            let output_id = self.clc_task.out_id(&loaded_id.clone());

            debug!("LoadAndMulTask::run: loaded_id={loaded_id:?} output_id={output_id:?}");

            let loaded_id_name = "LoadedTensorId"; //loaded.id().id_name().clone();
            let _output_id_name = "MulTaskOutputId"; //loaded.id().id_name().clone();

            debug!(
                "LoadAndMulTask::run: loaded_id_name={loaded_id_name:?} output_id.name()={:?}",
                output_id.name()
            );

            let _set_loaded = [&loaded_id as &dyn CachableIdTr]
                .into_iter()
                .collect::<HashSet<&dyn CachableIdTr>>();
            let _set_mul = [&output_id as &dyn CachableIdTr]
                .into_iter()
                .collect::<HashSet<&dyn CachableIdTr>>();

            let required_ids = HashMap::from([
                // (loaded_id_name, &set_loaded),
                // (output_id.name(), &set_mul),
            ]);

            let result = self
                .clc_oom_runner
                .run(&self.clc_task, loaded, Arc::clone(&ctx), &required_ids)
                .await?;

            let (device_idx, cache_area) = {
                let guard = ctx.read();
                (guard.device_idx(), guard.cache_area())
            };
            let num_devices = GlobalCtx::global().device_count();

            Ok(Arc::new(DevicesWrap::wrap(
                result,
                cache_area,
            )))
        }
    }

    #[async_trait]
    impl Cached for LoadAndClcTask {
        async fn cache() -> &'static DashMap<
            <<Self as Stage>::Out as WithId>::ID,
            Shared<BoxFuture<'static, Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>>>,
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
            oom_runner: ClcOOMRunner<ClcTask, LoadedRange>,
            kind: Kind,
            device: Device,
            use_matmul: bool,
        ) -> Self {
            let dim_1 = dim.dim_1();
            Self {
                id,
                loader: Arc::new(TestTensorLoader::new(dim, kind)),
                clc_task: Arc::new(ClcTask::new(
                    SimpleId::from("mul_task_0"),
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

    #[tokio::test]
    async fn test_load_tensor_cuda_to_cuda() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let _ = GlobalCtx::init_devices(2);
        let loader = Arc::new(TestTensorLoader::new([10, 3].into(), Float));

        do_test_loaded(1, 1, &loader).await
    }

    #[tokio::test]
    async fn test_load_tensor_cpu_to_cuda() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let _ = GlobalCtx::init_devices(2);
        let loader = Arc::new(TestTensorLoader::new([10, 3].into(), Float));

        do_test_loaded(0, 1, &loader).await
    }

    async fn do_test_loaded(
        loader_ctx_device: usize,
        load_to_device: usize,
        loader: &Arc<TestTensorLoader>,
    ) -> Result<(), ThoosaError> {
        let ctx_device_0 = Arc::new(RwLock::new(TorchRunCtx::new(
            loader_ctx_device,
            0,
            "device_1_ctx",
        )));

        let (ctx_device_idx, cache_area) = {
            let guard = ctx_device_0.read();
            (guard.device_idx(), guard.cache_area())
        };
        let loader_input = new_loader_input_cache("resource_1", ctx_device_idx, cache_area);

        let loaded_tensor_0_0 = Arc::clone(&loader)
            .run_cached(
                Arc::clone(&loader_input),
                Arc::clone(&ctx_device_0),
            )
            .await?;

        debug!("loaded_tensor_0_0={loaded_tensor_0_0:?}");

            let tensor_data_0_0 = loaded_tensor_0_0
            .with_data_ref(load_to_device, None, |tensor_data_0_0: &ThoTensor<LoadedTensor, SimpleId>| {
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
                Arc::clone(&loader_input),
                Arc::clone(&ctx_device_0),
            )
            .await?;

        loaded_tensor_0_0
            .with_data_ref(load_to_device, None, |tensor_data_0_1: &ThoTensor<LoadedTensor, SimpleId>| {
                // let tensor_data_0_1 = loaded_tensor_0_1.load_from_any_device(load_to_device, None).await?;

                assert_eq!(&tensor_data_0_0, tensor_data_0_1.t_ref());
                Ok(())
            })
            .await?;
        // let tensor_data_0_0 = loaded_tensor_0_0.load_from_any_device(load_to_device, None).await?;

        Ok(())
    }

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_no_oom_tensor() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;
        const MUL_K: f64 = 2.;

        let size = [3, 10];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::Permits { max: 20, step: 10 },
                Box::new(move |id| {
                    Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(
                        id,
                        1,
                        &size,
                        (Float, Device::Cuda(0)),
                        BuildAs::Append,
                    ))
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
            |_device_idx, _cache_area, _name| TorchRunCtx::new(1, 0, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
            .await?;

        Ok(())
    }

    fn mul_check(
        mul_k: f64,
        size: MonoMetrics<1, 2>,
        res: &str,
        result: &ThoTensor<ClcTaskOutput, SimpleId>,
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
        result: &ThoTensor<ClcTaskOutput, SimpleId>,
        rtol: f64,
        atol: f64,
        exp_f: F,
    ) where
        F: Fn(f64, &Tensor) -> Tensor,
    {
        let self_k = parse_numeric_part(res).unwrap_or(1.0);

        let expected = exp_f(self_k * mul_k, &result.t_ref());

        debug!(
            "mul_check: res={res}  result_t.size()={:?} \n\n",
            result.size()
        );

        let ok = expected.allclose(&result.t_ref(), rtol, atol, false);
        // if !ok {
        debug!(
            "mul_check: res={res} expected=\n{expected} result_t=\n{} \n\n",
            result.t_ref()
        );
        // }
        assert!(ok);
    }

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_till_min_permits_tensor() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;
        const MUL_K: f64 = 2.;

        let size = [3, 10];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::MinPermits(20)],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::Permits { max: 20, step: 10 },
                Box::new(move |id| {
                    Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(
                        id,
                        1,
                        &size,
                        (Float, Device::Cuda(0)),
                        BuildAs::Append,
                    ))
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
            |_device_idx, _cache_area, _name| TorchRunCtx::new(1, 0, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
        .await?;

        Ok(())
    }

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_compete_for_permits_tensor() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;

        const MUL_K: f64 = 2.;
        let size = [3, 10];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::MinPermits(MAX_PERMITS)],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::Permits {
                    max: MAX_PERMITS,
                    step: *PermitStep::TEN,
                },
                Box::new(move |id| {
                    Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(
                        id,
                        1,
                        &size,
                        (Float, Device::Cuda(0)),
                        BuildAs::Append,
                    ))
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
            |_device_idx, _cache_area, _name| TorchRunCtx::new(1, 0, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
        .await?;

        Ok::<_, ThoosaError>(())
    }

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_splits_tensor() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let _global_ctx = GlobalCtx::init_devices(2);

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
                    Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(
                        id,
                        1,
                        &size,
                        (Float, Device::Cuda(0)),
                        BuildAs::Append,
                    ))
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
            |_device_idx, _cache_area, _name| TorchRunCtx::new(1, 0, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
        .await?;

        Ok::<_, ThoosaError>(())
    }

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_permits_and_splits_tensor() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;
        const MUL_K: f64 = 2.;
        let size = [3, 100];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::MinPermits(MAX_PERMITS), FailUntil::MemLimit(12)],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::PermitsAndSplits {
                    max: MAX_PERMITS,
                    step: *PermitStep::TEN,
                },
                Box::new(move |id| {
                    Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(
                        id,
                        1,
                        &size,
                        (Float, Device::Cuda(0)),
                        BuildAs::Append,
                    ))
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
            |_device_idx, _cache_area, _name| TorchRunCtx::new(1, 0, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
        .await?;

        Ok(())
    }

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn heavy_test_load_and_mul_oom_dump_to_cpu() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // init_tracing();
        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;

        const MUL_K: f64 = 2.;
        let size = [3, 100];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![MinPermits(MAX_PERMITS), FailUntil::DumpedToCpu],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::DumpedToCpu {},
                Box::new(move |id| {
                    Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(
                        id,
                        1,
                        &[3, 100],
                        (Float, Device::Cuda(0)),
                        BuildAs::Append,
                    ))
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
            |_device_idx, _cache_area, name| TorchRunCtx::new(1, 0, name),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        )
        .await?;

        Ok::<_, ThoosaError>(())
    }

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn heavy_test_purge_gpu_with_lock() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // init_tracing();
        let _global_ctx = GlobalCtx::init_devices(2);

        const MUL_K: f64 = 2.;
        let size = [100000, 3];

        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![],
            size.clone().into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::DumpedToCpu {},
                Box::new(move |id| {
                    Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(
                        id,
                        1,
                        &size,
                        (Float, Device::Cuda(0)),
                        BuildAs::Append,
                    ))
                }),
            ),
            Float,
            Device::Cuda(0),
            false,
        ));

        let ctx_device = Arc::new(RwLock::new(TorchRunCtx::new(1, 0, "test")));

        for res in (0..10).map(|i| &*format!("resource_{i}").leak()) {
            let (device_idx, cache_area) = {
                let guard = ctx_device.read();
                (guard.device_idx(), guard.cache_area())
            };
            Arc::clone(&load_and_mut)
                .run_cached(
                    new_loader_input_cache(res, device_idx, cache_area),
                    Arc::clone(&ctx_device),
                )
                .await?;
        }

        for i in 0..10 {
            let cached_loaded = <TestTensorLoader as Cached>::cache()
                .await
                .get(&SimpleId::from(format!("loaded_resource_{i}")));
            let reff = cached_loaded.expect(&format!("no entry for loaded_resource_{i}"));
            debug!(
                "i={i} 0 cached_loaded: key={:?} val={:?}",
                reff.key(),
                reff.value().clone().await?
            );

            assert!(!reff.value().clone().await?.is_all_on_cpu().await);

            let cached_multed = <ClcTask as Cached>::cache()
                .await
                .get(&SimpleId::from(format!(
                    "mult_mul_task_0_inp_loaded_resource_{i}"
                )));
            let reff = cached_multed.expect(&format!("no entry for loaded_resource_{i}"));
            debug!(
                "i={i} 0 cached_multed: key={:?} val={:?}",
                reff.key(),
                reff.value().clone().await?
            );

            assert!(!reff.value().clone().await?.is_all_on_cpu().await);
        }

        let mut loaded_tensors_ids = HashSet::new();
        let loaded_tensor_3 = SimpleId::from("loaded_resource_3");
        loaded_tensors_ids.insert(&loaded_tensor_3 as &dyn CachableIdTr);

        let mut multed_tensors_ids = HashSet::new();
        let mult_mul_task_0_inp_loaded_resource_3 =
            SimpleId::from("mult_mul_task_0_inp_loaded_resource_3");
        multed_tensors_ids.insert(&mult_mul_task_0_inp_loaded_resource_3 as &dyn CachableIdTr);

        let mut required_ids = HashMap::new();
        required_ids.insert("LoadedTensorId", &loaded_tensors_ids);
        required_ids.insert("MulTaskOutputId", &multed_tensors_ids);

        _global_ctx.dump_all_to_cpu(&required_ids, 1).await?;

        for i in 0..10 {
            let cached_loaded = <TestTensorLoader as Cached>::cache()
                .await
                .get(&SimpleId::from(format!("loaded_resource_{i}")));
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
                .get(&SimpleId::from(format!(
                    "mult_mul_task_0_inp_loaded_resource_{i}"
                )));
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

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn heavy_test_real_oom_dump_and_split() -> Result<(), ThoosaError> {
        debug!("test_real_oom_dump_and_split");
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // init_tracing();
        let _global_ctx = GlobalCtx::init_devices(2);

        let (free_mem, _total_mem) = cuda_mem();
        let init_type = Kind::Int64;

        let total_size = (free_mem as f64 * 0.7) as i64 / init_type.elt_size_in_bytes() as i64; //123456; //25000; //123456; //
        let dim_1 = 3i64;
        let input_n = total_size / dim_1;

        debug!("test_real_oom_dump_and_split: input_n={input_n} dim_1={dim_1}");

        const MUL_K: i64 = 1;
        let size = [dim_1, input_n,];

        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![],
            size.clone().into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::DumpedToCpu {},
                Box::new(move |id| {
                    Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(
                        id,
                        1,
                        &[dim_1, 1],
                        (init_type, Device::Cuda(0)),
                        BuildAs::Sum,
                    ))
                }),
            ),
            Kind::Int64,
            Device::Cuda(0),
            true,
        ));

        let ctx_device = Arc::new(RwLock::new(TorchRunCtx::new(1, 0, "test")));

        let (device_idx, cache_area) = {
            let guard = ctx_device.read();
            (guard.device_idx(), guard.cache_area())
        };

        let multed = Arc::clone(&load_and_mut)
            .run_cached(
                new_loader_input_cache("resource_1", device_idx, cache_area),
                Arc::clone(&ctx_device),
            )
            .await?;

        debug!("multed={multed:?}");

        multed
            .with_data_ref(1, None, |r: &ThoTensor<ClcTaskOutput, SimpleId>| {
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

    #[allow(dead_code)]
    fn init_tracing() {
        /*let filter = EnvFilter::from_default_env()
            // Tokio’s internal resource events:
            .add_directive("runtime::resource::state_update=trace".parse().unwrap())
            // Optional: see more runtime resource logs
            .add_directive("runtime::resource=trace".parse().unwrap())
            // Your own app logs:
            .add_directive("tokio_rwlock_tracing=trace".parse().unwrap())
            .add_directive("trace".parse().unwrap());

        fmt().with_env_filter(filter).compact().init();*/

        let filter = EnvFilter::from_default_env()
            // .add_directive("runtime::resource::state_update=trace".parse().unwrap())
            // .add_directive("runtime::resource=trace".parse().unwrap())
            .add_directive("tokio::sync::rwlock=trace".parse().unwrap())
            .add_directive("debug".parse().unwrap());

        fmt()
            .with_env_filter(filter)
            .with_line_number(true)
            .with_span_events(FmtSpan::FULL)
            // .event_format(InlineSpanNames) - doesn't really add more info. with #[instrument] put on methods span names apper in the log
            .init();
    }
}
