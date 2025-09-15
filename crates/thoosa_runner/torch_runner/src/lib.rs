#![feature(lazy_get)]

extern crate core;

mod cuda_utils;
mod copy_ctx;
mod output_builders;
pub mod data;
#[cfg(test)]
mod model_test;

#[allow(unused_imports)]
use caching_runner::adjusters::splits::OutputBuilder;
#[allow(unused_imports)]
use caching_runner::common::{CopyToDevice, Data, Input, MyError, Output, WithId};
use std::fmt::Debug;
use std::ops::Deref;
use tch::{Device, Kind, TchError, Tensor};
// test-only imports live under cfg(test)
use data::ThoTensor;
use thoosa_common::dims::Dims;
use thoosa_shared::traits::{Named, SimpleId};

#[derive(Debug)]
pub enum MyTchError {
    MyError(MyError),
    TchError(TchError)
}

impl From<TchError> for MyTchError {
    fn from(value: TchError) -> Self {
        MyTchError::TchError(value)
    }
}

impl Into<MyError> for MyTchError{
    fn into(self) -> MyError {
        MyError::SomethingElse(format!("error: {self:?}"))
    }
}

struct ThoDevice(Device);

impl From<usize> for ThoDevice {
    fn from(value: usize) -> Self {
        if value == 0 {
            ThoDevice(Device::Cpu)
        } else {
            ThoDevice(Device::Cuda(value - 1))
        }
    }
}

impl ThoDevice {
    fn ordinal(&self)-> usize {
        match &self.0 {
            Device::Cpu => 0,
            Device::Cuda(v) => v+1,
            _=> {
                unimplemented!()
            }
        }
    }
}

impl Deref for ThoDevice {
    type Target = Device;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}


pub struct ThoTensorView<'a, D:Dims, Id: Named + Clone> {
    t: Tensor,
    _owner: &'a ThoTensor<D, Id>,
}
unsafe impl<'a, D:Dims, Id: Named + Clone> Sync for ThoTensorView<'a, D, Id> {}
unsafe impl<'a, D:Dims, Id: Named + Clone> Send for ThoTensorView<'a, D, Id> {}
impl<'a, D:Dims, Id: Named + Clone> ThoTensorView<'a, D, Id> {

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

#[cfg(test)]
mod tests {
    use super::{output_builders::SingleDimTensorOutputBuilder, ThoDevice};
    use async_trait::async_trait;
    use caching_runner::common::cachable::{Cached, DataCacheWrapper, Runner};
    use caching_runner::global_ctx::GlobalCtx;
    use caching_runner::oom_handler::Adjustable;
    use caching_runner::permits::{NumPermits, PermitStep, RunnerPerm, MAX_PERMITS};
    use caching_runner_test::FailUntil::MinPermits;
    use caching_runner_test::{run_mul_task_invocations, ClcOOMRunner, FailUntil, OOMRunnerConfig};
    use caching_runner::common::{Data, DropIf, Input, MyError, OutId, RunCtx, WithId};
    use dashmap::DashMap;
    use futures_util::future::{BoxFuture, Shared};
    use parking_lot::RwLock;
    use std::cmp::{min, Ordering};
    use std::collections::{HashMap, HashSet};
    use std::mem;
    use std::ops::{Add, AddAssign, DivAssign};
    use std::sync::Arc;
    use tch::Kind::Float;
    use tch::{Device, IndexOp, Kind, Scalar, Tensor};
    use caching_runner::test_utils::parse_numeric_part;
    use tracing_subscriber::{fmt, EnvFilter};
    use tracing_subscriber::fmt::format::FmtSpan;
    use caching_runner::common::runnable::Stage;
    use caching_runner::panic_to_err;
    use thoosa_common::dims::Dims;
    use thoosa_shared::traits::{CachableIdTr, Named, SimpleId};
    use crate::cuda_utils::cuda_mem;
    use crate::data::ThoTensor;
    use crate::output_builders::BuildAs;

    #[derive(Debug)]
    pub struct TorchRunCtx {
        device_idx: usize,
        permits: Vec<RunnerPerm>,
        loaded_tensors_cached: Vec<SimpleId>,
        multed_tensors_cached: Vec<SimpleId>,
        cache_area: u64,
        name: &'static str
    }

    impl TorchRunCtx {
        pub fn new(device_idx: usize, cache_area: u64, name: &'static str) -> Self {
            Self {
                device_idx,
                permits: vec![],
                loaded_tensors_cached: vec![],
                multed_tensors_cached: vec![],
                cache_area,
                name,
            }
        }
    }

    impl Adjustable for TorchRunCtx {

    }

    #[async_trait]
    impl RunCtx for TorchRunCtx {
        fn device_idx(&self) -> usize {
            self.device_idx
        }

        fn add_permit(&mut self, permit: RunnerPerm) {
            self.permits.push(permit);
        }

        fn drop_permits(&mut self, drop_if: DropIf) {
            //once we've got all permits try all adjusters - others weill wait - no danger of blocking.
            let acquired_permits = self.acquired_permits();
            println!("TorchRunCtx: acquired_permits={acquired_permits} ctx.name={} drop_if={drop_if:?}", self.name);

            match drop_if{
                DropIf::Always => {
                    println!("TorchRunCtx: ctx.name={} dropping permits", self.name);
                    self.permits.clear();
                }
                DropIf::IfNotMax => {
                    if acquired_permits < MAX_PERMITS {//TODO param
                        println!("TorchRunCtx: ctx.name={} dropping permits", self.name);
                        self.permits.clear();
                    }

                }
            }
        }

        fn acquired_permits(&self) -> NumPermits {
            self.permits.iter().map(|p| p.num_permits()).sum()
        }

        async fn purge_gpu_caches(ctx: Arc<RwLock<Self>>) -> Result<(), MyError>
        where
            Self: Sized
        {
            // let _purge_lock = GlobalCtx::global().purge_lock().await;

            let (device_idx, ctx_name) = {
                let guard = ctx.read();
                (guard.device_idx, guard.name)
            };
            assert_ne!(device_idx, 0);//shouldn't purge CPU

            println!("TorchRunCtx::purge_gpu_caches: 0 ctx_name={ctx_name} device_idx={device_idx} ");

            let (cache_area, loaded_tensors_cached, mul_output_cached) = {
                let mut guard = ctx.write();

                (
                    guard.cache_area,
                    mem::replace(&mut guard.loaded_tensors_cached, Vec::new()),
                    mem::replace(&mut guard.multed_tensors_cached, Vec::new()),
                )
            };
            println!("purge_gpu_caches: 1 ctx_name={ctx_name}device_idx={device_idx} loaded_tensors_cached={loaded_tensors_cached:?} mul_output_cached={mul_output_cached:?}");

            tokio::try_join!(
                TestTensorLoader::purge_gpu_cache_area(loaded_tensors_cached, cache_area, device_idx),
                ClcTask::purge_gpu_cache_area(mul_output_cached, cache_area, device_idx)
            )?;

            println!("purge_gpu_caches: 2 ctx_name={ctx_name} device_idx={device_idx}");

            Ok(())
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
                id: LoaderInputId(value)
            }
        }
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
    impl Dims for LoadedTensor {}
    impl LoadedTensor{

    }

    impl Adjustable for ThoTensor<LoadedTensor, SimpleId> {

    }

    #[derive(Debug, Clone, Default)]
    struct LoadedTensorMetrics{
        dim_0:i64,
        dim_1:i64
    }

    impl Add<& LoadedTensorMetrics> for &LoadedTensorMetrics {
        type Output = LoadedTensorMetrics;

        fn add(self, rhs: &LoadedTensorMetrics) -> Self::Output {
            LoadedTensorMetrics {
                dim_0: self.dim_0 + rhs.dim_0,
                dim_1: self.dim_1,
            }
        }
    }

    impl DivAssign<i64> for LoadedTensorMetrics {
        fn div_assign(&mut self, rhs: i64) {
            self.dim_0 /= rhs;
        }
    }

    impl PartialEq<i64> for LoadedTensorMetrics {
        fn eq(&self, other: &i64) -> bool {
            self.dim_0.eq(other)
        }
    }

    impl PartialOrd<i64> for LoadedTensorMetrics {
        fn partial_cmp(&self, other: &i64) -> Option<Ordering> {
            self.dim_0.partial_cmp( other)
        }
    }

    impl PartialEq<LoadedTensorMetrics> for LoadedTensorMetrics {
        fn eq(&self, other: &LoadedTensorMetrics) -> bool {
            self.dim_0.eq(&other.dim_0)
        }
    }

    impl PartialOrd<LoadedTensorMetrics> for LoadedTensorMetrics {
        fn partial_cmp(&self, other: &LoadedTensorMetrics) -> Option<Ordering> {
            self.dim_0.partial_cmp(&other.dim_0)
        }
    }

    impl <'a> AddAssign<&'a LoadedTensorMetrics> for LoadedTensorMetrics {
        fn add_assign(&mut self, rhs: &'a LoadedTensorMetrics) {
            self.dim_0+=rhs.dim_0;
            println!("AddAssign for LoadedTensorMetrics::add_assign rhs.dim_0={} self.dim_0={}", rhs.dim_0, self.dim_0);
            // self.dim_1+=rhs.dim_1;TODO dimension
        }
    }

    impl TryFrom<Vec<i64>> for LoadedTensorMetrics {
        type Error = MyError;

        fn try_from(value: Vec<i64>) -> Result<Self, Self::Error> {
            let &[dim_0, dim_1, ..] = value.as_slice() else {
                return Err(MyError::SomethingElse(format!("can't make LoadedTensorMetrics from {value:?}")));
            };

            Ok(Self {
                dim_0,
                dim_1
            })
        }
    }

    impl From<&LoadedTensorMetrics> for [i64;2] {
        fn from(value: &LoadedTensorMetrics) -> Self {
            [value.dim_0, value.dim_1]
        }
    }


    impl From<[i64;2]> for LoadedTensorMetrics {
        fn from([dim_0, dim_1]: [i64; 2]) -> Self {
            Self {
                dim_0,
                dim_1
            }
        }
    }


    #[async_trait]
    impl Data for ThoTensor<LoadedTensor, SimpleId> {
        type Metrics = LoadedTensorMetrics;

        type Sub = Self;

        type SubSelect = std::ops::Range<Self::Metrics>;

        fn size(&self) -> Self::Metrics {
            self.size().try_into().expect("can't get size")
        }

        fn subview(&self, range: &Self::SubSelect) -> Self
        where
            Self: Sized
        {
            let size:LoadedTensorMetrics = self.t().size().try_into().unwrap();


            let dim_0_range = range.start.dim_0..min(size.dim_0, range.end.dim_0);



            println!("Data for LoadedTensor::subview: dim_0_range={dim_0_range:?} size={size:?} range={range:?}");
            println!("Data for LoadedTensor::subview: t=\n{}", self.t());

            let sub = ThoTensor::new(self.id().clone(), self.t().i((dim_0_range.clone(), ..)));


            println!("Data for LoadedTensor::subview: sub.t=\n{}", self.t().i((dim_0_range, ..)));

            sub
        }

        fn track_used_in(&self, token: &'static str) {
            self._track_used_in(token);
        }

        fn device(&self) -> usize {
            self.device()
        }
    }

    #[derive(Debug, cached_derive::Cached)]
    #[cache_name = "LoadedTensorId"]
    struct TestTensorLoader {
        size:LoadedTensorMetrics,
        kind:Kind,
    }



    impl TestTensorLoader {
        pub fn new(size: LoadedTensorMetrics, kind: Kind) -> Self {
            Self { size, kind }
        }
    }

    impl OutId for TestTensorLoader {
        fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID {
            SimpleId::from(format!("loaded_{}", input_id.0))
        }

        fn on_cached(&self, input: &impl WithId<ID=<<Self as Stage>::In as WithId>::ID>, ctx: &Arc<RwLock<Self::Ctx>>) {
            ctx.write().loaded_tensors_cached.push(self.out_id(input.id()));
        }
    }

    pub fn lin_t(k:f64, size:[i64;2], kind:Kind, device:Device)->Tensor {

        let end = size.iter().fold(1, |a, v| a * *v);

        println!("lin_t: end={end}");

        let mut t = panic_to_err!(Tensor::f_arange(end, (kind, device))).map_err(|e| Into::<MyError>::into(e)).expect("f_arange failed").view(&size[..]);

        t = t.set_requires_grad(false);
        match kind {
            Kind::Uint8 | Kind::Int8 | Kind::Int16 | Kind::Int | Kind::Int64 => {
                t *= k as i32;
            }
            _=> {
                t *= k;
            }
        }


        println!("lin_t: t.size()={:?} t=\n{t}", t.size());

        t

    }

    #[async_trait]
    impl Stage for TestTensorLoader {
        type In = LoaderInput;
        type Out = ThoTensor<LoadedTensor, SimpleId>;
        type Ctx = TorchRunCtx;

        async fn run<I>(self: Arc<Self>, input: Arc<I>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, MyError>
        where
            I: WithId<ID=<<Self as Stage>::In as WithId>::ID> + Send + Sync + 'static
        {
            let k = parse_numeric_part(input.id().0).unwrap_or(1.0);

            Ok(ThoTensor::new(self.out_id(input.id()), lin_t(k, (&self.size).into(), self.kind, ThoDevice::from(ctx.read().device_idx).0)))
        }

    }

    

    // //////////////////////////////
    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct MulK;

    impl Dims for MulK {}

    #[derive(Debug, cached_derive::Cached)]
    #[cache_name = "MulTaskOutputId"]
    struct ClcTask {
        id: SimpleId,
        k: ThoTensor<MulK, SimpleId>,
        fail_until: Vec<FailUntil>,
        use_matmul: bool
    }


    impl ClcTask {
        pub fn new(id: SimpleId, k: impl Into<Scalar>, fail_until: Vec<FailUntil>, dim:i64, kind:Kind, device:Device, use_matmul: bool) -> Self {
            Self {
                id,
                k: ThoTensor::new(SimpleId::from("mul_k"), Tensor::ones([dim], (kind, device)).f_mul_scalar_(k).unwrap()),
                fail_until,
                use_matmul
            }
        }
    }


    impl Adjustable for ClcTask {}

    #[async_trait]
    impl Data for ThoTensor<ClcTaskOutput, SimpleId> {
        type Metrics = LoadedTensorMetrics;

        type Sub = Self;
        type SubSelect = std::ops::Range<Self::Metrics>;

        fn size(&self) -> Self::Metrics {
            self.size().try_into().expect("can't get size")
        }
        fn track_used_in(&self, _token: &'static str) {
            // no-op for tests
        }

        fn device(&self) -> usize {
            self.device()
        }
    }

    impl OutId for ClcTask {
        fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID {
            SimpleId::from(format!("mult_{}_inp_{}", self.id, input_id))
        }

        fn on_cached(&self, input: &impl WithId<ID=<<Self as Stage>::In as WithId>::ID>, ctx: &Arc<RwLock<Self::Ctx>>) {
            ctx.write().multed_tensors_cached.push(self.out_id(input.id()));
        }

    }

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct ClcTaskOutput;
    impl Dims for ClcTaskOutput {}

    impl LoadedTensor {
        fn clc_hadamard(id: SimpleId, loaded: &ThoTensor<LoadedTensor, SimpleId>, k: &ThoTensor<MulK, SimpleId>) -> Result<ThoTensor<ClcTaskOutput, SimpleId>, MyError> {

            loaded.run_inner_with_params(
                id,
                &|t:&Tensor, params:&[&Tensor]|{
                        Ok(t.multiply(&params[0]))
                },
                &[k]
            )
        }

        fn clc_sum(id: SimpleId, loaded: &ThoTensor<LoadedTensor, SimpleId>, k: &ThoTensor<MulK, SimpleId>) -> Result<ThoTensor<ClcTaskOutput, SimpleId>, MyError> {
            println!("LoadedTensor::clc_sum: loaded.t.size={:?} loaded=\n{}", loaded.size(), loaded);

            loaded.run_inner_with_params(
                id,
                &|t:&Tensor, params:&[&Tensor]|{
                    Ok(t
                        .f_multiply(&params[0])?
                        .f_sum_dim_intlist(0, false, t.kind())?
                        .view((-1, 1)))

                },
                &[k]
            )
        }

    }

    #[async_trait]
    impl Stage for ClcTask {
        type In = ThoTensor<LoadedTensor, SimpleId>;
        type Out = ThoTensor<ClcTaskOutput, SimpleId>;
        type Ctx = TorchRunCtx;


        fn do_run(self: Arc<Self>, input: &Self::In, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, MyError>
        where
            Self: Sized
        {
            // tokio::time::sleep(tokio::time::Duration::from_millis(random_range(0..10))).await;
            println!("MulTask::run: loaded.id={:?}", input.id());

            let deive_idx = ctx.read().device_idx;

            for fail_until in &self.fail_until {
                match fail_until {
                    FailUntil::MemLimit(mem_limit) => {
                        let _device_idx = deive_idx;
                        // println!("MulTask::do_run: ctx={ctx:?} input.len={}", input.data.0.as_ref().unwrap().size()[0]);

                        if input.size()[0] > *mem_limit as i64 {
                            println!("MulTask::do_run: MyError::OutOfMemory on MemLimit mem_limit={mem_limit}");
                            return Err(MyError::OutOfMemory);
                        }
                    }
                    FailUntil::MinPermits(min_permits) => {
                        let acquired_permits = ctx.read().acquired_permits();
                        println!("MulTask::do_run: ctx={ctx:?} acquired_permits={acquired_permits} min_permits={min_permits}");

                        if acquired_permits < *min_permits {
                            println!("MulTask::do_run: MyError::OutOfMemory on MinPermits min_permits={min_permits}");
                            return Err(MyError::OutOfMemory);
                        }
                    }

                    FailUntil::DumpedToCpu => {
                        let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(ctx.read().device_idx());
                        println!("MulTask::do_run: ctx.name={:?} device_idx={} is_all_on_cpu={is_all_on_cpu}", ctx.read().name(), ctx.read().device_idx());

                        if !is_all_on_cpu {
                            println!("MulTask::do_run: MyError::OutOfMemory on DumpedToCpu is_all_on_cpu={is_all_on_cpu}");
                            return Err(MyError::OutOfMemory);
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


    



    //////////////////////////////

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct LoadAndClcTaskId(&'static str);

    #[derive(Debug)]
    struct LoadAndClcTask {
        #[allow(dead_code)]
        id: LoadAndClcTaskId,

        loader: Arc<TestTensorLoader>,
        mul: Arc<ClcTask>,
        mul_oom_runner: ClcOOMRunner<ClcTask>,
    }

    #[async_trait]
    impl Stage for LoadAndClcTask {
        type In = LoaderInput;
        type Out = ThoTensor<ClcTaskOutput, SimpleId>;
        type Ctx = TorchRunCtx;

        async fn run<I>(self: Arc<Self>, input: Arc<I>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, MyError>
        where
            I: WithId<ID=<<Self as Stage>::In as WithId>::ID> + Send + Sync + 'static,

        {
            let loaded = Arc::clone(&self.loader)
                .run_cached(input, Arc::clone(&ctx))
                .await?;

            let loaded_id = loaded.id().clone();
            let output_id = self.mul.out_id(&loaded_id.clone());

            println!("LoadAndMulTask::run: loaded_id={loaded_id:?} output_id={output_id:?}");


            let loaded_id_name = "LoadedTensorId"; //loaded.id().id_name().clone();
            let _output_id_name = "MulTaskOutputId"; //loaded.id().id_name().clone();

            println!("LoadAndMulTask::run: loaded_id_name={loaded_id_name:?} output_id.name()={:?}", output_id.name());

            let _set_loaded = [&loaded_id as &dyn CachableIdTr].into_iter().collect::<HashSet<&dyn CachableIdTr>>();
            let _set_mul = [&output_id as &dyn CachableIdTr].into_iter().collect::<HashSet<&dyn CachableIdTr>>();

            let required_ids = HashMap::from([
                // (loaded_id_name, &set_loaded),
                // (output_id.name(), &set_mul),
            ]);

            async move {
                self.mul_oom_runner.run(&self.mul, loaded, ctx, &required_ids).await
            }.await
        }

    }

    #[async_trait]
    impl Cached for LoadAndClcTask {

        async fn cache() -> &'static DashMap<
            <<Self as Stage>::Out as WithId>::ID,
            Shared<BoxFuture<'static, Result<Arc<DataCacheWrapper<Self::Out>>, MyError>>>,
        >
        where
            Self: Sized
        {
            <ClcTask as Cached>::cache().await
        }

    }

    impl OutId for LoadAndClcTask {
        fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID {
            self.mul.out_id(&self.loader.out_id(input_id))
        }
    }

    impl LoadAndClcTask {
        pub fn new(
            id: LoadAndClcTaskId,
            k: impl Into<Scalar>,
            fail_until: Vec<FailUntil>,
            dim: LoadedTensorMetrics,
            oom_runner: ClcOOMRunner<ClcTask>,
            kind:Kind,
            device:Device,
            use_matmul: bool
        ) -> Self {
            let dim_1 = dim.dim_1;
            Self {
                id,
                loader: Arc::new(TestTensorLoader::new(dim, kind)),
                mul: Arc::new(ClcTask::new(SimpleId::from("mul_task_0"), k, fail_until, dim_1, kind, device, use_matmul)),
                mul_oom_runner: oom_runner,
            }
        }
    }
    // //////////////////////////////

    #[tokio::test]
    async fn test_load_tensor_cuda_to_cuda() ->Result<(), MyError>{
        unsafe { let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so"); }

        let _ = GlobalCtx::init_devices(2);
        let loader = Arc::new(TestTensorLoader::new([10, 3].into(), Float));


        do_test_loaded(1, 1, &loader).await
    }

    #[tokio::test]
    async fn heavy_test_load_tensor_cpu_to_cuda() ->Result<(), MyError>{
        unsafe { let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so"); }

        let _ = GlobalCtx::init_devices(2);
        let loader = Arc::new(TestTensorLoader::new([10, 3].into(), Float));


        do_test_loaded(0, 1, &loader).await
    }


    async fn do_test_loaded(loader_ctx_device: usize, load_to_device: usize, loader: &Arc<TestTensorLoader>) -> Result<(), MyError> {
        let ctx_device_0 = Arc::new(RwLock::new(TorchRunCtx::new(loader_ctx_device, 0, "device_1_ctx")));

        let loaded_tensor_0_0 = Arc::clone(&loader)
            .run_cached(
                Arc::new(LoaderInput {
                    id: LoaderInputId("resource_1"),
                }),
                Arc::clone(&ctx_device_0),
            )
            .await?;


        println!("loaded_tensor_0_0={loaded_tensor_0_0:?}");

        let tensor_data_0_0 = loaded_tensor_0_0.with_data(load_to_device, None, None,  |tensor_data_0_0|{
            println!("tensor_data={}", (*tensor_data_0_0).t());

            assert!(
                Tensor::from_slice(
                    &[
                        0., 1., 2.,
                        3., 4., 5.,
                        6., 7., 8.,
                        9., 10., 11.,
                        12., 13., 14.,
                        15., 16., 17.,
                        18., 19., 20.,
                        21., 22., 23.,
                        24., 25., 26.,
                        27., 28., 29.
                    ]
                ).totype(Float).to_device(Device::Cuda(0)).view((10, 3)).allclose(
                    &(*tensor_data_0_0).t(),
                    0.01, 0.01, false
                )
            );

            Ok(tensor_data_0_0.t().shallow_clone())
        }).await?;


        let _loaded_tensor_0_1 = Arc::clone(&loader)
            .run_cached(
                Arc::new(LoaderInput {
                    id: LoaderInputId("resource_1"),
                }),
                Arc::clone(&ctx_device_0),
            )
            .await?;

        loaded_tensor_0_0.with_data(load_to_device, None, None, |tensor_data_0_1|{
            // let tensor_data_0_1 = loaded_tensor_0_1.load_from_any_device(load_to_device, None).await?;

            assert_eq!(
                &tensor_data_0_0,
                tensor_data_0_1.t()
            );
            Ok(())

        }).await?;
        // let tensor_data_0_0 = loaded_tensor_0_0.load_from_any_device(load_to_device, None).await?;



        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_no_oom_tensor() -> Result<(), MyError> {
        unsafe { let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so"); }

        let _global_ctx = GlobalCtx::init_devices(2);

        let _device_idx = 0;

        let k = 2.;
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            k,
            vec![FailUntil::Never],
            [10,3].into(),
            ClcOOMRunner::new(OOMRunnerConfig::Permits{max: MAX_PERMITS, step:10}, Box::new(|id| Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(id, &[10,3], (Float, Device::Cuda(0)), BuildAs::Append))) ),
            Float,
            Device::Cuda(0), false
        ));
        let ctx_device_0 = Arc::new(RwLock::new(TorchRunCtx::new(1, 0, "noname")));

        let multed = load_and_mut
            .run_cached(
                Arc::new(LoaderInput {
                    id: LoaderInputId("resource_1"),
                }),
                ctx_device_0,
            )
            .await?;


        multed.with_data(1, None, None, |multed_tensor|{

            // let multed_tensor = multed.load_from_any_device(1, None).await?;

            assert!(
                Tensor::from_slice(
                    &[
                        0.,  1.,  2.,
                        3.,  4.,  5.,
                        6.,  7.,  8.,
                        9., 10., 11.,
                        12., 13., 14.,
                        15., 16., 17.,
                        18., 19., 20.,
                        21., 22., 23.,
                        24., 25., 26.,
                        27., 28., 29.
                    ]
                ).totype(Float).multiply_scalar(k as f64).to_device(Device::Cuda(0)).view((10, 3)).allclose(
                    &(*multed_tensor).t(),
                    0.01, 0.01, false
                )
            );
            Ok(())
        }).await?;



        // println!("multed={multed_tensor:?}");

        Ok(())
    }

    fn mul_check(mul_k:f64, size:LoadedTensorMetrics, res: &str, result: &ThoTensor<ClcTaskOutput, SimpleId>, rtol: f64, atol: f64) {

        mul_check_with_exp_f(mul_k, res, result, rtol, atol, |k, t|{
            lin_t(k, (&size).into(), t.kind(), t.device())
        })
    }

    fn mul_check_with_exp_f<F>(mul_k:f64, res: &str, result: &ThoTensor<ClcTaskOutput, SimpleId>, rtol: f64, atol: f64, exp_f:F)
    where F:Fn( f64, &Tensor)->Tensor
    {
        let self_k = parse_numeric_part(res).unwrap_or(1.0);

        let expected = exp_f(self_k*mul_k, &result.t());


        println!("mul_check: res={res}  result_t.size()={:?} \n\n", result.size());


        let ok = expected.allclose(&result.t(), rtol, atol, false);
        // if !ok {
            println!("mul_check: res={res} expected=\n{expected} result_t=\n{} \n\n", result.t());
        // }
        assert!(ok);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_till_min_permits_tensor() -> Result<(), MyError> {
        unsafe { let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so"); }
        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;
        const MUL_K:f64 = 2.;

        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::MinPermits(20)],
            [10, 3].into(),
            ClcOOMRunner::new(OOMRunnerConfig::Permits{max: 20, step:10}, Box::new(|id| Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(id, &[10,3], (Float, Device::Cuda(0)), BuildAs::Append)))),
            Float,
            Device::Cuda(0), false
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            |_device_idx, _cache_area, _name| TorchRunCtx::new(1, 0, "noname"),
            |res, out| {
                mul_check(MUL_K, [10, 3].into(), res, out, 0.1, 0.1);
            }
        )
            .await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn heavy_test_load_and_mul_oom_compete_for_permits_tensor() -> Result<(), MyError> {
        unsafe { let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so"); }

        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;

        const MUL_K:f64 = 2.;
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::MinPermits(MAX_PERMITS)],
            [10, 3].into(),
            ClcOOMRunner::new(OOMRunnerConfig::Permits { max: MAX_PERMITS, step: *PermitStep::TEN }, Box::new(|id| Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(id, &[10,3], (Float, Device::Cuda(0)), BuildAs::Append))) ),
            Float,
            Device::Cuda(0), false
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            |_device_idx, _cache_area, _name| TorchRunCtx::new(1, 0, "noname"),
            |res, out| {
                mul_check(MUL_K, [10, 3].into(), res, out, 0.1, 0.1);
            }
        ).await?;

        Ok::<_, MyError>(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_splits_tensor() -> Result<(), MyError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;

        const MUL_K:f64 = 2.;
        let size = [100, 3];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::MemLimit(12)],
            size.into(),
            ClcOOMRunner::new(OOMRunnerConfig::Splits {  }, Box::new(|id| Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(id, &[100,3], (Float, Device::Cuda(0)), BuildAs::Append)))),
            Float,
            Device::Cuda(0), false
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3", "resource_4", "resource_5"], //,
            device_idx,
            |_device_idx, _cache_area, _name| TorchRunCtx::new(1, 0, "noname"),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            }
        ).await?;

        Ok::<_, MyError>(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn heavy_test_load_and_mul_oom_permits_and_splits_tensor() -> Result<(), MyError>
    {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;
        const MUL_K:f64 = 2.;
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![FailUntil::MinPermits(MAX_PERMITS), FailUntil::MemLimit(12)],
            [10,3].into(),
            ClcOOMRunner::new(OOMRunnerConfig::PermitsAndSplits {max: MAX_PERMITS, step: *PermitStep::TEN}, Box::new(|id| Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(id, &[10,3], (Float, Device::Cuda(0)), BuildAs::Append)))),
            Float,
            Device::Cuda(0), false
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            (0..3).map(|i|&*format!("resource_{i}").leak()).collect::<Vec<_>>().as_slice(),//&["resource_1", "resource_2", "resource_3"],
            device_idx,
            |_device_idx, _cache_area, _name| TorchRunCtx::new(1, 0, "noname"),
            |res, out| {
                mul_check(MUL_K, [10, 3].into(), res, out, 0.1, 0.1);
            }
        ).await?;


        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn heavy_test_load_and_mul_oom_dump_to_cpu() -> Result<(), MyError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // init_tracing();
        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;

        const MUL_K:f64 = 2.;
        let size = [100000, 3];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![MinPermits(MAX_PERMITS), FailUntil::DumpedToCpu],
            size.into(),
            ClcOOMRunner::new(OOMRunnerConfig::DumpedToCpu {  }, Box::new(|id| Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(id, &[100000,3], (Float, Device::Cuda(0)), BuildAs::Append)))),
            Float,
            Device::Cuda(0), false
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            (0..5).map(|i|&*format!("resource_{i}").leak()).collect::<Vec<_>>().as_slice(), //&["resource_1","resource_2",  "resource_3"], //,
            device_idx,
            |_device_idx, _cache_area, name| TorchRunCtx::new(1, 0, name),
            move |res, out| {
                mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            }
        ).await?;

        Ok::<_, MyError>(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn heavy_test_purge_gpu_with_lock() -> Result<(), MyError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // init_tracing();
        let _global_ctx = GlobalCtx::init_devices(2);


        const MUL_K:f64 = 2.;
        let size = [100000, 3];

        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![],
            size.clone().into(),
            ClcOOMRunner::new(OOMRunnerConfig::DumpedToCpu {  }, Box::new( move |id| Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(id, &size, (Float, Device::Cuda(0)), BuildAs::Append)))),
            Float,
            Device::Cuda(0), false
        ));

        let ctx_device = Arc::new(RwLock::new(TorchRunCtx::new(1, 0, "test")));

        for res in (0..10).map(|i|&*format!("resource_{i}").leak()) {


            Arc::clone(&load_and_mut).run_cached(
                Arc::new(<&str as Into<LoaderInput>>::into(res)),
                Arc::clone(&ctx_device),
            )
                .await?;
        }

        for i in 0..10 {
            let cached_loaded = <TestTensorLoader as Cached>::cache().await.get(&SimpleId::from(format!("loaded_resource_{i}")));
            let reff = cached_loaded.expect(&format!("no entry for loaded_resource_{i}"));
            println!("i={i} 0 cached_loaded: key={:?} val={:?}", reff.key(), reff.value().clone().await?);

            assert!(!reff.value().clone().await?.is_all_on_cpu().await);


            let cached_multed = <ClcTask as Cached>::cache().await.get(&SimpleId::from(format!("mult_mul_task_0_inp_loaded_resource_{i}")));
            let reff = cached_multed.expect(&format!("no entry for loaded_resource_{i}"));
            println!("i={i} 0 cached_multed: key={:?} val={:?}", reff.key(), reff.value().clone().await?);

            assert!(!reff.value().clone().await?.is_all_on_cpu().await);

        }

        let mut loaded_tensors_ids = HashSet::new();
        let loaded_tensor_3 = SimpleId::from("loaded_resource_3");
        loaded_tensors_ids.insert(&loaded_tensor_3 as &dyn CachableIdTr);

        let mut multed_tensors_ids = HashSet::new();
        let mult_mul_task_0_inp_loaded_resource_3 = SimpleId::from("mult_mul_task_0_inp_loaded_resource_3");
        multed_tensors_ids.insert(&mult_mul_task_0_inp_loaded_resource_3 as &dyn CachableIdTr);


        let mut required_ids = HashMap::new();
        required_ids.insert("LoadedTensorId", &loaded_tensors_ids);
        required_ids.insert("MulTaskOutputId", &multed_tensors_ids);

        _global_ctx.dump_all_to_cpu(&required_ids, 1).await?;

        for i in 0..10 {
            let cached_loaded = <TestTensorLoader as Cached>::cache().await.get(&SimpleId::from(format!("loaded_resource_{i}")));
            let reff = cached_loaded.expect(&format!("no entry for loaded_resource_{i}"));
            println!("i={i} 1 cached_loaded: key={:?} val={:?}", reff.key(), reff.value().clone().await?);

            if i == 3 {
                assert!(!reff.value().clone().await?.is_all_on_cpu().await);
            } else {
                assert!(reff.value().clone().await?.is_all_on_cpu().await);
            }

            let cached_multed = <ClcTask as Cached>::cache().await.get(&SimpleId::from(format!("mult_mul_task_0_inp_loaded_resource_{i}")));
            let reff = cached_multed.expect(&format!("no entry for loaded_resource_{i}"));
            println!("i={i} 1 cached_multed: key={:?} val={:?}", reff.key(), reff.value().clone().await?);

            if i == 3 {
                assert!(!reff.value().clone().await?.is_all_on_cpu().await);
            } else {
                assert!(reff.value().clone().await?.is_all_on_cpu().await);
            }

        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn heavy_test_real_oom_dump_and_split() -> Result<(), MyError> {
        println!("test_real_oom_dump_and_split");
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }
        // init_tracing();
        let _global_ctx = GlobalCtx::init_devices(2);

        let (free_mem, _total_mem) = cuda_mem();
        let init_type = Kind::Int64;

        let total_size = (free_mem as f64 * 0.9) as i64 / init_type.elt_size_in_bytes() as i64; //123456; //25000; //123456; //
        let dim_1 = 4i64;
        let input_n = total_size / dim_1;

        println!("test_real_oom_dump_and_split: input_n={input_n} dim_1={dim_1}");

        const MUL_K:i64 = 1;
        let size = [input_n, dim_1];

        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![],
            size.clone().into(),
            ClcOOMRunner::new(OOMRunnerConfig::DumpedToCpu {  }, Box::new( move |id| Box::new(SingleDimTensorOutputBuilder::<ClcTaskOutput>::new(id, &[dim_1,1], (Kind::Int64, Device::Cuda(0)), BuildAs::Sum)))),
            Kind::Int64,
            Device::Cuda(0),
            true
        ));

        let ctx_device = Arc::new(RwLock::new(TorchRunCtx::new(1, 0, "test")));

        let multed = Arc::clone(&load_and_mut).run_cached(
            Arc::new(<&str as Into<LoaderInput>>::into("resource_1")),
            Arc::clone(&ctx_device),
        ).await?;

        println!("multed={multed:?}");

        multed.with_data(
            1,
            None,
            None,
            |r:&ThoTensor<ClcTaskOutput, SimpleId>| {

                mul_check_with_exp_f(MUL_K as f64, "resource_1", r, 0.01, 0.01, |k, t|{
                    lin_t(k, size, t.kind(), t.device()).sum_dim_intlist(0, false, t.kind()).view((-1, 1))
                });

                println!("Ok");

                Ok(())
            }
        ).await?;

           

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
