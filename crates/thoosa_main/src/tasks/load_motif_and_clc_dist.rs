use std::fmt::{Debug, Formatter};
use std::ops::{Add, Deref, DivAssign, Range};
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::IndexOp;
use cached_derive::Cached;
use caching_runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContext, SplitsAdjusterContextImpl};
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use caching_runner::common::{Data, OutId, Select};
use caching_runner::common::runnable::Stage;
use caching_runner::oom_handler::{Adjustable, Adjuster, Adjusters, Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use thoosa_common::dims::Dims;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::traits::{DataId, SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::cross_dims::MotifSlices;
use crate::tasks::motif_loader::MotifLoader;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::TensorLoader;

#[async_trait]
pub trait MotifLoaderStage: Send + Sync + Debug {
    type Output: MotifStageInput + 'static;

    async fn run_cached(
        self: Arc<Self>,
        input: Arc<DataId>,
        ctx: Arc<RwLock<RunCtx2>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Output>>, ThoosaError>;
}

#[async_trait]
impl<SRC> MotifLoaderStage for MotifLoader<SRC>
where
    SRC: TensorLoader + 'static,
{
    type Output = ThoTensor2<MotifSlices, DataId>;

    async fn run_cached(
        self: Arc<Self>,
        input: Arc<DataId>,
        ctx: Arc<RwLock<RunCtx2>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Output>>, ThoosaError> {
        <Self as Cached>::run_cached(self, input, ctx).await
    }
}



#[derive(Cached)]
pub struct LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: MotifStageInput,
    CLC::In<'static>: 'static,
{
    id: SimpleId,
    motif_loader: Arc<dyn MotifLoaderStage<Output = <CLC as Stage>::In<'static>>>,
    clc_dist: Arc<CLC>,
    output_builder:
        Box<dyn Fn(<Self as WithId>::ID) -> Box<dyn OutputBuilder<O = AverageDistanceValue>> + Send + Sync>,
}

#[async_trait]
impl CopyToDevice for AverageDistanceValue {
    async fn copy_to_device(&self, dst_device: usize) -> Self {
        self.clone()
    }
}

impl Data for AverageDistanceValue {
    type Metrics = ();

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }
}

impl<CLC> Debug for LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: MotifStageInput,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadMotifAndClcDistanceStage").finish()
    }
}

impl<CLC> WithId for LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: MotifStageInput,
{
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<CLC> LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: MotifStageInput,
{
    pub fn new(
        motif_loader: Arc<dyn MotifLoaderStage<Output = <CLC as Stage>::In<'static>>>,
        clc_dist: Arc<CLC>,
    ) -> Self {
        Self {
            id: SimpleId("LoadMotifAndClcDistanceStage"),
            motif_loader,
            clc_dist,
            output_builder: Box::new(|id| Box::new(DistOutputBuilder::new(id))),
        }
    }
}

impl<CLC> OutId for LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: MotifStageInput,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("{}_avg_ec_dist", input_id))
    }
}


#[async_trait]
impl<CLC> Stage for LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: MotifStageInput,
{

    type In<'a> = DataId;

    type Out = AverageDistanceValue;
    type Ctx = RunCtx2;


    async fn run(self: Arc<Self>, input: Arc<Self::In<'static>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
        let out_id = self.out_id(input.id());

        let motif = Arc::clone(&self.motif_loader).run_cached(input, Arc::clone(&ctx)).await?;

        let motif_size = motif.size();

        debug!("LoadMotifAndClcDistanceStage::run: out_id={} motif_size={motif_size:?}", out_id);

        let dist = OOMHandler::new(
            vec![Adjusters::Input(Box::new(
                SplitsAdjuster::<<CLC as Stage>::In<'static>, MotifSlicesSelector>::new(
                    [(motif_size.dim_0() as f32 * 0.1) as i64 + 2, -1].into(),
                ),
            ))],
            None,
        )
            .run(
                &self.clc_dist,
                Arc::clone(&motif),
                ctx,
                Contexts::new(
                    SplitsAdjusterContextImpl::new(motif_size.clone(), (self.output_builder)(out_id.clone()/*it should be dist id*/)),
                    NoopRunnableContext {},
                    NoopRunCtxContext {},
                ),
            )
            .await?;

        Ok(dist)

    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MotifSlicesSelector {
    slices: Range<TwoDimsMetrics<0>>
}

impl From<Range<TwoDimsMetrics<0>>>  for MotifSlicesSelector{
    fn from(value: Range<TwoDimsMetrics<0>>) -> Self {
        MotifSlicesSelector { slices: value }
    }
}

impl Adjustable for MotifSlicesSelector  {}


impl Select<MotifSlicesSelector> for ThoTensor2<MotifSlices, DataId> {
    type Output = ThoTensor2<MotifSlices, DataId>;

    fn select(&self, selector: MotifSlicesSelector) -> Self::Output {

        use std::cmp::min;

        let size: TwoDimsMetrics<0> = self
            .size()
            .try_into()
            .expect("LoadedTensor::select unable to convert size");

        let dim_0_range = selector.slices.start.dim_0()..min(size.dim_0(), selector.slices.end.dim_0());

        let sub_tensor = self.t_ref().i((dim_0_range.clone(), ..));

        ThoTensor2::new(self.id().clone(), sub_tensor)
    }
}

#[derive(Debug)]
struct DistOutputBuilder {
    id:SimpleId,
    values:Vec<f32>
}

impl DistOutputBuilder {
    pub fn new(id:SimpleId) -> Self {
        Self {id, values: vec![] }
    }
}

impl OutputBuilder for DistOutputBuilder {
    type O = AverageDistanceValue;

    fn add_partial_output(&mut self, o: Self::O) {
        debug!("DistOutputBuilder::add_partial_output: o={o:?}");
        self.values.push(o.value());
    }

    fn build(self: Box<Self>) -> Self::O {
        debug!("DistOutputBuilder::build: self.values={:?}", self.values);

        AverageDistanceValue::new(self.id.clone(), if self.values.is_empty() {0.0} else {self.values.iter().sum::<f32>()/self.values.len() as f32})
    }
}



pub(crate) trait MotifStageInput:
    WithId
    + Data<Metrics = TwoDimsMetrics<0>>
    + CopyToDevice
    + Select<MotifSlicesSelector, Output = Self>
    + Adjustable
    + Debug
    + Send
    + Sync
{
}

impl MotifStageInput for ThoTensor2<MotifSlices, DataId> {}

#[cfg(test)]
mod tests {
    use super::{AverageDistanceValue, LoadMotifAndClcDistanceStage, MotifLoaderStage, MotifSlices, RunCtx2};
    use async_trait::async_trait;
    use caching_runner::common::cachable::DataCacheWrapper;
    use caching_runner::common::runnable::Stage;
    use caching_runner::global_ctx::GlobalCtx;
    use parking_lot::RwLock;
    use std::collections::VecDeque;
    use std::fmt;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use tch::Tensor;
    use thoosa_shared::MyError;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::traits::{DataId, SimpleId, WithId};
    use torch_runner::data::ThoTensor2;

    mod mock_motif_loader {
        use super::*;

        type RunCachedFuture = Pin<
            Box<
                dyn Future<
                        Output = Result<Arc<DataCacheWrapper<ThoTensor2<MotifSlices, DataId>>>, ThoosaError>,
                    > + Send,
            >,
        >;

        type RunCachedFn = dyn Fn(Arc<DataId>, Arc<RwLock<RunCtx2>>) -> RunCachedFuture + Send + Sync;

        #[derive(Clone)]
        pub struct MockMotifLoader {
            run_cached_fn: Arc<RunCachedFn>,
        }

        impl fmt::Debug for MockMotifLoader {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.debug_struct("MockMotifLoader").finish()
            }
        }

        #[async_trait]
        impl MotifLoaderStage for MockMotifLoader {
            type Output = ThoTensor2<MotifSlices, DataId>;

            async fn run_cached(
                self: Arc<Self>,
                input: Arc<DataId>,
                ctx: Arc<RwLock<RunCtx2>>,
            ) -> Result<Arc<DataCacheWrapper<Self::Output>>, ThoosaError> {
                (self.run_cached_fn)(input, ctx).await
            }
        }

        pub struct MockMotifLoaderBuilder {
            run_cached_fn: Option<Arc<RunCachedFn>>,
        }

        impl MockMotifLoaderBuilder {
            pub fn with_run_cached_fn<F, Fut>(mut self, f: F) -> Self
            where
                F: Fn(Arc<DataId>, Arc<RwLock<RunCtx2>>) -> Fut + Send + Sync + 'static,
                Fut: Future<
                        Output = Result<Arc<DataCacheWrapper<ThoTensor2<MotifSlices, DataId>>>, ThoosaError>,
                    > + Send
                    + 'static,
            {
                self.run_cached_fn = Some(Arc::new(move |input, ctx| Box::pin(f(input, ctx))));
                self
            }

            pub fn with_result(
                self,
                data: Arc<DataCacheWrapper<ThoTensor2<MotifSlices, DataId>>>,
            ) -> Self {
                self.with_run_cached_fn(move |_input, _ctx| {
                    let data = Arc::clone(&data);
                    async move { Ok(data) }
                })
            }

            pub fn finish(self) -> Arc<MockMotifLoader> {
                let run_cached_fn = self
                    .run_cached_fn
                    .expect("MockMotifLoaderBuilder requires run_cached_fn to be set");
                Arc::new(MockMotifLoader { run_cached_fn })
            }
        }

        pub fn build() -> MockMotifLoaderBuilder {
            MockMotifLoaderBuilder { run_cached_fn: None }
        }
    }

    mod mock_average_stage {
        use super::*;
        use caching_runner::common::OutId;
        use caching_runner::oom_handler::Adjustable;
        use torch_runner::metrics::TwoDimsMetrics;

        #[derive(Debug)]
        pub struct MockAverageStage {
            id: SimpleId,
            outputs: Mutex<VecDeque<f32>>,
            fail_first: bool,
            fail_while_dim0_gt: Option<i64>,
            call_log: Arc<Mutex<Vec<i64>>>,
            counter: AtomicUsize,
        }

        impl MockAverageStage {
            fn next_value(&self) -> Option<f32> {
                self.outputs.lock().unwrap().pop_front()
            }
        }

        impl Adjustable for MockAverageStage {}

        impl WithId for MockAverageStage {
            type ID = SimpleId;

            fn id(&self) -> &Self::ID {
                &self.id
            }
        }

        impl OutId for MockAverageStage {
            fn out_id(
                &self,
                input_id: &<<Self as Stage>::In<'static> as WithId>::ID,
            ) -> <Self::Out as WithId>::ID {
                SimpleId::from(format!("{}_avg_ec_dist", input_id))
            }
        }

        #[async_trait]
        impl Stage for MockAverageStage {
            type Ctx = RunCtx2;
            type In<'a> = ThoTensor2<MotifSlices, DataId>;
            type Out = AverageDistanceValue;

            fn do_run<'a>(
                self: Arc<Self>,
                input: &Self::In<'a>,
                ctx: Arc<RwLock<Self::Ctx>>,
            ) -> Result<Self::Out, ThoosaError>
            where
                Self: Sized,
            {
                let _ = ctx;
                let call_idx = self.counter.fetch_add(1, Ordering::SeqCst);
                let metrics: TwoDimsMetrics<0> = input
                    .size()
                    .try_into()
                    .expect("MockAverageStage expects motif metrics");
                let dim_0 = metrics.dim_0();
                self.call_log.lock().unwrap().push(dim_0);

                if self.fail_first && call_idx == 0 {
                    return Err(MyError::OutOfMemory.into());
                }

                if let Some(threshold) = self.fail_while_dim0_gt {
                    if dim_0 > threshold {
                        return Err(MyError::OutOfMemory.into());
                    }
                }

                let value = self
                    .next_value()
                    .expect("mock outputs exhausted");

                Ok(AverageDistanceValue::new(
                    SimpleId::from(format!("partial_{call_idx}")),
                    value,
                ))
            }
        }

        pub struct MockAverageStageBuilder {
            outputs: Vec<f32>,
            fail_first: bool,
            fail_while_dim0_gt: Option<i64>,
            call_log: Option<Arc<Mutex<Vec<i64>>>>,
        }

        impl MockAverageStageBuilder {
            pub fn with_outputs<I>(mut self, outputs: I) -> Self
            where
                I: Into<Vec<f32>>,
            {
                self.outputs = outputs.into();
                self
            }

            pub fn fail_first(mut self) -> Self {
                self.fail_first = true;
                self
            }

            pub fn fail_while_dim0_gt(mut self, threshold: i64) -> Self {
                self.fail_while_dim0_gt = Some(threshold);
                self
            }

            pub fn with_call_log(mut self, call_log: Arc<Mutex<Vec<i64>>>) -> Self {
                self.call_log = Some(call_log);
                self
            }

            pub fn finish(self) -> Arc<MockAverageStage> {
                let mut deque = VecDeque::new();
                deque.extend(self.outputs);
                let outputs = Mutex::new(deque);
                let call_log = self
                    .call_log
                    .unwrap_or_else(|| Arc::new(Mutex::new(Vec::new())));

                Arc::new(MockAverageStage {
                    id: SimpleId("MockAverageStage"),
                    outputs,
                    fail_first: self.fail_first,
                    fail_while_dim0_gt: self.fail_while_dim0_gt,
                    call_log,
                    counter: AtomicUsize::new(0),
                })
            }
        }

        pub fn build() -> MockAverageStageBuilder {
            MockAverageStageBuilder {
                outputs: Vec::new(),
                fail_first: false,
                fail_while_dim0_gt: None,
                call_log: None,
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_load_and_clc_average_euclidean_distance() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);

        let ctx = Arc::new(RwLock::new(RunCtx2::new(0, 0, "test")));

        let motif_tensor = ThoTensor2::new(
            DataId::from(999u32),
            Tensor::from_slice(&[
                0.0, 1.0, 2.0,
                3.0, 4.0, 5.0,
                6.0, 7.0, 8.0,
                9.0, 10.0, 11.0,
                12.0, 13.0, 14.0,
                15.0, 16.0, 17.0,
            ])
            .view((6, 3)),
        );

        let motif_wrapper = Arc::new(DataCacheWrapper::new(
            motif_tensor,
            0,
            GlobalCtx::global().device_count(),
            0,
        ));

        let motif_loader = mock_motif_loader::build()
            .with_result(Arc::clone(&motif_wrapper))
            .finish();

        let call_log = Arc::new(Mutex::new(Vec::new()));

        let clc_stage = mock_average_stage::build()
            .fail_while_dim0_gt(3)
            .with_outputs(vec![10.0, 30.0])
            .with_call_log(Arc::clone(&call_log))
            .finish();

        let stage = Arc::new(LoadMotifAndClcDistanceStage::new(
            motif_loader as Arc<dyn MotifLoaderStage<Output = ThoTensor2<MotifSlices, DataId>>>,
            clc_stage,
        ));

        let input_id = Arc::new(DataId::from(42u32));
        let result = stage
            .run(Arc::clone(&input_id), Arc::clone(&ctx))
            .await?;

        assert_eq!(result.value(), 20.0);
        let expected_id = SimpleId::from(format!("{}_avg_ec_dist", input_id));
        assert_eq!(result.id(), &expected_id);

        let recorded = call_log.lock().unwrap().clone();
        assert!(recorded.iter().any(|&d| d > 3), "expected at least one oversized run, got {recorded:?}");
        let split_calls = recorded.iter().filter(|&&d| d == 3).count();
        assert_eq!(split_calls, 2, "expected two split runs, got {recorded:?}");
        assert_eq!(recorded.last().copied(), Some(3));

        Ok(())
    }

}
