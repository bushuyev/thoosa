use crate::tasks::cross_dims::MotifSlices;
use crate::tasks::run_ctx_2::RunCtx2;
use async_trait::async_trait;
use caching_runner::adjusters::splits::OutputBuilder;
use caching_runner::common::runnable::{Stage, SyncStage};
use caching_runner::common::OutId;
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use caching_runner::oom_handler::Adjustable;
use thoosa_motifs::distance::average_euclidean_distance;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::traits::{DataId, SimpleId, WithId};
use thoosa_shared::ThoosaError;
use torch_runner::data::ThoTensor2;
use crate::tasks::motif_loader::MotifWithData;
use thoosa_dummy_data_derive::DummyData;

pub struct AverageEuclideanDistanceStage {
    id:SimpleId,
}

impl Debug for AverageEuclideanDistanceStage {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AverageEuclideanDistance").finish()
    }
}

impl Adjustable for AverageEuclideanDistanceStage {

}

impl WithId for AverageEuclideanDistanceStage {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl AverageEuclideanDistanceStage {
    pub fn new() -> Self {
        Self {
            id: SimpleId("AverageEuclideanDistanceStage"),
        }
    }
}

#[derive(Debug, Clone, DummyData)]
pub struct AverageDistanceValue {
    id: SimpleId,
    value: f32,
}

impl AverageDistanceValue {
    pub fn new(id: SimpleId, value: f32) -> Self {
        Self {
            id,
            value
        }
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

impl WithId for AverageDistanceValue {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl Output for AverageDistanceValue {}

impl OutId for AverageEuclideanDistanceStage {
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("{}_avg_ec_dist", input_id))
    }
}


impl Stage for AverageEuclideanDistanceStage {
    type Ctx = RunCtx2;
    type In<'a> = MotifWithData;
    type Out = AverageDistanceValue;
}
impl SyncStage for AverageEuclideanDistanceStage {

    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {

        input.t().run_inner(|t|{
            Ok(AverageDistanceValue::new("todo".into(), average_euclidean_distance(t)?))
        })

    }
}


#[cfg(test)]
mod tests {


    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_average_euclidean_distance(){

    }

}
#[cfg(test)]
pub mod mock_average_stage {
    use std::collections::VecDeque;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use async_trait::async_trait;
    use log::debug;
    use parking_lot::{Mutex, RwLock};
    use caching_runner::common::OutId;
    use caching_runner::common::runnable::{Stage, SyncStage};
    use caching_runner::oom_handler::Adjustable;
    use thoosa_shared::{MyError, ThoosaError};
    use thoosa_shared::traits::{DataId, SimpleId, WithId};
    use torch_runner::data::ThoTensor2;
    use thoosa_shared::traits::MonoMetrics;
    use crate::tasks::average_euclidean_distance::AverageDistanceValue;
    use crate::tasks::cross_dims::MotifSlices;
    use crate::tasks::motif_loader::MotifWithData;
    use crate::tasks::run_ctx_2::RunCtx2;

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
            self.outputs.lock().pop_front()
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

    impl Stage for MockAverageStage {
        type Ctx = RunCtx2;
        type In<'a> = MotifWithData;
        type Out = AverageDistanceValue;
    }
    impl SyncStage for MockAverageStage {

        fn do_run<'a>(
            self: Arc<Self>,
            input: &Self::In<'a>,
            ctx: Arc<RwLock<Self::Ctx>>,
        ) -> Result<Self::Out, ThoosaError> {
            let _ = ctx;
            let call_idx = self.counter.fetch_add(1, Ordering::SeqCst);
            let metrics: MonoMetrics<0, 2> = input.t()
                .size()
                .try_into()
                .expect("MockAverageStage expects motif metrics");

            debug!("MockAverageStage::do_run: metrics={metrics:?}");

            let dim_0 = metrics.dim_0();
            self.call_log.lock().push(dim_0);

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

    pub fn builder() -> MockAverageStageBuilder {
        MockAverageStageBuilder {
            outputs: Vec::new(),
            fail_first: false,
            fail_while_dim0_gt: None,
            call_log: None,
        }
    }
}
