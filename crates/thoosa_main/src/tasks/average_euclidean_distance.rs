use crate::tasks::cross_dims::MotifSlices;
use crate::tasks::motif_loader::MotifMeta;
use crate::tasks::run_ctx_2::RunCtx2;
use async_trait::async_trait;
use thoosa_shared::runner::adjusters::splits::OutputBuilder;
use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_dummy_data_derive::DummyData;
use thoosa_motifs::distance::average_euclidean_distance;
use thoosa_shared::{mono_key, panic_to_err, CopyToDevice, ThoosaError};
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::traits::Adjustable;
use thoosa_shared::traits::{DataId, WithId};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::comp_id::SynKey;

pub struct AverageEuclideanDistanceStage<C>
where
    C: RunCtx + 'static,
{
    id: SynKey,
    _ctx: PhantomData<C>,
}

impl<C> Debug for AverageEuclideanDistanceStage<C>
where
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AverageEuclideanDistance").finish()
    }
}

impl<C> Adjustable for AverageEuclideanDistanceStage<C> where C: RunCtx + 'static {}

impl<C> WithId for AverageEuclideanDistanceStage<C>
where
    C: RunCtx + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<C> AverageEuclideanDistanceStage<C>
where
    C: RunCtx + 'static,
{
    pub fn new() -> Self {
        Self {
            id: SynKey::new([mono_key!("AverageEuclideanDistanceStage", "stage")]),
            _ctx: PhantomData,
        }
    }
}

#[derive(Debug, Clone, DummyData)]
#[dummy_data(clone)]
pub struct AverageDistanceValue {
    id: SynKey,
    value: f32,
}

impl AverageDistanceValue {
    pub fn new(id: SynKey, value: f32) -> Self {
        Self { id, value }
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

impl WithId for AverageDistanceValue {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl Output for AverageDistanceValue {}

#[async_trait]
impl CopyToDevice for AverageDistanceValue {
    async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoosaError> {
        Ok(self.clone())
    }
}

impl<C> OutId for AverageEuclideanDistanceStage<C>
where
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SynKey::new([mono_key!(input_id.0, format!("{input_id}_avg_ec_dist"))])
    }
}

impl<C> Stage for AverageEuclideanDistanceStage<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>;
    type Out = AverageDistanceValue;
}
impl<C> SyncStage for AverageEuclideanDistanceStage<C>
where
    C: RunCtx + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {
        Ok(AverageDistanceValue::new(
            SynKey::new([mono_key!("todo", "todo")]),
            panic_to_err!(average_euclidean_distance(input.data().t_ref()))?
        ))
    }
}

#[cfg(test)]
mod tests {

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_average_euclidean_distance() {
        //TODO
    }
}
#[cfg(test)]
pub mod mock_average_stage {
    use crate::tasks::average_euclidean_distance::AverageDistanceValue;
    use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
    use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
    use crate::tasks::tensor_loader::mock_tensor_loader::MockTensorLoaderBuilder;
    use async_trait::async_trait;
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
    use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use thoosa_shared::runner::common::{MockRunCtx, OutId};
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use dashmap::DashMap;
    use futures_util::future::{BoxFuture, Shared};
    use log::debug;
    use mockall::mock;
    use parking_lot::{Mutex, RwLock};
    use std::collections::HashMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{Adjustable, WithId};
    use thoosa_shared::traits::{DataId, MonoMetrics, Named};
    use thoosa_shared::{mono_key, CopyToDevice, MyError, ThoosaError};
    use thoosa_shared::comp_id::SynKey;
    use tokio::runtime::Handle;
    use tokio::task::block_in_place;
    use thoosa_common::tho_tensor::ThoTensor;

    mock! {
        #[derive(Debug)]
        pub AverageEuclideanDistanceStage {}

        impl Stage for AverageEuclideanDistanceStage {
            type Ctx = MockRunCtx;
            type In<'a> = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>;
            type Out = AverageDistanceValue;
        }

        impl OutId for AverageEuclideanDistanceStage {
            fn out_id<'a>(&self,input_id: &<<Self as Stage>::In<'a> as WithId>::ID,) -> <<MockAverageEuclideanDistanceStage as Stage>::Out as WithId>::ID;
        }


        impl SyncStage for AverageEuclideanDistanceStage {
            fn do_run<'f>(self: Arc<Self>, _input: &<Self as Stage>::In<'f>, _ctx: Arc<parking_lot::RwLock<<Self as Stage>::Ctx>>) -> Result<<Self as Stage>::Out, ThoosaError>;
        }


        impl Adjustable for AverageEuclideanDistanceStage {

        }


    }

    pub struct MockAverageEuclideanDistanceStageBuilder {
        outputs: Vec<AverageDistanceValue>,
        fail_first: bool,
        fail_while_dim0_gt: Option<i64>,
        call_log: Option<Arc<Mutex<Vec<i64>>>>,
        counter: AtomicUsize,

        mock: MockAverageEuclideanDistanceStage,
    }

    impl MockAverageEuclideanDistanceStageBuilder {
        pub fn with_do_run<I>(mut self, outputs: I) -> Self
        where
            I: Into<Vec<AverageDistanceValue>>,
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

        pub fn finish(mut self) -> MockAverageEuclideanDistanceStage {
            self.mock.expect_do_run().returning(move |input, _ctx| {
                let call_idx = self.counter.fetch_add(1, Ordering::SeqCst);
                let metrics: MonoMetrics<0, 2> = input.data().size().try_into().expect("MockAverageStage expects motif metrics");

                debug!("MockAverageStage::do_run: metrics={metrics:?}");

                let dim_0 = metrics.dim_0();
                if let Some(call_log) = self.call_log.as_ref() {
                    call_log.lock().push(dim_0);
                }

                if self.fail_first && call_idx == 0 {
                    return Err(MyError::OutOfMemory.into());
                }

                if let Some(threshold) = self.fail_while_dim0_gt {
                    if dim_0 > threshold {
                        return Err(MyError::OutOfMemory.into());
                    }
                }

                let value = self.outputs.pop().expect("mock outputs exhausted");

                Ok(value)
            });

            self.mock.expect_out_id().return_const(SynKey::new([mono_key!("whatever", "mock")]));

            self.mock
        }
    }

    pub fn builder() -> MockAverageEuclideanDistanceStageBuilder {
        MockAverageEuclideanDistanceStageBuilder {
            outputs: Vec::new(),
            fail_first: false,
            fail_while_dim0_gt: None,
            call_log: None,
            counter: Default::default(),
            mock: MockAverageEuclideanDistanceStage::new(),
        }
    }
}
