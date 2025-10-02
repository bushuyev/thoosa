use crate::tasks::cross_dims::MotifSlices;
use crate::tasks::run_ctx_2::RunCtx2;
use async_trait::async_trait;
use caching_runner::adjusters::splits::OutputBuilder;
use caching_runner::common::runnable::Stage;
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

#[derive(Debug, Clone)]
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

#[async_trait]
impl Stage for AverageEuclideanDistanceStage {
    type Ctx = RunCtx2;
    type In<'a> = ThoTensor2<MotifSlices, DataId>;
    type Out = AverageDistanceValue;




    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized
    {

        input.run_inner(|t|{
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