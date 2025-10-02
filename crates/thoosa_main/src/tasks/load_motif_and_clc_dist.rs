use std::fmt::{Debug, Formatter};
use std::ops::{Add, Deref, DivAssign, Range};
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
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



#[derive(Cached)]
struct LoadMotifAndClcDistanceStage<SRC>
where
    SRC: TensorLoader + 'static
{
    id:SimpleId,
    motif_loader:Arc<MotifLoader<SRC>>,
    clc_dist:Arc<AverageEuclideanDistanceStage>,
    output_builder: Box<
        dyn Fn(<Self as WithId>::ID) -> Box<dyn OutputBuilder<O = AverageDistanceValue>>
        + Send
        + Sync>

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

impl <SRC> Debug for LoadMotifAndClcDistanceStage<SRC>
where
    SRC: TensorLoader + 'static {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadMotifAndClcDistanceStage").finish()
    }
}

impl <SRC> WithId for LoadMotifAndClcDistanceStage<SRC>
where
    SRC: TensorLoader + 'static
{
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl <SRC> LoadMotifAndClcDistanceStage<SRC>
where
    SRC: TensorLoader + 'static
{
    fn new(motif_loader:Arc<MotifLoader<SRC>>, clc_dist:Arc<AverageEuclideanDistanceStage>) -> Self {
        Self {
            id: SimpleId("LoadMotifAndClcDistanceStage"),
            motif_loader,
            clc_dist,
            output_builder: Box::new(|id| Box::new(DistOutputBuilder::new(id))),
        }
    }
}
/*
#[derive(Debug)]
struct MotifWIdhDistance {
    id: SimpleId,
    dist: AverageDistanceValue,
    motif: ThoTensor2<MotifSlices, DataId>,
}

#[async_trait]
impl CopyToDevice for MotifWIdhDistance {
    async fn copy_to_device(&self, dst_device: usize) -> Self {
        MotifWIdhDistance {
            id: self.id.clone(),
            dist: self.dist.clone(),
            motif: self.motif.copy_to_device(dst_device).await,
        }

    }
}

impl Data for MotifWIdhDistance {
    type Metrics = <ThoTensor2<MotifSlices, SimpleId> as Data>::Metrics;

    fn size(&self) -> Self::Metrics {
        self.motif.size()
    }

    fn device(&self) -> usize {
        0
    }
}

impl MotifWIdhDistance {
    pub fn new(id: SimpleId, dist: AverageDistanceValue, motif: ThoTensor2<MotifSlices, DataId>) -> Self {
        Self {
            id,
            dist,
            motif,
        }
    }
}

impl WithId for MotifWIdhDistance {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl Output for MotifWIdhDistance {}
*/
impl <SRC> OutId for LoadMotifAndClcDistanceStage<SRC>
where
    SRC: TensorLoader + 'static
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("{}_avg_ec_dist", input_id))
    }
}




#[async_trait]
impl <SRC>  Stage for  LoadMotifAndClcDistanceStage<SRC>
where
    SRC: TensorLoader + 'static {

    type In<'a> = DataId;

    type Out = AverageDistanceValue;
    type Ctx = RunCtx2;


    async fn run(self: Arc<Self>, input: Arc<Self::In<'static>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
        let out_id = self.out_id(input.id());

        let motif = Arc::clone(&self.motif_loader).run_cached(input, Arc::clone(&ctx)).await?;


        let dist = OOMHandler::new(
            vec![Adjusters::Input(Box::new(
                SplitsAdjuster::<ThoTensor2<MotifSlices, DataId>, MotifSlicesSelector>::new([10, -1].into()),
            ))],
            None,
        )
            .run(
                &self.clc_dist,
                motif,
                ctx,
                Contexts::new(
                    // NoopInputContext(None),
                    SplitsAdjusterContextImpl::new([10, -1].into(), (self.output_builder)(out_id.clone()/*it shold be dist id*/)),
                    NoopRunnableContext {},
                    NoopRunCtxContext {},
                ),
            )
            .await?;

        Ok(dist)

    }
}

#[derive(Clone, Debug, Default)]
struct MotifSlicesSelector {

}

impl From<Range<TwoDimsMetrics>>  for MotifSlicesSelector{
    fn from(value: Range<TwoDimsMetrics>) -> Self {
        todo!()
    }
}

impl Adjustable for MotifSlicesSelector  {}


impl Select<MotifSlicesSelector> for ThoTensor2<MotifSlices, DataId> {
    type Output = ThoTensor2<MotifSlices, DataId>;

    fn select(&self, selector: MotifSlicesSelector) -> Self::Output {
        todo!()
    }
}

#[derive(Debug)]
struct DistOutputBuilder {
    id:SimpleId,
}

impl DistOutputBuilder {
    pub fn new(id:SimpleId) -> Self {
        Self {id}
    }
}

impl OutputBuilder for DistOutputBuilder {
    type O = AverageDistanceValue;

    fn add_partial_output(&mut self, o: Self::O) {

    }

    fn build(self: Box<Self>) -> Self::O {
        AverageDistanceValue::new(self.id.clone(), 0.0)
    }
}



#[cfg(test)]
mod tests {


    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_load_and_clc_average_euclidean_distance(){

    }

}