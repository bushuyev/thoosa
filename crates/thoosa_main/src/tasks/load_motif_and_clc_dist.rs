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



#[derive(Cached)]
pub struct LoadMotifAndClcDistanceStage<SRC>
where
    SRC: TensorLoader + 'static
{
    id:SimpleId,
    motif_loader:Arc<MotifLoader<SRC>>,
    clc_dist:Arc<AverageEuclideanDistanceStage>,
    output_builder: Box<dyn Fn(<Self as WithId>::ID) -> Box<dyn OutputBuilder<O = AverageDistanceValue>> + Send + Sync>

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
    pub fn new(motif_loader:Arc<MotifLoader<SRC>>, clc_dist:Arc<AverageEuclideanDistanceStage>) -> Self {
        Self {
            id: SimpleId("LoadMotifAndClcDistanceStage"),
            motif_loader,
            clc_dist,
            output_builder: Box::new(|id| Box::new(DistOutputBuilder::new(id))),
        }
    }
}

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

        let motif_size = motif.size();

        debug!("LoadMotifAndClcDistanceStage::run: motif_id={} motif_size={motif_size:?}", input.id());

        let dist = OOMHandler::new(
            vec![Adjusters::Input(Box::new(
                SplitsAdjuster::<ThoTensor2<MotifSlices, DataId>, MotifSlicesSelector>::new([(motif_size.dim_0() as f32 * 0.1) as i64 + 2, -1].into()),
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
struct MotifSlicesSelector {
    slices: Range<TwoDimsMetrics>
}

impl From<Range<TwoDimsMetrics>>  for MotifSlicesSelector{
    fn from(value: Range<TwoDimsMetrics>) -> Self {
        MotifSlicesSelector { slices: value }
    }
}

impl Adjustable for MotifSlicesSelector  {}


impl Select<MotifSlicesSelector> for ThoTensor2<MotifSlices, DataId> {
    type Output = ThoTensor2<MotifSlices, DataId>;

    fn select(&self, selector: MotifSlicesSelector) -> Self::Output {

        use std::cmp::min;

        let size: TwoDimsMetrics = self
            .size()
            .try_into()
            .expect("LoadedTensor::select unable to convert size");

        let dim_0_range = selector.slices.start.dim_0()..min(size.dim_0(), selector.slices.start.dim_0());

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



#[cfg(test)]
mod tests {


    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_load_and_clc_average_euclidean_distance(){

    }

}