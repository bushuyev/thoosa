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
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::global_ctx::GlobalCtx;
use caching_runner::oom_handler::{Adjustable, Adjuster, Adjusters, Contexts, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use thoosa_common::dims::Dims;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Data, DataId, Select, SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use thoosa_shared::traits::MonoMetrics;
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
use crate::tasks::motif_loader::{MotifLoader, MotifWithData};
use crate::tasks::run_ctx_2::RunCtx2;

#[async_trait]
pub trait MotifLoaderStage: Send + Sync + Debug {
    type Output: ClcMotifInput + 'static;

    async fn run_cached(
        self: Arc<Self>,
        input: Arc<DataCacheWrapper<DataId>>,
        ctx: Arc<RwLock<RunCtx2>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Output>>, ThoosaError>;
}

#[async_trait]
impl<SRC> MotifLoaderStage for MotifLoader<SRC, RunCtx2>
where
    SRC: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2>  + 'static,
{
    type Output = MotifWithData;

    async fn run_cached(
        self: Arc<Self>,
        input: Arc<DataCacheWrapper<DataId>>,
        ctx: Arc<RwLock<RunCtx2>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Output>>, ThoosaError> {
        <Self as Cached>::run_cached(self, input, ctx).await
    }
}



#[derive(Cached)]
pub struct LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + SyncStage
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: ClcMotifInput,
    CLC::In<'static>: 'static,
{
    id: SimpleId,
    motif_loader: Arc<dyn MotifLoaderStage<Output = <CLC as Stage>::In<'static>>>,
    clc_dist: Arc<CLC>,
    output_builder:
        Box<dyn Fn(<Self as WithId>::ID) -> Box<dyn OutputBuilder<O = AverageDistanceValue>> + Send + Sync>,
}
impl<CLC> Debug for LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + SyncStage
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: ClcMotifInput,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadMotifAndClcDistanceStage").finish()
    }
}

impl<CLC> WithId for LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + SyncStage
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: ClcMotifInput,
{
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<CLC> LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + SyncStage
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: ClcMotifInput,
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
        + SyncStage
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: ClcMotifInput,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("{}_avg_ec_dist", input_id))
    }
}



impl<CLC> Stage for LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + SyncStage
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: ClcMotifInput,
{
    type In<'a> = DataId;

    type Out = AverageDistanceValue;
    type Ctx = RunCtx2;
}

#[async_trait]
impl<CLC> AsyncStage for LoadMotifAndClcDistanceStage<CLC>
where
    CLC: Stage<Ctx = RunCtx2, Out = AverageDistanceValue>
        + SyncStage
        + Adjustable
        + OutId
        + Send
        + Sync
        + 'static,
    for<'a> CLC::In<'a>: ClcMotifInput,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DataCacheWrapper<Self::In<'static>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let out_id = self.out_id(input.id());

        let motif = Arc::clone(&self.motif_loader)
            .run_cached(Arc::clone(&input), Arc::clone(&ctx))
            .await?;

        let motif_size = motif.size().clone();

        debug!(
            "LoadMotifAndClcDistanceStage::run: out_id={} motif_size={motif_size:?}",
            out_id
        );

        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };
        let num_devices = GlobalCtx::global().device_count();

        let dist = OOMHandler::new(
            vec![Adjusters::Input(Box::new(
                SplitsAdjuster::<<CLC as Stage>::In<'static>, MotifSlicesSelector>::new(
                    [2, -1].into(),
                ),
            ))],
            None,
        )
        .run(
            &self.clc_dist,
            Arc::clone(&motif),
            Arc::clone(&ctx),
            Contexts::new(
                SplitsAdjusterContextImpl::new(
                    motif_size.clone(),
                    (self.output_builder)(out_id.clone()),
                ),
                NoopRunnableContext {},
                NoopRunCtxContext {},
            ),
        )
        .await?;

        Ok(Arc::new(DataCacheWrapper::new(
            dist,
            device_idx,
            num_devices,
            cache_area,
        )))
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct MotifSlicesSelector {
    slices: Range<MonoMetrics<0, 2>>
}

impl From<Range<MonoMetrics<0, 2>>>  for MotifSlicesSelector{
    fn from(value: Range<MonoMetrics<0, 2>>) -> Self {
        MotifSlicesSelector { slices: value }
    }
}

impl Adjustable for MotifSlicesSelector  {}


impl Select<MotifSlicesSelector> for ThoTensor2<MotifSlices, DataId> {
    type Output = ThoTensor2<MotifSlices, DataId>;

    fn subview(&self, selector: MotifSlicesSelector) -> Result<Self::Output, ThoosaError> {

        let size: MonoMetrics<0, 2> = self
            .size()
            .try_into()?;

        let ranges = MonoMetrics::<0,2>::map_to_ranges(&selector.slices.start, &selector.slices.end)?;

        let sub_tensor = self.t_ref().f_i((ranges[0].clone(), ranges[1].clone()))?;

        Ok(ThoTensor2::new(self.id().clone(), sub_tensor))
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



pub(crate) trait ClcMotifInput:
    WithId
    + Data<Metrics = MonoMetrics<0, 2>>
    + CopyToDevice
    + Select<MotifSlicesSelector, Output = Self>
    + Adjustable
    + Debug
    + Send
    + Sync
{
}

// impl ClcMotifInput for ThoTensor2<MotifSlices, DataId> {}

#[cfg(test)]
mod tests {
    use super::{LoadMotifAndClcDistanceStage, MotifLoaderStage, MotifSlices, RunCtx2};
    use caching_runner::common::cachable::{Cached, DataCacheWrapper};
    use caching_runner::common::RunCtx;
    use caching_runner::global_ctx::GlobalCtx;
    use parking_lot::{RwLock, Mutex};
    use std::sync::{Arc};
    use tch::Tensor;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::traits::{DataId, SimpleId, WithId};
    use torch_runner::data::ThoTensor2;
    use crate::tasks::average_euclidean_distance::mock_average_stage;
    use crate::tasks::motif_loader::{mock_motif_loader, wrap_data_id_for_cache, MotifWithData};

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
            MotifWithData::new(motif_tensor, 0, 1),
            0,
            GlobalCtx::global().device_count(),
            0,
        ));

        let motif_loader = mock_motif_loader::build()
            .with_result(Arc::clone(&motif_wrapper))
            .finish();

        let call_log = Arc::new(Mutex::new(Vec::new()));

        let clc_stage = mock_average_stage::builder()
            .fail_while_dim0_gt(3)
            .with_outputs(vec![10.0, 30.0])
            .with_call_log(Arc::clone(&call_log))
            .finish();

        let stage = Arc::new(LoadMotifAndClcDistanceStage::new(
            motif_loader as Arc<dyn MotifLoaderStage<Output = MotifWithData>>,
            clc_stage,
        ));

        let cache_area = ctx.read().cache_area();
        let num_devices = GlobalCtx::global().device_count();
        let input_id_value = DataId::from(&42u32);
        let input_id = wrap_data_id_for_cache(input_id_value.clone(), num_devices, cache_area);
        let result = stage
            .run_cached(Arc::clone(&input_id), Arc::clone(&ctx))
            .await?;

        let result_value = result
            .with_data_ref(0, None, |val| Ok(val.clone()))
            .await?;

        assert_eq!(result_value.value(), 20.0);
        let expected_id = SimpleId::from(format!("{}_avg_ec_dist", input_id_value));
        assert_eq!(result_value.id(), &expected_id);

        let recorded = call_log.lock().clone();
        assert!(recorded.iter().any(|&d| d > 3), "expected at least one oversized run, got {recorded:?}");

        let split_calls = recorded.iter().filter(|&&d| d == 3).count();
        assert_eq!(split_calls, 2, "expected two split runs, got {recorded:?}");
        assert_eq!(recorded.last().copied(), Some(3));

        Ok(())
    }

}
