use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
use crate::tasks::run_ctx_2::RunCtx2;
use async_trait::async_trait;
use cached_derive::Cached;
use caching_runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContext, SplitsAdjusterContextImpl};
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::global_ctx::GlobalCtx;
use caching_runner::oom_handler::{Adjuster, Adjusters, Contexts, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use log::debug;
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{Add, Deref, DivAssign, Range};
use std::pin::Pin;
use std::sync::Arc;
use tch::IndexOp;
use thoosa_common::dims::Dims;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::MonoMetrics;
use thoosa_shared::traits::{Adjustable, Data, DataId, Select, SimpleId, WithId};
use thoosa_shared::{CopyToDevice, ThoosaError};
use torch_runner::data::ThoTensor2;


#[derive(Cached)]
pub struct LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = DataId,
            Out = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    id: SimpleId,
    motif_loader: Arc<ML>,
    clc_dist: Arc<CLC>,
    output_builder: Box<dyn Fn(<Self as WithId>::ID) -> Box<dyn OutputBuilder<O = AverageDistanceValue>> + Send + Sync>,
    _ctx: PhantomData<C>,
}
impl<ML, CLC, C> Debug for LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = DataId,
            Out = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadMotifAndClcDistanceStage").finish()
    }
}

impl<ML, CLC, C> WithId for LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = DataId,
            Out = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<ML, CLC, C> LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = DataId,
            Out = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    pub fn new(motif_loader: Arc<ML>, clc_dist: Arc<CLC>) -> Self {
        Self {
            id: SimpleId("LoadMotifAndClcDistanceStage"),
            motif_loader,
            clc_dist,
            output_builder: Box::new(|id| Box::new(DistOutputBuilder::new(id))),
            _ctx: PhantomData,
        }
    }
}

impl<ML, CLC, C> OutId for LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = DataId,
            Out = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("{}_avg_ec_dist", input_id))
    }
}

impl<ML, CLC, C> Stage for LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = DataId,
            Out = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    type In<'a> = DataId;

    type Out = AverageDistanceValue;
    type Ctx = C;
}

#[async_trait]
impl<ML, CLC, C> AsyncStage for LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = DataId,
            Out = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>
    {
        let out_id = self.out_id(input.id());

        let motif = Arc::clone(&self.motif_loader).run_cached(Arc::clone(&input), Arc::clone(&ctx)).await?;

        let motif_size = motif.size().clone();

        debug!("LoadMotifAndClcDistanceStage::run: out_id={} motif_size={motif_size:?}", out_id);

        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };
        let num_devices = GlobalCtx::global().device_count();

        let dist = OOMHandler::new(
            vec![
                Adjusters::Input(Box::new(SplitsAdjuster::<DataWithExtra<ThoTensor2<MotifSlices, DataId>, MotifMeta>, MotifSlicesSelector>::new([2, -1].into())))
            ], None)
            .run(
                &self.clc_dist,
                Arc::clone(&motif),
                Arc::clone(&ctx),
                Contexts::new(SplitsAdjusterContextImpl::new(motif_size.clone(), (self.output_builder)(out_id.clone())), NoopRunnableContext {}, NoopRunCtxContext {}),
            )
            .await?;

        Ok(Arc::new(DevicesWrap::wrap_cpu(dist)))
    }
}


#[derive(Clone, Debug, Default)]
pub(crate) struct MotifSlicesSelector {
    slices: Range<MonoMetrics<0, 2>>,
}

impl From<Range<MonoMetrics<0, 2>>> for MotifSlicesSelector {
    fn from(value: Range<MonoMetrics<0, 2>>) -> Self {
        MotifSlicesSelector { slices: value }
    }
}

impl Adjustable for MotifSlicesSelector {}

impl Select<MotifSlicesSelector> for ThoTensor2<MotifSlices, DataId> {
    type Output = ThoTensor2<MotifSlices, DataId>;

    fn subview(&self, selector: MotifSlicesSelector) -> Result<Self::Output, ThoosaError> {
        let size: MonoMetrics<0, 2> = self.size().try_into()?;

        let ranges = MonoMetrics::<0, 2>::map_to_ranges(&selector.slices.start, &selector.slices.end)?;

        let sub_tensor = self.t_ref().f_i((ranges[0].clone(), ranges[1].clone()))?;

        Ok(ThoTensor2::new(self.id().clone(), sub_tensor))
    }
}

#[derive(Debug)]
pub struct DistOutputBuilder {
    id: SimpleId,
    values: Vec<f32>,
}

impl DistOutputBuilder {
    pub fn new(id: SimpleId) -> Self {
        Self { id, values: vec![] }
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

        AverageDistanceValue::new(self.id.clone(), if self.values.is_empty() { 0.0 } else { self.values.iter().sum::<f32>() / self.values.len() as f32 })
    }
}


#[cfg(test)]
mod tests {
    use super::{LoadMotifAndClcDistanceStage, RunCtx2};
    use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use crate::tasks::average_euclidean_distance::{mock_average_stage, AverageDistanceValue};
    use crate::tasks::motif_loader::{MotifMeta, mock_motif_loader};
    use caching_runner::common::{mock_run_ctx, MockRunCtx, RunCtx};
    use caching_runner::global_ctx::GlobalCtx;
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use tch::Tensor;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::traits::{DataId, SimpleId, WithId};
    use torch_runner::data::ThoTensor2;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_and_clc_average_euclidean_distance() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));


        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (42u32, (MotifMeta { channel: 0, run_id: 1, window: 3 }, Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]).view((6, 3))))
        ]).finish());

        let call_log = Arc::new(Mutex::new(Vec::new()));

        let clc_stage = Arc::new(mock_average_stage::builder().fail_while_dim0_gt(3).with_do_run(vec![
            AverageDistanceValue::new("0".into(), 10.0),
            AverageDistanceValue::new("1".into(), 30.0)
        ]).with_call_log(Arc::clone(&call_log)).finish());

        let stage = Arc::new(LoadMotifAndClcDistanceStage::new(
            motif_loader,
            clc_stage,
        ));

        let cache_area = ctx.read().cache_area();
        let num_devices = GlobalCtx::global().device_count();
        let input_id_value = DataId::from(&42u32);
        let input_id = Arc::new(DevicesWrap::wrap_cpu(input_id_value.clone()));
        let result = stage.run_cached(Arc::clone(&input_id), Arc::clone(&ctx)).await?;

        let result_value: AverageDistanceValue = result.with_data_ref(0, None, |val: &AverageDistanceValue| Ok(val.clone())).await?;

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
