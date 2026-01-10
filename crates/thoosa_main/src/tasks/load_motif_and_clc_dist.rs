use std::cmp::Ordering;
use crate::tasks::average_euclidean_distance::{AverageDistanceValue, AverageEuclideanDistanceStage};
use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
use crate::tasks::run_ctx_2::RunCtx2;
use async_trait::async_trait;
use cached_derive::Cached;
use thoosa_named_derive::Named;
use thoosa_shared::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContext, SplitsAdjusterContextImpl};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::oom_handler::{Adjuster, Adjusters, Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use log::debug;
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
use thoosa_shared::traits::{Adjustable, Data, DataId, Select, WithId};
use thoosa_shared::{mono_key, ThoError};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::comp_id::SynKey;

#[derive(Cached, Named)]
pub struct LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = MotifLoaderInput,
            Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    id: SynKey,
    pub motif_loader: Arc<ML>,
    clc_dist: Arc<CLC>,
    // output_builder: Box<dyn Fn(<Self as WithId>::ID) -> Box<dyn OutputBuilder<O = AverageDistanceValue>> + Send + Sync>,
    _ctx: PhantomData<C>,
}
impl<ML, CLC, C> Debug for LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = MotifLoaderInput,
            Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
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
            In<'i> = MotifLoaderInput,
            Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<ML, CLC, C> LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = MotifLoaderInput,
            Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    pub fn new(motif_loader: Arc<ML>, clc_dist: Arc<CLC>) -> Self {
        Self {
            id: SynKey::new([mono_key!("LoadMotifAndClcDistanceStage", "stage")]),
            motif_loader,
            clc_dist,
            // output_builder: Box::new(|id| Box::new(DistOutputBuilder::new(id))),
            _ctx: PhantomData,
        }
    }
}

impl<ML, CLC, C> OutId for LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = MotifLoaderInput,
            Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SynKey::new([mono_key!(input_id.0, format!("{input_id}_avg_ec_dist"))])
    }
}

impl<ML, CLC, C> Stage for LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = MotifLoaderInput,
            Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    type In<'a> = MotifLoaderInput;

    type Out = AverageDistanceValue;
    type Ctx = C;
}

// #[async_trait]
impl<ML, CLC, C> AsyncStage for LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = MotifLoaderInput,
            Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<
            In<'i> = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Out = AverageDistanceValue,
            Ctx = C,
        > + Adjustable + OutId + Sync + Send + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="LoadMotifAndClcDistanceStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    {
        let out_id = self.out_id(input.id());
        let mut ctx = ctx;
        let motif_ctx = ctx.branch("load_motif_and_clc_dist_motif", false);
        let motif = Arc::clone(&self.motif_loader).run_cached(Arc::clone(&input), motif_ctx).await?;

        let motif_size = motif.size().clone();

        debug!("LoadMotifAndClcDistanceStage::run: out_id={out_id:?} motif_size={motif_size:?}");


        let dist_ctx = ctx.branch("load_motif_and_clc_dist_dist", false);
        let dist = OOMHandler::new(
            vec![
                Adjusters::Input(Box::new(SplitsAdjuster::<MotifSlicesSelector, MonoMetrics<0, 2>>::new([2, -1].into())))
            ], None)
            .run(
                &self.clc_dist,
                Arc::clone(&motif),
                dist_ctx,
                Contexts::new(
                    SplitsAdjusterContextImpl::new(motif_size.clone(), DistOutputBuilder::new(out_id.clone())),
                    // NoopInputContext(None),
                    NoopRunnableContext {},
                    NoopRunCtxContext {}
                ),
            )
            .await?;

        Ok(Arc::new(DevicesWrap::wrap(dist, 0)))
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

impl Select<MotifSlicesSelector> for ThoTensor<MotifSlices, DataId> {
    type Output = ThoTensor<MotifSlices, DataId>;

    fn subview(&self, selector: MotifSlicesSelector) -> Result<Self::Output, ThoError> {
        let size: MonoMetrics<0, 2> = self.size().try_into()?;

        let ranges = MonoMetrics::<0, 2>::map_to_ranges(&selector.slices.start, &selector.slices.end)?;

        let sub_tensor = self.t_ref().f_i((ranges[0].clone(), ranges[1].clone()))?;

        Ok(ThoTensor::new(self.id().clone(), sub_tensor)?)
    }
}

#[derive(Debug)]
pub struct DistOutputBuilder {
    id: SynKey,
    values: Vec<f32>,
}

impl DistOutputBuilder {
    pub fn new(id: SynKey) -> Self {
        Self { id, values: vec![] }
    }
}

impl OutputBuilder for DistOutputBuilder {
    type O = AverageDistanceValue;

    async fn add_partial_output(&mut self, o: Self::O) {
        debug!("DistOutputBuilder::add_partial_output: o={o:?}");
        self.values.push(o.value());
    }

    async fn build(self) -> Result<Self::O, ThoError> {
        debug!("DistOutputBuilder::build: self.values={:?}", self.values);

        Ok(AverageDistanceValue::new(self.id.clone(), if self.values.is_empty() { 0.0 } else { self.values.iter().sum::<f32>() / self.values.len() as f32 }))
    }
}


#[cfg(test)]
mod tests {
    use super::LoadMotifAndClcDistanceStage;
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use crate::tasks::average_euclidean_distance::{mock_average_stage, AverageDistanceValue};
    use crate::tasks::motif_loader::{mock_motif_loader, MotifLoaderInput, MotifMeta};
    use thoosa_shared::runner::common::{mock_run_ctx, RunCtx};
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use tch::Tensor;
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::{DataId, WithId};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_and_clc_average_euclidean_distance() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(1)));


        let motif_loader = Arc::new(mock_motif_loader::build().with_run_cached([
            (MotifLoaderInput::all_slices(DataId(42u32)), (MotifMeta { channel: 0, find_motifs_run_id: 1, window: 3 }, Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]).view((6, 3))))
        ]).finish());

        let call_log = Arc::new(Mutex::new(Vec::new()));

        let clc_stage = Arc::new(mock_average_stage::builder().fail_while_dim0_gt(3).with_do_run(vec![
            AverageDistanceValue::new(SynKey::new([mono_key!(0u32, "avg_0")]), 10.0),
            AverageDistanceValue::new(SynKey::new([mono_key!(1u32, "avg_1")]), 30.0)
        ]).with_call_log(Arc::clone(&call_log)).finish());

        let stage = Arc::new(LoadMotifAndClcDistanceStage::new(
            motif_loader,
            clc_stage,
        ));

        let cache_area = ctx.read().cache_area();
        let num_devices = GlobalCtx::global().device_count();
        let input_id_value = DataId::from(&42u32);
        let input_id = Arc::new(DevicesWrap::wrap(MotifLoaderInput::all_slices(input_id_value.clone()), 0));
        let stage_ctx = ctx.write().branch("load_motif_and_clc_dist", false);
        let result = stage.run_cached(Arc::clone(&input_id), stage_ctx).await?;

        let mut ctx_for_data = ctx.write().branch("load_motif_and_clc_dist_data", false);
        let result_value: AverageDistanceValue = result.with_data_ref(0, &mut ctx_for_data, None, |val: &AverageDistanceValue, _ctx| Ok(val.clone())).await?;

        assert_eq!(result_value.value(), 20.0);
        let expected_id = SynKey::new([mono_key!(input_id_value.0, format!("{input_id_value}_avg_ec_dist"))]);
        assert_eq!(result_value.id(), &expected_id);

        let recorded = call_log.lock().clone();
        assert!(recorded.iter().any(|&d| d > 3), "expected at least one oversized run, got {recorded:?}");

        let split_calls = recorded.iter().filter(|&&d| d == 3).count();
        assert_eq!(split_calls, 2, "expected two split runs, got {recorded:?}");
        assert_eq!(recorded.last().copied(), Some(3));

        Ok(())
    }
}
