use crate::tasks::cross_dims::MotifSlices;
use crate::tasks::motif_loader::{MotifLoaderInput, MotifMeta};
use crate::tasks::slices_stats::{Aggr, ScalarStatsValue, Metrics, SlicesStats, SlicesStatsInput};
use cached_derive::Cached;
use log::debug;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use tch::IndexOp;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::Resource;
use thoosa_shared::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::oom_handler::{Adjusters, Contexts, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::traits::{Adjustable, Data, DataId, MonoMetrics, Select, WithId};
use thoosa_shared::{mono_key, ThoError};
use thoosa_with_data::with_data;

#[derive(Debug, Clone, DummyData)]
pub struct LoadMotifAndClcDistanceInput {
    id: SynKey,
    motif_loader_input: MotifLoaderInput,
    metrics: Metrics,
    aggr: Aggr,
}

impl LoadMotifAndClcDistanceInput {
    pub fn new(motif_loader_input: MotifLoaderInput, metrics: Metrics, aggr: Aggr) -> Self {
        Self {
            id: SynKey::new([
                mono_key!(motif_loader_input.id().0, format!("motif_id={}", motif_loader_input.id().0)),
                mono_key!(format!("{metrics:?}"), format!("metrics={metrics:?}")),
                mono_key!(format!("{aggr:?}"), format!("aggr={aggr:?}")),//TODO f32 should not be in key
            ]),
            motif_loader_input,
            metrics,
            aggr,
        }
    }

    pub fn motif_loader_input(&self) -> &MotifLoaderInput {
        &self.motif_loader_input
    }

    pub fn metrics(&self) -> Metrics {
        self.metrics
    }

    pub fn aggr(&self) -> Aggr {
        self.aggr
    }
}

impl Input for LoadMotifAndClcDistanceInput {}

impl WithId for LoadMotifAndClcDistanceInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Named)]//Cached,  is removed because input contains f32 in LoadMotifAndClcDistanceInput::Aggr
pub struct LoadMotifAndClcDistanceStage<ML, CLC, C>
where
    C: RunCtx + Adjustable + 'static,
    for<'i> ML: AsyncStage<
            In<'i> = MotifLoaderInput,
            Out = DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
            Ctx = C,
        > + Cached + OutId + 'static,

    for<'i> CLC: SyncStage<In<'i> = SlicesStatsInput, Out =ScalarStatsValue, Ctx = C> + Adjustable + OutId + Sync + Send + 'static,
{
    id: SynKey,
    pub motif_loader: Arc<ML>,
    clc_dist: Arc<CLC>,
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

    for<'i> CLC: SyncStage<In<'i> = SlicesStatsInput, Out =ScalarStatsValue, Ctx = C> + Adjustable + OutId + Sync + Send + 'static,
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

    for<'i> CLC: SyncStage<In<'i> = SlicesStatsInput, Out =ScalarStatsValue, Ctx = C> + Adjustable + OutId + Sync + Send + 'static,
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

    for<'i> CLC: SyncStage<In<'i> = SlicesStatsInput, Out =ScalarStatsValue, Ctx = C> + Adjustable + OutId + Sync + Send + 'static,
{
    pub fn new(motif_loader: Arc<ML>, clc_dist: Arc<CLC>) -> Self {
        Self {
            id: SynKey::new([mono_key!("LoadMotifAndClcDistanceStage", "stage")]),
            motif_loader,
            clc_dist,
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

    for<'i> CLC: SyncStage<In<'i> = SlicesStatsInput, Out =ScalarStatsValue, Ctx = C> + Adjustable + OutId + Sync + Send + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SynKey::new([mono_key!(format!("{input_id:?}"), format!("{input_id:?}_avg_ec_dist"))])
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

    for<'i> CLC: SyncStage<In<'i> = SlicesStatsInput, Out =ScalarStatsValue, Ctx = C> + Adjustable + OutId + Sync + Send + 'static,
{
    type In<'a> = LoadMotifAndClcDistanceInput;

    type Out = ScalarStatsValue;
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

    for<'i> CLC: SyncStage<In<'i> = SlicesStatsInput, Out =ScalarStatsValue, Ctx = C> + Adjustable + OutId + Sync + Send + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="LoadMotifAndClcDistanceStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let out_id = self.out_id(input.id());

        let (motif_loader_input, metrics, aggr) = with_data!(ctx.branch(Some("load_motif_and_clc_dist_input"), false, false), input; |_ctx, input: &LoadMotifAndClcDistanceInput| {
            Ok::<(MotifLoaderInput, Metrics, Aggr), ThoError>((input.motif_loader_input().clone(), input.metrics(), input.aggr()))
        })?;

        let motif_input = Arc::new(DevicesWrap::wrap(motif_loader_input, input.cache_area().clone()));
        let motif = Arc::clone(&self.motif_loader).run_cached(Arc::clone(&motif_input), ctx.branch(Some("load_motif_and_clc_dist_motif"), false, false)).await?;

        let stats_input = with_data!(ctx.branch(Some("load_motif_and_clc_dist_stats_input"), false, false), motif; |_ctx, motif: &DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>| {
            let motif_tensor = ThoTensor::new(motif.data().id().clone(), motif.data().t_ref().shallow_clone())?;//TODO add *Wrapper like ClcResourcesMotifsDistInputDeviceWrapper
            Ok::<SlicesStatsInput, ThoError>(SlicesStatsInput::new(metrics, aggr, DataWithExtra::new(motif_tensor, motif.extra().clone())))
        })?;

        let stats_input = Arc::new(DevicesWrap::wrap(stats_input, input.cache_area().clone()));
        let stats_input_size = stats_input.size().clone();

        debug!("LoadMotifAndClcDistanceStage::run: out_id={out_id:?} stats_input_size={stats_input_size:?} metrics={metrics:?} aggr={aggr:?}");

        let dist = OOMHandler::new(vec![Adjusters::Input(Box::new(SplitsAdjuster::<MotifSlicesSelector, MonoMetrics<0, 2>>::new([2, -1].into())))], None)
            .run(
                &self.clc_dist,
                Arc::clone(&stats_input),
                ctx.branch(Some("load_motif_and_clc_dist_dist"), false, false),
                Contexts::new(
                    SplitsAdjusterContextImpl::new(stats_input_size, DistOutputBuilder::new(out_id.clone(), aggr)),
                    NoopRunnableContext {},
                    NoopRunCtxContext {},
                ),
            )
            .await?;

        Ok(Arc::new(DevicesWrap::wrap(dist, CacheArea::default())))
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
        let ranges = MonoMetrics::<0, 2>::map_to_ranges(&selector.slices.start, &selector.slices.end)?;
        let sub_tensor = self.t_ref().f_i((ranges[0].clone(), ranges[1].clone()))?;

        Ok(ThoTensor::new(self.id().clone(), sub_tensor)?)
    }
}

#[derive(Debug)]
pub struct DistOutputBuilder {
    id: SynKey,
    values: Vec<f32>,
    aggr: Aggr,
}

impl DistOutputBuilder {
    pub fn new(id: SynKey, aggr: Aggr) -> Self {
        Self { id, values: vec![], aggr }
    }
}

impl OutputBuilder for DistOutputBuilder {
    type O = ScalarStatsValue;

    async fn add_partial_output(&mut self, o: Self::O) {
        debug!("DistOutputBuilder::add_partial_output: o={o:?}");
        self.values.push(o.value());
    }

    async fn build(self) -> Result<Self::O, ThoError> {
        debug!("DistOutputBuilder::build: self.values={:?} self.aggr={:?}", self.values, self.aggr);

        let value = if self.values.is_empty() {
            0.0
        } else {
            match self.aggr {
                Aggr::Mean(k) => k * self.values.iter().sum::<f32>() / self.values.len() as f32,
                Aggr::Min(k) => k * self.values.iter().copied().reduce(f32::min).unwrap_or(0.0),
                Aggr::Max(k) => k * self.values.iter().copied().reduce(f32::max).unwrap_or(0.0),
                Aggr::Quantile(_) => self.values.iter().copied().reduce(f32::max).unwrap_or(0.0),
            }
        };

        Ok(ScalarStatsValue::new(self.id.clone(), value))
    }
}

#[cfg(test)]
mod tests {
    use super::{DistOutputBuilder, LoadMotifAndClcDistanceInput, LoadMotifAndClcDistanceStage};
    use crate::tasks::motif_loader::{mock_motif_loader, MotifLoaderInput, MotifMeta};
    use crate::tasks::slices_stats::{mock_average_stage, Aggr, ScalarStatsValue, Metrics};
    use parking_lot::{Mutex, RwLock};
    use std::sync::Arc;
    use tch::Tensor;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use thoosa_shared::runner::common::{mock_run_ctx, OutId, RunCtx};
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_shared::traits::{DataId, WithId};
    use thoosa_shared::ThoError;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_load_and_clc_average_euclidean_distance() -> Result<(), ThoError> {
        GlobalCtx::init_devices(1);

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(0)));

        let motif_loader = Arc::new(
            mock_motif_loader::build()
                .with_run_cached([(
                    MotifLoaderInput::all_slices(DataId(42u32)),
                    (
                        MotifMeta { channel: 0, find_motifs_run_id: 1, window: 3 },
                        Tensor::from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]).view((6, 3)),
                    ),
                )])
                .finish(),
        );

        let call_log = Arc::new(Mutex::new(Vec::new()));

        let clc_stage = Arc::new(
            mock_average_stage::builder()
                .fail_while_dim0_gt(3)
                .with_do_run(vec![
                    ScalarStatsValue::new(SynKey::new([mono_key!(0u32, "avg_0")]), 10.0),
                    ScalarStatsValue::new(SynKey::new([mono_key!(1u32, "avg_1")]), 30.0),
                ])
                .with_call_log(Arc::clone(&call_log))
                .finish(),
        );

        let stage = Arc::new(LoadMotifAndClcDistanceStage::new(motif_loader, clc_stage));

        let input = Arc::new(DevicesWrap::wrap(
            LoadMotifAndClcDistanceInput::new(MotifLoaderInput::all_slices(DataId::from(&42u32)), Metrics::EucDist, Aggr::Mean(0.1)),
            CacheArea::default(),
        ));
        let expected_id = stage.out_id(input.id());

        let stage_ctx = ctx.write().branch(Some("load_motif_and_clc_dist"), false, false);
        let result = stage.run/*_cached*/(Arc::clone(&input), stage_ctx).await?;

        let mut ctx_for_data = ctx.write().branch(Some("load_motif_and_clc_dist_data"), false, false);
        let result_value: ScalarStatsValue = result.with_data_ref(0, ctx_for_data, None, |val: &ScalarStatsValue, _ctx| Ok(val.clone())).await?;

        assert_eq!(result_value.value(), 20.0);
        assert_eq!(result_value.id(), &expected_id);

        let recorded = call_log.lock().clone();
        assert!(recorded.iter().any(|&d| d > 3), "expected at least one oversized run, got {recorded:?}");

        let split_calls = recorded.iter().filter(|&&d| d == 3).count();
        assert_eq!(split_calls, 2, "expected two split runs, got {recorded:?}");
        assert_eq!(recorded.last().copied(), Some(3));

        Ok(())
    }
}
