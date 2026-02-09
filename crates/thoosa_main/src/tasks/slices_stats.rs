use crate::tasks::cross_dims::MotifSlices;
use crate::tasks::load_motif_and_clc_dist::MotifSlicesSelector;
use crate::tasks::motif_loader::MotifMeta;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Kind;
use tch::Tensor;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::runner::adjusters::splits::OutputBuilder;
use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::traits::{Adjustable, Data, DataId, MonoMetrics, Select, WithId};
use thoosa_shared::{mono_key, panic_to_err, CopyToDevice, ThoError};
use thoosa_motifs::pearson_corr::pearson_corr_distance_matrix_0_2_single as pearson_corr_distance_matrix_0_2_single_matrix;

#[derive(Debug, Copy, Clone, DummyData)]
pub enum Metrics {
    EucDist,
    PearsonCorr,//not used yet
}

impl Default for Metrics {
    fn default() -> Self {
        Self::EucDist
    }
}

#[derive(Debug, Copy, Clone, DummyData)]
pub enum Aggr {
    Mean(f32),
    Quantile(f32),
}

impl Default for Aggr {
    fn default() -> Self {
        Self::Mean(0.1)
    }
}

#[derive(Debug)]
pub struct SlicesStatsInput {
    metrics: Metrics,
    aggr: Aggr,
    slices: DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>,
}

impl SlicesStatsInput {
    pub fn new(metrics: Metrics, aggr: Aggr, slices: DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>) -> Self {
        Self { metrics, aggr, slices }
    }

    pub fn metrics(&self) -> Metrics {
        self.metrics
    }

    pub fn aggr(&self) -> Aggr {
        self.aggr
    }

    pub fn slices(&self) -> &DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta> {
        &self.slices
    }
}

#[async_trait]
impl CopyToDevice for SlicesStatsInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(Self {
            metrics: self.metrics,
            aggr: self.aggr,
            slices: self.slices.copy_to_device(dst_device).await?,
        })
    }
}

impl WithId for SlicesStatsInput {
    type ID = DataId;

    fn id(&self) -> &Self::ID {
        self.slices.id()
    }
}

impl Data for SlicesStatsInput {
    type Metrics = <DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta> as Data>::Metrics;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        self.slices.size()
    }

    fn device(&self) -> usize {
        self.slices.device()
    }
}

impl Input for SlicesStatsInput {}

impl Select<MotifSlicesSelector> for SlicesStatsInput {
    type Output = SlicesStatsInput;

    fn subview(&self, selector: MotifSlicesSelector) -> Result<Self::Output, ThoError> {
        Ok(Self {
            metrics: self.metrics,
            aggr: self.aggr,
            slices: self.slices.subview(selector)?,
        })
    }
}

impl Adjustable for SlicesStatsInput {}

pub struct SlicesStats<C>
where
    C: RunCtx + 'static,
{
    id: SynKey,
    _ctx: PhantomData<C>,
}

impl<C> Debug for SlicesStats<C>
where
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SlicesStats").finish()
    }
}

impl<C> Adjustable for SlicesStats<C> where C: RunCtx + 'static {}

impl<C> WithId for SlicesStats<C>
where
    C: RunCtx + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<C> SlicesStats<C>
where
    C: RunCtx + 'static,
{
    pub fn new() -> Self {
        Self {
            id: SynKey::new([mono_key!("SlicesStats", "stage")]),
            _ctx: PhantomData,
        }
    }
}

#[derive(Debug, Clone, DummyData)]
#[dummy_data(clone)]
pub struct ScalarStatsValue {
    //TODO add type
    id: SynKey,
    value: f32,
}

impl ScalarStatsValue {
    pub fn new(id: SynKey, value: f32) -> Self {
        Self { id, value }
    }

    pub fn value(&self) -> f32 {
        self.value
    }
}

impl WithId for ScalarStatsValue {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
impl Output for ScalarStatsValue {}

#[async_trait]
impl CopyToDevice for ScalarStatsValue {
    async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
        Ok(self.clone())
    }
}

impl<C> OutId for SlicesStats<C>
where
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SynKey::new([mono_key!(input_id.0, format!("{input_id}_avg_ec_dist"))])
    }
}

impl<C> Stage for SlicesStats<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = SlicesStatsInput;
    type Out = ScalarStatsValue;
}
impl<C> SyncStage for SlicesStats<C>
where
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SlicesStats::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let value = match input.metrics() {
            Metrics::EucDist => panic_to_err!(euclidean_distance(input.slices().data().t_ref(), input.aggr()))?,
            Metrics::PearsonCorr => panic_to_err!(pearson_corr_distance_matrix_0_2_single(input.slices().data().t_ref(), 1e-8, input.aggr()))?,
        };

        Ok(ScalarStatsValue::new(SynKey::new([mono_key!("AverageDistanceValue", "AverageDistanceValue")]), value))
    }
}

pub fn euclidean_distance(t: &Tensor, aggr: Aggr) -> Result<f32, ThoError> {
    let n = t.size()[0];
    if n < 2 {
        return Ok(0.0);
    }

    let squared_norms = t.f_pow_tensor_scalar(2)?.f_sum_dim_intlist(1, true, Kind::Float)?;
    let squared_norms_sum = squared_norms.f_add(&squared_norms.f_transpose(0, 1)?)?;

    let gram = t.f_mm(&t.f_transpose(0, 1)?)?;
    let gram = gram.f_add(&gram)?;

    let dists_squared = squared_norms_sum.f_subtract(&gram)?.f_clamp_min(0.0)?;

    let triu_indices = Tensor::f_triu_indices(n, n, 1, (Kind::Int64, t.device()))?;
    let i = triu_indices.f_get(0)?;
    let j = triu_indices.f_get(1)?;

    let pairwise_squared_dists = dists_squared.f_index_select(0, &i)?.f_gather(1, &j.f_unsqueeze(1)?, false)?.f_squeeze()?;
    let distances = pairwise_squared_dists.f_sqrt()?;

    aggregate_distances(&distances, aggr)
}

pub fn pearson_corr_distance_matrix_0_2_single(x: &Tensor, eps: f64, aggr: Aggr) -> Result<f32, ThoError> {
    let sx = x.size();
    if sx[0] < 2 {
        return Ok(0.0);
    }

    let matrix = pearson_corr_distance_matrix_0_2_single_matrix(x, eps)?;
    let pairwise = upper_triangle_without_diagonal(&matrix)?;

    aggregate_distances(&pairwise, aggr)
}

fn upper_triangle_without_diagonal(matrix: &Tensor) -> Result<Tensor, ThoError> {
    let n = matrix.size()[0];
    let triu_indices = Tensor::f_triu_indices(n, n, 1, (Kind::Int64, matrix.device()))?;
    let i = triu_indices.f_get(0)?;
    let j = triu_indices.f_get(1)?;

    Ok(matrix.f_index_select(0, &i)?.f_gather(1, &j.f_unsqueeze(1)?, false)?.f_squeeze()?)
}

fn aggregate_distances(distances: &Tensor, aggr: Aggr) -> Result<f32, ThoError> {
    if distances.numel() == 0 {
        return Ok(0.0);
    }

    let scalar = match aggr {
        Aggr::Mean(k) =>k * distances.f_mean(Kind::Float)?,
        Aggr::Quantile(q) => {
            let q = q.clamp(0.0, 1.0) as f64;
            let quantile = distances.f_quantile_scalar(q, 0, true, "lower")?;
            let mask = distances.f_le_tensor(&quantile)?;
            let selected = distances.f_masked_select(&mask)?;
            if selected.numel() == 0 {
                quantile
            } else {
                selected.f_max()?
            }
        }
    };

    Ok(scalar.f_double_value(&[])? as f32)
}

#[cfg(test)]
mod tests {
    use log::debug;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_debug() {
        let syn_key: SynKey = [mono_key!(0, "aaa"), mono_key!(1, "bbb"), mono_key!(2, "ccc"), mono_key!("ddd", "ddd")].into();

        debug!("syn_key: {syn_key:?}");
    }
}
#[cfg(test)]
pub mod mock_average_stage {
    use crate::tasks::cross_dims::MotifSlices;
    use crate::tasks::motif_loader::MotifMeta;
    use crate::tasks::slices_stats::{ScalarStatsValue, SlicesStatsInput};
    use async_trait::async_trait;
    use log::debug;
    use mockall::mock;
    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap};
    use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use thoosa_shared::runner::common::{MockRunCtx, OutId};
    use thoosa_shared::traits::{Adjustable, Data, MonoMetrics, WithId};
    use thoosa_shared::{mono_key, Cause, CopyToDevice, ThoError};
    use thoosa_shared::comp_id::SynKey;

    mock! {
        #[derive(Debug)]
        pub SlicesStats {}

        impl Stage for SlicesStats {
            type Ctx = MockRunCtx;
            type In<'a> = SlicesStatsInput;
            type Out = ScalarStatsValue;
        }

        impl OutId for SlicesStats {
            fn out_id<'a>(&self,input_id: &<<Self as Stage>::In<'a> as WithId>::ID,) -> <<MockSlicesStats as Stage>::Out as WithId>::ID;
        }


        impl SyncStage for SlicesStats {
            fn do_run<'f>(self: Arc<Self>, _input: &<Self as Stage>::In<'f>, _ctx: &<Self as Stage>::Ctx) -> Result<<Self as Stage>::Out, ThoError>;
        }


        impl Adjustable for SlicesStats {

        }


    }

    pub struct MockSlicesStatsBuilder {
        outputs: Vec<ScalarStatsValue>,
        fail_first: bool,
        fail_while_dim0_gt: Option<i64>,
        call_log: Option<Arc<Mutex<Vec<i64>>>>,
        counter: AtomicUsize,

        mock: MockSlicesStats,
    }

    impl MockSlicesStatsBuilder {
        pub fn with_do_run<I>(mut self, outputs: I) -> Self
        where
            I: Into<Vec<ScalarStatsValue>>,
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

        pub fn finish(mut self) -> MockSlicesStats {
            self.mock.expect_do_run().returning(move |input, _ctx| {
                let metrics: MonoMetrics<0, 2> = input.slices().data().size().try_into().expect("MockSlicesStats expects motif metrics");
                debug!("MockSlicesStats::do_run: metrics={metrics:?}");

                let call_idx = self.counter.fetch_add(1, Ordering::SeqCst);
                let dim_0 = metrics.dim_0();
                if let Some(call_log) = self.call_log.as_ref() {
                    call_log.lock().push(dim_0);
                }

                if self.fail_first && call_idx == 0 {
                    return Err(Cause::OutOfMemory.into());
                }

                if let Some(threshold) = self.fail_while_dim0_gt {
                    if dim_0 > threshold {
                        return Err(Cause::OutOfMemory.into());
                    }
                }

                let value = self.outputs.pop().expect("mock outputs exhausted");

                Ok(value)
            });

            self.mock.expect_out_id().return_const(SynKey::new([mono_key!("whatever", "mock")]));

            self.mock
        }
    }

    pub fn builder() -> MockSlicesStatsBuilder {
        MockSlicesStatsBuilder {
            outputs: Vec::new(),
            fail_first: false,
            fail_while_dim0_gt: None,
            call_log: None,
            counter: Default::default(),
            mock: MockSlicesStats::new(),
        }
    }
}
