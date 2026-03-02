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
use tracing::debug;
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
use thoosa_motifs::common_stats::Stats as CommonStatsValue;
use thoosa_motifs::criteria::{
    coherence, inlier_ratio, interesting_amp_01, interesting_band_energy_ratio_01, interesting_curvature_norm_01,
    interesting_entropy_band_01, interesting_entropy_gauss_01, interesting_turning_points_ratio_peaked_01,
    interesting_tv_norm_01, norm_thickness,
};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, DummyData)]
pub enum Metrics {
    EucDist,
    PearsonCorr,//not used yet
    CommonEucDist,
    CommonPearson,
    CommonA0,
    CommonTv0,
    CommonCurv0,
    CommonAmpP10,
    CommonAmpP50,
    CommonAmpP90,
    CommonTvnP10,
    CommonTvnP50,
    CommonTvnP90,
    CommonCurvnP10,
    CommonCurvnP50,
    CommonCurvnP90,
    NormThickness,
    InlierRatioRunConfig,
    CoherenceRunConfig,
    InterestingAmp01,
    InterestingTvNorm01,
    InterestingCurvatureNorm01,
    InterestingTurningPointsRatio,
    InterestingBandEnergyRatio01,
    InterestingEntropyGauss01,
    InterestingEntropyBand01,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::EucDist
    }
}

impl Metrics {
    pub fn is_common_stats_metric(&self) -> bool {
        matches!(
            self,
            Metrics::CommonEucDist
                | Metrics::CommonPearson
                | Metrics::CommonA0
                | Metrics::CommonTv0
                | Metrics::CommonCurv0
                | Metrics::CommonAmpP10
                | Metrics::CommonAmpP50
                | Metrics::CommonAmpP90
                | Metrics::CommonTvnP10
                | Metrics::CommonTvnP50
                | Metrics::CommonTvnP90
                | Metrics::CommonCurvnP10
                | Metrics::CommonCurvnP50
                | Metrics::CommonCurvnP90
        )
    }

    pub fn common_stats_value(&self, stats: &CommonStatsValue) -> Result<f32, ThoError> {
        match self {
            Metrics::CommonEucDist => Ok(stats.euc_dist),
            Metrics::CommonPearson => Ok(stats.pearson),
            Metrics::CommonA0 => Ok(stats.a0),
            Metrics::CommonTv0 => Ok(stats.tv0),
            Metrics::CommonCurv0 => Ok(stats.curv0),
            Metrics::CommonAmpP10 => Ok(stats.amp_p10),
            Metrics::CommonAmpP50 => Ok(stats.amp_p50),
            Metrics::CommonAmpP90 => Ok(stats.amp_p90),
            Metrics::CommonTvnP10 => Ok(stats.tvn_p10),
            Metrics::CommonTvnP50 => Ok(stats.tvn_p50),
            Metrics::CommonTvnP90 => Ok(stats.tvn_p90),
            Metrics::CommonCurvnP10 => Ok(stats.curvn_p10),
            Metrics::CommonCurvnP50 => Ok(stats.curvn_p50),
            Metrics::CommonCurvnP90 => Ok(stats.curvn_p90),
            _ => Err(ThoError::new(format!("common_stats_value called with non-common metric: {self:?}"))),
        }
    }
}

pub fn metric_scalar_for_motif(motif_slices: &Tensor, metric: Metrics, aggr: Aggr) -> Result<f32, ThoError> {
    let score_t = match metric {
        Metrics::EucDist => return panic_to_err!(euclidean_distance(motif_slices, aggr)),
        Metrics::PearsonCorr => return panic_to_err!(pearson_corr_distance_matrix_0_2_single(motif_slices, 1e-8, aggr)),

        Metrics::CommonEucDist
        | Metrics::CommonPearson
        | Metrics::CommonA0
        | Metrics::CommonTv0
        | Metrics::CommonCurv0
        | Metrics::CommonAmpP10
        | Metrics::CommonAmpP50
        | Metrics::CommonAmpP90
        | Metrics::CommonTvnP10
        | Metrics::CommonTvnP50
        | Metrics::CommonTvnP90
        | Metrics::CommonCurvnP10
        | Metrics::CommonCurvnP50
        | Metrics::CommonCurvnP90 => {
            let common_stats = thoosa_motifs::common_stats::estimate_channel_stats_from_windows(
                &motif_slices.f_unsqueeze(0)?,
                0.05,
                0.95,
                0.50,
                1e-8,
            )?;
            let stats = common_stats
                .first()
                .ok_or_else(|| ThoError::new("metric_scalar_for_motif: no common stats for motif"))?;
            return metric.common_stats_value(stats);
        }

        Metrics::NormThickness => norm_thickness(motif_slices, 1.0, 1e-8)?,
        Metrics::InlierRatioRunConfig => inlier_ratio(motif_slices, 4.5, 1e-8)?,
        Metrics::CoherenceRunConfig => coherence(motif_slices, 10, 1e-8)?,
        Metrics::InterestingAmp01 => interesting_amp_01(motif_slices, 0.05, 0.95, 1.0, 1e-8)?,
        Metrics::InterestingTvNorm01 => interesting_tv_norm_01(motif_slices, 0.05, 0.95, 1.0, 1e-8)?,
        Metrics::InterestingCurvatureNorm01 => interesting_curvature_norm_01(motif_slices, 0.05, 0.95, 1.0, 1e-8)?,
        Metrics::InterestingTurningPointsRatio => interesting_turning_points_ratio_peaked_01(motif_slices, 4)?,
        Metrics::InterestingBandEnergyRatio01 => interesting_band_energy_ratio_01(motif_slices, 250.0, 8.0, 30.0, 1e-8)?,
        Metrics::InterestingEntropyGauss01 => interesting_entropy_gauss_01(motif_slices, 0.6, 0.15, 1e-8)?,
        Metrics::InterestingEntropyBand01 => interesting_entropy_band_01(motif_slices, 0.3, 0.85, 1e-8)?,
    };

    let raw = score_t.f_double_value(&[])?;
    if raw.is_finite() {
        Ok(raw as f32)
    } else {
        Ok(0.0)
    }
}

#[derive(Debug, Copy, Clone, DummyData)]
pub enum Aggr {
    Mean(f32),
    Min(f32),
    Max(f32),
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
        let value = metric_scalar_for_motif(input.slices().data().t_ref(), input.metrics(), input.aggr())?;

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
        Aggr::Mean(k) =>{
            let mean = distances.f_mean(Kind::Float)?;
            debug!("aggregate_distances, Mean: k={k} distances=\n{distances} mean=\n{mean}");

            k * mean
        },
        Aggr::Min(k) => {
            let min = distances.f_min()?;
            debug!("aggregate_distances, Min: k={k} distances=\n{distances} min=\n{min}");

            k * min
        },
        Aggr::Max(k) => {
            let max = distances.f_max()?;
            debug!("aggregate_distances, Max: k={k} distances=\n{distances} max=\n{max}");
            k * max
        },
        Aggr::Quantile(q) => {
            debug!("aggregate_distances, Quantile: q={q} distances=\n{distances}");
            let q = q.clamp(0.0, 1.0) as f64;
            let quantile = distances.f_quantile_scalar(q, 0, true, "lower")?;
            debug!("aggregate_distances, Quantile: distances=\n{distances}, quantile=\n{quantile}");

            let mask = distances.f_le_tensor(&quantile)?;
            let selected = distances.f_masked_select(&mask)?;
            if selected.numel() == 0 {
                debug!("aggregate_distances, Quantile: selected.numel==0, quantile=\n{quantile}");
                quantile
            } else {
                debug!("aggregate_distances, Quantile: selected.numel!=0, quantile=\n{quantile} selected.f_max={}", selected.max());
                selected.f_max()?
            }
        }
    };

    Ok(scalar.f_double_value(&[])? as f32)
}

#[cfg(test)]
mod tests {
    use super::{aggregate_distances, Aggr};
    use log::debug;
    use tch::Tensor;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;

    #[test]
    fn test_aggregate_distances() {
        let distances = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);

        let mean_k_half = aggregate_distances(&distances, Aggr::Mean(0.5)).unwrap();
        assert!((mean_k_half - 1.25).abs() < 1e-6, "expected mean([1,2,3,4]) * 0.5 = 1.25, got {mean_k_half}");

        let min_k_2 = aggregate_distances(&distances, Aggr::Min(2.0)).unwrap();
        assert!((min_k_2 - 2.0).abs() < 1e-6, "expected min([1,2,3,4]) * 2.0 = 2.0, got {min_k_2}");

        let max_k_2 = aggregate_distances(&distances, Aggr::Max(2.0)).unwrap();
        assert!((max_k_2 - 8.0).abs() < 1e-6, "expected max([1,2,3,4]) * 2.0 = 8.0, got {max_k_2}");

        let median_lower = aggregate_distances(&distances, Aggr::Quantile(0.5)).unwrap();
        assert!((median_lower - 2.0).abs() < 1e-6, "expected quantile(0.5, lower) for [1,2,3,4] to be 2.0, got {median_lower}");
    }

    #[test]
    fn test_aggregate_distances_with_zeros() {
        let distances = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        //
        // let mean_k_half = aggregate_distances(&distances, Aggr::Mean(0.0)).unwrap();
        // assert!((mean_k_half ).abs() < 1e-6, "expected mean([1,2,3,4]) * 0.5 = 1.25, got {mean_k_half}");
        //
        // let min_k_2 = aggregate_distances(&distances, Aggr::Min(0.0)).unwrap();
        // assert!((min_k_2 ).abs() < 1e-6, "expected min([1,2,3,4]) * 2.0 = 2.0, got {min_k_2}");
        //
        // let max_k_2 = aggregate_distances(&distances, Aggr::Max(0.0)).unwrap();
        // assert!((max_k_2).abs() < 1e-6, "expected max([1,2,3,4]) * 2.0 = 8.0, got {max_k_2}");

        let median_lower = aggregate_distances(&distances, Aggr::Quantile(0.0)).unwrap();
        assert!((median_lower).abs() < 1e-6, "expected quantile(0.0, lower) for [1,2,3,4] to be 2.0, got {median_lower}");
    }


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
                    thoosa_shared::with_perf_log(
                        || call_log.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                        "MockSlicesStats/do_run/call_log",
                    )
                    .push(dim_0);
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
