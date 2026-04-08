use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;

use async_trait::async_trait;
use cached_derive::Cached;
use log::{debug, warn};
use parking_lot::RwLock;
use tch::Tensor;
use rand::prelude::{IndexedRandom, SliceRandom, StdRng};
use thoosa_common::dims::MotifSlices;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, MotifCriterion, MotifCriterionConstraint, SelectMotifsRunConfig};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_motifs::criteria::{coherence, inlier_ratio, interesting_amp_01, interesting_band_energy_ratio_01, interesting_curvature_norm_01, interesting_entropy_band_01, interesting_entropy_gauss_01, interesting_turning_points_ratio_trapezoid_01, interesting_tv_norm_01, norm_thickness, periodicity_acf_01_frac};
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::resource::{Resource, ResourceId};
use thoosa_shared::run_configs::RunId;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::traits::{Adjustable, Data, DataId, MonoMetrics, Select, WithId};
use thoosa_shared::{mono_key, CopyToDevice, TensorIdSliceIdx, ThoError};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use tracing::error;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists, ChannelsMotifsSlices};
use crate::tasks::motif_loader::{load_motif_slices_, MotifLoaderInput, MotifMeta};
use crate::tasks::slices_stats::{metric_scalar_for_motif, Aggr, Metrics};
use crate::tasks::stage_ids::{MOTIF_STATS, MOTIF_STATS_SYNC_STAGE};

pub const RANDOM_STATS_GROUP: &str = "random_avg_dist"; //dummy group to load random resources, TODO - add proper method to SqlStore

const DEFAULT_Q_LO: f64 = 0.05;
const DEFAULT_Q_HI: f64 = 0.95;
const DEFAULT_SCALE_Q: f64 = 0.50;
const DEFAULT_EPS: f64 = 1e-8;

fn criterion_metric(criterion: &MotifCriterion) -> Metrics {
    match criterion {
        MotifCriterion::NormThickness(_) => Metrics::NormThickness,
        MotifCriterion::InlierRatio(_) => Metrics::InlierRatioRunConfig,
        MotifCriterion::Coherence(_) => Metrics::CoherenceRunConfig,
        MotifCriterion::InterestingAmp01(_) => Metrics::InterestingAmp01,
        MotifCriterion::InterestingTvNorm01(_) => Metrics::InterestingTvNorm01,
        MotifCriterion::InterestingCurvatureNorm01(_) => Metrics::InterestingCurvatureNorm01,
        MotifCriterion::InterestingTurningPointsRatio(_) => Metrics::InterestingTurningPointsRatio,
        MotifCriterion::InterestingBandEnergyRatio01(_) => Metrics::InterestingBandEnergyRatio01,
        MotifCriterion::InterestingEntropyGauss01(_) => Metrics::InterestingEntropyGauss01,
        MotifCriterion::InterestingEntropyBand01(_) => Metrics::InterestingEntropyBand01,
        MotifCriterion::PeriodicityAcf(_) => Metrics::PeriodicityAcf,
    }
}

fn criterion_score_for_motif(motif_slices: &Tensor, criterion: &MotifCriterion) -> Result<f64, ThoError> {
    let score_t = match criterion {
        MotifCriterion::NormThickness(v) => norm_thickness(motif_slices, v.eps)?,
        MotifCriterion::InlierRatio(v) => inlier_ratio(motif_slices, v.k, v.eps)?,
        MotifCriterion::Coherence(v) => coherence(motif_slices, v.k.round().max(1.0) as i64, v.eps)?,
        MotifCriterion::InterestingAmp01(v) => interesting_amp_01(motif_slices, v.q_lo, v.q_hi, v.a0, v.eps)?,
        MotifCriterion::InterestingTvNorm01(v) => interesting_tv_norm_01(motif_slices, v.amp_q_lo, v.amp_q_hi, v.tv0, v.eps)?,
        MotifCriterion::InterestingCurvatureNorm01(v) => interesting_curvature_norm_01(motif_slices, v.amp_q_lo, v.amp_q_hi, v.curv0, v.eps)?,
        MotifCriterion::InterestingTurningPointsRatio(_) => interesting_turning_points_ratio_trapezoid_01(motif_slices, 2, 3, 5, 7)?,
        MotifCriterion::InterestingBandEnergyRatio01(v) => interesting_band_energy_ratio_01(motif_slices, v.fs, v.f_lo, v.f_hi, v.eps)?,
        MotifCriterion::InterestingEntropyGauss01(v) => interesting_entropy_gauss_01(motif_slices, v.h0, v.sigma, v.eps)?,
        MotifCriterion::InterestingEntropyBand01(v) => interesting_entropy_band_01(motif_slices, v.h_min, v.h_max, v.eps)?,
        MotifCriterion::PeriodicityAcf(v) => periodicity_acf_01_frac(motif_slices, v.struct_lag_max, v.lag_min, v.lag_max, v.struct0, v.peak_good, v.peak_bad, v.eps)?,
    };
    let raw_score = score_t.f_double_value(&[])?;
    if raw_score.is_finite() {
        Ok(raw_score.clamp(0.0, 1.0))
    } else {
        warn!("MotifStatsSyncStage::criterion_score_for_motif: got non-finite criterion score={raw_score}, treating it as 0.0");
        Ok(0.0)
    }
}

#[derive(Debug, Clone)]
pub enum MotifStatsSource {
    Motif(MotifLoaderInput),
    Random { max_slices: u32, find_motifs_run_id: RunId },
    ChannelsMotifsSlices(Arc<ThoTensor<ChannelsMotifsSlices, SynKey>>),
}

#[derive(Debug, Clone)]
pub struct MotifStatsInput {
    id: SynKey,
    source: MotifStatsSource,
    metrics: Vec<Metrics>,
    aggr: Aggr,
}

impl MotifStatsInput {
    pub fn from_motif_input(
        motif_loader_input: MotifLoaderInput,
        metrics: Vec<Metrics>,
        aggr: Aggr,
        select_motifs_run_config: Option<&SelectMotifsRunConfig>,
    ) -> Self {
        let metrics_sfx = metrics.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>().join(",");
        let aggr_sfx = format!("{aggr:?}");
        let id = match select_motifs_run_config.and_then(|cfg| cfg.id) {
            Some(select_motifs_run_id) => &motif_loader_input.id + mono_key!(select_motifs_run_id, format!("select_motifs_run_id={select_motifs_run_id}")),
            None => &(&motif_loader_input.id + mono_key!(metrics_sfx.clone(), format!("metrics={metrics_sfx}"))) + mono_key!(aggr_sfx.clone(), format!("aggr={aggr_sfx}")),
        };
        Self {
            id,
            source: MotifStatsSource::Motif(motif_loader_input),
            metrics,
            aggr,
        }
    }

    pub fn from_random(max_slices: u32, find_motifs_run_id: RunId, metrics: Vec<Metrics>, aggr: Aggr) -> Self {
        let metrics_sfx = metrics.iter().map(|m| format!("{m:?}")).collect::<Vec<_>>().join(",");
        Self {
            id: SynKey::new([
                mono_key!("MotifStatsInput", "stage"),
                mono_key!(find_motifs_run_id, format!("find_motifs_run_id={find_motifs_run_id}")),
                mono_key!(max_slices, format!("max_slices={max_slices}")),
                mono_key!(metrics_sfx.clone(), format!("metrics={metrics_sfx}")),
                mono_key!(format!("{aggr:?}"), format!("aggr={aggr:?}")),
            ]),
            source: MotifStatsSource::Random { max_slices, find_motifs_run_id },
            metrics,
            aggr,
        }
    }


    pub fn source(&self) -> &MotifStatsSource {
        &self.source
    }

    pub fn metrics(&self) -> &Vec<Metrics> {
        &self.metrics
    }

    pub fn aggr(&self) -> Aggr {
        self.aggr
    }
}

impl Input for MotifStatsInput {}

impl WithId for MotifStatsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for MotifStatsInput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        1usize.into()
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        match &self.source {
            MotifStatsSource::Motif(_) | MotifStatsSource::Random { .. } => 0,
            MotifStatsSource::ChannelsMotifsSlices(channels_motifs_slices) => channels_motifs_slices.mem_size(),
        }
    }
}

impl Select<()> for MotifStatsInput {
    type Output = MotifStatsInput;

    fn subview(&self, _selector: ()) -> Result<Self::Output, ThoError> {
        Ok(self.clone())
    }
}

impl Adjustable for MotifStatsInput {}

#[async_trait]
impl CopyToDevice for MotifStatsInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        let source = match &self.source {
            MotifStatsSource::Motif(motif_loader_input) => MotifStatsSource::Motif(motif_loader_input.clone()),
            MotifStatsSource::Random { max_slices, find_motifs_run_id } => MotifStatsSource::Random { max_slices: *max_slices, find_motifs_run_id: *find_motifs_run_id },
            MotifStatsSource::ChannelsMotifsSlices(t) => MotifStatsSource::ChannelsMotifsSlices(Arc::new(t.copy_to_device(dst_device).await?)),
        };

        Ok(Self {
            id: self.id.clone(),
            source,
            metrics: self.metrics.clone(),
            aggr: self.aggr,
        })
    }
}

#[derive(Debug)]
pub struct MotifStatsCalcInput {
    id: SynKey,
    channels_motifs_slices: Arc<ThoTensor<ChannelsMotifsSlices, SynKey>>,
    metrics: Vec<Metrics>,
    aggr: Aggr,
}

impl MotifStatsCalcInput {
    pub fn new(
        id: SynKey,
        channels_motifs_slices: Arc<ThoTensor<ChannelsMotifsSlices, SynKey>>,
        metrics: Vec<Metrics>,
        aggr: Aggr,
    ) -> Self {
        Self { id, channels_motifs_slices, metrics, aggr }
    }

    pub fn channels_motifs_slices(&self) -> &Arc<ThoTensor<ChannelsMotifsSlices, SynKey>> {
        &self.channels_motifs_slices
    }

    pub fn metrics(&self) -> &Vec<Metrics> {
        &self.metrics
    }

    pub fn aggr(&self) -> Aggr {
        self.aggr
    }
}

impl Input for MotifStatsCalcInput {}

impl WithId for MotifStatsCalcInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for MotifStatsCalcInput {
    type Metrics = <ThoTensor<ChannelsMotifsSlices, SynKey> as Data>::Metrics;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        <ThoTensor<ChannelsMotifsSlices, SynKey> as Data>::size(self.channels_motifs_slices())
    }

    fn device(&self) -> usize {
        self.channels_motifs_slices.device()
    }

    fn mem_size(&self) -> u64 {
        self.channels_motifs_slices.mem_size()
    }
}

impl Select<()> for MotifStatsCalcInput {
    type Output = MotifStatsCalcInput;

    fn subview(&self, _selector: ()) -> Result<Self::Output, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            channels_motifs_slices: Arc::new(ThoTensor::new(self.channels_motifs_slices.id().clone(), self.channels_motifs_slices.t_ref().shallow_clone())?),
            metrics: self.metrics.clone(),
            aggr: self.aggr,
        })
    }
}

impl Adjustable for MotifStatsCalcInput {}

#[async_trait]
impl CopyToDevice for MotifStatsCalcInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            channels_motifs_slices: Arc::new(self.channels_motifs_slices.copy_to_device(dst_device).await?),
            metrics: self.metrics.clone(),
            aggr: self.aggr,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MotifCriteriaEvaluation {
    avg_score: f64,
    criterion_scores: Vec<(&'static str, f64)>,
}

impl MotifCriteriaEvaluation {
    pub fn new(avg_score: f64, criterion_scores: Vec<(&'static str, f64)>) -> Self {
        Self { avg_score, criterion_scores }
    }

    pub fn avg_score(&self) -> f64 {
        self.avg_score
    }

    pub fn criterion_scores(&self) -> &Vec<(&'static str, f64)> {
        &self.criterion_scores
    }
}

#[derive(Debug)]
pub struct MotifStatsOutput {
    id: SynKey,
    values: Vec<(Metrics, ThoTensor<ChannelsMotifsDists, SynKey>)>,
    criteria_evaluation: Option<MotifCriteriaEvaluation>,
}

impl MotifStatsOutput {
    pub fn new(id: SynKey, values: Vec<(Metrics, ThoTensor<ChannelsMotifsDists, SynKey>)>) -> Self {
        Self { id, values, criteria_evaluation: None }
    }

    pub fn with_criteria_evaluation(id: SynKey, values: Vec<(Metrics, ThoTensor<ChannelsMotifsDists, SynKey>)>, criteria_evaluation: Option<MotifCriteriaEvaluation>) -> Self {
        Self { id, values, criteria_evaluation }
    }

    pub fn metrics(&self) -> Vec<Metrics> {
        self.values.iter().map(|(metric, _)| *metric).collect()
    }

    pub fn criteria_evaluation(&self) -> Option<&MotifCriteriaEvaluation> {
        self.criteria_evaluation.as_ref()
    }

    pub(crate) fn metric_values(&self, metric: Metrics) -> Option<&ThoTensor<ChannelsMotifsDists, SynKey>> {
        self.values.iter().find_map(|(cur, values)| if *cur == metric { Some(values) } else { None })
    }

    pub fn channel_value(&self, metric: Metrics, channel: usize) -> Result<f32, ThoError> {
        let values = self.metric_values(metric).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: metric {metric:?} was not requested")))?;
        let channels = values.t_ref().size().first().copied().unwrap_or(0).max(0) as usize;
        if channel >= channels {
            return Err(ThoError::new(format!("MotifStatsOutput: channel index out of bounds: channel={channel} channels={channels}")));
        }
        Ok(values.t_ref().f_double_value(&[channel as i64, 0])? as f32)
    }

    pub fn channels_values(&self, metric: Metrics) -> Result<Vec<f32>, ThoError> {
        let values = self.metric_values(metric).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: metric {metric:?} was not requested")))?;
        let channels = values.t_ref().size().first().copied().unwrap_or(0).max(0) as i64;
        (0..channels).map(|channel| Ok(values.t_ref().f_double_value(&[channel, 0])? as f32)).collect()
    }

    pub fn channels_motifs_dists(&self, metric: Metrics) -> Result<ThoTensor<ChannelsMotifsDists, SynKey>, ThoError> {
        let values = self.metric_values(metric).ok_or_else(|| ThoError::new(format!("MotifStatsOutput: metric {metric:?} was not requested")))?;
        ThoTensor::new(values.id().clone(), values.t_ref().shallow_clone())
    }
}

impl WithId for MotifStatsOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for MotifStatsOutput {}

impl Data for MotifStatsOutput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        self.values.len().into()
    }

    fn device(&self) -> usize {
        self.values.first().map(|(_, t)| t.device()).unwrap_or(0)
    }

    fn mem_size(&self) -> u64 {
        self.values.iter().map(|(_, t)| t.mem_size()).sum()
    }
}

#[async_trait]
impl CopyToDevice for MotifStatsOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        let mut values = Vec::with_capacity(self.values.len());
        for (metric, tensor) in &self.values {
            values.push((*metric, tensor.copy_to_device(dst_device).await?));
        }
        Ok(Self {
            id: self.id.clone(),
            values,
            criteria_evaluation: self.criteria_evaluation.clone(),
        })
    }
}

#[derive(Debug, Clone)]
struct MotifStatsCriteriaConfig {
    constraints: Vec<MotifCriterionConstraint>,
    min_slices: i64,
    max_slices: i64,
}

#[derive(Debug, Named)]
pub struct MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    q_lo: f64,
    q_hi: f64,
    scale_q: f64,
    eps: f64,
    criteria_config: Option<MotifStatsCriteriaConfig>,
    _ctx: PhantomData<C>,
}

impl<C> MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn with_common_stats_params(q_lo: f64, q_hi: f64, scale_q: f64, eps: f64) -> Self {
        Self {
            id: (MOTIF_STATS_SYNC_STAGE, "MotifStatsSyncStage").into(),
            q_lo,
            q_hi,
            scale_q,
            eps,
            criteria_config: None,
            _ctx: PhantomData,
        }
    }

    pub fn with_criteria(criteria_constraints: Vec<MotifCriterionConstraint>, min_slices: i64, max_slices: i64, select_motifs_run_id: Option<RunId>) -> Self {
        assert!(min_slices <= max_slices, "MotifStatsSyncStage::with_criteria: min_slices={min_slices} must be <= max_slices={max_slices}");
        let id = match select_motifs_run_id {
            Some(select_motifs_run_id) => SynKey::new([
                mono_key!(MOTIF_STATS_SYNC_STAGE, "MotifStatsSyncStage"),
                mono_key!("criteria", "mode"),
                mono_key!(select_motifs_run_id, format!("select_motifs_run_id={select_motifs_run_id}")),
            ]),
            None => SynKey::new([
                mono_key!(MOTIF_STATS_SYNC_STAGE, "MotifStatsSyncStage"),
                mono_key!("criteria", "mode"),
            ]),
        };
        Self {
            id,
            q_lo: DEFAULT_Q_LO,
            q_hi: DEFAULT_Q_HI,
            scale_q: DEFAULT_SCALE_Q,
            eps: DEFAULT_EPS,
            criteria_config: Some(MotifStatsCriteriaConfig { constraints: criteria_constraints, min_slices, max_slices }),
            _ctx: PhantomData,
        }
    }
}

impl<C> Default for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn default() -> Self {
        Self::with_common_stats_params(DEFAULT_Q_LO, DEFAULT_Q_HI, DEFAULT_SCALE_Q, DEFAULT_EPS)
    }
}

impl<C> Adjustable for MotifStatsSyncStage<C> where C: RunCtx + Adjustable + 'static {}

impl<C> OutId for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = MotifStatsCalcInput;
    type Out = MotifStatsOutput;
}

impl<C> SyncStage for MotifStatsSyncStage<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStatsSyncStage::do_run"))]
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let channels_motifs_slices = input.channels_motifs_slices().t_ref();
        let channels = channels_motifs_slices
            .size()
            .first()
            .copied()
            .ok_or_else(|| ThoError::new(format!("MotifStatsSyncStage::do_run: expected [channels, slices, window], got {:?}", channels_motifs_slices.size())))? as usize;

        if let Some(criteria_config) = &self.criteria_config {
            let motif_slices_count = channels_motifs_slices.size().get(1).copied().unwrap_or(0);
            if motif_slices_count < criteria_config.min_slices || motif_slices_count > criteria_config.max_slices {
                return Ok(MotifStatsOutput::with_criteria_evaluation(self.out_id(input.id()), Vec::new(), None));
            }

            let motif_slices = channels_motifs_slices.f_get(0)?;
            let mut values = Vec::with_capacity(criteria_config.constraints.len());
            let mut criterion_scores = Vec::with_capacity(criteria_config.constraints.len());
            let mut score_sum = 0.0f64;

            for (idx, criterion_constraint) in criteria_config.constraints.iter().enumerate() {
                let criterion_score = criterion_score_for_motif(&motif_slices, &criterion_constraint.criterion)?;
                if criterion_score < criterion_constraint.min as f64 {
                    return Ok(MotifStatsOutput::with_criteria_evaluation(self.out_id(input.id()), values, None));
                }
                let metric = criterion_metric(&criterion_constraint.criterion);
                let t = Tensor::f_from_slice(&[criterion_score as f32])?.f_view([1, 1])?;
                values.push((
                    metric,
                    ThoTensor::new(
                        SynKey::new([
                            mono_key!("motif_stats", "motif_stats"),
                            mono_key!(format!("{:?}", input.id()), format!("input_id={:?}", input.id())),
                            mono_key!(format!("{metric:?}"), format!("metric={metric:?}")),
                        ]),
                        t,
                    )?,
                ));
                score_sum += criterion_score;
                let avg_score = score_sum / (idx + 1) as f64;
                criterion_scores.push((criterion_constraint.key, criterion_score));
                debug!("MotifStatsSyncStage::do_run: criterion_constraint={criterion_constraint:?} criterion_score={criterion_score:.6} avg_score={avg_score:.6}");
            }

            let avg_score = if criterion_scores.is_empty() { 1.0 } else { score_sum / criterion_scores.len() as f64 };
            return Ok(MotifStatsOutput::with_criteria_evaluation(
                self.out_id(input.id()),
                values,
                Some(MotifCriteriaEvaluation::new(avg_score, criterion_scores)),
            ));
        }

        let mut common_stats_cache: Option<Vec<thoosa_motifs::common_stats::Stats>> = None;
        let mut values = Vec::with_capacity(input.metrics().len());

        for metric in input.metrics() {
            let channel_values = if metric.is_common_stats_metric() {
                if common_stats_cache.is_none() {
                    common_stats_cache = Some(thoosa_motifs::common_stats::estimate_channel_stats_from_windows(
                        channels_motifs_slices,
                        self.q_lo,
                        self.q_hi,
                        self.scale_q,
                        self.eps,
                    )?);
                }
                let common_stats = common_stats_cache.as_ref().expect("set above");
                common_stats.iter().map(|stats| metric.common_stats_value(stats)).collect::<Result<Vec<_>, ThoError>>()?
            } else {
                let mut channel_values = Vec::with_capacity(channels);
                for channel_idx in 0..channels {
                    let motif_slices = channels_motifs_slices.f_get(channel_idx as i64)?;
                    channel_values.push(metric_scalar_for_motif(&motif_slices, *metric, input.aggr())?);
                }
                channel_values
            };
            let t = Tensor::f_from_slice(channel_values.as_slice())?.f_view([channels as i64, 1])?;
            values.push((
                *metric,
                ThoTensor::new(
                    SynKey::new([
                        mono_key!("motif_stats", "motif_stats"),
                        mono_key!(format!("{:?}", input.id()), format!("input_id={:?}", input.id())),
                        mono_key!(format!("{metric:?}"), format!("metric={metric:?}")),
                    ]),
                    t,
                )?,
            ));
        }

        Ok(MotifStatsOutput::new(self.out_id(input.id()), values))
    }
}

#[derive(Cached, Named)]
pub struct MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    id: SynKey,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    pub motif_loader: Arc<ML>,
    clc_stats: Arc<MotifStatsSyncStage<C>>,
    _ctx: PhantomData<C>,
}

impl<ML, L, C> Debug for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MotifStats").finish()
    }
}

impl<ML, L, C> MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    pub fn new(
        store: Arc<Box<dyn Store>>,
        loader: Arc<L>,
        ws: Arc<RwLock<Box<dyn Workspace>>>,
        motif_loader: Arc<ML>,
    ) -> Self {
        Self {
            id: (MOTIF_STATS, "MotifStats").into(),
            store,
            loader,
            ws,
            motif_loader,
            clc_stats: Arc::new(MotifStatsSyncStage::default()),
            _ctx: PhantomData,
        }
    }

    pub fn with_criteria_constraints(
        store: Arc<Box<dyn Store>>,
        loader: Arc<L>,
        ws: Arc<RwLock<Box<dyn Workspace>>>,
        motif_loader: Arc<ML>,
        select_motifs_run_id: Option<RunId>,
        criteria_constraints: Vec<MotifCriterionConstraint>,
        min_slices: i64,
        max_slices: i64,
    ) -> Self {
        let id = match select_motifs_run_id {
            Some(select_motifs_run_id) => SynKey::new([
                mono_key!(MOTIF_STATS, "MotifStats"),
                mono_key!("criteria", "mode"),
                mono_key!(select_motifs_run_id, format!("select_motifs_run_id={select_motifs_run_id}")),
            ]),
            None => SynKey::new([
                mono_key!(MOTIF_STATS, "MotifStats"),
                mono_key!("criteria", "mode"),
            ]),
        };
        Self {
            id,
            store,
            loader,
            ws,
            motif_loader,
            clc_stats: Arc::new(MotifStatsSyncStage::with_criteria(criteria_constraints, min_slices, max_slices, select_motifs_run_id)),
            _ctx: PhantomData,
        }
    }
}

impl<ML, L, C> OutId for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    fn out_id(&self, input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<ML, L, C> Stage for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    type Ctx = C;
    type In<'a> = MotifStatsInput;
    type Out = MotifStatsOutput;
}

impl<ML, L, C> AsyncStage for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MotifStats::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<MotifStatsInput>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        let input_id = input.id().clone();
        let (source, metrics, aggr) = with_data!(ctx.branch(Some("motif_stats_input"), false), input; |_ctx, input: &MotifStatsInput| {
            Ok::<_, ThoError>((input.source().clone(), input.metrics().clone(), input.aggr()))
        })?;

        let channels_motifs_slices = match source {
            MotifStatsSource::Motif(motif_loader_input) => {
                let motif = Arc::clone(&self.motif_loader)
                    .run_cached(
                        Arc::new(DevicesWrap::wrap(motif_loader_input)),
                        ctx.branch(Some("motif_stats_motif_loader"), false),
                    )
                    .await?;

                with_data!(ctx.branch(Some("motif_stats_motif_to_channels"), false), motif; |_ctx, motif: &thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>| {
                    let motif_2d = motif.data().t_ref().shallow_clone();
                    let channels_3d = motif_2d.f_unsqueeze(0)?;
                    Ok::<_, ThoError>(Arc::new(ThoTensor::new(motif.data().id().clone(), channels_3d)?))
                })?
            }
            MotifStatsSource::Random { max_slices, find_motifs_run_id } => {
                if max_slices < 2 {
                    return Err(ThoError::new(format!("MotifStats::run: max_slices must be >= 2, got {max_slices}")));
                }

                let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
                    find_motifs_run_id,
                    &self.store.list_run_config_values(find_motifs_run_id).await?,
                )?;
                let input_len_adjuster = find_motifs_run_config.input_len_adjuster();
                let resources = self.store.list_random_resources(None).await?;
                let (slices, resources_map) = random_non_overlapping_slices(&resources, max_slices, find_motifs_run_config.window, input_len_adjuster)?;

                let (_window, channels_slices) = load_motif_slices_(
                    DataId(0),
                    find_motifs_run_id,
                    resources_map,
                    slices,
                    Arc::clone(&self.store),
                    Arc::clone(&self.loader),
                    ctx.branch(Some("motif_stats_random_load_slices"), false),
                    &self.ws,
                    "motif_stats_random",
                )
                .await?;

                Arc::new(ThoTensor::new(channels_slices.id().clone(), channels_slices.t_ref().shallow_clone())?)
            }
            MotifStatsSource::ChannelsMotifsSlices(t) => t,
        };

        let clc_input = Arc::new(DevicesWrap::wrap(MotifStatsCalcInput::new(input_id.clone(), channels_motifs_slices, metrics, aggr)));
        let required_ids: HashMap<&'static str, &std::collections::HashSet<&dyn thoosa_shared::traits::CachableIdTr>> = HashMap::new();
        let stats = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                Adjusters::RunContext(Box::new(DumpToCpu::new("clc_stats"))),
            ],
            None,
        )
        .run(
            &self.clc_stats,
            clc_input,
            ctx.branch(Some("motif_stats_sync"), false),
            Contexts::new(
                NoopInputContext(None),
                NoopRunnableContext {},
                MaxParallelAndDumpToCpuAdjustableContex::new(MaxParallelAdjustContextImpl::new(), DumpToCpuContextImpl::new(&required_ids)),
            ),
        )
        .await?;

        debug!("MotifStats::run: done input_id={input_id:?} stats.id={:?}", stats.id());

        Ok(Arc::new(DevicesWrap::wrap(stats)))
    }
}

impl<ML, L, C> Adjustable for MotifStats<ML, L, C>
where
    C: RunCtx + Adjustable + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + 'static,
    ML: for<'i> AsyncStage<In<'i> = MotifLoaderInput, Out = thoosa_shared::data_with_extra::DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>, Ctx = C> + Cached + OutId + 'static,
{}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="random_non_overlapping_slices"))]
pub fn random_non_overlapping_slices(
    resources: &[Resource],
    number_of_slices: u32,
    slice_width: u32,
    input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync>,
) -> Result<(Vec<TensorIdSliceIdx>, HashMap<ResourceId, Resource>), ThoError> {

    debug!("random_non_overlapping_slices: resources={resources:?} number_of_slices={number_of_slices:?} slice_width={slice_width:?}");

    assert_ne!(number_of_slices, 0, "number_of_slices is 0");
    assert_ne!(slice_width, 0, "slice_width is 0");

    #[derive(Clone)]
    struct ResState {
        res: Resource,
        starts: Vec<u32>, // available slice starts (aligned, non-overlapping by construction)
        picked: usize,
    }

    let mut rng:StdRng = rand::make_rng();

    let mut states: Vec<ResState> = resources
        .iter()
        .filter_map(|r| {
            let resource_len = input_len_adjuster(r.length());

            if resource_len < slice_width {
                return None;
            }
            let bins = resource_len / slice_width; // number of non-overlapping aligned slices
            if bins == 0 {
                return None;
            }
            let mut starts: Vec<u32> = (0..bins).map(|i| i * slice_width).collect();
            starts.shuffle(&mut rng);
            Some(ResState {
                res: r.clone(),
                starts,
                picked: 0,
            })
        })
        .collect();

    if states.is_empty() {

        return Err(ThoError::new(format!("random_non_overlapping_slices: no suitable resource for {number_of_slices} of {slice_width} width")));
    }

    let available_total: usize = states.iter().map(|s| s.starts.len()).sum();
    if available_total < number_of_slices as usize {
        error!("random_non_overlapping_slices: no suitable resource for {number_of_slices} of {slice_width} width, slice_width={slice_width} resources={resources:?}");
        return Err(ThoError::new(format!("random_non_overlapping_slices: not enough resources ({available_total}) available for {number_of_slices} random slices")));
        // warn!("random_non_overlapping_slices: not enough resources ({available_total}) available for {number_of_slices} random slices");
    }

    let mut chosen: Vec<TensorIdSliceIdx> = Vec::with_capacity(number_of_slices as usize);
    let mut used_resources: HashMap<ResourceId, Resource> = HashMap::new();

    while chosen.len() < number_of_slices as usize {
        // Find the minimum picked-count among resources that still have available slices.
        let mut min_picked: Option<usize> = None;
        for s in &states {
            if s.starts.is_empty() {
                continue;
            }
            min_picked = Some(match min_picked {
                None => s.picked,
                Some(m) => m.min(s.picked),
            });
        }


        let min_p = min_picked.expect("no slices");

        // Candidates = those with min_picked and remaining starts.
        let mut candidates: Vec<usize> = Vec::new();
        for (i, s) in states.iter().enumerate() {
            if !s.starts.is_empty() && s.picked == min_p {
                candidates.push(i);
            }
        }

        let pick_i = *candidates.choose(&mut rng).expect("cant choose slices");

        let s = &mut states[pick_i];
        let start = s.starts.pop().expect("no more slices");
        s.picked += 1;

        chosen.push(TensorIdSliceIdx(s.res.id(), start));
        used_resources.entry(s.res.id()).or_insert(s.res.clone());
    }

    Ok((chosen, used_resources))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::kind::FLOAT_CPU;
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_properties_map::ToHashmap;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::cachable::WithDataRef;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::motif_loader::MotifLoader;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_random_stats_basic() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        unsafe { mock_global_ctx::USE_REAL_CTX = true; }
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 1000)));
        let resources = vec![
            Resource::new(1, 1, "resource_a", 10),
            Resource::new(2, 1, "resource_b", 10),
            Resource::new(3, 1, "resource_c", 10),
            Resource::new(4, 1, "resource_d", 10),
        ];

        let run_id = 42;
        let motif_width = 6;
        let store = Arc::new(
            MockStoreBuilder::new()
                .with_list_random_resources([(None, Ok(resources.clone()))])
                .with_list_run_config_values(
                    run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(run_id), 1, motif_width, None, None)
                        .to_properties_iter(None)
                        .collect()),
                )
                .build(),
        );

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resources[0].clone(), Tensor::full(30, 0., FLOAT_CPU).view((3, -1))),
                    (resources[1].clone(), Tensor::full(30, 1., FLOAT_CPU).view((3, -1))),
                    (resources[2].clone(), Tensor::full(30, 2., FLOAT_CPU).view((3, -1))),
                    (resources[3].clone(), Tensor::full(30, 3., FLOAT_CPU).view((3, -1))),
                ])
                .finish(),
        );

        // let stage = Arc::new(RandomStats::new(Arc::clone(&store), loader, &ws));
        let stage = Arc::new(MotifStats::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            Arc::clone(&ws),
            Arc::new(MotifLoader::new(Arc::clone(&store), Arc::clone(&loader), &ws)),
        ));

        let ctx = Arc::new(RwLock::new(mock_run_ctx::default(1)));
        // let input = Arc::new(DevicesWrap::wrap(RandomStatsInput::new(4, run_id, Metrics::EucDist, Aggr::Mean(1.0))));
        let input = Arc::new(DevicesWrap::wrap(MotifStatsInput::from_random(4, run_id, vec![Metrics::EucDist], Aggr::Mean(1.0))));

        let output = Arc::clone(&stage)
            .run(
                input,
                thoosa_shared::with_perf_log(
                    || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                    "RandomStats/test/write_ctx",
                )
                    .branch(Some("random_stats"), false),
            )
            .await?;

        let mut ctx_for_data = thoosa_shared::with_perf_log(
            || ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "RandomStats/test/write_ctx",
        )
            .branch(Some("random_stats_data"), false);
        let size = output
            .with_data_ref(0, ctx_for_data, None, |stats: &MotifStatsOutput, _ctx| {
                let t = stats.metric_values(Metrics::EucDist).expect("no dists");
                debug!("t=\n{}", t.as_ref());
                assert!(Tensor::from_slice(&[4.0825f32, 4.0825, 4.0825]).view((3, 1)).allclose(t.as_ref(), 0.02, 0.02, false));

                Ok::<_, ThoError>(t.t_ref().size())
            })
            .await?;

        assert_eq!(size, vec![3, 1]);

        Ok(())
    }

    #[test]
    fn test_random_non_overlapping_slices_uses_adjusted_length() -> Result<(), ThoError> {
        let resources = vec![
            Resource::new(1, 1, "resource_a", 10),
            Resource::new(2, 1, "resource_b", 10),
        ];
        let motif_width = 10;
        let input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync> = Arc::new(|len| len);

        let (slices, resources_map) = random_non_overlapping_slices(&resources, 2, motif_width, input_len_adjuster)?;
        assert_eq!(slices.len(), 2);
        assert_eq!(resources_map.len(), 2);
        assert!(slices.iter().all(|slice| slice.1 == 0));

        Ok(())
    }

    #[test]
    fn test_random_non_overlapping_slices_filters_resources_by_adjusted_length() {
        let resources = vec![Resource::new(1, 1, "resource_a", 10)];
        let motif_width = 10;
        let input_len_adjuster: Arc<dyn Fn(u32) -> u32 + Send + Sync> = Arc::new(|len| len - 1);

        let err = random_non_overlapping_slices(&resources, 2, motif_width, input_len_adjuster).expect_err("expected adjusted-length filtering error");

        assert!(format!("{err:?}").contains("no suitable resource for 2 of 10 width"));
    }
}