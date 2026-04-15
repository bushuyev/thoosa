#![allow(non_snake_case, unused_variables, unused_assignments, non_camel_case_types)]

use crate::transforms::filters::*;
use log::debug;
use serde::Deserialize;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use thoosa_properties_map::{FromHashmap, ToHashmap};
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::F64BitsKey;
use thoosa_shared::run_configs::{RunConfig, RunId};
use thoosa_shared::traits::Named;
use thoosa_shared::val::Val;

//TODO move to find, + nets move out of common
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
#[run_config_grid]
pub struct FindMotifsRunConfig {
    #[in_map]
    pub id: Option<RunId>,

    #[run_config]
    #[in_map]
    pub step: u32,

    #[run_config]
    #[in_map]
    pub window: u32,

    #[run_config]
    #[in_map]
    pub freq_filter: Option<Vec<InputFilter>>,

    #[run_config]
    #[in_map]
    pub similarity_metric: SimilarityMetric,

    #[run_config]
    #[in_map]
    pub slices_filter: SlicesFilter,

    #[run_config]
    #[in_map]
    pub max_slices: Option<MaxSlices>
}

//keys which affect motif statistics - avg distance etc - used to cache random motifs
#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub struct StatsKeys {
    keys:Vec<StatsKey>
}

#[derive(Debug, Hash, PartialEq, Eq, Clone)]
pub enum StatsKey {
    Window(u32),
    ComparisonMethod(SimilarityMetric),
    NotchBandFilterKey{low_hz:F64BitsKey, high_hz:F64BitsKey},
    BandPassFilter{
        low_hz: F64BitsKey,


        band_hz: F64BitsKey,

        kernel_size: u32,
    },
    SavGolFilter {
        window_len: u32,
        poly_order: i64,
        stride: u32,
        padding: Padding,
    },
    MedianFilter {
        window: u32,
    },
    MovingAverageFilter {
        window: u32,
    },
}

impl Named for StatsKeys {//for  CachableIdTr

}

impl From<&NotchBandFilterParams> for StatsKey {
    fn from(value: &NotchBandFilterParams) -> Self {
        StatsKey::NotchBandFilterKey {
            low_hz: value.low_hz.into(), high_hz: value.high_hz.into()
        }
    }
}

impl From<&BandPassFilterParams> for StatsKey {
    fn from(value: &BandPassFilterParams) -> Self {
        StatsKey::BandPassFilter {
            low_hz: value.low_hz.into(),
            band_hz: value.band_hz.into(),
            kernel_size: value.kernel_size,
        }
    }
}

impl From<&SavGolFilterParams> for StatsKey {
    fn from(value: &SavGolFilterParams) -> Self {
        StatsKey::SavGolFilter {
            window_len: value.window_len,
            poly_order: value.poly_order,
            stride: value.stride,
            padding: value.padding,
        }
    }
}

impl From<&MedianFilterParams> for StatsKey {
    fn from(value: &MedianFilterParams) -> Self {
        StatsKey::MedianFilter {
            window: value.window,
        }
    }
}

impl From<&AverageFilterParams> for StatsKey {
    fn from(value: &AverageFilterParams) -> Self {
        StatsKey::MovingAverageFilter {
            window: value.window,
        }
    }
}

impl From<&InputFilter> for StatsKey {
    fn from(value: &InputFilter) -> Self {
        match value {
            InputFilter::NotchBandFilter(filter) =>filter.into(),
            InputFilter::BandPassFilter(filter) => filter.into(),
            InputFilter::SavGolFilter(filter) => filter.into(),
            InputFilter::MedianFilter(filter) => filter.into(),
            InputFilter::AverageFilter(filter) => filter.into(),
        }
    }
}


impl FindMotifsRunConfig {

    pub fn from_properties_map_with_id(
        id: RunId,
        properties: &HashMap<String, Val>,
    ) -> Result<FindMotifsRunConfig, thoosa_shared::ThoError> {
        let mut find_motifs_run_config = FindMotifsRunConfig::from_properties_map(None, properties)?;

        debug!("FindMotifsRunConfig::from_properties_map_with_id: find_motifs_run_config={find_motifs_run_config:?}");
        
        find_motifs_run_config.set_id(id);

        Ok(find_motifs_run_config)
    }

    pub fn to_stats_keys(&self) -> StatsKeys {

        StatsKeys {
            keys: std::iter::once(StatsKey::Window(self.window))
                .chain(std::iter::once(StatsKey::ComparisonMethod(self.similarity_metric)))
                .chain(
                self.freq_filter
                    .as_deref()
                    .into_iter()
                    .flatten()
                    .map(Into::into),

            ).collect(),
        }
    }
}

impl FindMotifsRunConfig {
    //TODO rename to new_with_accumulate_mean
    pub fn new_with_running_mean(
        id: Option<u32>,
        step: u32,
        window: u32,
        freq_filter: Option<Vec<InputFilter>>,
        mean_k: Option<f32>,
    ) -> Self {
        Self {
            id,
            step,
            window,
            freq_filter,
            similarity_metric: SimilarityMetric::default(),
            slices_filter: SlicesFilter::Accumulate(AccumulateStats {
                aggr: Aggr::Mean(MeanAggr { k: mean_k.unwrap_or(0.5) }),
            }),
            max_slices: None,
        }
    }

    //TODO rename to new_with_precalculate_stats
    pub fn new_with_max_dist(
        id: Option<u32>,
        step: u32,
        window: u32,
        freq_filter: Option<Vec<InputFilter>>,
        k: f32,
        slices: u32,
    ) -> Self {
        Self {
            id,
            step,
            window,
            freq_filter,
            similarity_metric: SimilarityMetric::default(),
            slices_filter: SlicesFilter::Precalculate(PrecalculateStats { aggr: Aggr::Mean(MeanAggr { k }), slices}),
            max_slices: None,
        }
    }


    pub fn window(&self) -> u32 {
        self.window
    }

    pub fn step(&self) -> u32 {
        self.step
    }

    pub fn input_len_adjuster(&self) -> Arc<dyn Fn(u32) -> u32 + Send + Sync> {
        let adjusters = self
            .freq_filter
            .as_ref()
            .map(|filters| {
                filters
                    .iter()
                    .map(|filter| filter.adjust_length())
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        Arc::new(move |len: u32| adjusters.iter().fold(len, |acc, adjust| adjust(acc)))
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub enum InputFilter {
    NotchBandFilter(NotchBandFilterParams),
    BandPassFilter(BandPassFilterParams),
    SavGolFilter(SavGolFilterParams),
    MedianFilter(MedianFilterParams),
    AverageFilter(AverageFilterParams),
}

impl InputFilterLengthAdjuster for InputFilter {
    fn adjust_length(&self) -> Box<dyn Fn(u32) -> u32 + Send + Sync> {
        match self {
            InputFilter::NotchBandFilter(filter) => filter.adjust_length(),
            InputFilter::BandPassFilter(filter) => filter.adjust_length(),
            InputFilter::SavGolFilter(filter) => filter.adjust_length(),
            InputFilter::MedianFilter(filter) => filter.adjust_length(),
            InputFilter::AverageFilter(filter) => filter.adjust_length(),
        }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Eq, Hash, Clone, Copy, Default)]
pub struct EucDistConfig {
    #[in_map]
    #[run_config]
    pub z_normalize: bool,
}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Eq, Hash, Clone, Copy, Default)]
pub struct PearsonCorrConfig {

}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Eq, Hash, Clone, Copy, Default)]
pub struct DotProductConfig {

}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum SimilarityMetric {
    #[run_config_default]
    EucDist(EucDistConfig),
    PearsonCorr(PearsonCorrConfig),
    DotProduct(DotProductConfig),
}

impl Default for SimilarityMetric {
    fn default() -> Self {
        Self::EucDist(EucDistConfig::default())
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub enum Aggr {
    Mean(MeanAggr),
    Min(MinAggr),
    Max(MaxAggr),
    Quantile(QuantileAggr),
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub struct MeanAggr {
    #[in_map]
    #[run_config]
    pub k: f32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub struct MinAggr {
    #[in_map]
    #[run_config]
    pub k: f32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub struct MaxAggr {
    #[in_map]
    #[run_config]
    pub k: f32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub struct QuantileAggr {
    #[in_map]
    #[run_config]
    pub q: f32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub struct MaxSlicesByResources {
    #[in_map]
    #[run_config]
    pub k:u32
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub struct MaxSlicesAbs {
    #[in_map]
    #[run_config]
    pub count:u32
}
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub enum MaxSlices{//discard motifs which slices counts bigger than
    ByResourcesCount(MaxSlicesByResources),// K * resources count - if motif has more than k slices in each resource it's probably noise
    Abs(MaxSlicesAbs) // by exact value - if motif has more than count slices it's garbage
}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub enum SlicesFilter {
    Accumulate(AccumulateStats),
    Precalculate(PrecalculateStats),
    ExactS(ExactStats)//same for all motifs
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct AccumulateStats {
    #[in_map]
    #[run_config]
    pub aggr: Aggr,

}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct PrecalculateStats {
    #[in_map]
    #[run_config]
    pub aggr: Aggr,

    #[in_map]
    #[run_config]
    pub slices: u32, //number of random slices to load for statistics calculations
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct ExactStats {


    #[in_map]
    #[run_config]
    pub euc_dist: f32, //euclidian distance threshold, same for all motifs
}


#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use crate::find_motifs_run_config::{AccumulateStats, AccumulateStatsRangeOrValue, AggrRangeOrValue, SimilarityMetric, SimilarityMetricRangeOrValue, EucDistConfigRangeOrValue, ExtendMotifsConfigRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, InlierRatioConstraintRangeOrValue, InlierRatioRunConfigRangeOrValue, InputFilterRangeOrValue, MaxSlicesAbs, MeanAggr, MeanAggrRangeOrValue, MotifCriteria, MotifCriteriaRangeOrValue, NormThicknessConstraintRangeOrValue, NormThicknessRunConfigRangeOrValue, PrecalculateStats, SelectMotifsRunConfig, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue, StatsKey, StatsKeys};
    use crate::find_motifs_run_config::InputFilter::{BandPassFilter, MedianFilter, AverageFilter, NotchBandFilter, SavGolFilter};
    use crate::transforms::filters::{BandPassFilterParams, BandPassFilterParamsRangeOrValue, MedianFilterParams, MedianFilterParamsRangeOrValue, AverageFilterParams, AverageFilterParamsRangeOrValue, NotchBandFilterParams, SavGolFilterParams, Padding, PaddingRangeOrValue, SgZeroPaddingRangeOrValue};
    use log::debug;
    use tch::Tensor;
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::GridRunConfigSource;
    use thoosa_shared::{F64BitsKey, RangeOrValue};
    use crate::find_motifs_run_config::Aggr::Mean;
    use crate::find_motifs_run_config::MaxSlices::Abs;
    use crate::find_motifs_run_config::SlicesFilter::Precalculate;
    use crate::find_motifs_run_config::StatsKey::Window;

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_find_motifs_run_config() {
        let rc = FindMotifsRunConfigRangeOrValue {
            id: None,
            step: 100.into(),
            window: 200.into(),
            freq_filter: Some(vec![InputFilterRangeOrValue::BandPassFilter(
                BandPassFilterParamsRangeOrValue {
                    low_hz: 10f64.into(),
                    band_hz: 300f64.into(),
                    kernel_size: 12000u32.into(),
                    stride: 1u32.into(),
                    padding: PaddingRangeOrValue::Zero(SgZeroPaddingRangeOrValue::default()),
                },
            ), InputFilterRangeOrValue::MedianFilter(MedianFilterParamsRangeOrValue { window: 11u32.into() })]),
            similarity_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
            slices_filter: SlicesFilterRangeOrValue::Accumulate(
                AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: 0.5.into() }) },
            ),
            max_slices: None,
        };
        // debug!("{:#}", serde_json5::to_string(&rc).unwrap());
        let grid = rc.grid_run_configs().expect("FindMotifsRunConfigRangeOrValue::grid_run_configs should not fail");
        // debug!("grid: {grid:?}");

        let run_configs:Vec<FindMotifsRunConfig> = grid.to_run_configs();

        let map = run_configs[0].to_properties_iter(None);

        debug!("run_configs {run_configs:?}");

        for v in map {
            println!("{:?}", v);
        }

    }

    #[test]
    fn test_input_len_adjuster_applies_filters_in_sequence() {
        let config = FindMotifsRunConfig::new_with_running_mean(
            None,
            100,
            100,
            Some(vec![
                NotchBandFilter(NotchBandFilterParams { low_hz: 58.0, high_hz: 62.0 }),
                BandPassFilter(BandPassFilterParams {
                    low_hz: 10.0,
                    band_hz: 300.0,
                    kernel_size: 100,
                    stride: 5,
                    padding: Padding::zero(),
                }),
            ]),
            None,
        );

        let adjust = config.input_len_adjuster();
        assert_eq!(201, adjust(1000));
    }

    #[test]
    fn test_input_len_adjuster_without_filters_is_identity() {
        let config = FindMotifsRunConfig::new_with_running_mean(None, 100, 100, None, None);
        let adjust = config.input_len_adjuster();
        assert_eq!(1000, adjust(1000));
    }

    #[test]
    fn test_input_len_adjuster_savgol_is_identity() {
        let config = FindMotifsRunConfig::new_with_running_mean(
            None,
            100,
            100,
            Some(vec![SavGolFilter(SavGolFilterParams { window_len: 11, poly_order: 3, stride: 1, padding: Padding::replicate() })]),
            None,
        );
        let adjust = config.input_len_adjuster();
        assert_eq!(1000, adjust(1000));
    }

    #[test]
    fn test_input_len_adjuster_median() {
        let config = FindMotifsRunConfig::new_with_running_mean(
            None,
            100,
            100,
            Some(vec![MedianFilter(MedianFilterParams { window: 11 })]),
            None,
        );
        let adjust = config.input_len_adjuster();
        assert_eq!(989, adjust(1000));
    }

    #[test]
    fn test_input_len_adjuster_moving_average() {
        let config = FindMotifsRunConfig::new_with_running_mean(
            None,
            100,
            100,
            Some(vec![AverageFilter(AverageFilterParams { window: 11, step: 1 })]),
            None,
        );
        let adjust = config.input_len_adjuster();
        assert_eq!(989, adjust(1000));
    }

    #[test]
    fn test_select_motifs_run_config_grid_with_criteria() {
        let rc = SelectMotifsRunConfigRangeOrValue {
            id: Some(7),
            extend_config: Some(ExtendMotifsConfigRangeOrValue {
                slices_selector_metric: SimilarityMetricRangeOrValue::EucDist(EucDistConfigRangeOrValue { z_normalize: RangeOrValue::Value(false) }),
                slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Value(1.0) }),
            }),
            motif_criteria: MotifCriteriaRangeOrValue {
                norm_thickness: Some(NormThicknessConstraintRangeOrValue {
                    criterion: NormThicknessRunConfigRangeOrValue {eps: RangeOrValue::Value(1e-8) },
                    min: RangeOrValue::Range { range: 0.4..0.6, step: Some((0.1, 1)) },
                }),
                inlier_ratioRunConfig: Some(InlierRatioConstraintRangeOrValue {
                    criterion: InlierRatioRunConfigRangeOrValue { k: RangeOrValue::Value(3.0), eps: RangeOrValue::Value(1e-8) },
                    min: RangeOrValue::Value(MotifCriteria::DEFAULT_CRITERION_MIN),
                }),
                coherenceRunConfig: None,
                interesting_amp_01: None,
                interesting_tv_norm_01: None,
                interesting_curvature_norm_01: None,
                interesting_turning_points_ratio: None,
                interesting_band_energy_ratio_01: None,
                interesting_entropy_gauss_01: None,
                interesting_entropy_band_01: None,
                periodicity_acf_01: None,
            },
            extend_max_slices: None,
        };

        let run_configs: Vec<SelectMotifsRunConfig> = rc.grid_run_configs().expect("SelectMotifsRunConfigRangeOrValue::grid_run_configs should not fail").to_run_configs();
        assert_eq!(run_configs.len(), 3);
        assert_eq!(run_configs[0].id, Some(7));
        assert!(run_configs[0].motif_criteria.norm_thickness.is_some());
        assert!(run_configs[0].motif_criteria.inlier_ratioRunConfig.is_some());
        assert!(run_configs[0].motif_criteria.coherenceRunConfig.is_none());
        assert!((run_configs[0].motif_criteria.norm_thickness.as_ref().expect("missing norm_thickness").min - 0.4).abs() < 1e-6);
        assert!((run_configs[1].motif_criteria.norm_thickness.as_ref().expect("missing norm_thickness").min - 0.5).abs() < 1e-6);
        assert!((run_configs[2].motif_criteria.norm_thickness.as_ref().expect("missing norm_thickness").min - 0.6).abs() < 1e-6);
    }

    #[test]
    fn test_to_stats_keys(){
        let run_config = FindMotifsRunConfig { id: Some(246), step: 100, window: 1000, freq_filter: Some(vec![BandPassFilter(BandPassFilterParams { low_hz: 20.0, band_hz: 2.0, kernel_size: 10, stride: 10, padding: Padding::zero() }), NotchBandFilter(NotchBandFilterParams { low_hz: 59.0, high_hz: 62.0 })]), similarity_metric: SimilarityMetric::default(), slices_filter: Precalculate(PrecalculateStats { aggr: Mean(MeanAggr { k: 0.09006707 }), slices: 1000 }), max_slices: Some(Abs(MaxSlicesAbs { count: 1000 })) };

        let keys = run_config.to_stats_keys();
        
        assert_eq!(
            keys,
            StatsKeys { keys: vec![
                StatsKey::Window(1000),
                StatsKey::ComparisonMethod(SimilarityMetric::default()),
                StatsKey::BandPassFilter { low_hz: F64BitsKey(4626322717216342016), band_hz: F64BitsKey(4611686018427387904), kernel_size: 10 },
                StatsKey::NotchBandFilterKey { low_hz: F64BitsKey(4633500329122463744), high_hz: F64BitsKey(4633922541587529728) }
            ] }
        );

    }

    #[test]
    fn test_to_stats_keys_with_savgol() {
        let run_config = FindMotifsRunConfig {
            id: Some(246),
            step: 100,
            window: 1000,
            freq_filter: Some(vec![SavGolFilter(SavGolFilterParams { window_len: 11, poly_order: 2, stride: 1, padding: Padding::reflect() })]),
            similarity_metric: SimilarityMetric::default(),
            slices_filter: Precalculate(PrecalculateStats { aggr: Mean(MeanAggr { k: 0.09006707 }), slices: 1000 }),
            max_slices: Some(Abs(MaxSlicesAbs { count: 1000 })),
        };

        let keys = run_config.to_stats_keys();

        assert_eq!(
            keys,
            StatsKeys {
                keys: vec![
                    StatsKey::Window(1000),
                    StatsKey::ComparisonMethod(SimilarityMetric::default()),
                    StatsKey::SavGolFilter { window_len: 11, poly_order: 2, stride: 1, padding: Padding::reflect() },
                ]
            }
        );
    }

    #[test]
    fn test_to_stats_keys_with_median() {
        let run_config = FindMotifsRunConfig {
            id: Some(246),
            step: 100,
            window: 1000,
            freq_filter: Some(vec![MedianFilter(MedianFilterParams { window: 11 })]),
            similarity_metric: SimilarityMetric::default(),
            slices_filter: Precalculate(PrecalculateStats { aggr: Mean(MeanAggr { k: 0.09006707 }), slices: 1000 }),
            max_slices: Some(Abs(MaxSlicesAbs { count: 1000 })),
        };

        let keys = run_config.to_stats_keys();

        assert_eq!(
            keys,
            StatsKeys {
                keys: vec![
                    StatsKey::Window(1000),
                    StatsKey::ComparisonMethod(SimilarityMetric::default()),
                    StatsKey::MedianFilter { window: 11 },
                ]
            }
        );
    }

    #[test]
    fn test_to_stats_keys_with_moving_average() {
        let run_config = FindMotifsRunConfig {
            id: Some(246),
            step: 100,
            window: 1000,
            freq_filter: Some(vec![AverageFilter(AverageFilterParams { window: 11, step: 1 })]),
            similarity_metric: SimilarityMetric::default(),
            slices_filter: Precalculate(PrecalculateStats { aggr: Mean(MeanAggr { k: 0.09006707 }), slices: 1000 }),
            max_slices: Some(Abs(MaxSlicesAbs { count: 1000 })),
        };

        let keys = run_config.to_stats_keys();

        assert_eq!(
            keys,
            StatsKeys {
                keys: vec![
                    StatsKey::Window(1000),
                    StatsKey::ComparisonMethod(SimilarityMetric::default()),
                    StatsKey::MovingAverageFilter { window: 11 },
                ]
            }
        );
    }

    #[test]
    fn test_motif_criteria_into_vec_with_interesting_variants() {
        let criteria = crate::find_motifs_run_config::MotifCriteria {
            norm_thickness: None,
            inlier_ratioRunConfig: None,
            coherenceRunConfig: None,
            interesting_amp_01: Some(crate::find_motifs_run_config::InterestingAmp01Constraint {
                criterion: crate::find_motifs_run_config::InterestingAmp01RunConfig { q_lo: 0.05, q_hi: 0.95, a0: 0.1, eps: 1e-8 },
                min: crate::find_motifs_run_config::MotifCriteria::DEFAULT_CRITERION_MIN,
            }),
            interesting_tv_norm_01: Some(crate::find_motifs_run_config::InterestingTvNorm01Constraint {
                criterion: crate::find_motifs_run_config::InterestingTvNorm01RunConfig { amp_q_lo: 0.05, amp_q_hi: 0.95, tv0: 0.2, eps: 1e-8 },
                min: crate::find_motifs_run_config::MotifCriteria::DEFAULT_CRITERION_MIN,
            }),
            interesting_curvature_norm_01: Some(crate::find_motifs_run_config::InterestingCurvatureNorm01Constraint {
                criterion: crate::find_motifs_run_config::InterestingCurvatureNorm01RunConfig { amp_q_lo: 0.05, amp_q_hi: 0.95, curv0: 0.3, eps: 1e-8 },
                min: crate::find_motifs_run_config::MotifCriteria::DEFAULT_CRITERION_MIN,
            }),
            interesting_turning_points_ratio: Some(crate::find_motifs_run_config::InterestingTurningPointsRatioConstraint {
                criterion: crate::find_motifs_run_config::InterestingTurningPointsRatioRunConfig { peak: 4 },
                min: crate::find_motifs_run_config::MotifCriteria::DEFAULT_CRITERION_MIN,
            }),
            interesting_band_energy_ratio_01: Some(crate::find_motifs_run_config::InterestingBandEnergyRatio01Constraint {
                criterion: crate::find_motifs_run_config::InterestingBandEnergyRatio01RunConfig { fs: 100.0, f_lo: 5.0, f_hi: 20.0, eps: 1e-8 },
                min: crate::find_motifs_run_config::MotifCriteria::DEFAULT_CRITERION_MIN,
            }),
            interesting_entropy_gauss_01: Some(crate::find_motifs_run_config::InterestingEntropyGauss01Constraint {
                criterion: crate::find_motifs_run_config::InterestingEntropyGauss01RunConfig { h0: 0.6, sigma: 0.2, eps: 1e-8 },
                min: crate::find_motifs_run_config::MotifCriteria::DEFAULT_CRITERION_MIN,
            }),
            interesting_entropy_band_01: Some(crate::find_motifs_run_config::InterestingEntropyBand01Constraint {
                criterion: crate::find_motifs_run_config::InterestingEntropyBand01RunConfig { h_min: 0.2, h_max: 0.8, eps: 1e-8 },
                min: crate::find_motifs_run_config::MotifCriteria::DEFAULT_CRITERION_MIN,
            }),
            periodicity_acf_01: Some(crate::find_motifs_run_config::PeriodicityAcfConstraint {
                criterion: crate::find_motifs_run_config::PeriodicityAcfRunConfig {
                    struct_lag_max: 1./64.0,
                    lag_min: 1./32.0,
                    lag_max: 0.5,
                    struct0: 0.2,
                    peak_good: 0.25,
                    peak_bad: 0.65,
                    eps: 1e-8,
                },
                min: crate::find_motifs_run_config::MotifCriteria::DEFAULT_CRITERION_MIN,
            }),
        };

        let variants: Vec<crate::find_motifs_run_config::MotifCriterion> = Vec::from(&criteria);

        assert_eq!(8, variants.len());
        assert!(matches!(variants[0], crate::find_motifs_run_config::MotifCriterion::InterestingAmp01(_)));
        assert!(matches!(variants[1], crate::find_motifs_run_config::MotifCriterion::InterestingTvNorm01(_)));
        assert!(matches!(variants[2], crate::find_motifs_run_config::MotifCriterion::InterestingCurvatureNorm01(_)));
        assert!(matches!(variants[3], crate::find_motifs_run_config::MotifCriterion::InterestingTurningPointsRatio(_)));
        assert!(matches!(variants[4], crate::find_motifs_run_config::MotifCriterion::InterestingBandEnergyRatio01(_)));
        assert!(matches!(variants[5], crate::find_motifs_run_config::MotifCriterion::InterestingEntropyGauss01(_)));
        assert!(matches!(variants[6], crate::find_motifs_run_config::MotifCriterion::InterestingEntropyBand01(_)));
        assert!(matches!(variants[7], crate::find_motifs_run_config::MotifCriterion::PeriodicityAcf(_)));
    }

    #[test]
    fn test_motif_criteria_periodicity_acf_roundtrips_properties() {
        let criteria = crate::find_motifs_run_config::MotifCriteria {
            periodicity_acf_01: Some(crate::find_motifs_run_config::PeriodicityAcfConstraint {
                criterion: crate::find_motifs_run_config::PeriodicityAcfRunConfig {
                    struct_lag_max: 1./64.,
                    lag_min: 1./32.,
                    lag_max: 0.5,
                    struct0: 0.2,
                    peak_good: 0.25,
                    peak_bad: 0.65,
                    eps: 1e-8,
                },
                min: crate::find_motifs_run_config::MotifCriteria::DEFAULT_CRITERION_MIN,
            }),
            ..Default::default()
        };

        let properties = criteria.to_properties_iter(None).collect::<std::collections::HashMap<_, _>>();
        let restored = <crate::find_motifs_run_config::MotifCriteria as thoosa_properties_map::FromHashmap>::from_properties_map(None, &properties).unwrap();

        assert_eq!(criteria, restored);
    }
}

#[derive(ThoosaRunConfig, Clone, PartialEq, Debug, Deserialize)]
pub struct ExtendMotifsConfig {
    #[run_config]
    pub slices_selector_metric: SimilarityMetric,

    #[run_config]
    pub slices_selector_aggr: Aggr,
}

impl Default for ExtendMotifsConfig {
    fn default() -> Self {
        Self {
            slices_selector_metric: SimilarityMetric::EucDist(EucDistConfig::default()),
            slices_selector_aggr: Aggr::Mean(MeanAggr { k: 1.0 }),
        }
    }
}

impl ToHashmap for ExtendMotifsConfig {
    fn to_properties_iter(&self, parent: Option<String>) -> Box<dyn Iterator<Item=(String, thoosa_shared::val::Val)>> {
        let prefix = parent.unwrap_or("ExtendMotifsConfig::".to_string());
        Box::new(
            self
                .slices_selector_metric
                .to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "slices_selector_metric.")))
                .chain(
                    self.slices_selector_aggr
                        .to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "slices_selector_aggr."))),
                ),
        )
    }
}

impl FromHashmap for ExtendMotifsConfig {
    fn from_properties_map(parent: Option<String>, properties: &HashMap<String, thoosa_shared::val::Val>) -> Result<Self, thoosa_shared::ThoError> {
        let prefix = parent.unwrap_or("ExtendMotifsConfig::".to_string());
        Ok(Self {
            slices_selector_metric: SimilarityMetric::from_properties_map(Some(thoosa_shared::conc_str(&prefix, "slices_selector_metric.")), properties)?,
            slices_selector_aggr: Aggr::from_properties_map(Some(thoosa_shared::conc_str(&prefix, "slices_selector_aggr.")), properties)?,
        })
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct NormThicknessRunConfig {

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct InlierRatioRunConfig {
    #[in_map]
    #[run_config]
    pub k: f64,

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct CoherenceRunConfig {
    #[in_map]
    #[run_config]
    pub k: f64,

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct InterestingAmp01RunConfig {
    #[in_map]
    #[run_config]
    pub q_lo: f64,

    #[in_map]
    #[run_config]
    pub q_hi: f64,

    #[in_map]
    #[run_config]
    pub a0: f64,

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct InterestingTvNorm01RunConfig {
    #[in_map]
    #[run_config]
    pub amp_q_lo: f64,

    #[in_map]
    #[run_config]
    pub amp_q_hi: f64,

    #[in_map]
    #[run_config]
    pub tv0: f64,

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct InterestingCurvatureNorm01RunConfig {
    #[in_map]
    #[run_config]
    pub amp_q_lo: f64,

    #[in_map]
    #[run_config]
    pub amp_q_hi: f64,

    #[in_map]
    #[run_config]
    pub curv0: f64,

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct InterestingTurningPointsRatioRunConfig {
    #[in_map]
    #[run_config]
    pub peak: u32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct InterestingBandEnergyRatio01RunConfig {
    #[in_map]
    #[run_config]
    pub fs: f64,

    #[in_map]
    #[run_config]
    pub f_lo: f64,

    #[in_map]
    #[run_config]
    pub f_hi: f64,

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct InterestingEntropyGauss01RunConfig {
    #[in_map]
    #[run_config]
    pub h0: f64,

    #[in_map]
    #[run_config]
    pub sigma: f64,

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct InterestingEntropyBand01RunConfig {
    #[in_map]
    #[run_config]
    pub h_min: f64,

    #[in_map]
    #[run_config]
    pub h_max: f64,

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct PeriodicityAcfRunConfig {
    #[in_map]
    #[run_config]
    pub struct_lag_max: f64,

    #[in_map]
    #[run_config]
    pub lag_min: f64,

    #[in_map]
    #[run_config]
    pub lag_max: f64,

    #[in_map]
    #[run_config]
    pub struct0: f64,

    #[in_map]
    #[run_config]
    pub peak_good: f64,

    #[in_map]
    #[run_config]
    pub peak_bad: f64,

    #[in_map]
    #[run_config]
    pub eps: f64,
}

#[derive(Clone, PartialEq, Debug, Deserialize, Default)]
pub struct Constraint<T> {
    pub criterion: T,
    pub min: f32,
}

macro_rules! define_constraint_run_config {
    ($name:ident, $criterion_ty:ty) => {
        #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
        pub struct $name {
            #[in_map]
            #[run_config]
            pub criterion: $criterion_ty,

            #[in_map]
            #[run_config]
            pub min: f32,
        }

        impl From<$name> for Constraint<$criterion_ty> {
            fn from(value: $name) -> Self {
                Self {
                    criterion: value.criterion,
                    min: value.min,
                }
            }
        }

        impl From<&$name> for Constraint<$criterion_ty> {
            fn from(value: &$name) -> Self {
                Self {
                    criterion: value.criterion.clone(),
                    min: value.min,
                }
            }
        }
    };
}

define_constraint_run_config!(NormThicknessConstraint, NormThicknessRunConfig);
define_constraint_run_config!(InlierRatioConstraint, InlierRatioRunConfig);
define_constraint_run_config!(CoherenceConstraint, CoherenceRunConfig);
define_constraint_run_config!(InterestingAmp01Constraint, InterestingAmp01RunConfig);
define_constraint_run_config!(InterestingTvNorm01Constraint, InterestingTvNorm01RunConfig);
define_constraint_run_config!(InterestingCurvatureNorm01Constraint, InterestingCurvatureNorm01RunConfig);
define_constraint_run_config!(InterestingTurningPointsRatioConstraint, InterestingTurningPointsRatioRunConfig);
define_constraint_run_config!(InterestingBandEnergyRatio01Constraint, InterestingBandEnergyRatio01RunConfig);
define_constraint_run_config!(InterestingEntropyGauss01Constraint, InterestingEntropyGauss01RunConfig);
define_constraint_run_config!(InterestingEntropyBand01Constraint, InterestingEntropyBand01RunConfig);
define_constraint_run_config!(PeriodicityAcfConstraint, PeriodicityAcfRunConfig);

#[derive(ThoosaRunConfig, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct MotifCriteria {
    #[run_config]
    pub norm_thickness: Option<NormThicknessConstraint>,

    #[run_config]
    pub inlier_ratioRunConfig: Option<InlierRatioConstraint>,

    #[run_config]
    pub coherenceRunConfig: Option<CoherenceConstraint>,

    #[run_config]
    pub interesting_amp_01: Option<InterestingAmp01Constraint>,

    #[run_config]
    pub interesting_tv_norm_01: Option<InterestingTvNorm01Constraint>,

    #[run_config]
    pub interesting_curvature_norm_01: Option<InterestingCurvatureNorm01Constraint>,

    #[run_config]
    pub interesting_turning_points_ratio: Option<InterestingTurningPointsRatioConstraint>,

    #[run_config]
    pub interesting_band_energy_ratio_01: Option<InterestingBandEnergyRatio01Constraint>,

    #[run_config]
    pub interesting_entropy_gauss_01: Option<InterestingEntropyGauss01Constraint>,

    #[run_config]
    pub interesting_entropy_band_01: Option<InterestingEntropyBand01Constraint>,

    #[run_config]
    pub periodicity_acf_01: Option<PeriodicityAcfConstraint>,
}

impl MotifCriteria {
    pub const DEFAULT_CRITERION_MIN: f32 = 0.5;
    pub const NORM_THICKNESS_MIN_KEY: &str = "norm_thickness_min";
    pub const INLIER_RATIO_MIN_KEY: &str = "inlier_ratio_min";
    pub const COHERENCE_MIN_KEY: &str = "coherence_min";
    pub const INTERESTING_AMP_01_MIN_KEY: &str = "interesting_amp_01_min";
    pub const INTERESTING_TV_NORM_01_MIN_KEY: &str = "interesting_tv_norm_01_min";
    pub const INTERESTING_CURVATURE_NORM_01_MIN_KEY: &str = "interesting_curvature_norm_01_min";
    pub const INTERESTING_TURNING_POINTS_RATIO_MIN_KEY: &str = "interesting_turning_points_ratio_min";
    pub const INTERESTING_BAND_ENERGY_RATIO_01_MIN_KEY: &str = "interesting_band_energy_ratio_01_min";
    pub const INTERESTING_ENTROPY_GAUSS_01_MIN_KEY: &str = "interesting_entropy_gauss_01_min";
    pub const INTERESTING_ENTROPY_BAND_01_MIN_KEY: &str = "interesting_entropy_band_01_min";
    pub const PERIODICITY_ACF_01_MIN_KEY: &str = "periodicity_acf_01_min";
    pub const CONSTRAINT_KEYS: [&str; 11] = [
        Self::NORM_THICKNESS_MIN_KEY,
        Self::INLIER_RATIO_MIN_KEY,
        Self::COHERENCE_MIN_KEY,
        Self::INTERESTING_AMP_01_MIN_KEY,
        Self::INTERESTING_TV_NORM_01_MIN_KEY,
        Self::INTERESTING_CURVATURE_NORM_01_MIN_KEY,
        Self::INTERESTING_TURNING_POINTS_RATIO_MIN_KEY,
        Self::INTERESTING_BAND_ENERGY_RATIO_01_MIN_KEY,
        Self::INTERESTING_ENTROPY_GAUSS_01_MIN_KEY,
        Self::INTERESTING_ENTROPY_BAND_01_MIN_KEY,
        Self::PERIODICITY_ACF_01_MIN_KEY,
    ];

    pub fn is_empty(&self) -> bool {
        self.norm_thickness.is_none()
            && self.inlier_ratioRunConfig.is_none()
            && self.coherenceRunConfig.is_none()
            && self.interesting_amp_01.is_none()
            && self.interesting_tv_norm_01.is_none()
            && self.interesting_curvature_norm_01.is_none()
            && self.interesting_turning_points_ratio.is_none()
            && self.interesting_band_energy_ratio_01.is_none()
            && self.interesting_entropy_gauss_01.is_none()
            && self.interesting_entropy_band_01.is_none()
            && self.periodicity_acf_01.is_none()
    }

    pub fn optimizer_constraints() -> HashMap<String, f32> {
        Self::CONSTRAINT_KEYS.into_iter().map(|key| (key.to_string(), 0.0)).collect()
    }

    pub fn failed_constraint_metrics() -> HashMap<String, f32> {
        Self::CONSTRAINT_KEYS.into_iter().map(|key| (key.to_string(), 1.0)).collect()
    }
}

impl ToHashmap for MotifCriteria {
    fn to_properties_iter(&self, parent: Option<String>) -> Box<dyn Iterator<Item=(String, thoosa_shared::val::Val)>> {
        let prefix = parent.unwrap_or("MotifCriteria::".to_string());
        Box::new(
            self
                .norm_thickness
                .as_ref()
                .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "norm_thickness."))))
                .unwrap_or(Box::new(std::iter::empty()))
                .chain(
                    self
                        .inlier_ratioRunConfig
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "inlier_ratioRunConfig."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                )
                .chain(
                    self
                        .coherenceRunConfig
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "coherenceRunConfig."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                )
                .chain(
                    self
                        .interesting_amp_01
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "interesting_amp_01."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                )
                .chain(
                    self
                        .interesting_tv_norm_01
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "interesting_tv_norm_01."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                )
                .chain(
                    self
                        .interesting_curvature_norm_01
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "interesting_curvature_norm_01."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                )
                .chain(
                    self
                        .interesting_turning_points_ratio
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "interesting_turning_points_ratio."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                )
                .chain(
                    self
                        .interesting_band_energy_ratio_01
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "interesting_band_energy_ratio_01."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                )
                .chain(
                    self
                        .interesting_entropy_gauss_01
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "interesting_entropy_gauss_01."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                )
                .chain(
                    self
                        .interesting_entropy_band_01
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "interesting_entropy_band_01."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                )
                .chain(
                    self
                        .periodicity_acf_01
                        .as_ref()
                        .map(|cfg| cfg.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "periodicity_acf_01."))))
                        .unwrap_or(Box::new(std::iter::empty())),
                ),
        )
    }
}

impl FromHashmap for MotifCriteria {
    fn from_properties_map(parent: Option<String>, properties: &HashMap<String, thoosa_shared::val::Val>) -> Result<Self, thoosa_shared::ThoError> {
        let prefix = parent.unwrap_or("MotifCriteria::".to_string());
        let norm_thickness_prefix = thoosa_shared::conc_str(&prefix, "norm_thickness.");
        let inlier_ratio_prefix = thoosa_shared::conc_str(&prefix, "inlier_ratioRunConfig.");
        let coherence_prefix = thoosa_shared::conc_str(&prefix, "coherenceRunConfig.");
        let interesting_amp_01_prefix = thoosa_shared::conc_str(&prefix, "interesting_amp_01.");
        let interesting_tv_norm_01_prefix = thoosa_shared::conc_str(&prefix, "interesting_tv_norm_01.");
        let interesting_curvature_norm_01_prefix = thoosa_shared::conc_str(&prefix, "interesting_curvature_norm_01.");
        let interesting_turning_points_ratio_prefix = thoosa_shared::conc_str(&prefix, "interesting_turning_points_ratio.");
        let interesting_band_energy_ratio_01_prefix = thoosa_shared::conc_str(&prefix, "interesting_band_energy_ratio_01.");
        let interesting_entropy_gauss_01_prefix = thoosa_shared::conc_str(&prefix, "interesting_entropy_gauss_01.");
        let interesting_entropy_band_01_prefix = thoosa_shared::conc_str(&prefix, "interesting_entropy_band_01.");
        let periodicity_acf_01_prefix = thoosa_shared::conc_str(&prefix, "periodicity_acf_01.");

        Ok(Self {
            norm_thickness: if properties.keys().any(|key| key.starts_with(norm_thickness_prefix.as_str())) {
                Some(NormThicknessConstraint::from_properties_map(Some(norm_thickness_prefix), properties)?)
            } else {
                None
            },
            inlier_ratioRunConfig: if properties.keys().any(|key| key.starts_with(inlier_ratio_prefix.as_str())) {
                Some(InlierRatioConstraint::from_properties_map(Some(inlier_ratio_prefix), properties)?)
            } else {
                None
            },
            coherenceRunConfig: if properties.keys().any(|key| key.starts_with(coherence_prefix.as_str())) {
                Some(CoherenceConstraint::from_properties_map(Some(coherence_prefix), properties)?)
            } else {
                None
            },
            interesting_amp_01: if properties.keys().any(|key| key.starts_with(interesting_amp_01_prefix.as_str())) {
                Some(InterestingAmp01Constraint::from_properties_map(Some(interesting_amp_01_prefix), properties)?)
            } else {
                None
            },
            interesting_tv_norm_01: if properties.keys().any(|key| key.starts_with(interesting_tv_norm_01_prefix.as_str())) {
                Some(InterestingTvNorm01Constraint::from_properties_map(Some(interesting_tv_norm_01_prefix), properties)?)
            } else {
                None
            },
            interesting_curvature_norm_01: if properties.keys().any(|key| key.starts_with(interesting_curvature_norm_01_prefix.as_str())) {
                Some(InterestingCurvatureNorm01Constraint::from_properties_map(Some(interesting_curvature_norm_01_prefix), properties)?)
            } else {
                None
            },
            interesting_turning_points_ratio: if properties.keys().any(|key| key.starts_with(interesting_turning_points_ratio_prefix.as_str())) {
                Some(InterestingTurningPointsRatioConstraint::from_properties_map(Some(interesting_turning_points_ratio_prefix), properties)?)
            } else {
                None
            },
            interesting_band_energy_ratio_01: if properties.keys().any(|key| key.starts_with(interesting_band_energy_ratio_01_prefix.as_str())) {
                Some(InterestingBandEnergyRatio01Constraint::from_properties_map(Some(interesting_band_energy_ratio_01_prefix), properties)?)
            } else {
                None
            },
            interesting_entropy_gauss_01: if properties.keys().any(|key| key.starts_with(interesting_entropy_gauss_01_prefix.as_str())) {
                Some(InterestingEntropyGauss01Constraint::from_properties_map(Some(interesting_entropy_gauss_01_prefix), properties)?)
            } else {
                None
            },
            interesting_entropy_band_01: if properties.keys().any(|key| key.starts_with(interesting_entropy_band_01_prefix.as_str())) {
                Some(InterestingEntropyBand01Constraint::from_properties_map(Some(interesting_entropy_band_01_prefix), properties)?)
            } else {
                None
            },
            periodicity_acf_01: if properties.keys().any(|key| key.starts_with(periodicity_acf_01_prefix.as_str())) {
                Some(PeriodicityAcfConstraint::from_properties_map(Some(periodicity_acf_01_prefix), properties)?)
            } else {
                None
            },
        })
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct NormThickness {
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InlierRatio {
    pub k: f64,
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Coherence {
    pub k: f64,
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InterestingAmp01 {
    pub q_lo: f64,
    pub q_hi: f64,
    pub a0: f64,
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InterestingTvNorm01 {
    pub amp_q_lo: f64,
    pub amp_q_hi: f64,
    pub tv0: f64,
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InterestingCurvatureNorm01 {
    pub amp_q_lo: f64,
    pub amp_q_hi: f64,
    pub curv0: f64,
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InterestingTurningPointsRatio {
    pub peak:u32
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InterestingBandEnergyRatio01 {
    pub fs: f64,
    pub f_lo: f64,
    pub f_hi: f64,
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InterestingEntropyGauss01 {
    pub h0: f64,
    pub sigma: f64,
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct InterestingEntropyBand01 {
    pub h_min: f64,
    pub h_max: f64,
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct PeriodicityAcf {
    pub struct_lag_max: f64,
    pub lag_min: f64,
    pub lag_max: f64,
    pub struct0: f64,
    pub peak_good: f64,
    pub peak_bad: f64,
    pub eps: f64,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum MotifCriterion {
    NormThickness(NormThickness),
    InlierRatio(InlierRatio),
    Coherence(Coherence),
    InterestingAmp01(InterestingAmp01),
    InterestingTvNorm01(InterestingTvNorm01),
    InterestingCurvatureNorm01(InterestingCurvatureNorm01),
    InterestingTurningPointsRatio(InterestingTurningPointsRatio),
    InterestingBandEnergyRatio01(InterestingBandEnergyRatio01),
    InterestingEntropyGauss01(InterestingEntropyGauss01),
    InterestingEntropyBand01(InterestingEntropyBand01),
    PeriodicityAcf(PeriodicityAcf),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct MotifCriterionConstraint {
    pub key: &'static str,
    pub criterion: MotifCriterion,
    pub min: f32,
}

impl MotifCriteria {
    pub fn constraints(&self) -> Vec<MotifCriterionConstraint> {
        let mut constraints = Vec::new();
        if let Some(v) = self.norm_thickness.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::NORM_THICKNESS_MIN_KEY,
                criterion: MotifCriterion::NormThickness(NormThickness { eps: v.criterion.eps }),
                min: v.min,
            });
        }
        if let Some(v) = self.inlier_ratioRunConfig.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::INLIER_RATIO_MIN_KEY,
                criterion: MotifCriterion::InlierRatio(InlierRatio { k: v.criterion.k, eps: v.criterion.eps }),
                min: v.min,
            });
        }
        if let Some(v) = self.coherenceRunConfig.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::COHERENCE_MIN_KEY,
                criterion: MotifCriterion::Coherence(Coherence { k: v.criterion.k, eps: v.criterion.eps }),
                min: v.min,
            });
        }
        if let Some(v) = self.interesting_amp_01.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::INTERESTING_AMP_01_MIN_KEY,
                criterion: MotifCriterion::InterestingAmp01(InterestingAmp01 { q_lo: v.criterion.q_lo, q_hi: v.criterion.q_hi, a0: v.criterion.a0, eps: v.criterion.eps }),
                min: v.min,
            });
        }
        if let Some(v) = self.interesting_tv_norm_01.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::INTERESTING_TV_NORM_01_MIN_KEY,
                criterion: MotifCriterion::InterestingTvNorm01(InterestingTvNorm01 { amp_q_lo: v.criterion.amp_q_lo, amp_q_hi: v.criterion.amp_q_hi, tv0: v.criterion.tv0, eps: v.criterion.eps }),
                min: v.min,
            });
        }
        if let Some(v) = self.interesting_curvature_norm_01.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::INTERESTING_CURVATURE_NORM_01_MIN_KEY,
                criterion: MotifCriterion::InterestingCurvatureNorm01(InterestingCurvatureNorm01 { amp_q_lo: v.criterion.amp_q_lo, amp_q_hi: v.criterion.amp_q_hi, curv0: v.criterion.curv0, eps: v.criterion.eps }),
                min: v.min,
            });
        }
        if let Some(v) = self.interesting_turning_points_ratio.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::INTERESTING_TURNING_POINTS_RATIO_MIN_KEY,
                criterion: MotifCriterion::InterestingTurningPointsRatio(InterestingTurningPointsRatio { peak: v.criterion.peak }),
                min: v.min,
            });
        }
        if let Some(v) = self.interesting_band_energy_ratio_01.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::INTERESTING_BAND_ENERGY_RATIO_01_MIN_KEY,
                criterion: MotifCriterion::InterestingBandEnergyRatio01(InterestingBandEnergyRatio01 { fs: v.criterion.fs, f_lo: v.criterion.f_lo, f_hi: v.criterion.f_hi, eps: v.criterion.eps }),
                min: v.min,
            });
        }
        if let Some(v) = self.interesting_entropy_gauss_01.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::INTERESTING_ENTROPY_GAUSS_01_MIN_KEY,
                criterion: MotifCriterion::InterestingEntropyGauss01(InterestingEntropyGauss01 { h0: v.criterion.h0, sigma: v.criterion.sigma, eps: v.criterion.eps }),
                min: v.min,
            });
        }
        if let Some(v) = self.interesting_entropy_band_01.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::INTERESTING_ENTROPY_BAND_01_MIN_KEY,
                criterion: MotifCriterion::InterestingEntropyBand01(InterestingEntropyBand01 { h_min: v.criterion.h_min, h_max: v.criterion.h_max, eps: v.criterion.eps }),
                min: v.min,
            });
        }
        if let Some(v) = self.periodicity_acf_01.as_ref() {
            constraints.push(MotifCriterionConstraint {
                key: Self::PERIODICITY_ACF_01_MIN_KEY,
                criterion: MotifCriterion::PeriodicityAcf(PeriodicityAcf {
                    struct_lag_max: v.criterion.struct_lag_max,
                    lag_min: v.criterion.lag_min,
                    lag_max: v.criterion.lag_max,
                    struct0: v.criterion.struct0,
                    peak_good: v.criterion.peak_good,
                    peak_bad: v.criterion.peak_bad,
                    eps: v.criterion.eps,
                }),
                min: v.min,
            });
        }
        constraints
    }
}

impl From<&MotifCriteria> for Vec<MotifCriterion> {
    fn from(value: &MotifCriteria) -> Self {
        let mut criteria = Vec::new();
        if let Some(v) = value.norm_thickness.as_ref() {
            criteria.push(MotifCriterion::NormThickness(NormThickness { eps: v.criterion.eps }));
        }
        if let Some(v) = value.inlier_ratioRunConfig.as_ref() {
            criteria.push(MotifCriterion::InlierRatio(InlierRatio { k: v.criterion.k, eps: v.criterion.eps }));
        }
        if let Some(v) = value.coherenceRunConfig.as_ref() {
            criteria.push(MotifCriterion::Coherence(Coherence { k: v.criterion.k, eps: v.criterion.eps }));
        }
        if let Some(v) = value.interesting_amp_01.as_ref() {
            criteria.push(MotifCriterion::InterestingAmp01(InterestingAmp01 { q_lo: v.criterion.q_lo, q_hi: v.criterion.q_hi, a0: v.criterion.a0, eps: v.criterion.eps }));
        }
        if let Some(v) = value.interesting_tv_norm_01.as_ref() {
            criteria.push(MotifCriterion::InterestingTvNorm01(InterestingTvNorm01 { amp_q_lo: v.criterion.amp_q_lo, amp_q_hi: v.criterion.amp_q_hi, tv0: v.criterion.tv0, eps: v.criterion.eps }));
        }
        if let Some(v) = value.interesting_curvature_norm_01.as_ref() {
            criteria.push(MotifCriterion::InterestingCurvatureNorm01(InterestingCurvatureNorm01 { amp_q_lo: v.criterion.amp_q_lo, amp_q_hi: v.criterion.amp_q_hi, curv0: v.criterion.curv0, eps: v.criterion.eps }));
        }
        if let Some(v) = value.interesting_turning_points_ratio.as_ref() {
            criteria.push(MotifCriterion::InterestingTurningPointsRatio(InterestingTurningPointsRatio { peak: v.criterion.peak }));
        }
        if let Some(v) = value.interesting_band_energy_ratio_01.as_ref() {
            criteria.push(MotifCriterion::InterestingBandEnergyRatio01(InterestingBandEnergyRatio01 { fs: v.criterion.fs, f_lo: v.criterion.f_lo, f_hi: v.criterion.f_hi, eps: v.criterion.eps }));
        }
        if let Some(v) = value.interesting_entropy_gauss_01.as_ref() {
            criteria.push(MotifCriterion::InterestingEntropyGauss01(InterestingEntropyGauss01 { h0: v.criterion.h0, sigma: v.criterion.sigma, eps: v.criterion.eps }));
        }
        if let Some(v) = value.interesting_entropy_band_01.as_ref() {
            criteria.push(MotifCriterion::InterestingEntropyBand01(InterestingEntropyBand01 { h_min: v.criterion.h_min, h_max: v.criterion.h_max, eps: v.criterion.eps }));
        }
        if let Some(v) = value.periodicity_acf_01.as_ref() {
            criteria.push(MotifCriterion::PeriodicityAcf(PeriodicityAcf {
                struct_lag_max: v.criterion.struct_lag_max,
                lag_min: v.criterion.lag_min,
                lag_max: v.criterion.lag_max,
                struct0: v.criterion.struct0,
                peak_good: v.criterion.peak_good,
                peak_bad: v.criterion.peak_bad,
                eps: v.criterion.eps,
            }));
        }
        criteria
    }
}

impl From<MotifCriteria> for Vec<MotifCriterion> {
    fn from(value: MotifCriteria) -> Self {
        Vec::from(&value)
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_grid]
pub struct SelectMotifsRunConfig {
    #[in_map]
    pub id: Option<RunId>,

    #[in_map]
    #[run_config]
    pub extend_config: Option<ExtendMotifsConfig>,

    #[in_map]
    #[run_config]
    pub motif_criteria: MotifCriteria,

    #[in_map]
    #[run_config]
    pub extend_max_slices: Option<MaxSlices>//TODO move into extend_config
}

impl SelectMotifsRunConfig {
    pub fn new_with_fields(id: Option<RunId>, extend_config: Option<ExtendMotifsConfig>, motif_criteria: MotifCriteria) -> Self {
        Self { id, extend_config, motif_criteria, extend_max_slices: None }
    }

    pub fn new(id: Option<RunId>) -> Self {
        Self::new_with_fields(id, Some(ExtendMotifsConfig::default()), MotifCriteria::default())
    }

}
