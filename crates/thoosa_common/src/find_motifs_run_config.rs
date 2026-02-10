#![allow(non_snake_case, unused_variables, unused_assignments, non_camel_case_types)]

use crate::transforms::filters::*;
use log::debug;
use serde::Deserialize;
use std::collections::HashMap;
use thoosa_properties_map::{FromHashmap, ToHashmap};
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::run_configs::{RunConfig, RunId};
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
    pub filter: Option<InputFilter>,

    #[run_config]
    #[in_map]
    pub slices_filter: SlicesFilter,
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

}

impl FindMotifsRunConfig {
    //TODO rename to new_with_accumulate_mean
    pub fn new_with_running_mean(
        id: Option<u32>,
        step: u32,
        window: u32,
        filter: Option<InputFilter>,
        mean_k: Option<f32>,
    ) -> Self {
        Self {
            id,
            step,
            window,
            filter,
            slices_filter: SlicesFilter::AccumulateStats(AccumulateStats {
                aggr: Aggr::Mean(MeanAggr { k: mean_k.unwrap_or(0.5) }),
            }),
        }
    }

    //TODO rename to new_with_precalculate_stats
    pub fn new_with_max_dist(
        id: Option<u32>,
        step: u32,
        window: u32,
        filter: Option<InputFilter>,
        k: f32,
        slices: u32,
    ) -> Self {
        Self {
            id,
            step,
            window,
            filter,
            slices_filter: SlicesFilter::PrecalculateStats(PrecalculateStats { aggr: Aggr::Mean(MeanAggr { k }), slices}),
        }
    }


    pub fn window(&self) -> u32 {
        self.window
    }

    pub fn step(&self) -> u32 {
        self.step
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub enum InputFilter {
    Conv1(Conv1Params),
    BandPassFilter(BandPassFilterParams),
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy, Default)]
pub struct EucDistConfig {

}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy, Default)]
pub struct PearsonCorrConfig {

}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub enum ComparisonMethod {
    #[run_config_default]
    EucDist(EucDistConfig),
    PearsonCorr(PearsonCorrConfig),
}

impl Default for ComparisonMethod {
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

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct Conv1Params {
    #[in_map]
    #[run_config]
    kernel: i64,

    #[in_map]
    #[run_config]
    stride: i64,

    #[in_map]
    #[run_config]
    padding: i64,
}

impl Conv1Params {
    pub fn new(kernel: i64, stride: i64, padding: i64) -> Self {
        Self {
            kernel,
            stride,
            padding,
        }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub enum SlicesFilter {
    AccumulateStats(AccumulateStats),
    PrecalculateStats(PrecalculateStats),
}

#[derive(ThoosaRunConfig, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct AccumulateStats {
    #[run_config]
    pub aggr: Aggr,
}

impl ToHashmap for AccumulateStats {
    fn to_properties_iter(&self, parent: Option<String>) -> Box<dyn Iterator<Item=(String, thoosa_shared::val::Val)>> {
        let prefix = parent.unwrap_or("AccumulateStats::".to_string());
        self.aggr.to_properties_iter(Some(thoosa_shared::conc_str(&prefix, "aggr.")))
    }
}

impl FromHashmap for AccumulateStats {
    fn from_properties_map(parent: Option<String>, properties: &HashMap<String, thoosa_shared::val::Val>) -> Result<Self, thoosa_shared::ThoError> {
        let prefix = parent.unwrap_or("AccumulateStats::".to_string());
        Ok(Self {
            aggr: Aggr::from_properties_map(Some(thoosa_shared::conc_str(&prefix, "aggr.")), properties)?,
        })
    }
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

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use crate::find_motifs_run_config::{AccumulateStats, AccumulateStatsRangeOrValue, AggrRangeOrValue, ComparisonMethodRangeOrValue, EucDistConfigRangeOrValue, ExtendMotifsConfigRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, InlierRatioRunConfigRangeOrValue, InputFilterRangeOrValue, MeanAggrRangeOrValue, MotifCriteriaRangeOrValue, NormThicknessRunConfigRangeOrValue, SelectMotifsRunConfig, SelectMotifsRunConfigRangeOrValue, SlicesFilterRangeOrValue};
    use crate::transforms::filters::BandPassFilterParamsRangeOrValue;
    use log::debug;
    use tch::Tensor;
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::GridRunConfigSource;
    use thoosa_shared::RangeOrValue;

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
            filter: Some(InputFilterRangeOrValue::BandPassFilter(
                BandPassFilterParamsRangeOrValue {
                    low_hz: 10f64.into(),
                    band_hz: 300f64.into(),
                    kernel_size: 12000i64.into(),
                    stride: 1i64.into(),
                    padding: 1i64.into(),
                },
            )),
            slices_filter: SlicesFilterRangeOrValue::AccumulateStats(
                AccumulateStatsRangeOrValue { aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: 0.5.into() }) },
            ),
        };
        // debug!("{:#}", serde_json5::to_string(&rc).unwrap());
        let grid = rc.grid_run_configs();
        // debug!("grid: {grid:?}");

        let run_configs:Vec<FindMotifsRunConfig> = grid.to_run_configs();

        let map = run_configs[0].to_properties_iter(None);

        debug!("run_configs {run_configs:?}");

        for v in map {
            println!("{:?}", v);
        }

    }

    #[test]
    fn test_select_motifs_run_config_grid_with_criteria() {
        let rc = SelectMotifsRunConfigRangeOrValue {
            id: Some(7),
            extend_config: Some(ExtendMotifsConfigRangeOrValue {
                slices_selector_metric: ComparisonMethodRangeOrValue::EucDist(EucDistConfigRangeOrValue {}),
                slices_selector_aggr: AggrRangeOrValue::Mean(MeanAggrRangeOrValue { k: RangeOrValue::Value(1.0) }),
            }),
            motif_criteria: MotifCriteriaRangeOrValue {
                norm_thickness: Some(NormThicknessRunConfigRangeOrValue { eps: RangeOrValue::Value(1e-8) }),
                inlier_ratioRunConfig: Some(InlierRatioRunConfigRangeOrValue { k: RangeOrValue::Value(3.0), eps: RangeOrValue::Value(1e-8) }),
                coherenceRunConfig: None,
            },
            threshold: RangeOrValue::Range { range: 0.4..0.6, step: Some((0.1, 1)) },
        };

        let run_configs: Vec<SelectMotifsRunConfig> = rc.grid_run_configs().to_run_configs();
        assert_eq!(run_configs.len(), 3);
        assert_eq!(run_configs[0].id, Some(7));
        assert!(run_configs[0].motif_criteria.norm_thickness.is_some());
        assert!(run_configs[0].motif_criteria.inlier_ratioRunConfig.is_some());
        assert!(run_configs[0].motif_criteria.coherenceRunConfig.is_none());
        assert!((run_configs[0].threshold - 0.4).abs() < 1e-6, "unexpected threshold: {}", run_configs[0].threshold);
        assert!((run_configs[1].threshold - 0.5).abs() < 1e-6, "unexpected threshold: {}", run_configs[1].threshold);
        assert!((run_configs[2].threshold - 0.6).abs() < 1e-6, "unexpected threshold: {}", run_configs[2].threshold);
    }
}

#[derive(ThoosaRunConfig, Clone, PartialEq, Debug, Deserialize)]
pub struct ExtendMotifsConfig {
    #[run_config]
    pub slices_selector_metric: ComparisonMethod,

    #[run_config]
    pub slices_selector_aggr: Aggr,
}

impl Default for ExtendMotifsConfig {
    fn default() -> Self {
        Self {
            slices_selector_metric: ComparisonMethod::EucDist(EucDistConfig::default()),
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
            slices_selector_metric: ComparisonMethod::from_properties_map(Some(thoosa_shared::conc_str(&prefix, "slices_selector_metric.")), properties)?,
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

#[derive(ThoosaRunConfig, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct MotifCriteria {
    #[run_config]
    pub norm_thickness: Option<NormThicknessRunConfig>,

    #[run_config]
    pub inlier_ratioRunConfig: Option<InlierRatioRunConfig>,

    #[run_config]
    pub coherenceRunConfig: Option<CoherenceRunConfig>,
}

impl MotifCriteria {
    pub fn is_empty(&self) -> bool {
        self.norm_thickness.is_none() && self.inlier_ratioRunConfig.is_none() && self.coherenceRunConfig.is_none()
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

        Ok(Self {
            norm_thickness: if properties.keys().any(|key| key.starts_with(norm_thickness_prefix.as_str())) {
                Some(NormThicknessRunConfig::from_properties_map(Some(norm_thickness_prefix), properties)?)
            } else {
                None
            },
            inlier_ratioRunConfig: if properties.keys().any(|key| key.starts_with(inlier_ratio_prefix.as_str())) {
                Some(InlierRatioRunConfig::from_properties_map(Some(inlier_ratio_prefix), properties)?)
            } else {
                None
            },
            coherenceRunConfig: if properties.keys().any(|key| key.starts_with(coherence_prefix.as_str())) {
                Some(CoherenceRunConfig::from_properties_map(Some(coherence_prefix), properties)?)
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
pub enum MotifCriterion {
    NormThickness(NormThickness),
    InlierRatio(InlierRatio),
    Coherence(Coherence),
}

impl From<&MotifCriteria> for Vec<MotifCriterion> {
    fn from(value: &MotifCriteria) -> Self {
        let mut criteria = Vec::new();
        if let Some(v) = value.norm_thickness.as_ref() {
            criteria.push(MotifCriterion::NormThickness(NormThickness { eps: v.eps }));
        }
        if let Some(v) = value.inlier_ratioRunConfig.as_ref() {
            criteria.push(MotifCriterion::InlierRatio(InlierRatio { k: v.k, eps: v.eps }));
        }
        if let Some(v) = value.coherenceRunConfig.as_ref() {
            criteria.push(MotifCriterion::Coherence(Coherence { k: v.k, eps: v.eps }));
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
    pub threshold: f32,
}

impl SelectMotifsRunConfig {
    fn validate_threshold(threshold: f32) {
        assert!((0.0..=1.0).contains(&threshold), "SelectMotifsRunConfig::threshold must be within 0..1, got {threshold}");
    }

    pub fn new_with_fields(id: Option<RunId>, extend_config: Option<ExtendMotifsConfig>, motif_criteria: MotifCriteria, threshold: f32) -> Self {
        Self::validate_threshold(threshold);
        Self { id, extend_config, motif_criteria, threshold }
    }

    // #[cfg(test)]
    pub fn new_(id: Option<RunId>, _motif_selector: MotifSelector) -> Self {
        Self::new_with_fields(id, Some(ExtendMotifsConfig::default()), MotifCriteria::default(), 0.5)
    }

    pub fn new() -> Self {
        Self::new_with_fields(None, Some(ExtendMotifsConfig::default()), MotifCriteria::default(), 0.5)
    }
}
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct ByAvgEucDist {
    #[in_map]
    #[run_config]
    pub k: f32,
}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub enum MotifSelector {
    ByAvgEucDist(ByAvgEucDist), //TODO
}
