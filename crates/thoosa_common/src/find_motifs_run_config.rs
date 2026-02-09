#![allow(non_snake_case, unused_variables, unused_assignments)]

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
pub struct MetricConfig {
    #[in_map]
    #[run_config]
    pub v: u32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub enum Metrics {
    EucDist(MetricConfig),
    PearsonCorr(MetricConfig),
}

impl Default for Metrics {
    fn default() -> Self {
        Self::EucDist(MetricConfig::default())
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub enum Aggr {
    Mean(MeanAggr),
    Quantile(QuantileAggr),
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub struct MeanAggr {
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
    use crate::find_motifs_run_config::{AccumulateStats, AccumulateStatsRangeOrValue, AggrRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, InputFilterRangeOrValue, MeanAggrRangeOrValue, SlicesFilterRangeOrValue};
    use crate::transforms::filters::BandPassFilterParamsRangeOrValue;
    use log::debug;
    use tch::Tensor;
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::GridRunConfigSource;

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
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_grid]
pub struct SelectMotifsRunConfig {
    #[in_map]
    pub id: Option<RunId>,

    #[in_map]
    #[run_config]
    pub slices_selector_metric: Metrics,

    #[in_map]
    #[run_config]
    pub slices_selector_aggr: Aggr,

    #[in_map]
    #[run_config]
    pub global_k: f32,
}

impl SelectMotifsRunConfig {

    // #[cfg(test)]
    pub fn new_(id: Option<RunId>, motif_selector: MotifSelector) -> Self {
        Self {
            id,
            slices_selector_metric: Metrics::EucDist(MetricConfig::default()),
            slices_selector_aggr: Aggr::Mean(MeanAggr { k: 1.0 }),
            global_k: 0.5,
        }
    }

    pub fn new() -> Self {
        Self {
            id: None,
            slices_selector_metric: Metrics::EucDist(MetricConfig::default()),
            slices_selector_aggr: Aggr::Mean(MeanAggr { k: 1.0 }),
            global_k: 0.5,
        }
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
