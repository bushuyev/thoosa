#![allow(non_snake_case, unused_variables, unused_assignments, non_camel_case_types)]

use crate::transforms::filters::{InputFilter, InputFilterBuilder, InputFilterFields, InputFilterRangeOrValue};
use crate::transforms::filters::InputFilterLengthAdjuster;
use log::debug;
use serde::Deserialize;
use std::collections::HashMap;
use std::str::FromStr;
use std::sync::Arc;
use thoosa_properties_map::FromHashmap;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::run_configs::{RunConfig, RunId};
use thoosa_shared::val::Val;

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
#[run_config_grid]
pub struct SeqScanRunConfig {
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
    pub metric: SeqScanMetric,

    #[run_config]
    #[in_map]
    pub buckets_count: u32,
}

impl SeqScanRunConfig {
    pub fn from_properties_map_with_id(
        id: RunId,
        properties: &HashMap<String, Val>,
    ) -> Result<Self, thoosa_shared::ThoError> {
        let mut seq_scan_run_config = Self::from_properties_map(None, properties)?;

        debug!("SeqScanRunConfig::from_properties_map_with_id: seq_scan_run_config={seq_scan_run_config:?}");

        seq_scan_run_config.set_id(id);

        Ok(seq_scan_run_config)
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
            .map(|filters| filters.iter().map(|filter| filter.adjust_length()).collect::<Vec<_>>())
            .unwrap_or_default();

        Arc::new(move |len: u32| adjusters.iter().fold(len, |acc, adjust| adjust(acc)))
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)]
pub enum SeqScanMetric {
    #[run_config_default]
    MeanAbs(MeanAbsSeqScanMetricConfig),
    MeanPower(MeanPowerSeqScanMetricConfig),
}

impl Default for SeqScanMetric {
    fn default() -> Self {
        Self::MeanAbs(MeanAbsSeqScanMetricConfig::default())
    }
}

impl FromStr for SeqScanMetric {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        serde_json5::from_str(s).map_err(|e| format!("parse SeqScanMetric from `{s}`: {e}"))
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy, Default)]
pub struct MeanAbsSeqScanMetricConfig {
    #[in_map]
    #[run_config]
    pub center: bool,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone, Copy)]
pub struct MeanPowerSeqScanMetricConfig {
    #[in_map]
    #[run_config]
    pub power: f64,

    #[in_map]
    #[run_config]
    pub center: bool,
}

impl Default for MeanPowerSeqScanMetricConfig {
    fn default() -> Self {
        Self { power: 2.0, center: false }
    }
}

#[cfg(test)]
mod tests {
    use crate::seq_scan_run_config::{MeanAbsSeqScanMetricConfig, MeanPowerSeqScanMetricConfig, MeanPowerSeqScanMetricConfigRangeOrValue, SeqScanMetric, SeqScanMetricRangeOrValue, SeqScanRunConfig, SeqScanRunConfigRangeOrValue};
    use crate::transforms::filters::{AverageFilterParamsRangeOrValue, InputFilterRangeOrValue};
    use log::debug;
    use std::collections::HashMap;
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::GridRunConfigSource;
    use thoosa_shared::RangeOrValue;

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_seq_scan_run_config_grid() {
        let rc = SeqScanRunConfigRangeOrValue {
            id: None,
            step: 25u32.into(),
            window: 100u32.into(),
            freq_filter: Some(vec![InputFilterRangeOrValue::AverageFilter(AverageFilterParamsRangeOrValue { window: 5u32.into(), step: 1u32.into() })]),
            metric: SeqScanMetricRangeOrValue::MeanPower(MeanPowerSeqScanMetricConfigRangeOrValue { power: 3.0.into(), center: RangeOrValue::Value(true) }),
            buckets_count: 8u32.into(),
        };

        let grid = rc.grid_run_configs().expect("SeqScanRunConfigRangeOrValue::grid_run_configs should not fail");
        let run_configs: Vec<SeqScanRunConfig> = grid.to_run_configs();

        debug!("run_configs {run_configs:?}");

        assert_eq!(1, run_configs.len());
        assert_eq!(25, run_configs[0].step);
        assert_eq!(8, run_configs[0].buckets_count);
        assert_eq!(SeqScanMetric::MeanPower(MeanPowerSeqScanMetricConfig { power: 3.0, center: true }), run_configs[0].metric);
    }

    #[test]
    fn test_seq_scan_input_len_adjuster_without_filters_is_identity() {
        let config = SeqScanRunConfig {
            id: None,
            step: 10,
            window: 50,
            freq_filter: None,
            metric: SeqScanMetric::MeanAbs(MeanAbsSeqScanMetricConfig::default()),
            buckets_count: 4,
        };

        let adjust = config.input_len_adjuster();

        assert_eq!(1000, adjust(1000));
    }

    #[test]
    fn test_seq_scan_run_config_to_properties_iter() {
        let config = SeqScanRunConfig {
            id: Some(11),
            step: 10,
            window: 50,
            freq_filter: None,
            metric: SeqScanMetric::MeanAbs(MeanAbsSeqScanMetricConfig { center: true }),
            buckets_count: 4,
        };

        let properties = config.to_properties_iter(None).collect::<HashMap<_, _>>();

        assert!(!properties.is_empty());
        assert!(properties.keys().any(|key| key.contains("step")));
        assert!(properties.keys().any(|key| key.contains("window")));
        assert!(properties.keys().any(|key| key.contains("metric")));
        assert!(properties.keys().any(|key| key.contains("buckets_count")));
    }
}
