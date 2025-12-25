use crate::applicables::{ThoDevice};
use crate::dims::{ChannelsData, ChannelsDataFiltered, ChannelsSlicesData, Dims, TCD, TCDF, TCSD};
use crate::loader::SyncWrapper;
use crate::tho_tensor::ThoTensor;
use crate::transforms::filters::*;
use crate::ws::Workspace;
use async_trait::async_trait;
use log::Level::Debug as DebugLevel;
use log::{debug, log_enabled, warn};
use parking_lot::{RwLock, RwLockReadGuard};
use serde::Deserialize;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use tch::Device::Cpu;
use tch::Kind::Float;
use tch::{Device, Tensor};
use thoosa_properties_map::FromHashmap;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{CachableId, Input, Output, ApplicableRunCtx, Runner, ToOrdinal};
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::run_configs::{RunConfig, RunId};
use thoosa_shared::traits::WithId;
use thoosa_shared::val::Val;

//TODO move to find, + nets move out of common
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
#[run_config_grid]
pub struct FindMotifsRunConfig {
    #[in_map]
    pub id: Option<RunId>, //TODO move to RunConfigHolder

    #[run_config]
    #[in_map]
    pub step: u32,

    #[run_config]
    #[in_map]
    pub window: u32,

    #[run_config]
    #[in_map]
    pub min_tracker_step: Option<u32>,

    #[run_config]
    #[in_map]
    pub filter: Option<InputFilter>,

    #[run_config]
    #[in_map]
    pub slices_filter: SlicesFilter,
}

impl FindMotifsRunConfig {
    //TODO RunConfigHolder
    pub fn from_properties_map__(
        id: RunId,
        properties: &HashMap<String, Val>,
    ) -> Result<FindMotifsRunConfig, thoosa_shared::ThoosaError> {
        let mut find_motifs_run_config = FindMotifsRunConfig::from_properties_map(None, properties)?;

        debug!("FindMotifsRunConfig::from_properties_map__: find_motifs_run_config={find_motifs_run_config:?}");
        
        find_motifs_run_config.set_id(id);

        Ok(find_motifs_run_config)
    }

}

pub enum FindMotifsRunConfigOptions {
    Filter,
    FilterUnfold,
}


pub struct DummyRunner;
#[derive(Debug)]
pub struct DummyRunnerCtx;
impl ApplicableRunCtx for DummyRunnerCtx {
    fn cache_area(&self) -> u64 {
        todo!()
    }

    fn into_cached_iter_mut(self) -> impl Iterator<Item = Vec<CachableId>> {
        [].into_iter()
    }
}
impl Runner for DummyRunner {
    type Ctx = DummyRunnerCtx;
}

impl FindMotifsRunConfig {
    pub fn new(
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
            min_tracker_step: None,
            slices_filter: SlicesFilter::DistanceRunningMeanK(DistanceRunningMeanK {
                mean: mean_k.unwrap_or(0.5),
            }),
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
    DistanceRunningMeanK(DistanceRunningMeanK),
    MaxDistance(MaxDistance),
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct DistanceRunningMeanK {
    #[in_map]
    #[run_config]
    pub mean: f32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct MaxDistance {
    #[in_map]
    #[run_config]
    pub distance: f32,
}

#[cfg(test)]
mod tests {
    use crate::find_motifs_run_config::{DistanceRunningMeanK, DistanceRunningMeanKRangeOrValue, FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, InputFilterRangeOrValue, SlicesFilterRangeOrValue};
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
            min_tracker_step: None,
            filter: Some(InputFilterRangeOrValue::BandPassFilter(
                BandPassFilterParamsRangeOrValue {
                    low_hz: 10f64.into(),
                    band_hz: 300f64.into(),
                    kernel_size: 12000i64.into(),
                    stride: 1i64.into(),
                    padding: 1i64.into(),
                },
            )),
            slices_filter: SlicesFilterRangeOrValue::DistanceRunningMeanK(
                DistanceRunningMeanKRangeOrValue { mean: 0.5.into() },
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
    pub motif_selector: MotifSelector,
}

impl SelectMotifsRunConfig {

    // #[cfg(test)]
    pub fn new_(id: Option<RunId>, motif_selector: MotifSelector) -> Self {
        Self { id, motif_selector }
    }

    pub fn new() -> Self {
        Self {
            id: None,
            motif_selector: MotifSelector::ByLabelVariance(ByLabelVarianceConfig::default()),
        }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct ByLabelVarianceConfig {
    #[in_map]
    pub _dummy: u32, //TODO fix ThoosaRunConfig to let empty configs
}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub enum MotifSelector {
    ByLabelVariance(ByLabelVarianceConfig), //TODO
}
