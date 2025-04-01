use std::collections::HashMap;
use tch::{Device, Tensor};
use log::{debug, log_enabled, warn};
use log::Level::Debug as DebugLevel;
use tch::Kind::Float;
use tch::Device::Cpu;
use serde::Deserialize;
use parking_lot::RwLockReadGuard;
use std::ops::Deref;
use thoosa_properties_map::FromHashmap;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::run_configs::{Applicable, RunConfig, RunId};
use thoosa_shared::ThoosaError;
use thoosa_shared::val::Val;
use crate::loader::SyncWrapper;
use crate::train_nets::FindMotifsRunConfigCtx;
use crate::transforms::filters::*;
use crate::ws::Workspace;

//TODO move to find, + nets move out of common
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
#[run_config_grid]
pub struct FindMotifsRunConfig {
    #[in_map]
    pub id:Option<RunId>,//TODO move to RunConfigHolder

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
    pub fn from_properties_map__(id:RunId, properties: &HashMap<String, Val>) -> Result<FindMotifsRunConfig, thoosa_shared::ThoosaError> {
        let mut find_motifs_run_config = FindMotifsRunConfig::from_properties_map(None, properties)?;
        find_motifs_run_config.set_id(id);

        Ok(find_motifs_run_config)
    }

    pub fn to_applicable<C:FindMotifsRunConfigCtx>(&self, ctx:C, device: Device) -> FindMotifsRunConfigApplicable {
        FindMotifsRunConfigApplicable::new(&self, ctx, device)
    }

}

//TODO use workspace directly
pub struct FindMotifsRunConfigWsCtx {
    channels:u32,
    sample_rate:u32
}

impl From<RwLockReadGuard<'_, Box<dyn Workspace>>> for FindMotifsRunConfigWsCtx {
    fn from(value: RwLockReadGuard<'_, Box<dyn Workspace>>) -> Self {
        FindMotifsRunConfigWsCtx {
            channels: value.channels() as u32,
            sample_rate: value.sample_rate(),
        }
    }
}

impl FindMotifsRunConfigCtx for FindMotifsRunConfigWsCtx {
    fn channels(&self) -> u32 {
        self.channels
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

#[derive(Debug)]
pub struct FindMotifsRunConfigApplicable {
    run_config:FindMotifsRunConfig,
    kernel:Option<SyncWrapper<Box<Tensor>>>,
    groups:i64,
    device:Device
}

impl FindMotifsRunConfigApplicable {

    pub fn new<C:FindMotifsRunConfigCtx>(run_config:&FindMotifsRunConfig, ctx:C, device:Device) -> Self {

        match &run_config.filter {
            None => {

                Self {
                    run_config: run_config.clone(),
                    kernel:None,
                    groups:0,
                    device
                }
            }

            Some(filter) => {
                match filter {
                    InputFilter::Conv1(_) => {
                        todo!()
                    }
                    InputFilter::BandPassFilter(band_pass_filter_config) => {
                        let kernel = band_pass_filter(
                            band_pass_filter_config.low_hz,
                            band_pass_filter_config.low_hz + band_pass_filter_config.band_hz,
                            band_pass_filter_config.kernel_size,
                            ctx.sample_rate() as f64,
                            1e-6//TODO config?
                        ).view((1, 1, -1)).expand([ctx.channels() as i64,1,-1], true);


                        Self {
                            run_config: run_config.clone(),
                            kernel:Some(kernel.to_device(device).into()),
                            groups: ctx.channels() as i64,
                            device
                        }
                    }
                }
            }
        }

    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn id(&self) -> RunId {
        self.run_config.id.expect("no run id")
    }

    pub  fn step(&self) -> u32{
        self.run_config.step()
    }

    pub fn run_config(&self) -> &FindMotifsRunConfig {
        &self.run_config
    }

    pub fn kernel(&self) -> &Option<SyncWrapper<Box<Tensor>>> {
        &self.kernel
    }

    pub fn groups(&self) -> i64 {
        self.groups
    }
}

impl Applicable<&Box<Tensor>, Box<Tensor>> for FindMotifsRunConfigApplicable {
    fn apply(&self, input: &Box<Tensor>) -> Result<Box<Tensor>, ThoosaError> {

        let t = match self.kernel.as_ref() {
            None => {
                input.to_device(self.device).shallow_clone()
            }
            Some(kernel) => {
                input.to_device(self.device).conv1d(&kernel, None::<Tensor>, 1, 0, 1, self.groups)
            }
        };

        let input_length = t.size()[1];

        if input_length >= self.run_config.window as i64 {

            let unfolded = t.unfold(1, self.run_config.window as i64, self.run_config.step as i64);

            debug!("Applicable for FindMotifsRunConfigApplicable::apply filter applied.size={:?} unfolded.size={:?}", t.size(), unfolded.size());

            Ok(Box::new(unfolded))
        } else {
            Err(ThoosaError::new(format!("can't unfold tensor - length ({}) is less than window ({})", input_length, self.run_config.window)))
        }
    }
}

impl Applicable<&[SyncWrapper<Box<Tensor>>], (Tensor, Vec<i64>)> for FindMotifsRunConfigApplicable {

    //TODO first cat then apply?
    fn apply(&self, tensors_chunk: &[SyncWrapper<Box<Tensor>>]) -> Result<(Tensor, Vec<i64>), ThoosaError> {
        let mut slices = Vec::with_capacity(tensors_chunk.len());
        let mut slice_sizes = Vec::with_capacity(tensors_chunk.len());
        if log_enabled!(DebugLevel) {
            debug!("Applicable for FindMotifsRunConfigApplicable::apply tensors_chunk.sizes={:?}", tensors_chunk.iter().map(|t| t.size()).collect::<Vec<_>>());
        }

        for tensor_wrapped in tensors_chunk {
            match self.apply(tensor_wrapped.deref()){
                Ok(t) => {
                    slice_sizes.push(t.size()[1]);
                    slices.push(t);

                }
                Err(e) => {
                    warn!("Applicable for FindMotifsRunConfigApplicable::apply: skipping tensor, e={e:?}");
                }
            };
        }

        let tensor= if slices.is_empty() {
            //channels
            Tensor::zeros(&[0, tensors_chunk[0].size()[0], self.run_config.window as i64], (Float, Cpu))
        } else {
            //channels x slices x window
            Tensor::cat(&slices, 1)//cat along slices
        };

        Ok(
            (tensor, slice_sizes) //num slices in each tensor,
        )
    }
}

impl FindMotifsRunConfig {
    pub fn new(id: Option<u32>,
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
            slices_filter: SlicesFilter::DistanceRunningMeanK(DistanceRunningMeanK{ mean:  mean_k.unwrap_or(0.5) })
        }
    }

    pub fn window(&self) -> i64 {
        self.window as i64
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
        Self { kernel, stride, padding }
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
    pub distance:f32
}

#[cfg(test)]
mod tests {
    use log::debug;
    use crate::find_motifs_run_config::{DistanceRunningMeanK, DistanceRunningMeanKRangeOrValue, FindMotifsRunConfigRangeOrValue, InputFilterRangeOrValue, SlicesFilterRangeOrValue};
    use crate::transforms::filters::BandPassFilterParamsRangeOrValue;

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_find_motifs_run_config(){
        let rc = FindMotifsRunConfigRangeOrValue {
            id: None,
            step: 100.into(),
            window: 200.into(),
            min_tracker_step: None,
            filter: Some(InputFilterRangeOrValue::BandPassFilter(BandPassFilterParamsRangeOrValue{
                low_hz: (10. ..100. ).into(),
                band_hz: (300. ..400. ).into(),
                kernel_size: (12000i64).into(),
                stride: (1i64).into(),
                padding: (1i64).into(),
            })),
            slices_filter: SlicesFilterRangeOrValue::DistanceRunningMeanK(DistanceRunningMeanKRangeOrValue { mean: 0.5.into()}),
        };
        debug!("{:#}", serde_json5::to_string(&rc).unwrap());
    }

}