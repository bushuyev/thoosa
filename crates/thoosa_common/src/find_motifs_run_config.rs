use crate::applicables::{ThoDevice, ThoRunner};
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
use thoosa_shared::applicables_runner::{CachableId, Input, Output, RunCtx, Runner, ToOrdinal};
use thoosa_shared::run_configs::{Cachable, RunConfig, RunId, Runnable};
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

    pub fn to_applicable(&self, ctx: &Box<dyn Workspace>, device: Device) -> FindMotifsApplicable {
        FindMotifsApplicable::new(&self, ctx, device)
    }
}


//TODO rename to apply DSP Filter
#[derive(Debug)]
pub struct FindMotifsApplicable {
    run_config: FindMotifsRunConfig,
    kernel: Option<SyncWrapper<Tensor>>,
    groups: i64,
    device: Device,
}

impl FindMotifsApplicable {
    pub fn new(run_config: &FindMotifsRunConfig, ctx: &Box<dyn Workspace>, device: Device) -> Self {
        match &run_config.filter {
            None => Self {
                run_config: run_config.clone(),
                kernel: None,
                groups: 0,
                device,
            },

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
                            1e-6, //TODO config?
                        )
                        .view((1, 1, -1))
                        .expand([ctx.channels() as i64, 1, -1], true);

                        Self {
                            run_config: run_config.clone(),
                            kernel: Some(kernel.to_device(device).into()),
                            groups: ctx.channels() as i64,
                            device,
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

    pub fn step(&self) -> u32 {
        self.run_config.step()
    }

    pub fn run_config(&self) -> &FindMotifsRunConfig {
        &self.run_config
    }

    pub fn kernel(&self) -> &Option<SyncWrapper<Tensor>> {
        &self.kernel
    }

    pub fn groups(&self) -> i64 {
        self.groups
    }

    pub fn unfold(&self, input: &TCDF) -> TCSD {
        // let input_length = input.size()[1];
        // input
        //     .t
        //     .unfold(
        //         1,
        //         self.run_config.window as i64,
        //         self.run_config.step as i64,
        //     )
        //     .into()
        todo!()
    }
}

pub enum FindMotifsRunConfigOptions {
    Filter,
    FilterUnfold,
}

#[async_trait]
impl Runnable<TCD, TCSD, FindMotifsRunConfigOptions> for FindMotifsApplicable {
    type R = ThoRunner;

    fn run(&self, input: TCD, options: FindMotifsRunConfigOptions) -> Result<TCSD, ThoosaError> {
        // debug!(
        //     "Applicable for FindMotifsRunConfigApplicable::apply self.kernel={:?} tensor.size={:?}",
        //     self.kernel,
        //     input.size()
        // );
        //
        // let t = match self.kernel.as_ref() {
        //     None => input.as_ref().to_device(self.device).shallow_clone(),
        //     Some(kernel) => input.as_ref().to_device(self.device).conv1d(
        //         &kernel,
        //         None::<Tensor>,
        //         1,
        //         0,
        //         1,
        //         self.groups,
        //     ),
        // };
        //
        // match options {
        //     FindMotifsRunConfigOptions::Filter => Ok(t.into()),
        //     FindMotifsRunConfigOptions::FilterUnfold => {
        //         let input_length = t.size()[1];
        //
        //         if input_length >= self.run_config.window as i64 {
        //             let unfolded = t.unfold(
        //                 1,
        //                 self.run_config.window as i64,
        //                 self.run_config.step as i64,
        //             );
        //
        //             debug!(
        //                 "Applicable for FindMotifsRunConfigApplicable::apply filter applied.size={:?} unfolded.size={:?}",
        //                 t.size(),
        //                 unfolded.size()
        //             );
        //
        //             Ok(unfolded.into())
        //         } else {
        //             Err(ThoosaError::new(format!(
        //                 "can't unfold tensor - length ({}) is less than window ({})",
        //                 input_length, self.run_config.window
        //             )))
        //         }
        //     }
        // }
        todo!()
    }

    // async fn async_run(
    //     self: Arc<Self>,
    //     input: Arc<TCD>,
    //     runner: Arc<Self::R>,
    //     ctx: Option<
    //         Arc<
    //             RwLock<
    //                 <<Self as Runnable<TCD, TCSD, FindMotifsRunConfigOptions>>::R as Runner>::Ctx,
    //             >,
    //         >,
    //     >,
    // ) -> Arc<TCSD> {
    //     todo!()
    // }
    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(self.device)
    }
}

#[async_trait]
impl Cachable<TCD, TCSD, FindMotifsRunConfigOptions> for FindMotifsApplicable {
    fn applied_id(&self, input_id: &CachableId) -> CachableId {
        todo!()
    }

    fn name(&self) -> String {
        todo!()
    }

    fn run_id(&self) -> String {
        todo!()
    }
}

#[async_trait]
impl Runnable<TCD, TCDF, ()> for FindMotifsApplicable {
    type R = ThoRunner;

    fn run(&self, input: TCD, _: ()) -> Result<TCDF, ThoosaError> {
        todo!()
    }

    // async fn async_run(
    //     self: Arc<Self>,
    //     input: Arc<TCD>,
    //     runner: Arc<Self::R>,
    //     ctx: Option<Arc<RwLock<<<Self as Runnable<TCD, TCDF, ()>>::R as Runner>::Ctx>>>,
    // ) -> Arc<TCDF> {
    //     debug!(
    //         "Applicable<TCD, TCSD, ()> for FindMotifsRunConfigApplicable::apply self.kernel={:?} tensor.size={:?}",
    //         self.kernel,
    //         input.size()
    //     );
    //     
    //     let t = match self.kernel.as_ref() {
    //         None => input.t.to_device(self.device).shallow_clone(),
    //         Some(kernel) => {
    //             input
    //                 .t
    //                 .to_device(self.device)
    //                 .conv1d(&kernel, None::<Tensor>, 1, 0, 1, self.groups)
    //         }
    //     };
    //     
    //     // match options {
    //     //     FindMotifsRunConfigOptions::Filter => {
    //     //         Ok(t.into())
    //     //     }
    //     //     FindMotifsRunConfigOptions::FilterUnfold => {
    //     let input_length = t.size()[1];
    //     
    //     if input_length >= self.run_config.window as i64 {
    //         // let unfolded = t.unfold(1, self.run_config.window as i64, self.run_config.step as i64);
    //     
    //         // debug!("Applicable<TCD, TCD, ()> for FindMotifsRunConfigApplicable::apply filter applied.size={:?} unfolded.size={:?}", t.size(), unfolded.size());
    //     
    //         Arc::new(ThoTensor::<ChannelsDataFiltered, CachableId>::new(
    //             <FindMotifsApplicable as Cachable<TCD, TCDF, ()>>::applied_id(
    //                 &self,
    //                 Input::id(input.as_ref()),
    //             ),
    //             t,
    //         ))
    //     } else {
    //         // Err(ThoosaError::new(format!("can't unfold tensor - length ({}) is less than window ({})", input_length, self.run_config.window)))
    //         panic!("todo")
    //     }
    //     // }
    //     // }
    //     todo!()
    // }

    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(self.device)
    }
}

#[async_trait]
impl Cachable<TCD, TCDF, ()> for FindMotifsApplicable {
    fn applied_id(&self, input_id: &CachableId) -> CachableId {
        Cachable::<TCD, TCDF, ()>::format_id(self, input_id)
    }

    fn name(&self) -> String {
        "FindMotifsApplicable".to_string()
    }

    fn run_id(&self) -> String {
        self.run_config.id.as_ref().expect("run_id").to_string()
    }
}

#[derive(Debug)]
pub struct FindMotifsApplicableInput(Vec<SyncWrapper<Tensor>>);

impl From<Vec<SyncWrapper<Tensor>>> for FindMotifsApplicableInput {
    fn from(value: Vec<SyncWrapper<Tensor>>) -> Self {
        FindMotifsApplicableInput(value)
    }
}

impl From<&[SyncWrapper<Tensor>]> for FindMotifsApplicableInput {
    fn from(value: &[SyncWrapper<Tensor>]) -> Self {
        FindMotifsApplicableInput(
            value
                .iter()
                .map(|t| SyncWrapper::wrap_tensor(t.shallow_clone()))
                .collect::<Vec<_>>(),
        )
    }
}

impl WithId for FindMotifsApplicableInput {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl Input for FindMotifsApplicableInput {
    fn id(&self) -> &CachableId {
        todo!()
    }
}

#[derive(Debug)]
pub struct FindMotifsApplicableOutput {
    pub t: Tensor,
    pub sizes: Vec<i64>,
}
unsafe impl Send for FindMotifsApplicableOutput {}
unsafe impl Sync for FindMotifsApplicableOutput {}

impl WithId for FindMotifsApplicableOutput {
    type ID = CachableId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

#[async_trait]
impl Output for FindMotifsApplicableOutput {
    async fn copy_to_device(self: Arc<Self>, device_ordinal: usize) -> Arc<Self> {
        todo!()
    }
}

pub struct DummyRunner;
#[derive(Debug)]
pub struct DummyRunnerCtx;
impl RunCtx for DummyRunnerCtx {
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

#[async_trait]
impl Runnable<FindMotifsApplicableInput, FindMotifsApplicableOutput, ()> for FindMotifsApplicable {
    type R = DummyRunner;

    /// applying transform (Filter) and unfolding tensors one by one since they have varying sizes - can't concat into batch to avoid slices spanning different tensors
    fn run(
        &self,
        tensors_chunk: FindMotifsApplicableInput,
        options: (),
    ) -> Result<FindMotifsApplicableOutput, ThoosaError> {
        // let mut slices = Vec::with_capacity(tensors_chunk.0.len());
        // let mut slice_sizes = Vec::with_capacity(tensors_chunk.0.len());
        // if log_enabled!(DebugLevel) {
        //     debug!(
        //         "Applicable for FindMotifsRunConfigApplicable::apply tensors_chunk.sizes={:?}",
        //         tensors_chunk.0.iter().map(|t| t.size()).collect::<Vec<_>>()
        //     );
        // }
        //
        // for tensor_wrapped in tensors_chunk.0.iter() {
        //     // let tcd:TCD = ThoTensor::<ChannelsData>::from(tensor_wrapped.deref());
        //     match self.run(
        //         tensor_wrapped.deref().into(),
        //         FindMotifsRunConfigOptions::FilterUnfold,
        //     ) {
        //         Ok(t) => {
        //             slice_sizes.push(t.size()[1]);
        //             slices.push(t.as_ref().shallow_clone());
        //         }
        //         Err(e) => {
        //             warn!(
        //                 "Applicable for FindMotifsRunConfigApplicable::apply: skipping tensor, e={e:?}"
        //             );
        //         }
        //     };
        // }
        //
        // let tensor = if slices.is_empty() {
        //     //channels
        //     Tensor::zeros(
        //         &[
        //             0,
        //             tensors_chunk.0[0].size()[0],
        //             self.run_config.window as i64,
        //         ],
        //         (Float, Cpu),
        //     )
        // } else {
        //     //channels x slices x window
        //     Tensor::cat(&slices, 1) //cat along slices
        // };
        //
        // Ok(
        //     FindMotifsApplicableOutput {
        //         t: tensor.into(),
        //         sizes: slice_sizes,
        //     }, //num slices in each tensor,
        // )
        todo!()
    }

    // async fn async_run(
    //     self: Arc<Self>,
    //     input: Arc<FindMotifsApplicableInput>,
    //     runner: Arc<Self::R>,
    //     ctx: Option<Arc<RwLock<<<Self as Runnable<FindMotifsApplicableInput, FindMotifsApplicableOutput, ()>>::R as Runner>::Ctx>>>,
    // ) -> Arc<FindMotifsApplicableOutput> {
    //     todo!()
    // }

    fn device(&self) -> impl ToOrdinal {
        ThoDevice::from(self.device)
    }
}

#[async_trait]
impl Cachable<FindMotifsApplicableInput, FindMotifsApplicableOutput, ()> for FindMotifsApplicable {
    fn applied_id(&self, input_id: &CachableId) -> CachableId {
        todo!()
    }

    fn name(&self) -> String {
        todo!()
    }

    fn run_id(&self) -> String {
        todo!()
    }
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
