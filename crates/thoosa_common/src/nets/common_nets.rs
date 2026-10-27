use thoosa_common::tch_error::tch_result;
use crate::dims::Dims;
use crate::nets::batches::BatchLengths;
use crate::nets::nodes::{BinaryHeadPool, ThoModule};
use crate::nets::tho_varstore::clone_var_store_to;
use crate::nets::state_transfer::{StateTensors, VarStoreState};
use thoosa_common::runner::sync_stage::StageStateInfo;
use crate::parzen::optimizer::{AsyncTPEOptimizer, CompletedTrial, ConfigurationSpace, StudyDirection};
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::utils::{ListenerResponse, TrainListener, mask_from_lengths, plot_series, tensor_mem_size};
use async_trait::async_trait;
use clap::Args;
use log::{debug, error, trace, warn};
use parking_lot::RwLock;
use plotters::prelude::BLACK;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::fmt::{Debug, Formatter};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::Arc;
use tch::Device::Cpu;
use tch::Kind::Float;
use tch::nn::{Optimizer, VarStore};
use tch::{Device, Kind, Tensor, nn};
use thoosa_properties_map::ToHashmap;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::collect_futures_eta::{CollectEtaInvocationGuard, collect_eta_record_completion};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Data, DebugId, WithId};
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_shared::{CopyToDevice, ThoError, panic_to_err_async};
use thoosa_shims::inference_mode::InferenceModeGuard;

pub trait WithCommonTrainingRunConfig {
    fn config(&self)->&CommonTrainingRunConfig;
}

type ModuleFactory<DI, DO> = dyn Fn(&nn::Path) -> anyhow::Result<Box<dyn ThoModule<DI, DO>>> + Send + Sync;

pub struct ThoModuleWithStore<DI: Dims + Send + 'static + ?Sized, DO: Dims + Send> {
    id:SynKey,
    module: parking_lot::Mutex<Option<Box<dyn ThoModule<DI, DO>>>>,
    var_store: VarStore,
    offloaded: Option<(VarStoreState, StateTensors)>,
    quantized: bool,
    module_factory: Option<Arc<ModuleFactory<DI, DO>>>,
}

impl <DI, DO> WithId for ThoModuleWithStore<DI, DO>
where
    DI: Dims + Send + 'static + ?Sized,
    DO: Dims + Send, {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl <DI, DO> Output for ThoModuleWithStore<DI, DO>
where
    DI: Dims + Send + 'static + ?Sized,
    DO: Dims + Send + 'static, {

}

impl<DI, DO> Debug for ThoModuleWithStore<DI, DO>
where
    DI: Dims + Send + 'static + ?Sized,
    DO: Dims + Send,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThoModuleWithStore")
            .field("module", &self.module)
            .field("var_store", &self.var_store)
            .field("has_factory", &self.module_factory.is_some())
            .finish()
    }
}

// Auto traits follow the factory bounds and the mutex protecting the non-Sync module.

impl<DI: Dims + Send + 'static, DO: Dims + Send + 'static> ThoModule<DI, DO> for ThoModuleWithStore<DI, DO>
{
    fn tho_forward(&self, t: &ThoTensor<DI, DebugId>, lengths: Option<&mut BatchLengths>, train: bool) -> anyhow::Result<ThoTensor<DO, DebugId>> {
        let _guard = InferenceModeGuard::new(!train);

        //TODO move to ThoVarStore?
        let store_float_precision = var_store_float_precision(&self.var_store);
        debug!("ThoModuleWithStore::tho_forward: t.id={:?} store_float_precision={:?} t.as_ref().kind={:?}", t.id(), store_float_precision, t.as_ref().kind());

        let float_precision = match store_float_precision {
            VarStoreFloatPrecision::NoFloatVars => {
                warn!("ThoModuleWithStore::tho_forward: store_float_precision={store_float_precision:?}"); //dummy nets may have no tensors
                Float
            }
            VarStoreFloatPrecision::Uniform(kind) => {
                kind
            }
            VarStoreFloatPrecision::Mixed(_) => {
                return Err(anyhow::anyhow!("ThoModule {:?} contains tensors of unexpected precision: store_float_precision={store_float_precision:?}", self.id))
            }
        };

        let t = if !float_precision.eq(&t.as_ref().kind()) {
            debug!("ThoModuleWithStore::tho_forward: t.id={:?}, converting to {:?}", t.id(), float_precision);

            &t.to_kind(float_precision)?
        } else {
            t
        };
        debug!("ThoModuleWithStore::tho_forward: t.id={:?}, t.as_ref().kind={:?}", t.id(), t.as_ref().kind());

        self.ensure_ready()?;
        self.module().tho_forward(t, lengths, train)
    }
}

impl<DI: Dims + Send + 'static, DO: Dims + Send + 'static> ThoModuleWithStore<DI, DO> {
    // #[cfg(test)]
    pub fn new(id: SynKey, module: Box<dyn ThoModule<DI, DO>>) -> Self {
        Self::with_factory(id, module, VarStore::new(Cpu), None)
    }

    /// The factory is shared when models are copied or rebuilt on another device.
    /// ```compile_fail
    /// use std::{cell::Cell, rc::Rc, sync::Arc};
    /// use tch::{nn::VarStore, Device};
    /// use thoosa_common::{dims::AnyDims, nets::{common_nets::ThoModuleWithStore, nodes::{ThoFuncModule, ThoModule}}};
    /// use thoosa_shared::comp_id::SynKey;
    /// let state = Rc::new(Cell::new(0));
    /// let retained = state.clone();
    /// let module: Box<dyn ThoModule<AnyDims, AnyDims>> = Box::new(ThoFuncModule::new(|t| Ok(t.f_alias()?)));
    /// let factory = Arc::new(move |_: &tch::nn::Path<'_>| -> anyhow::Result<Box<dyn ThoModule<AnyDims, AnyDims>>> {
    ///     state.set(state.get() + 1);
    ///     Ok(Box::new(ThoFuncModule::new(|t| Ok(t.f_alias()?))))
    /// });
    /// let model = ThoModuleWithStore::with_factory(SynKey::new([]), module, VarStore::new(Device::Cpu), Some(factory));
    /// retained.set(7);
    /// ```
    pub fn with_factory<I:Into<SynKey>>(
        id: I,
        module: Box<dyn ThoModule<DI, DO>>,
        var_store: VarStore,
        module_factory: Option<Arc<ModuleFactory<DI, DO>>>,
    ) -> Self {
        Self {
            id: id.into(),
            module: parking_lot::Mutex::new(Some(module)),
            var_store,
            module_factory,
            offloaded: None,
            quantized: false,
        }
    }

    pub fn load<F>(id: SynKey, builder: F, model: PathBuf, device: Device, quantize: bool) -> anyhow::Result<Self>
    where
        F: Fn(&nn::Path) -> anyhow::Result<Box<dyn ThoModule<DI, DO>>> + Send + Sync + 'static,
    {
        let _guard = InferenceModeGuard::enabled();

        let builder: Arc<ModuleFactory<DI, DO>> = Arc::new(builder);
        let mut vs = VarStore::new(device);



        let mut module = builder(&vs.root())?;
        vs.load(&model)?;
        vs.freeze();

        {
            let variables = vs.variables();
            debug!("ThoModuleWithStore::load: id={:?}, variables={:?}", id, variables);

            for tensor in variables.values() {
                debug!("ThoModule::load: model={model:?} tensor={:?}", tensor);
            }
        }

        //TODO config/arg to enable
        if quantize {
            module.quantize();
        }

        let mut result = Self::with_factory(id, module, vs, Some(builder));
        result.quantized = quantize;
        Ok(result)
    }

    #[allow(dead_code)]
    fn mem_total(&self) -> usize {
        self.var_store
            .variables()
            .iter()
            .map(|t| tensor_mem_size(t.1))
            .sum()
    }

    #[allow(dead_code)]
    fn debug(&self) {
        self.module().debug();
    }

    pub fn infer(&self, t: &ThoTensor<DI, DebugId>) -> ThoTensor<DO, DebugId> {
        self.module().tho_forward(t, None, false).unwrap()
    }

    pub fn var_store(&self) -> &VarStore {
        &self.var_store
    }

    pub fn module(&self) -> parking_lot::MappedMutexGuard<'_, Box<dyn ThoModule<DI, DO>>> {
        // Shared cached models must retain the lock for every access to the non-Sync module.
        parking_lot::MutexGuard::map(self.module.lock(), |module| module.as_mut().expect("model state must be restored before accessing its module"))
    }

    pub fn ensure_ready(&self) -> Result<(), ThoError> {
        if self.module.lock().is_none() {
            return Err(ThoError::new("model state is offloaded or incompletely restored"));
        }
        Ok(())
    }

    pub fn has_gpu_data(&self, device: usize) -> bool {
        device != 0 && (self.module.lock().is_some() && self.tho_device_idx() == device
            || self.offloaded.as_ref().is_some_and(|(_, tensors)| tensors.has_device(device)))
    }

    pub fn state_info(&self) -> Result<StageStateInfo, ThoError> {
        self.ensure_ready()?;
        let mut tensors = StateTensors::default();
        VarStoreState::capture(&self.var_store, &mut tensors)?;
        Ok(tensors.info())
    }

    pub fn state_tensor(&self, key: &str) -> Result<Tensor, ThoError> {
        self.ensure_ready()?;
        let variables = self.var_store.variables();
        let tensor = key.strip_prefix("var/").and_then(|name| variables.get(name)).ok_or_else(|| ThoError::NotFound(Some(key.to_owned())))?;
        Ok(tch_result(tensor.f_detach())?)
    }

    pub fn drop_gpu(&mut self, device: usize) -> Result<(), ThoError> {
        if !self.has_gpu_data(device) { return Ok(()); }
        if let Some((_, tensors)) = &mut self.offloaded {
            tensors.clear();
            return Ok(());
        }
        if self.module_factory.is_none() {
            return Err(ThoError::new("offloading a model requires its module factory"));
        }
        let mut tensors = StateTensors::default();
        let variables = VarStoreState::capture(&self.var_store, &mut tensors)?;
        if tensors.info().tensors.is_empty() { return Ok(()); }
        tensors.clear();
        // Both the module (including derived/quantized caches) and VarStore own
        // GPU tensors. Drop both after the parent's D2H copies have completed.
        *self.module.get_mut() = None;
        self.var_store = VarStore::new(Cpu);
        self.offloaded = Some((variables, tensors));
        Ok(())
    }

    pub fn set_state_tensor(&mut self, key: &str, tensor: Tensor, device: usize) -> Result<(), ThoError> {
        let (variables, tensors) = self.offloaded.as_mut().ok_or_else(|| ThoError::new("model state is already loaded"))?;
        tensors.set(key, tensor, device)?;
        if tensors.complete() {
            // Build from the restored store, so module and store share exactly
            // the same parameters; never reload the original checkpoint.
            let vs = variables.restore(tensors, device)?;
            let builder = self.module_factory.as_ref().ok_or_else(|| ThoError::new("model has no module factory"))?;
            let mut module = builder(&vs.root())?;
            if self.quantized { module.quantize(); }
            self.var_store = vs;
            *self.module.get_mut() = Some(module);
            self.offloaded = None;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarStoreFloatPrecision {
    NoFloatVars,
    Uniform(Kind),
    Mixed(Vec<(Kind, usize)>),
}

fn is_float_kind(kind: Kind) -> bool {
    matches!(
            kind,
            Kind::Half | Kind::BFloat16 | Kind::Float | Kind::Double
        )
}

pub fn var_store_float_precision(vs: &nn::VarStore) -> VarStoreFloatPrecision {
    let mut counts: Vec<(Kind, usize)> = Vec::new();

    for (_name, tensor) in vs.variables() {
        let kind = tensor.kind();

        if !is_float_kind(kind) {
            continue;
        }

        if let Some((_k, count)) = counts.iter_mut().find(|(k, _)| *k == kind) {
            *count += 1;
        } else {
            counts.push((kind, 1));
        }
    }

    debug!("var_store_float_precision: counts={counts:?}");

    match counts.len() {
        0 => VarStoreFloatPrecision::NoFloatVars,
        1 => VarStoreFloatPrecision::Uniform(counts[0].0),
        _ => VarStoreFloatPrecision::Mixed(counts),
    }
}

#[async_trait]
impl <DI, DO> CopyToDevice for ThoModuleWithStore<DI, DO>
where
DI: Dims + Send + 'static,
DO: Dims + Send + 'static {

    async fn copy_to_device(&self, tho_dst_device: usize) -> Result<Self, ThoError> {

        debug!("ThoModuleWithStore::copy_to_device: has module_factory={}", self.module_factory.is_some());

        self.ensure_ready()?;
        let var_store = clone_var_store_to(&self.var_store, tho_dst_device.into()).await?;
        // let mut var_store = VarStore::new(ThoDevice::from(tho_dst_device).into());
        let builder = self
            .module_factory
            .as_ref()
            .unwrap_or_else(|| panic!("copy_to_device requires module_factory; construct modules via ThoModuleWithStore::load to enable copying"));

        let mut module = builder(&var_store.root())?;
        if self.quantized { module.quantize(); }
        let mut result = Self::with_factory(self.id.clone(), module, var_store, Some(Arc::clone(builder)));
        result.quantized = self.quantized;
        Ok(result)
    }
}

impl <DI, DO> Data for ThoModuleWithStore<DI, DO>
where
    DI: Dims + Send + 'static,
    DO: Dims + Send + 'static{
    type Metrics = ();

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn tho_device_idx(&self) -> usize {
        ThoDevice::from(self.var_store.device()).tho_device_idx()
    }

    fn mem_size(&self) -> u64 {
        if let Ok( guard) = self.var_store.variables_.lock() {
            guard.trainable_variables.iter().map(|v|tensor_mem_size(&v.tensor) as u64).sum::<u64>()
          + guard.named_variables.iter().map(|v|tensor_mem_size(v.1) as u64).sum::<u64>()

        } else {
            error!("ThoModuleWithStore::mem_size: can't get variables");
            0
        }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BinaryHeadMax{}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BinaryHeadMean{}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct BinaryHeadLse{
    #[in_map]
    #[run_config]
    pub tau: f64
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum BinaryHeadPoolArgs {
    Max(BinaryHeadMax),
    Mean(BinaryHeadMean),
    Lse(BinaryHeadLse),
}

impl From<&BinaryHeadPoolArgs> for BinaryHeadPool {
    fn from(value: &BinaryHeadPoolArgs) -> Self {
        match value {
            BinaryHeadPoolArgs::Max(_) => {
                BinaryHeadPool::Max
            }
            BinaryHeadPoolArgs::Mean(_) => {
                BinaryHeadPool::Mean
            }
            BinaryHeadPoolArgs::Lse(BinaryHeadLse{tau}) => {
                BinaryHeadPool::Lse {tau: *tau}
            }
        }
    }
}

impl Default for BinaryHeadPoolArgs {
    fn default() -> Self {
        BinaryHeadPoolArgs::Max(BinaryHeadMax{})
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
pub struct AdamW {
    #[in_map]
    #[run_config]
    beta1:f32,

    #[in_map]
    #[run_config]
    beta2:f32,

    #[in_map]
    #[run_config]
    pub wd: f64,

    #[in_map]
    #[run_config]
    amsgrad:bool
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum Optimizers {
    AdamW(AdamW)
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct CommonTrainingRunConfig {
    #[in_map]
    #[run_config]
    pub epochs: u32,

    #[in_map]
    #[run_config]
    pub batch_size: u32, //number of the slices in a batch

    #[in_map]
    #[run_config]
    pub lr: f64,

    #[in_map]
    #[run_config]
    pub patience:u32,

    #[in_map]
    #[run_config]
    pub min_delta:f32,

    #[in_map]
    #[run_config]
    optimizer: Option<Optimizers>
}

impl CommonTrainingRunConfig {
    fn default_optimizer() -> Optimizers {
        Optimizers::AdamW(AdamW {
            beta1: 0.9, beta2: 0.999, wd: 0.01, amsgrad: false
        })
    }

    pub fn new(
        cycles: u32,
        batch_size: u32,
        lr: f64,
    ) -> Self {
        Self {
            epochs: cycles,
            batch_size,
            lr,
            patience: 0,
            min_delta: 0.0,
            optimizer: Some(Self::default_optimizer()),
        }
    }

    pub fn tho_optimizer(&self) -> crate::nets::tho_varstore::ThoOptimizers {
        let optimizer = self.optimizer.clone().unwrap_or_else(Self::default_optimizer);

        debug!("CommonTrainingRunConfig::tho_optimizer: optimizer={optimizer:?}");

        match optimizer {
            Optimizers::AdamW(AdamW{ beta1, beta2, wd, amsgrad }) => {
                tch::nn::AdamW{
                    beta1: beta1 as f64,
                    beta2: beta2 as f64,
                    wd,
                    eps: 1e-8,
                    amsgrad,
                }.into()
            }
        }
    }
}

//for tests
impl Default for CommonTrainingRunConfig {
    fn default() -> Self {
        CommonTrainingRunConfig {
            epochs: 10,
            batch_size: 10,
            lr: 0.0001,
            patience: 5,
            min_delta: 1e-4_f32,
            optimizer: Some(CommonTrainingRunConfig::default_optimizer()),
        }
    }
}

#[derive(Args, Debug, Clone)]
pub struct OptConfig {
    #[arg(long, default_value_t = 100)]
    pub opt_max_evals: u32,

    #[arg(long, default_value_t = 10)]
    pub opt_num_init: u32,

    #[arg(long, default_value_t = 20)]
    pub patience: u32,

    #[arg(long, default_value_t = 1e-5_f32)]
    pub min_delta: f32,
}

impl OptConfig {
    pub fn new(max_evals: u32, n_init: u32) -> Self {
        Self {
            opt_max_evals: max_evals,
            opt_num_init: n_init,
            patience: 0,
            min_delta: 0.0,
        }
    }

    pub fn new_with_early_stopping(max_evals: u32, n_init: u32, patience: u32, min_delta: f32) -> Self {
        Self {
            opt_max_evals: max_evals,
            opt_num_init: n_init,
            patience,
            min_delta,
        }
    }
}

#[derive(Debug, Copy, Clone)]
pub enum NetsPadding {
    Max,
    Fixed(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ConvNetJoint {
    AttnPool1d,
    MinMax
}

impl From<&ConvNetJointArgs> for ConvNetJoint {
    fn from(value: &ConvNetJointArgs) -> Self {
        match value {
            ConvNetJointArgs::AttnPool1d(_) => {
                ConvNetJoint::AttnPool1d
            }
            ConvNetJointArgs::MinMax(_) => {
                ConvNetJoint::MinMax
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvNetParams {
    pub num_conv_layers:u32,
    pub kernel:u32,
    pub hidden_conv_channels:u32,

    pub num_head_layers:u32,
    pub num_head_features:u32,

    pub joint: ConvNetJoint
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionNetParams {
    pub head_size: u32, //d_model- times n_heads
    pub n_heads: u32,
    pub ff_mult: u32,
    pub dropout: f64,
}

impl SelfAttentionNetParams {
    pub fn new(head_size: u32, n_heads: u32, ff_mult: u32, dropout: f64) -> Self {
        Self { head_size, n_heads, ff_mult, dropout }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinLenNetParams {
    pub layers: u32,
    pub features: u32,
}

impl LinLenNetParams {
    pub fn new(features: u32, layers: u32) -> Self {
        Self { layers, features }
    }
}

impl ConvNetParams {
    pub fn new(num_conv_layers:u32, hidden_conv_channels:u32, kernel: u32, num_head_layers: u32, num_head_features: u32, joint: ConvNetJoint) -> Self {
        Self {
            num_conv_layers,
            num_head_layers,
            kernel,
            hidden_conv_channels,
            num_head_features,
            joint
        }
    }
}

pub fn self_attention_masked_mean(h: &Tensor, lengths: Option<&BatchLengths>) -> anyhow::Result<Tensor> {
    let h_size = h.size();
    if h_size.len() != 3 {
        return Err(anyhow::anyhow!("self_attention_masked_mean expected [B,L,D], got {:?}", h_size));
    }
    if let Some(lengths) = lengths {
        let kind = h.kind();
        let mask = mask_from_lengths(lengths.as_slice(), h_size[1], h.device())?.f_squeeze_dim(1)?;
        let mask_f = mask.f_to_kind(kind)?.f_unsqueeze(-1)?;
        let sum = h.f_mul(&mask_f)?.f_sum_dim_intlist([1].as_slice(), false, kind)?;
        let denom = mask_f.f_sum_dim_intlist([1].as_slice(), false, kind)?.f_clamp_min(1.0)?;
        Ok(sum.f_div(&denom)?)
    } else {
        Ok(h.f_mean_dim(1, false, None)?)
    }
}

pub struct PlotVecf32 {
    #[allow(dead_code)]
    id: String,
    data: Vec<f32>,
}

impl PlotVecf32 {
    pub fn new(id: &str) -> Self {
        Self {
            id: id.into(),
            data: vec![],
        }
    }

}

impl TrainListener for PlotVecf32 {
    fn on_valid_cycle(&mut self, _cycle: u32, loss: f32) -> ListenerResponse {
        self.data.push(loss);

        //TODO move into another listener
        if self.data.len() > 10 {
            let len_half = self.data.len() / 2;
            if self.data[0..len_half].iter().sum::<f32>()
                < self.data[len_half..].iter().sum::<f32>()
            {
                debug!(
                    "TrainListener for PlotVecf32::on_valid_cycle stopping self.data={:?}",
                    self.data
                );
                return ListenerResponse::Stop;
            }
        }

        ListenerResponse::Nope
    }
}

fn opt_param_property_keys<P, R>(const_vals: &[NamedVal<P>], grid_vals: &[NamedVal<P>], cs: &ConfigurationSpace<P>) -> Vec<String>
where
    P: FieldName + 'static,
    R: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + ToHashmap,
{
    let base_opt_vals = cs
        .into_iter()
        .map(|hp| NamedVal::new(hp.name().clone(), opt_param_base_value(hp)))
        .collect::<Vec<_>>();
    let base_map = run_config_property_map::<P, R>(const_vals, grid_vals, &base_opt_vals);

    cs.into_iter()
        .enumerate()
        .map(|(idx, hp)| {
            let fallback = hp.name().to_string();
            let Some(alt_value) = opt_param_alt_value(hp) else {
                return fallback;
            };

            let mut alt_opt_vals = base_opt_vals.clone();
            alt_opt_vals[idx] = NamedVal::new(hp.name().clone(), alt_value);
            let alt_map = run_config_property_map::<P, R>(const_vals, grid_vals, &alt_opt_vals);
            changed_property_key(&base_map, &alt_map).unwrap_or_else(|| {
                debug!("opt_param_property_keys: could not resolve property key for hp={hp:?}; falling back to display key {fallback}");
                fallback
            })
        })
        .collect()
}

fn run_config_property_map<P, R>(const_vals: &[NamedVal<P>], grid_vals: &[NamedVal<P>], opt_vals: &[NamedVal<P>]) -> HashMap<String, Val>
where
    P: FieldName + 'static,
    R: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + ToHashmap,
{
    let run_config: R = [const_vals, grid_vals, opt_vals].concat().into_iter().into();
    run_config.to_properties_iter(None).collect()
}

fn changed_property_key(base_map: &HashMap<String, Val>, alt_map: &HashMap<String, Val>) -> Option<String> {
    let mut changed = alt_map
        .iter()
        .filter(|(key, val)| base_map.get(*key) != Some(*val))
        .map(|(key, _)| key.clone())
        .collect::<Vec<_>>();
    changed.extend(base_map.keys().filter(|key| !alt_map.contains_key(*key)).cloned());
    changed.sort();
    changed.dedup();
    (changed.len() == 1).then(|| changed.remove(0))
}

fn opt_param_base_value<P: FieldName>(hp: &Hyperparameter<P>) -> Val {
    match hp {
        Hyperparameter::UniformFloat { lower, .. } => Val::Real(*lower),
        Hyperparameter::UniformInt { lower, .. } => Val::Int(*lower),
        Hyperparameter::Categorical { .. } => Val::Int(0),
    }
}

fn opt_param_alt_value<P: FieldName>(hp: &Hyperparameter<P>) -> Option<Val> {
    match hp {
        Hyperparameter::UniformFloat { lower, upper, .. } if upper > lower => Some(Val::Real(lower + ((upper - lower) / 2.0))),
        Hyperparameter::UniformInt { lower, upper, .. } if upper > lower => Some(Val::Int(lower + 1)),
        Hyperparameter::Categorical { values, .. } if values.len() > 1 => Some(Val::Int(1)),
        _ => None,
    }
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="opt_train"))]
pub async fn opt_train<P, R, F, N: Debug + Default, A: Debug + Send + Sync + 'static>(
    opt_config: Arc<OptConfig>, //N node parameters
    mut train: F,
    run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >,
    descr: &str,
    reusable_completed_trials: Vec<CompletedTrial>,
) -> anyhow::Result<(A, N, R, f32)>
where
    P: FieldName + 'static,
    R: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + ToHashmap,
    F: for<'a, 'b> AsyncFnMut(
            R,
            &'a mut (dyn TrainListener + 'b),
        ) -> Result<(A, f32, N), ThoError>,
{
    let const_vals = run_config_source
        .grid_run_config_source()
        .const_params()
        .clone();
    const MAX_LOSS:f32 = f32::MAX;
    let min_loss = Arc::new(RwLock::new(MAX_LOSS));

    let min_var_store: Arc<RwLock<Option<(A, N, R, f32)>>> = Arc::new(RwLock::new(None));
    // Preserve the real failure when every trial fails, and never turn a
    // device recovery failure into a valid optimization penalty/result.
    let last_error = Arc::new(RwLock::new(None::<ThoError>));
    let recovery_error = Arc::new(RwLock::new(None::<ThoError>));

    let grid_runs = run_config_source.grid_to_vals_and_opt_params()?;

    debug!("opt_train: running opt train {descr} grid_runs={grid_runs:?}");

    let grids_eta_guard = CollectEtaInvocationGuard::new(&format!("grid_sets_{descr}"), grid_runs.len(), 1);
    for (grid_vals, opt_params) in grid_runs.into_iter() {
        let const_vals = const_vals.clone();
        let train_clone = &mut train;

        let min_var_store_clone = Arc::clone(&min_var_store);
        let min_loss_moved = Arc::clone(&min_loss);
        let last_error_moved = Arc::clone(&last_error);
        let recovery_error_moved = Arc::clone(&recovery_error);
        let config_space = ConfigurationSpace::new(opt_params);
        let reusable_configs = if reusable_completed_trials.is_empty() {
            vec![]
        } else {
            let reusable_param_keys = opt_param_property_keys::<P, R>(const_vals.as_slice(), grid_vals.as_slice(), &config_space);
            config_space.reusable_configs_from_completed_trials_with_param_keys(
                reusable_completed_trials.iter().cloned(),
                StudyDirection::Minimize,
                opt_config.opt_max_evals as usize,
                &reusable_param_keys,
            )
        };

        let mut opt = AsyncTPEOptimizer::new(
            async move |opt_vals| {
                if recovery_error_moved.read().is_some() { return MAX_LOSS; }
                debug!("opt_train: opt_vals={opt_vals:?}  ");

                let run_config: R = [const_vals.as_slice(), grid_vals.as_slice(), opt_vals]
                    .concat()
                    .into_iter()
                    .into();

                let mut listener = PlotVecf32::new(
                    format!("_params_{:?}", "id" /*TODO run_config.get_id()*/).as_str(),//get_id - it's not set 
                );

                let train_result = panic_to_err_async!(
                    train_clone(
                        run_config.clone(),
                        &mut listener
                    )
                );
                let (var_store, loss, node_params) = match train_result {
                    Ok(result) => result,
                    Err(e) => {
                        warn!("opt_train failed with {e:?} on {run_config:?} ");
                        if e.is_cuda_recovery_failed() || e.needs_cuda_restart() || matches!(e.what(), ThoError::ContextReset) {
                            *recovery_error_moved.write() = Some(e.clone());
                        }
                        *last_error_moved.write() = Some(e);
                        return MAX_LOSS;
                    }
                };
                // listener.plot();

                let min_loss_val = *min_loss_moved.read();
                debug!("opt_train: run_config_values={run_config:?} loss={loss} min_loss_val={min_loss_val:?} "); //descr={descr}

                if loss < min_loss_val {
                    *min_loss_moved.write() = loss;
                    min_var_store_clone
                        .write()
                        .replace((var_store, node_params, run_config, loss));
                }

                loss
            },
            config_space,
            opt_config.as_ref().clone(),
            descr,
        );
        opt.enqueue_configs(reusable_configs);
        debug!("opt_train: wait on optimize ");
        let best_config = track_await!(opt.optimize()).await;
        if let Some(error) = recovery_error.write().take() {
            debug!("opt_train: recovery_error={recovery_error:?} ");

            return Err(error.into());
        }

        debug!("opt_train: best_config={:?}", best_config);

        debug!("opt_train: done opt, descr={descr} best_config={best_config:?} min_loss={}", min_loss.read());
        trace!("opt_train: done opt, descr={descr} min_var_store={min_var_store:?}");//can cause Internal torch error: CUDA error: device-side assert triggered
        collect_eta_record_completion(grids_eta_guard.id());
    }
    Arc::into_inner(min_var_store)
        .expect("min_var_store is used somewhere else")
        .into_inner()
        .ok_or_else(|| last_error.write().take().map(anyhow::Error::from).unwrap_or_else(|| anyhow::anyhow!("no min_var_store")))
}

pub fn is_nonzero(t_1: &Tensor) -> anyhow::Result<bool> {
    Ok(t_1
        .flatten(0, (t_1.size().len() - 1) as i64)
        .iter::<f64>()?
        .find(|v| v > &0.)
        .is_some())
}

#[allow(non_snake_case, unused_assignments, unused_variables)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct ConvNetArgs {
    #[in_map]
    #[run_config]
    pub num_conv_layers: u32,

    #[in_map]
    #[run_config]
    pub hidden_conv_channels: u32,

    #[in_map]
    #[run_config]
    pub num_head_layers: u32,

    #[in_map]
    #[run_config]
    pub num_head_features: u32,

    #[in_map]
    #[run_config]
    pub kernel: u32,

    #[in_map]
    #[run_config]
    pub joint:ConvNetJointArgs
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct AttnPool1dArgs{

}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct MinMaxArgs{
    #[in_map]
    pub _dummy: u32,
}

#[allow(non_camel_case_types)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum ConvNetJointArgs {
    AttnPool1d(AttnPool1dArgs),
    MinMax(MinMaxArgs)
}

impl ConvNetArgs {
    pub fn new(num_conv_layers: u32, hidden_conv_channels: u32, kernel: u32, num_head_layers: u32, num_head_features:u32, joint:ConvNetJointArgs) -> Self {

        Self { num_conv_layers, hidden_conv_channels, num_head_layers, kernel, num_head_features, joint }
    }
}

#[allow(non_snake_case, unused_assignments, unused_variables)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct SelfAttentionNetArgs {
    #[in_map]
    #[run_config]
    pub head_size: u32,
    #[in_map]
    #[run_config]
    pub n_heads: u32,
    #[in_map]
    #[run_config]
    pub ff_mult: u32,
    #[in_map]
    #[run_config]
    pub dropout: f64,
}

impl SelfAttentionNetArgs {
    pub fn new(d_model: u32, n_heads: u32, ff_mult: u32, dropout: f64) -> Self {
        Self { head_size: d_model, n_heads, ff_mult, dropout }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Clone, PartialEq, Debug, Hash)]
pub struct StftRunConfig {
    #[in_map]
    #[run_config]
    pub n_fft: i64,

    #[in_map]
    #[run_config]
    pub hop_length: Option<i64>,

    #[in_map]
    #[run_config]
    pub win_length: Option<i64>,

    #[in_map]
    #[run_config]
    pub normalized: bool,

    #[in_map]
    #[run_config]
    pub onesided: bool,

    #[in_map]
    #[run_config]
    pub align_to_window: bool,
}

impl StftRunConfig {
    pub fn new(n_fft: i64, hop_length: Option<i64>, win_length: Option<i64>, normalized: bool, onesided: bool, align_to_window: bool) -> Self {
        Self { n_fft, hop_length, win_length, normalized, onesided, align_to_window }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if self.n_fft <= 0 {
            return Err(anyhow::anyhow!("StftRunConfig::validate: n_fft must be positive, got {}", self.n_fft));
        }
        if matches!(self.hop_length, Some(v) if v <= 0) {
            return Err(anyhow::anyhow!("StftRunConfig::validate: hop_length must be positive, got {:?}", self.hop_length));
        }
        if matches!(self.win_length, Some(v) if v <= 0) {
            return Err(anyhow::anyhow!("StftRunConfig::validate: win_length must be positive, got {:?}", self.win_length));
        }
        Ok(())
    }

    pub fn hash_value(&self) ->u64{
        let mut hasher = DefaultHasher::new();
        self.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for StftRunConfig {
    fn default() -> Self {
        Self { n_fft: 16000, hop_length: Some(4000), win_length: Some(4000), normalized: false, onesided: true, align_to_window: true }
    }
}
