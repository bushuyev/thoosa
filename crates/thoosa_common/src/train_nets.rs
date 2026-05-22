#![allow(non_snake_case, unused_assignments, unused_variables, non_camel_case_types)]


use crate::builder::{Node, SeqBuilder};
use crate::batches::BatchLengths;
use crate::my_tch::{BinaryHeadPool, MotifInitMode, ThoModule};
use crate::parzen::optimizer::{AsyncTPEOptimizer, CompletedTrial, ConfigurationSpace, StudyDirection};
use crate::runs::{Empty, Labeled, Logits, NetResources, SavableResult};
use crate::utils::{mask_from_lengths, masked_max, masked_mean, plot_spectrogram, tensor_mem_size};
use crate::utils::{plot_series, ListenerResponse, TrainListener};
use crate::ws::Workspace;
use clap::clap_derive::Args;
use thoosa_shared::{mono_key, panic_to_err_async, CopyToDevice};

use log::{debug, error, info, trace, warn};
use parking_lot::{Mutex, RwLock};
use plotters::prelude::BLACK;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::{fmt, fs};
use std::path::PathBuf;

use tch::nn::{init, ConvConfig, LinearConfig, Optimizer, OptimizerConfig, VarStore};
use tch::Device::Cpu;
use tch::{nn, Device, Kind, TchError, Tensor};
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_shared::ThoError;

use async_trait::async_trait;
use std::sync::Arc;
use tch::Kind::Float;
use torch_sys::at_grad_set_enabled;
use thoosa_dummy_data_derive::DummyData;
use thoosa_store::Store;

use crate::dims::{Batch, Dims, LabelProbability, LabelsProbabilities, MotifProbability, MotifSlices, Slice, Spectorgram};
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use thoosa_properties_map::ToHashmap;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::run_configs::RunConfig;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfigId};
use thoosa_shared::traits::{Data, SimpleId, WithId};
use thoosa_shims::inference_mode::InferenceModeGuard;
use crate::runner::global_ctx::GlobalCtx;
use crate::my_tch::MotifInitModeRangeOrValue;
use crate::my_tch::MotifInitModeFields;
use crate::my_tch::MotifInitModeBuilder;

#[derive(Args, Debug, Clone)]
pub struct TrainLabelNetsArgs {
    #[arg(long, value_delimiter = ',', value_parser = crate::utils::parse_run_config_ids)]
    pub train_motif_nets_run_config_ids: Vec<RunConfigId>,

    #[arg(long)]
    pub train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[command(flatten)]
    pub opt_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt: String, //TrainLabelNetRunConfig

    #[arg(long)]
    pub max_motif_net_loss: Option<f32>,

    #[arg(long)]
    pub selected_motif_nets_limit: Option<usize>,

    #[arg(long)]
    pub motif_slices_step: Option<u32>,

    #[arg(long)]
    pub labels: Option<String>,
}

impl TrainLabelNetsArgs {
    pub fn resource_config(&self) -> NetResources<Labeled> {
        NetResources::new(
            Labeled::new(self.train_resource_group.clone()),
            Labeled::new(self.valid_resource_group.clone()),
        )
    }

    fn parse_labels_csv(raw: &str) -> Vec<String> {
        raw.split(',').map(|label| label.trim().to_string()).filter(|label| !label.is_empty()).collect()
    }

    pub fn labels_filter(&self) -> Option<Vec<String>> {
        self.labels.as_deref().map(Self::parse_labels_csv).filter(|labels| !labels.is_empty())
    }
}

#[allow(non_snake_case, unused_assignments, unused_variables)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainLabelNetRunConfig {//TODO rename to TrainLabelNetRunConfig
    #[in_map]
    pub id: Option<RunConfigId>,

    #[in_map]
    #[run_config]
    pub common_training_args: CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    pub net: LabelNetTypeArgs,
}

impl AsRef<CommonTrainingRunConfig> for TrainLabelNetRunConfig {
    fn as_ref(&self) -> &CommonTrainingRunConfig {
        &self.common_training_args
    }
}

impl TrainLabelNetRunConfig {
    pub fn new(
        id: Option<RunConfigId>,
        common_training_args: CommonTrainingRunConfig,
        net: LabelNetTypeArgs,
    ) -> Self {
        Self {
            id,
            common_training_args,
            net,
        }
    }
}

#[derive(Debug)]
pub struct TrainLabelNetRunConfigResult {
    train_label_nets_run_config_id: Option<RunConfigId>,
    pub label: String,
    pub loss: f32,
    label_node_params: LabelNetParams,
    train_label_nets_run_config: TrainLabelNetRunConfig,
    var_store: VarStore,
    motif_nets_ids:Vec<u32>
}

impl WithId for Box<TrainLabelNetRunConfigResult> {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}


impl TrainLabelNetRunConfigResult {
    pub fn new(
        label: String,
        loss: f32,
        label_node_params: LabelNetParams,
        train_label_nets_run_config: TrainLabelNetRunConfig,
        var_store: VarStore,
    ) -> Self {
        Self {
            train_label_nets_run_config_id: None,
            label,
            loss,
            label_node_params,
            train_label_nets_run_config,
            var_store,
            motif_nets_ids: vec![],
        }
    }
}

#[derive(Debug)]
pub struct TrainLabelNetRunConfigResultParams<'a> {
    pub train_motif_nets_run_config_ids: &'a Vec<RunConfigId>,
    pub motif_nets_ids: &'a Vec<u32>,
}

#[async_trait]
impl SavableResult for TrainLabelNetRunConfigResult {
    type P<'a> = TrainLabelNetRunConfigResultParams<'a>;

    type R = u32;

    async fn save(
        &mut self,
        store: &Arc<Box<dyn Store>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        p: Self::P<'_>
    ) -> Result<u32, ThoError> {

        let run_config_id = store.save_run_config(
            Some(p.train_motif_nets_run_config_ids), //it's used as input for TrainLabelNetsOutput
            &TrainLabelNetRunConfig::type_name(),
            "Label",
            self.train_label_nets_run_config.to_properties_iter(None).collect(),
        ).await?;

        let path = ws.read()
            .save_label_net(
                run_config_id,
                &self.label,
                &self.var_store,
                &self.label_node_params,
            )?;

        debug!("TrainLabelNetRunConfigResult::save: self.motif_nets_ids={:?}", self.motif_nets_ids);

        store.save_trained_net(run_config_id, path, self.loss, Some(p.motif_nets_ids.clone())).await?;

        Ok(run_config_id)
    }
}

#[derive(Debug)]
pub struct TrainFinalNetsRunConfigResult {
    train_label_nodes_run_config_ids: Vec<RunConfigId>,
    pub loss: f32,
    var_store: VarStore,
    final_node_params: FinalNetParams,
    train_final_net_run_config: TrainFinalNetRunConfig,
}

impl TrainFinalNetsRunConfigResult {
    pub fn new(
        train_label_nodes_run_config_ids: Option<&Vec<RunConfigId>>,
        loss: f32,
        var_store: VarStore,
        final_node_params: FinalNetParams,
        train_final_nodes_values: TrainFinalNetRunConfig,
    ) -> Self {
        Self {
            train_label_nodes_run_config_ids: train_label_nodes_run_config_ids
                .expect("train_label_nodes_run_config_ids")
                .clone(),
            loss,
            var_store,
            final_node_params,
            train_final_net_run_config: train_final_nodes_values,
        }
    }
}

#[async_trait]
impl SavableResult for TrainFinalNetsRunConfigResult {
    type R = u32;

    async fn save(
        &mut self,
        store: &Arc<Box<dyn Store>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        p: Self::P<'_>
    ) -> Result<u32, ThoError> {
        let run_config_id = store.save_run_config(
            Some(&self.train_label_nodes_run_config_ids),
            &TrainFinalNetRunConfig::type_name(),
            "Final",
            self.train_final_net_run_config.to_properties_iter(None).collect(),
        ).await?;

        let path = ws.read()
            .save_final_net(run_config_id, &self.var_store, &self.final_node_params)?;

        store.save_trained_net(run_config_id, path, self.loss, None).await?;

        Ok(run_config_id)
    }
}
type ModuleFactory<DI, DO> = dyn Fn(&nn::Path) -> Result<Box<dyn ThoModule<DI, DO>>, ThoError>;

pub struct ThoModuleWithStore<DI: Dims + Send + 'static + ?Sized, DO: Dims + Send> {
    id:SynKey,
    module: Box<dyn ThoModule<DI, DO>>,
    var_store: VarStore,
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

//TODO
unsafe impl<DI: Dims + Send, DO: Dims + Send> Send for ThoModuleWithStore<DI, DO> {}
unsafe impl<DI: Dims + Send, DO: Dims + Send> Sync for ThoModuleWithStore<DI, DO> {}

impl<DI: Dims + Send + 'static, DO: Dims + Send + 'static> ThoModule<DI, DO> for ThoModuleWithStore<DI, DO>
{
    fn tho_forward(&self, t: &ThoTensor<DI, SimpleId>, lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoTensor<DO, SimpleId>, ThoError> {
        let _guard = InferenceModeGuard::enabled();
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
                return Err(ThoError::new(format!("ThoModule {:?} contains tensors of unexpected precision: store_float_precision={store_float_precision:?}", self.id)))
            }
        };

        let t = if !float_precision.eq(&t.as_ref().kind()) {
            debug!("ThoModuleWithStore::tho_forward: t.id={:?}, converting to {:?}", t.id(), float_precision);

            &t.to_kind(float_precision)?
        } else {
            t
        };
        debug!("ThoModuleWithStore::tho_forward: t.id={:?}, t.as_ref().kind={:?}", t.id(), t.as_ref().kind());

        self.module.tho_forward(t, lengths, train)
    }
}

impl<DI: Dims + Send + 'static, DO: Dims + Send + 'static> ThoModuleWithStore<DI, DO> {
    // #[cfg(test)]
    pub fn new(id: SynKey, module: Box<dyn ThoModule<DI, DO>>) -> Self {
        Self::with_factory(id, module, VarStore::new(Cpu), None)
    }

    pub fn with_factory<I:Into<SynKey>>(
        id: I,
        module: Box<dyn ThoModule<DI, DO>>,
        var_store: VarStore,
        module_factory: Option<Arc<ModuleFactory<DI, DO>>>,
    ) -> Self {
        Self {
            id: id.into(),
            module,
            var_store,
            module_factory,
        }
    }

    pub fn load<F>(id: SynKey, builder: F, model: PathBuf, device: Device, quantize: bool) -> Result<Self, ThoError>
    where
        F: Fn(&nn::Path) -> Result<Box<dyn ThoModule<DI, DO>>, ThoError> + Send + Sync + 'static,
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

        Ok(Self::with_factory(id, module, vs, Some(builder)))
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
        self.module.debug();
    }

    pub fn infer(&self, t: &ThoTensor<DI, SimpleId>) -> ThoTensor<DO, SimpleId> {
        self.module.tho_forward(t, None, false).unwrap()
    }

    pub fn var_store(&self) -> &VarStore {
        &self.var_store
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

        let builder = self
            .module_factory
            .as_ref()
            .unwrap_or_else(|| panic!("copy_to_device requires module_factory; construct modules via ThoModuleWithStore::load to enable copying"));

        let mut var_store = VarStore::new(ThoDevice::from(tho_dst_device).into());

        let module = builder(&var_store.root())?;

        var_store.copy(&self.var_store)?;
        // copy_vs(&self.var_store, &var_store).await?;

        var_store.freeze();

        Ok(Self::with_factory(self.id.clone(), module, var_store, Some(Arc::clone(builder))))
    }
}


impl <DI, DO> Data for ThoModuleWithStore<DI, DO>
where
    DI: Dims + Send + 'static,
    DO: Dims + Send + 'static{
    type Metrics = ();
    type M = CpuOrGpu<ThoModuleWithStore<DI, DO>>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        ThoDevice::from(self.var_store.device()).ordinal()
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

//TODO move to ws
pub fn load_final_model(
    run_config_id: RunConfigId,
    ws: &Arc<RwLock<Box<dyn Workspace>>>,
    device: &Device,
) -> Result<
    (
        FinalNetParams,
        ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>,
    ),
    ThoError,
> {
    let (model_path, params_path) = ws.read().final_model(run_config_id)?;
    debug!("load_final_model: params_path={model_path:?} run_config_id={run_config_id}");

    let final_node_params = FinalNetParams::from_file(&params_path)?;
    debug!("load_final_model:  final_node_params={final_node_params:?}");

    let final_params_for_builder = final_node_params.clone();
    let node = ThoModuleWithStore::load(
        mono_key!(run_config_id, "final_model").into(),
        move |path: &nn::Path| final_net(&final_params_for_builder)?.build_tch_module(path),
        model_path,
        *device,
        false,
    )?;

    Ok((final_node_params, node))
}

#[derive(Args, Debug, Clone)]
pub struct TrainMotifsNetsArgs {
    #[arg(long, value_delimiter = ',', value_parser = crate::utils::parse_run_config_ids)]
    pub select_motifs_run_config_ids: Option<Vec<RunConfigId>>,

    #[arg(long)]
    pub motif_id: Option<u32>,

    #[command(flatten)]
    pub opt_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt: String, //TrainMotifNetRunConfig

    #[arg(long, default_value_t = 10)]
    pub epoch_size: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, DummyData)]
pub enum MotifsBy {
    SelectMotifsRunIds { select_motifs_run_config_ids: Vec<RunConfigId> },
    MotifId { motif_id: u32, select_motifs_run_config_ids: Option<Vec<RunConfigId>> },
}

impl TrainMotifsNetsArgs {
    pub fn resource_config(&self) -> Empty {
        Empty {}
    }

    pub fn motifs_by(&self) -> Result<MotifsBy, ThoError> {
        match (self.motif_id, self.select_motifs_run_config_ids.clone()) {
            (Some(motif_id), select_motifs_run_config_ids) => Ok(MotifsBy::MotifId { motif_id, select_motifs_run_config_ids }),
            (None, Some(select_motifs_run_config_ids)) => Ok(MotifsBy::SelectMotifsRunIds { select_motifs_run_config_ids }),
            (None, None) => Err(ThoError::new("TrainMotifsNetsArgs: either --select-motifs-run-ids or --motif-id must be provided")),
        }
    }
}

#[cfg(test)]
mod train_motifs_nets_args_tests {
    use super::{MotifsBy, OptConfig, TrainMotifsNetsArgs};

    #[test]
    fn motifs_by_from_select_run_config_ids() {
        let args = TrainMotifsNetsArgs {
            select_motifs_run_config_ids: Some(vec![1, 2]),
            motif_id: None,
            opt_config: OptConfig::new(10, 1),
            run_config_cnt: "{}".to_string(),
            epoch_size: 10,
        };

        assert_eq!(
            args.motifs_by().unwrap(),
            MotifsBy::SelectMotifsRunIds {
                select_motifs_run_config_ids: vec![1, 2],
            }
        );
    }

    #[test]
    fn motifs_by_from_motif_id_and_optional_select_run_config_ids() {
        let args = TrainMotifsNetsArgs {
            select_motifs_run_config_ids: Some(vec![10, 11]),
            motif_id: Some(7),
            opt_config: OptConfig::new(10, 1),
            run_config_cnt: "{}".to_string(),
            epoch_size: 10,
        };

        assert_eq!(
            args.motifs_by().unwrap(),
            MotifsBy::MotifId {
                motif_id: 7,
                select_motifs_run_config_ids: Some(vec![10, 11]),
            }
        );
    }

    #[test]
    fn motifs_by_requires_one_source() {
        let args = TrainMotifsNetsArgs {
            select_motifs_run_config_ids: None,
            motif_id: None,
            opt_config: OptConfig::new(10, 1),
            run_config_cnt: "{}".to_string(),
            epoch_size: 10,
        };

        assert!(args.motifs_by().is_err());
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct MotifConvNetArgs {
    #[in_map]
    #[run_config]
    pub conv_layers: u32,

    #[in_map]
    #[run_config]
    pub head_layers: u32,

    #[in_map]
    #[run_config]
    pub step: u32,//TODO renamed to Conv stride

    #[in_map]
    #[run_config]
    pub channels:u32,

    #[in_map]
    #[run_config]
    pub kernel: u32,

    #[in_map]
    #[run_config]
    pub head_pool:Option<BinaryHeadPoolArgs>
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct MotifLinNetArgs {
    #[in_map]
    #[run_config]
    pub num_layers: u32,

    #[in_map]
    #[run_config]
    pub step: u32,//TODO renamed to Conv stride

    #[in_map]
    #[run_config]
    pub channels:u32,

    #[in_map]
    #[run_config]
    pub kernel: u32
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct MotifCosineNetArgs {

    #[in_map]
    #[run_config]
    pub a:f64,

    #[in_map]
    #[run_config]
    pub b: f64,

    #[in_map]
    #[run_config]
    pub init_mode:MotifInitMode

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

#[allow(non_snake_case, unused_assignments, unused_variables, non_camel_case_types)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MotifNetTypeArgs {
    ConvNet(MotifConvNetArgs),
    LinNet(MotifLinNetArgs),
    CosineNet(MotifCosineNetArgs),
    Cosine2DNet(MotifCosineNetArgs),
}

impl Default for MotifNetTypeArgs {
    fn default() -> Self {
        MotifNetTypeArgs::ConvNet(MotifConvNetArgs::default())
    }
}

#[allow(non_snake_case, unused_assignments, unused_variables)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainMotifNetRunConfig {
    #[in_map]
    pub id: Option<RunConfigId>,

    #[in_map]
    #[run_config]
    pub common_training_args: CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    #[serde(default)]
    pub net: MotifNetTypeArgs,


}

impl AsRef<CommonTrainingRunConfig> for TrainMotifNetRunConfig {
    fn as_ref(&self) -> &CommonTrainingRunConfig {
        &self.common_training_args
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Deserialize, Debug, PartialEq, Clone)] //PartialEq - for tests
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

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub enum Optimizers {
    AdamW(AdamW)
}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub struct CommonTrainingRunConfig {
    #[in_map]
    #[run_config]
    pub epochs: u32,

    #[in_map]
    #[run_config]
    pub batch_size: u32, //number of the motif slices in a batch

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

    pub fn build_optimizer(&self, vs: &VarStore)->Result<Optimizer, ThoError>{
        let optimizer = self.optimizer.clone().unwrap_or_else(Self::default_optimizer);

        debug!("CommonTrainingRunConfig::build_optimizer: optimizer={optimizer:?}");

        match optimizer {
            Optimizers::AdamW(AdamW{ beta1, beta2, wd, amsgrad }) => {
                tch::nn::AdamW{
                    beta1: beta1 as f64,
                    beta2: beta2 as f64,
                    wd,
                    eps: 1e-8,
                    amsgrad,
                }.build(&vs, self.lr).map_err(|e|e.into())
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

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MotifConvNetParams {
    pub num_features: u32,
    pub conv_layers: u32,
    pub head_layers: u32,
    pub stride: u32,//stride within slice
    pub channels: u32,
    pub kernel: u32,
    pub head_pool: Option<BinaryHeadPool>
}

impl MotifConvNetParams {
    pub fn new(num_features: u32, conv_layers: u32, head_layers: u32, stride: u32, channels: u32, kernel: u32) -> Self {
        Self { num_features, conv_layers, head_layers, stride, channels, kernel, head_pool:None }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MotifLinNetParams {
    pub num_features: u32,
    pub num_layers: u32,
}

impl MotifLinNetParams {
    pub fn new(num_features: u32, num_layers: u32) -> Self {
        Self { num_features, num_layers }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MotifCosineNetParams {
    pub num_features: u32,
    pub a:f64,
    pub b:f64,
    pub init_mode:MotifInitMode
}

impl MotifCosineNetParams {
    pub fn new(num_features: u32, a: f64, b: f64, init_mode: MotifInitMode) -> Self {
        Self { num_features, a, b, init_mode }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MotifCosine2dNetParams {
    pub kw: u32,
    pub kh: u32,
    pub a:f64,
    pub b:f64,
    pub init_mode:MotifInitMode
}

impl MotifCosine2dNetParams {
    pub fn new(kw: u32, kh: u32, a: f64, b: f64, init_mode: MotifInitMode) -> Self {
        Self { kw, kh, a, b, init_mode }
    }
}


#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub enum MotifNetTypeParams {
    LinNetParams(MotifLinNetParams),
    ConvNetParams(MotifConvNetParams),
    CosineNetParams(MotifCosineNetParams),
    CosineNet2DParams(MotifCosine2dNetParams),
}

impl Default for MotifNetTypeParams {
    fn default() -> Self {
        MotifNetTypeParams::ConvNetParams(MotifConvNetParams::new(0, 1, 1, 100, 32, 10))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct MotifNetParams {
    #[serde(default)]
    pub net_type: MotifNetTypeParams,
}

impl MotifNetParams {
    pub fn new(num_features: u32, conv_layers: u32, head_layers: u32, stride: u32, channels: u32, kernel: u32) -> Self {
        Self::new_conv(num_features, conv_layers, head_layers, stride, channels, kernel)
    }

    pub fn new_with_net_type(net_type: MotifNetTypeParams) -> Self {
        Self { net_type }
    }

    pub fn new_conv(num_features: u32, conv_layers: u32, head_layers: u32, stride: u32, channels: u32, kernel: u32) -> Self {//head_pool: BinaryHeadPool
        Self::new_with_net_type(MotifNetTypeParams::ConvNetParams(MotifConvNetParams::new(num_features, conv_layers, head_layers, stride, channels, kernel)))
    }

    pub fn new_lin(num_features: u32, num_layers: u32, stride: u32) -> Self {
        Self::new_with_net_type(MotifNetTypeParams::LinNetParams(MotifLinNetParams::new(num_features, num_layers)))
    }

    pub fn new_cosine(num_features: u32, a: f64, b: f64, init_mode: MotifInitMode) -> Self {
        Self::new_with_net_type(MotifNetTypeParams::CosineNetParams(MotifCosineNetParams::new(num_features, a, b, init_mode)))
    }

    pub fn new_cosine_2d(num_features: u32, a: f64, b: f64, init_mode: MotifInitMode) -> Self {
        Self::new_with_net_type(MotifNetTypeParams::CosineNetParams(MotifCosineNetParams::new(num_features, a, b, init_mode)))
    }


    pub fn from_file(path: &PathBuf) -> Result<Self, ThoError> {
        serde_json5::from_str::<MotifNetParams>(fs::read_to_string(&path)?.as_str()).map_err(
            |serde_err| ThoError::from_error(format!("read motif net params from {path:?}"), serde_err),
        )
    }

    //TODO
    pub fn num_features(&self) -> u32 {//this is used to group motif nets + calculate overlap for StepAlignedSplitsAdjuster::new_with_overlap at thoosa_main/src/tasks/apply_motif_nets.rs
        match &self.net_type {
            MotifNetTypeParams::LinNetParams(v) => v.num_features,
            MotifNetTypeParams::ConvNetParams(v) => v.num_features,
            MotifNetTypeParams::CosineNetParams(v) => v.num_features,
            MotifNetTypeParams::CosineNet2DParams(v) => {v.kw * v.kh}
        }
    }

    //TODO
    pub fn motif_net(&self, motif: Option<Arc<ThoTensor<MotifSlices, SynKey>>>) -> Result<Node, ThoError> {
        match &self.net_type {
            MotifNetTypeParams::LinNetParams(v) => lin_motif_node(v),
            MotifNetTypeParams::ConvNetParams(v) => conv_motif_node(v),
            MotifNetTypeParams::CosineNetParams(v) => cosine_motif_node(v, motif),
            MotifNetTypeParams::CosineNet2DParams(v) => {cosine_2d_motif_node(v, motif.map(|tt|Arc::new(ThoTensor::new(tt.id().clone(), Arc::into_inner(tt).unwrap().t).unwrap())))}//TODO
        }
    }
}


pub fn conv_motif_node(params: &MotifConvNetParams) -> Result<Node, ThoError> {
    debug!("motif_node: params={params:?}");

    let mut builder = SeqBuilder::new(params.num_features, None)?;
    let nd = ConvConfig {
        stride: params.stride as i64,
        ..ConvConfig::default()
    };

    let hidden_config = ConvConfig::default();

    let mut net = builder
        // .batch_norm(BatchNormConfig::default())?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                let r = t.unsqueeze(1);

                debug!("conv_motif_node: 0. t.size={:?} r.size={:?}", t.size(), r.size());

                Ok(r)
            })),
            None,
        )?;


    net = net
        .conv_1d("motif_conv_1", params.kernel, 1, params.channels, nd, None)?
        .func(|t| Tensor::f_relu(t).map_err(|e| e.into()))?;

    let mut net = builder
        // .batch_norm(BatchNormConfig::default())?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                let r = t.shallow_clone();

                debug!("conv_motif_node, after motif_conv_1: t.size={:?} r.size={:?}", t.size(), r.size());

                Ok(r)
            })),
            None,
        )?;

    for i in 0..params.conv_layers as usize {
        net = net
            .conv_1d(format!("motif_{i}_conv"), params.kernel, params.channels, params.channels, hidden_config, None)?
            .closure(
                Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {

                    debug!("conv_motif_node after conv_1d motif_{i}_conv, t.size={:?} _lengths={_lengths:?}", t.size());

                    let r = t.shallow_clone();

                    Ok(r)
                })),
                None,
            )?
            // .batch_norm(BatchNormConfig::default(), Some(params.channels))?
            .func(|t| Tensor::f_gelu(t, "none").map_err(|e| e.into()))?
            // .dropout(0.1)?
        ;
    }

    net = net.conv_1d("score_head", 1, params.channels, 1, ConvConfig::default(), None)?;

    net = net.avg_pool(params.kernel, 1, params.kernel/2)?;



    // net = net.adaptive_avg_pool(1, params.channels)?;


net.closure(
        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
            let r = t.squeeze();

            // let max = r.f_amax(-1, true)?;
            // let min = r.f_mean_dim(-1, true, None)?;
            //
            // let r1 = Tensor::f_cat(&[max.shallow_clone(), min.shallow_clone()], 1)?;
            //
            // debug!("conv_motif_node: 1. t.size={:?} r.size={:?} r1.size={:?} max.size={:?}, min.size={:?}", t.size(), r.size(), r1.size(), max.size(), min.size());
            debug!("conv_motif_node: 1. t.size={:?}  r.size={:?}", t.size(), r.size());

            Ok(r) //r/max
        })),
        None,
    )?
    // .linear_feats(
    //     "motif_first_head",
    //     2,1
    // )?
  // .closure(
  //     Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
  //         let r = t.shallow_clone();
  //
  //         debug!("conv_motif_node, after motif_first_head: t.size={:?} r.size={:?}", t.size(), r.size());
  //
  //         Ok(r)
  //     })),
  //     None,
  // )?
    // .linear("motif_first_head")?
    .func(|t| Tensor::f_relu(t).map_err(|e| e.into()))?;

    for i in 0..params.head_layers as usize {
        net = net
            .linear_f(
                format!("motif_{i}_head"),
                if i < 1 + (params.head_layers / 2) as usize {
                    2.
                } else {
                    0.5
                },
            )?
            .func(|t| Tensor::f_gelu(t, "none").map_err(|e| e.into()))?;
    }

    net
        // .func(|t| Tensor::f_relu(t).map_err(|e| e.into()))?
        .linear_s(
            1,
            "motif_last_head",
        )?
        .end_seq()

}

pub fn cosine_2d_motif_node(params: &MotifCosine2dNetParams, motif: Option<Arc<ThoTensor<Spectorgram, SynKey>>>) -> Result<Node, ThoError> {
    debug!("cosine_2d_motif_node: params={params:?} motif={motif:?}");

    let mut builder = SeqBuilder::new(0, None)?;

    let params =  MotifCosine2dNetParams{
        kw: motif.as_ref().map(|v|v.as_ref().as_ref().size()[2] as u32).unwrap_or(params.kw),//time
        kh: motif.as_ref().map(|v|v.as_ref().as_ref().size()[1] as u32).unwrap_or(params.kh),//freq
        ..params.clone()
    };


    builder
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("cosine_2d_motif_node: 1. t.size={:?}", t.size());
                Ok(t.unsqueeze(1))
            })),
            None,
        )?
        .cosine_conv_2d("cosine_bank", params.kw, params.kh, motif, params.a, params.b, params.init_mode)?
        // .closure(
        //     Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
        //         debug!("cosine_motif_node: 2. t.size={:?}", t.size());
        //         Ok(t.shallow_clone())
        //     })),
        //     None,
        // )?
        .end_seq()
}

pub fn cosine_motif_node(params: &MotifCosineNetParams, motif: Option<Arc<ThoTensor<MotifSlices, SynKey>>>) -> Result<Node, ThoError> {
    debug!("cosine_motif_node: params={params:?}");

    let mut builder = SeqBuilder::new(params.num_features, None)?;

    let params =  MotifCosineNetParams{
        num_features: motif.as_ref().map(|v|v.size().dim_1() as u32).unwrap_or(params.num_features),
        ..params.clone()
    };


builder
    .closure(
        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
            debug!("cosine_motif_node: 1. t.size={:?}", t.size());
            Ok(t.unsqueeze(1))
        })),
        None,
    )?
    .cosine_conv_1d("cosine_bank", params.num_features, motif, params.a, params.b, params.init_mode)?
    // .closure(
    //     Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
    //         debug!("cosine_motif_node: 2. t.size={:?}", t.size());
    //         Ok(t.shallow_clone())
    //     })),
    //     None,
    // )?
    .end_seq()

}


pub fn lin_motif_node(params: &MotifLinNetParams) -> Result<Node, ThoError> {
    debug!("motif_node: params={params:?}");

    let mut builder = SeqBuilder::new(params.num_features, None)?;
    let mut net = builder
        // .batch_norm()
        .linear_s(
            params.num_features * 2,
            "todo",
        )?
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?;

    for i in 0..params.num_layers as usize {
        net = net
            .linear_s(
                params.num_features * 2,
                "todo",
            )?
            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?;
    }

    net
        // .linear_s(180)
        // .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
        .linear_s(
            1,
            "todo",
        )?
        .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))?
        .end_seq()

}

#[derive(Debug, Copy, Clone)]
pub enum NetsPadding {
    Max,
    Fixed(u32),
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabelNetParams {
    motif_nets: u32,

    pub net_type: LabelNetTypeParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LabelNetTypeParams {

    LinNetParams(LinLenNetParams),
    ConvNetParams(ConvNetParams),
    SelfAttentionNetParams(SelfAttentionNetParams),
}

impl Default for LabelNetTypeParams {
    fn default() -> Self {
        LabelNetTypeParams::LinNetParams(LinLenNetParams::new(0, 3))
    }
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
    num_conv_layers:u32,
    kernel:u32,
    hidden_conv_channels:u32,

    num_head_layers:u32,
    num_head_features:u32,

    joint: ConvNetJoint
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionNetParams {
    head_size: u32, //d_model- times n_heads
    n_heads: u32,
    ff_mult: u32,
    dropout: f64,
}

impl SelfAttentionNetParams {
    pub fn new(head_size: u32, n_heads: u32, ff_mult: u32, dropout: f64) -> Self {
        Self { head_size, n_heads, ff_mult, dropout }
    }
}


#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinLenNetParams {
    pub motif_layers: u32,
    pub features: u32,
}

impl LinLenNetParams {
    pub fn new(features: u32, motif_layers: u32) -> Self {
        Self { motif_layers, features }
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

impl LabelNetParams {
    pub fn new(
        motif_nets: u32, //TODO move to LinNetParams
        net_type: LabelNetTypeParams,
    ) -> Self {
        Self {
            motif_nets,
            net_type,
        }
    }

    pub fn from_file(path: &PathBuf) -> Result<Self, ThoError> {
        serde_json5::from_str::<LabelNetParams>(fs::read_to_string(&path)?.as_str())
            .map_err(|serde_err| {
                ThoError::from_error(format!("read label net params from {path:?}"), serde_err)
            })
    }

    pub fn label_net(&self) -> Result<Node, ThoError> {
        match &self.net_type {
            LabelNetTypeParams::LinNetParams(_) => {
                lin_label_node(&self)
            }
            LabelNetTypeParams::ConvNetParams(_) => {
                conv_label_node(&self)
            }
            LabelNetTypeParams::SelfAttentionNetParams(_) => {
                self_attention_label_node(&self)
            }
        }
    }


}

fn self_attention_masked_mean(h: &Tensor, lengths: Option<&BatchLengths>) -> Result<Tensor, ThoError> {
    let h_size = h.size();
    if h_size.len() != 3 {
        return Err(ThoError::new(format!("self_attention_masked_mean expected [B,L,D], got {:?}", h_size)));
    }
    if let Some(lengths) = lengths {
        let kind = h.kind();
        let mask = mask_from_lengths(lengths.as_slice(), h_size[1], h.device())?.f_squeeze_dim(1)?;
        let mask_f = mask.f_to_kind(kind)?.f_unsqueeze(-1)?;
        let sum = h.f_mul(&mask_f)?.f_sum_dim_intlist([1].as_slice(), false, kind)?;
        let denom = mask_f.f_sum_dim_intlist([1].as_slice(), false, kind)?.f_clamp_min(1.0)?;
        sum.f_div(&denom).map_err(|e| e.into())
    } else {
        h.f_mean_dim(1, false, None).map_err(|e| e.into())
    }
}

pub fn conv_label_node(params: &LabelNetParams) -> Result<Node, ThoError> {
    let LabelNetTypeParams::ConvNetParams(cnn_params) = &params.net_type else {
        panic!("Unexpected params type: {:?}", params.net_type);
    };

    debug!("cnn_label_node: params={params:?}");


    let mut seq_builder = SeqBuilder::new(0, None)?;
    let mut builder = seq_builder
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("cnn_label_node: input={t:?}");
                Ok(t.shallow_clone())
            })),
            None,
        )?;

    builder = builder.conv_1d(
        "cnn_label_conv_1d",
        cnn_params.kernel,
        params.motif_nets,
        cnn_params.hidden_conv_channels,
        ConvConfig::default(),
        None,
    )?.closure(
        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
            debug!("cnn_label_node: cnn_label_conv_1d output={t:?}");
            Ok(t.shallow_clone())
        })),
        None,
    )?;

    for layer in 0..cnn_params.num_conv_layers {
        builder = builder.conv_1d(
            format!("cnn_label_conv_1d_{layer}"),
            cnn_params.kernel,
            cnn_params.hidden_conv_channels,
            cnn_params.hidden_conv_channels,
            ConvConfig::default(),
            None,
        )?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("cnn_label_node: cnn_label_conv_1d_{layer} output={t:?}");
                Ok(t.shallow_clone())
            })),
            None,
        )?;
    }

    builder = match &cnn_params.joint{
        ConvNetJoint::AttnPool1d => {
            builder.attn_pool1d(
                "attn_pool",
                cnn_params.hidden_conv_channels,
                2*cnn_params.hidden_conv_channels
            )?
            .closure(
                Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                    debug!("cnn_label_node: attn_pool output={t:?}");
                    Ok(t.shallow_clone())
                })),
                None,
            )?.linear_feats(
                "cnn_label_head",
                cnn_params.hidden_conv_channels, 2*cnn_params.num_head_features,
            )?
            .closure(
                Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                    debug!("cnn_label_node: cnn_label_head output={t:?}");
                    Ok(t.shallow_clone())
                })),
                None,
            )?
        }

        ConvNetJoint::MinMax => {
            builder.closure(
                Arc::new(Box::new(move |t: &Tensor, lengths: Option<&BatchLengths>| {
                    debug!("cnn_label_node: MinMax leaner input={t:?} lengths={lengths:?}");
                    let lengths = lengths.map(BatchLengths::as_slice).unwrap_or(&[]);//TODO
                    let mask = mask_from_lengths(lengths, *lengths.iter().max().unwrap_or(&0), t.device())?;

                    let mean = t.f_mean_dim(2, false, None)?;
                    let max = t.f_max_dim(2, false)?;
                    let mean = masked_mean(t, &mask)?;
                    let max = masked_max(t, &mask)?;

                    Tensor::f_cat(&[mean, max], 1).map_err(|e|e.into())

                })),
                None,
            )?
            .closure(
                Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                    debug!("cnn_label_node: MinMax output={t:?}");
                    Ok(t.shallow_clone())
                })),
                None,
            )?
            .linear_feats(
                "cnn_label_head",
                2* cnn_params.hidden_conv_channels, 2*cnn_params.num_head_features,
            )?
                .closure(
                    Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                        debug!("cnn_label_node: cnn_label_head output={t:?}");
                        Ok(t.shallow_clone())
                    })),
                    None,
                )?
        }
    };

    for layer in 0..cnn_params.num_head_layers {
        builder = builder.linear_s(
            2*cnn_params.num_head_features,
            format!("cnn_label_head_{layer}"),
        )?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("cnn_label_node: cnn_label_head_{layer} output={t:?}");
                Ok(t.shallow_clone())
            })),
            None,
        )?
        .func(|t| Tensor::f_relu(t).map_err(|e| e.into()))?;
    }

    builder.linear_s(
        1,
        "cnn_label_head_last",
    )?
    .closure(
        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
            debug!("cnn_label_node: cnn_label_head_last={t:?}");
            Ok(t.shallow_clone())
        })),
        None,
    )?
    // .func(|t|Tensor::f_sigmoid(t).map_err(|e|e.into()))
    .end_seq()
}

pub fn self_attention_label_node(params: &LabelNetParams) -> Result<Node, ThoError> {
    let LabelNetTypeParams::SelfAttentionNetParams(sa_params) = &params.net_type else {
        panic!("Unexpected params type: {:?}", params.net_type);
    };

    debug!("self_attention_label_node: params={params:?}");


    let mut seq_builder = SeqBuilder::new(0, None)?;
    let builder = seq_builder
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("self_attention_label_node: transpose input={t:?} _lengths={_lengths:?}");
                Ok(t.transpose(1, 2))
            })),
            Some(params.motif_nets),
        )?
        .linear_feats(
            "self_attention_in_proj",
            params.motif_nets,
            sa_params.head_size * sa_params.n_heads,
        )?
        .self_attention(
            "self_attention_block",
            sa_params.head_size * sa_params.n_heads,
            sa_params.n_heads,
            sa_params.ff_mult,
            sa_params.dropout,
        )?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, lengths: Option<&BatchLengths>| {
                debug!("self_attention_label_node: pooling input={t:?} lengths={lengths:?}");
                self_attention_masked_mean(t, lengths)
            })),
            None,
        )?
        .linear_s(
            1,
            "self_attention_label_head_last",
        )?;

        #[cfg(any(debug_assertions, tho_debug))] {
        let builder = builder.closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("self_attention_label_node: self_attention_label_head_last={t:?}");
                Ok(t.shallow_clone())
            })),
            None,
        )?;}


    builder .end_seq()
}

pub fn lin_label_node(params: &LabelNetParams) -> Result<Node, ThoError> {
    debug!("lin_label_node: params={params:?}");

    let LabelNetTypeParams::LinNetParams(lin_params) = &params.net_type else {
        // return Err(ThoError::new(format!("Unexpected params type: {cnn_params:?}")));
        panic!("Unexpected params type: {:?}", params.net_type);
    };

    let num_features = lin_params.features;//params.motif_nets;
    debug!("lin_label_node: num_features={num_features}");

    let mut seq_builder = SeqBuilder::new(0, None)?;
    let mut builder = seq_builder
        // .closure(
        //     Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
        //         debug!("lin_label_node: transpose input={:?} _lengths={_lengths:?}", t.size());
        //         // Ok(t.transpose(1, 2))
        //         Ok(t.shallow_clone())
        //     })),
        //     Some(params.motif_nets),
        // )?
        .adaptive_avg_pool(lin_params.features,lin_params.features )?;

        for layer in 0..lin_params.motif_layers {
            builder = builder.linear_feats(
                "lin_label_node",
                lin_params.features,
                lin_params.features,
            )?
            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?;
        }


    builder.linear_s(
        1,
        "lin_label_node last",
    )?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("lin_label_node: out={:?}", t.size());
                Ok(t.f_mean_dim(1, false, None)?)
            })),
            Some(params.motif_nets),
        )?
        .end_seq()

}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct FinalNetParams {
    in_features: u32,
    out_features: u32,
    num_layers: u32,
}

impl FinalNetParams {
    pub fn new(in_features: u32, out_features: u32, num_layers: u32) -> Self {
        Self {
            in_features,
            out_features,
            num_layers,
        }
    }

    pub fn from_file(path: &PathBuf) -> Result<Self, ThoError> {
        serde_json5::from_str::<FinalNetParams>(fs::read_to_string(&path)?.as_str())
            .map_err(|serde_err| {
                ThoError::from_error(format!("read final net params from {path:?}"), serde_err)
            })
    }
}

pub fn final_net(params: &FinalNetParams) -> Result<Node, ThoError> {
    debug!("final_node: params={:?}", params);

    let mut builder = SeqBuilder::new(params.in_features, None)?;
    let mut net = builder
        .linear_s(
            2 * params.in_features,
            "final_net_first_layer".to_string(),
        )?
        .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?;

    for i in 0..params.num_layers {
        net = net
            .linear_s(
                params.in_features,
                format!("final_net_{i}th_layer"),
            )?
            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?;
    }

net
    // .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
    .linear_s(
        params.out_features,
        "final_net_last_layer",
    )?
    .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
    .end_seq()
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

    #[allow(dead_code)]
    fn plot(&self) {
        plot_series(
            vec![(
                self.id.clone(),
                BLACK,
                self.data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i as f32, *v))
                    .collect::<Vec<(f32, f32)>>(),
            )],
            self.id.clone().as_str(),
            (2000, 1000),
            &format!("/tmp/plots/motif_train_{}.png", self.id),
        )
        .unwrap();
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

#[derive(Debug)]
pub struct TrainMotifNetResult {
    pub loss: f32, //TODO pub
    channel: u32,
    motif_id: u32,
    pub vs: VarStore,
    pub params: MotifNetParams,
    run_config_values: TrainMotifNetRunConfig,
    #[allow(dead_code)]
    select_motifs_run_config_id: RunConfigId,
    descr: Option<String>
}

impl TrainMotifNetResult {
    pub fn new(
        loss: f32,
        channel: u32,
        motif_id: u32,
        vs: VarStore,
        select_motifs_run_config_id: RunConfigId,
        params: MotifNetParams,
        run_config_values: TrainMotifNetRunConfig,
        descr: Option<String>,
    ) -> Self {
        Self {
            loss,
            channel,
            motif_id,
            vs,
            params,
            run_config_values,
            select_motifs_run_config_id,
            descr
        }
    }


}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainMotifsNetsResult {//TODO move to train_motif_nets.rs
    select_motifs_run_config_ids: Vec<RunConfigId>,
    pub train_motif_nets_results:Vec<TrainMotifNetResult>
}

impl TrainMotifsNetsResult {
    pub fn new(select_motifs_run_config_ids: Vec<RunConfigId>, train_motif_nets_results: Vec<TrainMotifNetResult>) -> Self {
        Self {  select_motifs_run_config_ids, train_motif_nets_results }
    }
}

#[async_trait]
impl SavableResult for TrainMotifsNetsResult {
    type R = Vec<u32>;

    async fn save(&mut self, store: &Arc<Box<dyn Store>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, p: Self::P<'_>) -> Result<Vec<u32>, ThoError> {

        let mut result = Vec::with_capacity(self.train_motif_nets_results.len());
        for r in self.train_motif_nets_results.iter_mut() {
            result.push(
            r.save(store, ws, self.select_motifs_run_config_ids.as_slice()).await?
            );
        }

        Ok(result)
    }
}


impl WithId for TrainMotifsNetsResult {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

#[async_trait]
impl Output for TrainMotifsNetsResult {
}


#[async_trait]
impl SavableResult for TrainMotifNetResult {
    type P<'a> =  &'a [RunConfigId];
    type R = u32;

    async fn save(
        &mut self,
        store: &Arc<Box<dyn Store>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        p: Self::P<'_>
    ) -> Result<u32, ThoError> {

        debug!("TrainMotifNetResult::save: self={:?}", self);

        let train_motif_net_run_config_id = store.save_run_config(
            Some(p),
            &TrainMotifNetRunConfig::type_name(), //TODO separate type for TrainMotifNetResult?
            self.descr.as_ref().map(|v|v.as_str()).unwrap_or("TrainMotifNetResult"),
            self.run_config_values.to_properties_iter(None).collect(),
        ).await?;

        let path = ws.read().save_motif_net(
            &self.channel,
            &train_motif_net_run_config_id,
            &self.motif_id,
            &self.vs,
            &self.params,
        )?;

        store.save_trained_net(train_motif_net_run_config_id, path, self.loss, None).await?;

        Ok(train_motif_net_run_config_id)

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

pub async fn opt_train<P, R, F, N: Debug + Default>(
    opt_config: Arc<OptConfig>, //N node parameters
    mut train: F,
    run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >,
    device: Device,
    descr: &str,
    reusable_completed_trials: Vec<CompletedTrial>,
) -> Result<(VarStore, N, R, f32), ThoError>
where
    P: FieldName + 'static,
    R: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + ToHashmap,
    F: for<'a, 'b> AsyncFnMut(
            Arc<Mutex<VarStore>>,
            R,
            &'a mut (dyn TrainListener + 'b),
        ) -> Result<(f32, N), ThoError>,
{
    let const_vals = run_config_source
        .grid_run_config_source()
        .const_params()
        .clone();
    let min_loss = Arc::new(RwLock::new(f32::MAX));

    let min_var_store: Arc<RwLock<Option<(VarStore, N, R, f32)>>> = Arc::new(RwLock::new(None));

    let grid_runs = run_config_source.grid_to_vals_and_opt_params();

    info!("opt_train: running opt train {descr} grid_runs={grid_runs:?}");

    let grids_eta_guard = CollectEtaInvocationGuard::new(&format!("grid_sets_{descr}"), grid_runs.len(), 1);
    for (grid_vals, opt_params) in grid_runs.into_iter() {
        let const_vals = const_vals.clone();
        let train_clone = &mut train;

        let min_var_store_clone = Arc::clone(&min_var_store);
        let min_loss_moved = Arc::clone(&min_loss);
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
                debug!("opt_train: opt_vals={opt_vals:?}  "); //descr={descr}

                let run_config: R = [const_vals.as_slice(), grid_vals.as_slice(), opt_vals]
                    .concat()
                    .into_iter()
                    .into();

                let vs = Arc::new(Mutex::new(VarStore::new(device)));

                let mut listener = PlotVecf32::new(
                    format!("_params_{:?}", "id" /*TODO run_config.get_id()*/).as_str(),
                ); //motif_id ?train_net

                let (loss, node_params) = panic_to_err_async!(
                    train_clone(
                        Arc::clone(&vs),
                        run_config.clone(),
                        &mut listener
                    )
                ).unwrap_or_else(|e| {
                    warn!("opt_train failed with {e:?} on {run_config:?} ");
                    (1.0, Default::default())
                });
                // listener.plot();

                let min_loss_val = *min_loss_moved.read();
                debug!("opt_train: run_config_values={run_config:?} loss={loss} min_loss_val={min_loss_val:?} "); //descr={descr}

                if loss < min_loss_val {
                    *min_loss_moved.write() = loss;
                    let vs = Arc::try_unwrap(vs)
                        .expect("opt_train: VarStore still shared after train")
                        .into_inner();

                    min_var_store_clone
                        .write()
                        .replace((vs, node_params, run_config, loss));
                }

                loss
            },
            config_space,
            opt_config.as_ref().clone(),
            descr,
        );
        opt.enqueue_configs(reusable_configs);
        info!("opt_train: wait on optimize ");
        let best_config = opt.optimize().await;

        debug!("opt_train: best_config={:?}", best_config);

        debug!("opt_train: done opt, descr={descr} best_config={best_config:?} min_loss={}", min_loss.read());
        trace!("opt_train: done opt, descr={descr} min_var_store={min_var_store:?}");//can cause Internal torch error: CUDA error: device-side assert triggered
        collect_eta_record_completion(grids_eta_guard.id());
    }
    Arc::into_inner(min_var_store)
        .unwrap()
        .into_inner()
        .ok_or_else(|| ThoError::new("opt was not run"))
}

pub fn slices_in_all_resources<ResourceId>(slices: Vec<(ResourceId, Vec<Tensor>)>) -> Vec<Tensor> {
    slices
        .into_iter()
        .flat_map(|(_resource_key, slices)| slices.into_iter())
        .collect::<Vec<Tensor>>()
}


#[derive(Debug)]
pub struct LoadMotifNetOutput {
    #[allow(dead_code)]
    channel: u32,
    #[allow(dead_code)]
    nets: Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>, //single motif's nets (of different types)
}

impl LoadMotifNetOutput {
    pub fn new(
        channel: u32,
        nets: Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>,
    ) -> Self {
        Self { channel, nets }
    }
}

impl WithId for LoadMotifNetOutput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}





#[derive(Args, Debug, Clone)]
pub struct TrainFinalNetArgs {
    #[arg(long, value_delimiter = ',', value_parser = crate::utils::parse_run_config_ids)]
    pub train_label_nets_run_config_ids: Vec<RunConfigId>,

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[command(flatten)]
    pub opt_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt: String, //TrainLabelNetRunConfig

    #[arg(long)]
    pub selected_motif_nets_limit: Option<usize>,
}

impl TrainFinalNetArgs {
    pub fn resource_config(&self) -> NetResources<Logits> {
        NetResources::new(
            Logits::new(self.train_resource_group.clone()),
            Logits::new(self.valid_resource_group.clone()),
        )
    }
}

#[allow(non_snake_case, unused_assignments, unused_variables)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainFinalNetRunConfig {
    #[in_map]
    pub id: Option<RunConfigId>,

    #[in_map]
    #[run_config]
    pub common_training_args: CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    pub net: FinalNetArgs,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub struct FinalNetArgs {
    #[in_map]
    #[run_config]
    pub num_layers: u32,
}

impl AsRef<CommonTrainingRunConfig> for TrainFinalNetRunConfig {
    fn as_ref(&self) -> &CommonTrainingRunConfig {
        &self.common_training_args
    }
}

impl FinalNetArgs {
    #[cfg(test)]
    pub fn new(num_layers: u32) -> Self {
        Self { num_layers }
    }
}

impl TrainFinalNetRunConfig {
    pub fn new(
        id: Option<RunConfigId>,
        common_training_args: CommonTrainingRunConfig,
        net: FinalNetArgs,
    ) -> Self {
        Self {
            id,
            common_training_args,
            net,
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct InferArgs {
    #[arg(long, default_value_t = 0)]
    pub final_run_config_id: RunConfigId,

    #[arg(long, default_value_t = 1000)]
    pub motif_slices_step: usize,

    #[arg(long)]
    pub input_dir: String,

    #[arg(long, default_value = "submission.csv")]
    pub output_dir: String,

    #[arg(long)]
    pub limit:Option<usize>,

    #[arg(long)]
    pub models_dir: Option<String>,
}

impl InferArgs {
    pub fn input_data(&self) -> &str {
        &self.input_dir
    }
    pub fn output(&self) -> &str {
        &self.output_dir
    }

    pub fn new(
        final_run_config_id: RunConfigId,
        input_data: String,
        output: String,
        input_slice_step: usize,
    ) -> Self {
        Self {
            final_run_config_id,
            motif_slices_step: input_slice_step,
            input_dir: input_data,
            output_dir: output,
            limit: None,
            models_dir: None,
        }
    }
}

pub fn is_nonzero(t_1: &Tensor) -> Result<bool, ThoError> {
    Ok(t_1
        .flatten(0, (t_1.size().len() - 1) as i64)
        .iter::<f64>()?
        .find(|v| v > &0.)
        .is_some())
}

#[allow(non_snake_case, unused_assignments, unused_variables, non_camel_case_types)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LabelNetTypeArgs {
    ConvNet(ConvNetArgs),

    LinNet(LinNetArgs),

    SelfAttentionNet(SelfAttentionNetArgs)
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
    #[in_map]
    pub _dummy: u32,
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
pub struct LinNetArgs {
    #[in_map]
    #[run_config]
    pub num_layers: u32,

    #[in_map]
    #[run_config]
    pub features: u32,
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



impl LinNetArgs {
    pub fn new(num_layers: u32, slices_len: u32) -> Self {
        Self {
            num_layers,
            features: slices_len,
        }
    }
}

impl SelfAttentionNetArgs {
    pub fn new(d_model: u32, n_heads: u32, ff_mult: u32, dropout: f64) -> Self {
        Self { head_size: d_model, n_heads, ff_mult, dropout }
    }
}

///sums and multiplies inputs by k
pub fn dummy_label_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
    num_features: u32,
    k: f64,
    device: Device,
    descr: &str,
) -> Result<Box<dyn ThoModule<DI, DO>>, ThoError> {
    let vs = VarStore::new(device);

    SeqBuilder::new(num_features, None)?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!(
                        "test_label_net: merged final in flatten t.size()={:?}",
                        t.size()
                    );

                let t_0 = t.sum_dim_intlist(&[1, 2][..], false, None);
                debug!(
                        "test_label_net: merged final out flatten t_0.size()={:?}",
                        t_0.size()
                    );

                Ok(t_0)
            })),
            None,
        )?
        .linear_s_with_conf(
            "todo",
            1,
            LinearConfig {
                ws_init: init::Init::Const(k), //conflicts with ctor's init
                bs_init: None,
                bias: false,
            },
        )?
        .end_seq()?
        .build_tch_module(&vs.root())
}

pub fn motifs_to_specs(x_train: &Tensor, desc: &str/*, path_buf: &PathBuf*/) -> Tensor {
    let motifs = x_train.unbind(0, );
    let mut specs = vec![];
    // let path = path_buf.join(desc);
    // fs::create_dir_all(&path).unwrap();
    for (i, motif) in motifs.iter().enumerate() {
        let motif = motif.unsqueeze(0);
        let spec_i = motif.stft(
            16000,
            4000,
            16000,
            None::<Tensor>,
            false,
            true,
            true,
            true
        ).squeeze_dim(0).abs(); //.transpose(0, 1);

        debug!("heavy_test_train_cosine_motif_node_2d: motif={motif:?} spec_i={spec_i:?}");
        // fs::create_dir_all(format!("/tmp/plots/spectrogram_{}")


        //
        // plot_spectrogram(
        //     &spec_i,
        //     path.join(format!("{i}_.png")),
        //     1200,
        //     600,
        // ).unwrap();

        specs.push(spec_i);
    }

    let specs_x = Tensor::stack(&specs, 0);
    specs_x
}


#[cfg(test)]
#[allow(unused_imports, unused_variables, unused_mut, dead_code)]
mod train_tests {
    use crate::batches::BatchLengths;
    use crate::builder::{Node, SeqBuilder};
    use crate::dims::{
        AnyDims, Batch, Dims, LabelProbability, MotifProbability, MotifsNetsSelectionDims, Slice,
        ThoBatch,
    };
    use crate::my_tch::{InitDummy, MotifInitMode, ThoModule};
    use crate::tho_tensor::ThoTensor;
    use crate::train_nets::{lin_label_node, dummy_label_net, lin_motif_node, LinLenNetParams, CommonTrainingRunConfig, CommonTrainingRunConfigRangeOrValue, FinalNetArgs, FinalNetParams, LinNetArgs, ConvNetParams, LabelNetTypeArgs, LabelNetTypeArgsRangeOrValue, LabelNetTypeParams, LoadMotifNetOutput, ConvNetArgs, NetsPadding, OptConfig, PlotVecf32, SelfAttentionNetParams, TrainFinalNetRunConfig, TrainLabelNetRunConfig, TrainLabelNetRunConfigRangeOrValue, conv_label_node, self_attention_label_node, ConvNetJoint, TrainLabelNetRunConfigResultParams, TrainMotifNetRunConfig, cosine_motif_node, MotifCosineNetParams, cosine_2d_motif_node, MotifCosine2dNetParams, motifs_to_specs};
    use crate::train_nets::{
        InferArgs, LabelNetParams, MotifLinNetParams, MotifNetParams, ThoModuleWithStore,
    };
    use crate::transforms::filters::BandPassFilterParamsRangeOrValue;
    use crate::utils::{debug_plot_impl, from, mix_x_binary_with_labels, optimizer_mem_size, pad_stack_with_mask, plot_spectrogram, set_requires_grad_in_subpath, vs_mem_size, TrainListener};
    use crate::utils::{
        lin_t, pad_to, reset_print_options, t_0_0, tensor_mem_size,
    };
    use crate::ws::mock_ws_simp;
    use crate::ws::{FsWorkspace, Workspace, WsMockBuilder, NET_WEIGHTS_FILE};
    use crate::{train_nets, utils};
    use async_trait::async_trait;
    use futures::executor::block_on;
    use futures_util::FutureExt;
    use log::LevelFilter::{Debug, Info, Trace};
    use log::{debug, error, info, set_max_level, trace};
    use mockall::automock;
    use mockall_double::double;
    use parking_lot::{Mutex, RwLock};
    use rand::{random_range, Rng, RngExt, SeedableRng};
    use rayon::iter::IntoParallelRefMutIterator;
    use std::collections::HashMap;
    use std::ops::{Deref, RangeInclusive};
    use std::path::{Path, PathBuf};
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::{fs, mem};
    use rand::prelude::SliceRandom;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::nn::{init, AdamW, LinearConfig, Module, OptimizerConfig, VarStore};
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Bool, Float, Half};
    use tch::{nn, Device, Tensor};
    use tch::{IndexOp, Kind, Reduction};
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{OptRunConfigs, RunConfig, RunConfigId, RUN_ID};
    use thoosa_shared::perf_logging::set_panic_hook;
    use thoosa_shared::val::Val;
    use thoosa_shared::{batch_windows, cdt_for_dir, clc_num_batches, mono_key, pulse_in_rand, sin_v, sin_v_f, slices_len, CopyToDevice, ThoError};
    use thoosa_store::{mock_store_with_find_motifs_run_config, MockStore, SqlStore};
    use torch_sys::at_autocast_clear_cache;
    use thoosa_shared::traits::SimpleId;
    use crate::train_nets::ConvNetArgsFields::Joint;


    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_shims::inference_mode::InferenceModeGuard;
    use crate::tch_test_utils::motifs_tensor;

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }



    #[test]
    fn test_test_net() -> Result<(), ThoError> {
        let t_0 = lin_test_tensor(1000, 5);
        debug!("t_0.size={:?}", t_0.size());

        let t: ThoTensor<AnyDims, SimpleId> = t_0.unfold(1, 100, 10).into();

        debug!("t.size={:?}", t.size());

        //sums and multiplies by k
        let r_0: ThoTensor<AnyDims, SimpleId> =
            utils::test_net(100, 1., Cpu, "test_net")?.tho_forward(&t, None, true)?;
        debug!("r_0.size={:?}", r_0.size());

        assert_eq!(Tensor::from_slice(&[4950.]), r_0.as_ref().i((0, 0, ..)));
        assert_eq!(
            Tensor::from_slice(&[9900.]),
            utils::test_net::<AnyDims, AnyDims>(100, 2., Cpu, "test_net")?
                .tho_forward(&t, None, true)?
                .as_ref()
                .i((0, 0, ..))
        );

        Ok(())
    }
    
    //5 ch
    fn lin_test_tensor(input_n: i64, channels: u32) -> Tensor {
        Tensor::vstack(
            (1..=channels)
                .map(|c| {
                    Tensor::from_slice(
                        (0..input_n)
                            .map(|i| i as f32 * c as f32)
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )
                })
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    #[tokio::test]
    async fn test_save_load_lin_label_node() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_dir={test_dir:?}");
        let channels = 3;
        let max_motif_nodes = 19;
        let features = 23;

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            channels,
            1000,
            32000,
        )?));
        let params = LabelNetParams::new(
            max_motif_nodes,
            LabelNetTypeParams::LinNetParams(LinLenNetParams::new(features, 3)),
        );

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> =
            lin_label_node(&params)?.build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        let batch = 11i64;
        for features in [features, 123, 17] {
            let _ = net.tho_forward(
                &Tensor::rand(
                    [
                        batch,
                        channels as i64,
                        features as i64,
                    ],
                    FLOAT_CPU,
                )
                    .into(),
                None,
                false,
            )?;
        }


        let net_id = ws
            .write()
            .save_label_net(1, &"test_label", &vs_0, &params)?;

        debug!("net_id={net_id:?} vs_0.variables().len()={}",vs_0.variables().len());

        let model_base_path = ws.read().labels_nets_path().join(&net_id);
        let params_for_builder = params.clone();
        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load("".into(),
                                                                                           move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder)?.build_tch_module(path),
                                                                                           model_base_path.join(NET_WEIGHTS_FILE),
                                                                                           Cpu,
                                                                                           false,
        )?;

        assert!(vs_0.variables().len() > 0);
        assert_eq!(vs_0.variables(), loaded_module.var_store().variables());

        Ok(())
    }

    #[tokio::test]
    async fn test_save_load_conv_label_node() -> Result<(), ThoError> {
        _test_save_load_conv_label_node(
            3, 19, &[2000, 12030, 17000], 11i64,
            ConvNetParams::new(3, 64, 10, 10, 256, ConvNetJoint::MinMax)
        ).await?;

        _test_save_load_conv_label_node(
            3, 19, &[2000, 12030, 17000], 11i64,
            ConvNetParams::new(3, 64, 10, 10, 256, ConvNetJoint::AttnPool1d)
        ).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_save_load_conv_label_node_1() -> Result<(), ThoError> {
        _test_save_load_conv_label_node(
            360, 1541, &[1541], 101i64,
            ConvNetParams::new(7, 64, 100, 10, 256, ConvNetJoint::MinMax)
        ).await?;

        _test_save_load_conv_label_node(
            360, 1541, &[1541], 101i64,
            ConvNetParams::new(7, 64, 100, 10, 256, ConvNetJoint::AttnPool1d)
        ).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_save_load_self_attention_label_node() -> Result<(), ThoError> {
        _test_save_load_self_attention_label_node(
            19,
            &[128, 256, 384],
            11i64,
            SelfAttentionNetParams::new(64, 4, 4, 0.1),
            false,
        ).await?;

        _test_save_load_self_attention_label_node(
            19,
            &[128, 256, 384],
            11i64,
            SelfAttentionNetParams::new(64, 4, 4, 0.1),
            true,
        ).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_test_train_conv_label_node() -> Result<(), ThoError> {
        let motif_nets= 100;
        let conv_net_params = ConvNetParams::new(8, 8, 10, 8, 8, ConvNetJoint::AttnPool1d);
        let params = LabelNetParams::new(
            motif_nets,
            LabelNetTypeParams::ConvNetParams(conv_net_params),
        );

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = conv_label_node(&params)?.build_tch_module(&vs_0.root())?;
        let mut opt = AdamW::default().amsgrad(true).build(&vs_0, 1e-3)?;

        let batch = 64usize;
        let (x_train, y_train, lengths_train) = gen_motifs_output(
            batch,
            motif_nets as usize,
            1200..=1400,
            LabelSignal::ByMotifs,
            0.01,
            Some(11),
        )?;
        let target_train = y_train.unsqueeze(1);
        let lengths_train = lengths_train.iter().map(|v| *v as i64).collect::<Vec<_>>();

        let (x_eval, y_eval, lengths_eval) = gen_motifs_output(
            batch,
            motif_nets as usize,
            1200..=1400,
            LabelSignal::ByMotifs,
            0.15,
            Some(12),
        )?;
        let target_eval = y_eval.unsqueeze(1);
        let lengths_eval = lengths_eval.iter().map(|v| *v as i64).collect::<Vec<_>>();

        let mut eval_lengths_before = BatchLengths::new(lengths_eval.clone());
        let logits_eval_before = net.tho_forward(&x_eval.as_ref().into(), Some(&mut eval_lengths_before), false)?;
        let eval_loss_before = logits_eval_before.as_ref().binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);

        let mut first_loss = None;
        let mut last_loss = 0.0;
        for _ in 0..50 {
            let mut batch_lengths = BatchLengths::new(lengths_train.clone());
            let logits = net.tho_forward(&x_train.as_ref().into(), Some(&mut batch_lengths), true)?;
            let loss = logits.as_ref().binary_cross_entropy_with_logits::<Tensor>(&target_train, None, None, Reduction::Mean);
            if first_loss.is_none() {
                first_loss = Some(loss.double_value(&[]));
            }
            last_loss = loss.double_value(&[]);
            opt.backward_step(&loss);
        }

        let mut lengths_eval = BatchLengths::new(lengths_eval.clone());
        let logits_eval = net.tho_forward(&x_eval.as_ref().into(), Some(&mut lengths_eval), false)?;
        let eval_loss = logits_eval.as_ref().binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);

        debug!("heavy_test_train_conv_label_node: first_loss={:?} last_loss={last_loss:.5} eval_loss_before={eval_loss_before:.5} eval_loss_after={eval_loss:.5}",first_loss);

        assert!(first_loss.unwrap_or(last_loss) > last_loss, "expected train loss to decrease, first_loss={first_loss:?}, last_loss={last_loss}");
        assert!(eval_loss < eval_loss_before, "expected eval loss to improve, before={eval_loss_before}, after={eval_loss}");

        Ok(())
    }

    fn gen_motifs_batch(batch: usize, motif_len: usize, motifs: &Tensor) -> Result<(Tensor, Option<BatchLengths>, Tensor), ThoError>{
        assert_eq!(batch%2, 0);

        debug_plot_impl(motifs, "motifs", 0)?;
        let random = motifs_tensor(motif_len, batch/2, MotifKind::Triangle, 1.0, 2.0);
        debug_plot_impl(&random, "random", 0)?;

        mix_x_binary_with_labels(motifs, None, &random, None)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_test_train_cosine_motif_node() -> Result<(), ThoError> {
        let motif_width= 100usize;
        let batch = 64usize;

        let conv_net_params = ConvNetParams::new(8, 8, 10, 8, 8, ConvNetJoint::AttnPool1d);



        let mut vs_0 = VarStore::new(Cpu);
        let motifs = motifs_tensor(motif_width, batch / 2, MotifKind::ChirpUp, 1.0, 2.0);

        let (x_train, lengths_train,  y_train) = gen_motifs_batch(batch, motif_width, &motifs)?;
        debug_plot_impl(&x_train, "x_train", 0)?;

        let (x_eval,lengths_eval, y_eval, ) = gen_motifs_batch(batch, motif_width, &motifs_tensor(motif_width, batch / 2, MotifKind::ChirpUp, 1.5, 3.0))?;
        debug_plot_impl(&x_eval, "x_eval", 0)?;

        for init_mode in &crate::my_tch::MotifInitMode::list(0.4) {
            let net = cosine_motif_node(&MotifCosineNetParams::new(motif_width as u32, 10.0, 1.0, init_mode.clone()), Some(Arc::new(ThoTensor::new("motifs".into(), motifs.shallow_clone())?)))?.build_tch_module(&vs_0.root())?;

            let result = run_cosine_motif_node(
                &mut vs_0, &x_train, &y_train, &x_eval, &y_eval,
                &net
            );
            debug!("heavy_test_train_cosine_motif_node: init_mode={init_mode:?} result={result:?}");
        }

        Ok(())
    }

    fn run_cosine_motif_node(vs_0: &mut VarStore, x_train: &Tensor, y_train: &Tensor, x_eval: &Tensor, y_eval: &Tensor, net: &Box<dyn ThoModule<AnyDims, AnyDims>>) -> Result<u32, ThoError> {
        let mut opt = AdamW::default().amsgrad(true).build(&vs_0, 1e-3)?;

        let logits_eval_before = net.tho_forward(&x_eval.as_ref().into(), None, false)?;
        let eval_loss_before = logits_eval_before.as_ref().binary_cross_entropy_with_logits::<Tensor>(&y_eval, None, None, Reduction::Mean).double_value(&[]);
        assert!(eval_loss_before > 0.5);
        let mut first_loss = None;
        let mut last_loss = 0.0;
        for i in 0..50u32 {
            let logits = net.tho_forward(&x_train.as_ref().into(), None, true)?;

            let loss = logits.as_ref().binary_cross_entropy_with_logits::<Tensor>(&y_train, None, None, Reduction::Mean);
            if first_loss.is_none() {
                first_loss = Some(loss.double_value(&[]));
            }
            last_loss = loss.double_value(&[]);

            let logits_eval = net.tho_forward(&x_eval.as_ref().into(), None, false)?;
            let eval_loss = logits_eval.as_ref().binary_cross_entropy_with_logits::<Tensor>(&y_eval, None, None, Reduction::Mean).double_value(&[]);
            if eval_loss < 0.1 {
                debug!("heavy_test_train_cosine_motif_node: first_loss={:?} last_loss={last_loss:.5} eval_loss_before={eval_loss_before:.5} eval_loss_after={eval_loss:.5}",first_loss);
                return Ok(i);
            }

            debug!("heavy_test_train_cosine_motif_node: x_train={x_train:?} y_train={y_train:?} logits={logits:?} logits.isnan={} last_loss={last_loss}", logits.as_ref().isnan().any());
            trace!("heavy_test_train_cosine_motif_node: x_train={x_train} y_train={y_train}");

            opt.backward_step(&loss);
        }
        Err(ThoError::new("failed"))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_test_train_cosine_motif_node_2d() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp/plots/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;

        let motif_width= 160_000usize;
        let batch = 64usize;

        let conv_net_params = ConvNetParams::new(8, 8, 10, 8, 8, ConvNetJoint::AttnPool1d);



        let mut vs_0 = VarStore::new(Cpu);
        // let motifs = motifs_tensor(motif_width, batch / 2, MotifKind::ChirpUp, 1.0, 2.0);
        let motifs = Tensor::stack((0..batch/2).map(|i| {
            Tensor::from_slice(sin_v_f(160_000, 8000.0, 32_000.0).as_slice())
        }).collect::<Vec<_>>().as_slice(), 0);

        let kernel = motifs_to_specs(&motifs, "kernel", &test_dir);

        let (x_train, lengths_train,  y_train) = gen_motifs_batch(batch, motif_width, &motifs)?;
        debug_plot_impl(&x_train, "x_train", 0)?;

        let specs_x = motifs_to_specs(&x_train, "specs_x", &test_dir);

        debug!("heavy_test_train_cosine_motif_node_2d: all_spects_t={specs_x:?}");


        // let (x_eval,lengths_eval, y_eval, ) = gen_motifs_batch(batch, motif_width, &motifs_tensor(motif_width, batch / 2, MotifKind::ChirpUp, 1.5, 3.0))?;
        // debug_plot_impl(&x_eval, "x_eval", 0)?;

        for init_mode in [MotifInitMode::Mean(InitDummy{})] {// &crate::my_tch::MotifInitMode::list(0.4) {
            let net = cosine_2d_motif_node(&MotifCosine2dNetParams::new(motif_width as u32, 10, 10.0, 1.0, init_mode.clone()), Some(Arc::new(ThoTensor::new("motifs".into(), kernel.shallow_clone())?)))?.build_tch_module(&vs_0.root())?;

            let result = run_cosine_motif_node(
                // &mut vs_0, &x_train, &y_train, &x_eval, &y_eval,
                &mut vs_0, &specs_x, &y_train, &specs_x, &y_train,
                &net
            );
            debug!("heavy_test_train_cosine_motif_node: init_mode={init_mode:?} result={result:?}");
        }
        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_self_attention_label_node() -> Result<(), ThoError> {
        tch::manual_seed(42);
        let motif_nets = 2;



        let batch = 10usize;
        let eval_x_y = gen_motifs_output(
            batch,
            motif_nets as usize,
            100..=100,
            LabelSignal::ByOrder,
            0.1,
            None,
        )?;
        // crate::utils::debug_plot_impl(&eval_x_y.0.i((.., 0, ..)), &format!("eval_random_motif_net_0_output_label_{:?}",  eval_x_y.1), 0)?;
        // crate::utils::debug_plot_impl(&eval_x_y.0.i((.., 1, ..)), &format!("eval_random_motif_net_1_output_label_{:?}",  eval_x_y.1), 0)?;

        let (eval_loss_before, first_loss, last_loss, eval_loss) = train_self_attention(
            motif_nets,
            SelfAttentionNetParams::new(20, 1, 1, 0.1),
            batch,
            eval_x_y,
            |step| {
                let v = gen_motifs_output(
                    batch,
                    motif_nets as usize,
                    100..=100,
                    LabelSignal::ByOrder,
                    0.1,
                    None,
                )?;
                debug!("y {}", v.1);
                // crate::utils::debug_plot_impl(&v.0.i((.., 0, ..)), &format!("{step}_random_motif_net_0_output_label_{:?}",  v.1), 0)?;
                // crate::utils::debug_plot_impl(&v.0.i((.., 1, ..)), &format!("{step}_random_motif_net_1_output_label_{:?}",  v.1), 0)?;
                Ok(v)
            },
            100,
        )?;

        debug!(
            "heavy_test_train_self_attention_label_node: first_loss={:?} last_loss={last_loss:.5} eval_loss_before={eval_loss_before:.5} eval_loss_after={eval_loss:.5}",
            first_loss,
        );



        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_test_train_self_attention_label_node() -> Result<(), ThoError> {
        tch::manual_seed(42);
        let motif_nets = 30;

        let (eval_loss_before, first_loss, last_loss, eval_loss) = train_self_attention(
            motif_nets,
            SelfAttentionNetParams::new(64, 4, 2, 0.1),
            32usize,
            gen_motifs_output(
                32,
                motif_nets as usize,
                220..=320,
                LabelSignal::ByOrder,
                0.35,
                Some(22),
            )?, |step| gen_motifs_output(
                32,
                motif_nets as usize,
                220..=320,
                LabelSignal::ByOrder,
                0.35,
                Some(1_000 + step as u64),
            ), 120,
        )?;

        debug!(
            "heavy_test_train_self_attention_label_node: first_loss={:?} last_loss={last_loss:.5} eval_loss_before={eval_loss_before:.5} eval_loss_after={eval_loss:.5}",
            first_loss,
        );

        assert!(first_loss.unwrap_or(last_loss) > last_loss, "expected train loss to decrease, first_loss={first_loss:?}, last_loss={last_loss}");
        assert!(eval_loss < eval_loss_before, "expected eval loss to improve, before={eval_loss_before}, after={eval_loss}");

        Ok(())
    }

    fn train_self_attention<XYF>(motif_nets: u32, attention_net_params: SelfAttentionNetParams, batch: usize, x_y_lengths_eval: (Tensor, Tensor, Vec<usize>), x_y_lengths_train: XYF, steps: i32) -> Result<(f64, Option<f64>, f64, f64), ThoError>
    where
        XYF: Fn(i32)->Result<(Tensor, Tensor, Vec<usize>), ThoError>
    {

        let params = LabelNetParams::new(
            motif_nets,
            LabelNetTypeParams::SelfAttentionNetParams(attention_net_params),
        );

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = self_attention_label_node(&params)?.build_tch_module(&vs_0.root())?;
        let mut opt = AdamW::default().amsgrad(true).build(&vs_0, 1e-2)?;


        let (x_eval, y_eval, lengths_eval) = x_y_lengths_eval;
        let target_eval = y_eval.unsqueeze(1);
        let lengths_eval = lengths_eval.iter().map(|v| *v as i64).collect::<Vec<_>>();

        let mut eval_lengths_before = BatchLengths::new(lengths_eval.clone());
        let logits_eval_before = net.tho_forward(&x_eval.as_ref().into(), Some(&mut eval_lengths_before), false)?;
        let eval_loss_before = logits_eval_before.as_ref().binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);

        let mut first_loss = None;
        let mut last_loss = 0.0;
        for step in 0..steps {
            let (x_train, y_train, lengths_train) = x_y_lengths_train(step)?;
            let target_train = y_train.unsqueeze(1);
            let mut batch_lengths = BatchLengths::new(lengths_train.iter().map(|v| *v as i64).collect::<Vec<_>>());
            let logits = net.tho_forward(&x_train.as_ref().into(), Some(&mut batch_lengths), true)?;
            let loss = logits.as_ref().binary_cross_entropy_with_logits::<Tensor>(&target_train, None, None, Reduction::Mean);
            if first_loss.is_none() {
                first_loss = Some(loss.double_value(&[]));
            }
            last_loss = loss.double_value(&[]);
            opt.backward_step(&loss);
        }

        let mut lengths_eval = BatchLengths::new(lengths_eval.clone());
        let logits_eval = net.tho_forward(&x_eval.as_ref().into(), Some(&mut lengths_eval), false)?;
        let eval_loss = logits_eval.as_ref().binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);
        Ok((eval_loss_before, first_loss, last_loss, eval_loss))
    }

    async fn _test_save_load_conv_label_node(motif_nets: u32, max_motif_nodes: i32, all_slices: &[i32], batch: i64, conv_net_params: ConvNetParams) -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_dir={test_dir:?}");

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            motif_nets,
            1000,//not used
            32000,
        )?));
        let params = LabelNetParams::new(
          motif_nets,
          LabelNetTypeParams::ConvNetParams(conv_net_params),
        );

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = conv_label_node(&params)?.build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        for slices in all_slices {
            let mut lengths = BatchLengths::new(vec![*slices as i64; batch as usize]);
            let _ = net.tho_forward(
                &Tensor::rand(
                    [
                        batch,
                        motif_nets as i64,
                        *slices as i64,
                    ],
                    FLOAT_CPU,
                ).into(),

                Some(&mut lengths),
                false,
            )?;
        }

        let net_id = ws
            .write()
            .save_label_net(1, &"test_label", &vs_0, &params)?;

        debug!("net_id={net_id:?} vs_0.variables().len()={}",vs_0.variables().len());

        let model_base_path = ws.read().labels_nets_path().join(&net_id);
        let params_for_builder = params.clone();
        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load("".into(),
                                                                                           move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder)?.build_tch_module(path),
                                                                                           model_base_path.join(NET_WEIGHTS_FILE),
                                                                                           Cpu,
                                                                                           false,
        )?;

        assert!(vs_0.variables().len() > 0);
        assert_eq!(vs_0.variables(), loaded_module.var_store().variables());


        let cuda_net = loaded_module.copy_to_device(1).await?;
        for slices in all_slices {
            let mut lengths = BatchLengths::new(vec![*slices as i64; batch as usize]);
            let _ = cuda_net.tho_forward(
                &Tensor::rand(
                    [
                        batch,
                        motif_nets as i64,
                        *slices as i64,
                    ],
                    FLOAT_CUDA,
                ).into(),

                Some(&mut lengths),
                false,
            )?;
        }


        Ok(())
    }

    async fn _test_save_load_self_attention_label_node(motif_nets: u32, all_slices: &[i32], batch: i64, self_attention_net_params: SelfAttentionNetParams, half: bool) -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_dir={test_dir:?}");

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            motif_nets,
            1000,
            32000,
        )?));
        let params = LabelNetParams::new(
            motif_nets,
            LabelNetTypeParams::SelfAttentionNetParams(self_attention_net_params),
        );

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = self_attention_label_node(&params)?.build_tch_module(&vs_0.root())?;
        if half {
            vs_0.half();
        }
        vs_0.freeze();

        let module_with_store = ThoModuleWithStore::with_factory(mono_key!(0, "test"), net, vs_0, None);

        for slices in all_slices {
            let mut lengths = BatchLengths::new(vec![*slices as i64; batch as usize]);
            let _ = module_with_store.tho_forward(
                &Tensor::rand(
                    [
                        batch,
                        motif_nets as i64,
                        *slices as i64,
                    ],
                    FLOAT_CPU,
                ).into(),
                Some(&mut lengths),
                false,
            )?;
        }

        let net_id = ws
            .write()
            .save_label_net(1, &"test_label", module_with_store.var_store(), &params)?;

        debug!("net_id={net_id:?} vs_0.variables().len()={}",module_with_store.var_store().variables().len());

        let model_base_path = ws.read().labels_nets_path().join(&net_id);
        let params_for_builder = params.clone();
        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load("".into(),
                                                                                           move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder)?.build_tch_module(path),
                                                                                           model_base_path.join(NET_WEIGHTS_FILE),
                                                                                           Cpu,
                                                                                           false,
        )?;

        assert!(module_with_store.var_store().variables().len() > 0);
        assert_eq!(module_with_store.var_store().variables(), loaded_module.var_store().variables());

        if matches!(Device::cuda_if_available(), Cuda(_)) {
            let cuda_net = loaded_module.copy_to_device(1).await?;
            for slices in all_slices {
                let mut lengths = BatchLengths::new(vec![*slices as i64; batch as usize]);
                let _ = cuda_net.tho_forward(
                    &Tensor::rand(
                        [
                            batch,
                            motif_nets as i64,
                            *slices as i64,
                        ],
                        FLOAT_CUDA,
                    ).into(),
                    Some(&mut lengths),
                    false,
                )?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_cnn_label_node_with_mask() -> Result<(), ThoError>{
        reset_print_options();
        let motif_nets = 3;
        let params = LabelNetParams::new(
            motif_nets,
            LabelNetTypeParams::ConvNetParams(ConvNetParams::new(3, 4, 3, 10, 256, ConvNetJoint::MinMax)),
        );

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = conv_label_node(&params)?.build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        let max_len = 32;
        let batch_vec = std::iter::once(Tensor::rand(&[motif_nets as i64, max_len], FLOAT_CPU))
            .chain((1..4).map(|i|Tensor::rand(&[motif_nets as i64, random_range(max_len/2..max_len)], FLOAT_CPU)))
            .chain(std::iter::once(Tensor::rand(&[motif_nets as i64, max_len/2], FLOAT_CPU)))
            .collect::<Vec<_>>();

        let batch_max_len = batch_vec.iter().map(|t|t.size()[1]).max().unwrap();
        let batch_refs_vec = batch_vec.iter().collect::<Vec<_>>();

        debug!("batch_refs_vec={batch_refs_vec:?}");

        let (batch, lengths) = pad_stack_with_mask(batch_refs_vec.as_slice(), motif_nets as i64, batch_max_len)?;
        let mut lengths = BatchLengths::new(lengths);

        debug!("batch=\n{batch} lengths={lengths:?}");
        let _ = net.tho_forward(
            &batch.into(),

            Some(&mut lengths),
            false,
        )?;

        Ok(())
    }

    fn dummy_0(data: &mut [f64; 11]) -> (Tensor, f32) {
        (Tensor::from_slice(data).to_kind(Float), 0.0)
    }

    fn dummy_1(data: &mut [f64; 11]) -> (Tensor, f32) {
        (Tensor::from_slice(data).to_kind(Float) + 1.0, 1.0)
    }

    fn asc_t<const S: usize>(mut data: [f32; S]) -> Tensor {
        Tensor::from_slice(
            (0..S)
                .map(|i| data[i] + i as f32 * 2.)
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .to_kind(Float)
    }

    fn desc_t<const S: usize>(mut data: [f32; S]) -> Tensor {
        Tensor::from_slice(
            (0..S)
                .rev()
                .map(|i| data[i] + i as f32 * 2.)
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .to_kind(Float)
    }



    fn dummy_labels_net_input<'a>(
        labels: &[bool],
        channels: usize,
        max_motif_nets: usize,
        slices_len: usize,
        tr: Option<Box<dyn Fn(&Tensor) -> Tensor + 'a>>,
    ) -> Tensor {
        //batch size x channels x max motif nets x slices_len
        let mut labels_iter = labels.iter();
        let t = Tensor::stack(
            labels
                .iter()
                .map(|label| {
                    Tensor::stack(
                        (0..channels)
                            .map(|mb| {
                                Tensor::stack(
                                    (0..max_motif_nets)
                                        .map(|mmn| {
                                            if *label {
                                                motifs_1(slices_len)
                                            } else {
                                                motifs_0(slices_len)
                                            }
                                        })
                                        .collect::<Vec<_>>()
                                        .as_slice(),
                                    0,
                                )
                            })
                            .collect::<Vec<_>>()
                            .as_slice(),
                        0,
                    )
                })
                .map(|t| {
                    debug!("dummy_labels_net_input: tr.size={:?}", t.size());
                    tr.as_ref().map(|tr| tr(&t)).unwrap_or(t)
                })
                .collect::<Vec<_>>()
                .as_slice(),
            0,
        );
        debug!("dummy_labels_net_input: t.size={:?}", t.size());

        t
    }

    fn dummy_motif_net_input<'a>(
        labels: &[bool],
        channels: usize,
        n: usize,
        tr: Option<Box<dyn Fn(&Tensor) -> Tensor + 'a>>,
    ) -> Tensor {
        //batch size x channels x max motif nets x slices_len
        let batch = Tensor::stack(
            labels
                .iter()
                .map(|label| {
                    Tensor::stack(
                        (0..channels)
                            .map(|channel| if *label { motifs_1(n) } else { motifs_0(n) })
                            .collect::<Vec<_>>()
                            .as_slice(),
                        0,
                    )
                })
                .map(|t| {
                    debug!("dummy_motif_net_input: tr.size={:?}", t.size());
                    tr.as_ref().map(|tr| tr(&t)).unwrap_or(t)
                })
                .collect::<Vec<_>>()
                .as_slice(),
            0,
        );
        debug!("dummy_motif_net_input: t.size={:?}", batch.size());

        batch
    }

    fn motifs_1(n: usize) -> Tensor {
        // Tensor::rand(n as i64, FLOAT_CPU).multiply_scalar(0.1).f_add_scalar(0.8).unwrap()
        Tensor::ones(n as i64, FLOAT_CPU)
    }

    fn motifs_0(n: usize) -> Tensor {
        // Tensor::rand(n as i64, FLOAT_CPU).multiply_scalar(0.01)
        Tensor::zeros(n as i64, FLOAT_CPU)
    }

    const fn build_labels<const BATCH_SIZE: usize>() -> [bool; BATCH_SIZE] {
        let mut arr = [false; BATCH_SIZE];
        let mut i = 0;
        while i < BATCH_SIZE {
            arr[i] = i % 2 == 0;
            i += 1;
        }
        arr
    }


    fn dummy_final_net_input<const LABELS_SIZE: usize, D: Dims>(
        batch_size: usize,
    ) -> (ThoTensor<D, SimpleId>, Tensor) {
        let mut rng = rand::rng();
        let data = (0..batch_size)
            .map(|label| {
                let label = rng.random_range(0..LABELS_SIZE);

                let x = random_tensor_with_dominant(LABELS_SIZE as i64, label as i64, 0.99, Cpu);
                let mut y = [0.; LABELS_SIZE];
                y[label] = 1.0;
                let y = Tensor::from_slice(y.as_slice());

                (x, y)
            })
            .collect::<Vec<_>>();

        (
            Tensor::stack(
                data.iter()
                    .map(|d| d.0.shallow_clone())
                    .collect::<Vec<_>>()
                    .as_slice(),
                0,
            )
            .into(),
            Tensor::stack(
                data.iter()
                    .map(|d| d.1.shallow_clone())
                    .collect::<Vec<_>>()
                    .as_slice(),
                0,
            ),
        )
    }

    /// Generate an n-dimensional tensor whose elements sum to 1,
    /// with index m taking `fraction` of the total (e.g., 0.8),
    /// and the remaining (1 - fraction) spread among the other n-1 entries.
    ///
    /// # Arguments
    /// * `n`         - Size of the tensor (number of elements).
    /// * `m`         - The index that will hold the dominant fraction.
    /// * `fraction`  - The fraction of the total sum to assign to the m-th element (e.g., 0.8).
    /// * `device`    - tch::Device (e.g., Device::Cpu).
    ///
    /// # Panics
    /// If `m >= n` or `fraction <= 0.0` or `fraction >= 1.0`.
    pub fn random_tensor_with_dominant(n: i64, m: i64, fraction: f64, device: Device) -> Tensor {
        assert!(m < n, "m must be < n");
        assert!(fraction > 0.0 && fraction < 1.0, "fraction must be in (0,1)");

        let remainder = 1.0 - fraction;

        // Generate n-1 random values in [0,1]
        let other_values = Tensor::rand(&[n - 1], (Kind::Float, device));
        // Normalize them so they sum up to `remainder`
        let sum_others = other_values.sum(Kind::Float);
        let scaled_others = &other_values / &sum_others * remainder;

        // Create an output tensor of zeros
        let mut out = Tensor::zeros(&[n], (Kind::Float, device));

        // Fill the m-th entry with the dominant fraction
        let _ = out.get(m).fill_(fraction);

        // Prepare indices for all entries except m
        let indices: Vec<i64> = (0..n).filter(|&idx| idx != m).collect();

        // Put the scaled_others values into out at positions other than m
        let _ = out.index_put_(
            &[Some(Tensor::from_slice(indices.as_slice()))],
            &scaled_others,
            false,
        );

        out
    }

    //6 corrs 4 motif models
    //TODO remove?
    fn save_test_models(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        base_run_config_id: RunConfigId,
        channels: u32,
        max_motif_batches: u32,
        max_batch_size: u32,
        motif_node_params_run_config_id: RunConfigId,
        window: u32,
    ) -> Result<(), ThoError> {
        // fs::create_dir_all(models_path)?;

        // fs::write(models_path.join("motif_model.json"), serde_json5::to_string(&MotifNodeParams::new(window))?)?;

        save_test_label_model(
            &ws,
            base_run_config_id + 1,
            "seizure",
            3,
            4,
            max_motif_batches,
            max_batch_size,
        )?;
        // fs::write(models_path.join("motif_model.json"), serde_json5::to_string(&MotifNodeParams::new(window))?)?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "0", 1, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "0", 2, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "0", 3, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "1", 11, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_config_id)?;

        save_test_label_model(
            ws,
            base_run_config_id + 1,
            "lpd",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_config_id)?;

        save_test_label_model(
            ws,
            base_run_config_id + 1,
            "gpd",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_config_id)?;

        save_test_label_model(
            ws,
            base_run_config_id + 1,
            "lrda",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_config_id)?;

        save_test_label_model(
            ws,
            base_run_config_id + 1,
            "grda",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_config_id)?;

        save_test_label_model(
            ws,
            base_run_config_id + 1,
            "other",
            1,
            1,
            max_motif_batches,
            max_batch_size,
        )?;

        save_test_motif_model(ws, "0", 0, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "1", 10, window, motif_node_params_run_config_id)?;
        save_test_motif_model(ws, "2", 20, window, motif_node_params_run_config_id)?;
        Ok(())
    }

    pub fn save_test_tensor(input_path: &PathBuf, name: &str) {
        fs::create_dir_all(input_path).unwrap();
        fs::write(&input_path.join(name), "lalalal").unwrap();
    }

    fn save_test_final_model(models_path: &PathBuf, features: u32) -> Result<(), ThoError> {
        fs::create_dir_all(models_path.clone())?;

        let mut vs_0 = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> =
            train_nets::final_net(&FinalNetParams::new(features, 0, 1))?
                .build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!(
            "save_test_final_model: features={} vs_0.variables()={:?}",
            features,
            vs_0.variables()
        );
        vs_0.save(models_path.join("final.safetensors"))?;
        Ok(())
    }

    fn save_test_label_model(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        run_config_id: RunConfigId,
        label: &str,
        channels: u32,
        max_motif_nets: u32,
        max_motif_batches: u32,
        motif_batch_size: u32,
    ) -> Result<(), ThoError> {
        let label_node_params = LabelNetParams::new(
            max_motif_nets,
            train_nets::LabelNetTypeParams::LinNetParams(LinLenNetParams::new(0, 3)),
        ); //TODO slices_len, motif_layers

        let mut vs_0 = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> =
            LabelNetParams::label_net(&label_node_params)?.build_tch_module(&vs_0.root())?;
        vs_0.freeze();
        debug!(
            "save_test_corr_model: channels={}, max_motif_nets={}, max_motif_batches={}, motif_batch_size={} vs_0.variables()={:?}",
            channels,
            max_motif_nets,
            max_motif_batches,
            motif_batch_size,
            vs_0.variables()
        );

        ws.write()
            .save_label_net(run_config_id, &label, &vs_0, &label_node_params)?;
        Ok(())
    }

    fn save_test_motif_model(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        channel: &str,
        motif_id: u32,
        window: u32,
        motif_node_params_run_config_id: RunConfigId,
    ) -> Result<(), ThoError> {
        let mut vs_0 = VarStore::new(Cpu);
        let _test_motif_model: Box<dyn ThoModule<AnyDims, AnyDims>> =
            train_nets::lin_motif_node(&MotifLinNetParams::new(window, 1))?
                .build_tch_module(&vs_0.root())?;

        let test_motif_model: Box<dyn ThoModule<AnyDims, AnyDims>> = SeqBuilder::new(100, None)?
            .linear_s_with_conf(
                "todo",
                1,
                LinearConfig {
                    ws_init: init::Init::Const(1.0),
                    bs_init: None,
                    bias: false,
                },
            )?
            .end_seq()?
            .build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!(
            "save_test_motif_model: channel={channel} vs_0.variables()={:?}",
            vs_0.variables()
        );
        ws.write().save_motif_net(
            &channel,
            &motif_node_params_run_config_id,
            &motif_id,
            &vs_0,
            &MotifNetParams::new(100, 1, 1, 100, 32, 10),
        )?;
        Ok(())
    }

    #[test]
    fn test_is_nonzero() -> Result<(), ThoError> {
        assert!(super::is_nonzero(&Tensor::from_slice(&[0., 1., 0.]))?);
        assert!(super::is_nonzero(&Tensor::stack(
            &[
                Tensor::from_slice(&[0., 1., 0.]),
                Tensor::from_slice(&[0., 0., 0.])
            ],
            0
        ))?);
        assert!(!super::is_nonzero(&Tensor::from_slice(&[0., 0., 0.]))?);

        Ok(())
    }


    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum LabelSignal {
        /// Label is detectable by *which motifs* have peaks.
        ByMotifs,
        /// Label is detectable by the *temporal order* in which motifs peak.
        ByOrder,
    }

    #[inline]
    fn row_mut<'a>(
        data: &'a mut [f32],
        motif_nets: usize,
        b: usize,
        m: usize,
        l: usize,
    ) -> &'a mut [f32] {
        let off = (b * motif_nets + m) * l;
        &mut data[off..off + l]
    }


    /// Generate dummy outputs that look like "motif detector logits over time".
    ///
    /// Notes:
    /// - `L` is sampled once per call from `l_range` and used for the whole batch.
    /// - Logits are mostly negative (no-match) with a few positive peaks (matches),
    ///   which is typical for scanning.
    /// - `similarity` in [0,1] controls how hard the classification is.
    ///   At ~1.0, classes may become nearly indistinguishable (intentionally).
    /// Returns:
    /// - `x`: [batch, motif_nets, max_L] float logits (padded)
    /// - `y`: [batch] float labels in {0.0, 1.0}
    /// - `lengths`: Vec<usize> where lengths[b] == true L for sample b
    pub fn gen_motifs_output(
        batch: usize,
        motif_nets: usize,
        l_range: RangeInclusive<usize>,
        label_signal: LabelSignal,
        similarity: f32,
        seed: Option<u64>,
    ) -> Result<(Tensor, Tensor, Vec<usize>), ThoError> {
        assert_ne!(batch, 0, "batch must be > 0");
        assert_ne!(motif_nets, 0, "motif_nets must be > 0");

        let (l_lo, l_hi) = (*l_range.start(), *l_range.end());
        assert!(
            !(l_lo == 0 || l_hi == 0 || l_lo > l_hi),
            "l_range must be non-empty and >= 1"
        );

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let mut tr = rand::rng();     // rand 0.10
                StdRng::from_rng(&mut tr)     // infallible
            }
        };

        // --- sample per-sample lengths (try to make them distinct if possible) ---
        let mut lengths: Vec<usize> = Vec::with_capacity(batch);
        let range_len = l_hi - l_lo + 1;

        if range_len >= batch {
            let mut all: Vec<usize> = (l_lo..=l_hi).collect();
            all.shuffle(&mut rng);
            lengths.extend_from_slice(&all[..batch]);
        } else {
            for _ in 0..batch {
                lengths.push(rng.random_range(l_lo..=l_hi));
            }
        }

        let max_l = *lengths.iter().max().unwrap();

        // Difficulty knob.
        let sim = similarity.clamp(0.0, 1.0);

        // Background logits: mostly "no match" (negative).
        let base_mean = -2.0_f32;
        let noise_std = 1.0_f32;

        // Signal amplitude: big when sim=0, small when sim=1.
        let signal_amp = 8.0 * (1.0 - sim) + 1.5;

        // Leakage: chance that "other-class" motifs also get peaks (makes it harder).
        let leak_p = (0.60 * sim) as f64;

        // Peak shape.
        let sigma = 2.0_f32;
        let radius: isize = 6;

        let normal = Normal::new(base_mean as f64, noise_std as f64)
            .map_err(|e| ThoError::new(format!("Normal dist error: {e:?}")))?;

        // Allocate padded tensor storage: [batch * motif_nets * max_l]
        // Fill padding with a constant "no match" value (base_mean).
        let mut data = vec![base_mean; batch * motif_nets * max_l];

        // Add noise only in the valid prefix [0..lengths[b]] for each (b,m).
        for b in 0..batch {
            let l_b = lengths[b];
            for m in 0..motif_nets {
                let row = row_mut(&mut data, motif_nets, b, m, max_l);
                for v in &mut row[..l_b] {
                    *v = normal.sample(&mut rng) as f32;
                }
            }
        }

        let mut labels = vec![0.0_f32; batch];

        // Helper: add a gaussian bump (peak) into a row slice.
        let mut add_bump = |row: &mut [f32], center: usize, amp: f32| {
            let c = center as isize;
            for dt in -radius..=radius {
                let t = c + dt;
                if t < 0 || (t as usize) >= row.len() {
                    continue;
                }
                let x = (dt as f32) / sigma;
                let w = (-0.5 * x * x).exp(); // Gaussian
                row[t as usize] += amp * w;
            }
        };

        // Define motif groups used by ByMotifs.
        let k_group = (motif_nets / 4).max(1);
        let pos_group: Vec<usize> = (0..k_group).collect();
        let neg_group: Vec<usize> = (k_group..(2 * k_group).min(motif_nets)).collect();

        // Define motif group size used by ByOrder.
        let k_order = motif_nets.min(4).max(1);
        let canonical_order: Vec<usize> = (0..k_order).collect();

        for b in 0..batch {
            let l_b = lengths[b];
            let y = rng.random_bool(0.5);
            labels[b] = if y { 1.0 } else { 0.0 };

            match label_signal {
                LabelSignal::ByMotifs => {
                    let (primary, secondary) = if y {
                        (pos_group.as_slice(), neg_group.as_slice())
                    } else {
                        (neg_group.as_slice(), pos_group.as_slice())
                    };

                    // Clear peaks on primary motifs.
                    for &m in primary {
                        let center = rng.random_range(0..l_b);
                        let row = row_mut(&mut data, motif_nets, b, m, max_l);
                        add_bump(&mut row[..l_b], center, signal_amp);
                    }

                    // Leaked peaks on secondary motifs.
                    for &m in secondary {
                        if rng.random_bool(leak_p) {
                            let center = rng.random_range(0..l_b);
                            let row = row_mut(&mut data, motif_nets, b, m, max_l);
                            add_bump(&mut row[..l_b], center, signal_amp * 0.6);
                        }
                    }

                    // Spurious peaks everywhere (small).
                    let spur_p = (0.15 * sim) as f64;
                    for m in 0..motif_nets {
                        if rng.random_bool(spur_p) {
                            let center = rng.random_range(0..l_b);
                            let row = row_mut(&mut data, motif_nets, b, m, max_l);
                            add_bump(&mut row[..l_b], center, signal_amp * 0.25);
                        }
                    }
                }

                LabelSignal::ByOrder => {
                    // Choose time centers (in increasing time).
                    let mut centers = Vec::with_capacity(k_order);
                    for i in 0..k_order {
                        let base = ((i + 1) * l_b) / (k_order + 1);
                        let jitter = (l_b / (10 * (k_order + 1))).max(1);
                        let lo = base.saturating_sub(jitter);
                        let hi = (base + jitter).min(l_b - 1);
                        centers.push(rng.random_range(lo..=hi));
                    }
                    centers.sort_unstable();

                    // Determine motif order for this sample.
                    let mut order = canonical_order.clone();
                    if !y {
                        let swaps = ((1.0 - sim) * (k_order as f32)).ceil() as usize;
                        if swaps >= k_order && k_order >= 2 {
                            order.reverse();
                        } else {
                            for _ in 0..swaps {
                                let i = rng.random_range(0..k_order);
                                let j = rng.random_range(0..k_order);
                                order.swap(i, j);
                            }
                        }
                    }

                    // Add peaks: motif `order[i]` peaks at time `centers[i]`.
                    for i in 0..k_order {
                        let m = order[i];
                        let t = centers[i];
                        let row = row_mut(&mut data, motif_nets, b, m, max_l);
                        add_bump(&mut row[..l_b], t, signal_amp);
                    }

                    // Spurious peaks to make it harder.
                    let spur_p = (0.10 + 0.20 * sim) as f64;
                    for m in 0..motif_nets {
                        if rng.random_bool(spur_p) {
                            let center = rng.random_range(0..l_b);
                            let row = row_mut(&mut data, motif_nets, b, m, max_l);
                            add_bump(&mut row[..l_b], center, signal_amp * 0.20);
                        }
                    }
                }
            }
        }

        let x = Tensor::from_slice(&data).reshape([
            batch as i64,
            motif_nets as i64,
            max_l as i64,
        ]);

        let y = Tensor::from_slice(&labels).to_kind(Kind::Float);

        Ok((x, y, lengths))
    }

    #[test]
    fn shapes_ok() {
        let batch_len = 3;
        let (x, y, lengths) = gen_motifs_output(
            batch_len,
            5,
            70..=80,
            LabelSignal::ByMotifs,
            0.2,
            Some(123),
        ).unwrap();

        debug!("lengths={lengths:?}");
        debug!("x=\n{x} y=\n{y}");

        let xs = x.size();
        assert_eq!(xs.len(), 3);
        assert_eq!(xs[0], batch_len as i64);
        assert_eq!(xs[1], 5);
        assert!(xs[2] >= 50 && xs[2] <= 80);

        let ys = y.size();
        assert_eq!(ys, vec![batch_len as i64]);
    }

    #[tokio::test]
    async fn test_save_train_label_net_result_persists_trained_net_and_loss() -> Result<(), ThoError> {
        let store = Arc::new(SqlStore::in_mem().await?);
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws: Arc<RwLock<Box<dyn Workspace>>> = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 2, 1000, 32000)?));
        let train_motif_nets_run_config_id = store.save_run_config(None, TrainMotifNetRunConfig::type_name(), "source", HashMap::new()).await?;

        let mut result = crate::train_nets::TrainLabelNetRunConfigResult::new(
            // vec![source_run_config_id],
            "alpha".to_string(),
            0.125,
            LabelNetParams::new(4, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(16, 2))),
            TrainLabelNetRunConfig::new(None, CommonTrainingRunConfig::default(), LabelNetTypeArgs::LinNet(LinNetArgs::new(2, 16))),
            VarStore::new(Cpu),
        );

        let train_motif_nets_run_config_id_vec = vec![train_motif_nets_run_config_id];
        let motif_nets_ids= vec![];
        let train_label_nets_run_config_id = crate::runs::SavableResult::save(&mut result, &store, &ws, TrainLabelNetRunConfigResultParams{
            train_motif_nets_run_config_ids: &train_motif_nets_run_config_id_vec,
            motif_nets_ids: &motif_nets_ids,
        }).await?;
        let run_config_values = store.list_run_config_values(train_label_nets_run_config_id).await?;
        let trained_nets = store.list_trained_nets(vec![train_label_nets_run_config_id], None, None).await?;

        assert_eq!(vec![train_label_nets_run_config_id], store.list_output_run_config_ids(train_motif_nets_run_config_id).await?);
        assert_eq!(vec![(1, PathBuf::from(format!("alpha/{train_label_nets_run_config_id}")), 0.125)], trained_nets);

        Ok(())
    }

    #[tokio::test]
    async fn test_save_train_final_net_result_persists_loss_in_run_config_values() -> Result<(), ThoError> {
        let store = Arc::new(SqlStore::in_mem().await?);
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws: Arc<RwLock<Box<dyn Workspace>>> = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 2, 1000, 32000)?));
        let label_run_config_id_0 = store.save_run_config(None, TrainLabelNetRunConfig::type_name(), "label_0", HashMap::new()).await?;
        let label_run_config_id_1 = store.save_run_config(None, TrainLabelNetRunConfig::type_name(), "label_1", HashMap::new()).await?;
        let mut result = crate::train_nets::TrainFinalNetsRunConfigResult::new(
            Some(&vec![label_run_config_id_0, label_run_config_id_1]),
            0.125,
            VarStore::new(Cpu),
            FinalNetParams::new(4, 2, 3),
            TrainFinalNetRunConfig::new(None, CommonTrainingRunConfig::default(), FinalNetArgs::new(3)),
        );

        let run_config_id = crate::runs::SavableResult::save(&mut result, &store, &ws, ()).await?;
        let run_config_values = store.list_run_config_values(run_config_id).await?;
        let trained_nets = store.list_trained_nets(vec![run_config_id], None, None).await?;

        assert_eq!(vec![(1, PathBuf::from(format!("3")), 0.125)], trained_nets);
        Ok(())
    }


    #[test]
    fn test_inference_mode() -> Result<(), ThoError>{

        assert!(!Tensor::zeros(&[10, 100], FLOAT_CPU).is_inference());

        let _guard = InferenceModeGuard::enabled();

        assert!(Tensor::zeros(&[10, 100], FLOAT_CPU).is_inference());

        Ok(())
    }

}
