#![allow(non_snake_case, unused_assignments, unused_variables, non_camel_case_types)]

use anyhow::Context as _;
use clap::clap_derive::Args;
use thoosa_common::nets::batches::BatchLengths;
use thoosa_common::nets::builder::{Node, SeqBuilder};
use thoosa_common::nets::nodes::{BinaryHeadPool, InitMode, ThoModule};
use thoosa_common::resources::{Empty, Logits, NetResources};
use thoosa_common::utils::{mask_from_lengths, masked_max, masked_mean};

use log::debug;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::fs;
use std::path::PathBuf;
use tch::nn::{BatchNormConfig, ConvConfig, LinearConfig, VarStore, init};
use tch::{Device, Tensor};

use std::sync::Arc;
use thoosa_cpu_data_derive::CpuData;

use crate::dims::{MotifSlices, Spectorgram};
use thoosa_common::dims::Dims;
use thoosa_common::nets::common_nets;
use thoosa_common::nets::common_nets::{
    BinaryHeadPoolArgs, BinaryHeadPoolArgsBuilder, BinaryHeadPoolArgsFields, BinaryHeadPoolArgsRangeOrValue, CommonTrainingRunConfig, CommonTrainingRunConfigBuilder, CommonTrainingRunConfigFields, CommonTrainingRunConfigRangeOrValue, ConvNetArgs, ConvNetArgsBuilder, ConvNetArgsFields,
    ConvNetArgsRangeOrValue, ConvNetJoint, ConvNetParams, LinLenNetParams, OptConfig, SelfAttentionNetArgs, SelfAttentionNetArgsBuilder, SelfAttentionNetArgsFields, SelfAttentionNetArgsRangeOrValue, SelfAttentionNetParams, StftRunConfig, StftRunConfigBuilder, StftRunConfigFields,
    StftRunConfigRangeOrValue, WithCommonTrainingRunConfig,
};
use thoosa_common::nets::nodes::InitModeBuilder;
use thoosa_common::nets::nodes::InitModeFields;
use thoosa_common::nets::nodes::InitModeRangeOrValue;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::traits::{Data, ToJsonFile, WithId};

#[derive(Debug)]
pub struct TrainLabelNetRunConfigResultParams<'a> {
    pub train_motif_nets_run_config_ids: &'a Vec<RunConfigId>,
    pub motif_nets_ids: &'a Vec<u32>,
}

#[derive(Args, Debug, Clone)]
pub struct TrainMotifsNetsArgs {
    #[arg(long, value_delimiter = ',', value_parser = thoosa_common::utils::parse_run_config_ids)]
    pub select_motifs_run_config_ids: Option<Vec<RunConfigId>>,

    #[arg(long)]
    pub motif_id: Option<u32>,

    #[command(flatten)]
    pub opt_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt: String, //TrainMotifNetRunConfig

    #[arg(long, default_value_t = 10)]
    pub epoch_size: u32,

    #[arg(long)]
    pub motifs_limit: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, CpuData)]
#[cpu_data(shallow)]
pub enum MotifsBy {
    SelectMotifsRunIds {
        select_motifs_run_config_ids: Vec<RunConfigId>,
    },
    MotifId {
        motif_id: u32,
        select_motifs_run_config_ids: Option<Vec<RunConfigId>>, //if motif_id is in select_motifs_run_config_ids
    },
}

impl TrainMotifsNetsArgs {
    pub fn resource_config(&self) -> Empty {
        Empty {}
    }

    pub fn motifs_by(&self) -> anyhow::Result<MotifsBy> {
        match (self.motif_id, self.select_motifs_run_config_ids.clone()) {
            (Some(motif_id), select_motifs_run_config_ids) => Ok(MotifsBy::MotifId { motif_id, select_motifs_run_config_ids }),
            (None, Some(select_motifs_run_config_ids)) => Ok(MotifsBy::SelectMotifsRunIds { select_motifs_run_config_ids }),
            (None, None) => Err(anyhow::anyhow!("TrainMotifsNetsArgs: either --select-motifs-run-ids or --motif-id must be provided")),
        }
    }
}

#[cfg(test)]
mod train_motifs_nets_args_tests {
    use super::{MotifsBy, TrainMotifsNetsArgs};
    use thoosa_common::nets::common_nets::OptConfig;

    #[test]
    fn motifs_by_from_select_run_config_ids() {
        let args = TrainMotifsNetsArgs {
            select_motifs_run_config_ids: Some(vec![1, 2]),
            motif_id: None,
            opt_config: OptConfig::new(10, 1),
            run_config_cnt: "{}".to_string(),
            epoch_size: 10,
            motifs_limit: None,
        };

        assert_eq!(args.motifs_by().unwrap(), MotifsBy::SelectMotifsRunIds { select_motifs_run_config_ids: vec![1, 2] });
    }

    #[test]
    fn motifs_by_from_motif_id_and_optional_select_run_config_ids() {
        let args = TrainMotifsNetsArgs {
            select_motifs_run_config_ids: Some(vec![10, 11]),
            motif_id: Some(7),
            opt_config: OptConfig::new(10, 1),
            run_config_cnt: "{}".to_string(),
            epoch_size: 10,
            motifs_limit: None,
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
            motifs_limit: None,
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
    pub step: u32, //TODO renamed to Conv stride

    #[in_map]
    #[run_config]
    pub channels: u32,

    #[in_map]
    #[run_config]
    pub kernel: u32,

    #[in_map]
    #[run_config]
    pub head_pool: Option<BinaryHeadPoolArgs>,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct MotifLinNetArgs {
    #[in_map]
    #[run_config]
    pub num_layers: u32,

    #[in_map]
    #[run_config]
    pub step: u32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct MotifCosineNetArgs {
    #[in_map]
    #[run_config]
    pub a: f64,

    #[in_map]
    #[run_config]
    pub b: f64,

    #[in_map]
    #[run_config]
    pub init_mode: InitMode,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq, Default)]
pub struct MotifCosine2DNetArgs {
    #[in_map]
    #[run_config]
    pub a: f64,

    #[in_map]
    #[run_config]
    pub b: f64,

    #[in_map]
    #[run_config]
    pub init_mode: InitMode,

    #[in_map]
    #[run_config]
    pub stft_config: StftRunConfig,
}

#[allow(non_snake_case, unused_assignments, unused_variables, non_camel_case_types)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum MotifNetTypeArgs {
    ConvNet(MotifConvNetArgs),
    LinNet(MotifLinNetArgs),
    CosineNet(MotifCosineNetArgs),
    Cosine2DNet(MotifCosine2DNetArgs),
}

impl Default for MotifNetTypeArgs {
    fn default() -> Self {
        MotifNetTypeArgs::ConvNet(MotifConvNetArgs::default())
    }
}

#[allow(non_snake_case, unused_assignments, unused_variables)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Serialize, Deserialize)]
#[run_config_opt]
pub struct TrainMotifNetRunConfig {
    #[in_map]
    #[config_scalar]
    pub id: Option<RunConfigId>,

    #[in_map]
    #[run_config]
    pub common_training_args: CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    #[serde(default)]
    pub net: MotifNetTypeArgs,
}

impl WithCommonTrainingRunConfig for TrainMotifNetRunConfig {
    fn config(&self) -> &CommonTrainingRunConfig {
        &self.common_training_args
    }
}

impl TrainMotifNetRunConfig {
    pub fn new(id: Option<RunConfigId>, common_training_args: CommonTrainingRunConfig, net: MotifNetTypeArgs) -> Self {
        Self { id, common_training_args, net }
    }
}

impl AsRef<CommonTrainingRunConfig> for TrainMotifNetRunConfig {
    fn as_ref(&self) -> &CommonTrainingRunConfig {
        &self.common_training_args
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MotifConvNetParams {
    pub num_features: u32,
    pub conv_layers: u32,
    pub head_layers: u32,
    pub stride: u32, //stride within slice
    pub channels: u32,
    pub kernel: u32,
    pub head_pool: Option<BinaryHeadPool>,
}

impl MotifConvNetParams {
    pub fn new(num_features: u32, conv_layers: u32, head_layers: u32, stride: u32, channels: u32, kernel: u32) -> Self {
        Self {
            num_features,
            conv_layers,
            head_layers,
            stride,
            channels,
            kernel,
            head_pool: None,
        }
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
    pub a: f64,
    pub b: f64,
    pub init_mode: InitMode,
}

impl MotifCosineNetParams {
    pub fn new(num_features: u32, a: f64, b: f64, init_mode: InitMode) -> Self {
        Self { num_features, a, b, init_mode }
    }
}

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct MotifCosine2dNetParams {
    pub kw_time: u32,
    pub kh_freq: u32,
    pub a: f64,
    pub b: f64,
    pub init_mode: InitMode,
    pub motif_width: u32,
}

impl MotifCosine2dNetParams {
    pub fn new(kw: u32, kh: u32, a: f64, b: f64, init_mode: InitMode, motif_width: u32) -> Self {
        Self { kw_time: kw, kh_freq: kh, a, b, init_mode, motif_width }
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

impl ToJsonFile for MotifNetParams {
    fn to_file(&self, path: &PathBuf) -> anyhow::Result<()> {
        // Generic workspace persistence needs model parameters to provide the same JSON representation as the former specialized methods.
        fs::write(path, serde_json5::to_string(self)?)?;
        Ok(())
    }
}

impl MotifNetParams {
    pub fn new(num_features: u32, conv_layers: u32, head_layers: u32, stride: u32, channels: u32, kernel: u32) -> Self {
        Self::new_conv(num_features, conv_layers, head_layers, stride, channels, kernel)
    }

    pub fn new_with_net_type(net_type: MotifNetTypeParams) -> Self {
        Self { net_type }
    }

    pub fn new_conv(num_features: u32, conv_layers: u32, head_layers: u32, stride: u32, channels: u32, kernel: u32) -> Self {
        //head_pool: BinaryHeadPool
        Self::new_with_net_type(MotifNetTypeParams::ConvNetParams(MotifConvNetParams::new(num_features, conv_layers, head_layers, stride, channels, kernel)))
    }

    pub fn new_lin(num_features: u32, num_layers: u32, stride: u32) -> Self {
        Self::new_with_net_type(MotifNetTypeParams::LinNetParams(MotifLinNetParams::new(num_features, num_layers)))
    }

    pub fn new_cosine(num_features: u32, a: f64, b: f64, init_mode: InitMode) -> Self {
        Self::new_with_net_type(MotifNetTypeParams::CosineNetParams(MotifCosineNetParams::new(num_features, a, b, init_mode)))
    }

    pub fn new_cosine_2d(kw: u32, kh: u32, a: f64, b: f64, init_mode: InitMode, motif_width: u32) -> Self {
        Self::new_with_net_type(MotifNetTypeParams::CosineNet2DParams(MotifCosine2dNetParams::new(kw, kh, a, b, init_mode, motif_width)))
    }

    pub fn from_file(path: &PathBuf) -> anyhow::Result<Self> {
        serde_json5::from_str::<MotifNetParams>(fs::read_to_string(&path)?.as_str()).with_context(|| format!("read motif net params from {path:?}"))
    }

    pub fn num_features(&self) -> u32 {
        //this is used to group motif nets + calculate overlap for StepAlignedSplitsAdjuster::new_with_overlap at thoosa_motifs/src/stages/apply_motif_nets.rs
        match &self.net_type {
            MotifNetTypeParams::LinNetParams(v) => v.num_features,
            MotifNetTypeParams::ConvNetParams(v) => v.num_features,
            MotifNetTypeParams::CosineNetParams(v) => v.num_features,
            MotifNetTypeParams::CosineNet2DParams(v) => v.motif_width,
        }
    }

    pub fn motif_net(&self, motif: Option<Arc<ThoTensor<MotifSlices, SynKey>>>) -> anyhow::Result<Node> {
        debug!("motif_net: motif {motif:?}");
        match &self.net_type {
            MotifNetTypeParams::LinNetParams(v) => lin_motif_node(v),
            MotifNetTypeParams::ConvNetParams(v) => conv_motif_node(v),
            MotifNetTypeParams::CosineNetParams(v) => cosine_motif_node(v, motif),
            MotifNetTypeParams::CosineNet2DParams(v) => cosine_2d_motif_node(v, motif.map(|tt| -> anyhow::Result<_> { Ok(Arc::new(ThoTensor::new(tt.id().clone(), tt.local()?.f_alias()?)?)) }).transpose()?),
        }
    }
}

pub fn conv_motif_node(params: &MotifConvNetParams) -> anyhow::Result<Node> {
    debug!("motif_node: params={params:?}");

    let mut builder = SeqBuilder::new(params.num_features, None)?;
    let nd = ConvConfig { stride: params.stride as i64, ..ConvConfig::default() };

    let hidden_config = ConvConfig::default();

    let mut net = builder.batch_norm(BatchNormConfig::default(), None)?.closure(
        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
            let r = t.unsqueeze(1);

            debug!("conv_motif_node: 0. t.size={:?} r.size={:?}", t.size(), r.size());

            Ok(r)
        })),
        None,
    )?;

    net = net.conv_1d("motif_conv_1", params.kernel, 1, params.channels, nd, None)?.func(|t| Ok(Tensor::f_relu(t)?))?;

    let mut net = builder.batch_norm(BatchNormConfig::default(), None)?.closure(
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
            .func(|t| Ok(Tensor::f_gelu(t, "none")?))?
            // .dropout(0.1)?
        ;
    }

    net = net.conv_1d("score_head", 1, params.channels, 1, ConvConfig::default(), None)?;

    net = net.avg_pool(params.kernel, 1, params.kernel / 2)?;

    // net = net.adaptive_avg_pool(1, params.channels)?;

    net.closure(
        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
            let r = t.squeeze();

            let max = r.f_amax(-1, true)?;
            let min = r.f_mean_dim(-1, true, None)?;

            let r1 = Tensor::f_cat(&[max.shallow_clone(), min.shallow_clone()], 1)?;

            debug!("conv_motif_node: 1. t.size={:?}  r.size={:?}", t.size(), r.size());

            Ok(r1) //r/max
        })),
        None,
    )?
    .linear_feats("motif_first_head", 2, 1)?
    .func(|t| Ok(Tensor::f_relu(t)?))?;

    for i in 0..params.head_layers as usize {
        net = net.linear_f(format!("motif_{i}_head"), if i < 1 + (params.head_layers / 2) as usize { 2. } else { 0.5 })?.func(|t| Ok(Tensor::f_gelu(t, "none")?))?;
    }

    net
        // .func(|t| Ok(Tensor::f_relu(t)?))?
        .linear_s(1, "motif_last_head")?
        .end_seq()
}

pub fn cosine_2d_motif_node(params: &MotifCosine2dNetParams, motif: Option<Arc<ThoTensor<Spectorgram, SynKey>>>) -> anyhow::Result<Node> {
    debug!("cosine_2d_motif_node: params={params:?} motif={motif:?}");

    let mut builder = SeqBuilder::new(0, None)?;

    let params = MotifCosine2dNetParams {
        kh_freq: motif.as_ref().map(|v| v.as_ref().size()[1] as u32).unwrap_or(params.kh_freq), //freq
        kw_time: motif.as_ref().map(|v| v.as_ref().size()[2] as u32).unwrap_or(params.kw_time), //time

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
        .cosine_conv_2d("cosine_bank", params.kw_time, params.kh_freq, motif, params.a, params.b, params.init_mode)?
        // .closure(
        //     Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
        //         debug!("cosine_motif_node: 2. t.size={:?}", t.size());
        //         Ok(t.shallow_clone())
        //     })),
        //     None,
        // )?
        .end_seq()
}

pub static TRIGGER_RESET_IN_COSINE_MOTIF_NODE: &'static str = "TRIGGER_RESET_IN_COSINE_MOTIF_NODE.thoosa";

pub fn cosine_motif_node(params: &MotifCosineNetParams, motif: Option<Arc<ThoTensor<MotifSlices, SynKey>>>) -> anyhow::Result<Node> {
    debug!("cosine_motif_node: params={params:?}");

    let mut builder = SeqBuilder::new(params.num_features, None)?;

    let params = MotifCosineNetParams {
        num_features: motif.as_ref().map(|v| v.size().dim_1() as u32).unwrap_or(params.num_features),
        ..params.clone()
    };

    let builder = builder
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("cosine_motif_node: 1. t.size={:?}", t.size());

                #[cfg(feature = "test-restart-hooks")]
                {
                    use log::info;
                    use thoosa_shared::test_utils::{is_flag_set, set_flag};

                    static CALL_COUNTER: parking_lot::Mutex<u64> = parking_lot::Mutex::new(0);
                    let mut guard = CALL_COUNTER.lock();
                    *guard += 1;

                    if *guard == 10 && !is_flag_set(TRIGGER_RESET_IN_COSINE_MOTIF_NODE)? {
                        set_flag(TRIGGER_RESET_IN_COSINE_MOTIF_NODE)?;

                        info!("cosine_motif_node: trigger restart guard={guard:?} is_TRIGGER_RESET_IN_COSINE_MOTIF_NODE_set={:?}", is_flag_set(TRIGGER_RESET_IN_COSINE_MOTIF_NODE));

                        return Err(anyhow::anyhow!("device-side assert triggered"));
                    }
                }

                Ok(t.unsqueeze(1))
            })),
            None,
        )?
        .cosine_conv_1d("cosine_bank", params.num_features, motif, params.a, params.b, params.init_mode)?;

    #[cfg(test)]
    let builder = builder.closure(
        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
            debug!("cosine_motif_node: 2. t.size={:?}", t.size());
            Ok(t.shallow_clone())
        })),
        None,
    )?;

    builder.end_seq()
}

pub fn lin_motif_node(params: &MotifLinNetParams) -> anyhow::Result<Node> {
    debug!("motif_node: params={params:?}");

    let mut builder = SeqBuilder::new(params.num_features, None)?;
    let mut net = builder
        // .batch_norm()
        .linear_s(params.num_features * 2, "todo")?
        .func(|t| Ok(Tensor::f_relu(t)?))?;

    for i in 0..params.num_layers as usize {
        net = net.linear_s(params.num_features * 2, "todo")?.func(|t| Ok(Tensor::f_relu(t)?))?;
    }

    net
        // .linear_s(180)
        // .func(|t|Ok(Tensor::f_relu(t)?))
        .linear_s(1, "todo")?
        .func(|t| Ok(Tensor::f_sigmoid(t)?))?
        .end_seq()
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LabelNetParams {
    motif_nets: u32,

    pub net_type: LabelNetTypeParams,
}

impl ToJsonFile for LabelNetParams {
    fn to_file(&self, path: &PathBuf) -> anyhow::Result<()> {
        // Generic workspace persistence needs model parameters to provide the same JSON representation as the former specialized methods.
        fs::write(path, serde_json5::to_string(self)?)?;
        Ok(())
    }
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

impl LabelNetParams {
    pub fn new(
        motif_nets: u32, //TODO move to LinNetParams
        net_type: LabelNetTypeParams,
    ) -> Self {
        Self { motif_nets, net_type }
    }

    pub fn from_file(path: &PathBuf) -> anyhow::Result<Self> {
        Ok(serde_json5::from_str::<LabelNetParams>(fs::read_to_string(&path)?.as_str()).with_context(|| format!("read label net params from {path:?}"))?)
    }

    pub fn label_net(&self) -> anyhow::Result<Node> {
        match &self.net_type {
            LabelNetTypeParams::LinNetParams(_) => lin_label_node(&self),
            LabelNetTypeParams::ConvNetParams(_) => conv_label_node(&self),
            LabelNetTypeParams::SelfAttentionNetParams(_) => self_attention_label_node(&self),
        }
    }
}

pub fn conv_label_node(params: &LabelNetParams) -> anyhow::Result<Node> {
    let LabelNetTypeParams::ConvNetParams(cnn_params) = &params.net_type else {
        panic!("Unexpected params type: {:?}", params.net_type);
    };

    debug!("cnn_label_node: params={params:?}");

    let mut seq_builder = SeqBuilder::new(0, None)?;
    let mut builder = seq_builder.closure(
        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
            debug!("cnn_label_node: input={t:?}");
            Ok(t.shallow_clone())
        })),
        None,
    )?;

    builder = builder.conv_1d("cnn_label_conv_1d", cnn_params.kernel, params.motif_nets, cnn_params.hidden_conv_channels, ConvConfig::default(), None)?.closure(
        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
            debug!("cnn_label_node: cnn_label_conv_1d output={t:?}");
            Ok(t.shallow_clone())
        })),
        None,
    )?;

    for layer in 0..cnn_params.num_conv_layers {
        builder = builder.conv_1d(format!("cnn_label_conv_1d_{layer}"), cnn_params.kernel, cnn_params.hidden_conv_channels, cnn_params.hidden_conv_channels, ConvConfig::default(), None)?.closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("cnn_label_node: cnn_label_conv_1d_{layer} output={t:?}");
                Ok(t.shallow_clone())
            })),
            None,
        )?;
    }

    builder = match &cnn_params.joint {
        ConvNetJoint::AttnPool1d => builder
            .attn_pool1d("attn_pool", cnn_params.hidden_conv_channels, 2 * cnn_params.hidden_conv_channels)?
            .closure(
                Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                    debug!("cnn_label_node: attn_pool output={t:?}");
                    Ok(t.shallow_clone())
                })),
                None,
            )?
            .linear_feats("cnn_label_head", cnn_params.hidden_conv_channels, 2 * cnn_params.num_head_features)?
            .closure(
                Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                    debug!("cnn_label_node: cnn_label_head output={t:?}");
                    Ok(t.shallow_clone())
                })),
                None,
            )?,

        ConvNetJoint::MinMax => {
            builder
                .closure(
                    Arc::new(Box::new(move |t: &Tensor, lengths: Option<&BatchLengths>| {
                        debug!("cnn_label_node: MinMax leaner input={t:?} lengths={lengths:?}");
                        let lengths = lengths.map(BatchLengths::as_slice).unwrap_or(&[]); //TODO
                        let mask = mask_from_lengths(lengths, *lengths.iter().max().unwrap_or(&0), t.device())?;

                        let mean = t.f_mean_dim(2, false, None)?;
                        let max = t.f_max_dim(2, false)?;
                        let mean = masked_mean(t, &mask)?;
                        let max = masked_max(t, &mask)?;

                        Ok(Tensor::f_cat(&[mean, max], 1)?)
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
                .linear_feats("cnn_label_head", 2 * cnn_params.hidden_conv_channels, 2 * cnn_params.num_head_features)?
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
        builder = builder
            .linear_s(2 * cnn_params.num_head_features, format!("cnn_label_head_{layer}"))?
            .closure(
                Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                    debug!("cnn_label_node: cnn_label_head_{layer} output={t:?}");
                    Ok(t.shallow_clone())
                })),
                None,
            )?
            .func(|t| Ok(Tensor::f_relu(t)?))?;
    }

    builder
        .linear_s(1, "cnn_label_head_last")?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("cnn_label_node: cnn_label_head_last={t:?}");
                Ok(t.shallow_clone())
            })),
            None,
        )?
        // .func(|t|Ok(Tensor::f_sigmoid(t)?))
        .end_seq()
}

pub fn self_attention_label_node(params: &LabelNetParams) -> anyhow::Result<Node> {
    let LabelNetTypeParams::SelfAttentionNetParams(sa_params) = &params.net_type else {
        panic!("Unexpected params type: {:?}", params.net_type);
    };

    debug!("self_attention_label_node: params={params:?}");

    let mut seq_builder = SeqBuilder::new(0, None)?;
    let builder = seq_builder
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("self_attention_label_node: transpose input={t:?} _lengths={_lengths:?}");
                Ok(t.f_transpose(1, 2)?)
            })),
            Some(params.motif_nets),
        )?
        .linear_feats("self_attention_in_proj", params.motif_nets, sa_params.head_size * sa_params.n_heads)?
        .self_attention("self_attention_block", sa_params.head_size * sa_params.n_heads, sa_params.n_heads, sa_params.ff_mult, sa_params.dropout)?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, lengths: Option<&BatchLengths>| {
                debug!("self_attention_label_node: pooling input={t:?} lengths={lengths:?}");
                common_nets::self_attention_masked_mean(t, lengths)
            })),
            None,
        )?
        .linear_s(1, "self_attention_label_head_last")?;

    #[cfg(any(debug_assertions, tho_debug))]
    {
        let builder = builder.closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("self_attention_label_node: self_attention_label_head_last={t:?}");
                Ok(t.shallow_clone())
            })),
            None,
        )?;
    }

    builder.end_seq()
}

pub fn lin_label_node(params: &LabelNetParams) -> anyhow::Result<Node> {
    debug!("lin_label_node: params={params:?}");

    let LabelNetTypeParams::LinNetParams(lin_params) = &params.net_type else {
        panic!("Unexpected params type: {:?}", params.net_type);
    };

    let num_features = lin_params.features;
    debug!("lin_label_node: num_features={num_features}");

    let mut seq_builder = SeqBuilder::new(0, None)?;
    let mut builder = seq_builder.adaptive_avg_pool(lin_params.features, lin_params.features)?;

    for layer in 0..lin_params.layers {
        builder = builder.linear_feats("lin_label_node", lin_params.features, lin_params.features)?.func(|t| Ok(Tensor::f_relu(t)?))?;
    }

    builder
        .linear_s(1, "lin_label_node last")?
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
    // Array position is the downstream feature index, independent of model discovery order.
    #[serde(default)]
    label_feature_schema: Option<Vec<RunConfigId>>,
    in_features: u32,
    mid_features: u32,
    out_features: u32,
    num_layers: u32,
}

impl ToJsonFile for FinalNetParams {
    fn to_file(&self, path: &PathBuf) -> anyhow::Result<()> {
        // Generic workspace persistence needs model parameters to provide the same JSON representation as the former specialized methods.
        fs::write(path, serde_json5::to_string(self)?)?;
        Ok(())
    }
}

impl FinalNetParams {
    pub fn new(in_features: u32, mid_features: u32, out_features: u32, num_layers: u32) -> Self {
        Self {
            label_feature_schema: None,
            in_features,
            mid_features,
            out_features,
            num_layers,
        }
    }

    pub fn set_label_feature_schema(&mut self, ids: Vec<RunConfigId>) -> anyhow::Result<()> {
        self.validate_label_feature_schema(&ids)?;
        self.label_feature_schema = Some(ids);
        Ok(())
    }

    pub fn label_feature_schema(&self) -> anyhow::Result<&[RunConfigId]> {
        // Legacy files can still load weights, but guessing their feature order would silently change predictions.
        let ids = self.label_feature_schema.as_deref().context("final model has no label feature schema; retrain it or supply its known original feature order")?;
        self.validate_label_feature_schema(ids)?;
        Ok(ids)
    }

    fn validate_label_feature_schema(&self, ids: &[RunConfigId]) -> anyhow::Result<()> {
        anyhow::ensure!(ids.len() == self.in_features as usize, "label feature schema has {} entries, expected {}", ids.len(), self.in_features);
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        anyhow::ensure!(unique.len() == ids.len(), "label feature schema contains duplicate run IDs");
        Ok(())
    }

    pub fn from_file(path: &PathBuf) -> anyhow::Result<Self> {
        Ok(serde_json5::from_str::<FinalNetParams>(fs::read_to_string(&path)?.as_str()).with_context(|| format!("read final net params from {path:?}"))?)
    }
}

pub fn final_net(params: &FinalNetParams) -> anyhow::Result<Node> {
    debug!("final_node: params={:?}", params);

    let mut builder = SeqBuilder::new(params.in_features, None)?;
    let mut net = builder
        //TODO enabled for tho_debug
        // .closure(
        //     Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
        //         trace!("final_net: input=\n{t}");
        //         Ok(t.shallow_clone())
        //     })),
        //     None,
        // )?
        .linear_s(params.mid_features * params.in_features, "final_net_first_layer".to_string())?
        .func(|t| Ok(Tensor::f_relu(t)?))?;

    for i in 0..params.num_layers {
        net = net.linear_s(params.mid_features * params.in_features, format!("final_net_{i}th_layer"))?.func(|t| Ok(Tensor::f_relu(t)?))?;
    }

    net.linear_s(params.out_features, "final_net_last_layer")?.end_seq()
}

pub fn slices_in_all_resources<ResourceId>(slices: Vec<(ResourceId, Vec<Tensor>)>) -> Vec<Tensor> {
    slices.into_iter().flat_map(|(_resource_key, slices)| slices.into_iter()).collect::<Vec<Tensor>>()
}

#[derive(Args, Debug, Clone)]
pub struct TrainFinalNetArgs {
    #[arg(long, value_delimiter = ',', value_parser = thoosa_common::utils::parse_run_config_ids)]
    pub train_label_nets_run_config_ids: Vec<RunConfigId>,

    #[arg(long)]
    train_resource_group: String,

    #[arg(long)]
    valid_resource_group: String,

    #[command(flatten)]
    pub opt_config: OptConfig,

    #[arg(long)]
    pub run_config_cnt: String, //TrainLabelNetRunConfig

    #[arg(long, default_value_t = 10)]
    pub motif_slices_step: u32,
}

impl TrainFinalNetArgs {
    pub fn resource_config(&self) -> NetResources<Logits> {
        NetResources::new(Logits::new(self.train_resource_group.clone()), Logits::new(self.valid_resource_group.clone()))
    }
}

#[allow(non_snake_case, unused_assignments, unused_variables)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_opt]
pub struct TrainFinalNetRunConfig {
    #[in_map]
    #[config_scalar]
    pub id: Option<RunConfigId>,

    #[in_map]
    #[run_config]
    pub common_training_args: CommonTrainingRunConfig,

    #[in_map]
    #[run_config]
    pub net: FinalNetArgs,
}

impl WithCommonTrainingRunConfig for TrainFinalNetRunConfig {
    fn config(&self) -> &CommonTrainingRunConfig {
        &self.common_training_args
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub struct FinalNetArgs {
    #[in_map]
    #[run_config]
    pub num_layers: u32,

    #[in_map]
    #[run_config]
    pub mid_features: u32,
}

impl AsRef<CommonTrainingRunConfig> for TrainFinalNetRunConfig {
    fn as_ref(&self) -> &CommonTrainingRunConfig {
        &self.common_training_args
    }
}

impl FinalNetArgs {
    pub fn new(num_layers: u32, mid_features: u32) -> Self {
        Self { num_layers, mid_features }
    }
}

impl TrainFinalNetRunConfig {
    pub fn new(id: Option<RunConfigId>, common_training_args: CommonTrainingRunConfig, net: FinalNetArgs) -> Self {
        Self { id, common_training_args, net }
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
    pub limit: Option<usize>,

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

    pub fn new(final_run_config_id: RunConfigId, input_data: String, output: String, input_slice_step: usize) -> Self {
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

#[allow(non_snake_case, unused_assignments, unused_variables, non_camel_case_types)]
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub enum LabelNetTypeArgs {
    ConvNet(ConvNetArgs),

    LinNet(LinNetArgs),

    SelfAttentionNet(SelfAttentionNetArgs),
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

impl LinNetArgs {
    pub fn new(num_layers: u32, slices_len: u32) -> Self {
        Self { num_layers, features: slices_len }
    }
}

///sums and multiplies inputs by k
pub fn dummy_label_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(num_features: u32, k: f64, device: Device, descr: &str) -> anyhow::Result<Box<dyn ThoModule<DI, DO>>> {
    let vs = VarStore::new(device);

    SeqBuilder::new(num_features, None)?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                debug!("test_label_net: merged final in flatten t.size()={:?}", t.size());

                let t_0 = t.sum_dim_intlist(&[1, 2][..], false, None);
                debug!("test_label_net: merged final out flatten t_0.size()={:?}", t_0.size());

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

///returns [batch x frequencies x time]
pub fn motifs_to_specs(x_train: &Tensor, desc: &str, hop_length: i64 /*, path_buf: &PathBuf*/) -> Tensor {
    let motifs = x_train.unbind(0);
    let mut specs = vec![];
    // let path = path_buf.join(desc);
    // fs::create_dir_all(&path).unwrap();
    for (i, motif) in motifs.iter().enumerate() {
        let motif = motif.unsqueeze(0);
        let spec_i = motif.stft(16000, hop_length, 16000, None::<Tensor>, false, true, true, true).squeeze_dim(0).abs(); //.transpose(0, 1);

        debug!("motifs_to_specs: motif={motif:?} spec_i={spec_i:?}");
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
    use super::{
        CommonTrainingRunConfigRangeOrValue, FinalNetArgs, FinalNetParams, LabelNetTypeArgs, LabelNetTypeArgsRangeOrValue, LabelNetTypeParams, LinNetArgs, MotifCosine2dNetParams, MotifCosineNetParams, TrainFinalNetRunConfig, TrainLabelNetRunConfigResultParams, TrainMotifNetRunConfig,
        conv_label_node, cosine_2d_motif_node, cosine_motif_node, dummy_label_net, lin_label_node, lin_motif_node, motifs_to_specs, self_attention_label_node,
    };
    use super::{InferArgs, LabelNetParams, MotifLinNetParams, MotifNetParams};
    use crate::dims::{ChannelsMotifsData, LabelProbability, MotifProbability};
    use anyhow::Context as _;
    use async_trait::async_trait;
    use futures::executor::block_on;
    use futures_util::FutureExt;
    use log::LevelFilter::{Debug, Info, Trace};
    use log::{debug, error, info, set_max_level, trace};
    use mockall::automock;
    use mockall_double::double;
    use parking_lot::{Mutex, RwLock};
    use rand::prelude::SliceRandom;
    use rand::{Rng, RngExt, SeedableRng, random_range};
    use rayon::iter::IntoParallelRefMutIterator;
    use std::cmp::max;
    use std::collections::HashMap;
    use std::ops::{Deref, RangeInclusive};
    use std::path::{Path, PathBuf};
    use std::pin::Pin;
    use std::rc::Rc;
    use std::sync::Arc;
    use std::{fs, mem};
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Bool, Float, Half};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::nn::{AdamW, LinearConfig, Module, OptimizerConfig, VarStore, init};
    use tch::{Device, Tensor, nn};
    use tch::{IndexOp, Kind, Reduction};
    use thoosa_common::dims::{AnyDims, Batch, Dims, Slice, ThoBatch};
    use thoosa_common::nets::batches::BatchLengths;
    use thoosa_common::nets::builder::{Node, SeqBuilder};
    use thoosa_common::nets::common_nets;
    use thoosa_common::nets::common_nets::ConvNetArgsFields::Joint;
    use thoosa_common::nets::nodes::{InitDummy, InitMode, ThoModule};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::transforms::filters::BandPassFilterParamsRangeOrValue;
    use thoosa_common::utils::{TrainListener, debug_plot_impl, from, mix_x_binary_with_labels, optimizer_mem_size, pad_stack_with_mask, plot_spectrogram, set_requires_grad_in_subpath, stretch, vs_mem_size};
    use thoosa_common::utils::{lin_t, pad_to, reset_print_options, t_0_0, tensor_mem_size};
    use thoosa_common::ws::{FsWorkspace, NET_WEIGHTS_FILE, Workspace};
    use thoosa_common::ws::{WsConfigs, mock_ws_simp};
    use thoosa_properties_map::ToHashmap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::perf_logging::{PERF, format_perf_map, set_panic_hook};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::{OptRunConfigs, RunConfig, RunConfigId};
    use thoosa_shared::traits::DebugId;
    use thoosa_shared::val::Val;
    use thoosa_shared::{CopyToDevice, ThoError, batch_windows, cdt_for_dir, clc_num_batches, mono_key, pulse_in_rand, sin_v, sin_v_f, slices_len};
    use thoosa_store::{MockStore, SqlStore, TrainedNet, TrainedNetsBy};
    use torch_sys::at_autocast_clear_cache;

    use super::LabelNetTypeParams::LinNetParams;
    use crate::tch_test_utils::motifs_tensor;
    use rand::rngs::StdRng;
    use rand_distr::{Distribution, Normal};
    use thoosa_common::nets::common_nets::{CommonTrainingRunConfig, ConvNetArgs, ConvNetJoint, ConvNetParams, LinLenNetParams, NetsPadding, OptConfig, PlotVecf32, SelfAttentionNetParams, ThoModuleWithStore};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::utils;
    use thoosa_shared::test_utils::TestSignalKind;
    use thoosa_shims::inference_mode::InferenceModeGuard;

    #[ctor::ctor]
    fn test_init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_test_net() -> anyhow::Result<()> {
        let t_0 = lin_test_tensor(1000, 5);
        debug!("t_0.size={:?}", t_0.size());

        let t: ThoTensor<AnyDims, DebugId> = t_0.unfold(1, 100, 10).into();

        debug!("t.size={:?}", t.size());

        //sums and multiplies by k
        let r_0: ThoTensor<AnyDims, DebugId> = utils::test_net(100, 1., Cpu, "test_net")?.tho_forward(&t, None, true)?;
        debug!("r_0.size={:?}", r_0.size());

        assert_eq!(Tensor::from_slice(&[4950.]), r_0.local()?.i((0, 0, ..)));
        assert_eq!(Tensor::from_slice(&[9900.]), utils::test_net::<AnyDims, AnyDims>(100, 2., Cpu, "test_net")?.tho_forward(&t, None, true)?.local()?.i((0, 0, ..)));

        Ok(())
    }

    //5 ch
    fn lin_test_tensor(input_n: i64, channels: u32) -> Tensor {
        Tensor::vstack((1..=channels).map(|c| Tensor::from_slice((0..input_n).map(|i| i as f32 * c as f32).collect::<Vec<_>>().as_slice())).collect::<Vec<_>>().as_slice())
    }

    #[tokio::test]
    async fn test_save_load_lin_label_node() -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_dir={test_dir:?}");
        let channels = 3;
        let max_motif_nodes = 19;
        let features = 23;

        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, WsConfigs::new(channels, 1000, 32000))?));
        let params = LabelNetParams::new(max_motif_nodes, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(features, 3)));

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = lin_label_node(&params)?.build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        let batch = 11i64;
        for features in [features, 123, 17] {
            let _ = net.tho_forward(&Tensor::rand([batch, channels as i64, features as i64], FLOAT_CPU).into(), None, false)?;
        }

        let model_path = ws.write().save_net(1, (&vs_0).into(), "labels/test_label", &params)?;

        debug!("model_path={model_path:?} vs_0.variables().len()={}", vs_0.variables().len());

        let params_for_builder = params.clone();
        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load("".into(), move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder)?.build_tch_module(path), model_path, Cpu, false)?;

        assert!(vs_0.variables().len() > 0);
        assert_eq!(vs_0.variables(), loaded_module.var_store().variables());

        Ok(())
    }

    #[tokio::test]
    async fn test_save_load_conv_label_node() -> anyhow::Result<()> {
        thoosa_common::runner::global_ctx::GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        _test_save_load_conv_label_node(3, 19, &[2000, 12030, 17000], 11i64, ConvNetParams::new(3, 64, 10, 10, 256, ConvNetJoint::MinMax)).await?;

        _test_save_load_conv_label_node(3, 19, &[2000, 12030, 17000], 11i64, ConvNetParams::new(3, 64, 10, 10, 256, ConvNetJoint::AttnPool1d)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_save_load_conv_label_node_1() -> anyhow::Result<()> {
        thoosa_common::runner::global_ctx::GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        _test_save_load_conv_label_node(360, 1541, &[1541], 101i64, ConvNetParams::new(7, 64, 100, 10, 256, ConvNetJoint::MinMax)).await?;

        _test_save_load_conv_label_node(360, 1541, &[1541], 101i64, ConvNetParams::new(7, 64, 100, 10, 256, ConvNetJoint::AttnPool1d)).await?;

        Ok(())
    }

    #[tokio::test]
    async fn test_save_load_self_attention_label_node() -> anyhow::Result<()> {
        thoosa_common::runner::global_ctx::GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        _test_save_load_self_attention_label_node(19, &[128, 256, 384], 11i64, SelfAttentionNetParams::new(64, 4, 4, 0.1), false).await?;

        _test_save_load_self_attention_label_node(19, &[128, 256, 384], 11i64, SelfAttentionNetParams::new(64, 4, 4, 0.1), true).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_test_train_conv_label_node() -> anyhow::Result<()> {
        let motif_nets = 100;
        let conv_net_params = ConvNetParams::new(8, 8, 10, 8, 8, ConvNetJoint::AttnPool1d);
        let params = LabelNetParams::new(motif_nets, LabelNetTypeParams::ConvNetParams(conv_net_params));

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = conv_label_node(&params)?.build_tch_module(&vs_0.root())?;
        let mut opt = AdamW::default().amsgrad(true).build(&vs_0, 1e-3)?;

        let batch = 64usize;
        let (x_train, y_train, lengths_train) = gen_motifs_output(batch, motif_nets as usize, 1200..=1400, LabelSignal::ByMotifs, 0.01, Some(11))?;
        let target_train = y_train.unsqueeze(1);
        let lengths_train = lengths_train.iter().map(|v| *v as i64).collect::<Vec<_>>();

        let (x_eval, y_eval, lengths_eval) = gen_motifs_output(batch, motif_nets as usize, 1200..=1400, LabelSignal::ByMotifs, 0.15, Some(12))?;
        let target_eval = y_eval.unsqueeze(1);
        let lengths_eval = lengths_eval.iter().map(|v| *v as i64).collect::<Vec<_>>();

        let mut eval_lengths_before = BatchLengths::new(lengths_eval.clone());
        let logits_eval_before = net.tho_forward(&x_eval.shallow_clone().into(), Some(&mut eval_lengths_before), false)?;
        let eval_loss_before = logits_eval_before.local()?.binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);

        let mut first_loss = None;
        let mut last_loss = 0.0;
        for _ in 0..50 {
            let mut batch_lengths = BatchLengths::new(lengths_train.clone());
            let logits = net.tho_forward(&x_train.shallow_clone().into(), Some(&mut batch_lengths), true)?;
            let loss = logits.local()?.binary_cross_entropy_with_logits::<Tensor>(&target_train, None, None, Reduction::Mean);
            if first_loss.is_none() {
                first_loss = Some(loss.double_value(&[]));
            }
            last_loss = loss.double_value(&[]);
            opt.backward_step(&loss);
        }

        let mut lengths_eval = BatchLengths::new(lengths_eval.clone());
        let logits_eval = net.tho_forward(&x_eval.shallow_clone().into(), Some(&mut lengths_eval), false)?;
        let eval_loss = logits_eval.local()?.binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);

        debug!("heavy_test_train_conv_label_node: first_loss={:?} last_loss={last_loss:.5} eval_loss_before={eval_loss_before:.5} eval_loss_after={eval_loss:.5}", first_loss);

        assert!(first_loss.unwrap_or(last_loss) > last_loss, "expected train loss to decrease, first_loss={first_loss:?}, last_loss={last_loss}");
        assert!(eval_loss < eval_loss_before, "expected eval loss to improve, before={eval_loss_before}, after={eval_loss}");

        info!("worker perf:\n{}", format_perf_map(&PERF));

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_test_train_lin_label_node_on_kernels() -> anyhow::Result<()> {
        let motif_nets = 100;
        let conv_net_params = LinLenNetParams::new(256, 4);
        let params = LabelNetParams::new(motif_nets, LabelNetTypeParams::LinNetParams(conv_net_params));

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = lin_label_node(&params)?.build_tch_module(&vs_0.root())?;
        let mut opt = AdamW::default().amsgrad(true).build(&vs_0, 1e-3)?;

        let batch = 64usize;
        let (x_train, y_train, lengths_train) = gen_motifs_output(batch, motif_nets as usize, 1200..=1400, LabelSignal::ByMotifs, 0.01, Some(11))?;
        let target_train = y_train.unsqueeze(1);
        let lengths_train = lengths_train.iter().map(|v| *v as i64).collect::<Vec<_>>();

        let (x_eval, y_eval, lengths_eval) = gen_motifs_output(batch, motif_nets as usize, 1200..=1400, LabelSignal::ByMotifs, 0.15, Some(12))?;
        let target_eval = y_eval.unsqueeze(1);
        let lengths_eval = lengths_eval.iter().map(|v| *v as i64).collect::<Vec<_>>();

        let mut eval_lengths_before = BatchLengths::new(lengths_eval.clone());
        let logits_eval_before = net.tho_forward(&x_eval.shallow_clone().into(), Some(&mut eval_lengths_before), false)?;
        let eval_loss_before = logits_eval_before.local()?.binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);

        let mut first_loss = None;
        let mut last_loss = 0.0;
        for _ in 0..50 {
            let mut batch_lengths = BatchLengths::new(lengths_train.clone());
            let logits = net.tho_forward(&x_train.shallow_clone().into(), Some(&mut batch_lengths), true)?;
            let loss = logits.local()?.binary_cross_entropy_with_logits::<Tensor>(&target_train, None, None, Reduction::Mean);
            if first_loss.is_none() {
                first_loss = Some(loss.double_value(&[]));
            }
            last_loss = loss.double_value(&[]);
            opt.backward_step(&loss);
        }

        let mut lengths_eval = BatchLengths::new(lengths_eval.clone());
        let logits_eval = net.tho_forward(&x_eval.shallow_clone().into(), Some(&mut lengths_eval), false)?;
        let eval_loss = logits_eval.local()?.binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);

        debug!("heavy_test_train_conv_label_node: first_loss={:?} last_loss={last_loss:.5} eval_loss_before={eval_loss_before:.5} eval_loss_after={eval_loss:.5}", first_loss);

        assert!(first_loss.unwrap_or(last_loss) > last_loss, "expected train loss to decrease, first_loss={first_loss:?}, last_loss={last_loss}");
        assert!(eval_loss < eval_loss_before, "expected eval loss to improve, before={eval_loss_before}, after={eval_loss}");

        Ok(())
    }

    fn gen_motifs_batch(batch: usize, motif_len: usize, motifs: &Tensor) -> anyhow::Result<(Tensor, Option<BatchLengths>, Tensor)> {
        assert_eq!(batch % 2, 0);

        // debug_plot_impl(motifs, "motifs", 0)?;
        let random = motifs_tensor(motif_len, batch / 2, TestSignalKind::Triangle, 1.0, 2.0);
        // debug_plot_impl(&random, "random", 0)?;

        mix_x_binary_with_labels(motifs, None, &random, None)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_test_train_cosine_motif_node() -> anyhow::Result<()> {
        let motif_width = 100usize;
        let batch = 64usize;

        let conv_net_params = ConvNetParams::new(8, 8, 10, 8, 8, ConvNetJoint::AttnPool1d);

        let mut vs_0 = VarStore::new(Cpu);
        let motifs = motifs_tensor(motif_width, batch / 2, TestSignalKind::ChirpUp, 1.0, 2.0);

        let (x_train, lengths_train, y_train) = gen_motifs_batch(batch, motif_width, &motifs)?;
        // debug_plot_impl(&x_train, "x_train", 0)?;

        let (x_eval, lengths_eval, y_eval) = gen_motifs_batch(batch, motif_width, &motifs_tensor(motif_width, batch / 2, TestSignalKind::ChirpUp, 1.5, 3.0))?;
        // debug_plot_impl(&x_eval, "x_eval", 0)?;

        for init_mode in &thoosa_common::nets::nodes::InitMode::list(0.4) {
            let net = cosine_motif_node(&MotifCosineNetParams::new(motif_width as u32, 10.0, 1.0, init_mode.clone()), Some(Arc::new(ThoTensor::new("motifs".into(), motifs.shallow_clone())?)))?.build_tch_module(&vs_0.root())?;

            let result = run_cosine_motif_node(&mut vs_0, &x_train, &y_train, &x_eval, &y_eval, &net);
            debug!("heavy_test_train_cosine_motif_node: init_mode={init_mode:?} result={result:?}");
        }

        Ok(())
    }

    fn run_cosine_motif_node(vs_0: &mut VarStore, x_train: &Tensor, y_train: &Tensor, x_eval: &Tensor, y_eval: &Tensor, net: &Box<dyn ThoModule<AnyDims, AnyDims>>) -> anyhow::Result<u32> {
        let mut opt = AdamW::default().amsgrad(true).build(&vs_0, 1e-3)?;

        let logits_eval_before = net.tho_forward(&x_eval.shallow_clone().into(), None, false)?;
        let eval_loss_before = logits_eval_before.local()?.binary_cross_entropy_with_logits::<Tensor>(&y_eval, None, None, Reduction::Mean).double_value(&[]);
        assert!(eval_loss_before > 0.5);
        let mut first_loss = None;
        let mut last_loss = 0.0;
        for i in 0..50u32 {
            let logits = net.tho_forward(&x_train.shallow_clone().into(), None, true)?;

            let loss = logits.local()?.binary_cross_entropy_with_logits::<Tensor>(&y_train, None, None, Reduction::Mean);
            if first_loss.is_none() {
                first_loss = Some(loss.double_value(&[]));
            }
            last_loss = loss.double_value(&[]);

            let logits_eval = net.tho_forward(&x_eval.shallow_clone().into(), None, false)?;
            let eval_loss = logits_eval.local()?.binary_cross_entropy_with_logits::<Tensor>(&y_eval, None, None, Reduction::Mean).double_value(&[]);
            if eval_loss < 0.1 {
                debug!("heavy_test_train_cosine_motif_node: first_loss={:?} last_loss={last_loss:.5} eval_loss_before={eval_loss_before:.5} eval_loss_after={eval_loss:.5}", first_loss);
                return Ok(i);
            }

            debug!("heavy_test_train_cosine_motif_node: x_train={x_train:?} y_train={y_train:?} logits={logits:?} logits.isnan={} last_loss={last_loss}", logits.local()?.isnan().any());
            trace!("heavy_test_train_cosine_motif_node: x_train={x_train} y_train={y_train}");

            opt.backward_step(&loss);
        }
        Err(anyhow::anyhow!("failed"))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_test_train_cosine_motif_node_2d() -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp/plots/").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;

        let motif_width = 160_000usize;
        let batch = 64usize;

        let mut vs_0 = VarStore::new(Cpu);
        // let motifs = motifs_tensor(motif_width, batch / 2, MotifKind::ChirpUp, 1.0, 2.0);
        let motifs = Tensor::stack((0..batch / 2).map(|i| Tensor::from_slice(sin_v_f(160_000, 8000.0, 32_000.0).as_slice())).collect::<Vec<_>>().as_slice(), 0);

        let kernel = motifs_to_specs(&motifs, "kernel" /*, &test_dir*/, 4000);

        let (x_train, lengths_train, y_train) = gen_motifs_batch(batch, motif_width, &motifs)?;
        // debug_plot_impl(&x_train, "x_train", 0)?;

        let specs_x = motifs_to_specs(&x_train, "specs_x" /*, &test_dir*/, 4000);

        debug!("heavy_test_train_cosine_motif_node_2d: all_spects_t={specs_x:?} kernel={kernel:?}");

        // let (x_eval,lengths_eval, y_eval, ) = gen_motifs_batch(batch, motif_width, &motifs_tensor(motif_width, batch / 2, MotifKind::ChirpUp, 1.5, 3.0))?;
        // debug_plot_impl(&x_eval, "x_eval", 0)?;

        for init_mode in [InitMode::Mean(InitDummy {})] {
            // &crate::my_tch::MotifInitMode::list(0.4) {
            let net = cosine_2d_motif_node(&MotifCosine2dNetParams::new(0, 0, 10.0, 1.0, init_mode.clone(), motif_width as u32), Some(Arc::new(ThoTensor::new("motifs".into(), kernel.shallow_clone())?)))?.build_tch_module(&vs_0.root())?;

            let result = run_cosine_motif_node(
                // &mut vs_0, &x_train, &y_train, &x_eval, &y_eval,
                &mut vs_0, &specs_x, &y_train, &specs_x, &y_train, &net,
            );
            debug!("heavy_test_train_cosine_motif_node: init_mode={init_mode:?} result={result:?}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_self_attention_label_node() -> anyhow::Result<()> {
        tch::manual_seed(42);
        let motif_nets = 2;

        let batch = 10usize;
        let eval_x_y = gen_motifs_output(batch, motif_nets as usize, 100..=100, LabelSignal::ByOrder, 0.1, None)?;
        // crate::utils::debug_plot_impl(&eval_x_y.0.i((.., 0, ..)), &format!("eval_random_motif_net_0_output_label_{:?}",  eval_x_y.1), 0)?;
        // crate::utils::debug_plot_impl(&eval_x_y.0.i((.., 1, ..)), &format!("eval_random_motif_net_1_output_label_{:?}",  eval_x_y.1), 0)?;

        let (eval_loss_before, first_loss, last_loss, eval_loss) = train_self_attention(
            motif_nets,
            SelfAttentionNetParams::new(20, 1, 1, 0.1),
            batch,
            eval_x_y,
            |step| {
                let v = gen_motifs_output(batch, motif_nets as usize, 100..=100, LabelSignal::ByOrder, 0.1, None)?;
                // debug!("y {}", v.1);
                // crate::utils::debug_plot_impl(&v.0.i((.., 0, ..)), &format!("{step}_random_motif_net_0_output_label_{:?}",  v.1), 0)?;
                // crate::utils::debug_plot_impl(&v.0.i((.., 1, ..)), &format!("{step}_random_motif_net_1_output_label_{:?}",  v.1), 0)?;
                Ok(v)
            },
            100,
        )?;

        debug!("test_train_self_attention_label_node: first_loss={first_loss:?} last_loss={last_loss:.5} eval_loss_before={eval_loss_before:.5} eval_loss_after={eval_loss:.5}");

        assert!(last_loss < first_loss.expect("no first_loss"));

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_test_train_self_attention_label_node() -> anyhow::Result<()> {
        tch::manual_seed(42);
        let motif_nets = 30;

        let (eval_loss_before, first_loss, last_loss, eval_loss) = train_self_attention(
            motif_nets,
            SelfAttentionNetParams::new(64, 4, 2, 0.1),
            32usize,
            gen_motifs_output(32, motif_nets as usize, 220..=320, LabelSignal::ByOrder, 0.35, Some(22))?,
            |step| gen_motifs_output(32, motif_nets as usize, 220..=320, LabelSignal::ByOrder, 0.35, Some(1_000 + step as u64)),
            120,
        )?;

        debug!("heavy_test_train_self_attention_label_node: first_loss={:?} last_loss={last_loss:.5} eval_loss_before={eval_loss_before:.5} eval_loss_after={eval_loss:.5}", first_loss,);

        assert!(first_loss.unwrap_or(last_loss) > last_loss, "expected train loss to decrease, first_loss={first_loss:?}, last_loss={last_loss}");
        assert!(eval_loss < eval_loss_before, "expected eval loss to improve, before={eval_loss_before}, after={eval_loss}");

        Ok(())
    }

    fn train_self_attention<XYF>(motif_nets: u32, attention_net_params: SelfAttentionNetParams, batch: usize, x_y_lengths_eval: (Tensor, Tensor, Vec<usize>), x_y_lengths_train: XYF, steps: i32) -> anyhow::Result<(f64, Option<f64>, f64, f64)>
    where
        XYF: Fn(i32) -> anyhow::Result<(Tensor, Tensor, Vec<usize>)>,
    {
        let params = LabelNetParams::new(motif_nets, LabelNetTypeParams::SelfAttentionNetParams(attention_net_params));

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = self_attention_label_node(&params)?.build_tch_module(&vs_0.root())?;
        let mut opt = AdamW::default().amsgrad(true).build(&vs_0, 1e-2)?;

        let (x_eval, y_eval, lengths_eval) = x_y_lengths_eval;
        let target_eval = y_eval.unsqueeze(1);
        let lengths_eval = lengths_eval.iter().map(|v| *v as i64).collect::<Vec<_>>();

        let mut eval_lengths_before = BatchLengths::new(lengths_eval.clone());
        let logits_eval_before = net.tho_forward(&x_eval.shallow_clone().into(), Some(&mut eval_lengths_before), false)?;
        let eval_loss_before = logits_eval_before.local()?.binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);

        let mut first_loss = None;
        let mut last_loss = 0.0;
        for step in 0..steps {
            let (x_train, y_train, lengths_train) = x_y_lengths_train(step)?;
            let target_train = y_train.unsqueeze(1);
            let mut batch_lengths = BatchLengths::new(lengths_train.iter().map(|v| *v as i64).collect::<Vec<_>>());
            let logits = net.tho_forward(&x_train.shallow_clone().into(), Some(&mut batch_lengths), true)?;
            let loss = logits.local()?.binary_cross_entropy_with_logits::<Tensor>(&target_train, None, None, Reduction::Mean);
            if first_loss.is_none() {
                first_loss = Some(loss.double_value(&[]));
            }
            last_loss = loss.double_value(&[]);
            opt.backward_step(&loss);
        }

        let mut lengths_eval = BatchLengths::new(lengths_eval.clone());
        let logits_eval = net.tho_forward(&x_eval.shallow_clone().into(), Some(&mut lengths_eval), false)?;
        let eval_loss = logits_eval.local()?.binary_cross_entropy_with_logits::<Tensor>(&target_eval, None, None, Reduction::Mean).double_value(&[]);
        Ok((eval_loss_before, first_loss, last_loss, eval_loss))
    }

    async fn _test_save_load_conv_label_node(motif_nets: u32, max_motif_nodes: i32, all_slices: &[i32], batch: i64, conv_net_params: ConvNetParams) -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_dir={test_dir:?}");

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            WsConfigs::new(
                motif_nets, 1000, //not used
                32000,
            ),
        )?));
        let params = LabelNetParams::new(motif_nets, LabelNetTypeParams::ConvNetParams(conv_net_params));

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = conv_label_node(&params)?.build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        for slices in all_slices {
            let mut lengths = BatchLengths::new(vec![*slices as i64; batch as usize]);
            let _ = net.tho_forward(&Tensor::rand([batch, motif_nets as i64, *slices as i64], FLOAT_CPU).into(), Some(&mut lengths), false)?;
        }

        let model_path = ws.write().save_net(1, (&vs_0).into(), "labels/test_label", &params)?;

        debug!("model_path={model_path:?} vs_0.variables().len()={}", vs_0.variables().len());

        let params_for_builder = params.clone();
        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load("".into(), move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder)?.build_tch_module(path), model_path, Cpu, false)?;

        assert!(vs_0.variables().len() > 0);
        assert_eq!(vs_0.variables(), loaded_module.var_store().variables());

        let cuda_net = loaded_module.copy_to_device(1).await?;
        for slices in all_slices {
            let mut lengths = BatchLengths::new(vec![*slices as i64; batch as usize]);
            let _ = cuda_net.tho_forward(&Tensor::rand([batch, motif_nets as i64, *slices as i64], FLOAT_CUDA).into(), Some(&mut lengths), false)?;
        }

        Ok(())
    }

    async fn _test_save_load_self_attention_label_node(motif_nets: u32, all_slices: &[i32], batch: i64, self_attention_net_params: SelfAttentionNetParams, half: bool) -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("_test_save_load_self_attention_label_node: test_dir={test_dir:?}");

        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, WsConfigs::new(motif_nets, 1000, 32000))?));
        let params = LabelNetParams::new(motif_nets, LabelNetTypeParams::SelfAttentionNetParams(self_attention_net_params));

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = self_attention_label_node(&params)?.build_tch_module(&vs_0.root())?;
        if half {
            vs_0.half();
        }
        vs_0.freeze();

        let module_with_store = ThoModuleWithStore::with_factory(mono_key!(0u32, "test"), net, vs_0, None);

        for slices in all_slices {
            let mut lengths = BatchLengths::new(vec![*slices as i64; batch as usize]);
            let _ = module_with_store.tho_forward(&Tensor::rand([batch, motif_nets as i64, *slices as i64], FLOAT_CPU).into(), Some(&mut lengths), false)?;
        }

        let model_path = ws.write().save_net(1, module_with_store.var_store().into(), "labels/test_label", &params)?;

        debug!("_test_save_load_self_attention_label_node: model_path={model_path:?} vs_0.variables().len()={}", module_with_store.var_store().variables().len());

        let params_for_builder = params.clone();
        let loaded_module: ThoModuleWithStore<AnyDims, AnyDims> = ThoModuleWithStore::load("".into(), move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder)?.build_tch_module(path), model_path, Cpu, false)?;

        assert!(module_with_store.var_store().variables().len() > 0);
        assert_eq!(module_with_store.var_store().variables(), loaded_module.var_store().variables());

        debug!("_test_save_load_self_attention_label_node: module_with_store.var_store().variables()={:?}", module_with_store.var_store().variables());

        if matches!(Device::cuda_if_available(), Cuda(_)) {
            let cuda_net = loaded_module.copy_to_device(1).await?;

            debug!("_test_save_load_self_attention_label_node: cuda_net.var_store().variables()={:?}", cuda_net.var_store().variables());

            for slices in all_slices {
                let mut lengths = BatchLengths::new(vec![*slices as i64; batch as usize]);
                let _ = cuda_net.tho_forward(&Tensor::rand([batch, motif_nets as i64, *slices as i64], FLOAT_CUDA).into(), Some(&mut lengths), false)?;
            }
        }

        Ok(())
    }

    #[test]
    fn test_cnn_label_node_with_mask() -> anyhow::Result<()> {
        reset_print_options();
        let motif_nets = 3;
        let params = LabelNetParams::new(motif_nets, LabelNetTypeParams::ConvNetParams(ConvNetParams::new(3, 4, 3, 10, 256, ConvNetJoint::MinMax)));

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = conv_label_node(&params)?.build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        let max_len = 32;
        let batch_vec = std::iter::once(Tensor::rand(&[motif_nets as i64, max_len], FLOAT_CPU))
            .chain((1..4).map(|i| Tensor::rand(&[motif_nets as i64, random_range(max_len / 2..max_len)], FLOAT_CPU)))
            .chain(std::iter::once(Tensor::rand(&[motif_nets as i64, max_len / 2], FLOAT_CPU)))
            .collect::<Vec<_>>();

        let batch_max_len = batch_vec.iter().map(|t| t.size()[1]).max().unwrap();
        let batch_refs_vec = batch_vec.iter().collect::<Vec<_>>();

        debug!("batch_refs_vec={batch_refs_vec:?}");

        let (batch, lengths) = pad_stack_with_mask(batch_refs_vec.as_slice(), motif_nets as i64, batch_max_len)?;
        let mut lengths = BatchLengths::new(lengths);

        debug!("batch=\n{batch} lengths={lengths:?}");
        let _ = net.tho_forward(&batch.into(), Some(&mut lengths), false)?;

        Ok(())
    }

    fn dummy_0(data: &mut [f64; 11]) -> (Tensor, f32) {
        (Tensor::from_slice(data).to_kind(Float), 0.0)
    }

    fn dummy_1(data: &mut [f64; 11]) -> (Tensor, f32) {
        (Tensor::from_slice(data).to_kind(Float) + 1.0, 1.0)
    }

    fn asc_t<const S: usize>(mut data: [f32; S]) -> Tensor {
        Tensor::from_slice((0..S).map(|i| data[i] + i as f32 * 2.).collect::<Vec<_>>().as_slice()).to_kind(Float)
    }

    fn desc_t<const S: usize>(mut data: [f32; S]) -> Tensor {
        Tensor::from_slice((0..S).rev().map(|i| data[i] + i as f32 * 2.).collect::<Vec<_>>().as_slice()).to_kind(Float)
    }

    fn dummy_labels_net_input<'a>(labels: &[bool], channels: usize, max_motif_nets: usize, slices_len: usize, tr: Option<Box<dyn Fn(&Tensor) -> Tensor + 'a>>) -> Tensor {
        //batch size x channels x max motif nets x slices_len
        let mut labels_iter = labels.iter();
        let t = Tensor::stack(
            labels
                .iter()
                .map(|label| {
                    Tensor::stack(
                        (0..channels)
                            .map(|mb| Tensor::stack((0..max_motif_nets).map(|mmn| if *label { motifs_1(slices_len) } else { motifs_0(slices_len) }).collect::<Vec<_>>().as_slice(), 0))
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

    fn dummy_motif_net_input<'a>(labels: &[bool], channels: usize, n: usize, tr: Option<Box<dyn Fn(&Tensor) -> Tensor + 'a>>) -> Tensor {
        //batch size x channels x max motif nets x slices_len
        let batch = Tensor::stack(
            labels
                .iter()
                .map(|label| Tensor::stack((0..channels).map(|channel| if *label { motifs_1(n) } else { motifs_0(n) }).collect::<Vec<_>>().as_slice(), 0))
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

    fn dummy_final_net_input<const LABELS_SIZE: usize, D: Dims>(batch_size: usize) -> (ThoTensor<D, DebugId>, Tensor) {
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
            Tensor::stack(data.iter().map(|d| d.0.shallow_clone()).collect::<Vec<_>>().as_slice(), 0).into(),
            Tensor::stack(data.iter().map(|d| d.1.shallow_clone()).collect::<Vec<_>>().as_slice(), 0),
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
        let _ = out.index_put_(&[Some(Tensor::from_slice(indices.as_slice()))], &scaled_others, false);

        out
    }

    //6 corrs 4 motif models
    //TODO remove?
    pub fn save_test_tensor(input_path: &PathBuf, name: &str) {
        fs::create_dir_all(input_path).unwrap();
        fs::write(&input_path.join(name), "lalalal").unwrap();
    }

    fn save_test_final_model(models_path: &PathBuf, features: u32) -> anyhow::Result<()> {
        fs::create_dir_all(models_path.clone())?;

        let mut vs_0 = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = super::final_net(&FinalNetParams::new(features, 100, 0, 1))?.build_tch_module(&vs_0.root())?;

        vs_0.freeze();
        debug!("save_test_final_model: features={} vs_0.variables()={:?}", features, vs_0.variables());
        vs_0.save(models_path.join("final.safetensors"))?;
        Ok(())
    }

    #[test]
    fn test_is_nonzero() -> anyhow::Result<()> {
        assert!(common_nets::is_nonzero(&Tensor::from_slice(&[0., 1., 0.]))?);
        assert!(common_nets::is_nonzero(&Tensor::stack(&[Tensor::from_slice(&[0., 1., 0.]), Tensor::from_slice(&[0., 0., 0.])], 0))?);
        assert!(!common_nets::is_nonzero(&Tensor::from_slice(&[0., 0., 0.]))?);

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
    fn row_mut<'a>(data: &'a mut [f32], motif_nets: usize, b: usize, m: usize, l: usize) -> &'a mut [f32] {
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
    pub fn gen_motifs_output(batch: usize, motif_nets: usize, l_range: RangeInclusive<usize>, label_signal: LabelSignal, similarity: f32, seed: Option<u64>) -> anyhow::Result<(Tensor, Tensor, Vec<usize>)> {
        assert_ne!(batch, 0, "batch must be > 0");
        assert_ne!(motif_nets, 0, "motif_nets must be > 0");

        let (l_lo, l_hi) = (*l_range.start(), *l_range.end());
        assert!(!(l_lo == 0 || l_hi == 0 || l_lo > l_hi), "l_range must be non-empty and >= 1");

        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => {
                let mut tr = rand::rng(); // rand 0.10
                StdRng::from_rng(&mut tr) // infallible
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

        let normal = Normal::new(base_mean as f64, noise_std as f64).context("construct normal distribution")?;

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
                    let (primary, secondary) = if y { (pos_group.as_slice(), neg_group.as_slice()) } else { (neg_group.as_slice(), pos_group.as_slice()) };

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

        let x = Tensor::from_slice(&data).reshape([batch as i64, motif_nets as i64, max_l as i64]);

        let y = Tensor::from_slice(&labels).to_kind(Kind::Float);

        Ok((x, y, lengths))
    }

    #[test]
    fn shapes_ok() {
        let batch_len = 3;
        let (x, y, lengths) = gen_motifs_output(batch_len, 5, 70..=80, LabelSignal::ByMotifs, 0.2, Some(123)).unwrap();

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

    #[test]
    fn test_inference_mode() -> anyhow::Result<()> {
        assert!(!Tensor::zeros(&[10, 100], FLOAT_CPU).is_inference());

        let _guard = InferenceModeGuard::enabled()?;

        assert!(Tensor::zeros(&[10, 100], FLOAT_CPU).is_inference());

        Ok(())
    }

    #[test]
    fn test_mix_masks_align_with_rows_2() -> anyhow::Result<()> {
        // set_max_level(Trace);
        //resource lengths (resources are the net's inputs, i.e. what needs to be classified)
        let resource_0_len = 100;
        let resource_1_len = 150;

        let num_motif_nets = 3i64;
        let motif_a_width = 20;
        let motif_b_width = 10;
        let motif_c_width = 30;

        //all motif nets are applied using same stride (TrainLabelNetsArgs/RunAllArgs .motif_slices_step )
        let stride = 2;
        //motif_a on resource_0_len produces
        let motif_a_on_resource_0_len = (resource_0_len - motif_a_width) / stride + 1; //
        let motif_b_on_resource_0_len = (resource_0_len - motif_b_width) / stride + 1; //
        let motif_c_on_resource_0_len = (resource_0_len - motif_c_width) / stride + 1; //

        let motif_a_on_resource_0_t = Tensor::ones(motif_a_on_resource_0_len, FLOAT_CPU) * 0.1; //what is returned by ApplyMotifNets::run
        let motif_b_on_resource_0_t = Tensor::ones(motif_b_on_resource_0_len, FLOAT_CPU) * 0.2; //what is returned by ApplyMotifNets::run
        let motif_c_on_resource_0_t = Tensor::ones(motif_c_on_resource_0_len, FLOAT_CPU) * 0.3; //what is returned by ApplyMotifNets::run

        let max_len_0 = *[motif_a_on_resource_0_len, motif_b_on_resource_0_len, motif_c_on_resource_0_len].iter().max().unwrap();
        let resource_0_t_after_nets_applied = Tensor::f_vstack(&[
            stretch(&motif_a_on_resource_0_t, max_len_0 as usize, 0)?, //corresponds to crates/thoosa_motifs/src/stages/apply_motif_nets.rs:525
            stretch(&motif_b_on_resource_0_t, max_len_0 as usize, 0)?, //corresponds to crates/thoosa_motifs/src/stages/apply_motif_nets.rs:525
            stretch(&motif_c_on_resource_0_t, max_len_0 as usize, 0)?, //corresponds to crates/thoosa_motifs/src/stages/apply_motif_nets.rs:525
        ])?;

        debug!("motif_a_on_resource_0_len={motif_a_on_resource_0_len} motif_b_on_resource_0_len={motif_b_on_resource_0_len} motif_c_on_resource_0_len={motif_c_on_resource_0_len}");

        let motif_a_on_resource_1_len = (resource_1_len - motif_a_width) / stride + 1; //
        let motif_b_on_resource_1_len = (resource_1_len - motif_b_width) / stride + 1; //
        let motif_c_on_resource_1_len = (resource_1_len - motif_c_width) / stride + 1; //

        let motif_a_on_resource_1_t = Tensor::ones(motif_a_on_resource_1_len, FLOAT_CPU) * 1.1; //what is returned by ApplyMotifNets::run
        let motif_b_on_resource_1_t = Tensor::ones(motif_b_on_resource_1_len, FLOAT_CPU) * 1.2; //what is returned by ApplyMotifNets::run
        let motif_c_on_resource_1_t = Tensor::ones(motif_c_on_resource_1_len, FLOAT_CPU) * 1.3; //what is returned by ApplyMotifNets::run

        let max_len_1 = *[motif_a_on_resource_1_len, motif_b_on_resource_1_len, motif_c_on_resource_1_len].iter().max().unwrap();
        let resource_1_t_after_nets_applied = Tensor::f_vstack(&[
            stretch(&motif_a_on_resource_1_t, max_len_1 as usize, 0)?, //corresponds to crates/thoosa_motifs/src/stages/apply_motif_nets.rs:525
            stretch(&motif_b_on_resource_1_t, max_len_1 as usize, 0)?, //corresponds to crates/thoosa_motifs/src/stages/apply_motif_nets.rs:525
            stretch(&motif_c_on_resource_1_t, max_len_1 as usize, 0)?, //corresponds to crates/thoosa_motifs/src/stages/apply_motif_nets.rs:525
        ])?;

        debug!("motif_a_on_resource_1_len={motif_a_on_resource_1_len} motif_b_on_resource_1_len={motif_b_on_resource_1_len} motif_c_on_resource_1_len={motif_c_on_resource_1_len}");

        debug!("resource_0_t_after_nets_applied=\n{resource_0_t_after_nets_applied}");
        debug!("resource_1_t_after_nets_applied=\n{resource_1_t_after_nets_applied}");
        debug!("resource_0_t_after_nets_applied.size={:?}, resource_1_t_after_nets_applied.size={:?}", resource_0_t_after_nets_applied.size(), resource_1_t_after_nets_applied.size());

        // TrainLabelNets, ResourcesBatch
        let max_input = max(resource_0_t_after_nets_applied.size()[1], resource_1_t_after_nets_applied.size()[1]);
        debug!("max_input={max_input}");

        let (x_0_batch, lengths_0) = pad_stack_with_mask(&[&resource_0_t_after_nets_applied], num_motif_nets, max_input)?;
        let (x_1_batch, lengths_1) = pad_stack_with_mask(&[&resource_1_t_after_nets_applied], num_motif_nets, max_input)?;

        let lengths_0 = BatchLengths::new(lengths_0);
        let lengths_1 = BatchLengths::new(lengths_1);

        debug!("max_input={max_input}");
        let (x, mut lengths, y) = mix_x_binary_with_labels(&x_1_batch, Some(&lengths_1), &x_0_batch, Some(&lengths_0))?;

        let idx_0 = y.eq(0.0).nonzero().double_value(&[0, 0]) as i64;
        let idx_1 = idx_0 ^ 1;

        debug!("x=\n{x}");
        debug!("y=\n{y}");
        debug!("lengths={lengths:?}");
        debug!("idx_0={idx_0} idx_1={idx_1}");

        let self_attention_net_params = SelfAttentionNetParams::new(64, 4, 4, 0.1);

        let params = LabelNetParams::new(3, LabelNetTypeParams::SelfAttentionNetParams(self_attention_net_params));

        let mut vs_0 = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = self_attention_label_node(&params)?.build_tch_module(&vs_0.root())?;
        vs_0.freeze();

        let _ = net.tho_forward(&x.into(), lengths.as_mut(), false)?;

        Ok(())
    }
}
