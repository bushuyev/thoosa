use thoosa_common::tch_error::tch_result;
use clap::ValueEnum;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use thoosa_common::stages::convert_precision::{
    ConvertPrecisionArgs as CommonConvertPrecisionArgs,
    ConvertPrecisionBy as CommonConvertPrecisionBy,
    ConvertPrecisionOutput as CommonConvertPrecisionOutput,
    ConvertPrecisionStage as CommonConvertPrecisionStage,
    ConvertedNet as CommonConvertedNet,
    PrecisionModelType,
};
use thoosa_common::ws::NET_WEIGHTS_FILE;
use thoosa_shared::ThoError;
use thoosa_shared::run_configs::RunConfigId;

use crate::motifs_ws::MotifsWorkspace;

pub use thoosa_common::stages::convert_precision::{ConvertPrecisionInput, ConvertPrecisionTarget};

/// Motif model categories remain here because only `MotifsWorkspace` knows their artifact layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum)]
pub enum ConvertPrecisionNetType {
    Motifs,
    Labels,
    Final,
    All,
}

impl FromStr for ConvertPrecisionNetType {
    type Err = anyhow::Error;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        match raw.trim().trim_matches(|ch| ch == '\'' || ch == '"').to_ascii_lowercase().as_str() {
            "motif" | "motifs" => Ok(Self::Motifs),
            "label" | "labels" => Ok(Self::Labels),
            "final" => Ok(Self::Final),
            "all" => Ok(Self::All),
            other => anyhow::bail!("unexpected net type {other:?}; expected motifs, labels, final, or all"),
        }
    }
}

impl PrecisionModelType<dyn MotifsWorkspace> for ConvertPrecisionNetType {
    fn all() -> &'static [Self] {
        &[Self::Motifs, Self::Labels, Self::Final]
    }

    fn concrete(self) -> &'static [Self] {
        match self {
            Self::Motifs => &[Self::Motifs],
            Self::Labels => &[Self::Labels],
            Self::Final => &[Self::Final],
            Self::All => Self::all(),
        }
    }

    fn root(self, ws: &dyn MotifsWorkspace) -> anyhow::Result<PathBuf> {
        match self {
            Self::Motifs => Ok(ws.motifs_nets_path()?),
            Self::Labels => Ok(ws.labels_nets_path()?),
            Self::Final => ws.final_nets_path(),
            Self::All => Ok(ws.models_path().clone()),
        }
    }

    fn run_config_id(self, weights_path: &Path) -> Option<RunConfigId> {
        let base_path = if weights_path.file_name().and_then(|v| v.to_str()) == Some(NET_WEIGHTS_FILE) {
            weights_path.parent()?
        } else {
            weights_path
        };

        match self {
            Self::Motifs => base_path
                .file_stem()
                .and_then(|v| v.to_str())
                .and_then(|v| v.rsplit_once('_').map(|(_, run_config_id)| run_config_id).or(Some(v)))
                .and_then(|v| v.parse::<RunConfigId>().ok()),
            Self::Labels | Self::Final => base_path.file_stem().and_then(|v| v.to_str()).and_then(|v| v.parse::<RunConfigId>().ok()),
            Self::All => None,
        }
    }
}

pub type ConvertPrecisionBy = CommonConvertPrecisionBy<ConvertPrecisionNetType>;
pub type ConvertPrecisionArgs = CommonConvertPrecisionArgs<ConvertPrecisionNetType>;
pub type ConvertedNet = CommonConvertedNet<ConvertPrecisionNetType>;
pub type ConvertPrecisionOutput = CommonConvertPrecisionOutput<ConvertPrecisionNetType>;
pub type ConvertPrecisionStage<C> = CommonConvertPrecisionStage<C, dyn MotifsWorkspace, ConvertPrecisionNetType>;

#[cfg(test)]
mod tests {
    use thoosa_common::tch_error::tch_result;
    use super::{ConvertPrecisionArgs, ConvertPrecisionBy, ConvertPrecisionInput, ConvertPrecisionNetType, ConvertPrecisionStage, ConvertPrecisionTarget};
    use crate::train_nets::{FinalNetParams, LabelNetParams, LabelNetTypeParams, final_net, lin_label_node};
    use parking_lot::RwLock;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};
    use thoosa_common::dims::AnyDims;
    use thoosa_common::nets::common_nets::LinLenNetParams;
    use thoosa_common::nets::nodes::ThoModule;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::run_ctx::RunCtxImpl;
    use thoosa_common::ws::WsConfigs;
    use thoosa_shared::traits::AsyncStage;
    use thoosa_shared::{ThoError, cdt_for_dir};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_convert_precision_stage_converts_saved_final_net() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        run_convert_precision_stage_final_net(ConvertPrecisionTarget::Fp64, Kind::Double).await?;
        run_convert_precision_stage_final_net(ConvertPrecisionTarget::Bf16, Kind::BFloat16).await?;
        run_convert_precision_stage_label_net(ConvertPrecisionTarget::Bf16, Kind::BFloat16).await?;

        Ok(())
    }

    async fn run_convert_precision_stage_final_net(precision_target: ConvertPrecisionTarget, kind: Kind) -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(format!("convert_precision_{}", cdt_for_dir()));
        let ws_box = crate::motifs_ws::new_motifs_workspace(test_dir, WsConfigs::new(2, 100, 100))?;

        let params = FinalNetParams::new(8, 100, 2, 1);
        let mut vs = VarStore::new(Device::Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = final_net(&params)?.build_tch_module(&vs.root())?;
        vs.freeze();
        let net_id = ws_box.save_final_net(3, (&vs).into(), &params)?;

        let ws = Arc::new(RwLock::new(ws_box));
        let args = ConvertPrecisionArgs::new(ConvertPrecisionBy::Path(PathBuf::from("final").join(&net_id)), precision_target);
        let stage = Arc::new(ConvertPrecisionStage::<RunCtxImpl>::new(&ws, args));
        let output = stage
            .run(
                ConvertPrecisionInput::new(),
                RunCtxImpl::new(0, "convert_precision_test"),
            )
            .await?;
        let output = Arc::into_inner(output).expect("stage output should not be shared");
        assert_eq!(output.converted().len(), 1);

        let (_, loaded_module) = ws.read().load_final_net(&net_id, &Device::Cpu)?;
        let variables = loaded_module.var_store().variables();
        assert!(!variables.is_empty());
        for tensor in variables.values() {
            assert_eq!(tensor.kind(), kind);
        }
        Ok(())
    }

    async fn run_convert_precision_stage_label_net(precision_target: ConvertPrecisionTarget, kind: Kind) -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(format!("convert_precision_{}", cdt_for_dir()));
        let ws_box = crate::motifs_ws::new_motifs_workspace(test_dir, WsConfigs::new(2, 100, 100))?;

        let params = LabelNetParams::new(2, LabelNetTypeParams::LinNetParams(LinLenNetParams{ layers: 2, features: 10 }));
        let mut vs = VarStore::new(Device::Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = lin_label_node(&params)?.build_tch_module(&vs.root())?;
        vs.freeze();
        let net_id = ws_box.save_label_net(3, &"label_a", (&vs).into(), &params)?;

        let ws = Arc::new(RwLock::new(ws_box));
        let args = ConvertPrecisionArgs::new(ConvertPrecisionBy::Path(PathBuf::from(&net_id)), precision_target);
        let stage = Arc::new(ConvertPrecisionStage::<RunCtxImpl>::new(&ws, args));
        let output = stage
            .run(
                ConvertPrecisionInput::new(),
                RunCtxImpl::new(0, "convert_precision_test"),
            )
            .await?;
        let output = Arc::into_inner(output).expect("stage output should not be shared");
        assert_eq!(output.converted().len(), 1);

        let (_, params, loaded_module) = ws.read().load_label_net(&net_id, &Device::Cpu, false, true)?;
        let variables = loaded_module.expect("loaded_module").var_store().variables();
        assert!(!variables.is_empty());
        for tensor in variables.values() {
            assert_eq!(tensor.kind(), kind);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_convert_precision_stage_preserves_existing_tensor_names() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let test_dir = Path::new("/tmp").join(format!("convert_precision_{}", cdt_for_dir()));
        let ws_box = crate::motifs_ws::new_motifs_workspace(test_dir, WsConfigs::new(2, 100, 100))?;
        let net_dir = ws_box.motifs_nets_path()?.join("0").join("34603_36472");
        std::fs::create_dir_all(&net_dir)?;
        std::fs::write(net_dir.join("parameters.json"), "{}")?;
        let weights_path = net_dir.join("weights.safetensors");
        let weight = Tensor::randn([1, 1, 4], (Kind::Float, Device::Cpu));
        let bias = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        tch_result(Tensor::write_safetensors(
            &[
                ("seq_4.motif_0_conv.weight", &weight),
                ("seq_4.motif_0_conv.bias", &bias),
            ],
            &weights_path,
        ))?;

        let ws = Arc::new(RwLock::new(ws_box));
        let args = ConvertPrecisionArgs::new(ConvertPrecisionBy::Type(ConvertPrecisionNetType::Motifs), ConvertPrecisionTarget::Fp16);
        let stage = Arc::new(ConvertPrecisionStage::<RunCtxImpl>::new(&ws, args));
        let output = stage
            .run(
                ConvertPrecisionInput::new(),
                RunCtxImpl::new(0, "convert_precision_names_test"),
            )
            .await?;
        let output = Arc::into_inner(output).expect("stage output should not be shared");
        assert_eq!(output.converted().len(), 1);

        let converted = tch_result(Tensor::read_safetensors(&weights_path))?;
        assert!(converted.iter().any(|(name, _)| name == "seq_4.motif_0_conv.weight"));
        for (_, tensor) in converted {
            assert_eq!(tensor.kind(), Kind::Half);
        }

        Ok(())
    }
}
