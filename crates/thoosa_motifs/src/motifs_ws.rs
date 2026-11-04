use anyhow::Context as _;
use crate::dims::{ChannelsMotifsData, LabelProbability, LabelsProbabilities, MotifProbability};
use crate::train_nets::{FinalNetParams, LabelNetParams, MotifNetParams, final_net};
use log::{debug, info};
use mockall::mock;
use std::collections::{HashMap, HashSet};
use std::fmt::Display;
use std::fs;
use std::path::{Path, PathBuf};
use tch::nn;
use tch::{Device, Tensor};
use thoosa_common::dims::{Batch, Slice};
use thoosa_common::nets::common_nets::ThoModuleWithStore;
use thoosa_common::ws::{generation_weights, FsWorkspace, VarStoreOrPath, Workspace, WsConfigs, NET_PARAMS_FILE, NET_WEIGHTS_FILE, publish_net, resolve_net_base_path, net_files_from_base, nested_weights, discover_nets, published_generation};
use thoosa_shared::resource::{Resource, ResourceStats};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::traits::ToJsonFile;
use thoosa_shared::{cdt_for_dir, mono_key, recurse_files};

const MOTIFS_MODELS: &str = "motifs";
const LABELS_MODELS: &str = "labels";
const FINAL_MODELS: &str = "final";

/// Workspace boundary for model artifacts owned by the motif-search workflow.
pub trait MotifsWorkspace: Workspace {
    fn motifs_nets_path(&self) -> anyhow::Result<PathBuf>;
    fn labels_nets_path(&self) -> anyhow::Result<PathBuf>;
    fn final_nets_path(&self) -> anyhow::Result<PathBuf>;
    fn save_final_net<'a>(&self, run_config_id: RunConfigId, vs_or_path: VarStoreOrPath<'a>, params: &FinalNetParams) -> anyhow::Result<PathBuf>;
    fn save_motif_net<'a>(&self, channel: &dyn Display, train_motif_net_run_config_id: &dyn Display, motif_id: &dyn Display, vs_or_path: VarStoreOrPath<'a>, params: &MotifNetParams) -> anyhow::Result<PathBuf>;
    fn load_motif_net(&self, net_path: &Path, device: &Device, quantize: bool, load_net: bool) -> anyhow::Result<(MotifNetParams, Option<ThoModuleWithStore<Batch<Slice>, MotifProbability>>)>;
    fn load_label_net(&self, net_path: &Path, device: &Device, quantize: bool, load_net: bool) -> anyhow::Result<(RunConfigId, LabelNetParams, Option<ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>>)>;
    fn load_final_net(&self, net_path: &Path, device: &Device) -> anyhow::Result<(FinalNetParams, ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>)>;
    fn load_motif_nets(&self, channel: u32, motif_id: u32, device: &Device) -> anyhow::Result<Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>>;
    fn load_label_nets<'a>(&self, train_label_nets_run_config_ids: Option<&'a [RunConfigId]>, device: &'a Device, quantize: bool, load_net: bool) -> anyhow::Result<HashMap<String, Vec<(RunConfigId, LabelNetParams, Option<ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>>)>>>;
    fn motif_models(&self, channel: u32, motif_id: u32) -> anyhow::Result<Vec<(PathBuf, PathBuf)>>;
    fn save_label_net<'a>(&self, label_node_run_config_id: u32, label: &dyn Display, vs_or_path: VarStoreOrPath<'a>, params: &LabelNetParams) -> anyhow::Result<PathBuf>;
    fn label_models(&self) -> anyhow::Result<HashMap<String, Vec<(PathBuf, PathBuf)>>>;
    fn final_model(&self, id: RunConfigId) -> anyhow::Result<(PathBuf, PathBuf)>;
}

/// Constructs the motif-aware trait object without making the common crate depend on this extension.
pub fn new_motifs_workspace(base_path: impl Into<PathBuf>, ws_configs: WsConfigs) -> anyhow::Result<Box<dyn MotifsWorkspace>> {
    Ok(Box::new(FsWorkspace::new_concrete(base_path, ws_configs)?))
}

/// Constructs the motif-aware trait object while keeping model artifacts in an external directory.
pub fn new_motifs_workspace_with_models_dir(base_path: impl Into<PathBuf>, app_configs: &WsConfigs, models_dir: &String) -> anyhow::Result<Box<dyn MotifsWorkspace>> {
    Ok(Box::new(FsWorkspace::new_concrete_with_models_dir(base_path, app_configs, models_dir)?))
}

pub fn net_path_to_run_config_id(model_path: &Path) -> RunConfigId {
    model_path
        .file_stem()
        .and_then(|s| s.to_str().and_then(|v| v.parse::<RunConfigId>().ok()))
        .or_else(|| model_path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str().and_then(|v| v.parse::<RunConfigId>().ok())))
        .unwrap_or_else(|| panic!("can't parse run_config_id from {model_path:?}"))
}

pub fn parse_motif_path(net_id: &Path) -> anyhow::Result<(u32, u32)> {
    let base_dir = if generation_weights(net_id) || matches!(net_id.file_name().and_then(|v| v.to_str()), Some(NET_WEIGHTS_FILE | NET_PARAMS_FILE)) {
        net_id.parent().ok_or_else(|| anyhow::anyhow!("parse_motif_path: no parent for {net_id:?}"))?
    } else {
        net_id
    };
    let channel = base_dir.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).ok_or_else(|| anyhow::anyhow!("parse_motif_path: no channel in {net_id:?}"))?.parse::<u32>().with_context(|| format!("parse_motif_path: bad channel in {net_id:?}"))?;
    let motif_part = base_dir.file_name().and_then(|s| s.to_str()).ok_or_else(|| anyhow::anyhow!("parse_motif_path: no motif id in {net_id:?}"))?;
    let motif_id = motif_part.split('_').next().unwrap_or(motif_part).parse::<u32>().with_context(|| format!("parse_motif_path: bad motif id in {net_id:?}"))?;
    Ok((channel, motif_id))
}






impl MotifsWorkspace for FsWorkspace {
    fn motifs_nets_path(&self) -> anyhow::Result<PathBuf> { self.model_type_path(MOTIFS_MODELS) }
    fn labels_nets_path(&self) -> anyhow::Result<PathBuf> { self.model_type_path(LABELS_MODELS) }
    fn final_nets_path(&self) -> anyhow::Result<PathBuf> { self.model_type_path(FINAL_MODELS) }

    fn save_final_net<'a>(&self, run_config_id: RunConfigId, vs_or_path: VarStoreOrPath<'a>, params: &FinalNetParams) -> anyhow::Result<PathBuf> {
        let final_models_path = self.final_nets_path()?;
        debug!("FsWorkspace::save_final_net: run_config_id={run_config_id} params={params:?} final_models_path={final_models_path:?}");
        let net_id = PathBuf::from(format!("{run_config_id}"));
        let base_path = final_models_path.join(&net_id);
        publish_net(&base_path, &serde_json5::to_string(params)?, |path| vs_or_path.save(&path.to_path_buf()))?;
        Ok(net_id)
    }

    fn save_motif_net<'a>(&self, channel: &dyn Display, train_motif_net_run_config_id: &dyn Display, motif_id: &dyn Display, vs_or_path: VarStoreOrPath<'a>, params: &MotifNetParams) -> anyhow::Result<PathBuf> {
        let channel_part = format!("{channel}");
        let channel_models = self.motifs_nets_path()?.join(&channel_part);
        fs::create_dir_all(&channel_models)?;
        let net_dir_name = format!("{motif_id}_{train_motif_net_run_config_id}");
        let net_id = PathBuf::from(&channel_part).join(&net_dir_name);
        let base_path = channel_models.join(&net_dir_name);
        fs::create_dir_all(&base_path)?;
        let vs_path = base_path.join(NET_WEIGHTS_FILE);
        debug!("FsWorkspace::save_motif_net: motif_id={motif_id} motif_model_run_config_id={train_motif_net_run_config_id} saved to {vs_path:?}");
        publish_net(&base_path, &serde_json5::to_string(params)?, |path| vs_or_path.save(&path.to_path_buf()))?;
        Ok(net_id)
    }

    fn load_motif_net(&self, net_path: &Path, device: &Device, quantize: bool, load_net: bool) -> anyhow::Result<(MotifNetParams, Option<ThoModuleWithStore<Batch<Slice>, MotifProbability>>)> {
        let base_path = resolve_net_base_path(&self.motifs_nets_path()?, net_path);
        let (model_path, params_path) = net_files_from_base(&base_path)?;
        let params = MotifNetParams::from_file(&params_path)?;
        let params_for_builder = params.clone();
        let module = load_net
            .then(|| ThoModuleWithStore::load(
                mono_key!(model_path.to_string_lossy().to_string(), "motif_model").into(),
                move |path: &nn::Path| params_for_builder.motif_net(None)?.build_tch_module(path), model_path, *device, quantize
            ))
            .transpose()?;
        Ok((params, module))
    }

    fn load_label_net(&self, net_path: &Path, device: &Device, quantize: bool, load_net: bool) -> anyhow::Result<(RunConfigId, LabelNetParams, Option<ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>>)> {
        let base_path = resolve_net_base_path(&self.labels_nets_path()?, net_path);
        let (model_path, params_path) = net_files_from_base(&base_path)?;
        // Keep path validation in release builds even though only debug keys retain the label.
        let _label = if nested_weights(&model_path) { model_path.parent().and_then(|p| p.parent()).and_then(|p| p.file_name()).and_then(|s| s.to_str()) } else { model_path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()) }.ok_or_else(|| anyhow::anyhow!("load_label_net: can't parse label from {base_path:?}"))?;
        let run_config_id = net_path_to_run_config_id(&model_path);
        let params = LabelNetParams::from_file(&params_path)?;
        let params_for_builder = params.clone();
        let module = load_net.then(|| ThoModuleWithStore::load(mono_key!(run_config_id, format!("Label_model_{_label}_{run_config_id}")).into(), move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder)?.build_tch_module(path), model_path.clone(), *device, quantize)).transpose()?;
        Ok((run_config_id, params, module))
    }

    fn load_final_net(&self, net_path: &Path, device: &Device) -> anyhow::Result<(FinalNetParams, ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>)> {
        let base_path = resolve_net_base_path(&self.final_nets_path()?, net_path);
        let (model_path, params_path) = net_files_from_base(&base_path)?;
        let run_config_id = net_path_to_run_config_id(&model_path);
        let params = FinalNetParams::from_file(&params_path)?;
        let params_for_builder = params.clone();
        // This legacy loader exposes only its supported unquantized representation.
        let module = ThoModuleWithStore::load(mono_key!(run_config_id, "final_model").into(), move |path: &nn::Path| final_net(&params_for_builder)?.build_tch_module(path), model_path, *device, false)?;
        Ok((params, module))
    }

    fn load_motif_nets(&self, channel: u32, motif_id: u32, device: &Device) -> anyhow::Result<Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>> {
        debug!("load_motif_nets: channel={channel} motif_id={motif_id}");
        self.motif_models(channel, motif_id)?.into_iter().map(|(model, config)| {
            let params = MotifNetParams::from_file(&config)?;
            let params_for_builder = params.clone();
            // Quantized loading is available through the single-model loader instead.
            let module = ThoModuleWithStore::load(mono_key!("MotifNet", "MotifNet") + &mono_key!(motif_id, "motif_id"), move |path: &nn::Path| params_for_builder.motif_net(None)?.build_tch_module(path), model, *device, false)?;
            Ok((params, module))
        }).collect()
    }

    fn load_label_nets<'a>(&self, train_label_nets_run_config_ids: Option<&'a [RunConfigId]>, device: &'a Device, quantize: bool, load_net: bool) -> anyhow::Result<HashMap<String, Vec<(RunConfigId, LabelNetParams, Option<ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>>)>>> {
        let allowed = train_label_nets_run_config_ids.map(|ids| ids.iter().copied().collect::<HashSet<_>>());
        self.label_models()?.into_iter().map(|(label, paths)| {
            let models = paths.into_iter().filter_map(|(model_path, params_path)| {
                let run_config_id = net_path_to_run_config_id(&model_path);
                if allowed.as_ref().is_some_and(|ids| !ids.contains(&run_config_id)) { return None; }
                Some((|| {
                    let params = LabelNetParams::from_file(&params_path).with_context(|| format!("can't read label net params from {params_path:?}"))?;
                    let params_for_builder = params.clone();
                    let module = load_net.then(|| ThoModuleWithStore::load(mono_key!(run_config_id, format!("Label_model_{label}_{run_config_id}")).into(), move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder)?.build_tch_module(path), model_path, *device, quantize)).transpose()?;
                    Ok((run_config_id, params, module))
                })())
            }).collect::<anyhow::Result<Vec<_>>>()?;
            Ok((label, models))
        }).collect()
    }

    fn motif_models(&self, channel: u32, motif_id: u32) -> anyhow::Result<Vec<(PathBuf, PathBuf)>> {
        let motif_prefix = format!("{motif_id}_");
        Ok(discover_nets(&self.motifs_nets_path()?.join(channel.to_string()))?.into_iter().filter(|(path, _)| {
            if nested_weights(path) {
                path.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()).is_some_and(|s| s.starts_with(&motif_prefix))
            } else { path.file_stem().and_then(|s| s.to_str()).is_some_and(|s| s.starts_with(&motif_prefix)) }
        }).collect())
    }

    fn save_label_net<'a>(&self, label_node_run_config_id: u32, label: &dyn Display, vs_or_path: VarStoreOrPath<'a>, params: &LabelNetParams) -> anyhow::Result<PathBuf> {
        let label_part = format!("{label}");
        let label_models_path = self.labels_nets_path()?.join(&label_part);
        fs::create_dir_all(&label_models_path)?;
        let net_dir_name = format!("{label_node_run_config_id}");
        let net_id = PathBuf::from(&label_part).join(&net_dir_name);
        let base_path = label_models_path.join(&net_dir_name);
        fs::create_dir_all(&base_path)?;
        let vs_path = base_path.join(NET_WEIGHTS_FILE);
        info!("save_label_net: run_config_id={label_node_run_config_id} label={label} saved to {vs_path:?}");
        publish_net(&base_path, &serde_json5::to_string(params)?, |path| vs_or_path.save(&path.to_path_buf()))?;
        Ok(net_id)
    }

    fn label_models(&self) -> anyhow::Result<HashMap<String, Vec<(PathBuf, PathBuf)>>> {
        discover_nets(&self.labels_nets_path()?)?.into_iter().try_fold(HashMap::new(), |mut models, (weights_path, params_path)| {
            let parent = weights_path.parent().context("model has no parent")?;
            let label_dir = if nested_weights(&weights_path) { parent.parent().context("model has no label")? } else { parent };
            // Labels are directory names, not filenames with removable extensions.
            let label = label_dir.file_name().and_then(|name| name.to_str()).with_context(|| format!("invalid UTF-8 label directory: {}", label_dir.display()))?;
            models.entry(label.to_string()).or_insert_with(Vec::new).push((weights_path, params_path));
            Ok(models)
        })
    }

    fn final_model(&self, id: RunConfigId) -> anyhow::Result<(PathBuf, PathBuf)> {
        let final_models_path = self.final_nets_path()?;
        let base_path = final_models_path.join(format!("{id}"));
        if let Some(pair) = published_generation(&base_path)? { return Ok(pair); }
        let nested = (base_path.join(NET_WEIGHTS_FILE), base_path.join(NET_PARAMS_FILE));
        if nested.0.exists() || nested.1.exists() { return Ok(nested); }
        Ok((final_models_path.join(format!("{id}.safetensors")), final_models_path.join(format!("{id}.json"))))
    }
}

/// Loads a final model through the motif-aware workspace boundary.
pub fn load_final_model(run_config_id: RunConfigId, ws: &parking_lot::RwLock<Box<dyn MotifsWorkspace>>, device: &Device) -> anyhow::Result<(FinalNetParams, ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>)> {
    let (model_path, params_path) = ws.read().final_model(run_config_id)?;
    let params = FinalNetParams::from_file(&params_path)?;
    let params_for_builder = params.clone();
    let module = ThoModuleWithStore::load(mono_key!(run_config_id, "final_model").into(), move |path: &nn::Path| final_net(&params_for_builder)?.build_tch_module(path), model_path, *device, false)?;
    Ok((params, module))
}

mock! {
    #[derive(Debug)]
    pub MotifsWorkspace {}

    impl MotifsWorkspace for MotifsWorkspace {
        fn motifs_nets_path(&self) -> anyhow::Result<PathBuf>;
        fn labels_nets_path(&self) -> anyhow::Result<PathBuf>;
        fn final_nets_path(&self) -> anyhow::Result<PathBuf>;
        fn save_final_net<'a>(&self, run_config_id: RunConfigId, vs_or_path: VarStoreOrPath<'a>, params: &FinalNetParams) -> anyhow::Result<PathBuf>;
        fn save_motif_net<'a>(&self, channel: &dyn Display, train_motif_net_run_config_id: &dyn Display, motif_id: &dyn Display, vs_or_path: VarStoreOrPath<'a>, params: &MotifNetParams) -> anyhow::Result<PathBuf>;
        fn load_motif_net(&self, net_path: &Path, device: &Device, quantize: bool, load_net: bool) -> anyhow::Result<(MotifNetParams, Option<ThoModuleWithStore<Batch<Slice>, MotifProbability>>)>;
        fn load_label_net(&self, net_path: &Path, device: &Device, quantize: bool, load_net: bool) -> anyhow::Result<(RunConfigId, LabelNetParams, Option<ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>>)>;
        fn load_final_net(&self, net_path: &Path, device: &Device) -> anyhow::Result<(FinalNetParams, ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>)>;
        fn load_motif_nets(&self, channel: u32, motif_id: u32, device: &Device) -> anyhow::Result<Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>>;
        fn load_label_nets<'a>(&self, train_label_nets_run_config_ids: Option<&'a [RunConfigId]>, device: &'a Device, quantize: bool, load_net: bool) -> anyhow::Result<HashMap<String, Vec<(RunConfigId, LabelNetParams, Option<ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>>)>>>;
        fn motif_models(&self, channel: u32, motif_id: u32) -> anyhow::Result<Vec<(PathBuf, PathBuf)>>;
        fn save_label_net<'a>(&self, label_node_run_config_id: u32, label: &dyn Display, vs_or_path: VarStoreOrPath<'a>, params: &LabelNetParams) -> anyhow::Result<PathBuf>;
        fn label_models(&self) -> anyhow::Result<HashMap<String, Vec<(PathBuf, PathBuf)>>>;
        fn final_model(&self, id: RunConfigId) -> anyhow::Result<(PathBuf, PathBuf)>;
    }

    impl Workspace for MotifsWorkspace {
        fn import_tensor<'a>(&self, t: Tensor, import_id: u32, key: &'a str) -> anyhow::Result<ResourceStats>;
        fn base_path_as_str(&self) -> String;
        fn base_path(&self) -> &PathBuf;
        fn store_path(&self) -> &PathBuf;
        fn log_path(&self) -> &PathBuf;
        fn tmp_path(&self) -> &PathBuf;
        fn tmp_run_store_path(&self, run_config_id: RunConfigId, channel: u32) -> PathBuf;
        fn tensors_path(&self) -> &PathBuf;
        fn tensor_path(&self, key: &Resource) -> PathBuf;
        fn models_path(&self) -> &PathBuf;
        fn plots_path(&self) -> &PathBuf;
        fn save_net<'a>(&self, run_config_id: RunConfigId, vs_or_path: VarStoreOrPath<'a>, model_type: &str, params: &dyn ToJsonFile) -> anyhow::Result<PathBuf>;
        fn channels(&self) -> u32;
        fn sample_rate(&self) -> u32;
        fn max_input(&self) -> u32;
        fn clc_records(&self, resource_ids: &Vec<String>) -> anyhow::Result<i64>;
        fn ws_configs(&self)->WsConfigs;
    }
}

pub fn mock_motifs_ws_base(path: Option<&PathBuf>, channels: u32, sample_rate: u32, max_input: u32) -> MockMotifsWorkspace {
    let base = path.cloned().unwrap_or_else(|| {
        let path = Path::new("/tmp").join(cdt_for_dir());
        fs::create_dir_all(&path).expect("can't create tmp dir");
        path
    });
    let mut ws = MockMotifsWorkspace::new();
    ws.expect_base_path().return_const(base.clone());
    ws.expect_store_path().return_const(base.join("main_db.db3"));
    ws.expect_log_path().return_const(base.clone());
    ws.expect_tmp_path().return_const(base.clone());
    ws.expect_tensors_path().return_const(base.clone());
    ws.expect_models_path().return_const(base.clone());
    ws.expect_plots_path().return_const(base.clone());
    ws.expect_sample_rate().return_const(sample_rate);
    ws.expect_channels().return_const(channels);
    ws.expect_max_input().return_const(max_input);
    ws.expect_ws_configs().return_const(WsConfigs::new(channels, max_input, sample_rate));
    ws.expect_import_tensor().returning(|_, _, _|Ok(ResourceStats::new(vec![20, 100])));
    ws
}

/// Creates the common motif-workspace mock defaults used by stage tests.
pub fn mock_motifs_ws_simp(path: Option<&PathBuf>, channels: u32, sample_rate: u32, max_input: u32) -> Box<dyn MotifsWorkspace> {
    Box::new(mock_motifs_ws_base(path, channels, sample_rate, max_input))
}

pub struct MotifsWsMockBuilder {
    mock: MockMotifsWorkspace,
}

impl MotifsWsMockBuilder {
    pub fn new(path: Option<&PathBuf>, channels: u32, sample_rate: u32, max_input: u32) -> Self {
        Self { mock: mock_motifs_ws_base(path, channels, sample_rate, max_input) }
    }

    #[deprecated]
    pub fn with_load_motif_nets(mut self, channels_motif_nets: Vec<Vec<(Box<dyn Fn() -> MotifNetParams + Sync + Send + 'static>, Box<dyn Fn() -> anyhow::Result<ThoModuleWithStore<Batch<Slice>, MotifProbability>> + Sync + Send + 'static>)>>) -> Self {
        let channels_motif_nets = std::sync::Arc::new(channels_motif_nets);
        self.mock.expect_load_motif_nets().returning(move |channel, motif_id, _device| {
            let entry = &channels_motif_nets[channel as usize][motif_id as usize];
            Ok(vec![((entry.0)(), (entry.1)()?)])
        });
        self
    }

    pub fn with_load_motif_net(self, nets: Vec<(PathBuf, Box<dyn Fn() -> MotifNetParams + Sync + Send + 'static>, Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send + 'static>)>) -> Self {
        self.with_try_load_motif_net(nets.into_iter().map(|(path, params, module)| (path, params, Box::new(move || Ok(module())) as Box<dyn Fn() -> anyhow::Result<ThoModuleWithStore<Batch<Slice>, MotifProbability>> + Send + Sync>)).collect())
    }

    // Tensor-backed fixtures can fail to acquire local storage; pass that failure
    // through the mocked workspace instead of panicking inside a model factory.
    pub fn with_try_load_motif_net(mut self, nets: Vec<(PathBuf, Box<dyn Fn() -> MotifNetParams + Sync + Send + 'static>, Box<dyn Fn() -> anyhow::Result<ThoModuleWithStore<Batch<Slice>, MotifProbability>> + Sync + Send + 'static>)>) -> Self {
        let nets = std::sync::Arc::new(nets);
        self.mock.expect_load_motif_net().returning(move |net_path, _device, _quantize, _load_net| {
            let entry = nets.iter().find(|(path, _, _)| path == net_path).ok_or_else(|| anyhow::anyhow!("Not found: {net_path:?}"))?;
            // Preserve the previous WsMockBuilder contract, which always returned configured modules.
            Ok(((entry.1)(), Some((entry.2)()?)))
        });
        self
    }

    pub fn with_load_label_nets(mut self, label_nets: HashMap<String, Vec<(RunConfigId, Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>, Box<dyn Fn() -> ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>> + Sync + Send + 'static>)>>) -> Self {
        self.mock.expect_load_label_nets().returning(move |_ids, _device, _quantize, _load_net| {
            // Preserve the previous WsMockBuilder contract, which always returned configured modules.
            Ok(label_nets.iter().map(|(label, nets)| (label.clone(), nets.iter().map(|(id, params, module)| (*id, params(), Some(module()))).collect())).collect())
        });
        self
    }

    pub fn with_load_label_net(mut self, nets: Vec<(PathBuf, RunConfigId, Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>, Box<dyn Fn() -> ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>> + Sync + Send + 'static>)>) -> Self {
        // Individual inference stages load exact checkpoints, so mocks preserve path-sensitive lookup.
        let nets = std::sync::Arc::new(nets);
        self.mock.expect_load_label_net().returning(move |net_path, _device, _quantize, _load_net| {
            let entry = nets.iter().find(|(path, _, _, _)| path == net_path).ok_or_else(|| anyhow::anyhow!("Not found: {net_path:?}"))?;
            Ok((entry.1, (entry.2)(), Some((entry.3)())))
        });
        self
    }

    pub fn build(self) -> Box<dyn MotifsWorkspace> { Box::new(self.mock) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train_nets::{FinalNetParams, LabelNetTypeParams, MotifNetTypeParams, conv_motif_node, cosine_motif_node};
    use std::path::Path;
    use std::sync::Arc;
    use tch::Device::Cpu;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;
    use tch::nn::VarStore;
    use thoosa_common::dims::AnyDims;
    use thoosa_common::nets::common_nets::LinLenNetParams;
    use thoosa_common::nets::nodes::{InitDummy, InitMode, ThoModule};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_shared::cdt_for_dir;

    fn test_workspace(channels: u32) -> anyhow::Result<Box<dyn MotifsWorkspace>> {
        new_motifs_workspace(Path::new("/tmp").join(cdt_for_dir()), WsConfigs::new(channels, 684400, 32000))
    }

    #[test]
    fn test_model_publication_keeps_complete_generations() -> anyhow::Result<()> {
        let ws = test_workspace(1)?;
        let root = ws.final_nets_path()?;
        let base = root.join("1");
        assert!(publish_net(&base, "new", |path| { fs::write(path, b"partial")?; anyhow::bail!("injected failure") }).is_err());
        assert!(discover_nets(&root)?.is_empty());
        publish_net(&base, "old", |path| Ok(fs::write(path, b"old")?))?;
        let old = net_files_from_base(&base)?;
        assert!(publish_net(&base, "new", |path| { fs::write(path, b"partial")?; anyhow::bail!("injected failure") }).is_err());
        assert_eq!(net_files_from_base(&base)?, old);
        publish_net(&base, "new", |path| Ok(fs::write(path, b"new")?))?;
        let new = net_files_from_base(&base)?;
        assert_ne!(old, new);
        assert_eq!(fs::read_to_string(old.0)?, "old");
        assert_eq!(fs::read_to_string(old.1)?, "old");
        assert_eq!(fs::read_to_string(&new.0)?, "new");
        assert_eq!(fs::read_to_string(&new.1)?, "new");
        assert_eq!(discover_nets(&root)?, vec![new]);
        Ok(())
    }

    #[test]
    fn test_model_paths_are_created_lazily() -> anyhow::Result<()> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = FsWorkspace::new_concrete(&test_dir, WsConfigs::new(2, 100, 100))?;
        assert!(!test_dir.join("models").join(MOTIFS_MODELS).exists());
        assert_eq!(ws.motifs_nets_path()?, test_dir.join("models").join(MOTIFS_MODELS));
        assert!(test_dir.join("models").join(MOTIFS_MODELS).exists());
        assert!(!test_dir.join("models").join(LABELS_MODELS).exists());
        assert!(!test_dir.join("models").join(FINAL_MODELS).exists());
        Ok(())
    }

    #[test]
    fn test_save_and_list_motif_models() -> anyhow::Result<()> {
        let ws = test_workspace(2)?;
        ws.save_motif_net(&0, &0, &0, (&VarStore::new(Cpu)).into(), &MotifNetParams::new_conv(10, 1, 1, 100, 32, 10))?;
        ws.save_motif_net(&1, &1, &0, (&VarStore::new(Cpu)).into(), &MotifNetParams::new_conv(20, 1, 1, 100, 32, 10))?;
        ws.save_motif_net(&1, &2, &2, (&VarStore::new(Cpu)).into(), &MotifNetParams::new_conv(30, 1, 1, 100, 32, 10))?;
        ws.save_motif_net(&1, &3, &2, (&VarStore::new(Cpu)).into(), &MotifNetParams::new_conv(40, 1, 1, 100, 32, 10))?;
        ws.save_motif_net(&1, &4, &2, (&VarStore::new(Cpu)).into(), &MotifNetParams::new_conv(50, 1, 1, 100, 32, 10))?;
        ws.save_motif_net(&1, &3, &4, (&VarStore::new(Cpu)).into(), &MotifNetParams::new_conv(50, 1, 1, 100, 32, 10))?;
        assert_eq!(ws.motif_models(0, 0)?.len(), 1);
        assert_eq!(ws.motif_models(1, 0)?.len(), 1);
        assert_eq!(ws.motif_models(1, 2)?.len(), 3);
        assert_eq!(ws.motif_models(1, 4)?.len(), 1);
        Ok(())
    }

    #[test]
    fn test_save_and_list_label_models() -> anyhow::Result<()> {
        let ws = test_workspace(20)?;
        let params = LabelNetParams::new(4, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(20, 10)));
        ws.save_label_net(1, &"label_0", (&VarStore::new(Cpu)).into(), &params)?;
        ws.save_label_net(2, &"label_0", (&VarStore::new(Cpu)).into(), &params)?;
        ws.save_label_net(3, &"label_1", (&VarStore::new(Cpu)).into(), &params)?;
        ws.save_label_net(4, &"label_1", (&VarStore::new(Cpu)).into(), &params)?;
        let models = ws.label_models()?;
        assert_eq!(models.len(), 2);
        assert_eq!(models.get("label_0").expect("label_0").len(), 2);
        assert_eq!(models.get("label_1").expect("label_1").len(), 2);
        Ok(())
    }

    #[test]
    fn test_dotted_labels_remain_distinct_during_discovery() -> anyhow::Result<()> {
        let ws = test_workspace(2)?;
        let params = LabelNetParams::new(4, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(20, 10)));
        for label in ["class.a", "class.b"] {
            ws.save_label_net(1, &label, (&VarStore::new(Cpu)).into(), &params)?;
        }
        let models = ws.label_models()?;
        assert!(models.contains_key("class.a"));
        assert!(models.contains_key("class.b"));
        assert!(!models.contains_key("class"));
        Ok(())
    }

    #[test]
    fn test_load_motif_net() -> anyhow::Result<()> {
        let ws = test_workspace(2)?;
        let params = MotifNetParams::new_conv(1000, 3, 1, 10, 27, 10);
        let mut vs = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = conv_motif_node(match &params.net_type {
            MotifNetTypeParams::ConvNetParams(params) => params,
            _ => unreachable!(),
        })?.build_tch_module(&vs.root())?;
        vs.freeze();
        let net_id = ws.save_motif_net(&0, &0, &0, (&vs).into(), &params)?;
        let (loaded_params, Some(loaded_module)) = ws.load_motif_net(&net_id, &Cpu, false, true)? else { panic!("motif module was not loaded") };
        assert_eq!(format!("{params:?}"), format!("{loaded_params:?}"));
        assert_eq!(vs.variables(), loaded_module.var_store().variables());
        Ok(())
    }

    #[test]
    fn test_load_cosine_motif_net() -> anyhow::Result<()> {
        let ws = test_workspace(2)?;
        let params = MotifNetParams::new_cosine(1234, 5.0, 0.0, InitMode::Mean(InitDummy {}));
        let mut vs = VarStore::new(Cpu);
        let motif = Arc::new(ThoTensor::new(((0u32), "motif_id").into(), Tensor::rand([123, 1234], FLOAT_CPU))?);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> = cosine_motif_node(match &params.net_type {
            MotifNetTypeParams::CosineNetParams(params) => params,
            _ => unreachable!(),
        }, Some(motif))?.build_tch_module(&vs.root())?;
        let _ = net.tho_forward(&Tensor::rand([789, 1234], FLOAT_CPU).into(), None, false)?;
        vs.freeze();
        let net_id = ws.save_motif_net(&0, &0, &0, (&vs).into(), &params)?;
        let (loaded_params, Some(loaded_module)) = ws.load_motif_net(&net_id, &Cpu, false, true)? else { panic!("motif module was not loaded") };
        assert_eq!(format!("{params:?}"), format!("{loaded_params:?}"));
        assert_eq!(vs.variables(), loaded_module.var_store().variables());
        Ok(())
    }

    #[test]
    fn test_load_label_net() -> anyhow::Result<()> {
        let ws = test_workspace(20)?;
        let params = LabelNetParams::new(4, LabelNetTypeParams::LinNetParams(LinLenNetParams::new(20, 10)));
        let mut vs = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = LabelNetParams::label_net(&params)?.build_tch_module(&vs.root())?;
        vs.freeze();
        let net_id = ws.save_label_net(7, &"label_0", (&vs).into(), &params)?;
        let (run_config_id, loaded_params, Some(loaded_module)) = ws.load_label_net(&net_id, &Cpu, false, true)? else { panic!("label module was not loaded") };
        assert_eq!(run_config_id, 7);
        assert_eq!(format!("{params:?}"), format!("{loaded_params:?}"));
        assert_eq!(vs.variables(), loaded_module.var_store().variables());
        Ok(())
    }

    #[test]
    fn test_load_final_net() -> anyhow::Result<()> {
        let ws = test_workspace(2)?;
        let params = FinalNetParams::new(8, 100, 2, 1);
        let mut vs = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = final_net(&params)?.build_tch_module(&vs.root())?;
        vs.freeze();
        let net_id = ws.save_final_net(3, (&vs).into(), &params)?;
        let (loaded_params, loaded_module) = ws.load_final_net(&net_id, &Cpu)?;
        assert_eq!(format!("{params:?}"), format!("{loaded_params:?}"));
        assert_eq!(vs.variables(), loaded_module.var_store().variables());
        Ok(())
    }
}
