use crate::dims::{Batch, ChannelsMotifsData, LabelProbability, LabelsProbabilities, MotifProbability, Slice};
use crate::train_nets::{
    FinalNetParams, LabelNetParams, MotifNetParams, ThoModuleWithStore, final_net,
};
use log::{debug, error, info, trace};
use mockall::automock;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::fs;
use std::fs::read_dir;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tch::nn::{self, VarStore};
use tch::{Device, Tensor};
use thoosa_shared::resource::{Resource, ResourceStats};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{ThoError, cdt_for_dir, recurse_files, mono_key, panic_to_err};

pub const NET_PARAMS_FILE: &str = "parameters.json";
pub const NET_WEIGHTS_FILE: &str = "weights.safetensors";


#[derive(Debug)]
pub struct FsWorkspace {
    base_path: PathBuf,
    store_path: PathBuf,
    tensors_path: PathBuf,
    nets_path: PathBuf,
    plots_path: PathBuf,
    motifs_nets_path: PathBuf,
    labels_nets_path: PathBuf,
    final_nets_path: PathBuf,
    log_path: PathBuf,
    tmp_path: PathBuf,

    channels: u32, //TODO move into input meta
    sample_rate: u32,
    max_input: u32, //TODO move into input meta
}

impl FsWorkspace {
    pub fn new(
        base_path: impl Into<PathBuf>,
        channels: u32,
        max_input: u32,
        sample_rate: u32,
    ) -> Result<Box<dyn Workspace>, ThoError> {
        let base_path = base_path.into();
        fs::create_dir_all(&base_path)?;

        let store_path: PathBuf = base_path.join("main_db.db3");

        let models_path: PathBuf = base_path.join("models");

        let plots_path: PathBuf = base_path.join("plots");
        fs::create_dir_all(&plots_path)?;

        let tensors_path: PathBuf = base_path.join("tensors");
        fs::create_dir_all(&tensors_path)?;

        let tmp_path: PathBuf = base_path.join("tmp");
        fs::create_dir_all(&tmp_path)?;

        let log_path: PathBuf = base_path.join("log");
        fs::create_dir_all(&log_path)?;

        let motifs_models_path: PathBuf = models_path.join("motifs");
        fs::create_dir_all(&motifs_models_path)?;

        let labels_models_path: PathBuf = models_path.join("labels");
        fs::create_dir_all(&labels_models_path)?;

        let final_models_path: PathBuf = models_path.join("final");
        fs::create_dir_all(&final_models_path)?;

        Ok(Box::new(Self {
            base_path,
            store_path,
            tensors_path,
            nets_path: models_path,
            plots_path,
            motifs_nets_path: motifs_models_path,
            labels_nets_path: labels_models_path,
            final_nets_path: final_models_path,
            log_path,
            tmp_path,
            channels,
            sample_rate,
            max_input,
        }))
    }

    pub fn new_with_models_dir(
        base_path: impl Into<PathBuf>,
        channels: u32,
        max_input: u32,
        sample_rate: u32,
        models_dir:&String
    ) -> Result<Box<dyn Workspace>, ThoError> {
        let base_path = base_path.into();
        fs::create_dir_all(&base_path)?;

        let store_path: PathBuf = base_path.join("main_db.db3");

        let models_path: PathBuf = models_dir.into();

        let plots_path: PathBuf = base_path.join("plots");
        fs::create_dir_all(&plots_path)?;

        let tensors_path: PathBuf = base_path.join("tensors");
        fs::create_dir_all(&tensors_path)?;

        let tmp_path: PathBuf = base_path.join("tmp");
        fs::create_dir_all(&tmp_path)?;

        let log_path: PathBuf = base_path.join("log");
        fs::create_dir_all(&log_path)?;

        let motifs_models_path: PathBuf = models_path.join("motifs");


        let labels_models_path: PathBuf = models_path.join("labels");


        let final_models_path: PathBuf = models_path.join("final");


        Ok(Box::new(Self {
            base_path,
            store_path,
            tensors_path,
            nets_path: models_path,
            plots_path,
            motifs_nets_path: motifs_models_path,
            labels_nets_path: labels_models_path,
            final_nets_path: final_models_path,
            log_path,
            tmp_path,
            channels,
            sample_rate,
            max_input,
        }))
    }


    /*
       select_motifs_run_config_id/channel/motif_id
    */

    pub fn net_path_to_run_config_id(model_path: &Path) -> RunConfigId {
        model_path
            .file_stem()
            .and_then(|s| s.to_str().and_then(|v| v.parse::<RunConfigId>().ok()))
            .or_else(|| {
                model_path
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str().and_then(|v| v.parse::<RunConfigId>().ok()))
            })
            .expect("can't parse run_config_id from {model_path}")
    }

    pub fn parse_motif_path(net_id: &Path) -> Result<(u32, u32), ThoError> {
        let base_dir = if net_id.file_name().and_then(|v| v.to_str()) == Some(NET_WEIGHTS_FILE)
            || net_id.file_name().and_then(|v| v.to_str()) == Some(NET_PARAMS_FILE)
        {
            net_id.parent().ok_or_else(|| ThoError::new(format!("parse_motif_path: no parent for {net_id:?}")))?
        } else {
            net_id
        };

        let channel = base_dir
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|s| s.to_str())
            .ok_or_else(|| ThoError::new(format!("parse_motif_path: no channel in {net_id:?}")))?
            .parse::<u32>()
            .map_err(|e| ThoError::from_error(format!("parse_motif_path: bad channel in {net_id:?}"), e))?;

        let motif_part = base_dir
            .file_name()
            .and_then(|s| s.to_str())
            .ok_or_else(|| ThoError::new(format!("parse_motif_path: no motif id in {net_id:?}")))?;
        let motif_id_str = motif_part.split('_').next().unwrap_or(motif_part);
        let motif_id = motif_id_str
            .parse::<u32>()
            .map_err(|e| ThoError::from_error(format!("parse_motif_path: bad motif id in {net_id:?}"), e))?;

        Ok((channel, motif_id))
    }

    pub fn resource_tensor_path(tensors_path: &PathBuf, resource: &Resource) -> PathBuf {
        tensors_path.join(format!(
            "{}_{}.pt",
            resource.import_id(),
            resource.file_stem()
        ))
    }

    pub fn load_motif_net(
        &self,
        net_path: &Path,
        device: &Device,
        quantize: bool,
    ) -> Result<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>), ThoError> {
        let base_path = resolve_net_base_path(&self.motifs_nets_path, net_path);
        let (model_path, params_path) = net_files_from_base(&base_path);

        let motif_node_params = MotifNetParams::from_file(&params_path)?;
        let params_for_builder = motif_node_params.clone();

        let module = ThoModuleWithStore::load(
            mono_key!("MotifNet", "MotifNet") + &mono_key!(model_path.to_string_lossy().to_string(), "motif_model"),
            move |path: &nn::Path| params_for_builder.motif_net(None)?.build_tch_module(path),
            model_path,
            *device,
            quantize,
        )?;

        Ok((motif_node_params, module))
    }

    pub fn load_label_net(
        &self,
        net_path: &Path,
        device: &Device,
        quantize: bool,
    ) -> Result<
        (
            RunConfigId,
            LabelNetParams,
            ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>,
        ),
        ThoError,
    > {
        let base_path = resolve_net_base_path(&self.labels_nets_path, net_path);
        let (model_path, params_path) = net_files_from_base(&base_path);

        let label = if model_path.file_name().and_then(|v| v.to_str()) == Some(NET_WEIGHTS_FILE) {
            model_path
                .parent()
                .and_then(|p| p.parent())
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
        } else {
            model_path
                .parent()
                .and_then(|p| p.file_name())
                .and_then(|s| s.to_str())
        }
        .ok_or_else(|| ThoError::new(format!("load_label_net: can't parse label from {base_path:?}")))?;
        #[cfg(not(debug_assertions))]
        let _ = &label;

        let run_config_id = FsWorkspace::net_path_to_run_config_id(&model_path);
        let label_node_params = LabelNetParams::from_file(&params_path)?;
        let params_for_builder = label_node_params.clone();

        let module = ThoModuleWithStore::load(
            mono_key!(run_config_id, format!("Label_model_{label}_{run_config_id}")).into(),
            move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder)?.build_tch_module(path),
            model_path,
            *device,
            quantize,
        )?;

        Ok((run_config_id, label_node_params, module))
    }

    pub fn load_final_net(
        &self,
        net_path: &Path,
        device: &Device,
        quantize: bool,
    ) -> Result<
        (
            FinalNetParams,
            ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>,
        ),
        ThoError,
    > {
        let base_path = resolve_net_base_path(&self.final_nets_path, net_path);
        let (model_path, params_path) = net_files_from_base(&base_path);

        let run_config_id = FsWorkspace::net_path_to_run_config_id(&model_path);
        let final_node_params = FinalNetParams::from_file(&params_path)?;
        let params_for_builder = final_node_params.clone();

        let module = ThoModuleWithStore::load(
            mono_key!(run_config_id, "final_model").into(),
            move |path: &nn::Path| final_net(&params_for_builder)?.build_tch_module(path),
            model_path,
            *device,
            quantize,
        )?;

        Ok((final_node_params, module))
    }
}


fn net_files_from_base(base_path: &Path) -> (PathBuf, PathBuf) {
    let file_name = base_path.file_name().and_then(|v| v.to_str());
    if file_name == Some(NET_WEIGHTS_FILE) {
        let base_dir = base_path.parent().unwrap_or(base_path);
        return (
            base_path.to_path_buf(),
            base_dir.join(NET_PARAMS_FILE),
        );
    }
    if file_name == Some(NET_PARAMS_FILE) {
        let base_dir = base_path.parent().unwrap_or(base_path);
        return (
            base_dir.join(NET_WEIGHTS_FILE),
            base_path.to_path_buf(),
        );
    }
    match base_path.extension().and_then(|ext| ext.to_str()) {
        Some("safetensors") => {
            let base_buf = base_path.to_path_buf();
            let (_, params_path) = thoosa_shared::replace_extension(&base_buf, "json");
            (base_buf, params_path)
        }
        Some("json") => {
            let base_buf = base_path.to_path_buf();
            let (_, weights_path) = thoosa_shared::replace_extension(&base_buf, "safetensors");
            (weights_path, base_buf)
        }
        _ => (
            base_path.join(NET_WEIGHTS_FILE),
            base_path.join(NET_PARAMS_FILE),
        ),
    }
}

fn resolve_net_base_path(root: &Path, net_path: &Path) -> PathBuf {
    if net_path.is_absolute() {
        net_path.to_path_buf()
    } else {
        root.join(net_path)
    }
}

impl Workspace for FsWorkspace {
    fn import_tensor(
        &self,
        t: Tensor,
        import_id: u32,
        key: &str,
    ) -> Result<ResourceStats, ThoError> {
        let path = self.tensors_path.join(format!("{import_id}_{key}.pt"));
        trace!("Workspace:import_tensor {path:?} t.size={:?}", t.size());
        t.save(&path)
            .map_err(|e| {
                error!("import_tensor, save tensor error {e}");
                ThoError::from_error(format!("save tensor to {path:?}"), e)
            })?;

        Ok(ResourceStats::new(
            t.size().into_iter().map(|v| v as u32).collect::<Vec<_>>(),
        ))
    }

    fn base_path_as_str(&self) -> String {
        self.base_path.to_string_lossy().to_string()
    }
    fn base_path(&self) -> &PathBuf {
        &self.base_path
    }
    fn store_path(&self) -> &PathBuf {
        &self.store_path
    }

    fn tmp_run_store_path(&self, run_config_id: RunConfigId, channel: u32) -> PathBuf {
        self.tmp_path.join(format!("{run_config_id}_{channel}.db3"))
    }

    fn tmp_path(&self) -> &PathBuf {
        &self.tmp_path
    }

    fn tensors_path(&self) -> &PathBuf {
        &self.tensors_path
    }
    fn tensor_path(&self, key: &Resource) -> PathBuf {
        self.tensors_path
            .join(format!("{}_{}.pt", key.import_id(), key.file_stem()))
    }
    fn models_path(&self) -> &PathBuf {
        &self.nets_path
    }

    fn plots_path(&self) -> &PathBuf {
        &self.plots_path
    }

    fn motifs_nets_path(&self) -> &PathBuf {
        &self.motifs_nets_path
    }
    fn labels_nets_path(&self) -> &PathBuf {
        &self.labels_nets_path
    }
    fn final_nets_path(&self) -> &PathBuf {
        &self.final_nets_path
    }
    fn save_final_net(
        &self,
        run_config_id: RunConfigId,
        vs: &VarStore,
        params: &FinalNetParams,
    ) -> Result<PathBuf, ThoError> {
        debug!("FsWorkspace::save_final_net: run_config_id={run_config_id}  params={params:?} final_models_path={:?}",self.final_nets_path);

        let net_id = PathBuf::from(format!("{run_config_id}"));
        let base_path = self.final_nets_path.join(&net_id);

        fs::create_dir_all(&base_path)?;

        vs.save(base_path.join(NET_WEIGHTS_FILE))?;

        fs::write(
            base_path.join(NET_PARAMS_FILE),
            serde_json5::to_string(params)?,
        )?;

        Ok(net_id)
    }
    fn save_motif_net(
        &self,
        channel: &dyn Display,
        train_motif_net_run_config_id: &dyn Display,
        motif_id: &dyn Display,
        vs: &VarStore,
        params: &MotifNetParams,
    ) -> Result<PathBuf, ThoError> {
        let channel_part = format!("{channel}");
        let channel_models = self.motifs_nets_path.join(&channel_part);
        fs::create_dir_all(&channel_models)?;

        let net_dir_name = format!("{motif_id}_{train_motif_net_run_config_id}");
        let net_id = PathBuf::from(&channel_part).join(&net_dir_name);
        let base_path = channel_models.join(&net_dir_name);
        fs::create_dir_all(&base_path)?;
        let vs_path = base_path.join(NET_WEIGHTS_FILE);

        info!("FsWorkspace::save_motif_net:  motif_id={motif_id} motif_model_run_config_id={train_motif_net_run_config_id} saved to {vs_path:?}");

        vs.save(vs_path)?;

        fs::write(
            base_path.join(NET_PARAMS_FILE),
            serde_json5::to_string(&params)?,
        )?;

        Ok(net_id)
    }

    fn load_motif_net(
        &self,
        net_path: &Path,
        device: &Device,
        quantize: bool,
    ) -> Result<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>), ThoError> {
        FsWorkspace::load_motif_net(self, net_path, device, quantize)
    }

    fn load_label_net(
        &self,
        net_path: &Path,
        device: &Device,
        quantize: bool,
    ) -> Result<
        (
            RunConfigId,
            LabelNetParams,
            ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>,
        ),
        ThoError,
    > {
        FsWorkspace::load_label_net(self, net_path, device, quantize)
    }

    fn load_final_net(
        &self,
        net_path: &Path,
        device: &Device,
        quantize: bool,
    ) -> Result<
        (
            FinalNetParams,
            ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>,
        ),
        ThoError,
    > {
        FsWorkspace::load_final_net(self, net_path, device, false)
    }

    //TODO filter by loss and/or train_motif_nets_run_config_id
    fn load_motif_nets<'a>(
        &self,
        channel: u32,
        motif_id: u32,
        device: &'a Device,
        quantize:bool
    ) -> Result<Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>, ThoError> {
        debug!("load_motif_nets: channel={channel} motif_id={motif_id}");

        let channel_motif_models_paths = self.motif_models(channel, motif_id)?;

        channel_motif_models_paths.into_iter().map(|(model, config)| {
            let motif_node_params = MotifNetParams::from_file(&config)?;

            debug!("FsWorkspace::load_motif_nets: model={model:?} config={config:?} motif_node_params={motif_node_params:?}");

            let params_for_builder = motif_node_params.clone();

            let module = ThoModuleWithStore::load(
                mono_key!("MotifNet", "MotifNet") + &mono_key!(motif_id, "motif_id"),
                move |path: &nn::Path| params_for_builder.motif_net(None)?.build_tch_module(path),
                model,
                device.clone(),
                false,
            )?;

            Ok((motif_node_params, module))
        }).collect()
    }

    fn load_label_nets(
        &self,
        train_label_nets_run_config_ids: Option<&[RunConfigId]>,
        device: &Device,
        quantize: bool,
    ) -> Result<
        HashMap<
            String,
            Vec<(
                RunConfigId,
                LabelNetParams,
                ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>,
            )>,
        >,
        ThoError,
    > {
        let label_models_paths = self.label_models()?;

        let train_label_nodes_run_config_ids_set = train_label_nets_run_config_ids
            .map(|v| v.iter().cloned().collect::<HashSet<_>>());

        debug!("load_label_nets: label_models_paths={label_models_paths:?} train_label_nodes_run_config_ids={train_label_nets_run_config_ids:?} train_label_nodes_run_config_ids_hash={train_label_nodes_run_config_ids_set:?}");

        label_models_paths
            .into_iter()
            .try_fold(HashMap::new(), |mut acc, (label, paths)| {
                let models = paths.into_iter().try_fold(
                    Vec::new(),
                    |mut models, (model_path, params_path)| -> Result<_, ThoError> {
                        let run_config_id = FsWorkspace::net_path_to_run_config_id(&model_path);


                        if train_label_nodes_run_config_ids_set
                            .as_ref()
                            .is_some_and(|set| !set.contains(&run_config_id))
                        {
                            debug!("load_label_nets: skipping run_config_id={run_config_id}");
                            return Ok(models);
                        }

                        let label_node_params = LabelNetParams::from_file(&params_path)
                            .map_err(|e| {
                                ThoError::from_msg(format!("can't read label net params from {params_path:?}: {e}")) })?;

                        debug!("load_label_nets: model={model_path:?} run_config_id={run_config_id} label_node_params={label_node_params:?}");

                        let params_for_builder = label_node_params.clone();

                        let module = ThoModuleWithStore::load(
                            mono_key!(run_config_id, format!("Label_model_{label}_{run_config_id}")).into(),
                            move |path: &nn::Path| {
                                LabelNetParams::label_net(&params_for_builder)?
                                    .build_tch_module(path)
                            },
                            model_path.clone(),
                            *device,
                            quantize,
                        )
                        .map_err(|e| {
                            ThoError::from_msg(format!("can't load label net model from {model_path:?}: {e}"))
                        })?;

                        debug!("load_label_nets: model={model_path:?} run_config_id={run_config_id} loaded");

                        models.push((run_config_id, label_node_params, module));

                        Ok(models)
                    },
                )?;

                acc.insert(label, models);

                Ok(acc)
            })
    }
    /**
    returns Vec<safetensors_path, config jsonpath>
     */
    fn motif_models(
        &self,
        channel: u32,
        motif_id: u32,
    ) -> Result<Vec<(PathBuf, PathBuf)>, ThoError> {
        let motif_prefix = format!("{}_", motif_id);

        recurse_files(&self.motifs_nets_path.join(channel.to_string()), &|path| {
            if !path.extension().map(|v| v.eq("safetensors")).unwrap_or(false) {
                return false;
            }
            if path.file_name().and_then(|v| v.to_str()) == Some(NET_WEIGHTS_FILE) {
                return path
                    .parent()
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                    .map(|s| s.starts_with(motif_prefix.as_str()))
                    .unwrap_or(false);
            }
            path.file_stem().and_then(|s| s.to_str()).map(|s| {
                s.starts_with(motif_prefix.as_str())
            }).unwrap_or(false)
        })?.into_iter().try_fold(vec![], |mut a, safetensors_path| {
            debug!("motif_models: safetensors_path={safetensors_path:?}");

            let (file_stem, json) = if safetensors_path.file_name().and_then(|v| v.to_str()) == Some(NET_WEIGHTS_FILE) {
                let base_path = safetensors_path.parent().expect("parent");
                ("weights", base_path.join(NET_PARAMS_FILE))
            } else {
                thoosa_shared::replace_extension(&safetensors_path, "json")
            };

            debug!("motif_models: channel={channel} safetensors_path={safetensors_path:?} file_stem={file_stem} json={json:?}  {}", json.exists());

            a.push((safetensors_path, json));
            Ok(a)
        })
    }

    fn save_label_net(
        &self,
        label_node_run_config_id: u32,
        label: &dyn Display,
        vs: &VarStore,
        params: &LabelNetParams,
    ) -> Result<PathBuf, ThoError> {
        let label_part = format!("{label}");
        let label_models_path = self.labels_nets_path.join(&label_part);

        fs::create_dir_all(&label_models_path)?;

        let net_dir_name = format!("{label_node_run_config_id}");
        let net_id = PathBuf::from(&label_part).join(&net_dir_name);
        let base_path = label_models_path.join(&net_dir_name);
        fs::create_dir_all(&base_path)?;
        let vs_path = base_path.join(NET_WEIGHTS_FILE);

        info!("save_label_net:  run_config_id={label_node_run_config_id} label={label} saved to {vs_path:?}");

        panic_to_err!(vs.save(vs_path))?; //View for SafeView unwraps, it should be retried

        let path = base_path.join(NET_PARAMS_FILE);
        fs::write(&path, serde_json5::to_string(params)?)?;

        debug!("save_label_net: all saved, path={path:?}");

        Ok(net_id)
    }

    /// returns vec of (safetensors_path, json_path)
    fn label_models(&self) -> Result<HashMap<String, Vec<(PathBuf, PathBuf)>>, ThoError> {
        recurse_files(&self.labels_nets_path, &|path| {
            debug!("label_models:  path={path:?}");
            path.extension().map(|v| v.eq("safetensors")).unwrap_or(false)
        })?.into_iter().try_fold(HashMap::new(), |mut a, safetensors_path| {
            let (label_dir, file_stem, json_path) = if safetensors_path.file_name().and_then(|v| v.to_str()) == Some(NET_WEIGHTS_FILE) {
                let run_dir = safetensors_path.parent().expect("parent");
                let label_dir = run_dir.parent().expect("label");
                let json_path = run_dir.join(NET_PARAMS_FILE);
                (label_dir, "weights".to_string(), json_path)
            } else {
                let parent = safetensors_path.parent().expect("parent");
                let file_stem = safetensors_path.file_stem().expect("file_stem").to_str().expect("to_str").to_string();
                let json = format!("{}.json", file_stem);
                let json_path = parent.join(json);
                (parent, file_stem, json_path)
            };
            let label = label_dir.file_stem().expect("label");

            debug!("label_models: label={label:?} safetensors_path={safetensors_path:?} file_stem={file_stem} json={json_path:?}  {}", json_path.exists());

            a.entry(label.to_str().expect("label").to_string()).or_insert(vec![]).push((safetensors_path, json_path));
            Ok(a)
        })
    }

    /// returns (safetensors_path, json_path)
    fn final_model(&self, id: RunConfigId) -> Result<(PathBuf, PathBuf), ThoError> {
        let base_path = self.final_nets_path.join(format!("{id}"));
        let safetensors_path = base_path.join(NET_WEIGHTS_FILE);
        let json_path = base_path.join(NET_PARAMS_FILE);

        if safetensors_path.exists() || json_path.exists() {
            return Ok((safetensors_path, json_path));
        }

        let safetensors_path = self.final_nets_path.join(format!("{id}.safetensors"));
        let json_path = self.final_nets_path.join(format!("{id}.json"));

        Ok((safetensors_path, json_path))
    }

    #[inline]
    fn channels(&self) -> u32 {
        self.channels
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    #[inline]
    fn max_input(&self) -> u32 {
        self.max_input
    }

    fn clc_records(&self, _resource_ids: &Vec<String>) -> Result<i64, ThoError> {
        debug!(
            "FsWorkspace::clc_records: self.tensors_path={:?}",
            self.tensors_path
        );

        Ok(read_dir(&self.tensors_path)?
            .enumerate()
            .map(|(i, dir)| {
                (
                    i,
                    dir.unwrap()
                        .path()
                        .as_os_str()
                        .to_str()
                        .unwrap()
                        .to_string(),
                )
            })
            .collect::<Vec<(usize, String)>>()
            .into_par_iter()
            .map(|(i, path)| {
                let result = Tensor::load(path);
                match result {
                    Ok(t) => {
                        let tensor = t; //.to_device(Device::Cuda(0));

                        debug!("clc_records: i={i} tensor={:?}", tensor.size());
                        tensor.size()[1]
                    }
                    Err(e) => {
                        debug!("clc_records: i={i} warning e={e:?}");
                        0
                    }
                }
            })
            .sum())
    }

    fn log_path(&self) -> &PathBuf {
        &self.log_path
    }
}

#[automock]
pub trait Workspace: Sync + Send + Debug {
    fn import_tensor<'a>(
        &self,
        t: Tensor,
        import_id: u32,
        key: &'a str,
    ) -> Result<ResourceStats, ThoError>;
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
    fn motifs_nets_path(&self) -> &PathBuf;
    fn labels_nets_path(&self) -> &PathBuf;
    fn final_nets_path(&self) -> &PathBuf;
    fn save_final_net(
        &self,
        run_config_id: RunConfigId,
        vs: &VarStore,
        params: &FinalNetParams,
    ) -> Result<PathBuf, ThoError>;
    fn save_motif_net(
        &self,
        channel: &dyn Display,
        train_motif_net_run_config_id: &dyn Display,
        motif_id: &dyn Display,
        vs: &VarStore,
        params: &MotifNetParams,
    ) -> Result<PathBuf, ThoError>;
    fn load_motif_net(
        &self,
        net_path: &Path,
        device: &Device,
        quantize: bool
    ) -> Result<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>), ThoError>;

    fn load_label_net(
        &self,
        net_path: &Path,
        device: &Device,
        quantize: bool
    ) -> Result<
        (
            RunConfigId,
            LabelNetParams,
            ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>,
        ),
        ThoError,
    >;
    fn load_final_net(
        &self,
        net_path: &Path,
        device: &Device,
        quantize: bool,
    ) -> Result<
        (
            FinalNetParams,
            ThoModuleWithStore<Batch<LabelProbability>, Batch<LabelsProbabilities>>,
        ),
        ThoError,
    >;

    //TODO remove
    fn load_motif_nets<'a>(
        &self,
        channel: u32,
        motif_id: u32,
        device: &'a Device,
        quantize:bool
    ) -> Result<Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>, ThoError>;

    fn load_label_nets<'a>(
        &self,
        train_label_nodes_run_config_ids: Option<&'a [RunConfigId]>,
        device: &'a Device,
        quantize:bool
    ) -> Result<
        HashMap<
            String,
            Vec<(
                RunConfigId,
                LabelNetParams,
                ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>,
            )>,
        >,
        ThoError,
    >; //TODO remove
    /**
    returns Vec<safetensors_path, config jsonpath>
     */
    fn motif_models(
        &self,
        channel: u32,
        motif_id: u32,
    ) -> Result<Vec<(PathBuf, PathBuf)>, ThoError>;
    fn save_label_net(
        &self,
        label_node_run_config_id: u32,
        label: &dyn Display,
        vs: &VarStore,
        params: &LabelNetParams,
    ) -> Result<PathBuf, ThoError>;
    fn label_models(&self) -> Result<HashMap<String, Vec<(PathBuf, PathBuf)>>, ThoError>;
    fn final_model(&self, id: RunConfigId) -> Result<(PathBuf, PathBuf), ThoError>;

    fn channels(&self) -> u32;
    fn sample_rate(&self) -> u32;
    fn max_input(&self) -> u32;
    fn clc_records(&self, resource_ids: &Vec<String>) -> Result<i64, ThoError>;
}

pub fn mock_ws_simp(
    path: Option<&PathBuf>,
    channels: u32,
    sample_rate: u32,
    max_input: u32,
) -> Box<dyn Workspace> {
    let ws_mock = _mock_simp(path, channels, sample_rate, max_input);

    Box::new(ws_mock)
}

fn _mock_simp(
    path: Option<&PathBuf>,
    channels: u32,
    sample_rate: u32,
    max_input: u32,
) -> MockWorkspace {
    let base = path.map(|v| v.clone()).unwrap_or_else(|| {
        let p = Path::new("/tmp").join(cdt_for_dir());
        fs::create_dir_all(&p).expect("can't create tmp dir");
        p
    });

    let mut ws_mock = MockWorkspace::new();

    ws_mock.expect_tensors_path().return_const(base.clone());
    ws_mock.expect_tmp_path().return_const(base.clone());
    ws_mock.expect_tensors_path().return_const(base.clone());

    ws_mock.expect_sample_rate().return_const(sample_rate);
    ws_mock.expect_channels().return_const(channels);
    ws_mock.expect_max_input().return_const(max_input);
    ws_mock.expect_import_tensor().return_const(Ok(ResourceStats::new(vec![20, 100])));
    ws_mock
}

pub struct WsMockBuilder {
    mock: MockWorkspace,
}

impl WsMockBuilder {
    pub fn new(path: Option<&PathBuf>, channels: u32, sample_rate: u32, max_input: u32) -> Self {
        Self {
            mock: _mock_simp(path, channels, sample_rate, max_input),
        }
    }

    #[deprecated]//use with_load_motif_net
    pub fn with_load_motif_nets(
        mut self,
        channels_motif_nets: Vec<
            Vec<(
                Box<dyn Fn() -> MotifNetParams + Sync + Send + 'static>,
                Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send + 'static>,
            )>,
        >,
    ) -> Self {
        let channels_motif_nets_clone = Arc::new(channels_motif_nets);

        self.mock
            .expect_load_motif_nets()
            .returning(move |channel: u32, motif_id: u32, _device, quantize| {
                let channels_motif_nets_clone = Arc::clone(&channels_motif_nets_clone);

                Ok(vec![(
                    (channels_motif_nets_clone[channel as usize][motif_id as usize].0)(),
                    (channels_motif_nets_clone[channel as usize][motif_id as usize].1)(),
                )])
            });

        self
    }

    pub fn with_load_motif_net(
        mut self,
        nets: Vec<(
            PathBuf,
            Box<dyn Fn() -> MotifNetParams + Sync + Send + 'static>,
            Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send + 'static>,
        )>,
    ) -> Self {
        let nets = Arc::new(nets);
        self.mock
            .expect_load_motif_net()
            .returning(move |net_path: &Path, _device, quantize| {
                let net_path = net_path.to_path_buf();
                let entry = nets
                    .iter()
                    .find(|(path, _, _)| *path == net_path)
                    .ok_or_else(|| ThoError::new(format!("Not found: {net_path:?}")))?;
                Ok(((entry.1)(), (entry.2)()))
            });

        self
    }

    pub fn with_load_label_nets(
        mut self,
        label_nets: HashMap<
            String,
            Vec<(
                RunConfigId,
                Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>,
                Box<
                    dyn Fn() -> ThoModuleWithStore<ChannelsMotifsData, Batch<LabelProbability>>
                        + Sync
                        + Send
                        + 'static,
                >,
            )>,
        >,
    ) -> Self {
        self.mock
            .expect_load_label_nets()
            .returning(move |_train_label_nodes_run_config_ids, _device, quantize| {
                Ok(HashMap::from_iter(label_nets.iter().map(
                    |(label, nodes)| {
                        (
                            label.clone(),
                            nodes
                                .into_iter()
                                .map(|(run_config_id, params, module)| (*run_config_id, params(), module()))
                                .collect::<Vec<_>>(),
                        )
                    },
                )))
            });
        self
    }

    pub fn build(self) -> Box<dyn Workspace> {
        Box::new(self.mock)
    }
}

#[cfg(test)]
#[allow(unused_imports, unused_variables)]
mod tests {
    use super::{FsWorkspace, Workspace};
    use crate::dims::AnyDims;
    use crate::my_tch::{BinaryHeadPool, InitDummy, MotifInitMode, MyCosineConv1d, ThoModule};
    use crate::train_nets::{conv_motif_node, cosine_motif_node, FinalNetParams, LabelNetParams, LabelNetTypeParams, LinLenNetParams, MotifNetParams, MotifNetTypeParams};
    use log::debug;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::nn::{ConvConfig, ModuleT, VarStore};
    use tch::Tensor;
    use thoosa_shared::{ThoError, cdt_for_dir, simple_id};
    use thoosa_shared::traits::{DataId, SimpleId};
    use crate::tho_tensor::ThoTensor;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_motifs() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_run_train_on_motifs: test_dir={test_dir:?}");

        let tensors_path = Some(test_dir.join("tensors"));
        let ws = FsWorkspace::new(test_dir, 2, 684400, 32000)?;

        debug!("store_path={:?}", ws.store_path());

        assert!(ws.store_path().ends_with("main_db.db3"));

        ws.save_motif_net(&0, &0, &0, &VarStore::new(Cpu), &MotifNetParams::new(10, 1, 1, 100, 32, 10))?;
        ws.save_motif_net(&1, &1, &0, &VarStore::new(Cpu), &MotifNetParams::new(20, 1, 1, 100, 32, 10))?;

        ws.save_motif_net(&1, &2, &2, &VarStore::new(Cpu), &MotifNetParams::new(30, 1, 1, 100, 32, 10))?;
        ws.save_motif_net(&1, &3, &2, &VarStore::new(Cpu), &MotifNetParams::new(40, 1, 1, 100, 32, 10))?;
        ws.save_motif_net(&1, &4, &2, &VarStore::new(Cpu), &MotifNetParams::new(50, 1, 1, 100, 32, 10))?;

        ws.save_motif_net(&1, &3, &4, &VarStore::new(Cpu), &MotifNetParams::new(50, 1, 1, 100, 32, 10))?;

        let models = ws.motif_models(0, 0)?;

        assert_eq!(ws.motif_models(0, 0)?.len(), 1);
        assert_eq!(ws.motif_models(1, 0)?.len(), 1);
        assert_eq!(ws.motif_models(1, 2)?.len(), 3);
        assert_eq!(ws.motif_models(1, 4)?.len(), 1);

        Ok(())
    }

    #[test]
    fn test_labels() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_run_train_on_motifs: test_dir={test_dir:?}");

        let ws = FsWorkspace::new(test_dir, 20, 684400, 32000)?;

        ws.save_label_net(
            1,
            &"label_0",
            &VarStore::new(Cpu),
            &LabelNetParams::new(
                4,
                crate::train_nets::LabelNetTypeParams::LinNetParams(LinLenNetParams::new(20, 10)),
            ),
        )?;
        ws.save_label_net(
            2,
            &"label_0",
            &VarStore::new(Cpu),
            &LabelNetParams::new(
                4,
                crate::train_nets::LabelNetTypeParams::LinNetParams(LinLenNetParams::new(20, 10)),
            ),
        )?;
        ws.save_label_net(
            3,
            &"label_1",
            &VarStore::new(Cpu),
            &LabelNetParams::new(
                4,
                crate::train_nets::LabelNetTypeParams::LinNetParams(LinLenNetParams::new(20, 10)),
            ),
        )?;
        ws.save_label_net(
            4,
            &"label_1",
            &VarStore::new(Cpu),
            &LabelNetParams::new(
                4,
                crate::train_nets::LabelNetTypeParams::LinNetParams(LinLenNetParams::new(20, 10)),
            ),
        )?;

        let models = ws.label_models()?;
        // for mm in &models{
        //     debug!("mm={mm:?}");
        // }

        assert_eq!(models.len(), 2);
        assert_eq!(models.get("label_0").expect("label_0").len(), 2);
        assert_eq!(models.get("label_1").expect("label_1").len(), 2);

        Ok(())
    }

    #[test]
    fn test_load_conv_motif_net() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = FsWorkspace::new(test_dir, 2, 684400, 32000)?;

        let params = MotifNetParams::new_conv(1000, 3, 1, 10, 27, 10);
        let mut vs = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> =
            conv_motif_node(match &params.net_type {
                MotifNetTypeParams::ConvNetParams(v) => v,
                _ => unreachable!(),
            })?.build_tch_module(&vs.root())?;
        vs.freeze();

        let net_id = ws.save_motif_net(&0, &0, &0, &vs, &params)?;
        debug!("net_id={net_id:?}");

        let (loaded_params, loaded_module) = ws.load_motif_net(&net_id, &Cpu, false)?;

        assert_eq!(format!("{:?}", params), format!("{:?}", loaded_params));
        assert_eq!(vs.variables(), loaded_module.var_store().variables());

        loaded_module.tho_forward(&ThoTensor::new(simple_id!(""), Tensor::rand(&[127, 1000], FLOAT_CPU))?, None, true)?;

        Ok(())
    }

    // #[test]
    // fn test_x()-> Result<(), ThoError>{
    //    let m =  MyCosineConv1d::new(&VarStore::new(Cpu).root(), 8, 10, ConvConfig::default())?;
    //
    //     <MyCosineConv1d as ThoModule<AnyDims, AnyDims>>::tho_forward(&m, &ThoTensor::<AnyDims, SimpleId>::new("".into(), Tensor::rand(
    //         [
    //             10,
    //             1234i64,
    //         ],
    //         FLOAT_CPU,
    //     )).unwrap(), None, true).unwrap();
    //
    //     Ok(())
    // }

    #[test]
    fn test_load_cosine_motif_net() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = FsWorkspace::new(test_dir, 2, 684400, 32000)?;

        let params = MotifNetParams::new_cosine(1234, 5.0, 0.0, MotifInitMode::Mean(InitDummy{}));
        let mut vs = VarStore::new(Cpu);
        let net: Box<dyn ThoModule<AnyDims, AnyDims>> =
            cosine_motif_node(match &params.net_type {
                MotifNetTypeParams::CosineNetParams(v) => v,
                _ => unreachable!(),
            },
              Some(Arc::new(ThoTensor::new(((0u32), "motif_id").into(), Tensor::rand(
                  [
                      123,
                      1234i64,
                  ],
                  FLOAT_CPU,
              ))?
              ))
            )?.build_tch_module(&vs.root())?;

        let out = net.tho_forward(
            &Tensor::rand(
                [
                    789,
                    1234i64,
                ],
                FLOAT_CPU,
            )
                .into(),
            None,
            false,
        )?;

        debug!("out={out:?}");


        vs.freeze();

        let net_id = ws.save_motif_net(&0, &0, &0, &vs, &params)?;
        debug!("net_id={net_id:?}");

        let (loaded_params, loaded_module) = ws.load_motif_net(&net_id, &Cpu, false)?;

        assert_eq!(format!("{:?}", params), format!("{:?}", loaded_params));
        assert_eq!(vs.variables(), loaded_module.var_store().variables());

        Ok(())
    }


    #[test]
    fn test_load_label_net() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = FsWorkspace::new(test_dir, 20, 684400, 32000)?;

        let params = LabelNetParams::new(
            4,
            LabelNetTypeParams::LinNetParams(LinLenNetParams::new(20, 10)),
        );
        let mut vs = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = LabelNetParams::label_net(&params)?.build_tch_module(&vs.root())?;
        vs.freeze();

        let net_id = ws.save_label_net(7, &"label_0", &vs, &params)?;

        debug!("test_load_label_net:net_id={net_id:?}");

        let (run_config_id, loaded_params, loaded_module) = ws.load_label_net(&net_id, &Cpu, false)?;

        assert_eq!(run_config_id, 7);
        assert_eq!(format!("{:?}", params), format!("{:?}", loaded_params));
        assert_eq!(vs.variables(), loaded_module.var_store().variables());

        Ok(())
    }

    #[test]
    fn test_load_final_net() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = FsWorkspace::new(test_dir, 2, 684400, 32000)?;

        let params = FinalNetParams::new(8, 2, 1);
        let mut vs = VarStore::new(Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = crate::train_nets::final_net(&params)?.build_tch_module(&vs.root())?;
        vs.freeze();

        let net_id = ws.save_final_net(3, &vs, &params)?;

        let (loaded_params, loaded_module) = ws.load_final_net(&net_id, &Cpu, false)?;

        assert_eq!(format!("{:?}", params), format!("{:?}", loaded_params));
        assert_eq!(vs.variables(), loaded_module.var_store().variables());

        Ok(())
    }
}
