use crate::dims::{
    Batch, LabelProbability, MotifProbability, MotifsNetsSelectionDims, Slice,
};
use crate::train_nets::{FinalNetParams, LabelNetParams, MotifNetParams, ThoModuleWithStore, motif_node_2};
use log::{debug, info, trace};
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
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{ThoError, cdt_for_dir, recurse_files, mono_key};

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

    /*
       select_motifs_run_id/channel/motif_id
    */

    pub fn net_path_to_run_id(model_path: &Path) -> RunId {
        model_path
            .file_stem()
            .and_then(|s| s.to_str().and_then(|v| v.parse::<RunId>().ok()))
            .expect("can't parse run_id from {model_path}")
    }

    pub fn resource_tensor_path(tensors_path: &PathBuf, resource: &Resource) -> PathBuf {
        tensors_path.join(format!(
            "{}_{}.pt",
            resource.import_id(),
            resource.file_stem()
        ))
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
            .map_err(|e| ThoError::from_error(format!("save tensor to {path:?}"), e))?;

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

    fn tmp_run_store_path(&self, run_id: RunId, channel: u32) -> PathBuf {
        self.tmp_path.join(format!("{run_id}_{channel}.db3"))
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
        run_id: RunId,
        vs: &VarStore,
        params: &FinalNetParams,
    ) -> Result<(), ThoError> {
        debug!(
            "save_final_model: run_id={run_id}  params={params:?} final_models_path={:?}",
            self.final_nets_path
        );

        fs::create_dir_all(&self.final_nets_path)?;

        vs.save(&self.final_nets_path.join(format!("{run_id}.safetensors")))?;

        fs::write(
            self.final_nets_path.join(format!("{run_id}.json")),
            serde_json5::to_string(params)?,
        )?;

        Ok(())
    }
    fn save_motif_net(
        &self,
        channel: &dyn Display,
        train_motif_net_run_id: &dyn Display,
        motif_id: &dyn Display,
        vs: &VarStore,
        params: &MotifNetParams,
    ) -> Result<(), ThoError> {
        let channel_models = self.motifs_nets_path.join(format!("{}", channel));
        fs::create_dir_all(&channel_models)?;

        let vs_path = channel_models.join(format!("{motif_id}_{train_motif_net_run_id}.safetensors"));

        info!("train_on_motif:  motif_id={motif_id} motif_model_run_id={train_motif_net_run_id} saved to {vs_path:?}");

        vs.save(vs_path)?;

        fs::write(
            channel_models.join(format!("{motif_id}_{train_motif_net_run_id}.json")),
            serde_json5::to_string(&params)?,
        )?;

        Ok(())
    }

    //TODO filter by loss and/or train_motif_nets_run_id
    fn load_motif_nets<'a>(
        &self,
        channel: u32,
        motif_id: u32,
        device: &'a Device,
    ) -> Result<Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>, ThoError> {
        debug!("load_motif_nets: channel={channel} motif_id={motif_id}");

        let channel_motif_models_paths = self.motif_models(channel, motif_id)?;

        channel_motif_models_paths.into_iter().map(|(model, config)| {
            let motif_node_params = MotifNetParams::from_file(&config)?;

            debug!("load_motif_models_ws: model={model:?} config={config:?} motif_node_params={motif_node_params:?}");

            let params_for_builder = motif_node_params.clone();

            let module = ThoModuleWithStore::load(
                mono_key!("MotifNet", "MotifNet") + &mono_key!(motif_id, "motif_id"),
                move |path: &nn::Path| motif_node_2(&params_for_builder).build_tch_module(path),
                model,
                device.clone(),
            )?;

            Ok((motif_node_params, module))
        }).collect()
    }

    fn load_label_nets(
        &self,
        train_label_nets_run_ids: Option<&[RunId]>,
        device: &Device,
    ) -> Result<
        HashMap<
            String,
            Vec<(
                RunId,
                LabelNetParams,
                ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>,
            )>,
        >,
        ThoError,
    > {
        let label_models_paths = self.label_models()?;

        let train_label_nodes_run_ids_hash = train_label_nets_run_ids.map(|v| v.into_iter().collect::<HashSet<_>>());

        debug!("load_all_label_models_ws: label_models_paths={label_models_paths:?} train_label_nodes_run_ids={train_label_nets_run_ids:?} train_label_nodes_run_ids_hash={train_label_nodes_run_ids_hash:?}");

        label_models_paths
            .into_iter()
            .try_fold(HashMap::new(), |mut a, (label, paths)| {

                let models: Vec<(RunId, LabelNetParams, ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>)> = paths
                    .into_iter()
                    .filter_map(|(model_path, params_path)|
                {
                    let run_id = FsWorkspace::net_path_to_run_id(&model_path);
                    if train_label_nodes_run_ids_hash.as_ref().map(|set| !set.contains(&run_id)).unwrap_or(true) {
                        debug!("load_all_label_models_ws: skipping run_id={run_id}");
                        return None;
                    }

                    let label_node_params = LabelNetParams::from_file(&params_path).expect(format!("no {params_path:?}").as_str());

                    debug!("load_all_label_models_ws: model={model_path:?} run_id={run_id} label_node_params={label_node_params:?}");

                //TODO ChannelsMotifsFlattenData
                // let module = LabelModule::ChannelsMotifsData(ThoModuleWithStore::load(LabelNetParams::label_net(&label_node_params), model_path.clone(), *device).expect(format!("can't load {model_path:?}").as_str()));
                let params_for_builder = label_node_params.clone();
                let module = ThoModuleWithStore::load(
                    mono_key!(run_id, format!("Label_model_{label}_{run_id}")).into(),
                      move |path: &nn::Path| LabelNetParams::label_net(&params_for_builder).build_tch_module(path),
                      model_path.clone(),
                      *device,
                )
                .expect(format!("can't load {model_path:?}").as_str());

                    debug!("load_all_label_models_ws: model={model_path:?} run_id={run_id} loaded");

                    Some((run_id, label_node_params, module))
                }).collect();

                a.insert(
                    label,
                    models
                );

            Ok(a)
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
            path.extension().map(|v| v.eq("safetensors")).unwrap_or(false) &&
                path.file_stem().and_then(|s| s.to_str()).map(|s| {
                    s.starts_with(motif_prefix.as_str())
                }).unwrap_or(false)
        })?.into_iter().try_fold(vec![], |mut a, safetensors_path| {
            debug!("motif_models: safetensors_path={safetensors_path:?}");

            let (file_stem, json) = thoosa_shared::replace_extension(&safetensors_path, "json");

            debug!("motif_models: channel={channel} safetensors_path={safetensors_path:?} file_stem={file_stem} json={json:?}  {}", json.exists());

            a.push((safetensors_path, json));
            Ok(a)
        })
    }

    fn save_label_model(
        &self,
        label_node_run_id: u32,
        label: &dyn Display,
        vs: &VarStore,
        params: &LabelNetParams,
    ) -> Result<PathBuf, ThoError> {
        let label_models_path = self.labels_nets_path.join(format!("{}", label));

        fs::create_dir_all(&label_models_path)?;

        let vs_path = label_models_path.join(format!("{label_node_run_id}.safetensors"));

        info!("save_label_model:  run_id={label_node_run_id} label={label} saved to {vs_path:?}");

        vs.save(vs_path)?;

        let path = label_models_path.join(format!("{}.json", label_node_run_id));
        fs::write(&path, serde_json5::to_string(params)?)?;

        debug!("save_label_model: all saved, path={path:?}");

        Ok(path)
    }

    /// returns vec of (safetensors_path, json_path)
    fn label_models(&self) -> Result<HashMap<String, Vec<(PathBuf, PathBuf)>>, ThoError> {
        recurse_files(&self.labels_nets_path, &|path| {
            debug!("label_models:  path={path:?}");
            path.extension().map(|v| v.eq("safetensors")).unwrap_or(false)
        })?.into_iter().try_fold(HashMap::new(), |mut a, safetensors_path| {
            let parent = safetensors_path.parent().expect("parent");
            let label = parent.file_stem().expect("label");
            let file_stem = safetensors_path.file_stem().expect("file_stem").to_str().expect("to_str");
            let json = format!("{}.json", file_stem);
            let json_path = parent.join(json);

            debug!("label_models: label={label:?} safetensors_path={safetensors_path:?} file_stem={file_stem} json={json_path:?}  {}", json_path.exists());

            a.entry(label.to_str().expect("label").to_string()).or_insert(vec![]).push((safetensors_path, json_path));
            Ok(a)
        })
    }

    /// returns (safetensors_path, json_path)
    fn final_model(&self, id: RunId) -> Result<(PathBuf, PathBuf), ThoError> {
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
    fn tmp_run_store_path(&self, run_id: RunId, channel: u32) -> PathBuf;
    fn tensors_path(&self) -> &PathBuf;
    fn tensor_path(&self, key: &Resource) -> PathBuf;
    fn models_path(&self) -> &PathBuf;
    fn plots_path(&self) -> &PathBuf;
    fn motifs_nets_path(&self) -> &PathBuf;
    fn labels_nets_path(&self) -> &PathBuf;
    fn final_nets_path(&self) -> &PathBuf;
    fn save_final_net(
        &self,
        run_id: RunId,
        vs: &VarStore,
        params: &FinalNetParams,
    ) -> Result<(), ThoError>;
    fn save_motif_net(
        &self,
        channel: &dyn Display,
        motif_node_params_run_id: &dyn Display,
        motif_id: &dyn Display,
        vs: &VarStore,
        params: &MotifNetParams,
    ) -> Result<(), ThoError>;

    fn load_motif_nets<'a>(
        &self,
        channel: u32,
        motif_id: u32,
        device: &'a Device,
    ) -> Result<Vec<(MotifNetParams, ThoModuleWithStore<Batch<Slice>, MotifProbability>)>, ThoError>;
    fn load_label_nets<'a>(
        &self,
        train_label_nodes_run_ids: Option<&'a [RunId]>,
        device: &'a Device,
    ) -> Result<
        HashMap<
            String,
            Vec<(
                RunId,
                LabelNetParams,
                ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>,
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
    fn save_label_model(
        &self,
        label_node_run_id: u32,
        label: &dyn Display,
        vs: &VarStore,
        params: &LabelNetParams,
    ) -> Result<PathBuf, ThoError>;
    fn label_models(&self) -> Result<HashMap<String, Vec<(PathBuf, PathBuf)>>, ThoError>;
    fn final_model(&self, id: RunId) -> Result<(PathBuf, PathBuf), ThoError>;

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
            .returning(move |channel: u32, motif_id: u32, _device| {
                let channels_motif_nets_clone = Arc::clone(&channels_motif_nets_clone);

                Ok(vec![(
                    (channels_motif_nets_clone[channel as usize][motif_id as usize].0)(),
                    (channels_motif_nets_clone[channel as usize][motif_id as usize].1)(),
                )])
            });

        self
    }

    pub fn with_label_nets(
        mut self,
        label_nets: HashMap<
            String,
            Vec<(
                RunId,
                Box<dyn Fn() -> LabelNetParams + Sync + Send + 'static>,
                Box<
                    dyn Fn() -> ThoModuleWithStore<MotifsNetsSelectionDims, Batch<LabelProbability>>
                        + Sync
                        + Send
                        + 'static,
                >,
            )>,
        >,
    ) -> Self {
        self.mock
            .expect_load_label_nets()
            .returning(move |_train_label_nodes_run_ids, _device| {
                Ok(HashMap::from_iter(label_nets.iter().map(
                    |(label, nodes)| {
                        (
                            label.clone(),
                            nodes
                                .into_iter()
                                .map(|(run_id, params, module)| (*run_id, params(), module()))
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
    use crate::train_nets::{LabelNetParams, MaxPaddingNetParams, MotifNetParams};
    use log::debug;
    use std::path::Path;
    use tch::Device::Cpu;
    use tch::nn::VarStore;
    use thoosa_shared::{ThoError, cdt_for_dir};

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

        ws.save_motif_net(&0, &0, &0, &VarStore::new(Cpu), &MotifNetParams::new(10, 1, 100))?;
        ws.save_motif_net(&1, &1, &0, &VarStore::new(Cpu), &MotifNetParams::new(20, 1, 100))?;

        ws.save_motif_net(&1, &2, &2, &VarStore::new(Cpu), &MotifNetParams::new(30, 1, 100))?;
        ws.save_motif_net(&1, &3, &2, &VarStore::new(Cpu), &MotifNetParams::new(40, 1, 100))?;
        ws.save_motif_net(&1, &4, &2, &VarStore::new(Cpu), &MotifNetParams::new(50, 1, 100))?;

        ws.save_motif_net(&1, &3, &4, &VarStore::new(Cpu), &MotifNetParams::new(50, 1, 100))?;

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

        ws.save_label_model(
            1,
            &"label_0",
            &VarStore::new(Cpu),
            &LabelNetParams::new(
                10,
                4,
                10,
                20,
                crate::train_nets::LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new()),
            ),
        )?;
        ws.save_label_model(
            2,
            &"label_0",
            &VarStore::new(Cpu),
            &LabelNetParams::new(
                20,
                4,
                10,
                20,
                crate::train_nets::LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new()),
            ),
        )?;
        ws.save_label_model(
            3,
            &"label_1",
            &VarStore::new(Cpu),
            &LabelNetParams::new(
                30,
                4,
                10,
                20,
                crate::train_nets::LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new()),
            ),
        )?;
        ws.save_label_model(
            4,
            &"label_1",
            &VarStore::new(Cpu),
            &LabelNetParams::new(
                40,
                4,
                10,
                20,
                crate::train_nets::LabelNetTypeParams::MaxPadding(MaxPaddingNetParams::new()),
            ),
        )?;

        let models = ws.label_models()?;
        // for mm in &models{
        //     debug!("mm={mm:?}");
        // }

        assert_eq!(models.len(), 2);
        assert_eq!(models.get("label_0".into()).expect("label_0").len(), 2);
        assert_eq!(models.get("label_1".into()).expect("label_1").len(), 2);

        Ok(())
    }
}
