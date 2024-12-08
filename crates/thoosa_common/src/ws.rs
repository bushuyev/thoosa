use crate::train_nets::{FinalNodeParams, LabelNodeParams, MotifNodeParams};
use lazy_static::lazy_static;
use log::{debug, info, trace, warn};
use parking_lot::RwLock;
use rayon::iter::ParallelIterator;
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelBridge};
use std::collections::HashMap;
use std::fmt::Display;
use std::fs;
use std::fs::{read_dir, ReadDir};
use std::num::ParseIntError;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use tch::nn::VarStore;
use tch::{Device, TchError, Tensor};
use thoosa_shared::resource::{Resource, ResourceStats};
use thoosa_shared::{cdt_for_dir, recurse_files, ThoosaError};

pub struct FsWorkspace {
    base_path:PathBuf,
    store_path:PathBuf,
    tensors_path:PathBuf,
    models_path:PathBuf,
    plots_path:PathBuf,
    motifs_models_path:PathBuf,
    labels_models_path:PathBuf,
    final_models_path:PathBuf,
    log_path:PathBuf,
    tmp_path:PathBuf,

    channels:usize, //TODO move into input meta
    max_input:usize //TODO move into input meta
}

impl FsWorkspace {

    pub fn new(base_path: impl Into<PathBuf>, channels: usize, max_input: usize) -> Result<Box<dyn Workspace>, ThoosaError> {

        let base_path = base_path.into();
        fs::create_dir_all(&base_path)?;

        let store_path:PathBuf = base_path.join("main_db.db3");

        let models_path:PathBuf = base_path.join("models");

        let plots_path:PathBuf = base_path.join("plots");
        fs::create_dir_all(&plots_path)?;

        let tensors_path:PathBuf = base_path.join("tensors");
        fs::create_dir_all(&tensors_path)?;

        let tmp_path:PathBuf = base_path.join("tmp");
        fs::create_dir_all(&tmp_path)?;

        let log_path:PathBuf = base_path.join("log");
        fs::create_dir_all(&log_path)?;

        let motifs_models_path:PathBuf = models_path.join("motifs");
        fs::create_dir_all(&motifs_models_path)?;

        let labels_models_path:PathBuf = models_path.join("labels");
        fs::create_dir_all(&labels_models_path)?;

        let final_models_path:PathBuf = models_path.join("final");
        fs::create_dir_all(&final_models_path)?;



        Ok(Box::new(Self {
            base_path,
            store_path,
            tensors_path,
            models_path,
            plots_path,
            motifs_models_path,
            labels_models_path,
            final_models_path,
            log_path,
            tmp_path,
            channels,
            max_input
        }))
    }


    /*
        motif_selection_run_id/channel/motif_id
     */

    pub fn model_path_to_run_id(model_path:&Path) -> usize {
        model_path.file_stem().and_then(|s|s.to_str().and_then(|v|v.parse::<usize>().ok())).expect("can't parse run_id from {model_path}")
    }
}

impl Workspace for FsWorkspace {

    fn import_tensor(&self,  t:Tensor, import_id:u32, key:&str)-> Result<ResourceStats, ThoosaError>{
        let path = self.tensors_path.join(format!("{import_id}_{key}.pt"));
        trace!("Workspace:import_tensor {path:?} t.size={:?}", t.size());
        t.save(path).map_err(|e|ThoosaError::new(e.to_string()))?;

        Ok(ResourceStats::new(t.size().into_iter().map(|v|v as u32).collect::<Vec<_>>()))
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

    fn tmp_run_store_path(&self, run_id: u32, channel: u32) -> PathBuf {
        self.tmp_path.join(format!("{run_id}_{channel}.db3"))
    }

    fn tmp_path(&self) -> &PathBuf {
        &self.tmp_path
    }

    fn tensors_path(&self) -> &PathBuf {
        &self.tensors_path
    }
    fn tensor_path(&self, key: &Resource) -> PathBuf {
        self.tensors_path.join(format!("{}_{}.pt", key.import_id(), key.file_stem()))
    }
    fn models_path(&self) -> &PathBuf {
        &self.models_path
    }

    fn plots_path(&self) -> &PathBuf {
        &self.plots_path
    }

    fn motifs_models_path(&self) -> &PathBuf {
        &self.motifs_models_path
    }
    fn labels_models_path(&self) -> &PathBuf {
        &self.labels_models_path
    }
    fn final_models_path(&self) -> &PathBuf {
        &self.final_models_path
    }
    fn save_final_model(&self, run_id: usize, vs: VarStore, params: FinalNodeParams) -> Result<(), ThoosaError> {
        fs::create_dir_all(&self.final_models_path)?;

        vs.save(&self.final_models_path.join(format!("{run_id}.safetensors")))?;

        fs::write(
            self.final_models_path.join(format!("{run_id}.json")),
            serde_json::to_string(&params)?
        )?;


        Ok(())
    }
    fn save_motif_model(&self, channel: &dyn Display, motif_node_params_run_id: &dyn Display, motif_id: &dyn Display, vs: &VarStore, params: &MotifNodeParams) -> Result<(), ThoosaError> {
        let channel_models = self.motifs_models_path.join(format!("{}", channel));
        fs::create_dir_all(&channel_models)?;

        let vs_path = channel_models.join(format!("{motif_id}_{motif_node_params_run_id}.safetensors"));

        info!("train_on_motif:  motif_id={motif_id} motif_model_run_id={motif_node_params_run_id} saved to {vs_path:?}");

        vs.save(vs_path)?;

        fs::write(
            channel_models.join(format!("{motif_id}_{motif_node_params_run_id}.json")),
            serde_json::to_string(&params)?
        )?;

        Ok(())
    }
    /**
    returns Vec<safetensors_path, config jsonpath>
     */
    fn motif_models(&self, channel: &dyn Display, motif_id: usize) -> Result<Vec<(PathBuf, PathBuf)>, ThoosaError> {
        let motif_prefix = format!("{}_", motif_id);

        recurse_files(&self.motifs_models_path.join(channel.to_string()), &|path| {
            path.extension().map(|v| v.eq("safetensors")).unwrap_or(false) &&
                path.file_stem().and_then(|s| s.to_str()).map(|s| {
                    s.starts_with(motif_prefix.as_str())
                }).unwrap_or(false)
        })?.into_iter().try_fold(vec![], |mut a, safetensors_path| {
            debug!("motif_models: safetensors_path={safetensors_path:?}");

            let parent = safetensors_path.parent().expect("parent");
            // let channel = parent.file_stem().expect("channel");
            let file_stem = safetensors_path.file_stem().expect("file_stem").to_str().expect("to_str");
            let json = format!("{}.json", file_stem);
            let json = parent.join(json);

            debug!("motif_models: channel={channel} safetensors_path={safetensors_path:?} file_stem={file_stem} json={json:?}  {}", json.exists());

            a.push((safetensors_path, json));
            Ok(a)
        })
    }
    fn save_label_model(&self, label_node_run_id: usize, label: &dyn  Display, vs: &VarStore, params: &LabelNodeParams) -> Result<(), ThoosaError> {
        let label_models_path = self.labels_models_path.join(format!("{}", label));

        fs::create_dir_all(&label_models_path)?;

        let vs_path = label_models_path.join(format!("{label_node_run_id}.safetensors"));

        info!("save_label_model:  run_id={label_node_run_id} label={label} saved to {vs_path:?}");

        vs.save(vs_path)?;

        fs::write(
            label_models_path.join(format!("{}.json", label_node_run_id)),
            serde_json::to_string(params)?
        )?;

        Ok(())
    }
    fn label_models(&self) -> Result<HashMap<String, Vec<(PathBuf, PathBuf)>>, ThoosaError> {
        recurse_files(&self.labels_models_path, &|path| {
            println!("path={path:?}");
            path.extension().map(|v| v.eq("safetensors")).unwrap_or(false)
        })?.into_iter().try_fold(HashMap::new(), |mut a, safetensors_path| {
            let parent = safetensors_path.parent().expect("parent");
            let label = parent.file_stem().expect("label");
            let file_stem = safetensors_path.file_stem().expect("file_stem").to_str().expect("to_str");
            let json = format!("{}.json", file_stem);
            let json = parent.join(json);

            debug!("label_models: label={label:?} safetensors_path={safetensors_path:?} file_stem={file_stem} json={json:?}  {}", json.exists());

            a.entry(label.to_str().expect("label").to_string()).or_insert(vec![]).push((safetensors_path, json));
            Ok(a)
        })
    }
    #[inline]
    fn channels(&self) -> usize {
        self.channels
    }
    #[inline]
    fn max_input(&self) -> usize {
        self.max_input
    }
    fn clc_records(&self, resource_ids: &Vec<String>) -> Result<i64, ThoosaError> {
        debug!("FsWorkspace::clc_records: self.tensors_path={:?}", self.tensors_path);

        Ok(read_dir(&self.tensors_path)?.enumerate().map(|(i, dir)| (i, dir.unwrap().path().as_os_str().to_str().unwrap().to_string())).collect::<Vec<(usize, String)>>()
            .into_par_iter().map(|(i, path)| {
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

pub trait Workspace:Sync + Send {
    fn import_tensor(&self, t:Tensor, import_id:u32, key:&str)-> Result<ResourceStats, ThoosaError>;
    fn base_path_as_str(&self) -> String;
    fn base_path(&self) -> &PathBuf;
    fn store_path(&self) -> &PathBuf;
    fn log_path(&self) -> &PathBuf;
    fn tmp_path(&self) -> &PathBuf;
    fn tmp_run_store_path(&self, run_id:u32, channel:u32) -> PathBuf;
    fn tensors_path(&self) -> &PathBuf;
    fn tensor_path(&self, key: &Resource) -> PathBuf;
    fn models_path(&self) -> &PathBuf;
    fn plots_path(&self) -> &PathBuf;
    fn motifs_models_path(&self) -> &PathBuf;
    fn labels_models_path(&self) -> &PathBuf;
    fn final_models_path(&self) -> &PathBuf;
    fn save_final_model(&self, run_id: usize, vs: VarStore, params: FinalNodeParams) -> Result<(), ThoosaError>;
    fn save_motif_model(&self, channel: &dyn Display, motif_node_params_run_id: &dyn Display, motif_id: &dyn Display, vs: &VarStore, params: &MotifNodeParams) -> Result<(), ThoosaError>;
    /**
    returns Vec<safetensors_path, config jsonpath>
     */
    fn motif_models(&self, channel: &dyn Display, motif_id: usize) -> Result<Vec<(PathBuf, PathBuf)>, ThoosaError>;
    fn save_label_model(&self, label_node_run_id: usize, label: &dyn  Display, vs: &VarStore, params: &LabelNodeParams) -> Result<(), ThoosaError>;
    fn label_models(&self) -> Result<HashMap<String, Vec<(PathBuf, PathBuf)>>, ThoosaError>;
    #[inline]
    fn channels(&self) -> usize;
    #[inline]
    fn max_input(&self) -> usize;
    fn clc_records(&self, resource_ids: &Vec<String>) -> Result<i64, ThoosaError>;
}

#[derive(Debug)]
pub struct DummyWorkspace {
    base:PathBuf,
    channels:usize,
    n:usize
}

impl DummyWorkspace {
    pub fn new(channels: usize, n: usize) ->Result<Box<dyn Workspace>, ThoosaError> {
        let base = Path::new("/tmp").join(cdt_for_dir());
        fs::create_dir_all(&base)?;

        Ok(Box::new(Self {
            channels,
            base,
            n
        }))
    }
}

impl Workspace for DummyWorkspace {
    fn import_tensor(&self,  t:Tensor, import_id: u32, key: &str) -> Result<ResourceStats, ThoosaError>{
        Ok(ResourceStats::new(vec![self.channels as u32, self.n as u32]))
    }

    fn base_path_as_str(&self) -> String {
        todo!()
    }

    fn base_path(&self) -> &PathBuf {
        todo!()
    }

    fn store_path(&self) -> &PathBuf {
        todo!()
    }

    fn tmp_run_store_path(&self, run_id: u32, channel: u32) -> PathBuf {
        self.base.join(format!("{run_id}_{channel}.db3"))
    }

    fn tmp_path(&self) -> &PathBuf {
        &self.base
    }


    fn tensors_path(&self) -> &PathBuf {
        &self.base
    }

    fn tensor_path(&self, key: &Resource) -> PathBuf {
        todo!()
    }

    fn models_path(&self) -> &PathBuf {
        todo!()
    }

    fn plots_path(&self) -> &PathBuf {
        todo!()
    }

    fn motifs_models_path(&self) -> &PathBuf {
        todo!()
    }

    fn labels_models_path(&self) -> &PathBuf {
        todo!()
    }

    fn final_models_path(&self) -> &PathBuf {
        todo!()
    }

    fn save_final_model(&self, run_id: usize, vs: VarStore, params: FinalNodeParams) -> Result<(), ThoosaError> {
        todo!()
    }

    fn save_motif_model(&self, channel: &dyn Display, motif_node_params_run_id: &dyn Display, motif_id: &dyn Display, vs: &VarStore, params: &MotifNodeParams) -> Result<(), ThoosaError> {
        todo!()
    }

    fn motif_models(&self, channel: &dyn Display, motif_id: usize) -> Result<Vec<(PathBuf, PathBuf)>, ThoosaError> {
        todo!()
    }

    fn save_label_model(&self, label_node_run_id: usize, label: &dyn  Display, vs: &VarStore, params: &LabelNodeParams) -> Result<(), ThoosaError> {
        todo!()
    }

    fn label_models(&self) -> Result<HashMap<String, Vec<(PathBuf, PathBuf)>>, ThoosaError> {
        todo!()
    }

    fn channels(&self) -> usize {
        self.channels
    }

    fn max_input(&self) -> usize {
        todo!()
    }

    fn clc_records(&self, resource_ids: &Vec<String>) -> Result<i64, ThoosaError> {
        Ok(1000)
    }

    fn log_path(&self) -> &PathBuf {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::{DummyWorkspace, FsWorkspace, Workspace};
    use crate::train_nets::{LabelNodeParams, MotifNodeParams};
    use log::debug;
    use std::fs;
    use std::path::Path;
    use tch::nn::VarStore;
    use tch::Device::Cpu;
    use thoosa_shared::{cdt_for_dir, ThoosaError};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
    }


    #[test]
    fn test_motifs() -> Result<(), ThoosaError>{

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_run_train_on_motifs: test_dir={test_dir:?}");

        let tensors_path = Some(test_dir.join("tensors"));
        let ws = FsWorkspace::new(test_dir, 2, 684400)?;


        debug!("store_path={:?}", ws.store_path());

        assert!(ws.store_path().ends_with("main_db.db3"));


        ws.save_motif_model(&0, &0, &0, &VarStore::new(Cpu), &MotifNodeParams::new(10, 1))?;
        ws.save_motif_model(&1, &1, &0, &VarStore::new(Cpu), &MotifNodeParams::new(20, 1))?;

        ws.save_motif_model(&1, &2, &2, &VarStore::new(Cpu), &MotifNodeParams::new(30, 1))?;
        ws.save_motif_model(&1, &3, &2, &VarStore::new(Cpu), &MotifNodeParams::new(40, 1))?;
        ws.save_motif_model(&1, &4, &2, &VarStore::new(Cpu), &MotifNodeParams::new(50, 1))?;

        ws.save_motif_model(&1, &3, &4, &VarStore::new(Cpu), &MotifNodeParams::new(50, 1))?;


        let models = ws.motif_models(&0, 0)?;


        assert_eq!(ws.motif_models(&0, 0)?.len(), 1);
        assert_eq!(ws.motif_models(&1, 0)?.len(), 1);
        assert_eq!(ws.motif_models(&1, 2)?.len(), 3);
        assert_eq!(ws.motif_models(&1, 4)?.len(), 1);

        Ok(())
    }

    #[test]
    fn test_labels() -> Result<(), ThoosaError> {

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_run_train_on_motifs: test_dir={test_dir:?}");

        let ws = FsWorkspace::new(test_dir, 20, 684400)?;


        ws.save_label_model(1, &"label_0", &VarStore::new(Cpu), &LabelNodeParams::new(10, 4, 3, 10))?;
        ws.save_label_model(2, &"label_0", &VarStore::new(Cpu), &LabelNodeParams::new(20, 4, 3, 10))?;
        ws.save_label_model(3, &"label_1", &VarStore::new(Cpu), &LabelNodeParams::new(30, 4, 3, 10))?;
        ws.save_label_model(4, &"label_1", &VarStore::new(Cpu), &LabelNodeParams::new(40, 4, 3, 10))?;

        let models =  ws.label_models()?;
        for mm in &models{
            debug!("mm={mm:?}");
        }

        assert_eq!(models.len(), 2);
        assert_eq!(models.get("label_0".into()).expect("label_0").len(), 2);
        assert_eq!(models.get("label_1".into()).expect("label_1").len(), 2);

        Ok(())
    }
}