use std::collections::HashMap;
use std::fmt::Display;
use std::fs;
use std::fs::read_dir;
use std::num::ParseIntError;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};
use log::{debug, info, warn};
use tch::nn::VarStore;
use crate::train_nets::{LabelNodeParams, MotifNodeParams};
use thoosa_shared::{recurse_files, ThoosaError};

pub struct Workspace {
    base_path:PathBuf,
    store_path:PathBuf,
    tensors_path:PathBuf,
    models_path:PathBuf,
    motifs_models_path:PathBuf,
    labels_models_path:PathBuf,
    final_models_path:PathBuf,

    channels:usize //TODO move into input meta
}

impl Workspace {

    pub fn new(base_path: impl Into<PathBuf>, channels_num: usize) -> Result<Self, ThoosaError> {
        let base_path = base_path.into();
        if !base_path.exists() {
            fs::create_dir_all(&base_path)?;

        }

        let store_path:PathBuf = base_path.join("main_db.db3");
        let tensors_path:PathBuf = base_path.join("tensors");
        fs::create_dir_all(&tensors_path)?;

        let models_path:PathBuf = base_path.join("models");
        let motifs_models_path:PathBuf = models_path.join("motifs");
        let labels_models_path:PathBuf = models_path.join("labels");
        let final_models_path:PathBuf = models_path.join("final");
        fs::create_dir_all(&motifs_models_path)?;

        Ok(Self {
            base_path,
            store_path,
            tensors_path,
            models_path,
            motifs_models_path,
            labels_models_path,
            final_models_path,
            channels: channels_num
        })
    }

    pub fn base_path_as_str(&self) -> String {
        self.base_path.to_string_lossy().to_string()
    }

    pub fn base_path(&self) -> &PathBuf {
        &self.base_path
    }

    pub fn store_path(&self) -> &PathBuf {
        &self.store_path
    }

    pub fn tensors_path(&self) -> &PathBuf {
        &self.tensors_path
    }

    pub fn tensor_path(&self, key:impl Display) -> PathBuf {
        self.tensors_path.join(format!("{key}.pt"))
    }

    pub fn models_path(&self) -> &PathBuf {
        &self.models_path
    }

    pub fn motifs_models_path(&self) -> &PathBuf {
        &self.motifs_models_path
    }

    pub fn labels_models_path(&self) -> &PathBuf {
        &self.labels_models_path
    }

    pub fn final_models_path(&self) -> &PathBuf {
        &self.final_models_path
    }


    pub fn save_motif_model(&self, channel:impl Display, motif_selection_id:impl Display, motif_id:impl Display, vs:VarStore, params:MotifNodeParams) -> Result<(), ThoosaError>{

        let channel_models = self.motifs_models_path.join(format!("{}", motif_selection_id)).join(format!("{}", channel));
        fs::create_dir_all(&channel_models)?;

        let vs_path = channel_models.join(format!("{}.safetensors", motif_id));

        info!("train_on_motif:  motif_id={motif_id} saved to {vs_path:?}");

        vs.save(vs_path);

        fs::write(
            channel_models.join(format!("{}.json", motif_id)),
            serde_json::to_string(&params)?
        )?;

        Ok(())
    }

    /**
    returns channels x Vec<safetensors_path, config jsonpath>
    */
    pub fn motif_models(&self, motif_selection_id:usize) -> Result<Vec<Vec<(PathBuf, PathBuf)>>, ThoosaError> {

        recurse_files(&self.motifs_models_path.join(motif_selection_id.to_string()), &|path|{
            path.extension().map(|v|v.eq("safetensors")).unwrap_or(false)

        })?.into_iter().try_fold(vec![vec![];self.channels], |mut a, safetensors_path|{
            debug!("motif_models: safetensors_path={safetensors_path:?}");

            let parent= safetensors_path.parent().expect("parent");
            let channel = parent.file_stem().expect("channel");
            let file_stem = safetensors_path.file_stem().expect("file_stem").to_str().expect("to_str");
            let json = format!("{}.json", file_stem);
            let json = parent.join(json);

            debug!("motif_models: channel={channel:?} safetensors_path={safetensors_path:?} file_stem={file_stem} json={json:?}  {}", json.exists());

            match channel.to_str().and_then(|c|c.parse::<usize>().ok()){
                None => {
                    warn!("unexpected channel: channel{channel:?}");
                }
                Some(channel) => {
                    a[channel].push((safetensors_path, json));

                }

            };
            Ok(a)

        })
    }

    pub fn save_label_model(&self, run_id:usize, label:impl Display, vs:VarStore, params:LabelNodeParams)-> Result<(), ThoosaError> {
        let label_models_path = self.labels_models_path.join(format!("{}", label));

        fs::create_dir_all(&label_models_path)?;

        let vs_path = label_models_path.join(format!("{}.safetensors", run_id));

        info!("save_label_model:  run_id={run_id} label={label} saved to {vs_path:?}");

        vs.save(vs_path);

        fs::write(
            label_models_path.join(format!("{}.json", run_id)),
            serde_json::to_string(&params)?
        )?;

        Ok(())
    }


    pub fn label_models(&self) -> Result<HashMap<String, Vec<(PathBuf, PathBuf)>>, ThoosaError> {
        recurse_files(&self.labels_models_path, &|path|{
            println!("path={path:?}");
            path.extension().map(|v|v.eq("safetensors")).unwrap_or(false)

        })?.into_iter().try_fold(HashMap::new(), |mut a, safetensors_path|{

            let parent= safetensors_path.parent().expect("parent");
            let label = parent.file_stem().expect("label");
            let file_stem = safetensors_path.file_stem().expect("file_stem").to_str().expect("to_str");
            let json = format!("{}.json", file_stem);
            let json = parent.join(json);

            debug!("label_models: label={label:?} safetensors_path={safetensors_path:?} file_stem={file_stem} json={json:?}  {}", json.exists());

            a.entry(label.to_str().expect("label").to_string()).or_insert(vec![]).push((safetensors_path, json));
            Ok(a)
        })
    }

    pub fn model_path_to_run_id(model_path:&Path) -> usize {
        model_path.file_stem().and_then(|s|s.to_str().and_then(|v|v.parse::<usize>().ok())).expect("can't parse run_id from {model_path}")
    }

    pub fn channels(&self) -> usize {
        self.channels
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use log::debug;
    use tch::Device::Cpu;
    use tch::nn::VarStore;
    use crate::utils::cdt_for_dir;
    use thoosa_shared::ThoosaError;
    use crate::train_nets::{LabelNodeParams, MotifNodeParams};
    use super::Workspace;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
    }


    #[test]
    fn test_motifs() -> Result<(), ThoosaError>{

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_run_train_on_motifs: test_dir={test_dir:?}");

        let ws = Workspace::new(test_dir, 2)?;


        debug!("store_path={:?}", ws.store_path());

        assert!(ws.store_path().ends_with("main_db.db3"));


        ws.save_motif_model(0, 0, 0, VarStore::new(Cpu), MotifNodeParams::new(10));
        ws.save_motif_model(1, 0, 1, VarStore::new(Cpu), MotifNodeParams::new(20));

        ws.save_motif_model(1, 0, 2, VarStore::new(Cpu), MotifNodeParams::new(30));
        ws.save_motif_model(0, 0, 3, VarStore::new(Cpu), MotifNodeParams::new(40));
        ws.save_motif_model(1, 1, 4, VarStore::new(Cpu), MotifNodeParams::new(50));


        let models = ws.motif_models(0)?;
        for mm in &models{
            debug!("mm={mm:?}");
        }

        assert_eq!(models[0].len(), 2);
        assert_eq!(models[1].len(), 2);

        let models = ws.motif_models(1)?;
        for mm in &models{
            debug!("mm={mm:?}");
        }

        assert_eq!(models[0].len(), 0);
        assert_eq!(models[1].len(), 1);

        Ok(())
    }

    #[test]
    fn test_labels() -> Result<(), ThoosaError> {

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_run_train_on_motifs: test_dir={test_dir:?}");

        let ws = Workspace::new(test_dir, 20)?;

        ws.save_label_model(1, "label_0", VarStore::new(Cpu), LabelNodeParams::new(10, 4, 3, 10));
        ws.save_label_model(2, "label_0", VarStore::new(Cpu), LabelNodeParams::new(20, 4, 3, 10));
        ws.save_label_model(3, "label_1", VarStore::new(Cpu), LabelNodeParams::new(30, 4, 3, 10));
        ws.save_label_model(4, "label_1", VarStore::new(Cpu), LabelNodeParams::new(40, 4, 3, 10));

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