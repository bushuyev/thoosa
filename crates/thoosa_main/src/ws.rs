use std::fmt::Display;
use std::fs;
use std::fs::read_dir;
use std::path::{Path, PathBuf};
use log::{debug, info};
use tch::nn::VarStore;
use thoosa_common::train_nets::MotifNodeParams;
use thoosa_shared::{recurse_files, ThoosaError};

pub struct Workspace {
    base_path:PathBuf,
    store_path:PathBuf,
    tensors_path:PathBuf,
    models_path:PathBuf,
    motifs_models_path:PathBuf
}

impl Workspace {

    pub fn new(base_path: impl Into<PathBuf>) -> Result<Self, ThoosaError> {
        let base_path = base_path.into();
        if !base_path.exists() {
            fs::create_dir_all(&base_path)?;

        }

        let store_path:PathBuf = base_path.join("main_db.db3");
        let tensors_path:PathBuf = base_path.join("tensors");
        fs::create_dir_all(&tensors_path)?;

        let models_path:PathBuf = base_path.join("models");
        let motifs_models_path:PathBuf = models_path.join("motifs");
        fs::create_dir_all(&motifs_models_path)?;

        Ok(Self {
            base_path,
            store_path,
            tensors_path,
            models_path,
            motifs_models_path
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


    pub fn save_motif_model(&self, channel:impl Display, motif_selection_id:impl Display, motif_id:impl Display, vs:VarStore, params:MotifNodeParams) -> Result<(), ThoosaError>{

        let channel_models = self.motifs_models_path.join(format!("channel_{}", channel));
        fs::create_dir_all(&channel_models)?;

        let vs_path = channel_models.join(format!("{}_{}.safetensors", motif_selection_id, motif_id));

        info!("train_on_motif:  motif_id={motif_id} saved to {vs_path:?}");

        vs.save(vs_path);

        fs::write(
            channel_models.join(format!("{}_{}.json", motif_selection_id, motif_id)),
            serde_json::to_string(&params)?
        )?;

        Ok(())
    }

    pub fn motif_models(&self) -> Result<Vec<(PathBuf, PathBuf)>, ThoosaError> {

        recurse_files(&self.motifs_models_path, &|path|{
            path.extension().map(|v|v.eq("safetensors")).unwrap_or(false)
        })?.into_iter().map(|safetensors_path|{
            let parent= safetensors_path.parent().expect("parent");
            let file_stem = safetensors_path.file_stem().expect("file_stem").to_str().expect("to_str");
            let json = format!("{}.json", file_stem);
            let json = parent.join(json);

             debug!("motif_models: safetensors_path={safetensors_path:?} file_stem={file_stem} json={json:?}  {}", json.exists());
            Ok((safetensors_path, json))
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use log::debug;
    use thoosa_common::utils::cdt_for_dir;
    use thoosa_shared::ThoosaError;
    use crate::ws::Workspace;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
    }


    #[test]
    fn test_0() -> Result<(), ThoosaError>{


        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_run_train_on_motifs: test_dir={test_dir:?}");

        let ws = Workspace::new(test_dir)?;


        debug!("store_path={:?}", ws.store_path());

        assert!(ws.store_path().ends_with("main_db.db3"));

        fs::write(ws.motifs_models_path().join("1_0.safetensors"), "zzz")?;
        fs::write(ws.motifs_models_path().join("1_0.json"), "zzz")?;

        for mm in ws.motif_models()?{
            debug!("mm={mm:?}");
        }

        assert_eq!(ws.motif_models()?.len(), 1);

        Ok(())
    }
}