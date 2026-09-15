use log::{debug, error, trace};
use mockall::automock;
use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;
use std::fmt::Debug;
use std::fs;
use std::fs::read_dir;
use std::path::{Path, PathBuf};
use tch::nn::VarStore;
use tch::Tensor;
use thoosa_shared::resource::{Resource, ResourceStats};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::tmp_file::TmpFile;
use thoosa_shared::{ThoError, cdt_for_dir, panic_to_err};
use thoosa_shared::traits::ToJsonFile;

pub const NET_PARAMS_FILE: &str = "parameters.json";
pub const NET_WEIGHTS_FILE: &str = "weights.safetensors";

#[derive(Debug, Clone, Copy)]
pub struct WsConfigs {
    pub channels: u32,
    pub max_input: u32,
    pub sample_rate: u32,
}

impl WsConfigs {
    pub fn new(channels: u32, max_input: u32, sample_rate: u32) -> Self {
        Self { channels, max_input, sample_rate }
    }
}


#[derive(Debug)]
pub struct FsWorkspace {
    base_path: PathBuf,
    store_path: PathBuf,
    tensors_path: PathBuf,
    nets_path: PathBuf,
    plots_path: PathBuf,
    log_path: PathBuf,
    tmp_path: PathBuf,

    ws_configs:WsConfigs
}

impl FsWorkspace {
    pub fn new(
        base_path: impl Into<PathBuf>,
        ws_configs: WsConfigs,
    ) -> Result<Box<dyn Workspace>, ThoError> {
        Ok(Box::new(Self::new_concrete(base_path, ws_configs)?))
    }

    /// Builds the concrete workspace so feature crates can expose their own workspace trait objects.
    pub fn new_concrete(
        base_path: impl Into<PathBuf>,
        ws_configs: WsConfigs,
    ) -> Result<Self, ThoError> {
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

        Ok(Self {
            base_path,
            store_path,
            tensors_path,
            nets_path: models_path,
            plots_path,
            log_path,
            tmp_path,
            ws_configs,
        })
    }

    pub fn new_with_models_dir(
        base_path: impl Into<PathBuf>,
        app_configs:&WsConfigs,
        models_dir:&String
    ) -> Result<Box<dyn Workspace>, ThoError> {
        Ok(Box::new(Self::new_concrete_with_models_dir(base_path, app_configs, models_dir)?))
    }

    /// Builds the concrete workspace with an external model directory for feature-specific trait objects.
    pub fn new_concrete_with_models_dir(
        base_path: impl Into<PathBuf>,
        app_configs:&WsConfigs,
        models_dir:&String
    ) -> Result<Self, ThoError> {
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

        Ok(Self {
            base_path,
            store_path,
            tensors_path,
            nets_path: models_path,
            plots_path,
            log_path,
            tmp_path,
            ws_configs:app_configs.clone()
        })
    }


    pub fn resource_tensor_path(tensors_path: &PathBuf, resource: &Resource) -> PathBuf {
        tensors_path.join(format!(
            "{}_{}.pt",
            resource.import_id(),
            resource.file_stem()
        ))
    }

    /// Resolves one model category lazily so feature crates own category names and creation timing.
    pub fn model_type_path(&self, model_type: &str) -> Result<PathBuf, ThoError> {
        let path = self.nets_path.join(model_type);
        fs::create_dir_all(&path)?;
        Ok(path)
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

    fn save_net<'a>(
        &self,
        run_config_id: RunConfigId,
        vs_or_path: VarStoreOrPath<'a>,
        model_type:&str,
        params: &dyn ToJsonFile,
    ) -> Result<PathBuf, ThoError> {


        let net_id = PathBuf::from(format!("{run_config_id}"));
        let base_path = self.models_path().join(model_type).join(&net_id);

        fs::create_dir_all(&base_path)?;
        let dst = base_path.join(NET_WEIGHTS_FILE);
        debug!("FsWorkspace::save_net: run_config_id={run_config_id} params={params:?} dst={dst:?}");

        panic_to_err!(vs_or_path.save(&dst))?;

        params.to_file(&base_path.join(NET_PARAMS_FILE))?;


        Ok(dst)
    }

    #[inline]
    fn channels(&self) -> u32 {
        self.ws_configs.channels
    }

    fn sample_rate(&self) -> u32 {
        self.ws_configs.sample_rate
    }

    #[inline]
    fn max_input(&self) -> u32 {
        self.ws_configs.max_input
    }

    fn ws_configs(&self)->WsConfigs {
        self.ws_configs
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

//TODO remove it - was used to migrate from saving VarStore directly to moving from tmp
pub enum VarStoreOrPath<'a> {
    VarStore(&'a VarStore),
    Path(&'a Path)
}

impl <'a> VarStoreOrPath<'a> {
    /// Saves or moves a checkpoint so feature-specific workspace extensions can reuse the common transport.
    pub fn save(&self, to:&PathBuf) -> Result<(), ThoError> {

        match self {
            VarStoreOrPath::VarStore(vs) => {
                vs.save(to)?;
            }
            VarStoreOrPath::Path(path) => {
                debug!("VarStoreOrPath::save: path={path:?} to={to:?}");
                std::fs::rename(path, to)?;
            }
        }

        Ok(())
    }
}



impl <'a>  From<&'a VarStore> for VarStoreOrPath<'a> {
    fn from(value: &'a VarStore) -> Self {
        Self::VarStore(value)
    }
}

impl <'a>  From<&'a PathBuf> for VarStoreOrPath<'a> {
    fn from(value: &'a PathBuf) -> Self {
        Self::Path(value.as_path())
    }
}

impl <'a> From<&'a TmpFile> for VarStoreOrPath<'a> {
    fn from(value: &'a TmpFile) -> Self {
        // Preserve TmpFile ownership while the workspace atomically moves its checkpoint.
        Self::Path(value.path())
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
    fn save_net<'a>(
        &self,
        run_config_id: RunConfigId,
        vs_or_path: VarStoreOrPath<'a>,
        model_type:&str,
        params: &dyn ToJsonFile,
    ) -> Result<PathBuf, ThoError>;
    fn channels(&self) -> u32;
    fn sample_rate(&self) -> u32;
    fn max_input(&self) -> u32;
    fn clc_records(&self, resource_ids: &Vec<String>) -> Result<i64, ThoError>;
    fn ws_configs(&self)->WsConfigs;
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

/// Builds a common workspace mock;
pub struct WsMockBuilder {
    mock: MockWorkspace,
}

impl WsMockBuilder {
    pub fn new(path: Option<&PathBuf>, channels: u32, sample_rate: u32, max_input: u32) -> Self {
        Self { mock: _mock_simp(path, channels, sample_rate, max_input) }
    }

    pub fn build(self) -> Box<dyn Workspace> {
        Box::new(self.mock)
    }
}
