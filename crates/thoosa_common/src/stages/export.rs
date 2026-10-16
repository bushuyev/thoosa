use anyhow::Context as _;
use async_trait::async_trait;
use clap::Args;
use log::{info, warn};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thoosa_cpu_data_derive::CpuData;
use thoosa_shared::ThoError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::Data;
use thoosa_shared::traits::{AsyncStage, RunCtx, Stage};
use thoosa_store::{SqlStore, Store, TrainedNetsBy};

use crate::stages::convert_precision::{ConvertPrecisionArgs, ConvertPrecisionBy, ConvertPrecisionInput, ConvertPrecisionStage, ConvertPrecisionTarget, ConvertedNet, PrecisionModelType};
use crate::ws::{Workspace, WsConfigs};

/// Supplies the workspace construction and model roles that vary between application crates.
pub trait ExportWorkspace: Workspace {
    type ModelType: PrecisionModelType<Self>;

    fn create_export_workspace(target_dir: &Path, configs: WsConfigs) -> anyhow::Result<Box<Self>>;
    fn exported_model_type() -> Self::ModelType;
    fn dependency_model_type() -> Self::ModelType;
    fn all_model_type() -> Self::ModelType;
}

#[async_trait]
/// Supplies the application-owned record operations while common code handles export orchestration.
pub trait ExportStore: Store {
    async fn list_export_groups(&self) -> anyhow::Result<Vec<String>>;
    async fn list_exported_model_run_config_ids(&self, group: &str) -> anyhow::Result<Vec<RunConfigId>>;
    async fn export_model_records(&self, target_db_path: PathBuf, run_config_ids: Vec<RunConfigId>) -> anyhow::Result<()>;
}

#[derive(Debug, Args, Clone)]
pub struct ExportArgs {
    #[arg(long)]
    pub target_dir: PathBuf,

    #[arg(long, value_enum)]
    pub target_precision: Option<ConvertPrecisionTarget>,
}

#[derive(Debug, Clone, CpuData)]
pub struct ExportInput {}

impl ExportInput {
    pub fn new() -> Self {
        Self {}
    }
}

impl Input for ExportInput {}

#[derive(Debug, Clone)]
pub struct ExportOutput<N>
where
    N: Debug,
{
    target_dir: PathBuf,
    exported_model_run_config_ids: Vec<RunConfigId>,
    dependency_model_run_config_ids: Vec<RunConfigId>,
    copied_models: Vec<PathBuf>,
    converted: Vec<ConvertedNet<N>>,
}

impl<N> ExportOutput<N>
where
    N: Debug,
{
    fn new(target_dir: PathBuf, exported_model_run_config_ids: Vec<RunConfigId>, dependency_model_run_config_ids: Vec<RunConfigId>, copied_models: Vec<PathBuf>, converted: Vec<ConvertedNet<N>>) -> Self {
        Self { target_dir, exported_model_run_config_ids, dependency_model_run_config_ids, copied_models, converted }
    }

    pub fn target_dir(&self) -> &Path {
        &self.target_dir
    }

    pub fn exported_model_run_config_ids(&self) -> &[RunConfigId] {
        &self.exported_model_run_config_ids
    }

    pub fn dependency_model_run_config_ids(&self) -> &[RunConfigId] {
        &self.dependency_model_run_config_ids
    }

    // These aliases preserve the existing motif export API while new common callers use model roles.
    pub fn label_net_run_config_ids(&self) -> &[RunConfigId] {
        self.exported_model_run_config_ids()
    }

    pub fn motif_net_run_config_ids(&self) -> &[RunConfigId] {
        self.dependency_model_run_config_ids()
    }

    pub fn copied_models(&self) -> &[PathBuf] {
        &self.copied_models
    }

    pub fn converted(&self) -> &[ConvertedNet<N>] {
        &self.converted
    }
}

impl<N> Data for ExportOutput<N>
where
    N: Debug,
{
    type Metrics = ();

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn tho_device_idx(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0
    }
}

impl<N> Output for ExportOutput<N> where N: Debug {}

#[derive(Debug)]
pub struct Export<C, W, S>
where
    C: RunCtx + 'static,
    W: ExportWorkspace + ?Sized + 'static,
    S: ExportStore + ?Sized + 'static,
{
    ws: Arc<RwLock<Box<W>>>,
    store: Arc<S>,
    args: ExportArgs,
    pd: PhantomData<C>,
}

impl<C, W, S> Export<C, W, S>
where
    C: RunCtx + 'static,
    W: ExportWorkspace + ?Sized + 'static,
    S: ExportStore + ?Sized + 'static,
{
    pub fn new(ws: &Arc<RwLock<Box<W>>>, store: &Arc<S>, args: ExportArgs) -> Self {
        Self { ws: Arc::clone(ws), store: Arc::clone(store), args, pd: PhantomData }
    }
}

impl<C, W, S> Stage for Export<C, W, S>
where
    C: RunCtx + 'static,
    W: ExportWorkspace + ?Sized + 'static,
    S: ExportStore + ?Sized + 'static,
{
    type Ctx = C;
    type In<'a> = ExportInput;
    type Out = Arc<ExportOutput<W::ModelType>>;
}

impl<C, W, S> AsyncStage for Export<C, W, S>
where
    C: RunCtx + 'static,
    W: ExportWorkspace + ?Sized + 'static,
    S: ExportStore + ?Sized + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "Export::run"))]
    async fn run(self: Arc<Self>, _input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        let target_ws = self.create_target_workspace()?;
        let target_store_path = { target_ws.read().store_path().clone() };
        {
            // Opening the target store initializes the common schema before application records are copied.
            let target_store = track_await!(SqlStore::from_path(&target_store_path)).await?;
            drop(target_store);
        }

        let exported_model_run_config_ids = track_await!(self.exported_model_run_config_ids()).await?;
        if exported_model_run_config_ids.is_empty() {
            warn!("Export::run: no model run configs found for export");
        }

        let (dependency_model_run_config_ids, copied_models) = track_await!(self.copy_required_models(&target_ws, exported_model_run_config_ids.as_slice())).await?;
        track_await!(self.store.export_model_records(target_store_path, exported_model_run_config_ids.clone())).await?;

        let converted = if let Some(target_precision) = self.args.target_precision {
            let output = track_await!(Arc::new(ConvertPrecisionStage::<C, W, W::ModelType>::new(
                &target_ws,
                ConvertPrecisionArgs::new(ConvertPrecisionBy::Type(W::all_model_type()), target_precision),
            ))
            .run(ConvertPrecisionInput::new(), ctx.branch(true))).await?;
            output.converted().to_vec()
        } else {
            Vec::new()
        };

        info!("Export::run: exported target_dir={:?} exported_model_run_config_ids={exported_model_run_config_ids:?} dependency_model_run_config_ids={dependency_model_run_config_ids:?} copied_models={} converted={}", self.args.target_dir, copied_models.len(), converted.len());
        Ok(Arc::new(ExportOutput::new(self.args.target_dir.clone(), exported_model_run_config_ids, dependency_model_run_config_ids, copied_models, converted)))
    }
}

impl<C, W, S> Export<C, W, S>
where
    C: RunCtx + 'static,
    W: ExportWorkspace + ?Sized + 'static,
    S: ExportStore + ?Sized + 'static,
{
    fn create_target_workspace(&self) -> anyhow::Result<Arc<RwLock<Box<W>>>> {
        let configs = {
            let ws = self.ws.read();
            WsConfigs::new(ws.channels(), ws.max_input(), ws.sample_rate())
        };
        Ok(Arc::new(RwLock::new(W::create_export_workspace(&self.args.target_dir, configs)?)))
    }

    async fn exported_model_run_config_ids(&self) -> anyhow::Result<Vec<RunConfigId>> {
        let groups = track_await!(self.store.list_export_groups()).await?;
        let mut ids = Vec::new();
        for group in groups {
            ids.extend(track_await!(self.store.list_exported_model_run_config_ids(&group)).await?);
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    async fn copy_required_models(&self, target_ws: &Arc<RwLock<Box<W>>>, exported_model_run_config_ids: &[RunConfigId]) -> anyhow::Result<(Vec<RunConfigId>, Vec<PathBuf>)> {
        let exported_models = track_await!(self.store.list_trained_nets(TrainedNetsBy::RunConfigIds(exported_model_run_config_ids.to_vec()), None, None)).await?;
        let mut dependency_model_run_config_ids = Vec::new();
        for model in &exported_models {
            dependency_model_run_config_ids.extend(track_await!(self.store.list_input_nets(model.id)).await?);
        }
        dependency_model_run_config_ids.sort_unstable();
        dependency_model_run_config_ids.dedup();

        let dependency_models = track_await!(self.store.list_trained_nets(TrainedNetsBy::RunConfigIds(dependency_model_run_config_ids.clone()), None, None)).await?;
        let mut copied = Vec::new();
        let mut copied_targets = HashSet::new();
        {
            let source_ws = self.ws.read();
            let target_ws = target_ws.read();
            let source_exported_root = W::exported_model_type().root(source_ws.as_ref())?;
            let target_exported_root = W::exported_model_type().root(target_ws.as_ref())?;
            let source_dependency_root = W::dependency_model_type().root(source_ws.as_ref())?;
            let target_dependency_root = W::dependency_model_type().root(target_ws.as_ref())?;
            for model in &exported_models {
                let target_path = copy_net_path(&source_exported_root, &target_exported_root, &model.net_path)?;
                if copied_targets.insert(target_path.clone()) {
                    copied.push(target_path);
                }
            }
            for model in &dependency_models {
                let target_path = copy_net_path(&source_dependency_root, &target_dependency_root, &model.net_path)?;
                if copied_targets.insert(target_path.clone()) {
                    copied.push(target_path);
                }
            }
        }

        Ok((dependency_model_run_config_ids, copied))
    }
}

fn copy_net_path(source_root: &Path, target_root: &Path, net_path: &Path) -> anyhow::Result<PathBuf> {
    let source = if net_path.is_absolute() { net_path.to_path_buf() } else { source_root.join(net_path) };
    let target = if net_path.is_absolute() {
        let relative = net_path.strip_prefix(source_root).unwrap_or(net_path);
        target_root.join(relative)
    } else {
        target_root.join(net_path)
    };

    if source.is_dir() {
        copy_dir_all(&source, &target)?;
    } else if source.is_file() {
        let parent = target.parent().ok_or_else(|| anyhow::anyhow!(format!("copy_net_path: target {target:?} has no parent")))?;
        std::fs::create_dir_all(parent)?;
        std::fs::copy(&source, &target).with_context(|| format!("copy model file {source:?} to {target:?}"))?;
    } else {
        return Err(anyhow::anyhow!(format!("copy_net_path: source model path {source:?} does not exist")));
    }

    Ok(target)
}

fn copy_dir_all(source: &Path, target: &Path) -> anyhow::Result<()> {
    std::fs::create_dir_all(target)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_all(&source_path, &target_path)?;
        } else {
            std::fs::copy(&source_path, &target_path).with_context(|| format!("copy model file {source_path:?} to {target_path:?}"))?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::copy_net_path;
    use std::path::PathBuf;
    use thoosa_shared::ThoError;

    #[test]
    fn test_copy_net_path_copies_model_file() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let source_root = tmp.path().join("source");
        let target_root = tmp.path().join("target");
        let net_path = PathBuf::from("labels/1.safetensors");
        std::fs::create_dir_all(source_root.join("labels"))?;
        std::fs::write(source_root.join(&net_path), b"weights")?;

        let copied = copy_net_path(&source_root, &target_root, &net_path)?;

        assert_eq!(copied, target_root.join(&net_path));
        assert_eq!(std::fs::read(copied)?, b"weights");
        Ok(())
    }

    #[test]
    fn test_copy_net_path_copies_model_directory_recursively() -> anyhow::Result<()> {
        let tmp = tempfile::tempdir()?;
        let source_root = tmp.path().join("source");
        let target_root = tmp.path().join("target");
        let net_path = PathBuf::from("labels/1");
        std::fs::create_dir_all(source_root.join(&net_path).join("nested"))?;
        std::fs::write(source_root.join(&net_path).join("parameters.json"), b"params")?;
        std::fs::write(source_root.join(&net_path).join("nested/weights.safetensors"), b"weights")?;

        let copied = copy_net_path(&source_root, &target_root, &net_path)?;

        assert_eq!(std::fs::read(copied.join("parameters.json"))?, b"params");
        assert_eq!(std::fs::read(copied.join("nested/weights.safetensors"))?, b"weights");
        Ok(())
    }
}
