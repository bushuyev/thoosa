use crate::tasks::convert_precision::{ConvertPrecisionArgs, ConvertPrecisionBy, ConvertPrecisionInput, ConvertPrecisionNetType, ConvertPrecisionStage, ConvertPrecisionTarget, ConvertedNet};
use clap::Args;
use log::{debug, info, warn};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use thoosa_common::runner::common::cachable::{ DevicesWrap, WithDataRef};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::RunCtx;
use thoosa_common::ws::{FsWorkspace, Workspace};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::traits::{Data, WithId};
use thoosa_shared::{mono_key, ThoError};
use thoosa_store::{SqlStore, Store};
use thoosa_shared::tracked_await::track_await;

#[derive(Debug, Args, Clone)]
pub struct ExportArgs {
    #[arg(long)]
    pub target_dir: PathBuf,

    #[arg(long, value_enum)]
    pub target_precision: Option<ConvertPrecisionTarget>,
}

#[derive(Debug, Clone, DummyData)]
pub struct ExportInput {
    id: SynKey,
}

impl ExportInput {
    pub fn new() -> Self {
        Self { id: SynKey::new([mono_key!("export", "export")]) }
    }
}

impl Input for ExportInput {}

impl WithId for ExportInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Clone)]
pub struct ExportOutput {
    id: SynKey,
    target_dir: PathBuf,
    label_net_run_config_ids: Vec<RunConfigId>,
    motif_net_run_config_ids: Vec<RunConfigId>,
    copied_models: Vec<PathBuf>,
    converted: Vec<ConvertedNet>,
}

impl ExportOutput {
    fn new(target_dir: PathBuf, label_net_run_config_ids: Vec<RunConfigId>, motif_net_run_config_ids: Vec<RunConfigId>, copied_models: Vec<PathBuf>, converted: Vec<ConvertedNet>) -> Self {
        Self { id: SynKey::new([mono_key!("export", "export_output")]), target_dir, label_net_run_config_ids, motif_net_run_config_ids, copied_models, converted }
    }

    pub fn target_dir(&self) -> &Path {
        &self.target_dir
    }

    pub fn label_net_run_config_ids(&self) -> &[RunConfigId] {
        &self.label_net_run_config_ids
    }

    pub fn motif_net_run_config_ids(&self) -> &[RunConfigId] {
        &self.motif_net_run_config_ids
    }

    pub fn copied_models(&self) -> &[PathBuf] {
        &self.copied_models
    }

    pub fn converted(&self) -> &[ConvertedNet] {
        &self.converted
    }
}

impl Data for ExportOutput {
    type Metrics = ();
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        0
    }
}

impl WithId for ExportOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for ExportOutput {}

#[derive(Debug)]
pub struct Export<C>
where
    C: RunCtx + 'static,
{
    id: SynKey,
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    args: ExportArgs,
    pd: PhantomData<C>,
}

impl<C> Export<C>
where
    C: RunCtx + 'static,
{
    pub fn new(ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &Arc<Box<dyn Store>>, args: ExportArgs) -> Self {
        Self { id: SynKey::new([mono_key!("export", "stage")]), ws: Arc::clone(ws), store: Arc::clone(store), args, pd: PhantomData }
    }
}

impl<C> WithId for Export<C>
where
    C: RunCtx + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<C> Stage for Export<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = ExportInput;
    type Out = ExportOutput;
}

impl<C> AsyncStage for Export<C>
where
    C: RunCtx + 'static,
{
    type RunInput = Self::In<'static>;
    type RunOutput = Arc<DevicesWrap<Self::Out>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as WithId>::ID {
        WithId::id(input)
    }
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "Export::run"))]
    async fn run(self: Arc<Self>, _input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let target_ws = self.create_target_workspace()?;
        let target_store_path = { target_ws.read().store_path().clone() };
        {
            let target_store = track_await(SqlStore::from_path(&target_store_path)).await?;
            drop(target_store);
        }

        let label_net_run_config_ids = track_await(self.label_net_run_config_ids()).await?;
        if label_net_run_config_ids.is_empty() {
            warn!("Export::run: no label net run configs found for export");
        }

        let (motif_net_run_config_ids, copied_models) = track_await(self.copy_required_models(&target_ws, label_net_run_config_ids.as_slice())).await?;
        track_await(self.store.export_apply_label_nets_records(target_store_path, label_net_run_config_ids.clone())).await?;

        let converted = if let Some(target_precision) = self.args.target_precision {
            let output = track_await(Arc::new(ConvertPrecisionStage::<C>::new(&target_ws, ConvertPrecisionArgs::new(ConvertPrecisionBy::Type(ConvertPrecisionNetType::All), target_precision)))
                .run(ConvertPrecisionInput::new(), ctx.branch(true))).await?;
            output.converted().to_vec()
        } else {
            Vec::new()
        };

        info!("Export::run: exported target_dir={:?} label_net_run_config_ids={label_net_run_config_ids:?} motif_net_run_config_ids={motif_net_run_config_ids:?} copied_models={} converted={}", self.args.target_dir, copied_models.len(), converted.len());
        Ok(Arc::new(DevicesWrap::wrap_plain(ExportOutput::new(self.args.target_dir.clone(), label_net_run_config_ids, motif_net_run_config_ids, copied_models, converted))))
    }
}

impl<C> Export<C>
where
    C: RunCtx + 'static,
{
    fn create_target_workspace(&self) -> Result<Arc<RwLock<Box<dyn Workspace>>>, ThoError> {
        let (channels, max_input, sample_rate) = {
            let ws = self.ws.read();
            (ws.channels(), ws.max_input(), ws.sample_rate())
        };
        Ok(Arc::new(RwLock::new(FsWorkspace::new(&self.args.target_dir, channels, max_input, sample_rate)?)))
    }

    async fn label_net_run_config_ids(&self) -> Result<Vec<RunConfigId>, ThoError> {
        let labels = track_await(self.store.list_labels(true)).await?;
        let mut ids = Vec::new();
        for label in labels {
            ids.extend(track_await(self.store.list_label_nets_run_config_ids(&label)).await?);
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    async fn copy_required_models(&self, target_ws: &Arc<RwLock<Box<dyn Workspace>>>, label_net_run_config_ids: &[RunConfigId]) -> Result<(Vec<RunConfigId>, Vec<PathBuf>), ThoError> {
        let label_nets = track_await(self.store.list_trained_nets(label_net_run_config_ids.to_vec(), None, None)).await?;
        let mut motif_net_run_config_ids = Vec::new();
        for label_net in &label_nets {
            motif_net_run_config_ids.extend(track_await(self.store.list_input_nets(label_net.id)).await?);
        }
        motif_net_run_config_ids.sort_unstable();
        motif_net_run_config_ids.dedup();

        let motif_nets = track_await(self.store.list_trained_nets(motif_net_run_config_ids.clone(), None, None)).await?;
        let mut copied = Vec::new();
        let mut copied_targets = HashSet::new();
        {
            let source_ws = self.ws.read();
            let target_ws = target_ws.read();
            for label_net in &label_nets {
                let target_path = copy_net_path(source_ws.labels_nets_path(), target_ws.labels_nets_path(), &label_net.net_path)?;
                if copied_targets.insert(target_path.clone()) {
                    copied.push(target_path);
                }
            }
            for motif_net in &motif_nets {
                let target_path = copy_net_path(source_ws.motifs_nets_path(), target_ws.motifs_nets_path(), &motif_net.net_path)?;
                if copied_targets.insert(target_path.clone()) {
                    copied.push(target_path);
                }
            }
        }

        Ok((motif_net_run_config_ids, copied))
    }
}

fn copy_net_path(source_root: &Path, target_root: &Path, net_path: &Path) -> Result<PathBuf, ThoError> {
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
        let parent = target.parent().ok_or_else(|| ThoError::new(format!("copy_net_path: target {target:?} has no parent")))?;
        std::fs::create_dir_all(parent)?;
        std::fs::copy(&source, &target).map_err(|e| ThoError::from_error(format!("copy model file {source:?} to {target:?}"), e))?;
    } else {
        return Err(ThoError::new(format!("copy_net_path: source model path {source:?} does not exist")));
    }

    Ok(target)
}

fn copy_dir_all(source: &Path, target: &Path) -> Result<(), ThoError> {
    std::fs::create_dir_all(target)?;
    for entry in std::fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let target_path = target.join(entry.file_name());
        if source_path.is_dir() {
            copy_dir_all(&source_path, &target_path)?;
        } else {
            std::fs::copy(&source_path, &target_path).map_err(|e| ThoError::from_error(format!("copy model file {source_path:?} to {target_path:?}"), e))?;
        }
    }
    Ok(())
}
