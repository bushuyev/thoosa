use clap::Args;
use log::{debug, info, warn};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::RunCtx;
use thoosa_common::ws::{Workspace, NET_PARAMS_FILE, NET_WEIGHTS_FILE};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Data, WithId};
use thoosa_shared::{collect_futures_with_progress, mono_key, recurse_files, FuturesOrdering, ThoError};
use thoosa_store::{Store, TrainedNetEntry};
use thoosa_shared::tracked_await::track_await;


#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurgeDate(String);

impl PurgeDate {
    pub fn new(date: impl Into<String>) -> Result<Self, String> {
        let date = date.into();
        validate_purge_date(&date)?;
        Ok(Self(date))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

impl FromStr for PurgeDate {
    type Err = String;

    fn from_str(v: &str) -> Result<Self, Self::Err> {
        Self::new(v.trim().trim_matches(|ch| ch == '\'' || ch == '"'))
    }
}

fn validate_purge_date(date: &str) -> Result<(), String> {
    let bytes = date.as_bytes();
    if bytes.len() == 10
        && bytes[4] == b'-'
        && bytes[7] == b'-'
        && bytes.iter().enumerate().all(|(idx, byte)| idx == 4 || idx == 7 || byte.is_ascii_digit())
    {
        Ok(())
    } else {
        Err("date must be YYYY-MM-DD".to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PurgeLabelModelsBy {
    Date(PurgeDate),
    Loss(f32),
}

impl FromStr for PurgeLabelModelsBy {
    type Err = String;

    fn from_str(v: &str) -> Result<Self, Self::Err> {
        let value = v.trim().trim_matches(|ch| ch == '\'' || ch == '"');
        let (kind, raw_value) = value
            .split_once(':')
            .ok_or_else(|| format!("unexpected label purge selector {v:?}; expected date:YYYY-MM-DD or loss:<value>"))?;
        match kind.trim().to_ascii_lowercase().as_str() {
            "date" | "run_on" | "before" => Ok(Self::Date(PurgeDate::from_str(raw_value)?)),
            "loss" => Ok(Self::Loss(raw_value.trim().parse::<f32>().map_err(|e| format!("bad loss value {raw_value:?}: {e}"))?)),
            other => Err(format!("unexpected label purge selector {other:?}; expected date or loss")),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum PurgeMode {
    /// Remove motif, label, and final model directories that are present on disk but absent from `trained_nets`.
    /// Example: `purge-models --purge-mode lost`
    PurgeLost,
    /// Remove label model directories and their `trained_nets` rows by training run date or loss threshold.
    /// Examples: `purge-models --purge-mode label:date:2026-05-18`, `purge-models --purge-mode label:loss:0.1`
    PurgeLabelModels(PurgeLabelModelsBy),
    /// Remove motif model directories and their `trained_nets` rows when no `trained_nets_input` row uses them.
    /// Example: `purge-models --purge-mode unused_motifs`
    PurgeUnusedMotifModels,
}

impl FromStr for PurgeMode {
    type Err = String;

    fn from_str(v: &str) -> Result<Self, Self::Err> {
        let value = v.trim().trim_matches(|ch| ch == '\'' || ch == '"');
        let lower = value.to_ascii_lowercase();
        match lower.as_str() {
            "lost" | "purge_lost" | "purge-lost" => return Ok(Self::PurgeLost),
            "unused_motifs" | "unused-motifs" | "unused_motif_models" | "unused-motif-models" | "purge_unused_motif_models" | "purge-unused-motif-models" => return Ok(Self::PurgeUnusedMotifModels),
            _ => {}
        }

        for prefix in ["label:", "labels:", "label_models:", "label-models:", "purge_label_models:", "purge-label-models:"] {
            if lower.starts_with(prefix) {
                return Ok(Self::PurgeLabelModels(PurgeLabelModelsBy::from_str(&value[prefix.len()..])?));
            }
        }

        Err(format!("unexpected purge mode {v:?}; expected lost, label:date:<YYYY-MM-DD>, label:loss:<value>, or unused_motifs"))
    }
}

#[derive(Debug, Args, Clone)]
pub struct PurgeModelsArgs {
    #[arg(long)]
    purge_mode: PurgeMode,
}

impl PurgeModelsArgs {
    pub fn new(purge_mode: PurgeMode) -> Self {
        Self { purge_mode }
    }

    pub fn purge_mode(&self) -> PurgeMode {
        self.purge_mode.clone()
    }
}

#[derive(Debug, Clone, DummyData)]
pub struct PurgeModelsInput {
    id: SynKey,
}

impl PurgeModelsInput {
    pub fn new() -> Self {
        Self { id: SynKey::new([mono_key!("purge_models", "purge_models")]) }
    }
}

impl Input for PurgeModelsInput {}

impl WithId for PurgeModelsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PurgeModelType {
    Motifs,
    Labels,
    Final,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PurgedModel {
    pub net_type: PurgeModelType,
    pub net_id: Option<u32>,
    pub net_path: PathBuf,
    pub disk_path: PathBuf,
    pub removed_from_store: bool,
    pub removed_from_disk: bool,
}

#[derive(Debug, Clone)]
pub struct PurgeModelsOutput {
    id: SynKey,
    purged: Vec<PurgedModel>,
}

impl PurgeModelsOutput {
    fn new(purged: Vec<PurgedModel>) -> Self {
        Self { id: SynKey::new([mono_key!("purge_models", "purge_models_output")]), purged }
    }

    pub fn purged(&self) -> &[PurgedModel] {
        &self.purged
    }
}

impl Data for PurgeModelsOutput {
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

impl WithId for PurgeModelsOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for PurgeModelsOutput {}

#[derive(Debug)]
pub struct PurgeModelsStage<C>
where
    C: RunCtx + 'static,
{
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    args: PurgeModelsArgs,
    pd: PhantomData<C>,
}

impl<C> PurgeModelsStage<C>
where
    C: RunCtx + 'static,
{
    pub fn new(ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &Arc<Box<dyn Store>>, args: PurgeModelsArgs) -> Self {
        Self {
            ws: Arc::clone(ws),
            store: Arc::clone(store),
            args,
            pd: PhantomData,
        }
    }
}

impl<C> Stage for PurgeModelsStage<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = PurgeModelsInput;
    type Out = PurgeModelsOutput;
}

impl<C> AsyncStage for PurgeModelsStage<C>
where
    C: RunCtx + 'static,
{
    type RunInput = Arc<DevicesWrap<Self::In<'static>>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        thoosa_shared::traits::WithId::id(input)
    }
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "PurgeModelsStage::run"))]
    async fn run(self: Arc<Self>, _input: Arc<DevicesWrap<Self::In<'static>>>, _ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let roots = {
            let ws_guard = self.ws.read();
            ModelRoots::from_ws(ws_guard.as_ref())
        };
        let purge_mode = self.args.purge_mode();
        let entries = track_await(selected_purge_entries(&roots, &self.store, purge_mode)).await?;

        let entries_len = entries.len();
        if entries_len == 0 {
            warn!("PurgeModelsStage::run: no models matched {:?}", self.args.purge_mode());
            return Ok(Arc::new(DevicesWrap::wrap(PurgeModelsOutput::new(vec![]))));
        }

        let ids_to_delete = entries
            .iter()
            .filter(|entry| entry.remove_from_store)
            .filter_map(|entry| entry.net_id)
            .collect::<Vec<_>>();
        let futures = entries.into_iter().map(|entry| {
            async move {
                track_await(tokio::task::spawn_blocking(move || {
                    tracing::info_span!(
                        "purge_models::purge_entry",
                        kind = "spawn_blocking",
                        file = file!(),
                        line = line!(),
                        task = "purge selected model entry",
                    ).in_scope(|| purge_entry(entry))
                })).await
                    .map_err(|e| ThoError::from_error("join purge models worker", e))?
            }
        });

        let purged = track_await(collect_futures_with_progress(
            futures,
            entries_len,
            FuturesOrdering::Unordered,
            "PurgeModelsStage::run",
        )).await;

        if purged.len() != entries_len {
            return Err(ThoError::new(format!("PurgeModelsStage::run: purged {} of {entries_len} models", purged.len())));
        }

        if !ids_to_delete.is_empty() {
            track_await(self.store.delete_trained_nets(ids_to_delete)).await?;
        }

        info!("PurgeModelsStage::run: purged {} models for {:?}", purged.len(), self.args.purge_mode());
        Ok(Arc::new(DevicesWrap::wrap(PurgeModelsOutput::new(purged))))
    }
}

#[derive(Debug, Clone)]
struct ModelRoots {
    motifs: PathBuf,
    labels: PathBuf,
    final_models: PathBuf,
}

impl ModelRoots {
    fn from_ws(ws: &dyn Workspace) -> Self {
        Self {
            motifs: ws.motifs_nets_path().clone(),
            labels: ws.labels_nets_path().clone(),
            final_models: ws.final_nets_path().clone(),
        }
    }

    fn root(&self, net_type: PurgeModelType) -> &Path {
        match net_type {
            PurgeModelType::Motifs => &self.motifs,
            PurgeModelType::Labels => &self.labels,
            PurgeModelType::Final => &self.final_models,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PurgeEntry {
    net_type: PurgeModelType,
    net_id: Option<u32>,
    net_path: PathBuf,
    disk_path: PathBuf,
    remove_from_store: bool,
}

async fn selected_purge_entries(roots: &ModelRoots, store: &Arc<Box<dyn Store>>, purge_mode: PurgeMode) -> Result<Vec<PurgeEntry>, ThoError> {
    match purge_mode {
        PurgeMode::PurgeLost => purge_lost_entries(roots, track_await(store.list_trained_net_entries()).await?),
        PurgeMode::PurgeLabelModels(by) => {
            let rows = match by {
                PurgeLabelModelsBy::Date(date) => track_await(store.list_label_trained_nets_before(date.as_str().to_string())).await?,
                PurgeLabelModelsBy::Loss(loss) => track_await(store.list_label_trained_nets_loss_gt(loss)).await?,
            };
            Ok(rows_to_purge_entries(roots, PurgeModelType::Labels, rows))
        }
        PurgeMode::PurgeUnusedMotifModels => {
            let rows = track_await(store.list_unused_motif_trained_nets()).await?;
            Ok(rows_to_purge_entries(roots, PurgeModelType::Motifs, rows))
        }
    }
}

fn purge_lost_entries(roots: &ModelRoots, stored_entries: Vec<TrainedNetEntry>) -> Result<Vec<PurgeEntry>, ThoError> {
    let stored_net_paths = stored_entries.into_iter().map(|entry| entry.net_path).collect::<HashSet<_>>();
    let mut entries = Vec::new();
    for net_type in [PurgeModelType::Motifs, PurgeModelType::Labels, PurgeModelType::Final] {
        entries.extend(
            disk_entries_for_type(roots, net_type)?
                .into_iter()
                .filter(|entry| !stored_net_paths.contains(&entry.net_path)),
        );
    }
    Ok(entries)
}

fn disk_entries_for_type(roots: &ModelRoots, net_type: PurgeModelType) -> Result<Vec<PurgeEntry>, ThoError> {
    let root = roots.root(net_type);
    if !root.exists() {
        return Ok(vec![]);
    }

    recurse_files(root, &|path| path.file_name().and_then(|v| v.to_str()) == Some(NET_WEIGHTS_FILE))?
        .into_iter()
        .filter_map(|weights_path| disk_entry_from_weights(root, net_type, &weights_path))
        .collect::<Vec<_>>()
        .dedup_purge_entries()
}

fn disk_entry_from_weights(root: &Path, net_type: PurgeModelType, weights_path: &Path) -> Option<PurgeEntry> {
    let disk_path = weights_path.parent()?.to_path_buf();
    if !disk_path.join(NET_PARAMS_FILE).exists() {
        return None;
    }
    let net_path = disk_path.strip_prefix(root).ok()?.to_path_buf();
    if net_path.as_os_str().is_empty() {
        return None;
    }
    Some(PurgeEntry {
        net_type,
        net_id: None,
        net_path,
        disk_path,
        remove_from_store: false,
    })
}

trait DedupPurgeEntries {
    fn dedup_purge_entries(self) -> Result<Vec<PurgeEntry>, ThoError>;
}

impl DedupPurgeEntries for Vec<PurgeEntry> {
    fn dedup_purge_entries(self) -> Result<Vec<PurgeEntry>, ThoError> {
        let mut seen = HashSet::new();
        Ok(self.into_iter().filter(|entry| seen.insert(entry.disk_path.clone())).collect())
    }
}

fn rows_to_purge_entries(roots: &ModelRoots, net_type: PurgeModelType, rows: Vec<TrainedNetEntry>) -> Vec<PurgeEntry> {
    rows.into_iter()
        .map(|row| PurgeEntry {
            net_type,
            net_id: Some(row.id),
            disk_path: roots.root(net_type).join(&row.net_path),
            net_path: row.net_path,
            remove_from_store: true,
        })
        .collect()
}

fn purge_entry(entry: PurgeEntry) -> Result<PurgedModel, ThoError> {
    debug!("purge_entry: entry={entry:?}");
    let removed_from_disk = remove_disk_path(&entry.disk_path)?;
    Ok(PurgedModel {
        net_type: entry.net_type,
        net_id: entry.net_id,
        net_path: entry.net_path,
        disk_path: entry.disk_path,
        removed_from_store: entry.remove_from_store,
        removed_from_disk,
    })
}

fn remove_disk_path(path: &Path) -> Result<bool, ThoError> {
    if !path.exists() {
        return Ok(false);
    }
    if path.is_dir() {
        std::fs::remove_dir_all(path)?;
    } else {
        std::fs::remove_file(path)?;
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::{PurgeDate, PurgeLabelModelsBy, PurgeMode, PurgeModelType, PurgeModelsArgs, PurgeModelsInput, PurgeModelsStage};
    use crate::cli::{Cli, Commands};
    use clap::Parser;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::train_nets::{TrainLabelNetRunConfig, TrainMotifNetRunConfig};
    use thoosa_common::ws::{FsWorkspace, Workspace, NET_PARAMS_FILE, NET_WEIGHTS_FILE};
    use thoosa_shared::run_configs::{RunConfig, RUN_ID};
    use thoosa_shared::{cdt_for_dir, ThoError};
    use thoosa_store::{SqlStore, Store};

    #[test]
    fn test_purge_models_command_parses() {
        let cli = Cli::parse_from(["thoosa_main", "purge-models", "--purge-mode", "label:loss:0.1"]);

        match cli.command {
            Some(Commands::PurgeModels(args)) => assert_eq!(args.purge_mode(), PurgeMode::PurgeLabelModels(PurgeLabelModelsBy::Loss(0.1))),
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_purge_models_stage_purges_lost_disk_models() -> Result<(), ThoError> {
        let (ws, store) = test_ws_store("purge_models_lost").await?;
        let kept_path = PathBuf::from("kept_label").join("1");
        let lost_motif_path = PathBuf::from("0").join("42_2");
        let lost_label_path = PathBuf::from("lost_label").join("3");
        let lost_final_path = PathBuf::from("4");

        {
            let ws_guard = ws.read();
            create_dummy_model(ws_guard.labels_nets_path(), &kept_path)?;
            create_dummy_model(ws_guard.motifs_nets_path(), &lost_motif_path)?;
            create_dummy_model(ws_guard.labels_nets_path(), &lost_label_path)?;
            create_dummy_model(ws_guard.final_nets_path(), &lost_final_path)?;
        }

        let run_config_id = store.save_run_config(None, &TrainLabelNetRunConfig::type_name(), "kept", HashMap::new()).await?;
        store.save_trained_net(run_config_id, kept_path.clone(), 0.01, None).await?;

        let output = run_stage(&ws, &store, PurgeMode::PurgeLost).await?;
        assert_eq!(output.purged().len(), 3);
        assert!(output.purged().iter().any(|model| model.net_type == PurgeModelType::Motifs && model.net_path == lost_motif_path));
        assert!(ws.read().labels_nets_path().join(&kept_path).exists());
        assert!(!ws.read().motifs_nets_path().join(&lost_motif_path).exists());
        assert!(!ws.read().labels_nets_path().join(&lost_label_path).exists());
        assert!(!ws.read().final_nets_path().join(&lost_final_path).exists());
        assert_eq!(store.list_trained_nets(vec![run_config_id], None, None).await?.len(), 1);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_purge_models_stage_purges_label_models_by_date() -> Result<(), ThoError> {
        let (ws, store) = test_ws_store("purge_models_label_date").await?;
        let old_path = PathBuf::from("label").join("1");
        let new_path = PathBuf::from("label").join("2");
        {
            let ws_guard = ws.read();
            create_dummy_model(ws_guard.labels_nets_path(), &old_path)?;
            create_dummy_model(ws_guard.labels_nets_path(), &new_path)?;
        }

        let old_run_config_id = store.save_run_config(None, &TrainLabelNetRunConfig::type_name(), "old", HashMap::new()).await?;
        let new_run_config_id = store.save_run_config(None, &TrainLabelNetRunConfig::type_name(), "new", HashMap::new()).await?;
        set_run_on(&store, old_run_config_id, "2026-05-17 12:00:00").await?;
        set_run_on(&store, new_run_config_id, "2026-05-18 12:00:00").await?;
        store.save_trained_net(old_run_config_id, old_path.clone(), 0.01, None).await?;
        store.save_trained_net(new_run_config_id, new_path.clone(), 0.01, None).await?;

        let output = run_stage(&ws, &store, PurgeMode::PurgeLabelModels(PurgeLabelModelsBy::Date(PurgeDate::new("2026-05-18").unwrap()))).await?;
        assert_eq!(output.purged().len(), 1);
        assert_eq!(output.purged()[0].net_path, old_path);
        assert_eq!(store.list_trained_nets(vec![old_run_config_id], None, None).await?.len(), 0);
        assert_eq!(store.list_trained_nets(vec![new_run_config_id], None, None).await?.len(), 1);
        assert!(!ws.read().labels_nets_path().join("label").join("1").exists());
        assert!(ws.read().labels_nets_path().join("label").join("2").exists());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_purge_models_stage_purges_label_models_by_loss() -> Result<(), ThoError> {
        let (ws, store) = test_ws_store("purge_models_label_loss").await?;
        let high_loss_path = PathBuf::from("label").join("1");
        let low_loss_path = PathBuf::from("label").join("2");
        {
            let ws_guard = ws.read();
            create_dummy_model(ws_guard.labels_nets_path(), &high_loss_path)?;
            create_dummy_model(ws_guard.labels_nets_path(), &low_loss_path)?;
        }

        let high_loss_run_config_id = store.save_run_config(None, &TrainLabelNetRunConfig::type_name(), "high_loss", HashMap::new()).await?;
        let low_loss_run_config_id = store.save_run_config(None, &TrainLabelNetRunConfig::type_name(), "low_loss", HashMap::new()).await?;
        store.save_trained_net(high_loss_run_config_id, high_loss_path.clone(), 0.2, None).await?;
        store.save_trained_net(low_loss_run_config_id, low_loss_path.clone(), 0.05, None).await?;

        let output = run_stage(&ws, &store, PurgeMode::PurgeLabelModels(PurgeLabelModelsBy::Loss(0.1))).await?;
        assert_eq!(output.purged().len(), 1);
        assert_eq!(output.purged()[0].net_path, high_loss_path);
        assert_eq!(store.list_trained_nets(vec![high_loss_run_config_id], None, None).await?.len(), 0);
        assert_eq!(store.list_trained_nets(vec![low_loss_run_config_id], None, None).await?.len(), 1);
        assert!(!ws.read().labels_nets_path().join("label").join("1").exists());
        assert!(ws.read().labels_nets_path().join("label").join("2").exists());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_purge_models_stage_purges_unused_motif_models() -> Result<(), ThoError> {
        let (ws, store) = test_ws_store("purge_models_unused_motifs").await?;
        let used_motif_path = PathBuf::from("0").join("10_1");
        let unused_motif_path = PathBuf::from("0").join("11_2");
        let label_path = PathBuf::from("label").join("3");
        {
            let ws_guard = ws.read();
            create_dummy_model(ws_guard.motifs_nets_path(), &used_motif_path)?;
            create_dummy_model(ws_guard.motifs_nets_path(), &unused_motif_path)?;
            create_dummy_model(ws_guard.labels_nets_path(), &label_path)?;
        }

        let used_motif_run_config_id = store.save_run_config(None, &TrainMotifNetRunConfig::type_name(), "used", HashMap::new()).await?;
        let unused_motif_run_config_id = store.save_run_config(None, &TrainMotifNetRunConfig::type_name(), "unused", HashMap::new()).await?;
        store.save_trained_net(used_motif_run_config_id, used_motif_path.clone(), 0.01, None).await?;
        store.save_trained_net(unused_motif_run_config_id, unused_motif_path.clone(), 0.01, None).await?;
        let used_motif_id = store.list_trained_nets(vec![used_motif_run_config_id], None, None).await?[0].id;

        let label_run_config_id = store.save_run_config(None, &TrainLabelNetRunConfig::type_name(), "label", HashMap::new()).await?;
        store.save_trained_net(label_run_config_id, label_path, 0.01, Some(vec![used_motif_id])).await?;

        let output = run_stage(&ws, &store, PurgeMode::PurgeUnusedMotifModels).await?;
        assert_eq!(output.purged().len(), 1);
        assert_eq!(output.purged()[0].net_path, unused_motif_path);
        assert_eq!(store.list_trained_nets(vec![used_motif_run_config_id], None, None).await?.len(), 1);
        assert_eq!(store.list_trained_nets(vec![unused_motif_run_config_id], None, None).await?.len(), 0);
        assert!(ws.read().motifs_nets_path().join("0").join("10_1").exists());
        assert!(!ws.read().motifs_nets_path().join("0").join("11_2").exists());

        Ok(())
    }

    async fn test_ws_store(name: &str) -> Result<(Arc<RwLock<Box<dyn Workspace>>>, Arc<Box<dyn Store>>), ThoError> {
        let test_dir = Path::new("/tmp").join(format!("{name}_{}", cdt_for_dir()));
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 2, 100, 100)?));
        let store = Arc::new(SqlStore::in_mem().await?);
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");
        Ok((ws, store))
    }

    async fn run_stage(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        purge_mode: PurgeMode,
    ) -> Result<super::PurgeModelsOutput, ThoError> {
        let stage = Arc::new(PurgeModelsStage::<RunCtxImpl>::new(ws, store, PurgeModelsArgs::new(purge_mode)));
        let output = stage
            .run(
                Arc::new(DevicesWrap::wrap(PurgeModelsInput::new())),
                RunCtxImpl::new(0, "purge_models_test"),
            )
            .await?;
        Arc::into_inner(output).expect("stage output should not be shared").take_data(0).await
    }

    fn create_dummy_model(root: &Path, net_path: &Path) -> Result<(), ThoError> {
        let model_dir = root.join(net_path);
        std::fs::create_dir_all(&model_dir)?;
        std::fs::write(model_dir.join(NET_WEIGHTS_FILE), "")?;
        std::fs::write(model_dir.join(NET_PARAMS_FILE), "{}")?;
        Ok(())
    }

    async fn set_run_on(store: &Arc<Box<dyn Store>>, run_config_id: u32, run_on: &str) -> Result<(), ThoError> {
        store.update_run_config_run_on(run_config_id, run_on.to_string()).await
    }
}
