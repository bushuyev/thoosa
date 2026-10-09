use anyhow::Context as _;
use clap::{Args, ValueEnum};
use log::{debug, info, warn};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::fmt::Debug;
use std::hash::Hash;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};
use thoosa_cpu_data_derive::CpuData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{AsyncStage, RunCtx, Stage};
use thoosa_shared::{FuturesOrdering, ThoError, collect_futures_with_progress, recurse_files};

use crate::utils::parse_run_config_ids_vec;
use crate::ws::{NET_PARAMS_FILE, NET_WEIGHTS_FILE, Workspace};

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum ConvertPrecisionTarget {
    #[value(alias = "half")]
    #[value(alias = "fp16")]
    Fp16,
    #[value(alias = "bfloat16")]
    #[value(alias = "bf16")]
    Bf16,
    #[value(alias = "float")]
    #[value(alias = "fp32")]
    Fp32,
    #[value(alias = "double")]
    #[value(alias = "fp64")]
    Fp64,
}

impl ConvertPrecisionTarget {
    pub fn kind(self) -> Kind {
        match self {
            Self::Fp16 => Kind::Half,
            Self::Bf16 => Kind::BFloat16,
            Self::Fp32 => Kind::Float,
            Self::Fp64 => Kind::Double,
        }
    }

    fn apply(self, vs: &mut VarStore) {
        match self {
            Self::Fp16 => vs.half(),
            Self::Bf16 => vs.bfloat16(),
            Self::Fp32 => vs.float(),
            Self::Fp64 => vs.double(),
        }
    }
}

/// Describes a workspace-specific model layout without exposing its concrete model types to common code.
pub trait PrecisionModelType<W>: Debug + Clone + Copy + Eq + Hash + Send + Sync + FromStr<Err = anyhow::Error> + 'static
where
    W: Workspace + ?Sized,
{
    fn all() -> &'static [Self];
    fn concrete(self) -> &'static [Self];
    fn root(self, ws: &W) -> anyhow::Result<PathBuf>;
    fn run_config_id(self, weights_path: &Path) -> Option<RunConfigId>;
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ConvertPrecisionBy<N> {
    Path(PathBuf),
    RunId(Vec<RunConfigId>),
    Type(N),
}

impl<N> FromStr for ConvertPrecisionBy<N>
where
    N: FromStr<Err = anyhow::Error>,
{
    type Err = anyhow::Error;

    fn from_str(v: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = v.strip_prefix("path:") {
            let path = rest.trim().trim_matches(|ch| ch == '\'' || ch == '"');
            if path.is_empty() {
                anyhow::bail!("path payload is empty");
            }
            Ok(Self::Path(PathBuf::from(path)))
        } else if let Some(rest) = v.strip_prefix("run_config_id:").or_else(|| v.strip_prefix("run_config_ids:")) {
            let run_config_ids = parse_run_config_ids_vec(rest)?;
            if run_config_ids.is_empty() {
                anyhow::bail!("run_config_id payload is empty");
            }
            Ok(Self::RunId(run_config_ids))
        } else if let Some(rest) = v.strip_prefix("type:").or_else(|| v.strip_prefix("net_type:")) {
            Ok(Self::Type(N::from_str(rest)?))
        } else {
            anyhow::bail!("unexpected selector {v:?}; expected path:<path>, run_config_id:<id[,id]>, or type:<model-type>")
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct ConvertPrecisionArgs<N>
where
    N: Clone + FromStr<Err = anyhow::Error> + Send + Sync + 'static,
{
    #[arg(long)]
    nets_by: ConvertPrecisionBy<N>,

    #[arg(long, value_enum)]
    target_precision: ConvertPrecisionTarget,
}

impl<N> ConvertPrecisionArgs<N>
where
    N: Clone + FromStr<Err = anyhow::Error> + Send + Sync + 'static,
{
    pub fn new(nets_by: ConvertPrecisionBy<N>, target_precision: ConvertPrecisionTarget) -> Self {
        Self { nets_by, target_precision }
    }

    pub fn nets_by(&self) -> ConvertPrecisionBy<N> {
        self.nets_by.clone()
    }

    pub fn target_precision(&self) -> ConvertPrecisionTarget {
        self.target_precision
    }
}

#[derive(Debug, Clone, CpuData)]
pub struct ConvertPrecisionInput {}

impl ConvertPrecisionInput {
    pub fn new() -> Self {
        Self {}
    }
}

impl Input for ConvertPrecisionInput {}

#[derive(Debug, Clone)]
pub struct ConvertedNet<N> {
    pub net_type: N,
    pub weights_path: PathBuf,
    pub float_vars: usize,
    pub total_vars: usize,
}

#[derive(Debug, Clone, CpuData)]
pub struct ConvertPrecisionOutput<N>
where
    N: Debug,
{
    converted: Vec<ConvertedNet<N>>,
}

impl<N> ConvertPrecisionOutput<N>
where
    N: Debug,
{
    fn new(converted: Vec<ConvertedNet<N>>) -> Self {
        Self { converted }
    }

    pub fn converted(&self) -> &[ConvertedNet<N>] {
        &self.converted
    }
}

impl<N> Output for ConvertPrecisionOutput<N> where N: Debug {}

#[derive(Debug)]
pub struct ConvertPrecisionStage<C, W, N>
where
    C: RunCtx + 'static,
    W: Workspace + ?Sized + 'static,
    N: PrecisionModelType<W>,
{
    ws: Arc<RwLock<Box<W>>>,
    args: ConvertPrecisionArgs<N>,
    pd: PhantomData<C>,
}

impl<C, W, N> ConvertPrecisionStage<C, W, N>
where
    C: RunCtx + 'static,
    W: Workspace + ?Sized + 'static,
    N: PrecisionModelType<W>,
{
    pub fn new(ws: &Arc<RwLock<Box<W>>>, args: ConvertPrecisionArgs<N>) -> Self {
        Self { ws: Arc::clone(ws), args, pd: PhantomData }
    }
}

impl<C, W, N> Stage for ConvertPrecisionStage<C, W, N>
where
    C: RunCtx + 'static,
    W: Workspace + ?Sized + 'static,
    N: PrecisionModelType<W>,
{
    type Ctx = C;
    type In<'a> = ConvertPrecisionInput;
    type Out = Arc<ConvertPrecisionOutput<N>>;
}

impl<C, W, N> AsyncStage for ConvertPrecisionStage<C, W, N>
where
    C: RunCtx + 'static,
    W: Workspace + ?Sized + 'static,
    N: PrecisionModelType<W>,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "ConvertPrecisionStage::run"))]
    async fn run(self: Arc<Self>, _input: Self::In<'static>, _ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        let entries = {
            let ws_guard = self.ws.read();
            selected_entries(ws_guard.as_ref(), self.args.nets_by())?
        };
        let entries_len = entries.len();
        if entries_len == 0 {
            warn!("ConvertPrecisionStage::run: no nets matched {:?}", self.args.nets_by());
            return Ok(Arc::new(ConvertPrecisionOutput::new(vec![])));
        }

        let target_precision = self.args.target_precision();
        let futures = entries.into_iter().map(move |entry| {
            async move {
                Ok::<_, ThoError>(track_await(tokio::task::spawn_blocking(move || {
                    tracing::info_span!(
                        "convert_precision::convert_entry",
                        kind = "spawn_blocking",
                        file = file!(),
                        line = line!(),
                        task = format!("convert net entry to {target_precision:?}"),
                    ).in_scope(|| convert_entry(entry, target_precision))
                })).await.context("join convert precision worker")??)
            }
        });

        let converted = track_await(collect_futures_with_progress(
            futures,
            entries_len,
            FuturesOrdering::Unordered,
            "ConvertPrecisionStage::run",
        )).await?;

        if converted.len() != entries_len {
            return Err(ThoError::new(format!("ConvertPrecisionStage::run: converted {} of {entries_len} nets", converted.len())));
        }

        info!("ConvertPrecisionStage::run: converted {} nets to {:?}", converted.len(), target_precision);
        Ok(Arc::new(ConvertPrecisionOutput::new(converted)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NetEntry<N> {
    net_type: N,
    weights_path: PathBuf,
}

fn selected_entries<W, N>(ws: &W, selector: ConvertPrecisionBy<N>) -> Result<Vec<NetEntry<N>>, ThoError>
where
    W: Workspace + ?Sized,
    N: PrecisionModelType<W>,
{
    match selector {
        ConvertPrecisionBy::Path(path) => entries_for_path(ws, &path),
        ConvertPrecisionBy::RunId(run_config_ids) => {
            let run_config_ids = run_config_ids.into_iter().collect::<HashSet<_>>();
            entries_for_all_types::<W, N>(ws)?
                .into_iter()
                .filter(|entry| entry.net_type.run_config_id(&entry.weights_path).is_some_and(|run_config_id| run_config_ids.contains(&run_config_id)))
                .collect::<Vec<_>>()
                .dedup_entries()
        }
        ConvertPrecisionBy::Type(net_type) => entries_for_type(ws, net_type),
    }
}

trait DedupEntries<N> {
    fn dedup_entries(self) -> Result<Vec<NetEntry<N>>, ThoError>;
}

impl<N> DedupEntries<N> for Vec<NetEntry<N>> {
    fn dedup_entries(self) -> Result<Vec<NetEntry<N>>, ThoError> {
        let mut seen = HashSet::new();
        Ok(self.into_iter().filter(|entry| seen.insert(entry.weights_path.clone())).collect())
    }
}

fn entries_for_path<W, N>(ws: &W, path: &Path) -> Result<Vec<NetEntry<N>>, ThoError>
where
    W: Workspace + ?Sized,
    N: PrecisionModelType<W>,
{
    let model_types = N::all();
    let candidates = if path.is_absolute() {
        vec![path.to_path_buf()]
    } else {
        let mut candidates = vec![ws.models_path().join(path)];
        for net_type in model_types {
            candidates.push(net_type.root(ws)?.join(path));
        }
        candidates
    };

    let mut entries = Vec::new();
    for candidate in candidates {
        for net_type in model_types {
            if let Some(entry) = entry_from_candidate(ws, *net_type, &candidate)? {
                entries.push(entry);
            }
        }
    }
    let entries = entries.dedup_entries()?;

    if entries.is_empty() {
        return Err(ThoError::new(format!("ConvertPrecision: no net found for path {path:?}")));
    }
    if entries.len() > 1 {
        return Err(ThoError::new(format!("ConvertPrecision: path {path:?} is ambiguous and matched {:?}", entries.iter().map(|entry| &entry.weights_path).collect::<Vec<_>>())));
    }
    Ok(entries)
}

fn entries_for_all_types<W, N>(ws: &W) -> Result<Vec<NetEntry<N>>, ThoError>
where
    W: Workspace + ?Sized,
    N: PrecisionModelType<W>,
{
    let mut entries = Vec::new();
    for net_type in N::all() {
        let root = net_type.root(ws)?;
        entries.extend(
            recurse_files(&root, &|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))?
                .into_iter()
                .filter_map(|weights_path| net_entry_from_base(*net_type, &weights_path)),
        );
    }
    entries.dedup_entries()
}

fn entries_for_type<W, N>(ws: &W, net_type: N) -> Result<Vec<NetEntry<N>>, ThoError>
where
    W: Workspace + ?Sized,
    N: PrecisionModelType<W>,
{
    let mut entries = Vec::new();
    for concrete_type in net_type.concrete() {
        let root = concrete_type.root(ws)?;
        entries.extend(
            recurse_files(&root, &|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))?
                .into_iter()
                .filter_map(|weights_path| net_entry_from_base(*concrete_type, &weights_path)),
        );
    }
    entries.dedup_entries()
}

fn entry_from_candidate<W, N>(ws: &W, net_type: N, candidate: &Path) -> Result<Option<NetEntry<N>>, ThoError>
where
    W: Workspace + ?Sized,
    N: PrecisionModelType<W>,
{
    if !candidate.starts_with(net_type.root(ws)?) {
        return Ok(None);
    }
    Ok(net_entry_from_base(net_type, candidate))
}

fn net_entry_from_base<N>(net_type: N, base_path: &Path) -> Option<NetEntry<N>> {
    let (weights_path, params_path) = net_files_from_base(base_path);
    if weights_path.exists() && params_path.exists() {
        Some(NetEntry { net_type, weights_path })
    } else {
        None
    }
}

fn net_files_from_base(base_path: &Path) -> (PathBuf, PathBuf) {
    let file_name = base_path.file_name().and_then(|v| v.to_str());
    if file_name == Some(NET_WEIGHTS_FILE) {
        let base_dir = base_path.parent().unwrap_or(base_path);
        return (base_path.to_path_buf(), base_dir.join(NET_PARAMS_FILE));
    }
    if file_name == Some(NET_PARAMS_FILE) {
        let base_dir = base_path.parent().unwrap_or(base_path);
        return (base_dir.join(NET_WEIGHTS_FILE), base_path.to_path_buf());
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
        _ => (base_path.join(NET_WEIGHTS_FILE), base_path.join(NET_PARAMS_FILE)),
    }
}

fn convert_entry<N>(entry: NetEntry<N>, target_precision: ConvertPrecisionTarget) -> anyhow::Result<ConvertedNet<N>>
where
    N: Debug,
{
    debug!("convert_entry: entry={entry:?} target_precision={target_precision:?}");
    let mut vs = load_var_store(&entry.weights_path)?;
    let total_vars = vs.variables().len();
    let float_vars = vs.variables().values().filter(|tensor| is_float_like(tensor.kind())).count();

    target_precision.apply(&mut vs);

    for (name, tensor) in vs.variables() {
        if is_float_like(tensor.kind()) && tensor.kind() != target_precision.kind() {
            anyhow::bail!("ConvertPrecision: variable {name} in {:?} has kind {:?}, expected {:?}", entry.weights_path, tensor.kind(), target_precision.kind());
        }
    }

    replace_var_store(&vs, &entry.weights_path)?;
    Ok(ConvertedNet { net_type: entry.net_type, weights_path: entry.weights_path, float_vars, total_vars })
}

fn load_var_store(weights_path: &Path) -> anyhow::Result<VarStore> {
    let mut vs = VarStore::new(Device::Cpu);
    for (name, tensor) in Tensor::read_safetensors(weights_path)? {
        add_named_tensor_to_var_store(&vs, &name, tensor)?;
    }
    vs.load(weights_path)?;
    Ok(vs)
}

fn add_named_tensor_to_var_store(vs: &VarStore, name: &str, tensor: Tensor) -> anyhow::Result<()> {
    let parts = name.split('.').collect::<Vec<_>>();
    if parts.iter().any(|part| part.is_empty()) {
        anyhow::bail!("ConvertPrecision: bad tensor name {name:?}");
    }

    let mut path = vs.root();
    for part in &parts[..parts.len().saturating_sub(1)] {
        path = &path / *part;
    }
    let _ = path.add(parts[parts.len() - 1], tensor, true);
    Ok(())
}

fn replace_var_store(vs: &VarStore, weights_path: &Path) -> anyhow::Result<()> {
    let tmp_path = tmp_weights_path(weights_path)?;
    vs.save(&tmp_path)?;
    std::fs::remove_file(weights_path)?;
    std::fs::rename(&tmp_path, weights_path)?;
    Ok(())
}

fn tmp_weights_path(weights_path: &Path) -> anyhow::Result<PathBuf> {
    let parent = weights_path.parent().ok_or_else(|| anyhow::anyhow!("ConvertPrecision: weights path {weights_path:?} has no parent"))?;
    let stem = weights_path.file_stem().and_then(|v| v.to_str()).unwrap_or("weights");
    Ok(parent.join(format!("{stem}.convert_precision_{}.safetensors", std::process::id())))
}

fn is_float_like(kind: Kind) -> bool {
    matches!(kind, Kind::Half | Kind::BFloat16 | Kind::Float | Kind::Double)
}
