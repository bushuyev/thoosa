use clap::{Args, ValueEnum};
use log::{debug, info, warn};
use parking_lot::RwLock;
use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};
use thoosa_common::runner::common::cachable::DevicesWrap;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::RunCtx;
use thoosa_common::ws::{Workspace, NET_PARAMS_FILE, NET_WEIGHTS_FILE};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::traits::{Data, WithId};
use thoosa_shared::{collect_futures_with_progress, mono_key, recurse_files, StreamBuffering, ThoError};
use thoosa_shared::tracked_await::track_await;


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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvertPrecisionNetType {
    Motifs,
    Labels,
    Final,
    All,
}

impl ConvertPrecisionNetType {
    fn concrete(self) -> &'static [ConvertPrecisionNetType] {
        match self {
            Self::Motifs => &[Self::Motifs],
            Self::Labels => &[Self::Labels],
            Self::Final => &[Self::Final],
            Self::All => &[Self::Motifs, Self::Labels, Self::Final],
        }
    }

    fn root(self, ws: &dyn Workspace) -> &PathBuf {
        match self {
            Self::Motifs => ws.motifs_nets_path(),
            Self::Labels => ws.labels_nets_path(),
            Self::Final => ws.final_nets_path(),
            Self::All => ws.models_path(),
        }
    }

    fn parse(raw: &str) -> Result<Self, String> {
        match raw.trim().trim_matches(|ch| ch == '\'' || ch == '"').to_ascii_lowercase().as_str() {
            "motif" | "motifs" => Ok(Self::Motifs),
            "label" | "labels" => Ok(Self::Labels),
            "final" => Ok(Self::Final),
            "all" => Ok(Self::All),
            other => Err(format!("unexpected net type {other:?}; expected motifs, labels, final, or all")),
        }
    }
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ConvertPrecisionBy {
    Path(PathBuf),
    RunId(Vec<RunConfigId>),
    Type(ConvertPrecisionNetType),
}

impl FromStr for ConvertPrecisionBy {
    type Err = String;

    fn from_str(v: &str) -> Result<Self, Self::Err> {
        if let Some(rest) = v.strip_prefix("path:") {
            let path = rest.trim().trim_matches(|ch| ch == '\'' || ch == '"');
            if path.is_empty() {
                return Err("path payload is empty".to_string());
            }
            Ok(Self::Path(PathBuf::from(path)))
        } else if let Some(rest) = v.strip_prefix("run_config_id:").or_else(|| v.strip_prefix("run_config_ids:")) {
            let run_config_ids = thoosa_common::utils::parse_run_config_ids_vec(rest)?;
            if run_config_ids.is_empty() {
                return Err("run_config_id payload is empty".to_string());
            }
            Ok(Self::RunId(run_config_ids))
        } else if let Some(rest) = v.strip_prefix("type:").or_else(|| v.strip_prefix("net_type:")) {
            Ok(Self::Type(ConvertPrecisionNetType::parse(rest)?))
        } else {
            Err(format!("unexpected selector {v:?}; expected path:<path>, run_config_id:<id[,id]>, or type:motifs|labels|final|all"))
        }
    }
}

#[derive(Debug, Args, Clone)]
pub struct ConvertPrecisionArgs {
    #[arg(long)]
    nets_by: ConvertPrecisionBy,

    #[arg(long, value_enum)]
    target_precision: ConvertPrecisionTarget,
}

impl ConvertPrecisionArgs {
    pub fn new(nets_by: ConvertPrecisionBy, target_precision: ConvertPrecisionTarget) -> Self {
        Self { nets_by, target_precision }
    }

    pub fn nets_by(&self) -> ConvertPrecisionBy {
        self.nets_by.clone()
    }

    pub fn target_precision(&self) -> ConvertPrecisionTarget {
        self.target_precision
    }
}

#[derive(Debug, Clone, DummyData)]
pub struct ConvertPrecisionInput {
    id: SynKey,
}

impl ConvertPrecisionInput {
    pub fn new() -> Self {
        Self { id: SynKey::new([mono_key!("convert_precision", "convert_precision")]) }
    }
}

impl Input for ConvertPrecisionInput {}

impl WithId for ConvertPrecisionInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Clone)]
pub struct ConvertedNet {
    pub net_type: ConvertPrecisionNetType,
    pub weights_path: PathBuf,
    pub float_vars: usize,
    pub total_vars: usize,
}

#[derive(Debug, Clone)]
pub struct ConvertPrecisionOutput {
    id: SynKey,
    converted: Vec<ConvertedNet>,
}

impl ConvertPrecisionOutput {
    fn new(converted: Vec<ConvertedNet>) -> Self {
        Self { id: SynKey::new([mono_key!("convert_precision", "convert_precision_output")]), converted }
    }

    pub fn converted(&self) -> &[ConvertedNet] {
        &self.converted
    }
}

impl Data for ConvertPrecisionOutput {
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

impl WithId for ConvertPrecisionOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for ConvertPrecisionOutput {}

#[derive(Debug)]
pub struct ConvertPrecisionStage<C>
where
    C: RunCtx + 'static,
{
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    args: ConvertPrecisionArgs,
    pd: PhantomData<C>,
}

impl<C> ConvertPrecisionStage<C>
where
    C: RunCtx + 'static,
{
    pub fn new(ws: &Arc<RwLock<Box<dyn Workspace>>>, args: ConvertPrecisionArgs) -> Self {
        Self {
            ws: Arc::clone(ws),
            args,
            pd: PhantomData,
        }
    }
}

impl<C> Stage for ConvertPrecisionStage<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = ConvertPrecisionInput;
    type Out = ConvertPrecisionOutput;
}

impl<C> AsyncStage for ConvertPrecisionStage<C>
where
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "ConvertPrecisionStage::run"))]
    async fn run(self: Arc<Self>, _input: Arc<DevicesWrap<Self::In<'static>>>, _ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let entries = {
            let ws_guard = self.ws.read();
            selected_entries(ws_guard.as_ref(), self.args.nets_by())?
        };
        let entries_len = entries.len();
        if entries_len == 0 {
            warn!("ConvertPrecisionStage::run: no nets matched {:?}", self.args.nets_by());
            return Ok(Arc::new(DevicesWrap::wrap(ConvertPrecisionOutput::new(vec![]))));
        }

        let target_precision = self.args.target_precision();
        let futures = entries.into_iter().map(move |entry| {
            async move {
                track_await(tokio::task::spawn_blocking(move || {
                    tracing::info_span!(
                        "convert_precision::convert_entry",
                        kind = "spawn_blocking",
                        file = file!(),
                        line = line!(),
                        task = format!("convert net entry to {target_precision:?}"),
                    ).in_scope(|| convert_entry(entry, target_precision))
                })).await
                    .map_err(|e| ThoError::from_error("join convert precision worker", e))?
            }
        });

        let converted = track_await(collect_futures_with_progress(
            futures,
            entries_len,
            StreamBuffering::Unordered,
            "ConvertPrecisionStage::run",
        )).await;

        if converted.len() != entries_len {
            return Err(ThoError::new(format!("ConvertPrecisionStage::run: converted {} of {entries_len} nets", converted.len())));
        }

        info!("ConvertPrecisionStage::run: converted {} nets to {:?}", converted.len(), target_precision);
        Ok(Arc::new(DevicesWrap::wrap(ConvertPrecisionOutput::new(converted))))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct NetEntry {
    net_type: ConvertPrecisionNetType,
    weights_path: PathBuf,
}

fn selected_entries(ws: &dyn Workspace, selector: ConvertPrecisionBy) -> Result<Vec<NetEntry>, ThoError> {
    match selector {
        ConvertPrecisionBy::Path(path) => entries_for_path(ws, &path),
        ConvertPrecisionBy::RunId(run_config_ids) => {
            let run_config_ids = run_config_ids.into_iter().collect::<HashSet<_>>();
            entries_for_type(ws, ConvertPrecisionNetType::All)?
                .into_iter()
                .filter(|entry| entry_run_config_id(entry).is_some_and(|run_config_id| run_config_ids.contains(&run_config_id)))
                .collect::<Vec<_>>()
                .dedup_entries()
        }
        ConvertPrecisionBy::Type(net_type) => entries_for_type(ws, net_type),
    }
}

trait DedupEntries {
    fn dedup_entries(self) -> Result<Vec<NetEntry>, ThoError>;
}

impl DedupEntries for Vec<NetEntry> {
    fn dedup_entries(self) -> Result<Vec<NetEntry>, ThoError> {
        let mut seen = HashSet::new();
        Ok(self.into_iter().filter(|entry| seen.insert(entry.weights_path.clone())).collect())
    }
}

fn entries_for_path(ws: &dyn Workspace, path: &Path) -> Result<Vec<NetEntry>, ThoError> {
    let candidates = if path.is_absolute() {
        vec![path.to_path_buf()]
    } else {
        vec![
            ws.models_path().join(path),
            ws.motifs_nets_path().join(path),
            ws.labels_nets_path().join(path),
            ws.final_nets_path().join(path),
        ]
    };

    let entries = candidates
        .into_iter()
        .flat_map(|candidate| {
            ConvertPrecisionNetType::All
                .concrete()
                .iter()
                .filter_map(move |net_type| entry_from_candidate(ws, *net_type, &candidate))
        })
        .collect::<Vec<_>>()
        .dedup_entries()?;

    if entries.is_empty() {
        return Err(ThoError::new(format!("ConvertPrecision: no net found for path {path:?}")));
    }
    if entries.len() > 1 {
        return Err(ThoError::new(format!("ConvertPrecision: path {path:?} is ambiguous and matched {:?}", entries.iter().map(|entry| &entry.weights_path).collect::<Vec<_>>())));
    }
    Ok(entries)
}

fn entries_for_type(ws: &dyn Workspace, net_type: ConvertPrecisionNetType) -> Result<Vec<NetEntry>, ThoError> {
    let mut entries = Vec::new();
    for concrete_type in net_type.concrete() {
        let root = concrete_type.root(ws).clone();
        entries.extend(
            recurse_files(&root, &|path| path.extension().and_then(|ext| ext.to_str()) == Some("safetensors"))?
                .into_iter()
                .filter_map(|weights_path| net_entry_from_base(*concrete_type, &weights_path)),
        );
    }
    entries.dedup_entries()
}

fn entry_from_candidate(ws: &dyn Workspace, net_type: ConvertPrecisionNetType, candidate: &Path) -> Option<NetEntry> {
    if !candidate.starts_with(net_type.root(ws)) {
        return None;
    }
    net_entry_from_base(net_type, candidate)
}

fn net_entry_from_base(net_type: ConvertPrecisionNetType, base_path: &Path) -> Option<NetEntry> {
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

fn entry_run_config_id(entry: &NetEntry) -> Option<RunConfigId> {
    let base_path = if entry.weights_path.file_name().and_then(|v| v.to_str()) == Some(NET_WEIGHTS_FILE) {
        entry.weights_path.parent()?
    } else {
        entry.weights_path.as_path()
    };

    match entry.net_type {
        ConvertPrecisionNetType::Motifs => base_path
            .file_stem()
            .and_then(|v| v.to_str())
            .and_then(|v| v.rsplit_once('_').map(|(_, run_config_id)| run_config_id).or(Some(v)))
            .and_then(|v| v.parse::<RunConfigId>().ok()),
        ConvertPrecisionNetType::Labels | ConvertPrecisionNetType::Final => base_path
            .file_stem()
            .and_then(|v| v.to_str())
            .and_then(|v| v.parse::<RunConfigId>().ok()),
        ConvertPrecisionNetType::All => None,
    }
}

fn convert_entry(entry: NetEntry, target_precision: ConvertPrecisionTarget) -> Result<ConvertedNet, ThoError> {
    debug!("convert_entry: entry={entry:?} target_precision={target_precision:?}");
    let mut vs = load_var_store(&entry)?;
    let total_vars = vs.variables().len();
    let float_vars = vs.variables().values().filter(|tensor| is_float_like(tensor.kind())).count();

    target_precision.apply(&mut vs);

    for (name, tensor) in vs.variables() {
        if is_float_like(tensor.kind()) && tensor.kind() != target_precision.kind() {
            return Err(ThoError::new(format!("ConvertPrecision: variable {name} in {:?} has kind {:?}, expected {:?}", entry.weights_path, tensor.kind(), target_precision.kind())));
        }
    }

    replace_var_store(&vs, &entry.weights_path)?;
    Ok(ConvertedNet { net_type: entry.net_type, weights_path: entry.weights_path, float_vars, total_vars })
}

fn load_var_store(entry: &NetEntry) -> Result<VarStore, ThoError> {
    let mut vs = VarStore::new(Device::Cpu);
    for (name, tensor) in Tensor::read_safetensors(&entry.weights_path)? {
        add_named_tensor_to_var_store(&vs, &name, tensor)?;
    }
    vs.load(&entry.weights_path)?;
    Ok(vs)
}

fn add_named_tensor_to_var_store(vs: &VarStore, name: &str, tensor: Tensor) -> Result<(), ThoError> {
    let parts = name.split('.').collect::<Vec<_>>();
    if parts.iter().any(|part| part.is_empty()) {
        return Err(ThoError::new(format!("ConvertPrecision: bad tensor name {name:?}")));
    }

    let mut path = vs.root();
    for part in &parts[..parts.len().saturating_sub(1)] {
        path = &path / *part;
    }
    let _ = path.add(parts[parts.len() - 1], tensor, true);
    Ok(())
}

fn replace_var_store(vs: &VarStore, weights_path: &Path) -> Result<(), ThoError> {
    let tmp_path = tmp_weights_path(weights_path)?;
    vs.save(&tmp_path)?;
    std::fs::remove_file(weights_path)?;
    std::fs::rename(&tmp_path, weights_path)?;
    Ok(())
}

fn tmp_weights_path(weights_path: &Path) -> Result<PathBuf, ThoError> {
    let parent = weights_path.parent().ok_or_else(|| ThoError::new(format!("ConvertPrecision: weights path {weights_path:?} has no parent")))?;
    let stem = weights_path.file_stem().and_then(|v| v.to_str()).unwrap_or("weights");
    Ok(parent.join(format!("{stem}.convert_precision_{}.safetensors", std::process::id())))
}

fn is_float_like(kind: Kind) -> bool {
    matches!(kind, Kind::Half | Kind::BFloat16 | Kind::Float | Kind::Double)
}

#[cfg(test)]
mod tests {
    use super::{ConvertPrecisionArgs, ConvertPrecisionBy, ConvertPrecisionInput, ConvertPrecisionNetType, ConvertPrecisionStage, ConvertPrecisionTarget};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use parking_lot::RwLock;
    use tch::nn::VarStore;
    use tch::{Device, Kind, Tensor};
    use thoosa_common::dims::AnyDims;
    use thoosa_common::my_tch::ThoModule;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::train_nets::{final_net, lin_label_node, FinalNetParams, LabelNetParams, LabelNetTypeParams, LinLenNetParams};
    use thoosa_common::ws::{FsWorkspace, Workspace};
    use thoosa_shared::{cdt_for_dir, ThoError};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_convert_precision_stage_converts_saved_final_net() -> Result<(), ThoError> {
        run_convert_precision_stage_final_net(ConvertPrecisionTarget::Fp64, Kind::Double).await?;
        run_convert_precision_stage_final_net(ConvertPrecisionTarget::Bf16, Kind::BFloat16).await?;
        run_convert_precision_stage_label_net(ConvertPrecisionTarget::Bf16, Kind::BFloat16).await?;

        Ok(())
    }

    async fn run_convert_precision_stage_final_net(precision_target: ConvertPrecisionTarget, kind: Kind) -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(format!("convert_precision_{}", cdt_for_dir()));
        let ws_box = FsWorkspace::new(test_dir, 2, 100, 100)?;

        let params = FinalNetParams::new(8, 2, 1);
        let mut vs = VarStore::new(Device::Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = final_net(&params)?.build_tch_module(&vs.root())?;
        vs.freeze();
        let net_id = ws_box.save_final_net(3, &vs, &params)?;

        let ws = Arc::new(RwLock::new(ws_box));
        let args = ConvertPrecisionArgs::new(ConvertPrecisionBy::Path(PathBuf::from("final").join(&net_id)), precision_target);
        let stage = Arc::new(ConvertPrecisionStage::<RunCtxImpl>::new(&ws, args));
        let output = stage
            .run(
                Arc::new(DevicesWrap::wrap(ConvertPrecisionInput::new())),
                RunCtxImpl::new(0, "convert_precision_test"),
            )
            .await?;
        let output = Arc::into_inner(output).expect("stage output should not be shared").take_data(0).await?;
        assert_eq!(output.converted().len(), 1);

        let (_, loaded_module) = ws.read().load_final_net(&net_id, &Device::Cpu, false)?;
        let variables = loaded_module.var_store().variables();
        assert!(!variables.is_empty());
        for tensor in variables.values() {
            assert_eq!(tensor.kind(), kind);
        }
        Ok(())
    }

    async fn run_convert_precision_stage_label_net(precision_target: ConvertPrecisionTarget, kind: Kind) -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(format!("convert_precision_{}", cdt_for_dir()));
        let ws_box = FsWorkspace::new(test_dir, 2, 100, 100)?;

        let params = LabelNetParams::new(2, LabelNetTypeParams::LinNetParams(LinLenNetParams{ motif_layers: 2, features: 10 }));
        let mut vs = VarStore::new(Device::Cpu);
        let _: Box<dyn ThoModule<AnyDims, AnyDims>> = lin_label_node(&params)?.build_tch_module(&vs.root())?;
        vs.freeze();
        let net_id = ws_box.save_label_net(3, &"label_a", &vs, &params)?;

        let ws = Arc::new(RwLock::new(ws_box));
        let args = ConvertPrecisionArgs::new(ConvertPrecisionBy::Path(PathBuf::from(&net_id)), precision_target);
        let stage = Arc::new(ConvertPrecisionStage::<RunCtxImpl>::new(&ws, args));
        let output = stage
            .run(
                Arc::new(DevicesWrap::wrap(ConvertPrecisionInput::new())),
                RunCtxImpl::new(0, "convert_precision_test"),
            )
            .await?;
        let output = Arc::into_inner(output).expect("stage output should not be shared").take_data(0).await?;
        assert_eq!(output.converted().len(), 1);

        let (_, params, loaded_module) = ws.read().load_label_net(&net_id, &Device::Cpu, false)?;
        let variables = loaded_module.var_store().variables();
        assert!(!variables.is_empty());
        for tensor in variables.values() {
            assert_eq!(tensor.kind(), kind);
        }
        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_convert_precision_stage_preserves_existing_tensor_names() -> Result<(), ThoError> {
        let test_dir = Path::new("/tmp").join(format!("convert_precision_{}", cdt_for_dir()));
        let ws_box = FsWorkspace::new(test_dir, 2, 100, 100)?;
        let net_dir = ws_box.motifs_nets_path().join("0").join("34603_36472");
        std::fs::create_dir_all(&net_dir)?;
        std::fs::write(net_dir.join("parameters.json"), "{}")?;
        let weights_path = net_dir.join("weights.safetensors");
        let weight = Tensor::randn([1, 1, 4], (Kind::Float, Device::Cpu));
        let bias = Tensor::zeros([1], (Kind::Float, Device::Cpu));
        Tensor::write_safetensors(
            &[
                ("seq_4.motif_0_conv.weight", &weight),
                ("seq_4.motif_0_conv.bias", &bias),
            ],
            &weights_path,
        )?;

        let ws = Arc::new(RwLock::new(ws_box));
        let args = ConvertPrecisionArgs::new(ConvertPrecisionBy::Type(ConvertPrecisionNetType::Motifs), ConvertPrecisionTarget::Fp16);
        let stage = Arc::new(ConvertPrecisionStage::<RunCtxImpl>::new(&ws, args));
        let output = stage
            .run(
                Arc::new(DevicesWrap::wrap(ConvertPrecisionInput::new())),
                RunCtxImpl::new(0, "convert_precision_names_test"),
            )
            .await?;
        let output = Arc::into_inner(output).expect("stage output should not be shared").take_data(0).await?;
        assert_eq!(output.converted().len(), 1);

        let converted = Tensor::read_safetensors(&weights_path)?;
        assert!(converted.iter().any(|(name, _)| name == "seq_4.motif_0_conv.weight"));
        for (_, tensor) in converted {
            assert_eq!(tensor.kind(), Kind::Half);
        }

        Ok(())
    }
}
