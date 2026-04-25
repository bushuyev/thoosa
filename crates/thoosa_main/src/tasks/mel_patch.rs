use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs;
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use log::{debug, info, warn};
use tch::Tensor;
use serde::Deserialize;
use thoosa_common::audio::{assign_lsh_buckets_per_channel, audio_to_log_mel_patches, patch_sample_ranges_flat, LogMelPatchConfig};
use thoosa_common::audio::{LogMelPatchConfigRangeOrValue, LogMelPatchConfigFields, LogMelPatchConfigBuilder};
use thoosa_common::dims::Dims;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runs::SavableResult;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{TensorIdSliceIdx, ThoError, LOCK_TIMEOUT};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Data, Named, WithId};
use thoosa_store::{IdSlice, Store};
use thoosa_with_data::with_data;
use thoosa_named_derive::Named;


const MEL_PATCH_TEMP_MAGIC: [u8; 8] = *b"THOMELP1";

#[derive(Debug, Clone)]
pub struct MelPatchResourcesInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
    resources_config: RC,
}

impl<RC> MelPatchResourcesInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
    pub fn new(resources_config: RC) -> Self {
        Self { resources_config }
    }

    pub async fn check(self, store: &Arc<Box<dyn Store>>) -> Result<Self, ThoError> {
        self.resources_config.check(store).await?;
        Ok(self)
    }

    pub fn resources_config(&self) -> &RC {
        &self.resources_config
    }
}

impl<RC> WithId for MelPatchResourcesInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl<RC> Data for MelPatchResourcesInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
    type Metrics = ();
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        std::mem::size_of_val(self) as u64
    }
}

impl<RC> Input for MelPatchResourcesInput<RC> where RC: ResourcesConfig + Debug + Send + Sync + Clone {}

#[derive(Debug, Clone)]
pub struct MelPatchOutput {
    groups_by_run: Vec<(RunId, Vec<usize>)>,
}

impl MelPatchOutput {
    fn new(groups_by_run: Vec<(RunId, Vec<usize>)>) -> Self {
        Self { groups_by_run }
    }

    pub fn groups_by_run(&self) -> &[(RunId, Vec<usize>)] {
        &self.groups_by_run
    }
}

impl WithId for MelPatchOutput {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Data for MelPatchOutput {
    type Metrics = ();
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        std::mem::size_of_val(self) as u64
    }
}

impl Output for MelPatchOutput {}

#[async_trait]
impl SavableResult for MelPatchOutput {
    async fn save(
        &mut self,
        _store: &Arc<Box<dyn Store>>,
        _ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoError> {
        Ok(0)
    }
}

#[derive(Debug)]
pub struct MelPatchTempStore {
    path: PathBuf,
    writer: BufWriter<File>,
}

impl MelPatchTempStore {
    fn new(path: &Path, run_id: RunId, channel: u32) -> Result<Self, ThoError> {
        fs::create_dir_all(path)?;
        let file_path = path.join(format!("{run_id}_{channel}_mel_patches.bin"));
        let file = OpenOptions::new().create(true).truncate(true).read(true).write(true).open(&file_path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&MEL_PATCH_TEMP_MAGIC)?;
        writer.flush()?;
        debug!("MelPatchTempStore::new: file_path={file_path:?}");

        Ok(Self {
            path: file_path,
            writer,
        })
    }

    fn save_slices(&mut self, group: u16, slices: &[TensorIdSliceIdx]) -> Result<(), ThoError> {
        debug!("MelPatchTempStore::save_slices: path={:?} group={group} slices.len={}", self.path, slices.len());
        for slice in slices {
            self.writer.write_all(&group.to_le_bytes())?;
            self.writer.write_all(&slice.0.to_le_bytes())?;
            self.writer.write_all(&slice.1.to_le_bytes())?;
        }
        self.writer.flush()?;
        Ok(())
    }

    fn load_groups(&mut self) -> Result<HashMap<u16, HashSet<IdSlice>>, ThoError> {
        self.writer.flush()?;
        let mut reader = BufReader::new(File::open(&self.path)?);
        let mut magic = [0u8; MEL_PATCH_TEMP_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if magic != MEL_PATCH_TEMP_MAGIC {
            return Err(ThoError::new(format!("MelPatchTempStore::load_groups: unexpected file header in {:?}", self.path)));
        }

        let mut groups: HashMap<u16, HashSet<IdSlice>> = HashMap::new();
        let mut slice_id = 0u32;
        loop {
            let mut group_bytes = [0u8; 2];
            match reader.read(&mut group_bytes[0..1])? {
                0 => break,
                1 => reader.read_exact(&mut group_bytes[1..2])?,
                n => return Err(ThoError::new(format!("MelPatchTempStore::load_groups: unexpected read size {n}"))),
            }

            let mut resource_id_bytes = [0u8; 4];
            let mut slice_idx_bytes = [0u8; 4];
            reader.read_exact(&mut resource_id_bytes)?;
            reader.read_exact(&mut slice_idx_bytes)?;

            slice_id = slice_id.checked_add(1).ok_or_else(|| ThoError::new("MelPatchTempStore::load_groups: slice_id overflow"))?;
            groups.entry(u16::from_le_bytes(group_bytes)).or_default().insert(IdSlice(
                slice_id,
                u32::from_le_bytes(resource_id_bytes),
                u32::from_le_bytes(slice_idx_bytes),
            ));
        }

        Ok(groups)
    }

    fn remove(self) -> Result<(), ThoError> {
        fs::remove_file(&self.path)?;
        Ok(())
    }
}

#[derive(Debug, Named)]
pub struct MelPatchStage<RC, L, D, Id, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<D, Id>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    D: Dims + 'static,
    Id: Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + 'static,
{
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    run_configs: Vec<MelPatchRunConfig>,
    pd: PhantomData<(RC, D, Id, C)>,
}

impl<RC, L, D, Id, C> MelPatchStage<RC, L, D, Id, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<D, Id>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    D: Dims + 'static,
    Id: Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + 'static,
{
    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<L>,
        run_configs: Vec<MelPatchRunConfig>,
    ) -> Result<Self, ThoError> {
        if run_configs.is_empty() {
            return Err(ThoError::new("MelPatchStage::new: run_configs should not be empty"));
        }

        for run_config in &run_configs {
            run_config.run_id()?;
            if run_config.bucket_count == 0 {
                return Err(ThoError::new("MelPatchStage::new: bucket_count must be positive"));
            }
            if run_config.bucket_count > u16::MAX as u32 + 1 {
                return Err(ThoError::new(format!("MelPatchStage::new: bucket_count={} exceeds u16 group ord capacity", run_config.bucket_count)));
            }
            if run_config.min_slices > run_config.max_slices {
                return Err(ThoError::new(format!("MelPatchStage::new: min_slices={} exceeds max_slices={}", run_config.min_slices, run_config.max_slices)));
            }
        }

        Ok(Self {
            ws: Arc::clone(ws),
            store: Arc::clone(store),
            loader,
            run_configs,
            pd: PhantomData,
        })
    }

    fn save_resource_patches(
        &self,
        resource: &Resource,
        audio: &Tensor,
        tmp_path: &Path,
        temp_stores: &mut Vec<MelPatchTempStore>,
        run_config: &MelPatchRunConfig,
    ) -> Result<(), ThoError> {
        let mel_config = run_config.mel_config();
        let patches = audio_to_log_mel_patches(audio, mel_config)?;
        let patch_ranges = patch_sample_ranges_flat(&patches, mel_config)?;
        let bucket_ids = assign_lsh_buckets_per_channel(&patches, run_config.bucket_count_i64())?;

        if patch_ranges.len() != bucket_ids.len() {
            return Err(ThoError::new(format!("MelPatchStage::save_resource_patches: ranges channels={} bucket channels={}", patch_ranges.len(), bucket_ids.len())));
        }


        for channel in 0..bucket_ids.len() {
            if patch_ranges[channel].len() != bucket_ids[channel].len() {
                return Err(ThoError::new(format!("MelPatchStage::save_resource_patches: channel={channel} ranges len={} bucket len={}", patch_ranges[channel].len(), bucket_ids[channel].len())));
            }

            let mut slices_by_bucket: HashMap<u16, Vec<TensorIdSliceIdx>> = HashMap::new();
            for ((sample_start, _sample_end), bucket) in patch_ranges[channel].iter().zip(bucket_ids[channel].iter()) {
                let group = u16::try_from(*bucket).map_err(|_| ThoError::new(format!("MelPatchStage::save_resource_patches: bucket={bucket} does not fit into u16")))?;
                let slice_start = u32::try_from(*sample_start).map_err(|_| ThoError::new(format!("MelPatchStage::save_resource_patches: sample_start={sample_start} does not fit into u32")))?;
                slices_by_bucket.entry(group).or_default().push(TensorIdSliceIdx(resource.id(), slice_start));
            }

            for (group, slices) in slices_by_bucket {
                debug!("save_resource_patches: group={group} slices={slices:?}");
                temp_stores[channel].save_slices(group, &slices)?;
            }
        }

        Ok(())
    }
}

impl<RC, L, D, Id, C> Stage for MelPatchStage<RC, L, D, Id, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<D, Id>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    D: Dims + 'static,
    Id: Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = MelPatchResourcesInput<RC>
    where
        Self: 'a;
    type Out = MelPatchOutput;
}

impl<RC, L, D, Id, C> AsyncStage for MelPatchStage<RC, L, D, Id, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<D, Id>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    D: Dims + Send + Sync + 'static,
    Id: Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "MelPatchStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let resources_config = with_data!(ctx.branch(Some("mel_patch_resources"), false), input; |_ctx, input: &MelPatchResourcesInput<RC>| {
            Ok::<_, ThoError>(input.resources_config().clone())
        })?;
        let resources = resources_config.to_resources(&self.store).await?;

        debug!("MelPatchStage::run: resources_config={resources_config:?} resources={resources:?}");

        let tmp_path = thoosa_shared::with_perf_log(
            || self.ws.try_read_for(LOCK_TIMEOUT).expect("MelPatchStage::run: timed out waiting for workspace read lock").tmp_path().clone(),
            "MelPatchStage::run/read_ws_tmp_path",
        );
        let channels = self.ws.try_read_for(LOCK_TIMEOUT).expect("MelPatchStage::run: timed out waiting for workspace read lock").channels();

        let mut temp_stores_by_run: Vec<Vec<MelPatchTempStore>> = self.run_configs.iter()
            .map(|run_config|
                (0..channels).map(|channel|MelPatchTempStore::new(&tmp_path, run_config.id.expect("no run id").clone(), channel))
                    .collect::<Result<Vec<_>, ThoError>>()
            ).collect::<Result<Vec<_>, ThoError>>()?;

        for resource in &resources {
            let loaded = Arc::clone(&self.loader)
                .run_cached(
                    Arc::new(DevicesWrap::wrap(resource.clone())),
                    ctx.branch(Some(&format!("mel_patch_load_resource_{}", resource.id())), false),
                )
                .await?;

            with_data!(ctx.branch(Some(&format!("mel_patch_bucket_resource_{}", resource.id())), false), loaded; |_ctx, tensor: &ThoTensor<D, Id>| {
                for (run_config, temp_stores) in self.run_configs.iter().zip(temp_stores_by_run.iter_mut()) {
                    self.save_resource_patches(resource, tensor.t_ref(), &tmp_path, temp_stores, run_config)?;
                }
                Ok::<_, ThoError>(())
            })?;
        }

        let mut groups_by_run = Vec::with_capacity(self.run_configs.len());
        for (run_config, temp_stores) in self.run_configs.iter().zip(temp_stores_by_run.into_iter()) {
            let run_id = run_config.run_id()?;
            let mut groups_by_channel = Vec::with_capacity(temp_stores.len());
            for (channel, mut temp_store) in temp_stores.into_iter().enumerate() {
                let groups = filter_mel_patch_groups(temp_store.load_groups()?, run_config.min_slices_usize(), run_config.max_slices_usize());
                let groups_len = groups.len();
                self.store.save_groups_as_motifs(run_id, channel as i64, groups).await?;
                temp_store.remove()?;
                groups_by_channel.push(groups_len);
            }

            info!("MelPatchStage::run: run_id={run_id} resources={} groups_by_channel={groups_by_channel:?}", resources.len());
            groups_by_run.push((run_id, groups_by_channel));
        }

        Ok(Arc::new(DevicesWrap::wrap(MelPatchOutput::new(groups_by_run))))
    }
}

fn ensure_mel_patch_temp_stores(
    temp_stores: &mut Vec<MelPatchTempStore>,
    tmp_path: &Path,
    run_id: RunId,
    channels: usize,
) -> Result<(), ThoError> {
    while temp_stores.len() < channels {
        let channel = u32::try_from(temp_stores.len()).map_err(|_| ThoError::new("ensure_mel_patch_temp_stores: channel index does not fit into u32"))?;
        temp_stores.push(MelPatchTempStore::new(tmp_path, run_id, channel)?);
    }
    Ok(())
}

pub fn filter_mel_patch_groups(groups: HashMap<u16, HashSet<IdSlice>>, min_slices: usize, max_slices: usize) -> HashMap<u16, HashSet<IdSlice>> {
    groups
        .into_iter()
        .filter_map(|(group, slices)| {
            let len = slices.len();
            if (min_slices..=max_slices).contains(&len) {
                Some((group, slices))
            } else {
                warn!("MelPatchStage: skipping group={group} with {len} slices");
                None
            }
        })
        .collect()
}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_grid]
pub struct MelPatchRunConfig {
    #[in_map]
    pub id: Option<RunId>,

    #[run_config]
    #[in_map]
    pub bucket_count: u32,

    #[run_config]
    #[in_map]
    pub min_slices: u32,

    #[run_config]
    #[in_map]
    pub max_slices: u32,

    #[run_config]
    #[in_map]
    mel_config: LogMelPatchConfig
}

impl MelPatchRunConfig {

    pub fn run_id(&self) -> Result<RunId, ThoError> {
        self.id.ok_or_else(|| ThoError::new("MelPatchRunConfig::run_id: missing run id"))
    }

    pub fn bucket_count_i64(&self) -> i64 {
        i64::from(self.bucket_count)
    }

    pub fn min_slices_usize(&self) -> usize {
        self.min_slices as usize
    }

    pub fn max_slices_usize(&self) -> usize {
        self.max_slices as usize
    }

    pub fn mel_config(&self) -> &LogMelPatchConfig {
        &self.mel_config
    }

    pub fn new(id: Option<RunId>, bucket_count: u32, min_slices: u32, max_slices: u32, mel_config:LogMelPatchConfig) -> Self {
        Self { id, bucket_count, min_slices, max_slices, mel_config }
    }
}

impl Default for MelPatchRunConfig {
    fn default() -> Self {
        Self {
            id: None,
            bucket_count: 256,
            min_slices: 10,
            max_slices: 100,
            mel_config: Default::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use futures_util::StreamExt;
    use log::debug;
    use parking_lot::RwLock;
    use rand::random_range;
    use tch::{Device, Tensor};
    use tch::Device::Cuda;
    use thoosa_common::audio::{LogMelPatchConfig, LogMelPatchConfigRangeOrValue};
    use thoosa_common::dims::MotifSlices;
    use thoosa_common::find_motifs_run_config::{AccumulateStats, Aggr, EucDistConfig, FindMotifsRunConfig, MaxAggr, Metrics, NormThicknessRunConfig, SimilarityMetric, SlicesFilter};
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::RunCtx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::runs::Simple;
    use thoosa_common::tch_test_utils;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::ws::FsWorkspace;
    use thoosa_motifs::distance::clc_d2_similarity;
    use crate::cli::plot_resource;
    use thoosa_motifs::fragments::Resolution;
    use thoosa_properties_map::ToHashmap;
    use super::{filter_mel_patch_groups, MelPatchTempStore, MelPatchRunConfig, MelPatchRunConfigRangeOrValue, MelPatchStage, MelPatchResourcesInput};
    use thoosa_run_config_args::GridRunConfigSource;
    use thoosa_shared::{cdt_for_dir, mono_key, ResourceType, TensorIdSliceIdx, ThoError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::{Resource, ResourcesConfig};
    use thoosa_shared::run_configs::RunConfig;
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use thoosa_store::{IdSlice, SqlStore};
    use thoosa_store::rusqlite::Transaction;
    use thoosa_with_data::with_data;
    use crate::cli::{import_resources, plot_motif};
    use crate::RawTensorLoader;
    use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
    use crate::tasks::motif_stats::{MotifStats, MotifStatsInput};
    use crate::tasks::slices_stats::euclidean_distance;

    fn temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time before UNIX_EPOCH")
            .as_nanos();
        std::env::temp_dir().join(format!("{name}_{}_{}", std::process::id(), nanos))
    }

    fn slice(resource_id: u32, slice_idx: u32) -> TensorIdSliceIdx {
        TensorIdSliceIdx(resource_id, slice_idx)
    }

    #[test]
    fn mel_patch_temp_store_loads_saved_slices_as_groups() -> Result<(), ThoError> {
        let dir = temp_dir("mel_patch_temp_store_loads_saved_slices_as_groups");
        let mut store = MelPatchTempStore::new(&dir, 7, 1)?;

        store.save_slices(3, &[slice(10, 100), slice(11, 110)])?;
        store.save_slices(5, &[slice(12, 120)])?;
        store.save_slices(3, &[slice(13, 130)])?;

        let groups = store.load_groups()?;
        assert_eq!(groups.len(), 2);
        assert_eq!(groups.get(&3).expect("missing group 3").len(), 3);
        assert_eq!(groups.get(&5).expect("missing group 5").len(), 1);

        store.remove()?;
        fs::remove_dir_all(&dir)?;
        Ok(())
    }

    #[test]
    fn filter_mel_patch_groups_keeps_only_groups_in_configured_slice_range() {
        let groups = [
            (1u16, 9usize),
            (2u16, 10usize),
            (3u16, 100usize),
            (4u16, 101usize),
        ]
            .into_iter()
            .map(|(group, len)| {
                let slices = (0..len)
                    .map(|idx| IdSlice(idx as u32, group as u32, idx as u32))
                    .collect::<HashSet<_>>();
                (group, slices)
            })
            .collect::<HashMap<_, _>>();

        let filtered = filter_mel_patch_groups(groups, 10, 100);
        assert_eq!(filtered.keys().copied().collect::<HashSet<_>>(), [2u16, 3u16].into_iter().collect::<HashSet<_>>());
    }

    #[test]
    fn mel_patch_run_config_grid_uses_slice_and_bucket_parameters() {
        let default_log_mel_patch_config = LogMelPatchConfig::default();
        let rc = MelPatchRunConfigRangeOrValue {
            id: None,
            bucket_count: 256u32.into(),
            min_slices: 10u32.into(),
            max_slices: 100u32.into(),
            mel_config: LogMelPatchConfigRangeOrValue {
                sample_rate: default_log_mel_patch_config.sample_rate.into(),
                n_fft: default_log_mel_patch_config.n_fft.into(),
                hop_length: default_log_mel_patch_config.hop_length.into(),
                n_mels: default_log_mel_patch_config.n_mels.into(),
                f_min: default_log_mel_patch_config.f_min.into(),
                f_max: default_log_mel_patch_config.f_max.into(),
                patch_frames: default_log_mel_patch_config.patch_frames.into(),
                patch_step: default_log_mel_patch_config.patch_step.into(),
                log_eps: default_log_mel_patch_config.log_eps.into(),
            },
        };

        let grid = rc.grid_run_configs().expect("MelPatchRunConfigRangeOrValue::grid_run_configs should not fail");
        let run_configs: Vec<MelPatchRunConfig> = grid.to_run_configs();

        assert_eq!(1, run_configs.len());
        assert_eq!(256, run_configs[0].bucket_count);
        assert_eq!(10, run_configs[0].min_slices);
        assert_eq!(100, run_configs[0].max_slices);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_run_mel_path() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        unsafe { mock_global_ctx::USE_REAL_CTX = true; }
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        debug!("test_dir={test_dir:?}");
        // let store = Arc::new(SqlStore::in_mem().await?);
        let store = Arc::new(SqlStore::from_path(test_dir.join(&"main_db.db3")).await?);

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            3,
            u32::MAX,
            32000,
        )?));



        let imported_ids = import_resources(
            &ws,
            &store,
            &["label_0", "label_1"],
            || {
                Ok(HashMap::from_iter([
                    (0..10).map(|i| (format!("t_0_{i}"), vec![("label_0".to_string(), 1.0)])).collect::<Vec<_>>(),
                    (0..10).map(|i| (format!("t_1_{i}"), vec![("label_1".to_string(), 1.0)])).collect::<Vec<_>>()
                ].concat().into_iter()))
            },
            |key: String| {
                debug!("key={key}");
                let num_motifs = random_range(5usize..9usize);
                let motif_len = 64usize;
                let motif_gap = 64usize;
                let resource_len = num_motifs * motif_len + (num_motifs - 1) * motif_gap;

                if key.starts_with("t_0") {

                    let (resource_t, _resource_intervals) = tch_test_utils::resource_tensor(resource_len, &[
                        vec![(num_motifs, motif_len, MotifKind::ChirpUp)],
                        vec![(num_motifs, motif_len, MotifKind::ChirpDown)],
                        vec![(num_motifs, motif_len, MotifKind::HarmonicStack)]
                    ], motif_gap, 10.0, false, true)?;//TODO set motif_from_zero to false and z-normalize

                    Ok(resource_t)
                } else if key.starts_with("t_1") {

                    let (resource_t, _resource_intervals) = tch_test_utils::resource_tensor(resource_len, &[
                        vec![(num_motifs, motif_len, MotifKind::ChirpDown)],
                        vec![(num_motifs, motif_len, MotifKind::ChirpUp)],
                        vec![(num_motifs, motif_len, MotifKind::Trill)]
                    ], motif_gap, 10.0, false, true)?;

                    Ok(resource_t)
                } else {
                    //TODO random tensor
                    todo!()
                }

            },
            None,
            true,
            |t: &Transaction|{
                SqlStore::save_all_sets(&t)?;
                Ok(())
            },
        ).await?;

        let tensors_path = ws.read().tensors_path().clone();

        let loader = Arc::new(RawTensorLoader::<RunCtxImpl>::new(
            move |key, _| {
                let tensor = Tensor::load(tensors_path.join(format!("1_{}.pt", key.file_stem())))?;//1_ for import id
                debug!("test_motifs: key={key:?} loaded tensor={tensor:?}");
                Ok(ThoTensor::new(SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]), tensor)?)
            },
            ThoDevice::from(Device::Cpu),
        ));

        let mut ctx = RunCtxImpl::new(1, "heavy_run_mel_path");

        let mut patch_run_config = MelPatchRunConfig {
            id: None,

            bucket_count: 16,
            min_slices: 10,
            max_slices: 100,

            mel_config: LogMelPatchConfig {
                n_fft: 32,
                hop_length: 4,
                n_mels: 16,
                patch_frames: 9,
                patch_step: 32,
                ..LogMelPatchConfig::default()
            }
        };

        let find_motifs_run_id = store.save_run_config(
            None,
            &FindMotifsRunConfig::type_name(),
            "FindMotifsRunConfig",
            FindMotifsRunConfig::new(None, 0, 64, None, SimilarityMetric::EucDist(EucDistConfig{ z_normalize: true }), SlicesFilter::Accumulate(AccumulateStats{ aggr: Aggr::Max(MaxAggr{ k: 0.0 }) }), None).to_properties_iter(None).collect(),
        ).await?;


        let id = store.save_run_config(
            Some(&[find_motifs_run_id]),
            &MelPatchRunConfig::type_name(),
            "MelPatchRunConfig",
            patch_run_config.to_properties_iter(None).collect(),
        ).await?;

        patch_run_config.id = Some(id);

        let mel_patch_stage = Arc::new(MelPatchStage::new(&ws, &store, Arc::clone(&loader), vec![
            patch_run_config

        ])?);

        let res = mel_patch_stage.run(Arc::new(DevicesWrap::wrap(MelPatchResourcesInput::new(Simple::new(ResourceType::Group("all group".into()))))), ctx.branch("test".into(), false)).await?;

        let motifs = store.list_motifs(id).await?;

        debug!("res={res:?} motifs={motifs:?}");

        let motif_loader_impl = Arc::new(MotifLoader::<RawTensorLoader<RunCtxImpl>, RunCtxImpl>::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            &ws
        ));

        let motif_stats = Arc::new(MotifStats::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            Arc::clone(&ws),
            Arc::clone(&motif_loader_impl),
        ));

        //it's slow, uncomment if raw tensors data is needed
        for resource in Simple::new(ResourceType::Group("all group".into())).to_resources(&store).await? {
            // resource

            plot_resource(
                &ws,
                &store,
                &loader,
                resource.id(),
                None,
                None,
                Resolution::Normal,
                false,
                ctx.branch("plot".into(), false)
            ).await?

        }

        let all_motifs = [MotifKind::ChirpUp, MotifKind::ChirpDown, MotifKind::Trill, MotifKind::HarmonicStack].as_slice().into_iter().map(move |motif|(motif.clone(), Tensor::from_slice(gen_motif(64, &motif).as_slice()).view((1, 1, -1)).to_device(Cuda(0)))).collect::<Vec<(MotifKind, Tensor)>>();
        let mut best_reference_distances = vec![f64::INFINITY; all_motifs.len()];

        for motif_id in &motifs {
            let motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>> = Arc::clone(&motif_loader_impl).run(Arc::new(DevicesWrap::wrap(MotifLoaderInput::new(mono_key!(*motif_id, "").into(), None))), ctx.branch("motif_load".into(), false)).await?;
            debug!("heavy_run_mel_path: motif={motif:?}");
            with_data!(&ctx, motif; |ctx, motif:&DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>|{
                for (ref_idx, ref_motif) in all_motifs.iter().enumerate() {
                    let motif_t = motif.data().as_ref().unsqueeze(0);
                    debug!("heavy_run_mel_path: motif_t={motif_t:?} ref_motif={:?}", ref_motif.1);
                    let distance = clc_d2_similarity(&ref_motif.1, &motif_t, SimilarityMetric::EucDist(EucDistConfig{z_normalize: true}))?;
                    best_reference_distances[ref_idx] = best_reference_distances[ref_idx].min(distance.f_min()?.f_double_value(&[])?);
                    debug!("heavy_run_mel_path: motif_id={motif_id:?} x {:?}:  distance=\n{distance}", ref_motif.0);
                }
                Ok::<_, ThoError>(())
            })?;


            let metrics = Arc::clone(&motif_stats).run(Arc::new(DevicesWrap::wrap(MotifStatsInput::from_motif_input(MotifLoaderInput::new(mono_key!(*motif_id, "").into(), None), vec![
                Metrics::NormThickness(NormThicknessRunConfig::default())
            ]))), ctx.branch("motif_load".into(), false)).await?;

            debug!("heavy_run_mel_path: metrics={metrics:?}");

            plot_motif(
                &ws,
                ctx.branch("plot".into(), false),
                *motif_id as usize,
                &motif_loader_impl,
                Arc::clone(&motif_stats),
                None,
                Resolution::Normal,
                false
            ).await?;

        }

        assert!(!motifs.is_empty());
        debug!("heavy_run_mel_path: best_reference_distances={best_reference_distances:?}");
        assert!(best_reference_distances.iter().filter(|distance| **distance < 1.0).count() >= 2, "expected MelPatch to recover at least two generated motif kinds, best_reference_distances={best_reference_distances:?}");

        Ok(())
    }

}
