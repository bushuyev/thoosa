use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs;
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use parking_lot::RwLock;
use log::{debug, info, warn};
use tch::Tensor;
use thoosa_common::audio::{assign_lsh_buckets_per_channel, audio_to_log_mel_patches, patch_sample_ranges_flat, LogMelPatchConfig};
use thoosa_common::dims::Dims;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_named_derive::Named;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{TensorIdSliceIdx, ThoError, LOCK_TIMEOUT};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Data, Named, WithId};
use thoosa_store::{IdSlice, Store};
use thoosa_with_data::with_data;

const DEFAULT_MEL_PATCH_BUCKET_COUNT: i64 = 256;
const MEL_PATCH_MIN_SLICES: usize = 10;
const MEL_PATCH_MAX_SLICES: usize = 100;
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
    run_id: RunId,
    groups_by_channel: Vec<usize>,
}

impl MelPatchOutput {
    fn new(run_id: RunId, groups_by_channel: Vec<usize>) -> Self {
        Self {
            run_id,
            groups_by_channel,
        }
    }

    pub fn run_id(&self) -> RunId {
        self.run_id
    }

    pub fn groups_by_channel(&self) -> &[usize] {
        &self.groups_by_channel
    }
}

impl WithId for MelPatchOutput {
    type ID = RunId;

    fn id(&self) -> &Self::ID {
        &self.run_id
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
    run_id: RunId,
    mel_config: LogMelPatchConfig,
    bucket_count: i64,
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
        run_id: RunId,
    ) -> Result<Self, ThoError> {
        Self::new_with_bucket_count(ws, store, loader, run_id, DEFAULT_MEL_PATCH_BUCKET_COUNT)
    }

    pub fn new_with_bucket_count(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<L>,
        run_id: RunId,
        bucket_count: i64,
    ) -> Result<Self, ThoError> {
        if bucket_count <= 0 {
            return Err(ThoError::new(format!("MelPatchStage::new_with_bucket_count: bucket_count must be positive, got {bucket_count}")));
        }
        if bucket_count > u16::MAX as i64 + 1 {
            return Err(ThoError::new(format!("MelPatchStage::new_with_bucket_count: bucket_count={bucket_count} exceeds u16 group ord capacity")));
        }

        Ok(Self {
            ws: Arc::clone(ws),
            store: Arc::clone(store),
            loader,
            run_id,
            mel_config: LogMelPatchConfig::default(),
            bucket_count,
            pd: PhantomData,
        })
    }

    fn save_resource_patches(
        &self,
        resource: &Resource,
        audio: &Tensor,
        tmp_path: &Path,
        temp_stores: &mut Vec<MelPatchTempStore>,
    ) -> Result<(), ThoError> {
        let patches = audio_to_log_mel_patches(audio, &self.mel_config)?;
        let patch_ranges = patch_sample_ranges_flat(&patches, &self.mel_config)?;
        let bucket_ids = assign_lsh_buckets_per_channel(&patches, self.bucket_count)?;

        if patch_ranges.len() != bucket_ids.len() {
            return Err(ThoError::new(format!("MelPatchStage::save_resource_patches: ranges channels={} bucket channels={}", patch_ranges.len(), bucket_ids.len())));
        }

        ensure_mel_patch_temp_stores(temp_stores, tmp_path, self.run_id, bucket_ids.len())?;

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
        let tmp_path = thoosa_shared::with_perf_log(
            || self.ws.try_read_for(LOCK_TIMEOUT).expect("MelPatchStage::run: timed out waiting for workspace read lock").tmp_path().clone(),
            "MelPatchStage::run/read_ws_tmp_path",
        );

        let mut temp_stores: Vec<MelPatchTempStore> = Vec::new();
        for resource in &resources {
            let loaded = Arc::clone(&self.loader)
                .run_cached(
                    Arc::new(DevicesWrap::wrap(resource.clone())),
                    ctx.branch(Some(&format!("mel_patch_load_resource_{}", resource.id())), false),
                )
                .await?;

            with_data!(ctx.branch(Some(&format!("mel_patch_bucket_resource_{}", resource.id())), false), loaded; |_ctx, tensor: &ThoTensor<D, Id>| {
                self.save_resource_patches(resource, tensor.t_ref(), &tmp_path, &mut temp_stores)
            })?;
        }

        let mut groups_by_channel = Vec::with_capacity(temp_stores.len());
        for (channel, mut temp_store) in temp_stores.into_iter().enumerate() {
            let groups = filter_mel_patch_groups(temp_store.load_groups()?);
            let groups_len = groups.len();
            self.store.save_groups_as_motifs(self.run_id, channel as i64, groups).await?;
            temp_store.remove()?;
            groups_by_channel.push(groups_len);
        }

        info!("MelPatchStage::run: run_id={} resources={} groups_by_channel={groups_by_channel:?}", self.run_id, resources.len());

        Ok(Arc::new(DevicesWrap::wrap(MelPatchOutput::new(self.run_id, groups_by_channel))))
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

pub fn filter_mel_patch_groups(groups: HashMap<u16, HashSet<IdSlice>>) -> HashMap<u16, HashSet<IdSlice>> {
    groups
        .into_iter()
        .filter_map(|(group, slices)| {
            let len = slices.len();
            if (MEL_PATCH_MIN_SLICES..=MEL_PATCH_MAX_SLICES).contains(&len) {
                Some((group, slices))
            } else {
                warn!("MelPatchStage: skipping group={group} with {len} slices");
                None
            }
        })
        .collect()
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    use super::{filter_mel_patch_groups, MelPatchTempStore};

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
    fn filter_mel_patch_groups_keeps_only_groups_between_ten_and_one_hundred_slices() {
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

        let filtered = filter_mel_patch_groups(groups);
        assert_eq!(filtered.keys().copied().collect::<HashSet<_>>(), [2u16, 3u16].into_iter().collect::<HashSet<_>>());
    }
}
