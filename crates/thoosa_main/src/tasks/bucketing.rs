#![allow(non_camel_case_types)]

use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::fs;
use std::fs::{File, OpenOptions};
use std::hash::Hash;
use std::io::{BufReader, BufWriter, Read, Write};
use std::marker::PhantomData;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::{Mutex, RawMutex, RwLock};
use log::{debug, info, warn};
use tch::{Device, Kind, TchError, Tensor};
use serde::Deserialize;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContext};
use thoosa_common::audio::{assign_lsh_buckets_per_channel, build_mel_filterbank, fft_band_amplitudes_sliced, lsh_bucket_one_hot_3d_exact_buckets, BandAggregation, LshPlanes, LOG_EPS};
use thoosa_common::dims::Dims;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, InputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::runs::SavableResult;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::{Resource, ResourceId, ResourcesConfig};
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::{collect_futures_with_progress, panic_to_err, StreamBuffering, TensorIdSliceIdx, ThoError};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Named, WithId};
use thoosa_store::{IdSlice, Store};
use thoosa_with_data::with_data;
use thoosa_named_derive::Named;
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use crate::buffer_limits::{BUCKETING_LIMIT, STACKED_MOTIFS_LOADER_RUN};

const MEL_PATCH_TEMP_MAGIC: [u8; 8] = *b"THOBUCK1";

#[derive(Debug, Clone)]
pub struct BucketingInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
    resources_config: RC,
}

impl<RC> BucketingInput<RC>
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

impl<RC> WithId for BucketingInput<RC>
where
    RC: ResourcesConfig + Debug + Send + Sync + Clone,
{
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl<RC> Data for BucketingInput<RC>
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

impl<RC> Input for BucketingInput<RC> where RC: ResourcesConfig + Debug + Send + Sync + Clone {}

#[derive(Debug, Clone)]
pub struct BucketingOutput {
    pub(crate) motif_ids: (RunConfigId, Vec<u32>),
}

impl BucketingOutput {
    fn new(run_config_ids_motif_ids: (RunConfigId, Vec<u32>)) -> Self {
        Self { motif_ids: run_config_ids_motif_ids }
    }

    pub fn run_config_ids_motif_ids(&self) -> &(RunConfigId, Vec<u32>) {
        &self.motif_ids
    }
}

impl WithId for BucketingOutput {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Data for BucketingOutput {
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

impl Output for BucketingOutput {}


#[derive(Debug)]
pub struct MelPatchTempStore {//TODO it's similar to FileMotifTrackerStore  but saves single slices instead of pairs. Rename and put together with FileMotifTrackerStore in same mod
    path: PathBuf,
    writer: BufWriter<File>,
}

impl MelPatchTempStore {
    fn new(path: &Path, run_config_id: RunConfigId, bucket: u16) -> Result<Self, ThoError> {
        fs::create_dir_all(path)?;
        let file_path = path.join(format!("{run_config_id}_bucket_{bucket}_bucketing.bin"));
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

    fn save_slices(&mut self, channel: u32, slices: &[TensorIdSliceIdx]) -> Result<(), ThoError> {
        debug!("MelPatchTempStore::save_slices: path={:?} channel={channel} slices.len={}", self.path, slices.len());
        for slice in slices {
            self.writer.write_all(&channel.to_le_bytes())?;
            self.writer.write_all(&slice.0.to_le_bytes())?;
            self.writer.write_all(&slice.1.to_le_bytes())?;
        }
        self.writer.flush()?;
        Ok(())
    }

    fn load_channel_groups(&mut self, group: u16) -> Result<HashMap<u32, HashSet<IdSlice>>, ThoError> {
        self.writer.flush()?;
        let mut reader = BufReader::new(File::open(&self.path)?);
        let mut magic = [0u8; MEL_PATCH_TEMP_MAGIC.len()];
        reader.read_exact(&mut magic)?;
        if magic != MEL_PATCH_TEMP_MAGIC {
            return Err(ThoError::new(format!("MelPatchTempStore::load_channel_groups: unexpected file header in {:?}", self.path)));
        }

        let mut groups: HashMap<u32, HashSet<IdSlice>> = HashMap::new();
        let mut slice_id = 0u32;
        loop {
            let mut channel_bytes = [0u8; 4];
            match reader.read(&mut channel_bytes[0..1])? {
                0 => break,
                1 => reader.read_exact(&mut channel_bytes[1..4])?,
                n => return Err(ThoError::new(format!("MelPatchTempStore::load_channel_groups: unexpected read size {n}"))),
            }

            let mut resource_id_bytes = [0u8; 4];
            let mut slice_idx_bytes = [0u8; 4];
            reader.read_exact(&mut resource_id_bytes)?;
            reader.read_exact(&mut slice_idx_bytes)?;

            slice_id = slice_id.checked_add(1).ok_or_else(|| ThoError::new("MelPatchTempStore::load_channel_groups: slice_id overflow"))?;
            groups.entry(u32::from_le_bytes(channel_bytes)).or_default().insert(IdSlice(
                slice_id,
                u32::from_le_bytes(resource_id_bytes),
                u32::from_le_bytes(slice_idx_bytes),
            ));
        }

        debug!("MelPatchTempStore::load_channel_groups: group={group} channels={}", groups.len());
        Ok(groups)
    }

    fn remove(self) -> Result<(), ThoError> {
        fs::remove_file(&self.path)?;
        Ok(())
    }
}

#[derive(Debug, Named)]
pub struct BucketingStage<RC, L, D, Id, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<D, Id>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    run_config: BucketRunConfig,
    sync_bucketing_stage: Arc<SyncBucketingStage<D, Id, C>>,
    pd: PhantomData<(RC, D, Id, C)>,
    channels:u32,
    bucket_window_samples: i64,
    bucket_step_samples: i64,
    tmp_path:PathBuf,
    temp_stores_by_bucket: Arc<parking_lot::lock_api::Mutex<RawMutex, Vec<MelPatchTempStore>>>
}

impl<RC, L, D, Id, C> BucketingStage<RC, L, D, Id, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<D, Id>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<L>,
        run_config: BucketRunConfig,
        window: u32,
        step: u32,
    ) -> Result<Self, ThoError> {


        if run_config.bucket_count == 0 {
            return Err(ThoError::new("BucketingStage::new: bucket_count must be positive"));
        }
        if run_config.bucket_count > u16::MAX as u32 + 1 {
            return Err(ThoError::new(format!("BucketingStage::new: bucket_count={} exceeds u16 group ord capacity", run_config.bucket_count)));
        }
        if run_config.min_slices > run_config.max_slices {
            return Err(ThoError::new(format!("BucketingStage::new: min_slices={} exceeds max_slices={}", run_config.min_slices, run_config.max_slices)));
        }
        if window == 0 || step == 0 {
            return Err(ThoError::new(format!("BucketingStage::new: window={window} step={step} must be positive")));
        }


        let bucket_window_samples = i64::from(run_config.window_samples(window, step)?);
        let bucket_step_samples = bucket_step_samples_for_config(&run_config, step)?;
        let window = i64::from(window);
        let step = i64::from(step);

        let (channels, sample_rate, tmp_path) = {
            let ws = thoosa_shared::perf_log!(thoosa_shared::sync_read_lock!(ws));
            (ws.channels(), ws.sample_rate(), ws.tmp_path().clone())
        };



        let temp_stores_by_bucket =
            (0..run_config.bucket_count).map(|bucket| {
                let bucket = u16::try_from(bucket).map_err(|_| ThoError::new(format!("BucketingStage::run: bucket={bucket} does not fit into u16")))?;
                MelPatchTempStore::new(&tmp_path, run_config.id.expect("no run id").clone(), bucket)
            })
            .collect::<Result<Vec<_>, ThoError>>()?;

        let temp_stores_by_bucket: Arc<parking_lot::lock_api::Mutex<RawMutex, Vec<MelPatchTempStore>>> = Arc::new(Mutex::new(temp_stores_by_bucket));

        Ok(Self {
            ws: Arc::clone(ws),
            store: Arc::clone(store),
            loader,
            sync_bucketing_stage: Arc::new(SyncBucketingStage::new(run_config.clone(), window, step, sample_rate, Arc::clone(&temp_stores_by_bucket))),
            run_config,
            pd: PhantomData,
            channels,
            bucket_window_samples,
            bucket_step_samples,
            tmp_path,
            temp_stores_by_bucket
        })
    }

}

#[derive(Debug)]
struct BucketingSyncInput<'a, D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    resource_id: ResourceId,
    tensor: &'a ThoTensor<D, Id>,
    sample_offset: i64,
}

impl<'a, D, Id> BucketingSyncInput<'a, D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    fn new(resource_id: ResourceId, tensor: &'a ThoTensor<D, Id>, sample_offset: i64) -> Self {
        Self { resource_id, tensor, sample_offset }
    }
}

impl<D, Id> WithId for BucketingSyncInput<'_, D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    type ID = Id;

    fn id(&self) -> &Self::ID {
        self.tensor.id()
    }
}

impl<D, Id> Data for BucketingSyncInput<'_, D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    type Metrics = D::Metrics;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        Data::size(self.tensor)
    }

    fn device(&self) -> usize {
        Data::device(self.tensor)
    }

    fn mem_size(&self) -> u64 {
        Data::mem_size(self.tensor)
    }
}

impl<D, Id> Input for BucketingSyncInput<'_, D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{}

#[derive(Debug)]
struct BucketingTensorInput<D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    resource_id: ResourceId,
    tensor: Arc<DevicesWrap<ThoTensor<D, Id>>>,
    bucket_window_samples: i64,
    bucket_step_samples: i64,
}

impl<D, Id> BucketingTensorInput<D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    fn new(resource_id: ResourceId, tensor: Arc<DevicesWrap<ThoTensor<D, Id>>>, bucket_window_samples: i64, bucket_step_samples: i64) -> Self {
        Self { resource_id, tensor, bucket_window_samples, bucket_step_samples }
    }
}

impl<D, Id> WithId for BucketingTensorInput<D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    type ID = Id;

    fn id(&self) -> &Self::ID {
        self.tensor.id()
    }
}

impl<D, Id> WithDataRef for BucketingTensorInput<D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    type O<'o> = BucketingSyncInput<'o, D, Id>;

    fn with_data_ref<'a, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move {
            let resource_id = self.resource_id;

            self.tensor.with_data_ref(to_device_idx, permits_ctx, trace_ctx, move |tensor, ctx| {
                let input = BucketingSyncInput::new(resource_id, tensor, 0);
                do_run(&input, ctx)
            }).await
        }
    }
}

#[derive(Debug, Clone)]
struct SelectBucketsInputSelector {
    bucket_slice_range: Range<i64>,
}

impl Default for SelectBucketsInputSelector {
    fn default() -> Self {
        Self { bucket_slice_range: 0..0 }
    }
}

impl From<Range<i64>> for SelectBucketsInputSelector {
    fn from(bucket_slice_range: Range<i64>) -> Self {
        Self { bucket_slice_range }
    }
}

impl Adjustable for SelectBucketsInputSelector {}

impl<D, Id> WithSelectDataRef<()> for BucketingTensorInput<D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, permits_ctx: P, _selector: (), trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move {
            self.with_data_ref(to_device_idx, permits_ctx, trace_ctx, do_run).await
        }
    }
}

impl<D, Id> WithSelectDataRef<SelectBucketsInputSelector> for BucketingTensorInput<D, Id>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
{
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, permits_ctx: P, selector: SelectBucketsInputSelector, trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move {
            let resource_id = self.resource_id;
            let bucket_window_samples = self.bucket_window_samples;
            let bucket_step_samples = self.bucket_step_samples;
            self.tensor.with_data_ref(to_device_idx, permits_ctx, trace_ctx, move |tensor, ctx| {
                let audio = tensor.t_ref();
                let audio_size = audio.size();
                if audio_size.len() != 2 {
                    return Err(ThoError::new(format!("BucketingTensorInput::with_selected_data_ref: expected audio shape [channels, samples], got {audio_size:?}")));
                }
                let sample_range = bucket_selector_sample_range(&selector, audio_size[1], bucket_window_samples, bucket_step_samples)?;
                debug!("BucketingTensorInput::with_selected_data_ref: selector={selector:?} sample_range={sample_range:?} audio_size={audio_size:?}");
                let sample_len = sample_range.end.checked_sub(sample_range.start).ok_or_else(|| ThoError::new(format!("BucketingTensorInput::with_selected_data_ref: invalid sample_range={sample_range:?}")))?;
                let selected_audio = audio.f_narrow(1, sample_range.start, sample_len)?;
                let selected_tensor = ThoTensor::new(tensor.id().clone(), selected_audio)?;
                let input = BucketingSyncInput::new(resource_id, &selected_tensor, sample_range.start);
                do_run(&input, ctx)
            }).await
        }
    }
}

#[derive(Debug)]
struct SelectBucketsSplitterContext<'a, ST>
where
    ST: Stage<Out = ()> + Adjustable + 'a,
{
    input_len: i64,
    split_len: i64,
    done_to: i64,
    pd: PhantomData<&'a ST>,
}

impl<'a, ST> SelectBucketsSplitterContext<'a, ST>
where
    ST: Stage<Out = ()> + Adjustable + 'a,
{
    fn new(input_len: i64) -> Self {
        Self { input_len, split_len: input_len, done_to: 0, pd: PhantomData }
    }
}

impl<'a, ST> SplitsAdjusterContext for SelectBucketsSplitterContext<'a, ST>
where
    ST: Stage<Out = ()> + Adjustable + 'a,
{
    type Metrics = i64;

    fn input_len(&self) -> &Self::Metrics {
        &self.input_len
    }

    fn split_len(&self) -> &Self::Metrics {
        &self.split_len
    }

    fn done_to(&self) -> &Self::Metrics {
        &self.done_to
    }

    fn split_len_mut(&mut self) -> &mut Self::Metrics {
        &mut self.split_len
    }
}

impl<'a, 'z, ST> InputContext<'a, ST> for SelectBucketsSplitterContext<'z, ST>
where
    ST: Stage<Out = ()> + Adjustable + 'a,
{
    async fn add_partial_output(&mut self, _out: ST::Out) {
        self.done_to += self.split_len;
        debug!("SelectBucketsSplitterContext::add_partial_output: done_to={} split_len={} input_len={}", self.done_to, self.split_len, self.input_len);
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }

    async fn build_final_output(self) -> Result<ST::Out, ThoError> {
        Ok(())
    }
}

#[derive(Debug)]
struct SelectBucketsAdjuster {
    bucket_slice_adjuster: SplitsAdjuster<SelectBucketsInputSelector, i64>,
}

impl SelectBucketsAdjuster {
    fn new() -> Self {
        Self { bucket_slice_adjuster: SplitsAdjuster::new(1) }
    }
}

#[async_trait]
impl<'a, ST> Adjuster<SelectBucketsSplitterContext<'a, ST>> for SelectBucketsAdjuster
where
    ST: Stage<Out = ()> + Adjustable + 'a,
{
    type A = SelectBucketsInputSelector;

    async fn adjust(&self, adjustable: Self::A, adjust_context: &mut SelectBucketsSplitterContext<'a, ST>) -> Result<Self::A, AdjusterError<Self::A>> {
        let result = self.bucket_slice_adjuster.adjust(adjustable, adjust_context).await;
        debug!("SelectBucketsAdjuster::adjust: adjust_context={adjust_context:?} result={result:?}");
        result
    }

    fn degrade(&self, adjust_context: &mut SelectBucketsSplitterContext<'a, ST>) -> bool {
        self.bucket_slice_adjuster.degrade(adjust_context)
    }
}

#[derive(Debug, Named)]
struct SyncBucketingStage<D, Id, C>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{
    run_config: BucketRunConfig,
    window: i64,
    step: i64,
    sample_rate:u32,
    temp_stores_by_bucket: Arc<parking_lot::lock_api::Mutex<RawMutex, Vec<MelPatchTempStore>>>,
    pd: PhantomData<(D, Id, C)>,
}

impl<D, Id, C> SyncBucketingStage<D, Id, C>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn new(run_config: BucketRunConfig, window: i64, step: i64, sample_rate:u32, temp_stores_by_bucket: Arc<parking_lot::lock_api::Mutex<RawMutex, Vec<MelPatchTempStore>>>) -> Self {
        Self { run_config, window, step, temp_stores_by_bucket, sample_rate, pd: PhantomData }
    }

    fn save_resource_patches(&self, input: &BucketingSyncInput<'_, D, Id>, mel_config:&LogMelPatchConfig) -> Result<(), ThoError> {
        let patches = audio_to_log_mel_patches(input.tensor.t_ref(), mel_config, self.sample_rate as f64, self.window, self.step)?;
        let patch_ranges = patch_sample_ranges_flat(&patches, mel_config, self.window, self.step)?;
        let channels_bucket_ids = assign_lsh_buckets_per_channel(&patches, self.run_config.bucket_count_i64())?;

        if patch_ranges.len() != channels_bucket_ids.len() {
            return Err(ThoError::new(format!("SyncBucketingStage::save_resource_patches: ranges channels={} bucket channels={}", patch_ranges.len(), channels_bucket_ids.len())));
        }

        // TODO(perf): temp_stores_by_bucket is locked while slices are appended; shard by bucket or buffer per resource and flush outside this global mutex.
        let mut temp_stores = self.temp_stores_by_bucket.lock();
        if temp_stores.len() != self.run_config.bucket_count as usize {
            return Err(ThoError::new(format!("SyncBucketingStage::save_resource_patches: temp stores={} bucket_count={}", temp_stores.len(), self.run_config.bucket_count)));
        }

        for channel in 0..channels_bucket_ids.len() {
            if patch_ranges[channel].len() != channels_bucket_ids[channel].len() {
                return Err(ThoError::new(format!("SyncBucketingStage::save_resource_patches: channel={channel} ranges len={} bucket len={}", patch_ranges[channel].len(), channels_bucket_ids[channel].len())));
            }

            let mut slices_by_bucket: HashMap<u16, Vec<TensorIdSliceIdx>> = HashMap::new();

            for ((sample_start, _sample_end), bucket) in patch_ranges[channel].iter().zip(channels_bucket_ids[channel].iter()) {
                let group = u16::try_from(*bucket).map_err(|_| ThoError::new(format!("SyncBucketingStage::save_resource_patches: bucket={bucket} does not fit into u16")))?;
                let sample_start = i64::try_from(*sample_start).map_err(|_| ThoError::new(format!("SyncBucketingStage::save_resource_patches: sample_start={sample_start} does not fit into i64")))?;
                let slice_start = input.sample_offset.checked_add(sample_start).ok_or_else(|| ThoError::new("SyncBucketingStage::save_resource_patches: sample_start overflow"))?;
                let slice_start = u32::try_from(slice_start).map_err(|_| ThoError::new(format!("SyncBucketingStage::save_resource_patches: sample_start={slice_start} does not fit into u32")))?;
                slices_by_bucket.entry(group).or_default().push(TensorIdSliceIdx(input.resource_id, slice_start));
            }

            for (group, slices) in slices_by_bucket {
                debug!("SyncBucketingStage::save_resource_patches: group={group} slices={slices:?}");
                temp_stores[group as usize].save_slices(channel as u32, &slices)?;
            }
        }

        Ok(())
    }

    fn save_resource_plain_buckets(&self, input: &BucketingSyncInput<'_, D, Id>, plain_config: &PlainConfig) -> Result<(), ThoError> {
        let audio = input.tensor.t_ref();
        let audio_size = audio.size();
        debug!("SyncBucketingStage::save_resource_plain_buckets: id={:?} audio_size={audio_size:?}", input.id());

        if audio_size.len() != 2 {
            return Err(ThoError::new(format!("SyncBucketingStage::save_resource_plain_buckets: expected audio shape [channels, samples], got {audio_size:?}")));
        }
        if self.window <= 0 || self.step <= 0 {
            return Err(ThoError::new(format!("SyncBucketingStage::save_resource_plain_buckets: invalid window={} step={} plain_config={plain_config:?}", self.window, self.step)));
        }
        if audio_size[1] < self.window {
            return Err(ThoError::new(format!("SyncBucketingStage::save_resource_plain_buckets: audio too short samples={} window={}", audio_size[1], self.window)));
        }

        let slices = audio.f_unfold(1, self.window, self.step)?;
        debug!("SyncBucketingStage::save_resource_plain_buckets: id={:?} slices={slices:?}", input.id());

        let channels_slices_freqamp = panic_to_err!(fft_band_amplitudes_sliced(&slices, self.sample_rate as f64, plain_config.band_hz, plain_config.aggregation.to_audio(), plain_config.apply_hann_window, plain_config.remove_dc_offset))?;
        debug!("SyncBucketingStage::save_resource_plain_buckets: channels_slices_freqamp={channels_slices_freqamp:?}");

        let planes = deterministic_plain_lsh_planes_for_buckets(channels_slices_freqamp.size()[2], self.run_config.bucket_count as i64, Kind::Float, channels_slices_freqamp.device())?;
        debug!("SyncBucketingStage::save_resource_plain_buckets: planes={planes:?}");

        let buckets_one_hot = panic_to_err!(lsh_bucket_one_hot_3d_exact_buckets(&channels_slices_freqamp, &planes, plain_config.normalize_lsh))?;
        debug!("SyncBucketingStage::save_resource_plain_buckets: buckets_one_hot={buckets_one_hot:?}");

        let channels_bucket_ids = panic_to_err!(bucket_ids_from_one_hot_3d(&buckets_one_hot))?;

        // TODO(perf): this uses the same global bucket-store lock as mel patches; accumulate per-resource groups first and keep the critical section to the final save calls.
        let mut temp_stores = self.temp_stores_by_bucket.lock();
        if temp_stores.len() != self.run_config.bucket_count as usize {
            return Err(ThoError::new(format!("SyncBucketingStage::save_resource_plain_buckets: temp stores={} bucket_count={}", temp_stores.len(), self.run_config.bucket_count)));
        }

        for (channel, bucket_ids) in channels_bucket_ids.iter().enumerate() {
            let mut slices_by_bucket: HashMap<u16, Vec<TensorIdSliceIdx>> = HashMap::new();
            for (slice_idx, bucket) in bucket_ids.iter().enumerate() {
                let group = u16::try_from(*bucket).map_err(|_| ThoError::new(format!("SyncBucketingStage::save_resource_plain_buckets: bucket={bucket} does not fit into u16")))?;
                let sample_start = (slice_idx as i64).checked_mul(self.step).and_then(|v| v.checked_add(input.sample_offset)).ok_or_else(|| ThoError::new("SyncBucketingStage::save_resource_plain_buckets: sample_start overflow"))?;
                let slice_start = u32::try_from(sample_start).map_err(|_| ThoError::new(format!("SyncBucketingStage::save_resource_plain_buckets: sample_start={sample_start} does not fit into u32")))?;
                slices_by_bucket.entry(group).or_default().push(TensorIdSliceIdx(input.resource_id, slice_start));
            }

            for (group, slices) in slices_by_bucket {
                debug!("SyncBucketingStage::save_resource_plain_buckets: group={group} slices={slices:?}");
                temp_stores[group as usize].save_slices(channel as u32, &slices)?;
            }
        }

        Ok(())
    }
}

impl<D, Id, C> Adjustable for SyncBucketingStage<D, Id, C>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{}

impl<D, Id, C> OutId for SyncBucketingStage<D, Id, C>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn out_id(&self, _input_id: &<<Self as Stage>::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        ()
    }
}

impl<D, Id, C> Stage for SyncBucketingStage<D, Id, C>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = BucketingSyncInput<'a, D, Id>;
    type Out = ();
}

impl<D, Id, C> SyncStage for SyncBucketingStage<D, Id, C>
where
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncBucketingStage::do_run"))]
    fn  do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.run_config.mel_config() {
            BucketingConfig::Mel(mel_config) => self.save_resource_patches(input, mel_config)?,
            BucketingConfig::Plain(plain_config) => self.save_resource_plain_buckets(input, plain_config)?,
        }

        Ok(())
    }
}

/// Convert audio `[channels, samples]` into log-mel patches:
///
/// Output shape:
///
/// `[channels, patches, n_mels, patch_frames]`
pub fn audio_to_log_mel_patches(
    audio: &Tensor,
    cfg: &LogMelPatchConfig,
    sample_rate: f64,
    window: i64,
    step: i64,
) -> Result<Tensor, ThoError> {
    debug!("audio_to_log_mel_patches: audio={audio:?} cfg={cfg:?} window={window} step={step}");

    let size = audio.size();

    if size.len() != 2 {
        return Err(ThoError::new(format!("expected audio shape [channels, samples], got {size:?}")));
    }

    let channels = size[0];
    let samples = size[1];

    if samples < window {
        return Err(ThoError::new(format!("audio too short: samples={samples}, n_fft={window}")));
    }

    if window <= 0 || step <= 0 || cfg.n_mels <= 0 {
        return Err(ThoError::new(format!("invalid config: n_fft={window}, hop_length={step}, n_mels={}", cfg.n_mels)));
    }

    if cfg.mel_patch_size <= 0 || cfg.mel_patch_step <= 0 {
        return Err(ThoError::new(format!("invalid patch config: patch_frames={}, patch_step={}", cfg.mel_patch_size, cfg.mel_patch_step)));
    }

    let device = audio.device();

    // Work in float.
    let audio = audio.f_to_kind(Kind::Float)?;

    // ---------------------------------------------------------------------
    // 1. Frame audio with unfold.
    //
    // audio:  [C, samples]
    // frames: [C, frames, n_fft]
    // ---------------------------------------------------------------------
    let frames = audio.f_unfold(1, window, step)?;
    debug!("audio_to_log_mel_patches: frames={frames:?}");
    // ---------------------------------------------------------------------
    // 2. Apply Hann window.
    //
    // window: [n_fft]
    // frames: [C, frames, n_fft]
    // ---------------------------------------------------------------------
    let hann = Tensor::f_hann_window(window, (Kind::Float, device))?;
    debug!("audio_to_log_mel_patches: hann={hann:?}");

    let frames = frames.f_mul(&hann.f_view([1, 1, window])?)?;
    debug!("audio_to_log_mel_patches: frames x hann={frames:?}");

    // ---------------------------------------------------------------------
    // 3. RFFT over the last dimension.
    //
    // spectrum: [C, frames, freq_bins] complex
    // power:    [C, frames, freq_bins] real
    // ---------------------------------------------------------------------
    let spectrum = frames.f_fft_rfft(None, -1, "backward")?;
    debug!("audio_to_log_mel_patches: spectrum={spectrum:?}");

    let power = spectrum.f_abs()?.f_pow_tensor_scalar(2.0)?;

    debug!("audio_to_log_mel_patches: power={power:?}");

    let freq_bins = window / 2 + 1;

    // ---------------------------------------------------------------------
    // 4. Mel projection.
    //
    // power:        [C, frames, F]
    // mel_fb:       [M, F]
    // power_flat:   [C * frames, F]
    // mel_flat:     [C * frames, M]
    // mel:          [C, M, frames]
    // ---------------------------------------------------------------------
    //called once per RunConfig
    let mel_fb = build_mel_filterbank(
        sample_rate,
        window,
        cfg.n_mels,
        cfg.f_min,
        cfg.f_max,
        device,
    )?;

    debug!("audio_to_log_mel_patches: mel_fb=\n{mel_fb}");

    let n_frames = power.size()[1];

    let power_flat = power.f_view([channels * n_frames, freq_bins])?;

    debug!("audio_to_log_mel_patches: power_flat={power_flat:?}");

    let mel_flat = power_flat.f_matmul(&mel_fb.f_transpose(0, 1)?)?;

    debug!("audio_to_log_mel_patches: mel_flat={mel_flat:?}");

    let mel = mel_flat
        .f_view([channels, n_frames, cfg.n_mels])?
        .f_transpose(1, 2)?; // [C, M, frames]

    debug!("audio_to_log_mel_patches: mel={mel:?}");
    // ---------------------------------------------------------------------
    // 5. Log compression.
    //
    // log_mel: [C, M, frames]
    // ---------------------------------------------------------------------
    let log_mel = mel.f_clamp_min(LOG_EPS)?.f_log()?;

    // ---------------------------------------------------------------------
    // 6. Slice time axis into patches.
    //
    // log_mel: [C, M, frames]
    // patches: [C, M, patches, patch_frames]
    // output:  [C, patches, M, patch_frames]
    // ---------------------------------------------------------------------
    if log_mel.size()[2] < cfg.mel_patch_size {
        return Err(ThoError::new(format!("not enough STFT frames for one patch: frames={}, patch_frames={}", log_mel.size()[2], cfg.mel_patch_size)));
    }

    let patches = log_mel.f_unfold(2, cfg.mel_patch_size, cfg.mel_patch_step)?;
    let patches = patches.f_permute(&[0, 2, 1, 3])?;

    Ok(patches)
}

/// For patches shaped [channels, patches, n_mels, patch_frames],
/// returns [channels x patches, (sample_start, sample_end)].
pub fn patch_sample_ranges_flat(
    patches: &Tensor,
    cfg: &LogMelPatchConfig,
    window: i64,
    step: i64,
) -> Result<Vec<Vec<(usize, usize)>>, TchError> {
    let size = patches.size();

    if size.len() != 4 {
        return Err(TchError::Shape(format!(
            "expected [channels, patches, n_mels, patch_frames], got {size:?}"
        )));
    }

    let channels = size[0];
    let n_patches = size[1];

    // let mut values = Vec::<i64>::with_capacity((channels * n_patches * 4) as usize);
    let mut values = vec![vec![]; channels as usize];

    for c in 0..channels {
        for p in 0..n_patches {
            let first_frame = p * cfg.mel_patch_step;
            let last_frame = first_frame + cfg.mel_patch_size - 1;

            let sample_start = first_frame * step;
            let sample_end = last_frame * step + window;


            values[c as usize].push((sample_start as usize, sample_end as usize))
        }
    }

    Ok(values)
}


impl<RC, L, D, Id, C> Stage for BucketingStage<RC, L, D, Id, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<D, Id>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = BucketingInput<RC>
    where
        Self: 'a;
    type Out = BucketingOutput;
}

impl<RC, L, D, Id, C> AsyncStage for BucketingStage<RC, L, D, Id, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<D, Id>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    D: Dims + Send + Sync + 'static,
    Id: CachableIdTr + Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "BucketingStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let resources_config = with_data!(ctx.branch(false), input; |_ctx, input: &BucketingInput<RC>| {
            Ok::<_, ThoError>(input.resources_config().clone())
        })?;
        let resources = resources_config.to_resources(&self.store).await?;

        debug!("BucketingStage::run: resources_config={resources_config:?} resources={resources:?}");


        let eta_guard = CollectEtaInvocationGuard::new("BucketingStage/resources", resources.len(), 1);
        let futures = resources.iter().map(|resource|{
            let mut sync_bucketing_ctx = ctx.branch(false);
            let loader = Arc::clone(&self.loader);
            let sync_stage = Arc::clone(&self.sync_bucketing_stage);
            let bucket_window_samples = self.bucket_window_samples;
            let bucket_step_samples = self.bucket_step_samples;
            async move {

                let loaded = loader
                    .run_cached(
                        Arc::new(DevicesWrap::wrap(resource.clone())),
                        sync_bucketing_ctx.branch(false),
                    )
                    .await?;

                let required_ids: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
                let bucket_slice_count = bucket_slice_count(i64::from(resource.length()), bucket_window_samples, bucket_step_samples)?;
                let oom_handler: OOMHandler<'_, '_, _, _, _, _, SelectBucketsInputSelector> = OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                        Adjusters::RunContext(Box::new(DumpToCpu::new("bucketing"))),
                        Adjusters::Input(Box::new(SelectBucketsAdjuster::new())),
                    ],
                    None,
                );

                match oom_handler
                    .run(
                        &sync_stage,
                        Arc::new(BucketingTensorInput::new(resource.id(), loaded, bucket_window_samples, bucket_step_samples)),
                        sync_bucketing_ctx,
                        Contexts::new(
                            SelectBucketsSplitterContext::new(bucket_slice_count),
                            NoopRunnableContext {},
                            MaxParallelAndDumpToCpuAdjustableContex::new(
                                MaxParallelAdjustContextImpl::new(),
                                DumpToCpuContextImpl::new(&required_ids),
                            ),
                        ),
                    )
                    .await
                {
                    Ok(_) => {}
                    Err(e) => {
                        warn!("BucketingStage::run e={e}");
                    }
                };
                Ok(())
            }
        });

        let loaded_motifs = collect_futures_with_progress(
            futures,
            resources.len(),
            BUCKETING_LIMIT,
            StreamBuffering::Unordered,
            "BucketingStage/resources",
        )
        .await
        .into_iter()
        .collect::<Vec<_>>();


        let run_config_id = self.run_config.run_config_id()?;
        let mut motif_ids = Vec::new();
        let mut groups_by_channel = (0..self.channels).map(|_| HashMap::<u16, HashSet<IdSlice>>::new()).collect::<Vec<_>>();

        {
            let mut temp_stores_by_bucket = self.temp_stores_by_bucket.lock();
            let eta_guard = CollectEtaInvocationGuard::new("BucketingStage/buckets", temp_stores_by_bucket.len() as usize, 1);
            for (bucket, mut temp_store) in temp_stores_by_bucket.drain(..).enumerate() {
                let group = u16::try_from(bucket).map_err(|_| ThoError::new(format!("BucketingStage::run: bucket={bucket} does not fit into u16")))?;
                for (channel, slices) in temp_store.load_channel_groups(group)? {
                    let channel = usize::try_from(channel).map_err(|_| ThoError::new(format!("BucketingStage::run: channel={channel} does not fit into usize")))?;
                    if channel >= groups_by_channel.len() {
                        return Err(ThoError::new(format!("BucketingStage::run: temp store channel={channel} exceeds workspace channels={}", self.channels)));
                    }
                    groups_by_channel[channel].entry(group).or_default().extend(slices);
                }
                temp_store.remove()?;
                collect_eta_record_completion(eta_guard.id());
            }
        }

        for (channel, groups) in groups_by_channel.into_iter().enumerate() {
            let groups = filter_mel_patch_groups(groups, self.run_config.min_slices_usize(), self.run_config.max_slices_usize());
            let group_motif_ids = self.store.save_groups_as_motifs(run_config_id, channel as i64, groups).await?;
            motif_ids.extend(group_motif_ids);
        }

        debug!("BucketingStage::run: run_config_id={run_config_id} resources={} motif_ids={motif_ids:?}", resources.len());


        info!("BucketingStage::run: run_config_id={run_config_id:?} done");

        Ok(Arc::new(DevicesWrap::wrap(BucketingOutput::new((run_config_id, motif_ids)))))
    }
}

pub fn filter_mel_patch_groups(groups: HashMap<u16, HashSet<IdSlice>>, min_slices: usize, max_slices: usize) -> HashMap<u16, HashSet<IdSlice>> {
    groups
        .into_iter()
        .filter_map(|(group, slices)| {
            let len = slices.len();
            if (min_slices..=max_slices).contains(&len) {
                Some((group, slices))
            } else {
                warn!("BucketingStage: skipping group={group} with {len} slices");
                None
            }
        })
        .collect()
}

fn bucket_ids_from_one_hot_3d(buckets: &Tensor) -> Result<Vec<Vec<usize>>, ThoError> {
    let size = buckets.size();
    if size.len() != 3 {
        return Err(ThoError::new(format!("bucket_ids_from_one_hot_3d: expected [channels, slices, buckets], got {size:?}")));
    }

    let channels = size[0];
    let slices = size[1];
    // TODO(perf): this forces a device-to-CPU roundtrip for every bucketed tensor; keep bucket assignment/grouping on device or batch the CPU conversion per resource.
    let bucket_ids = buckets.f_argmax(Some(2), false)?.f_to_device(Device::Cpu)?.f_to_kind(Kind::Int64)?.f_contiguous()?;
    let flat = Vec::<i64>::try_from(&bucket_ids.f_reshape([-1])?)?;
    let mut result = vec![Vec::with_capacity(slices as usize); channels as usize];
    for channel in 0..channels {
        for slice in 0..slices {
            let idx = (channel * slices + slice) as usize;
            let bucket = usize::try_from(flat[idx]).map_err(|_| ThoError::new(format!("bucket_ids_from_one_hot_3d: bucket={} does not fit into usize", flat[idx])))?;
            result[channel as usize].push(bucket);
        }
    }

    Ok(result)
}

fn deterministic_plain_lsh_planes(data_dim: i64, num_bits: i64, kind: Kind, device: Device) -> Result<Tensor, ThoError> {
    if data_dim <= 0 {
        return Err(ThoError::new("deterministic_plain_lsh_planes: data_dim must be > 0"));
    }
    if num_bits <= 0 || num_bits > 30 {
        return Err(ThoError::new("deterministic_plain_lsh_planes: num_bits must be in 1..=30"));
    }

    let count = usize::try_from(data_dim.checked_mul(num_bits).ok_or_else(|| ThoError::new("deterministic_plain_lsh_planes: plane size overflow"))?).map_err(|_| ThoError::new("deterministic_plain_lsh_planes: plane size does not fit into usize"))?;
    let mut seed = 0x9e3779b97f4a7c15u64 ^ (data_dim as u64).wrapping_mul(0xbf58476d1ce4e5b9) ^ (num_bits as u64).wrapping_mul(0x94d049bb133111eb);
    let mut values = Vec::with_capacity(count);
    for idx in 0..count {
        seed ^= (idx as u64).wrapping_mul(0x9e3779b97f4a7c15);
        seed = seed.wrapping_mul(0xbf58476d1ce4e5b9).rotate_left(27);
        let unit = ((seed >> 40) as f32) / ((1u32 << 24) as f32);
        values.push(unit * 2.0 - 1.0);
    }

    Ok(Tensor::f_from_slice(&values)?.f_to_kind(kind)?.f_to_device(device)?.f_view([data_dim, num_bits])?)
}

fn deterministic_plain_lsh_planes_for_buckets(data_dim: i64, wanted_buckets: i64, kind: Kind, device: Device) -> Result<LshPlanes, ThoError> {
    if wanted_buckets <= 1 {
        return Err(ThoError::new("deterministic_plain_lsh_planes_for_buckets: wanted_buckets must be > 1"));
    }
    let mut num_bits = 0_i64;
    let mut raw_buckets = 1_i64;
    while raw_buckets < wanted_buckets {
        num_bits += 1;
        if num_bits > 30 {
            return Err(ThoError::new("deterministic_plain_lsh_planes_for_buckets: too many buckets requested; refusing num_bits > 30"));
        }
        raw_buckets <<= 1;
    }
    Ok(LshPlanes {
        planes: deterministic_plain_lsh_planes(data_dim, num_bits, kind, device)?,
        num_bits,
        raw_buckets,
        wanted_buckets,
    })
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub enum PlainBandAggregation {
    #[run_config_default]
    Sum(PlainBandAggregationSum),
    Mean(PlainBandAggregationMean),
}

impl Default for PlainBandAggregation {
    fn default() -> Self {
        Self::Sum(PlainBandAggregationSum {})
    }
}

impl PlainBandAggregation {
    fn to_audio(&self) -> BandAggregation {
        match self {
            Self::Sum(_) => BandAggregation::Sum,
            Self::Mean(_) => BandAggregation::Mean,
        }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct PlainBandAggregationSum {}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Default)]
pub struct PlainBandAggregationMean {}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub struct PlainConfig {
    #[run_config]
    #[in_map]
    pub band_hz: f64,

    #[run_config]
    #[in_map]
    pub aggregation: PlainBandAggregation,

    #[run_config]
    #[in_map]
    pub apply_hann_window: bool,

    #[run_config]
    #[in_map]
    pub remove_dc_offset: bool,

    #[run_config]
    #[in_map]
    pub normalize_lsh: bool,
}

impl Default for PlainConfig {
    fn default() -> Self {
        Self {
            band_hz: 1000.0,
            aggregation: PlainBandAggregation::default(),
            apply_hann_window: false,
            remove_dc_offset: true,
            normalize_lsh: true,
        }
    }
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[allow(non_camel_case_types)]
pub enum BucketingConfig {
    #[run_config_default]
    Mel(LogMelPatchConfig),
    Plain(PlainConfig),
}

impl Default for BucketingConfig {
    fn default() -> Self {
        Self::Mel(LogMelPatchConfig::default())
    }
}


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
#[run_config_grid]
pub struct BucketRunConfig {
    #[in_map]
    pub id: Option<RunConfigId>,

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
    bucket_config: BucketingConfig
}

impl BucketRunConfig {

    pub fn run_config_id(&self) -> Result<RunConfigId, ThoError> {
        self.id.ok_or_else(|| ThoError::new("BucketRunConfig::run_config_id: missing run id"))
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

    pub fn mel_config(&self) -> &BucketingConfig {
        &self.bucket_config
    }


    pub fn window_samples(&self, window: u32, step: u32) -> Result<u32, ThoError> {
        match &self.bucket_config {
            BucketingConfig::Mel(mel_config) => {
                if window == 0 || step == 0 || mel_config.mel_patch_size <= 0 {
                    return Err(ThoError::new(format!("BucketRunConfig::window_samples: invalid window={window} step={step} mel_config={mel_config:?}")));
                }
                let patch_frames = u32::try_from(mel_config.mel_patch_size).map_err(|_| ThoError::new(format!("BucketRunConfig::window_samples: patch_frames={} does not fit into u32", mel_config.mel_patch_size)))?;

                step.checked_mul(patch_frames.saturating_sub(1)).and_then(|tail| tail.checked_add(window)).ok_or_else(|| ThoError::new(format!("BucketRunConfig::window_samples: sample span overflow for window={window} step={step} mel_config={mel_config:?}")))
            }
            BucketingConfig::Plain(plain_config) => {
                if window == 0 {
                    return Err(ThoError::new(format!("BucketRunConfig::window_samples: invalid window={window} plain_config={plain_config:?}")));
                }
                Ok(window)
            }
        }
    }

    pub fn new(id: Option<RunConfigId>, bucket_count: u32, min_slices: u32, max_slices: u32, mel_config: BucketingConfig) -> Self {
        Self { id, bucket_count, min_slices, max_slices, bucket_config: mel_config }
    }
}

fn bucket_step_samples_for_config(run_config: &BucketRunConfig, step: u32) -> Result<i64, ThoError> {
    let step = i64::from(step);
    match run_config.mel_config() {
        BucketingConfig::Plain(_) => Ok(step),
        BucketingConfig::Mel(mel_config) => {
            if mel_config.mel_patch_step <= 0 {
                return Err(ThoError::new(format!("bucket_step_samples_for_config: invalid mel_patch_step={}", mel_config.mel_patch_step)));
            }
            step.checked_mul(mel_config.mel_patch_step).ok_or_else(|| ThoError::new(format!("bucket_step_samples_for_config: sample step overflow for step={step} mel_patch_step={}", mel_config.mel_patch_step)))
        }
    }
}

fn bucket_slice_count(samples: i64, bucket_window_samples: i64, bucket_step_samples: i64) -> Result<i64, ThoError> {
    if bucket_window_samples <= 0 || bucket_step_samples <= 0 {
        return Err(ThoError::new(format!("bucket_slice_count: invalid bucket_window_samples={bucket_window_samples} bucket_step_samples={bucket_step_samples}")));
    }
    if samples < bucket_window_samples {
        return Ok(0);
    }
    Ok((samples - bucket_window_samples) / bucket_step_samples + 1)
}

fn bucket_selector_sample_range(selector: &SelectBucketsInputSelector, samples: i64, bucket_window_samples: i64, bucket_step_samples: i64) -> Result<Range<i64>, ThoError> {
    if selector.bucket_slice_range.start < 0 || selector.bucket_slice_range.end <= selector.bucket_slice_range.start {
        return Err(ThoError::new(format!("bucket_selector_sample_range: invalid selector={selector:?}")));
    }
    if bucket_window_samples <= 0 || bucket_step_samples <= 0 {
        return Err(ThoError::new(format!("bucket_selector_sample_range: invalid bucket_window_samples={bucket_window_samples} bucket_step_samples={bucket_step_samples}")));
    }
    let sample_start = selector.bucket_slice_range.start.checked_mul(bucket_step_samples).ok_or_else(|| ThoError::new(format!("bucket_selector_sample_range: sample_start overflow selector={selector:?} bucket_step_samples={bucket_step_samples}")))?;
    let last_bucket_slice = selector.bucket_slice_range.end.checked_sub(1).ok_or_else(|| ThoError::new(format!("bucket_selector_sample_range: selector end overflow selector={selector:?}")))?;
    let sample_end = last_bucket_slice.checked_mul(bucket_step_samples).and_then(|v| v.checked_add(bucket_window_samples)).ok_or_else(|| ThoError::new(format!("bucket_selector_sample_range: sample_end overflow selector={selector:?} bucket_window_samples={bucket_window_samples} bucket_step_samples={bucket_step_samples}")))?;
    let sample_end = sample_end.min(samples);
    if sample_start >= sample_end {
        return Err(ThoError::new(format!("bucket_selector_sample_range: selected sample range is empty selector={selector:?} samples={samples} bucket_window_samples={bucket_window_samples} bucket_step_samples={bucket_step_samples}")));
    }
    Ok(sample_start..sample_end)
}

impl Default for BucketRunConfig {
    fn default() -> Self {
        Self {
            id: None,
            bucket_count: 256,
            min_slices: 10,
            max_slices: 100,
            bucket_config: BucketingConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fs;
    use std::ops::Range;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};
    use futures_util::{StreamExt, TryStreamExt};
    use log::debug;
    use parking_lot::{RawRwLock, RwLock};
    use rand::random_range;
    use tch::{Device, Tensor};
    use tch::Device::Cuda;
    use thoosa_common::dims::MotifSlices;
    use crate::tasks::find_motifs_run_config::{AccumulateStats, Aggr, FindMotifsRunConfig, MaxAggr, Metrics, NormThicknessRunConfig, SlicesFilter};
    use thoosa_common::parzen::optimizer::{AsyncTPEOptimizer, ConfigurationSpace};
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
    use thoosa_common::train_nets::OptConfig;
    use thoosa_common::ws::{FsWorkspace, Workspace};
    use thoosa_motifs::distance::{clc_d2_similarity, EucDistConfig, SimilarityMetric};
    use crate::cli::{plot_resource, run_opt};
    use thoosa_motifs::fragments::Resolution;
    use thoosa_properties_map::ToHashmap;
    use super::{bucket_selector_sample_range, bucket_slice_count, bucket_step_samples_for_config, filter_mel_patch_groups, BucketRunConfig, BucketRunConfigRangeOrValue, BucketingConfig, BucketingConfigRangeOrValue, BucketingStage, LogMelPatchConfig, LogMelPatchConfigRangeOrValue, BucketingInput, MelPatchTempStore, SelectBucketsInputSelector};
    use thoosa_run_config_args::GridRunConfigSource;
    use thoosa_shared::{cdt_for_dir, mono_key, RangeOrValue, ResourceType, TensorIdSliceIdx, ThoError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::{Resource, ResourcesConfig};
    use thoosa_shared::run_configs::{RunConfig, RunConfigId};
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use thoosa_store::{IdSlice, SqlStore, Store};
    use thoosa_store::rusqlite::Transaction;
    use thoosa_with_data::with_data;
    use crate::cli::{import_resources, plot_motif};
    use crate::RawTensorLoader;
    use crate::tasks::find_and_select_motifs::{FindAndSelectMotifsInput, FindAndSelectMotifsRunConfig, FindAndSelectMotifsStage};
    use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
    use crate::tasks::motif_stats::{MotifStats, MotifStatsInput, MotifStatsOutput};
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
        let mut store = MelPatchTempStore::new(&dir, 7, 3)?;

        store.save_slices(1, &[slice(10, 100), slice(11, 110)])?;
        store.save_slices(2, &[slice(12, 120)])?;
        store.save_slices(1, &[slice(13, 130)])?;

        let groups = store.load_channel_groups(3)?;
        assert_eq!(groups.len(), 2);
        assert_eq!(groups.get(&1).expect("missing channel 1").len(), 3);
        assert_eq!(groups.get(&2).expect("missing channel 2").len(), 1);

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
        let rc = BucketRunConfigRangeOrValue {
            id: None,
            bucket_count: 256u32.into(),
            min_slices: 10u32.into(),
            max_slices: 100u32.into(),
            bucket_config: BucketingConfigRangeOrValue::Mel(LogMelPatchConfigRangeOrValue {
                n_mels: default_log_mel_patch_config.n_mels.into(),
                f_min: default_log_mel_patch_config.f_min.into(),
                f_max: default_log_mel_patch_config.f_max.into(),
                mel_patch_size: default_log_mel_patch_config.mel_patch_size.into(),
                mel_patch_step: default_log_mel_patch_config.mel_patch_step.into(),
            }),
        };

        let grid = rc.grid_run_configs().expect("BucketRunConfigRangeOrValue::grid_run_configs should not fail");
        let run_configs: Vec<BucketRunConfig> = grid.to_run_configs();

        assert_eq!(1, run_configs.len());
        assert_eq!(256, run_configs[0].bucket_count);
        assert_eq!(10, run_configs[0].min_slices);
        assert_eq!(100, run_configs[0].max_slices);
    }

    #[test]
    fn bucket_selector_sample_range_uses_bucket_slice_units() -> Result<(), ThoError> {
        let selector = SelectBucketsInputSelector::from(2..5);
        assert_eq!(bucket_selector_sample_range(&selector, 100, 10, 4)?, 8..26);
        assert_eq!(bucket_slice_count(26, 10, 4)?, 5);
        Ok(())
    }

    #[test]
    fn bucket_step_samples_for_config_uses_mel_patch_stride() -> Result<(), ThoError> {
        let run_config = BucketRunConfig::new(None, 16, 1, 100, BucketingConfig::Mel(LogMelPatchConfig { mel_patch_step: 3, ..LogMelPatchConfig::default() }));
        assert_eq!(bucket_step_samples_for_config(&run_config, 4)?, 12);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_run_mel_path_1_channel() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        fs::create_dir_all(&test_dir)?;
        debug!("test_dir={test_dir:?}");
        // let store = Arc::new(SqlStore::in_mem().await?);
        let store = Arc::new(SqlStore::from_path(test_dir.join(&"main_db.db3")).await?);

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir,
            1,
            u32::MAX,
            32000,
        )?));
        let motif_len = 64usize;

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
                let num_motifs = 4; //random_range(5usize..9usize);

                let motif_gap = 3*motif_len;
                let resource_len = num_motifs * motif_len + (num_motifs - 1) * motif_gap;

                if key.starts_with("t_0") {

                    let (resource_t, _resource_intervals) = tch_test_utils::resource_tensor(resource_len, &[
                        vec![(num_motifs, motif_len, MotifKind::ChirpUp)],
                    ], motif_gap, 10.0, false, true)?;//TODO set motif_from_zero to false and z-normalize

                    Ok(resource_t)
                } else if key.starts_with("t_1") {

                    let (resource_t, _resource_intervals) = tch_test_utils::resource_tensor(resource_len, &[
                        vec![(num_motifs, motif_len, MotifKind::Trill)],
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
        let window = 64;
        let step = 4;

        let mut patch_run_config = BucketRunConfig {
            id: None,

            bucket_count: 16,
            min_slices: 2,
            max_slices: 100,

            bucket_config: BucketingConfig::Mel(LogMelPatchConfig {
                n_mels: 32,
                mel_patch_size: 18,
                mel_patch_step: 32,
                ..LogMelPatchConfig::default()
            })
        };

        let find_motifs_run_config_id = store.save_run_config(
            None,
            &FindMotifsRunConfig::type_name(),
            "FindMotifsRunConfig",
            FindMotifsRunConfig::new(None, step, window, None, SimilarityMetric::EucDist(EucDistConfig{ z_normalize: true }), SlicesFilter::Accumulate(AccumulateStats{ aggr: Aggr::Max(MaxAggr{ k: 0.0 }) }), None).to_properties_iter(None).collect(),
        ).await?;


        let id = store.save_run_config(
            Some(&[find_motifs_run_config_id]),
            &BucketRunConfig::type_name(),
            "BucketRunConfig",
            patch_run_config.to_properties_iter(None).collect(),
        ).await?;

        patch_run_config.id = Some(id);

        let mel_patch_stage = Arc::new(BucketingStage::new(&ws, &store, Arc::clone(&loader), patch_run_config, window, step)?);

        let res = mel_patch_stage.run(Arc::new(DevicesWrap::wrap(BucketingInput::new(Simple::new(ResourceType::Group("all group".into()))))), ctx.branch(false)).await?;

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

        let avg_norm_thickness = futures::stream::iter(motifs.iter().map(|motif_id| {
            let motif_stats = Arc::clone(&motif_stats);
            let mut motif_load_ctx = ctx.branch(false);
            async move {
                let norm_thickness = Metrics::NormThickness(NormThicknessRunConfig::default());
                let metrics: Arc<DevicesWrap<MotifStatsOutput>> = Arc::clone(&motif_stats).run(Arc::new(DevicesWrap::wrap(MotifStatsInput::from_motif_input(MotifLoaderInput::new(mono_key!(*motif_id, "").into(), None), vec![
                    norm_thickness.clone()
                ]))), motif_load_ctx.branch(false)).await?;

                let norm_thickness = with_data!(motif_load_ctx.branch(false), metrics; |ctx, metrics:&MotifStatsOutput|{
                    Ok::<_, ThoError>(metrics.channel_metric_value(&norm_thickness, 0)?.value())
                })?;

                debug!("heavy_run_mel_path: norm_thickness={norm_thickness:?}");
                Ok::<_, ThoError>(norm_thickness)
            }
        })).buffer_unordered(10).try_fold(0.0_f32, |acc, value| async move {
            Ok(acc + value)
        }).await?/motifs.len() as f32;

        debug!("heavy_run_mel_path: avg_norm_thickness={avg_norm_thickness:?}");
        let run_config: crate::tasks::bucketing::BucketRunConfigRangeOrValue<crate::tasks::bucketing::BucketRunConfigFields> = crate::tasks::bucketing::BucketRunConfigRangeOrValue {
            id: None,
            bucket_count: RangeOrValue::Range { range: Range {start:64, end:512}, step: None },
            min_slices: RangeOrValue::Value(2),
            max_slices: RangeOrValue::Value(100),
            bucket_config: BucketingConfigRangeOrValue::Mel(LogMelPatchConfigRangeOrValue {
                n_mels: RangeOrValue::Range { range: Range {start:64, end:256}, step: None },
                f_min: RangeOrValue::Value(20.0),
                f_max: RangeOrValue::Value(16000.0),
                mel_patch_size: RangeOrValue::Range { range: Range {start:64, end:128}, step: None },
                mel_patch_step: RangeOrValue::Range { range: Range {start:4, end:32}, step: None },
            }),
        };

        let find_and_select_input = Arc::new(DevicesWrap::wrap(BucketingInput::new(
            Simple::new(ResourceType::Group("all group".into())), //run_all_args.find_resources.clone(),
        )));

       /*plot_test_resources(&store, &ws, &loader, &mut ctx).await?;

        let all_motifs = [MotifKind::ChirpUp, MotifKind::ChirpDown, MotifKind::Trill, MotifKind::HarmonicStack].as_slice().into_iter().map(move |motif|(motif.clone(), Tensor::from_slice(gen_motif(motif_len, &motif).as_slice()).view((1, 1, -1)).to_device(Cuda(0)))).collect::<Vec<(MotifKind, Tensor)>>();
        let mut best_reference_distances = vec![f64::INFINITY; all_motifs.len()];

        for motif_id in &motifs {
            let motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>> = Arc::clone(&motif_loader_impl).run(Arc::new(DevicesWrap::wrap(MotifLoaderInput::new(mono_key!(*motif_id, "").into(), None))), ctx.branch(false)).await?;
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



            plot_motif(
                &ws,
                ctx.branch(false),
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
*/
        Ok(())
    }

    async fn plot_test_resources(store: &Arc<Box<dyn Store>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, loader: &Arc<RawTensorLoader<RunCtxImpl>>, mut ctx: &mut RunCtxImpl) -> Result<(), ThoError> {
        for resource in Simple::new(ResourceType::Set("all_label_0".into())).to_resources(&store).await? {
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
                ctx.branch(false),
                "0_"
            ).await?
        }

        for resource in Simple::new(ResourceType::Set("all_label_1".into())).to_resources(&store).await? {
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
                ctx.branch(false),
                "1_"
            ).await?
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn heavy_run_mel_path_3_channel() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

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
        let window = 32;
        let step = 4;

        let mut patch_run_config = BucketRunConfig {
            id: None,

            bucket_count: 16,
            min_slices: 10,
            max_slices: 100,

            bucket_config: BucketingConfig::Mel(LogMelPatchConfig {
                n_mels: 16,
                mel_patch_size: 9,
                mel_patch_step: 32,
                ..LogMelPatchConfig::default()
            })
        };

        let find_motifs_run_config_id = store.save_run_config(
            None,
            &FindMotifsRunConfig::type_name(),
            "FindMotifsRunConfig",
            FindMotifsRunConfig::new(None, step, window, None, SimilarityMetric::EucDist(EucDistConfig{ z_normalize: true }), SlicesFilter::Accumulate(AccumulateStats{ aggr: Aggr::Max(MaxAggr{ k: 0.0 }) }), None).to_properties_iter(None).collect(),
        ).await?;


        let id = store.save_run_config(
            Some(&[find_motifs_run_config_id]),
            &BucketRunConfig::type_name(),
            "BucketRunConfig",
            patch_run_config.to_properties_iter(None).collect(),
        ).await?;

        patch_run_config.id = Some(id);

        let mel_patch_stage = Arc::new(BucketingStage::new(&ws, &store, Arc::clone(&loader), patch_run_config, window, step)?);

        let res = mel_patch_stage.run(Arc::new(DevicesWrap::wrap(BucketingInput::new(Simple::new(ResourceType::Group("all group".into()))))), ctx.branch(false)).await?;

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
                ctx.branch(false),
                ""
            ).await?

        }

        let all_motifs = [MotifKind::ChirpUp, MotifKind::ChirpDown, MotifKind::Trill, MotifKind::HarmonicStack].as_slice().into_iter().map(move |motif|(motif.clone(), Tensor::from_slice(gen_motif(64, &motif).as_slice()).view((1, 1, -1)).to_device(Cuda(0)))).collect::<Vec<(MotifKind, Tensor)>>();
        let mut best_reference_distances = vec![f64::INFINITY; all_motifs.len()];

        for motif_id in &motifs {
            let motif: Arc<DevicesWrap<DataWithExtra<ThoTensor<MotifSlices, SynKey>, MotifMeta>>> = Arc::clone(&motif_loader_impl).run(Arc::new(DevicesWrap::wrap(MotifLoaderInput::new(mono_key!(*motif_id, "").into(), None))), ctx.branch(false)).await?;
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
            ]))), ctx.branch(false)).await?;

            debug!("heavy_run_mel_path: metrics={metrics:?}");

            plot_motif(
                &ws,
                ctx.branch(false),
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

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize)]
pub struct LogMelPatchConfig {
    /// Number of mel bands, e.g. 64 or 128.
    #[run_config]
    #[in_map]
    pub n_mels: i64,

    /// Lowest mel frequency.
    #[run_config]
    #[in_map]
    pub f_min: f64,

    /// Highest mel frequency. If <= 0.0, sample_rate / 2.0 is used.
    #[run_config]
    #[in_map]
    pub f_max: f64,

    /// Number of STFT frames per output patch.
    #[run_config]
    #[in_map]
    pub mel_patch_size: i64,

    /// Step between patches in STFT frames.
    #[run_config]
    #[in_map]
    pub mel_patch_step: i64,


}

impl Default for LogMelPatchConfig {
    fn default() -> Self {
        Self {
            n_mels: 128,
            f_min: 20.0,
            f_max: 16_000.0,
            mel_patch_size: 64,
            mel_patch_step: 32,
        }
    }
}
