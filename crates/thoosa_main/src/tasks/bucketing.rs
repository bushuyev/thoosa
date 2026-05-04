#![allow(non_camel_case_types)]

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
use tch::{Device, Kind, TchError, Tensor};
use serde::Deserialize;
use thoosa_common::audio::{assign_lsh_buckets_per_channel, build_mel_filterbank, fft_band_amplitudes_sliced, lsh_bucket_one_hot_3d, lsh_bucket_one_hot_3d_exact_buckets, make_lsh_planes_for_buckets, BandAggregation, LOG_EPS};
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


const MEL_PATCH_TEMP_MAGIC: [u8; 8] = *b"THOBUCK1";

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
    pub(crate) motif_ids: (RunId, Vec<u32>),
}

impl MelPatchOutput {
    fn new(run_ids_motif_ids: (RunId, Vec<u32>)) -> Self {
        Self { motif_ids: run_ids_motif_ids }
    }

    pub fn run_ids_motif_ids(&self) -> &(RunId, Vec<u32>) {
        &self.motif_ids
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
    fn new(path: &Path, run_id: RunId, bucket: u16) -> Result<Self, ThoError> {
        fs::create_dir_all(path)?;
        let file_path = path.join(format!("{run_id}_bucket_{bucket}_bucketing.bin"));
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
    D: Dims + 'static,
    Id: Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + 'static,
{
    ws: Arc<RwLock<Box<dyn Workspace>>>,
    store: Arc<Box<dyn Store>>,
    loader: Arc<L>,
    run_config: BucketRunConfig,
    window: i64,
    step: i64,
    pd: PhantomData<(RC, D, Id, C)>,
}

impl<RC, L, D, Id, C> BucketingStage<RC, L, D, Id, C>
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


        Ok(Self {
            ws: Arc::clone(ws),
            store: Arc::clone(store),
            loader,
            run_config,
            window: i64::from(window),
            step: i64::from(step),
            pd: PhantomData,
        })
    }

    fn save_resource_patches(
        &self,
        resource: &Resource,
        audio: &Tensor,
        _tmp_path: &Path,
        temp_stores: &mut [MelPatchTempStore],
        run_config: &BucketRunConfig,
        sample_rate: f64,
    ) -> Result<(), ThoError> {
        let mel_config = run_config.log_mel_config()?;
        let patches = audio_to_log_mel_patches(audio, mel_config, sample_rate, self.window, self.step)?;
        let patch_ranges = patch_sample_ranges_flat(&patches, mel_config, self.window, self.step)?;
        let channels_bucket_ids = assign_lsh_buckets_per_channel(&patches, run_config.bucket_count_i64())?;

        if patch_ranges.len() != channels_bucket_ids.len() {
            return Err(ThoError::new(format!("BucketingStage::save_resource_patches: ranges channels={} bucket channels={}", patch_ranges.len(), channels_bucket_ids.len())));
        }


        for channel in 0..channels_bucket_ids.len() {
            if patch_ranges[channel].len() != channels_bucket_ids[channel].len() {
                return Err(ThoError::new(format!("BucketingStage::save_resource_patches: channel={channel} ranges len={} bucket len={}", patch_ranges[channel].len(), channels_bucket_ids[channel].len())));
            }

            let mut slices_by_bucket: HashMap<u16, Vec<TensorIdSliceIdx>> = HashMap::new();

            for ((sample_start, _sample_end), bucket) in patch_ranges[channel].iter().zip(channels_bucket_ids[channel].iter()) {
                let group = u16::try_from(*bucket).map_err(|_| ThoError::new(format!("BucketingStage::save_resource_patches: bucket={bucket} does not fit into u16")))?;
                let slice_start = u32::try_from(*sample_start).map_err(|_| ThoError::new(format!("BucketingStage::save_resource_patches: sample_start={sample_start} does not fit into u32")))?;
                slices_by_bucket.entry(group).or_default().push(TensorIdSliceIdx(resource.id(), slice_start));
            }

            for (group, slices) in slices_by_bucket {
                debug!("save_resource_patches: group={group} slices={slices:?}");
                temp_stores[group as usize].save_slices(channel as u32, &slices)?;
            }
        }

        Ok(())
    }

    fn save_resource_plain_buckets(
        &self,
        resource: &Resource,
        audio: &Tensor,
        temp_stores: &mut [MelPatchTempStore],
        run_config: &BucketRunConfig,
        plain_config: &PlainConfig,
        sample_rate: f64,
        window: i64,
        step: i64,
    ) -> Result<(), ThoError> {
        if temp_stores.len() != run_config.bucket_count as usize {
            return Err(ThoError::new(format!("BucketingStage::save_resource_plain_buckets: temp stores={} bucket_count={}", temp_stores.len(), run_config.bucket_count)));
        }

        let audio_size = audio.size();
        if audio_size.len() != 2 {
            return Err(ThoError::new(format!("BucketingStage::save_resource_plain_buckets: expected audio shape [channels, samples], got {audio_size:?}")));
        }
        if window <= 0 || step <= 0 {
            return Err(ThoError::new(format!("BucketingStage::save_resource_plain_buckets: invalid window={window} step={step} plain_config={plain_config:?}")));
        }
        if audio_size[1] < window {
            return Err(ThoError::new(format!("BucketingStage::save_resource_plain_buckets: audio too short samples={} window={window}", audio_size[1])));
        }

        let slices = audio.f_unfold(1, window, step)?;
        debug!("save_resource_plain_buckets: slices={slices:?}");

        let channels_slices_freqamp = fft_band_amplitudes_sliced(&slices, sample_rate, plain_config.band_hz, plain_config.aggregation.to_audio(), plain_config.apply_hann_window, plain_config.remove_dc_offset)?;
        debug!("save_resource_plain_buckets: channels_slices_freqamp={channels_slices_freqamp:?}");

        let planes = make_lsh_planes_for_buckets(channels_slices_freqamp.size()[2], run_config.bucket_count as i64, Kind::Float, channels_slices_freqamp.device())?;
        debug!("save_resource_plain_buckets: planes={planes:?}");

        let buckets_one_hot = lsh_bucket_one_hot_3d_exact_buckets(&channels_slices_freqamp, &planes, plain_config.normalize_lsh)?;
        debug!("save_resource_plain_buckets: buckets_one_hot={buckets_one_hot:?}");

        let channels_bucket_ids = bucket_ids_from_one_hot_3d(&buckets_one_hot)?;

        for (channel, bucket_ids) in channels_bucket_ids.iter().enumerate() {
            let mut slices_by_bucket: HashMap<u16, Vec<TensorIdSliceIdx>> = HashMap::new();
            for (slice_idx, bucket) in bucket_ids.iter().enumerate() {
                let group = u16::try_from(*bucket).map_err(|_| ThoError::new(format!("BucketingStage::save_resource_plain_buckets: bucket={bucket} does not fit into u16")))?;
                let sample_start = (slice_idx as i64).checked_mul(step).ok_or_else(|| ThoError::new("BucketingStage::save_resource_plain_buckets: sample_start overflow"))?;
                let slice_start = u32::try_from(sample_start).map_err(|_| ThoError::new(format!("BucketingStage::save_resource_plain_buckets: sample_start={sample_start} does not fit into u32")))?;
                slices_by_bucket.entry(group).or_default().push(TensorIdSliceIdx(resource.id(), slice_start));
            }

            for (group, slices) in slices_by_bucket {
                debug!("save_resource_plain_buckets: group={group} slices={slices:?}");
                temp_stores[group as usize].save_slices(channel as u32, &slices)?;
            }
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

impl<RC, L, D, Id, C> AsyncStage for BucketingStage<RC, L, D, Id, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<D, Id>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    D: Dims + Send + Sync + 'static,
    Id: Named + Debug + Clone + Send + Sync + Eq + Hash + 'static,
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "BucketingStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let resources_config = with_data!(ctx.branch(Some("mel_patch_resources"), false), input; |_ctx, input: &MelPatchResourcesInput<RC>| {
            Ok::<_, ThoError>(input.resources_config().clone())
        })?;
        let resources = resources_config.to_resources(&self.store).await?;

        debug!("BucketingStage::run: resources_config={resources_config:?} resources={resources:?}");

        let tmp_path = thoosa_shared::with_perf_log(
            || self.ws.try_read_for(LOCK_TIMEOUT).expect("BucketingStage::run: timed out waiting for workspace read lock").tmp_path().clone(),
            "BucketingStage::run/read_ws_tmp_path",
        );
        let (channels, sample_rate) = {
            let ws = self.ws.try_read_for(LOCK_TIMEOUT).expect("BucketingStage::run: timed out waiting for workspace read lock");
            (ws.channels(), ws.sample_rate())
        };

        let mut temp_stores_by_bucket: Vec<MelPatchTempStore> =
            (0..self.run_config.bucket_count).map(|bucket| {
                let bucket = u16::try_from(bucket).map_err(|_| ThoError::new(format!("BucketingStage::run: bucket={bucket} does not fit into u16")))?;
                MelPatchTempStore::new(&tmp_path, self.run_config.id.expect("no run id").clone(), bucket)
            })
                .collect::<Result<Vec<_>, ThoError>>()?;

        for resource in &resources {
            let loaded = Arc::clone(&self.loader)
                .run_cached(
                    Arc::new(DevicesWrap::wrap(resource.clone())),
                    ctx.branch(Some(&format!("BucketingStage_{}", resource.id())), false),
                )
                .await?;

            match with_data!(ctx.branch(Some(&format!("mel_patch_bucket_resource_{}", resource.id())), false), loaded; |_ctx, tensor: &ThoTensor<D, Id>| {

                    match self.run_config.mel_config() {
                        BucketingConfig::Mel(_) => self.save_resource_patches(resource, tensor.t_ref(), &tmp_path, temp_stores_by_bucket.as_mut_slice(), &self.run_config, sample_rate as f64)?,
                        BucketingConfig::Plain(plain_config) => self.save_resource_plain_buckets(resource, tensor.t_ref(), temp_stores_by_bucket.as_mut_slice(), &self.run_config, plain_config, sample_rate as f64, self.window, self.step)?,
                    }

                Ok::<_, ThoError>(())
            }) {
                Ok(_) => {}
                Err(e) => {
                    warn!("BucketingStage::run e={e}");
                }
            };
        }


        // self.store.up
        let run_id = self.run_config.run_id()?;
        let mut motif_ids = Vec::new();
        let mut groups_by_channel = (0..channels).map(|_| HashMap::<u16, HashSet<IdSlice>>::new()).collect::<Vec<_>>();
        for (bucket, mut temp_store) in temp_stores_by_bucket.into_iter().enumerate() {
            let group = u16::try_from(bucket).map_err(|_| ThoError::new(format!("BucketingStage::run: bucket={bucket} does not fit into u16")))?;
            for (channel, slices) in temp_store.load_channel_groups(group)? {
                let channel = usize::try_from(channel).map_err(|_| ThoError::new(format!("BucketingStage::run: channel={channel} does not fit into usize")))?;
                if channel >= groups_by_channel.len() {
                    return Err(ThoError::new(format!("BucketingStage::run: temp store channel={channel} exceeds workspace channels={channels}")));
                }
                groups_by_channel[channel].entry(group).or_default().extend(slices);
            }
            temp_store.remove()?;
        }

        for (channel, groups) in groups_by_channel.into_iter().enumerate() {
            let groups = filter_mel_patch_groups(groups, self.run_config.min_slices_usize(), self.run_config.max_slices_usize());
            let group_motif_ids = self.store.save_groups_as_motifs(run_id, channel as i64, groups).await?;
            motif_ids.extend(group_motif_ids);
        }

        debug!("BucketingStage::run: run_id={run_id} resources={} motif_ids={motif_ids:?}", resources.len());


        info!("BucketingStage::run: run_id={run_id:?} done");

        Ok(Arc::new(DevicesWrap::wrap(MelPatchOutput::new((run_id, motif_ids)))))
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
    bucket_config: BucketingConfig
}

impl BucketRunConfig {

    pub fn run_id(&self) -> Result<RunId, ThoError> {
        self.id.ok_or_else(|| ThoError::new("BucketRunConfig::run_id: missing run id"))
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

    pub fn log_mel_config(&self) -> Result<&LogMelPatchConfig, ThoError> {
        match &self.bucket_config {
            BucketingConfig::Mel(mel_config) => Ok(mel_config),
            BucketingConfig::Plain(_) => Err(ThoError::new("BucketRunConfig::log_mel_config: bucketing config is Plain")),
        }
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

    pub fn new(id: Option<RunId>, bucket_count: u32, min_slices: u32, max_slices: u32, mel_config: BucketingConfig) -> Self {
        Self { id, bucket_count, min_slices, max_slices, bucket_config: mel_config }
    }
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
    use super::{filter_mel_patch_groups, BucketRunConfig, BucketRunConfigRangeOrValue, BucketingConfig, BucketingConfigRangeOrValue, BucketingStage, LogMelPatchConfig, LogMelPatchConfigRangeOrValue, MelPatchResourcesInput, MelPatchTempStore};
    use thoosa_run_config_args::GridRunConfigSource;
    use thoosa_shared::{cdt_for_dir, mono_key, RangeOrValue, ResourceType, TensorIdSliceIdx, ThoError};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::resource::{Resource, ResourcesConfig};
    use thoosa_shared::run_configs::{RunConfig, RunId};
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use thoosa_store::{IdSlice, SqlStore, Store};
    use thoosa_store::rusqlite::Transaction;
    use thoosa_with_data::with_data;
    use crate::cli::{import_resources, plot_motif};
    use crate::RawTensorLoader;
    use crate::tasks::find_and_select_motifs::{FindAndSelectMotifsInput, FindAndSelectMotifsRunConfig, FindAndSelectMotifsStage};
    use crate::tasks::bucketing_opt::BucketingOptStage;
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

        let find_motifs_run_id = store.save_run_config(
            None,
            &FindMotifsRunConfig::type_name(),
            "FindMotifsRunConfig",
            FindMotifsRunConfig::new(None, step, window, None, SimilarityMetric::EucDist(EucDistConfig{ z_normalize: true }), SlicesFilter::Accumulate(AccumulateStats{ aggr: Aggr::Max(MaxAggr{ k: 0.0 }) }), None).to_properties_iter(None).collect(),
        ).await?;


        let id = store.save_run_config(
            Some(&[find_motifs_run_id]),
            &BucketRunConfig::type_name(),
            "BucketRunConfig",
            patch_run_config.to_properties_iter(None).collect(),
        ).await?;

        patch_run_config.id = Some(id);

        let mel_patch_stage = Arc::new(BucketingStage::new(&ws, &store, Arc::clone(&loader), patch_run_config, window, step)?);

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

        let avg_norm_thickness = futures::stream::iter(motifs.iter().map(|motif_id| {
            let motif_stats = Arc::clone(&motif_stats);
            let mut motif_load_ctx = ctx.branch("motif_load".into(), false);
            async move {
                let norm_thickness = Metrics::NormThickness(NormThicknessRunConfig::default());
                let metrics: Arc<DevicesWrap<MotifStatsOutput>> = Arc::clone(&motif_stats).run(Arc::new(DevicesWrap::wrap(MotifStatsInput::from_motif_input(MotifLoaderInput::new(mono_key!(*motif_id, "").into(), None), vec![
                    norm_thickness.clone()
                ]))), motif_load_ctx.branch("motif_load".into(), false)).await?;

                let norm_thickness = with_data!(motif_load_ctx.branch("metrics".into(), false), metrics; |ctx, metrics:&MotifStatsOutput|{
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

        let find_and_select_input = Arc::new(DevicesWrap::wrap(MelPatchResourcesInput::new(
            Simple::new(ResourceType::Group("all group".into())), //run_all_args.find_resources.clone(),
        )));

         let run_id = run_opt::</*Simple,*/ _, _, BucketRunConfig, _, _>(
            None,
            run_config,
            find_and_select_input,
            &ws,
            &store,
            async |run_configs| {
                Ok(Arc::new(BucketingOptStage::new(loader, &ws, &store, run_configs, Arc::new(OptConfig::new(1000, 30)))))
            },
            || ctx.branch(Some("opt mel patch"), false),
        ).await?;


        /*plot_test_resources(&store, &ws, &loader, &mut ctx).await?;

        let all_motifs = [MotifKind::ChirpUp, MotifKind::ChirpDown, MotifKind::Trill, MotifKind::HarmonicStack].as_slice().into_iter().map(move |motif|(motif.clone(), Tensor::from_slice(gen_motif(motif_len, &motif).as_slice()).view((1, 1, -1)).to_device(Cuda(0)))).collect::<Vec<(MotifKind, Tensor)>>();
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
                ctx.branch("plot".into(), false),
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
                ctx.branch("plot".into(), false),
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

        let find_motifs_run_id = store.save_run_config(
            None,
            &FindMotifsRunConfig::type_name(),
            "FindMotifsRunConfig",
            FindMotifsRunConfig::new(None, step, window, None, SimilarityMetric::EucDist(EucDistConfig{ z_normalize: true }), SlicesFilter::Accumulate(AccumulateStats{ aggr: Aggr::Max(MaxAggr{ k: 0.0 }) }), None).to_properties_iter(None).collect(),
        ).await?;


        let id = store.save_run_config(
            Some(&[find_motifs_run_id]),
            &BucketRunConfig::type_name(),
            "BucketRunConfig",
            patch_run_config.to_properties_iter(None).collect(),
        ).await?;

        patch_run_config.id = Some(id);

        let mel_patch_stage = Arc::new(BucketingStage::new(&ws, &store, Arc::clone(&loader), patch_run_config, window, step)?);

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
                ctx.branch("plot".into(), false),
                ""
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
