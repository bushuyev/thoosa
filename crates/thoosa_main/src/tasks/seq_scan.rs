use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::cross_loader::ResourcesConfigInput;
use crate::tasks::find_motifs::{FilteredLoaderStage, LoaderStageChoice};
use crate::tasks::stages_seq::{ABStage, StageChainOption};
use crate::tasks::transforms::freq_filter::FreqFilter;
use log::{debug, info};
use parking_lot::RwLock;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use tch::kind::Kind;
use tch::Device::Cpu;
use tch::Tensor;
use thoosa_common::find_motifs_run_config::InputFilter;
use thoosa_common::loader::LoadRunner;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::{OutId, RunCtx};
use thoosa_common::seq_scan_run_config::{MeanAbsSeqScanMetricConfig, MeanPowerSeqScanMetricConfig, SeqScanMetric, SeqScanRunConfig};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::mono_key;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_shared::traits::WithId;
use thoosa_shared::ThoError;
use thoosa_store::Store;
use thoosa_with_data::with_data;
use thoosa_common::runs::SavableResult;
use async_trait::async_trait;

#[derive(Debug, Clone, DummyData)]
#[dummy_data(clone)]
pub struct SeqScanOutput {}

impl SeqScanOutput {
    pub fn new() -> Self {
        Self {}
    }
}

impl Output for SeqScanOutput {}

#[async_trait]
impl SavableResult for SeqScanOutput {
    async fn save(
        &mut self,
        _store: &Arc<Box<dyn Store>>,
        _ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoError> {
        Ok(0)
    }
}

impl WithId for SeqScanOutput {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

#[derive(Debug, Clone)]
struct SeqScanBucketEntry {
    resource: Resource,
    slice_idx: usize,
    slice_start: u32,
    score: f32,
}

#[derive(Debug)]
struct SeqScanBuckets {
    buckets: Vec<Vec<SeqScanBucketEntry>>,
}

impl SeqScanBuckets {
    fn new(buckets_count: u32) -> Result<Self, ThoError> {
        if buckets_count == 0 {
            return Err(ThoError::new("SeqScanBuckets::new: buckets_count must be greater than zero"));
        }

        Ok(Self { buckets: vec![Vec::new(); buckets_count as usize] })
    }

    fn push(&mut self, bucket_idx: usize, entry: SeqScanBucketEntry) -> Result<(), ThoError> {
        let bucket = self.buckets.get_mut(bucket_idx).ok_or_else(|| ThoError::new(format!("SeqScanBuckets::push: invalid bucket_idx={bucket_idx}")))?;
        bucket.push(entry);
        Ok(())
    }

    fn counts(&self) -> Vec<usize> {
        self.buckets.iter().map(Vec::len).collect()
    }
}

#[derive(Debug)]
struct SeqScanConfiguredStage<L, C>
where
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    run_config: Arc<SeqScanRunConfig>,
    configured_loader: Arc<LoaderStageChoice<L, C>>,
}

const DEFAULT_SEQ_SCAN_CHUNK_SIZE: usize = 20_000;

#[derive(Debug, Named)]
pub struct SeqScanStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    store: Arc<Box<dyn Store>>,
    run_config_stages: Vec<SeqScanConfiguredStage<L, C>>,
    chunk_size: usize,
    pd: PhantomData<(RC, C)>,
}

impl<RC, L, C> SeqScanStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<L>,
        run_configs: Vec<SeqScanRunConfig>,
    ) -> Result<Self, ThoError> {
        Self::new_with_chunk_size(ws, store, loader, run_configs, DEFAULT_SEQ_SCAN_CHUNK_SIZE)
    }

    pub fn new_with_chunk_size(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<L>,
        run_configs: Vec<SeqScanRunConfig>,
        chunk_size: usize,
    ) -> Result<Self, ThoError> {
        if chunk_size == 0 {
            return Err(ThoError::new("SeqScanStage::new_with_chunk_size: chunk_size must be greater than zero"));
        }

        let run_config_stages = run_configs
            .into_iter()
            .map(|run_config| {
                let configured_loader = config_to_stage(&loader, &run_config, ws)?;

                Ok::<_, ThoError>(SeqScanConfiguredStage {
                    run_config: Arc::new(run_config),
                    configured_loader,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            store: Arc::clone(store),
            run_config_stages,
            chunk_size,
            pd: PhantomData,
        })
    }
}

impl<RC, L, C> Stage for SeqScanStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L, C>: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + 'static,
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<RC>
    where
        Self: 'a;
    type Out = SeqScanOutput;
}

impl<RC, L, C> AsyncStage for SeqScanStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L, C>: for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C> + 'static,
    C: RunCtx + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "SeqScanStage::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let max_chunk_len = i64::try_from(self.chunk_size).unwrap_or(i64::MAX);
        let resources_config = with_data!(ctx.branch(Some("seq_scan_resources"), false), input; |_ctx, input: &ResourcesConfigInput<RC>| {
            Ok::<_, ThoError>(input.resources_config().clone())
        })?;
        let resources = resources_config.to_resources(&self.store).await?;

        for configured_stage in &self.run_config_stages {
            let mut buckets = SeqScanBuckets::new(configured_stage.run_config.buckets_count)?;
            let mut batch_resources: Vec<Resource> = Vec::new();
            let mut batch_tensors: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>> = Vec::new();
            let mut batch_len = 0i64;

            for resource in &resources {
                let loaded = Arc::clone(&configured_stage.configured_loader)
                    .run_cached(
                        Arc::new(DevicesWrap::wrap(resource.clone())),
                        ctx.branch(Some(&format!("seq_scan_load_resource_{}", resource.id())), false),
                    )
                    .await?;
                let resource_len = loaded.size().dim_1();

                if resource_len > max_chunk_len {
                    flush_batch_to_buckets(
                        &batch_resources,
                        batch_tensors.clone(),
                        &configured_stage.run_config,
                        &mut buckets,
                        ctx.branch(Some("seq_scan_bucket_accumulated_batch_before_large_resource"), false),
                    ).await?;
                    batch_resources.clear();
                    batch_tensors.clear();
                    batch_len = 0;

                    with_data!(ctx.branch(Some("seq_scan_bucket_large_resource"), false), loaded; |_ctx, tensor: &ThoTensor<ChannelsData2, SynKey>| {
                        assign_resource_to_buckets_chunked(resource, tensor.t_ref(), &configured_stage.run_config, max_chunk_len, &mut buckets)
                    })?;
                    continue;
                }

                batch_len += resource_len;
                batch_resources.push(resource.clone());
                batch_tensors.push(loaded);

                if batch_len < max_chunk_len {
                    continue;
                }

                flush_batch_to_buckets(
                    &batch_resources,
                    batch_tensors.clone(),
                    &configured_stage.run_config,
                    &mut buckets,
                    ctx.branch(Some("seq_scan_bucket_accumulated_batch"), false),
                ).await?;
                batch_resources.clear();
                batch_tensors.clear();
                batch_len = 0;
            }

            flush_batch_to_buckets(
                &batch_resources,
                batch_tensors,
                &configured_stage.run_config,
                &mut buckets,
                ctx.branch(Some("seq_scan_bucket_accumulated_batch_tail"), false),
            ).await?;

            info!(
                "SeqScanStage::run: run_config_id={:?} metric={:?} chunk_size={} bucket_sizes={:?}",
                configured_stage.run_config.id,
                configured_stage.run_config.metric,
                self.chunk_size,
                buckets.counts()
            );
        }

        Ok(Arc::new(DevicesWrap::wrap(SeqScanOutput::new())))
    }
}

async fn flush_batch_to_buckets<C>(
    batch_resources: &[Resource],
    batch_tensors: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>,
    run_config: &SeqScanRunConfig,
    buckets: &mut SeqScanBuckets,
    ctx: C,
) -> Result<(), ThoError>
where
    C: RunCtx,
{
    if batch_resources.is_empty() {
        return Ok(());
    }

    with_data!(ctx, [batch_tensors];|_ctx, tensors: &[&ThoTensor<ChannelsData2, SynKey>]| {
        for (resource, tensor) in batch_resources.iter().zip(tensors.iter()) {
            assign_resource_to_buckets(resource, tensor.t_ref(), run_config, buckets)?;
        }
        Ok::<_, ThoError>(())
    })?;

    Ok(())
}

fn assign_resource_to_buckets(
    resource: &Resource,
    resource_t: &Tensor,
    run_config: &SeqScanRunConfig,
    buckets: &mut SeqScanBuckets,
) -> Result<(), ThoError> {
    assign_resource_to_buckets_chunked(resource, resource_t, run_config, i64::MAX, buckets)
}

fn assign_resource_to_buckets_chunked(
    resource: &Resource,
    resource_t: &Tensor,
    run_config: &SeqScanRunConfig,
    max_chunk_len: i64,
    buckets: &mut SeqScanBuckets,
) -> Result<(), ThoError> {
    let chunk_scores = score_resource_in_chunks(resource, resource_t, run_config, max_chunk_len)?;

    if chunk_scores.is_empty() {
        return Ok(());
    }

    let min_score = chunk_scores
        .iter()
        .flat_map(|(_, _, score_values)| score_values.iter().copied())
        .fold(f32::INFINITY, f32::min);
    let max_score = chunk_scores
        .iter()
        .flat_map(|(_, _, score_values)| score_values.iter().copied())
        .fold(f32::NEG_INFINITY, f32::max);
    let buckets_count = usize::try_from(run_config.buckets_count).map_err(|_| ThoError::new(format!("assign_resource_to_buckets_chunked: invalid buckets_count={}", run_config.buckets_count)))?;

    for (first_slice_idx, first_slice_start, score_values) in chunk_scores {
        for (local_slice_idx, score) in score_values.into_iter().enumerate() {
            let bucket_idx = bucket_idx(score, min_score, max_score, buckets_count);
            let slice_start_offset = u32::try_from(local_slice_idx)
                .map_err(|_| ThoError::new(format!("assign_resource_to_buckets_chunked: local_slice_idx={local_slice_idx} does not fit into u32")))?
                .saturating_mul(run_config.step);
            buckets.push(
                bucket_idx,
                SeqScanBucketEntry {
                    resource: resource.clone(),
                    slice_idx: first_slice_idx + local_slice_idx,
                    slice_start: first_slice_start.saturating_add(slice_start_offset),
                    score,
                },
            )?;
        }
    }

    Ok(())
}

fn score_resource_in_chunks(
    resource: &Resource,
    resource_t: &Tensor,
    run_config: &SeqScanRunConfig,
    max_chunk_len: i64,
) -> Result<Vec<(usize, u32, Vec<f32>)>, ThoError> {
    let size = resource_t.size();
    let resource_len = *size.get(1).ok_or_else(|| ThoError::new(format!("score_resource_in_chunks: expected [channels, seq], got size={size:?}")))?;
    let window = i64::from(run_config.window);
    let step = i64::from(run_config.step);

    if step <= 0 {
        return Err(ThoError::new("score_resource_in_chunks: step must be greater than zero"));
    }

    if max_chunk_len <= 0 {
        return Err(ThoError::new(format!("score_resource_in_chunks: max_chunk_len must be greater than zero, got {max_chunk_len}")));
    }

    if resource_len < window {
        debug!("score_resource_in_chunks: skipping resource={} because resource_len={resource_len} < window={window}", resource.id());
        return Ok(Vec::new());
    }

    let effective_chunk_len = max_chunk_len.max(window);
    let last_slice_start = resource_len - window;
    let mut next_slice_start = 0i64;
    let mut first_slice_idx = 0usize;
    let mut chunk_scores = Vec::new();

    while next_slice_start <= last_slice_start {
        let max_start_in_chunk = next_slice_start
            .saturating_add(effective_chunk_len.saturating_sub(window))
            .min(last_slice_start);
        let last_slice_start_in_chunk = next_slice_start + ((max_start_in_chunk - next_slice_start) / step) * step;
        let chunk_end = last_slice_start_in_chunk.saturating_add(window);
        let chunk_t = resource_t.f_slice(1, next_slice_start, chunk_end, 1)?;
        let slices = chunk_t.f_unfold(1, window, step)?;
        let score_values = score_slices_to_vec(&run_config.metric, &slices)?;

        debug!(
            "score_resource_in_chunks: resource_id={} next_slice_start={} last_slice_start_in_chunk={} chunk_end={} chunk_scores={}",
            resource.id(),
            next_slice_start,
            last_slice_start_in_chunk,
            chunk_end,
            score_values.len()
        );

        if score_values.is_empty() {
            return Err(ThoError::new(format!(
                "score_resource_in_chunks: resource={} produced no scores for next_slice_start={next_slice_start} chunk_end={chunk_end}",
                resource.id()
            )));
        }

        let chunk_len = score_values.len();
        chunk_scores.push((
            first_slice_idx,
            u32::try_from(next_slice_start).map_err(|_| ThoError::new(format!("score_resource_in_chunks: next_slice_start={next_slice_start} does not fit into u32")))?,
            score_values,
        ));

        first_slice_idx += chunk_len;
        next_slice_start = last_slice_start_in_chunk.saturating_add(step);
    }

    Ok(chunk_scores)
}

fn bucket_idx(score: f32, min_score: f32, max_score: f32, buckets_count: usize) -> usize {
    if buckets_count <= 1 || (max_score - min_score).abs() <= f32::EPSILON {
        return 0;
    }

    let normalized = ((score - min_score) / (max_score - min_score)).clamp(0.0, 1.0);
    let bucket = (normalized * buckets_count as f32).floor() as usize;

    bucket.min(buckets_count - 1)
}

fn score_slices(metric: &SeqScanMetric, slices: &Tensor) -> Result<Tensor, ThoError> {
    match metric {
        SeqScanMetric::MeanAbs(config) => score_mean_abs(slices, config),
        SeqScanMetric::MeanPower(config) => score_mean_power(slices, config),
    }
}

fn score_slices_to_vec(metric: &SeqScanMetric, slices: &Tensor) -> Result<Vec<f32>, ThoError> {
    let scores = score_slices(metric, slices)?;
    let scores_cpu = scores.f_to_device(Cpu)?;
    let scores_flat = scores_cpu.f_reshape([-1])?;

    Ok(Vec::<f32>::try_from(&scores_flat)?)
}

fn score_mean_abs(slices: &Tensor, config: &MeanAbsSeqScanMetricConfig) -> Result<Tensor, ThoError> {
    let absolute = if config.center {
        let mean = slices.f_mean_dim([2].as_slice(), true, Kind::Float)?;
        slices.f_sub(&mean)?.f_abs()?
    } else {
        slices.f_abs()?
    };

    reduce_slice_scores(&absolute)
}

fn score_mean_power(slices: &Tensor, config: &MeanPowerSeqScanMetricConfig) -> Result<Tensor, ThoError> {
    let values = if config.center {
        let mean = slices.f_mean_dim([2].as_slice(), true, Kind::Float)?;
        slices.f_sub(&mean)?
    } else {
        slices.f_abs()?
    };
    let powered = values.f_abs()?.f_pow_tensor_scalar(config.power)?;

    reduce_slice_scores(&powered)
}

fn reduce_slice_scores(values: &Tensor) -> Result<Tensor, ThoError> {
    let per_channel = values.f_mean_dim([2].as_slice(), false, Kind::Float)?;
    let per_slice = per_channel.f_mean_dim([0].as_slice(), false, Kind::Float)?;

    Ok(per_slice)
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "seq_scan_config_to_stage"))]
pub fn config_to_stage<L, C>(loader: &Arc<L>, run_config: &SeqScanRunConfig, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<Arc<LoaderStageChoice<L, C>>, ThoError>
where
    C: 'static + RunCtx,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SynKey>, Ctx = C>,
{
    let configurable_loader: Arc<LoaderStageChoice<L, C>> = match run_config.freq_filter.as_deref() {
        None | Some([]) => Arc::new(StageChainOption::<L, FilteredLoaderStage<L, C>>::Single(Arc::clone(loader))),
        Some(freq_filter) => {
            let run_config_id = run_config.id.ok_or_else(|| ThoError::new("seq_scan::config_to_stage: freq_filter requires SeqScanRunConfig.id"))?;

            Arc::new(StageChainOption::<L, FilteredLoaderStage<L, C>>::Chained(Arc::new(ABStage::new(
                SynKey::new([mono_key!(run_config_id, "seq_scan_ab_stage")]),
                Arc::clone(loader),
                Arc::new(FreqFilter::<C>::new(freq_filter, ws, run_config_id)?),
            ))))
        }
    };

    Ok(configurable_loader)
}

#[cfg(test)]
mod tests {
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::seq_scan::{assign_resource_to_buckets, assign_resource_to_buckets_chunked, bucket_idx, SeqScanBuckets, SeqScanOutput, SeqScanStage};
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::Tensor;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::find_motifs_run_config::InputFilter;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runs::Simple;
    use thoosa_common::seq_scan_run_config::{MeanAbsSeqScanMetricConfig, MeanPowerSeqScanMetricConfig, SeqScanMetric, SeqScanRunConfig};
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::WithId;
    use thoosa_shared::ResourceType;
    use thoosa_shared::ThoError;
    use thoosa_store::MockStoreBuilder;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_bucket_idx_caps_last_bucket() {
        assert_eq!(0, bucket_idx(0.0, 0.0, 1.0, 4));
        assert_eq!(3, bucket_idx(1.0, 0.0, 1.0, 4));
        assert_eq!(2, bucket_idx(0.74, 0.0, 1.0, 4));
    }

    #[test]
    fn test_assign_resource_to_buckets() -> Result<(), ThoError> {
        let resource = Resource::new(1, 1, "r0", 6);
        let resource_t = Tensor::from_slice(&[1.0f32, 1.0, 2.0, 2.0, 4.0, 4.0]).view([1, 6]);
        let run_config = SeqScanRunConfig {
            id: Some(5),
            step: 2,
            window: 2,
            freq_filter: None,
            metric: SeqScanMetric::MeanAbs(MeanAbsSeqScanMetricConfig::default()),
            buckets_count: 3,
        };
        let mut buckets = SeqScanBuckets::new(run_config.buckets_count)?;

        assign_resource_to_buckets(&resource, &resource_t, &run_config, &mut buckets)?;

        assert_eq!(vec![1, 1, 1], buckets.counts());

        Ok(())
    }

    #[test]
    fn test_assign_resource_to_buckets_chunked_matches_full_scan() -> Result<(), ThoError> {
        let resource = Resource::new(1, 1, "r0", 10);
        let resource_t = Tensor::from_slice(&[1.0f32, 3.0, 2.0, 6.0, 4.0, 9.0, 5.0, 7.0, 8.0, 10.0]).view([1, 10]);
        let run_config = SeqScanRunConfig {
            id: Some(7),
            step: 2,
            window: 4,
            freq_filter: None,
            metric: SeqScanMetric::MeanPower(MeanPowerSeqScanMetricConfig { power: 2.0, center: false }),
            buckets_count: 3,
        };
        let mut full_scan_buckets = SeqScanBuckets::new(run_config.buckets_count)?;
        let mut chunked_buckets = SeqScanBuckets::new(run_config.buckets_count)?;

        assign_resource_to_buckets(&resource, &resource_t, &run_config, &mut full_scan_buckets)?;
        assign_resource_to_buckets_chunked(&resource, &resource_t, &run_config, 5, &mut chunked_buckets)?;

        let full_scan_entries = full_scan_buckets
            .buckets
            .iter()
            .map(|bucket| bucket.iter().map(|entry| (entry.slice_idx, entry.slice_start, entry.score)).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let chunked_entries = chunked_buckets
            .buckets
            .iter()
            .map(|bucket| bucket.iter().map(|entry| (entry.slice_idx, entry.slice_start, entry.score)).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        assert_eq!(full_scan_entries, chunked_entries);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_seq_scan_stage_runs() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let resources = vec![Resource::new(1, 1, "r0", 6), Resource::new(2, 1, "r1", 6)];
        let resources_group = ResourceType::List(resources.iter().map(|resource| resource.file_stem().to_string()).collect());
        let store = Arc::new(
            MockStoreBuilder::new()
                .with_list_resources_by_file_stems(
                    resources.iter().map(|resource| resource.file_stem().to_string()).collect(),
                    Ok(resources.clone()),
                )
                .build(),
        );
        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached([
                    (resources[0].clone(), Tensor::from_slice(&[1.0f32, 1.0, 2.0, 2.0, 4.0, 4.0]).view([1, 6])),
                    (resources[1].clone(), Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0, 3.0, 3.0]).view([1, 6])),
                ])
                .finish(),
        );
        let stage: Arc<SeqScanStage<Simple, _, _>> = Arc::new(SeqScanStage::new_with_chunk_size(
            &Arc::new(RwLock::new(mock_ws_simp(None, 1, 100, 1000))),
            &store,
            loader,
            vec![SeqScanRunConfig {
                id: Some(10),
                step: 2,
                window: 2,
                freq_filter: None::<Vec<InputFilter>>,
                metric: SeqScanMetric::MeanPower(MeanPowerSeqScanMetricConfig { power: 2.0, center: false }),
                buckets_count: 3,
            }],
            8,
        )?);

        let out = stage
            .run(
                Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(resources_group)).check(&store).await?)),
                mock_run_ctx::default(0),
            )
            .await?;

        assert_eq!(&(), out.as_ref().id());
        assert_eq!(&(), SeqScanOutput::new().id());

        Ok(())
    }
}
