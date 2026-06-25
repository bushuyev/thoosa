use std::collections::HashMap;
use crate::tasks::bucketing::{BucketingInput, BucketingStage};
use crate::tasks::cross_batch_clc::{CrossBatchFn, CrossInput, CrossUnfoldClc, CrossUnfoldClcOutput, UnfoldParams};
use crate::tasks::cross_loader::{
    CrossLoader, CrossUnfoldClcOutputMergeMode, ResourcesConfigInput,
};
use crate::tasks::dists_tracker::{DistsMask, DistsMaskType, DistsTracker};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use crate::tasks::tensor_loader::RawTensorLoader;
use async_trait::async_trait;
use thoosa_common::runner::common::{OutId, RunCtx};

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use futures_util::FutureExt;
use log::{debug, error, info, trace, warn};
use parking_lot::{Mutex, RwLock};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use tch::Kind::{Bool, Float};
use tch::{Device, Kind, Tensor};
use tch::Device::Cpu;
use tch::kind::FLOAT_CPU;
use tokio::runtime::Handle;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::copy_ctx::CopyCtx;
use crate::tasks::find_motifs_run_config::{Aggr as FilterAggr, FindMotifsRunConfig, FinderKind, MaxSlices, SlicesFilter};
use thoosa_common::runs::Simple;
use thoosa_common::ws::Workspace;
use thoosa_motifs::distance::MaskMaker::{Le, LeAccumulatedMean, LeAccumulatedQuantile, LocalMin, Tril};
use thoosa_motifs::find::{FindMotifsSavableResult, FixedFindMotifsConfig};
use thoosa_shared::{collect_futures_with_progress, mono_key, perf_log, sync_read_lock, FuturesOrdering, ThoError};
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_store::{SqlStore, Store};
use tokio::task::{JoinError, JoinSet};
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::cuda_utils::pack_bits_bool;
use thoosa_common::dims::{ChannelsData, MotifSlices};
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::data_with_extra::DataWithExtra;
use crate::tasks::motif_loader::{MotifLoader, MotifMeta};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_named_derive::Named;
use crate::tasks::cross_dims::ChannelsMotifsDists;
use crate::tasks::slices_stats::{Aggr as StatsAggr, Metrics};
use crate::tasks::transforms::freq_filter::FreqFilter;
use thoosa_common::stages_seq::{ABStage, StageChainOption};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_with_data::with_data;
use crate::tasks::motif_stats::{MotifStats, MotifStatsInput, MotifStatsOutput};
use thoosa_shared::tracked_await::track_await;

pub type FilteredLoaderStage<L, C> = ABStage<L, FreqFilter<C>, C>;
pub type LoaderStageChoice<L, C> = StageChainOption<L, FilteredLoaderStage<L, C>>;
pub type CrossUnfoldMaskFn<C> = Box<dyn CrossBatchFn<DistsMask, MasksCtx, C> + 'static>;
type CrossUnfoldMaskStage<C> = CrossUnfoldClc<CrossUnfoldMaskFn<C>, DistsMask, MasksCtx, C>;
type DistsTrackerStage<C> = DistsTracker<C>;



#[derive(Debug)]
enum FindMotifsStages<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    CrossLoader {
        id: RunConfigId,
        cross_loader: Arc<CrossLoader<
            RC,
            L,
            CrossUnfoldMaskStage<C>,
            DistsTrackerStage<C>,
            DistsMask,
            C,
        >>
    },
    Bucketing {
        id: RunConfigId,
        bucketing_stage: Arc<BucketingStage<RC, L, ChannelsData, SynKey, C>>,
    },
}

impl<RC, L, C> Clone for FindMotifsStages<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    fn clone(&self) -> Self {
        match self {
            Self::CrossLoader { id, cross_loader } => Self::CrossLoader { id: *id, cross_loader: Arc::clone(cross_loader) },
            Self::Bucketing { id, bucketing_stage } => Self::Bucketing { id: *id, bucketing_stage: Arc::clone(bucketing_stage) },
        }
    }
}

impl<RC, L, C> FindMotifsStages<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new_cross_loader(
        id: RunConfigId,
        cross_loader: Arc<CrossLoader<
            RC,
            L,
            CrossUnfoldMaskStage<C>,
            DistsTrackerStage<C>,
            DistsMask,
            C,
        >>
    ) -> Result<Self, ThoError> {
        Ok(Self::CrossLoader {
            id,
            cross_loader
        })
    }

    pub fn new_bucketing(
        id: RunConfigId,
        bucketing_stage: Arc<BucketingStage<RC, L, ChannelsData, SynKey, C>>,
    ) -> Result<Self, ThoError> {
        Ok(Self::Bucketing {
            id,
            bucketing_stage,
        })
    }

    pub fn id(&self) -> &RunConfigId {
        match self {
            Self::CrossLoader { id, .. } => id,
            Self::Bucketing { id, .. } => id,
        }
    }

    pub async fn run(&self, input: Arc<ResourcesConfigInput<RC>>, mut ctx:C) -> Result<Arc<()>, ThoError> {
        match self {
            Self::CrossLoader { cross_loader, .. } => track_await(Arc::clone(cross_loader).run(input, ctx)).await,
            Self::Bucketing { bucketing_stage, .. } => {
                let input = input.resources_config().clone();
                track_await(Arc::clone(bucketing_stage).run(Arc::new(BucketingInput::new(input)), ctx)).await?;
                Ok(Arc::new(()))
            }
        }
    }

}

pub enum MasksCtx {
    AccumulatedMeanDist {k:f32, ctx:MeanDistsCtx},
    AccumulatedQuantileDist {q:f32, ctx:QuantileDistsCtx},
    PrecalculatedDist {dists:ThoTensor<ChannelsMotifsDists, SynKey>}
}

pub struct MeanDistsCtx {
    means_acc: Mutex<Tensor>,
    total_samples: Mutex<Tensor>,
}

impl MeanDistsCtx {
    pub fn new(channels: u32, device: Device) -> Result<Self, ThoError> {
        let means_acc = Mutex::new(Tensor::f_zeros([channels as i64, 1], (Kind::Float, device))?);
        let total_samples = Mutex::new(Tensor::f_zeros([channels as i64, 1], (Kind::Int64, device))?);

        Ok(Self {
            means_acc,
            total_samples,
        })
    }
}

pub struct QuantileDistsCtx {
    quantiles_acc: Mutex<Tensor>,
    total_samples: Mutex<Tensor>,
}

impl QuantileDistsCtx {
    pub fn new(channels: u32, device: Device) -> Result<Self, ThoError> {
        let quantiles_acc = Mutex::new(Tensor::f_zeros([channels as i64, 1], (Kind::Float, device))?);
        let total_samples = Mutex::new(Tensor::f_zeros([channels as i64, 1], (Kind::Int64, device))?);

        Ok(Self {
            quantiles_acc,
            total_samples,
        })
    }
}

fn to_stats_aggr(aggr: FilterAggr) -> StatsAggr {
    match aggr {
        FilterAggr::Mean(v) => StatsAggr::Mean(v),
        FilterAggr::Min(v) => StatsAggr::Min(v),
        FilterAggr::Max(v) => StatsAggr::Max(v),
        FilterAggr::Quantile(v) => StatsAggr::Quantile(v),
    }
}

/// Searches for repeating patterns (motifs) in the resources by cross-comparing slices within and across resources, computing a metric (currently only Euclidean distance), building a mask of minima, and then saving it (see DistsTracker).
#[derive(Debug, Named)]
pub struct FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    run_config_stages: Vec<FindMotifsStages<RC, LoaderStageChoice<L, C>, C>>,
    pd: PhantomData<(RC, L, C)>,
}

impl<RC, L, C> FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync+ Debug + Clone + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>  + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub async fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        loader: Arc<L>,
        run_configs: Vec<FindMotifsRunConfig>,
        fixed_config: FixedFindMotifsConfig,
        device: ThoDevice, //TODO remove, make MeanDistsCtx cachable in DataCacheWrapper
        mut init_ctx:C
    ) -> Result<Self, ThoError> {
        
        debug!("FindMotifsStage::new: fixed_config={fixed_config:?} init_ctx.id={:?}", init_ctx.id());

        let motif_stats = Arc::new(MotifStats::new(
            Arc::clone(&store),
            Arc::clone(&loader),
            Arc::clone(ws),
            Arc::new(MotifLoader::new(Arc::clone(&store), Arc::clone(&loader), ws)),
        ));


        //gpt Build per-run-config loaders in parallel; each run_config becomes its own CrossLoader pipeline.
        let run_configs_len = run_configs.len();
        let ws = Arc::clone(ws);
        let store = Arc::clone(store);
        let futures = run_configs
            .into_iter()
            .map( |run_config| {
                let fixed_config = fixed_config.clone();
                let loader = Arc::clone(&loader);
                let motif_stats = Arc::clone(&motif_stats);
                let ws = Arc::clone(&ws);
                let store = Arc::clone(&store);
                let run_config_id = run_config.id.expect("should be pre saved");
                let mut ctx = init_ctx.branch(false);

                async move {
                    let ctx_id = ctx.id().clone();
                    debug!("FindMotifsStage::new: run_config={run_config:?} ctx.id={:?}", ctx.id());
                    let find_motifs_run_config_id = run_config.id.expect("find motif run config without id");
                    let configured_loader = config_to_stage(&loader, &run_config, &ws)?;

                     Ok(match &run_config.finder_kind {
                        FinderKind::Bucket(bucket_run_config) => {
                            let mut bucket_run_config = bucket_run_config.clone();
                            bucket_run_config.id = Some(find_motifs_run_config_id);

                            debug!("FindMotifsStage::new: using BucketingStage for run_config_id={find_motifs_run_config_id} ctx.id={:?}", ctx_id);
                            FindMotifsStages::new_bucketing(
                                find_motifs_run_config_id,
                                Arc::new(BucketingStage::new(
                                    &ws,
                                    &store,
                                    configured_loader,
                                    bucket_run_config,
                                    run_config.window,
                                    run_config.step,
                                )?),
                            )?
                        }
                        FinderKind::CrossLoader(cross_loader_run_config) => {
                            let cross_loader_run_config = cross_loader_run_config.clone();
                            let channels = perf_log!(sync_read_lock!(ws).channels());

                            let filter = cross_loader_run_config.slices_filter.clone();

                            let masks_ctx = match &filter {
                                SlicesFilter::Accumulate(v) => {
                                    match v.aggr {
                                        FilterAggr::Mean(aggr) => MasksCtx::AccumulatedMeanDist { k: aggr.k, ctx: MeanDistsCtx::new(channels, device.into())? },
                                        FilterAggr::Min(_) => return Err(ThoError::new("FindMotifsStage::new: Aggr::Min is not supported for SlicesFilter::AccumulateStats")),
                                        FilterAggr::Max(_) => return Err(ThoError::new("FindMotifsStage::new: Aggr::Max is not supported for SlicesFilter::AccumulateStats")),
                                        FilterAggr::Quantile(aggr) => MasksCtx::AccumulatedQuantileDist { q: aggr.q, ctx: QuantileDistsCtx::new(channels, device.into())? },
                                    }
                                }
                                SlicesFilter::Precalculate(v) => {
                                    debug!("FindMotifsStage::new: SlicesFilter::Precalculate ctx.id={:?}", ctx_id);

                                    let comparison_metric = Metrics::from_similarity_metric(cross_loader_run_config.similarity_metric, to_stats_aggr(v.aggr));
                                    //TODO can be effectively cached because run configs recreated for each input in  run_all/run_cli
                                    let motif_stats = track_await(motif_stats.run_cached(Arc::new(MotifStatsInput::from_random(v.slices, run_config_id, vec![comparison_metric.clone()])), ctx.branch(true))).await?;

                                    debug!("FindMotifsStage::new: motif_stats={motif_stats:?} ctx.id={:?}", ctx_id);

                                    let dists: ThoTensor<ChannelsMotifsDists, SynKey> = with_data!(ctx, motif_stats; |ctx: C, t:&MotifStatsOutput| {
                                        let metric_val = t.channels_motifs_dists(&comparison_metric)?;
                                        let metric_val_id = metric_val.id().clone();
                                        debug!("FindMotifsStage::new, SlicesFilter::PrecalculateStats: metric_val=\n{}", metric_val.t_ref());

                                        ThoTensor::new(metric_val_id, metric_val.t().f_view((-1, 1, 1))?.f_to_device(ThoDevice::from(ctx.tho_device_idx()).into())?)
                                    })?;
                                    MasksCtx::PrecalculatedDist { dists }
                                }
                                SlicesFilter::ExactS(v) => {
                                    let exact_stats_t = Tensor::f_from_slice(&[v.euc_dist])?
                                        .f_to_device(ThoDevice::from(ctx.tho_device_idx()).into())?
                                        .f_broadcast_to([channels as i64, 1])?;
                                    debug!("FindMotifsStage::new, SlicesFilter::PrecalculateStats: exact_stats_t=\n{}", exact_stats_t);
                                    MasksCtx::PrecalculatedDist { //rename to PreSet
                                        dists: ThoTensor::new("ExactStats".into(), exact_stats_t)?
                                    }
                                }
                            };
                            let radius = run_config.step() as u64 * 2;

                            debug!("FindMotifsStage::new: masks_ctx done  ctx.id={:?}", ctx_id);

                            let cross_unfold_clc: Arc<CrossUnfoldMaskStage<C>> = Arc::new(CrossUnfoldClc::new(
                                Box::new(move |t_0: Tensor, t_1: Tensor, input: &CrossInput<Arc<Resource>>, masks_ctx: &MasksCtx, ctx: &C| {
                                    let device_idx = ctx.tho_device_idx();
                                    let is_diag = input.batch_0()[0].id().eq(&input.batch_1()[0].id()); //TODO asserts, non-square crosses

                                    let mut mask_makers = match &masks_ctx {
                                        MasksCtx::AccumulatedMeanDist { k, ctx } => {

                                            vec![
                                                LocalMin { dim: 1, radius },
                                                // LocalMin { dim: 2, radius: run_config.window as u64 / 2 },
                                                LeAccumulatedMean {
                                                    k: *k,
                                                    avg: &ctx.means_acc,
                                                    total_samples: &ctx.total_samples,
                                                },
                                            ]
                                        }
                                        MasksCtx::AccumulatedQuantileDist { q, ctx } => {
                                            vec![
                                                LocalMin { dim: 1, radius },
                                                // LocalMin { dim: 2, radius: run_config.window as u64 / 2 },
                                                LeAccumulatedQuantile {
                                                    q: *q,
                                                    quantiles: &ctx.quantiles_acc,
                                                    total_samples: &ctx.total_samples,
                                                },
                                            ]
                                        }
                                        MasksCtx::PrecalculatedDist { dists } => {
                                            vec![
                                                LocalMin { dim: 1, radius },
                                                // LocalMin { dim: 2, radius: run_config.window as u64 / 2 },
                                                Le {
                                                    v: dists.t_ref(),
                                                }
                                            ]
                                        }
                                    };

                                    if is_diag {
                                        mask_makers.push(Tril);
                                    }

                                    //copy to device is done on streams created in CopyCtx and used in DistsMask::new_async. Should wait for calculation stream to finish.
                                    debug!("FindMotifsStage::new, batch_clc: t_0.size={:?}, t_1.size={:?}", t_0.size(), t_1.size());

                                    //it's invoked from CrossLoader::run/OOMHandler/CrossInputSelectorWrap//with_[selected_]data_ref where stream is synchronized
                                    let (mask, dists) = thoosa_motifs::distance::clc_d2_similarity_and_mask(
                                         &t_0,
                                         &t_1,
                                         cross_loader_run_config.similarity_metric,
                                         fixed_config.keep_slice_matches, //if true dists are returned + slices are stored in sqlite
                                         &mut mask_makers,
                                    )?;


                                    trace!("FindMotifsStage::new, batch_clc: mask=\n{mask}");
                                    debug!("FindMotifsStage::new, batch_clc: mask={mask:?} dists={dists:?}");
                                    let mask_type = if device_idx > 0 {
                                        DistsMaskType::MaskIndexes(mask.size())
                                    } else {
                                        DistsMaskType::AsIs
                                    };
                                    Ok(DistsMask::new(mask, dists, mask_type)?)
                                }),
                                masks_ctx,
                                UnfoldParams::new(1, run_config.window as i64, run_config.step as i64),
                            ));
                            debug!("FindMotifsStage::new: cross_unfold_clc={cross_unfold_clc:?} ctx.id={:?}", ctx_id);

                            let run_config_clone = run_config.clone();
                            let step = cross_unfold_clc.as_ref().unfold_params.step();

                            debug!("FindMotifsStage::new: ctx.id={:?} done", ctx_id);

                            FindMotifsStages::new_cross_loader(
                                find_motifs_run_config_id,
                                Arc::new(CrossLoader::new(
                                    find_motifs_run_config_id,
                                    configured_loader,
                                    &store,
                                    Arc::clone(&cross_unfold_clc),
                                    {
                                        let ws = Arc::clone(&ws);
                                        let store = Arc::clone(&store);
                                        let run_config = run_config_clone.clone();
                                        let fixed_config = fixed_config.clone();
                                        move |resources: &[Arc<Resource>]| {
                                            debug!("FindMotifsStages::new: new DistsTracker, resources.len={}", resources.len());
                                            Arc::new(DistsTracker::<C>::new(
                                                &ws,
                                                &store,
                                                &run_config,
                                                &fixed_config,
                                                run_config.max_slices.as_ref().map(|max_slices| {
                                                    match max_slices {
                                                        MaxSlices::ByResourcesCount(max_by_resources) => { resources.len() as u32 * max_by_resources.k }
                                                        MaxSlices::Abs(max_slices_ab) => max_slices_ab.count
                                                    }
                                                }),
                                            ))
                                        }
                                    },
                                    fixed_config.max_chunk_size as usize,
                                    |batch| {
                                        async { track_await(batch.to_cpu()).await }.boxed()
                                    },
                                    |receiver| {
                                        let dists_tracker = Arc::clone(receiver);
                                        async move {
                                            track_await(dists_tracker.build_groups()).await
                                        }
                                            .boxed()
                                    },
                                    CrossUnfoldClcOutputMergeMode::CpuMerge,
                                    step as u32,
                                )),
                            )?
                        }
                    })
                }

            });
        debug!("FindMotifsStage::new: fixed_config={fixed_config:?} futures_done");

        let cross_loaders = track_await(collect_futures_with_progress(
            futures,
            run_configs_len,
            FuturesOrdering::Unordered,
            "FindMotifsStage::new",
        )).await
        .into_iter()
        .collect::<Vec<_>>();

        debug!("FindMotifsStage::new: cross_loaders={cross_loaders:?}");

        Ok(Self {
            run_config_stages: cross_loaders,
            pd: Default::default(),
        })
    }
}

impl<RC, L, C> Stage for FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L, C>: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + 'static,
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ResourcesConfigInput<RC>
    where
        Self: 'a;
    type Out = FindMotifsSavableResult;
}


// #[async_trait]
impl<RC, L, C> AsyncStage for FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L, C>: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> +  'static,
    C: RunCtx + Adjustable + 'static,
{
    type RunInput = Arc<Self::In<'static>>;
    type RunOutput = Arc<Self::Out>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        thoosa_shared::traits::WithId::id(input)
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="FindMotifsStage::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<ResourcesConfigInput<RC>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<Self::Out>, ThoError> {



        let run_config_stages_len = self.run_config_stages.len();
        debug!("FindMotifsStage::run: run_config_stages.len={run_config_stages_len}");

        //gpt Run each per-config CrossLoader; failures are logged but do not abort the stage.
        // TODO(perf): order or group run configs by shared resource/filter pipeline so repeated RawTensorLoader/config_to_stage work can be reused across FindMotifsStage children.
        let futures = self
            .run_config_stages
            .iter()
            .cloned()
            .map(|stage|{
                info!("running find motifs on run config {:?}", stage.id());

                let ctx_for_stage = ctx.branch(true);
                let input = Arc::clone(&input);

                async move { stage.run(input, ctx_for_stage).await }
            });

        let _/*cross loader's out is ()*/ = track_await(collect_futures_with_progress(
            futures,
            run_config_stages_len,
            FuturesOrdering::Unordered,
            "FindMotifsStage::run",
        )).await;


        //gpt FindMotifsStage output is a placeholder; the real artifacts are persisted by DistsTracker side effects.
        Ok(Arc::new(FindMotifsSavableResult::new()))
    }
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="config_to_stage"))]
pub fn config_to_stage<L, C>(loader: &Arc<L>, find_motifs_run_config: &FindMotifsRunConfig, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<Arc<LoaderStageChoice<L, C>>, ThoError>
where
    C: 'static + RunCtx + Adjustable,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData, SynKey>, Ctx=C, RunInput=Arc<Resource>, RunOutput=Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
{
    let configurable_loader: Arc<LoaderStageChoice<L, C>> = match find_motifs_run_config.freq_filter.as_deref() {
        None | Some([]) => {
            debug!("MotifLoader::run: find_motifs_run_config_id={:?} no filter", find_motifs_run_config.id);
            Arc::new(StageChainOption::<L, FilteredLoaderStage<L, C>>::Single(Arc::clone(loader)))
        },
        Some(freq_filter) => {
            debug!("MotifLoader::run: find_motifs_run_config_id={:?} freq_filter={freq_filter:?}", find_motifs_run_config.id);

            Arc::new(StageChainOption::<L, FilteredLoaderStage<L, C>>::Chained(Arc::new(ABStage::new(
                SynKey::new([mono_key!(find_motifs_run_config.id.expect("find_motifs_run_config"), "ab_stage")]),
                Arc::clone(loader),
                Arc::new(FreqFilter::<C>::new(freq_filter, ws, find_motifs_run_config.id.expect("find_motifs_run_config"))?),
            ))))
        },
    };
    Ok(configurable_loader)
}


#[cfg(test)]
mod tests {
    use crate::cli::{import_resources, run_grid};
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_motifs::FindMotifsStage;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use crate::tasks::tensor_loader::{mock_tensor_loader, RawTensorLoader};
    use thoosa_common::runner::common::{mock_run_ctx, MaxParallelCtx, MockRunCtx, RunCtx};
    use thoosa_common::runner::common::cachable::{DevicesWrap, WithDataRef};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use log::{debug, set_max_level};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use tch::Device::Cuda;
    use tch::{Device, IndexOp, Tensor};
    use tch::display::set_print_options_full;
    use tch::kind::INT64_CPU;
    use thoosa_common::runner::common::cachable::Cached;
    use thoosa_common::tho_device::ThoDevice;
    use crate::tasks::find_motifs_run_config::{FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfigRangeOrValue};
    use thoosa_common::runs::{Labeled, Simple};
    use thoosa_common::{debug_plot, tch_test_utils};
    use thoosa_common::utils::{debug_plot_impl, lin_t, plot_series, reset_print_options, three_sigs_tensor, to_vec_f32};
    use thoosa_common::ws::{mock_ws_simp, FsWorkspace};
    use thoosa_motifs::find::{FindMotifsSavableResult, FixedFindMotifsConfig};
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::DataId;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::{cdt_for_dir, sig_one_imp, sig_three_imp, sig_two_imp, ResourceType, ThoError};
    use thoosa_store::{CheckGroups, MockStoreBuilder, RunConfigRow, SqlStore};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::ResourceType::Group;
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_with_data::with_data;
    use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
    use thoosa_common::dims::MotifSlices;
    use crate::tasks::motif_stats::RANDOM_STATS_GROUP;
    use crate::tasks::select_motifs::SelectMotifsStage;

    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn test_find_motifs_running_mean() -> Result<(), ThoError> {
        setup_find_motifs(FindMotifsRunConfig::new_with_running_mean(Some(1), 10, 100, None, Some(0.5))).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn test_find_motifs_max_dist() -> Result<(), ThoError> {
        setup_find_motifs(FindMotifsRunConfig::new_with_precalculate_stats(Some(1), 10, 100, None, 0.1, 60)).await?;

        Ok(())
    }

    async fn setup_find_motifs(find_motifs_run_config: FindMotifsRunConfig) -> Result<(), ThoError> {
        // set_max_level(Trace);
        // reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 10000)));
        let find_motifs_run_config_id = 1;


        let resources = (0..3).map(|i| {
            let size = i + 2;
            let resource = Resource::new(i, 1, format!("resource_{i}"), size * 1000);


            let (resource_t, resource_intervals) = tch_test_utils::resource_tensor(resource.length() as usize, &[
                            vec![(size as usize, 100, MotifKind::Pulse)],
                            vec![(size as usize, 100, MotifKind::Sin)],
                            vec![(size as usize, 100, MotifKind::Triangle)]
                        ], 100, 10.0, true, true)?;


            debug_plot_impl(&resource_t, &format!("resource_{i}"), 0)?;

            Ok((resource, resource_t, resource_intervals))
        }).collect::<Result<Vec<_>, ThoError>>()?;

        let resources_group = "group";

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([(
                    find_motifs_run_config_id,
                    Ok(RunConfigRow {
                        id: find_motifs_run_config_id,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find motifs".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_list_run_config_values([(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_list_group_labeled_resources_sets([
                    (resources_group, Ok(vec![
                        ("X".to_string(), "set".to_string()),
                    ])),
                ])
                .with_list_resource_set_group_sets(
                    [(resources_group, Ok(vec![(1, "set".into())]))]
                )
                .with_list_resource_set_resources_by_descr([
                    ("set", Ok(resources.iter().map(|(resource, _resource_t, _resource_intervals)| {
                        resource.clone()
                    }).collect())),
                ])
                //for max dist
                .with_list_resources_not_in_group([(RANDOM_STATS_GROUP, Ok(resources.iter().map(|(resource, _resource_t, _resource_intervals)| {
                    resource.clone()
                }).collect()))])
                .with_list_random_resources([(None, Ok(resources.iter().map(|r|r.0.clone()).collect::<Vec<_>>()))])
                .with_save_groups_as_motifs(vec![CheckGroups::MinGroups(1), CheckGroups::MinGroups(1), CheckGroups::MinGroups(1)]) //<-assert TODO check if too many groups
                .build());

        let loader = Arc::new(
            mock_tensor_loader::build()
                .with_out_id()
                .with_run_cached(resources.iter().map(|(resource, resource_t, _resource_intervals)| {
                    (resource.clone(), resource_t.shallow_clone())
                }))
                .finish(),
        );
        let mut ctx = mock_run_ctx::default(1);

        let stage: Arc<FindMotifsStage<Simple, _, _>> = Arc::new(FindMotifsStage::new(
            &ws,
            &store,
            Arc::clone(&loader),
            vec![
                find_motifs_run_config
            ],
            FixedFindMotifsConfig::new(30000, false),
            ThoDevice::from(1),
            ctx.branch(false)
        ).await.expect("FindMotifsStage::new failed in test"));


        let result = stage.run(
            Arc::new(ResourcesConfigInput::new(Simple::new(Group(resources_group.into())))), ctx.branch(false),
        ).await?;

        with_data!(ctx.branch(false), result; |ctx, result:&FindMotifsSavableResult|{
            debug!("result={result:?}");
            Ok::<_, ThoError>(())
        })?;


        Ok(())
    }
}
