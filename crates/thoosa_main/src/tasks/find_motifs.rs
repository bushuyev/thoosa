use std::collections::HashMap;
use crate::tasks::cross_batch_clc::{CrossInput, CrossUnfoldClc, CrossUnfoldClcOutput, UnfoldParams};
use crate::tasks::cross_loader::{
    CrossLoader, CrossUnfoldClcOutputMergeMode, ResourcesConfigInput,
};
use crate::tasks::dists_tracker::{DistsMask, DistsMaskType, DistsTracker};
use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
use crate::tasks::tensor_loader::RawTensorLoader;
use async_trait::async_trait;
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
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
use thoosa_common::find_motifs_run_config::{Aggr as FilterAggr, FindMotifsRunConfig, MaxSlices, SlicesFilter};
use thoosa_common::loader::LoadRunner;
use thoosa_common::runs::Simple;
use thoosa_common::ws::Workspace;
use thoosa_motifs::distance::MaskMaker::{Le, LeAccumulatedMean, LeAccumulatedQuantile, LocalMin, Tril};
use thoosa_motifs::find::{FindMotifsSavableResult, FixedFindMotifsConfig};
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_store::{SqlStore, Store};
use tokio::task::{JoinError, JoinSet};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_common::cuda_utils::{pack_bits_bool, with_cuda_stream};
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::data_with_extra::DataWithExtra;
use crate::tasks::slices_stats::ScalarStatsValue;
use crate::tasks::motif_loader::MotifMeta;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_named_derive::Named;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsMotifsDists, MotifSlices};
use crate::tasks::slices_stats::{Aggr as StatsAggr, Metrics};
use crate::tasks::transforms::freq_filter::FreqFilter;
use crate::tasks::stages_seq::{ABStage, StageChainOption};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::run_configs::RunId;
use thoosa_with_data::with_data;
use crate::buffer_limits::{CROSS_LOADERS_LIMIT, FIND_MOTIFS_STAGES_BUFFER_UNORDERED_LIMIT};
use crate::tasks::random_slices_stats::{RandomStats, RandomStatsInput};

pub type FilteredLoaderStage<L, C> = ABStage<L, FreqFilter<C>, C>;
pub type LoaderStageChoice<L, C> = StageChainOption<L, FilteredLoaderStage<L, C>>;

//TODO remove - use cross_loader directly
#[derive(Debug)]
struct FindMotifsStages<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for <'i> Cached<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C>  + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    run_config: Arc<FindMotifsRunConfig>,
    cross_loader: Arc<CrossLoader<
            RC,
        L,
            CrossUnfoldClc<Box<dyn Fn(Tensor, Tensor, &CrossInput<Arc<Resource>>, &MasksCtx, &C) -> Result<DistsMask, ThoError> + Send + Sync>, DistsMask, MasksCtx, C>,
            DistsTracker<C>,
            DistsMask,
            C,
        >>
}

impl<RC, L, C> FindMotifsStages<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for <'i> Cached<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C>  + OutId + 'static,
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(
        run_config: FindMotifsRunConfig,
        cross_loader: Arc<CrossLoader<
            RC,
            L,
            CrossUnfoldClc<Box<dyn Fn(Tensor, Tensor, &CrossInput<Arc<Resource>>, &MasksCtx, &C) -> Result<DistsMask, ThoError> + Send + Sync>, DistsMask, MasksCtx, C>,
            DistsTracker<C>,
            DistsMask,
            C,
        >>
    ) -> Result<Self, ThoError> {
        Ok(Self {
            run_config: Arc::new(run_config),
            cross_loader
        })
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
    pub fn new(channels: u32, device: Device) -> Self {
        let means_acc = Mutex::new(Tensor::zeros([channels as i64, 1], (Kind::Float, device)));
        let total_samples = Mutex::new(Tensor::zeros([channels as i64, 1], (Kind::Int64, device)));

        Self {
            means_acc,
            total_samples,
        }
    }
}

pub struct QuantileDistsCtx {
    quantiles_acc: Mutex<Tensor>,
    total_samples: Mutex<Tensor>,
}

impl QuantileDistsCtx {
    pub fn new(channels: u32, device: Device) -> Self {
        let quantiles_acc = Mutex::new(Tensor::zeros([channels as i64, 1], (Kind::Float, device)));
        let total_samples = Mutex::new(Tensor::zeros([channels as i64, 1], (Kind::Int64, device)));

        Self {
            quantiles_acc,
            total_samples,
        }
    }
}

fn to_stats_aggr(aggr: FilterAggr) -> StatsAggr {
    match aggr {
        FilterAggr::Mean(v) => StatsAggr::Mean(v.k),
        FilterAggr::Min(v) => StatsAggr::Min(v.k),
        FilterAggr::Max(v) => StatsAggr::Max(v.k),
        FilterAggr::Quantile(v) => StatsAggr::Quantile(v.q),
    }
}

/// Searches for repeating patterns (motifs) in the resources by cross-comparing slices within and across resources, computing a metric (currently only Euclidean distance), building a mask of minima, and then saving it (see DistsTracker).
#[derive(Debug, Named)]
pub struct FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + Adjustable + 'static,
{
    run_config_stages: Vec<FindMotifsStages<RC, LoaderStageChoice<L, C>, C>>,
    pd: PhantomData<(RC, L, C)>,
}

impl<RC, L, C> FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug + Clone,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C>  + Cached + OutId + Send + Sync + 'static,
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
        
        debug!("FindMotifsStage::new: fixed_config={fixed_config:?}");

        let random_stats = Arc::new(RandomStats::new(Arc::clone(&store), Arc::clone(&loader), &ws));


        //gpt Build per-run-config loaders in parallel; each run_config becomes its own CrossLoader pipeline.
        let run_configs_len = run_configs.len();
        let futures = run_configs
            .into_iter()
            .map( |run_config| {
                let fixed_config = fixed_config.clone();
                let loader = Arc::clone(&loader);
                let random_stats_arc = Arc::clone(&random_stats);
                let run_id = run_config.id.expect("should be pre saved");
                let mut ctx = init_ctx.branch(Some(&format!("init ctx for run_config {run_id}")), false, false);

                async move {
                    debug!("FindMotifsStage::new: run_config={run_config:?}");
                    let channels = thoosa_shared::with_perf_log(
                        || ws.try_read_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout").channels(),
                        "FindMotifsStage::new/ws_read_channels",
                    );

                    let filter = run_config.slices_filter.clone();

                    let masks_ctx = match &filter {
                        SlicesFilter::AccumulateStats(v) => {
                            match v.aggr {
                                FilterAggr::Mean(aggr) => MasksCtx::AccumulatedMeanDist { k: aggr.k, ctx: MeanDistsCtx::new(channels, device.into()) },
                                FilterAggr::Min(_) => return Err(ThoError::new("FindMotifsStage::new: Aggr::Min is not supported for SlicesFilter::AccumulateStats")),
                                FilterAggr::Max(_) => return Err(ThoError::new("FindMotifsStage::new: Aggr::Max is not supported for SlicesFilter::AccumulateStats")),
                                FilterAggr::Quantile(aggr) => MasksCtx::AccumulatedQuantileDist { q: aggr.q, ctx: QuantileDistsCtx::new(channels, device.into()) },
                            }
                        }
                        SlicesFilter::PrecalculateStats(v) => {
                            let dists: Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>> = random_stats_arc.run(Arc::new(DevicesWrap::wrap(RandomStatsInput::new(v.slices, run_id, Metrics::EucDist, to_stats_aggr(v.aggr)), CacheArea::default())), ctx.branch(Some("cxt for avg_dists_arc"), true, false)).await?;
                            let dists = with_data!(ctx, dists; |ctx, t:&ThoTensor<ChannelsMotifsDists, SynKey>|{

                                    debug!("FindMotifsStage::new, SlicesFilter::PrecalculateStats: dists=\n{}", t.t);

                                    ThoTensor::new(t.id().clone(), t.t.view((-1, 1, 1)))
                                })?;
                            MasksCtx::PrecalculatedDist { dists }
                        }
                        SlicesFilter::ExactStats(v) => {
                            let exact_stats_t = Tensor::from_slice(&[v.euc_dist]).to_device(ThoDevice::from(ctx.device_idx()).into()).broadcast_to(&[channels as i64, 1]);
                            debug!("FindMotifsStage::new, SlicesFilter::PrecalculateStats: exact_stats_t=\n{}", exact_stats_t);
                            MasksCtx::PrecalculatedDist { //rename to PreSet
                                dists: ThoTensor::new("ExactStats".into(), exact_stats_t)?
                            }
                        }
                    };


                    let cross_unfold_clc: CrossUnfoldClc<
                        Box<dyn Fn(Tensor, Tensor, &CrossInput<Arc<Resource>>, &MasksCtx, &C) -> Result<DistsMask, ThoError> + Send + Sync>,
                        DistsMask,
                        MasksCtx,
                        C,
                    > = CrossUnfoldClc::new(
                        Box::new(move |t_0, t_1, input, masks_ctx, ctx| {
                            let device_idx = ctx.device_idx();
                            let is_diag = input.batch_0()[0].id().eq(&input.batch_1()[0].id()); //TODO asserts, non-square crosses

                            let mut mask_makers = match &masks_ctx {
                                MasksCtx::AccumulatedMeanDist { k, ctx } => {
                                    vec![
                                        LocalMin { dim: 1, radius: run_config.window as u64 / 2 },
                                        LocalMin { dim: 2, radius: run_config.window as u64 / 2 },
                                        LeAccumulatedMean {
                                            k: *k,
                                            avg: &ctx.means_acc,
                                            total_samples: &ctx.total_samples,
                                        },
                                    ]
                                }
                                MasksCtx::AccumulatedQuantileDist { q, ctx } => {
                                    vec![
                                        LocalMin { dim: 1, radius: run_config.window as u64 / 2 },
                                        LocalMin { dim: 2, radius: run_config.window as u64 / 2 },
                                        LeAccumulatedQuantile {
                                            q: *q,
                                            quantiles: &ctx.quantiles_acc,
                                            total_samples: &ctx.total_samples,
                                        },
                                    ]
                                }
                                MasksCtx::PrecalculatedDist { dists } => {
                                    vec![
                                        LocalMin { dim: 1, radius: run_config.window as u64 / 2 },
                                        LocalMin { dim: 2, radius: run_config.window as u64 / 2 },
                                        Le {
                                            v: dists.t.shallow_clone(),
                                        }
                                    ]
                                }
                            };

                            if is_diag {
                                mask_makers.push(Tril);
                            }

                            //copy to device is done on streams created in CopyCtx and used in DistsMask::new_async. Should wait for calculation stream to finish.
                            debug!("FindMotifsStage::new, batch_clc: t_0.size={:?}, t_1.size={:?}", t_0.size(), t_1.size());

                            let (clc_dist_d2_and_mask_result, motifs_stream) = with_cuda_stream(device_idx, || {
                                thoosa_motifs::distance::clc_dist_d2_and_mask(
                                    t_0.shallow_clone(),
                                    t_1.shallow_clone(),
                                    fixed_config.keep_slice_matches, //if true dists are returned + slices are stored in sqlite
                                    &mut mask_makers,
                                )
                            });

                            let (mask, dists) = clc_dist_d2_and_mask_result?;

                            trace!("FindMotifsStage::new, batch_clc: mask=\n{mask}");
                            debug!("FindMotifsStage::new, batch_clc: mask={mask:?} dists={dists:?}");
                            let mask_type = if device_idx > 0 {
                                DistsMaskType::MaskIndexes(mask.size())
                            } else {
                                DistsMaskType::AsIs
                            };
                            let stream = if device_idx > 0 { motifs_stream } else { None };
                            Ok(DistsMask::new(mask, dists, mask_type, stream)?)
                        }),
                        masks_ctx,
                        UnfoldParams::new(1, run_config.window as i64, run_config.step as i64),
                    );

                    let configured_loader = config_to_stage(&loader, &run_config, ws)?;
                    let run_config_clone = run_config.clone();
                    let step = cross_unfold_clc.unfold_params.step();

                    //gpt TODO!! relies on run_config.id being set (presave); consider erroring instead of panicking.
                    let find_motifs_run_config_id = run_config_clone.id.expect("find motif run config without id");
                    FindMotifsStages::new(
                        run_config,
                        Arc::new(CrossLoader::new(
                            find_motifs_run_config_id,
                            configured_loader,
                            &store,
                            Arc::new(cross_unfold_clc),
                            {
                                let ws = Arc::clone(ws);
                                let store = Arc::clone(store);
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
                                async { batch.to_cpu().await.unwrap() }.boxed() //TODO unwrap
                            },
                            |receiver| {
                                let dists_tracker = Arc::clone(receiver);
                                async move {
                                    dists_tracker.build_groups().await
                                }
                                    .boxed()
                            },
                            CrossUnfoldClcOutputMergeMode::CpuMerge,
                            step as u32,
                        )),
                    )
                }

            });

        let cross_loaders = collect_futures_with_progress(
            futures,
            run_configs_len,
            CROSS_LOADERS_LIMIT,//TODO config
            StreamBuffering::Unordered,
            "FindMotifsStage::new",
        )
        .await
        .into_iter()
        .collect::<Vec<_>>();

        Ok(Self {
            run_config_stages: cross_loaders,
            pd: Default::default(),
        })
    }
}

impl<RC, L, C> Stage for FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + Clone + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L, C>: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + 'static,
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
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L, C>: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C> +  'static,
    C: RunCtx + Adjustable + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="FindMotifsStage::run"))]
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<ResourcesConfigInput<RC>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {

        let mut js: JoinSet<Result<Arc<DevicesWrap<()>>, ThoError>> = JoinSet::new();//TODO to stream + spawn

        let run_config_stages_len = self.run_config_stages.len();
        debug!("FindMotifsStage::run: run_config_stages.len={run_config_stages_len}");
        let failed = Arc::new(AtomicU8::new(0));
        //gpt Run each per-config CrossLoader; failures are logged but do not abort the stage.
        let futures = self
            .run_config_stages
            .iter()
            .map(|stage|{
                info!("running find motifs on run config {:?}", stage.run_config.id);

                let ctx_for_stage = ctx.branch(Some(&format!("find_motifs_for_rc_{}", stage.run_config.id.expect("no run_config"))), true, false);
                let input_arc = Arc::clone(&input);
                let cross_loader_arc = Arc::clone(&stage.cross_loader);

                let failed = Arc::clone(&failed);
                async move {
                    match cross_loader_arc.run(input_arc, ctx_for_stage).await {
                        Ok(v) => Ok::<_, ThoError>(Some(v)),
                        Err(e) => {
                            failed.fetch_add(1, Ordering::Relaxed);
                            warn!("Find motifs failed: {e:?}");
                            //gpt TODO!! errors are swallowed; downstream stages still run using pre-saved run ids.
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let _ = collect_futures_with_progress(
            futures,
            run_config_stages_len,
            FIND_MOTIFS_STAGES_BUFFER_UNORDERED_LIMIT,
            StreamBuffering::Unordered,
            "FindMotifsStage::run",
        )
        .await;


        info!("failed {} out of {}",failed.load(Ordering::Relaxed),run_config_stages_len);

        //gpt FindMotifsStage output is a placeholder; the real artifacts are persisted by DistsTracker side effects.
        Ok(Arc::new(DevicesWrap::wrap(FindMotifsSavableResult::new(), CacheArea::default())))
    }
}


pub fn config_to_stage<L, C>(loader: &Arc<L>, find_motifs_run_config: &FindMotifsRunConfig, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<Arc<LoaderStageChoice<L, C>>, ThoError>
where
    C: 'static + RunCtx,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SynKey>, Ctx=C>
{
    let configurable_loader: Arc<LoaderStageChoice<L, C>> = match find_motifs_run_config.freq_filter.as_deref() {
        None | Some([]) => {
            debug!("MotifLoader::run: find_motifs_run_id={:?} no filter", find_motifs_run_config.id);
            Arc::new(StageChainOption::<L, FilteredLoaderStage<L, C>>::Single(Arc::clone(loader)))
        },
        Some(freq_filter) => {
            debug!("MotifLoader::run: find_motifs_run_id={:?} freq_filter={freq_filter:?}", find_motifs_run_config.id);

            Arc::new(StageChainOption::<L, FilteredLoaderStage<L, C>>::Chained(Arc::new(ABStage::new(
                SynKey::new([mono_key!("ab", "stage")]),
                Arc::clone(loader),
                Arc::new(FreqFilter::<C>::new(freq_filter, ws)?),
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
    use thoosa_shared::runner::run_ctx_2::RunCtxWithPermits;
    use crate::tasks::tensor_loader::{mock_tensor_loader, RawTensorLoader};
    use thoosa_shared::runner::common::{mock_run_ctx, MockRunCtx, PermitsCtx, RunCtx};
    use thoosa_shared::runner::common::cachable::{DevicesWrap, WithDataRef};
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use log::{debug, set_max_level};
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use rand::random_range;
    use tch::Device::Cuda;
    use tch::{Device, IndexOp, Tensor};
    use tch::display::set_print_options_full;
    use tch::kind::INT64_CPU;
    use thoosa_shared::runner::common::cachable::Cached;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfigRangeOrValue};
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::{Labeled, Simple};
    use thoosa_common::tch_test_utils;
    use thoosa_common::utils::{lin_t, plot_series, plot_t, reset_print_options, three_sigs_tensor, to_vec_f32};
    use thoosa_common::ws::{mock_ws_simp, FsWorkspace};
    use thoosa_motifs::find::{FindMotifsSavableResult, FixedFindMotifsConfig};
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::DataId;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::{cdt_for_dir, sig_one_imp, sig_three_imp, sig_two_imp, ResourceType, ThoError};
    use thoosa_store::{MockStoreBuilder, SqlStore};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::ResourceType::Group;
    use thoosa_shared::test_utils::MotifKind;
    use thoosa_with_data::with_data;
    use crate::tasks::slices_stats::SlicesStats;
    use crate::tasks::load_motif_and_clc_dist::LoadMotifAndClcDistanceStage;
    use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
    use crate::tasks::cross_dims::MotifSlices;
    use crate::tasks::slices_stats::ScalarStatsValue;
    use crate::tasks::random_slices_stats::RANDOM_AVG_DIST_GROUP;
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
        setup_find_motifs(FindMotifsRunConfig::new_with_max_dist(Some(1), 10, 100, None, 0.5, 1000)).await?;

        Ok(())
    }

    async fn setup_find_motifs(find_motifs_run_config: FindMotifsRunConfig) -> Result<(), ThoError> {
        // set_max_level(Trace);
        GlobalCtx::init_devices(2);
        // reset_print_options();

        let ws = Arc::new(RwLock::new(mock_ws_simp(None, 3, 100, 10000)));
        let find_motifs_run_id = 1;


        let resources = (0..3).map(|i| {
            let size = random_range(2..5);
            let resource = Resource::new(i, 1, format!("resource_{i}"), size * 1000);


            let (resource_t, resource_intervals) = tch_test_utils::resource_tensor(resource.length() as usize, &[
                vec![(size as usize, 100, MotifKind::Pulse)],
                vec![(size as usize, 100, MotifKind::Sin)],
                vec![(size as usize, 100, MotifKind::Triangle)]
            ], 100, 10.0)?;


            // plot_t(&resource_t, &format!("resource_{i}"))?;

            Ok((resource, resource_t, resource_intervals))
        }).collect::<Result<Vec<_>, ThoError>>()?;

        let resources_group = "group";

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), 1, 100, None, None).to_properties_iter(None).collect()),
                )
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
                .with_list_resources_not_in_group([(RANDOM_AVG_DIST_GROUP, Ok(resources.iter().map(|(resource, _resource_t, _resource_intervals)| {
                    resource.clone()
                }).collect()))])
                .with_save_groups_as_motifs() //TODO check if resource_intervals are in found groups
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
            ctx.branch(Some("init"), false, false)
        ).await.expect("FindMotifsStage::new failed in test"));


        let result = stage.run(
            Arc::new(DevicesWrap::wrap(ResourcesConfigInput::new(Simple::new(Group(resources_group.into()))), CacheArea::default())), ctx.branch(Some("run"), false, false),
        ).await?;

        with_data!(ctx.branch(Some("check"), false, false), result; |ctx, result:&FindMotifsSavableResult|{
            debug!("result={result:?}");
            Ok::<_, ThoError>(())
        })?;
        Ok(())
    }
}
