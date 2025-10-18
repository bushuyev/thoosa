use crate::tasks::cross_batch_clc::{CrossInput, CrossUnfoldClc, CrossUnfoldClcOutput, UnfoldParams};
use crate::tasks::cross_loader::{
    CrossLoader, CrossResult, ResourcesConfigInput,
};
use crate::tasks::dists_tracker::{DistsMask, DistsTracker};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::{RawTensorLoader};
use async_trait::async_trait;
use caching_runner::common::{OutId, RunCtx};
use caching_runner::global_ctx::GlobalCtx;
use caching_runner::common::runnable::{AsyncStage, Stage};
use futures_util::FutureExt;
use log::{debug, error, info};
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
use thoosa_common::applicables::ThoDevice;
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, SlicesFilter};
use thoosa_common::loader::LoadRunner;
use thoosa_common::runs::Simple;
use thoosa_common::ws::Workspace;
use thoosa_motifs::distance::MaskMaker::{Le, LeRunningMean, LocalMin, Tril};
use thoosa_motifs::find::{FindMotifsSavableResult, FixedFindMotifsConfig};
use thoosa_shared::ThoosaError;
use thoosa_shared::resource::{Resource, ResourcesConfig};
use thoosa_store::{SqlStore, Store};
use tokio::task::{JoinError, JoinSet};
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use thoosa_common::cuda_utils::pack_bits_bool;
use thoosa_shared::traits::{Data, SimpleId, WithId};
use thoosa_shims::stream::{StreamGuard, ThoCudaStream};
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::transforms::freq_filter::FreqFilter;
use crate::tasks::stages_seq::ABStage;

type FilteredLoaderStage<L> = ABStage<L, FreqFilter, RunCtx2>;
type LoaderStageChoice<L> = StageChainOption<L, FilteredLoaderStage<L>>;


enum StageChainOption<Base, Chain>
where
    Base: AsyncStage + Send + Sync + Debug + 'static,
    Chain: for<'i> AsyncStage<
            In<'i>=<Base as Stage>::In<'i>,
            Out=<Base as Stage>::Out,
            Ctx=<Base as Stage>::Ctx,
        > + Send + Sync + Debug + 'static,
    for<'i> <Base as Stage>::In<'i>: Data + WithId + Send + Sync,
    <Base as Stage>::Out: Data + WithId + Send + Sync,
{
    Single(Arc<Base>),
    Chained(Arc<Chain>),
}

impl<Base, Chain> Debug for StageChainOption<Base, Chain>
where
    Base: AsyncStage + Send + Sync + Debug + 'static,
    Chain: for<'i> AsyncStage<
            In<'i>=<Base as Stage>::In<'i>,
            Out=<Base as Stage>::Out,
            Ctx=<Base as Stage>::Ctx,
        > + Send + Sync + Debug + 'static,
    for<'i> <Base as Stage>::In<'i>: Data + WithId + Send + Sync,
    <Base as Stage>::Out: Data + WithId + Send + Sync,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageChainOption::Single(stage) => f
                .debug_tuple("StageChainOption::Single")
                .field(stage)
                .finish(),
            StageChainOption::Chained(stage) => f
                .debug_tuple("StageChainOption::Chained")
                .field(stage)
                .finish(),
        }
    }
}

impl<Base, Chain> Stage for StageChainOption<Base, Chain>
where
    Base: AsyncStage + Send + Sync + Debug + 'static,
    Chain: for<'i> AsyncStage<
            In<'i>=<Base as Stage>::In<'i>,
            Out=<Base as Stage>::Out,
            Ctx=<Base as Stage>::Ctx,
        > + Send + Sync + Debug + 'static,
    for<'i> <Base as Stage>::In<'i>: Data + WithId + Send + Sync,
    <Base as Stage>::Out: Data + WithId + Send + Sync,
{
    type Ctx = <Base as Stage>::Ctx;
    type In<'a> = <Base as Stage>::In<'a>
    where
        Self: 'a;
    type Out = <Base as Stage>::Out;
}

#[async_trait]
impl<Base, Chain> AsyncStage for StageChainOption<Base, Chain>
where
    Base: AsyncStage + Send + Sync + Debug + 'static,
    Chain: for<'i> AsyncStage<
            In<'i>=<Base as Stage>::In<'i>,
            Out=<Base as Stage>::Out,
            Ctx=<Base as Stage>::Ctx,
        > + Send + Sync + Debug + 'static,
    for<'i> <Base as Stage>::In<'i>: Data + WithId + Send + Sync,
    <Base as Stage>::Out: Data + WithId + Send + Sync,
{
    async fn run(
        self: Arc<Self>,
        input: Arc<DataCacheWrapper<Self::In<'static>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {
        match &*self {
            StageChainOption::Single(stage) => Arc::clone(stage).run(input, ctx).await,
            StageChainOption::Chained(stage) => Arc::clone(stage).run(input, ctx).await,
        }
    }
}


#[derive(Debug)]
struct FindMotifsStages<RC, SRC>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    SRC: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + 'static,
{
    run_config: FindMotifsRunConfig,
    cross_loader: Arc<CrossLoader<
            RC,
            SRC,
            CrossUnfoldClc<Box<dyn Fn(Tensor, Tensor, &CrossInput<&Arc<Resource>>, &MeanDistsCtx) -> Result<DistsMask, ThoosaError> + Send + Sync>, DistsMask, MeanDistsCtx>,
            DistsTracker,
            DistsMask,
        >>
}

impl<RC, SRC> FindMotifsStages<RC, SRC>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    SRC: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + 'static,
{
    pub fn new(
        run_config: FindMotifsRunConfig,
        cross_loader: Arc<CrossLoader<
            RC,
            SRC,
            CrossUnfoldClc<Box<dyn Fn(Tensor, Tensor, &CrossInput<&Arc<Resource>>, &MeanDistsCtx) -> Result<DistsMask, ThoosaError> + Send + Sync>, DistsMask, MeanDistsCtx>,
            DistsTracker,
            DistsMask,
        >>
    ) -> Self {
        Self {
            run_config,
            cross_loader
        }
    }
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

#[derive(Debug)]
pub struct FindMotifsStage<RC, L>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + Cached + OutId + Send + Sync + 'static,
{
    run_config_stages: Vec<FindMotifsStages<RC, LoaderStageChoice<L>>>,
    pd: PhantomData<(RC, L)>,
}

impl<RC, L> FindMotifsStage<RC, L>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2>  + Cached + OutId + Send + Sync + 'static,
{
    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader: Arc<L>,
        run_configs: Vec<FindMotifsRunConfig>,
        fixed_config: FixedFindMotifsConfig,
        device: ThoDevice, //TODO remove, make MeanDistsCtx cachable in DataCacheWrapper
    ) -> Self {
        let band_filter = Arc::new(FreqFilter::new());

        let cross_loaders = run_configs
            .into_iter()
            .map(|run_config| {
                let channels = ws.read().channels();

                let filter = run_config.slices_filter.clone();

                let cross_unfold_clc: CrossUnfoldClc<Box<dyn Fn(Tensor, Tensor, &CrossInput<&Arc<Resource>>, &MeanDistsCtx) -> Result<DistsMask, ThoosaError> + Send + Sync >,
                    DistsMask,
                    MeanDistsCtx,
                > = CrossUnfoldClc::new(
                    Box::new(move |t_0, t_1, input, means_ctx| {
                        let is_diag = input.batch_0()[0].id().eq(&input.batch_1()[0].id()); //TODO asserts, non-square crosses

                        let mut mask_makers = match &filter {
                            SlicesFilter::DistanceRunningMeanK(v) => {
                                vec![
                                    LocalMin {dim: 1},
                                    LocalMin {dim: 2},
                                    LeRunningMean {
                                        k: v.mean,
                                        avg: &means_ctx.means_acc,
                                        total_samples: &means_ctx.total_samples,
                                    },
                                ]
                            }
                            SlicesFilter::MaxDistance(v) => {
                                vec![Le {
                                    v: Tensor::full(
                                        &[channels as i64, 1],
                                        v.distance as f64,
                                        (Float, t_0.device()),
                                    ),
                                }]
                            }
                        };

                        if is_diag {
                            mask_makers.push(Tril);
                        }

                        let motifs_stream = ThoCudaStream::new(0);//TODO
                        let guard = StreamGuard::set(&motifs_stream);//TODO Device

                        let (mask, dists) = thoosa_motifs::distance::clc_dist_d2_and_mask(
                            t_0,
                            t_1,
                            fixed_config.keep_slice_matches,//if true dists are returned
                            &mut mask_makers,
                        )?;
                        drop(guard);


                        debug!("CrossBatchClc::do_run:  dists={dists:?}");
                        if mask.device().ne(&Cpu) {

                            Ok(
                                tokio::task::block_in_place(|| {
                                    Handle::current().block_on(async {
                                        motifs_stream.synchronize();

                                        DistsMask::new_async(mask, dists, true).await

                                    })
                                })
                            )
                        } else {
                            Ok(DistsMask::new(mask, dists))
                        }

                    }),
                    MeanDistsCtx::new(channels, device.into()),
                    UnfoldParams::new(1, run_config.window as i64, run_config.step as i64),
                );

                let receiver = Arc::new(DistsTracker::new(
                    ws,
                    store,
                    &run_config,
                    &fixed_config,
                    None,
                ));

                let configurable_loader: Arc<LoaderStageChoice<L>> = match run_config.filter.as_ref() {
                    None => Arc::new(StageChainOption::<L, FilteredLoaderStage<L>>::Single(Arc::clone(&loader))),
                    Some(_) => Arc::new(StageChainOption::<L, FilteredLoaderStage<L>>::Chained(Arc::new(ABStage::new(
                        SimpleId::from("ab"),
                        Arc::clone(&loader),
                        Arc::clone(&band_filter),
                    )))),
                };

                FindMotifsStages::new(
                    run_config,
                    Arc::new(CrossLoader::new(
                        configurable_loader,
                        &store,
                        Arc::new(cross_unfold_clc),
                        receiver,
                        fixed_config.max_chunk_size as usize,
                        |dists_tracker| {
                            let dists_tracker = Arc::clone(dists_tracker);
                            async move {
                                dists_tracker.build_groups().await;
                            }
                            .boxed()
                        },
                    )),
                )
            })
            .collect();

        Self {
            run_config_stages: cross_loaders,
            pd: Default::default(),
        }
    }
}

impl<RC, L> Stage for FindMotifsStage<RC, L>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L>: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + 'static,
{
    type Ctx = RunCtx2;
    type In<'a> = ResourcesConfigInput<RC>
    where
        Self: 'a;
    type Out = FindMotifsSavableResult;
}


#[async_trait]
impl<RC, L> AsyncStage for FindMotifsStage<RC, L>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L>: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor2<ChannelsData2, SimpleId>, Ctx=RunCtx2> +  'static,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DataCacheWrapper<ResourcesConfigInput<RC>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {
        let best = 0;
        let mut js: JoinSet<Result<Arc<DataCacheWrapper<CrossResult>>, ThoosaError>> = JoinSet::new();

        for stage in self.run_config_stages.iter() {
            let ctx_arc = Arc::clone(&ctx);
            let input_arc = Arc::clone(&input);
            let cross_loader_arc = Arc::clone(&stage.cross_loader);

            js.spawn(async move { cross_loader_arc.run(input_arc, ctx_arc).await });
        }

        // let results =js.join_all().await;

        let mut failed = AtomicU8::new(0);

        let mut successes = 0u32;
        while let Some(res) = js.join_next().await {
            match res? {
                Ok(_) => {
                    successes += 1;
                }
                Err(e) => {
                    failed.fetch_add(1, Ordering::Relaxed);
                    error!("Find motifs failed: {e:?}");
                }
            }
        }

        info!(
            "failed {} out of {}",
            failed.load(Ordering::Relaxed),
            self.run_config_stages.len()
        );

        let cache_area = ctx.read().cache_area();
        let num_devices = GlobalCtx::global().device_count();

        let result = FindMotifsSavableResult::new(Arc::new(
            self.run_config_stages[best].run_config.clone(),
        ));

        Ok(Arc::new(DataCacheWrapper::new(
            result,
            0,
            num_devices,
            cache_area,
        )))
    }
}

#[cfg(test)]
mod tests {
    use crate::cli::{import_resources, run_grid};
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_motifs::FindMotifsStage;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use caching_runner::common::RunCtx;
    use caching_runner::common::cachable::DataCacheWrapper;
    use caching_runner::common::runnable::{AsyncStage, Stage};
    use caching_runner::global_ctx::GlobalCtx;
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use tch::Device::Cuda;
    use tch::{Device, Tensor};
    use tch::kind::INT64_CPU;
    use caching_runner::common::cachable::Cached;
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::find_motifs_run_config::{
        FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue,
    };
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::Simple;
    use thoosa_common::utils::{lin_t, plot_series, three_sigs_tensor, vec_f32_from};
    use thoosa_common::ws::FsWorkspace;
    use thoosa_motifs::find::{plot_motif_slices, FixedFindMotifsConfig};
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{DataId, SimpleId};
    use thoosa_shared::{
        cdt_for_dir, sig_one_imp, sig_three_imp, sig_two_imp, ResourceType, ThoosaError,
    };
    use thoosa_store::SqlStore;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::average_euclidean_distance::AverageEuclideanDistanceStage;
    use crate::tasks::load_motif_and_clc_dist::{LoadMotifAndClcDistanceStage};
    use crate::tasks::motif_loader::{wrap_data_id_for_cache, MotifLoader};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn test_find_motifs() -> Result<(), ThoosaError> {
        unsafe {
            let _lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");
        }

        // log::set_max_level(log::LevelFilter::Trace);
        GlobalCtx::init_devices(2);

        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        debug!("test_motifs: test_dir={test_dir:?}");

        fs::create_dir_all(&test_dir)?;
        let base_path = test_dir.to_string_lossy().to_string();

        let n = 1000;

        let ws = Arc::new(RwLock::new(FsWorkspace::new(
            test_dir.clone(),
            3,
            n as u32,
            32000,
        )?));
        let ws_rc = Arc::clone(&ws);
        let max_y = 30.0;

        let sig_0_v = sig_one_imp(n, max_y, None, None, 0.1);
        let sig_1_v = sig_two_imp(n, max_y, None, None, 0.1);
        let sig_2_v = sig_three_imp(n, max_y, None, None);

        let x = lin_t(n);
        plot_series(
            vec![
                (
                    format!("Sig0"),
                    plotters::prelude::RED,
                    vec_f32_from(&x)
                        .into_iter()
                        .zip(sig_0_v.clone())
                        .collect::<Vec<(f32, f32)>>(),
                ),
                (
                    format!("Sig1"),
                    plotters::prelude::GREEN,
                    vec_f32_from(&x)
                        .into_iter()
                        .zip(sig_1_v.clone())
                        .collect::<Vec<(f32, f32)>>(),
                ),
                (
                    format!("Sig2"),
                    plotters::prelude::BLACK,
                    vec_f32_from(&x)
                        .into_iter()
                        .zip(sig_2_v.clone())
                        .collect::<Vec<(f32, f32)>>(),
                ),
            ],
            "test_find_motifs_euc",
            (2000, 1000),
            &format!("/tmp/plots/0_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str(),
        )?;

        let (t_0_0, t_0_1, t_1_0, t_1_1) = three_sigs_tensor(n, &sig_0_v, &sig_1_v, &sig_2_v);
        debug!(
            "t_0_0.size()={:?}, t_0_1.size()={:?}, t_1_0.size()={:?}, t_1_1.size()={:?}",
            t_0_0.size(),
            t_0_1.size(),
            t_1_0.size(),
            t_1_1.size()
        );

        t_0_0.save(ws.read().tensors_path().join(format!("t_0_0.pt")))?;
        t_0_1.save(ws.read().tensors_path().join(format!("t_0_1.pt")))?;
        t_1_0.save(ws.read().tensors_path().join(format!("t_1_0.pt")))?;
        t_1_1.save(ws.read().tensors_path().join(format!("t_1_1.pt")))?;

        // debug!("t_1_1 mean={} max-min={}", t_1_1.mean(None).double_value(&[]), t_1_1.max() - t_1_1.min());
        let channels = 3;
        let tensors_path = ws.read().tensors_path().clone();
        let tensors_path_for_import = ws.read().tensors_path().clone();

        let mut store = Arc::new(RwLock::new(SqlStore::from_path(ws.read().store_path())?));

        let imported_ids = import_resources(
            &ws,
            &store,
            &["label_0", "label_1"],
            || {
                Ok(HashMap::from([
                    ("t_0_0".to_string(), vec![("label_0".to_string(), 1.0)]),
                    ("t_0_1".to_string(), vec![("label_0".to_string(), 1.0)]),
                    ("t_1_0".to_string(), vec![("label_1".to_string(), 1.0)]),
                    ("t_1_1".to_string(), vec![("label_1".to_string(), 1.0)]),
                ]))
            },
            // None::<Box<dyn Fn(String) -> Result<Tensor, ThoosaError> + Sync + Send>>,
            |key| {
                Tensor::load(tensors_path_for_import.join(format!("{}.pt", key)))
                    .map_err(|e| ThoosaError::new(e.to_string()))
            },
            true,
        )?;

        let resources = store.read().list_resources_by_ids(&imported_ids)?;

        let resources_0_0 = resources
            .iter()
            .find(|r| r.file_stem().eq("t_0_0"))
            .expect("t_0_0");
        let resources_0_1 = resources
            .iter()
            .find(|r| r.file_stem().eq("t_0_1"))
            .expect("t_0_1");
        let resources_1_0 = resources
            .iter()
            .find(|r| r.file_stem().eq("t_1_0"))
            .expect("t_1_0");
        let resources_1_1 = resources
            .iter()
            .find(|r| r.file_stem().eq("t_1_1"))
            .expect("t_1_1");

        debug!("test_motifs: resources_0_0={resources_0_0:?}");

        let loader = Arc::new(RawTensorLoader::new(
            move |key, _| {
                ThoTensor2::new(
                    Input::id(key).clone(),
                    Tensor::load(tensors_path.join(format!("{}.pt", key.file_stem()))).unwrap(),
                )
            },
            ThoDevice::from(Device::Cpu),
        ));

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let cache_area = ctx.read().cache_area();
        let resources_input = Arc::new(DataCacheWrapper::new(
            ResourcesConfigInput::new(
                SimpleId::from("resources"),
                Simple::new(ResourceType::List(
                    vec!["t_0_0", "t_0_1", "t_1_0", "t_1_1"]
                        .into_iter()
                        .map(|v| v.to_string())
                        .collect(),
                )),
            ),
            0,
            GlobalCtx::global().device_count(),
            cache_area,
        ));

        let run_ids = run_grid::<Simple, _, _, _, _, _>(
            None,
            serde_json5::from_str::<FindMotifsRunConfigRangeOrValue>(
                &r#"{
                step:"100",
                window:"1000",
                slices_filter: {'DistanceRunningMeanK':{'mean':'0.2'}}

            }"#,
            )
            .unwrap(),
            resources_input,
            &store,
            true,
            |run_configs| {
                Arc::new(FindMotifsStage::new(
                    &ws,
                    &store,
                    Arc::clone(&loader),
                    run_configs,
                    FixedFindMotifsConfig::new(None, 2, false, None),
                    ThoDevice::from(0),
                ))
            },
            || Arc::clone(&ctx),
        )
        .await?;

        let motif_loader_impl = Arc::new(MotifLoader::<RawTensorLoader, RunCtx2>::new(
            Arc::clone(&store),
            Arc::clone(&loader)
        ));

        let clc_dist = Arc::new(LoadMotifAndClcDistanceStage::new(
            Arc::clone(&motif_loader_impl),
            Arc::new(AverageEuclideanDistanceStage::new())
        ));

        let motifs = store.read().list_motifs(run_ids[0])?;

        for motif_id_raw in motifs {
            let cache_area_loop = ctx.read().cache_area();
            let num_devices_loop = GlobalCtx::global().device_count();
            let motif_id_value = DataId::from(&(motif_id_raw as u32));
            let motif_id = wrap_data_id_for_cache(
                motif_id_value.clone(),
                num_devices_loop,
                cache_area_loop,
            );

            let motif = Arc::clone(&motif_loader_impl).run(
                Arc::clone(&motif_id),
                Arc::clone(&ctx),
            )
            .await?;

            motif
                .with_data_ref(1, None, |t| {
                    debug!("motif_id={motif_id_raw} slices={:?}", t.data());

                    Ok(())
                })
                .await?;

            let _dist = Arc::clone(&clc_dist)
                .run_cached(Arc::clone(&motif_id), Arc::clone(&ctx))
                .await?
                .with_data_ref(0, None, |dist_val| Ok(dist_val.value()))
                .await?;
        }

        // let mut slices = store.read().list_motifs_slices(run_ids[0])?;
        //
        // debug!("slices={slices:?}");
        //
        // assert!(slices.len() >= 4);

        Ok(())
    }

}
