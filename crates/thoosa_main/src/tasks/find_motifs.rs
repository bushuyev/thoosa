use crate::tasks::cross_batch_clc::{CrossInput, CrossUnfoldClc, CrossUnfoldClcOutput, UnfoldParams};
use crate::tasks::cross_loader::{
    CrossLoader, CrossResult, ResourcesConfigInput,
};
use crate::tasks::dists_tracker::{DistsMask, DistsTracker};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::RawTensorLoader;
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
use thoosa_common::applicables::{with_cuda_stream, ThoDevice};
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
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_common::cuda_utils::pack_bits_bool;
use thoosa_shared::traits::{Data, SimpleId, WithId};
use thoosa_shared::data_with_extra::DataWithExtra;
use crate::tasks::average_euclidean_distance::AverageDistanceValue;
use crate::tasks::motif_loader::MotifMeta;
use thoosa_shims::stream::{ThoStreamGuard, ThoCudaStream};
use thoosa_common::tho_tensor::ThoTensor;
use crate::tasks::cross_dims::{ChannelsData2, MotifSlices};
use crate::tasks::transforms::freq_filter::FreqFilter;
use crate::tasks::stages_seq::{ABStage, StageChainOption};

pub type FilteredLoaderStage<L, C> = ABStage<L, FreqFilter<C>, C>;
pub type LoaderStageChoice<L, C> = StageChainOption<L, FilteredLoaderStage<L, C>>;


#[derive(Debug)]
struct FindMotifsStages<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for <'i> Cached<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C>  + OutId + 'static,
    C: RunCtx + 'static,
{
    run_config: FindMotifsRunConfig,
    cross_loader: Arc<CrossLoader<
            RC,
        L,
            CrossUnfoldClc<Box<dyn Fn(Tensor, Tensor, &CrossInput<&Arc<Resource>>, &MeanDistsCtx, &Arc<RwLock<C>>) -> Result<DistsMask, ThoosaError> + Send + Sync>, DistsMask, MeanDistsCtx, C>,
            DistsTracker<C>,
            DistsMask,
            C,
        >>
}

impl<RC, L, C> FindMotifsStages<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for <'i> Cached<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C>  + OutId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(
        run_config: FindMotifsRunConfig,
        cross_loader: Arc<CrossLoader<
            RC,
            L,
            CrossUnfoldClc<Box<dyn Fn(Tensor, Tensor, &CrossInput<&Arc<Resource>>, &MeanDistsCtx, &Arc<RwLock<C>>) -> Result<DistsMask, ThoosaError> + Send + Sync>, DistsMask, MeanDistsCtx, C>,
            DistsTracker<C>,
            DistsMask,
            C,
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
pub struct FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    run_config_stages: Vec<FindMotifsStages<RC, LoaderStageChoice<L, C>, C>>,
    pd: PhantomData<(RC, L, C)>,
}

impl<RC, L, C> FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C>  + Cached + OutId + Send + Sync + 'static,
    C: RunCtx + 'static,
{
    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader: Arc<L>,
        run_configs: Vec<FindMotifsRunConfig>,
        fixed_config: FixedFindMotifsConfig,
        device: ThoDevice, //TODO remove, make MeanDistsCtx cachable in DataCacheWrapper
    ) -> Self {
        // let band_filter = Arc::new(FreqFilter::new());
        debug!("FindMotifsStage::new: fixed_config={fixed_config:?}");

        let cross_loaders = run_configs
            .into_iter()
            .map(|run_config| {
                debug!("FindMotifsStage::new: run_config={run_config:?}");
                let channels = ws.read().channels();

                let filter = run_config.slices_filter.clone();

                let cross_unfold_clc: CrossUnfoldClc<
                    Box<dyn Fn(Tensor, Tensor, &CrossInput<&Arc<Resource>>, &MeanDistsCtx, &Arc<RwLock<C>>) -> Result<DistsMask, ThoosaError> + Send + Sync>,
                    DistsMask,
                    MeanDistsCtx,
                    C,
                > = CrossUnfoldClc::new(

                    Box::new(move |t_0, t_1, input, means_ctx, ctx| {
                        let device_idx = ctx.read().device_idx();
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

                        //copy to device is done on streams created in CopyCtx and used in DistsMask::new_async. Should wait for calculation stream to finish.


                        let ((mask, dists), motifs_stream) = with_cuda_stream(device_idx, ||{
                            thoosa_motifs::distance::clc_dist_d2_and_mask(
                                t_0.shallow_clone(),
                                t_1.shallow_clone(),
                                fixed_config.keep_slice_matches,//if true dists are returned + slices are stored in sqlite
                                &mut mask_makers,
                            ).unwrap()
                        });

                        debug!("CrossBatchClc::do_run:  dists={dists:?}");
                        if device_idx > 0 {
                            Ok(DistsMask::new(mask, dists, true, motifs_stream))
                        } else {
                            Ok(DistsMask::new(mask, dists, false, None))
                        }

                    }),
                    MeanDistsCtx::new(channels, device.into()),
                    UnfoldParams::new(1, run_config.window as i64, run_config.step as i64),
                );

                let configured_loader = config_to_stage(&loader, &run_config);
                let run_config_clone = run_config.clone();

                FindMotifsStages::new(
                    run_config,
                    Arc::new(CrossLoader::new(
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
                                // let total_seq_len = resources.iter().map(|r|r.length() as u64).sum::<u64>();
                                Arc::new(DistsTracker::<C>::new(
                                    &ws,
                                    &store,
                                    &run_config,
                                    &fixed_config,
                                    Some(resources.len() as u32 * 10),//TODO. max_slices_count - if motif has more than ten slices in each resource it's probably noise
                                ))
                            }
                        },
                        fixed_config.max_chunk_size as usize,
                        |batch|{
                            async {batch.to_cpu().await.unwrap()}.boxed()//TODO unwrap
                        },
                        |receiver| {
                            let dists_tracker = Arc::clone(receiver);
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

impl<RC, L, C> Stage for FindMotifsStage<RC, L, C>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L, C>: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + 'static,
    C: RunCtx + 'static,
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
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + Send + Sync + 'static,
    LoaderStageChoice<L, C>: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> +  'static,
    C: RunCtx + 'static,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<ResourcesConfigInput<RC>>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let best = 0;
        let mut js: JoinSet<Result<Arc<DevicesWrap<CrossResult>>, ThoosaError>> = JoinSet::new();

        debug!("FindMotifsStage::run: run_config_stages.len={}", self.run_config_stages.len());
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

        info!("failed {} out of {}",failed.load(Ordering::Relaxed),self.run_config_stages.len());

        let cache_area = ctx.read().cache_area();
        let num_devices = GlobalCtx::global().device_count();

        let result = FindMotifsSavableResult::new(Arc::new(
            self.run_config_stages[best].run_config.clone(),
        ));

        Ok(Arc::new(DevicesWrap::wrap_cpu(result)))
    }
}


pub fn config_to_stage<L, C>(loader: &Arc<L>, find_motifs_run_config: &FindMotifsRunConfig) -> Arc<LoaderStageChoice<L, C>>
where
    C: 'static + RunCtx,
    L: 'static + Cached + OutId + for<'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C>
{
    let configurable_loader: Arc<LoaderStageChoice<L, C>> = match find_motifs_run_config.filter.as_ref() {
        None => {
            debug!("MotifLoader::run: find_motifs_run_id={:?} no filter", find_motifs_run_config.id);
            Arc::new(StageChainOption::<L, FilteredLoaderStage<L, C>>::Single(Arc::clone(loader)))
        },
        Some(filter_config) => {
            debug!("MotifLoader::run: find_motifs_run_id={:?} filter_config={filter_config:?}", find_motifs_run_config.id);

            Arc::new(StageChainOption::<L, FilteredLoaderStage<L, C>>::Chained(Arc::new(ABStage::new(
                SimpleId::from("ab"),
                Arc::clone(loader),
                Arc::new(FreqFilter::<C>::new(filter_config)),
            ))))
        },
    };
    configurable_loader
}


#[cfg(test)]
mod tests {
    use crate::cli::{import_resources, run_grid};
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_motifs::FindMotifsStage;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use caching_runner::common::{PermitsCtx, RunCtx};
    use caching_runner::common::cachable::{DevicesWrap, WithDataRef};
    use caching_runner::common::runnable::{AsyncStage, Stage};
    use caching_runner::global_ctx::{GlobalCtx};
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
    use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue, SelectMotifsRunConfigRangeOrValue};
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::{Labeled, Simple};
    use thoosa_common::utils::{lin_t, plot_series, three_sigs_tensor, vec_f32_from};
    use thoosa_common::ws::FsWorkspace;
    use thoosa_motifs::find::{plot_motif_slices, FixedFindMotifsConfig};
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{DataId, SimpleId};
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::{
        cdt_for_dir, sig_one_imp, sig_three_imp, sig_two_imp, ResourceType, ThoosaError,
    };
    use thoosa_store::SqlStore;
    use thoosa_common::tho_tensor::ThoTensor;
    use crate::tasks::average_euclidean_distance::AverageEuclideanDistanceStage;
    use crate::tasks::load_motif_and_clc_dist::LoadMotifAndClcDistanceStage;
    use crate::tasks::motif_loader::{MotifLoader, MotifLoaderInput, MotifMeta};
    use crate::tasks::cross_dims::MotifSlices;
    use crate::tasks::average_euclidean_distance::AverageDistanceValue;
    use crate::tasks::select_motifs::SelectMotifsStage;

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
        let global_ctx = GlobalCtx::init_devices(2);

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
        // plot_series(
        //     vec![
        //         (
        //             format!("Sig0"),
        //             plotters::prelude::RED,
        //             vec_f32_from(&x)
        //                 .into_iter()
        //                 .zip(sig_0_v.clone())
        //                 .collect::<Vec<(f32, f32)>>(),
        //         ),
        //         (
        //             format!("Sig1"),
        //             plotters::prelude::GREEN,
        //             vec_f32_from(&x)
        //                 .into_iter()
        //                 .zip(sig_1_v.clone())
        //                 .collect::<Vec<(f32, f32)>>(),
        //         ),
        //         (
        //             format!("Sig2"),
        //             plotters::prelude::BLACK,
        //             vec_f32_from(&x)
        //                 .into_iter()
        //                 .zip(sig_2_v.clone())
        //                 .collect::<Vec<(f32, f32)>>(),
        //         ),
        //     ],
        //     "test_find_motifs_euc",
        //     (2000, 1000),
        //     &format!("/tmp/plots/0_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str(),
        // )?;

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

        let loader = Arc::new(RawTensorLoader::<RunCtx2>::new(
            move |key, _| {
                ThoTensor::new(
                    Input::id(key).clone(),
                    Tensor::load(tensors_path.join(format!("{}.pt", key.file_stem()))).unwrap(),
                )
            },
            ThoDevice::from(Device::Cpu),
        ));

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));
        ctx.write().add_permit(global_ctx.acquire_many(0, 1, Arc::new(SimpleId("test"))).await?);

        let cache_area = ctx.read().cache_area();
        let resources_input = Arc::new(DevicesWrap::wrap_cpu(
            ResourcesConfigInput::new(
                SimpleId::from("resources"),
                Simple::new(ResourceType::List(
                    vec!["t_0_0", "t_0_1", "t_1_0", "t_1_1"]
                        .into_iter()
                        .map(|v| v.to_string())
                        .collect(),
                )),
            )
        ));

        let run_ids = run_grid::<_, _, _, _, _>(
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
            &ws,
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

        let motif_loader_impl = Arc::new(MotifLoader::<RawTensorLoader<RunCtx2>, RunCtx2>::new(
            Arc::clone(&store),
            Arc::clone(&loader)
        ));

        let clc_dist = Arc::new(LoadMotifAndClcDistanceStage::new(
            Arc::clone(&motif_loader_impl),
            Arc::new(AverageEuclideanDistanceStage::<RunCtx2>::new())
        ));

        let motifs = store.read().list_motifs(run_ids[0])?;

        for motif_id_raw in motifs {
            let cache_area_loop = ctx.read().cache_area();
            let num_devices_loop = GlobalCtx::global().device_count();
            let motif_id_value = DataId::from(&(motif_id_raw as u32));
            let motif_id = Arc::new(DevicesWrap::wrap_cpu(MotifLoaderInput::all_slices(motif_id_value.clone())));

            let motif = Arc::clone(&motif_loader_impl).run(
                Arc::clone(&motif_id),
                Arc::clone(&ctx),
            )
            .await?;

            motif
                .with_data_ref(1,
                Arc::clone(&ctx), None, |t: &DataWithExtra<ThoTensor<MotifSlices, DataId>, MotifMeta>| {
                   debug!("motif_id={motif_id_raw} slices={:?}", t.data());

                   Ok(())
               })
                .await?;

            let _dist = Arc::clone(&clc_dist)
                .run_cached(Arc::clone(&motif_id), Arc::clone(&ctx))
                .await?
                .with_data_ref(0, Arc::clone(&ctx), None, |dist_val: &AverageDistanceValue| Ok(dist_val.value()))
                .await?;
        }
        ////////////////Select
        let select_motifs_run_config_range_or_value: SelectMotifsRunConfigRangeOrValue =
            serde_json5::from_str(
            &r#"{
                motif_selector:{'ByLabelVariance': {"_dummy":0}}
            }"#
            ).expect("SelectMotifs2");

        let input_wrapper = Arc::new(DevicesWrap::wrap_cpu(
            ResourcesConfigInput::new(
                SimpleId::from("resources"),
                Labeled::new("test".to_string()),
            )
        ));

        // let run_ids = run_grid::</*Labeled,*/ _, _, _, _, _>(
        //     None,
        //     select_motifs_run_config_range_or_value,
        //     input_wrapper,
        //     &store,
        //     true,
        //     |run_configs| {
        //         Arc::new(SelectMotifsStage::<RawTensorLoader, RunCtx2>::new(
        //             &ws,
        //             &store,
        //             Arc::clone(&loader),
        //             run_configs,
        //         ))
        //     },
        //     || Arc::new(parking_lot::RwLock::new(RunCtx2::new(1, 0, "find motifs"))),
        // ).await?;

        Ok(())
    }

}
