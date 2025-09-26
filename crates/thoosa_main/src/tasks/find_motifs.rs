use crate::tasks::cross_batch_clc::{BatchInput, CrossBatchClc, CrossBatchClcOutput, UnfoldParams};
use crate::tasks::cross_loader::{
    CrossLoader, CrossResult, CrossTensorLoader, ResourcesConfigInput,
};
use crate::tasks::dists_tracker::{DistWithMask, DistsTracker};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::TensorLoader;
use async_trait::async_trait;
use caching_runner::common::RunCtx;
use caching_runner::common::runnable::Stage;
use futures_util::FutureExt;
use log::debug;
use parking_lot::{Mutex, RwLock};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Kind::Float;
use tch::{Device, Kind, Tensor};
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
use tokio::task::JoinSet;

#[derive(Debug)]
struct FindMotifsStages<RC, SRC>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    SRC: CrossTensorLoader + 'static,
{
    run_config: FindMotifsRunConfig,
    cross_loader: Arc<
        CrossLoader<
            RC,
            SRC,
            CrossBatchClc<
                Box<
                    dyn Fn(
                            Tensor,
                            Tensor,
                            &BatchInput<&Arc<Resource>>,
                            &MeanDistsCtx,
                        ) -> Result<DistWithMask, ThoosaError>
                        + Send
                        + Sync,
                >,
                DistWithMask,
                MeanDistsCtx,
            >,
            DistsTracker,
        >,
    >,
}

impl<RC, SRC> FindMotifsStages<RC, SRC>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    SRC: CrossTensorLoader + 'static,
{
    pub fn new(
        run_config: FindMotifsRunConfig,
        cross_loader: Arc<
            CrossLoader<
                RC,
                SRC,
                CrossBatchClc<
                    Box<
                        dyn Fn(
                                Tensor,
                                Tensor,
                                &BatchInput<&Arc<Resource>>,
                                &MeanDistsCtx,
                            ) -> Result<DistWithMask, ThoosaError>
                            + Send
                            + Sync,
                    >,
                    DistWithMask,
                    MeanDistsCtx,
                >,
                DistsTracker,
            >,
        >,
    ) -> Self {
        Self {
            run_config,
            cross_loader,
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
pub struct FindMotifsStage<ResConf, SRC>
where
    ResConf: ResourcesConfig<R = Vec<Resource>> + Debug,
    SRC: CrossTensorLoader + 'static + Debug,
{
    run_config_stages: Vec<FindMotifsStages<ResConf, SRC>>,
    pd: PhantomData<(ResConf, SRC)>,
}

impl<RC, SRC> FindMotifsStage<RC, SRC>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Debug,
    SRC: CrossTensorLoader + 'static,
{
    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        loader: Arc<SRC>,
        run_configs: Vec<FindMotifsRunConfig>,
        fixed_config: FixedFindMotifsConfig,
        device: ThoDevice, //TODO remove, make MeanDistsCtx cachable in DataCacheWrapper
    ) -> Self {
        let cross_loaders = run_configs
            .into_iter()
            .map(|run_config| {
                let channels = ws.read().channels();

                let filter = run_config.slices_filter.clone();

                let batch_clc: CrossBatchClc<
                    Box<
                        dyn Fn(
                                Tensor,
                                Tensor,
                                &BatchInput<&Arc<Resource>>,
                                &MeanDistsCtx,
                            ) -> Result<DistWithMask, ThoosaError>
                            + Send
                            + Sync,
                    >,
                    DistWithMask,
                    MeanDistsCtx,
                > = CrossBatchClc::new(
                    Box::new(move |t_0, t_1, input, ctx| {
                        let is_diag = input.batch_0()[0].id().eq(&input.batch_1()[0].id()); //TODO asserts, non-square crosses

                        let mut mask_makers = match &filter {
                            SlicesFilter::DistanceRunningMeanK(v) => {
                                vec![
                                    LocalMin,
                                    LeRunningMean {
                                        k: v.mean,
                                        avg: &ctx.means_acc,
                                        total_samples: &ctx.total_samples,
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

                        let dists = thoosa_motifs::distance::clc_dist_d2_and_mask(
                            t_0,
                            t_1,
                            true,
                            &mut mask_makers,
                        )?;

                        debug!("CrossBatchClc::do_run:  dists={dists:?}");
                        Ok(tokio::task::block_in_place(|| {
                            tokio::runtime::Handle::current().block_on(async {
                                DistWithMask::new(
                                    CopyCtx::instance().copy(&dists.0, 0, Kind::Bool).await,
                                    match dists.1 {
                                        None => None,
                                        Some(t) => {
                                            Some(CopyCtx::instance().copy(&t, 0, Kind::Bool).await)
                                        }
                                    },
                                )
                            })
                        }))
                        // Ok(DistWithMask::new(dists.0, dists.1))
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

                FindMotifsStages::new(
                    run_config,
                    Arc::new(CrossLoader::new(
                        Arc::clone(&loader),
                        &store,
                        Arc::new(batch_clc),
                        receiver,
                        2,
                        20,
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
//
// impl <RC,SRC> From<(Arc<RwLock<Box<dyn Workspace>>>, Arc<RwLock<Box<dyn Store>>>, Arc<SRC>, Vec<FindMotifsRunConfig>, FixedFindMotifsConfig, Arc<RwLock<RunCtx2>>)> for FindMotifsStage<RC,SRC>
// where
//     RC: ResourcesConfig<R = Vec<Resource>> + Debug,
//     SRC: CrossTensorLoader + 'static,
// {
//     fn from(value: (Arc<RwLock<Box<dyn Workspace>>>, Arc<RwLock<Box<dyn Store>>>, Arc<SRC>, Vec<FindMotifsRunConfig>, FixedFindMotifsConfig, Arc<RwLock<RunCtx2>>)) -> Self {
//         let (ws, store, loader, run_config, fixed_config, ctx) = value;
//         FindMotifsStage::new(&ws, &store, loader,  run_config, fixed_config, ThoDevice::from(ctx.read().device_idx()))
//     }
// }

#[async_trait]
impl<RC, SRC> Stage for FindMotifsStage<RC, SRC>
where
    RC: ResourcesConfig<R = Vec<Resource>> + Send + Sync + Debug + 'static,
    SRC: CrossTensorLoader + 'static,
{
    type Ctx = RunCtx2;
    type In<'a>
        = ResourcesConfigInput<RC>
    where
        Self: 'a;
    type Out = FindMotifsSavableResult;

    async fn run(
        self: Arc<Self>,
        input: Arc<ResourcesConfigInput<RC>>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError> {
        let best = 0;
        let mut js: JoinSet<Result<CrossResult, ThoosaError>> = JoinSet::new();

        for stage in self.run_config_stages.iter() {
            let ctx_arc = Arc::clone(&ctx);
            let input_arc = Arc::clone(&input);
            let stage_arc = Arc::clone(&stage.cross_loader);

            js.spawn(async {
                let out = stage_arc.run(input_arc, ctx_arc).await?;

                Ok(out)
            });
        }

        let results = js.join_all().await;

        Ok(FindMotifsSavableResult::new(Arc::new(
            self.run_config_stages[best].run_config.clone(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use crate::cli::{import_resources, run_grid};
    use crate::tasks::cross_loader::ResourcesConfigInput;
    use crate::tasks::find_motifs::FindMotifsStage;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::TensorLoader;
    use caching_runner::common::RunCtx;
    use caching_runner::common::runnable::Stage;
    use caching_runner::global_ctx::GlobalCtx;
    use log::debug;
    use parking_lot::RwLock;
    use std::collections::HashMap;
    use std::fs;
    use std::path::Path;
    use std::sync::Arc;
    use tch::Device::Cuda;
    use tch::{Device, Tensor};
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::find_motifs_run_config::{
        FindMotifsRunConfig, FindMotifsRunConfigRangeOrValue,
    };
    use thoosa_common::loader::{LoadRunner, SyncWrapper};
    use thoosa_common::runs::Simple;
    use thoosa_common::utils::{lin_t, plot_series, three_sigs_tensor, vec_f32_from};
    use thoosa_common::ws::FsWorkspace;
    use thoosa_motifs::find::FixedFindMotifsConfig;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::SimpleId;
    use thoosa_shared::{
        ResourceType, ThoosaError, cdt_for_dir, sig_one_imp, sig_three_imp, sig_two_imp,
    };
    use thoosa_store::SqlStore;
    use torch_runner::data::ThoTensor2;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError> {
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

        let loader = Arc::new(TensorLoader::new(
            move |key, _| {
                ThoTensor2::new(
                    Input::id(key).clone(),
                    Tensor::load(tensors_path.join(format!("{}.pt", key.file_stem()))).unwrap(),
                )
            },
            ThoDevice::from(Device::Cpu),
        ));

        let run_ids = run_grid::<Simple, _, _, _, _, _>(
            None,
            serde_json5::from_str::<FindMotifsRunConfigRangeOrValue>(
                &r#"{
                step:"200",
                window:"1000",
                slices_filter: {'DistanceRunningMeanK':{'mean':'0.2'}}

            }"#,
            )
            .unwrap(),
            Arc::new(ResourcesConfigInput::new(
                SimpleId::from("resources"),
                Simple::new(ResourceType::List(
                    vec!["t_0_0", "t_0_1", "t_1_0", "t_1_1"]
                        .into_iter()
                        .map(|v| v.to_string())
                        .collect(),
                )),
            )),
            &store,
            true,
            |run_configs| {
                Arc::new(FindMotifsStage::new(
                    &ws,
                    &store,
                    loader,
                    run_configs,
                    FixedFindMotifsConfig::new(None, 2, false, None),
                    ThoDevice::from(0),
                ))
            },
            || Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa"))),
        )
        .await?;

        let mut slices = store.read().list_motifs_slices(run_ids[0])?;

        debug!("slices={slices:?}");

        assert!(slices.len() >= 4);

        Ok(())
    }
}
