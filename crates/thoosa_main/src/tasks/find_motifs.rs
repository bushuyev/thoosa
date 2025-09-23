use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::{Mutex, RwLock};
use tch::Kind::Float;
use tch::{Kind, Tensor};
use caching_runner::common::runnable::Stage;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig, SlicesFilter};
use thoosa_common::ws::Workspace;
use thoosa_motifs::distance::MaskMaker::{Le, LeRunningMean, LocalMin, Tril};
use thoosa_motifs::find::{FindMotifsSavableResult, FixedFindMotifsConfig};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use crate::tasks::cross_batch_clc::{BatchInput, CrossBatchClc, CrossBatchClcOutput};
use crate::tasks::cross_loader::{CrossLoader, CrossLoaderTensorSource, ResourcesVec};
use crate::tasks::dists_tracker::{DistWithMask, DistsTracker};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::TensorLoader;

#[derive(Debug)]
struct FindMotifsCtx<SRC>
where
    SRC: CrossLoaderTensorSource + 'static
{
    run_config:FindMotifsRunConfig,
    cross_loader:Arc<CrossLoader<SRC, CrossBatchClc<Box<dyn Fn(Tensor, Tensor, &BatchInput<&Arc<Resource>>) -> Result<DistWithMask, ThoosaError> + Send + Sync>, DistWithMask>, DistsTracker>>,
    means_acc: Mutex<Tensor>,
    total_samples: Mutex<Tensor>,
}

impl <SRC> FindMotifsCtx <SRC>
where
    SRC: CrossLoaderTensorSource + 'static
{
    pub fn new(
        run_config: FindMotifsRunConfig,
        cross_loader: Arc<CrossLoader<SRC, CrossBatchClc<Box<dyn Fn(Tensor, Tensor, &BatchInput<&Arc<Resource>>) -> Result<DistWithMask, ThoosaError> + Send + Sync>, DistWithMask>, DistsTracker>>,
        means_acc: Mutex<Tensor>,
        total_samples: Mutex<Tensor>,
    ) -> Self {
        Self {
            run_config,
            cross_loader,
            means_acc,
            total_samples,
        }
    }
}

#[derive(Debug)]
pub struct FindMotifsStage<SRC>
where
    SRC: CrossLoaderTensorSource + 'static,
{
    cross_loaders:  Vec<FindMotifsCtx<SRC>>,
    pd:PhantomData<SRC>,
}

impl <SRC> FindMotifsStage<SRC>
where
    SRC: CrossLoaderTensorSource + 'static,
{
    pub fn new(ws: &Arc<RwLock<Box<dyn Workspace>>>, loader: Arc<SRC>, run_configs:Vec<FindMotifsRunConfig>, fixed_config:FixedFindMotifsConfig, device:ThoDevice) -> Self {//TODO remove device, make means_acc & total_samples cachable stage?

        let cross_loaders = run_configs.into_iter().map(|run_config|{
            let channels = ws.read().channels();
            let device = device.clone();

            let means_acc =  Mutex::new(Tensor::zeros(
                [channels as i64, 1],
            (Kind::Float, device.clone().into()),
            ));
            let total_samples = Mutex::new(Tensor::zeros(
                [channels as i64, 1],
            (Kind::Int64, device.clone().into()),
            ));
            let filter = run_config.slices_filter.clone();

            let batch_clc:CrossBatchClc<
                Box<dyn Fn(Tensor, Tensor, &BatchInput<&Arc<Resource>>) -> Result<DistWithMask, ThoosaError> + Send + Sync>,
                DistWithMask
            > = CrossBatchClc::new(Box::new( move |t_0, t_1, is_diag| {

                let mut mask_makers = vec![];

                // let mut mask_makers = match &filter {
                //     SlicesFilter::DistanceRunningMeanK(v) => {
                //         vec![
                //             LocalMin,
                //             // LeRunningMean {
                //             //     k: v.mean,
                //             //     avg: &means_acc,
                //             //     total_samples: &total_samples,
                //             // },
                //         ]
                //     }
                //     SlicesFilter::MaxDistance(v) => {
                //         vec![Le {
                //             v: Tensor::full(
                //                 &[channels as i64, 1],
                //                 v.distance as f64,
                //                 (Float, device.clone().into()),
                //             ),
                //         }]
                //     }
                // };
                //
                //
                // if is_diag {
                //     mask_makers.push(Tril);
                // }
                let dists = thoosa_motifs::distance::clc_dist_d2_and_mask(
                    t_0,
                    t_1,
                    true,
                    &mut mask_makers,
                )?;

                debug!("CrossBatchClc::do_run:  dists={dists:?}");
                Ok(DistWithMask::new(dists.0, dists.1))
            }));

            let receiver = Arc::new(DistsTracker::new_2(&ws, &run_config, &fixed_config, None));

            let arc = Arc::new(CrossLoader::new(Arc::clone(&loader), Arc::new(batch_clc), receiver, 2, 20));
            FindMotifsCtx::new(run_config, arc, means_acc, total_samples)
        }).collect();


        Self { cross_loaders, pd: Default::default() }
    }
}

#[async_trait]
impl <SRC> Stage for FindMotifsStage<SRC>
where
    SRC: CrossLoaderTensorSource + 'static,
{
    type Ctx = RunCtx2;
    type In<'a> = ResourcesVec
    where
        Self: 'a;
    type Out = FindMotifsSavableResult;

    async fn run(self: Arc<Self>, input: Arc<Self::In<'static>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
        let best = 0;
        let out = Arc::clone(&self.cross_loaders[0].cross_loader).run(input, ctx).await?;

        Ok(FindMotifsSavableResult::new(Arc::new(self.cross_loaders[best].run_config.clone())))//just dummy result for now
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use log::debug;
    use parking_lot::RwLock;
    use tch::{Device, Tensor};
    use tch::Device::Cuda;
    use caching_runner::common::runnable::Stage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::ws::FsWorkspace;
    use thoosa_motifs::find::FixedFindMotifsConfig;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::{cdt_for_dir, ThoosaError};
    use torch_runner::data::ThoTensor2;
    use crate::tasks::cross_loader::ResourcesVec;
    use crate::tasks::find_motifs::FindMotifsStage;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::TensorLoader;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError>{
        GlobalCtx::init_devices(2);

        let test_dir = Path::new("/tmp/").join(cdt_for_dir());

        let resources: Vec<Resource> = vec!["111", "222", "333", "444", "555", "666"]
            .into_iter()
            .enumerate()
            .map(|(i, v)| (i as u32, 1u32, v, 10).into())
            .collect();
        let resources_to_move = resources.clone();

        let loader = Arc::new(TensorLoader::new(
            move |key, _| {
                let i = resources_to_move
                    .iter()
                    .position(|v| v.eq(key))
                    .expect(format!("no {key:?}").as_str())
                    + 1;
                let v = (0..10)
                    .map(|v| i as f32 + v as f32 / 10.)
                    .collect::<Vec<f32>>();
                debug!("executor: key={key:?} i={i} v={v:?}");

                ThoTensor2::new(
                    Input::id(key).clone(),
                    Tensor::from_slice(v.as_slice()).view((1, 10)),
                )
            },
            ThoDevice::from(Device::Cpu),
        ));

        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, 1, 1000, 32000)?));

        let run_config = FindMotifsRunConfig::new(Some(0), 1, 2, None, None);
        let fixed_config = FixedFindMotifsConfig::new(None, 2, false, None);

        let find_motifs = Arc::new(FindMotifsStage::new(&ws, loader,  vec![run_config], fixed_config, ThoDevice::from(Cuda(0))));


        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let _ = find_motifs
            .run(Arc::new(ResourcesVec::new("".into(), resources)), ctx)
            .await?;

        Ok(())
    }
}



