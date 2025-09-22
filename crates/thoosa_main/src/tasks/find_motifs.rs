use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::Tensor;
use caching_runner::common::runnable::Stage;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_common::ws::Workspace;
use thoosa_motifs::distance::MaskMaker::Tril;
use thoosa_motifs::find::{FindMotifsSavableResult, FixedFindMotifsConfig};
use thoosa_shared::ThoosaError;
use crate::tasks::cross_batch_clc::{CrossBatchClc, CrossBatchClcOutput};
use crate::tasks::cross_loader::{CrossLoader, CrossLoaderTensorSource};
use crate::tasks::dists_tracker::{DistWithMask, DistsTracker};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::TensorLoader;

#[derive(Debug)]
struct FindMotifs<SRC>
where
    SRC: CrossLoaderTensorSource + 'static,
{
    cross_loader:  Arc<CrossLoader<SRC, CrossBatchClc<Box<dyn Fn(Tensor, Tensor, bool) -> Result<DistWithMask, ThoosaError> + Send + Sync>, DistWithMask>, DistsTracker>>,
    pd:PhantomData<SRC>
}

impl <SRC> FindMotifs<SRC>
where
    SRC: CrossLoaderTensorSource + 'static,
{
    pub fn new(ws: &Arc<RwLock<Box<dyn Workspace>>>, loader: Arc<SRC>, run_configs:Vec<FindMotifsRunConfig>, fixed_config:FixedFindMotifsConfig) -> Self {
        let batch_clc:CrossBatchClc<
            Box<dyn Fn(Tensor, Tensor, bool) -> Result<DistWithMask, ThoosaError> + Send + Sync>,
            DistWithMask
        > = CrossBatchClc::new(Box::new(|t_0, t_1, is_diag| {
            let mut mask_makers = vec![];
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
            Ok(DistWithMask::new(dists.0, dists.1))
        }));

        let receiver = Arc::new(DistsTracker::new_2(&ws, &run_configs[0], &fixed_config, None));

        Self { cross_loader: Arc::new(CrossLoader::new(loader, Arc::new(batch_clc), receiver, 2, 1, 20)), pd: Default::default() }
    }
}

#[async_trait]
impl <SRC> Stage for FindMotifs<SRC>
where
    SRC: CrossLoaderTensorSource + 'static,
{
    type Ctx = RunCtx2;
    type In<'a> = FindMotifsRunConfig;
    type Out = FindMotifsSavableResult;

    async fn run(self: Arc<Self>, input: Arc<Self::In<'static>>, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
       Ok(FindMotifsSavableResult::new(input))//just dummy result for now
    }
}

#[cfg(test)]
mod tests {
    use thoosa_shared::ThoosaError;

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError>{

        Ok(())
    }
}



