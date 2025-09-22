use std::fmt::Debug;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use tch::Tensor;
use caching_runner::common::runnable::Stage;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_motifs::find::FindMotifsSavableResult;
use thoosa_shared::ThoosaError;
use crate::tasks::cross_batch_clc::{CrossBatchClc, CrossBatchClcOutput};
use crate::tasks::cross_loader::CrossLoader;
use crate::tasks::dists_tracker::{DistWithMask, DistsTracker};
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::TensorLoader;

#[derive(Debug)]
struct FindMotifs<F>
where  F: Fn(Tensor, Tensor, bool) -> Result<CrossBatchClcOutput<DistWithMask>, ThoosaError>
{
    cross_loader: CrossLoader<TensorLoader, CrossBatchClc<F, CrossBatchClcOutput<DistWithMask>>, DistsTracker>
}

#[async_trait]
impl <F> Stage for FindMotifs<F>
where
    F: Fn(Tensor, Tensor, bool) -> Result<CrossBatchClcOutput<DistWithMask>, ThoosaError> + Send + Sync + Debug + 'static
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



