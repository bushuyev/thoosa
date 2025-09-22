use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use caching_runner::common::runnable::Stage;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_motifs::find::FindMotifsSavableResult;
use thoosa_shared::ThoosaError;
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
struct FindMotifs {

}

#[async_trait]
impl Stage for FindMotifs {
    type Ctx = RunCtx2;
    type In<'a> = FindMotifsRunConfig;
    type Out = FindMotifsSavableResult;

    async fn run(self: Arc<Self>, _input: Arc<Self::In<'static>>, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
        todo!()
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



