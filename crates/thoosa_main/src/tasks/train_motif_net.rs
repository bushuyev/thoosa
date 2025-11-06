use std::marker::PhantomData;
use caching_runner::common::RunCtx;
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::train_nets::TrainMotifsNetsRunConfigResult;
use thoosa_shared::traits::{Adjustable, DataId};
use crate::tasks::run_ctx_2::MotifsCtx;

#[derive(Debug)]
struct TrainMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    pd:PhantomData<C>

}

impl <C> Stage for TrainMotifNet<C>
where
C: RunCtx + MotifsCtx + Adjustable + 'static{
    type Ctx = C;
    type In<'a> = DataId;//MotifID
    type Out = TrainMotifsNetsRunConfigResult;
}


// impl SyncStage

#[cfg(test)]
mod tests {
    use thoosa_shared::ThoosaError;

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError>{


        Ok(())
    }

}