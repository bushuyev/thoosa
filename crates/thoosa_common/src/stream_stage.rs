use std::sync::Arc;
use thoosa_shared::runner::common::RunCtx;
use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
use thoosa_shared::ThoError;
use crate::cuda_utils::with_cuda_stream;

#[derive(Debug)]
pub struct StreamStage<S>
where
 S:SyncStage
{
    stage:Arc<S>
}

impl <S> StreamStage<S>
where
    S:SyncStage
{
    pub fn new(stage: Arc<S>) -> Self {
        Self { stage }
    }
}

impl <S> Stage for StreamStage<S>
where
    S:SyncStage
{
    type Ctx = S::Ctx;
    type In<'a> = S::In<'a>;
    type Out = S::Out;
}

impl <S> SyncStage for StreamStage<S>
where
    S:SyncStage
{
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {

        let (out, stream) = with_cuda_stream(ctx.device_idx(), move ||{
            Arc::clone(&self.stage).do_run(input, ctx)
        });
        if let Some(stream) = stream {
            stream.synchronize().map_err(|msg|ThoError::new(msg))?;
        }

        out
    }
}

impl <S> AsRef<S> for StreamStage<S>
where
    S:SyncStage
{
    fn as_ref(&self) -> &S {
        self.stage.as_ref()
    }
}