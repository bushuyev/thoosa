use cached_derive::Cached;
use caching_runner::common::runnable::Stage;
use caching_runner::common::{Data, OutId, RunCtx};
use std::fmt::Debug;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use caching_runner::common::cachable::Cached;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::WithId;

#[derive(Debug, Cached)]
struct StagesSeq<I, M, O, C, SIn, SOut>
where
    I: Input + Debug + WithId + 'static,
    M: Input + Output + Data+ Debug + Send + Sync+ 'static,
    O: Output + Debug + Data + Send + Sync + 'static,
    C: RunCtx + Debug + 'static,
    for<'a> SIn: Stage<In<'a> = I, Out = M, Ctx = C> + OutId + Cached + Send + Sync + 'a,
    for<'a> SOut: Stage<In<'a> = M, Out = O, Ctx = C> + OutId + Cached + Send + Sync +'a,
{
    stage_in: Arc<SIn>,
    stage_out: Arc<SOut>,
}

#[async_trait]
impl<I, M, O, C, SIn, SOut> Stage for StagesSeq<I, M, O, C, SIn, SOut>
where
    I: Input + Debug + WithId + 'static,
    M: Input + Output + Data+ Debug + Send + Sync + 'static,
    O: Output + Debug + Data + Send + Sync + 'static,
    C: RunCtx + Debug + 'static,
    for<'a> SIn: Stage<In<'a> = I, Out = M, Ctx = C>  + OutId+ Cached+ Send + Sync+ 'a,
    for<'a> SOut: Stage<In<'a> = M, Out = O, Ctx = C> + OutId+ Cached + Send + Sync + 'a,
{
    type Ctx = C;
    type In<'a>
        = I
    where
        I: 'a,
        M: 'a,
        O: 'a,
        C: 'a;
    type Out = O;

    async fn run(self: Arc<Self>, _input: &'static Self::In<'static>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
        // let m = self.stage_in.run_cached(_input, Arc::clone(&_ctx)).await?;
        //
        //
        // self.stage_in.run_cached(m, Arc::clone(&_ctx)).await
        todo!()

    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_0() {}
}
