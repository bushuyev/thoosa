use cached_derive::Cached;
use caching_runner::common::runnable::Stage;
use caching_runner::common::{Data, OutId, RunCtx};
use std::fmt::Debug;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use thoosa_shared::applicables_runner::*;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::traits::WithId;

#[derive(Debug)]
pub struct StagesSeq<SIn, SOut>
where
    SIn:Stage + Cached + OutId + Sync + Send + 'static,
    SOut: Stage<Ctx=SIn::Ctx> + Cached + OutId + Sync + Send + 'static,
    for <'a> SOut: Stage<In<'a>=DataCacheWrapper<SIn::Out>>,
    // SOut: Stage + Cached + Send + Sync + OutId + 'static,
    <SIn as Stage>::In<'static>: WithId,
    <SOut as Stage>::In<'static>: WithId,
    <SIn as Stage>::Out: CopyToDevice + Data +  Send + Sync + 'static,
    <SOut as Stage>::Out: CopyToDevice + Data + Send + Sync + 'static,
    // <<SOut as Stage>::Out as Stage>::In<'static>: WithId,
    for <'a> <SOut as Stage>::In<'a>: Send + Sync + 'static,
{
    stage_in: Arc<SIn>,
    stage_out: Arc<SOut>,
}

impl<SIn, SOut>  StagesSeq<SIn, SOut>
where
    SIn:Stage + Cached + OutId + Sync + Send + 'static,
    SOut: Stage<Ctx=SIn::Ctx> + Cached + OutId + Sync + Send + 'static,
    for <'a> SOut: Stage<In<'a>=DataCacheWrapper<SIn::Out>>,
    // SOut: Stage + Cached + Send + Sync + OutId + 'static,
    <SIn as Stage>::In<'static>: WithId,
    <SOut as Stage>::In<'static>: WithId,
    <SIn as Stage>::Out: CopyToDevice + Data +  Send + Sync + 'static,
    <SOut as Stage>::Out: CopyToDevice + Data + Send + Sync + 'static,
    // <<SOut as Stage>::Out as Stage>::In<'static>: WithId,
    for <'a> <SOut as Stage>::In<'a>: Send + Sync + 'static,
{
    pub fn new(stage_in: Arc<SIn>, stage_out: Arc<SOut>) -> Self {
        Self { stage_in, stage_out }
    }
}

#[async_trait]
impl<SIn, SOut> Stage for StagesSeq<SIn, SOut>
where
    SIn:Stage + Cached + OutId + Sync + Send + 'static,
    SOut: Stage<Ctx=SIn::Ctx> + Cached + OutId + Sync + Send + 'static,
    for <'a> SOut: Stage<In<'a>=DataCacheWrapper<SIn::Out>>,
    // SOut: Stage + Cached + Send + Sync + OutId + 'static,
    <SIn as Stage>::In<'static>: WithId,
    <SOut as Stage>::In<'static>: WithId,
    <SIn as Stage>::Out: CopyToDevice + Data +  Send + Sync + 'static,
    <SOut as Stage>::Out: CopyToDevice + Data  + Send + Sync + 'static,
    // <<SOut as Stage>::Out as Stage>::In<'static>: WithId,
    for <'a> <SOut as Stage>::In<'a>: Send + Sync + 'static,
{
    type Ctx = SIn::Ctx;
    type In<'a> = SIn::In<'a>;
    type Out = Arc<DataCacheWrapper<SOut::Out>>;

    async fn run(self: Arc<Self>, _input: Arc<Self::In<'static>>, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
            for<'a> Self::In<'a>: Send + Sync
    {
        let m = Arc::clone(&self.stage_in).run_cached(_input, Arc::clone(&_ctx)).await?;

        Arc::clone(&self.stage_out).run_cached(m, Arc::clone(&_ctx)).await
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_0() {}
}
