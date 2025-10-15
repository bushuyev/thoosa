use cached_derive::Cached;
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::common::{OutId, RunCtx};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use thoosa_shared::applicables_runner::*;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::traits::{Data, SimpleId, WithId};

#[derive(Debug)]
pub struct ABStage<A, B, C>
where
    C:RunCtx,
    for<'i> <A as Stage>::In<'i>: Send + Sync,
    A:Cached<Ctx=C> + Sync + Send + 'static,
    for <'i> B:Cached<In<'i>=A::Out, Ctx=C>  + Send + Sync + 'static,
    for <'i> A::In<'i>: Data + WithId,
    A::Out: Data + WithId + Send + Sync + 'static,

    for <'i> B::In<'i>: Data + WithId,
    B::Out: Data + WithId + Send + Sync + 'static
{
    id:SimpleId,
    stage_a: Arc<A>,
    stage_b: Arc<B>,

    pd:PhantomData<C>
}


impl <A, B, C> Stage for ABStage<A, B, C>
where
    C:RunCtx,
    for<'i> <A as Stage>::In<'i>: Send + Sync,
    A:Cached<Ctx=C> + Sync + Send + 'static,
    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId,
    A::Out: Data + WithId + Send + Sync + 'static,

    for <'i> B::In<'i>: Data + WithId,
    B::Out: Data + WithId + Send + Sync + 'static,

{
    type Ctx = C;
    type In<'i> =  A::In<'i> where A:'i, B:'i, C:'i;
    type Out = B::Out;
}


impl<A, B, C> OutId for ABStage<A, B, C>
where
    C:RunCtx + 'static,
    for<'i> <A as Stage>::In<'i>: Send + Sync,
    A:Cached<Ctx=C> + Sync + Send + 'static,
    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId,
    A::Out: Data + WithId + Send + Sync + 'static,

    for <'i> B::In<'i>: Data + WithId,
    B::Out: Data + WithId + Send + Sync + 'static,

    <<B as Stage>::Out as WithId>::ID: From<String>,
    for <'i> <<A as Stage>::In<'i> as WithId>::ID: std::fmt::Display
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID  {
        format!("me_{}_input_{}", self.id, input_id).into()
    }
}


impl<A, B, C> ABStage<A, B, C>
where
    C:RunCtx,
    for<'i> <A as Stage>::In<'i>: Send + Sync,
    A:Cached<Ctx=C> + Sync + Send + 'static,
    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId,
    A::Out: Data + WithId + Send + Sync + 'static,

    for <'i> B::In<'i>: Data + WithId,
    B::Out: Data + WithId + Send + Sync + 'static,
{
    pub fn new<I:Into<SimpleId>>(id:I, stage_a: Arc<A>, stage_b: Arc<B>) -> ABStage<A, B, C> {
        ABStage {
            id: id.into(),
            stage_a,
            stage_b,
            pd: PhantomData
        }
    }
}

impl<A, B, C> WithId for ABStage<A, B, C>
where
    C:RunCtx,
    for<'i> <A as Stage>::In<'i>: Send + Sync,
    A:Cached<Ctx=C> + Sync + Send + 'static,
    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId,
    A::Out: Data + WithId + Send + Sync + 'static,

    for <'i> B::In<'i>: Data + WithId,
    B::Out: Data + WithId + Send + Sync + 'static,
{
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


#[async_trait]
impl<A, B, C> AsyncStage for ABStage<A, B, C>
where
    C:RunCtx + 'static,
    for<'i> <A as Stage>::In<'i>: Send + Sync,
    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId + Sync + Send + 'static,
    A::Out: Data + WithId + Send + Sync + 'static,

    for <'i> B::In<'i>: Data + WithId,
    B::Out: Data + WithId + Send + Sync + 'static,
{
    async fn run(self: Arc<Self>, input: Arc<DataCacheWrapper<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {

        let output_a = Arc::clone(&self.stage_a).run_cached(input, Arc::clone(&ctx)).await?;
        let output_b = Arc::clone(&self.stage_b).run_cached(output_a, ctx).await?;

        Ok(output_b)
    }

}



#[test]
fn test_0(){

}