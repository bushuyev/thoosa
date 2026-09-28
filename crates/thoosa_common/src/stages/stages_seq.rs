use crate::runner::common::cachable::CachedPurgeOutput;
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::future::{BoxFuture, Shared};
use log::debug;
use parking_lot::RwLock;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::*;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{AsyncStage, Data, OutId, RunCtx, Stage, WithId};
use thoosa_shared::{ThoError, ThoosaError};

#[derive(Debug, Named)]
pub struct PairStage<A, B, C> //TODO rename to Pair/Chained
where
    C: RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>:  WithId + Send + Sync,
    A::Out: Clone + CachedPurgeOutput + WithId + Send + Sync + 'static,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Clone + CachedPurgeOutput + WithId<ID = SynKey> + Send + Sync + 'static,
{
    stage_a: Arc<A>,
    stage_b: Arc<B>,

    pd:PhantomData<C>
}


impl <A, B, C> Stage for PairStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>:  WithId + Send + Sync,
    A::Out: Clone + CachedPurgeOutput + WithId + Send + Sync + 'static,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Clone + CachedPurgeOutput + WithId<ID = SynKey> + Send + Sync + 'static,

{
    type Ctx = C;
    type In<'i> =  A::In<'i>;
    type Out = B::Out;
}


impl<A, B, C> OutId for PairStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>:  WithId + Send + Sync,
    A::Out: Clone + CachedPurgeOutput + WithId + Send + Sync + 'static,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Clone + CachedPurgeOutput + WithId<ID = SynKey> + Send + Sync + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID
    {
        self.stage_b.out_id(&self.stage_a.out_id(input_id))
    }
}


impl<A, B, C> PairStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>:  WithId + Send + Sync,
    A::Out: Clone + CachedPurgeOutput + WithId + Send + Sync + 'static,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Clone + CachedPurgeOutput + WithId<ID = SynKey> + Send + Sync + 'static,
{
    pub fn new(stage_a: Arc<A>, stage_b: Arc<B>) -> PairStage<A, B, C> {
        PairStage {
            stage_a,
            stage_b,
            pd: PhantomData
        }
    }
}


impl<A, B, C> AsyncStage for PairStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>:  WithId + Send + Sync,
    A::Out: Clone + CachedPurgeOutput + WithId + Send + Sync + 'static,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Clone + CachedPurgeOutput + WithId<ID = SynKey> + Send + Sync + 'static,
{

    async fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        let input_id = input.id().clone();
        debug!("ABStage::run: input.id={input_id:?} ctx={ctx:?}");

        let output_a = track_await(Arc::clone(&self.stage_a).run_cached(input, ctx.branch(false))).await?;

        debug!("ABStage::run: input.id={input_id:?} a done");

        let output_b = track_await(Arc::clone(&self.stage_b).run_cached(output_a, ctx.branch(false))).await?;

        debug!("ABStage::run: input.id={input_id:?} b done");

        Ok(output_b)
    }

}

#[async_trait]
impl<A, B, C> Cached for PairStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>:  WithId + Send + Sync,
    A::Out: Clone + CachedPurgeOutput + WithId + Send + Sync + 'static,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Clone + CachedPurgeOutput + WithId<ID = SynKey> + Send + Sync + 'static,

    Self::Out: WithId
{
    async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, thoosa_common::runner::common::cachable::CachedFuture<Self::Out>>
    where
        Self: Sized,
        Self::Out: WithId + Sync + Send,
    {
        unimplemented!()
    }

    async fn run_cached(
        self: Arc<Self>,
        input: Self::In<'static>,
        mut ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        let input_id = input.id().clone();
        debug!("ABStage::run_cached: input.id={input_id:?}");

        let mut ctx_c: C = ctx;
        let output_a: A::Out = track_await(Arc::clone(&self.stage_a).run_cached(input, ctx_c.branch(false))).await?;
        debug!("ABStage::run_cached: input.id={input_id:?} output_a={output_a:?} a done");

        let output_b = track_await(Arc::clone(&self.stage_b).run_cached(output_a, ctx_c.branch(false))).await?;

        debug!("ABStage::run_cached: input.id={input_id:?} output_b={output_b:?} b done");

        Ok(output_b)
    }
}

#[derive(Named)]//TODO delegate to B's cache
pub enum SingleOrPair<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>:  WithId + Send + Sync,
    <A as Stage>::Out: Clone + CachedPurgeOutput + WithId + Send + Sync,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,
{
    Single(Arc<A>),
    Pair(Arc<B>),
}

impl<A, B> Debug for SingleOrPair<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>:  WithId + Send + Sync,
    <A as Stage>::Out: Clone + CachedPurgeOutput + WithId + Send + Sync,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SingleOrPair::Single(stage) => f
                .debug_tuple("StageChainOption::Single")
                .field(stage)
                .finish(),
            SingleOrPair::Pair(stage) => f
                .debug_tuple("StageChainOption::Chained")
                .field(stage)
                .finish(),
        }
    }
}

impl<A, B> Stage for SingleOrPair<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>:  WithId + Send + Sync,
    <A as Stage>::Out: Clone + CachedPurgeOutput + WithId + Send + Sync,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,
{
    type Ctx = <A as Stage>::Ctx;
    type In<'a> = <A as Stage>::In<'a>
    where
        Self: 'a;
    type Out = <A as Stage>::Out;
}


impl<A, B> AsyncStage for SingleOrPair<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>:  WithId + Send + Sync,
    <A as Stage>::Out: Clone + CachedPurgeOutput + WithId + Send + Sync,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,
{
//    // type RunInput = A::RunInput;
//     // type RunOutput = A::Out;
    async fn run(
        self: Arc<Self>,
        input: Self::In<'static>,
        ctx: <A as Stage>::Ctx,
    ) -> Result<Self::Out, ThoError> {
        match &*self {
            SingleOrPair::Single(stage) => track_await(Arc::clone(stage).run_cached(input, ctx)).await,
            SingleOrPair::Pair(stage) => track_await(Arc::clone(stage).run_cached(input, ctx)).await,
        }
    }
}

impl<A, B> OutId for SingleOrPair<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>:  WithId + Send + Sync,
    <A as Stage>::Out: Clone + CachedPurgeOutput + WithId + Send + Sync,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,

    Self::Out: WithId
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        match &*self {
            SingleOrPair::Single(stage) =>  stage.out_id(input_id),
            SingleOrPair::Pair(stage) => stage.out_id(input_id),
        }
    }
}

#[async_trait]
impl<A, B> Cached for SingleOrPair<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>:  WithId + Send + Sync,
    <A as Stage>::Out: Clone + CachedPurgeOutput + WithId + Send + Sync,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,

    Self::Out: WithId
{
    async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, thoosa_common::runner::common::cachable::CachedFuture<Self::Out>>
    where
        Self: Sized,
        Self::Out: Sync + Send
    {
        unimplemented!()
    }

    async fn run_cached(
        self: Arc<Self>,
        input: Self::In<'static>,
        ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError>
    // where
    //     Self: Sized + OutId,
    //     for<'a> <Self as Stage>::In<'a>: WithId + Send + Sync,
    //     <Self as Stage>::Out: Send + Sync,
    //     <<Self as Stage>::Out as WithId>::ID: 'static,
    //     <Self as Stage>::Out: 'static,
    //     Self::Out: Data,
    {
        let ctx_a: <A as Stage>::Ctx = ctx;

        Ok(match &*self {
            SingleOrPair::Single(stage) => track_await(Arc::clone(stage).run_cached(input, ctx_a)).await?,
            SingleOrPair::Pair(stage) => track_await(Arc::clone(stage).run_cached(input, ctx_a)).await?,
        })
    }
}
