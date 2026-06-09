use thoosa_common::runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::runner::common::{OutId, RunCtx};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use dashmap::DashMap;
use futures_util::future::{BoxFuture, Shared};
use parking_lot::RwLock;
use log::debug;
use thoosa_named_derive::Named;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, DevicesWrapCopyMode};
use thoosa_shared::applicables_runner::*;
use thoosa_shared::{CopyToDevice, ThoError, ThoosaError};
use thoosa_shared::traits::{Data, WithId};
use thoosa_shared::comp_id::SynKey;
use tracing::Instrument;
use thoosa_shared::tracked_await::track_await;

#[derive(Debug, Named)]
pub struct ABStage<A, B, C>//TODO rename to Pair/Chained
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId + Send + Sync,
    A::Out: Data + WithId + Send + Sync + 'static,
    <A::Out as Data>::M: DevicesWrapCopyMode<A::Out>,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Data + WithId<ID = SynKey> + Send + Sync + CopyToDevice + 'static,
    <B::Out as Data>::M: DevicesWrapCopyMode<B::Out>
{
    id: SynKey,
    stage_a: Arc<A>,
    stage_b: Arc<B>,

    pd:PhantomData<C>
}


impl <A, B, C> Stage for ABStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId + Send + Sync,
    A::Out: Data + WithId + Send + Sync + 'static,
    <A::Out as Data>::M: DevicesWrapCopyMode<A::Out>,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Data + WithId<ID = SynKey> + Send + Sync + CopyToDevice + 'static,
    <B::Out as Data>::M: DevicesWrapCopyMode<B::Out>

{
    type Ctx = C;
    type In<'i> =  A::In<'i>;
    type Out = B::Out;
}


impl<A, B, C> OutId for ABStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId + Send + Sync,
    A::Out: Data + WithId + Send + Sync + 'static,
    <A::Out as Data>::M: DevicesWrapCopyMode<A::Out>,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Data + WithId<ID = SynKey> + Send + Sync + CopyToDevice + 'static,
    <B::Out as Data>::M: DevicesWrapCopyMode<B::Out>,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID
    {
        &self.id + &self.stage_b.out_id(&self.stage_a.out_id(input_id))
    }
}


impl<A, B, C> ABStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId + Send + Sync,
    A::Out: Data + WithId + Send + Sync + 'static,
    <A::Out as Data>::M: DevicesWrapCopyMode<A::Out>,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Data + WithId<ID = SynKey> + Send + Sync + CopyToDevice + 'static,
    <B::Out as Data>::M: DevicesWrapCopyMode<B::Out>
{
    pub fn new<I:Into<SynKey>>(id:I, stage_a: Arc<A>, stage_b: Arc<B>) -> ABStage<A, B, C> {
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
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId + Send + Sync,
    A::Out: Data + WithId + Send + Sync + 'static,
    <A::Out as Data>::M: DevicesWrapCopyMode<A::Out>,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Data + WithId<ID = SynKey> + Send + Sync + CopyToDevice + 'static,
    <B::Out as Data>::M: DevicesWrapCopyMode<B::Out>
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


// #[async_trait]
impl<A, B, C> AsyncStage for ABStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    for <'i> A::In<'i>: Data + WithId + Send + Sync,
    A::Out: Data + WithId + Send + Sync + 'static,
    <A::Out as Data>::M: DevicesWrapCopyMode<A::Out>,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Data + WithId<ID = SynKey> + Send + Sync + CopyToDevice + 'static,
    <B::Out as Data>::M: DevicesWrapCopyMode<B::Out>
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let input_id = input.id().clone();
        debug!("ABStage::run: input.id={input_id:?} ctx={ctx:?}");

        let output_a = track_await("thoosa_main::tasks::stages_seq::ABStage::AsyncStage::run/1", Arc::clone(&self.stage_a).run_cached(input, ctx.branch(false))).instrument(tracing::debug_span!("run/1", file = file!(), line = line!())).await?;

        debug!("ABStage::run: input.id={input_id:?} a done");

        let output_b = track_await("thoosa_main::tasks::stages_seq::ABStage::AsyncStage::run/2", Arc::clone(&self.stage_b).run_cached(output_a, ctx.branch(false))).instrument(tracing::debug_span!("run/2", file = file!(), line = line!())).await?;

        debug!("ABStage::run: input.id={input_id:?} b done");

        Ok(output_b)
    }

}

#[async_trait]
impl<A, B, C> Cached for ABStage<A, B, C>
where
    C:RunCtx + 'static,

    A:Cached<Ctx=C> + OutId + Sync + Send + 'static,
    A::Out: Data + WithId + Send + Sync + 'static,
    <A::Out as Data>::M: DevicesWrapCopyMode<A::Out>,

    for <'i> B:Cached<In<'i>=A::Out, Ctx=C> + OutId + Send + Sync + 'static,
    B::Out: Data + WithId<ID = SynKey> + Send + Sync + CopyToDevice + 'static,
    <B::Out as Data>::M: DevicesWrapCopyMode<B::Out>,
    for<'i> ABStage<A, B, C>: Stage<In<'i>=A::In<'i>>,
    ABStage<A, B, C>: Stage<Out=B::Out, Ctx=C>,
    for<'i> <ABStage<A, B, C> as Stage>::In<'i>: Data + WithId + Send + Sync,
{
    async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>>>>
    where
        Self: Sized,
        Self::Out: Data + Sync + Send,
    {
        unimplemented!()
    }

    async fn run_cached(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let input_a: Arc<DevicesWrap<A::In<'static>>> = input;
        let mut ctx_c: C = ctx;
        let output_a: Arc<DevicesWrap<A::Out>> = track_await("thoosa_main::tasks::stages_seq::ABStage::Cached::run_cached/1", Arc::clone(&self.stage_a).run_cached(input_a, ctx_c.branch(false))).instrument(tracing::debug_span!("run_cached/1", file = file!(), line = line!())).await?;
        let output_b: Arc<DevicesWrap<B::Out>> = track_await("thoosa_main::tasks::stages_seq::ABStage::Cached::run_cached/2", Arc::clone(&self.stage_b).run_cached(output_a, ctx_c.branch(false))).instrument(tracing::debug_span!("run_cached/2", file = file!(), line = line!())).await?;
        let output: Arc<DevicesWrap<Self::Out>> = output_b;
        Ok(output)
    }
}

#[derive(Named)]//TODO delegate to B's cache
pub enum StageChainOption<A, B>//TODO it's not Chained, rename to Or/AB?
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>: Data + WithId + Send + Sync,
    <A as Stage>::Out: Data + WithId + Send + Sync + CopyToDevice,
    <<A as Stage>::Out as Data>::M: DevicesWrapCopyMode<<A as Stage>::Out>,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,
{
    Single(Arc<A>),
    Chained(Arc<B>),
}

impl<A, B> Debug for StageChainOption<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>: Data + WithId + Send + Sync,
    <A as Stage>::Out: Data + WithId + Send + Sync + CopyToDevice,
    <<A as Stage>::Out as Data>::M: DevicesWrapCopyMode<<A as Stage>::Out>,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StageChainOption::Single(stage) => f
                .debug_tuple("StageChainOption::Single")
                .field(stage)
                .finish(),
            StageChainOption::Chained(stage) => f
                .debug_tuple("StageChainOption::Chained")
                .field(stage)
                .finish(),
        }
    }
}

impl<A, B> Stage for StageChainOption<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>: Data + WithId + Send + Sync,
    <A as Stage>::Out: Data + WithId + Send + Sync + CopyToDevice,
    <<A as Stage>::Out as Data>::M: DevicesWrapCopyMode<<A as Stage>::Out>,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,
{
    type Ctx = <A as Stage>::Ctx;
    type In<'a> = <A as Stage>::In<'a>
    where
        Self: 'a;
    type Out = <A as Stage>::Out;
}

// #[async_trait]
impl<A, B> AsyncStage for StageChainOption<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>: Data + WithId + Send + Sync,
    <A as Stage>::Out: Data + WithId + Send + Sync + CopyToDevice,
    <<A as Stage>::Out as Data>::M: DevicesWrapCopyMode<<A as Stage>::Out>,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId + Send + Sync + Debug + 'static,
{
    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: <A as Stage>::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        match &*self {
            StageChainOption::Single(stage) => track_await("thoosa_main::tasks::stages_seq::StageChainOption::AsyncStage::run/1", Arc::clone(stage).run_cached(input, ctx)).instrument(tracing::debug_span!("run/1", file = file!(), line = line!())).await,
            StageChainOption::Chained(stage) => track_await("thoosa_main::tasks::stages_seq::StageChainOption::AsyncStage::run/2", Arc::clone(stage).run_cached(input, ctx)).instrument(tracing::debug_span!("run/2", file = file!(), line = line!())).await,
        }
    }
}

impl<A, B> OutId for StageChainOption<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    for<'i> <A as Stage>::In<'i>: Data + WithId + Send + Sync,
    <A as Stage>::Out: Data + WithId + Send + Sync + CopyToDevice,
    <<A as Stage>::Out as Data>::M: DevicesWrapCopyMode<<A as Stage>::Out>,

    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId  + Send + Sync + Debug + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        match &*self {
            StageChainOption::Single(stage) =>  stage.out_id(input_id),
            StageChainOption::Chained(stage) => stage.out_id(input_id),
        }
    }
}

#[async_trait]
impl<A, B> Cached for StageChainOption<A, B>
where
    A: Cached + OutId + Send + Sync + Debug + 'static,
    <A as Stage>::Out: CopyToDevice,
    B: for<'i> Cached<In<'i>=<A as Stage>::In<'i>, Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx > + OutId + Send + Sync + Debug + 'static,
    for<'i> StageChainOption<A, B>: Stage<In<'i>=<A as Stage>::In<'i>>,
    StageChainOption<A, B>: Stage<Out=<A as Stage>::Out, Ctx=<A as Stage>::Ctx>,
    for<'i> <StageChainOption<A, B> as Stage>::In<'i>: Data + WithId + Send + Sync,
    <StageChainOption<A, B> as Stage>::Out: Data + WithId + Send + Sync + 'static,
    <<StageChainOption<A, B> as Stage>::Out as Data>::M: DevicesWrapCopyMode<<StageChainOption<A, B> as Stage>::Out>,
{
    async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>>>>
    where
        Self: Sized,
        Self::Out: Data + Sync + Send
    {
        unimplemented!()
    }

    async fn run_cached(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError>
    where
        Self: Sized + OutId,
        for<'a> <Self as Stage>::In<'a>: Send + Sync,
        <Self as Stage>::Out: Send + Sync,
        <<Self as Stage>::Out as WithId>::ID: 'static,
        <Self as Stage>::Out: 'static,
        Self::Out: Data,
    {
        let input_a: Arc<DevicesWrap<<A as Stage>::In<'static>>> = input;
        let ctx_a: <A as Stage>::Ctx = ctx;

        let output_a: Arc<DevicesWrap<<A as Stage>::Out>> = match &*self {
            StageChainOption::Single(stage) => track_await("thoosa_main::tasks::stages_seq::StageChainOption::Cached::run_cached/1", Arc::clone(stage).run_cached(input_a, ctx_a)).instrument(tracing::debug_span!("run_cached/1", file = file!(), line = line!())).await?,
            StageChainOption::Chained(stage) => track_await("thoosa_main::tasks::stages_seq::StageChainOption::Cached::run_cached/2", Arc::clone(stage).run_cached(input_a, ctx_a)).instrument(tracing::debug_span!("run_cached/2", file = file!(), line = line!())).await?,
        };

        let output: Arc<DevicesWrap<Self::Out>> = output_a;
        Ok(output)
    }
}
