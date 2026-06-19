use crate::runner::common::cachable::{DevicesWrap, DevicesWrapCopyMode};
use crate::runner::common::{OutId, RunCtx};
use std::fmt::Debug;
use std::sync::Arc;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{Adjustable, Data, WithId};
use thoosa_shared::ThoError;
//some operation on data - pattern search, dnn train, whatever what takes input and produces output

pub trait Stage: Debug {
    type Ctx: RunCtx;

    type In<'a>: Input;

    type Out: Output;

}

// #[async_trait] - it's anyway not obj safe, and it forces Output to be Send
pub trait AsyncStage:Stage + Debug + Send + Sync + 'static 
where
        for <'a> Self::In<'a>: Data + WithId + Send + Sync,
        Self::Out: Data + WithId + Send + Sync,
        <Self::Out as Data>::M: DevicesWrapCopyMode<Self::Out>
{

    //the DevicesWrap is in input and output so that data could be dumped to CPU at any time (if other stages wins 100 GPU)
    //it means 'plain' inputs like resource ID should have dummy Data impl, TODO look for the better way (enum based solution appear to be essentially same dummy wrapper)
    // Returning DevicesWrap also forces callers to specify the device to load data from, keeping the logic clear and transparent (as opposed to implicitly taking the device from the stage context).
    fn run(self: Arc<Self>, _input: Arc<DevicesWrap<Self::In<'static>>>, _ctx: Self::Ctx) -> impl Future<Output = Result<Arc<DevicesWrap<Self::Out>>, ThoError>> + Send

    {
        async {
            unimplemented!()
        }
    }
}

#[derive(Debug)]
pub enum ConsumeResult<I, O> {
    Failed{input:I, error:ThoError},
    Ok(O)
}

pub trait SyncStage: Stage + Debug {
    fn do_run<'f>(self: Arc<Self>, _input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        unimplemented!()
    }

    fn consume<'f>(
        self: Arc<Self>,
        _input: Self::In<'f>,
        _ctx: Self::Ctx,
    ) -> ConsumeResult<Self::In<'f>, Self::Out> {
        unimplemented!()
    }
}

pub trait DynSyncStage<I, O, C>: Debug + Send + Sync + 'static
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    fn do_run_dyn(self: Arc<Self>, input: &I, ctx: &C) -> Result<O, ThoError>;

    fn out_id_dyn(&self, input_id: &I::ID) -> O::ID;
}

impl<T, I, O, C> DynSyncStage<I, O, C> for T
where
    T: SyncStage + OutId + Send + Sync + 'static,
    for<'a> T: Stage<In<'a> = I, Out = O, Ctx = C>,
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    fn do_run_dyn(self: Arc<Self>, input: &I, ctx: &C) -> Result<O, ThoError> {
        self.do_run(input, ctx)
    }

    fn out_id_dyn(&self, input_id: &I::ID) -> O::ID {
        self.out_id(input_id)
    }
}

#[derive(Debug)]
pub struct DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    inner: Arc<dyn DynSyncStage<I, O, C>>,
}

impl<I, O, C> DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    pub fn new(inner: Arc<dyn DynSyncStage<I, O, C>>) -> Self {
        Self { inner }
    }
}

impl<I, O, C> Stage for DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    type Ctx = C;
    type In<'a> = I;
    type Out = O;
}

impl<I, O, C> SyncStage for DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        Arc::clone(&self.inner).do_run_dyn(input, ctx)
    }
}

impl<I, O, C> OutId for DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        self.inner.out_id_dyn(input_id)
    }
}

impl<I, O, C> Adjustable for DynSyncStageBox<I, O, C>
where
    I: Input + WithId + 'static,
    O: Output + WithId + 'static,
    C: RunCtx + 'static,
{
}
