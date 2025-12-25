use std::fmt::Debug;
use std::sync::Arc;
use parking_lot::RwLock;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{Data, WithId};
use crate::runner::common::cachable::{DevicesWrap, DevicesWrapCopyMode};
use crate::runner::common::RunCtx;
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
    fn run(self: Arc<Self>, _input: Arc<DevicesWrap<Self::In<'static>>>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> impl Future<Output = Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>> + Send

    {
        async {
            unimplemented!()
        }
    }


    //should be in Cachable, but it's implemented as derive macro
    fn on_cached<'a>(&self, _input: &impl WithId<ID = <<Self as Stage>::In<'a> as WithId>::ID>, _ctx: &Arc<RwLock<Self::Ctx>>)
    where
        Self:'a
    {
        //TODO remove default impl
    }

}


pub trait SyncStage: Stage + Debug {
    fn do_run<'f>(self: Arc<Self>, _input: &Self::In<'f>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {
        unimplemented!()
    }

    fn consume<'f>(self: Arc<Self>, _input: Self::In<'f>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {
        unimplemented!()
    }
}
