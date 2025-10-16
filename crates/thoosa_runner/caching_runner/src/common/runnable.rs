use crate::common::{RunCtx};
use async_trait::async_trait;
use std::fmt::Debug;
use std::sync::Arc;
use parking_lot::RwLock;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{Data, WithId};
use crate::common::cachable::DataCacheWrapper;
use crate::common::stage_input::StageInput;
use crate::global_ctx::GlobalCtx;

//some operation on data - pattern search, dnn train, whatever what takes input and produces output
#[async_trait]
pub trait Stage: Debug {
    type Ctx: RunCtx;

    type In<'a>: Input
    where
        Self: 'a;

    type Out: Output;

}


#[async_trait]
pub trait AsyncStage:Stage + Debug + Send + Sync + 'static 
where
    for<'a> Self::In<'a>: WithId + Send + Sync,
    Self::In<'static>: Sized + 'static,
    Self::Out: Data + WithId + Send + Sync,
    Self: 'static,
{



    //should implement either run for multistep or do_run for single step (can be oom-handled)
    async fn run(self: Arc<Self>, input: StageInput<Self::In<'static>>, ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {
        unimplemented!()
    }

    //should be in Cached, but it's implemented as derive macro
    fn on_cached_2<'a>(&self, input: &<Self as Stage>::In<'a>, ctx: &Arc<RwLock<Self::Ctx>>)
    where
        Self:'a
    {

    }



    //should be in Cachable, it's here to avoid adding extra impls
    fn on_cached<'a>(&self, input: &<<Self as Stage>::In<'static> as WithId>::ID, ctx: &Arc<RwLock<Self::Ctx>>)
    where
        Self:'a
    {

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
