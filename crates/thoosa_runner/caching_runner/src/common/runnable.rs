use crate::common::{Data, RunCtx};
use async_trait::async_trait;
use std::fmt::Debug;
use std::sync::Arc;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::WithId;
use crate::common::cachable::DataCacheWrapper;

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
pub trait AsyncStage:Stage + Debug
where
        for <'a> Self::In<'a>: Data + WithId + Send + Sync,
        Self::Out: Data + WithId + Send + Sync
{


    //should implement either run for multistep or do_run for single step (can be oom-handled)
    async fn run(self: Arc<Self>, input: Arc<DataCacheWrapper<Self::In<'static>>>, ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {
        unimplemented!()
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
