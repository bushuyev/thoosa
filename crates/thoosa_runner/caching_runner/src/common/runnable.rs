use crate::common::{Data, RunCtx, WithId};
use async_trait::async_trait;
use std::fmt::Debug;
use std::sync::Arc;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::ThoosaError;

//some operation on data - pattern search, dnn train, whatever what takes input and produces output
#[async_trait]
pub trait Stage: Debug  + Send + Sync + 'static {
    type In: Input + Sync + Send  + 'static;
    type Out: Output + Data<Sub=Self::Out> + Sync + Send + 'static;
    type Ctx: RunCtx;


    //should implement either run for multistep or do_run for single step (can be oom-handled)

    async fn run(self: Arc<Self>, _input: Arc<Self::In>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {
        unimplemented!()
    }


    fn do_run(self: Arc<Self>, _input:&Self::In, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> where Self:Sized {
        unimplemented!()
    }
}
