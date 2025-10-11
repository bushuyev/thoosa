use crate::common::{Data, RunCtx};
use async_trait::async_trait;
use std::fmt::Debug;
use std::sync::Arc;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};

//some operation on data - pattern search, dnn train, whatever what takes input and produces output
#[async_trait]
pub trait Stage: Debug {
    type Ctx: RunCtx;

    type In<'i>: Input where Self: 'i;

    type Out: Output;

    //should implement either run for multistep or do_run for single step (can be oom-handled)
    async fn run<'f>(
        self: Arc<Self>,
        _input: &'f Self::In<'f>,
        _ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self::In<'f>: Send + Sync,
    {
        unimplemented!()
    }

    fn do_run<'f>(self: Arc<Self>, _input: &Self::In<'f>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn consume<'f>(self: Arc<Self>, _input: Self::In<'f>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}
