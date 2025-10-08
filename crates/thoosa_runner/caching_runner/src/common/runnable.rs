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

    type In<'a>: Input
    where
        Self: 'a;

    type Out: Output;

    //should implement either run for multistep or do_run for single step (can be oom-handled)
    async fn run(
        self: Arc<Self>,
        _input: Arc<Self::In<'static>>,
        _ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        for<'a> Self::In<'a>: Send + Sync,
    {
        unimplemented!()
    }

    fn do_run<'a>(
        self: Arc<Self>,
        _input: &Self::In<'a>,
        _ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized,
    {
        unimplemented!()
    }

    fn consume<'a>(
        self: Arc<Self>,
        _input: Self::In<'a>,
        _ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized,
    {
        unimplemented!()
    }
}
