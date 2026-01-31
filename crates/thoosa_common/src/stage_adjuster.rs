use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use thoosa_shared::runner::common::runnable::Stage;
use thoosa_shared::runner::oom_handler::{Adjuster, AdjusterError};
use thoosa_shared::traits::Adjustable;

#[derive(Debug)]
pub struct StageAdjuster<ST>
where
    ST:Stage + Adjustable + Debug
{
    fd:PhantomData<ST>
}

impl<ST> StageAdjuster<ST>
where
    ST:Stage + Adjustable + Debug
{
    pub fn new() -> Self {
        Self { fd:PhantomData }
    }
}

#[async_trait]
impl <ST, AC> Adjuster<AC> for StageAdjuster<ST>
where
    ST:Stage + Adjustable + Debug
{
    type A = Arc<ST>;

    async fn adjust(&self, _adjustable: Self::A, _adjust_context: &mut AC) -> Result<Self::A, AdjusterError<Self::A>> {
        todo!()
    }
}
