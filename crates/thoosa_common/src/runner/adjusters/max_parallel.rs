use crate::runner::global_ctx::CacheKey;
use crate::runner::adjusters::dump_to_cpu::*;
use crate::runner::oom_handler::{Adjuster, AdjusterError, RunCtxContext};
use ambassador::{Delegate, delegatable_trait};
use async_trait::async_trait;
use log::debug;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use thoosa_shared::traits::{MaxParallelCtx, RunCtx};
// use tokio::io::{AsyncReadExt, AsyncWriteExt};
// use crate::common::runnable::Runnable;
use thoosa_shared::traits::{Adjustable, CachableIdTr};


#[derive(Debug)]
pub enum MaxParallelStep {
    SLOW,
    MID,
    FAST,
}

impl Deref for MaxParallelStep {
    type Target = MaxParallel;

    fn deref(&self) -> &Self::Target {
        match self {
            MaxParallelStep::SLOW => &2,
            MaxParallelStep::MID => &4,
            MaxParallelStep::FAST => &16,
        }
    }
}

pub const MAX_PARALLEL: MaxParallel = 32;

pub type MaxParallel = u8;
pub struct MaxParallelAdjuster<C>
where
    C: Adjustable + RunCtx,
{
    pub step: MaxParallel,
    pd: PhantomData<C>,
}

impl<C> Debug for MaxParallelAdjuster<C>
where
    C: Adjustable + RunCtx,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PermitsAdjuster")
            .field("step", &self.step)
            .finish()
    }
}

impl<C> MaxParallelAdjuster<C>
where
    C: Adjustable + RunCtx,
{
    pub fn new(step: MaxParallel) -> Self {
        Self {
            step,
            pd: Default::default(),
        }
    }
}

#[delegatable_trait]
pub trait MaxParallelAdjustContext: Debug + Send + Sync {
    fn max_other_parallel(&self) -> MaxParallel;
    fn max_other_parallel_mut(&mut self) -> &mut MaxParallel;
}

#[derive(Debug)]
pub struct MaxParallelAdjustContextImpl {
    max_other_parallel: MaxParallel,
}

impl MaxParallelAdjustContextImpl {
    pub fn new() -> Self {
        Self {
            max_other_parallel: MAX_PARALLEL,
        }
    }
}

impl MaxParallelAdjustContext for MaxParallelAdjustContextImpl {
    fn max_other_parallel(&self) -> MaxParallel {
        self.max_other_parallel
    }

    fn max_other_parallel_mut(&mut self) -> &mut MaxParallel {
        &mut self.max_other_parallel
    }

}

impl RunCtxContext for MaxParallelAdjustContextImpl {}

#[async_trait]
impl<A, C> Adjuster<C> for MaxParallelAdjuster<A>
where
    A: Adjustable + MaxParallelCtx + RunCtx,
    C: MaxParallelAdjustContext,
{
    type A = A;

    async fn adjust(
        &self,
        mut adjustable: <Self as Adjuster<C>>::A,
        adjust_context: &mut C,
    ) -> Result<<Self as Adjuster<C>>::A, AdjusterError<<Self as Adjuster<C>>::A>> {

        adjustable.set_max_other_parallel(adjust_context.max_other_parallel());

        debug!("PermitsAdjuster::adjust: adjust_context={adjust_context:?} adjustable={adjustable:?}");

        Ok(adjustable)
    }

    fn degrade(&self, adjust_context: &mut C) -> bool {

        if adjust_context.max_other_parallel() > 0 {
            *adjust_context.max_other_parallel_mut() = adjust_context.max_other_parallel().saturating_sub(self.step);
            debug!("PermitsAdjuster::degrade: adjust_context={adjust_context:?} true");

            true
        } else {
            debug!("PermitsAdjuster::degrade: adjust_context={adjust_context:?} false");

            false
        }

    }
}

#[derive(Debug, Delegate)]
#[delegate(MaxParallelAdjustContext, target = "max_parallel")]
#[delegate(DumpToCpuContext<'a>, target = "dump_to_cpu")]
pub struct MaxParallelAndDumpToCpuAdjustableContex<'a> {
    max_parallel: MaxParallelAdjustContextImpl,
    dump_to_cpu: DumpToCpuContextImpl<'a>,
}

impl<'a> MaxParallelAndDumpToCpuAdjustableContex<'a> {
    pub fn new(max_parallel: MaxParallelAdjustContextImpl, cpu: DumpToCpuContextImpl<'a>) -> Self {
        Self {
            max_parallel,
            dump_to_cpu: cpu,
        }
    }
}

impl<'a> RunCtxContext for MaxParallelAndDumpToCpuAdjustableContex<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use thoosa_shared::traits::MaxParallelCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::run_ctx::RunCtxImpl;

    fn ctx(_name: &'static str, want: MaxParallel) -> MaxParallelAdjustContextImpl {
        let mut c = MaxParallelAdjustContextImpl::new();
        *c.max_other_parallel_mut() = want;
        c
    }

    #[tokio::test]
    async fn adjust_adds_dummy_max_parallel() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

        let adjustable = RunCtxImpl::new(0, "permits_adjust_dummy");
        let adjuster = MaxParallelAdjuster::<RunCtxImpl>::new(10);
        let mut ac = ctx("test", 10);

        let res = adjuster.adjust(adjustable, &mut ac).await.unwrap();

        assert_eq!(res.max_other_parallel(), 10);
        assert_eq!(ac.max_other_parallel(), 10);
    }

    #[tokio::test]
    async fn adjust_is_noop_when_max_parallel_already_set() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

        let mut adjustable = RunCtxImpl::new(0, "permits_adjust_noop");
        adjustable.set_max_other_parallel(10);
        let adjuster = MaxParallelAdjuster::<RunCtxImpl>::new(10);
        let mut ac = ctx("test", 10);

        let res = adjuster.adjust(adjustable, &mut ac).await.unwrap();

        assert_eq!(res.max_other_parallel(), 10);
        assert_eq!(ac.max_other_parallel(), 10);
    }
}
