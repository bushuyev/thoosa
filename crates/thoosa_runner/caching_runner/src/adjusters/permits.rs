use crate::adjusters::dump_to_cpu::*;
// use crate::adjusters::splits::*;
use crate::common::DropIf;
use crate::common::RunCtx;
use crate::oom_handler::{Adjuster, RunCtxContext};
use ambassador::{delegatable_trait, Delegate};
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_shared::{MyError, ThoosaError};
// use tokio::io::{AsyncReadExt, AsyncWriteExt};
// use crate::common::runnable::Runnable;
use crate::permits::{NumPermits, RunnerPerm};
use thoosa_shared::traits::{Adjustable, CachableIdTr};

pub struct PermitsAdjuster<F, Fut, C>
where
    F: Fn(NumPermits) -> Fut + Send + Sync,
    Fut: Future<Output = Result<RunnerPerm, ThoosaError>> + Send,
    C: Adjustable + RunCtx,
{
    pub(crate) max: NumPermits,
    pub(crate) step: NumPermits,
    pub acquire_fn: F,
    pd: PhantomData<C>,
}

impl<F, Fut, C> Debug for PermitsAdjuster<F, Fut, C>
where
    F: Fn(NumPermits) -> Fut + Send + Sync,
    Fut: Future<Output = Result<RunnerPerm, ThoosaError>> + Send,
    C: Adjustable + RunCtx,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PermitsAdjuster")
            .field("max", &self.max)
            .field("step", &self.step)
            .finish()
    }
}

impl<F, Fut, C> PermitsAdjuster<F, Fut, C>
where
    F: Fn(NumPermits) -> Fut + Send + Sync,
    Fut: Future<Output = Result<RunnerPerm, ThoosaError>> + Send,
    C: Adjustable + RunCtx,
{
    pub fn new(
        max: NumPermits,
        step: NumPermits, /*TODO replace with PermitStep*/
        acquire_fn: F,
    ) -> Self {
        Self {
            max,
            step,
            acquire_fn,
            pd: Default::default(),
        }
    }
}

#[delegatable_trait]
pub trait PermitsAdjustContext: Debug + Send + Sync {
    fn to_acquire(&self) -> NumPermits;
    fn to_acquire_mut(&mut self) -> &mut NumPermits;
}

#[derive(Debug)]
pub struct PermitsAdjustContextImpl {
    to_acquire: NumPermits,

    _ctx_name: &'static str, //for debug
                             // statistics: Option<Arc<Statistics>>
}

impl PermitsAdjustContextImpl {
    pub fn new(ctx_name: &'static str) -> Self {
        Self {
            to_acquire: 0,

            _ctx_name: ctx_name, // statistics,
        }
    }
}

impl PermitsAdjustContext for PermitsAdjustContextImpl {
    fn to_acquire(&self) -> NumPermits {
        self.to_acquire
    }

    fn to_acquire_mut(&mut self) -> &mut NumPermits {
        &mut self.to_acquire
    }
}

impl RunCtxContext for PermitsAdjustContextImpl {}

#[async_trait]
impl<F, Fut, A, C> Adjuster<C> for PermitsAdjuster<F, Fut, A>
where
    F: Fn(NumPermits) -> Fut + Send + Sync,
    Fut: Future<Output = Result<RunnerPerm, ThoosaError>> + Send,
    A: Adjustable + RunCtx,
    C: PermitsAdjustContext,
{
    type A = Arc<RwLock<A>>;

    async fn adjust(
        &self,
        adjustable: &<Self as Adjuster<C>>::A,
        adjust_context: &mut C,
    ) -> Result<<Self as Adjuster<C>>::A, ThoosaError> {
        let acquired_permits = adjustable.read().acquired_permits();
        if adjust_context.to_acquire() <= acquired_permits {
            //TODO warn
            return Ok(Arc::clone(adjustable));
        }

        let permits_to_acquire = adjust_context.to_acquire() - acquired_permits;

        let adjustable_name = adjustable.read().name();
        debug!("PermitsAdjuster::adjust: name={adjustable_name} thread={:?} self.max={} permits_to_acquire/{}/ = adjust_context.to_acquire()/{}/ - acquired_permits/{acquired_permits}/",std::thread::current().id(),self.max,   permits_to_acquire,      adjust_context.to_acquire());

        if permits_to_acquire > 0 {
            match (self.acquire_fn)(permits_to_acquire).await {
                Ok(permit) => {
                    debug!("PermitsAdjuster::adjust: acquire_fn name={adjustable_name} acquired_permits={acquired_permits} permits_to_acquire={permits_to_acquire} acquired ok ");
                    adjustable.write().add_permit(permit);
                }

                //can't acquire permits,
                Err(te) => {
                    match te.what() {
                        MyError::OutOfPermits => {
                            debug!("PermitsAdjuster::adjust: acquire_fn name={adjustable_name} acquired_permits={acquired_permits} Error: {te:?} device_idx={}", adjustable.read().device_idx());

                           /* if adjustable.read().device_idx() != 0 {
                                //not on CPU, TODO
                                if let Err(e) = A::purge_gpu_caches(Arc::clone(adjustable)).await {
                                    debug!("PermitsAdjuster::adjust: purge_gpu_caches failed: {e:?}");
                                };
                            }*/
                            adjustable.write().drop_permits(DropIf::Always);

                            *adjust_context.to_acquire_mut() = 0;

                            debug!("PermitsAdjuster::adjust: acquire_fn name={adjustable_name} yielding");

                            tokio::task::yield_now().await;
                        }
                        e => {
                            // Propagate unexpected errors to let caller decide.
                            return Err(te);
                        }
                    }
                }
            };
        }

        Ok(Arc::clone(adjustable))
    }

    fn degrade(&self, adjust_context: &mut C) -> bool {
        debug!("PermitsAdjuster::degrade: adjust_context={adjust_context:?}");

        if adjust_context.to_acquire() < self.max {
            *adjust_context.to_acquire_mut() += self.step;

            true
        } else {
            false
        }
    }
}

#[derive(Debug, Delegate)]
#[delegate(PermitsAdjustContext, target = "permits")]
#[delegate(DumpToCpuContext<'a>, target = "dump_to_cpu")]
pub struct PermitsAndDumpToCpuAdjustableContex<'a> {
    permits: PermitsAdjustContextImpl,
    dump_to_cpu: DumpToCpuContextImpl<'a>,
}

impl<'a> PermitsAndDumpToCpuAdjustableContex<'a> {
    pub fn new(permits: PermitsAdjustContextImpl, cpu: DumpToCpuContextImpl<'a>) -> Self {
        Self {
            permits,
            dump_to_cpu: cpu,
        }
    }
}

impl<'a> RunCtxContext for PermitsAndDumpToCpuAdjustableContex<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::PermitsCtx;
    use crate::permits::NumPermits;
    use futures_util::future::BoxFuture;
    use parking_lot::RwLock;
    use std::sync::Arc;
    use thoosa_shared::MyError;
    use thoosa_shared::traits::Named;

    #[derive(Debug)]
    struct TestRunCtx {
        device_idx: usize,
        acquired: NumPermits,
    }

    impl TestRunCtx {
        fn new(device_idx: usize, acquired: NumPermits) -> Self {
            Self {
                device_idx,
                acquired,
            }
        }
    }

    impl Adjustable for TestRunCtx {}

    impl PermitsCtx for TestRunCtx {
        fn add_permit(&mut self, permit: RunnerPerm) {
            self.acquired += permit.num_permits();
        }

        fn drop_permits(&mut self, _drop_if: DropIf) {
            self.acquired = 0;
        }

        fn acquired_permits(&self) -> NumPermits {
            self.acquired
        }
    }

    impl RunCtx for TestRunCtx {
        fn device_idx(&self) -> usize {
            self.device_idx
        }

        fn purge_gpu_caches(_ctx: Arc<RwLock<Self>>) -> BoxFuture<'static, Result<(), ThoosaError>>
        where
            Self: Sized,
        {
            Box::pin(async { Ok(()) })
        }

        fn cache_area(&self) -> u64 {
            0
        }
    }

    impl Named for TestRunCtx {}

    fn ctx(name: &'static str, want: NumPermits) -> PermitsAdjustContextImpl {
        let mut c = PermitsAdjustContextImpl::new(name);
        *c.to_acquire_mut() = want;
        c
    }

    #[tokio::test]
    async fn adjust_handles_out_of_permits() {
        let adjustable = Arc::new(RwLock::new(TestRunCtx::new(0, 0)));
        let adjuster = PermitsAdjuster::<_, _, TestRunCtx>::new(100, 10, |_n| async {
            Err(MyError::OutOfPermits.into())
        });
        let mut ac = ctx("test", 10);

        let res = adjuster.adjust(&adjustable, &mut ac).await;
        assert!(res.is_ok());
        // OutOfPermits path resets requested acquire to 0
        assert_eq!(ac.to_acquire(), 0);
    }

    #[tokio::test]
    async fn adjust_propagates_other_errors() {
        let adjustable = Arc::new(RwLock::new(TestRunCtx::new(0, 0)));
        let adjuster = PermitsAdjuster::<_, _, TestRunCtx>::new(100, 10, |_n| async {
            Err(MyError::SomethingElse("boom".into()).into())
        });
        let mut ac = ctx("test", 10);

        let res = adjuster.adjust(&adjustable, &mut ac).await;
        assert!(matches!(
            res,
            Err(ThoosaError {
                what: MyError::SomethingElse(_),
                ..
            })
        ));
    }
}
