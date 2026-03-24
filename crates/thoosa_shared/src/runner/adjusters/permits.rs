
use crate::runner::oom_handler::{Adjuster, AdjusterError, RunCtxContext};
use ambassador::{delegatable_trait, Delegate};
use async_trait::async_trait;
use log::{debug, error};
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use mockall_double::double;
use crate::Cause;
// use tokio::io::{AsyncReadExt, AsyncWriteExt};
// use crate::common::runnable::Runnable;
use crate::traits::{Adjustable, CachableIdTr};
use crate::runner::common::{DropIf, RunCtx};
use crate::runner::permits::{NumPermits, MAX_PERMITS};
use crate::runner::adjusters::dump_to_cpu::*;
#[double]
use crate::runner::global_ctx::GlobalCtx;

pub struct PermitsAdjuster<C>
where
    C: Adjustable + RunCtx,
{
    pub max: NumPermits,
    pub step: NumPermits,
    pd: PhantomData<C>,
}

impl<C> Debug for PermitsAdjuster<C>
where
    C: Adjustable + RunCtx,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PermitsAdjuster")
            .field("max", &self.max)
            .field("step", &self.step)
            .finish()
    }
}

impl<C> PermitsAdjuster<C>
where
    C: Adjustable + RunCtx,
{
    pub fn new(
        max: NumPermits,
        step: NumPermits,
    ) -> Self {
        Self {
            max,
            step,
            pd: Default::default(),
        }
    }
}

#[delegatable_trait]
pub trait PermitsAdjustContext: Debug + Send + Sync {
    fn to_acquire(&self) -> NumPermits;
    fn to_acquire_mut(&mut self) -> &mut NumPermits;

    fn retry_count_mut(&mut self) -> &mut usize;
}

#[derive(Debug)]
pub struct PermitsAdjustContextImpl {
    to_acquire: NumPermits,
    retry_count: usize,
}

impl PermitsAdjustContextImpl {
    pub fn new() -> Self {
        Self {
            to_acquire: 0,
            retry_count: 0,
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

    fn retry_count_mut(&mut self) -> &mut usize {
        &mut self.retry_count
    }
}

impl RunCtxContext for PermitsAdjustContextImpl {}

#[async_trait]
impl<A, C> Adjuster<C> for PermitsAdjuster<A>
where
    A: Adjustable + RunCtx,
    C: PermitsAdjustContext,
{
    type A = A;

    async fn adjust(
        &self,
        mut adjustable: <Self as Adjuster<C>>::A,
        adjust_context: &mut C,
    ) -> Result<<Self as Adjuster<C>>::A, AdjusterError<<Self as Adjuster<C>>::A>> {
        let acquired_permits = adjustable.acquired_permits();
        if adjust_context.to_acquire() <= acquired_permits {
            //TODO warn
            return Ok(adjustable);
        }

        let permits_to_acquire = adjust_context.to_acquire() - acquired_permits;

        let adjustable_name = adjustable.id().clone();//TODO clone for debug only
        debug!("PermitsAdjuster::adjust: name={adjustable_name:?} thread={:?} self.max={} permits_to_acquire/{}/ = adjust_context.to_acquire()/{}/ - acquired_permits/{acquired_permits}/",std::thread::current().id(),self.max,   permits_to_acquire,      adjust_context.to_acquire());

        if permits_to_acquire > 0 {

            match super::acquire_many_waiting_once(
                adjustable.device_idx(),
                permits_to_acquire,
                Arc::new(adjustable.id().clone()) as Arc<dyn CachableIdTr>
            ).await {
                Ok(permit) => {
                    debug!("PermitsAdjuster::adjust: acquire_fn name={adjustable_name:?} acquired_permits={acquired_permits} permits_to_acquire={permits_to_acquire} acquired ok ");
                    adjustable.add_permit(permit);
                }

                //can't acquire permits,
                Err(te) => {
                    match te.what() {
                        Cause::OutOfPermits => {
                            #[cfg(not(tho_debug))]
                            debug!("PermitsAdjuster::adjust: acquire_fn name={adjustable_name:?} acquired_permits={acquired_permits} Error: {te:?} device_idx={}", adjustable.device_idx());
                            #[cfg(tho_debug)]
                            debug!("PermitsAdjuster::adjust: acquire_fn name={adjustable_name:?} acquired_permits={acquired_permits} Error: {te:?} device_idx={} debug_semaphore={:?}", adjustable.device_idx(), GlobalCtx::global().debug_semaphore(adjustable.device_idx()));

                           /* if adjustable.read().device_idx() != 0 {
                                //not on CPU, TODO
                                if let Err(e) = A::purge_gpu_caches(Arc::clone(adjustable)).await {
                                    debug!("PermitsAdjuster::adjust: purge_gpu_caches failed: {e:?}");
                                };
                            }*/
                            adjustable.drop_permits(DropIf::Always);

                            *adjust_context.to_acquire_mut() = 0;

                            debug!("PermitsAdjuster::adjust: acquire_fn name={adjustable_name:?} about to return AdjustFailed");

                            return Err(AdjusterError::AdjustFailed(adjustable));
                        }
                        inner => {
                            error!("PermitsAdjuster::adjust: e={inner:?}");
                            // Propagate unexpected errors to let caller decide.
                            return Err(AdjusterError::SomethingElse);
                        }
                    }
                }
            };
        }

        Ok(adjustable)
    }

    fn degrade(&self, adjust_context: &mut C) -> bool {
        let retry_count = {//if adjust_context.to_acquire() is reset in OutOfPermits/out of permits
            let retry_count = adjust_context.retry_count_mut();
            *retry_count += 1;
            *retry_count
        };

        if retry_count > 1000 {
            return false
        }


        let degrade = if adjust_context.to_acquire() < self.max {
            *adjust_context.to_acquire_mut() += self.step;
            if adjust_context.to_acquire() > MAX_PERMITS {
                *adjust_context.to_acquire_mut() = MAX_PERMITS;
            }

            true
        } else {
            false
        };

        debug!("PermitsAdjuster::degrade: adjust_context={adjust_context:?} degrade={degrade} retry_count={retry_count}");

        degrade
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
    use crate::runner::global_ctx::{mock_global_ctx, MockGlobalCtx};
    use crate::runner::global_ctx::mock_global_ctx::{AcquireBehavior, MockGlobalCtxBuildParams};
    use crate::runner::run_ctx_2::RunCtxWithPermits;

    fn ctx(_name: &'static str, want: NumPermits) -> PermitsAdjustContextImpl {
        let mut c = PermitsAdjustContextImpl::new();
        *c.to_acquire_mut() = want;
        c
    }

    #[tokio::test]
    async fn adjust_handles_out_of_permits() {
        let mock_ctx = MockGlobalCtx::global_context();
        mock_ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_with_params(MockGlobalCtxBuildParams {
                    acquire_many: AcquireBehavior::Deny,
                    ..Default::default()
                }),
            )) as &'static MockGlobalCtx
        });

        let adjustable = RunCtxWithPermits::new(0, "permits_adjust_out_of_permits");
        let adjuster = PermitsAdjuster::<RunCtxWithPermits>::new(100, 10);
        let mut ac = ctx("test", 10);

        let res = adjuster.adjust(adjustable, &mut ac).await;
        assert!(matches!(res, Err(AdjusterError::AdjustFailed(_))));
        // OutOfPermits path resets requested acquire to 0
        assert_eq!(ac.to_acquire(), 0);
    }

    #[tokio::test]
    async fn adjust_propagates_other_errors() {
        let mock_ctx = MockGlobalCtx::global_context();
        mock_ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build_with_params(MockGlobalCtxBuildParams {
                    acquire_many: AcquireBehavior::Boom("ups"),
                    ..Default::default()
                }),
            )) as &'static MockGlobalCtx
        });

        let adjustable = RunCtxWithPermits::new(0, "permits_adjust_other_errors");
        let adjuster = PermitsAdjuster::<RunCtxWithPermits>::new(100, 10);
        let mut ac = ctx("test", 10);

        let res = adjuster.adjust(adjustable, &mut ac).await;
        debug!("adjust_propagates_other_errors: res={res:?}");
        assert!(matches!(
            res,
            Err(AdjusterError::SomethingElse)
        ));
    }
}
