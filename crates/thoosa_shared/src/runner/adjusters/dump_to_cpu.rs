use ambassador::delegatable_trait;
use async_trait::async_trait;
use log::{debug, warn};
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use mockall_double::double;
use thoosa_shared::traits::{Adjustable, CachableIdTr};
use crate::runner::common::{DropIf, RunCtx};
#[double]
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{Adjuster, AdjusterError, RunCtxContext};
use crate::runner::permits::MAX_PERMITS;

#[derive(Debug)]
pub struct DumpToCpu<A>
where
    A: Adjustable + RunCtx,
{
    pd: PhantomData<A>,
}

impl<A> DumpToCpu<A>
where
    A: Adjustable + RunCtx,
{
    pub fn new() -> Self {
        DumpToCpu {
            pd: Default::default(),
        }
    }
}

#[delegatable_trait]
pub trait DumpToCpuContext<'a>: Debug + RunCtxContext {
    fn ids_to_exclude(&self) -> &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>;
    fn dump_to_cpu_retry_count_mut(&mut self) -> &mut usize;
}

#[derive(Debug)]
pub struct DumpToCpuContextImpl<'a> {
    ids_to_exclude: &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>,
    dump_to_cpu_retry_count: usize,
}

impl<'a> DumpToCpuContextImpl<'a> {
    pub fn new(
        ids_to_exclude: &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>,
    ) -> Self {
        Self {
            ids_to_exclude,
            dump_to_cpu_retry_count: 0,
        }
    }
}

impl<'a> DumpToCpuContext<'a> for DumpToCpuContextImpl<'a> {
    fn ids_to_exclude(&self) -> &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>> {
        self.ids_to_exclude
    }

    fn dump_to_cpu_retry_count_mut(&mut self) -> &mut usize {
        &mut self.dump_to_cpu_retry_count
    }
}

impl<'a> RunCtxContext for DumpToCpuContextImpl<'a> {}

#[async_trait]
impl<'a, A, C> Adjuster<C> for DumpToCpu<A>
where
    A: Adjustable + RunCtx + Send,
    C: DumpToCpuContext<'a> + Send,
{
    type A = A;

    async fn adjust(
        &self,
        mut adjustable: <Self as Adjuster<C>>::A,
        adjust_context: &mut C,
    ) -> Result<<Self as Adjuster<C>>::A, AdjusterError<<Self as Adjuster<C>>::A>> {


        let (device_idx, acquired_permits)  = {
            (adjustable.device_idx(), adjustable.acquired_permits())
        };

        debug!("DumpToCpu::adjust: adjustable={adjustable:?} acquired_permits={acquired_permits}");

        if adjustable.acquired_permits() < MAX_PERMITS {
            // DumpToCpu is special: only dump when we hold MAX_PERMITS to avoid
            // dumping tensors still in use by other tasks.
            let requested_permits = MAX_PERMITS - adjustable.acquired_permits();
            match GlobalCtx::global().acquire_many(
                adjustable.device_idx(),
                requested_permits,
                Arc::new(adjustable.id().clone()) as Arc<dyn CachableIdTr>,
            ).await {
                Ok(permits) => {
                    debug!("DumpToCpu::adjust: acquired {permits:?} adjustable.id={:?}", adjustable.id());
                    adjustable.add_permit(permits);
                }
                Err(e) => {
                    debug!("DumpToCpu::adjust: can't acquire {requested_permits} permits, adjustable.id={:?} retry_count={} got {e:?}  ", adjustable.id(), adjust_context.dump_to_cpu_retry_count_mut());
                    adjustable.drop_permits(DropIf::Always);

                    GlobalCtx::global().wait_for_permits_hint(adjustable.device_idx()).await;

                    return Err(AdjusterError::AdjustFailed(adjustable));
                }
            }
        }

        let ids_to_exclude = adjust_context.ids_to_exclude();
        debug!("DumpToCpu::adjust: acquired 100 permits, dumping");

        GlobalCtx::global().debug_caches(device_idx).await;
        GlobalCtx::global()
            .dump_all_to_cpu(ids_to_exclude, device_idx)
            .await.map_err(|e|{
            warn!("DumpToCpu::adjust: dump_all_to_cpu failed: {e:?}");
            AdjusterError::<A>::SomethingElse
        })?;
        GlobalCtx::global().debug_caches(device_idx).await;

        *adjust_context.dump_to_cpu_retry_count_mut() = 10001;//TODO to force false from degrade and proceed to next adjuster
        // adjustable.write().drop_permits(DropIf::Always);

        debug!("DumpToCpu::adjust done: is_all_on_cpu={}",GlobalCtx::global().is_all_on_cpu(device_idx));

        Ok(adjustable.branch(None, false, false))
    }

    fn degrade(&self, adjust_context: &mut C) -> bool {
        // DumpToCpu can't reduce work; we just retry acquiring MAX_PERMITS.
        let retry_count = adjust_context.dump_to_cpu_retry_count_mut();
        *retry_count += 1;

        debug!("DumpToCpu::degrade: retry_count={retry_count}");

        *retry_count <= 1000 //TODO config
    }
}
