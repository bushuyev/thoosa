use ambassador::delegatable_trait;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use mockall_double::double;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, CachableIdTr, SimpleId};
use crate::runner::common::{DropIf, RunCtx};
#[double]
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{Adjuster, RunCtxContext};
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
}

#[derive(Debug)]
pub struct DumpToCpuContextImpl<'a> {
    ids_to_exclude: &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>,
}

impl<'a> DumpToCpuContextImpl<'a> {
    pub fn new(
        ids_to_exclude: &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>,
    ) -> Self {
        Self { ids_to_exclude }
    }
}

impl<'a> DumpToCpuContext<'a> for DumpToCpuContextImpl<'a> {
    fn ids_to_exclude(&self) -> &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>> {
        self.ids_to_exclude
    }
}

impl<'a> RunCtxContext for DumpToCpuContextImpl<'a> {}

#[async_trait]
impl<'a, A, C> Adjuster<C> for DumpToCpu<A>
where
    A: Adjustable + RunCtx + Send,
    C: DumpToCpuContext<'a> + Send,
{
    type A = Arc<RwLock<A>>;

    async fn adjust(
        &self,
        adjustable: &<Self as Adjuster<C>>::A,
        adjust_context: &mut C,
    ) -> Result<<Self as Adjuster<C>>::A, ThoosaError> {


        let (device_idx, mut acquired_permits,  ctx_name)  = {
            let guard = adjustable.read();
            // assert_eq!(acquired_permits, MAX_PERMITS, "not enough permits {}", acquired_permits);

            (guard.device_idx(), guard.acquired_permits(), guard.name())
        };

        debug!("DumpToCpu::adjust: adjustable={adjustable:?} acquired_permits={acquired_permits}");

        if acquired_permits < MAX_PERMITS {
            let mut try_count = 0;
            loop {

                match GlobalCtx::global().acquire_many(device_idx, MAX_PERMITS - acquired_permits, Arc::new(SimpleId(format!("dump_to_cpu_for_{}", ctx_name).leak()))).await {
                    Ok(permits) => {
                        debug!("DumpToCpu::adjust: acquired {permits:?} on {try_count}");

                        adjustable.write().add_permit(permits);
                        break
                    }
                    Err(e) => {
                        debug!("DumpToCpu::adjust: can't acquire {acquired_permits} on {try_count} try, got {e:?}");
                        adjustable.write().drop_permits(DropIf::Always);
                        acquired_permits = 0;
                    }
                }

                try_count+=1;
                if try_count > 1000 {
                    return Err(ThoosaError::new("Can't get permits for dumping to CPU"));
                }
            }
        }

        let ids_to_exclude = adjust_context.ids_to_exclude();
        GlobalCtx::global()
            .dump_all_to_cpu(ids_to_exclude, device_idx)
            .await?;

        debug!("DumpToCpu::adjust done: is_all_on_cpu={}",GlobalCtx::global().is_all_on_cpu(device_idx));

        Ok(Arc::clone(adjustable))
    }
}
