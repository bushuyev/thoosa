use ambassador::delegatable_trait;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use mockall_double::double;
use thoosa_shared::{Cause, ThoError};
use thoosa_shared::traits::{Adjustable, CachableIdTr};
use crate::runner::common::{DropIf, RunCtx};
#[double]
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{Adjuster, RunCtxContext};
use crate::runner::permits::MAX_PERMITS;
use crate::simple_id;

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
    ) -> Result<<Self as Adjuster<C>>::A, ThoError> {


        let (device_idx, acquired_permits,  _ctx_name)  = {
            // assert_eq!(acquired_permits, MAX_PERMITS, "not enough permits {}", acquired_permits);

            (adjustable.device_idx(), adjustable.acquired_permits(), adjustable.name())
        };

        debug!("DumpToCpu::adjust: adjustable={adjustable:?} acquired_permits={acquired_permits}");

        if acquired_permits < MAX_PERMITS {
            // DumpToCpu is special: only dump when we hold MAX_PERMITS to avoid
            // dumping tensors still in use by other tasks.
            match GlobalCtx::global().acquire_many(
                device_idx,
                MAX_PERMITS - acquired_permits,
                Arc::new(simple_id!(format!("dump_to_cpu_for_{}", _ctx_name))),
            ).await {
                Ok(permits) => {
                    debug!("DumpToCpu::adjust: acquired {permits:?}");
                    adjustable.add_permit(permits);
                }
                Err(e) => {
                    debug!("DumpToCpu::adjust: can't acquire {acquired_permits} permits, got {e:?}");
                    adjustable.drop_permits(DropIf::Always);
                    return Err(ThoError::new_with_error(
                        "DumpToCpu::adjust: can't acquire MAX_PERMITS",
                        Cause::AdjusterFailed,
                    ));
                }
            }
        }

        let ids_to_exclude = adjust_context.ids_to_exclude();
        GlobalCtx::global()
            .dump_all_to_cpu(ids_to_exclude, device_idx)
            .await?;

        *adjust_context.dump_to_cpu_retry_count_mut() = 10001;//TODO to force false from degrade and proceed to next adjuster
        // adjustable.write().drop_permits(DropIf::Always);

        debug!("DumpToCpu::adjust done: is_all_on_cpu={}",GlobalCtx::global().is_all_on_cpu(device_idx));

        Ok(adjustable.branch("", false))
        // todo!()
    }

    fn degrade(&self, adjust_context: &mut C) -> bool {
        // DumpToCpu can't reduce work; we just retry acquiring MAX_PERMITS.
        let retry_count = adjust_context.dump_to_cpu_retry_count_mut();
        *retry_count += 1;

        *retry_count <= 1000 //TODO config
    }
}
