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

use crate as thoosa_common;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use crate::runner::adjusters::max_parallel::MAX_PARALLEL;
use crate::runner::oom_handler::{Adjuster, AdjusterError, RunCtxContext};
#[derive(Debug)]
pub struct DumpToCpu<A>
where
    A: Adjustable + RunCtx,
{
    descr:String,
    pd: PhantomData<A>,
}

impl<A> DumpToCpu<A>
where
    A: Adjustable + RunCtx,
{
    pub fn new(decr: &str) -> Self {
        DumpToCpu {
            pd: Default::default(),
            descr: decr.to_string()
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
    pub fn new(ids_to_exclude: &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>) -> Self {
        Self {
            ids_to_exclude,
            dump_to_cpu_retry_count: 0
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


        let (device_idx)  = adjustable.device_idx();

        debug!("DumpToCpu::adjust: adjustable={adjustable:?}");

        adjustable.set_max_other_parallel(0);

        let ids_to_exclude = adjust_context.ids_to_exclude();
        debug!("DumpToCpu::adjust: acquired 100 max_parallel, dumping");

        #[cfg(tho_debug)]
        GlobalCtx::global().debug_caches(device_idx, "before dump_all_to_cpu", None).await;


        GlobalCtx::global()
            .dump_all_to_cpu(ids_to_exclude, device_idx, &self.descr)
            .await.map_err(|e|{
            warn!("DumpToCpu::adjust: dump_all_to_cpu failed: {e:?}");
            AdjusterError::<A>::SomethingElse
        })?;

        #[cfg(tho_debug)]
        GlobalCtx::global().debug_caches(device_idx, "after dump_all_to_cpu", None).await;


        Ok(adjustable.branch(None, false))
    }

    fn degrade(&self, adjust_context: &mut C) -> bool {

        false
    }
}
