use ambassador::delegatable_trait;
use async_trait::async_trait;
use log::{debug, warn};
use mockall_double::double;
use std::fmt::Debug;
use std::marker::PhantomData;
use thoosa_shared::traits::Adjustable;
use thoosa_shared::traits::{MaxParallelCtx, RunCtx};

use crate as thoosa_common;
use crate::runner::oom_handler::{Adjuster, AdjusterError, RunCtxContext};
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_shared::tracked_await::t;
#[derive(Debug)]
pub struct DumpToCpu<A>
where
    A: Adjustable + RunCtx,
{
    descr: String,
    pd: PhantomData<A>,
}

impl<A> DumpToCpu<A>
where
    A: Adjustable + RunCtx,
{
    pub fn new(decr: &str) -> Self {
        DumpToCpu { pd: Default::default(), descr: decr.to_string() }
    }
}

#[delegatable_trait]
pub trait DumpToCpuContext: Debug + RunCtxContext {
    fn dump_to_cpu_retry_count_mut(&mut self) -> &mut usize;
}

#[derive(Debug)]
pub struct DumpToCpuContextImpl {
    dump_to_cpu_retry_count: usize,
}

impl DumpToCpuContextImpl {
    pub fn new() -> Self {
        Self { dump_to_cpu_retry_count: 0 }
    }
}

impl DumpToCpuContext for DumpToCpuContextImpl {
    fn dump_to_cpu_retry_count_mut(&mut self) -> &mut usize {
        &mut self.dump_to_cpu_retry_count
    }
}

impl RunCtxContext for DumpToCpuContextImpl {}

#[async_trait]
impl<'a, A, C> Adjuster<C> for DumpToCpu<A>
where
    A: Adjustable + RunCtx + MaxParallelCtx + Send,
    C: DumpToCpuContext + Send,
{
    type A = A;

    async fn adjust(&self, mut adjustable: <Self as Adjuster<C>>::A, _adjust_context: &mut C) -> Result<<Self as Adjuster<C>>::A, AdjusterError<<Self as Adjuster<C>>::A>> {
        let device_idx = adjustable.tho_device_idx();

        debug!("DumpToCpu::adjust: adjustable={adjustable:?}");

        adjustable.set_max_other_parallel(0);

        debug!("DumpToCpu::adjust: acquired 100 max_parallel, dumping");

        #[cfg(tho_debug)]
        t!(GlobalCtx::global().debug_caches(device_idx, "before dump_all_to_cpu")).await;

        if let Err(error) = t!(GlobalCtx::global().dump_all_to_cpu(device_idx, &self.descr)).await {
            warn!("DumpToCpu::adjust: dump_all_to_cpu failed: {error:?}");
            return Err(AdjusterError::SomethingElse(error));
        }

        #[cfg(tho_debug)]
        t!(GlobalCtx::global().debug_caches(device_idx, "after dump_all_to_cpu")).await;

        Ok(adjustable.branch(false))
    }

    fn degrade(&self, _adjust_context: &mut C) -> bool {
        false
    }
}
