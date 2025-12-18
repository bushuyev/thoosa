use crate::common::RunCtx;
use crate::global_ctx::GlobalCtx;
use crate::oom_handler::{Adjuster, RunCtxContext};
use crate::permits::MAX_PERMITS;
use ambassador::delegatable_trait;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, CachableIdTr};

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
        debug!("DumpToCpu::adjust: adjustable={adjustable:?}");

        let device_idx  = {
            let guard = adjustable.read();
            let acquired_permits = guard.acquired_permits();
            assert_eq!(acquired_permits, MAX_PERMITS, "not enough permits {}", acquired_permits);
            guard.device_idx()
        };

        let ids_to_exclude = adjust_context.ids_to_exclude();
        GlobalCtx::global()
            .dump_all_to_cpu(ids_to_exclude, device_idx)
            .await?;

        debug!("DumpToCpu::adjust done: is_all_on_cpu={}",GlobalCtx::global().is_all_on_cpu(device_idx));

        Ok(Arc::clone(adjustable))
    }
}
