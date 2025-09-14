use crate::common::{MyError, RunCtx};
use crate::global_ctx::GlobalCtx;
use crate::oom_handler::{Adjustable, Adjuster, RunCtxContext};
use crate::permits::MAX_PERMITS;
use ambassador::delegatable_trait;
use async_trait::async_trait;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_shared::traits::CachableIdTr;

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
    ids_to_exclude: &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>
}

impl <'a> DumpToCpuContextImpl<'a> {
    pub fn new(ids_to_exclude:&'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>) -> Self {
        Self { ids_to_exclude }
    }
}

impl <'a> DumpToCpuContext<'a> for DumpToCpuContextImpl<'a> {
    fn ids_to_exclude(&self) -> &'a HashMap<&'static str, &'a HashSet<&'a dyn CachableIdTr>>{
        self.ids_to_exclude
    }
}

impl <'a> RunCtxContext for DumpToCpuContextImpl<'a> {}


#[async_trait]
impl<'a, A, C> Adjuster<C> for DumpToCpu<A>
where
    A: Adjustable + RunCtx + Send,
    C: DumpToCpuContext<'a> + Send,
{
    type A = Arc<RwLock<A>>;

    async fn adjust(
        &self,
        adjustable: &Self::A,
        adjust_context: &mut C,
    ) -> Result<Self::A, MyError> {
        println!("DumpToCpu::adjust: adjustable={adjustable:?}");

        assert_eq!(adjustable.read().acquired_permits(), MAX_PERMITS);

        let ids_to_exclude = adjust_context.ids_to_exclude();
        let device_idx = adjustable.read().device_idx();
        GlobalCtx::global()
            .dump_all_to_cpu(ids_to_exclude, device_idx)
            .await?;


        println!("DumpToCpu::adjust done: is_all_on_cpu={}", GlobalCtx::global().is_all_on_cpu(device_idx));

        Ok(Arc::clone(adjustable))
    }
}
