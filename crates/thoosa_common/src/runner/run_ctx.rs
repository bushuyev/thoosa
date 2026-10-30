use thoosa_shared::traits::MaxParallelCtx;
use async_trait::async_trait;
use futures_util::future::BoxFuture;
use log::debug;
use mockall_double::double;
use parking_lot::Mutex;
use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;

use thoosa_shared::traits::{Adjustable, RunCtx, WithId};
use thoosa_shared::ThoError;
use tracing::error;

use crate as thoosa_common;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;

use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallel};
use thoosa_shared::mono_key;

type PurgeGpuClosure = Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>;
// Arc drops this owner exactly once, after every branch has submitted its
// pending callbacks. A snapshot of strong_count cannot elect the final branch.
#[derive(Default)]
struct CacheAreaCleanup {
    closures: Mutex<Vec<PurgeGpuClosure>>,
    purger: Option<Arc<crate::runner::purger::Purger>>,
}

impl Drop for CacheAreaCleanup {
    fn drop(&mut self) {
        let closures = std::mem::take(self.closures.get_mut());
        if !closures.is_empty() {
            // Explicit sessions keep their cleanup owner alive through the last
            // cache-area branch. The facade remains for legacy context callers.
            let result = match &self.purger { Some(purger) => purger.run_purge_gpu(closures), None => GlobalCtx::global().run_purge_gpu(closures) };
            if let Err(error) = result {
                error!("cache-area cleanup submission failed: {error:?}");
            }
        }
    }
}

type SharedPurgeGpuClosures = Arc<CacheAreaCleanup>;


#[derive(Named)]
pub struct RunCtxImpl {
    id: SynKey,
    tho_device_idx: usize,

    max_other_parallel: MaxParallel,

    purge_gpu_closures:Option<SharedPurgeGpuClosures>,
    pending_purge_gpu_closures: Vec<PurgeGpuClosure>,
    residency: Option<Arc<crate::runner::global_ctx::ResidencyService>>,
    branches:AtomicU32//for unique branch ids, never decreased
}

impl Debug for RunCtxImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunCtxImpl")
            .field("device_idx", &self.tho_device_idx)
            .finish()
    }
}

impl Adjustable for RunCtxImpl {}

impl WithId for RunCtxImpl {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl RunCtxImpl {
    pub fn new_in(runtime: &crate::runner::global_ctx::Runtime, device: usize, name: &str) -> Self {
        let mut ctx = Self::new(device, name);
        ctx.set_residency(runtime.residency());
        ctx
    }

    fn set_residency(&mut self, residency: Arc<crate::runner::global_ctx::ResidencyService>) {
        self.purge_gpu_closures = Some(Arc::new(CacheAreaCleanup { closures: Mutex::new(vec![]), purger: Some(residency.cleanup()) }));
        self.residency = Some(residency);
    }

    pub fn new(tho_device_idx: usize, _id: &str,) -> Self {

        // mono_key only retains the diagnostic name in debug builds.
        Self::with_cache_area(tho_device_idx, mono_key!(0u32, _id).into())
    }


    fn with_cache_area(tho_device_idx: usize, id: SynKey) -> Self {


        debug!("RunCtxImpl::with_cache_area: device_idx={tho_device_idx} branch_name={id:?}");

        Self {
            id,
            tho_device_idx,
            purge_gpu_closures: Some(Arc::new(CacheAreaCleanup::default())),
            pending_purge_gpu_closures: vec![],
            max_other_parallel: MAX_PARALLEL,
            residency: None,
            branches: AtomicU32::new(0)
        }
    }
}

impl MaxParallelCtx for RunCtxImpl {

    fn set_max_other_parallel(&mut self, max_other_parallel: MaxParallel) {
        self.max_other_parallel = max_other_parallel
    }


    fn max_other_parallel(&self) -> MaxParallel {
        self.max_other_parallel
    }
}

#[async_trait]
impl RunCtx for RunCtxImpl {
    #[inline]
    fn tho_device_idx(&self) -> usize {
        self.tho_device_idx
    }

    fn branch(
        &mut self,
        with_new_cache_area: bool
    ) -> Self {

        let branches = self.branches.fetch_add(1, Ordering::SeqCst);

        let branch_full_id = &self.id + mono_key!(branches, concat!(module_path!(), ":", line!(), ":", column!()));

        debug!("RunCtxImpl::branch: branch_full_id={branch_full_id:?}");

        let mut branched_ctx= if with_new_cache_area {

            let mut branched_ctx = RunCtxImpl::with_cache_area(self.tho_device_idx, branch_full_id);
            if let Some(residency) = &self.residency { branched_ctx.set_residency(Arc::clone(residency)); }

            debug!("RunCtxImpl::branch: branch_name={:?}", branched_ctx.id);


            branched_ctx

        } else {
            let mut branched_ctx = RunCtxImpl::with_cache_area(self.tho_device_idx, branch_full_id);

            branched_ctx.residency = self.residency.clone();
            branched_ctx.purge_gpu_closures = self.purge_gpu_closures.as_ref().map(|v| Arc::clone(v));

            branched_ctx
        };


        let max_other_parallel = self.max_other_parallel.clone();

        debug!("RunCtxImpl::branch: max_other_parallel={max_other_parallel:?} set_return_to={:?}  from={}", branched_ctx.id, self.id, );

        branched_ctx.max_other_parallel = max_other_parallel;


        branched_ctx.tho_device_idx = self.tho_device_idx;


        branched_ctx
    }

    fn add_purge_gpu_closure(&mut self, cl:Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>>+ Send + Sync>) {
        if self.purge_gpu_closures.is_some() {
            self.pending_purge_gpu_closures.push(cl)
        }
    }

}

impl Drop for RunCtxImpl {
    fn drop(&mut self) {
        if let Some(owner) = self.purge_gpu_closures.take() {
            // Append before releasing this branch's ownership. The owner's Drop
            // cannot run until all concurrent branches have finished appending.
            owner.closures.lock().append(&mut self.pending_purge_gpu_closures);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::FutureExt;
    use std::sync::{Barrier, mpsc};
    use crate::runner::global_ctx::{MockGlobalCtx, mock_global_ctx};

    #[tokio::test]
    async fn test_concurrent_final_branches_submit_all_cleanup_exactly_once() {
        let (tx, rx) = mpsc::channel();
        let global = Box::leak(Box::new(mock_global_ctx::build().with_run_purge_gpu_handler(move |closures| {
            tx.send(closures).unwrap();
            Ok(())
        }).finish()));
        let global_context = MockGlobalCtx::global_context();
        global_context.expect().return_const(&*global);
        let executed = Arc::new(AtomicU32::new(0));

        for _ in 0..64 {
            let mut first = RunCtxImpl::new(1, "concurrent cleanup");
            let mut second = first.branch(false);
            for ctx in [&mut first, &mut second] {
                let executed = Arc::clone(&executed);
                ctx.add_purge_gpu_closure(Box::new(move || {
                    let executed = Arc::clone(&executed);
                    async move { executed.fetch_add(1, Ordering::SeqCst); Ok(()) }.boxed()
                }));
            }
            let barrier = Barrier::new(2);
            std::thread::scope(|scope| {
                let barrier = &barrier;
                scope.spawn(move || { barrier.wait(); drop(first); });
                scope.spawn(move || { barrier.wait(); drop(second); });
            });
            let closures = rx.try_recv().expect("final owner must submit cleanup");
            assert_eq!(closures.len(), 2);
            assert!(rx.try_recv().is_err(), "one submission per area");
            for cleanup in closures { cleanup().await.unwrap(); }
        }
        assert_eq!(executed.load(Ordering::SeqCst), 128);
    }

    #[test]
    fn test_new_cache_area_cleans_up_independently_of_parent() {
        let (tx, rx) = mpsc::channel();
        let global = Box::leak(Box::new(mock_global_ctx::build().with_run_purge_gpu_handler(move |closures| {
            tx.send(closures.len()).unwrap();
            Ok(())
        }).finish()));
        let global_context = MockGlobalCtx::global_context();
        global_context.expect().return_const(&*global);
        let mut parent = RunCtxImpl::new(1, "parent");
        let mut shared = parent.branch(false);
        let mut independent = parent.branch(true);
        shared.add_purge_gpu_closure(Box::new(|| async { Ok(()) }.boxed()));
        independent.add_purge_gpu_closure(Box::new(|| async { Ok(()) }.boxed()));
        drop(shared);
        assert!(rx.try_recv().is_err());
        drop(independent);
        assert_eq!(rx.try_recv().unwrap(), 1);
        drop(parent);
        assert_eq!(rx.try_recv().unwrap(), 1);
        assert!(rx.try_recv().is_err());
    }
}
