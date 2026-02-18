use async_trait::async_trait;
use futures_util::future::BoxFuture;
use log::{debug, warn};
use parking_lot::Mutex;
use std::fmt::{Debug, Formatter};
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use mockall_double::double;
use thoosa_named_derive::Named;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::runner::common::{DropIf, PermitsCtx, RunCtx};
#[double]
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::permits::{NumPermits, RunnerPerm};
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_shared::ThoError;
use crate::LOCK_TIMEOUT;
use tracing::error;
#[cfg(any(test, feature = "mock_global_ctx"))]
use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use crate::mono_key;
use crate::runner::permits::MAX_PERMITS;


///was supposed to be execution-long context, now used mostly to keep permits so to avoid parallel async calls 'sharing' same permit, which may lead to deadlock, should never be passed as &mut to an async function - use RunCtx::branch instead.
#[derive(Named)]
pub struct RunCtxWithPermits {
    id: SynKey,
    device_idx: usize,
    cache_area: u64,

    permits: Arc<Mutex<Vec<RunnerPerm>>>,
    return_to: Option<(SynKey, Arc<Mutex<Vec<RunnerPerm>>>)>,

    purge_gpu_closures:Option<Arc<Mutex<Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>>>>,
    branches:AtomicU32//for unique branch ids, never decreased
}

impl Debug for RunCtxWithPermits {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunCtx2")
            .field("device_idx", &self.device_idx)
            .field("cache_area", &self.cache_area)
            .finish()
    }
}

impl Adjustable for RunCtxWithPermits {}

impl WithId for RunCtxWithPermits {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl RunCtxWithPermits {
    pub fn new(device_idx: usize, _id: &str,) -> Self {
        Self::with_cache_area(device_idx, mono_key!(0, _id).into(), GlobalCtx::global().next_cache_area().expect("no more cache areas"))
    }

    #[cfg(test)]
    pub fn with_cache_area_(device_idx: usize, id: SynKey, cache_area: u64) -> Self {
        RunCtxWithPermits::with_cache_area(device_idx, id, cache_area)
    }

    fn with_cache_area(device_idx: usize, id: SynKey, cache_area: u64) -> Self {
        assert_ne!(cache_area, 0);


        debug!("RunCtx2::with_cache_area: device_idx={device_idx} branch_name={id:?}");

        Self {
            id,
            device_idx,
            cache_area,
            purge_gpu_closures: Some(Arc::new(Mutex::new(vec![]))),
            permits: Arc::new(Mutex::new(Vec::new())),
            return_to: None,
            branches: AtomicU32::new(0)
        }
    }


    pub fn set_return_to(&mut self, return_to: (SynKey, Arc<Mutex<Vec<RunnerPerm>>>)) {
        self.return_to = Some(return_to);
    }

}

impl PermitsCtx for RunCtxWithPermits {

    fn add_permit(&mut self, permit: RunnerPerm) {
        self.permits.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::add_permit: timed out waiting for permits lock").push(permit);
    }

    fn drop_permits(&mut self, drop_if: DropIf) {
        let mut permits = self.permits.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::drop_permits: timed out waiting for permits lock");
        let acquired_permits: NumPermits = permits.iter().map(|p| p.num_permits()).sum();
        debug!("TorchRunCtx: acquired_permits={acquired_permits} ctx.id={:?} drop_if={drop_if:?}",self.id);

        match drop_if {
            DropIf::Always => {
                debug!("TorchRunCtx: ctx.id={:?} dropping permits", self.id);
                permits.clear();
            }
            DropIf::IfNotMax => {
                if acquired_permits < MAX_PERMITS {
                    debug!("TorchRunCtx: ctx.id={:?} dropping permits", self.id);
                    permits.clear();
                }
            }
        }
    }

    fn acquired_permits(&self) -> NumPermits {
        self.permits.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::acquired_permits: timed out waiting for permits lock").iter().map(|p| p.num_permits()).sum()
    }
}

#[async_trait]
impl RunCtx for RunCtxWithPermits {
    fn device_idx(&self) -> usize {
        self.device_idx
    }

    fn cache_area(&self) -> u64 {
        self.cache_area
    }

    fn branch(
        &mut self,
        new_name: Option<&str>,
        with_new_cache_area: bool,
        give_permits_back_to_parent: bool//for OOM handler to pass permits between DumpToCpu(needs MAX_PERMITS) and further cycles
    ) -> Self {

        let branches = self.branches.fetch_add(1, Ordering::SeqCst);

        let branch_full_id = match new_name {
            None => {
                self.id.clone()
            }
            Some(branch_name) => {
                &self.id + mono_key!(branches, branch_name)
            }
        };
        debug!("RunCtx2::branch: branch_full_id={branch_full_id:?}");

        let mut branched_ctx= if with_new_cache_area {
            let new_cache_area = GlobalCtx::global().next_cache_area().expect("ran out of cache areas");

            let mut branched_ctx = RunCtxWithPermits::with_cache_area(self.device_idx, branch_full_id, new_cache_area);

            debug!("RunCtx2::branch: branch_name={:?}  new_cache_area={new_cache_area}", branched_ctx.id);

            branched_ctx.cache_area = new_cache_area;
            branched_ctx.purge_gpu_closures = Some(Arc::new(Mutex::new(vec![])));//Arc::strong_count == 1 causes it to purge gpu caches on Drop

            branched_ctx

        } else {
            let mut branched_ctx = RunCtxWithPermits::with_cache_area(self.device_idx, branch_full_id, self.cache_area);

            branched_ctx.purge_gpu_closures = self.purge_gpu_closures.as_ref().map(|v| Arc::clone(v));

            branched_ctx
        };


        let parent_permits = mem::replace(&mut self.permits, Arc::new(Mutex::new(Vec::new())));

        branched_ctx.permits = parent_permits;

        if give_permits_back_to_parent {
            branched_ctx.set_return_to((self.id.clone(), Arc::clone(&self.permits)));
        }
        branched_ctx.device_idx = self.device_idx;

        GlobalCtx::global().debug_semaphore(self.device_idx).move_permits(&self.id, &branched_ctx.id);


        branched_ctx
    }

    fn add_purge_gpu_closure(&mut self, cl:Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>>+ Send + Sync>) {
        if let Some(purge_gpu_closures) = self.purge_gpu_closures.as_ref() {
            purge_gpu_closures.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::add_purge_gpu_closure: timed out waiting for purge_gpu_closures lock").push(cl);
        }
    }

}

impl Drop for RunCtxWithPermits {
    fn drop(&mut self) {

        if let Some(purge_gpu_closures) = self.purge_gpu_closures.take() {
            let count = Arc::strong_count(&purge_gpu_closures);

            debug!("Drop for RunCtx2::drop: id={:?} count={count} purge_gpu_closures.len={}", self.id, purge_gpu_closures.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::drop: timed out waiting for purge_gpu_closures lock").len());

            if count == 1 {// it's topmost ctx
                match Arc::into_inner(purge_gpu_closures) {
                    None => {
                        warn!("can't get inner from Arc with 1 strong ref in Drop for RunCtx2");
                    }
                    Some(m) => {
                        #[cfg(any(test, feature = "mock_global_ctx"))]
                        let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

                        if let Err(e) = GlobalCtx::global().run_purge_gpu(m.into_inner(), self.cache_area) {
                            error!("run_purge_gpu failed: {e:?} ");
                        }
                    }
                }
            }
        }

        if let Some((return_to_id, return_to_perms)) = self.return_to.take() {
            debug!("Drop for RunCtx2::drop: id={:?} giving permits back to parent self.perms.permits.len={}", self.id, self.permits.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::drop: timed out waiting for permits lock").len());

            let mut local_perms = mem::take(&mut *self.permits.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::drop: timed out waiting for permits lock"));
            if !local_perms.is_empty() {
                return_to_perms.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::drop: timed out waiting for return_to_perms lock").append(&mut local_perms);

                // #[cfg(debug_assertions)] TODO
                {
                    let id_len = self.id.as_slice().len();
                    assert!(id_len > 1);
                    GlobalCtx::global().debug_semaphore(self.device_idx).move_permits(&self.id, &return_to_id);
                }
            }

        } else {
            debug!("Drop for RunCtx2::drop: id={:?} dropping permits self.perms.permits.len={}", self.id, self.permits.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::drop: timed out waiting for permits lock").len());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RunCtxWithPermits;
    use crate::runner::common::{PermitsCtx, RunCtx};
    use crate::runner::permits::RunnerPerm;
    use std::sync::Arc;
    use tokio::sync::Semaphore;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;

    fn make_ctx(name: &str) -> RunCtxWithPermits {
        let mut ctx = RunCtxWithPermits::with_cache_area(0, name.into(), 1);
        ctx.purge_gpu_closures = None; // avoid GlobalCtx::global() in Drop
        ctx
    }

    fn make_permit(sem: &Arc<Semaphore>, permits: u32) -> RunnerPerm {
        RunnerPerm::new(Arc::clone(sem).try_acquire_many_owned(permits).expect("permit"))
    }

    #[test]
    fn branch_moves_permits_to_child() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let sem = Arc::new(Semaphore::new(4));
        let mut parent = make_ctx("parent");
        parent.add_permit(make_permit(&sem, 2));
        assert_eq!(parent.acquired_permits(), 2);
        let child = parent.branch(Some("child"), false, false);
        assert_eq!(parent.acquired_permits(), 0);
        assert_eq!(child.acquired_permits(), 2);
        assert!(!Arc::ptr_eq(&parent.permits, &child.permits));
    }

    #[test]
    fn branched_ctx_returns_permits_to_parent_on_drop() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let sem = Arc::new(Semaphore::new(4));
        let mut parent = make_ctx("parent");
        parent.add_permit(make_permit(&sem, 3));
        {
            let child = parent.branch(Some("child"), false, true);
            assert_eq!(parent.acquired_permits(), 0);
            assert_eq!(child.acquired_permits(), 3);
        }
        assert_eq!(parent.acquired_permits(), 3);
    }
}
