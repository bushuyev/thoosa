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
use crate::runner::common::{DropIf, MaxParallelCtx, RunCtx};

use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_shared::{with_perf_log, ThoError, LOCK_TIMEOUT};
use tracing::error;

use crate as thoosa_common;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;

use thoosa_shared::mono_key;
use crate::runner::adjusters::max_parallel::MaxParallel;
use crate::runner::adjusters::max_parallel::MAX_PARALLEL;


///was supposed to be execution-long context, now used mostly to keep max_parallel so to avoid parallel async calls 'sharing' same permit, which may lead to deadlock, should never be passed as &mut to an async function - use RunCtx::branch instead.
#[derive(Named)]
pub struct RunCtxImpl {
    id: SynKey,
    device_idx: usize,

    max_other_parallel: MaxParallel,

    purge_gpu_closures:Option<Arc<Mutex<Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>>>>,
    branches:AtomicU32//for unique branch ids, never decreased
}

impl Debug for RunCtxImpl {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunCtxWithPermits")
            .field("device_idx", &self.device_idx)
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
    pub fn new(device_idx: usize, _id: &str,) -> Self {
        // #[cfg(any(test, feature = "mock_global_ctx"))]
        // let _ctx = crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder::install_global_ctx_mock();

        Self::with_cache_area(device_idx, mono_key!(0, _id).into())
    }


    fn with_cache_area(device_idx: usize, id: SynKey) -> Self {


        debug!("RunCtxWithPermits::with_cache_area: device_idx={device_idx} branch_name={id:?}");

        Self {
            id,
            device_idx,
            purge_gpu_closures: Some(Arc::new(Mutex::new(vec![]))),
            max_other_parallel: MaxParallel::MAX,
            branches: AtomicU32::new(0)
        }
    }
}

impl MaxParallelCtx for RunCtxImpl {

    fn set_max_other_parallel(&mut self, permit: MaxParallel) {
        self.max_other_parallel = permit
    }


    fn max_other_parallel(&self) -> MaxParallel {
        self.max_other_parallel
    }
}

#[async_trait]
impl RunCtx for RunCtxImpl {
    fn device_idx(&self) -> usize {
        self.device_idx
    }

    fn branch(
        &mut self,
        new_name: Option<&str>,
        with_new_cache_area: bool
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
        debug!("RunCtxWithPermits::branch: branch_full_id={branch_full_id:?}");

        let mut branched_ctx= if with_new_cache_area {

            let mut branched_ctx = RunCtxImpl::with_cache_area(self.device_idx, branch_full_id);

            debug!("RunCtxWithPermits::branch: branch_name={:?}", branched_ctx.id);

            branched_ctx.purge_gpu_closures = Some(Arc::new(Mutex::new(vec![])));//Arc::strong_count == 1 causes it to purge gpu caches on Drop

            branched_ctx

        } else {
            let mut branched_ctx = RunCtxImpl::with_cache_area(self.device_idx, branch_full_id);

            branched_ctx.purge_gpu_closures = self.purge_gpu_closures.as_ref().map(|v| Arc::clone(v));

            branched_ctx
        };


        let parent_permits = self.max_other_parallel.clone();

        debug!("RunCtxWithPermits::branch: parent_permits={parent_permits:?} set_return_to={:?}  from={}", branched_ctx.id, self.id, );

        branched_ctx.max_other_parallel = parent_permits;


        branched_ctx.device_idx = self.device_idx;


        branched_ctx
    }

    fn add_purge_gpu_closure(&mut self, cl:Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>>+ Send + Sync>) {
        if let Some(purge_gpu_closures) = self.purge_gpu_closures.as_ref() {
            with_perf_log(
                || purge_gpu_closures.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::add_purge_gpu_closure: timed out waiting for purge_gpu_closures lock"),
                "RunCtxWithPermits/add_purge_gpu_closure/purge_gpu_closures",
            )
            .push(cl);
        }
    }

}

impl Drop for RunCtxImpl {
    fn drop(&mut self) {

        if let Some(purge_gpu_closures) = self.purge_gpu_closures.take() {
            let count = Arc::strong_count(&purge_gpu_closures);

            debug!(
                "Drop for RunCtxWithPermits::drop: id={:?} count={count} purge_gpu_closures.len={}",
                self.id,
                with_perf_log(
                    || purge_gpu_closures.try_lock_for(LOCK_TIMEOUT).expect("RunCtxWithPermits::drop: timed out waiting for purge_gpu_closures lock"),
                    "RunCtxWithPermits/drop/purge_gpu_closures",
                )
                .len()
            );

            if count == 1 {// it's topmost ctx
                match Arc::into_inner(purge_gpu_closures) {
                    None => {
                        warn!("can't get inner from Arc with 1 strong ref in Drop for RunCtxWithPermits");
                    }
                    Some(m) => {
                        // #[cfg(any(test, feature = "mock_global_ctx"))]
                        // let _mock_ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

                        if let Err(e) = GlobalCtx::global().run_purge_gpu(m.into_inner()) {
                            error!("run_purge_gpu failed: {e:?} ");
                        }
                    }
                }
            }
        }
    }
}