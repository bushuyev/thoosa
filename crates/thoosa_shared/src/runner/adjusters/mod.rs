use std::sync::Arc;
use mockall_double::double;
use crate::runner::permits::{NumPermits, RunnerPerm};
use crate::traits::CachableIdTr;
use crate::{Cause, ThoError};
#[double]
use crate::runner::global_ctx::GlobalCtx;

pub mod dump_to_cpu;
pub mod permits;
pub mod splits;

pub(crate) async fn acquire_many_waiting_once(device_idx: usize, num_permits: NumPermits, requester_id: Arc<dyn CachableIdTr>) -> Result<RunnerPerm, ThoError> {
    let global_ctx = GlobalCtx::global();
    match global_ctx.acquire_many(device_idx, num_permits, Arc::clone(&requester_id)).await {
        Err(te) if matches!(te.what(), Cause::OutOfPermits) => {
            global_ctx.wait_for_permits_hint(device_idx).await;
            global_ctx.acquire_many(device_idx, num_permits, requester_id).await
        }
        other => other,
    }
}
