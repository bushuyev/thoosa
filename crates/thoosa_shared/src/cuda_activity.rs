use std::{future::Future, sync::Arc};

use crate::{Cause, CudaActivityPauseTicket, ThoError, cuda_activity_gate};

tokio::task_local! {
    static CURRENT_ACTIVITY: CudaActivityScope;
}

/// Admitted work and its children may drain while admission is paused. Clones
/// share one registration, so reset also waits for detached children to finish.
#[derive(Clone, Debug)]
pub struct CudaActivityScope {
    _lease: Option<Arc<ActivityLease>>,
}

#[derive(Debug)]
struct ActivityLease;

impl Drop for ActivityLease {
    fn drop(&mut self) {
        let gate = cuda_activity_gate();
        let mut state = gate.state.lock();
        state.active -= 1;
        gate.notify.notify_waiters();
    }
}

impl CudaActivityScope {
    pub fn current() -> Option<Self> {
        CURRENT_ACTIVITY.try_with(Clone::clone).ok()
    }

    /// A new root must atomically register before a reset can close admission.
    /// Executors leave unadmitted jobs queued rather than awaiting inside a job.
    pub fn try_enter() -> Option<Self> {
        let mut state = cuda_activity_gate().state.lock();
        if state.paused {
            return None;
        }
        state.active += 1;
        Some(Self { _lease: Some(Arc::new(ActivityLease)) })
    }

    /// Copy callers can already hold tensor/cache locks needed by reset. Reject
    /// an unadmitted operation during pause rather than wait while holding them.
    pub fn for_operation() -> Result<Self, ThoError> {
        Self::current().or_else(Self::try_enter).ok_or_else(|| {
            ThoError::new_with_cause("CUDA operation interrupted by context reset", Cause::ContextReset)
        })
    }

    pub async fn run<F: Future>(self, future: F) -> F::Output {
        CURRENT_ACTIVITY.scope(self, future).await
    }

    /// Reset cleanup runs only after admitted activity drained. It must not wait
    /// for the admission gate that this same reset owns.
    pub async fn run_reset<F: Future>(ticket: CudaActivityPauseTicket, future: F) -> F::Output {
        {
            let state = cuda_activity_gate().state.lock();
            assert!(ticket.owner && state.paused && ticket.generation == state.generation && state.active == 0, "reset scope requires an idle owned pause");
        }
        Self { _lease: None }.run(future).await
    }
}
