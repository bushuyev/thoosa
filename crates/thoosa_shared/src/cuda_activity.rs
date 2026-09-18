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

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use tokio::sync::mpsc;
    use crate::CudaActivityGate;

    #[test]
    fn test_reserved_enqueue_is_rejected_during_pause_and_retried_after() {
        let gate = CudaActivityGate::default();
        let (tx, mut rx) = mpsc::channel(1);
        let permit = tx.try_reserve().unwrap();
        let job = Box::new(42);
        let address = job.as_ref() as *const i32;
        let pause = gate.request_pause();
        assert!(rx.try_recv().is_err()); // Reset has already drained this queue.

        let rejected = gate.try_enqueue(permit, job).unwrap_err();
        assert_eq!(rejected.as_ref() as *const i32, address, "rejection must retain the original job");
        assert!(rx.try_recv().is_err());
        assert_eq!(tx.capacity(), 1, "rejection must release the reserved channel slot");

        gate.finish_pause(pause);
        gate.try_enqueue(tx.try_reserve().unwrap(), rejected).unwrap();
        assert_eq!(*rx.try_recv().unwrap(), 42);
    }

    #[test]
    fn test_enqueue_racing_pause_is_visible_to_purge_or_rejected() {
        // Synchronize the contenders rather than depending on sleeps. After
        // pause returns, a successful publication must already be in the queue.
        for _ in 0..64 {
            let gate = CudaActivityGate::default();
            let (tx, mut rx) = mpsc::channel(1);
            let permit = tx.try_reserve().unwrap();
            let start = Arc::new(Barrier::new(2));
            std::thread::scope(|threads| {
                let enqueue = threads.spawn({
                    let start = Arc::clone(&start);
                    let gate = &gate;
                    move || {
                        start.wait();
                        gate.try_enqueue(permit, 42)
                    }
                });
                start.wait();
                let pause = gate.request_pause();
                let purged = rx.try_recv().ok();
                match enqueue.join().unwrap() {
                    Ok(()) => assert_eq!(purged, Some(42)),
                    Err(job) => {
                        assert_eq!(job, 42);
                        assert_eq!(purged, None);
                    }
                }
                assert!(rx.try_recv().is_err(), "nothing may be enqueued after the purge");
                gate.finish_pause(pause);
            });
        }
    }

    #[test]
    fn test_old_reset_completion_cannot_reopen_new_pause() {
        let gate = CudaActivityGate::default();
        let (tx, mut rx) = mpsc::channel(1);
        gate.try_enqueue(tx.try_reserve().unwrap(), 1).unwrap();
        let first = gate.request_pause();
        assert_eq!(rx.try_recv().unwrap(), 1, "jobs accepted before pause remain visible to purge");
        gate.finish_pause(first);
        let second = gate.request_pause();
        gate.finish_pause(first);
        assert_eq!(gate.try_enqueue(tx.try_reserve().unwrap(), 2), Err(2));
        gate.finish_pause(second);
        gate.try_enqueue(tx.try_reserve().unwrap(), 3).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 3);
    }
}
