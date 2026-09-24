use std::{future::Future, sync::Arc};

use crate::{CudaActivityPauseTicket, CudaRecoveryState, ThoError, cuda_activity_gate};

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
    pub fn try_enter() -> Result<Option<Self>, ThoError> {
        let mut state = cuda_activity_gate().state.lock();
        match &state.recovery {
            CudaRecoveryState::Recovering { .. } => return Ok(None),
            CudaRecoveryState::Failed(failure) => return Err(ThoError::CudaRecoveryFailed(Arc::clone(failure))),
            CudaRecoveryState::Running => {}
        }
        state.active += 1;
        Ok(Some(Self { _lease: Some(Arc::new(ActivityLease)) }))
    }

    /// Copy callers can already hold tensor/cache locks needed by reset. Reject
    /// an unadmitted operation during pause rather than wait while holding them.
    pub fn for_operation() -> Result<Self, ThoError> {
        // Existing scopes may drain during recovery, but terminal failure must
        // never authorize another CUDA operation through an inherited scope.
        if let Some(error) = cuda_activity_gate().failure() {
            return Err(error);
        }
        if let Some(scope) = Self::current() {
            return Ok(scope);
        }
        Self::try_enter()?.ok_or_else(|| {
            ThoError::new_with_cause("CUDA operation interrupted by context reset", ThoError::ContextReset)
        })
    }

    pub async fn run<F: Future>(self, future: F) -> F::Output {
        CURRENT_ACTIVITY.scope(self, future).await
    }

    /// Reset cleanup runs only after admitted activity drained. It must not wait
    /// for the admission gate that this same reset owns.
    pub async fn run_reset<T, F: Future<Output = Result<T, ThoError>>>(ticket: CudaActivityPauseTicket, future: F) -> Result<T, ThoError> {
        {
            let state = cuda_activity_gate().state.lock();
            if let CudaRecoveryState::Failed(failure) = &state.recovery {
                return Err(ThoError::CudaRecoveryFailed(Arc::clone(failure)));
            }
            assert!(ticket.owner && matches!(state.recovery, CudaRecoveryState::Recovering { .. }) && ticket.cycle == state.cycle && state.active == 0, "reset scope requires an idle owned pause");
        }
        Self { _lease: None }.run(future).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use tokio::sync::mpsc;
    use crate::{CudaAction, CudaActivityGate, CudaDeviceGeneration, CudaRecoveryFailure, CudaRecoveryState, SyncStageRunnerError, ThoError};

    #[test]
    fn test_reserved_enqueue_is_rejected_during_pause_and_retried_after() {
        let gate = CudaActivityGate::default();
        let (tx, mut rx) = mpsc::channel(1);
        let permit = tx.try_reserve().unwrap();
        let job = Box::new(42);
        let address = job.as_ref() as *const i32;
        let pause = gate.request_pause(ThoError::ContextReset).unwrap();
        assert!(rx.try_recv().is_err()); // Reset has already drained this queue.

        let (rejected, error) = gate.try_enqueue(permit, job).unwrap_err();
        assert!(matches!(error.what(), ThoError::ContextReset));
        assert_eq!(rejected.as_ref() as *const i32, address, "rejection must retain the original job");
        assert!(rx.try_recv().is_err());
        assert_eq!(tx.capacity(), 1, "rejection must release the reserved channel slot");

        gate.finish_pause(pause, &[]);
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
                let pause = gate.request_pause(ThoError::ContextReset).unwrap();
                let purged = rx.try_recv().ok();
                match enqueue.join().unwrap() {
                    Ok(()) => assert_eq!(purged, Some(42)),
                    Err((job, error)) => {
                        assert_eq!(job, 42);
                        assert!(matches!(error.what(), ThoError::ContextReset));
                        assert_eq!(purged, None);
                    }
                }
                assert!(rx.try_recv().is_err(), "nothing may be enqueued after the purge");
                gate.finish_pause(pause, &[]);
            });
        }
    }

    #[test]
    fn test_old_reset_completion_cannot_reopen_new_pause() {
        let gate = CudaActivityGate::default();
        let (tx, mut rx) = mpsc::channel(1);
        gate.try_enqueue(tx.try_reserve().unwrap(), 1).unwrap();
        let first = gate.request_pause(ThoError::ContextReset).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 1, "jobs accepted before pause remain visible to purge");
        gate.finish_pause(first, &[]);
        let second = gate.request_pause(ThoError::ContextReset).unwrap();
        gate.finish_pause(first, &[]);
        let (job, error) = gate.try_enqueue(tx.try_reserve().unwrap(), 2).unwrap_err();
        assert_eq!(job, 2);
        assert!(matches!(error.what(), ThoError::ContextReset));
        gate.finish_pause(second, &[]);
        gate.try_enqueue(tx.try_reserve().unwrap(), 3).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 3);
    }

    fn original_failure() -> ThoError {
        ThoError::Cuda { action: CudaAction::NeedsRestart, msg: "illegal address".into() }.context("stage failed")
    }

    #[derive(Default)]
    struct WakeCount(std::sync::atomic::AtomicUsize);

    impl std::task::Wake for WakeCount {
        fn wake(self: Arc<Self>) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn test_failed_recovery_wakes_all_waiters_and_keeps_admission_closed() {
        use std::future::Future;
        use std::sync::atomic::Ordering::SeqCst;

        // Use a private gate so a terminal state cannot poison unrelated tests.
        let gate = CudaActivityGate::default();
        let ticket = gate.request_pause(original_failure()).unwrap();
        let mut admission = Box::pin(gate.wait_if_paused());
        let mut completion = Box::pin(gate.wait_for_pause_to_finish(ticket));
        let wakes = Arc::new(WakeCount::default());
        let waker = std::task::Waker::from(Arc::clone(&wakes));
        let mut cx = std::task::Context::from_waker(&waker);
        assert!(admission.as_mut().poll(&mut cx).is_pending());
        assert!(completion.as_mut().poll(&mut cx).is_pending());

        gate.fail_pause(ticket, Some(ThoError::oom().context("replacement failed")));
        assert_eq!(wakes.0.load(SeqCst), 2, "both registered waiters must be notified");
        for error in [admission.await.unwrap_err(), completion.await.unwrap_err(), gate.check_admission().unwrap_err()] {
            assert!(error.is_cuda_recovery_failed());
            assert!(!error.is_oom() && !error.is_retryable_oom() && !error.needs_cuda_ctx_reset());
            let ThoError::CudaRecoveryFailed(failure) = error.what() else { panic!("terminal error was lost") };
            assert!(failure.original.needs_cuda_ctx_reset());
            assert!(failure.recovery.as_ref().unwrap().is_oom());
            assert!(error.to_string().contains("stage failed") && error.to_string().contains("replacement failed"));
        }

        gate.finish_pause(ticket, &[1]);
        assert!(matches!(gate.recovery_state(), CudaRecoveryState::Failed(_)));
        assert!(gate.is_paused() && !gate.is_recovering());
        assert_eq!(gate.device_generation(1), CudaDeviceGeneration(0));
        assert!(gate.request_pause(ThoError::ContextReset).unwrap_err().is_cuda_recovery_failed());
        assert!(gate.wait_until_paused().await.unwrap_err().is_cuda_recovery_failed());
        let (tx, mut rx) = mpsc::channel(1);
        let (job, error) = gate.try_enqueue(tx.try_reserve().unwrap(), 42).unwrap_err();
        assert_eq!(job, 42);
        assert!(error.is_cuda_recovery_failed());
        assert_eq!(tx.capacity(), 1);
        assert!(rx.try_recv().is_err());
    }

    #[tokio::test]
    async fn test_successful_recovery_advances_only_recovered_device_generations() {
        let gate = CudaActivityGate::default();
        assert!(matches!(gate.recovery_state(), CudaRecoveryState::Running));
        gate.wait_if_paused().await.unwrap();
        let first = gate.request_pause(original_failure()).unwrap();
        let joined = gate.request_pause(ThoError::oom()).unwrap();
        assert!(!joined.is_owner());
        assert_eq!(first.cycle(), joined.cycle());
        assert!(matches!(gate.check_admission().unwrap_err().what(), ThoError::ContextReset));
        gate.wait_until_paused().await.unwrap();
        gate.finish_pause(joined, &[2]);
        gate.fail_pause(joined, None);
        assert!(gate.is_recovering());
        assert_eq!(gate.device_generation(2), CudaDeviceGeneration(0));

        let mut waiter = Box::pin(gate.wait_if_paused());
        assert!(futures::poll!(waiter.as_mut()).is_pending());
        gate.finish_pause(first, &[1, 1, 3]);
        waiter.await.unwrap();
        gate.wait_for_pause_to_finish(joined).await.unwrap();
        assert_eq!(gate.device_generation(1), CudaDeviceGeneration(1));
        assert_eq!(gate.device_generation(2), CudaDeviceGeneration(0));
        assert_eq!(gate.device_generation(3), CudaDeviceGeneration(1));

        let second = gate.request_pause(original_failure()).unwrap();
        assert_ne!(first.cycle(), second.cycle());
        gate.finish_pause(first, &[2]);
        gate.fail_pause(first, Some(ThoError::new("stale failure")));
        assert!(gate.is_recovering());
        gate.wait_for_pause_to_finish(first).await.unwrap();
        gate.finish_pause(second, &[2]);
        assert_eq!(gate.device_generation(1), CudaDeviceGeneration(1));
        assert_eq!(gate.device_generation(2), CudaDeviceGeneration(1));
        gate.check_admission().unwrap();
    }

    #[test]
    fn test_terminal_recovery_error_survives_context_and_job_wrapping() {
        let original = original_failure();
        let error = ThoError::CudaRecoveryFailed(Arc::new(CudaRecoveryFailure { original, recovery: None }));
        let wrapped = error.clone().context("recovery task").context("application");
        assert!(matches!(wrapped.what(), ThoError::CudaRecoveryFailed(_)));
        let job = ThoError::from(SyncStageRunnerError::JobError { job_id: 7, inner: Box::new(wrapped) }).context("runner");
        assert!(job.is_cuda_recovery_failed());
        assert!(!job.is_oom() && !job.is_retryable_oom() && !job.needs_cuda_ctx_reset());
        assert!(std::error::Error::source(&error).unwrap().to_string().contains("stage failed"));
    }
}
