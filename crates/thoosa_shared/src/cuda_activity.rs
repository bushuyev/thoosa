use std::{future::Future, sync::Arc};

use crate::{CudaActivityGate, CudaActivityPauseTicket, CudaRecoveryState, ThoError};

tokio::task_local! {
    static CURRENT_ACTIVITY: CudaActivityScope;
}

/// Admitted work and its children may drain while admission is paused. Clones
/// share one registration, so reset also waits for detached children to finish.
#[derive(Clone, Debug)]
pub struct CudaActivityScope {
    gate: Arc<CudaActivityGate>,
    _lease: Option<Arc<ActivityLease>>,
}

#[derive(Debug)]
struct ActivityLease {
    gate: Arc<CudaActivityGate>,
}

impl Drop for ActivityLease {
    fn drop(&mut self) {
        let gate = &self.gate;
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
    pub fn try_enter(gate: &Arc<CudaActivityGate>) -> Result<Option<Self>, ThoError> {
        let mut state = gate.state.lock();
        match &state.recovery {
            CudaRecoveryState::Recovering { .. } => return Ok(None),
            CudaRecoveryState::Failed(failure) => return Err(ThoError::CudaRecoveryFailed(Arc::clone(failure))),
            CudaRecoveryState::Running => {}
        }
        state.active += 1;
        Ok(Some(Self { gate: Arc::clone(gate), _lease: Some(Arc::new(ActivityLease { gate: Arc::clone(gate) })) }))
    }

    /// Copy callers can already hold tensor/cache locks needed by reset. Reject
    /// an unadmitted operation during pause rather than wait while holding them.
    pub fn for_operation(gate: &Arc<CudaActivityGate>) -> Result<Self, ThoError> {
        // Existing scopes may drain during recovery, but terminal failure must
        // never authorize another CUDA operation through an inherited scope.
        if let Some(error) = gate.failure() {
            return Err(error);
        }
        // A scope from another device grants no admission on this gate.
        if let Some(scope) = Self::current().filter(|scope| Arc::ptr_eq(&scope.gate, gate)) {
            return Ok(scope);
        }
        Self::try_enter(gate)?.ok_or_else(|| {
            ThoError::new_with_cause("CUDA operation interrupted by context reset", ThoError::ContextReset)
        })
    }

    pub async fn run<F: Future>(self, future: F) -> F::Output {
        CURRENT_ACTIVITY.scope(self, future).await
    }

    /// Reset cleanup runs only after admitted activity drained. It must not wait
    /// for the admission gate that this same reset owns.
    pub async fn run_reset<T, F: Future<Output = Result<T, ThoError>>>(gate: &Arc<CudaActivityGate>, ticket: CudaActivityPauseTicket, future: F) -> Result<T, ThoError> {
        {
            let state = gate.state.lock();
            if let CudaRecoveryState::Failed(failure) = &state.recovery {
                return Err(ThoError::CudaRecoveryFailed(Arc::clone(failure)));
            }
            if ticket.gate_id != gate.id || !ticket.owner || !matches!(state.recovery, CudaRecoveryState::Recovering { .. }) || ticket.cycle != state.cycle || state.active != 0 {
                return Err(ThoError::new("reset scope requires an idle pause owned by this gate"));
            }
        }
        Self { gate: Arc::clone(gate), _lease: None }.run(future).await
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use tokio::sync::mpsc;
    use crate::{CudaAction, CudaActivityGate, CudaDeviceGeneration, CudaRecoveryFailure, CudaRecoveryState, SyncStageRunnerError, ThoError};

    #[test]
    fn test_reserved_enqueue_is_rejected_during_pause_and_retried_after() {
        let gate = CudaActivityGate::new(1);
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

        gate.finish_pause(pause);
        gate.try_enqueue(tx.try_reserve().unwrap(), rejected).unwrap();
        assert_eq!(*rx.try_recv().unwrap(), 42);
    }

    #[test]
    fn test_enqueue_racing_pause_is_visible_to_purge_or_rejected() {
        // Synchronize the contenders rather than depending on sleeps. After
        // pause returns, a successful publication must already be in the queue.
        for _ in 0..64 {
            let gate = CudaActivityGate::new(1);
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
                gate.finish_pause(pause);
            });
        }
    }

    #[test]
    fn test_old_reset_completion_cannot_reopen_new_pause() {
        let gate = CudaActivityGate::new(1);
        let (tx, mut rx) = mpsc::channel(1);
        gate.try_enqueue(tx.try_reserve().unwrap(), 1).unwrap();
        let first = gate.request_pause(ThoError::ContextReset).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 1, "jobs accepted before pause remain visible to purge");
        gate.finish_pause(first);
        let second = gate.request_pause(ThoError::ContextReset).unwrap();
        gate.finish_pause(first);
        let (job, error) = gate.try_enqueue(tx.try_reserve().unwrap(), 2).unwrap_err();
        assert_eq!(job, 2);
        assert!(matches!(error.what(), ThoError::ContextReset));
        gate.finish_pause(second);
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
        let gate = CudaActivityGate::new(1);
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

        gate.finish_pause(ticket);
        assert!(matches!(gate.recovery_state(), CudaRecoveryState::Failed(_)));
        assert!(gate.is_paused() && !gate.is_recovering());
        assert_eq!(gate.device_generation(), CudaDeviceGeneration(0));
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
    async fn test_recovery_tickets_and_generations_are_gate_local() {
        use super::CudaActivityScope;
        // Even two independent contexts for the same ordinal cannot exchange tickets.
        let first_gate = Arc::new(CudaActivityGate::new(1));
        let other_gate = Arc::new(CudaActivityGate::new(1));
        let first = first_gate.request_pause(original_failure()).unwrap();
        let other = other_gate.request_pause(original_failure()).unwrap();
        assert_eq!(first.cycle(), other.cycle());
        assert_ne!(first, other);
        let joined = first_gate.request_pause(ThoError::oom()).unwrap();
        assert!(!joined.is_owner());
        first_gate.finish_pause(joined);
        first_gate.fail_pause(joined, None);
        first_gate.finish_pause(other);
        first_gate.fail_pause(other, None);
        assert!(first_gate.is_recovering());
        assert!(first_gate.wait_for_pause_to_finish(other).await.is_err());
        assert!(CudaActivityScope::run_reset::<(), _>(&first_gate, other, async { panic!("foreign reset ran") }).await.is_err());
        assert_eq!(first_gate.device_generation(), CudaDeviceGeneration(0));

        first_gate.finish_pause(first);
        first_gate.wait_for_pause_to_finish(joined).await.unwrap();
        assert_eq!(first_gate.device_generation(), CudaDeviceGeneration(1));
        assert_eq!(other_gate.device_generation(), CudaDeviceGeneration(0));
        assert!(other_gate.is_recovering());

        let second = first_gate.request_pause(original_failure()).unwrap();
        first_gate.finish_pause(first);
        first_gate.fail_pause(first, Some(ThoError::new("stale failure")));
        assert!(first_gate.is_recovering());
        first_gate.wait_for_pause_to_finish(first).await.unwrap();
        first_gate.finish_pause(second);
        assert_eq!(first_gate.device_generation(), CudaDeviceGeneration(2));
        other_gate.finish_pause(other);
    }

    #[tokio::test]
    async fn test_activity_registration_and_reset_authority_are_gate_local() {
        use super::CudaActivityScope;
        let first = Arc::new(CudaActivityGate::new(1));
        let second = Arc::new(CudaActivityGate::new(2));
        let activity = CudaActivityScope::for_operation(&first).unwrap();
        let pause = second.request_pause(original_failure()).unwrap();
        activity.run(async {
            // A first-device lease neither drains nor authorizes the second GPU.
            second.wait_until_idle().await;
            assert!(CudaActivityScope::for_operation(&second).is_err());
            let _child = CudaActivityScope::for_operation(&first).unwrap();
        }).await;
        let first_pause = first.request_pause(original_failure()).unwrap();
        CudaActivityScope::run_reset(&first, first_pause, async {
            let _cleanup = CudaActivityScope::for_operation(&first)?;
            assert!(CudaActivityScope::for_operation(&second).is_err());
            Ok(())
        }).await.unwrap();
        first.finish_pause(first_pause);
        second.fail_pause(pause, None);
        first.check_admission().unwrap();
        assert!(second.check_admission().unwrap_err().is_cuda_recovery_failed());
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
