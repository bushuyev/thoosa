use std::{future::Future, sync::Arc};

use crate::{CudaActivityGate, CudaActivityPauseTicket, CudaDeviceGeneration, CudaRecoveryState, ThoError};

tokio::task_local! {
    static CURRENT_ACTIVITY: CudaActivityScope;
}

/// A child retains every device its ancestors admitted. Each lease records the
/// actual gate and generation, so switching devices cannot transfer authority or
/// release an ancestor's activity while detached descendants still use it.
#[derive(Clone, Debug)]
pub struct CudaActivityScope {
    leases: Vec<Arc<ActivityLease>>,
}

#[derive(Debug)]
struct ActivityLease {
    gate: Arc<CudaActivityGate>,
    generation: CudaDeviceGeneration,
    reset: Option<CudaActivityPauseTicket>,
}

impl Drop for ActivityLease {
    fn drop(&mut self) {
        if self.reset.is_none() {
            let mut state = self.gate.state.lock();
            state.active -= 1;
            self.gate.notify.notify_waiters();
        }
    }
}

impl CudaActivityScope {
    pub fn current() -> Option<Self> {
        CURRENT_ACTIVITY.try_with(Clone::clone).ok()
    }

    pub fn recovery_ticket(gate: &Arc<CudaActivityGate>) -> Result<CudaActivityPauseTicket, ThoError> {
        let ticket = Self::current()
            .and_then(|scope| scope.leases.iter().find(|lease| Arc::ptr_eq(&lease.gate, gate)).and_then(|lease| lease.reset))
            .ok_or_else(|| ThoError::new("backend reload requires the coordinator's recovery scope"))?;
        gate.check_recovery_authority(ticket, true)?;
        Ok(ticket)
    }

    /// Register dispatch atomically with closing admission, before spawning work.
    pub fn try_enter(gate: &Arc<CudaActivityGate>) -> Result<Option<Self>, ThoError> {
        let mut state = gate.state.lock();
        match &state.recovery {
            CudaRecoveryState::Recovering { .. } => return Ok(None),
            CudaRecoveryState::Failed(failure) => return Err(ThoError::CudaRecoveryFailed(Arc::clone(failure))),
            CudaRecoveryState::Running => {}
        }
        state.active += 1;
        Ok(Some(Self {
            leases: vec![Arc::new(ActivityLease {
                gate: Arc::clone(gate),
                generation: state.generation,
                reset: None,
            })],
        }))
    }

    pub fn with_inherited(mut self, inherited: Option<Self>) -> Self {
        if let Some(inherited) = inherited {
            for lease in inherited.leases {
                if !self.leases.iter().any(|own| Arc::ptr_eq(&own.gate, &lease.gate)) {
                    self.leases.push(lease);
                }
            }
        }
        self
    }

    pub fn generation(&self, gate: &Arc<CudaActivityGate>) -> Option<CudaDeviceGeneration> {
        self.leases.iter().find(|lease| Arc::ptr_eq(&lease.gate, gate)).map(|lease| lease.generation)
    }

    /// Admitted operations may drain during recovery, but never run against a
    /// replacement generation or borrow another gate's cleanup authority.
    pub fn check_operation(&self, gate: &Arc<CudaActivityGate>) -> Result<(), ThoError> {
        let lease = self.leases.iter().find(|lease| Arc::ptr_eq(&lease.gate, gate)).ok_or_else(|| ThoError::new("CUDA operation has no activity lease for its device"))?;

        let state = gate.state.lock();
        if let CudaRecoveryState::Failed(failure) = &state.recovery {
            return Err(ThoError::CudaRecoveryFailed(Arc::clone(failure)));
        }
        if state.generation != lease.generation {
            return Err(ThoError::CudaBeingRecovered);
        }
        if let Some(ticket) = lease.reset {
            if !matches!(state.recovery, CudaRecoveryState::Recovering { .. }) || state.cycle != ticket.cycle {
                return Err(ThoError::CudaBeingRecovered);
            }
        }
        Ok(())
    }

    /// Queued jobs are dependencies, not yet executing operations. Interrupt them
    /// when any admitted ancestor must unwind, including on a different device.
    pub fn check_dependencies(&self) -> Result<(), ThoError> {
        for lease in &self.leases {
            if lease.reset.is_none() {
                lease.gate.check_generation(lease.generation)?;
            }
        }
        Ok(())
    }

    pub async fn wait_for_interruption(&self) -> ThoError {
        let waits: Vec<_> = self.leases.iter().filter(|lease| lease.reset.is_none()).map(|lease| Box::pin(lease.gate.wait_for_generation_end(lease.generation))).collect();
        if waits.is_empty() {
            return std::future::pending().await;
        }
        futures::future::select_all(waits).await.0
    }

    /// Copy callers can hold locks needed by recovery: reject a new root instead
    /// of waiting, but preserve all existing leases when adding another device.
    pub fn for_operation(gate: &Arc<CudaActivityGate>) -> Result<Self, ThoError> {
        let inherited = Self::current();
        if let Some(scope) = &inherited
            && scope.generation(gate).is_some()
        {
            scope.check_operation(gate)?;
            return Ok(scope.clone());
        }
        Self::try_enter(gate)?
            .map(|scope| scope.with_inherited(inherited))
            .ok_or_else(|| ThoError::new_with_cause("CUDA operation interrupted by Cuda recovery", ThoError::CudaBeingRecovered))
    }

    pub async fn run<F: Future>(self, future: F) -> F::Output {
        CURRENT_ACTIVITY.scope(self, future).await
    }

    /// Cleanup is authorized only for this idle owned cycle, never another gate.
    pub async fn run_recovery<T, F: Future<Output = Result<T, ThoError>>>(gate: &Arc<CudaActivityGate>, ticket: CudaActivityPauseTicket, future: F) -> Result<T, ThoError> {
        let generation = {
            let state = gate.state.lock();
            if let CudaRecoveryState::Failed(failure) = &state.recovery {
                return Err(ThoError::CudaRecoveryFailed(Arc::clone(failure)));
            }
            if ticket.gate_id != gate.id || !ticket.owner || !matches!(state.recovery, CudaRecoveryState::Recovering { .. }) || ticket.cycle != state.cycle || state.active != 0 {
                return Err(ThoError::new("reset scope requires an idle pause owned by this gate"));
            }
            state.generation
        };
        Self {
            leases: vec![Arc::new(ActivityLease {
                gate: Arc::clone(gate),
                generation,
                reset: Some(ticket),
            })],
        }
        .run(future)
        .await
    }
}

#[cfg(test)]
mod tests {
    use crate::{CudaAction, CudaActivityGate, CudaDeviceGeneration, CudaRecoveryFailure, CudaRecoveryState, SyncStageRunnerError, ThoError, tho_error};
    use std::sync::{Arc, Barrier};
    use tokio::sync::mpsc;

    #[test]
    fn test_reserved_enqueue_is_rejected_during_pause_and_retried_after() {
        let gate = CudaActivityGate::new(1);
        let (tx, mut rx) = mpsc::channel(1);
        let permit = tx.try_reserve().unwrap();
        let job = Box::new(42);
        let address = job.as_ref() as *const i32;
        let pause = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        assert!(rx.try_recv().is_err()); // Reset has already drained this queue.

        let (rejected, error) = gate.try_enqueue(permit, job).unwrap_err();
        assert!(matches!(error.what(), ThoError::CudaBeingRecovered));
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
                let pause = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
                let purged = rx.try_recv().ok();
                match enqueue.join().unwrap() {
                    Ok(()) => assert_eq!(purged, Some(42)),
                    Err((job, error)) => {
                        assert_eq!(job, 42);
                        assert!(matches!(error.what(), ThoError::CudaBeingRecovered));
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
        let first = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 1, "jobs accepted before pause remain visible to purge");
        gate.finish_pause(first);
        let second = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        gate.finish_pause(first);
        let (job, error) = gate.try_enqueue(tx.try_reserve().unwrap(), 2).unwrap_err();
        assert_eq!(job, 2);
        assert!(matches!(error.what(), ThoError::CudaBeingRecovered));
        gate.finish_pause(second);
        gate.try_enqueue(tx.try_reserve().unwrap(), 3).unwrap();
        assert_eq!(rx.try_recv().unwrap(), 3);
    }

    fn original_failure() -> ThoError {
        tho_error::cuda_error(CudaAction::NeedsRestart, "illegal address".into()).context("stage failed")
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
            assert!(!error.is_oom() && !error.is_retryable_oom() && !error.needs_cuda_recovery());
            let ThoError::CudaRecoveryFailed(failure) = error.what() else { panic!("terminal error was lost") };
            assert!(failure.original.needs_cuda_recovery());
            assert!(failure.recovery.as_ref().unwrap().is_oom());
            assert!(error.to_string().contains("stage failed") && error.to_string().contains("replacement failed"));
        }

        gate.finish_pause(ticket);
        assert!(matches!(gate.recovery_state(), CudaRecoveryState::Failed(_)));
        assert!(gate.is_paused() && !gate.is_recovering());
        assert_eq!(gate.device_generation(), CudaDeviceGeneration(0));
        assert!(gate.request_pause(ThoError::CudaBeingRecovered).unwrap_err().is_cuda_recovery_failed());
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
        assert!(CudaActivityScope::run_recovery::<(), _>(&first_gate, other, async { panic!("foreign reset ran") }).await.is_err());
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
        activity
            .run(async {
                // A first-device lease neither drains nor authorizes the second GPU.
                second.wait_until_idle().await;
                assert!(CudaActivityScope::for_operation(&second).is_err());
                let _child = CudaActivityScope::for_operation(&first).unwrap();
            })
            .await;
        let first_pause = first.request_pause(original_failure()).unwrap();
        CudaActivityScope::run_recovery(&first, first_pause, async {
            let _cleanup = CudaActivityScope::for_operation(&first)?;
            assert!(CudaActivityScope::for_operation(&second).is_err());
            Ok(())
        })
        .await
        .unwrap();
        first.finish_pause(first_pause);
        second.fail_pause(pause, None);
        first.check_admission().unwrap();
        assert!(second.check_admission().unwrap_err().is_cuda_recovery_failed());
    }

    #[tokio::test]
    async fn test_cross_device_descendants_retain_all_ancestor_leases() {
        use super::CudaActivityScope;
        let first = Arc::new(CudaActivityGate::new(1));
        let second = Arc::new(CudaActivityGate::new(2));
        let scope = CudaActivityScope::for_operation(&first).unwrap().run(async { CudaActivityScope::for_operation(&second).unwrap() }).await;
        assert_eq!(scope.generation(&first), Some(CudaDeviceGeneration(0)));
        assert_eq!(scope.generation(&second), Some(CudaDeviceGeneration(0)));
        let first_pause = first.request_pause(original_failure()).unwrap();
        let second_pause = second.request_pause(original_failure()).unwrap();
        assert!(futures::poll!(Box::pin(first.wait_until_idle()).as_mut()).is_pending());
        assert!(futures::poll!(Box::pin(second.wait_until_idle()).as_mut()).is_pending());
        // Already-admitted descendants can finish on either ancestor's device.
        scope
            .run(async {
                CudaActivityScope::for_operation(&first).unwrap();
                CudaActivityScope::for_operation(&second).unwrap();
            })
            .await;
        assert!(futures::poll!(Box::pin(first.wait_until_idle()).as_mut()).is_ready());
        assert!(futures::poll!(Box::pin(second.wait_until_idle()).as_mut()).is_ready());
        first.finish_pause(first_pause);
        second.finish_pause(second_pause);
    }

    #[tokio::test]
    async fn test_cleanup_scope_cannot_authorize_a_later_generation() {
        use super::CudaActivityScope;
        let gate = Arc::new(CudaActivityGate::new(1));
        let ticket = gate.request_pause(original_failure()).unwrap();
        let escaped = CudaActivityScope::run_recovery(&gate, ticket, async { Ok(CudaActivityScope::current().unwrap()) }).await.unwrap();
        gate.finish_pause(ticket);
        let next = gate.request_pause(original_failure()).unwrap();
        escaped
            .run(async {
                assert!(matches!(CudaActivityScope::for_operation(&gate).unwrap_err().what(), ThoError::CudaBeingRecovered));
            })
            .await;
        gate.finish_pause(next);
    }

    #[test]
    fn test_terminal_recovery_error_survives_context_and_job_wrapping() {
        let original = original_failure();
        let error = ThoError::CudaRecoveryFailed(Arc::new(CudaRecoveryFailure { original, recovery: None }));
        let wrapped = error.clone().context("recovery task").context("application");
        assert!(matches!(wrapped.what(), ThoError::CudaRecoveryFailed(_)));
        let job = ThoError::from(SyncStageRunnerError::JobError { job_id: 7, inner: Box::new(wrapped) }).context("runner");
        assert!(job.is_cuda_recovery_failed());
        assert!(!job.is_oom() && !job.is_retryable_oom() && !job.needs_cuda_recovery());
        assert!(std::error::Error::source(&error).unwrap().to_string().contains("stage failed"));
    }
}
