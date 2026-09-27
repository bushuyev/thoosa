use std::sync::Arc;

use futures_util::future::BoxFuture;
use parking_lot::Mutex;
use thoosa_shared::{CudaActivityGate, CudaActivityPauseTicket, CudaActivityScope, CudaDeviceGeneration, ThoError};
use tokio::sync::oneshot;
use tokio::task::AbortHandle;

pub(crate) type ResetFn = Arc<dyn Fn(usize) -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>;

struct RecoveryCycle {
    ticket: CudaActivityPauseTicket,
    task: Option<AbortHandle>,
}

/// One coordinator owns one device's gate and executor drain. This mutex always
/// precedes the gate mutex; neither is held while draining or rebuilding.
pub(crate) struct CudaRecoveryCoordinator {
    gate: Arc<CudaActivityGate>,
    cycle: Mutex<Option<RecoveryCycle>>,
    reset: ResetFn,
}

impl std::fmt::Debug for CudaRecoveryCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRecoveryCoordinator").field("gate", &self.gate).finish_non_exhaustive()
    }
}

impl CudaRecoveryCoordinator {
    pub(crate) fn new(gate: Arc<CudaActivityGate>, reset: ResetFn) -> Arc<Self> {
        Arc::new(Self { gate, cycle: Mutex::new(None), reset })
    }

    pub(crate) fn gate(&self) -> &Arc<CudaActivityGate> {
        &self.gate
    }

    /// Closing admission and arranging this device's executor drain are one
    /// synchronous operation. Publication uses the same mutex, so late reports
    /// either join this cycle or are recognized as belonging to a retired worker.
    pub(crate) fn request(
        self: &Arc<Self>,
        generation: CudaDeviceGeneration,
        cause: ThoError,
        drain: impl FnOnce() -> Result<oneshot::Receiver<()>, ThoError>,
    ) -> Result<Option<CudaActivityPauseTicket>, ThoError> {
        let mut cycle = self.cycle.lock();
        let gate = &self.gate;
        if let Some(error) = gate.failure() {
            return Err(error);
        }
        let current = gate.device_generation();
        if generation.0 < current.0 {
            return Ok(None);
        }
        if generation != current {
            return Err(ThoError::new("CUDA recovery report refers to a future device generation"));
        }
        if let Some(recovery) = cycle.as_ref() {
            return Ok(Some(recovery.ticket));
        }

        let ticket = gate.request_pause(cause.clone())?;
        if !ticket.is_owner() {
            return Err(ThoError::new("CUDA recovery is owned by another coordinator"));
        }
        *cycle = Some(RecoveryCycle { ticket, task: None });
        let drained = match drain() {
            Ok(drained) => drained,
            Err(error) => {
                gate.fail_pause(ticket, Some(error));
                cycle.take();
                return Err(gate.failure().expect("failed recovery"));
            }
        };

        // Spawn outside the lock: a shutting-down runtime may immediately drop
        // the future, whose guard also acquires this mutex to publish failure.
        drop(cycle);
        let runtime = match tokio::runtime::Handle::try_current() {
            Ok(runtime) => runtime,
            Err(error) => {
                self.fail(ticket, ThoError::new(error.to_string()));
                return Err(gate.failure().expect("failed recovery"));
            }
        };
        // Construct before spawning so cancellation before the first poll also
        // fails admission. The reporter does not own this task's lifetime.
        let guard = RecoveryGuard { coordinator: Arc::clone(self), ticket };
        let task = runtime.spawn(async move {
            let _guard = guard;
            if let Err(error) = _guard.coordinator.run(ticket, drained, generation, cause).await {
                _guard.coordinator.fail(ticket, error);
            }
        });
        let mut cycle = self.cycle.lock();
        if let Some(recovery) = cycle.as_mut().filter(|recovery| recovery.ticket == ticket) {
            recovery.task = Some(task.abort_handle());
        }
        Ok(Some(ticket))
    }

    async fn run(&self, ticket: CudaActivityPauseTicket, drained: oneshot::Receiver<()>, generation: CudaDeviceGeneration, cause: ThoError) -> Result<(), ThoError> {
        drained.await.map_err(|_| ThoError::new("CUDA recovery reporting executor stopped before draining"))?;
        self.gate.wait_until_idle().await;
        let device = self.gate.device_idx();
        CudaActivityScope::run_reset(&self.gate, ticket, async {
            (self.reset)(device).await
        }).await.map_err(|error| error.context(format!("recovering CUDA device {device}, generation {}, after {cause:#}", generation.0)))?;

        // Generation publication and report deduplication share this lock. No
        // other device participates in this completion or delays its reopening.
        let mut cycle = self.cycle.lock();
        if let Some(error) = self.gate.failure() {
            return Err(error);
        }
        if cycle.as_ref().is_some_and(|cycle| cycle.ticket == ticket) {
            self.gate.finish_pause(ticket);
            cycle.take();
        }
        Ok(())
    }

    fn fail(&self, ticket: CudaActivityPauseTicket, error: ThoError) {
        let mut cycle = self.cycle.lock();
        if cycle.as_ref().is_some_and(|cycle| cycle.ticket == ticket) {
            self.gate.fail_pause(ticket, Some(error));
            cycle.take();
        }
    }
}

struct RecoveryGuard {
    coordinator: Arc<CudaRecoveryCoordinator>,
    ticket: CudaActivityPauseTicket,
}

impl Drop for RecoveryGuard {
    fn drop(&mut self) {
        // Success already removed this cycle. Panic, abort, and runtime shutdown
        // leave it owned, and must wake this device's waiters with a terminal error.
        self.coordinator.fail(self.ticket, ThoError::new("CUDA recovery coordinator was interrupted"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use thoosa_shared::CudaAction;
    use tokio::sync::Notify;

    fn fatal() -> ThoError {
        ThoError::Cuda { action: CudaAction::NeedsRestart, msg: "original CUDA failure".into() }
    }

    fn drained() -> Result<oneshot::Receiver<()>, ThoError> {
        let (tx, rx) = oneshot::channel();
        tx.send(()).unwrap();
        Ok(rx)
    }

    #[tokio::test]
    async fn test_devices_recover_independently_and_deduplicate_own_reports() {
        tokio::time::timeout(Duration::from_secs(2), async {
            let (started, mut starts) = tokio::sync::mpsc::unbounded_channel();
            let releases = [Arc::new(Notify::new()), Arc::new(Notify::new())];
            let reset: ResetFn = Arc::new({
                let releases = releases.clone();
                move |device| {
                    let started = started.clone();
                    let release = Arc::clone(&releases[device - 1]);
                    Box::pin(async move {
                        started.send(device).unwrap();
                        release.notified().await;
                        Ok(())
                    })
                }
            });
            let first = CudaRecoveryCoordinator::new(Arc::new(CudaActivityGate::new(1)), Arc::clone(&reset));
            let second = CudaRecoveryCoordinator::new(Arc::new(CudaActivityGate::new(2)), reset);
            let generation = CudaDeviceGeneration(0);
            let first_ticket = first.request(generation, fatal(), drained).unwrap().unwrap();
            assert_eq!(starts.recv().await, Some(1));
            second.gate().check_admission().unwrap();
            let second_ticket = second.request(generation, fatal(), drained).unwrap().unwrap();
            assert_eq!(starts.recv().await, Some(2), "second recovery must not wait for the first backend");
            assert_eq!(first_ticket.cycle(), second_ticket.cycle());
            assert_ne!(first_ticket, second_ticket);
            assert_eq!(first.request(generation, fatal(), || panic!("duplicate drain")).unwrap(), Some(first_ticket));
            releases[0].notify_one();
            first.gate().wait_for_pause_to_finish(first_ticket).await.unwrap();
            assert_eq!(first.gate().device_generation(), CudaDeviceGeneration(1));
            assert_eq!(second.gate().device_generation(), generation);
            assert!(second.gate().is_recovering());
            assert!(first.request(generation, fatal(), || panic!("retired report drained")).unwrap().is_none());
            releases[1].notify_one();
            second.gate().wait_for_pause_to_finish(second_ticket).await.unwrap();
            assert_eq!(second.gate().device_generation(), CudaDeviceGeneration(1));
            assert!(starts.try_recv().is_err());
            assert!(first.request(CudaDeviceGeneration(2), fatal(), || panic!("future report drained")).is_err());
            first.gate().check_admission().unwrap();
        }).await.expect("devices must complete recovery independently");
    }

    #[tokio::test]
    async fn test_reporting_task_cancellation_does_not_cancel_recovery() {
        let coordinator = CudaRecoveryCoordinator::new(Arc::new(CudaActivityGate::new(1)), Arc::new(|_| Box::pin(async { Ok(()) })));
        let (drain_tx, drain_rx) = oneshot::channel();
        let (reported_tx, reported_rx) = oneshot::channel();
        let reporter = tokio::spawn({
            let coordinator = Arc::clone(&coordinator);
            async move {
                let ticket = coordinator.request(coordinator.gate().device_generation(), fatal(), || Ok(drain_rx)).unwrap().unwrap();
                reported_tx.send(ticket).unwrap();
                std::future::pending::<()>().await;
            }
        });
        let ticket = reported_rx.await.unwrap();
        reporter.abort();
        assert!(reporter.await.unwrap_err().is_cancelled());
        assert!(coordinator.gate().is_recovering());
        drain_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), coordinator.gate().wait_for_pause_to_finish(ticket)).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_coordinator_interruption_fails_admission_and_wakes_waiters() {
        // Private device gates allow terminal failure cases in one process.
        for mode in ["abort_before_poll", "abort_backend", "panic", "backend_error", "drain_closed", "drain_send_error"] {
            tokio::time::timeout(Duration::from_secs(2), async {
                let started = Arc::new(Notify::new());
                let panics = mode == "panic";
                let coordinator = CudaRecoveryCoordinator::new(Arc::new(CudaActivityGate::new(1)), Arc::new({
                    let started = Arc::clone(&started);
                    move |_| {
                        let started = Arc::clone(&started);
                        Box::pin(async move {
                            if panics {
                                panic!("injected coordinator backend panic");
                            }
                            if mode == "backend_error" {
                                return Err(ThoError::new("injected backend failure"));
                            }
                            started.notify_one();
                            std::future::pending::<Result<(), ThoError>>().await
                        })
                    }
                }));
                let gate = coordinator.gate();
                let healthy = Arc::new(CudaActivityGate::new(2));
                let generation = gate.device_generation();
                let request = coordinator.request(generation, fatal(), || {
                    if mode == "drain_send_error" {
                        return Err(ThoError::new("reporting executor is closed"));
                    }
                    let (tx, rx) = oneshot::channel();
                    if mode != "drain_closed" {
                        tx.send(()).unwrap();
                    }
                    Ok(rx)
                });
                if mode == "drain_send_error" {
                    assert!(request.unwrap_err().is_cuda_recovery_failed());
                } else {
                    let ticket = request.unwrap().unwrap();
                    let mut waiter = Box::pin(gate.wait_for_pause_to_finish(ticket));
                    assert!(futures::poll!(waiter.as_mut()).is_pending());
                    if mode == "abort_backend" {
                        started.notified().await;
                    }
                    if mode.starts_with("abort") {
                        coordinator.cycle.lock().as_ref().unwrap().task.as_ref().unwrap().abort();
                    }
                    assert!(waiter.await.unwrap_err().is_cuda_recovery_failed());
                }
                let failure = gate.wait_if_paused().await.unwrap_err();
                let ThoError::CudaRecoveryFailed(failure) = failure.what() else { panic!("terminal error lost") };
                assert!(failure.original.needs_cuda_ctx_reset());
                assert!(failure.recovery.is_some());
                assert_eq!(gate.device_generation(), generation);
                assert!(gate.check_admission().unwrap_err().is_cuda_recovery_failed());
                healthy.check_admission().unwrap();
                let _activity = CudaActivityScope::for_operation(&healthy).unwrap();
                assert_eq!(healthy.device_generation(), CudaDeviceGeneration(0));
            }).await.expect("coordinator failure must wake waiters");
        }
    }
}
