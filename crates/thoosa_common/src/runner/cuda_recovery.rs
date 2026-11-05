use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

use futures_util::future::BoxFuture;
use log::{debug, error, info, warn};
use parking_lot::Mutex;
use thoosa_shared::{CudaActivityGate, CudaActivityPauseTicket, CudaActivityScope, CudaDeviceGeneration, ThoError};
use tokio::sync::oneshot;
use tokio::task::AbortHandle;

pub(crate) type RestartFn = Arc<dyn Fn(usize) -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>;
pub(crate) type RetireFn = Arc<dyn Fn(usize, CudaActivityPauseTicket) -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>;
pub(crate) type ProcessFailureFn = Arc<dyn Fn(ThoError) + Send + Sync>;
const RECOVERY_DRAIN_TIMEOUT: Duration = Duration::from_secs(30);
const RECOVERY_BACKEND_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug)]
struct RecoveryCycle {
    ticket: CudaActivityPauseTicket,
    task: Option<AbortHandle>,
}

/// One coordinator owns one device's gate and executor drain. This mutex always
/// precedes the gate mutex; neither is held while draining or rebuilding.
pub(crate) struct CudaRecoveryCoordinator {
    gate: Arc<CudaActivityGate>,
    cycle: Mutex<Option<RecoveryCycle>>,
    restart_fn: RestartFn,
    retire: Option<RetireFn>,
    process_failure: Option<ProcessFailureFn>,//to tear down main process in local mode.
    drain_timeout: Duration,
}

impl std::fmt::Debug for CudaRecoveryCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRecoveryCoordinator").field("gate", &self.gate).finish_non_exhaustive()
    }
}

impl CudaRecoveryCoordinator {
    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub(crate) fn new(gate: Arc<CudaActivityGate>, reset: RestartFn) -> Arc<Self> {
        Self::with_backend(gate, reset, None, None)
    }

    pub(crate) fn with_backend(gate: Arc<CudaActivityGate>, restart_fn: RestartFn, retire: Option<RetireFn>, process_failure: Option<ProcessFailureFn>) -> Arc<Self> {
        Arc::new(Self { gate, cycle: Mutex::new(None), restart_fn, retire, process_failure, drain_timeout: RECOVERY_DRAIN_TIMEOUT })
    }

    pub(crate) fn gate(&self) -> &Arc<CudaActivityGate> {
        &self.gate
    }

    /// Closing admission and arranging this device's SyncStageRunner drain are one
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
        debug!("CudaRecoveryCoordinator::request: needs_cuda_ctx_reset={:?} is_cuda_recovery_failed={:?} process_failure={:?} cycle={:?}", cause.needs_cuda_recovery(), cause.is_cuda_recovery_failed(), self.process_failure.is_some(), cycle.deref());

        if (cause.needs_cuda_recovery() || cause.is_cuda_recovery_failed()) && let Some(process_failure) = &self.process_failure {
            // Local fatal errors or an unsuccessful stream drain invalidate the
            // process; neither can be repaired by resetting live LibTorch state.
            // Publish synchronously,
            // notify the application, and never attempt an in-process repair.
            // A maintenance cycle may already own the pause. Failure must also
            // supersede that cycle, without borrowing a non-owner pause ticket.
            gate.fail_process(cause);
            let failure = gate.failure().expect("failed local CUDA process");
            let task = cycle.take().and_then(|cycle| cycle.task);
            let _ = drain();
            drop(cycle);
            if let Some(task) = task { task.abort(); }
            process_failure(failure.clone());
            return Err(failure);
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
                error!("CudaRecoveryCoordinator::request: drain failed: e={error:?}");
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
            if let Err(error) = _guard.coordinator.run(ticket, drained, generation, cause).await {//<-- calls thoosa_common::runner::global_ctx::GlobalCtx::cuda_restart
                error!("CudaRecoveryCoordinator::request: run failed, error={error:?}");
                _guard.coordinator.fail(ticket, error);
            }
        });
        let mut cycle = self.cycle.lock();
        if let Some(recovery) = cycle.as_mut().filter(|recovery| recovery.ticket == ticket) {
            recovery.task = Some(task.abort_handle());
        } else {
            // A local fatal report may have superseded this cycle while spawning.
            task.abort();
        }
        Ok(Some(ticket))
    }

    async fn run(&self, ticket: CudaActivityPauseTicket, drained: oneshot::Receiver<()>, generation: CudaDeviceGeneration, cause: ThoError) -> Result<(), ThoError> {
        debug!("CudaRecoveryCoordinator::run: ticket={ticket:?} cause={cause:?}");
        let device = self.gate.device_idx();
        let drain = async {
            drained.await.map_err(|_| ThoError::new("CUDA recovery reporting executor stopped before draining"))?;
            info!("CudaRecoveryCoordinator::run: drained ready");
            self.gate.wait_until_idle().await;
            Ok::<_, ThoError>(())
        };
        tokio::pin!(drain);
        debug!("CudaRecoveryCoordinator::run: drain done");

        match tokio::time::timeout(self.drain_timeout, &mut drain).await {
            Ok(result) => result?,
            Err(e) => {
                warn!("CudaRecoveryCoordinator::run: drain_timeout e={e:?}");
                // A wedged IPC operation holds its lease and shared buffers until
                // the worker is dead. Retirement uses a separate control channel.
                // Only a drain timeout retires early; after a normal drain the
                // Restart control itself must retire the old worker.
                let retire = self.retire.as_ref().ok_or_else(|| ThoError::new("CUDA activity did not drain before the recovery deadline"))?;
                tokio::time::timeout(RECOVERY_BACKEND_TIMEOUT, retire(device, ticket)).await.map_err(|_| ThoError::new("CUDA worker retirement timed out"))??;
                tokio::time::timeout(self.drain_timeout, &mut drain).await.map_err(|_| ThoError::new("CUDA activity did not drain after worker retirement"))??;
            }
        }
        info!("CudaRecoveryCoordinator::run: calling restart_fn");

        tokio::time::timeout(RECOVERY_BACKEND_TIMEOUT, CudaActivityScope::run_recovery(&self.gate, ticket, async {
            (self.restart_fn)(device).await
        })).await
        .map_err(|e| {
            ThoError::new(format!("CUDA backend recovery timed out: ({e:?})"))
        })?
        .map_err(|error| {
            error.context(format!("recovering CUDA device {device}, generation {}, after {cause:#}", generation.0))
        })?;

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
            let failure = self.gate.failure().expect("failed CUDA recovery");
            drop(cycle);
            // A failed local maintenance operation cannot be repaired by a
            // successor worker either; terminate the application in this case.
            if let Some(process_failure) = &self.process_failure { process_failure(failure); }
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
        // Use the constructor so injected CUDA errors capture their backtrace.
        thoosa_shared::tho_error::cuda_error(CudaAction::NeedsRestart, "original CUDA failure".into())
    }

    fn drained() -> Result<oneshot::Receiver<()>, ThoError> {
        let (tx, rx) = oneshot::channel();
        tx.send(()).unwrap();
        Ok(rx)
    }

    #[tokio::test]
    async fn test_drain_timeout_retires_backend_before_reopening() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let gate = Arc::new(CudaActivityGate::new(1));
        let activity = Arc::new(Mutex::new(CudaActivityScope::try_enter(&gate).unwrap()));
        let retired = Arc::new(AtomicBool::new(false));
        let (done, drained_rx) = oneshot::channel();
        let done = Arc::new(Mutex::new(Some(done)));
        let retire: RetireFn = Arc::new({
            let gate = Arc::clone(&gate);
            let activity = Arc::clone(&activity);
            let retired = Arc::clone(&retired);
            move |device, authority| {
                assert_eq!(device, 1);
                gate.check_recovery_authority(authority, false).unwrap();
                assert!(gate.check_recovery_authority(authority, true).is_err());
                // This models the ordering promised by the real backend: exit
                // first, then request completion and physical activity release.
                retired.store(true, Ordering::Release);
                activity.lock().take();
                done.lock().take().unwrap().send(()).unwrap();
                Box::pin(async { Ok(()) })
            }
        });
        let reset: RestartFn = Arc::new({
            let retired = Arc::clone(&retired);
            move |_| { assert!(retired.load(Ordering::Acquire)); Box::pin(async { Ok(()) }) }
        });
        let mut coordinator = CudaRecoveryCoordinator::with_backend(Arc::clone(&gate), reset, Some(retire), None);
        Arc::get_mut(&mut coordinator).unwrap().drain_timeout = Duration::from_millis(10);
        let ticket = coordinator.request(gate.device_generation(), fatal(), || Ok(drained_rx)).unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(2), gate.wait_for_pause_to_finish(ticket)).await.unwrap().unwrap();
        assert_eq!(gate.device_generation(), CudaDeviceGeneration(1));
    }

    #[tokio::test]
    async fn test_local_fatal_failure_is_terminal_and_notifies_application_without_reload() {
        let gate = Arc::new(CudaActivityGate::new(1));
        let (failure_tx, mut failure_rx) = tokio::sync::mpsc::unbounded_channel();
        let coordinator = CudaRecoveryCoordinator::with_backend(Arc::clone(&gate), Arc::new(|_| panic!("fatal local errors must never reload")), None, Some(Arc::new(move |failure| { failure_tx.send(failure).unwrap(); })));
        let error = coordinator.request(gate.device_generation(), fatal(), drained).unwrap_err();
        assert!(error.is_cuda_recovery_failed());
        assert!(failure_rx.recv().await.unwrap().is_cuda_recovery_failed());
        assert_eq!(gate.device_generation(), CudaDeviceGeneration(0));
        assert!(gate.check_admission().unwrap_err().is_cuda_recovery_failed());
    }

    #[tokio::test]
    async fn test_failed_local_stream_drain_is_terminal_without_reset() {
        let gate = Arc::new(CudaActivityGate::new(1));
        let (failure_tx, mut failure_rx) = tokio::sync::mpsc::unbounded_channel();
        let coordinator = CudaRecoveryCoordinator::with_backend(Arc::clone(&gate), Arc::new(|_| panic!("uncertain local completion must not reset live LibTorch")), None, Some(Arc::new(move |failure| { failure_tx.send(failure).unwrap(); })));
        let failure = crate::copy_ctx::complete_cuda_stream(Err(ThoError::oom()), || Err(ThoError::oom())).result.unwrap_err();
        assert!(!failure.needs_cuda_recovery());
        let error = coordinator.request(gate.device_generation(), failure, drained).unwrap_err();
        assert!(error.is_cuda_recovery_failed());
        assert!(failure_rx.recv().await.unwrap().is_cuda_recovery_failed());
        assert_eq!(gate.device_generation(), CudaDeviceGeneration(0));
    }

    #[tokio::test]
    async fn test_local_fatal_report_supersedes_maintenance_without_reopening() {
        let gate = Arc::new(CudaActivityGate::new(1));
        let (failure_tx, mut failure_rx) = tokio::sync::mpsc::unbounded_channel();
        let coordinator = CudaRecoveryCoordinator::with_backend(Arc::clone(&gate), Arc::new(|_| panic!("maintenance must not run after a fatal report")), None, Some(Arc::new(move |failure| { failure_tx.send(failure).unwrap(); })));
        let (_held, pending) = oneshot::channel();
        let ticket = coordinator.request(gate.device_generation(), ThoError::CudaBeingRecovered, || Ok(pending)).unwrap().unwrap();
        let error = coordinator.request(gate.device_generation(), fatal(), drained).unwrap_err();
        assert!(error.is_cuda_recovery_failed());
        assert!(failure_rx.recv().await.unwrap().is_cuda_recovery_failed());
        assert!(gate.wait_for_pause_to_finish(ticket).await.unwrap_err().is_cuda_recovery_failed());
        gate.finish_pause(ticket);
        tokio::task::yield_now().await;
        assert_eq!(gate.device_generation(), CudaDeviceGeneration(0));
        assert!(gate.check_admission().unwrap_err().is_cuda_recovery_failed());
        assert!(failure_rx.try_recv().is_err(), "superseded coordinator cannot publish another failure");
    }

    #[tokio::test]
    async fn test_local_maintenance_still_rebuilds_and_failed_retirement_stays_closed() {
        let gate = Arc::new(CudaActivityGate::new(1));
        let coordinator = CudaRecoveryCoordinator::with_backend(Arc::clone(&gate), Arc::new(|_| Box::pin(async { Ok(()) })), None, Some(Arc::new(|_| panic!("healthy maintenance cannot fail the process"))));
        let ticket = coordinator.request(gate.device_generation(), ThoError::CudaBeingRecovered, drained).unwrap().unwrap();
        gate.wait_for_pause_to_finish(ticket).await.unwrap();
        let mut coordinator = CudaRecoveryCoordinator::with_backend(Arc::clone(&gate), Arc::new(|_| panic!("cannot reload an unreaped worker")), Some(Arc::new(|_, _| Box::pin(async { Err(ThoError::new("reap failed")) }))), None);
        Arc::get_mut(&mut coordinator).unwrap().drain_timeout = Duration::from_millis(10);
        let (_held, pending) = oneshot::channel();
        let ticket = coordinator.request(gate.device_generation(), fatal(), || Ok(pending)).unwrap().unwrap();
        assert!(tokio::time::timeout(Duration::from_secs(2), gate.wait_for_pause_to_finish(ticket)).await.unwrap().unwrap_err().is_cuda_recovery_failed());
        assert_eq!(gate.device_generation(), CudaDeviceGeneration(1));
    }

    #[tokio::test]
    async fn test_devices_recover_independently_and_deduplicate_own_reports() {
        tokio::time::timeout(Duration::from_secs(2), async {
            let (started, mut starts) = tokio::sync::mpsc::unbounded_channel();
            let releases = [Arc::new(Notify::new()), Arc::new(Notify::new())];
            let reset: RestartFn = Arc::new({
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
                assert!(failure.original.needs_cuda_recovery());
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
