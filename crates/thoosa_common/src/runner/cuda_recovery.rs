use std::collections::BTreeMap;
use std::sync::Arc;

use futures_util::future::BoxFuture;
use parking_lot::Mutex;
use thoosa_shared::{CudaActivityPauseTicket, CudaActivityScope, CudaDeviceGeneration, ThoError, cuda_activity_gate};
use tokio::sync::oneshot;
use tokio::task::AbortHandle;

pub(crate) type ResetFn = Arc<dyn Fn(usize) -> BoxFuture<'static, Result<(), ThoError>> + Send + Sync>;

struct DeviceReport {
    generation: CudaDeviceGeneration,
    cause: ThoError,
    recovered: bool,
}

struct RecoveryCycle {
    ticket: CudaActivityPauseTicket,
    devices: BTreeMap<usize, DeviceReport>,
    drains: Vec<oneshot::Receiver<()>>,
    task: Option<AbortHandle>,
}

/// GlobalCtx owns one coordinator shared by all its executors. This mutex always
/// precedes the activity-gate mutex; neither is held while draining or rebuilding.
pub(crate) struct CudaRecoveryCoordinator {
    cycle: Mutex<Option<RecoveryCycle>>,
    reset: ResetFn,
}

impl std::fmt::Debug for CudaRecoveryCoordinator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaRecoveryCoordinator").finish_non_exhaustive()
    }
}

impl CudaRecoveryCoordinator {
    pub(crate) fn new(reset: ResetFn) -> Arc<Self> {
        Arc::new(Self { cycle: Mutex::new(None), reset })
    }

    /// Closing admission, recording the report, and arranging the reporting
    /// executor's drain are one synchronous operation. The same mutex protects
    /// final publication, so a late report is either included or already retired.
    pub(crate) fn request(
        self: &Arc<Self>,
        device: usize,
        generation: CudaDeviceGeneration,
        cause: ThoError,
        drain: impl FnOnce() -> Result<oneshot::Receiver<()>, ThoError>,
    ) -> Result<Option<CudaActivityPauseTicket>, ThoError> {
        let mut cycle = self.cycle.lock();
        let gate = cuda_activity_gate();
        if let Some(error) = gate.failure() {
            return Err(error);
        }
        let current = gate.device_generation(device);
        if generation.0 < current.0 {
            return Ok(None);
        }
        if generation != current {
            return Err(ThoError::new("CUDA recovery report refers to a future device generation"));
        }
        if let Some(recovery) = cycle.as_ref() && recovery.devices.contains_key(&device) {
            // This generation is already recorded, including while its backend
            // is running or has completed. Repeated reports must not delay finish.
            return Ok(Some(recovery.ticket));
        }

        let start = cycle.is_none();
        if start {
            let ticket = gate.request_pause(cause.clone())?;
            // A second coordinator must never steal another owner's pause.
            if !ticket.is_owner() {
                return Err(ThoError::new("CUDA recovery is owned by another coordinator"));
            }
            *cycle = Some(RecoveryCycle { ticket, devices: BTreeMap::new(), drains: Vec::new(), task: None });
        }
        let recovery = cycle.as_mut().expect("recovery cycle");
        let ticket = recovery.ticket;
        recovery.devices.entry(device).or_insert(DeviceReport { generation, cause, recovered: false });
        match drain() {
            Ok(drained) => recovery.drains.push(drained),
            Err(error) => {
                gate.fail_pause(ticket, Some(error));
                // A running backend must finish unwinding before its task exits.
                // Admission stays failed even if that backend subsequently succeeds.
                if start {
                    cycle.take();
                }
                return Err(gate.failure().expect("failed recovery"));
            }
        }

        // Spawn outside the lock: a shutting-down runtime may immediately drop
        // the future, whose guard also acquires this mutex to publish failure.
        drop(cycle);
        if start {
            let runtime = match tokio::runtime::Handle::try_current() {
                Ok(runtime) => runtime,
                Err(error) => {
                    self.fail(ticket, ThoError::new(error.to_string()));
                    return Err(gate.failure().expect("failed recovery"));
                }
            };
            // Construct the guard before spawning: abort before the first poll
            // must also publish failure. Dropping the JoinHandle detaches recovery
            // from the reporting task's cancellation and lifetime.
            let guard = RecoveryGuard { coordinator: Arc::clone(self), ticket };
            let task = runtime.spawn(async move {
                let _guard = guard;
                if let Err(error) = _guard.coordinator.run(ticket).await {
                    _guard.coordinator.fail(ticket, error);
                }
            });
            let mut cycle = self.cycle.lock();
            if let Some(recovery) = cycle.as_mut().filter(|recovery| recovery.ticket.cycle() == ticket.cycle()) {
                recovery.task = Some(task.abort_handle());
            }
        }
        Ok(Some(ticket))
    }

    async fn run(&self, ticket: CudaActivityPauseTicket) -> Result<(), ThoError> {
        loop {
            let drains = {
                let mut cycle = self.cycle.lock();
                std::mem::take(&mut cycle.as_mut().expect("recovery cycle").drains)
            };
            for drained in drains {
                drained.await.map_err(|_| ThoError::new("CUDA recovery reporting executor stopped before draining"))?;
            }
            cuda_activity_gate().wait_until_idle().await;

            let next = {
                let mut cycle = self.cycle.lock();
                if let Some(error) = cuda_activity_gate().failure() {
                    return Err(error);
                }
                let recovery = cycle.as_mut().expect("recovery cycle");
                if !recovery.drains.is_empty() {
                    continue;
                }
                match recovery.devices.iter().find(|(_, report)| !report.recovered) {
                    Some((&device, report)) => Some((device, report.generation, report.cause.clone())),
                    None => {
                        let devices = recovery.devices.keys().copied().collect::<Vec<_>>();
                        // Checking for new reports and reopening share the request
                        // lock. No failed device can be lost at this boundary.
                        cuda_activity_gate().finish_pause(ticket, &devices);
                        cycle.take();
                        None
                    }
                }
            };
            let Some((device, generation, cause)) = next else { return Ok(()) };
            CudaActivityScope::run_reset(ticket, async {
                (self.reset)(device).await
            }).await.map_err(|error| error.context(format!("recovering CUDA device {device}, generation {}, after {cause:#}", generation.0)))?;
            self.cycle.lock().as_mut().expect("recovery cycle").devices.get_mut(&device).expect("reported device").recovered = true;
        }
    }

    fn fail(&self, ticket: CudaActivityPauseTicket, error: ThoError) {
        let mut cycle = self.cycle.lock();
        if cycle.as_ref().is_some_and(|cycle| cycle.ticket.cycle() == ticket.cycle()) {
            cuda_activity_gate().fail_pause(ticket, Some(error));
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
        // leave it owned, and must wake admission waiters with a terminal error.
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
    async fn test_all_reported_devices_recover_once_before_reopening() {
        tokio::time::timeout(Duration::from_secs(2), async {
            let (started, mut starts) = tokio::sync::mpsc::unbounded_channel();
            let release = Arc::new(Notify::new());
            let coordinator = CudaRecoveryCoordinator::new(Arc::new({
                let release = Arc::clone(&release);
                move |device| {
                    let started = started.clone();
                    let release = Arc::clone(&release);
                    Box::pin(async move {
                        started.send(device).unwrap();
                        release.notified().await;
                        Ok(())
                    })
                }
            }));
            let gate = cuda_activity_gate();
            let first_generation = gate.device_generation(1);
            let second_generation = gate.device_generation(2);
            let first = coordinator.request(1, first_generation, fatal(), drained).unwrap().unwrap();
            assert_eq!(starts.recv().await, Some(1));
            // A second device reports after the first backend has already begun.
            let second = coordinator.request(2, second_generation, fatal(), drained).unwrap().unwrap();
            assert_eq!(first.cycle(), second.cycle());
            coordinator.request(1, first_generation, fatal(), || panic!("duplicate drain")).unwrap();
            release.notify_one();
            assert_eq!(starts.recv().await, Some(2));
            assert_eq!(gate.device_generation(1), first_generation);
            let mut admission = Box::pin(gate.wait_if_paused());
            assert!(futures::poll!(admission.as_mut()).is_pending());
            // Even a report for an already recovered member of this cycle is
            // deduplicated until all affected devices can be published together.
            coordinator.request(1, first_generation, fatal(), || panic!("duplicate completed device")).unwrap();
            release.notify_one();
            admission.await.unwrap();
            assert_eq!(gate.device_generation(1).0, first_generation.0 + 1);
            assert_eq!(gate.device_generation(2).0, second_generation.0 + 1);
            assert!(starts.try_recv().is_err());
            assert!(coordinator.request(1, first_generation, fatal(), || panic!("retired report drained")).unwrap().is_none());
            assert!(!gate.is_paused());
        }).await.expect("every reported device must complete in one cycle");
    }

    #[tokio::test]
    async fn test_reporting_task_cancellation_does_not_cancel_recovery() {
        let coordinator = CudaRecoveryCoordinator::new(Arc::new(|_| Box::pin(async { Ok(()) })));
        let (drain_tx, drain_rx) = oneshot::channel();
        let (reported_tx, reported_rx) = oneshot::channel();
        let reporter = tokio::spawn({
            let coordinator = Arc::clone(&coordinator);
            async move {
                let ticket = coordinator.request(1, cuda_activity_gate().device_generation(1), fatal(), || Ok(drain_rx)).unwrap().unwrap();
                reported_tx.send(ticket).unwrap();
                std::future::pending::<()>().await;
            }
        });
        let ticket = reported_rx.await.unwrap();
        reporter.abort();
        assert!(reporter.await.unwrap_err().is_cancelled());
        assert!(cuda_activity_gate().is_recovering());
        drain_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), cuda_activity_gate().wait_for_pause_to_finish(ticket)).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_coordinator_interruption_fails_admission_and_wakes_waiters() {
        const CHILD_ENV: &str = "THOOSA_TEST_COORDINATOR_FAILURE";
        let Some(mode) = std::env::var_os(CHILD_ENV) else {
            // Permanent failure is process-wide. Keep each case isolated even
            // under cargo test, which shares a process between unrelated tests.
            for mode in ["abort_before_poll", "abort_backend", "panic", "drain_closed", "drain_send_error"] {
                let output = tokio::process::Command::new(std::env::current_exe().unwrap())
                    .args(["--exact", "runner::cuda_recovery::tests::test_coordinator_interruption_fails_admission_and_wakes_waiters", "--nocapture"])
                    .env(CHILD_ENV, mode)
                    .kill_on_drop(true)
                    .output();
                let output = tokio::time::timeout(Duration::from_secs(10), output).await.unwrap().unwrap();
                assert!(output.status.success(), "{mode}: stdout={} stderr={}", String::from_utf8_lossy(&output.stdout), String::from_utf8_lossy(&output.stderr));
            }
            return;
        };
        let mode = mode.to_str().unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            let started = Arc::new(Notify::new());
            let panics = mode == "panic";
            let coordinator = CudaRecoveryCoordinator::new(Arc::new({
                let started = Arc::clone(&started);
                move |_| {
                    let started = Arc::clone(&started);
                    Box::pin(async move {
                        if panics {
                            panic!("injected coordinator backend panic");
                        }
                        started.notify_one();
                        std::future::pending::<Result<(), ThoError>>().await
                    })
                }
            }));
            let gate = cuda_activity_gate();
            let generation = gate.device_generation(1);
            let request = coordinator.request(1, generation, fatal(), || {
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
            assert_eq!(gate.device_generation(1), generation);
            assert!(gate.check_admission().unwrap_err().is_cuda_recovery_failed());
        }).await.expect("coordinator failure must wake waiters");
    }
}
