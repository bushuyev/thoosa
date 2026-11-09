use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fmt::{Debug, Formatter},
    future::Future,
    pin::Pin,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
};

#[cfg(test)]
#[cfg(any(test, feature = "mock_global_ctx"))]
use futures::FutureExt;
use log::error;
use parking_lot::{Mutex, RwLock};
use tokio::{
    sync::{mpsc, oneshot, watch},
    task::{JoinHandle, JoinSet},
};
use tracing::{Instrument, debug, warn};

use thoosa_shared::job_context::{CurrentJob, JobContext, ManagerCmd, ParentSuspendGuard};

use super::cuda_recovery::{CudaRecoveryCoordinator, RestartFn};
use thoosa_shared::futures_limits::{FuturesLimits, SyncExecutionPermit, SyncExecutionWaiter, futures_limits};
use thoosa_shared::tracked_await::t;
use thoosa_shared::{CudaActivityGate, CudaActivityPauseTicket, CudaActivityScope, CudaDeviceGeneration, CudaRecoveryState, LOCK_TIMEOUT, SyncStageRunnerError, ThoError};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum JobFeedback {
    /// Job failed with OOM; dynamic parallelism should decrease.
    Oom,
    /// Job completed successfully; dynamic parallelism may increase.
    Success,
    /// Job failed for a non-OOM reason, or requested CUDA reset; do not tune parallelism.
    Neutral,
}

type JobFuture = Pin<Box<dyn Future<Output = JobFeedback> + Send + 'static>>;
type PendingJobFailure = Box<dyn FnOnce(ThoError) + Send + 'static>;
type JobId = u64;

fn executor_closed_error(msg: impl Into<String>) -> ThoError {
    ThoError::new_with_cause(msg, ThoError::SyncStageRunner(SyncStageRunnerError::SubmitErrorClosed))
}



static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

/// Owns one sequential retry chain, including the waits between attempts. Tokens
/// held by detached jobs do not extend this owner's lifetime.
pub struct RetryScope {
    token: RetryToken,
}

#[derive(Clone, Debug)]
pub struct RetryToken {
    inner: Arc<RetryState>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ExclusiveOwner {
    job_id: JobId,
    family_id: JobId,
}

#[derive(Debug)]
struct RetryState {
    job_id: JobId,
    family_id: Mutex<Option<JobId>>,
    alive: Mutex<bool>,
    generation: Mutex<Option<CudaDeviceGeneration>>,
    generation_bound: tokio::sync::Notify,
    gate: Option<Arc<CudaActivityGate>>,
    exclusive_job_id: Arc<RwLock<Option<ExclusiveOwner>>>,
    cmd_tx: mpsc::UnboundedSender<ManagerCmd>,
}

impl RetryScope {
    pub fn token(&self) -> RetryToken {
        self.token.clone()
    }

    pub(crate) async fn run<T>(&self, future: impl Future<Output = Result<T, ThoError>>) -> Result<T, ThoError> {
        // Adjusters may await CPU work between attempts. Stop the whole chain at
        // recovery instead of carrying old inputs/selectors into a new generation.
        tokio::select! {
            biased;
            error = self.token.wait_for_invalidation() => Err(error),
            result = future => result,
        }
    }
}

impl Drop for RetryScope {
    fn drop(&mut self) {
        // The same lock serializes abandonment with a late OOM reservation. An
        // already-running job may finish, but can never resurrect its reservation.
        let mut alive = self.token.inner.alive.lock();
        *alive = false;
        self.token.clear_exclusive();
    }
}

impl RetryToken {
    fn bind_generation(&self) -> Result<(), ThoError> {
        let Some(gate) = &self.inner.gate else { return Ok(()) };
        let mut generation = self.inner.generation.lock();
        if let Some(generation) = *generation {
            gate.check_generation(generation)?;
        } else {
            gate.check_admission()?;
            *generation = Some(gate.device_generation());
            self.inner.generation_bound.notify_waiters();
        }
        Ok(())
    }

    fn check_generation(&self) -> Result<(), ThoError> {
        if let Some(gate) = &self.inner.gate
            && let Some(generation) = *self.inner.generation.lock()
        {
            gate.check_generation(generation)?;
        }
        Ok(())
    }

    async fn wait_for_invalidation(&self) -> ThoError {
        let Some(gate) = &self.inner.gate else { return std::future::pending().await };
        let generation = loop {
            let notified = {
                let generation = self.inner.generation.lock();
                if let Some(generation) = *generation {
                    break generation;
                }
                self.inner.generation_bound.notified()
            };
            notified.await;
        };
        gate.wait_for_generation_end(generation).await
    }

    fn reserve_exclusive(&self) {
        let alive = self.inner.alive.lock();
        if *alive && self.check_generation().is_ok() {
            // A descendant must not replace its ancestor's reservation: finishing
            // that child cannot release exclusivity owned by an outer retry loop.
            let mut exclusive = self.inner.exclusive_job_id.write();
            if exclusive.is_none() {
                *exclusive = Some(ExclusiveOwner { job_id: self.inner.job_id, family_id: self.inner.family_id.lock().expect("submitted retry family") });
            }
        }
    }

    fn clear_exclusive(&self) {
        let mut exclusive = self.inner.exclusive_job_id.write();
        if exclusive.is_some_and(|owner| owner.job_id == self.inner.job_id) {
            exclusive.take();
            let _ = self.inner.cmd_tx.send(ManagerCmd::Recheck);
        }
    }
}


struct Job {
    id: JobId,
    family_id: JobId,
    fut: JobFuture,
    pending_failure: Option<PendingJobFailure>,
    /// Maximum number of *other* jobs allowed to run together with this one.
    ///
    /// Examples:
    /// - 0 => exclusive job
    /// - 1 => may run with at most 1 other job
    /// - usize::MAX => no per-job restriction beyond global limit
    max_other_parallel: usize,
    inherited: Option<CudaActivityScope>,
    cancellation: Option<Arc<AttemptCancellation>>,
}

// Cancelling a submitter removes queued work, but never aborts an executing
// CUDA future. Its activity leases must survive until physical completion.
#[derive(Debug)]
struct AttemptCancellation {
    reason: Mutex<Option<ThoError>>,
    cmd_tx: mpsc::UnboundedSender<ManagerCmd>,
}

impl AttemptCancellation {
    fn cancel(&self, error: ThoError) {
        let mut reason = self.reason.lock();
        if reason.is_none() {
            *reason = Some(error);
            let _ = self.cmd_tx.send(ManagerCmd::Recheck);
        }
    }
}

struct SubmissionGuard(Arc<AttemptCancellation>);

impl Drop for SubmissionGuard {
    fn drop(&mut self) {
        self.0.cancel(ThoError::SyncStageRunner(SyncStageRunnerError::SubmitErrorCancelled));
    }
}

async fn await_dependency<F: Future>(activity: &Option<CudaActivityScope>, future: F) -> Result<F::Output, ThoError> {
    match activity {
        Some(activity) => tokio::select! {
            biased;
            error = activity.wait_for_interruption() => Err(error),
            result = future => Ok(result),
        },
        None => Ok(future.await),
    }
}

impl Job {
    fn new<F>(id: JobId, max_other_parallel: usize, fut: F, pending_failure: PendingJobFailure) -> Self
    where
        F: Future<Output = JobFeedback> + Send + 'static,
    {
        Self {
            id,
            family_id: id,
            fut: Box::pin(fut),
            pending_failure: Some(pending_failure),
            max_other_parallel,
            inherited: None,
            cancellation: None,
        }
    }

    fn total_parallel_cap(&self) -> usize {
        self.max_other_parallel.saturating_add(1)
    }

    fn cancelled(&self) -> Option<ThoError> {
        self.cancellation.as_ref().and_then(|cancel| cancel.reason.lock().clone()).or_else(|| self.inherited.as_ref().and_then(|scope| scope.check_dependencies().err()))
    }

    async fn run(self) -> JobFeedback {
        let job_id = self.id;
        debug!("SyncStageRunner, Job::run: running job {self:?}");

        let r = t!(self.fut).await;

        debug!("SyncStageRunner, Job::run: done job {job_id:?}");

        r
    }

    fn fail_pending(mut self, error: ThoError) {
        debug!("SyncStageRunner, Job::fail_pending: job_id={} error={error:?}", self.id);
        if let Some(pending_failure) = self.pending_failure.take() {
            pending_failure(error);
        }
    }
}

struct JobResultSender<T> {
    tx: Mutex<Option<oneshot::Sender<Result<T, ThoError>>>>,
}

impl<T> JobResultSender<T> {
    fn new(tx: oneshot::Sender<Result<T, ThoError>>) -> Self {
        Self { tx: Mutex::new(Some(tx)) }
    }

    fn send(&self, result: Result<T, ThoError>) {
        if let Some(tx) = self.tx.lock().take() {
            let _ = tx.send(result);
        }
    }
}

impl Debug for Job {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Job").field("id", &self.id).field("max_other_parallel", &self.max_other_parallel).finish()
    }
}

#[derive(Debug)]
struct LiveJob {
    family_id: JobId,
    /// max_other_parallel + 1
    cap: usize,
    /// Logical scheduler-active state. A suspended job is still physically alive,
    /// but is not counted against scheduling capacity.
    active: bool,
    /// Held for its Drop implementation: releasing it returns one slot to the shared limiter.
    sync_execution_permit: Option<SyncExecutionPermit>,
    suspend_depth: usize,
    waiting_resumes: Vec<oneshot::Sender<()>>,
    unwinding: bool,
    interrupted: Arc<AtomicBool>,
}


#[derive(Debug)]
#[allow(dead_code)]
pub enum SubmitError {
    Closed,
    Cancelled,
}

impl std::fmt::Display for SubmitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SubmitError::Closed => write!(f, "executor is closed"),
            SubmitError::Cancelled => write!(f, "job was cancelled before producing a result"),
        }
    }
}

impl std::error::Error for SubmitError {}

/// Its goal is to run synchronous stages limiting parallelism.
/// Maximum allowed number of parallel stages is specified in max_other_parallel parameter to submit method
/// There is single executor instance per device. The aggregate number of  active jobs is bounded by that adaptive limit.
/// Currently, it's used from OOMHandler only.
pub struct SyncStageRunner {
    tx: mpsc::Sender<Job>,
    cmd_tx: mpsc::UnboundedSender<ManagerCmd>,
    worker: JoinHandle<()>,
    exclusive_job_id: Arc<RwLock<Option<ExclusiveOwner>>>,
    tho_device_idx: usize,
    recovery: Option<Arc<CudaRecoveryCoordinator>>, //None for CPU
}

impl Debug for SyncStageRunner {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncStageRunner")
            .field("tx", &self.tx)
            .field("cmd_tx", &self.cmd_tx)
            .field("worker", &self.worker)
            .field("exclusive_job_id", &self.exclusive_job_id)
            .field("tho_device_idx", &self.tho_device_idx)
            .finish_non_exhaustive()
    }
}

impl SyncStageRunner {
    pub fn retry_scope(&self) -> RetryScope {
        let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
        RetryScope {
            token: RetryToken {
                inner: Arc::new(RetryState {
                    job_id,
                    family_id: Mutex::new(None),
                    alive: Mutex::new(true),
                    generation: Mutex::new(None),
                    generation_bound: tokio::sync::Notify::new(),
                    gate: self.cuda_gate().cloned(),
                    exclusive_job_id: Arc::clone(&self.exclusive_job_id),
                    cmd_tx: self.cmd_tx.clone(),
                }),
            },
        }
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub fn new(queue_capacity: usize) -> Self {
        Self::new_cpu(queue_capacity)
    }

    pub(crate) fn new_cpu(queue_capacity: usize) -> Self {
        Self::new_inner_with_buffer_limits(queue_capacity, 0, None, Arc::clone(futures_limits()))
    }

    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub fn new_with_reset_hook(queue_capacity: usize, tho_device_idx: usize, reset_fn: RestartFn) -> Self {
        Self::new_with_recovery(queue_capacity, CudaRecoveryCoordinator::new(Arc::new(CudaActivityGate::new(tho_device_idx)), reset_fn))
    }

    pub(crate) fn new_with_recovery(queue_capacity: usize, recovery: Arc<CudaRecoveryCoordinator>) -> Self {
        Self::new_inner_with_buffer_limits(queue_capacity, recovery.gate().device_idx(), Some(recovery), Arc::clone(futures_limits()))
    }

    fn new_inner_with_buffer_limits(queue_capacity: usize, tho_device_idx: usize, recovery: Option<Arc<CudaRecoveryCoordinator>>, buffer_limits: Arc<FuturesLimits>) -> Self {
        debug!("SyncStageRunner::new_inner: queue_capacity={queue_capacity} tho_device_idx={tho_device_idx}");

        let (tx, rx) = mpsc::channel(queue_capacity);
        let sync_capacity_rx = buffer_limits.sync_capacity();
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();

        let cmd_tx_for_tasks = cmd_tx.clone();
        let exclusive_job_id = Arc::new(RwLock::new(None));
        let exclusive_job_id_for_manager = Arc::clone(&exclusive_job_id);

        let gate = recovery.as_ref().map(|recovery| Arc::clone(recovery.gate()));
        let worker = tokio::spawn(
            async move {
                t!(Self::run_manager(rx, sync_capacity_rx, cmd_rx, cmd_tx_for_tasks, exclusive_job_id_for_manager, buffer_limits, gate,)).await;
            }
            .instrument(tracing::info_span!("dyn_mpsc::manager", kind = "spawn", file = file!(), line = line!(), task = format!("run manager for tho_device_idx={tho_device_idx}"),)),
        );

        Self {
            tx,
            cmd_tx,
            worker,
            exclusive_job_id,
            tho_device_idx,
            recovery,
        }
    }

    fn closed_error(msg: impl Into<String>) -> ThoError {
        executor_closed_error(msg)
    }

    fn context_reset_error() -> ThoError {
        ThoError::new_with_cause("Resetting cuda context", ThoError::CudaBeingRecovered)
    }

    pub(crate) fn cuda_gate(&self) -> Option<&Arc<CudaActivityGate>> {
        self.recovery.as_ref().map(|recovery| recovery.gate())
    }

    async fn wait_until_accepting(&self, parent: Option<&CurrentJob>) -> Result<(), ThoError> {
        let Some(gate) = self.cuda_gate() else { return Ok(()) };
        if parent.is_some() || CudaActivityScope::current().is_some() {
            // Reset may be waiting for this admitted parent/child to unwind.
            // Its nested submissions must fail instead of waiting for reset.
            gate.check_admission()?;
        } else {
            t!(gate.wait_if_paused()).await?;
        }
        Ok(())
    }

    pub async fn submit<F, T>(&self, max_other_parallel: usize, fut: F, retry: Option<RetryToken>) -> Result<T, ThoError>
    where
        F: Future<Output = Result<T, ThoError>> + Send + 'static,
        T: Send + Debug + 'static,
    {
        self.submit_observed(max_other_parallel, fut, retry, |_| None).await
    }

    pub(crate) async fn submit_consume<F, I, T>(&self, max_other_parallel: usize, fut: F, retry: Option<RetryToken>) -> Result<thoosa_common::runner::sync_stage::ConsumeResult<I, T>, ThoError>
    where
        F: Future<Output = Result<thoosa_common::runner::sync_stage::ConsumeResult<I, T>, ThoError>> + Send + 'static,
        I: Send + Debug + 'static,
        T: Send + Debug + 'static,
    {
        self.submit_observed(max_other_parallel, fut, retry, |result| match result {
            thoosa_common::runner::sync_stage::ConsumeResult::Failed { error, .. } => Some(error),
            thoosa_common::runner::sync_stage::ConsumeResult::Ok(_) => None,
        })
        .await
    }

    async fn submit_observed<F, T>(
        &self,
        max_other_parallel: usize, //how many other jobs SyncStageRunner can run in parallel with the job being submitted. 0 means job wants exclusive access to GPU.
        fut: F,
        retry: Option<RetryToken>,
        nested_error: fn(&mut T) -> Option<&mut ThoError>,
    ) -> Result<T, ThoError>
    where
        F: Future<Output = Result<T, ThoError>> + Send + 'static, //TODO change to closure taking ticket to let function a way to ensure they're running in Executor
        T: Send + Debug + 'static,
    {
        let parent = JobContext::current().job;
        let inherited = CudaActivityScope::current();
        let cancellation = Arc::new(AttemptCancellation { reason: Mutex::new(None), cmd_tx: self.cmd_tx.clone() });
        let _submission = SubmissionGuard(Arc::clone(&cancellation));
        // Direct submissions own their scope until return; OOMHandler supplies a
        // scope that also spans error processing and adjustments between retries.
        let owned_scope = retry.is_none().then(|| self.retry_scope());
        let retry = retry.unwrap_or_else(|| owned_scope.as_ref().expect("local retry scope").token());

        debug_assert!(Arc::ptr_eq(&retry.inner.exclusive_job_id, &self.exclusive_job_id), "retry scope belongs to another executor"); //it compares locks' addresses, not job ids - if it fails it's a bug in the app and we can't handle it

        // Retry ownership is stable, but a physical attempt needs a distinct ID:
        // late dependency resumes must never change a successor attempt's slot.
        let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
        // Bind at first submission, since callers may construct a retry scope
        // before entering the scheduled frame whose work it depends on.
        let family_id = *retry.inner.family_id.lock().get_or_insert_with(|| JobContext::current().family_id.unwrap_or(retry.inner.job_id));
        if max_other_parallel != 0 {
            retry.clear_exclusive();
        }

        debug!("SyncStageRunner::submit: job_id={job_id} max_other_parallel={max_other_parallel:?} parent={parent:?}");

        // Admission can close while reserve/suspend awaits. Keep ownership of the
        // unqueued job until the gate atomically accepts it, including across resets.
        let (result_tx, result_rx) = oneshot::channel::<Result<T, ThoError>>();
        let result_sender = Arc::new(JobResultSender::new(result_tx));
        retry.check_generation()?;
        let retry_for_job = retry.clone();
        let cmd_tx = self.cmd_tx.clone();
        let recovery = self.recovery.clone();

        let result_sender_for_job = Arc::clone(&result_sender);
        let wrapped = async move {
            let retry = retry_for_job;
            debug!("SyncStageRunner::submit: job_id={job_id} waiting on wrapped fut");
            if !*retry.inner.alive.lock() {
                result_sender.send(Err(ThoError::new_with_cause("retry owner was cancelled", ThoError::SyncStageRunner(SyncStageRunnerError::SubmitErrorCancelled))));
                return JobFeedback::Neutral;
            }
            // Capture the generation of the admitted attempt, not the generation
            // observed after its failure (which could otherwise be a replacement).
            if let Err(error) = retry.bind_generation() {
                retry.clear_exclusive();
                result_sender.send(Err(error));
                return JobFeedback::Neutral;
            }
            let generation = recovery.as_ref().map(|recovery| CudaActivityScope::current().and_then(|scope| scope.generation(recovery.gate())).expect("admitted CUDA generation"));
            let mut result = t!(fut).await; //<--- wrapped
            debug!("SyncStageRunner::submit: result={result:?}");
            let error = match &mut result {
                Err(error) => Some(error),
                Ok(value) => nested_error(value),
            };
            let is_oom = error.as_ref().is_some_and(|error| error.is_oom());
            let is_ok = error.is_none();
            if let Some(error) = error {
                debug!("SyncStageRunner::submit: needs_cuda_ctx_reset={} is_cuda_recovery_failed={} recovery={recovery:?}", error.needs_cuda_recovery(), error.is_cuda_recovery_failed());

                if (error.needs_cuda_recovery() || error.is_cuda_recovery_failed())
                    && let Some(recovery) = &recovery
                {
                    retry.clear_exclusive();
                    *error = Self::report_failure(recovery, &cmd_tx, generation.expect("CUDA generation"), error.clone()).err().unwrap_or_else(Self::context_reset_error);
                    // Preserve a consume failure's owned input for the adapter.
                    result_sender.send(result);
                    return JobFeedback::Neutral;
                }
                if is_oom && max_other_parallel == 0 {
                    retry.reserve_exclusive();
                    *error = SyncStageRunnerError::JobError { job_id: retry.inner.job_id, inner: Box::new(error.clone()) }.into();
                } else {
                    retry.clear_exclusive();
                }
            } else {
                retry.clear_exclusive();
            }

            result_sender.send(result);

            if is_oom {
                JobFeedback::Oom
            } else if is_ok {
                JobFeedback::Success
            } else {
                JobFeedback::Neutral
            }
        };

        let pending_failure = Box::new(move |error| result_sender_for_job.send(Err(error)));
        let mut job = Job::new(job_id, max_other_parallel, wrapped, pending_failure);
        job.family_id = family_id;
        job.inherited = inherited.clone();
        job.cancellation = Some(Arc::clone(&cancellation));

        loop {
            await_dependency(&inherited, t!(self.wait_until_accepting(parent.as_ref()))).await??;
            retry.check_generation()?;

            debug!("SyncStageRunner::submit: job_id={job_id} accepted");

            // Reserve queue capacity before suspending a parent. Otherwise a nested submit
            // could remove its parent from active accounting while still waiting for space
            // in the input channel, with no child yet queued to make progress.
            //
            //   1. submit() calls suspend_guard.maybe_suspend().
            //   2. It sends ManagerCmd::Suspend at thoosa_common/src/runner/syn_stage_runner:199.
            //   3. The manager handles it through handle_suspend().
            //   4. handle_suspend() sets job.active = false and calls remove_active() at thoosa_common/src/runner/syn_stage_runner:1218.
            //

            let send_permit = match await_dependency(&inherited, t!(tokio::time::timeout(LOCK_TIMEOUT, self.tx.reserve()))).await? {
                Ok(Ok(permit)) => permit,
                Ok(Err(e)) => {
                    error!("SyncStageRunner::submit: reserve job_id={job_id} failed: e={e:?}");
                    retry.clear_exclusive();
                    return Err(Self::closed_error(format!("Can't submit job: executor is_closed={}", self.tx.is_closed())));
                }
                Err(e) => {
                    error!("SyncStageRunner::submit: reserve job_id={job_id} timed out: e={e:?}");
                    retry.clear_exclusive();
                    return Err(ThoError::new("Can't submit job: timed out"));
                }
            };

            // Reset may have started while reserve() was awaiting channel capacity. The
            // permit alone does not authorize enqueueing, so recheck before suspension.
            if let Some(gate) = self.cuda_gate()
                && let Err(error) = gate.check_admission()
            {
                drop(send_permit);
                retry.clear_exclusive();
                if error.is_cuda_recovery_failed() || parent.is_some() || CudaActivityScope::current().is_some() {
                    return Err(error);
                }
                debug!("SyncStageRunner::submit: reserve job_id={job_id} cuda_activity_gate is paused 1");

                continue;
            }

            // A nested submit must stop counting its blocked parent as active before the
            // child is enqueued, or the parent's own parallelism cap can block the child.
            let mut suspend_guard = ParentSuspendGuard::new(parent.clone());
            if let Err(e) = await_dependency(&inherited, t!(suspend_guard.maybe_suspend())).await.and_then(|result| result) {
                drop(send_permit);
                retry.clear_exclusive();
                return Err(e);
            }

            // This check and publication share the pause mutex. Any job accepted
            // before reset closes admission is visible to the manager's purge.
            let enqueue = match self.cuda_gate() {
                Some(gate) => gate.try_enqueue(send_permit, job),
                None => {
                    send_permit.send(job);
                    Ok(())
                }
            };
            if let Err((rejected, error)) = enqueue {
                job = rejected;
                let _ = t!(suspend_guard.resume()).await;
                retry.clear_exclusive();
                if error.is_cuda_recovery_failed() || parent.is_some() || CudaActivityScope::current().is_some() {
                    return Err(error);
                }
                continue;
            }

            debug!("SyncStageRunner::submit: job_id={job_id} submitted job_id={job_id}");

            // Always try to resume the parent before returning, even if the job panicked
            // and closed result_tx without sending a value.
            let result = match await_dependency(&inherited, t!(result_rx)).await {
                Ok(Ok(result)) => result,
                Ok(Err(_)) => Err(ThoError::new_with_cause("job was cancelled before producing a result", ThoError::SyncStageRunner(SyncStageRunnerError::SubmitErrorCancelled))),
                Err(error) => {
                    cancellation.cancel(error.clone());
                    Err(error)
                }
            };
            t!(suspend_guard.resume()).await?;

            return result;
        }
    }

    pub fn clear_exclusive_job_id(&self, job_id: u64) {
        debug!("SyncStageRunner::clear_exclusive_job_id: job_id={job_id}");

        let mut exclusive_job_id = self.exclusive_job_id.write();

        if exclusive_job_id.is_some_and(|owner| owner.job_id == job_id) {
            exclusive_job_id.take();
            let _ = self.cmd_tx.send(ManagerCmd::Recheck);
        }
    }

    #[cfg(test)]
    pub(crate) fn request_cuda_reset(&self, cause: ThoError) -> Result<Option<CudaActivityPauseTicket>, ThoError> {
        let gate = self.cuda_gate().ok_or_else(|| ThoError::new("CPU executor has no CUDA recovery"))?;
        self.request_cuda_reset_for_generation(gate.device_generation(), cause)
    }

    pub(crate) fn request_cuda_reset_for_generation(&self, generation: CudaDeviceGeneration, cause: ThoError) -> Result<Option<CudaActivityPauseTicket>, ThoError> {
        let recovery = self.recovery.as_ref().ok_or_else(|| ThoError::new("CPU executor has no CUDA recovery"))?;
        Self::report_failure(recovery, &self.cmd_tx, generation, cause)
    }

    fn report_failure(recovery: &Arc<CudaRecoveryCoordinator>, sender: &mpsc::UnboundedSender<ManagerCmd>, generation: CudaDeviceGeneration, cause: ThoError) -> Result<Option<CudaActivityPauseTicket>, ThoError> {
        recovery.request(generation, cause, || {
            let (reply, drained) = oneshot::channel();
            sender.send(ManagerCmd::DrainForRecovery { reply }).map_err(|_| Self::closed_error("CUDA recovery reporting executor is closed"))?;
            Ok(drained)
        })
    }

    pub async fn shutdown(self) {
        drop(self.tx);
        drop(self.cmd_tx);
        let _ = t!(self.worker).await;
    }

    async fn run_manager(
        mut rx: mpsc::Receiver<Job>,
        mut sync_capacity_rx: watch::Receiver<usize>,
        mut cmd_rx: mpsc::UnboundedReceiver<ManagerCmd>,
        cmd_tx_for_tasks: mpsc::UnboundedSender<ManagerCmd>,
        exclusive_job_id: Arc<RwLock<Option<ExclusiveOwner>>>,
        buffer_limits: Arc<FuturesLimits>,
        gate: Option<Arc<CudaActivityGate>>,
    ) {
        let mut pending = VecDeque::<Job>::new();
        let mut running = JoinSet::<()>::new();

        let mut jobs = HashMap::<JobId, LiveJob>::new();
        let mut active_caps = BTreeMap::<usize, usize>::new();
        let mut active_count = 0usize;

        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<JobId>();

        let mut input_closed = false;
        let mut cmd_closed = false;
        let mut sync_capacity_closed = false;
        let mut pending_drains = Vec::<oneshot::Sender<()>>::new();
        let mut execution_waiter = buffer_limits.sync_execution_waiter();

        loop {
            execution_waiter.begin_turn();
            let max_parallel_jobs = *sync_capacity_rx.borrow_and_update();
            debug!(
                "SyncStageRunner::run_manager, loop: active_caps={active_caps:?} active_count={active_count} running.len={} pending.len={} max_parallel_jobs={max_parallel_jobs} jobs.len={}",
                running.len(),
                pending.len(),
                jobs.len()
            );

            Self::drain_finished(&mut running, &mut done_rx, &mut jobs, &mut active_caps, &mut active_count);

            // Critical: drain already queued commands before spawning pending jobs.
            // Otherwise a job sent immediately before DrainForRecovery could be spawned
            // before the manager observes the reset request.
            Self::drain_commands(&mut cmd_rx, &mut cmd_closed, &mut jobs, &mut active_caps, &mut active_count, max_parallel_jobs, &mut execution_waiter, &mut pending_drains, gate.as_ref(), *exclusive_job_id.read());

            for job in jobs.values_mut() {
                if job.interrupted.load(Ordering::Acquire) {
                    Self::mark_unwinding(job, &mut active_caps, &mut active_count);
                }
            }

            let mut index = 0;
            while index < pending.len() {
                if let Some(error) = pending[index].cancelled() {
                    pending.remove(index).expect("pending job").fail_pending(error);
                } else {
                    index += 1;
                }
            }

            let recovery_state = gate.as_ref().map_or(CudaRecoveryState::Running, |gate| gate.recovery_state());
            if let CudaRecoveryState::Failed(failure) = &recovery_state {
                // Terminal admission must resolve queued results as well as submitters
                // still waiting outside the channel. Already-running jobs unwind below.
                pending_drains.clear();
                exclusive_job_id.write().take();
                Self::purge_pending_jobs(&mut pending, &mut rx, &mut input_closed, ThoError::CudaRecoveryFailed(Arc::clone(failure)));
            } else if matches!(recovery_state, CudaRecoveryState::Recovering { .. }) {
                exclusive_job_id.write().take();
                // This device's gate closed admission atomically with enqueueing,
                // so no submitter can publish another job after this purge.
                Self::purge_pending_jobs(&mut pending, &mut rx, &mut input_closed, Self::context_reset_error());
            }

            // While reset is pending, no queued jobs are started. Already-running jobs
            // continue to completion, and suspended parents are allowed to resume so they
            // can observe a child's ContextReset and unwind before reset_fn runs.
            let cuda_activity_gate_is_paused = !matches!(recovery_state, CudaRecoveryState::Running);
            debug!("SyncStageRunner::run_manager: cuda_activity_gate_is_paused={cuda_activity_gate_is_paused}");

            if !cuda_activity_gate_is_paused {
                let current_exclusive_job_id = *exclusive_job_id.read();
                Self::try_resume_waiters(&mut jobs, &mut active_caps, &mut active_count, max_parallel_jobs, current_exclusive_job_id, &mut execution_waiter);
                Self::try_spawn_ready_jobs(
                    &mut pending,
                    &mut running,
                    &mut jobs,
                    &mut active_caps,
                    &mut active_count,
                    max_parallel_jobs,
                    &done_tx,
                    &cmd_tx_for_tasks,
                    current_exclusive_job_id,
                    &buffer_limits,
                    &mut execution_waiter,
                    gate.as_ref(),
                );
            } else {
                execution_waiter.cancel();
                Self::force_resume_waiters_for_unwind(&mut jobs, &mut active_caps, &mut active_count);
            }

            execution_waiter.end_turn();

            // Keep the reporting executor's existing physical-drain guarantee.
            // The coordinator owns backend work; this manager keeps servicing
            // suspension/resumption and completion messages while it drains.
            if !pending_drains.is_empty() && jobs.is_empty() && running.is_empty() {
                exclusive_job_id.write().take();
                for reply in pending_drains.drain(..) {
                    let _ = reply.send(());
                }
            }

            if input_closed && pending.is_empty() && jobs.is_empty() && !gate.as_ref().is_some_and(|gate| gate.is_recovering()) {
                break;
            }

            if !gate.as_ref().is_some_and(|gate| gate.is_paused()) && !pending.is_empty() && jobs.is_empty() {
                let exclusive = *exclusive_job_id.read();

                warn!(
                    "SyncStageRunner::run_manager: idle with pending jobs: pending_len={}, pending_front={:?}, exclusive_job_id={exclusive:?}, active_count={}, active_caps={active_caps:?}",
                    pending.len(),
                    pending.front().map(|j| (j.id, j.total_parallel_cap())),
                    active_count,
                );
            }

            // Subscribe to both closing and reopening admission. A failed gate is
            // terminal, so never repeatedly select its immediately-ready error.
            let waiting_for_pause = gate.is_some() && matches!(recovery_state, CudaRecoveryState::Running);
            let waiting_for_recovery = matches!(recovery_state, CudaRecoveryState::Recovering { .. });
            tokio::select! {
                maybe_job = rx.recv(), if !input_closed => {
                    match maybe_job {
                        Some(job) => pending.push_back(job),
                        None => input_closed = true,
                    }
                }

                maybe_done = done_rx.recv(), if !jobs.is_empty() => {
                    if let Some(job_id) = maybe_done {
                        Self::note_job_finished(
                            &mut jobs,
                            &mut active_caps,
                            &mut active_count,
                            job_id,
                        );
                    }
                }

                maybe_cmd = cmd_rx.recv(), if !cmd_closed => {
                    match maybe_cmd {
                        Some(cmd) => {
                            Self::process_command(
                                cmd,
                                &mut jobs,
                                &mut active_caps,
                                &mut active_count,
                                max_parallel_jobs,
                                &mut execution_waiter,
                                &mut pending_drains,
                                gate.as_ref(),
                                *exclusive_job_id.read(),
                            );
                        }
                        None => {
                            cmd_closed = true;
                        }
                    }
                }

                changed = sync_capacity_rx.changed(), if !sync_capacity_closed => {
                    match changed {
                        Ok(()) => {}
                        Err(_) => {
                            sync_capacity_closed = true;
                        }
                    }
                }

                _ = async { gate.as_ref().expect("CUDA gate").wait_until_paused().await }, if waiting_for_pause => {}
                _ = async { gate.as_ref().expect("CUDA gate").wait_if_paused().await }, if waiting_for_recovery => {}

                join_result = running.join_next(), if !running.is_empty() => {
                    if let Some(Err(err)) = join_result {
                        error!("job failed: {err}");
                    }
                }
            }
        }

        while let Some(res) = t!(running.join_next()).await {
            if let Err(err) = res {
                error!("job failed during shutdown: {err}");
            }
        }
    }

    fn process_command(cmd: ManagerCmd, jobs: &mut HashMap<JobId, LiveJob>, active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize, max_parallel_jobs: usize, execution_waiter: &mut SyncExecutionWaiter, pending_drains: &mut Vec<oneshot::Sender<()>>, gate: Option<&Arc<CudaActivityGate>>, exclusive: Option<ExclusiveOwner>) {
        debug!("SyncStageRunner::run_manager: manager cmd={cmd:?}");

        match cmd {
            ManagerCmd::Suspend { job_id, reply } => {
                Self::handle_suspend(jobs, active_caps, active_count, job_id);
                let _ = reply.send(());
            }
            ManagerCmd::Resume { job_id, reply, for_unwind } => {
                Self::handle_resume(jobs, active_caps, active_count, max_parallel_jobs, exclusive, job_id, reply, for_unwind || gate.is_some_and(|gate| gate.is_paused()), execution_waiter);
            }
            ManagerCmd::DrainForRecovery { reply } => pending_drains.push(reply),
            ManagerCmd::Recheck => {}
        }
    }

    fn drain_commands(
        cmd_rx: &mut mpsc::UnboundedReceiver<ManagerCmd>,
        cmd_closed: &mut bool,
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        max_parallel_jobs: usize,
        execution_waiter: &mut SyncExecutionWaiter,
        pending_drains: &mut Vec<oneshot::Sender<()>>,
        gate: Option<&Arc<CudaActivityGate>>,
        exclusive: Option<ExclusiveOwner>,
    ) {
        use tokio::sync::mpsc::error::TryRecvError;

        if *cmd_closed {
            return;
        }

        // A continuous stream of submissions/cancellations must not starve
        // completions, admission changes, or other executors on this runtime.
        for _ in 0..256 {
            //TODO 256?
            match cmd_rx.try_recv() {
                Ok(cmd) => Self::process_command(cmd, jobs, active_caps, active_count, max_parallel_jobs, execution_waiter, pending_drains, gate, exclusive),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    *cmd_closed = true;
                    break;
                }
            }
        }
    }

    fn purge_pending_jobs(pending: &mut VecDeque<Job>, rx: &mut mpsc::Receiver<Job>, input_closed: &mut bool, error: ThoError) {
        use tokio::sync::mpsc::error::TryRecvError;

        let mut purged = 0usize;
        while let Some(job) = pending.pop_front() {
            job.fail_pending(error.clone());
            purged += 1;
        }

        if !*input_closed {
            loop {
                match rx.try_recv() {
                    Ok(job) => {
                        job.fail_pending(error.clone());
                        purged += 1;
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        *input_closed = true;
                        break;
                    }
                }
            }
        }

        if purged > 0 {
            debug!("SyncStageRunner::purge_pending_jobs: purged={purged}");
        }
    }

    fn try_spawn_ready_jobs(
        pending: &mut VecDeque<Job>,
        running: &mut JoinSet<()>,
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        max_parallel_jobs: usize,
        done_tx: &mpsc::UnboundedSender<JobId>,
        cmd_tx_for_tasks: &mpsc::UnboundedSender<ManagerCmd>,
        exclusive_job_id: Option<ExclusiveOwner>,
        buffer_limits: &Arc<FuturesLimits>,
        execution_waiter: &mut SyncExecutionWaiter,
        gate: Option<&Arc<CudaActivityGate>>,
    ) {
        debug!("SyncStageRunner::try_spawn_ready_jobs: pending={pending:?}");

        loop {
            let mut spawned_any = false;
            let mut idx = 0usize;

            while idx < pending.len() {
                if let Some(error) = pending[idx].cancelled() {
                    pending.remove(idx).expect("pending job").fail_pending(error);
                    continue;
                }
                if jobs.contains_key(&pending[idx].id) {
                    idx += 1;
                    continue;
                }
                let sync_execution_permit = {
                    let job = &pending[idx];
                    Self::try_activate_now(job.family_id, *active_count, max_parallel_jobs, active_caps, job.total_parallel_cap(), exclusive_job_id, execution_waiter)
                };

                if let Some(sync_execution_permit) = sync_execution_permit {
                    let activity = match gate.map(CudaActivityScope::try_enter).transpose() {
                        Ok(None) => None, // CPU jobs only inherit their ancestors' leases.
                        Ok(Some(Some(activity))) => Some(activity),
                        Ok(Some(None)) => return, // Recovery raced the manager's check.
                        Err(error) => {
                            // Failed admission must not leave this queued job waiting.
                            pending.remove(idx).expect("pending job").fail_pending(error);
                            continue;
                        }
                    };
                    let mut job = pending.remove(idx).expect("pending index must be valid");
                    let inherited = job.inherited.take();
                    let activity = match activity {
                        Some(activity) => Some(activity.with_inherited(inherited)),
                        None => inherited,
                    };
                    let job_id = job.id;
                    let family_id = job.family_id;
                    let cap = job.total_parallel_cap();
                    let interrupted = Arc::new(AtomicBool::new(false));

                    jobs.insert(
                        job_id,
                        LiveJob {
                            family_id,
                            cap,
                            active: true,
                            sync_execution_permit: Some(sync_execution_permit),
                            suspend_depth: 0,
                            waiting_resumes: Vec::new(),
                            unwinding: false,
                            interrupted: Arc::clone(&interrupted),
                        },
                    );
                    Self::add_active(active_caps, active_count, cap);

                    let done_tx = done_tx.clone();
                    let cmd_tx_for_tasks = cmd_tx_for_tasks.clone();
                    let buffer_limits = Arc::clone(buffer_limits);

                    debug!("SyncStageRunner::try_spawn_ready_jobs: spawning job_id={job_id}");

                    running.spawn(
                        async move {
                            debug!("SyncStageRunner::try_spawn_ready_jobs: spawned job_id={job_id}");

                            struct DoneGuard {
                                job_id: JobId,
                                done_tx: mpsc::UnboundedSender<JobId>,
                            }

                            impl Drop for DoneGuard {
                                fn drop(&mut self) {
                                    let _ = self.done_tx.send(self.job_id);
                                }
                            }

                            let _guard = DoneGuard { job_id, done_tx };

                            let unwind_tx = cmd_tx_for_tasks.clone();
                            let future = JobContext::for_job(family_id, CurrentJob { job_id, cmd_tx: cmd_tx_for_tasks }).run(async move {
                                debug!("SyncStageRunner::try_spawn_ready_jobs: running job job_id={job_id}");

                                t!(job.run()).await
                            });
                            let feedback = match activity {
                                Some(activity) => {
                                    let dependencies = activity.clone();
                                    t!(activity.run(async move {
                                        tokio::pin!(future);
                                        tokio::select! {
                                            biased;
                                            _ = dependencies.wait_for_interruption() => {
                                                // This may be a healthy device with a recovering
                                                // ancestor. Release its scheduling slot even when
                                                // its parent is already waiting for a Resume ack.
                                                // Store interruption on the physical attempt so a
                                                // delayed wakeup cannot release another attempt's capacity.
                                                interrupted.store(true, Ordering::Release);
                                                let _ = unwind_tx.send(ManagerCmd::Recheck);
                                                future.await
                                            }
                                            feedback = &mut future => feedback,
                                        }
                                    }))
                                    .await
                                }
                                None => t!(future).await,
                            };

                            match feedback {
                                JobFeedback::Oom => {
                                    debug!("SyncStageRunner::try_spawn_ready_jobs: OOM job job_id={job_id}, max_parallel_jobs={}", buffer_limits.max_parallel_jobs());
                                    buffer_limits.on_oom();
                                }
                                JobFeedback::Success => {
                                    debug!("SyncStageRunner::try_spawn_ready_jobs: success job job_id={job_id}, max_parallel_jobs={}", buffer_limits.max_parallel_jobs());
                                    buffer_limits.on_ok();
                                }
                                JobFeedback::Neutral => {
                                    debug!("SyncStageRunner::try_spawn_ready_jobs: neutral job job_id={job_id}, max_parallel_jobs={}", buffer_limits.max_parallel_jobs());
                                }
                            }
                        }
                        .instrument(tracing::info_span!("dyn_mpsc::job", kind = "spawn", file = file!(), line = line!(), task = format!("run dynamic executor job_id={job_id}"),)),
                    );

                    spawned_any = true;
                    break;
                }

                // Any restrictive head needs the active group to drain. Bypassing
                // caps greater than one can starve them just like exclusive work.
                if idx == 0 {
                    match exclusive_job_id {
                        // If a specific OOM-rerun job is required, do not let some other
                        // exclusive job at the front block the scan.
                        Some(owner) if pending[0].family_id != owner.family_id => {
                            idx += 1;
                            continue;
                        }

                        // Normal fairness barrier: no sticky exclusive rerun is active.
                        _ => break,
                    }
                }
                idx += 1;
            }

            if !spawned_any {
                break;
            }
        }
    }

    fn can_activate_now(family_id: JobId, active_count: usize, max_parallel_jobs: usize, active_caps: &BTreeMap<usize, usize>, new_cap: usize, exclusive_job_id: Option<ExclusiveOwner>) -> bool {
        if exclusive_job_id.is_some_and(|owner| owner.family_id != family_id) {
            debug!("SyncStageRunner::can_activate_now: blocked by exclusive job family_id={family_id} exclusive_job_id={exclusive_job_id:?}");
            return false;
        }

        let next_total = active_count.saturating_add(1);
        let current_group_cap = Self::current_group_cap(max_parallel_jobs, active_caps);

        let min = current_group_cap.min(new_cap);
        let r = next_total <= min;

        debug!("SyncStageRunner::can_activate_now: family_id={family_id} active_count={active_count} next_total={next_total} max_parallel_jobs={max_parallel_jobs} min={min} r={r}");
        r
    }

    fn try_activate_now(family_id: JobId, active_count: usize, max_parallel_jobs: usize, active_caps: &BTreeMap<usize, usize>, new_cap: usize, exclusive_job_id: Option<ExclusiveOwner>, execution_waiter: &mut SyncExecutionWaiter) -> Option<SyncExecutionPermit> {
        if !Self::can_activate_now(family_id, active_count, max_parallel_jobs, active_caps, new_cap, exclusive_job_id) {
            return None;
        }
        execution_waiter.try_acquire()
    }

    fn current_group_cap(max_parallel_jobs: usize, active_caps: &BTreeMap<usize, usize>) -> usize {
        match active_caps.first_key_value() {
            Some((&smallest_active_cap, _)) => max_parallel_jobs.min(smallest_active_cap),
            None => max_parallel_jobs,
        }
    }

    fn add_active(active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize, cap: usize) {
        *active_caps.entry(cap).or_insert(0) += 1;
        *active_count += 1;
    }

    fn remove_active(active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize, cap: usize) {
        if let Some(count) = active_caps.get_mut(&cap) {
            *count -= 1;
            if *count == 0 {
                active_caps.remove(&cap);
            }
        }
        *active_count = active_count.saturating_sub(1);
    }

    fn flush_waiting_resumes(job: &mut LiveJob) {
        for tx in job.waiting_resumes.drain(..) {
            let _ = tx.send(());
        }
    }

    fn handle_suspend(jobs: &mut HashMap<JobId, LiveJob>, active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize, job_id: JobId) {
        let Some(job) = jobs.get_mut(&job_id) else {
            return;
        };

        job.suspend_depth += 1;

        if job.active {
            job.active = false;
            Self::remove_active(active_caps, active_count, job.cap);
            job.sync_execution_permit.take();
        }
    }

    fn handle_resume(jobs: &mut HashMap<JobId, LiveJob>, active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize, max_parallel_jobs: usize, exclusive_job_id: Option<ExclusiveOwner>, job_id: JobId, reply: oneshot::Sender<()>, activity_paused: bool, execution_waiter: &mut SyncExecutionWaiter) {
        let Some(job) = jobs.get_mut(&job_id) else {
            let _ = reply.send(());
            return;
        };

        // Every nested submit() that suspended this parent must wait until the
        // parent becomes active again before returning.
        job.waiting_resumes.push(reply);

        job.unwinding |= activity_paused;
        if job.unwinding {
            if job.suspend_depth > 0 {
                job.suspend_depth -= 1;
            }
            Self::mark_unwinding(job, active_caps, active_count);
            return;
        }

        if job.suspend_depth == 0 {
            if job.active {
                Self::flush_waiting_resumes(job);
            } else if let Some(sync_execution_permit) = Self::try_activate_now(job.family_id, *active_count, max_parallel_jobs, active_caps, job.cap, exclusive_job_id, execution_waiter) {
                job.active = true;
                job.sync_execution_permit = Some(sync_execution_permit);
                Self::add_active(active_caps, active_count, job.cap);
                Self::flush_waiting_resumes(job);
            }
            return;
        }

        job.suspend_depth -= 1;

        if job.suspend_depth == 0
            && let Some(sync_execution_permit) = Self::try_activate_now(job.family_id, *active_count, max_parallel_jobs, active_caps, job.cap, exclusive_job_id, execution_waiter)
        {
            job.active = true;
            job.sync_execution_permit = Some(sync_execution_permit);
            Self::add_active(active_caps, active_count, job.cap);
            Self::flush_waiting_resumes(job);
        }
    }

    fn force_resume_waiters_for_unwind(jobs: &mut HashMap<JobId, LiveJob>, active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize) {
        // Resumes queued before the pause may not receive a fresh Resume command. Flush
        // only fully-unsuspended jobs so parent stages that are already eligible to run
        // are not stranded behind a reset that may be waiting for them to drop state.
        for job in jobs.values_mut() {
            Self::mark_unwinding(job, active_caps, active_count);
        }
    }

    fn mark_unwinding(job: &mut LiveJob, active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize) {
        job.unwinding = true;
        job.sync_execution_permit.take();
        // Physical activity remains leased, but neither the shared budget nor
        // this healthy descendant's device capacity should block unrelated work.
        if job.active {
            job.active = false;
            Self::remove_active(active_caps, active_count, job.cap);
        }
        if job.suspend_depth == 0 {
            Self::flush_waiting_resumes(job);
        }
    }

    fn try_resume_waiters(jobs: &mut HashMap<JobId, LiveJob>, active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize, max_parallel_jobs: usize, exclusive_job_id: Option<ExclusiveOwner>, execution_waiter: &mut SyncExecutionWaiter) {
        for job in jobs.values_mut() {
            if job.active && !job.waiting_resumes.is_empty() {
                Self::flush_waiting_resumes(job);
            }
        }

        loop {
            let waiting_ids: Vec<JobId> = jobs.iter().filter_map(|(&id, job)| if !job.active && !job.unwinding && job.suspend_depth == 0 && !job.waiting_resumes.is_empty() { Some(id) } else { None }).collect();

            debug!("SyncStageRunner::try_resume_waiters: waiting_ids={waiting_ids:?}");

            let mut progressed = false;

            for job_id in waiting_ids {
                let (cap, family_id) = match jobs.get(&job_id) {
                    Some(job) if !job.active && job.suspend_depth == 0 && !job.waiting_resumes.is_empty() => (job.cap, job.family_id),
                    _ => continue,
                };

                if let Some(sync_execution_permit) = Self::try_activate_now(family_id, *active_count, max_parallel_jobs, active_caps, cap, exclusive_job_id, execution_waiter) {
                    let job = jobs.get_mut(&job_id).expect("job must exist");
                    job.active = true;
                    job.sync_execution_permit = Some(sync_execution_permit);
                    Self::add_active(active_caps, active_count, job.cap);
                    Self::flush_waiting_resumes(job);
                    progressed = true;
                }
            }

            if !progressed {
                break;
            }
        }
    }

    fn note_job_finished(jobs: &mut HashMap<JobId, LiveJob>, active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize, job_id: JobId) {
        if let Some(job) = jobs.remove(&job_id) {
            debug!("SyncStageRunner::note_job_finished: job={job:?}");

            if job.active {
                Self::remove_active(active_caps, active_count, job.cap);
            }
        } else {
            warn!("SyncStageRunner::note_job_finished: no {job_id}");
        }
    }

    fn drain_finished(running: &mut JoinSet<()>, done_rx: &mut mpsc::UnboundedReceiver<JobId>, jobs: &mut HashMap<JobId, LiveJob>, active_caps: &mut BTreeMap<usize, usize>, active_count: &mut usize) {
        while let Ok(job_id) = done_rx.try_recv() {
            Self::note_job_finished(jobs, active_caps, active_count, job_id);
        }

        while let Some(res) = running.try_join_next() {
            if let Err(err) = res {
                error!("SyncStageRunner::drain_finished: err={err:?}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{StreamExt, TryStreamExt};
    use std::{
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
        time::Duration,
    };
    use thoosa_shared::CudaAction;

    fn needs_restart_error() -> ThoError {
        ThoError::new_with_cause(
            "needs restart",
            // Use the constructor so injected CUDA errors capture their backtrace.
            thoosa_shared::tho_error::cuda_error(CudaAction::NeedsRestart, "reset me".into()),
        )
    }

    fn pending_test_job(job_id: JobId) -> (Job, oneshot::Receiver<Result<(), ThoError>>) {
        let (result_tx, result_rx) = oneshot::channel();
        let result_sender = Arc::new(JobResultSender::new(result_tx));
        let result_sender_for_failure = Arc::clone(&result_sender);
        let job = Job::new(
            job_id,
            usize::MAX,
            async move {
                result_sender.send(Ok(()));
                JobFeedback::Success
            },
            Box::new(move |error| result_sender_for_failure.send(Err(error))),
        );
        (job, result_rx)
    }

    #[cfg(test)]
    fn runner_with_limits(device: usize, limits: &Arc<FuturesLimits>) -> Arc<SyncStageRunner> {
        let recovery = (device != 0).then(|| CudaRecoveryCoordinator::new(Arc::new(CudaActivityGate::new(device)), Arc::new(|_| Box::pin(async { Ok(()) }))));
        Arc::new(SyncStageRunner::new_inner_with_buffer_limits(1, device, recovery, Arc::clone(limits)))
    }

    async fn manager_barrier(exec: &SyncStageRunner) {
        // Job zero is never allocated. Its acknowledgement fences earlier
        // Suspend commands without introducing a separate scheduler test API.
        let (reply, received) = oneshot::channel();
        exec.cmd_tx.send(ManagerCmd::Suspend { job_id: 0, reply }).unwrap();
        received.await.unwrap();
    }

    #[tokio::test]
    async fn test_exclusive_parent_awaits_spawned_collector_dependencies() {
        tokio::time::timeout(Duration::from_secs(5), async {
            // One shared slot forces suspension even when the child uses another device.
            for target in [1, 2] {
                let limits = Arc::new(FuturesLimits::new_with_parallelism(1, 1));
                let parent = runner_with_limits(1, &limits);
                let child = if target == 1 { Arc::clone(&parent) } else { runner_with_limits(target, &limits) };
                let child_run = Arc::clone(&child);
                let values = parent.submit(0, async move {
                    thoosa_shared::collect_futures_with_progress::collect_futures_with_progress(
                        (0..3).map(move |index| {
                            let child = Arc::clone(&child_run);
                            async move {
                                let value = child.submit(0, async move { Ok(index) }, None).await?;
                                // Post-submit work must finish while the collector's parent
                                // stays suspended, rather than each child trying to resume it.
                                tokio::task::yield_now().await;
                                Ok(value)
                            }
                        }), 3, thoosa_shared::FuturesOrdering::Ordered, thoosa_shared::CollectionErrorPolicy::Propagate, "scheduled descendants",
                    ).await
                }, None).await.unwrap();
                assert_eq!(values, vec![0, 1, 2]);
                if target == 1 { drop(child); } else { Arc::try_unwrap(child).unwrap().shutdown().await; }
                Arc::try_unwrap(parent).unwrap().shutdown().await;
            }
        }).await.expect("collector descendants must make progress with one scheduling slot");
    }

    #[tokio::test]
    async fn test_exclusive_retry_admits_descendants_but_keeps_unrelated_jobs_out() {
        tokio::time::timeout(Duration::from_secs(5), async {
            let limits = Arc::new(FuturesLimits::new_with_parallelism(1, 1));
            let exec = runner_with_limits(1, &limits);
            let retry = exec.retry_scope();
            assert!(exec.submit::<_, ()>(0, async { Err(ThoError::oom()) }, Some(retry.token())).await.unwrap_err().is_oom());
            let unrelated_ran = Arc::new(AtomicBool::new(false));
            let unrelated = tokio::spawn({
                let exec = Arc::clone(&exec);
                let ran = Arc::clone(&unrelated_ran);
                async move { exec.submit(0, async move { ran.store(true, Ordering::SeqCst); Ok(()) }, None).await }
            });
            let nested = Arc::clone(&exec);
            let ran = Arc::clone(&unrelated_ran);
            exec.submit(0, async move {
                // Direct child OOMs must neither replace nor clear the outer reservation.
                assert!(nested.submit::<_, ()>(0, async { Err(ThoError::oom()) }, None).await.unwrap_err().is_oom());
                let collector_runner = Arc::clone(&nested);
                let results = thoosa_shared::collect_futures_with_progress::collect_futures_with_progress(
                    (0..2).map(move |_| {
                        let exec = Arc::clone(&collector_runner);
                        async move { exec.submit(0, async { Ok(7) }, None).await }
                    }), 2, thoosa_shared::FuturesOrdering::Ordered, thoosa_shared::CollectionErrorPolicy::Propagate, "exclusive retry descendants",
                ).await?;
                assert_eq!(results, vec![7, 7]);
                assert!(!ran.load(Ordering::SeqCst), "unrelated work entered the reserved dependency tree");
                Err::<(), _>(ThoError::oom())
            }, Some(retry.token())).await.unwrap_err();
            assert_eq!(exec.exclusive_job_id.read().map(|owner| owner.job_id), Some(retry.token.inner.job_id));
            assert!(!unrelated_ran.load(Ordering::SeqCst));
            drop(retry);
            unrelated.await.unwrap().unwrap();
            Arc::try_unwrap(exec).unwrap().shutdown().await;
        }).await.expect("exclusive retries must admit their whole dependency tree");
    }

    #[tokio::test]
    async fn test_exclusive_parent_awaits_instance_and_cached_initializers() {
        tokio::time::timeout(Duration::from_secs(5), async {
            for device in [0, 1] {
                let limits = Arc::new(FuturesLimits::new_with_parallelism(1, 1));
                let exec = runner_with_limits(device, &limits);
                let nested = Arc::clone(&exec);
                exec.submit(0, async move {
                    let cache = crate::instance_cache::InstanceCache::new();
                    let child = Arc::clone(&nested);
                    let value = cache.get_or_try_insert_with::<(), _, _, _, _>(device, nested.cuda_gate().cloned(), 0, move || async move {
                        child.submit(0, async { Ok(Arc::new(17)) }, None).await
                    }).await?;
                    assert_eq!(*value, 17);
                    let gate = nested.cuda_gate().cloned();
                    let computation = async move { nested.submit(0, async { Ok(23) }, None).await }.boxed().shared();
                    let cached = crate::runner::common::cachable::CachedFuture::for_device(computation, gate);
                    assert_eq!(cached.await?, 23);
                    Ok(())
                }, None).await.unwrap();
                Arc::try_unwrap(exec).unwrap().shutdown().await;
            }
        }).await.expect("detached initializers must retain scheduling dependencies on CPU and CUDA");
    }

    #[tokio::test]
    async fn test_cancelled_dependency_wait_releases_only_its_parent_suspension() {
        tokio::time::timeout(Duration::from_secs(5), async {
            let limits = Arc::new(FuturesLimits::new_with_parallelism(1, 1));
            let exec = runner_with_limits(0, &limits);
            let nested = Arc::clone(&exec);
            exec.submit(0, async move {
                let cache = crate::instance_cache::InstanceCache::new();
                let (started, ready) = oneshot::channel();
                let (release, released) = oneshot::channel();
                let mut waiter = Box::pin(cache.get_or_try_insert_with::<(), _, _, _, _>(0, None, 0, move || async move {
                    started.send(()).unwrap();
                    released.await.unwrap();
                    Ok(Arc::new(17))
                }));
                tokio::select! { _ = ready => {}, _ = &mut waiter => panic!("initializer should still be pending") }
                drop(waiter);
                // The detached initializer remains alive, but its cancelled
                // waiter must not keep the parent suspended behind a new child.
                assert_eq!(nested.submit(0, async { Ok(1) }, None).await?, 1);
                release.send(()).unwrap();
                assert_eq!(*cache.get_or_try_insert_with::<(), _, _, _, _>(0, None, 0, || async { Ok(Arc::new(99)) }).await?, 17);

                let (started, ready) = oneshot::channel();
                let (release, released) = oneshot::channel();
                let computation = async move { started.send(()).unwrap(); released.await.unwrap(); Ok(23) }.boxed().shared();
                let cached = crate::runner::common::cachable::CachedFuture::for_device(computation, None);
                let mut waiter = Box::pin(cached.clone());
                tokio::select! { _ = ready => {}, _ = &mut waiter => panic!("cached computation should still be pending") }
                drop(waiter);
                assert_eq!(nested.submit(0, async { Ok(2) }, None).await?, 2);
                release.send(()).unwrap();
                assert_eq!(cached.await?, 23);
                Ok(())
            }, None).await.unwrap();
            Arc::try_unwrap(exec).unwrap().shutdown().await;
        }).await.expect("cancelled waits must not pin capacity behind detached computations");
    }

    #[tokio::test]
    async fn test_recovery_cancels_queued_cross_device_children() {
        tokio::time::timeout(Duration::from_secs(3), async {
            for target in [0, 2] {
                let limits = Arc::new(FuturesLimits::new_with_parallelism(2, 2));
                let parent_exec = runner_with_limits(1, &limits);
                let child_exec = runner_with_limits(target, &limits);
                let (started, ready) = oneshot::channel();
                let (release, released) = oneshot::channel();
                let blocker = tokio::spawn({
                    let exec = Arc::clone(&child_exec);
                    async move {
                        exec.submit(
                            0,
                            async move {
                                started.send(()).unwrap();
                                released.await.unwrap();
                                Ok(())
                            },
                            None,
                        )
                        .await
                    }
                });
                ready.await.unwrap();
                let (queued, queue_ready) = oneshot::channel();
                let child_ran = Arc::new(AtomicBool::new(false));
                let parent = tokio::spawn({
                    let exec = Arc::clone(&parent_exec);
                    let child_exec = Arc::clone(&child_exec);
                    let parent_manager = Arc::clone(&parent_exec);
                    let child_ran = Arc::clone(&child_ran);
                    async move {
                        exec.submit(
                            0,
                            async move {
                                let mut child = Box::pin(child_exec.submit(
                                    0,
                                    async move {
                                        child_ran.store(true, Ordering::SeqCst);
                                        Ok(())
                                    },
                                    None,
                                ));
                                assert!(futures::poll!(child.as_mut()).is_pending());
                                manager_barrier(&parent_manager).await;
                                assert!(futures::poll!(child.as_mut()).is_pending());
                                queued.send(()).unwrap();
                                child.await
                            },
                            None,
                        )
                        .await
                    }
                });
                queue_ready.await.unwrap();
                drop(child_exec.tx.reserve().await.unwrap());
                let ticket = parent_exec.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap().unwrap();
                assert!(matches!(parent.await.unwrap().unwrap_err().what(), ThoError::CudaBeingRecovered));
                // The healthy target's exclusive job is still physically running.
                // Only its queued dependent child should be discarded.
                parent_exec.cuda_gate().unwrap().wait_for_pause_to_finish(ticket).await.unwrap();
                assert!(!child_ran.load(Ordering::SeqCst));
                if let Some(gate) = child_exec.cuda_gate() {
                    assert_eq!(gate.device_generation(), CudaDeviceGeneration(0));
                }
                release.send(()).unwrap();
                blocker.await.unwrap().unwrap();
                Arc::try_unwrap(parent_exec).unwrap().shutdown().await;
                Arc::try_unwrap(child_exec).unwrap().shutdown().await;
            }
        })
        .await
        .expect("queued dependencies must not pin a recovering ancestor");
    }

    #[tokio::test]
    async fn test_running_cross_device_child_retains_ancestor_until_completion() {
        tokio::time::timeout(Duration::from_secs(3), async {
            let limits = Arc::new(FuturesLimits::new_with_parallelism(2, 2));
            let parent_exec = runner_with_limits(1, &limits);
            let child_exec = runner_with_limits(2, &limits);
            let (started, ready) = oneshot::channel();
            let (release, released) = oneshot::channel();
            let parent = tokio::spawn({
                let exec = Arc::clone(&parent_exec);
                let child_exec = Arc::clone(&child_exec);
                let ancestor = Arc::clone(parent_exec.cuda_gate().unwrap());
                async move {
                    exec.submit(
                        0,
                        async move {
                            child_exec
                                .submit(
                                    0,
                                    async move {
                                        assert_eq!(CudaActivityScope::current().unwrap().generation(&ancestor), Some(CudaDeviceGeneration(0)));
                                        started.send(()).unwrap();
                                        released.await.unwrap();
                                        // The caller has already unwound; the actual child still
                                        // owns the ancestor lease and can finish admitted work.
                                        CudaActivityScope::for_operation(&ancestor)?;
                                        Ok(())
                                    },
                                    None,
                                )
                                .await
                        },
                        None,
                    )
                    .await
                }
            });
            ready.await.unwrap();
            let ticket = parent_exec.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap().unwrap();
            assert!(matches!(parent.await.unwrap().unwrap_err().what(), ThoError::CudaBeingRecovered));
            let mut completion = Box::pin(parent_exec.cuda_gate().unwrap().wait_for_pause_to_finish(ticket));
            assert!(futures::poll!(completion.as_mut()).is_pending());
            // The descendant is still physically alive, but must not hold its
            // healthy device's exclusive slot while its ancestor is recovering.
            child_exec.submit(0, async { Ok(()) }, None).await.unwrap();
            release.send(()).unwrap();
            completion.await.unwrap();
            assert_eq!(child_exec.cuda_gate().unwrap().device_generation(), CudaDeviceGeneration(0));
            Arc::try_unwrap(parent_exec).unwrap().shutdown().await;
            Arc::try_unwrap(child_exec).unwrap().shutdown().await;
        })
        .await
        .expect("running descendants must drain physically without aborting");
    }

    #[tokio::test]
    async fn test_ancestor_recovery_releases_a_parent_already_waiting_to_resume() {
        tokio::time::timeout(Duration::from_secs(3), async {
            let limits = Arc::new(FuturesLimits::new_with_parallelism(2, 2));
            let healthy = runner_with_limits(2, &limits);
            let ancestor = Arc::new(CudaActivityGate::new(1));
            let activity = CudaActivityScope::for_operation(&ancestor).unwrap();
            let (suspended, suspend_ready) = oneshot::channel();
            let (resume, resume_ready) = oneshot::channel();
            let (queued, queue_ready) = oneshot::channel();
            let parent = tokio::spawn({
                let healthy = Arc::clone(&healthy);
                let manager = Arc::clone(&healthy);
                activity.run(async move {
                    healthy
                        .submit(
                            0,
                            async move {
                                let mut guard = ParentSuspendGuard::new(JobContext::current().job);
                                guard.maybe_suspend().await?;
                                suspended.send(()).unwrap();
                                resume_ready.await.unwrap();
                                let mut resuming = Box::pin(guard.resume());
                                assert!(futures::poll!(resuming.as_mut()).is_pending());
                                manager_barrier(&manager).await;
                                assert!(futures::poll!(resuming.as_mut()).is_pending());
                                queued.send(()).unwrap();
                                resuming.await
                            },
                            None,
                        )
                        .await
                })
            });
            suspend_ready.await.unwrap();
            let (started, ready) = oneshot::channel();
            let (release, released) = oneshot::channel();
            let blocker = tokio::spawn({
                let healthy = Arc::clone(&healthy);
                async move {
                    healthy
                        .submit(
                            0,
                            async move {
                                started.send(()).unwrap();
                                released.await.unwrap();
                                Ok(())
                            },
                            None,
                        )
                        .await
                }
            });
            ready.await.unwrap();
            resume.send(()).unwrap();
            queue_ready.await.unwrap();
            let ticket = ancestor.request_pause(ThoError::CudaBeingRecovered).unwrap();
            // The submitter may return as soon as the ancestor closes admission;
            // idle also proves that the physical job escaped its pending Resume.
            assert!(matches!(parent.await.unwrap().unwrap_err().what(), ThoError::CudaBeingRecovered));
            ancestor.wait_until_idle().await;
            ancestor.finish_pause(ticket);
            release.send(()).unwrap();
            blocker.await.unwrap().unwrap();
            Arc::try_unwrap(healthy).unwrap().shutdown().await;
        })
        .await
        .expect("ancestor recovery must wake already-queued resumes on a healthy device");
    }

    #[tokio::test]
    async fn test_recovery_releases_shared_capacity_for_healthy_devices() {
        tokio::time::timeout(Duration::from_secs(3), async {
            let limits = Arc::new(FuturesLimits::new_with_parallelism(2, 2));
            let recovering = runner_with_limits(1, &limits);
            let healthy = runner_with_limits(2, &limits);
            let cpu = runner_with_limits(0, &limits);
            let (started, mut ready) = mpsc::unbounded_channel();
            let mut releases = Vec::new();
            let mut callers = Vec::new();
            for _ in 0..2 {
                let (release, released) = oneshot::channel();
                releases.push(release);
                let started = started.clone();
                let exec = Arc::clone(&recovering);
                callers.push(tokio::spawn(async move {
                    exec.submit(
                        usize::MAX,
                        async move {
                            started.send(()).unwrap();
                            released.await.unwrap();
                            Ok(())
                        },
                        None,
                    )
                    .await
                }));
            }
            ready.recv().await.unwrap();
            ready.recv().await.unwrap();
            assert!(limits.try_acquire_sync_execution().is_none());
            let ticket = recovering.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap().unwrap();
            healthy.submit(0, async { Ok(()) }, None).await.unwrap();
            cpu.submit(0, async { Ok(()) }, None).await.unwrap();
            assert!(recovering.cuda_gate().unwrap().is_recovering());
            for release in releases {
                release.send(()).unwrap();
            }
            for caller in callers {
                caller.await.unwrap().unwrap();
            }
            recovering.cuda_gate().unwrap().wait_for_pause_to_finish(ticket).await.unwrap();
            Arc::try_unwrap(recovering).unwrap().shutdown().await;
            Arc::try_unwrap(healthy).unwrap().shutdown().await;
            Arc::try_unwrap(cpu).unwrap().shutdown().await;
        })
        .await
        .expect("recovery must not consume healthy devices' execution budget");
    }

    #[tokio::test]
    async fn test_restrictive_queue_head_is_not_bypassed_by_permissive_jobs() {
        tokio::time::timeout(Duration::from_secs(3), async {
            let limits = Arc::new(FuturesLimits::new_with_parallelism(4, 4));
            let exec = runner_with_limits(0, &limits);
            let (started, mut starts) = mpsc::unbounded_channel();
            let mut releases = Vec::new();
            for (id, cap) in [(1, usize::MAX), (2, usize::MAX), (3, usize::MAX), (4, 1), (5, usize::MAX)] {
                let (release, released) = oneshot::channel();
                releases.push(release);
                let started = started.clone();
                exec.tx
                    .send(Job::new(
                        id,
                        cap,
                        async move {
                            started.send(id).unwrap();
                            released.await.unwrap();
                            JobFeedback::Neutral
                        },
                        Box::new(|error| panic!("unexpected purge: {error}")),
                    ))
                    .await
                    .unwrap();
                // Capacity one makes this a deterministic acknowledgement that the
                // manager received each job while the original group remains active.
                drop(exec.tx.reserve().await.unwrap());
                if id <= 3 {
                    assert_eq!(starts.recv().await, Some(id));
                }
            }
            releases.remove(0).send(()).unwrap();
            releases.remove(0).send(()).unwrap();
            assert_eq!(starts.recv().await, Some(4), "a cap of two needs the same fairness barrier as an exclusive head");
            for release in releases {
                release.send(()).unwrap();
            }
            assert_eq!(starts.recv().await, Some(5));
            Arc::try_unwrap(exec).unwrap().shutdown().await;
        })
        .await
        .expect("restrictive jobs must eventually obtain their requested concurrency");
    }

    #[tokio::test]
    async fn test_consume_failures_reach_oom_and_recovery_classification() {
        use thoosa_common::runner::sync_stage::ConsumeResult;
        let exec = SyncStageRunner::new_with_reset_hook(8, 1, Arc::new(|_| Box::pin(async { Ok(()) })));
        let scope = exec.retry_scope();
        let result = exec.submit_consume(0, async { Ok(ConsumeResult::<_, ()>::Failed { input: vec![7], error: ThoError::oom() }) }, Some(scope.token())).await.unwrap();
        let ConsumeResult::Failed { input, error } = result else { panic!("missing OOM") };
        assert_eq!(input, vec![7]);
        assert!(error.is_oom());
        assert_eq!(exec.exclusive_job_id.read().map(|owner| owner.job_id), Some(scope.token.inner.job_id));
        let generation = exec.cuda_gate().unwrap().device_generation();
        let result = exec.submit_consume(0, async move { Ok(ConsumeResult::<_, ()>::Failed { input, error: needs_restart_error() }) }, Some(scope.token())).await.unwrap();
        let ConsumeResult::Failed { input, error } = result else { panic!("missing fatal failure") };
        assert_eq!(input, vec![7]);
        assert!(matches!(error.what(), ThoError::CudaBeingRecovered));
        tokio::time::timeout(Duration::from_secs(2), exec.cuda_gate().unwrap().wait_if_paused()).await.unwrap().unwrap();
        assert_eq!(exec.cuda_gate().unwrap().device_generation().0, generation.0 + 1);
        // An old retry chain must not run even after admission has reopened.
        assert!(matches!(
            exec.submit(
                0,
                async {
                    panic!("stale retry ran");
                    #[allow(unreachable_code)]
                    Ok(())
                },
                Some(scope.token())
            )
            .await
            .unwrap_err()
            .what(),
            ThoError::CudaBeingRecovered
        ));
        assert_eq!(*exec.exclusive_job_id.read(), None);
        drop(scope);
        exec.shutdown().await;
    }

    #[tokio::test]
    async fn test_cancelled_oom_owner_cannot_reserve_after_completion() {
        // Exercise both the first attempt and a retry whose reservation already exists.
        for retrying in [false, true] {
            let exec = Arc::new(SyncStageRunner::new(8));
            let scope = exec.retry_scope();
            if retrying {
                assert!(exec.submit(0, async { Err::<(), _>(ThoError::oom()) }, Some(scope.token())).await.unwrap_err().is_oom());
            }
            let (started_tx, started_rx) = oneshot::channel();
            let (release_tx, release_rx) = oneshot::channel();
            let caller = tokio::spawn({
                let exec = Arc::clone(&exec);
                async move {
                    let scope = scope;
                    exec.submit(
                        0,
                        async move {
                            started_tx.send(()).unwrap();
                            release_rx.await.unwrap();
                            Err::<(), _>(ThoError::oom())
                        },
                        Some(scope.token()),
                    )
                    .await
                }
            });
            started_rx.await.unwrap();
            caller.abort();
            assert!(caller.await.unwrap_err().is_cancelled());
            release_tx.send(()).unwrap();
            // The exclusive running job must finish before this job can start.
            tokio::time::timeout(Duration::from_secs(2), exec.submit(0, async { Ok(()) }, None)).await.unwrap().unwrap();
            assert_eq!(*exec.exclusive_job_id.read(), None);
            Arc::try_unwrap(exec).unwrap().shutdown().await;
        }
    }

    #[tokio::test]
    async fn test_cancelled_submit_with_unread_oom_releases_reservation() {
        let exec = SyncStageRunner::new(8);
        let (finished_tx, finished_rx) = oneshot::channel();
        let mut caller = Box::pin(async {
            let scope = exec.retry_scope();
            exec.submit(
                0,
                async move {
                    finished_tx.send(()).unwrap();
                    Err::<(), _>(ThoError::oom())
                },
                Some(scope.token()),
            )
            .await
        });
        assert!(futures::poll!(caller.as_mut()).is_pending());
        // On the current-thread runtime the job sends its OOM result before this
        // receiver is polled again; the submission itself remains unpolled.
        finished_rx.await.unwrap();
        assert!(exec.exclusive_job_id.read().is_some());
        drop(caller);
        assert_eq!(*exec.exclusive_job_id.read(), None);
        tokio::time::timeout(Duration::from_secs(2), exec.submit(0, async { Ok(()) }, None)).await.unwrap().unwrap();
        exec.shutdown().await;
    }

    #[tokio::test]
    async fn test_retry_reservation_survives_adjustment_and_releases_on_exit() {
        for succeed in [false, true] {
            let exec = SyncStageRunner::new(8);
            let scope = exec.retry_scope();
            let job_id = scope.token.inner.job_id;
            assert!(exec.submit(0, async { Err::<(), _>(ThoError::oom()) }, Some(scope.token())).await.unwrap_err().is_oom());
            assert_eq!(exec.exclusive_job_id.read().map(|owner| owner.job_id), Some(job_id));
            let mut other = Box::pin(exec.submit(0, async { Ok(42) }, None));
            assert!(futures::poll!(other.as_mut()).is_pending());
            // A further OOM keeps the same reservation through the next adjustment.
            assert!(exec.submit(0, async { Err::<(), _>(ThoError::oom()) }, Some(scope.token())).await.unwrap_err().is_oom());
            assert_eq!(exec.exclusive_job_id.read().map(|owner| owner.job_id), Some(job_id));
            assert!(futures::poll!(other.as_mut()).is_pending());
            if succeed {
                exec.submit(0, async { Ok(()) }, Some(scope.token())).await.unwrap();
            }
            drop(scope); // Cancellation/terminal error during an adjustment also releases it.
            assert_eq!(tokio::time::timeout(Duration::from_secs(2), other).await.unwrap().unwrap(), 42);
            exec.shutdown().await;
        }
    }

    #[tokio::test]
    async fn test_cancel_before_suspend_ack_allows_next_nested_submit() {
        let exec = Arc::new(SyncStageRunner::new(8));
        let parent = exec.submit(
            0,
            {
                let exec = Arc::clone(&exec);
                async move {
                    let mut cancelled = Box::pin(exec.submit(0, async { Ok(()) }, None));
                    assert!(futures::poll!(cancelled.as_mut()).is_pending());
                    drop(cancelled);
                    exec.submit(0, async { Ok(42) }, None).await
                }
            },
            None,
        );
        assert_eq!(tokio::time::timeout(Duration::from_secs(2), parent).await.unwrap().unwrap(), 42);
        Arc::try_unwrap(exec).unwrap().shutdown().await;
    }

    #[tokio::test]
    async fn test_cancel_before_resume_ack_does_not_resume_twice() {
        let (cmd_tx, mut cmd_rx) = mpsc::unbounded_channel();
        let parent = CurrentJob { job_id: 1, cmd_tx };
        let mut first = ParentSuspendGuard::new(Some(parent.clone()));
        let mut second = ParentSuspendGuard::new(Some(parent));
        for guard in [&mut first, &mut second] {
            let mut suspend = Box::pin(guard.maybe_suspend());
            assert!(futures::poll!(suspend.as_mut()).is_pending());
            let ManagerCmd::Suspend { reply, .. } = cmd_rx.try_recv().unwrap() else { panic!("expected Suspend") };
            reply.send(()).unwrap();
            suspend.await.unwrap();
        }
        let mut resume = Box::pin(first.resume());
        assert!(futures::poll!(resume.as_mut()).is_pending());
        let ManagerCmd::Resume { reply, .. } = cmd_rx.try_recv().unwrap() else { panic!("expected Resume") };
        // Keep the acknowledgement pending, as when another suspension still exists.
        drop(resume);
        drop(first);
        assert!(matches!(cmd_rx.try_recv(), Err(mpsc::error::TryRecvError::Empty)));
        drop(second);
        assert!(matches!(cmd_rx.try_recv(), Ok(ManagerCmd::Resume { .. })));
        assert!(cmd_rx.try_recv().is_err());
        drop(reply);
    }

    #[tokio::test]
    async fn test_reset_drains_nested_collectors_in_running_stage() {
        for ordering in [thoosa_shared::FuturesOrdering::Ordered, thoosa_shared::FuturesOrdering::Unordered] {
            let completed = Arc::new(AtomicBool::new(false));
            let reset_ran = Arc::new(AtomicBool::new(false));
            let exec = Arc::new(SyncStageRunner::new_with_reset_hook(
                8,
                1,
                Arc::new({
                    let completed = Arc::clone(&completed);
                    let reset_ran = Arc::clone(&reset_ran);
                    move |_| {
                        assert!(completed.load(Ordering::SeqCst), "reset must wait for all copy children");
                        reset_ran.store(true, Ordering::SeqCst);
                        Box::pin(async { Ok(()) })
                    }
                }),
            ));
            let (started_tx, started_rx) = oneshot::channel();
            let (collect_tx, collect_rx) = oneshot::channel();
            let caller = tokio::spawn({
                let exec = Arc::clone(&exec);
                let nested_exec = Arc::clone(&exec);
                let gate = Arc::clone(exec.cuda_gate().unwrap());
                async move {
                    exec.submit(
                        0,
                        async move {
                            started_tx.send(()).unwrap();
                            collect_rx.await.unwrap();
                            let results = thoosa_shared::collect_futures_with_progress(
                                std::iter::once(async move {
                                    assert!(gate.is_paused());
                                    // This child has no CURRENT_JOB task local. It must still
                                    // reject a nested submission rather than wait for reset.
                                    assert!(nested_exec.submit(0, async { Ok(()) }, None).await.is_err());
                                    let values = thoosa_shared::collect_futures_with_progress(
                                        (0..3).map(|i| {
                                            let gate = Arc::clone(&gate);
                                            async move {
                                                let _copy = CudaActivityScope::for_operation(&gate)?;
                                                Ok::<_, ThoError>(i)
                                            }
                                        }),
                                        3,
                                        ordering,
                                        thoosa_shared::CollectionErrorPolicy::Propagate,
                                        "nested copy chunks",
                                    )
                                    .await?;
                                    assert_eq!(values.len(), 3);
                                    Ok::<_, ThoError>(values)
                                }),
                                1,
                                ordering,
                                thoosa_shared::CollectionErrorPolicy::Propagate,
                                "copy chunks",
                            )
                            .await?;
                            assert_eq!(results.len(), 1);
                            completed.store(true, Ordering::SeqCst);
                            Ok(())
                        },
                        None,
                    )
                    .await
                }
            });
            started_rx.await.unwrap();
            exec.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap();
            collect_tx.send(()).unwrap();
            tokio::time::timeout(Duration::from_secs(2), caller).await.unwrap().unwrap().unwrap();
            tokio::time::timeout(Duration::from_secs(2), exec.cuda_gate().unwrap().wait_if_paused()).await.unwrap().unwrap();
            assert!(reset_ran.load(Ordering::SeqCst));
            exec.submit(0, async { Ok(()) }, None).await.unwrap();
            Arc::try_unwrap(exec).unwrap().shutdown().await;
        }
    }

    #[tokio::test]
    async fn test_reset_waits_for_activity_outside_executor() {
        let reset_ran = Arc::new(AtomicBool::new(false));
        let gate = Arc::new(CudaActivityGate::new(1));
        let recovery = CudaRecoveryCoordinator::new(
            Arc::clone(&gate),
            Arc::new({
                let reset_ran = Arc::clone(&reset_ran);
                let gate = Arc::clone(&gate);
                move |_| {
                    let gate = Arc::clone(&gate);
                    reset_ran.store(true, Ordering::SeqCst);
                    Box::pin(async move {
                        // Cleanup uses only the gate owned by this reset.
                        let _cleanup = CudaActivityScope::for_operation(&gate)?;
                        Ok(())
                    })
                }
            }),
        );
        let exec = SyncStageRunner::new_with_recovery(8, recovery);
        let activity = CudaActivityScope::for_operation(&gate).unwrap();
        let descendant = activity.clone();
        exec.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap();
        assert!(CudaActivityScope::for_operation(&gate).is_err());
        drop(activity);
        let mut idle = Box::pin(gate.wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending());
        assert!(!reset_ran.load(Ordering::SeqCst));
        drop(descendant);
        tokio::time::timeout(Duration::from_secs(2), idle).await.unwrap();
        tokio::time::timeout(Duration::from_secs(2), gate.wait_if_paused()).await.unwrap().unwrap();
        assert!(reset_ran.load(Ordering::SeqCst));
        exec.submit(0, async { Ok(()) }, None).await.unwrap();
        exec.shutdown().await;
    }

    #[tokio::test]
    async fn test_purge_pending_jobs_sends_context_reset_to_all_results() {
        let (queued_tx, mut queued_rx) = mpsc::channel(2);
        let (pending_job, pending_result_rx) = pending_test_job(1);
        let (channel_job, channel_result_rx) = pending_test_job(2);
        let mut pending = VecDeque::from([pending_job]);
        queued_tx.send(channel_job).await.expect("channel must be open");
        drop(queued_tx);
        let mut input_closed = false;

        SyncStageRunner::purge_pending_jobs(&mut pending, &mut queued_rx, &mut input_closed, SyncStageRunner::context_reset_error());

        assert!(pending.is_empty());
        assert!(input_closed);
        for result_rx in [pending_result_rx, channel_result_rx] {
            let result = result_rx.await.expect("purged job must send a result");
            assert!(matches!(result.as_ref().err().map(ThoError::what), Some(ThoError::CudaBeingRecovered)), "unexpected reset result: {result:?}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_pending_job_is_purged_before_reset() -> Result<(), ThoError> {
        let exec = Arc::new(SyncStageRunner::new_with_reset_hook(8, 1, Arc::new(|_| Box::pin(async { Ok(()) }))));
        let pending_ran = Arc::new(AtomicBool::new(false));
        let (running_started_tx, running_started_rx) = oneshot::channel();
        let (release_running_tx, release_running_rx) = oneshot::channel();

        let running = {
            let exec = Arc::clone(&exec);
            tokio::spawn(async move {
                exec.submit(
                    0,
                    async move {
                        let _ = running_started_tx.send(());
                        let _ = release_running_rx.await;
                        Ok::<_, ThoError>(())
                    },
                    None,
                )
                .await
            })
        };
        running_started_rx.await.expect("exclusive job must start");

        let pending = {
            let exec = Arc::clone(&exec);
            let pending_ran = Arc::clone(&pending_ran);
            tokio::spawn(async move {
                exec.submit(
                    usize::MAX,
                    async move {
                        pending_ran.store(true, Ordering::SeqCst);
                        Ok::<_, ThoError>(())
                    },
                    None,
                )
                .await
            })
        };
        tokio::time::sleep(Duration::from_millis(50)).await;

        exec.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap();

        let pending_result = tokio::time::timeout(Duration::from_secs(2), pending).await.expect("pending job must be purged")?;
        assert!(matches!(pending_result.as_ref().err().map(ThoError::what), Some(ThoError::CudaBeingRecovered)), "unexpected reset result: {pending_result:?}");
        assert!(!pending_ran.load(Ordering::SeqCst));

        let _ = release_running_tx.send(());
        running.await??;
        tokio::time::timeout(Duration::from_secs(2), exec.cuda_gate().unwrap().wait_if_paused()).await.expect("device recovery must complete").unwrap();

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_reset_waits_for_running_jobs() -> Result<(), ThoError> {
        let reset_calls = Arc::new(AtomicUsize::new(0));
        let exec = Arc::new(SyncStageRunner::new_with_reset_hook(64, 1, {
            let reset_calls = Arc::clone(&reset_calls);
            Arc::new(move |_device_idx| {
                let reset_calls = Arc::clone(&reset_calls);
                async move {
                    reset_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
                .boxed()
            })
        }));

        let (long_started_tx1, long_started_rx1) = oneshot::channel();
        let (long_started_tx2, long_started_rx2) = oneshot::channel();
        let (release_tx1, release_rx1) = oneshot::channel();
        let (release_tx2, release_rx2) = oneshot::channel();

        let long1 = {
            let exec = Arc::clone(&exec);
            tokio::spawn(async move {
                exec.submit(
                    usize::MAX,
                    async move {
                        let _ = long_started_tx1.send(());
                        let _ = release_rx1.await;
                        Ok::<_, ThoError>(())
                    },
                    None,
                )
                .await
            })
        };

        let long2 = {
            let exec = Arc::clone(&exec);
            tokio::spawn(async move {
                exec.submit(
                    usize::MAX,
                    async move {
                        let _ = long_started_tx2.send(());
                        let _ = release_rx2.await;
                        Ok::<_, ThoError>(())
                    },
                    None,
                )
                .await
            })
        };

        let _ = long_started_rx1.await;
        let _ = long_started_rx2.await;

        let reset_err: Result<(), ThoError> = exec.submit(usize::MAX, async move { Err(needs_restart_error()) }, None).await;

        assert!(matches!(reset_err.as_ref().err().map(ThoError::what), Some(ThoError::CudaBeingRecovered)), "unexpected reset result: {reset_err:?}");

        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(reset_calls.load(Ordering::SeqCst), 0);

        let _ = release_tx1.send(());
        long1.await??;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(reset_calls.load(Ordering::SeqCst), 0);

        let _ = release_tx2.send(());
        long2.await??;

        tokio::time::timeout(Duration::from_secs(2), async {
            while reset_calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("reset hook should start after all running jobs finish");
        assert_eq!(reset_calls.load(Ordering::SeqCst), 1);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_running_reset_failures_are_coalesced() -> Result<(), ThoError> {
        let reset_calls = Arc::new(AtomicUsize::new(0));
        let finished_jobs = Arc::new(AtomicUsize::new(0));
        let exec = Arc::new(SyncStageRunner::new_with_reset_hook(64, 1, {
            let reset_calls = Arc::clone(&reset_calls);
            let finished_jobs = Arc::clone(&finished_jobs);
            Arc::new(move |_device_idx| {
                let reset_calls = Arc::clone(&reset_calls);
                let finished_jobs = Arc::clone(&finished_jobs);
                async move {
                    assert_eq!(finished_jobs.load(Ordering::SeqCst), 2);
                    reset_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
                .boxed()
            })
        }));

        let (started_tx1, started_rx1) = oneshot::channel();
        let (started_tx2, started_rx2) = oneshot::channel();
        let (release_tx1, release_rx1) = oneshot::channel();
        let (release_tx2, release_rx2) = oneshot::channel();

        let job1 = {
            let exec = Arc::clone(&exec);
            let finished_jobs = Arc::clone(&finished_jobs);
            tokio::spawn(async move {
                exec.submit(
                    usize::MAX,
                    async move {
                        let _ = started_tx1.send(());
                        let _ = release_rx1.await;
                        finished_jobs.fetch_add(1, Ordering::SeqCst);
                        Err::<(), _>(needs_restart_error())
                    },
                    None,
                )
                .await
            })
        };

        let job2 = {
            let exec = Arc::clone(&exec);
            let finished_jobs = Arc::clone(&finished_jobs);
            tokio::spawn(async move {
                exec.submit(
                    usize::MAX,
                    async move {
                        let _ = started_tx2.send(());
                        let _ = release_rx2.await;
                        finished_jobs.fetch_add(1, Ordering::SeqCst);
                        Err::<(), _>(needs_restart_error())
                    },
                    None,
                )
                .await
            })
        };

        let _ = started_rx1.await;
        let _ = started_rx2.await;
        let _ = release_tx1.send(());
        let _ = release_tx2.send(());

        for result in [job1.await?, job2.await?] {
            assert!(matches!(result.as_ref().err().map(ThoError::what), Some(ThoError::CudaBeingRecovered)), "unexpected reset result: {result:?}");
        }

        tokio::time::timeout(Duration::from_secs(2), async {
            while reset_calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("one reset should run after both failing jobs finish");
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(reset_calls.load(Ordering::SeqCst), 1);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_nested_needs_reset_unwinds_parent_before_reset() -> Result<(), ThoError> {
        let reset_calls = Arc::new(AtomicUsize::new(0));
        let parent_exited = Arc::new(AtomicBool::new(false));

        let exec = Arc::new(SyncStageRunner::new_with_reset_hook(64, 1, {
            let reset_calls = Arc::clone(&reset_calls);
            let parent_exited = Arc::clone(&parent_exited);
            Arc::new(move |_device_idx| {
                let reset_calls = Arc::clone(&reset_calls);
                let parent_exited = Arc::clone(&parent_exited);
                async move {
                    debug!("test_nested_needs_reset_unwinds_parent_before_reset: reset invoked");
                    assert!(parent_exited.load(Ordering::SeqCst), "parent must unwind before reset starts");
                    reset_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
                .boxed()
            })
        }));

        let result: Result<(), ThoError> = {
            let exec_for_child = Arc::clone(&exec);
            let parent_exited = Arc::clone(&parent_exited);
            let reset_calls = Arc::clone(&reset_calls);
            tokio::time::timeout(
                Duration::from_secs(2),
                exec.submit(
                    0,
                    async move {
                        debug!("test_nested_needs_reset_unwinds_parent_before_reset: Parent job running");
                        let child_result: Result<(), ThoError> = exec_for_child
                            .submit(
                                usize::MAX,
                                async move {
                                    debug!("test_nested_needs_reset_unwinds_parent_before_reset: Child job running");

                                    if reset_calls.load(Ordering::Relaxed) == 0 {
                                        debug!("test_nested_needs_reset_unwinds_parent_before_reset: Child job failed");
                                        Err(needs_restart_error())
                                    } else {
                                        debug!(":test_nested_needs_reset_unwinds_parent_before_reset: Child job done");
                                        Ok(())
                                    }
                                },
                                None,
                            )
                            .await;

                        parent_exited.store(true, Ordering::SeqCst);

                        debug!("test_nested_needs_reset_unwinds_parent_before_reset: Parent job done");

                        child_result
                    },
                    None,
                ),
            )
            .await
            .expect("parent should resume and return reset error")
        };

        assert!(matches!(result.as_ref().err().map(ThoError::what), Some(ThoError::CudaBeingRecovered)), "unexpected reset result: {result:?}");

        tokio::time::timeout(Duration::from_secs(2), async {
            while reset_calls.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("reset should eventually run");

        assert_eq!(1, reset_calls.load(Ordering::Relaxed));
        assert!(parent_exited.load(Ordering::SeqCst), "parent exited");

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_external_reset_does_not_deadlock_suspended_parent() -> Result<(), ThoError> {
        let reset_calls = Arc::new(AtomicUsize::new(0));
        let parent_exited = Arc::new(AtomicBool::new(false));

        let exec = Arc::new(SyncStageRunner::new_with_reset_hook(64, 1, {
            let reset_calls = Arc::clone(&reset_calls);
            let parent_exited = Arc::clone(&parent_exited);
            Arc::new(move |_device_idx| {
                let reset_calls = Arc::clone(&reset_calls);
                let parent_exited = Arc::clone(&parent_exited);
                async move {
                    while !parent_exited.load(Ordering::SeqCst) {
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                    reset_calls.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }
                .boxed()
            })
        }));

        let (child_started_tx, child_started_rx) = oneshot::channel();
        let (release_child_tx, release_child_rx) = oneshot::channel();
        let parent = {
            let exec_for_parent = Arc::clone(&exec);
            let parent_exited = Arc::clone(&parent_exited);
            tokio::spawn(async move {
                let exec_for_child = Arc::clone(&exec_for_parent);
                exec_for_parent
                    .submit(
                        0,
                        async move {
                            let child_result: Result<(), ThoError> = exec_for_child
                                .submit(
                                    usize::MAX,
                                    async move {
                                        let _ = child_started_tx.send(());
                                        let _ = release_child_rx.await;
                                        Ok(())
                                    },
                                    None,
                                )
                                .await;
                            parent_exited.store(true, Ordering::SeqCst);
                            child_result
                        },
                        None,
                    )
                    .await
            })
        };

        let _ = child_started_rx.await;
        exec.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap();
        let _ = release_child_tx.send(());

        let result = tokio::time::timeout(Duration::from_secs(2), parent).await.expect("parent task should not deadlock")?;
        assert!(matches!(result.unwrap_err().what(), ThoError::CudaBeingRecovered));

        tokio::time::timeout(Duration::from_secs(2), async {
            while reset_calls.load(Ordering::SeqCst) == 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("reset should finish after parent unwinds");

        assert_eq!(1, reset_calls.load(Ordering::Relaxed));

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test]
    async fn test_retired_generation_report_does_not_close_accepting() {
        let exec = SyncStageRunner::new_with_reset_hook(8, 1, Arc::new(|_| Box::pin(async { Ok(()) })));
        let generation = exec.cuda_gate().unwrap().device_generation();
        let ticket = exec.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap().unwrap();
        exec.cuda_gate().unwrap().wait_for_pause_to_finish(ticket).await.unwrap();
        assert!(exec.request_cuda_reset_for_generation(generation, needs_restart_error()).unwrap().is_none());
        assert!(!exec.cuda_gate().unwrap().is_paused());
        exec.shutdown().await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_nested_exclusive_oom_does_not_deadlock_parent_resume() -> Result<(), ThoError> {
        let exec = Arc::new(SyncStageRunner::new(64));

        let result = tokio::time::timeout(Duration::from_secs(2), {
            let exec_for_parent = Arc::clone(&exec);
            exec.submit(
                0,
                async move {
                    let child_result: Result<(), ThoError> = exec_for_parent.submit(0, async move { Err(ThoError::oom()) }, None).await;

                    match child_result {
                        Err(e) => match e.what() {
                            ThoError::SyncStageRunner(SyncStageRunnerError::JobError { job_id, inner }) => {
                                assert!(inner.is_oom());
                                assert!(*job_id > 0);
                                Ok::<_, ThoError>(())
                            }
                            other => panic!("unexpected error cause: {other:?}"),
                        },
                        Ok(_) => panic!("child should fail"),
                    }
                },
                None,
            )
        })
        .await
        .expect("parent resume must not deadlock")?;

        assert_eq!(result, ());

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_max_other_parallel_is_observed() -> Result<(), ThoError> {
        let exec = Arc::new(SyncStageRunner::new(64));
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let results = futures_util::stream::iter(0..8)
            .map(|_| {
                let exec = Arc::clone(&exec);
                let active = Arc::clone(&active);
                let max_seen = Arc::clone(&max_seen);
                async move {
                    exec.submit(
                        0,
                        async move {
                            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                            max_seen.fetch_max(now, Ordering::SeqCst);
                            tokio::time::sleep(Duration::from_millis(30)).await;
                            active.fetch_sub(1, Ordering::SeqCst);
                            Ok::<_, ThoError>(())
                        },
                        None,
                    )
                    .await
                }
            })
            .buffer_unordered(8)
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(results.len(), 8);
        assert!(max_seen.load(Ordering::SeqCst) <= 1);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test]
    async fn test_reset_without_waiters_reopens_admission() {
        let reset_invoked = Arc::new(AtomicBool::new(false));
        let gate = Arc::new(CudaActivityGate::new(1));
        let recovery = CudaRecoveryCoordinator::new(
            Arc::clone(&gate),
            Arc::new({
                let gate = Arc::clone(&gate);
                let reset_invoked = Arc::clone(&reset_invoked);
                move |_| {
                    let reset_invoked = Arc::clone(&reset_invoked);
                    let gate = Arc::clone(&gate);
                    Box::pin(async move {
                        assert!(gate.is_paused(), "admission must remain closed throughout reset");
                        reset_invoked.store(true, Ordering::SeqCst);
                        Ok(())
                    })
                }
            }),
        );
        let exec = SyncStageRunner::new_with_recovery(8, recovery);
        exec.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap();
        // No subscriber owns admission state; completion persists in the device gate.
        tokio::time::timeout(Duration::from_secs(2), exec.wait_until_accepting(None)).await.unwrap().unwrap();
        assert!(reset_invoked.load(Ordering::SeqCst));
        assert!(!exec.cuda_gate().unwrap().is_paused());
        assert_eq!(exec.submit(0, async { Ok(42) }, None).await.unwrap(), 42);
        exec.shutdown().await;
    }

    #[tokio::test]
    async fn test_failed_recovery_is_confined_to_its_device_and_allows_shutdown() {
        tokio::time::timeout(Duration::from_secs(2), async {
            let started = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let owner = SyncStageRunner::new_with_reset_hook(
                8,
                1,
                Arc::new({
                    let started = Arc::clone(&started);
                    let release = Arc::clone(&release);
                    move |_| {
                        let started = Arc::clone(&started);
                        let release = Arc::clone(&release);
                        Box::pin(async move {
                            started.notify_one();
                            release.notified().await;
                            Err(ThoError::oom().context("restart failed"))
                        })
                    }
                }),
            );
            let other = SyncStageRunner::new_with_reset_hook(8, 2, Arc::new(|_| Box::pin(async { Ok(()) })));
            let cpu = SyncStageRunner::new_cpu(8);
            assert!(cpu.cuda_gate().is_none());
            assert!(cpu.request_cuda_reset(ThoError::CudaBeingRecovered).is_err());
            let generation = owner.cuda_gate().unwrap().device_generation();
            owner.request_cuda_reset(needs_restart_error()).unwrap();
            started.notified().await;
            let mut submit = Box::pin(owner.submit(0, async { Ok(()) }, None));
            assert!(futures::poll!(submit.as_mut()).is_pending());
            other.submit(0, async { Ok(()) }, None).await.unwrap();
            cpu.submit(0, async { Ok(()) }, None).await.unwrap();
            release.notify_one();
            let error = submit.await.unwrap_err();
            assert!(error.is_cuda_recovery_failed());
            assert!(!error.is_retryable_oom() && !error.needs_cuda_recovery());
            let ThoError::CudaRecoveryFailed(failure) = error.what() else { panic!("terminal error was lost") };
            assert!(failure.original.needs_cuda_recovery());
            assert!(failure.recovery.as_ref().unwrap().is_oom());
            assert_eq!(owner.cuda_gate().unwrap().device_generation(), generation);
            assert!(owner.submit(0, async { Ok(()) }, None).await.unwrap_err().is_cuda_recovery_failed());
            other.submit(0, async { Ok(()) }, None).await.unwrap();
            cpu.submit(0, async { Ok(()) }, None).await.unwrap();
            owner.shutdown().await;
            other.shutdown().await;
            cpu.shutdown().await;
        })
        .await
        .expect("one failed device must not close other executors");
    }

    #[tokio::test]
    async fn test_simultaneous_executor_failures_recover_independently() {
        tokio::time::timeout(Duration::from_secs(2), async {
            let (started, mut starts) = mpsc::unbounded_channel();
            let release = [Arc::new(tokio::sync::Notify::new()), Arc::new(tokio::sync::Notify::new())];
            let reset: RestartFn = Arc::new({
                let release = release.clone();
                move |device| {
                    let started = started.clone();
                    let release = Arc::clone(&release[device - 1]);
                    Box::pin(async move {
                        started.send(device).unwrap();
                        release.notified().await;
                        Ok(())
                    })
                }
            });
            let first = Arc::new(SyncStageRunner::new_with_reset_hook(8, 1, Arc::clone(&reset)));
            let second = Arc::new(SyncStageRunner::new_with_reset_hook(8, 2, reset));
            let start = Arc::new(tokio::sync::Barrier::new(3));
            let mut callers = Vec::new();
            for exec in [&first, &second] {
                let exec = Arc::clone(exec);
                let start = Arc::clone(&start);
                callers.push(tokio::spawn(async move {
                    exec.submit(
                        usize::MAX,
                        async move {
                            start.wait().await;
                            Err::<(), _>(needs_restart_error())
                        },
                        None,
                    )
                    .await
                }));
            }
            start.wait().await;
            for caller in callers {
                assert!(matches!(caller.await.unwrap().unwrap_err().what(), ThoError::CudaBeingRecovered));
            }
            let mut devices = vec![starts.recv().await.unwrap(), starts.recv().await.unwrap()];
            devices.sort();
            assert_eq!(devices, vec![1, 2]);
            release[0].notify_one();
            assert_eq!(first.submit(0, async { Ok(1) }, None).await.unwrap(), 1);
            assert!(second.cuda_gate().unwrap().is_recovering());
            assert_eq!(first.cuda_gate().unwrap().device_generation(), CudaDeviceGeneration(1));
            assert_eq!(second.cuda_gate().unwrap().device_generation(), CudaDeviceGeneration(0));
            release[1].notify_one();
            assert_eq!(second.submit(0, async { Ok(2) }, None).await.unwrap(), 2);
            Arc::try_unwrap(first).unwrap().shutdown().await;
            Arc::try_unwrap(second).unwrap().shutdown().await;
        })
        .await
        .expect("devices must recover without a shared completion barrier");
    }

    #[tokio::test]
    async fn test_held_recovery_allows_healthy_gpu_and_cpu_to_complete_work() {
        tokio::time::timeout(Duration::from_secs(3), async {
            let limits = Arc::new(FuturesLimits::new_with_parallelism(2, 2));
            let started = Arc::new(tokio::sync::Notify::new());
            let release = Arc::new(tokio::sync::Notify::new());
            let gate = Arc::new(CudaActivityGate::new(1));
            let recovery = CudaRecoveryCoordinator::new(
                Arc::clone(&gate),
                Arc::new({
                    let started = Arc::clone(&started);
                    let release = Arc::clone(&release);
                    move |_| {
                        let started = Arc::clone(&started);
                        let release = Arc::clone(&release);
                        Box::pin(async move {
                            started.notify_one();
                            release.notified().await;
                            Ok(())
                        })
                    }
                }),
            );
            let owner = SyncStageRunner::new_inner_with_buffer_limits(1, 1, Some(recovery), Arc::clone(&limits));
            let healthy = runner_with_limits(2, &limits);
            let cpu = runner_with_limits(0, &limits);
            owner.request_cuda_reset(needs_restart_error()).unwrap();
            started.notified().await;
            let mut queued = Box::pin(owner.submit(0, async { Ok(99) }, None));
            assert!(futures::poll!(queued.as_mut()).is_pending());
            // Complete several waves using the same shared execution budget,
            // with A's backend deliberately held until every wave has finished.
            for value in 0..8 {
                let (gpu_result, cpu_result) = tokio::join!(healthy.submit(0, async move { Ok(value) }, None), cpu.submit(0, async move { Ok(value) }, None));
                assert_eq!(gpu_result.unwrap(), value);
                assert_eq!(cpu_result.unwrap(), value);
                assert!(gate.is_recovering());
                assert_eq!(healthy.cuda_gate().unwrap().device_generation(), CudaDeviceGeneration(0));
                assert!(cpu.cuda_gate().is_none());
                assert!(futures::poll!(queued.as_mut()).is_pending());
            }
            release.notify_one();
            assert_eq!(queued.await.unwrap(), 99);
            assert_eq!(gate.device_generation(), CudaDeviceGeneration(1));
            owner.shutdown().await;
            Arc::try_unwrap(healthy).unwrap().shutdown().await;
            Arc::try_unwrap(cpu).unwrap().shutdown().await;
        })
        .await
        .expect("healthy devices must complete independent work throughout another device's recovery");
    }

    #[tokio::test]
    async fn test_reset_does_not_wait_for_unrelated_gpu_or_cpu_activity() {
        tokio::time::timeout(Duration::from_secs(2), async {
            let owner = SyncStageRunner::new_with_reset_hook(8, 1, Arc::new(|_| Box::pin(async { Ok(()) })));
            let other = Arc::new(SyncStageRunner::new_with_reset_hook(8, 2, Arc::new(|_| Box::pin(async { Ok(()) }))));
            let cpu = Arc::new(SyncStageRunner::new_cpu(8));
            let (started, mut starts) = mpsc::unbounded_channel();
            let release = Arc::new(tokio::sync::Notify::new());
            let mut callers = Vec::new();
            for exec in [&other, &cpu] {
                let exec = Arc::clone(exec);
                let started = started.clone();
                let release = Arc::clone(&release);
                callers.push(tokio::spawn(async move {
                    exec.submit(
                        usize::MAX,
                        async move {
                            // Register before notifying the test so release cannot be lost.
                            let released = release.notified();
                            started.send(()).unwrap();
                            released.await;
                            Ok(())
                        },
                        None,
                    )
                    .await
                }));
            }
            starts.recv().await.unwrap();
            starts.recv().await.unwrap();
            let ticket = owner.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap().unwrap();
            owner.cuda_gate().unwrap().wait_for_pause_to_finish(ticket).await.unwrap();
            assert_eq!(other.cuda_gate().unwrap().device_generation(), CudaDeviceGeneration(0));
            release.notify_waiters();
            for caller in callers {
                caller.await.unwrap().unwrap();
            }
            owner.shutdown().await;
            Arc::try_unwrap(other).unwrap().shutdown().await;
            Arc::try_unwrap(cpu).unwrap().shutdown().await;
        })
        .await
        .expect("recovery must drain only its own device");
    }

    #[tokio::test]
    async fn test_reset_between_suspend_and_enqueue_rejects_child_and_unwinds_parent() {
        let parent_finished = Arc::new(AtomicBool::new(false));
        let exec = Arc::new(SyncStageRunner::new_with_reset_hook(
            8,
            1,
            Arc::new({
                let parent_finished = Arc::clone(&parent_finished);
                move |_| {
                    assert!(parent_finished.load(Ordering::SeqCst), "parent must unwind before reset");
                    Box::pin(async { Ok(()) })
                }
            }),
        ));
        let child_ran = Arc::new(AtomicBool::new(false));
        let parent = exec.submit(
            0,
            {
                let exec = Arc::clone(&exec);
                let child_ran = Arc::clone(&child_ran);
                async move {
                    let mut child = Box::pin(exec.submit(
                        0,
                        async move {
                            child_ran.store(true, Ordering::SeqCst);
                            Ok(())
                        },
                        None,
                    ));
                    // Poll through capacity reservation and Suspend, but stop before
                    // its acknowledgement. Pause must then win the final enqueue check.
                    assert!(futures::poll!(child.as_mut()).is_pending());
                    exec.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap();
                    assert!(child.await.is_err());
                    parent_finished.store(true, Ordering::SeqCst);
                    Ok(())
                }
            },
            None,
        );
        tokio::time::timeout(Duration::from_secs(2), parent).await.unwrap().unwrap();
        tokio::time::timeout(Duration::from_secs(2), exec.cuda_gate().unwrap().wait_if_paused()).await.unwrap().unwrap();
        assert!(!child_ran.load(Ordering::SeqCst));
        exec.submit(0, async { Ok(()) }, None).await.unwrap();
        Arc::try_unwrap(exec).unwrap().shutdown().await;
    }
}
