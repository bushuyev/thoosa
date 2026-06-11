use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    fmt::{Debug, Formatter},
    future::Future,
    num::NonZeroUsize,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

#[cfg(any(test, feature = "mock_global_ctx"))]
use futures::FutureExt;
use futures_util::future::BoxFuture;
use log::{error, info};
use parking_lot::RwLock;
use tokio::{
    runtime::Handle,
    sync::{mpsc, oneshot, watch},
    task::{JoinHandle, JoinSet},
};
use tracing::{debug, warn, Instrument};

use thoosa_shared::{Cause, CudaActivityPauseTicket, ThoError, LOCK_TIMEOUT, cuda_activity_gate};
use thoosa_shared::tracked_await::track_await;

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
type JobId = u64;
type ResetFn = Arc<dyn Fn(usize /* tho_device */) -> BoxFuture<'static, Result<(), ThoError>> + Sync + Send + 'static>;

static NEXT_JOB_ID: AtomicU64 = AtomicU64::new(1);

tokio::task_local! {
    static CURRENT_JOB: CurrentJob;
}

#[derive(Clone, Debug)]
struct CurrentJob {
    job_id: JobId,
    cmd_tx: mpsc::UnboundedSender<ManagerCmd>,
}

struct Job {
    id: JobId,
    fut: JobFuture,
    /// Maximum number of *other* jobs allowed to run together with this one.
    ///
    /// Examples:
    /// - 0 => exclusive job
    /// - 1 => may run with at most 1 other job
    /// - usize::MAX => no per-job restriction beyond global limit
    max_other_parallel: usize,
}

impl Job {
    fn new<F>(id: JobId, max_other_parallel: usize, fut: F) -> Self
    where
        F: Future<Output = JobFeedback> + Send + 'static,
    {
        Self {
            id,
            fut: Box::pin(fut),
            max_other_parallel,
        }
    }

    fn total_parallel_cap(&self) -> usize {
        self.max_other_parallel.saturating_add(1)
    }

    fn is_exclusive(&self) -> bool {
        self.total_parallel_cap() == 1
    }

    async fn run(self) -> JobFeedback {
        let job_id = self.id;
        debug!("DynamicTaskExecutor, Job::run: running job {self:?}");

        let r = track_await(self.fut).await;

        debug!("DynamicTaskExecutor, Job::run: done job {job_id:?}");

        r
    }
}

impl Debug for Job {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Job")
            .field("id", &self.id)
            .field("max_other_parallel", &self.max_other_parallel)
            .finish()
    }
}

#[derive(Debug)]
struct LiveJob {
    /// max_other_parallel + 1
    cap: usize,
    /// Logical scheduler-active state. A suspended job is still physically alive,
    /// but is not counted against scheduling capacity.
    active: bool,
    suspend_depth: usize,
    waiting_resumes: Vec<oneshot::Sender<()>>,
}

#[derive(Debug)]
enum ManagerCmd {
    Suspend {
        job_id: JobId,
        reply: oneshot::Sender<()>,
    },
    Resume {
        job_id: JobId,
        reply: oneshot::Sender<()>,
    },
    RequestCudaReset {
        pause_ticket: CudaActivityPauseTicket,
        source: &'static str,
    },
    Recheck,
}

#[derive(Debug)]
struct PendingCudaReset {
    pause_ticket: CudaActivityPauseTicket,
    source: &'static str,
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

#[derive(Debug)]
pub enum SetLimitError {
    Closed,
}

impl std::fmt::Display for SetLimitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SetLimitError::Closed => write!(f, "executor is closed"),
        }
    }
}

impl std::error::Error for SetLimitError {}

/// Temporarily removes a parent job from the manager's active-capacity accounting
/// while that job awaits a nested `submit`.
///
/// The parent Tokio task is not paused: it is already blocked waiting for its child.
/// Without removing it from `active_count`/`active_caps`, an exclusive parent (or a
/// parent whose cap is already full) can prevent the child from ever starting. The
/// parent would then wait for the child while the child waits for the parent's slot.
///
/// `resume` waits for the manager to count the parent as active again before the
/// nested `submit` returns. `Drop` provides a best-effort asynchronous resume when
/// the nested submission unwinds or is cancelled after suspension was acknowledged.
struct ParentSuspendGuard {
    parent: Option<CurrentJob>,
    suspended: bool,
}

impl ParentSuspendGuard {
    fn new(parent: Option<CurrentJob>) -> Self {
        Self {
            parent,
            suspended: false,
        }
    }

    async fn maybe_suspend(&mut self) -> Result<(), ThoError> {
        let Some(parent) = &self.parent else {
            return Ok(());
        };

        let (ack_tx, ack_rx) = oneshot::channel();
        parent
            .cmd_tx
            .send(ManagerCmd::Suspend {
                job_id: parent.job_id,
                reply: ack_tx,
            })
            .map_err(|_| ThoError::new_with_cause("executor is closed", Cause::SubmitErrorClosed))?;

        // The flag is set only after the manager has applied the suspension. Cancellation
        // between sending Suspend and receiving this acknowledgement is not covered by Drop.
        track_await(ack_rx).await
            .map_err(|_| ThoError::new_with_cause("executor is closed", Cause::SubmitErrorClosed))?;

        self.suspended = true;
        Ok(())
    }

    async fn resume(&mut self) -> Result<(), ThoError> {
        if !self.suspended {
            return Ok(());
        }

        let Some(parent) = &self.parent else {
            self.suspended = false;
            return Ok(());
        };

        let (ack_tx, ack_rx) = oneshot::channel();
        parent
            .cmd_tx
            .send(ManagerCmd::Resume {
                job_id: parent.job_id,
                reply: ack_tx,
            })
            .map_err(|_| ThoError::new_with_cause("executor is closed", Cause::SubmitErrorClosed))?;

        track_await(ack_rx).await
            .map_err(|_| ThoError::new_with_cause("executor is closed", Cause::SubmitErrorClosed))?;

        self.suspended = false;
        Ok(())
    }
}

impl Drop for ParentSuspendGuard {
    fn drop(&mut self) {
        if !self.suspended {
            return;
        }

        let Some(parent) = self.parent.clone() else {
            return;
        };

        // Drop cannot await the manager acknowledgement, so cancellation/panic recovery
        // has to spawn a best-effort resume task on the current Tokio runtime.
        if let Ok(handle) = Handle::try_current() {
            handle.spawn(async move {
                let (ack_tx, ack_rx) = oneshot::channel();
                if parent
                    .cmd_tx
                    .send(ManagerCmd::Resume {
                        job_id: parent.job_id,
                        reply: ack_tx,
                    })
                    .is_ok()
                {
                    let _ = track_await(ack_rx).await;
                }
            }.instrument(tracing::info_span!(
                "dyn_mpsc::parent_suspend_guard",
                kind = "spawn",
                file = file!(),
                line = line!(),
                task = "resume parent job",
            )));
        }
    }
}

///Its goal is to run synchronous tasks limiting parallelism.
/// Maximum allowed number of parallel tasks is specified in max_other_parallel parameter to submit method
/// Currently, it's used from OOMHandler only.
pub struct DynamicTaskExecutor {
    tx: mpsc::Sender<Job>,
    limit_tx: watch::Sender<usize>,
    cmd_tx: mpsc::UnboundedSender<ManagerCmd>,
    accepting_tx: watch::Sender<bool>,
    accept_gate: Arc<tokio::sync::RwLock<()>>,
    worker: JoinHandle<()>,
    exclusive_job_id: Arc<RwLock<Option<JobId>>>,
    tho_device_idx: usize,
}

impl Debug for DynamicTaskExecutor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DynamicTaskExecutor")
            .field("tx", &self.tx)
            .field("limit_tx", &self.limit_tx)
            .field("cmd_tx", &self.cmd_tx)
            .field("accepting_tx", &self.accepting_tx)
            .field("accept_gate", &self.accept_gate)
            .field("worker", &self.worker)
            .field("exclusive_job_id", &self.exclusive_job_id)
            .field("tho_device_idx", &self.tho_device_idx)
            .finish_non_exhaustive()
    }
}

impl DynamicTaskExecutor {

    #[cfg(any(test, feature = "mock_global_ctx"))]
    pub fn new(queue_capacity: usize, initial_limit: NonZeroUsize) -> Self {
        Self::new_inner(
            queue_capacity,
            initial_limit,
            0,
            Arc::new(|_| async { Ok(()) }.boxed()),
        )
    }

    pub fn new_with_reset_hook(
        queue_capacity: usize,
        initial_limit: NonZeroUsize,
        tho_device_idx: usize,
        reset_fn: ResetFn,
    ) -> Self {
        Self::new_inner(queue_capacity, initial_limit, tho_device_idx, reset_fn)
    }

    fn new_inner(
        queue_capacity: usize,
        initial_limit: NonZeroUsize,
        tho_device_idx: usize,
        reset_fn: ResetFn,
    ) -> Self {
        debug!("DynamicTaskExecutor::new_inner: queue_capacity={queue_capacity} initial_limit={initial_limit} tho_device_idx={tho_device_idx}");

        let (tx, rx) = mpsc::channel(queue_capacity);
        let (limit_tx, limit_rx) = watch::channel(initial_limit.get());
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();
        let (accepting_tx, _) = watch::channel(true);
        let accept_gate = Arc::new(tokio::sync::RwLock::new(()));

        let cmd_tx_for_tasks = cmd_tx.clone();
        let exclusive_job_id = Arc::new(RwLock::new(None));
        let exclusive_job_id_for_manager = Arc::clone(&exclusive_job_id);
        let accepting_tx_for_manager = accepting_tx.clone();
        let accept_gate_for_manager = Arc::clone(&accept_gate);
        let reset_fn_for_manager = reset_fn;

        let worker = tokio::spawn(async move {
            track_await(Self::run_manager(
                rx,
                limit_rx,
                cmd_rx,
                cmd_tx_for_tasks,
                exclusive_job_id_for_manager,
                accepting_tx_for_manager,
                accept_gate_for_manager,
                tho_device_idx,
                reset_fn_for_manager,
            )).await;
        }.instrument(tracing::info_span!(
            "dyn_mpsc::manager",
            kind = "spawn",
            file = file!(),
            line = line!(),
            task = format!("run manager for tho_device_idx={tho_device_idx}"),
        )));

        Self {
            tx,
            limit_tx,
            cmd_tx,
            accepting_tx,
            accept_gate,
            worker,
            exclusive_job_id,
            tho_device_idx,
        }
    }

    fn closed_error(msg: impl Into<String>) -> ThoError {
        ThoError::new_with_cause(msg.into(), Cause::SubmitErrorClosed)
    }

    fn resetting_submit_error() -> ThoError {
        ThoError::new_with_cause(
            "Can't submit job: CUDA context reset is in progress",
            Cause::SubmitErrorClosed,
        )
    }

    async fn wait_until_accepting(&self, parent: Option<&CurrentJob>) -> Result<(), ThoError> {
        let mut accepting_rx = self.accepting_tx.subscribe();

        loop {
            if cuda_activity_gate().is_paused() {
                if parent.is_some() {
                    return Err(Self::resetting_submit_error());
                }
                track_await(cuda_activity_gate().wait_if_paused()).await;
                continue;
            }

            if *accepting_rx.borrow_and_update() {
                return Ok(());
            }

            // Nested submit while reset is requested would otherwise suspend a parent
            // which may be needed to complete reset-draining. Fail fast instead.
            if parent.is_some() {
                return Err(Self::resetting_submit_error());
            }

            track_await(accepting_rx
                .changed()).await
                .map_err(|_| Self::closed_error("Can't submit job: executor is closed"))?;
        }
    }

    pub async fn submit<F, T>(
        &self,
        max_other_parallel: usize,
        fut: F,
        job_id_to_rerun: Option<JobId>,//Some when invoked from OOMHandler on oom error and  max_other_parallel==0
    ) -> Result<T, ThoError>
    where
        F: Future<Output = Result<T, ThoError>> + Send + 'static,
        T: Send + Debug + 'static,
    {
        //currently jobs can only be submitted from OOMHandler::run and ::consume and are never nested (i.e. parent is always None)
        let parent = CURRENT_JOB.try_with(|job| job.clone()).ok();
        let job_id = job_id_to_rerun.unwrap_or_else(|| NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed));


        debug!("DynamicTaskExecutor::submit: job_id={job_id} max_other_parallel={max_other_parallel:?} parent={parent:?}");

        loop {
            track_await(self.wait_until_accepting(parent.as_ref())).await?;

            debug!("DynamicTaskExecutor::submit: job_id={job_id} accepted");

            // Reserve queue capacity before suspending a parent. Otherwise a nested submit
            // could remove its parent from active accounting while still waiting for space
            // in the input channel, with no child yet queued to make progress.
            //
            //   1. submit() calls suspend_guard.maybe_suspend().
            //   2. It sends ManagerCmd::Suspend at thoosa_common/src/runner/dyn_mpsc.rs:199.
            //   3. The manager handles it through handle_suspend().
            //   4. handle_suspend() sets job.active = false and calls remove_active() at thoosa_common/src/runner/dyn_mpsc.rs:1218.
            //

            let send_permit = match track_await(tokio::time::timeout(LOCK_TIMEOUT, self.tx.reserve())).await {
                Ok(Ok(permit)) => permit,
                Ok(Err(e)) => {
                    error!("DynamicTaskExecutor::submit: reserve job_id={job_id} failed: e={e:?}");
                    if let Some(job_id) = job_id_to_rerun {
                        self.clear_exclusive_job_id(job_id);
                    }
                    return Err(Self::closed_error(format!("Can't submit job: executor is_closed={}", self.tx.is_closed())));
                }
                Err(e) => {
                    error!("DynamicTaskExecutor::submit: reserve job_id={job_id} timed out: e={e:?}");
                    if let Some(job_id) = job_id_to_rerun {
                        self.clear_exclusive_job_id(job_id);
                    }
                    return Err(ThoError::new("Can't submit job: timed out"));
                }
            };

            // Reset may have started while reserve() was awaiting channel capacity. The
            // permit alone does not authorize enqueueing, so recheck before suspension.
            if !*self.accepting_tx.borrow() || cuda_activity_gate().is_paused() {
                drop(send_permit);
                if let Some(job_id) = job_id_to_rerun {
                    self.clear_exclusive_job_id(job_id);
                }
                if parent.is_some() {
                    return Err(Self::resetting_submit_error());
                }
                debug!("DynamicTaskExecutor::submit: reserve job_id={job_id} cuda_activity_gate is paused 1");

                continue;
            }

            // A nested submit must stop counting its blocked parent as active before the
            // child is enqueued, or the parent's own parallelism cap can block the child.
            let mut suspend_guard = ParentSuspendGuard::new(parent.clone());
            if let Err(e) = track_await(suspend_guard.maybe_suspend()).await {
                drop(send_permit);
                if let Some(job_id) = job_id_to_rerun {
                    self.clear_exclusive_job_id(job_id);
                }
                return Err(e);
            }

            // Suspension waits for a manager acknowledgement, during which reset may begin.
            // Back out and resume the parent rather than enqueueing a child into a paused executor.
            if !*self.accepting_tx.borrow() || cuda_activity_gate().is_paused() {
                drop(send_permit);
                let _ = track_await(suspend_guard.resume()).await;
                if let Some(job_id) = job_id_to_rerun {
                    self.clear_exclusive_job_id(job_id);
                }
                if parent.is_some() {
                    return Err(Self::resetting_submit_error());
                }
                debug!("DynamicTaskExecutor::submit: reserve job_id={job_id} cuda_activity_gate is paused 2");

                continue;
            }

            // Race-closing gate. During reset execution the manager owns the write lock.
            // Do not await here: if reset is starting, just back out and retry/fail.
            let accept_guard = match self.accept_gate.try_read() {
                Ok(guard) => guard,
                Err(_) => {
                    drop(send_permit);
                    let _ = track_await(suspend_guard.resume()).await;
                    if let Some(job_id) = job_id_to_rerun {
                        self.clear_exclusive_job_id(job_id);
                    }
                    if parent.is_some() {
                        return Err(Self::resetting_submit_error());
                    }
                    debug!("DynamicTaskExecutor::submit: reserve job_id={job_id} cuda_activity_gate is paused 3");

                    continue;
                }
            };

            if !*self.accepting_tx.borrow() || cuda_activity_gate().is_paused() {
                drop(accept_guard);
                drop(send_permit);
                let _ = track_await(suspend_guard.resume()).await;
                if let Some(job_id) = job_id_to_rerun {
                    self.clear_exclusive_job_id(job_id);
                }
                if parent.is_some() {
                    return Err(Self::resetting_submit_error());
                }
                debug!("DynamicTaskExecutor::submit: reserve job_id={job_id} cuda_activity_gate is paused 4");

                continue;
            }

            let (result_tx, result_rx) = oneshot::channel::<Result<T, ThoError>>();
            let exclusive_job_id = Arc::clone(&self.exclusive_job_id);
            let rerun_job_id = job_id_to_rerun;
            let accepting_tx = self.accepting_tx.clone();
            let cmd_tx = self.cmd_tx.clone();
            let tho_device_idx = self.tho_device_idx;


            let wrapped = async move {
                debug!("DynamicTaskExecutor::submit: job_id={job_id} waiting on wrapped fut");
                let result = track_await(fut).await;
                debug!("DynamicTaskExecutor::submit: job_id={job_id} got result.needs_cuda_ctx_reset={} wrapped fut", result.as_ref().is_err_and(|e| e.needs_cuda_ctx_reset()));

                if result.as_ref().is_err_and(|e| e.needs_cuda_ctx_reset()) {
                    if tho_device_idx > 0 {
                        warn!("DynamicTaskExecutor::submit: job_id={job_id} needs CUDA reset, result={result:?}");

                        if rerun_job_id == Some(job_id) {
                            let mut exclusive = exclusive_job_id.write();
                            if exclusive.as_ref() == Some(&job_id) {
                                warn!("DynamicTaskExecutor::submit: clearing exclusive_job_id={job_id} because rerun requested CUDA reset");
                                exclusive.take();
                            }
                        }

                        // The first failing job closes the shared gate and asks the manager
                        // to run one reset after every currently running job has unwound.
                        // Later failures in the same paused generation are coalesced into
                        // that request. The failing job must return ContextReset immediately:
                        // it may be a child of a suspended parent that also has to unwind
                        // before reset_fn can run exclusively.
                        let pause_ticket = cuda_activity_gate().request_pause();
                        if pause_ticket.is_owner() {
                            let _ = accepting_tx.send(false);
                            if cmd_tx.send(ManagerCmd::RequestCudaReset { pause_ticket, source: "submit" }).is_err() {
                                error!("DynamicTaskExecutor::submit: manager is closed while requesting CUDA reset");
                                let _ = accepting_tx.send(true);
                                cuda_activity_gate().finish_pause(pause_ticket);
                            }
                        }

                        let _ = result_tx.send(Err(ThoError::new_with_cause("Resetting cuda context", Cause::ContextReset)));
                        return JobFeedback::Neutral;
                    } else {
                        warn!("DynamicTaskExecutor::submit: job_id={job_id} reset requested on CPU/test executor");
                    }
                };

                let is_oom = result.as_ref().is_err_and(|e| e.is_oom());
                let is_ok = result.is_ok();

                let result = if is_oom && max_other_parallel == 0 {
                    let mut exclusive_job_id = exclusive_job_id.write();
                    exclusive_job_id.replace(job_id);

                    result.map_err(|e| {
                        ThoError::new_with_cause(
                            e.msg.clone(),
                            Cause::JobError {
                                job_id,
                                inner: Box::new(e.clone()),
                            },
                        )
                    })
                } else {
                    if rerun_job_id == Some(job_id) {
                        let mut exclusive_job_id = exclusive_job_id.write();
                        if exclusive_job_id.as_ref() == Some(&job_id) {
                            exclusive_job_id.take();
                        }
                    }
                    result
                };

                debug!("DynamicTaskExecutor::submit: job_id={job_id} wrapped result={result:?}");

                let _ = result_tx.send(result);

                if is_oom {
                    JobFeedback::Oom
                } else if is_ok {
                    JobFeedback::Success
                } else {
                    JobFeedback::Neutral
                }
            };

            send_permit.send(Job::new(job_id, max_other_parallel, wrapped));
            drop(accept_guard);

            debug!("DynamicTaskExecutor::submit: job_id={job_id} submitted job_id={job_id}");

            // Always try to resume the parent before returning, even if the job panicked
            // and closed result_tx without sending a value.
            let result = track_await(result_rx).await.map_err(|_| {
                ThoError::new_with_cause(
                    "job was cancelled before producing a result",
                    Cause::SubmitErrorCancelled,
                )
            })?;
            track_await(suspend_guard.resume()).await?;

            return result;
        }
    }

    async fn run_cuda_reset(
        tho_device_idx: usize,
        reset_fn: &ResetFn,
        accepting_tx: watch::Sender<bool>,
        accept_gate: Arc<tokio::sync::RwLock<()>>,
        exclusive_job_id: Arc<RwLock<Option<JobId>>>,
        pending_reset: PendingCudaReset,
    ) {
        // The manager calls this only after all jobs and their Tokio tasks have completed,
        // making reset equivalent to an exclusive max_other_parallel=0 job. Keeping the
        // reopen operation in Drop also covers cancellation or reset_fn failure.
        struct ResetPauseGuard {
            accepting_tx: watch::Sender<bool>,
            exclusive_job_id: Arc<RwLock<Option<JobId>>>,
            pause_ticket: CudaActivityPauseTicket,
        }

        impl Drop for ResetPauseGuard {
            fn drop(&mut self) {
                {
                    let mut exclusive = self.exclusive_job_id.write();
                    if exclusive.is_some() {
                        warn!("DynamicTaskExecutor::run_cuda_reset: clearing stale exclusive_job_id={exclusive:?} after reset");
                        exclusive.take();
                    }
                }
                let _ = self.accepting_tx.send(true);
                cuda_activity_gate().finish_pause(self.pause_ticket);
            }
        }

        let PendingCudaReset { pause_ticket, source } = pending_reset;
        let _reset_guard = ResetPauseGuard {
            accepting_tx,
            exclusive_job_id,
            pause_ticket,
        };
        // Hold the write side while resetting so new submissions cannot pass the
        // read-side accept gate until reset_fn has completed and the gate is reopened.
        let _accept_guard = track_await(Arc::clone(&accept_gate).write_owned()).await;

        if tho_device_idx > 0 {
            debug!("DynamicTaskExecutor::run_cuda_reset: sleep before running reset");

            log::logger().flush();
            tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;

            if let Err(e) = track_await(reset_fn(tho_device_idx)).await {
                error!("DynamicTaskExecutor::run_cuda_reset: CUDA context reset failed for source={source} tho_device_idx={tho_device_idx}: e={e:?}");
            } else {
                info!("DynamicTaskExecutor::run_cuda_reset: CUDA context was reset for source={source} tho_device_idx={tho_device_idx}");
            }
        } else {
            warn!("DynamicTaskExecutor::run_cuda_reset: reset requested on CPU/test executor source={source}");
        }
    }

    pub fn clear_exclusive_job_id(&self, job_id: u64) {
        debug!("DynamicTaskExecutor::clear_exclusive_job_id: job_id={job_id}");

        let mut exclusive_job_id = self.exclusive_job_id.write();

        if exclusive_job_id.as_ref() == Some(&job_id) {
            exclusive_job_id.take();
            let _ = self.cmd_tx.send(ManagerCmd::Recheck);
        }
    }

    pub fn request_cuda_reset(&self) {
        // Only the owner of this pause generation queues a reset. Concurrent requests
        // are already covered by that reset and must not close another executor's local
        // accepting flag without also owning the operation that reopens it.
        let pause_ticket = cuda_activity_gate().request_pause();
        if pause_ticket.is_owner() {
            let _ = self.accepting_tx.send(false);
            if self.cmd_tx.send(ManagerCmd::RequestCudaReset { pause_ticket, source: "request_cuda_reset" }).is_err() {
                error!("DynamicTaskExecutor::request_cuda_reset: manager is closed");
                let _ = self.accepting_tx.send(true);
                cuda_activity_gate().finish_pause(pause_ticket);
            }
        }
    }

    pub fn set_max_parallelism(&self, new_limit: NonZeroUsize) -> Result<(), SetLimitError> {
        self.limit_tx
            .send(new_limit.get())
            .map_err(|_| SetLimitError::Closed)
    }

    pub async fn shutdown(self) {
        drop(self.tx);
        drop(self.limit_tx);
        drop(self.cmd_tx);
        let _ = track_await(self.worker).await;
    }

    async fn run_manager(
        mut rx: mpsc::Receiver<Job>,
        mut limit_rx: watch::Receiver<usize>,
        mut cmd_rx: mpsc::UnboundedReceiver<ManagerCmd>,
        cmd_tx_for_tasks: mpsc::UnboundedSender<ManagerCmd>,
        exclusive_job_id: Arc<RwLock<Option<JobId>>>,
        accepting_tx: watch::Sender<bool>,
        accept_gate: Arc<tokio::sync::RwLock<()>>,
        tho_device_idx: usize,
        reset_fn: ResetFn,
    ) {
        let mut pending = VecDeque::<Job>::new();
        let mut running = JoinSet::<()>::new();

        let mut jobs = HashMap::<JobId, LiveJob>::new();
        let mut active_caps = BTreeMap::<usize, usize>::new();
        let mut active_count = 0usize;

        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<JobId>();

        let mut input_closed = false;
        let mut cmd_closed = false;
        let mut limit_closed = false;
        let mut pending_reset = None::<PendingCudaReset>;

        let global_limit = Arc::new(AtomicU64::new(*limit_rx.borrow() as u64));

        loop {
            debug!("DynamicTaskExecutor::run_manager, loop: active_caps={active_caps:?} active_count={active_count} running.len={} pending.len={} global_limit={} jobs.len={}",running.len(),pending.len(),global_limit.load(Ordering::Relaxed),jobs.len());

            Self::drain_finished(
                &mut running,
                &mut done_rx,
                &mut jobs,
                &mut active_caps,
                &mut active_count,
            );

            // Critical: drain already queued commands before spawning pending jobs.
            // Otherwise a job sent immediately before RequestCudaReset could be spawned
            // before the manager observes the reset request.
            Self::drain_commands(
                &mut cmd_rx,
                &mut cmd_closed,
                &mut jobs,
                &mut active_caps,
                &mut active_count,
                global_limit.load(Ordering::Relaxed) as usize,
                &mut pending_reset,
            );

            // While reset is pending, no queued jobs are started. Already-running jobs
            // continue to completion, and suspended parents are allowed to resume so they
            // can observe a child's ContextReset and unwind before reset_fn runs.
            let cuda_activity_gate_is_paused = cuda_activity_gate().is_paused();
            debug!("DynamicTaskExecutor::run_manager: cuda_activity_gate_is_paused={cuda_activity_gate_is_paused}" );

            if !cuda_activity_gate_is_paused {
                let current_exclusive_job_id = *exclusive_job_id.read();
                Self::try_resume_waiters(
                    &mut jobs,
                    &mut active_caps,
                    &mut active_count,
                    global_limit.load(Ordering::Relaxed) as usize,
                    None,
                );
                Self::try_spawn_ready_jobs(
                    &mut pending,
                    &mut running,
                    &mut jobs,
                    &mut active_caps,
                    &mut active_count,
                    Arc::clone(&global_limit),
                    &done_tx,
                    &cmd_tx_for_tasks,
                    current_exclusive_job_id,
                );
            } else {
                Self::force_resume_waiters_for_unwind(
                    &mut jobs,
                    &mut active_caps,
                    &mut active_count,
                );
            }

            // `jobs` can become empty before JoinSet observes physical task completion,
            // so require both collections to be empty before touching the CUDA context.
            if pending_reset.is_some() && jobs.is_empty() && running.is_empty() {
                let pending_reset = pending_reset.take().expect("pending reset");
                track_await(Self::run_cuda_reset(
                    tho_device_idx,
                    &reset_fn,
                    accepting_tx.clone(),
                    Arc::clone(&accept_gate),
                    Arc::clone(&exclusive_job_id),
                    pending_reset,
                )).instrument(tracing::info_span!(
                    "dyn_mpsc::cuda_reset",
                    file = file!(),
                    line = line!(),
                    task = format!("reset cuda context for tho_device_idx={tho_device_idx}"),
                )).await;
                continue;
            }

            if input_closed && pending.is_empty() && jobs.is_empty() && !cuda_activity_gate().is_paused() {
                break;
            }

            if !cuda_activity_gate().is_paused() && !pending.is_empty() && jobs.is_empty() {
                let exclusive = *exclusive_job_id.read();

                warn!("DynamicTaskExecutor::run_manager: idle with pending jobs: pending_len={}, pending_front={:?}, exclusive_job_id={exclusive:?}, active_count={}, active_caps={active_caps:?}",pending.len(),pending.front().map(|j| (j.id, j.total_parallel_cap())),active_count,);
            }

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
                                global_limit.load(Ordering::Relaxed) as usize,
                                &mut pending_reset,
                            );
                        }
                        None => {
                            cmd_closed = true;
                        }
                    }
                }

                changed = limit_rx.changed(), if !limit_closed => {
                    match changed {
                        Ok(()) => {
                            let new_limit = *limit_rx.borrow_and_update();
                            global_limit.store(new_limit as u64, Ordering::SeqCst);
                        }
                        Err(_) => {
                            limit_closed = true;
                        }
                    }
                }

                _ = cuda_activity_gate().wait_if_paused(), if cuda_activity_gate().is_paused() => {}

                join_result = running.join_next(), if !running.is_empty() => {
                    if let Some(Err(err)) = join_result {
                        error!("job failed: {err}");
                    }
                }
            }
        }

        while let Some(res) = track_await(running.join_next()).await {
            if let Err(err) = res {
                error!("job failed during shutdown: {err}");
            }
        }
    }

    fn process_command(
        cmd: ManagerCmd,
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        global_limit: usize,
        pending_reset: &mut Option<PendingCudaReset>,
    ) {
        debug!("DynamicTaskExecutor::run_manager: manager cmd={cmd:?}");

        match cmd {
            ManagerCmd::Suspend { job_id, reply } => {
                Self::handle_suspend(jobs, active_caps, active_count, job_id);
                let _ = reply.send(());
            }
            ManagerCmd::Resume { job_id, reply } => {
                Self::handle_resume(
                    jobs,
                    active_caps,
                    active_count,
                    global_limit,
                    None,
                    job_id,
                    reply,
                    cuda_activity_gate().is_paused(),
                );
            }
            ManagerCmd::RequestCudaReset { pause_ticket, source } => {
                // Only the owner ticket represents a new pause generation. Non-owner
                // requests are normally discarded by the caller; keeping this check here
                // also makes delayed/stale commands harmless. One paused generation thus
                // triggers exactly one reset_fn invocation.
                if pause_ticket.is_owner() && pending_reset.is_none() {
                    *pending_reset = Some(PendingCudaReset { pause_ticket, source });
                }
            }
            ManagerCmd::Recheck => {}
        }
    }

    fn drain_commands(
        cmd_rx: &mut mpsc::UnboundedReceiver<ManagerCmd>,
        cmd_closed: &mut bool,
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        global_limit: usize,
        pending_reset: &mut Option<PendingCudaReset>,
    ) {
        use tokio::sync::mpsc::error::TryRecvError;

        if *cmd_closed {
            return;
        }

        loop {
            match cmd_rx.try_recv() {
                Ok(cmd) => Self::process_command(
                    cmd,
                    jobs,
                    active_caps,
                    active_count,
                    global_limit,
                    pending_reset,
                ),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    *cmd_closed = true;
                    break;
                }
            }
        }
    }

    fn try_spawn_ready_jobs(
        pending: &mut VecDeque<Job>,
        running: &mut JoinSet<()>,
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        global_limit: Arc<AtomicU64>,
        done_tx: &mpsc::UnboundedSender<JobId>,
        cmd_tx_for_tasks: &mpsc::UnboundedSender<ManagerCmd>,
        exclusive_job_id: Option<JobId>,
    ) {
        debug!("DynamicTaskExecutor::try_spawn_ready_jobs: pending={pending:?}");

        loop {
            let mut spawned_any = false;
            let mut idx = 0usize;

            while idx < pending.len() {
                let can_spawn = {
                    let job = &pending[idx];
                    Self::can_activate_now(
                        job.id,
                        *active_count,
                        global_limit.load(Ordering::Relaxed) as usize,
                        active_caps,
                        job.total_parallel_cap(),
                        exclusive_job_id,
                    )
                };

                if can_spawn {
                    let job = pending.remove(idx).expect("pending index must be valid");
                    let job_id = job.id;
                    let cap = job.total_parallel_cap();

                    jobs.insert(
                        job_id,
                        LiveJob {
                            cap,
                            active: true,
                            suspend_depth: 0,
                            waiting_resumes: Vec::new(),
                        },
                    );
                    Self::add_active(active_caps, active_count, cap);

                    let done_tx = done_tx.clone();
                    let cmd_tx_for_tasks = cmd_tx_for_tasks.clone();
                    let global_limit = Arc::clone(&global_limit);

                    debug!("DynamicTaskExecutor::try_spawn_ready_jobs: spawning job_id={job_id}");

                    running.spawn(async move {
                        debug!("DynamicTaskExecutor::try_spawn_ready_jobs: spawned job_id={job_id}");

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

                        let feedback = track_await(CURRENT_JOB
                            .scope(
                                CurrentJob {
                                    job_id,
                                    cmd_tx: cmd_tx_for_tasks,
                                },
                                async move {
                                    debug!("DynamicTaskExecutor::try_spawn_ready_jobs: running job job_id={job_id}");

                                    track_await(job.run())
                                        .await
                                },
                            ))
                            .await;

                        match feedback {
                            JobFeedback::Oom => {
                                debug!("DynamicTaskExecutor::try_spawn_ready_jobs: OOM job job_id={job_id}, global_limit={}",global_limit.load(Ordering::Relaxed));

                                if global_limit.load(Ordering::Relaxed) > 16 {
                                    global_limit.fetch_sub(1, Ordering::SeqCst);
                                    thoosa_shared::buffer_limits::buffer_limits().on_oom();
                                }
                            }
                            JobFeedback::Success => {
                                debug!("DynamicTaskExecutor::try_spawn_ready_jobs: success job job_id={job_id}, global_limit={}",global_limit.load(Ordering::Relaxed));
                                if global_limit.load(Ordering::Relaxed) < 32 {
                                    global_limit.fetch_add(1, Ordering::SeqCst);
                                    thoosa_shared::buffer_limits::buffer_limits().on_ok();
                                }
                            }
                            JobFeedback::Neutral => {
                                debug!("DynamicTaskExecutor::try_spawn_ready_jobs: neutral job job_id={job_id}, global_limit={}",global_limit.load(Ordering::Relaxed));
                            }
                        }
                    }.instrument(tracing::info_span!(
                        "dyn_mpsc::job",
                        kind = "spawn",
                        file = file!(),
                        line = line!(),
                        task = format!("run dynamic executor job_id={job_id}"),
                    )));

                    spawned_any = true;
                    break;
                }

                // Fairness: do not keep bypassing an exclusive job at the queue head.
                // Otherwise a steady stream of non-exclusive jobs can keep it from ever
                // seeing an empty executor.
                if idx == 0 && pending[0].is_exclusive() {
                    match exclusive_job_id {
                        // If a specific OOM-rerun job is required, do not let some other
                        // exclusive job at the front block the scan.
                        Some(required_job_id) if pending[0].id != required_job_id => {
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

    fn can_activate_now(
        job_id: JobId,
        active_count: usize,
        global_limit: usize,
        active_caps: &BTreeMap<usize, usize>,
        new_cap: usize,
        exclusive_job_id: Option<JobId>,
    ) -> bool {
        if exclusive_job_id.is_some_and(|exclusive_job_id| exclusive_job_id != job_id) {
            debug!("DynamicTaskExecutor::can_activate_now: blocked by exclusive job current_job_id={job_id} exclusive_job_id={exclusive_job_id:?}");
            return false;
        }

        let next_total = active_count.saturating_add(1);
        let current_group_cap = Self::current_group_cap(global_limit, active_caps);

        let min = current_group_cap.min(new_cap);
        let r = next_total <= min;

        debug!("DynamicTaskExecutor::can_activate_now: job_id={job_id} active_count={active_count} next_total={next_total} global_limit={global_limit} min={min} r={r}");
        r
    }

    fn current_group_cap(global_limit: usize, active_caps: &BTreeMap<usize, usize>) -> usize {
        match active_caps.first_key_value() {
            Some((&smallest_active_cap, _)) => global_limit.min(smallest_active_cap),
            None => global_limit,
        }
    }

    fn add_active(
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        cap: usize,
    ) {
        *active_caps.entry(cap).or_insert(0) += 1;
        *active_count += 1;
    }

    fn remove_active(
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        cap: usize,
    ) {
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

    fn handle_suspend(
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        job_id: JobId,
    ) {
        let Some(job) = jobs.get_mut(&job_id) else {
            return;
        };

        job.suspend_depth += 1;

        if job.active {
            job.active = false;
            Self::remove_active(active_caps, active_count, job.cap);
        }
    }

    fn handle_resume(
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        global_limit: usize,
        exclusive_job_id: Option<JobId>,
        job_id: JobId,
        reply: oneshot::Sender<()>,
        activity_paused: bool,
    ) {
        let Some(job) = jobs.get_mut(&job_id) else {
            let _ = reply.send(());
            return;
        };

        // Every nested submit() that suspended this parent must wait until the
        // parent becomes active again before returning.
        job.waiting_resumes.push(reply);

        if activity_paused {
            // While reset is paused, capacity accounting is secondary to liveness. Let
            // already-running parents resume so they can observe the nested reset error
            // and unwind; no new pending jobs are spawned until the shared gate reopens.
            if job.suspend_depth > 0 {
                job.suspend_depth -= 1;
            }
            if job.suspend_depth == 0 {
                if !job.active {
                    job.active = true;
                    Self::add_active(active_caps, active_count, job.cap);
                }
                Self::flush_waiting_resumes(job);
            }
            return;
        }

        if job.suspend_depth == 0 {
            if job.active {
                Self::flush_waiting_resumes(job);
            } else if Self::can_activate_now(
                job_id,
                *active_count,
                global_limit,
                active_caps,
                job.cap,
                exclusive_job_id,
            ) {
                job.active = true;
                Self::add_active(active_caps, active_count, job.cap);
                Self::flush_waiting_resumes(job);
            }
            return;
        }

        job.suspend_depth -= 1;

        if job.suspend_depth == 0
            && Self::can_activate_now(
            job_id,
            *active_count,
            global_limit,
            active_caps,
            job.cap,
            exclusive_job_id,
        )
        {
            job.active = true;
            Self::add_active(active_caps, active_count, job.cap);
            Self::flush_waiting_resumes(job);
        }
    }

    fn force_resume_waiters_for_unwind(
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
    ) {
        // Resumes queued before the pause may not receive a fresh Resume command. Flush
        // only fully-unsuspended jobs so parent tasks that are already eligible to run
        // are not stranded behind a reset that may be waiting for them to drop state.
        for job in jobs.values_mut() {
            if job.suspend_depth == 0 && !job.waiting_resumes.is_empty() {
                if !job.active {
                    job.active = true;
                    Self::add_active(active_caps, active_count, job.cap);
                }
                Self::flush_waiting_resumes(job);
            }
        }
    }

    fn try_resume_waiters(
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        global_limit: usize,
        exclusive_job_id: Option<JobId>,
    ) {
        for job in jobs.values_mut() {
            if job.active && !job.waiting_resumes.is_empty() {
                Self::flush_waiting_resumes(job);
            }
        }

        loop {
            let waiting_ids: Vec<JobId> = jobs
                .iter()
                .filter_map(|(&id, job)| {
                    if !job.active && job.suspend_depth == 0 && !job.waiting_resumes.is_empty() {
                        Some(id)
                    } else {
                        None
                    }
                })
                .collect();

            debug!("DynamicTaskExecutor::try_resume_waiters: waiting_ids={waiting_ids:?}");

            let mut progressed = false;

            for job_id in waiting_ids {
                let cap = match jobs.get(&job_id) {
                    Some(job)
                    if !job.active
                        && job.suspend_depth == 0
                        && !job.waiting_resumes.is_empty() =>
                        {
                            job.cap
                        }
                    _ => continue,
                };

                if Self::can_activate_now(
                    job_id,
                    *active_count,
                    global_limit,
                    active_caps,
                    cap,
                    exclusive_job_id,
                ) {
                    let job = jobs.get_mut(&job_id).expect("job must exist");
                    job.active = true;
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

    fn note_job_finished(
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        job_id: JobId,
    ) {
        if let Some(job) = jobs.remove(&job_id) {
            debug!("DynamicTaskExecutor::note_job_finished: job={job:?}");

            if job.active {
                Self::remove_active(active_caps, active_count, job.cap);
            }
        } else {
            warn!("DynamicTaskExecutor::note_job_finished: no {job_id}");
        }
    }

    fn drain_finished(
        running: &mut JoinSet<()>,
        done_rx: &mut mpsc::UnboundedReceiver<JobId>,
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
    ) {
        while let Ok(job_id) = done_rx.try_recv() {
            Self::note_job_finished(jobs, active_caps, active_count, job_id);
        }

        while let Some(res) = running.try_join_next() {
            if let Err(err) = res {
                error!("DynamicTaskExecutor::drain_finished: err={err:?}");
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
            atomic::{AtomicBool, AtomicUsize, Ordering},
            Arc,
        },
        time::Duration,
    };
    use thoosa_shared::{CudaAction, ThoosaError};

    fn needs_reset_error() -> ThoError {
        ThoError::new_with_cause(
            "needs reset",
            Cause::Cuda {
                action: CudaAction::NeedsRestart,
                msg: "reset me".into(),
            },
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_reset_waits_for_running_jobs() -> Result<(), ThoError> {
        let reset_calls = Arc::new(AtomicUsize::new(0));
        let exec = Arc::new(DynamicTaskExecutor::new_with_reset_hook(
            64,
            NonZeroUsize::new(4).unwrap(),
            1,
            {
                let reset_calls = Arc::clone(&reset_calls);
                Arc::new(move |_device_idx| {
                    let reset_calls = Arc::clone(&reset_calls);
                    async move {
                        reset_calls.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    }
                        .boxed()
                })
            },
        ));

        let (long_started_tx1, long_started_rx1) = oneshot::channel();
        let (long_started_tx2, long_started_rx2) = oneshot::channel();
        let (release_tx1, release_rx1) = oneshot::channel();
        let (release_tx2, release_rx2) = oneshot::channel();

        let long1 = {
            let exec = Arc::clone(&exec);
            tokio::spawn(async move {
                exec.submit(usize::MAX, async move {
                    let _ = long_started_tx1.send(());
                    let _ = release_rx1.await;
                    Ok::<_, ThoError>(())
                }, None).await
            })
        };

        let long2 = {
            let exec = Arc::clone(&exec);
            tokio::spawn(async move {
                exec.submit(usize::MAX, async move {
                    let _ = long_started_tx2.send(());
                    let _ = release_rx2.await;
                    Ok::<_, ThoError>(())
                }, None).await
            })
        };

        let _ = long_started_rx1.await;
        let _ = long_started_rx2.await;

        let reset_err: Result<(), ThoError> = exec
            .submit(usize::MAX, async move { Err(needs_reset_error()) }, None)
            .await;

        let Err(ThoosaError {  what: Cause::ContextReset, ..}) = reset_err else {
            panic!("Unexpected result: {reset_err:?}")
        };

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
        }).await.expect("reset hook should start after all running jobs finish");
        assert_eq!(reset_calls.load(Ordering::SeqCst), 1);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_running_reset_failures_are_coalesced() -> Result<(), ThoError> {
        let reset_calls = Arc::new(AtomicUsize::new(0));
        let finished_jobs = Arc::new(AtomicUsize::new(0));
        let exec = Arc::new(DynamicTaskExecutor::new_with_reset_hook(
            64,
            NonZeroUsize::new(2).unwrap(),
            1,
            {
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
            },
        ));

        let (started_tx1, started_rx1) = oneshot::channel();
        let (started_tx2, started_rx2) = oneshot::channel();
        let (release_tx1, release_rx1) = oneshot::channel();
        let (release_tx2, release_rx2) = oneshot::channel();

        let job1 = {
            let exec = Arc::clone(&exec);
            let finished_jobs = Arc::clone(&finished_jobs);
            tokio::spawn(async move {
                exec.submit(usize::MAX, async move {
                    let _ = started_tx1.send(());
                    let _ = release_rx1.await;
                    finished_jobs.fetch_add(1, Ordering::SeqCst);
                    Err::<(), _>(needs_reset_error())
                }, None).await
            })
        };

        let job2 = {
            let exec = Arc::clone(&exec);
            let finished_jobs = Arc::clone(&finished_jobs);
            tokio::spawn(async move {
                exec.submit(usize::MAX, async move {
                    let _ = started_tx2.send(());
                    let _ = release_rx2.await;
                    finished_jobs.fetch_add(1, Ordering::SeqCst);
                    Err::<(), _>(needs_reset_error())
                }, None).await
            })
        };

        let _ = started_rx1.await;
        let _ = started_rx2.await;
        let _ = release_tx1.send(());
        let _ = release_tx2.send(());

        for result in [job1.await?, job2.await?] {
            let Err(ThoosaError { what: Cause::ContextReset, .. }) = result else {
                panic!("Unexpected result: {result:?}")
            };
        }

        tokio::time::timeout(Duration::from_secs(2), async {
            while reset_calls.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        }).await.expect("one reset should run after both failing jobs finish");
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

        let exec = Arc::new(DynamicTaskExecutor::new_with_reset_hook(
            64,
            NonZeroUsize::new(4).unwrap(),
            1,
            {
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
            },
        ));

        let result: Result<(), ThoError> = {
            let exec_for_child = Arc::clone(&exec);
            let parent_exited = Arc::clone(&parent_exited);
            let reset_calls = Arc::clone(&reset_calls);
            tokio::time::timeout(Duration::from_secs(2), exec.submit(0, async move {
                debug!("test_nested_needs_reset_unwinds_parent_before_reset: Parent job running");
                let child_result: Result<(), ThoError> = exec_for_child.submit(
                    usize::MAX,
                    async move {
                        debug!("test_nested_needs_reset_unwinds_parent_before_reset: Child job running");

                        if reset_calls.load(Ordering::Relaxed) == 0 {
                            debug!("test_nested_needs_reset_unwinds_parent_before_reset: Child job failed");
                            Err(needs_reset_error())
                        } else {
                            debug!(":test_nested_needs_reset_unwinds_parent_before_reset: Child job done");
                            Ok(())
                        }
                    },
                    None
                ).await;

                parent_exited.store(true, Ordering::SeqCst);

                debug!("test_nested_needs_reset_unwinds_parent_before_reset: Parent job done");

                child_result

            }, None)).await.expect("parent should resume and return reset error")
        };


        let Err(ThoosaError {  what: Cause::ContextReset, ..}) = result else {
            panic!("Unexpected result: {result:?}")
        };

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

        let exec = Arc::new(DynamicTaskExecutor::new_with_reset_hook(
            64,
            NonZeroUsize::new(4).unwrap(),
            1,
            {
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
            },
        ));

        let (child_started_tx, child_started_rx) = oneshot::channel();
        let (release_child_tx, release_child_rx) = oneshot::channel();
        let parent = {
            let exec_for_parent = Arc::clone(&exec);
            let parent_exited = Arc::clone(&parent_exited);
            tokio::spawn(async move {
                let exec_for_child = Arc::clone(&exec_for_parent);
                exec_for_parent.submit(0, async move {
                    let child_result: Result<(), ThoError> = exec_for_child.submit(
                        usize::MAX,
                        async move {
                            let _ = child_started_tx.send(());
                            let _ = release_child_rx.await;
                            Ok(())
                        },
                        None,
                    ).await;
                    parent_exited.store(true, Ordering::SeqCst);
                    child_result
                }, None).await
            })
        };

        let _ = child_started_rx.await;
        exec.request_cuda_reset();
        let _ = release_child_tx.send(());

        tokio::time::timeout(Duration::from_secs(2), parent)
            .await
            .expect("parent task should not deadlock")??;

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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_stale_non_owner_reset_request_does_not_close_accepting() -> Result<(), ThoError> {
        cuda_activity_gate().wait_if_paused().await;

        let owner_ticket = cuda_activity_gate().request_pause();
        let non_owner_ticket = cuda_activity_gate().request_pause();
        cuda_activity_gate().finish_pause(owner_ticket);

        let mut jobs = HashMap::new();
        let mut active_caps = BTreeMap::new();
        let mut active_count = 0;
        let mut pending_reset = None;

        DynamicTaskExecutor::process_command(
            ManagerCmd::RequestCudaReset {
                pause_ticket: non_owner_ticket,
                source: "test",
            },
            &mut jobs,
            &mut active_caps,
            &mut active_count,
            4,
            &mut pending_reset,
        );

        assert!(pending_reset.is_none());
        assert!(!cuda_activity_gate().is_paused());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_nested_exclusive_oom_does_not_deadlock_parent_resume() -> Result<(), ThoError> {
        let exec = Arc::new(DynamicTaskExecutor::new(64, NonZeroUsize::new(4).unwrap()));

        let result = tokio::time::timeout(Duration::from_secs(2), {
            let exec_for_parent = Arc::clone(&exec);
            exec.submit(0, async move {
                let child_result: Result<(), ThoError> = exec_for_parent
                    .submit(0, async move { Err(ThoError::oom()) }, None)
                    .await;

                match child_result {
                    Err(e) => match &e.what {
                        Cause::JobError { job_id, inner } => {
                            assert!(inner.is_oom());
                            assert!(*job_id > 0);
                            Ok::<_, ThoError>(())
                        }
                        other => panic!("unexpected error cause: {other:?}"),
                    },
                    Ok(_) => panic!("child should fail"),
                }
            }, None)
        })
            .await
            .expect("parent resume must not deadlock")?;

        assert_eq!(result, ());

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_set_max_parallelism_is_observed() -> Result<(), ThoError> {
        let exec = Arc::new(DynamicTaskExecutor::new(64, NonZeroUsize::new(1).unwrap()));
        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let results = futures_util::stream::iter(0..8)
            .map(|_| {
                let exec = Arc::clone(&exec);
                let active = Arc::clone(&active);
                let max_seen = Arc::clone(&max_seen);
                async move {
                    exec.submit(0, async move {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_seen.fetch_max(now, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(30)).await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok::<_, ThoError>(())
                    }, None).await
                }
            })
            .buffer_unordered(8)
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(results.len(), 8);
        assert!(max_seen.load(Ordering::SeqCst) <= 1);

        exec.set_max_parallelism(NonZeroUsize::new(4).unwrap()).unwrap();

        let active = Arc::new(AtomicUsize::new(0));
        let max_seen = Arc::new(AtomicUsize::new(0));

        let results = futures_util::stream::iter(0..8)
            .map(|_| {
                let exec = Arc::clone(&exec);
                let active = Arc::clone(&active);
                let max_seen = Arc::clone(&max_seen);
                async move {
                    exec.submit(usize::MAX, async move {
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_seen.fetch_max(now, Ordering::SeqCst);
                        tokio::time::sleep(Duration::from_millis(30)).await;
                        active.fetch_sub(1, Ordering::SeqCst);
                        Ok::<_, ThoError>(())
                    }, None).await
                }
            })
            .buffer_unordered(8)
            .try_collect::<Vec<_>>()
            .await?;

        assert_eq!(results.len(), 8);
        assert!(max_seen.load(Ordering::SeqCst) > 1);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }
}
