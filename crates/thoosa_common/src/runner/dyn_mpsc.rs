use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    future::Future,
    num::NonZeroUsize,
    pin::Pin,
    sync::atomic::{AtomicU64, Ordering},
};
use std::fmt::{Debug, Formatter};
use log::error;
use tokio::{
    runtime::Handle,
    sync::{mpsc, oneshot, watch},
    task::{JoinHandle, JoinSet},
};
use tracing::{debug, warn};
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use parking_lot::RwLock;
use thoosa_shared::{Cause, ThoError, LOCK_TIMEOUT};
use thoosa_shims::context::ThoCudaContext;

enum IsOOM {
    Yes,
    No
}

impl From<bool> for IsOOM {
    fn from(value: bool) -> Self {
        if value {
            IsOOM::Yes
        } else {
            IsOOM::No
        }
    }
}

type BoxFuture = Pin<Box<dyn Future<Output = IsOOM> + Send + 'static>>;
type JobId = u64;
type ResetFn = Arc<dyn Fn(usize) -> Result<(), ThoError> + Send + Sync>;

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
    fut: BoxFuture,
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
        F: Future<Output = IsOOM> + Send + 'static,
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

    async fn run(self) -> IsOOM {
        let job_id = self.id;
        debug!("DynamicTaskExecutor, Job::run: running job {self:?} ");

        let r = self.fut.await;

        debug!("DynamicTaskExecutor, Job::run: done job {job_id:?} ");

        r
    }
}

impl Debug for Job {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Jon")
            .field("id", &self.id)
            .field("max_other_parallel", &self.max_other_parallel)
            .finish()
    }
}

#[derive(Debug)]
struct LiveJob {
    cap: usize,//max_other_parallel + 1
    active: bool,
    suspend_depth: usize,
    waiting_resumes: Vec<oneshot::Sender<()>>,
}

enum ManagerCmd {
    Suspend {
        job_id: JobId,
        reply: oneshot::Sender<()>,
    },
    Resume {
        job_id: JobId,
        reply: oneshot::Sender<()>,
    },
    RequestCudaReset,
    Recheck,
}

#[derive(Debug)]
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
            .map_err(|_| SubmitError::Closed)?;

        ack_rx.await.map_err(|_| SubmitError::Closed)?;
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
            .map_err(|_| SubmitError::Closed)?;

        ack_rx.await.map_err(|_| SubmitError::Closed)?;
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
                    let _ = ack_rx.await;
                }
            });
        }
    }
}

#[derive(Debug)]
pub struct DynamicTaskExecutor {
    tx: mpsc::Sender<Job>,
    limit_tx: watch::Sender<usize>,
    cmd_tx: mpsc::UnboundedSender<ManagerCmd>,
    accepting_tx: watch::Sender<bool>,
    accept_gate: Arc<tokio::sync::RwLock<()>>,
    worker: JoinHandle<()>,
    exclusive_job_id: Arc<RwLock<Option<JobId>>>,
}

impl DynamicTaskExecutor {
    pub fn new(queue_capacity: usize, initial_limit: NonZeroUsize) -> Self {
        Self::new_inner(queue_capacity, initial_limit, None, Arc::new(|_| Ok(())))
    }

    pub fn new_for_device(queue_capacity: usize, initial_limit: NonZeroUsize, device_idx: usize) -> Self {
        Self::new_inner(queue_capacity, initial_limit, Some(device_idx), Arc::new(Self::reset_cuda_context))
    }

    #[cfg(test)]
    fn new_with_reset_hook(
        queue_capacity: usize,
        initial_limit: NonZeroUsize,
        device_idx: usize,
        reset_fn: ResetFn,
    ) -> Self {
        Self::new_inner(queue_capacity, initial_limit, Some(device_idx), reset_fn)
    }

    fn new_inner(
        queue_capacity: usize,
        initial_limit: NonZeroUsize,
        reset_device_idx: Option<usize>,
        reset_fn: ResetFn,
    ) -> Self {
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

        let worker = tokio::spawn(async move {
            Self::run_manager(
                rx,
                limit_rx,
                cmd_rx,
                cmd_tx_for_tasks,
                exclusive_job_id_for_manager,
                accepting_tx_for_manager,
                accept_gate_for_manager,
                reset_device_idx,
                reset_fn,
            ).await;
        });

        Self {
            tx,
            limit_tx,
            cmd_tx,
            accepting_tx,
            accept_gate,
            worker,
            exclusive_job_id,
        }
    }

    fn reset_cuda_context(device_idx: usize) -> Result<(), ThoError> {
        let cu_ctx = ThoCudaContext::new((device_idx - 1) as i32).map_err(ThoError::new)?;
        cu_ctx.reset()?;

        warn!("DynamicTaskExecutor::reset_cuda_context: done");
        Ok(())
    }

    fn resetting_submit_error() -> ThoError {
        ThoError::new_with_error("Can't submit job: CUDA context reset is in progress", Cause::SubmitErrorClosed)
    }

    async fn wait_until_accepting(&self, parent: Option<&CurrentJob>) -> Result<(), ThoError> {
        let mut accepting_rx = self.accepting_tx.subscribe();

        loop {
            if *accepting_rx.borrow_and_update() {
                return Ok(());
            }

            if parent.is_some() {
                return Err(Self::resetting_submit_error());
            }

            accepting_rx.changed().await.map_err(|_| ThoError::new_with_error("Can't submit job", Cause::SubmitErrorClosed))?;
        }
    }

    pub async fn submit<F, T>(
        &self,
        max_other_parallel: usize,
        fut: F,
        job_id_to_rerun: Option<JobId>,
    ) -> Result<T, ThoError>
    where
        F: Future<Output = Result<T, ThoError>> + Send + 'static,
        T: Send + Debug + 'static,
    {
        let parent = CURRENT_JOB.try_with(|job| job.clone()).ok();
        let job_id = job_id_to_rerun.unwrap_or_else(|| NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed));
        let mut fut = Some(fut);
        debug!("DynamicTaskExecutor::submit: job_id={job_id} max_other_parallel={max_other_parallel:?} parent={parent:?}");

        loop {
            self.wait_until_accepting(parent.as_ref()).await?;

            let permit = match tokio::time::timeout(LOCK_TIMEOUT, self.tx.reserve()).await {
                Ok(Ok(permit)) => permit,
                Ok(Err(e)) => {
                    error!("DynamicTaskExecutor::submit: reserve job_id={job_id} failed: e={e:?}");
                    if let Some(job_id) = job_id_to_rerun {
                        self.clear_exclusive_job_id(job_id);
                    }
                    return Err(ThoError::new_with_error("Can't submit job", Cause::SubmitErrorClosed));
                }
                Err(e) => {
                    error!("DynamicTaskExecutor::submit: reserve job_id={job_id} timed out: e={e:?}");
                    if let Some(job_id) = job_id_to_rerun {
                        self.clear_exclusive_job_id(job_id);
                    }
                    return Err(ThoError::new("Can't submit job: timed out"));
                }
            };

            if !*self.accepting_tx.borrow() {
                drop(permit);
                if parent.is_some() {
                    if let Some(job_id) = job_id_to_rerun {
                        self.clear_exclusive_job_id(job_id);
                    }
                    return Err(Self::resetting_submit_error());
                }
                continue;
            }

            let mut suspend_guard = ParentSuspendGuard::new(parent.clone());
            suspend_guard.maybe_suspend().await?;

            if !*self.accepting_tx.borrow() {
                drop(permit);
                let _ = suspend_guard.resume().await;
                if let Some(job_id) = job_id_to_rerun {
                    self.clear_exclusive_job_id(job_id);
                }
                if parent.is_some() {
                    return Err(Self::resetting_submit_error());
                }
                continue;
            }

            let accept_guard = match self.accept_gate.try_read() {
                Ok(guard) => guard,
                Err(_) => {
                    drop(permit);
                    let _ = suspend_guard.resume().await;
                    if let Some(job_id) = job_id_to_rerun {
                        self.clear_exclusive_job_id(job_id);
                    }
                    if parent.is_some() {
                        return Err(Self::resetting_submit_error());
                    }
                    continue;
                }
            };

            let (result_tx, result_rx) = oneshot::channel::<Result<T, ThoError>>();
            let exclusive_job_id = Arc::clone(&self.exclusive_job_id);
            let rerun_job_id = job_id_to_rerun;
            let cmd_tx = self.cmd_tx.clone();

            let fut = fut.take().expect("future must only be moved into a submitted job");
            let wrapped = async move {
                let result = fut.await;

                let needs_reset = result.as_ref().is_err_and(|e| e.needs_cuda_ctx_reset());
                if needs_reset {
                    warn!("DynamicTaskExecutor::submit: needs_reset");
                }
                if needs_reset && cmd_tx.send(ManagerCmd::RequestCudaReset).is_err() {
                    error!("DynamicTaskExecutor::submit: failed to request CUDA context reset for job_id={job_id}");
                }

                let is_oom = result.as_ref().is_err_and(|e| e.is_oom());
                let mut exclusive_job_id = exclusive_job_id.write();

                let result = if is_oom && max_other_parallel == 0 {
                    exclusive_job_id.replace(job_id);
                    result.map_err(|e| ThoError::new_with_error(e.msg.clone(), Cause::JobError { job_id, inner: Box::new(e.clone()) }))

                } else {
                    if rerun_job_id == Some(job_id) && exclusive_job_id.as_ref() == Some(&job_id) {
                        exclusive_job_id.take();
                    }
                    result
                };

                debug!("DynamicTaskExecutor::submit: wrapped  result={result:?}");

                let _ = result_tx.send(result);

                is_oom.into()
            };

            permit.send(Job::new(job_id, max_other_parallel, wrapped));
            drop(accept_guard);

            debug!("DynamicTaskExecutor::submit: submitted job_id={job_id}");

            let result = result_rx.await?;//TODO if wrapped panics this fails with "channel closed"

            suspend_guard.resume().await?;
            return result;
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
        if self.cmd_tx.send(ManagerCmd::RequestCudaReset).is_err() {
            error!("DynamicTaskExecutor::request_cuda_reset: manager is closed");
        }
    }

    pub fn set_max_parallelism(
        &self,
        new_limit: NonZeroUsize,
    ) -> Result<(), SetLimitError> {
        self.limit_tx
            .send(new_limit.get())
            .map_err(|_| SetLimitError::Closed)
    }

    pub async fn shutdown(self) {
        drop(self.tx);
        drop(self.limit_tx);
        drop(self.cmd_tx);
        let _ = self.worker.await;
    }

    async fn run_manager(
        mut rx: mpsc::Receiver<Job>,
        _limit_rx: watch::Receiver<usize>,
        mut cmd_rx: mpsc::UnboundedReceiver<ManagerCmd>,
        cmd_tx_for_tasks: mpsc::UnboundedSender<ManagerCmd>,
        exclusive_job_id: Arc<RwLock<Option<JobId>>>,
        accepting_tx: watch::Sender<bool>,
        accept_gate: Arc<tokio::sync::RwLock<()>>,
        reset_device_idx: Option<usize>,
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
        let _limit_closed = false;
        let global_limit = Arc::new(AtomicU64::new(2)); //*limit_rx.borrow();
        let mut reset_in_progress = false;
        let mut accept_gate_guard = None::<tokio::sync::OwnedRwLockWriteGuard<()>>;

        loop {
            debug!("DynamicTaskExecutor::run_manager, loop: active_caps={active_caps:?} running.len={}, pending.len={} global_limit={global_limit:?}", running.len(), pending.len());
            Self::drain_finished(
                &mut running,
                &mut done_rx,
                &mut jobs,
                &mut active_caps,
                &mut active_count,
            );

            let current_exclusive_job_id = *exclusive_job_id.read();
            Self::try_resume_waiters(
                &mut jobs,
                &mut active_caps,
                &mut active_count,
                global_limit.load(Relaxed) as usize,
                current_exclusive_job_id,
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

            if reset_in_progress && pending.is_empty() && jobs.is_empty() {
                if let Some(device_idx) = reset_device_idx {
                    if let Err(e) = reset_fn(device_idx) {
                        error!("DynamicTaskExecutor::run_manager: CUDA context reset failed for device_idx={device_idx}: e={e:?}");
                    }
                } else {
                    warn!("DynamicTaskExecutor::run_manager: CUDA context reset requested but executor has no bound device");
                }

                accept_gate_guard.take();
                reset_in_progress = false;
                let _ = accepting_tx.send(true);
            }

            if input_closed && pending.is_empty() && jobs.is_empty() && !reset_in_progress {
                break;
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
                            match cmd {
                                ManagerCmd::Suspend { job_id, reply } => {
                                    Self::handle_suspend(
                                        &mut jobs,
                                        &mut active_caps,
                                        &mut active_count,
                                        job_id,
                                    );
                                    let _ = reply.send(());
                                }
                                ManagerCmd::Resume { job_id, reply } => {
                                    Self::handle_resume(
                                        &mut jobs,
                                        &mut active_caps,
                                        &mut active_count,
                                        global_limit.load(Relaxed) as usize,
                                        *exclusive_job_id.read(),
                                        job_id,
                                        reply,
                                    );
                                }
                                ManagerCmd::RequestCudaReset => {
                                    if !reset_in_progress {
                                        reset_in_progress = true;
                                        let _ = accepting_tx.send(false);
                                        accept_gate_guard = Some(Arc::clone(&accept_gate).write_owned().await);
                                    }
                                }
                                ManagerCmd::Recheck => {}
                            }
                        }
                        None => {
                            cmd_closed = true;
                        }
                    }
                }

                // changed = limit_rx.changed(), if !limit_closed => {
                //     match changed {
                //         Ok(()) => {
                //             global_limit = *limit_rx.borrow_and_update();
                //         }
                //         Err(_) => {
                //             limit_closed = true;
                //         }
                //     }
                // }

                join_result = running.join_next(), if !running.is_empty() => {
                    if let Some(Err(err)) = join_result {
                        error!("job failed: {err}");
                    }
                }
            }
        }

        while let Some(res) = running.join_next().await {
            if let Err(err) = res {
                error!("job failed during shutdown: {err}");
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
                        global_limit.load(Relaxed) as usize,
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
                    running.spawn(async move {
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

                        match CURRENT_JOB
                            .scope(
                                CurrentJob {
                                    job_id,
                                    cmd_tx: cmd_tx_for_tasks,
                                },
                                async move {
                                    job.run().await
                                },
                            ).await {
                            IsOOM::Yes => {
                                if global_limit.load(Relaxed) > 6 {//TODO config
                                    global_limit.fetch_sub(1, Ordering::SeqCst);
                                }
                            }
                            IsOOM::No => {
                                if global_limit.load(Relaxed)  < 32 {//TODO config
                                    global_limit.fetch_add(1, Ordering::SeqCst);
                                }
                            }
                        }
                    });

                    spawned_any = true;
                    break;
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

    fn current_group_cap(
        global_limit: usize,
        active_caps: &BTreeMap<usize, usize>,
    ) -> usize {
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
    ) {
        let Some(job) = jobs.get_mut(&job_id) else {
            let _ = reply.send(());
            return;
        };

        // Every nested submit() that suspended this parent must wait until the
        // parent becomes active again before returning.
        job.waiting_resumes.push(reply);

        if job.suspend_depth == 0 {
            if job.active {
                Self::flush_waiting_resumes(job);
            } else if Self::can_activate_now(job_id, *active_count, global_limit, active_caps, job.cap, exclusive_job_id) {
                job.active = true;
                Self::add_active(active_caps, active_count, job.cap);
                Self::flush_waiting_resumes(job);
            }
            return;
        }

        job.suspend_depth -= 1;

        if job.suspend_depth == 0
            && Self::can_activate_now(job_id, *active_count, global_limit, active_caps, job.cap, exclusive_job_id)
        {
            job.active = true;
            Self::add_active(active_caps, active_count, job.cap);
            Self::flush_waiting_resumes(job);
        }
    }

    fn try_resume_waiters(
        jobs: &mut HashMap<JobId, LiveJob>,
        active_caps: &mut BTreeMap<usize, usize>,
        active_count: &mut usize,
        global_limit: usize,
        exclusive_job_id: Option<JobId>,
    ) {
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

                if Self::can_activate_now(job_id, *active_count, global_limit, active_caps, cap, exclusive_job_id) {
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

    pub fn spawn<F>(future: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        F::Output: Send + 'static,
    {
        let current_job = CURRENT_JOB.get();
        tokio::spawn(async move {
            CURRENT_JOB.scope(current_job, future).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{StreamExt, TryStreamExt};
    use parking_lot::Mutex;
    use std::{sync::Arc, time::Duration};
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
    use rand::prelude::IndexedRandom;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_returns_value() -> Result<(), ThoError> {
        let exec = DynamicTaskExecutor::new(64, NonZeroUsize::new(4).unwrap());

        let x = exec
            .submit(0, async move {
                            tokio::time::sleep(Duration::from_millis(50)).await;
                            Ok(123)
                        },
                    None)
            .await?;

        assert_eq!(x, 123);

        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_exhaust_runner() -> Result<(), ThoError> {
        let exec = Arc::new(DynamicTaskExecutor::new(64, NonZeroUsize::new(4).unwrap()));

        let x = Arc::clone(&exec).submit(0, async move {
                     tokio::spawn(async {
                        futures::stream::iter(0..10).map(|i|{
                            async move {
                                debug!("inner {i}");
                                Ok::<_, ThoError>(i)
                            }
                        })
                        .buffer_unordered(10)
                        .try_collect::<Vec<_>>()
                        .await?;

                         Ok(())
                     }).await?

                },
                                         None).await?;

        debug!("x={x:?}");
        // assert_eq!(x, 1);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_inner_spawn() -> Result<(), ThoError> {
        let exec = Arc::new(DynamicTaskExecutor::new(64, NonZeroUsize::new(4).unwrap()));

        let x = Arc::clone(&exec).submit(0, async move {

                        futures::stream::iter(0..10).map(|i|{
                            let exec = Arc::clone(&exec);

                            async move {
                                DynamicTaskExecutor::spawn(async move {

                                    Arc::clone(&exec).submit(0, async move {
                                        debug!("inner {i}");
                                        Ok::<_, ThoError>(i)
                                    }, None).await

                                }).await?

                            }
                        })
                            .buffer_unordered(10)
                            .try_collect::<Vec<_>>()
                            .await


                },
                                         None).await?;

        debug!("x={x:?}");
        // assert_eq!(x, 1);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_returns_error() -> Result<(), ThoError> {
        let exec = DynamicTaskExecutor::new(64, NonZeroUsize::new(4).unwrap());

        let x:Result<u32, ThoError> = exec
            .submit(0, async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Err::<u32, ThoError>(ThoError::new("test"))
            }, None)
            .await;


        debug!("x={x:?}");

        let Err(e) = x else {
            panic!("not an error")
        };
        assert_eq!("test", e.msg);


        let x:Result<u32, ThoError> = exec
            .submit(0, async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                Err::<u32, ThoError>(ThoError::oom())
            }, None)
            .await;

        debug!("x={x:?}");

        let Err(e) = x else {
            panic!("not an error")
        };
        assert!(e.is_oom());


        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_cuda_reset_blocks_new_jobs_until_running_jobs_finish() -> Result<(), ThoError> {
        let reset_calls = Arc::new(AtomicUsize::new(0));
        let reset_active = Arc::new(AtomicBool::new(false));
        let exec = Arc::new(DynamicTaskExecutor::new_with_reset_hook(
            64,
            NonZeroUsize::new(4).unwrap(),
            1,
            {
                let reset_calls = Arc::clone(&reset_calls);
                let reset_active = Arc::clone(&reset_active);
                Arc::new(move |_device_idx| {
                    reset_calls.fetch_add(1, Ordering::SeqCst);
                    reset_active.store(true, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(50));
                    reset_active.store(false, Ordering::SeqCst);
                    Ok(())
                })
            },
        ));

        let (running_started_tx, running_started_rx) = oneshot::channel();
        let (release_running_tx, release_running_rx) = oneshot::channel();
        let running_job = {
            let exec = Arc::clone(&exec);
            tokio::spawn(async move {
                exec.submit(usize::MAX, async move {
                    let _ = running_started_tx.send(());
                    let _ = release_running_rx.await;
                    Ok::<_, ThoError>(())
                }, None).await
            })
        };

        let _ = running_started_rx.await;

        let reset_err: Result<(), ThoError> = exec.submit(usize::MAX, async move {
            Err(ThoError::new_with_error(
                "needs reset",
                Cause::Cuda {
                    needs_reset: true,
                    msg: "reset me".into(),
                },
            ))
        }, None).await;
        assert!(reset_err.as_ref().is_err_and(|e| e.needs_cuda_ctx_reset()));

        let late_started = Arc::new(AtomicBool::new(false));
        let late_started_during_reset = Arc::new(AtomicBool::new(false));
        let late_job = {
            let exec = Arc::clone(&exec);
            let late_started = Arc::clone(&late_started);
            let late_started_during_reset = Arc::clone(&late_started_during_reset);
            let reset_active = Arc::clone(&reset_active);
            tokio::spawn(async move {
                exec.submit(usize::MAX, async move {
                    late_started_during_reset.store(reset_active.load(Ordering::SeqCst), Ordering::SeqCst);
                    late_started.store(true, Ordering::SeqCst);
                    Ok::<_, ThoError>(())
                }, None).await
            })
        };

        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(!late_started.load(Ordering::SeqCst));
        assert_eq!(reset_calls.load(Ordering::SeqCst), 0);

        let _ = release_running_tx.send(());
        running_job.await??;
        late_job.await??;

        assert_eq!(reset_calls.load(Ordering::SeqCst), 1);
        assert!(!late_started_during_reset.load(Ordering::SeqCst));

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_nested_submit_no_deadlock() -> Result<(), ThoError> {
        let exec = Arc::new(DynamicTaskExecutor::new(64, NonZeroUsize::new(4).unwrap()));

        let results = futures_util::stream::iter(0..3)
            .map(|i| {
                let exec = Arc::clone(&exec);
                async move {
                    let exec_for_inner = Arc::clone(&exec);

                    let values = exec
                        .submit(0, async move {
                            Ok(futures_util::stream::iter(0..3)
                                .map(|j| {
                                    let exec_for_inner = Arc::clone(&exec_for_inner);
                                    async move {
                                        let v = exec_for_inner
                                            .submit(usize::MAX, async move {
                                                tokio::time::sleep(Duration::from_millis(20)).await;
                                                Ok(i * j)
                                            }, None)
                                            .await?;
                                        Ok::<i32, ThoError>(v)
                                    }
                                })
                                .buffer_unordered(3)
                                .try_collect::<Vec<i32>>()
                                .await?)
                        }, None)
                        .await?;

                    let sum = values.into_iter().sum::<i32>();
                    Ok::<i32, ThoError>(sum)
                }
            })
            .buffer_unordered(3)
            .collect::<Vec<Result<i32, ThoError>>>()
            .await;

        let sums: Vec<i32> = results.into_iter().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(sums.len(), 3);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_load_ahead() -> Result<(), ThoError> {
        let exec = Arc::new(DynamicTaskExecutor::new(4, NonZeroUsize::new(4).unwrap()));

        let results = futures_util::stream::iter(0..100)
            .map(|i| {
                let exec = Arc::clone(&exec);
                async move {
                    let exec_for_inner = Arc::clone(&exec);

                    let max_par = *vec![0, 0, 0, 10, 100, usize::MAX].choose(&mut rand::rng()).unwrap_or(&0);
                    let sum = exec
                        .submit(max_par, async move {
                            tokio::time::sleep(Duration::from_millis(10)).await;

                            Ok(futures_util::stream::iter(0..12)
                                .map(|j| {
                                    let exec_for_inner = Arc::clone(&exec_for_inner);

                                    async move {
                                        let max_par_inner = *vec![0, 0, 0, 10, 100, usize::MAX].choose(&mut rand::rng()).unwrap_or(&0);
                                        exec_for_inner
                                            .submit(max_par_inner, async move {
                                                tokio::time::sleep(Duration::from_millis(10)).await;
                                               Ok(i * j)
                                            }, None)
                                            .await?;
                                        Ok::<i32, ThoError>(j)
                                    }
                                })
                                .buffer_unordered(3)
                                .try_collect::<Vec<i32>>()
                                .await)
                        }, None)
                        .await?
                        .into_iter()
                        .flat_map(|x|x.into_iter())
                        .sum::<i32>();

                    Ok::<i32, ThoError>(sum)
                }
            })
            .buffer_unordered(3)
            .collect::<Vec<Result<i32, ThoError>>>()
            .await;

        let sums: Vec<i32> = results.into_iter().collect::<Result<Vec<_>, _>>()?;
        // assert_eq!(sums, vec![66, 66, 66]);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_multistep() -> Result<(), ThoError> {
        async fn some_fn_1(i:i32, j:i32){
            println!("xxxxxxxxxxxx__{i}_{j}");
        }

        async fn some_fn_2(i:i32, j:i32){
            println!("______________{i}_{j}");
        }



        let exec = Arc::new(DynamicTaskExecutor::new(4, NonZeroUsize::new(4).unwrap()));
        let exec_1 = Arc::clone(&exec);


        for z in 0..10 {
            let exec = Arc::clone(&exec);
            let exec_1 = Arc::clone(&exec);

            let _ = tokio::join!(
                async move {
                    tokio::spawn(async move{

                        futures::stream::iter(1..10).for_each_concurrent(10, move |i|   {
                            let exec_1 = Arc::clone(&exec);
                            async move {
                                tokio::spawn(async move {


                                    exec_1.submit(usize::MAX, async move {

                                        for j in 0..10 {
                                            some_fn_1(i, j).await;
                                            tokio::task::yield_now().await;
                                        }

                                        tokio::time::sleep(Duration::from_millis(500)).await;

                                        Ok::<_, ThoError>(())
                                    }, None).await.expect("1");

                                }).await.unwrap();
                            }

                        }).await
                    }).await
                },
                async move {
                    tokio::spawn(async move {
                        futures::stream::iter(1..10).for_each_concurrent(10,move |i|   {
                            let exec_1 = Arc::clone(&exec_1);
                            async move {
                                tokio::spawn(async move {
                                exec_1.submit(0, async move {

                                    for j in 0..10 {
                                        some_fn_2(i, j).await;
                                        tokio::task::yield_now().await;
                                    }

                                    tokio::time::sleep(Duration::from_millis(500)).await;

                                    Ok::<_, ThoError>(())
                                }, None).await.expect("1");

                                }).await.unwrap();
                            }

                        }).await
                    }).await
                }
            );

        }

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_exec_0() -> Result<(), ThoError> {
        let exec = Arc::new(DynamicTaskExecutor::new(100, NonZeroUsize::new(4).unwrap()));
        let starts = Arc::new(Mutex::new(Vec::new()));

        futures::stream::iter(0..20).for_each_concurrent(10, |i| {
            let exec = Arc::clone(&exec);
            let starts = Arc::clone(&starts);
            async move {
                tokio::task::spawn(async move {
                    let max_other_parallel = if i == 3 {
                        0
                    } else {
                        usize::MAX
                    };
                    let mut job_id_to_rerun = None::<u64>;

                    loop {
                        tokio::time::sleep(Duration::from_millis(100)).await;

                        match exec.submit(max_other_parallel, {
                            let starts = Arc::clone(&starts);
                            async move {
                                debug!("starting  i={i}");
                                starts.lock().push(i);

                                if i == 3 {
                                    static ATTEMPTS: AtomicU8 = AtomicU8::new(0);

                                    if ATTEMPTS.fetch_add(1, Ordering::Relaxed) < 2 {
                                        debug!("triggering error i={i} {}", ATTEMPTS.load(Ordering::Relaxed));
                                        Err(ThoError::oom())
                                    } else {
                                        debug!("finishing i={i} after err");
                                        Ok::<_, ThoError>(())
                                    }
                                } else {
                                    debug!("finishing i={i}");
                                    Ok::<_, ThoError>(())
                                }
                            }
                        }, job_id_to_rerun).await {
                            Ok(_) => {
                                debug!("done i={i}");
                                break;
                            }
                            Err(e) => {
                                match &e.what {
                                    Cause::JobError { job_id, inner } => {
                                        if inner.is_oom() {
                                            job_id_to_rerun.replace(*job_id);
                                        } else {
                                            panic!("unexpected inner={inner:?}")
                                        }
                                    }
                                    _ => {
                                        panic!("all failed")
                                    }
                                }
                                debug!("err i={i} is_oom={}", e.is_oom());
                            }
                        };
                    }
                }).await.unwrap();
            }
        }).await;

        let starts = starts.lock().clone();
        let first_three = starts.iter().position(|i| *i == 3).expect("job 3 must run");
        assert_eq!(&starts[first_three..first_three + 3], &[3, 3, 3]);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }



}
