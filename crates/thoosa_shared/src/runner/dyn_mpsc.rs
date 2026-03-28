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
use tokio::sync::watch::Receiver;
use tracing::{debug, warn};
use std::borrow::BorrowMut;
use std::sync::Arc;
use std::sync::atomic::Ordering::Relaxed;
use crate::{Cause, ThoError};

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
type ExecutorId = u64;

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
        self.fut.await
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

    async fn maybe_suspend(&mut self) -> Result<(), SubmitError> {
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

    async fn resume(&mut self) -> Result<(), SubmitError> {
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
    worker: JoinHandle<()>,
}

impl DynamicTaskExecutor {
    pub fn new(queue_capacity: usize, initial_limit: NonZeroUsize) -> Self {

        let (tx, rx) = mpsc::channel(queue_capacity);
        let (limit_tx, limit_rx) = watch::channel(initial_limit.get());
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel();

        let cmd_tx_for_tasks = cmd_tx.clone();

        let worker = tokio::spawn(async move {
            Self::run_manager(rx, limit_rx, cmd_rx, cmd_tx_for_tasks).await;
        });

        Self {
            tx,
            limit_tx,
            cmd_tx,
            worker,
        }
    }

    pub async fn submit<F, T>(
        &self,
        max_other_parallel: usize,
        fut: F,
    ) -> Result<T, ThoError>
    where
        F: Future<Output = Result<T, ThoError>> + Send + 'static,
        T: Send + Debug + 'static,
    {

        let parent = CURRENT_JOB.try_with(|job| job.clone()).ok();

        debug!("DynamicTaskExecutor::submit: max_other_parallel={max_other_parallel:?} parent={parent:?}");

        let mut suspend_guard = ParentSuspendGuard::new(parent);
        suspend_guard.maybe_suspend().await?;

        let job_id = NEXT_JOB_ID.fetch_add(1, Ordering::Relaxed);
        let (result_tx, result_rx) = oneshot::channel::<Result<T, ThoError>>();

        let wrapped = async move {
            let value = fut.await;
            debug!("DynamicTaskExecutor::submit: wrapped value={value:?}");
            let is_oom = value.as_ref().is_err_and(|e|e.is_oom());
            let _ = result_tx.send(value);

            is_oom.into()
        };

        if self
            .tx
            .send(Job::new(job_id, max_other_parallel, wrapped))
            .await
            .is_err()
        {
            let _ = suspend_guard.resume().await;
            return Err(ThoError::new_with_error("", Cause::SubmitErrorClosed));
        }

        let result = result_rx.await?;

        suspend_guard.resume().await?;
        result
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
        mut limit_rx: watch::Receiver<usize>,
        mut cmd_rx: mpsc::UnboundedReceiver<ManagerCmd>,
        cmd_tx_for_tasks: mpsc::UnboundedSender<ManagerCmd>,
    ) {
        let mut pending = VecDeque::<Job>::new();
        let mut running = JoinSet::<()>::new();

        let mut jobs = HashMap::<JobId, LiveJob>::new();
        let mut active_caps = BTreeMap::<usize, usize>::new();
        let mut active_count = 0usize;

        let (done_tx, mut done_rx) = mpsc::unbounded_channel::<JobId>();

        let mut input_closed = false;
        let mut limit_closed = false;
        let mut cmd_closed = false;
        let mut global_limit = Arc::new(AtomicU64::new(2)); //*limit_rx.borrow();

        loop {
            debug!("DynamicTaskExecutor::run_manager, loop: active_caps={active_caps:?} running.len={}, pending.len={} global_limit={global_limit:?}", running.len(), pending.len());
            Self::drain_finished(
                &mut running,
                &mut done_rx,
                &mut jobs,
                &mut active_caps,
                &mut active_count,
            );

            Self::try_resume_waiters(
                &mut jobs,
                &mut active_caps,
                &mut active_count,
                global_limit.load(Relaxed) as usize,
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
            );

            if input_closed && pending.is_empty() && jobs.is_empty() {
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
                                        job_id,
                                        reply,
                                    );
                                }
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
                        eprintln!("job failed: {err}");
                    }
                }
            }
        }

        while let Some(res) = running.join_next().await {
            if let Err(err) = res {
                eprintln!("job failed during shutdown: {err}");
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
    ) {
        debug!("DynamicTaskExecutor::try_spawn_ready_jobs: pending={pending:?}");

        loop {
            let mut spawned_any = false;
            let mut idx = 0usize;

            while idx < pending.len() {
                let can_spawn = {
                    let job = &pending[idx];
                    Self::can_activate_now(
                        *active_count,
                        global_limit.load(Relaxed) as usize,
                        active_caps,
                        job.total_parallel_cap(),
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
                                if global_limit.load(Relaxed) > 2 {//TODO config
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
        active_count: usize,
        global_limit: usize,
        active_caps: &BTreeMap<usize, usize>,
        new_cap: usize,
    ) -> bool {
        let next_total = active_count.saturating_add(1);
        let current_group_cap = Self::current_group_cap(global_limit, active_caps);

        let min = current_group_cap.min(new_cap);
        let r = next_total <= min;

        debug!("DynamicTaskExecutor::can_activate_now: active_count={active_count} global_limit={global_limit} min={min} r={r}");
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
            } else if Self::can_activate_now(*active_count, global_limit, active_caps, job.cap) {
                job.active = true;
                Self::add_active(active_caps, active_count, job.cap);
                Self::flush_waiting_resumes(job);
            }
            return;
        }

        job.suspend_depth -= 1;

        if job.suspend_depth == 0
            && Self::can_activate_now(*active_count, global_limit, active_caps, job.cap)
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

                if Self::can_activate_now(*active_count, global_limit, active_caps, cap) {
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
    use futures_util::StreamExt;
    use std::{sync::Arc, time::Duration};
    use rand::prelude::IndexedRandom;

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_return_value() -> Result<(), SubmitError> {
        let exec = DynamicTaskExecutor::new(64, NonZeroUsize::new(4).unwrap());

        let x = exec
            .submit(0, async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                123
            })
            .await?;

        assert_eq!(x, 123);

        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_nested_submit_no_deadlock() -> Result<(), SubmitError> {
        let exec = Arc::new(DynamicTaskExecutor::new(64, NonZeroUsize::new(4).unwrap()));

        let results = futures_util::stream::iter(0..3)
            .map(|i| {
                let exec = Arc::clone(&exec);
                async move {
                    let exec_for_inner = Arc::clone(&exec);

                    let values = exec
                        .submit(0, async move {
                            futures_util::stream::iter(0..3)
                                .map(|j| {
                                    let exec_for_inner = Arc::clone(&exec_for_inner);
                                    async move {
                                        let v = exec_for_inner
                                            .submit(usize::MAX, async move {
                                                tokio::time::sleep(Duration::from_millis(20)).await;
                                                i * j
                                            })
                                            .await?;
                                        Ok::<i32, SubmitError>(v)
                                    }
                                })
                                .buffer_unordered(3)
                                .collect::<Vec<Result<i32, SubmitError>>>()
                                .await
                        })
                        .await?;

                    let sum = values.into_iter().filter_map(Result::ok).sum::<i32>();
                    Ok::<i32, SubmitError>(sum)
                }
            })
            .buffer_unordered(3)
            .collect::<Vec<Result<i32, SubmitError>>>()
            .await;

        let sums: Vec<i32> = results.into_iter().collect::<Result<Vec<_>, _>>()?;
        assert_eq!(sums.len(), 3);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_load_ahead() -> Result<(), SubmitError> {
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

                            futures_util::stream::iter(0..12)
                                .map(|j| {
                                    let exec_for_inner = Arc::clone(&exec_for_inner);

                                    async move {
                                        let max_par_inner = *vec![0, 0, 0, 10, 100, usize::MAX].choose(&mut rand::rng()).unwrap_or(&0);
                                        exec_for_inner
                                            .submit(max_par_inner, async move {
                                                tokio::time::sleep(Duration::from_millis(10)).await;
                                                i * j
                                            })
                                            .await?;
                                        Ok::<i32, SubmitError>(j)
                                    }
                                })
                                .buffer_unordered(3)
                                .collect::<Vec<Result<i32, SubmitError>>>()
                                .await
                        })
                        .await?
                        .into_iter()
                        .filter_map(Result::ok)
                        .sum::<i32>();

                    Ok::<i32, SubmitError>(sum)
                }
            })
            .buffer_unordered(3)
            .collect::<Vec<Result<i32, SubmitError>>>()
            .await;

        let sums: Vec<i32> = results.into_iter().collect::<Result<Vec<_>, _>>()?;
        // assert_eq!(sums, vec![66, 66, 66]);

        let exec = Arc::try_unwrap(exec).expect("all Arc clones must be dropped");
        exec.shutdown().await;
        Ok(())
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_multistep() -> Result<(), SubmitError> {
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

                                        Ok::<_, SubmitError>(())
                                    }).await.expect("1").expect("2");

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

                                    Ok::<_, SubmitError>(())
                                }).await.expect("1").expect("2");

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
    async fn test_oom() -> Result<(), SubmitError> {
        let exec = DynamicTaskExecutor::new(64, NonZeroUsize::new(4).unwrap());

        let x = exec
            .submit(0, async move {
                tokio::time::sleep(Duration::from_millis(50)).await;
                123
            })
            .await?;

        assert_eq!(x, 123);

        exec.shutdown().await;
        Ok(())
    }

}