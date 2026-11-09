//! Logical scheduling ownership, separate from physical CUDA activity leases.
use std::future::Future;
use tokio::sync::{mpsc, oneshot};
use crate::{CudaActivityScope, SyncStageRunnerError, ThoError};
use crate::tracked_await::t;

type JobId = u64;

#[derive(Clone, Debug)]
pub struct CurrentJob {
    pub job_id: JobId,
    pub cmd_tx: mpsc::UnboundedSender<ManagerCmd>,
}

/// A family identifies dependencies across runners and spawned tasks. Only the
/// currently scheduled frame owns capacity; children of a suspended wait must
/// not try to resume that frame individually before their siblings complete.
#[derive(Clone, Debug, Default)]
pub struct JobContext {
    pub family_id: Option<JobId>,
    pub job: Option<CurrentJob>,
}

tokio::task_local! { static CONTEXT: JobContext; }

impl JobContext {
    pub fn current() -> Self { CONTEXT.try_with(Clone::clone).unwrap_or_default() }

    pub fn for_job(family_id: JobId, job: CurrentJob) -> Self {
        Self { family_id: Some(family_id), job: Some(job) }
    }

    pub fn dependencies(&self) -> Self {
        Self { family_id: self.family_id, job: None }
    }

    pub async fn run<F: Future>(self, future: F) -> F::Output {
        CONTEXT.scope(self, future).await
    }

    /// Suspend once for the entire dependency wait, retaining the family in
    /// spawned children. CUDA leases are deliberately untouched. Resume before
    /// returning even when the dependency failed; Drop compensates cancellation.
    pub async fn wait<F: Future>(self, future: F) -> Result<F::Output, ThoError> {
        let mut suspended = ParentSuspendGuard::new(self.job);
        suspended.maybe_suspend().await?;
        let result = Self { family_id: self.family_id, job: None }.run(future).await;
        suspended.resume().await?;
        Ok(result)
    }
}

#[derive(Debug)]
pub enum ManagerCmd {
    Suspend { job_id: JobId, reply: oneshot::Sender<()> },
    Resume { job_id: JobId, reply: oneshot::Sender<()>, for_unwind: bool },
    DrainForRecovery { reply: oneshot::Sender<()> },
    Recheck,
}


/// Temporarily removes a parent job from the manager's active-capacity accounting
/// while that job awaits a nested `submit`.
///
/// The parent Tokio task is not paused: it is already blocked waiting for its child.
/// Without removing it from `active_count`/`active_caps`, an exclusive parent (or a
/// parent whose cap is already full) can prevent the child from ever starting. The
/// parent would then wait for the child while the child waits for the parent's slot.
///
/// `resume` waits for the manager to count the parent as active again before the
/// nested `submit` returns. Once Suspend is queued, Drop owes exactly one Resume,
/// including cancellation before either acknowledgement has been received.
pub struct ParentSuspendGuard {
    parent: Option<CurrentJob>,
    suspended: bool,
}

fn executor_closed_error(msg: impl Into<String>) -> ThoError {
    ThoError::new_with_cause(msg, ThoError::SyncStageRunner(SyncStageRunnerError::SubmitErrorClosed))
}

impl ParentSuspendGuard {
    pub fn new(parent: Option<CurrentJob>) -> Self {
        Self { parent, suspended: false }
    }

    pub async fn maybe_suspend(&mut self) -> Result<(), ThoError> {
        let Some(parent) = &self.parent else {
            return Ok(());
        };

        let (ack_tx, ack_rx) = oneshot::channel();
        if parent.cmd_tx.send(ManagerCmd::Suspend { job_id: parent.job_id, reply: ack_tx }).is_err() {
            return Err(executor_closed_error("executor is closed"));
        }

        // Sending is synchronous: arm compensation before the first cancellation point.
        self.suspended = true;
        if t!(ack_rx).await.is_err() {
            return Err(executor_closed_error("executor is closed"));
        }

        Ok(())
    }

    pub async fn resume(&mut self) -> Result<(), ThoError> {
        if !self.suspended {
            return Ok(());
        }

        let Some(parent) = &self.parent else {
            self.suspended = false;
            return Ok(());
        };

        let (ack_tx, ack_rx) = oneshot::channel();
        if parent
            .cmd_tx
            .send(ManagerCmd::Resume {
                job_id: parent.job_id,
                reply: ack_tx,
                for_unwind: CudaActivityScope::current().is_some_and(|scope| scope.check_dependencies().is_err()),
            })
            .is_err()
        {
            return Err(executor_closed_error("executor is closed"));
        }

        // The manager now owns the resume. Drop must not enqueue a second one while
        // we await capacity, since that could release another nested submission's slot.
        self.suspended = false;
        if t!(ack_rx).await.is_err() {
            return Err(executor_closed_error("executor is closed"));
        }

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

        // Unbounded sends do not await or need a runtime. FIFO ordering pairs this
        // command with our queued Suspend even when its acknowledgement was dropped.
        let (reply, _) = oneshot::channel();
        let _ = parent.cmd_tx.send(ManagerCmd::Resume {
            job_id: parent.job_id,
            reply,
            for_unwind: CudaActivityScope::current().is_some_and(|scope| scope.check_dependencies().is_err()),
        });
    }
}

