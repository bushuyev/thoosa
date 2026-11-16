//! Bounded blocking work without extending borrowed stage/input lifetimes.
use anyhow::Context as _;
use std::sync::{Arc, LazyLock};
use thoosa_shared::ThoError;
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::sync::{Semaphore, SemaphorePermit};

pub struct BlockingPool {
    slots: Arc<Semaphore>,
}

impl BlockingPool {
    pub fn new(parallelism: usize) -> Self {
        assert!(parallelism > 0, "blocking work requires at least one slot");
        Self { slots: Arc::new(Semaphore::new(parallelism)) }
    }

    /// Owned CPU/I/O work can outlive cancellation of its waiter. The closure
    /// owns the permit as well, so cancellation cannot admit excess work.
    pub async fn run_owned<T: Send + 'static>(&self, work: impl FnOnce() -> Result<T, ThoError> + Send + 'static) -> Result<T, ThoError> {
        let slot = Arc::clone(&self.slots).acquire_owned().await.context("acquire blocking work slot")?;
        let span = tracing::Span::current();
        tokio::task::spawn_blocking(move || {
            let _slot = slot;
            span.in_scope(work)
        })
        .await
        .context("join blocking work")?
    }

    /// Stage inputs and mapped state guards borrow their owners. Keep this
    /// scope synchronous once entered: cancellation must never detach CUDA work
    /// from those borrows, the stream lease, or the caller's recovery activity.
    pub(crate) async fn acquire(&self) -> Result<BlockingPermit<'_>, ThoError> {
        Ok(BlockingPermit { _slot: self.slots.acquire().await.context("acquire synchronous stage slot")? })
    }
}

pub(crate) struct BlockingPermit<'a> {
    _slot: SemaphorePermit<'a>,
}

impl BlockingPermit<'_> {
    pub(crate) fn run<T>(self, work: impl FnOnce() -> T) -> T {
        if Handle::current().runtime_flavor() == RuntimeFlavor::MultiThread {
            // Both application runtimes are multithreaded. Tokio hands off the
            // scheduler before blocking; task locals and panic propagation stay
            // intact, and the caller binds/drops its CUDA guard inside `work`.
            tokio::task::block_in_place(work)
        } else {
            // Preserve support for current-thread embedders/tests. Tokio cannot
            // hand off that scheduler, and borrowed work cannot be detached
            // safely. This compatibility path remains synchronous.
            work()
        }
    }
}

// This additional host-side bound does not replace adaptive GPU admission or
// stream leases. Leave blocking-thread capacity for file I/O and recovery too.
pub(crate) static STAGE_BLOCKING_POOL: LazyLock<BlockingPool> = LazyLock::new(|| {
    let cpus = std::thread::available_parallelism().map_or(2, |cpus| cpus.get());
    BlockingPool::new((cpus / 2).clamp(1, 8))
});

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::sync::oneshot;

    tokio::task_local! {
        static RECOVERY_CONTEXT: usize;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_borrowed_work_preserves_context_and_ownership_without_starving_scheduler() -> anyhow::Result<()> {
        let pool = Arc::new(BlockingPool::new(1));
        let state = Arc::new(tokio::sync::Mutex::new(0));
        let (entered_tx, entered_rx) = oneshot::channel();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel();
        let job_pool = Arc::clone(&pool);
        let job_state = Arc::clone(&state);
        let job = tokio::spawn(RECOVERY_CONTEXT.scope(7, async move {
            let mut state = job_state.lock().await;
            job_pool.acquire().await?.run(|| {
                assert_eq!(RECOVERY_CONTEXT.get(), 7, "blocking must retain the admitted task context");
                entered_tx.send(()).expect("test is waiting for execution");
                finish_rx.recv_timeout(Duration::from_secs(5)).expect("Tokio scheduler was starved by synchronous work");
                *state = 1;
                Ok::<_, ThoError>(())
            })
        }));
        entered_rx.await?;
        job.abort();
        // This must run on Tokio's sole scheduler worker while the stage is
        // blocked. Cancellation cannot release the stage's state or permit yet.
        let observed_state = Arc::clone(&state);
        let observed_pool = Arc::clone(&pool);
        tokio::spawn(async move {
            assert!(observed_state.try_lock().is_err(), "cancelled caller released state during execution");
            assert_eq!(observed_pool.slots.available_permits(), 0);
            finish_tx.send(()).expect("blocking stage is still waiting");
        }).await?;
        // An abort racing a synchronous completion can yield either result;
        // joining still guarantees the borrowed state is no longer in use.
        match job.await {
            Ok(result) => result?,
            Err(error) => assert!(error.is_cancelled(), "unexpected stage failure: {error:?}"),
        }
        assert_eq!(*state.lock().await, 1, "admitted work must finish before its borrowed state is released");
        assert_eq!(pool.slots.available_permits(), 1);
        Ok(())
    }

    #[tokio::test]
    async fn test_owned_work_retains_capacity_after_waiter_cancellation() -> anyhow::Result<()> {
        let pool = Arc::new(BlockingPool::new(1));
        let (entered_tx, entered_rx) = oneshot::channel();
        let (finish_tx, finish_rx) = std::sync::mpsc::channel();
        let job_pool = Arc::clone(&pool);
        let job = tokio::spawn(async move {
            job_pool.run_owned(move || {
                entered_tx.send(()).expect("test is waiting for file work");
                finish_rx.recv_timeout(Duration::from_secs(5)).expect("test releases file work");
                Ok(())
            }).await
        });
        entered_rx.await?;
        job.abort();
        assert!(job.await.unwrap_err().is_cancelled());
        assert_eq!(pool.slots.available_permits(), 0, "detached blocking work must still count against the limit");

        let mut cancelled = Box::pin(pool.run_owned(|| -> Result<(), ThoError> { panic!("unadmitted work must not start") }));
        assert!(futures_util::poll!(cancelled.as_mut()).is_pending());
        drop(cancelled);
        finish_tx.send(())?;
        assert_eq!(tokio::time::timeout(Duration::from_secs(5), pool.run_owned(|| Ok(42))).await??, 42);
        assert_eq!(pool.slots.available_permits(), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 1)]
    async fn test_blocking_failure_preserves_errors_and_returns_capacity() -> anyhow::Result<()> {
        let pool = BlockingPool::new(1);
        assert!(pool.run_owned(|| Err::<(), _>(ThoError::oom())).await.unwrap_err().is_oom());
        let slot = pool.acquire().await?;
        let panic: std::thread::Result<()> = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| slot.run(|| std::panic::panic_any("submission panic"))));
        assert_eq!(panic.unwrap_err().downcast_ref::<&str>(), Some(&"submission panic"));
        assert_eq!(pool.slots.available_permits(), 1, "unwinding must release the execution slot");
        Ok(())
    }
}
