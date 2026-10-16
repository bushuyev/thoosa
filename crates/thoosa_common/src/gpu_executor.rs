use anyhow::Context as _;
use futures::executor::ThreadPool;
use futures_util::FutureExt;
use std::future::{Future, poll_fn};
use std::panic::AssertUnwindSafe;
use std::sync::OnceLock;
use thoosa_shared::{CudaActivityScope, ThoError};
use tracing::Instrument;

// Match the execution-stream limit. These threads are separate from Tokio's
// scheduler and blocking pool, so a synchronizing GPU cannot starve IPC or copies.
const EXECUTION_THREADS: usize = 32;
static EXECUTOR: OnceLock<Result<ThreadPool, ThoError>> = OnceLock::new();

pub(crate) async fn run<F>(future: F) -> Result<F::Output, ThoError>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let pool = EXECUTOR.get_or_init(|| {
        Ok(ThreadPool::builder().pool_size(EXECUTION_THREADS).name_prefix("thoosa-gpu-").create()
            .context("create GPU execution threads")?)
    }).as_ref().map_err(Clone::clone)?;
    run_on(pool, future).await
}

async fn run_on<F>(pool: &ThreadPool, future: F) -> Result<F::Output, ThoError>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let runtime = tokio::runtime::Handle::current();
    let activity = CudaActivityScope::current();
    let future = crate::runner::inherit_current_job(async move {
        match activity {
            Some(activity) => activity.run(future).await,
            None => future.await,
        }
    });
    let (send, receive) = tokio::sync::oneshot::channel();
    let future = AssertUnwindSafe(future.instrument(tracing::Span::current())).catch_unwind();
    let mut future = Some(Box::pin(async move {
        let result = future.await;
        // An accepted job owns its borrows and activity until completion. A
        // dropped receiver must not cancel CUDA work or release its inputs early.
        if send.send(result).is_err() {
            tracing::debug!("GPU execution completed after its caller was cancelled");
        }
    }));
    // Enter the caller's runtime on every poll: timers and spawned copy tasks
    // still belong to that runtime, while pending preparation releases this
    // execution thread. No block_on bridge or extra runtime is needed.
    pool.spawn_ok(poll_fn(move |cx| {
        let _runtime = runtime.enter();
        let result = future.as_mut().expect("GPU job polled after completion").as_mut().poll(cx);
        if result.is_ready() {
            // Tensor/context destructors can spawn cleanup; drop them while the
            // caller's runtime is still entered, including cancelled replies.
            future.take();
        }
        result
    }));
    match receive.await.context("GPU execution thread dropped its result")? {
        Ok(output) => Ok(output),
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use thoosa_shared::CudaActivityGate;

    #[tokio::test(flavor = "current_thread")]
    async fn test_bounded_execution_keeps_tokio_responsive_and_drains_cancelled_jobs() {
        // One occupied execution thread must leave Tokio free, prevent a second
        // synchronous job from starting, and retain cancellation's recovery lease.
        let pool = ThreadPool::builder().pool_size(1).create().unwrap();
        let gate = Arc::new(CudaActivityGate::new(1));
        let activity = CudaActivityScope::for_operation(&gate).unwrap();
        let (started, start) = tokio::sync::oneshot::channel();
        let (release, finish) = std::sync::mpsc::channel();
        let mut first = Box::pin(activity.run(run_on(&pool, async move {
            assert!(CudaActivityScope::current().is_some());
            started.send(()).unwrap();
            finish.recv_timeout(Duration::from_secs(10)).unwrap();
        })));
        assert!(futures::poll!(first.as_mut()).is_pending());
        start.await.unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        let (second_started, mut second_start) = tokio::sync::oneshot::channel();
        let mut second = Box::pin(run_on(&pool, async move { second_started.send(()).unwrap(); }));
        assert!(futures::poll!(second.as_mut()).is_pending());
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(second_start.try_recv().is_err());
        drop(first);
        let pause = gate.request_pause(ThoError::ContextReset).unwrap();
        assert!(futures::poll!(Box::pin(gate.wait_until_idle()).as_mut()).is_pending());
        release.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(10), gate.wait_until_idle()).await.unwrap();
        second.await.unwrap();
        second_start.await.unwrap();
        gate.finish_pause(pause);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn test_pending_preparation_releases_execution_thread_and_preserves_runtime() {
        let pool = ThreadPool::builder().pool_size(1).create().unwrap();
        let (release, prepare) = tokio::sync::oneshot::channel();
        let mut first = Box::pin(run_on(&pool, async move {
            prepare.await.unwrap();
            tokio::time::sleep(Duration::from_millis(1)).await;
            tokio::spawn(async { 42 }).await.unwrap()
        }));
        assert!(futures::poll!(first.as_mut()).is_pending());
        // This job must run even though the first is waiting for preparation.
        tokio::time::timeout(Duration::from_secs(10), run_on(&pool, async move { release.send(()).unwrap(); })).await.unwrap().unwrap();
        assert_eq!(first.await.unwrap(), 42);
        let panic = AssertUnwindSafe(run_on(&pool, async { panic!("execution panic") })).catch_unwind().await;
        assert!(panic.is_err());
        assert_eq!(run_on(&pool, async { 7 }).await.unwrap(), 7, "panics must not kill execution threads");
    }
}
