use std::{
    collections::BTreeMap,
    fmt::Debug,
    future::Future,
    sync::Arc,
};

use tokio::sync::watch;
use tokio::task::{JoinError, JoinSet};
use tracing::{debug, error};
use crate::{cuda_activity_gate, StreamBuffering, ThoError};
use crate::collect_futures_eta::{collect_eta_maybe_log, collect_eta_record_completion, CollectEtaInvocationGuard};

pub async fn collect_futures_with_progress<T, F>(
    futures: impl Iterator<Item = F>,
    len: usize,
    buffering: StreamBuffering,
    descr: &str,
) -> Vec<T>
where
    T: Debug + Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    ThoError: Send + 'static,
{
    debug!("collect_futures_with_progress: descr={descr} len={len}");
    let registration = crate::buffer_limits::buffer_limits().register(descr);
    collect_futures_with_progress_with_limit(futures, len, registration.receiver(), buffering, descr).await
}

async fn collect_futures_with_progress_with_limit<T, F>(
    futures: impl Iterator<Item = F>,
    len: usize,
    buffer_limit: Arc<watch::Receiver<usize>>,
    buffering: StreamBuffering,
    descr: &str,
) -> Vec<T>
where
    T: Debug + Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    ThoError: Send + 'static,
{
    let mut buffer_limit = buffer_limit.as_ref().clone();
    let initial_buffer_limit = *buffer_limit.borrow_and_update();
    assert!(initial_buffer_limit > 0, "buffer_limit must be > 0; JoinSet version cannot make progress with 0");

    let mut results = Vec::with_capacity(len);
    let (mut ok, mut err) = (0usize, 0usize);
    let total = len;

    let eta_concurrency = initial_buffer_limit.min(total.max(1));

    let eta_guard = (total > 0).then(|| {
        CollectEtaInvocationGuard::new(descr, total, eta_concurrency)
    });

    let eta_invocation_id = eta_guard.as_ref().map(CollectEtaInvocationGuard::id);

    let mut futures = futures.enumerate();
    let mut set = JoinSet::<(usize, Result<T, ThoError>)>::new();

    match buffering {
        StreamBuffering::Unordered => {
            collect_joinset_unordered(
                &mut set,
                &mut futures,
                &mut buffer_limit,
                initial_buffer_limit,
                &mut results,
                &mut ok,
                &mut err,
                total,
                descr,
                eta_invocation_id,
            )
                .await;
        }

        StreamBuffering::Ordered => {
            collect_joinset_ordered(
                &mut set,
                &mut futures,
                &mut buffer_limit,
                initial_buffer_limit,
                &mut results,
                &mut ok,
                &mut err,
                total,
                descr,
                eta_invocation_id,
            )
                .await;
        }
    }

    drop(eta_guard);

    if eta_invocation_id.is_some() {
        collect_eta_maybe_log(true);
    }

    results
}

fn spawn_next<T, F, I>(
    set: &mut JoinSet<(usize, Result<T, ThoError>)>,
    futures: &mut I,
) -> bool
where
    T: Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    I: Iterator<Item = (usize, F)>,
    ThoError: Send + 'static,
{
    let Some((idx, fut)) = futures.next() else {
        return false;
    };

    set.spawn(async move {
        cuda_activity_gate().wait_if_paused().await;
        (idx, fut.await)
    });

    true
}

pub async fn collect_joinset_unordered<T, F, I>(
    set: &mut JoinSet<(usize, Result<T, ThoError>)>,
    futures: &mut I,
    buffer_limit_rx: &mut watch::Receiver<usize>,
    mut buffer_limit: usize,
    results: &mut Vec<T>,
    ok: &mut usize,
    err: &mut usize,
    total: usize,
    descr: &str,
    eta_invocation_id: Option<u64>,
)
where
    T: Debug + Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    I: Iterator<Item = (usize, F)>,
    ThoError: Send + 'static,
{
    let mut futures_exhausted = fill_joinset(set, futures, buffer_limit, 0);
    let mut buffer_limit_updates_open = true;

    while !set.is_empty() {
        tokio::select! {
            biased;
            changed = buffer_limit_rx.changed(), if buffer_limit_updates_open => {
                if changed.is_err() {
                    buffer_limit_updates_open = false;
                } else {
                    buffer_limit = *buffer_limit_rx.borrow_and_update();
                    assert!(buffer_limit > 0, "buffer_limit must be > 0; JoinSet version cannot make progress with 0");
                }
            }
            joined = set.join_next() => {
                let (_idx, item) = unwrap_join_result(joined.expect("JoinSet must contain a task"));

                collect_one_with_progress(
                    item,
                    results,
                    ok,
                    err,
                    total,
                    descr,
                    eta_invocation_id,
                );
            }
        }

        buffer_limit = *buffer_limit_rx.borrow_and_update();
        assert!(buffer_limit > 0, "buffer_limit must be > 0; JoinSet version cannot make progress with 0");
        if !futures_exhausted {
            futures_exhausted = fill_joinset(set, futures, buffer_limit, 0);
        }
    }
}

async fn collect_joinset_ordered<T, F, I>(
    set: &mut JoinSet<(usize, Result<T, ThoError>)>,
    futures: &mut I,
    buffer_limit_rx: &mut watch::Receiver<usize>,
    mut buffer_limit: usize,
    results: &mut Vec<T>,
    ok: &mut usize,
    err: &mut usize,
    total: usize,
    descr: &str,
    eta_invocation_id: Option<u64>,
)
where
    T: Debug + Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    I: Iterator<Item = (usize, F)>,
    ThoError: Send + 'static,
{
    let mut next_yield_idx = 0usize;
    let mut completed = BTreeMap::<usize, Result<T, ThoError>>::new();
    let mut futures_exhausted = fill_joinset(set, futures, buffer_limit, completed.len());
    let mut buffer_limit_updates_open = true;

    while !set.is_empty() {
        tokio::select! {
            biased;
            changed = buffer_limit_rx.changed(), if buffer_limit_updates_open => {
                if changed.is_err() {
                    buffer_limit_updates_open = false;
                } else {
                    buffer_limit = *buffer_limit_rx.borrow_and_update();
                    assert!(buffer_limit > 0, "buffer_limit must be > 0; JoinSet version cannot make progress with 0");
                }
            }
            joined = set.join_next() => {
                let (idx, item) = unwrap_join_result(joined.expect("JoinSet must contain a task"));
                completed.insert(idx, item);

                while let Some(item) = completed.remove(&next_yield_idx) {
                    collect_one_with_progress(
                        item,
                        results,
                        ok,
                        err,
                        total,
                        descr,
                        eta_invocation_id,
                    );

                    next_yield_idx += 1;
                }
            }
        }

        buffer_limit = *buffer_limit_rx.borrow_and_update();
        assert!(buffer_limit > 0, "buffer_limit must be > 0; JoinSet version cannot make progress with 0");
        if !futures_exhausted {
            // Completed out-of-order items still occupy ordered-buffer capacity until they can be yielded.
            futures_exhausted = fill_joinset(set, futures, buffer_limit, completed.len());
        }
    }

    debug_assert!(completed.is_empty());
}

fn fill_joinset<T, F, I>(
    set: &mut JoinSet<(usize, Result<T, ThoError>)>,
    futures: &mut I,
    buffer_limit: usize,
    completed_len: usize,
) -> bool
where
    T: Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    I: Iterator<Item = (usize, F)>,
    ThoError: Send + 'static,
{
    while set.len() + completed_len < buffer_limit {
        if !spawn_next(set, futures) {
            return true;
        }
    }

    false
}


fn unwrap_join_result<T>(joined: Result<T, JoinError>) -> T {
    match joined {
        Ok(value) => value,

        Err(err) if err.is_panic() => {
            std::panic::resume_unwind(err.into_panic());
        }

        Err(err) => {
            panic!("JoinSet task was cancelled while collecting futures: {err}");
        }
    }
}

fn collect_one_with_progress<T>(
    item: Result<T, ThoError>,
    results: &mut Vec<T>,
    ok: &mut usize,
    err: &mut usize,
    total: usize,
    descr: &str,
    eta_invocation_id: Option<u64>,
)
where
    T: Debug,
{
    match item {
        Ok(value) => {
            *ok += 1;
            results.push(value);
        }

        Err(e) => {
            *err += 1;
            error!("collect_futures_with_progress: descr={descr} error={e:?}");
        }
    }

    let done = *ok + *err;

    if let Some(invocation_id) = eta_invocation_id {
        collect_eta_record_completion(invocation_id);
    }

    debug!("collect_futures_with_progress: descr={descr} progress={done}/{total} ok={ok} err={err}");

    if eta_invocation_id.is_some() {
        collect_eta_maybe_log(false);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use tokio::sync::{Semaphore, watch};

    use super::collect_futures_with_progress_with_limit;
    use crate::{StreamBuffering, ThoError};

    async fn wait_for_value(value: &AtomicUsize, expected: usize) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while value.load(Ordering::SeqCst) != expected {
                tokio::task::yield_now().await;
            }
        }).await.expect("timed out waiting for expected value");
    }

    async fn assert_limit_increase_is_applied(buffering: StreamBuffering) {
        let started = Arc::new(AtomicUsize::new(0));
        let permits = Arc::new(Semaphore::new(0));
        let (limit_tx, limit_rx) = watch::channel(1);

        let collect = {
            let started = Arc::clone(&started);
            let permits = Arc::clone(&permits);
            tokio::spawn(async move {
                collect_futures_with_progress_with_limit(
                    (0..6).map(move |idx| {
                        let started = Arc::clone(&started);
                        let permits = Arc::clone(&permits);
                        async move {
                            started.fetch_add(1, Ordering::SeqCst);
                            permits.acquire().await.expect("semaphore must stay open").forget();
                            Ok::<_, ThoError>(idx)
                        }
                    }),
                    6,
                    Arc::new(limit_rx),
                    buffering,
                    "assert_limit_increase_is_applied",
                ).await
            })
        };

        wait_for_value(&started, 1).await;
        limit_tx.send(3).expect("collector must retain the receiver");
        wait_for_value(&started, 3).await;

        permits.add_permits(6);
        let results = tokio::time::timeout(Duration::from_secs(2), collect).await.expect("collector must finish").expect("collector task must not panic");
        assert_eq!(results.len(), 6);
    }

    #[tokio::test]
    async fn unordered_limit_increase_starts_additional_futures() {
        assert_limit_increase_is_applied(StreamBuffering::Unordered).await;
    }

    #[tokio::test]
    async fn ordered_limit_increase_starts_additional_futures() {
        assert_limit_increase_is_applied(StreamBuffering::Ordered).await;
    }

    #[tokio::test]
    async fn limit_decrease_waits_for_active_futures_before_starting_more() {
        let started = Arc::new(AtomicUsize::new(0));
        let active = Arc::new(AtomicUsize::new(0));
        let permits = Arc::new(Semaphore::new(0));
        let (limit_tx, limit_rx) = watch::channel(3);

        let collect = {
            let started = Arc::clone(&started);
            let active = Arc::clone(&active);
            let permits = Arc::clone(&permits);
            tokio::spawn(async move {
                collect_futures_with_progress_with_limit(
                    (0..5).map(move |idx| {
                        let started = Arc::clone(&started);
                        let active = Arc::clone(&active);
                        let permits = Arc::clone(&permits);
                        async move {
                            started.fetch_add(1, Ordering::SeqCst);
                            active.fetch_add(1, Ordering::SeqCst);
                            permits.acquire().await.expect("semaphore must stay open").forget();
                            active.fetch_sub(1, Ordering::SeqCst);
                            Ok::<_, ThoError>(idx)
                        }
                    }),
                    5,
                    Arc::new(limit_rx),
                    StreamBuffering::Unordered,
                    "limit_decrease_waits_for_active_futures_before_starting_more",
                ).await
            })
        };

        wait_for_value(&started, 3).await;
        limit_tx.send(1).expect("collector must retain the receiver");
        permits.add_permits(2);
        wait_for_value(&active, 1).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(started.load(Ordering::SeqCst), 3);

        permits.add_permits(1);
        wait_for_value(&started, 4).await;
        permits.add_permits(2);

        let results = tokio::time::timeout(Duration::from_secs(2), collect).await.expect("collector must finish").expect("collector task must not panic");
        assert_eq!(results.len(), 5);
    }
}
