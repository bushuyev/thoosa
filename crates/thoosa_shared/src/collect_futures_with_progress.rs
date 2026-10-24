use std::{
    collections::{BTreeMap, HashMap},
    fmt::Debug,
    future::Future,
    sync::Arc,
};
use log::warn;
use tokio::sync::watch;
use tokio::task::{Id, JoinSet};
use tracing::{debug, error};
use crate::{CudaActivityScope, FuturesOrdering, ThoError};
use crate::collect_futures_eta::{collect_eta_maybe_log, collect_eta_record_completion, CollectEtaInvocationGuard};

/// Required work never succeeds with missing items. Drain every submitted future
/// before reporting failure: dropping a JoinSet early can abandon CUDA completion
/// and request lifetimes owned by its tasks.
pub async fn collect_futures_with_progress<T, F>(
    futures: impl Iterator<Item = F>, len: usize, buffering: FuturesOrdering,
    ignore_failures: bool, descr: &str,
) -> Result<Vec<T>, ThoError>
where
    T: Debug + Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
{
    let registration = crate::futures_limits::futures_limits().register(descr);
    collect_futures_with_progress_with_limit(futures, len, registration.receiver(), buffering, ignore_failures, descr).await
}

/// Explicit opt-in for independent, best-effort operations such as plot export.
/// Failures are logged with their input index; all work is still drained.
pub async fn collect_futures_best_effort_with_progress<T, F>(
    futures: impl Iterator<Item = F>, len: usize, buffering: FuturesOrdering, descr: &str,
) -> Vec<T>
where
    T: Debug + Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
{
    let registration = crate::futures_limits::futures_limits().register(descr);
    let outcomes = collect_outcomes(futures, len, registration.receiver(), buffering, descr).await;
    outcomes.into_iter().filter_map(|(idx, result)| match result {
        Ok(value) => Some(value),
        Err(error) => {
            error!("best-effort collection: descr={descr} index={idx} error={error:?}");
            None
        }
    }).collect()
}

async fn collect_futures_with_progress_with_limit<T, F>(
    futures: impl Iterator<Item = F>, len: usize, buffer_limit: Arc<watch::Receiver<usize>>,
    buffering: FuturesOrdering,
    ignore_failures: bool,
    descr: &str,
) -> Result<Vec<T>, ThoError>
where
    T: Debug + Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
{
    let outcomes = collect_outcomes(futures, len, buffer_limit, buffering, descr).await;
    let mut values = Vec::with_capacity(outcomes.len());
    let mut failure: Option<ThoError> = None;
    for (idx, result) in outcomes {
        match result {
            Ok(value) => values.push(value),
            Err(error) => {
                warn!("collect_futures_with_progress_with_limit: error={error:?}");
                // A later fatal completion must not be hidden by an earlier OOM
                // or ordinary error. Context retains the original typed cause.
                let replace = failure.as_ref().is_none_or(|original| {
                    (error.is_cuda_recovery_failed() && !original.is_cuda_recovery_failed()) ||
                    (error.needs_cuda_restart() && !original.needs_cuda_restart() && !original.is_cuda_recovery_failed())
                });
                if replace {
                    failure = Some(error.context(format!("{descr}: task at input index {idx} failed")));
                }
            }
        }
    }
    debug!("collect_futures_with_progress_with_limit: ignore_failures={ignore_failures} values.is_empty={:?}", values.is_empty());
    if failure.is_some() && !ignore_failures {
        debug!("collect_futures_with_progress_with_limit: error: {failure:?}");
        Err(failure.unwrap())
    } else {
        Ok(values)
    }
}

async fn collect_outcomes<T, F>(
    futures: impl Iterator<Item = F>, len: usize, buffer_limit: Arc<watch::Receiver<usize>>,
    buffering: FuturesOrdering, descr: &str,
) -> Vec<(usize, Result<T, ThoError>)>
where
    T: Debug + Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
{
    let mut limit_rx = buffer_limit.as_ref().clone();
    let mut limit = *limit_rx.borrow_and_update();
    assert!(limit > 0, "buffer_limit must be > 0; JoinSet version cannot make progress with 0");
    let eta_guard = (len > 0).then(|| CollectEtaInvocationGuard::new(descr, len, limit.min(len.max(1))));
    let eta_id = eta_guard.as_ref().map(CollectEtaInvocationGuard::id);
    let mut futures = futures.enumerate();
    let mut tasks = JoinSet::new();
    // JoinError carries a Tokio task ID, not the input index. Retain the mapping
    // so a panic/cancellation occupies its ordered slot instead of leaving a gap.
    let mut indices = HashMap::new();
    let mut completed = BTreeMap::new();
    let mut next_idx = 0;
    let mut outcomes = Vec::with_capacity(len);
    let (mut ok, mut err) = (0, 0);
    let mut updates_open = true;
    let mut exhausted = fill_joinset(&mut tasks, &mut indices, &mut futures, limit, 0);
    while !tasks.is_empty() {
        tokio::select! {
            changed = limit_rx.changed(), if updates_open => {
                if changed.is_err() { updates_open = false; }
            }
            joined = tasks.join_next_with_id() => {
                let (id, item) = match joined.expect("JoinSet must contain a task") {
                    Ok((id, item)) => (id, item),
                    Err(error) => (error.id(), Err(ThoError::from_source(error).context("join parallel task"))),
                };
                let idx = indices.remove(&id).expect("every submitted task has an input index");
                if item.is_ok() { ok += 1; } else { err += 1; }
                if let Some(id) = eta_id { collect_eta_record_completion(id); }
                debug!("collect_futures_with_progress: descr={descr} progress={}/{len} ok={ok} err={err}", ok + err);
                if eta_id.is_some() { collect_eta_maybe_log(false); }
                match buffering {
                    FuturesOrdering::Unordered => outcomes.push((idx, item)),
                    FuturesOrdering::Ordered => {
                        completed.insert(idx, item);
                        while let Some(item) = completed.remove(&next_idx) {
                            outcomes.push((next_idx, item));
                            next_idx += 1;
                        }
                    }
                }
            }
        }
        limit = *limit_rx.borrow_and_update();
        assert!(limit > 0, "buffer_limit must be > 0; JoinSet version cannot make progress with 0");
        if !exhausted {
            // Out-of-order completions occupy ordered-buffer capacity until the
            // earlier slots finish, including slots that failed or panicked.
            exhausted = fill_joinset(&mut tasks, &mut indices, &mut futures, limit, completed.len());
        }
    }
    debug_assert!(completed.is_empty() && indices.is_empty());
    drop(eta_guard);
    if eta_id.is_some() { collect_eta_maybe_log(true); }
    outcomes
}

fn fill_joinset<T, F, I>(
    tasks: &mut JoinSet<Result<T, ThoError>>, indices: &mut HashMap<Id, usize>,
    futures: &mut I, limit: usize, completed_len: usize,
) -> bool
where
    T: Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    I: Iterator<Item = (usize, F)>,
{
    while tasks.len() + completed_len < limit {
        let Some((idx, future)) = futures.next() else { return true; };
        // Child tasks inherit admission, including nested copy collectors. They
        // must drain during recovery rather than wait on their own parent lease.
        let activity = CudaActivityScope::current();
        let task = tasks.spawn(async move {
            match activity {
                Some(activity) => activity.run(future).await,
                None => future.await,
            }
        });
        indices.insert(task.id(), idx);
    }
    false
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use tokio::sync::{Semaphore, watch};

    use super::collect_futures_with_progress_with_limit;
    use crate::{FuturesOrdering, ThoError};

    async fn wait_for_value(value: &AtomicUsize, expected: usize) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while value.load(Ordering::SeqCst) != expected {
                tokio::task::yield_now().await;
            }
        }).await.expect("timed out waiting for expected value");
    }

    async fn assert_limit_increase_is_applied(buffering: FuturesOrdering) {
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
                    true,
                    "assert_limit_increase_is_applied",
                ).await
            })
        };

        wait_for_value(&started, 1).await;
        limit_tx.send(3).expect("collector must retain the receiver");
        wait_for_value(&started, 3).await;

        permits.add_permits(6);
        let results = tokio::time::timeout(Duration::from_secs(2), collect).await.expect("collector must finish").expect("collector task must not panic").expect("all work must succeed");
        assert_eq!(results.len(), 6);
    }

    #[tokio::test]
    async fn unordered_limit_increase_starts_additional_futures() {
        assert_limit_increase_is_applied(FuturesOrdering::Unordered).await;
    }

    #[tokio::test]
    async fn ordered_limit_increase_starts_additional_futures() {
        assert_limit_increase_is_applied(FuturesOrdering::Ordered).await;
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
                    FuturesOrdering::Unordered,
                    true,
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

        let results = tokio::time::timeout(Duration::from_secs(2), collect).await.expect("collector must finish").expect("collector task must not panic").expect("all work must succeed");
        assert_eq!(results.len(), 5);
    }

    #[tokio::test]
    async fn test_failure_drains_started_and_remaining_work() {
        for ordering in [FuturesOrdering::Ordered, FuturesOrdering::Unordered] {
            let started = Arc::new(AtomicUsize::new(0));
            let completed = Arc::new(AtomicUsize::new(0));
            let permits = Arc::new(Semaphore::new(0));
            let (_limit_tx, limit_rx) = watch::channel(2);
            let mut collect = {
                let started = Arc::clone(&started);
                let completed = Arc::clone(&completed);
                let permits = Arc::clone(&permits);
                tokio::spawn(async move {
                    collect_futures_with_progress_with_limit((0..4).map(move |idx| {
                                            let started = Arc::clone(&started);
                                            let completed = Arc::clone(&completed);
                                            let permits = Arc::clone(&permits);
                                            async move {
                                                started.fetch_add(1, Ordering::SeqCst);
                                                if idx == 0 { return Err(ThoError::ContextReset); }
                                                permits.acquire().await.expect("test keeps semaphore open").forget();
                                                completed.fetch_add(1, Ordering::SeqCst);
                                                Ok(idx)
                                            }
                                        }), 4, Arc::new(limit_rx), ordering,
                                                             false, "required pipeline work").await
                })
            };
            // The error has freed a slot and another task has started, but the
            // caller must still wait for the operations holding completion leases.
            wait_for_value(&started, 3).await;
            assert!(tokio::time::timeout(Duration::from_millis(30), &mut collect).await.is_err());
            permits.add_permits(3);
            let error = tokio::time::timeout(Duration::from_secs(2), collect).await.expect("drain finishes").expect("collector does not panic").unwrap_err();
            assert!(matches!(error.what(), ThoError::ContextReset));
            assert!(error.to_string().contains("required pipeline work: task at input index 0"));
            assert_eq!(completed.load(Ordering::SeqCst), 3);
        }
    }

    #[tokio::test]
    async fn test_panicking_task_is_a_failure_and_does_not_stall_ordered_slots() {
        for ordering in [FuturesOrdering::Ordered, FuturesOrdering::Unordered] {
            let completed = Arc::new(AtomicUsize::new(0));
            let (_limit_tx, limit_rx) = watch::channel(2);
            let futures = (0..5).map(|idx| {
                let completed = Arc::clone(&completed);
                async move {
                    if idx == 1 { panic!("parallel task panic"); }
                    completed.fetch_add(1, Ordering::SeqCst);
                    Ok(idx)
                }
            });
            let error = tokio::time::timeout(Duration::from_secs(2), collect_futures_with_progress_with_limit(futures, 5, Arc::new(limit_rx), ordering, false, "panic regression")).await.expect("panic must not stall collection").unwrap_err();
            assert_eq!(completed.load(Ordering::SeqCst), 4);
            assert!(error.to_string().contains("input index 1"));
            assert!(format!("{error:#}").contains("parallel task panic"));
        }
    }

    #[tokio::test]
    async fn test_fatal_error_precedes_oom_after_all_work_finishes() {
        for ordering in [FuturesOrdering::Ordered, FuturesOrdering::Unordered] {
            let (_limit_tx, limit_rx) = watch::channel(1);
            let failures = [ThoError::oom(), ThoError::from_cuda_error("fatal completion", 700)];
            let futures = failures.into_iter().map(|error| async move { Err::<(), _>(error) });
            let error = collect_futures_with_progress_with_limit(futures, 2, Arc::new(limit_rx), ordering, false, "fatal precedence").await.unwrap_err();
            assert!(error.needs_cuda_restart());
            assert!(!error.is_oom());
            assert!(error.to_string().contains("input index 1"));
        }
    }

    #[tokio::test]
    async fn test_terminal_recovery_failure_precedes_restart_and_oom() {
        let (_limit_tx, limit_rx) = watch::channel(1);
        let failures = [
            ThoError::oom(),
            ThoError::CudaRecoveryFailed(Arc::new(crate::CudaRecoveryFailure {
                original: ThoError::from_cuda_error("device lost", 700),
                recovery: Some(ThoError::new("worker replacement failed")),
            })),
            ThoError::from_cuda_error("late restart error", 700),
        ];
        let futures = failures.into_iter().map(|error| async move { Err::<(), _>(error) });
        let error = collect_futures_with_progress_with_limit(futures, 3, Arc::new(limit_rx), FuturesOrdering::Unordered, false, "terminal precedence").await.unwrap_err();
        assert!(error.is_cuda_recovery_failed());
        assert!(error.to_string().contains("input index 1"));
    }

    #[tokio::test]
    async fn test_best_effort_requires_explicit_api() {
        let futures = || (0..3).map(|idx| async move {
            if idx == 1 { Err(ThoError::new("ordinary task failure")) } else { Ok(idx) }
        });
        let error = super::collect_futures_with_progress(futures(), 3, FuturesOrdering::Ordered, false, "strict collection").await.unwrap_err();
        assert!(format!("{error:#}").contains("ordinary task failure"));
        let values = super::collect_futures_best_effort_with_progress(futures(), 3, FuturesOrdering::Ordered, "best effort collection").await;
        assert_eq!(values, vec![0, 2]);
    }

    #[tokio::test]
    async fn test_ordered_success_retains_input_order_after_out_of_order_completion() {
        let (_limit_tx, limit_rx) = watch::channel(3);
        let completed = Arc::new(AtomicUsize::new(0));
        let futures = (0..3).map(|idx| {
            let completed = Arc::clone(&completed);
            async move {
                if idx == 0 { wait_for_value(&completed, 2).await; }
                else { completed.fetch_add(1, Ordering::SeqCst); }
                Ok::<_, ThoError>(idx)
            }
        });
        let values = collect_futures_with_progress_with_limit(futures, 3, Arc::new(limit_rx), FuturesOrdering::Ordered, true, "ordered success").await.expect("all work succeeds");
        assert_eq!(values, vec![0, 1, 2]);
    }

}
