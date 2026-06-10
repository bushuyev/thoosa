use std::{
    collections::BTreeMap,
    fmt::Debug,
    future::Future,
};

use tokio::task::{JoinError, JoinSet};
use tracing::{debug, error};
use crate::{cuda_activity_gate, StreamBuffering, ThoError};
use crate::collect_futures_eta::{collect_eta_maybe_log, collect_eta_record_completion, CollectEtaInvocationGuard};

pub async fn collect_futures_with_progress<T, F>(
    futures: impl Iterator<Item = F>,
    len: usize,
    buffer_limit: usize,
    buffering: StreamBuffering,
    descr: &str,
) -> Vec<T>
where
    T: Debug + Send + 'static,
    F: Future<Output = Result<T, ThoError>> + Send + 'static,
    ThoError: Send + 'static,
{
    debug!("collect_futures_with_progress: descr={descr} len={len}");

    assert!(buffer_limit > 0, "buffer_limit must be > 0; JoinSet version cannot make progress with 0");

    let mut results = Vec::with_capacity(len);
    let (mut ok, mut err) = (0usize, 0usize);
    let total = len;

    let eta_concurrency = buffer_limit.max(1).min(total.max(1));

    let eta_guard = (total > 0).then(|| {
        CollectEtaInvocationGuard::new(descr, total, eta_concurrency)
    });

    let eta_invocation_id = eta_guard.as_ref().map(CollectEtaInvocationGuard::id);

    let mut futures = futures.enumerate();
    let mut set = JoinSet::<(usize, Result<T, ThoError>)>::new();

    for _ in 0..buffer_limit {
        if !spawn_next(&mut set, &mut futures) {
            break;
        }
    }

    match buffering {
        StreamBuffering::Unordered => {
            collect_joinset_unordered(
                &mut set,
                &mut futures,
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
    while let Some(joined) = set.join_next().await {
        let (_idx, item) = unwrap_join_result(joined);

        collect_one_with_progress(
            item,
            results,
            ok,
            err,
            total,
            descr,
            eta_invocation_id,
        );

        spawn_next(set, futures);
    }
}

async fn collect_joinset_ordered<T, F, I>(
    set: &mut JoinSet<(usize, Result<T, ThoError>)>,
    futures: &mut I,
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

    while let Some(joined) = set.join_next().await {
        let (idx, item) = unwrap_join_result(joined);
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

            // Important:
            // For ordered/buffered semantics, only spawn the next future
            // after the next in-order item has been yielded.
            spawn_next(set, futures);
        }
    }
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
