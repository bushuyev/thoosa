use std::{
    any::type_name,
    fmt::{Display, Formatter},
    future::Future,
    panic::Location,
    sync::{
        Arc,
        LazyLock,
        atomic::{AtomicU64, Ordering},
    },
    task::{Context, Poll, Wake, Waker},
    time::Instant,
};
use std::pin::Pin;
use dashmap::DashMap;
use pin_project::{pin_project, pinned_drop};

// Select the expansion when this crate is compiled, so ordinary builds return
// the original expression without boxing, registry updates, or waker wrapping.
// In particular, do not add a block/async wrapper that shortens temporary
// lifetimes or changes the future's type and capture behavior.
#[cfg(not(tho_debug))]
#[macro_export]
macro_rules! track_await {
    ($future:expr $(,)?) => { $future };
}

/// Enable await diagnostics only in builds compiled with `--cfg tho_debug`.
#[cfg(tho_debug)]
#[macro_export]
macro_rules! track_await {
    ($future:expr $(,)?) => { $crate::tracked_await::track_await($future) };
}

// Keep the existing module import path usable for both the macro and the
// explicit tracking function used by its diagnostic expansion.
pub use crate::track_await;

static NEXT_ID: AtomicU64 = AtomicU64::new(1);
pub static TASKS: LazyLock<DashMap<u64, TaskInfo>> = LazyLock::new(DashMap::new);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskState {
    Polling,
    Pending,
    Woken,
}

#[derive(Debug, Clone)]
pub struct TaskInfo {
    pub id: u64,
    pub ty: &'static str,
    pub spawn_location: &'static Location<'static>,
    pub state: TaskState,
    pub poll_count: u64,
    pub wake_count: u64,
    pub last_change: Instant,
    pub last_pending: Option<Instant>,
}

impl Display for TaskInfo {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let state = match self.state {
            TaskState::Polling => "polling",
            TaskState::Pending => "pending",
            TaskState::Woken => "woken",
        };
        let location = self.spawn_location;

        write!(
            f,
            "task #{} {state} at {}:{}:{} polls={} wakes={} state_for={:?}",
            self.id,
            location.file(),
            location.line(),
            location.column(),
            self.poll_count,
            self.wake_count,
            self.last_change.elapsed(),
        )?;

        if let Some(last_pending) = self.last_pending {
            write!(f, " pending_for={:?}", last_pending.elapsed())?;
        }

        write!(f, " future={}", self.ty)
    }
}

#[pin_project(PinnedDrop)]
pub struct Tracked<F> {
    id: u64,
    #[pin]
    inner: F,
}

#[pinned_drop]
impl<F> PinnedDrop for Tracked<F> {
    fn drop(self: Pin<&mut Self>) {
        TASKS.remove(&self.id);
    }
}



impl<F> Tracked<F> {
    pub fn new_at(
        spawn_location: &'static Location<'static>,
        inner: F,
    ) -> Self {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);

        TASKS.insert(id, TaskInfo {
            id,
            ty: type_name::<F>(),
            spawn_location,
            state: TaskState::Pending,
            poll_count: 0,
            wake_count: 0,
            last_change: Instant::now(),
            last_pending: None,
        });

        Self { id, inner }
    }
}

#[track_caller]
pub fn spawn_tracked<F>(
    fut: F,
) -> tokio::task::JoinHandle<F::Output>
where
    F: Future + Send + 'static,
    F::Output: Send + 'static,
{
    let loc = Location::caller();
    tokio::spawn(Tracked::new_at(loc, fut))
}

impl<F: Future> Future for Tracked<F> {
    type Output = F::Output;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        let id = *this.id;

        let wake_count_before = mark_polling(id);

        let waker = Waker::from(Arc::new(TrackedWake {
            id,
            inner: cx.waker().clone(),
        }));
        let mut tracked_cx = Context::from_waker(&waker);

        match this.inner.poll(&mut tracked_cx) {
            Poll::Ready(v) => {
                TASKS.remove(&id);
                Poll::Ready(v)
            }
            Poll::Pending => {
                let woke_during_poll = TASKS
                    .get(&id)
                    .map(|e| e.wake_count != wake_count_before)
                    .unwrap_or(false);

                mark_pending(
                    id,
                    if woke_during_poll {
                        TaskState::Woken
                    } else {
                        TaskState::Pending
                    },
                );

                Poll::Pending
            }
        }
    }
}


struct TrackedWake {
    id: u64,
    inner: Waker,
}

impl Wake for TrackedWake {
    fn wake(self: Arc<Self>) {
        mark_woken(self.id);
        self.inner.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        mark_woken(self.id);
        self.inner.wake_by_ref();
    }
}

fn mark_polling(id: u64) -> u64 {
    if let Some(mut e) = TASKS.get_mut(&id) {
        e.state = TaskState::Polling;
        e.poll_count += 1;
        e.last_change = Instant::now();
        e.wake_count
    } else {
        0
    }
}

fn mark_pending(id: u64, state: TaskState) {
    if let Some(mut e) = TASKS.get_mut(&id) {
        e.state = state;
        e.last_change = Instant::now();
        e.last_pending = Some(Instant::now());
    }
}

fn mark_woken(id: u64) {
    if let Some(mut e) = TASKS.get_mut(&id) {
        e.state = TaskState::Woken;
        e.wake_count += 1;
        e.last_change = Instant::now();
    }
}

#[track_caller]
pub fn track_await<F>(
    fut: F,
) -> Tracked<Pin<Box<F>>>
where
    F: Future,
{
    Tracked::new_at(Location::caller(), Box::pin(fut))
}

pub fn snapshot_not_ready() -> Vec<TaskInfo> {
    TASKS
        .iter()
        .filter(|e| matches!(e.state, TaskState::Pending | TaskState::Woken))
        .map(|e| e.value().clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;

    #[cfg(not(tho_debug))]
    #[test]
    fn test_track_await_returns_the_original_future_type() {
        // A concrete type check rules out boxing and any tracking wrapper.
        let future: std::future::Ready<u32> = track_await!(std::future::ready(42));
        assert_eq!(futures::executor::block_on(future), 42);
    }

    #[test]
    fn test_track_await_evaluates_once_and_preserves_borrowed_temporaries() {
        let evaluations = Cell::new(0);
        let polls = Cell::new(0);
        let future = track_await!({
            evaluations.set(evaluations.get() + 1);
            std::future::poll_fn(|cx| {
                polls.set(polls.get() + 1);
                if polls.get() == 1 { cx.waker().wake_by_ref(); Poll::Pending }
                else { Poll::Ready(42) }
            })
        },);
        assert_eq!(evaluations.get(), 1);
        assert_eq!(polls.get(), 0, "constructing the wrapper must not poll the future");
        assert_eq!(futures::executor::block_on(future), 42);
        assert_eq!(evaluations.get(), 1);
        assert_eq!(polls.get(), 2);
        async fn length(value: &str) -> usize { value.len() }
        // Direct expansion must keep the temporary String alive through await.
        assert_eq!(futures::executor::block_on(async { track_await!(length(&String::from("abc"))).await }), 3);
    }

    #[cfg(tho_debug)]
    #[test]
    fn test_track_await_tracks_location_wakes_completion_and_cancellation() {
        let polls = Cell::new(0);
        let line = line!() + 1;
        let future = track_await!(std::future::poll_fn(|cx| {
            polls.set(polls.get() + 1);
            if polls.get() == 1 { cx.waker().wake_by_ref(); Poll::Pending }
            else { Poll::Ready(42) }
        }));
        let id = future.id;
        assert_eq!(TASKS.get(&id).unwrap().spawn_location.line(), line);
        let mut future = Box::pin(future);
        let mut cx = Context::from_waker(futures::task::noop_waker_ref());
        assert!(future.as_mut().poll(&mut cx).is_pending());
        {
            let task = TASKS.get(&id).unwrap();
            assert_eq!(task.state, TaskState::Woken);
            assert_eq!(task.poll_count, 1);
            assert_eq!(task.wake_count, 1);
        }
        assert_eq!(future.as_mut().poll(&mut cx), Poll::Ready(42));
        assert!(!TASKS.contains_key(&id));
        let cancelled = track_await!(std::future::pending::<()>());
        let id = cancelled.id;
        assert!(TASKS.contains_key(&id));
        drop(cancelled);
        assert!(!TASKS.contains_key(&id), "cancellation must remove diagnostic entries");
    }
}
