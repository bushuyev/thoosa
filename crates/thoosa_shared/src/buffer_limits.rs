use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use tokio::sync::watch;
use tracing::debug;

const INITIAL_GLOBAL_LIMIT: usize = 8;
const MIN_GLOBAL_LIMIT: usize = 1;
const MAX_GLOBAL_LIMIT: usize = 32;
// Increase only after a clean success streak; one OOM per 16 completed sync jobs is excessive.
const SUCCESSFUL_EXECUTIONS_PER_INCREASE: usize = 32;
const OOM_RATE_MIN_EXECUTIONS: usize = 16;
const OOM_RATE_DENOMINATOR: usize = 16;
const FEEDBACK_WINDOW_CAPACITY: usize = 64;

pub struct DynamicBufferLimit {
    sender: watch::Sender<usize>,
    receiver: Arc<watch::Receiver<usize>>,
}

impl DynamicBufferLimit {
    fn new(limit: usize) -> Self {
        assert!(limit > 0, "buffer limit must be greater than zero");
        let (sender, receiver) = watch::channel(limit);
        Self {
            sender,
            receiver: Arc::new(receiver),
        }
    }

    pub fn receiver(&self) -> Arc<watch::Receiver<usize>> {
        Arc::clone(&self.receiver)
    }

    fn set(&self, limit: usize) {
        assert!(limit > 0, "buffer limit must be greater than zero");
        self.sender.send_replace(limit);
    }

    pub fn current(&self) -> usize {
        *self.receiver.borrow()
    }
}

struct FutureLimitEntry {
    limit: DynamicBufferLimit,
    registrations: usize,
}

#[derive(Default)]
struct BufferLimitsState {
    futures_limits: HashMap<String, FutureLimitEntry>,
    active_registrations: usize,
    feedback_window: VecDeque<bool>,
    feedback_window_ooms: usize,
    consecutive_ok: usize,
    oom_seen: bool,
}

#[derive(Debug)]
struct SyncExecutionLimiterState {
    active: usize,
    limit: usize,
    generation: u64,
}

#[derive(Debug)]
struct SyncExecutionLimiter {
    state: Mutex<SyncExecutionLimiterState>,
    capacity_changed: watch::Sender<u64>,
}

impl SyncExecutionLimiter {
    fn new(limit: usize) -> Self {
        let (capacity_changed, _) = watch::channel(0);
        Self {
            state: Mutex::new(SyncExecutionLimiterState {
                active: 0,
                limit,
                generation: 0,
            }),
            capacity_changed,
        }
    }

    fn try_acquire(self: &Arc<Self>) -> Option<SyncExecutionPermit> {
        let mut state = self.state.lock();
        if state.active >= state.limit {
            return None;
        }
        state.active += 1;
        Some(SyncExecutionPermit { limiter: Arc::clone(self) })
    }

    fn force_acquire(self: &Arc<Self>) -> SyncExecutionPermit {
        self.state.lock().active += 1;
        SyncExecutionPermit { limiter: Arc::clone(self) }
    }

    fn set_limit(&self, limit: usize) {
        let mut state = self.state.lock();
        state.limit = limit;
        Self::notify_capacity_changed(&mut state, &self.capacity_changed);
    }

    fn subscribe(&self) -> watch::Receiver<u64> {
        self.capacity_changed.subscribe()
    }

    fn release(&self) {
        let mut state = self.state.lock();
        assert!(state.active > 0, "released an inactive sync execution slot");
        state.active -= 1;
        Self::notify_capacity_changed(&mut state, &self.capacity_changed);
    }

    fn notify_capacity_changed(state: &mut SyncExecutionLimiterState, capacity_changed: &watch::Sender<u64>) {
        state.generation = state.generation.wrapping_add(1);
        capacity_changed.send_replace(state.generation);
    }
}

#[derive(Debug)]
pub struct SyncExecutionPermit {
    limiter: Arc<SyncExecutionLimiter>,
}

impl Drop for SyncExecutionPermit {
    fn drop(&mut self) {
        self.limiter.release();
    }
}

/// Coordinates asynchronous iterator fan-out with the shared synchronous executor limit.
///
/// Active collectors share the current global limit equally. Registering a nested or concurrent
/// collector therefore lowers every collector's allowance; already-running futures are allowed to
/// drain, while new futures observe the reduced watch value. Dynamic task executors use shared
/// execution permits, so the same limit is enforced across all per-device executor instances.
pub struct BufferLimits {
    state: Mutex<BufferLimitsState>,
    global_limit: DynamicBufferLimit,
    sync_execution_limiter: Arc<SyncExecutionLimiter>,
}

pub struct BufferLimitRegistration<'a> {
    owner: &'a BufferLimits,
    descr: String,
    receiver: Arc<watch::Receiver<usize>>,
}

impl BufferLimitRegistration<'_> {
    pub fn receiver(&self) -> Arc<watch::Receiver<usize>> {
        Arc::clone(&self.receiver)
    }
}

impl Drop for BufferLimitRegistration<'_> {
    fn drop(&mut self) {
        self.owner.forget(&self.descr);
    }
}

impl BufferLimits {
    pub fn new() -> Self {
        Self {
            state: Mutex::new(BufferLimitsState::default()),
            global_limit: DynamicBufferLimit::new(INITIAL_GLOBAL_LIMIT),
            sync_execution_limiter: Arc::new(SyncExecutionLimiter::new(INITIAL_GLOBAL_LIMIT)),
        }
    }

    pub fn register<'a>(&'a self, descr: &str) -> BufferLimitRegistration<'a> {
        let mut state = self.state.lock();
        let receiver = {
            let entry = state.futures_limits.entry(descr.to_owned()).or_insert_with(|| FutureLimitEntry {
                limit: DynamicBufferLimit::new(1),
                registrations: 0,
            });
            entry.registrations += 1;
            entry.limit.receiver()
        };
        state.active_registrations += 1;
        self.rebalance_future_limits(&state);
        debug!(descr, active_registrations = state.active_registrations, global_limit = self.global_limit.current(), "registered future collector");

        BufferLimitRegistration {
            owner: self,
            descr: descr.to_owned(),
            receiver,
        }
    }

    pub fn forget(&self, descr: &str) {
        let mut state = self.state.lock();
        let mut remove = false;
        if let Some(entry) = state.futures_limits.get_mut(descr) {
            entry.registrations -= 1;
            remove = entry.registrations == 0;
            state.active_registrations -= 1;
        }
        if remove {
            state.futures_limits.remove(descr);
        }
        self.rebalance_future_limits(&state);
        debug!(descr, active_registrations = state.active_registrations, global_limit = self.global_limit.current(), "forgot future collector");
    }

    pub fn on_oom(&self) {
        let mut state = self.state.lock();
        state.consecutive_ok = 0;
        state.oom_seen = true;
        Self::push_feedback(&mut state, true);
        self.decrease_if_oom_rate_high(&mut state);
    }

    pub fn on_ok(&self) {
        let mut state = self.state.lock();
        Self::push_feedback(&mut state, false);
        if self.decrease_if_oom_rate_high(&mut state) {
            return;
        }
        if state.oom_seen {
            return;
        }
        state.consecutive_ok += 1;

        if state.consecutive_ok >= SUCCESSFUL_EXECUTIONS_PER_INCREASE {
            state.consecutive_ok = 0;
            let old_limit = self.global_limit.current();
            let new_limit = old_limit.saturating_add(1).min(MAX_GLOBAL_LIMIT);
            self.set_global_limit(new_limit);
            self.rebalance_future_limits(&state);
            debug!(old_limit, new_limit, "increased adaptive parallelism after successful executions");
        }
    }

    /// The shared limit for active synchronous jobs across all per-device executors.
    pub fn global_limit(&self) -> Arc<watch::Receiver<usize>> {
        self.global_limit.receiver()
    }

    pub fn sync_capacity_changes(&self) -> watch::Receiver<u64> {
        self.sync_execution_limiter.subscribe()
    }

    pub fn try_acquire_sync_execution(&self) -> Option<SyncExecutionPermit> {
        self.sync_execution_limiter.try_acquire()
    }

    /// Reset draining must let already-running parents unwind even if this temporarily exceeds
    /// the adaptive limit. No new queued jobs are started while the CUDA activity gate is paused.
    pub fn force_acquire_sync_execution(&self) -> SyncExecutionPermit {
        self.sync_execution_limiter.force_acquire()
    }

    fn push_feedback(state: &mut BufferLimitsState, oom: bool) {
        state.feedback_window.push_back(oom);
        state.feedback_window_ooms += usize::from(oom);
        if state.feedback_window.len() > FEEDBACK_WINDOW_CAPACITY {
            state.feedback_window_ooms -= usize::from(state.feedback_window.pop_front().expect("feedback window must not be empty"));
        }
    }

    fn decrease_if_oom_rate_high(&self, state: &mut BufferLimitsState) -> bool {
        if state.feedback_window.len() < OOM_RATE_MIN_EXECUTIONS
            || state.feedback_window_ooms.saturating_mul(OOM_RATE_DENOMINATOR) < state.feedback_window.len()
        {
            return false;
        }

        let old_limit = self.global_limit.current();
        let new_limit = old_limit.saturating_sub(1).max(MIN_GLOBAL_LIMIT);
        self.set_global_limit(new_limit);
        state.feedback_window.clear();
        state.feedback_window_ooms = 0;
        state.consecutive_ok = 0;
        self.rebalance_future_limits(state);
        debug!(old_limit, new_limit, "decreased adaptive parallelism after frequent OOMs");
        true
    }

    fn set_global_limit(&self, limit: usize) {
        self.sync_execution_limiter.set_limit(limit);
        self.global_limit.set(limit);
    }

    fn rebalance_future_limits(&self, state: &BufferLimitsState) {
        if state.active_registrations == 0 {
            return;
        }

        let collector_limit = (self.global_limit.current() / state.active_registrations).max(1);
        for entry in state.futures_limits.values() {
            entry.limit.set(collector_limit);
        }
    }
}

impl Default for BufferLimits {
    fn default() -> Self {
        Self::new()
    }
}

pub static BUFFER_LIMITS: LazyLock<Arc<BufferLimits>> = LazyLock::new(|| Arc::new(BufferLimits::new()));

pub fn buffer_limits() -> &'static Arc<BufferLimits> {
    &BUFFER_LIMITS
}

#[cfg(test)]
mod tests {
    use super::{BufferLimits, OOM_RATE_MIN_EXECUTIONS, SUCCESSFUL_EXECUTIONS_PER_INCREASE};

    #[test]
    fn active_collectors_share_global_limit() {
        let limits = BufferLimits::new();
        let first = limits.register("first");
        assert_eq!(*first.receiver().borrow(), 8);

        let second = limits.register("second");
        assert_eq!(*first.receiver().borrow(), 4);
        assert_eq!(*second.receiver().borrow(), 4);

        let second_again = limits.register("second");
        assert_eq!(*first.receiver().borrow(), 2);
        assert_eq!(*second.receiver().borrow(), 2);
        assert_eq!(*second_again.receiver().borrow(), 2);

        drop(second_again);
        assert_eq!(*first.receiver().borrow(), 4);
        assert_eq!(*second.receiver().borrow(), 4);
    }

    #[test]
    fn successful_execution_streak_increases_limits() {
        let limits = BufferLimits::new();
        let collector = limits.register("collector");

        for _ in 0..SUCCESSFUL_EXECUTIONS_PER_INCREASE {
            limits.on_ok();
        }

        assert_eq!(*limits.global_limit().borrow(), 9);
        assert_eq!(*collector.receiver().borrow(), 9);
    }

    #[test]
    fn oom_disables_future_increases() {
        let limits = BufferLimits::new();
        for _ in 0..SUCCESSFUL_EXECUTIONS_PER_INCREASE - 1 {
            limits.on_ok();
        }
        limits.on_oom();
        for _ in 0..SUCCESSFUL_EXECUTIONS_PER_INCREASE * 2 {
            limits.on_ok();
        }
        assert_eq!(*limits.global_limit().borrow(), 8);
    }

    #[test]
    fn frequent_ooms_decrease_limits_gradually() {
        let limits = BufferLimits::new();
        limits.on_oom();
        for _ in 1..OOM_RATE_MIN_EXECUTIONS {
            limits.on_ok();
        }
        assert_eq!(*limits.global_limit().borrow(), 7);

        limits.on_oom();
        for _ in 1..OOM_RATE_MIN_EXECUTIONS {
            limits.on_ok();
        }
        assert_eq!(*limits.global_limit().borrow(), 6);
    }

    #[test]
    fn sync_execution_permits_enforce_one_shared_limit() {
        let limits = BufferLimits::new();
        let permits = (0..8)
            .map(|_| limits.try_acquire_sync_execution().expect("initial shared capacity"))
            .collect::<Vec<_>>();
        assert!(limits.try_acquire_sync_execution().is_none());

        drop(permits);
        assert!(limits.try_acquire_sync_execution().is_some());
    }
}
