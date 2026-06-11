use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, LazyLock};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tokio::sync::watch;
use tracing::debug;

pub const COPY_D2H: &str = "cuda_utils::copy_d2h_from_ptr";
pub const COPY_H2D: &str = "cuda_utils::copy_h2d_from_ptr";
pub const ENTRY: &str = "entry";

const MIN_SYNC_PARALLELISM: usize = 1;
const MAX_SYNC_PARALLELISM_MULTIPLIER: usize = 4;
const COPY_FUTURE_WEIGHT: usize = 2;
const ENTRY_FUTURE_WEIGHT: usize = 1;
const DEFAULT_FUTURE_WEIGHT: usize = 1;
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

    fn current(&self) -> usize {
        *self.receiver.borrow()
    }
}

struct FutureLimitEntry {
    limit: DynamicBufferLimit,
    registrations: usize,
    weight: usize,
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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FutureLimitProgressEntry {
    pub descr: String,
    pub limit: usize,
    pub registrations: usize,
    pub weight: usize,
}

#[derive(Debug)]
struct SyncExecutionLimiterState {
    active: usize,
    limit: usize,
    max_limit: usize,
}

#[derive(Debug)]
struct SyncExecutionLimiter {
    state: Mutex<SyncExecutionLimiterState>,
    capacity_changed: watch::Sender<usize>,
}

impl SyncExecutionLimiter {
    fn new(limit: usize, max_limit: usize) -> Self {
        assert!(limit > 0 && limit <= max_limit, "invalid sync execution limits: initial={limit} max={max_limit}");
        let (capacity_changed, _) = watch::channel(limit);
        Self {
            state: Mutex::new(SyncExecutionLimiterState {
                active: 0,
                limit,
                max_limit,
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

    fn increase_limit(&self) -> (usize, usize) {
        let mut state = self.state.lock();
        let old_limit = state.limit;
        state.limit = state.limit.saturating_add(1).min(state.max_limit);
        Self::notify_capacity_changed(&state, &self.capacity_changed);
        (old_limit, state.limit)
    }

    fn decrease_limit(&self) -> (usize, usize) {
        let mut state = self.state.lock();
        let old_limit = state.limit;
        state.limit = state.limit.saturating_sub(1).max(MIN_SYNC_PARALLELISM);
        Self::notify_capacity_changed(&state, &self.capacity_changed);
        (old_limit, state.limit)
    }

    fn limit(&self) -> usize {
        self.state.lock().limit
    }

    fn subscribe(&self) -> watch::Receiver<usize> {
        self.capacity_changed.subscribe()
    }

    fn release(&self) {
        let mut state = self.state.lock();
        assert!(state.active > 0, "released an inactive sync execution slot");
        state.active -= 1;
        Self::notify_capacity_changed(&state, &self.capacity_changed);
    }

    fn notify_capacity_changed(state: &SyncExecutionLimiterState, capacity_changed: &watch::Sender<usize>) {
        capacity_changed.send_replace(state.limit);
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
/// Active collectors share the current sync limit according to their weights. Registering a nested
/// or concurrent collector therefore lowers existing allowances; already-running futures drain,
/// while new futures observe the reduced watch value. Dynamic task executors use shared execution
/// permits, so the same limit is enforced across all per-device executor instances.
pub struct BufferLimits {
    state: Mutex<BufferLimitsState>,
    sync_execution_limiter: Arc<SyncExecutionLimiter>,
    publish_progress: bool,
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
        let initial_limit = std::thread::available_parallelism().map(usize::from).unwrap_or(MIN_SYNC_PARALLELISM);
        let max_limit = initial_limit.saturating_mul(MAX_SYNC_PARALLELISM_MULTIPLIER).max(initial_limit);
        Self::new_with_parallelism_and_progress(initial_limit, max_limit, true)
    }

    #[cfg(test)]
    fn new_with_parallelism(initial_limit: usize, max_limit: usize) -> Self {
        Self::new_with_parallelism_and_progress(initial_limit, max_limit, false)
    }

    fn new_with_parallelism_and_progress(initial_limit: usize, max_limit: usize, publish_progress: bool) -> Self {
        Self {
            state: Mutex::new(BufferLimitsState::default()),
            sync_execution_limiter: Arc::new(SyncExecutionLimiter::new(initial_limit, max_limit)),
            publish_progress,
        }
    }

    pub fn register<'a>(&'a self, descr: &str) -> BufferLimitRegistration<'a> {
        let mut state = self.state.lock();
        let receiver = {
            let entry = state.futures_limits.entry(descr.to_owned()).or_insert_with(|| FutureLimitEntry {
                limit: DynamicBufferLimit::new(1),
                registrations: 0,
                weight: Self::future_limit_weight(descr),
            });
            entry.registrations += 1;
            entry.limit.receiver()
        };
        state.active_registrations += 1;
        self.rebalance_future_limits(&state);
        debug!(descr, active_registrations = state.active_registrations, max_parallel_jobs = self.sync_execution_limiter.limit(), "registered future collector");

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
        debug!(descr, active_registrations = state.active_registrations, max_parallel_jobs = self.sync_execution_limiter.limit(), "forgot future collector");
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
            let (old_limit, new_limit) = self.sync_execution_limiter.increase_limit();
            self.rebalance_future_limits(&state);
            debug!(old_limit, new_limit, "increased adaptive parallelism after successful executions");
        }
    }

    /// Reports the current shared maximum and wakes executors when the maximum or available
    /// capacity changes.
    pub fn sync_capacity(&self) -> watch::Receiver<usize> {
        self.sync_execution_limiter.subscribe()
    }

    pub fn max_parallel_jobs(&self) -> usize {
        self.sync_execution_limiter.limit()
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

        let (old_limit, new_limit) = self.sync_execution_limiter.decrease_limit();
        state.feedback_window.clear();
        state.feedback_window_ooms = 0;
        state.consecutive_ok = 0;
        self.rebalance_future_limits(state);
        debug!(old_limit, new_limit, "decreased adaptive parallelism after frequent OOMs");
        true
    }

    fn rebalance_future_limits(&self, state: &BufferLimitsState) {
        if state.active_registrations == 0 {
            self.publish_future_limits(Vec::new());
            return;
        }

        let total_weight = state.futures_limits.values().map(|entry| entry.registrations.saturating_mul(entry.weight)).sum::<usize>();
        for entry in state.futures_limits.values() {
            let collector_limit = (self.sync_execution_limiter.limit().saturating_mul(entry.weight) / total_weight).max(1);
            entry.limit.set(collector_limit);
        }
        self.publish_future_limits(Self::future_limits_snapshot(state));
    }

    fn future_limits_snapshot(state: &BufferLimitsState) -> Vec<FutureLimitProgressEntry> {
        let mut entries = state.futures_limits.iter().map(|(descr, entry)| FutureLimitProgressEntry {
            descr: descr.clone(),
            limit: entry.limit.current(),
            registrations: entry.registrations,
            weight: entry.weight,
        }).collect::<Vec<_>>();
        entries.sort_by(|a, b| a.descr.cmp(&b.descr));
        entries
    }

    fn publish_future_limits(&self, entries: Vec<FutureLimitProgressEntry>) {
        if self.publish_progress {
            crate::collect_futures_eta::collect_eta_set_future_limits(entries);
        }
    }

    fn future_limit_weight(descr: &str) -> usize {
        match descr {
            COPY_D2H | COPY_H2D => COPY_FUTURE_WEIGHT,
            ENTRY => ENTRY_FUTURE_WEIGHT,
            _ => DEFAULT_FUTURE_WEIGHT,
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
    use super::{BufferLimits, COPY_D2H, OOM_RATE_MIN_EXECUTIONS, SUCCESSFUL_EXECUTIONS_PER_INCREASE};

    #[test]
    fn active_collectors_share_sync_limit() {
        let limits = BufferLimits::new_with_parallelism(8, 32);
        let initial_limit = limits.max_parallel_jobs();
        let first = limits.register("first");
        assert_eq!(*first.receiver().borrow(), initial_limit);

        let second = limits.register("second");
        assert_eq!(*first.receiver().borrow(), (initial_limit / 2).max(1));
        assert_eq!(*second.receiver().borrow(), (initial_limit / 2).max(1));

        let second_again = limits.register("second");
        assert_eq!(*first.receiver().borrow(), (initial_limit / 3).max(1));
        assert_eq!(*second.receiver().borrow(), (initial_limit / 3).max(1));
        assert_eq!(*second_again.receiver().borrow(), (initial_limit / 3).max(1));

        drop(second_again);
        assert_eq!(*first.receiver().borrow(), (initial_limit / 2).max(1));
        assert_eq!(*second.receiver().borrow(), (initial_limit / 2).max(1));
    }

    #[test]
    fn copy_collectors_get_a_larger_share() {
        let limits = BufferLimits::new_with_parallelism(8, 32);
        let initial_limit = limits.max_parallel_jobs();
        let copy = limits.register(COPY_D2H);
        let regular = limits.register("regular");

        assert_eq!(*copy.receiver().borrow(), (initial_limit * 2 / 3).max(1));
        assert_eq!(*regular.receiver().borrow(), (initial_limit / 3).max(1));
    }

    #[test]
    fn successful_execution_streak_increases_limits() {
        let limits = BufferLimits::new_with_parallelism(8, 32);
        let initial_limit = limits.max_parallel_jobs();
        let collector = limits.register("collector");

        for _ in 0..SUCCESSFUL_EXECUTIONS_PER_INCREASE {
            limits.on_ok();
        }

        assert_eq!(limits.max_parallel_jobs(), initial_limit + 1);
        assert_eq!(*collector.receiver().borrow(), initial_limit + 1);
    }

    #[test]
    fn oom_disables_future_increases() {
        let limits = BufferLimits::new_with_parallelism(8, 32);
        let initial_limit = limits.max_parallel_jobs();
        for _ in 0..SUCCESSFUL_EXECUTIONS_PER_INCREASE - 1 {
            limits.on_ok();
        }
        limits.on_oom();
        for _ in 0..SUCCESSFUL_EXECUTIONS_PER_INCREASE * 2 {
            limits.on_ok();
        }
        assert_eq!(limits.max_parallel_jobs(), initial_limit);
    }

    #[test]
    fn frequent_ooms_decrease_limits_gradually() {
        let limits = BufferLimits::new_with_parallelism(8, 32);
        let initial_limit = limits.max_parallel_jobs();
        limits.on_oom();
        for _ in 1..OOM_RATE_MIN_EXECUTIONS {
            limits.on_ok();
        }
        assert_eq!(limits.max_parallel_jobs(), initial_limit.saturating_sub(1).max(1));

        limits.on_oom();
        for _ in 1..OOM_RATE_MIN_EXECUTIONS {
            limits.on_ok();
        }
        assert_eq!(limits.max_parallel_jobs(), initial_limit.saturating_sub(2).max(1));
    }

    #[test]
    fn sync_execution_permits_enforce_one_shared_limit() {
        let limits = BufferLimits::new_with_parallelism(8, 32);
        let initial_limit = limits.max_parallel_jobs();
        let mut capacity_rx = limits.sync_capacity();
        capacity_rx.borrow_and_update();
        let permits = (0..initial_limit)
            .map(|_| limits.try_acquire_sync_execution().expect("initial shared capacity"))
            .collect::<Vec<_>>();
        assert!(limits.try_acquire_sync_execution().is_none());

        drop(permits);
        assert!(capacity_rx.has_changed().expect("capacity sender must remain open"));
        assert_eq!(*capacity_rx.borrow_and_update(), initial_limit);
        assert!(limits.try_acquire_sync_execution().is_some());
    }
}
