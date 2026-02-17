use log::info;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

pub(crate) const COLLECT_ETA_LOG_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Debug)]
struct CollectEtaInvocation {
    descr: String,
    total: usize,
    completed: usize,
    started_at: Instant,
    last_completion_at: Option<Instant>,
    completion_gap_sum: Duration,
    completion_gap_count: u64,
}

impl CollectEtaInvocation {
    fn new(descr: &str, total: usize) -> Self {
        Self {
            descr: descr.to_string(),
            total,
            completed: 0,
            started_at: Instant::now(),
            last_completion_at: None,
            completion_gap_sum: Duration::ZERO,
            completion_gap_count: 0,
        }
    }

    fn remaining(&self) -> usize {
        self.total.saturating_sub(self.completed)
    }

    fn record_completion(&mut self, now: Instant) {
        if self.completed >= self.total {
            return;
        }
        self.completed = self.completed.saturating_add(1);
        if let Some(last_completion_at) = self.last_completion_at {
            let gap = now.saturating_duration_since(last_completion_at);
            self.completion_gap_sum = self.completion_gap_sum.checked_add(gap).unwrap_or(Duration::MAX);
            self.completion_gap_count = self.completion_gap_count.saturating_add(1);
        }
        self.last_completion_at = Some(now);
    }

    fn avg_completion_time(&self, now: Instant) -> Option<Duration> {
        if self.completed == 0 {
            return None;
        }
        let elapsed_avg = div_duration(now.saturating_duration_since(self.started_at), self.completed as u64);
        if self.completion_gap_count == 0 {
            return Some(elapsed_avg);
        }
        let gap_avg = div_duration(self.completion_gap_sum, self.completion_gap_count);
        Some(elapsed_avg.max(gap_avg))
    }

    fn eta(&self, now: Instant) -> Option<Duration> {
        let remaining = self.remaining();
        if remaining == 0 {
            return Some(Duration::ZERO);
        }
        self.avg_completion_time(now).map(|avg| mul_duration(avg, remaining))
    }
}

#[derive(Debug, Default)]
struct CollectEtaState {
    invocations: HashMap<u64, CollectEtaInvocation>,
    last_log_at: Option<Instant>,
    historical_avg_completion: Option<Duration>,
    historical_completed: u64,
}

#[derive(Debug, Default)]
struct CollectEtaSnapshot {
    active_collectors: usize,
    total_completed: usize,
    total_remaining: usize,
    max_eta: Option<Duration>,
    slowest_descr: Option<String>,
    slowest_completed: usize,
    slowest_total: usize,
}

#[derive(Debug)]
struct CollectEtaTracker {
    next_id: AtomicU64,
    state: Mutex<CollectEtaState>,
}

impl CollectEtaTracker {
    fn new() -> Self {
        Self {
            next_id: AtomicU64::new(1),
            state: Mutex::new(CollectEtaState::default()),
        }
    }

    fn register(&self, descr: &str, total: usize) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.state.lock().invocations.insert(id, CollectEtaInvocation::new(descr, total));
        id
    }

    fn unregister(&self, id: u64) {
        self.unregister_at(id, Instant::now());
    }

    fn record_completion(&self, id: u64) {
        self.record_completion_at(id, Instant::now());
    }

    fn record_completion_at(&self, id: u64, now: Instant) {
        if let Some(invocation) = self.state.lock().invocations.get_mut(&id) {
            invocation.record_completion(now);
        }
    }

    fn unregister_at(&self, id: u64, now: Instant) {
        let mut state = self.state.lock();
        if let Some(invocation) = state.invocations.remove(&id) {
            if invocation.completed > 0 {
                if let Some(avg) = invocation.avg_completion_time(now) {
                    Self::update_historical_avg(&mut state, avg, invocation.completed as u64);
                }
            }
        }
    }

    fn update_historical_avg(state: &mut CollectEtaState, avg: Duration, completed_weight: u64) {
        if completed_weight == 0 {
            return;
        }
        let next_completed = state.historical_completed.saturating_add(completed_weight);
        state.historical_avg_completion = Some(match (state.historical_avg_completion, state.historical_completed) {
            (Some(current_avg), current_weight) if current_weight > 0 => {
                weighted_duration_avg(current_avg, current_weight, avg, completed_weight)
            }
            _ => avg,
        });
        state.historical_completed = next_completed;
    }

    fn build_snapshot(state: &CollectEtaState, now: Instant) -> CollectEtaSnapshot {
        let mut snapshot = CollectEtaSnapshot {
            active_collectors: state.invocations.len(),
            ..Default::default()
        };
        let mut fallback_avg = state.historical_avg_completion;

        for invocation in state.invocations.values() {
            let remaining = invocation.remaining();
            snapshot.total_completed = snapshot.total_completed.saturating_add(invocation.completed);
            snapshot.total_remaining = snapshot.total_remaining.saturating_add(remaining);
            if let Some(avg) = invocation.avg_completion_time(now) {
                fallback_avg = Some(fallback_avg.map(|current| current.max(avg)).unwrap_or(avg));
            }
        }

        for invocation in state.invocations.values() {
            let remaining = invocation.remaining();
            if remaining == 0 {
                continue;
            }
            let eta = invocation.eta(now).or_else(|| fallback_avg.map(|avg| mul_duration(avg, remaining)));
            if let Some(eta) = eta {
                let is_slowest = snapshot.max_eta.map(|current| eta > current).unwrap_or(true);
                if is_slowest {
                    snapshot.max_eta = Some(eta);
                    snapshot.slowest_descr = Some(invocation.descr.clone());
                    snapshot.slowest_completed = invocation.completed;
                    snapshot.slowest_total = invocation.total;
                }
            }
        }
        snapshot
    }

    fn maybe_log(&self, force: bool) -> bool {
        self.maybe_log_at(force, Instant::now())
    }

    fn maybe_log_at(&self, force: bool, now: Instant) -> bool {
        let snapshot = {
            let mut state = self.state.lock();
            if state.invocations.is_empty() {
                return false;
            }
            if !force {
                if let Some(last_log_at) = state.last_log_at {
                    if now.saturating_duration_since(last_log_at) < COLLECT_ETA_LOG_INTERVAL {
                        return false;
                    }
                }
            }
            state.last_log_at = Some(now);
            Self::build_snapshot(&state, now)
        };

        if let Some(max_eta) = snapshot.max_eta {
            info!("collect_futures_with_progress ETA: active={} completed={} remaining={} max_eta={} slowest=\"{}\" ({}/{})", snapshot.active_collectors, snapshot.total_completed, snapshot.total_remaining, format_duration(max_eta), snapshot.slowest_descr.as_deref().unwrap_or("unknown"), snapshot.slowest_completed, snapshot.slowest_total);
        } else {
            info!("collect_futures_with_progress ETA: active={} completed={} remaining={} max_eta=unknown (waiting for first completions)", snapshot.active_collectors, snapshot.total_completed, snapshot.total_remaining);
        }
        true
    }
}

#[derive(Debug)]
pub(crate) struct CollectEtaInvocationGuard {
    id: u64,
}

impl CollectEtaInvocationGuard {
    pub(crate) fn new(descr: &str, total: usize) -> Self {
        Self {
            id: COLLECT_ETA_TRACKER.register(descr, total),
        }
    }

    pub(crate) fn id(&self) -> u64 {
        self.id
    }
}

impl Drop for CollectEtaInvocationGuard {
    fn drop(&mut self) {
        COLLECT_ETA_TRACKER.unregister(self.id);
    }
}

static COLLECT_ETA_TRACKER: Lazy<CollectEtaTracker> = Lazy::new(CollectEtaTracker::new);

pub(crate) fn collect_eta_record_completion(id: u64) {
    COLLECT_ETA_TRACKER.record_completion(id);
}

pub(crate) fn collect_eta_maybe_log(force: bool) {
    let _ = COLLECT_ETA_TRACKER.maybe_log(force);
}

fn duration_from_nanos_saturating(nanos: u128) -> Duration {
    let secs = nanos / 1_000_000_000;
    if secs > u64::MAX as u128 {
        return Duration::MAX;
    }
    Duration::new(secs as u64, (nanos % 1_000_000_000) as u32)
}

fn div_duration(duration: Duration, by: u64) -> Duration {
    if by == 0 {
        return Duration::ZERO;
    }
    duration_from_nanos_saturating(duration.as_nanos() / by as u128)
}

fn mul_duration(duration: Duration, mul: usize) -> Duration {
    duration_from_nanos_saturating(duration.as_nanos().saturating_mul(mul as u128))
}

fn weighted_duration_avg(a: Duration, a_weight: u64, b: Duration, b_weight: u64) -> Duration {
    if a_weight == 0 {
        return b;
    }
    if b_weight == 0 {
        return a;
    }
    let total_weight = a_weight.saturating_add(b_weight);
    if total_weight == 0 {
        return Duration::ZERO;
    }
    let weighted_sum = a
        .as_nanos()
        .saturating_mul(a_weight as u128)
        .saturating_add(b.as_nanos().saturating_mul(b_weight as u128));
    duration_from_nanos_saturating(weighted_sum / total_weight as u128)
}

fn format_duration(duration: Duration) -> String {
    if duration.is_zero() {
        return "0s".to_string();
    }

    let secs = duration.as_secs();
    if secs == 0 {
        let millis = duration.as_millis();
        if millis > 0 {
            return format!("{millis}ms");
        }
        return "<1ms".to_string();
    }

    let days = secs / 86_400;
    let hours = (secs % 86_400) / 3_600;
    let mins = (secs % 3_600) / 60;
    let sec = secs % 60;

    if days > 0 {
        format!("{days}d {hours:02}h {mins:02}m")
    } else if hours > 0 {
        format!("{hours}h {mins:02}m {sec:02}s")
    } else if mins > 0 {
        format!("{mins}m {sec:02}s")
    } else {
        format!("{sec}s")
    }
}

#[cfg(test)]
mod tests {
    use super::{CollectEtaTracker, COLLECT_ETA_LOG_INTERVAL};
    use std::time::{Duration, Instant};

    #[test]
    fn snapshot_uses_completion_gaps_for_eta() {
        let tracker = CollectEtaTracker::new();
        let id = tracker.register("self_attention", 4);
        let now = Instant::now();
        tracker.record_completion_at(id, now + Duration::from_secs(2));
        tracker.record_completion_at(id, now + Duration::from_secs(5));
        let state = tracker.state.lock();
        let snapshot = CollectEtaTracker::build_snapshot(&state, now + Duration::from_secs(5));
        assert_eq!(snapshot.total_completed, 2);
        assert_eq!(snapshot.total_remaining, 2);
        assert_eq!(snapshot.max_eta, Some(Duration::from_secs(6)));
        assert_eq!(snapshot.slowest_descr.as_deref(), Some("self_attention"));
        assert_eq!(snapshot.slowest_completed, 2);
        assert_eq!(snapshot.slowest_total, 4);
    }

    #[test]
    fn snapshot_uses_max_eta_across_collectors() {
        let tracker = CollectEtaTracker::new();
        let now = Instant::now();
        let fast = tracker.register("fast", 5);
        let slow = tracker.register("slow", 5);
        tracker.record_completion_at(fast, now + Duration::from_secs(1));
        tracker.record_completion_at(fast, now + Duration::from_secs(2));
        tracker.record_completion_at(slow, now + Duration::from_secs(1));
        tracker.record_completion_at(slow, now + Duration::from_secs(5));
        let state = tracker.state.lock();
        let snapshot = CollectEtaTracker::build_snapshot(&state, now + Duration::from_secs(5));
        assert_eq!(snapshot.total_completed, 4);
        assert_eq!(snapshot.total_remaining, 6);
        assert_eq!(snapshot.max_eta, Some(Duration::from_secs(12)));
        assert_eq!(snapshot.slowest_descr.as_deref(), Some("slow"));
    }

    #[test]
    fn maybe_log_respects_interval_and_force() {
        let tracker = CollectEtaTracker::new();
        let now = Instant::now();
        let _ = tracker.register("gated", 2);
        assert!(tracker.maybe_log_at(false, now));
        assert!(!tracker.maybe_log_at(false, now + Duration::from_secs(3)));
        assert!(tracker.maybe_log_at(false, now + COLLECT_ETA_LOG_INTERVAL));
        assert!(!tracker.maybe_log_at(false, now + COLLECT_ETA_LOG_INTERVAL + Duration::from_secs(1)));
        assert!(tracker.maybe_log_at(true, now + COLLECT_ETA_LOG_INTERVAL + Duration::from_secs(2)));
    }

    #[test]
    fn snapshot_uses_elapsed_average_when_gap_average_is_too_small() {
        let tracker = CollectEtaTracker::new();
        let id = tracker.register("bursty", 4);
        let now = Instant::now();
        tracker.record_completion_at(id, now + Duration::from_secs(10));
        tracker.record_completion_at(id, now + Duration::from_secs(10) + Duration::from_millis(1));
        let state = tracker.state.lock();
        let snapshot = CollectEtaTracker::build_snapshot(&state, now + Duration::from_secs(10) + Duration::from_millis(1));
        assert_eq!(snapshot.total_completed, 2);
        assert_eq!(snapshot.total_remaining, 2);
        let eta = snapshot.max_eta.expect("eta should be present");
        assert!(eta >= Duration::from_secs(10), "eta was unexpectedly small: {eta:?}");
    }

    #[test]
    fn snapshot_uses_historical_average_for_collectors_without_completions() {
        let tracker = CollectEtaTracker::new();
        let now = Instant::now();

        let done = tracker.register("done", 2);
        tracker.record_completion_at(done, now + Duration::from_secs(2));
        tracker.record_completion_at(done, now + Duration::from_secs(4));
        tracker.unregister_at(done, now + Duration::from_secs(4));

        let _pending = tracker.register("pending", 3);
        let state = tracker.state.lock();
        let snapshot = CollectEtaTracker::build_snapshot(&state, now + Duration::from_secs(4));
        assert_eq!(snapshot.total_completed, 0);
        assert_eq!(snapshot.total_remaining, 3);
        assert_eq!(snapshot.max_eta, Some(Duration::from_secs(6)));
        assert_eq!(snapshot.slowest_descr.as_deref(), Some("pending"));
    }
}
