use log::info;
use once_cell::sync::Lazy;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use crate::LOCK_TIMEOUT;

pub(crate) const COLLECT_ETA_LOG_INTERVAL: Duration = Duration::from_secs(10);

#[derive(Debug)]
struct CollectEtaInvocation {
    descr: String,
    total: usize,
    completed: usize,
    started_at: Instant,
    last_completion_at: Option<Instant>,
    warmup: usize,
    ewma_interval_s: Option<f64>,
    alpha: f64,
}

impl CollectEtaInvocation {
    fn new(descr: &str, total: usize, concurrency: usize) -> Self {
        let warmup = concurrency.max(1).min(total.max(1));
        Self {
            descr: descr.to_string(),
            total,
            completed: 0,
            started_at: Instant::now(),
            last_completion_at: None,
            warmup,
            ewma_interval_s: None,
            alpha: 0.15,
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
            let dt = now.saturating_duration_since(last_completion_at).as_secs_f64().max(1e-6);
            self.ewma_interval_s = Some(match self.ewma_interval_s {
                None => dt,
                Some(prev) => self.alpha * dt + (1.0 - self.alpha) * prev,
            });
        }
        self.last_completion_at = Some(now);
    }

    fn conservative_rate_per_sec(&self, now: Instant) -> Option<f64> {
        if self.completed == 0 {
            return None;
        }
        let elapsed_s = now.saturating_duration_since(self.started_at).as_secs_f64();
        if elapsed_s <= 0.0 {
            return None;
        }
        let elapsed_rate = self.completed as f64 / elapsed_s;
        let ewma_rate = self.ewma_interval_s.map(|interval| 1.0 / interval.max(1e-6));
        let rate = match ewma_rate {
            Some(r) if r.is_finite() && r > 0.0 => elapsed_rate.min(r),
            _ => elapsed_rate,
        };
        (rate.is_finite() && rate > 0.0).then_some(rate)
    }

    fn eta_interval_s(&self, now: Instant) -> Option<f64> {
        if self.completed < self.warmup {
            return None;
        }
        self.conservative_rate_per_sec(now).map(|rate| 1.0 / rate)
    }

    fn historical_interval_s(&self, now: Instant) -> Option<f64> {
        if self.completed == 0 {
            return None;
        }
        let elapsed_s = now.saturating_duration_since(self.started_at).as_secs_f64();
        if elapsed_s <= 0.0 {
            return None;
        }
        Some((elapsed_s / self.completed as f64).max(1e-6))
    }

    fn eta(&self, now: Instant) -> Option<Duration> {
        let remaining = self.remaining();
        if remaining == 0 {
            return Some(Duration::ZERO);
        }
        let interval_s = self.eta_interval_s(now)?;
        Some(Duration::from_secs_f64(remaining as f64 * interval_s))
    }
}

#[derive(Debug, Default)]
struct CollectEtaState {
    invocations: HashMap<u64, CollectEtaInvocation>,
    last_log_at: Option<Instant>,
    historical_interval_s: Option<f64>,
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

    fn state_guard(&self) -> parking_lot::MutexGuard<'_, CollectEtaState> {
        self.state.try_lock_for(LOCK_TIMEOUT).expect("CollectEtaTracker::state_guard: timed out waiting for tracker lock")
    }

    fn register(&self, descr: &str, total: usize, concurrency: usize) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.state_guard().invocations.insert(id, CollectEtaInvocation::new(descr, total, concurrency));
        id
    }

    fn unregister(&self, id: u64) {
        self.unregister_at(id, Instant::now());
    }

    fn record_completion(&self, id: u64) {
        self.record_completion_at(id, Instant::now());
    }

    fn record_completion_at(&self, id: u64, now: Instant) {
        if let Some(invocation) = self.state_guard().invocations.get_mut(&id) {
            invocation.record_completion(now);
        }
    }

    fn unregister_at(&self, id: u64, now: Instant) {
        let mut state = self.state_guard();
        if let Some(invocation) = state.invocations.remove(&id) {
            if invocation.completed > 0 {
                if let Some(interval_s) = invocation.historical_interval_s(now) {
                    Self::update_historical_interval(&mut state, interval_s, invocation.completed as u64);
                }
            }
        }
    }

    fn update_historical_interval(state: &mut CollectEtaState, interval_s: f64, completed_weight: u64) {
        if completed_weight == 0 {
            return;
        }
        let next_completed = state.historical_completed.saturating_add(completed_weight);
        state.historical_interval_s = Some(match (state.historical_interval_s, state.historical_completed) {
            (Some(current_interval), current_weight) if current_weight > 0 => {
                weighted_f64_avg(current_interval, current_weight, interval_s, completed_weight)
            }
            _ => interval_s,
        });
        state.historical_completed = next_completed;
    }

    fn build_snapshot(state: &CollectEtaState, now: Instant) -> CollectEtaSnapshot {
        let mut snapshot = CollectEtaSnapshot {
            active_collectors: state.invocations.len(),
            ..Default::default()
        };
        let mut fallback_interval_s = state.historical_interval_s;

        for invocation in state.invocations.values() {
            let remaining = invocation.remaining();
            snapshot.total_completed = snapshot.total_completed.saturating_add(invocation.completed);
            snapshot.total_remaining = snapshot.total_remaining.saturating_add(remaining);
            if let Some(interval_s) = invocation.eta_interval_s(now) {
                fallback_interval_s = Some(fallback_interval_s.map(|current| current.max(interval_s)).unwrap_or(interval_s));
            }
        }

        for invocation in state.invocations.values() {
            let remaining = invocation.remaining();
            if remaining == 0 {
                continue;
            }
            let eta = invocation
                .eta(now)
                .or_else(|| fallback_interval_s.map(|interval_s| Duration::from_secs_f64(remaining as f64 * interval_s)));
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
            let mut state = self.state_guard();
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
            let unknown_reason = if snapshot.total_completed == 0 { "waiting for first completions" } else { "warming up throughput model" };
            info!("collect_futures_with_progress ETA: active={} completed={} remaining={} max_eta=unknown ({})", snapshot.active_collectors, snapshot.total_completed, snapshot.total_remaining, unknown_reason);
        }
        true
    }
}

#[derive(Debug)]
pub(crate) struct CollectEtaInvocationGuard {
    id: u64,
}

impl CollectEtaInvocationGuard {
    pub(crate) fn new(descr: &str, total: usize, concurrency: usize) -> Self {
        Self {
            id: COLLECT_ETA_TRACKER.register(descr, total, concurrency),
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

fn weighted_f64_avg(a: f64, a_weight: u64, b: f64, b_weight: u64) -> f64 {
    if a_weight == 0 {
        return b;
    }
    if b_weight == 0 {
        return a;
    }
    let total_weight = a_weight.saturating_add(b_weight);
    if total_weight == 0 {
        return 0.0;
    }
    (a * a_weight as f64 + b * b_weight as f64) / total_weight as f64
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
    fn snapshot_waits_for_warmup_before_estimating() {
        let tracker = CollectEtaTracker::new();
        let id = tracker.register("warmup", 6, 3);
        let now = Instant::now();
        tracker.record_completion_at(id, now + Duration::from_secs(1));
        tracker.record_completion_at(id, now + Duration::from_secs(2));
        let state = tracker.state_guard();
        let snapshot = CollectEtaTracker::build_snapshot(&state, now + Duration::from_secs(2));
        assert_eq!(snapshot.max_eta, None);
        drop(state);

        tracker.record_completion_at(id, now + Duration::from_secs(3));
        let state = tracker.state_guard();
        let snapshot = CollectEtaTracker::build_snapshot(&state, now + Duration::from_secs(3));
        assert_eq!(snapshot.total_completed, 3);
        assert_eq!(snapshot.total_remaining, 3);
        let eta = snapshot.max_eta.expect("eta should be present after warmup");
        assert!(eta.as_secs_f64() >= 2.0 && eta.as_secs_f64() <= 4.0, "eta={eta:?}");
    }

    #[test]
    fn snapshot_uses_max_eta_across_collectors() {
        let tracker = CollectEtaTracker::new();
        let now = Instant::now();
        let fast = tracker.register("fast", 5, 2);
        let slow = tracker.register("slow", 5, 2);
        tracker.record_completion_at(fast, now + Duration::from_secs(1));
        tracker.record_completion_at(fast, now + Duration::from_secs(2));
        tracker.record_completion_at(slow, now + Duration::from_secs(1));
        tracker.record_completion_at(slow, now + Duration::from_secs(5));
        let state = tracker.state_guard();
        let snapshot = CollectEtaTracker::build_snapshot(&state, now + Duration::from_secs(5));
        assert_eq!(snapshot.total_completed, 4);
        assert_eq!(snapshot.total_remaining, 6);
        let max_eta = snapshot.max_eta.expect("max eta should be present");
        assert!(max_eta.as_secs_f64() >= 11.0 && max_eta.as_secs_f64() <= 13.0, "max_eta={max_eta:?}");
        assert_eq!(snapshot.slowest_descr.as_deref(), Some("slow"));
    }

    #[test]
    fn maybe_log_respects_interval_and_force() {
        let tracker = CollectEtaTracker::new();
        let now = Instant::now();
        let _ = tracker.register("gated", 2, 1);
        assert!(tracker.maybe_log_at(false, now));
        assert!(!tracker.maybe_log_at(false, now + Duration::from_secs(3)));
        assert!(tracker.maybe_log_at(false, now + COLLECT_ETA_LOG_INTERVAL));
        assert!(!tracker.maybe_log_at(false, now + COLLECT_ETA_LOG_INTERVAL + Duration::from_secs(1)));
        assert!(tracker.maybe_log_at(true, now + COLLECT_ETA_LOG_INTERVAL + Duration::from_secs(2)));
    }

    #[test]
    fn snapshot_estimates_from_completion_rate_without_zero_ms() {
        let tracker = CollectEtaTracker::new();
        let id = tracker.register("parallelish", 20, 5);
        let now = Instant::now();
        tracker.record_completion_at(id, now + Duration::from_millis(1000));
        tracker.record_completion_at(id, now + Duration::from_millis(1200));
        tracker.record_completion_at(id, now + Duration::from_millis(1400));
        tracker.record_completion_at(id, now + Duration::from_millis(1600));
        tracker.record_completion_at(id, now + Duration::from_millis(1800));
        let state = tracker.state_guard();
        let snapshot = CollectEtaTracker::build_snapshot(&state, now + Duration::from_millis(1800));
        assert_eq!(snapshot.total_completed, 5);
        assert_eq!(snapshot.total_remaining, 15);
        let eta = snapshot.max_eta.expect("eta should be present");
        assert!(eta.as_secs_f64() >= 4.0, "eta was unexpectedly small: {eta:?}");
    }

    #[test]
    fn snapshot_uses_historical_interval_for_collectors_without_completions() {
        let tracker = CollectEtaTracker::new();
        let now = Instant::now();

        let done = tracker.register("done", 2, 2);
        tracker.record_completion_at(done, now + Duration::from_secs(1));
        tracker.record_completion_at(done, now + Duration::from_secs(2));
        tracker.unregister_at(done, now + Duration::from_secs(2));

        let _pending = tracker.register("pending", 3, 3);
        let state = tracker.state_guard();
        let snapshot = CollectEtaTracker::build_snapshot(&state, now + Duration::from_secs(2));
        assert_eq!(snapshot.total_completed, 0);
        assert_eq!(snapshot.total_remaining, 3);
        let max_eta = snapshot.max_eta.expect("historical fallback eta should be present");
        assert!(max_eta.as_secs_f64() >= 2.0 && max_eta.as_secs_f64() <= 4.0, "max_eta={max_eta:?}");
        assert_eq!(snapshot.slowest_descr.as_deref(), Some("pending"));
    }
}
