use std::collections::HashMap;
use std::fmt::Debug;
use std::fs::File;
use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::atomic::Ordering::Relaxed;
use std::sync::{LazyLock, OnceLock};
use std::time::{Duration, Instant};
use dashmap::DashMap;
use tracing::{debug, Level};
use tracing::field::Field;
use tracing_subscriber::{fmt, fmt::{format::Writer, time::FormatTime}, layer::SubscriberExt, registry, EnvFilter, Layer};
use tracing_subscriber::layer::Context;
use tracing_subscriber::registry::LookupSpan;
use tracing_subscriber::util::SubscriberInitExt;
use crate::ThoosaError;

#[derive(Clone)]
pub struct SimplelogLikeEvent<T> {
    timer: T,
}

impl<T> SimplelogLikeEvent<T> {
    pub fn new(timer: T) -> Self {
        Self { timer }
    }
}



fn write_level_colored(
    w: &mut fmt::format::Writer<'_>,
    level: &tracing::Level,
) -> std::fmt::Result {

    let (start, end) = match *level {
        Level::ERROR => ("\x1b[31m", "\x1b[0m"), // red
        Level::WARN  => ("\x1b[33m", "\x1b[0m"), // yellow
        Level::INFO  => ("\x1b[32m", "\x1b[0m"), // green
        Level::DEBUG => ("\x1b[34m", "\x1b[0m"), // blue
        Level::TRACE => ("\x1b[35m", "\x1b[0m"), // magenta
    };
    write!(w, "{}[{}]{} ", start, level.as_str(), end)

}

impl<S, N, T> fmt::FormatEvent<S, N> for SimplelogLikeEvent<T>
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
    N: for<'a> fmt::FormatFields<'a> + 'static,
    T: FormatTime + Clone + 'static,
{
    fn format_event(
        &self,
        _ctx: &fmt::FmtContext<'_, S, N>,
        mut w: Writer<'_>,
        event: &tracing::Event<'_>,
    ) -> std::fmt::Result {
        // timestamp
        self.timer.format_time(&mut w)?;
        write!(w, " ")?;

        let meta = event.metadata();

        // level
        write_level_colored(&mut w, meta.level())?;

        // thread name
        let thread = std::thread::current();
        let tname = thread.name().unwrap_or("unknown");
        write!(w, "({}) ", tname)?;

        // target (module path)
        write!(w, "{}: ", meta.target())?;

        // print message first, then other fields (time.busy/time.idle, etc.)
        struct Visitor {
            msg: Option<String>,
            rest: Vec<(String, String)>,
        }
        impl tracing::field::Visit for Visitor {
            fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
                if field.name() == "message" {
                    self.msg = Some(value.to_string());
                } else {
                    self.rest.push((field.name().to_string(), value.to_string()));
                }
            }
            fn record_debug(
                &mut self,
                field: &tracing::field::Field,
                value: &dyn std::fmt::Debug,
            ) {
                if field.name() == "message" {
                    self.msg = Some(format!("{value:?}"));
                } else {
                    self.rest
                        .push((field.name().to_string(), format!("{value:?}")));
                }
            }
        }

        let mut v = Visitor { msg: None, rest: vec![] };
        event.record(&mut v);

        if let Some(msg) = v.msg {
            write!(w, "{msg}")?;
        }
        for (k, val) in v.rest {
            write!(w, " {k}={val}")?;
        }

        writeln!(w)
    }
}

/// Perf-only tracing to stdout, but formatted like your simplelog lines.
pub fn maybe_init_perf_tracing_stdout_simplelog_like(perf_filter: &str, perf_file: PathBuf) -> Result<bool, ThoosaError> {
    if perf_filter.trim() == "off" {
        return Ok((false));
    }

    let perf_f = File::create(perf_file)?;
    let (perf_writer, perf_guard) = tracing_appender::non_blocking(perf_f);

    // Local timestamp like: `2025-12-15 08:35:30`
    // (requires tracing-subscriber chrono support for ChronoLocal)
    let timer = tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S".to_string());

    let layer = fmt::layer()
        .with_ansi(true)
        .with_writer(perf_writer)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .event_format(SimplelogLikeEvent::new(timer))
        .with_filter(EnvFilter::new(perf_filter));

    registry()
        .with(layer)
        .with(PerfLayer::new())
        .try_init()
        .map(|v|true)
        .map_err(|e|ThoosaError::new(e.to_string()))
}


#[derive(Debug)]
struct SpanTiming {
    created_at: Instant,
    busy_ns: u64,//
    entered_at: Option<Instant>,
}

impl SpanTiming {
    fn new() -> Self {
        Self {
            created_at: Instant::now(),
            busy_ns: 0,
            entered_at: None,
        }
    }
}

#[derive(Debug)]
pub struct PerfAgg {
    busy_ns: AtomicU64,
    elapsed_ns: AtomicU64,
    busy_calls: AtomicU64,
}

impl PerfAgg {
    pub const fn new() -> Self {
        Self {
            busy_ns: AtomicU64::new(0),
            elapsed_ns: AtomicU64::new(0),
            busy_calls: AtomicU64::new(0),
        }
    }

    pub fn reset(&self) -> (u64, u64) {
        (
            self.busy_ns.swap(0, Relaxed),
            self.busy_calls.swap(0, Relaxed)
        )
    }

    pub fn busy_time(&self) -> Duration {
        Duration::from_nanos(self.busy_ns.load(Relaxed))
    }

    pub fn track(&self, span_timing:&SpanTiming) {
        self.busy_ns.fetch_add(span_timing.busy_ns, Relaxed);
        self.elapsed_ns.fetch_add(span_timing.created_at.elapsed().as_nanos() as u64, Relaxed);
        self.busy_calls.fetch_add(1, Relaxed);
    }

}

#[derive(Default)]
struct AggPerfFields {
    agg_perf: Option<String>,
}

impl tracing::field::Visit for AggPerfFields {
    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        if field.name() == "agg_perf" {
            self.agg_perf = Some(value.to_string());
        }
    }

    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "agg_perf" {
            self.agg_perf = Some(format!("{value:?}"));
        }
    }
}

/// Global, so tests can call reset/print between blocks.
pub static PERF: LazyLock<DashMap<String, PerfAgg>> =  LazyLock::new(||DashMap::new());

/// Accumulates busy time for spans with `target == "perf"`.
pub struct PerfLayer;

impl PerfLayer {
    pub fn new() -> Self {
        Self {}
    }
}

impl<S> Layer<S> for PerfLayer
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &tracing::span::Id, ctx: Context<'_, S>) {

        if let Some(span) = ctx.span(id) {

            let mut v = AggPerfFields { agg_perf: None };
            attrs.record(&mut v);

            if v.agg_perf.is_some() {
                log::debug!("PerfLayer::on_new_span: span.metadata={:?}", span.metadata());

                span.extensions_mut().insert(v);
                span.extensions_mut().insert(SpanTiming::new());
            }
        }
    }

    fn on_enter(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {

        let Some(span) = ctx.span(id) else {
            return
        };

        let mut exts = span.extensions_mut();
        let Some(t) = exts.get_mut::<SpanTiming>() else { return };
        t.entered_at = Some(Instant::now());

        log::debug!("PerfLayer::on_enter: t.entered_at={:?}", t.entered_at);
    }

    fn on_exit(&self, id: &tracing::span::Id, ctx: Context<'_, S>) {

        let Some(span) = ctx.span(id) else { return };

        let mut exts = span.extensions_mut();
        let Some(t) = exts.get_mut::<SpanTiming>() else { return };

        if let Some(entered) = t.entered_at.take() {
            log::debug!("PerfLayer::on_exit: entered={:?}", entered);
            t.busy_ns = t.busy_ns.saturating_add(entered.elapsed().as_nanos() as u64);
        }
    }

    fn on_close(&self, id: tracing::span::Id, ctx: Context<'_, S>) {

        let Some(span) = ctx.span(&id) else { return };

        let extensions = span.extensions();

        if let Some(agg_perf) = extensions
            .get::<AggPerfFields>()
            .and_then(|f| f.agg_perf.as_deref()) {

            log::debug!("PerfLayer::on_close: agg_perf={agg_perf:?}");

            if let Some(t)  = extensions.get::<SpanTiming>() {
                let perf = PERF.entry(agg_perf.to_string()).or_insert(PerfAgg::new());

                perf.track(t);
            }
        }
    }
}

fn fmt_grouped_u64(n: u64) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        out.push(b as char);
        let left = bytes.len() - i - 1;
        if left > 0 && left % 3 == 0 {
            out.push(' ');
        }
    }
    out
}

pub fn format_perf_map(map: &DashMap<String, PerfAgg>) -> String {
    use std::fmt::Write;
    use std::sync::atomic::Ordering;

    // Snapshot + sort by key
    let mut rows: Vec<(String, u64, u64, u64)> = map
        .iter()
        .map(|e| {
            let k = e.key().clone();
            let busy_mls = e.value().busy_ns.load(Ordering::Relaxed) / 1_000_000;
            let elapsed_mls = e.value().elapsed_ns.load(Ordering::Relaxed) / 1_000_000; // / 1000;
            let calls = e.value().busy_calls.load(Ordering::Relaxed);
            (k, busy_mls, elapsed_mls, calls)
        })
        .collect();

    rows.sort_by(|a, b| a.0.cmp(&b.0));

    // Headers
    let key_header = "Key";
    let busy_header = "busy time, mls";
    let elapsed_header = "elapsed time, mls";
    let calls_header = "invocations";
    let per_header = "per inv.";

    // Pre-format numbers once (keeps widths correct + avoids repeated allocs)
    let mut fmt_rows: Vec<(String, String, String, String, String)> = Vec::with_capacity(rows.len());
    for (k, busy, elapsed, calls) in rows {
        let per = if calls == 0 { 0 } else { busy / calls };
        fmt_rows.push((
            k,
            fmt_grouped_u64(busy),
            fmt_grouped_u64(elapsed),
            fmt_grouped_u64(calls),
            fmt_grouped_u64(per),
        ));
    }

    // Column widths
    let key_w = fmt_rows
        .iter()
        .map(|(k, _, _, _, _)| k.len())
        .max()
        .unwrap_or(0)
        .max(key_header.len())
        .min(140);

    let busy_w = fmt_rows
        .iter()
        .map(|(_, b, _, _, _)| b.len())
        .max()
        .unwrap_or(0)
        .max(busy_header.len());

    let elapsed_w = fmt_rows
        .iter()
        .map(|(_, _, e, _, _)| e.len())
        .max()
        .unwrap_or(0)
        .max(elapsed_header.len());

    let calls_w = fmt_rows
        .iter()
        .map(|(_, _, _, c, _)| c.len())
        .max()
        .unwrap_or(0)
        .max(calls_header.len());

    let per_w = fmt_rows
        .iter()
        .map(|(_, _, _, _, p)| p.len())
        .max()
        .unwrap_or(0)
        .max(per_header.len());

    let mut out = String::new();

    let _ = writeln!(
        out,
        "{:<key_w$} | {:>busy_w$} | {:>elapsed_w$} | {:>calls_w$} | {:>per_w$} |",
        key_header,
        busy_header,
        elapsed_header,
        calls_header,
        per_header,
        key_w = key_w,
        busy_w = busy_w,
        elapsed_w = elapsed_w,
        calls_w = calls_w,
        per_w = per_w
    );

    let total_w = key_w + 3 + busy_w + 3 + elapsed_w + 3 + calls_w + 3 + per_w + 3;
    let _ = writeln!(out, "{}", "-".repeat(total_w));

    for (k, busy_s, elapsed_s, calls_s, per_s) in fmt_rows {
        let key_disp = if k.len() > key_w {
            let keep = key_w.saturating_sub(1);
            format!("{}…", &k[..keep])
        } else {
            k
        };

        let _ = writeln!(
            out,
            "{:<key_w$} | {:>busy_w$} | {:>elapsed_w$} | {:>calls_w$} | {:>per_w$} |",
            key_disp,
            busy_s,
            elapsed_s,
            calls_s,
            per_s,
            key_w = key_w,
            busy_w = busy_w,
            elapsed_w = elapsed_w,
            calls_w = calls_w,
            per_w = per_w
        );
    }

    out
}
