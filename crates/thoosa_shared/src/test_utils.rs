use crate::LOG_INIT;
use log::{LevelFilter, debug, trace};
use num_traits::{Num, Signed, ToPrimitive};
use simplelog::{
    ColorChoice, CombinedLogger, ConfigBuilder, TermLogger, TerminalMode, ThreadLogMode,
    ThreadPadding, format_description,
};
use std::env;
use std::env::VarError;
use std::fmt::Debug;
use std::str::FromStr;
use std::sync::atomic::Ordering;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{
    fmt,
    layer::SubscriberExt,
    EnvFilter, Layer,
};
use crate::perf_logging::{PerfLayer, SimplelogLikeEvent};

pub fn init_test_logging() {
    if !LOG_INIT.load(Ordering::Relaxed) {
        LOG_INIT.store(true, Ordering::Relaxed);
        //TODO CombinedLogger not needed?
        CombinedLogger::init(vec![TermLogger::new(
            LevelFilter::Trace, //to let set trace level from tests with set_max_level
            ConfigBuilder::new()
                // Optional: pick the exact date/time layout you want
                .set_time_format_custom(format_description!(
                    "[year]-[month]-[day] [hour]:[minute]:[second]"
                ))
                // Print the *name* (or the ID if unnamed); use `Both` to fall back automatically
                .set_thread_mode(ThreadLogMode::Names)
                // Show it for every record (Error is the highest-priority level, so this means “always”)
                .set_thread_level(LevelFilter::Error)
                // Optional: pad so columns line up nicely
                .set_thread_padding(ThreadPadding::Right(12))
                .build(),
            TerminalMode::Mixed,
            ColorChoice::Auto,
        )])
        .unwrap_or_else(|e| {
            debug!("CombinedLogger::init failed: {:?}", e);
        });

        log::set_max_level(
            env::var("RUST_LOG")
                .and_then(|v| {
                    LevelFilter::from_str(v.as_str()).map_err(|e| {
                        println!("Can't find loglevel: {}", e);
                        VarError::NotPresent
                    })
                })
                .unwrap_or(LevelFilter::Debug),
        );
        unsafe {
            env::set_var("POLARS_FMT_MAX_COLS", "100");
            env::set_var("POLARS_FMT_MAX_ROWS", "100");
        }
    }
}


/// Initialize a perf-only tracing subscriber that writes to **stdout**.
/// Intended for tests (so you can `cargo test -- --nocapture`).
///
/// Returns `true` if it successfully installed the subscriber, `false` if a
/// subscriber was already installed (common in test harnesses).
pub fn maybe_init_perf_tracing_stdout(perf_filter: &str) -> bool {
    if perf_filter.trim() == "off" {
        return false;
    }

    let timer = tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S".to_string());

    let layer = fmt::layer()
        .with_ansi(true)
        .with_writer(std::io::stdout)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .event_format(SimplelogLikeEvent::new(timer))
        .with_filter(EnvFilter::new(perf_filter));

    tracing_subscriber::registry()
        .with(layer)
        .with(PerfLayer::new())
        .try_init()
        .is_ok()
}


pub fn set_panic_hook() {
    let default_hook = std::panic::take_hook();

    std::panic::set_hook(Box::new(move |info| {
        debug!("Panic info: {:?}", info);

        if let Some(s) = info.payload().downcast_ref::<&str>() {
            log::error!("panic payload: {s}");
        } else if let Some(s) = info.payload().downcast_ref::<String>() {
            log::error!("panic payload: {s}");
        }


        let backtrace = std::backtrace::Backtrace::force_capture();
        for frame in backtrace.frames() {
            debug!("{:?}\n", frame);
        }
        log::logger().flush(); // make sure async loggers flush

        default_hook(info);
    }));
}

pub fn are_close<T>(a: &[T], b: &[T], p: T) -> bool
where
    T: Num + Signed + PartialOrd + Copy + ToPrimitive + Debug,
{
    assert_eq!(a.len(), b.len());

    !a.iter().zip(b.iter()).any(|(&x, &y)| {
        if x == y {
            trace!("0. x={x:?} y={y:?}");

            false
        } else if let (Some(xf), Some(yf), Some(pf)) = (
            ToPrimitive::to_f64(&x),
            ToPrimitive::to_f64(&y),
            ToPrimitive::to_f64(&p),
        ) {
            let diff = (xf - yf).abs() / ((xf.abs() + yf.abs()) / 2.);
            trace!("1. x={x:?} y={y:?} diff={diff:?}");
            diff > pf
        } else {
            let diff = (x - y).abs();
            trace!("1. x={x:?} y={y:?} diff={diff:?}");

            diff > p
        }
    })
}

#[cfg(test)]
mod tests {
    use crate::test_utils::are_close;

    #[test]
    fn test_are_close() {
        assert!(are_close(&vec![1, 2, 3, 4, 5], &vec![1, 2, 3, 4, 5], 0));

        assert!(are_close(&vec![1, 2, 3, 4, 5], &vec![2, 3, 4, 5, 6], 1));

        assert!(are_close(
            &vec![1., 2., 3., 4., 5.],
            &vec![2., 3., 4., 5., 6.],
            0.7
        ));

        assert!(!are_close(
            &vec![1., 2., 3., 4., 5.],
            &vec![2., 3., 4., 5., 6.],
            0.1
        ));

        assert!(!are_close(&vec![1, 2, 3, 4, 5], &vec![4, 4, 4, 4, 4], 1));
    }
}
