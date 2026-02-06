use crate::LOG_INIT;
use crate::perf_logging::{
    init_log_tracer, perf_log_level, perf_output_filter, set_perf_log_level, PerfLayer,
    SimplelogLikeEvent,
};
use log::{debug, trace, LevelFilter};
use num_traits::{Num, Signed, ToPrimitive};
use std::env;
use std::env::VarError;
use std::fmt::Debug;
use std::str::FromStr;
use std::sync::atomic::Ordering;
use tracing::Level;
use tracing_subscriber::filter::{DynFilterFn, FilterFn};
use tracing_subscriber::{
    fmt,
    layer::SubscriberExt,
    Layer,
};
use time_series_generator::{generate_sine_wave, generate_triangle_wave};
use rand::Rng;

fn level_filter_allows(filter: LevelFilter, level: &Level) -> bool {
    if filter == LevelFilter::Off {
        return false;
    }
    let filter = match filter {
        LevelFilter::Error => 1,
        LevelFilter::Warn => 2,
        LevelFilter::Info => 3,
        LevelFilter::Debug => 4,
        LevelFilter::Trace => 5,
        LevelFilter::Off => 0,
    };
    let level = match *level {
        Level::ERROR => 1,
        Level::WARN => 2,
        Level::INFO => 3,
        Level::DEBUG => 4,
        Level::TRACE => 5,
    };
    level <= filter
}

//TODO per mod filtering like in EnvFilter::new("warn,my_crate::some_mod=trace")
pub fn init_test_logging() {

    if LOG_INIT.swap(true, Ordering::Relaxed) {
        return;
    }
    let log_level = env::var("RUST_LOG")
        .and_then(|v| {
            LevelFilter::from_str(v.as_str()).map_err(|e| {
                println!("Can't find loglevel: {}", e);
                VarError::NotPresent
            })
        })
        .unwrap_or(LevelFilter::Debug);

    if perf_log_level() == LevelFilter::Off {
        if let Ok(val) = env::var("RUST_LOG_PERF") {
            let _ = set_perf_log_level(&val);
        }
    }

    let timer = tracing_subscriber::fmt::time::ChronoLocal::new("%Y-%m-%d %H:%M:%S".to_string());
    let include_log_fields = false;

    let stdout_layer = fmt::layer()
        .with_ansi(true)
        .with_writer(std::io::stdout)
        .event_format(SimplelogLikeEvent::new(timer.clone(), true, include_log_fields))
        .with_filter(DynFilterFn::new(|meta, _| {
            let is_log_record = meta.fields().field("log.target").is_some();
            if *meta.level() == Level::TRACE && !is_log_record {
                return false;
            }
            level_filter_allows(log::max_level(), meta.level())
        }));

    let perf_layer = fmt::layer()
        .with_ansi(true)
        .with_writer(std::io::stdout)
        .with_span_events(fmt::format::FmtSpan::CLOSE)
        .event_format(SimplelogLikeEvent::new(timer, true, include_log_fields))
        .with_filter(FilterFn::new(perf_output_filter));

    let subscriber = tracing_subscriber::registry()
        .with(stdout_layer)
        .with(perf_layer)
        .with(PerfLayer::new());

    let _ = tracing::subscriber::set_global_default(subscriber);

    init_log_tracer(log_level);

    unsafe {
        env::set_var("POLARS_FMT_MAX_COLS", "100");
        env::set_var("POLARS_FMT_MAX_ROWS", "100");
    }
}


/// Initialize a perf-only tracing subscriber that writes to **stdout**.
/// Intended for tests (so you can `cargo test -- --nocapture`).
///
/// Returns `true` if it successfully installed the subscriber, `false` if a
/// subscriber was already installed (common in test harnesses).
pub fn maybe_init_perf_tracing_stdout(perf_filter: &str) -> bool {
    if perf_filter.trim() == "off" {
        let _ = set_perf_log_level(perf_filter);
        return false;
    }

    if set_perf_log_level(perf_filter).is_err() {
        return false;
    }

    if LOG_INIT.load(Ordering::Relaxed) {
        return false;
    }

    init_test_logging();
    true
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

pub enum MotifKind {
    Triangle,
    Sin,
    Pulse
}

pub fn gen_motif(motif_len: usize, motif_kind: &MotifKind) -> Vec<f32> {
    match motif_kind {
        MotifKind::Triangle => generate_triangle_wave(motif_len).into_iter().map(|v| v as f32).collect(),
        MotifKind::Sin => generate_sine_wave::<f32>(motif_len),
        MotifKind::Pulse => crate::pulse_v(motif_len as f32 / 2., motif_len as f32 / 4., motif_len as i64, 10.)
    }
}

pub fn random_v(n: u32, amplitude: f32, max_slope: f32) -> Vec<f32> {
    let mut signal = Vec::with_capacity(n as usize);

    let mut current_value = 0.0;
    let mut prev_value = 0.0;

    let rnd_vals_0 = (0..n)
        .flat_map(move |_i| {
            std::iter::repeat(rand::rng().random_range(-max_slope..max_slope)).take(1)
        })
        .collect::<Vec<f32>>();
    let rnd_vals_1 = (0..n / 10)
        .flat_map(move |_i| {
            std::iter::repeat(rand::rng().random_range(-max_slope..max_slope)).take(10)
        })
        .collect::<Vec<f32>>();
    let rnd_vals_2 = (0..n / 60)
        .flat_map(move |_i| {
            std::iter::repeat(rand::rng().random_range(-max_slope..max_slope)).take(100)
        })
        .collect::<Vec<f32>>();
    let rnd_vals_3 = (0..n / 120)
        .flat_map(move |_i| {
            std::iter::repeat(rand::rng().random_range(-max_slope..max_slope)).take(300)
        })
        .collect::<Vec<f32>>();

    rnd_vals_0
        .into_iter()
        .zip(
            rnd_vals_1
                .into_iter()
                .zip(rnd_vals_2.into_iter().zip(rnd_vals_3.into_iter())),
        )
        .for_each(|(r_0, (r_1, (r_2, r_3)))| {
            // debug!("random_v: r_0={:?}, r_1={:?}, r_2={:?}, r_3={:?}", r_0, r_1, r_2, r_3);

            current_value = prev_value + 10. * r_0 + r_1 + r_2 + r_3;
            signal.push(current_value * amplitude);
            prev_value = current_value;
        });

    signal
}