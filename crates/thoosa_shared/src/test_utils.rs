use crate::LOG_INIT;
use crate::perf_logging::{
    init_log_tracer, perf_log_level, perf_output_filter, set_perf_log_level, PerfLayer,
    SimplelogLikeEvent,
};
use log::{trace, LevelFilter};
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
use rand::RngExt;
use thoosa_shared::perf_logging::set_panic_hook;

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
    set_panic_hook();

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

#[derive(Clone, Debug)]
pub enum MotifKind {
    Triangle,
    Sin,
    Pulse,

    /// Stable harmonic stack: f0 + 2f0 + 3f0 ...
    HarmonicStack,

    /// Harmonic stack with amplitude envelope, closer to a bird syllable.
    HarmonicSyllable,

    /// Frequency rises during the motif.
    ChirpUp,

    /// Frequency falls during the motif.
    ChirpDown,

    /// Several short repeated harmonic syllables.
    Trill,
}

pub fn gen_motif(motif_len: usize, motif_kind: &MotifKind) -> Vec<f32> {
    match motif_kind {
        MotifKind::Triangle => generate_triangle_wave(motif_len)
            .into_iter()
            .map(|v| v as f32)
            .collect(),

        MotifKind::Sin => generate_sine_wave::<f32>(motif_len),

        MotifKind::Pulse => {
            crate::pulse_v(motif_len as f32 / 2., motif_len as f32 / 4., motif_len as i64, 10.)
        }

        MotifKind::HarmonicStack => {
            harmonic_stack(motif_len, 8.0, &[1.0, 0.55, 0.30, 0.18])
        }

        MotifKind::HarmonicSyllable => {
            let mut x = harmonic_stack(motif_len, 9.0, &[1.0, 0.60, 0.35, 0.20]);
            apply_syllable_envelope(&mut x);
            normalize_peak(&mut x, 1.0);
            x
        }

        MotifKind::ChirpUp => {
            let mut x = chirp(motif_len, 5.0, 16.0);
            apply_syllable_envelope(&mut x);
            normalize_peak(&mut x, 1.0);
            x
        }

        MotifKind::ChirpDown => {
            let mut x = chirp(motif_len, 16.0, 5.0);
            apply_syllable_envelope(&mut x);
            normalize_peak(&mut x, 1.0);
            x
        }

        MotifKind::Trill => {
            let mut x = trill(motif_len, 5, 10.0);
            normalize_peak(&mut x, 1.0);
            x
        }
    }
}

fn harmonic_stack(length: usize, f0_cycles: f32, harmonic_amps: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; length];

    for i in 0..length {
        let t = i as f32 / length as f32;

        let mut v = 0.0;
        for (h_idx, amp) in harmonic_amps.iter().enumerate() {
            let h = (h_idx + 1) as f32;
            let phase = std::f32::consts::TAU * f0_cycles * h * t;
            v += amp * phase.sin();
        }

        out[i] = v;
    }

    normalize_peak(&mut out, 1.0);
    out
}

fn chirp(length: usize, f_start_cycles: f32, f_end_cycles: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; length];

    for i in 0..length {
        let t = i as f32 / length as f32;

        // Linear frequency sweep.
        // Phase is integral of frequency:
        // phase(t) = 2π * length-normalized-cycles * (f0*t + 0.5*(f1-f0)*t²)
        let phase = std::f32::consts::TAU
            * (f_start_cycles * t + 0.5 * (f_end_cycles - f_start_cycles) * t * t);

        out[i] = phase.sin();
    }

    normalize_peak(&mut out, 1.0);
    out
}

fn trill(length: usize, syllables: usize, f0_cycles_per_syllable: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; length];

    if syllables == 0 || length == 0 {
        return out;
    }

    let syll_len = length / syllables;
    if syll_len == 0 {
        return out;
    }

    for s in 0..syllables {
        let from = s * syll_len;
        let to = if s + 1 == syllables {
            length
        } else {
            ((s + 1) * syll_len).min(length)
        };

        let mut syll = harmonic_stack(to - from, f0_cycles_per_syllable, &[1.0, 0.5, 0.25]);
        apply_syllable_envelope(&mut syll);

        for (j, v) in syll.into_iter().enumerate() {
            out[from + j] = v;
        }
    }

    normalize_peak(&mut out, 1.0);
    out
}

fn apply_syllable_envelope(x: &mut [f32]) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    for (i, v) in x.iter_mut().enumerate() {
        let t = i as f32 / (n - 1) as f32;

        // Hann envelope: smooth attack and decay.
        let env = 0.5 - 0.5 * (std::f32::consts::TAU * t).cos();

        *v *= env;
    }
}

fn normalize_peak(x: &mut [f32], peak: f32) {
    let max_abs = x
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);

    if max_abs > 1e-12 {
        let scale = peak / max_abs;
        for v in x {
            *v *= scale;
        }
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
