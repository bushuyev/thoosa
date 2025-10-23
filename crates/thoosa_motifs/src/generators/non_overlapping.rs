use std::ops::Deref;
use rand::{rngs::StdRng, Rng, SeedableRng};
use thoosa_shared::{gen_motif, pulse_v, MotifKind};

#[derive(Debug, Clone, Copy)]
pub struct Interval { pub start: usize, pub end: usize }

impl From<Interval> for (f32, f32) {
    fn from(value: Interval) -> Self {
        (value.start as f32, value.end as f32)
    }
}

#[inline]
fn normal_step(rng: &mut impl Rng, sigma: f32) -> f32 {
    // Box–Muller, single sample ~ N(0, sigma^2)
    //https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
    let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
    let u2: f32 = rng.random::<f32>();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f32::consts::TAU * u2;
    sigma * r * theta.cos()
}


pub fn generate_series_with_motif(
    len: usize,
    motif_len: usize,
    repeats: usize,
    min_gap: usize,
    step_sigma: f32,
    amp_range: (f32, f32),
    seed: Option<u64>,
    motif_kind: &MotifKind, // "triangle" or "sine"
) -> (Vec<f32>, Vec<Interval>) {
    let mut rng = seed.map(|seed|StdRng::seed_from_u64(seed)).unwrap_or(StdRng::from_os_rng());

    // random-walk background using our own Normal(0, step_sigma)
    let mut sig = vec![0f32; len];
    for i in 1..len {
        sig[i] = sig[i - 1] + normal_step(&mut rng, step_sigma);
    }

    // motif template (z-normalized)
    let mut motif: Vec<f32> = gen_motif(motif_len, motif_kind);

    let m = motif.iter().sum::<f32>() / motif.len() as f32;
    let s = (motif.iter().map(|v| (v - m).powi(2)).sum::<f32>() / motif.len() as f32).sqrt().max(1e-6);
    for v in &mut motif { *v = (*v - m) / s; }

    // place non-overlapping intervals
    let mut intervals = Vec::<Interval>::new();
    'outer: for _ in 0..repeats {
        for _ in 0..20_000 {
            let start = rng.random_range(0..=len.saturating_sub(motif_len));
            if intervals.iter().any(|iv| !(start + motif_len + min_gap <= iv.start || start >= iv.end + min_gap)) {
                continue;
            }
            let amp = rng.random_range(amp_range.0..amp_range.1);
            for (k, &mv) in motif.iter().enumerate() {
                sig[start + k] = sig[start] + amp * mv;
            }
            intervals.push(Interval { start, end: start + motif_len });
            continue 'outer;
        }
        break; // couldn’t place more with constraints
    }

    intervals.sort_by_key(|iv| iv.start);
    (sig, intervals)
}


#[cfg(test)]
mod demo {
    use std::path::Path;
    use log::debug;
    use plotters::prelude::BLACK;
    use thoosa_common::utils::{draw_function_plot, plot_series, plot_series_with_lanes};
    use thoosa_shared::{cdt, cdt_for_dir};
    use super::*;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn gen_demo() {
        let (sig, ivals) = generate_series_with_motif(
            20_000, 400, 4, 300, 0.4, (0.8, 1.8), None, &MotifKind::Pulse,
        );
        debug!("len={} intervals={:?}", sig.len(), ivals);

        let test_dir = Path::new("/tmp/random_sigs");//.join(cdt_for_dir());
        // plot_series_with_lanes(
        //     vec![
        //         (format!("sig"), BLACK, sig.into_iter().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>())
        //     ],
        //     format!("sig").as_str(),
        //     (2000, 1000),
        //     &test_dir.join(&format!("sig_{}.png", cdt_for_dir())),
        //     &[(0, 1000)]
        // ).unwrap();

        draw_function_plot(
            &test_dir.join(&format!("sig_{}.png", cdt_for_dir())),
            (2000, 1000),
            &sig.into_iter().enumerate().map(|(i, v)| (i as f32, v as f32)).collect::<Vec<(f32, f32)>>(),
            &ivals.into_iter().map(|v|v.into()).collect::<Vec<(f32, f32)>>(),
            "sig"
        ).unwrap();

        println!("saved to {test_dir:?}");
    }
}