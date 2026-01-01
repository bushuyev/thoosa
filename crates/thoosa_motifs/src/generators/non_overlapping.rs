use std::ops::Deref;
use log::debug;
use rand::{rngs::StdRng, Rng, SeedableRng};
use thoosa_shared::{gen_motif, pulse_v, MotifKind, ThoError};
use crate::generators::noise::generate_motifs_with_avg_distance;

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
) -> Result<(Vec<f32>, Vec<Interval>), ThoError> {
    let mut rng = seed.map(|seed|StdRng::seed_from_u64(seed)).unwrap_or(StdRng::from_os_rng());

    // random-walk background using our own Normal(0, step_sigma)
    let mut sig = vec![0f32; len];
    for i in 1..len {
        sig[i] = sig[i - 1] + normal_step(&mut rng, step_sigma);
    }

    let mut motif_0 =  generate_motifs_with_avg_distance(
        gen_motif(motif_len, &motif_kind).as_slice(), repeats, 0.1, 0.2, None
    );
    // motif template (z-normalized)
    let mut motif: Vec<f32> = gen_motif(motif_len, motif_kind);

    let m = motif.iter().sum::<f32>() / motif.len() as f32;
    let s = (motif.iter().map(|v| (v - m).powi(2)).sum::<f32>() / motif.len() as f32).sqrt().max(1e-6);
    for v in &mut motif { *v = (*v - m) / s; }

    // place non-overlapping intervals
    if repeats == 0 || motif_len == 0 {
        return Ok((sig, Vec::new()));
    }

    let total_motif_space = motif_len.checked_mul(repeats).ok_or_else(|| {
        ThoError::new("motif placement overflows usize")
    })?;
    let gaps_between = repeats.saturating_sub(1);
    let total_gap_space = min_gap.checked_mul(gaps_between).ok_or_else(|| {
        ThoError::new("motif gap sizing overflows usize")
    })?;
    let required_space = total_motif_space.checked_add(total_gap_space).ok_or_else(|| {
        ThoError::new("motif layout exceeds addressable range")
    })?;

    if required_space > len {
        return Err(ThoError::new(format!(
            "cannot place {repeats} motifs (len={motif_len}) with min_gap={min_gap} into series len={len}"
        )));
    }

    let extra_space = len - required_space;
    let leading_gap = extra_space / 2;

    let mut intervals = Vec::with_capacity(repeats);
    let mut start = leading_gap;
    for motif_idx in 0..repeats {
        let base = sig[start];
        for (k, &mv) in motif_0.0[motif_idx].iter().enumerate() {
            sig[start + k] = /*base +*/ mv;
        }
        intervals.push(Interval { start, end: start + motif_len });
        if motif_idx + 1 < repeats {
            start += motif_len + min_gap;
        }
    }

    Ok((sig, intervals))
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
    fn generate_series_with_motif_places_everything_when_space_allows() {
        let len = 200;
        let motif_len = 10;
        let repeats = 3;
        let min_gap = 5;
        let (_, intervals) = generate_series_with_motif(
            len,
            motif_len,
            repeats,
            min_gap,
            0.4,
            (0.8, 1.8),
            Some(42),
            &MotifKind::Triangle,
        ).expect("motifs should be placed");

        assert_eq!(intervals.len(), repeats);
        let starts: Vec<_> = intervals.into_iter().map(|iv| iv.start).collect();
        assert_eq!(starts, vec![80, 95, 110]);
    }

    #[test]
    fn generate_series_with_motif_errors_when_series_too_short() {
        let result = generate_series_with_motif(
            20,
            10,
            2,
            5,
            0.4,
            (0.8, 1.8),
            None,
            &MotifKind::Triangle,
        );

        assert!(result.is_err());
    }
}
