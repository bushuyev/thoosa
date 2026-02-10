use tch::Tensor;
use rand::{Rng, SeedableRng};
use rand::prelude::StdRng;
use thoosa_shared::test_utils::{gen_motif, MotifKind};
use thoosa_shared::ThoError;
use crate::utils::to_vec_f32;

pub fn deterministic_noise(len: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut out = Vec::with_capacity(len);
    let mut state = seed;
    for _ in 0..len {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        let u = (state >> 8) as f32 / ((u32::MAX >> 8) as f32);
        out.push((u * 2.0 - 1.0) * amp);
    }
    out
}

pub fn resource_tensor_from_patterns(resource_len: usize, channels_patterns: &[Vec<Tensor>], gap: usize, step: usize) -> Result<(Tensor, Vec<Vec<Vec<Interval>>>), ThoError> {
    let step = step.max(1);
    let mut channels_signals = Vec::with_capacity(channels_patterns.len());
    let mut channels_intervals = Vec::with_capacity(channels_patterns.len());
    for (channel_idx, channel_patterns) in channels_patterns.iter().enumerate() {
        let seed = 0x9E37_79B9u32 ^ (channel_idx as u32).wrapping_mul(0x85EB_CA6B);
        let mut signal = deterministic_noise(resource_len, seed, 0.2);
        if channel_patterns.is_empty() {
            channels_signals.push(signal);
            channels_intervals.push(Vec::new());
            continue;
        }
        let start = gap.saturating_sub(gap % step);
        let mut current = start;
        let mut motif_intervals = Vec::with_capacity(channel_patterns.len());
        for pattern in channel_patterns.iter() {
            assert_eq!(pattern.size().len(), 1, "pattern must be 1-D");
            let motif_len = pattern.size()[0] as usize;
            if motif_len == 0 {
                motif_intervals.push(Vec::new());
                continue;
            }
            let end = current.checked_add(motif_len).ok_or_else(|| ThoError::new("motif placement exceeds resource length"))?;
            if end > resource_len {
                return Err(ThoError::new("motif placement exceeds resource length"));
            }
            let values = to_vec_f32(pattern);
            for (offset, value) in values.iter().enumerate() {
                signal[current + offset] = *value;
            }
            motif_intervals.push(vec![Interval { start: current, end }]);
            current = end + gap;
        }
        channels_signals.push(signal);
        channels_intervals.push(motif_intervals);
    }
    let t = Tensor::f_stack(
        channels_signals
            .into_iter()
            .map(|data| Tensor::f_from_slice(data.as_slice()))
            .collect::<Result<Vec<Tensor>, _>>()?
            .as_slice(),
        0,
    )?;
    Ok((t, channels_intervals))
}

pub fn motifs_tensor_fixed(motif_len: usize, num_slices: usize, motif_kind: MotifKind) -> Tensor {
    let mut motif = vec![0.0f32; motif_len];
    let width = motif_len; //(motif_len / 10).max(1);
    let (start, end) = match motif_kind {
        MotifKind::Pulse => (0, width),
        MotifKind::Triangle => (motif_len / 3, (motif_len / 3 + width).min(motif_len)),
        MotifKind::Sin => (2 * motif_len / 3, (2 * motif_len / 3 + width).min(motif_len)),
    };
    for (v, motif_v) in motif.iter_mut().take(end).skip(start).zip(gen_motif(end-start, &motif_kind)) {
        *v = motif_v;
    }
    let slices = (0..num_slices)
        .map(|_| Tensor::from_slice(motif.as_slice()))
        .collect::<Vec<_>>();
    Tensor::stack(slices.as_slice(), 0)
}

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
        gen_motif(motif_len, &motif_kind).as_slice(), repeats, 0.1, 0.2, None, 10.0f32
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
mod tests {
    use thoosa_shared::test_utils::MotifKind;
    use crate::tch_test_utils::generate_series_with_motif;

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

pub fn z_norm(x: &mut [f32]) {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let mut var = 0.0;
    for v in x.iter() {
        var += (v - mean) * (v - mean);
    }
    let std = (var / n).sqrt().max(1e-6);
    for v in x.iter_mut() {
        *v = (*v - mean) / std;
    }
}

pub fn smooth_noise(len: usize, rng: &mut impl Rng) -> Vec<f32> {
    // white noise
    let mut n = (0..len).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect::<Vec<_>>();
    // simple 5-tap smoothing kernel [1,2,3,2,1] / 9
    let k = [1.0, 2.0, 3.0, 2.0, 1.0];
    let ks = 9.0;
    let mut s = vec![0.0; len];
    for i in 0..len {
        let mut acc = 0.0;
        for (t, w) in (-2i32..=2).zip(k) {
            let j = i as i32 + t;
            if j >= 0 && (j as usize) < len {
                acc += n[j as usize] * w;
            }
        }
        s[i] = acc / ks;
    }
    s
}

pub fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

pub fn avg_pairwise_distance(xs: &[Vec<f32>]) -> f32 {
    let n = xs.len();
    if n < 2 { return 0.0; }
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            sum += l2(&xs[i], &xs[j]);
            cnt += 1;
        }
    }
    sum / (cnt as f32)
}

/// Generate N z-normalized motifs by adding smoothed noise scaled by `amp`.
pub fn generate_motifs_with_amp(base: &[f32], n: usize, amp: f32, seed: Option<u64>) -> Vec<Vec<f32>> {
    let mut rng = seed.map(|seed|StdRng::seed_from_u64(seed)).unwrap_or(StdRng::from_os_rng());
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let mut y = base.to_vec();
        let dn = smooth_noise(y.len(), &mut rng);
        for (v, d) in y.iter_mut().zip(dn) {
            *v += amp * d; // shape distortion
        }
        z_norm(&mut y); // compare shapes fairly
        out.push(y);
    }
    out
}

/// Find an amplitude whose average pairwise distance hits target within [d0, d1].
pub fn generate_motifs_with_avg_distance(
    base: &[f32],
    n: usize,
    d0: f32,
    d1: f32,
    seed: Option<u64>,
    hight: f32,
) -> (Vec<Vec<f32>>, f32 /*avg_d*/, f32 /*amp_used*/) {
    assert!(n >= 2 && d0 > 0.0 && d1 >= d0);

    // Bisection over distortion amplitude.
    let target = 0.5 * (d0 + d1);
    let mut lo = 0.0f32;
    let mut hi = hight; // generous upper bound; bump if needed

    // Ensure hi is high enough.
    for _ in 0..12 {
        let motifs = generate_motifs_with_amp(base, n, hi, seed);
        let d = avg_pairwise_distance(&motifs);
        if d >= target { break; }
        hi *= 2.0;
    }

    let mut best = (Vec::new(), 0.0f32, 0.0f32);
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        let motifs = generate_motifs_with_amp(base, n, mid, seed);
        let d = avg_pairwise_distance(&motifs);
        best = (motifs, d, mid);
        if d < d0 {
            lo = mid;
        } else if d > d1 {
            hi = mid;
        } else {
            break; // in range
        }
    }
    best
}

///channels_motif_kinds -  channels x [(how many times motif should appear in the sequence, motif length , MotifKind)]
///returns Tensor [channels x resource_len] + channels[ motifs[ intervals[Interval] ] ]
pub fn resource_tensor(resource_len:usize, channels_motif_kinds:&[Vec<(usize, usize, MotifKind)>], gap: usize, motif_ampl: f32) -> Result<(Tensor, Vec<Vec<Vec<Interval>>>), ThoError> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use log::debug;
    use thoosa_shared::test_utils::gen_motif;
    const STEP_SIGMA: f32 = 0.4;

    fn normal_step(rng: &mut StdRng, sigma: f32) -> f32 {
        let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
        let u2: f32 = rng.random::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        sigma * r * theta.cos()
    }

    let mut channels_signals = Vec::with_capacity(channels_motif_kinds.len());
    let mut channels_intervals = Vec::with_capacity(channels_motif_kinds.len());

    for (channel_idx, channel_kinds) in channels_motif_kinds.iter().enumerate() {
        let total_repeats: usize = channel_kinds.iter().map(|(repeats, _, _)| *repeats).sum();

        let mut rng = StdRng::from_os_rng();
        let mut signal = vec![0f32; resource_len];
        for idx in 1..resource_len {
            signal[idx] = signal[idx - 1] + normal_step(&mut rng, STEP_SIGMA);
        }

        if total_repeats == 0 {
            channels_signals.push(signal);
            channels_intervals.push(channel_kinds.iter().map(|_| Vec::new()).collect());
            continue;
        }

        let total_motif_space = channel_kinds.iter().try_fold(0usize, |acc, (repeats, motif_len, _)| {
            let motif_space = motif_len.checked_mul(*repeats).ok_or_else(|| ThoError::new("motif length overflow"))?;
            acc.checked_add(motif_space).ok_or_else(|| ThoError::new("motif placement overflows usize"))
        })?;
        let total_gap_space = gap.checked_mul(total_repeats.saturating_sub(1)).ok_or_else(|| ThoError::new("motif gap sizing overflows usize"))?;
        let required_space = total_motif_space.checked_add(total_gap_space).ok_or_else(|| ThoError::new("motif layout exceeds addressable range"))?;

        if required_space > resource_len {
            return Err(ThoError::new(format!(
                "cannot place {total_repeats} motifs with min_gap={gap} into series len={resource_len}"
            )));
        }

        let available_motif_space = resource_len.checked_sub(total_gap_space).ok_or_else(|| ThoError::new("motif layout exceeds addressable range"))?;
        let slot_len_base = available_motif_space / total_repeats;
        let slot_len_remainder = available_motif_space % total_repeats;
        let mut slot_index = 0usize;
        let mut slot_cursor = 0usize;
        let mut channel_motif_intervals = Vec::with_capacity(channel_kinds.len());

        for (repeats, motif_len, kind) in channel_kinds.iter() {
            if *repeats == 0 {
                channel_motif_intervals.push(Vec::new());
                continue;
            }

            let motifs = if *motif_len == 0 {
                Vec::new()
            } else {
                generate_motifs_with_avg_distance(
                    gen_motif(*motif_len, kind).as_slice(), //height 10
                    (*repeats).max(2),
                    0.1,
                    0.11, //should be lower than in motif where it's 0.2-0.3, TODO move to params
                    None, motif_ampl,
                ).0
            };
            let mut motif_intervals = Vec::with_capacity(*repeats);

            for repeat_idx in 0..*repeats {
                let slot_len = slot_len_base + usize::from(slot_index < slot_len_remainder);
                if *motif_len > 0 {
                    if *motif_len > slot_len {
                        return Err(ThoError::new(format!(
                            "motif len={motif_len} exceeds slot_len={slot_len} (gap={gap}, total_repeats={total_repeats}, resource_len={resource_len})"
                        )));
                    }
                    let max_offset = slot_len - *motif_len;
                    let slot_offset = if max_offset == 0 { 0 } else { rng.random_range(0..=max_offset) };
                    let start = slot_cursor.checked_add(slot_offset).ok_or_else(|| ThoError::new(format!(
                        "motif placement exceeds resource length for channel {channel_idx}"
                    )))?;
                    let end = start.checked_add(*motif_len).ok_or_else(|| ThoError::new(format!(
                        "motif placement exceeds resource length for channel {channel_idx}"
                    )))?;
                    if end > resource_len {
                        return Err(ThoError::new(format!(
                            "motif placement exceeds resource length for channel {channel_idx}"
                        )));
                    }
                    let motif = motifs.get(repeat_idx).ok_or_else(|| ThoError::new(format!(
                        "missing motif for repeat {repeat_idx} in channel {channel_idx}"
                    )))?;
                    for (offset, value) in motif.iter().enumerate() {
                        signal[start + offset] = *value;//signal[start] +
                    }
                    motif_intervals.push(Interval { start, end });
                }
                slot_index += 1;
                let next_cursor = slot_cursor.checked_add(slot_len).ok_or_else(|| ThoError::new(format!(
                    "motif placement exceeds resource length for channel {channel_idx}"
                )))?;
                if slot_index < total_repeats {
                    slot_cursor = next_cursor.checked_add(gap).ok_or_else(|| ThoError::new(format!(
                        "motif placement exceeds resource length for channel {channel_idx}"
                    )))?;
                } else {
                    slot_cursor = next_cursor;
                }
            }

            channel_motif_intervals.push(motif_intervals);
        }

        // ensure we push placeholders for motifs that had zero repeats but were skipped
        while channel_motif_intervals.len() < channel_kinds.len() {
            channel_motif_intervals.push(Vec::new());
        }

        channels_signals.push(signal);
        channels_intervals.push(channel_motif_intervals);
    }

    let t = Tensor::f_stack(
        channels_signals
            .into_iter()
            .map(|data| Tensor::f_from_slice(data.as_slice()))
            .collect::<Result<Vec<Tensor>, _>>()?
            .as_slice(),
        0,
    )?;

    debug!("resource_tensor: t.size={:?} intervals={channels_intervals:?}", t.size());

    Ok((t, channels_intervals))
}
