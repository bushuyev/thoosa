
use rand::{rngs::StdRng, Rng, SeedableRng};
use tch::Tensor;
use thoosa_shared::{MotifKind, ThoError};
use crate::generators::non_overlapping::Interval;

fn z_norm(x: &mut [f32]) {
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

fn smooth_noise(len: usize, rng: &mut impl Rng) -> Vec<f32> {
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

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt()
}

fn avg_pairwise_distance(xs: &[Vec<f32>]) -> f32 {
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
fn generate_motifs_with_amp(base: &[f32], n: usize, amp: f32, seed: Option<u64>) -> Vec<Vec<f32>> {
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
) -> (Vec<Vec<f32>>, f32 /*avg_d*/, f32 /*amp_used*/) {
    assert!(n >= 2 && d0 > 0.0 && d1 >= d0);

    // Bisection over distortion amplitude.
    let target = 0.5 * (d0 + d1);
    let mut lo = 0.0f32;
    let mut hi = 10.0f32; // generous upper bound; bump if needed

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

// --- Example usage ---
// let base = (0..200).map(|i| (2.0 * std::f32::consts::PI * i as f32 / 50.0).sin()).collect::<Vec<_>>();
// let (motifs, avg_d, amp) = generate_motifs_with_avg_distance(&base, 6, 15.0, 17.0, 42);
// println!("avg_d={avg_d:.3}, amp={amp:.3}, generated {}", motifs.len());

///channels_motif_kinds -  channels x [(how many times motif should appear in the sequence, motif length , MotifKind)]
///returns Tensor [channels x resource_len] + channels[ motifs[ intervals[Interval] ] ]
pub fn resource_tensor(resource_len:usize, channels_motif_kinds:&[Vec<(usize, usize, MotifKind)>], gap: usize) -> Result<(Tensor, Vec<Vec<Vec<Interval>>>), ThoError> {
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use rand::seq::SliceRandom;
    use log::debug;
    use thoosa_shared::gen_motif;
    const STEP_SIGMA: f32 = 0.4;

    fn normal_step(rng: &mut StdRng, sigma: f32) -> f32 {
        let u1: f32 = rng.random::<f32>().max(f32::MIN_POSITIVE);
        let u2: f32 = rng.random::<f32>();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        sigma * r * theta.cos()
    }

    fn distribute_slack(rng: &mut StdRng, segments: usize, slack: usize) -> Vec<usize> {
        if segments == 0 {
            return Vec::new();
        }

        if slack == 0 {
            return vec![0; segments];
        }

        let mut offsets = vec![0usize; segments];
        let mut weights: Vec<f32> = (0..segments).map(|_| rng.random::<f32>().max(f32::MIN_POSITIVE)).collect();
        let weight_sum: f32 = weights.iter().sum();

        if !weight_sum.is_finite() || weight_sum <= f32::MIN_POSITIVE {
            offsets[0] = slack;
            return offsets;
        }

        let mut allocated = 0usize;
        for (offset, weight) in offsets.iter_mut().zip(weights.iter()) {
            let share = ((*weight / weight_sum) * slack as f32).floor() as usize;
            *offset = share;
            allocated = allocated.saturating_add(share);
        }

        let allocated = allocated.min(slack);
        let remainder = slack.saturating_sub(allocated);

        if remainder > 0 {
            let mut indices: Vec<usize> = (0..segments).collect();
            indices.shuffle(rng);
            for idx in indices.into_iter().cycle().take(remainder) {
                offsets[idx] += 1;
            }
        }

        offsets
    }

    let mut channels_signals = Vec::with_capacity(channels_motif_kinds.len());
    let mut channels_intervals = Vec::with_capacity(channels_motif_kinds.len());

    let channels_count = channels_motif_kinds.len().max(1);

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

        let extra_space = resource_len - required_space;
        let channel_start = resource_len * channel_idx / channels_count;
        let channel_end = resource_len * (channel_idx + 1) / channels_count;
        let segment_len = channel_end.saturating_sub(channel_start);
        let use_segment = segment_len >= required_space;
        let slack = if use_segment {
            segment_len - required_space
        } else {
            extra_space
        };
        let start_bound = if use_segment { channel_start } else { 0 };
        let end_bound = if use_segment { channel_end } else { resource_len };

        let segment_offsets = distribute_slack(&mut rng, total_repeats + 1, slack);
        let mut current = start_bound
            .checked_add(segment_offsets.get(0).copied().unwrap_or(0))
            .ok_or_else(|| ThoError::new(format!(
                "motif placement exceeds resource length for channel {channel_idx}"
            )))?;
        let mut channel_motif_intervals = Vec::with_capacity(channel_kinds.len());
        let mut placed = 0usize;

        for (repeats, motif_len, kind) in channel_kinds.iter() {
            if *repeats == 0 || *motif_len == 0 {
                channel_motif_intervals.push(Vec::new());
                continue;
            }

            let motifs = generate_motifs_with_avg_distance(
                gen_motif(*motif_len, kind).as_slice(),
                (*repeats).max(2),
                0.1,
                0.2,//should be lower than in motif where it's 0.2-0.3, TODO move to params
                None,
            ).0;

            let mut motif_intervals = Vec::with_capacity(*repeats);

            for motif in motifs.into_iter().take(*repeats) {
                let start = current;
                let end = start.checked_add(*motif_len).ok_or_else(|| ThoError::new(format!(
                    "motif placement exceeds resource length for channel {channel_idx}"
                )))?;
                if end > end_bound {
                    return Err(ThoError::new(format!(
                        "motif placement exceeds resource length for channel {channel_idx}"
                    )));
                }

                for (offset, value) in motif.iter().enumerate() {
                    signal[start + offset] = *value;
                }

                motif_intervals.push(Interval { start, end });
                placed += 1;
                current = end;
                if placed < total_repeats {
                    let extra_gap = segment_offsets.get(placed).copied().unwrap_or(0);
                    let spacing = gap.checked_add(extra_gap).ok_or_else(|| ThoError::new("motif gap sizing overflows usize"))?;
                    current = current.checked_add(spacing).ok_or_else(|| ThoError::new(format!(
                        "motif placement exceeds resource length for channel {channel_idx}"
                    )))?;
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
