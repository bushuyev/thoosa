
use rand::{rngs::StdRng, Rng, SeedableRng};

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
