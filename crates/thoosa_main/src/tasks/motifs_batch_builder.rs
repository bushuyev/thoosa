use log::debug;
use tch::{Tensor, Kind, Device, Result as TchResult};

/// Controls mixup+jitter behavior when N is small.
#[derive(Clone, Copy, Debug)]
pub struct MixupJitterCfg {
    /// Mixup Beta(alpha,alpha)
    pub alpha: f64,
    /// Gaussian noise std for jitter
    pub sigma: f64,
    /// Apply x <- lam*x + (1-lam)*x[perm]
    pub do_mixup: bool,
    /// Apply x <- x + Normal(0, sigma)
    pub do_jitter: bool,
}

impl Default for MixupJitterCfg {
    fn default() -> Self {
        Self { alpha: 0.4, sigma: 0.01, do_mixup: true, do_jitter: true }
    }
}

/// Convenience wrapper with defaults.
pub fn make_batch(in_tensor: &Tensor, batch_size: i64) -> TchResult<Tensor> {
    make_batch_with(in_tensor, batch_size, MixupJitterCfg::default())
}

/// Main entry. Expects `x` of shape [num_records X seq_len], float dtype.
#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="MixupJitterCfg::make_batch_with"))]
pub fn make_batch_with(in_tensor: &Tensor, batch_size: i64, cfg: MixupJitterCfg) -> TchResult<Tensor> {
    debug!("make_batch_with: batch_size={batch_size} in_tensor.size={:?}", in_tensor.size());
    let (num_records, seq_len) = in_tensor.size2()?; // ensures 2-D and returns (N, M)
    let device = in_tensor.device();
    let kind = in_tensor.kind();
    debug_assert!(matches!(kind, Kind::Float | Kind::BFloat16 | Kind::Half | Kind::Double));

    // Case 1: enough data — sample without replacement.
    if num_records > 2 * batch_size {
        let idx = Tensor::f_randperm(num_records, (Kind::Int64, device))?.f_narrow(0, 0, batch_size)?;
        return Ok(in_tensor.f_index_select(0, &idx)?);
    }

    // Case 2: data paucity — build a batch via sampling with replacement,
    // then mixup (optional) and jitter (optional).
    let base_idx = Tensor::f_randint(num_records, &[batch_size], (Kind::Int64, device))?;
    let mut out = in_tensor.f_index_select(0, &base_idx)?.f_to_kind(Kind::Float)?;

    if cfg.do_mixup {
        let perm_idx = Tensor::f_randint(num_records, &[batch_size], (Kind::Int64, device))?;
        let partner = in_tensor.f_index_select(0, &perm_idx)?.f_to_kind(Kind::Float)?;
        let lam = beta_tensor(cfg.alpha, batch_size, device)?.f_to_kind(out.kind())?; // [B]
        let w = lam.f_unsqueeze(1)?; // [B,1] to broadcast over M
        out = &w * out + (1.0 - &w) * partner;
    }

    if cfg.do_jitter && cfg.sigma > 0.0 {
        let noise = out.f_randn_like()? * cfg.sigma;
        out += noise;
    }

    // Cast back to the original float kind if needed (avoid integer kinds).
    Ok(match kind {
        Kind::Float | Kind::BFloat16 | Kind::Half | Kind::Double => out.f_to_kind(kind)?,
        _ => out, // leave as Float
    })
}

/// Samples Beta(alpha, alpha) for `batch_size` entries and returns a 1-D tensor on `device`.
fn beta_tensor(alpha: f64, batch_size: i64, device: Device) -> TchResult<Tensor> {
    use rand_distr::{Distribution, Gamma};

    assert!(alpha > 0.0, "alpha must be > 0");
    let ga = Gamma::new(alpha, 1.0).expect("valid gamma");
    let mut rng = rand::rng();

    let mut buf = Vec::with_capacity(batch_size as usize);
    for _ in 0..batch_size {
        let u: f64 = ga.sample(&mut rng);
        let v: f64 = ga.sample(&mut rng);
        let lam = if u + v > 0.0 { u / (u + v) } else { 0.5 };
        buf.push(lam as f32);
    }
    Ok(Tensor::f_from_slice(&buf)?.f_to_device(device)?)
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;
    use tch::{Device, IndexOp, Kind, Tensor};
    use tch::kind::FLOAT_CPU;
    use thoosa_motifs::criteria::norm_thickness;

    fn row_matrix(n: i64, m: i64) -> Tensor {
        // Rows encode their index so we can check uniqueness/bounds.
        // For m==1 -> [i], for m>=2 -> [i, 2i, 3i, ...] truncated to m.
        let base = Tensor::arange(n, (Kind::Float, Device::Cpu)).unsqueeze(1); // [N,1]
        if m == 1 {
            base
        } else {
            let mut cols = vec![base.shallow_clone()];
            for k in 2..=m {
                cols.push(base.shallow_clone() * (k as f64));
            }
            Tensor::cat(&cols.iter().collect::<Vec<_>>(), 1)
        }
    }

    #[test]
    fn samples_without_replacement_when_enough_rows() {
        tch::manual_seed(42);
        let n = 30;
        let b = 10;
        let m = 1;
        let x = row_matrix(n, m); // row i == [i]
        let out = make_batch(&x, b).expect("ok");
        assert_eq!(out.size().as_slice(), &[b, m]);

        // Out should contain b distinct row indices in [0, n).
        let mut seen = HashSet::new();
        let out_1d = out.squeeze_dim(1).to_kind(Kind::Int64);
        for i in 0..b {
            let v = out_1d.i(i).int64_value(&[]);
            assert!(0 <= v && v < n, "sampled index out of bounds: {}", v);
            seen.insert(v);
        }
        assert_eq!(seen.len() as i64, b, "duplicates found in sampled rows");
    }

    #[test]
    fn small_n_mixup_only_stays_in_convex_hull() {
        tch::manual_seed(123);
        let n = 3;
        let b = 8;
        let m = 2;
        let x = row_matrix(n, m); // rows: [0,0], [1,2], [2,4]
        let cfg = MixupJitterCfg { do_mixup: true, do_jitter: false, alpha: 0.4, sigma: 0.0 };
        let out = make_batch_with(&x, b, cfg).expect("ok");
        assert_eq!(out.size().as_slice(), &[b, m]);

        // Per-feature bounds from original rows.
        let mins = x.min_dim(0, false).0; // [M]
        let maxs = x.max_dim(0, false).0; // [M]

        // Every output value must lie within per-feature [min, max].
        for i in 0..b {
            for j in 0..m {
                let v = out.i((i, j)).double_value(&[]);
                let lo = mins.i(j).double_value(&[]);
                let hi = maxs.i(j).double_value(&[]);
                assert!(lo - 1e-6 <= v && v <= hi + 1e-6, "value not in convex hull");
            }
        }
    }

    #[test]
    fn small_n_jitter_only_adds_noise() {
        tch::manual_seed(7);
        let n = 2;
        let b = 12;
        let m = 3;
        let x = Tensor::zeros([n, m], (Kind::Float, Device::Cpu));
        let cfg = MixupJitterCfg { do_mixup: false, do_jitter: true, alpha: 0.4, sigma: 0.05 };
        let out = make_batch_with(&x, b, cfg).expect("ok");
        assert_eq!(out.size().as_slice(), &[b, m]);

        // Since input is all zeros and we only jitter, sum of abs must be > 0.
        let sum_abs = out.abs().sum(Kind::Float).double_value(&[]);
        assert!(sum_abs > 0.0, "jitter had no effect");
    }

    #[test]
    fn dtype_preserved_in_small_n_branch() {
        tch::manual_seed(9);
        let n = 2;
        let b = 5;
        let m = 2;
        let x = row_matrix(n, m).to_kind(Kind::Double); // input as f64
        // Turn off mixup & jitter to isolate dtype path in the small-N branch.
        let cfg = MixupJitterCfg { do_mixup: false, do_jitter: false, alpha: 0.4, sigma: 0.0 };
        let out = make_batch_with(&x, b, cfg).expect("ok");
        assert_eq!(out.kind(), Kind::Double, "output dtype should match input");
        assert_eq!(out.size().as_slice(), &[b, m]);
    }

    #[test]
    fn norm_thickness_k_can_flip_filter_for_generated_batch() -> Result<(), tch::TchError> {
        tch::manual_seed(17);

        let motif = Tensor::from_slice(&[
            0.0f32, 1.0, 2.0, 1.0, 0.0, -0.5,
            0.0,    1.1, 1.9, 1.0, 0.0, -0.4,
            0.0,    0.9, 2.1, 1.0, 0.0, -0.6,
        ])
        .view([3, 6]);

        // Jitter-only keeps the test deterministic (tch::manual_seed controls randomness here).
        let cfg = MixupJitterCfg { alpha: 0.4, sigma: 0.08, do_mixup: false, do_jitter: true };
        let batch = make_batch_with(&motif, 32, cfg)?;
        assert_eq!(batch.size().as_slice(), &[32, 6]);

        let score_k_lo = norm_thickness(&batch, 1.0, 1e-8)?.double_value(&[]);
        let score_k_hi = norm_thickness(&batch, 12.0, 1e-8)?.double_value(&[]);

        assert!(score_k_hi < score_k_lo, "expected stronger k to lower score, got k=1:{score_k_lo:.6} k=12:{score_k_hi:.6}");

        // Derive a threshold from observed scores to show that changing k can change keep/drop decisions.
        let threshold = (score_k_lo + score_k_hi) / 2.0;
        let keep_k_lo = score_k_lo < threshold;
        let keep_k_hi = score_k_hi < threshold;
        assert_ne!(keep_k_lo, keep_k_hi, "expected k to flip threshold decision: k=1:{score_k_lo:.6} k=12:{score_k_hi:.6} threshold={threshold:.6}");

        Ok(())
    }
}
