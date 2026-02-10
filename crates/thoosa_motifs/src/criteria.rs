use tch::{Kind, Tensor, TchError};

fn scalar_like(x: &Tensor, v: f64) -> Tensor {
    Tensor::from(v).to_device(x.device()).to_kind(x.kind())
}

/// Quantile for x over dim=0 (x is [N, L]).
/// Implemented via sort + select (no torch::quantile dependency).
fn f_quantile_dim0(x: &Tensor, q: f64) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expects 2 dim tensor");

    let n = sz[0];

    assert!(n >= 1, "N must be >= 1, n={n}");

    let (sorted, _idx) = x.f_sort(0, false)?;
    let mut i = ((n - 1) as f64 * q).round() as i64;
    if i < 0 {
        i = 0;
    }
    if i > n - 1 {
        i = n - 1;
    }
    // returns [L]
    sorted.f_select(0, i)
}

/// Quantile for 1D tensor v over dim=0 (v is [L]).
fn f_quantile_1d(v: &Tensor, q: f64) -> Result<Tensor, TchError> {
    let sz = v.size();
    assert_eq!(sz.len(), 1, "expects 1 dim tensor");

    let l = sz[0];

    assert!(l >= 1, "L must be >= 1");

    let (sorted, _idx) = v.f_sort(0, false)?;
    let mut i = ((l - 1) as f64 * q).round() as i64;
    if i < 0 {
        i = 0;
    }
    if i > l - 1 {
        i = l - 1;
    }
    sorted.f_select(0, i) // scalar
}

/// 1) Robust envelope thickness normalized by robust amplitude of the consensus.
/// Lower is better.
///
/// thickness = mean_t( q90(x[:,t]) - q10(x[:,t]) )
/// amp       = q95(consensus) - q05(consensus)
/// score     = thickness / (amp + eps)
pub fn norm_thickness(x: &Tensor, eps: f64) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expects 2 dim tensor");
    let l = sz[1];

    let q10 = f_quantile_dim0(x, 0.10)?;
    let q90 = f_quantile_dim0(x, 0.90)?;
    let thickness_time = q90.f_sub(&q10)?; // [L]

    // thickness = mean(thickness_time) = sum / L
    let thickness_sum = thickness_time.f_sum(x.kind())?; // scalar
    let thickness = thickness_sum.f_div(&scalar_like(x, l as f64))?; // scalar

    // consensus = median across occurrences per timepoint
    let consensus = f_quantile_dim0(x, 0.50)?; // [L]

    // amp = q95(consensus) - q05(consensus) over time
    let c_q05 = f_quantile_1d(&consensus, 0.05)?;
    let c_q95 = f_quantile_1d(&consensus, 0.95)?;
    let amp = c_q95.f_sub(&c_q05)?; // scalar

    thickness.f_div(&amp.f_add(&scalar_like(x, eps))?)
}

/// Helper: per-occurrence RMS residual to a consensus (both aligned in time).
/// x: [N, L], consensus: [L] -> resid: [N]
fn f_rms_residuals(x: &Tensor, consensus: &Tensor) -> Result<Tensor, TchError> {
    let sz = x.size();
    let l = sz[1];

    let diff = x.f_sub(consensus)?;       // [N,L] (broadcast)
    let diff2 = diff.f_mul(&diff)?;       // [N,L]
    let sum = diff2.f_sum_dim_intlist(1, false, x.kind())?; // [N]
    let mse = sum.f_div(&scalar_like(x, l as f64))?;    // [N]
    mse.f_sqrt()
}

/// 2) Inlier ratio (explicit “strays”):
/// - consensus = per-time median
/// - resid_i = RMS(x_i - consensus)
/// - med = median(resid), mad = median(|resid-med|)
/// - inlier if resid <= med + k*mad
/// Returns fraction in [0,1] (scalar). Higher is better.
pub fn inlier_ratio(x: &Tensor, k: f64, eps: f64) -> Result<Tensor, TchError> {
    let sz = x.size();

    assert_eq!(sz.len(), 2, "expected [N,L], got {:?}", sz);
    let n = sz[0];

    let consensus = f_quantile_dim0(x, 0.50)?; // [L]
    let resid = f_rms_residuals(x, &consensus)?; // [N]

    let med = f_quantile_1d(&resid, 0.50)?; // scalar
    let abs_dev = resid.f_sub(&med)?.f_abs()?; // [N]
    let mad = f_quantile_1d(&abs_dev, 0.50)?; // scalar

    let thr = med.f_add(&mad.f_mul(&scalar_like(x, k))?)?
        .f_add(&scalar_like(x, eps))?; // scalar

    let inlier_mask = resid.f_le_tensor(&thr)?;          // [N] bool
    let inlier_f = inlier_mask.f_to_kind(Kind::Float)?;  // [N]
    let inlier_sum = inlier_f.f_sum(Kind::Float)?;       // scalar
    inlier_sum.f_div(&Tensor::from(n as f64).to_device(x.device()))
}

/// 3) Coherence (“rank-1-ness”) without SVD:
/// Let Xc be x centered by column mean. Then:
///   G = Xc * Xc^T  (N×N), eigenvalues of G are σ_i^2
/// coherence = (largest eigenvalue of G) / trace(G)
///
/// We approximate largest eigenvalue by power iteration.
/// Higher is better.
pub fn coherence(x: &Tensor, iters: i64, eps: f64) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N,L], got {:?}", sz);

    let n = sz[0];

    // center by column mean: mean over N (dim=0)
    let mean = x.f_mean_dim(0, true, x.kind())?; // [1,L]
    let xc = x.f_sub(&mean)?;                // [N,L]

    // G = Xc * Xc^T : [N,N]
    let g = xc.f_matmul(&xc.f_transpose(0, 1)?)?;

    // trace(G) = sum diag
    let diag = g.f_diag(0)?; // [N]
    let trace = diag.f_sum(x.kind())?; // scalar

    // power iteration for top eigenvalue
    let mut v = Tensor::randn(&[n, 1], (x.kind(), x.device()));
    for _ in 0..iters {
        let gv = g.f_matmul(&v)?;                 // [N,1]
        let gv2 = gv.f_mul(&gv)?;          // [N,1]
        let norm2 = gv2.f_sum(x.kind())?;         // scalar
        let norm = norm2.f_add(&scalar_like(x, eps))?.f_sqrt()?; // scalar
        v = gv.f_div(&norm)?;              // [N,1]
    }

    // Rayleigh quotient λ = v^T G v
    let gv = g.f_matmul(&v)?; // [N,1]
    let vt = v.f_transpose(0, 1)?; // [1,N]
    let lambda = vt.f_matmul(&gv)?; // [1,1]
    let lambda = lambda.f_squeeze()?; // scalar

    // coherence = lambda / (trace + eps)
    lambda.f_div(&trace.f_add(&scalar_like(x, eps))?)
}

#[cfg(test)]
mod tests {
    #[cfg(test)]
    mod tests {
        use log::debug;
        use super::*;
        use tch::{Tensor, TchError};
        use crate::criteria::{coherence, inlier_ratio, norm_thickness};

        fn f64_of_scalar(t: &Tensor) -> f64 {
            assert!(
                t.size().is_empty(),
                "expected scalar tensor, got size={:?}",
                t.size()
            );
            t.double_value(&[])
        }

        fn assert_close(got: f64, expected: f64, tol: f64) {
            let diff = (got - expected).abs();
            assert!(
                diff <= tol,
                "got={got:.12}, expected={expected:.12}, diff={diff:.12}, tol={tol}",
            );
        }

        #[test]
        fn test_norm_thickness_identical_is_zero() -> Result<(), TchError> {
            // 3 identical occurrences => envelope thickness = 0 => score = 0
            let x = Tensor::from_slice(&[
                0.0f32, 1.0, 2.0, 1.0,
                0.0,    1.0, 2.0, 1.0,
                0.0,    1.0, 2.0, 1.0,
            ])
                .view([3, 4]);

            let s = norm_thickness(&x, 1e-8)?;
            assert_close(f64_of_scalar(&s), 0.0, 1e-12);
            Ok(())
        }

        #[test]
        fn test_norm_thickness_known_value_n3_rounding() -> Result<(), TchError> {
            // For N=3, our quantile impl uses sort+select with rounding:
            // q10 -> min, q90 -> max. So spread_t = max-min at each t.
            //
            // Occurrences:
            // t1: [1.0, 1.1, 0.9] spread=0.2
            // t2: [2.0, 1.9, 2.1] spread=0.2
            // thickness = mean([0,0.2,0.2,0]) = 0.1
            // consensus median = [0,1,2,1]
            // amp = q95-q05 over time = 2-0 = 2
            // norm_thickness = 0.1/2 = 0.05
            let x = Tensor::from_slice(&[
                0.0f32, 1.0, 2.0, 1.0,
                0.0,    1.1, 1.9, 1.0,
                0.0,    0.9, 2.1, 1.0,
            ])
                .view([3, 4]);

            let s = norm_thickness(&x, 1e-8)?;
            assert_close(f64_of_scalar(&s), 0.05, 1e-6);
            Ok(())
        }

        #[test]
        fn test_inlier_ratio_all_inliers_is_one() -> Result<(), TchError> {
            let x = Tensor::from_slice(&[
                0.0f32, 1.0, 2.0, 1.0,
                0.0,    1.0, 2.0, 1.0,
                0.0,    1.0, 2.0, 1.0,
                0.0,    1.0, 2.0, 1.0,
            ])
                .view([4, 4]);

            let r = inlier_ratio(&x, 3.0, 1e-8)?;
            assert_close(f64_of_scalar(&r), 1.0, 1e-12);
            Ok(())
        }

        #[test]
        fn test_inlier_ratio_one_stray() -> Result<(), TchError> {
            // 4 good occurrences + 1 stray far away
            // consensus (median per t) == good motif, residuals good = 0
            // threshold ~= eps, so only good are inliers => 4/5 = 0.8
            let x = Tensor::from_slice(&[
                0.0f32, 1.0, 2.0, 1.0,
                0.0,    1.0, 2.0, 1.0,
                0.0,    1.0, 2.0, 1.0,
                0.0,    1.0, 2.0, 1.0,
                10.0,  10.0,10.0,10.0,
            ])
                .view([5, 4]);

            let r = inlier_ratio(&x, 3.0, 1e-8)?;
            assert_close(f64_of_scalar(&r), 0.8, 1e-6);
            Ok(())
        }

        #[test]
        fn test_coherence_rank1_near_one() -> Result<(), TchError> {
            // Construct rank-1 data: x_i = a_i * p
            // After column-centering: still rank-1 => coherence ~ 1
            tch::manual_seed(0);

            let p = [0.0f32, 1.0, 2.0, 1.0];
            let a = [1.0f32, 2.0, 3.0, 4.0];

            let mut data = Vec::new();
            for &ai in &a {
                for &pt in &p {
                    data.push(ai * pt);
                }
            }
            let x = Tensor::from_slice(&data).view([4, 4]);

            let c = coherence(&x, 30, 1e-8)?;
            let cv = f64_of_scalar(&c);
            assert!(cv.is_finite());
            assert!(
                cv > 0.999,
                "expected coherence close to 1 for rank-1 data, got {cv}"
            );
            Ok(())
        }

        #[test]
        fn test_coherence_mixed_is_lower() -> Result<(), TchError> {
            // Start from rank-1, then add a strong “orthogonal-ish” component to two rows => rank-2-ish.
            // Expected coherence noticeably < 1. For this construction it should be about 0.769...
            tch::manual_seed(0);

            let p = [0.0f32, 1.0, 2.0, 1.0];
            let a = [1.0f32, 2.0, 3.0, 4.0];
            let q = [1.0f32, -1.0, 1.0, -1.0];

            let mut x = Tensor::zeros(&[4, 4], (tch::Kind::Float, tch::Device::Cpu));
            for i in 0..4 {
                let mut row = Vec::new();
                for t in 0..4 {
                    row.push(a[i] * p[t]);
                }
                x.get(i as i64).copy_(&Tensor::from_slice(&row));
            }

            // Add strong component to row 0 and row 3
            let add = Tensor::from_slice(&q) * 5.0;
            let _ = x.get(0).f_add_(&add)?;
            let _ = x.get(3).f_add_(&add)?;

            let c = coherence(&x, 40, 1e-8)?;
            let cv = f64_of_scalar(&c);
            assert!(cv.is_finite());

            // Should be substantially lower than 1
            assert!(
                (0.70..0.85).contains(&cv),
                "expected coherence in ~[0.70,0.85], got {cv}"
            );
            Ok(())
        }
    }

}