use std::fmt::Debug;
use std::sync::{Arc, OnceLock};
use parking_lot::RwLock;
use tch::{Kind, TchError, Tensor};
use tch::Kind::Float;
use tracing::debug;
use thoosa_common::utils::tensor_digest;

type NormThicknessMockFn = Arc<dyn Fn(&Tensor, f64) -> Result<Tensor, TchError> + Send + Sync>;
type InlierRatioMockFn = Arc<dyn Fn(&Tensor, f64, f64) -> Result<Tensor, TchError> + Send + Sync>;
type CoherenceMockFn = Arc<dyn Fn(&Tensor, i64, f64) -> Result<Tensor, TchError> + Send + Sync>;

#[derive(Default, Clone)]
struct CriteriaMocks {
    norm_thickness: Option<NormThicknessMockFn>,
    inlier_ratio: Option<InlierRatioMockFn>,
    coherence: Option<CoherenceMockFn>,
}

fn criteria_mocks() -> &'static RwLock<CriteriaMocks> {
    static MOCKS: OnceLock<RwLock<CriteriaMocks>> = OnceLock::new();
    MOCKS.get_or_init(|| RwLock::new(CriteriaMocks::default()))
}

fn maybe_norm_thickness_mock(x: &Tensor, eps: f64) -> Option<Result<Tensor, TchError>> {
    let mock_fn = criteria_mocks().read().norm_thickness.as_ref().map(Arc::clone);
    mock_fn.map(|mock_fn| mock_fn(x, eps))
}

fn maybe_inlier_ratio_mock(x: &Tensor, k: f64, eps: f64) -> Option<Result<Tensor, TchError>> {
    let mock_fn = criteria_mocks().read().inlier_ratio.as_ref().map(Arc::clone);
    mock_fn.map(|mock_fn| mock_fn(x, k, eps))
}

fn maybe_coherence_mock(x: &Tensor, iters: i64, eps: f64) -> Option<Result<Tensor, TchError>> {
    let mock_fn = criteria_mocks().read().coherence.as_ref().map(Arc::clone);
    mock_fn.map(|mock_fn| mock_fn(x, iters, eps))
}

pub mod mock_criteria {
    use super::{criteria_mocks, CoherenceMockFn, CriteriaMocks, InlierRatioMockFn, NormThicknessMockFn};
    use parking_lot::{Mutex, MutexGuard};
    use std::sync::{Arc, OnceLock};
    use tch::{TchError, Tensor};

    fn install_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    pub struct MockCriteriaGuard {
        _lock: MutexGuard<'static, ()>,
    }

    impl Drop for MockCriteriaGuard {
        fn drop(&mut self) {
            *criteria_mocks().write() = CriteriaMocks::default();
        }
    }

    #[derive(Default)]
    pub struct MockCriteriaBuilder {
        mocks: CriteriaMocks,
    }

    impl MockCriteriaBuilder {
        pub fn with_norm_thickness<F>(mut self, f: F) -> Self
        where
            F: Fn(&Tensor, f64) -> Result<Tensor, TchError> + Send + Sync + 'static,
        {
            self.mocks.norm_thickness = Some(Arc::new(f) as NormThicknessMockFn);
            self
        }

        pub fn with_inlier_ratio<F>(mut self, f: F) -> Self
        where
            F: Fn(&Tensor, f64, f64) -> Result<Tensor, TchError> + Send + Sync + 'static,
        {
            self.mocks.inlier_ratio = Some(Arc::new(f) as InlierRatioMockFn);
            self
        }

        pub fn with_coherence<F>(mut self, f: F) -> Self
        where
            F: Fn(&Tensor, i64, f64) -> Result<Tensor, TchError> + Send + Sync + 'static,
        {
            self.mocks.coherence = Some(Arc::new(f) as CoherenceMockFn);
            self
        }

        pub fn install(self) -> MockCriteriaGuard {
            let lock = install_lock().lock();
            *criteria_mocks().write() = self.mocks;
            MockCriteriaGuard { _lock: lock }
        }
    }

    pub fn build() -> MockCriteriaBuilder {
        MockCriteriaBuilder::default()
    }
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
/// score     = 1 / (1 + k * thickness / (amp + eps))
pub fn norm_thickness<ID:Debug>(id:&ID, x: &Tensor, eps: f64) -> Result<Tensor, TchError> {
    debug!("norm_thickness: id={id:?} tensor_digest={:?}", tensor_digest(x));

    if let Some(mocked) = maybe_norm_thickness_mock(x, eps) {
        return mocked;
    }
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expects [N, L]");
    let l = sz[1];

    let q10 = f_quantile_dim0(x, 0.10)?;
    let q90 = f_quantile_dim0(x, 0.90)?;
    let thickness_time = q90.f_sub(&q10)?;               // [L]

    let thickness_sum = thickness_time.f_sum(x.kind())?; // scalar
    let thickness = thickness_sum.f_div_scalar(l as f64)?; // scalar

    let consensus = f_quantile_dim0(x, 0.50)?;           // [L]
    let c_q05 = f_quantile_1d(&consensus, 0.05)?;
    let c_q95 = f_quantile_1d(&consensus, 0.95)?;
    let amp = c_q95.f_sub(&c_q05)?;                      // scalar

    // s = thickness / (amp + eps)#[cfg(test)]
    let denom = amp.f_add_scalar(eps)?;
    let s = thickness.f_div(&denom)?;                    // scalar

    debug!("norm_thickness: s=\n{s}");

    // Q = 1 / (1 + s)
    let one = Tensor::from(1.0).to_device(x.device()).to_kind(x.kind());
    let one_plus_s = one.f_add(&s)?;
    one.f_div(&one_plus_s)

    // let s_strict = s.f_mul_scalar(k)?;
    //
    // // Q = 1 / (1 + s_strict)
    // let one = Tensor::from(1.0).to_device(x.device()).to_kind(x.kind());
    // let denom = one.f_add(&s_strict)?;
    // one.f_div(&denom)
}


/// Helper: per-occurrence RMS residual to a consensus (both aligned in time).
/// x: [N, L], consensus: [L] -> resid: [N]
fn f_rms_residuals(x: &Tensor, consensus: &Tensor) -> Result<Tensor, TchError> {
    let sz = x.size();
    let l = sz[1];

    let diff = x.f_sub(consensus)?;       // [N,L] (broadcast)
    let diff2 = diff.f_mul(&diff)?;       // [N,L]
    let sum = diff2.f_sum_dim_intlist(1, false, x.kind())?; // [N]
    let mse = sum.f_div_scalar(l as f64)?;    // [N]
    mse.f_sqrt()
}

/// 2) Inlier ratio (explicit “strays”):
/// k - how many MADs above the median residual you still call “inlier”.
/// k ≈ 4.5 → “3σ-ish” robust cutoff (moderate)
///
/// k ≈ 3.0 → aggressive (will reject more, especially if alignment jitter exists)
///
/// k ≈ 6.0 → lenient (keeps more shifted/noisy slices)
/// - consensus = per-time median
/// - resid_i = RMS(x_i - consensus)
/// - med = median(resid), mad = median(|resid-med|)
/// - inlier if resid <= med + k*mad
/// Returns fraction in [0,1] (scalar). Higher is better.
pub fn inlier_ratio(x: &Tensor, k: f64, eps: f64) -> Result<Tensor, TchError> {
    if let Some(mocked) = maybe_inlier_ratio_mock(x, k, eps) {
        return mocked;
    }
    let sz = x.size();

    assert_eq!(sz.len(), 2, "expected [N,L], got {:?}", sz);
    let n = sz[0];

    let consensus = f_quantile_dim0(x, 0.50)?; // [L]
    let resid = f_rms_residuals(x, &consensus)?; // [N]

    let med = f_quantile_1d(&resid, 0.50)?; // scalar
    let abs_dev = resid.f_sub(&med)?.f_abs()?; // [N]
    let mad = f_quantile_1d(&abs_dev, 0.50)?; // scalar

    let thr = med.f_add(&mad.f_mul_scalar(k)?)?
        .f_add_scalar(eps)?; // scalar

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
    if let Some(mocked) = maybe_coherence_mock(x, iters, eps) {
        return mocked;
    }
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
        let norm = norm2.f_add_scalar(eps)?.f_sqrt()?; // scalar
        v = gv.f_div(&norm)?;              // [N,1]
    }

    // Rayleigh quotient λ = v^T G v
    let gv = g.f_matmul(&v)?; // [N,1]
    let vt = v.f_transpose(0, 1)?; // [1,N]
    let lambda = vt.f_matmul(&gv)?; // [1,1]
    let lambda = lambda.f_squeeze()?; // scalar

    // coherence = lambda / (trace + eps)
    lambda.f_div(&trace.f_add_scalar(eps)?)
}




fn scalar_like(x: &Tensor, v: f64) -> Tensor {
    Tensor::from(v).to_device(x.device()).to_kind(x.kind())
}

fn consensus_median(x: &Tensor) -> Result<Tensor, TchError> {
    // x: [N,L] -> c: [L]
    f_quantile_dim0(x, 0.50)
}

/// Robust amplitude of the consensus curve:
/// A = q_hi(c) - q_lo(c)   over time
/// Higher is "more interesting" (bigger signal).
pub fn interesting_amp_consensus(
    x: &Tensor,
    q_lo: f64,   // e.g. 0.05
    q_hi: f64,   // e.g. 0.95
) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N,L]");
    let c = consensus_median(x)?; // [L]
    let lo = f_quantile_1d(&c, q_lo)?;
    let hi = f_quantile_1d(&c, q_hi)?;
    hi.f_sub(&lo) // scalar
}

/// Total variation ("wiggliness") of the consensus:
/// TV = sum_{t=0..L-2} |c[t+1]-c[t]|
/// Returns TV normalized by robust amplitude: TV / (A + eps)
/// Higher => more (non-periodic-friendly) fluctuations.
pub fn interesting_tv_norm(
    x: &Tensor,
    amp_q_lo: f64, // e.g. 0.05
    amp_q_hi: f64, // e.g. 0.95
    eps: f64,
) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N,L]");
    let l = sz[1];
    assert!(l >= 2, "need L>=2");

    let c = consensus_median(x)?; // [L]

    let c1 = c.f_narrow(0, 1, l - 1)?;     // [L-1]
    let c0 = c.f_narrow(0, 0, l - 1)?;     // [L-1]
    let dv = c1.f_sub(&c0)?.f_abs()?;      // [L-1]
    let tv = dv.f_sum(x.kind())?;          // scalar

    let amp = interesting_amp_consensus(x, amp_q_lo, amp_q_hi)?; // scalar
    tv.f_div(&amp.f_add_scalar(eps)?) // scalar
}

/// Curvature / "bends" of the consensus via second differences:
/// CURV = sum_{t=1..L-2} |c[t+1] - 2c[t] + c[t-1]|
/// Returns CURV normalized by robust amplitude: CURV / (A + eps)
/// Higher => more turning / local structure.
pub fn interesting_curvature_norm(
    x: &Tensor,
    amp_q_lo: f64, // e.g. 0.05
    amp_q_hi: f64, // e.g. 0.95
    eps: f64,
) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N,L]");
    let l = sz[1];
    assert!(l >= 3, "need L>=3");

    let c = consensus_median(x)?; // [L]

    let cp = c.f_narrow(0, 2, l - 2)?; // c[t+1], t=1..L-2  -> [L-2]
    let cc = c.f_narrow(0, 1, l - 2)?; // c[t],   t=1..L-2  -> [L-2]
    let cm = c.f_narrow(0, 0, l - 2)?; // c[t-1], t=1..L-2  -> [L-2]

    let two = scalar_like(&c, 2.0);
    let d2 = cp.f_sub(&cc.f_mul(&two)?)?.f_add(&cm)?; // [L-2]
    let curv = d2.f_abs()?.f_sum(x.kind())?;          // scalar

    let amp = interesting_amp_consensus(x, amp_q_lo, amp_q_hi)?; // scalar
    curv.f_div(&amp.f_add_scalar(eps)?) // scalar
}

/// Turning-points ratio of the consensus:
/// Count sign changes in first difference (ignores zeros).
/// ratio = (# sign changes) / (L-2)
/// Higher => more "nonlinear" / wiggly.
/// NOTE: This is cheap and often very effective for rejecting lines.
pub fn interesting_turning_points_ratio(x: &Tensor) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N,L]");
    let l = sz[1];
    assert!(l >= 3, "need L>=3");

    let c = consensus_median(x)?; // [L]

    // d[t] = c[t+1]-c[t], length L-1
    let d = c
        .f_narrow(0, 1, l - 1)?
        .f_sub(&c.f_narrow(0, 0, l - 1)?)?; // [L-1]

    let s = d.f_sign()?; // [-1,0,1], [L-1]

    // sign change when s[t]*s[t+1] < 0
    let s0 = s.f_narrow(0, 0, l - 2)?; // [L-2]
    let s1 = s.f_narrow(0, 1, l - 2)?; // [L-2]
    let prod = s0.f_mul(&s1)?;         // [L-2]
    let ch = prod.f_lt(0.0)?.f_to_kind(Kind::Float)?; // [L-2] float mask
    let count = ch.f_sum(Kind::Float)?; // scalar

    debug!("count=\n{count}");

    count.f_div_scalar((l - 2) as f64)
}


/// Turning-points score with a "peak" turn count.
///
/// We count sign changes in the first difference of the consensus curve.
/// Let T = number of turns (sign changes).
///
/// Score is triangular:
///   if T <= peak_turns:         score = T / peak_turns
///   if peak_turns < T < 2P:     score = (2P - T) / peak_turns
///   if T >= 2*peak_turns:       score = 0
///
/// Output: scalar Tensor in [0,1], where 1 is "best" (at peak_turns).
pub fn interesting_turning_points_ratio_peaked_01(
    x: &Tensor,        // [N, L]
    peak_turns: u32,   // P > 0
) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N,L]");
    let l = sz[1];
    assert!(l >= 3, "need L>=3");
    assert!(peak_turns > 0, "peak_turns must be > 0");

    // consensus = median across occurrences per timepoint
    let c = f_quantile_dim0(x, 0.50)?; // [L]

    // d[t] = c[t+1]-c[t], length L-1
    let d = c
        .f_narrow(0, 1, l - 1)?
        .f_sub(&c.f_narrow(0, 0, l - 1)?)?; // [L-1]

    // s[t] in {-1,0,1}
    let s = d.f_sign()?; // [L-1]

    // sign change when s[t]*s[t+1] < 0
    let s0 = s.f_narrow(0, 0, l - 2)?; // [L-2]
    let s1 = s.f_narrow(0, 1, l - 2)?; // [L-2]
    let prod = s0.f_mul(&s1)?;         // [L-2]
    let turns_mask = prod.f_lt(0.0)?;  // [L-2] bool
    let turns = turns_mask
        .f_to_kind(Kind::Float)?
        .f_sum(Kind::Float)?;          // scalar (float)

    // Triangular score around peak_turns
    let p = peak_turns as f64;
    let p2 = 2.0 * p;

    let turns_k = turns.f_to_kind(x.kind())?; // scalar in x.kind()
    let p_k = scalar_like(x, p);
    let p2_k = scalar_like(x, p2);

    let score_left = turns_k.f_div(&p_k)?;                // T / P
    let score_right = p2_k.f_sub(&turns_k)?.f_div(&p_k)?; // (2P - T) / P

    let m_left = turns_k.f_le_tensor(&p_k)?.f_to_kind(x.kind())?; // 1 if T<=P else 0
    let m_mid = turns_k
        .f_gt_tensor(&p_k)?
        .f_bitwise_and_tensor(&turns_k.f_lt_tensor(&p2_k)?)?
        .f_to_kind(x.kind())?; // 1 if P<T<2P else 0

    // score = m_left*score_left + m_mid*score_right (else 0)
    let s_left = m_left.f_mul(&score_left)?;
    let s_mid = m_mid.f_mul(&score_right)?;
    s_left.f_add(&s_mid)
}

/// Flat-top turning-points score in [0,1].
///
/// Computes T = (# sign changes in first difference of consensus median curve).
/// Then applies trapezoid scoring with edges (a,b,c,d):
///   T <= a : 0
///   a < T < b : (T-a)/(b-a)
///   b <= T <= c : 1
///   c < T < d : (d-T)/(d-c)
///   T >= d : 0
///
/// Example: a=2,b=3,c=5,d=7 gives:
///   T=2.5 -> 0.5,  T in [3,5] -> 1,  T=6 -> 0.5
pub fn interesting_turning_points_ratio_trapezoid_01(
    x: &Tensor,  // [N, L]
    a: i64,
    b: i64,
    c: i64,
    d: i64,
) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N,L]");
    let l = sz[1];
    assert!(l >= 3, "need L>=3");
    assert!(a < b && b <= c && c < d, "need a < b <= c < d");

    // consensus = median across occurrences per timepoint
    let cns = f_quantile_dim0(x, 0.50)?; // [L]

    // d[t] = c[t+1] - c[t], length L-1
    let d1 = cns
        .f_narrow(0, 1, l - 1)?
        .f_sub(&cns.f_narrow(0, 0, l - 1)?)?; // [L-1]

    // sign in {-1,0,1}
    let s = d1.f_sign()?; // [L-1]

    // turn when s[t]*s[t+1] < 0
    let s0 = s.f_narrow(0, 0, l - 2)?; // [L-2]
    let s1 = s.f_narrow(0, 1, l - 2)?; // [L-2]
    let prod = s0.f_mul(&s1)?;         // [L-2]
    let turns = prod
        .f_lt(0.0)?                    // bool [L-2]
        .f_to_kind(Kind::Float)?       // float mask
        .f_sum(Kind::Float)?;          // scalar (float)

    let t = turns.f_to_kind(x.kind())?; // scalar in x.kind()

    // boundaries as scalars
    let a_k = scalar_like(x, a as f64);
    let b_k = scalar_like(x, b as f64);
    let c_k = scalar_like(x, c as f64);
    let d_k = scalar_like(x, d as f64);

    let one = scalar_like(x, 1.0);
    let zero = scalar_like(x, 0.0);

    // rise: (t-a)/(b-a)
    let rise = t
        .f_sub(&a_k)?
        .f_div(&b_k.f_sub(&a_k)?)?;

    // fall: (d-t)/(d-c)
    let fall = d_k
        .f_sub(&t)?
        .f_div(&d_k.f_sub(&c_k)?)?;

    // masks
    let m_rise = t
        .f_gt_tensor(&a_k)?
        .f_bitwise_and_tensor(&t.f_lt_tensor(&b_k)?)?
        .f_to_kind(x.kind())?;

    let m_flat = t
        .f_ge_tensor(&b_k)?
        .f_bitwise_and_tensor(&t.f_le_tensor(&c_k)?)?
        .f_to_kind(x.kind())?;

    let m_fall = t
        .f_gt_tensor(&c_k)?
        .f_bitwise_and_tensor(&t.f_lt_tensor(&d_k)?)?
        .f_to_kind(x.kind())?;

    // score = m_rise*rise + m_flat*1 + m_fall*fall
    let s_r = m_rise.f_mul(&rise)?;
    let s_f = m_fall.f_mul(&fall)?;
    let s_flat = m_flat.f_mul(&one)?;
    s_r.f_add(&s_flat)?.f_add(&s_f)?.f_maximum(&zero)?.f_minimum(&one)
}

/// Helper: power spectrum of consensus using torch.fft.rfft.
/// Returns power: [F] where F = floor(L/2)+1, includes DC bin at 0.
fn consensus_power_rfft(c: &Tensor) -> Result<Tensor, TchError> {
    // c: [L]
    let c2 = c.f_unsqueeze(0)?; // [1,L]
    let l = c2.size()[1];
    let spec = c2.f_fft_rfft(Some(l), 1, "forward")?; // [1,F] complex
    let p = spec.f_abs()?.f_pow_tensor_scalar(2.0)?; // [1,F]
    p.f_squeeze() // [F]
}

/// High-band energy ratio of the consensus:
/// ratio = sum_{f in [f_lo,f_hi]} P(f) / sum_{f>0} P(f)
/// Excludes DC from denominator.
/// Higher => more high-frequency / fast structure.
/// Needs sample rate to map Hz->bins.
pub fn interesting_band_energy_ratio(
    x: &Tensor,
    fs: f64,
    f_lo: f64,
    f_hi: f64,
    eps: f64,
) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N,L]");
    let l = sz[1];
    assert!(l >= 2, "need L>=2");
    assert!(fs > 0.0, "fs must be > 0");

    let c = consensus_median(x)?;     // [L]
    let p = consensus_power_rfft(&c)?; // [F]
    let f_bins = p.size()[0];         // F = L/2+1

    // denom: sum over bins 1..F-1 (exclude DC)
    if f_bins <= 1 {
        return Ok(scalar_like(x, 0.0));
    }
    let denom = p
        .f_narrow(0, 1, f_bins - 1)?
        .f_sum(x.kind())?
        .f_add_scalar(eps)?; // scalar

    // map Hz -> bin index k ~= f * L / fs, clamp to [1, F-1]
    let lo_hz = f_lo.max(0.0);
    let hi_hz = f_hi.max(0.0);
    let (lo_hz, hi_hz) = if lo_hz <= hi_hz { (lo_hz, hi_hz) } else { (hi_hz, lo_hz) };

    let n = l as f64;
    let mut k_lo = (lo_hz * n / fs).ceil() as i64;
    let mut k_hi = (hi_hz * n / fs).floor() as i64;

    let k_min = 1i64;
    let k_max = (f_bins - 1) as i64;
    if k_lo < k_min { k_lo = k_min; }
    if k_hi > k_max { k_hi = k_max; }

    if k_hi < k_lo {
        return Ok(scalar_like(x, 0.0));
    }

    let len = k_hi - k_lo + 1;
    let band = p.f_narrow(0, k_lo, len)?.f_sum(x.kind())?; // scalar

    band.f_div(&denom)
}

/// Spectral entropy of the consensus (normalized to [0,1]):
/// Compute power P(k) for k=1..F-1 (exclude DC),
/// p_k = P(k)/sum P
/// H = -sum p_k log(p_k) / log(K)
///
/// Interpretation:
/// - ~0 : very tonal / narrowband / simple
/// - ~1 : very flat / noise-like
/// Often "interesting but structured" is somewhere in the middle.
pub fn interesting_spectral_entropy(
    x: &Tensor,
    eps: f64,
) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N,L]");
    let l = sz[1];
    assert!(l >= 2, "need L>=2");

    let c = consensus_median(x)?;
    let p = consensus_power_rfft(&c)?; // [F]
    let f_bins = p.size()[0];
    if f_bins <= 2 {
        // no meaningful non-DC bins
        return Ok(scalar_like(x, 0.0));
    }

    // P_ndc: bins 1..F-1
    let p_ndc = p.f_narrow(0, 1, f_bins - 1)?; // [K]
    let sum = p_ndc.f_sum(x.kind())?.f_add_scalar(eps)?; // scalar
    let prob = p_ndc.f_div(&sum)?; // [K]

    // H = -sum p log p
    let logp = prob.f_add_scalar(eps)?.f_log()?; // [K]
    let plogp = prob.f_mul(&logp)?;              // [K]
    let h = plogp.f_sum(x.kind())?.f_neg()?;     // scalar

    let k = (f_bins - 1) as f64;
    let logk = k.ln().max(eps);
    h.f_div_scalar(logk) // scalar in ~[0,1]
}


/// Map s>=0 to [0,1): q = s / (s + s0)
/// If s0 is small, q saturates fast; if large, q is conservative.
fn squash_pos_to_01(s: &Tensor, s0: f64, eps: f64) -> Result<Tensor, TchError> {
    assert!(s0 > 0.0, "s0 must be > 0");
    let denom = s.f_add_scalar(s0)?.f_add_scalar(eps)?; // s + s0 + eps
    s.f_div(&denom)
}

/// 0..1, 1 good: high consensus amplitude is good.
pub fn interesting_amp_01(
    x: &Tensor,
    q_lo: f64,   // e.g. 0.05
    q_hi: f64,   // e.g. 0.95
    a0: f64,     // scale (amp == a0 => 0.5)
    eps: f64,
) -> Result<Tensor, TchError> {
    let a = interesting_amp_consensus(x, q_lo, q_hi)?;  // scalar >=0
    squash_pos_to_01(&a, a0, eps)
}

/// 0..1, 1 good: high wiggliness per amplitude is good.
pub fn interesting_tv_norm_01(
    x: &Tensor,
    amp_q_lo: f64,
    amp_q_hi: f64,
    tv0: f64,   // scale (tv_norm == tv0 => 0.5)
    eps: f64,
) -> Result<Tensor, TchError> {
    let tvn = interesting_tv_norm(x, amp_q_lo, amp_q_hi, eps)?; // scalar >=0
    squash_pos_to_01(&tvn, tv0, eps)
}

/// 0..1, 1 good: high curvature per amplitude is good.
pub fn interesting_curvature_norm_01(
    x: &Tensor,
    amp_q_lo: f64,
    amp_q_hi: f64,
    curv0: f64, // scale (curv_norm == curv0 => 0.5)
    eps: f64,
) -> Result<Tensor, TchError> {
    let cn = interesting_curvature_norm(x, amp_q_lo, amp_q_hi, eps)?; // scalar >=0
    squash_pos_to_01(&cn, curv0, eps)
}

/// Already in [0,1], higher => more energy in the chosen band.
pub fn interesting_band_energy_ratio_01(
    x: &Tensor,
    fs: f64,
    f_lo: f64,
    f_hi: f64,
    eps: f64,
) -> Result<Tensor, TchError> {
    interesting_band_energy_ratio(x, fs, f_lo, f_hi, eps)
}

/// Spectral entropy H is ~[0,1] but "best" is usually *middle*, not max.
/// Smooth "1 good" mapping: peak at h0 with width sigma.
pub fn interesting_entropy_gauss_01(
    x: &Tensor,
    h0: f64,     // preferred entropy (e.g. 0.5..0.7)
    sigma: f64,  // width (e.g. 0.10..0.25)
    eps: f64,
) -> Result<Tensor, TchError> {
    assert!(sigma > 0.0, "sigma must be > 0");
    let h = interesting_spectral_entropy(x, eps)?; // scalar ~[0,1]

    let z = h.f_sub_scalar(h0)?.f_div_scalar(sigma)?; // (h-h0)/sigma
    let z2 = z.f_mul(&z)?;                             // z^2
    // q = exp(-0.5*z^2)
    let neg_half = scalar_like(x, -0.5);
    let arg = z2.f_mul(&neg_half)?;                    // -0.5*z^2
    arg.f_exp()
}

/// Hard band-pass on entropy: 1 if H in [h_min, h_max], else 0.
/// Useful to reject "too tonal" (low H) and "too noisy" (high H).
pub fn interesting_entropy_band_01(
    x: &Tensor,
    h_min: f64,
    h_max: f64,
    eps: f64,
) -> Result<Tensor, TchError> {
    let (h_min, h_max) = if h_min <= h_max { (h_min, h_max) } else { (h_max, h_min) };
    let h = interesting_spectral_entropy(x, eps)?; // scalar
    let ge = h.f_ge(h_min)?.f_to_kind(Kind::Float)?; // scalar float mask {0,1}
    let le = h.f_le(h_max)?.f_to_kind(Kind::Float)?;
    ge.f_mul(&le)?.f_to_kind(x.kind())
}

pub fn periodicity_acf_01_frac(
    x: &Tensor,            // [N, L]
    struct_frac: f64,      // e.g. 1.0/64.0
    lag_min_frac: f64,     // e.g. 1.0/32.0
    lag_max_frac: f64,     // e.g. 0.5
    struct0: f64,
    peak_good: f64,
    peak_bad: f64,
    eps: f64,
) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N, L]");
    let l = sz[1];
    assert!(l >= 4, "need L>=4");

    assert!(struct_frac > 0.0, "struct_frac must be > 0");
    assert!(lag_min_frac > 0.0, "lag_min_frac must be > 0");
    assert!(lag_max_frac > 0.0, "lag_max_frac must be > 0");
    assert!(lag_max_frac >= lag_min_frac, "need lag_max_frac >= lag_min_frac");

    let lf = l as f64;

    let mut struct_lag_max = (struct_frac * lf).round() as i64;
    let mut lag_min = (lag_min_frac * lf).round() as i64;
    let mut lag_max = (lag_max_frac * lf).round() as i64;

    // clamp to valid ranges
    if struct_lag_max < 1 { struct_lag_max = 1; }
    if lag_min < 1 { lag_min = 1; }
    if lag_max < lag_min { lag_max = lag_min; }
    if lag_max > l - 1 { lag_max = l - 1; }

    // ensure periodicity window starts after structure window
    if lag_min <= struct_lag_max {
        lag_min = struct_lag_max + 1;
        if lag_min > lag_max {
            // minimal fix: push lag_max up if possible, else push lag_min down
            lag_max = (lag_min).min(l - 1);
        }
    }

    // If after adjustments the window is degenerate, you can return "not periodic" or error.
    // Here we keep it valid and call the integer version.
    periodicity_acf_01(
        x,
        struct_lag_max,
        lag_min,
        lag_max,
        struct0,
        peak_good,
        peak_bad,
        eps,
    )
}

/// Score in [0,1], where:
///   1 = structured AND not strongly periodic
///   0 = strongly periodic OR noise-like
///
/// x: [N, L] (batch of slices)
///
/// Params:
/// - struct_lag_max: short-lag window for "structure" (anti-noise), e.g. 16..64
/// - lag_min..lag_max: lags where periodicity is searched (sub-pattern repeat),
///                     IMPORTANT: choose lag_min > struct_lag_max.
/// - struct0: scale for structure squashing: structure01 = s/(s+struct0)
/// - peak_good/peak_bad: linear ramp for periodic peak:
///       peak <= peak_good => anti_period=1
///       peak >= peak_bad  => anti_period=0
///       else linear in between
///
/// eps: small number for stability
pub fn periodicity_acf_01(
    x: &Tensor,         // [N, L]
    struct_lag_max: i64,
    lag_min: i64,
    lag_max: i64,
    struct0: f64,
    peak_good: f64,
    peak_bad: f64,
    eps: f64,
) -> Result<Tensor, TchError> {
    let sz = x.size();
    assert_eq!(sz.len(), 2, "expected [N, L], sz={sz:?}");
    let l = sz[1];
    assert!(l >= 4, "need L({l})>=4");
    assert!(struct_lag_max >= 1, "struct_lag_max({struct_lag_max}) must be >= 1");
    assert!(lag_min >= 1 && lag_max >= lag_min, "bad lag_min({lag_min})/lag_max({lag_max})");
    assert!(lag_max < l, "need lag_max({lag_max} < L{l}");
    assert!(lag_min > struct_lag_max, "choose lag_min({lag_min}) > struct_lag_max({struct_lag_max})");
    assert!(struct0 > 0.0, "struct0({struct0}) must be > 0");
    assert!(peak_bad > peak_good, "need peak_bad({peak_bad}) > peak_good ({peak_good})");

    // consensus median across slices: c [L]
    let c = f_quantile_dim0(x, 0.50)?; // [L]

    // center: c0 = c - mean(c)
    let mean = c.f_mean_dim(0, false, x.kind())?; // scalar
    let c0 = c.f_sub(&mean)?;                        // [L]

    // energy = sum c0^2
    let energy = c0
        .f_mul(&c0)?
        .f_sum(x.kind())?
        .f_add_scalar(eps)?; // scalar

    // rho_raw[tau] for tau=1..=lag_max  (NO abs here)
    let mut rhos: Vec<Tensor> = Vec::with_capacity(lag_max as usize);
    for tau in 1..=lag_max {
        let len = l - tau;
        let a = c0.f_narrow(0, 0, len)?;
        let b = c0.f_narrow(0, tau, len)?;
        let dot = a.f_mul(&b)?.f_sum(x.kind())?; // scalar
        let rho = dot.f_div(&energy)?;           // scalar (signed)
        rhos.push(rho);
    }
    let rho_raw = Tensor::stack(&rhos, 0); // [lag_max], index i => tau=i+1

    // -------- (1) structure (anti-noise): mean |rho| at small lags
    let rho_abs = rho_raw.f_abs()?;
    let s_slice = rho_abs.f_narrow(0, 0, struct_lag_max)?; // tau=1..struct_lag_max
    let s_mean = s_slice.f_sum(x.kind())?.f_div_scalar(struct_lag_max as f64)?;

    // structure01 = s / (s + struct0)
    let structure01 = {
        let denom = s_mean.f_add_scalar(struct0)?.f_add_scalar(eps)?;
        s_mean.f_div(&denom)?
    };

    // -------- (2) periodic peak: max of *positive* local maxima of rho_raw in [lag_min..lag_max]
    // Local max at tau means: rho[tau] > rho[tau-1] AND rho[tau] >= rho[tau+1] AND rho[tau] > 0
    // We'll build peaks_full[tau] = rho[tau] if local max else 0, for tau=1..lag_max.

    let lag_max_usize = lag_max as i64;

    // interior maxima for tau=2..lag_max-1 (requires lag_max>=3)
    let one_slot = Tensor::zeros(&[1], (x.kind(), x.device()));
    let last_slot = {
        // boundary tau=lag_max: accept if rho[lag_max] > rho[lag_max-1] and >0
        if lag_max >= 2 {
            let last = rho_raw.f_select(0, lag_max - 1)?;     // scalar
            let prev = rho_raw.f_select(0, lag_max - 2)?;     // scalar
            let m = last.f_gt_tensor(&prev)?
                .f_bitwise_and_tensor(&last.f_gt(0.0)?)?
                .f_to_kind(x.kind())?;
            // make it shape [1]
            last.f_mul(&m)?.f_unsqueeze(0)?
        } else {
            one_slot.shallow_clone()
        }
    };

    let peaks_full = if lag_max >= 3 {
        let r_prev = rho_raw.f_narrow(0, 0, lag_max - 2)?; // tau=1..lag_max-2
        let r_mid  = rho_raw.f_narrow(0, 1, lag_max - 2)?; // tau=2..lag_max-1
        let r_next = rho_raw.f_narrow(0, 2, lag_max - 2)?; // tau=3..lag_max

        let m1 = r_mid.f_gt_tensor(&r_prev)?;
        let m2 = r_mid.f_ge_tensor(&r_next)?;
        let m3 = r_mid.f_gt(0.0)?;
        let mask = m1.f_bitwise_and_tensor(&m2)?.f_bitwise_and_tensor(&m3)?.f_to_kind(x.kind())?;

        let peaks_mid = r_mid.f_mul(&mask)?; // [lag_max-2], aligned to tau=2..lag_max-1

        // peaks_full: tau=1 slot is 0, tau=2..lag_max-1 from peaks_mid, tau=lag_max from last_slot
        Tensor::cat(&[&one_slot, &peaks_mid, &last_slot], 0)
    } else {
        // lag_max==1 or 2: no interior maxima; just [0, last_slot?]
        if lag_max == 2 {
            Tensor::cat(&[&one_slot, &last_slot], 0)
        } else {
            one_slot.shallow_clone() // lag_max==1 => [0]
        }
    };
    // peaks_full has shape [lag_max] (same as rho_raw)

    // slice [lag_min..lag_max] inclusive (indices lag_min-1 .. lag_max-1)
    let start = lag_min - 1;
    let len = lag_max - lag_min + 1;
    let p_slice = peaks_full.f_narrow(0, start, len)?;

    // peak = max local-max value in window; if none, this will be 0
    let (peak, _idx) = p_slice.f_max_dim(0, false)?; // scalar

    // anti_period01: 1 when peak <= peak_good, 0 when peak >= peak_bad, linear in between.
    let anti_period01 = {
        let one = scalar_like(x, 1.0);
        let zero = scalar_like(x, 0.0);
        let pg = scalar_like(x, peak_good);
        let pb = scalar_like(x, peak_bad);

        let m_good = peak.f_le_tensor(&pg)?.f_to_kind(x.kind())?;
        let m_bad  = peak.f_ge_tensor(&pb)?.f_to_kind(x.kind())?;
        let m_mid  = one.f_sub(&m_good)?.f_sub(&m_bad)?;

        let mid = pb.f_sub(&peak)?.f_div(&pb.f_sub(&pg)?)?;

        m_good.f_mul(&one)?
            .f_add(&m_mid.f_mul(&mid)?)?
            .f_maximum(&zero)?
            .f_minimum(&one)?
    };

    let score = structure01.f_mul(&anti_period01)?;
    let zero = scalar_like(x, 0.0);
    let one = scalar_like(x, 1.0);
    score.f_maximum(&zero)?.f_minimum(&one)
}
#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};
    use tracing::debug;
    use thoosa_common::transforms::utils::sin_t;
    use thoosa_common::utils;
    use thoosa_shared::ThoError;
    use crate::criteria::{interesting_turning_points_ratio, interesting_turning_points_ratio_trapezoid_01, periodicity_acf_01};

    #[cfg(test)]
    mod tests {
        use super::*;
        use tch::{TchError, Tensor};
        use thoosa_common::utils::debug_plot_impl;
        use thoosa_shared::ThoError;
        use crate::criteria::{coherence, inlier_ratio, interesting_turning_points_ratio_peaked_01, norm_thickness};

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

            let s = norm_thickness(&"identical", &x, 1e-8)?;
            assert_close(f64_of_scalar(&s), 1.0, 1e-12);
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
            // s = thickness / amp = 0.1/2 = 0.05
            // with k=5 => score = 1 / (1 + 5*s) = 1 / 1.25 = 0.8
            let x = Tensor::from_slice(&[
                0.0f32, 1.0, 2.0, 1.0,
                0.0,    1.1, 1.9, 1.0,
                0.0,    0.9, 2.1, 1.0,
            ])
            .view([3, 4]);

            let s = norm_thickness(&"known_value_n3_rounding", &x, 1e-8)?;
            assert_close(f64_of_scalar(&s), 0.952381014824, 1e-6);
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

        #[test]
        fn test_interesting_turning_points_ratio() -> Result<(), ThoError>{
            let t = Tensor::arange(1000, (Kind::Float, Device::Cpu)) / 200;
            let t_sin = sin_t(&t, 1.).view((1, -1));

            debug_plot_impl(&t_sin, "sinp.png", 0)?;

            let r = interesting_turning_points_ratio(&t_sin)?.double_value(&[]);

            // debug!("r=\n{r}");
            assert_close(r, 0.01, 0.01);

            Ok(())
        }

        #[test]
        fn test_interesting_turning_points_ratio_peaked() -> Result<(), ThoError>{
            let t = Tensor::arange(1000, (Kind::Float, Device::Cpu)) / 200;
            let t_sin = sin_t(&t, 1.).view((1, -1));

            debug_plot_impl(&t_sin, "sinp.png", 0)?;

            let r = interesting_turning_points_ratio_peaked_01(&t_sin, 1)?.double_value(&[]);
            assert_close(r, 0., 0.01);

            let r = interesting_turning_points_ratio_peaked_01(&t_sin, 10)?.double_value(&[]);
            assert_close(r, 1., 0.01);

            let r = interesting_turning_points_ratio_peaked_01(&t_sin, 100)?.double_value(&[]);
            assert_close(r, 0.1, 0.01);

            Ok(())
        }


    }


    fn f64_of_scalar(t: &Tensor) -> f64 {
        assert!(t.size().is_empty(), "expected scalar, got {:?}", t.size());
        t.double_value(&[])
    }

    fn assert_close(got: f64, expected: f64, tol: f64) {
        let diff = (got - expected).abs();
        assert!(
            diff <= tol,
            "got={got:.12}, expected={expected:.12}, diff={diff:.12}, tol={tol}"
        );
    }

    fn make_batch_from_consensus(c: &[f32], n: i64) -> Tensor {
        // x is [N, L] with all rows identical => median consensus == c exactly
        let l = c.len() as i64;
        let mut data = Vec::with_capacity((n * l) as usize);
        for _ in 0..n {
            data.extend_from_slice(c);
        }
        Tensor::from_slice(&data).view([n, l])
    }

    fn make_entropy_signal(width: u32, periods: u32) -> Tensor {
        let wave_length = width/periods;
        let t = Tensor::arange(width as i64, (Kind::Float, Device::Cpu)) / wave_length as i64;
        (sin_t(&t, 1.) + sin_t(&t, 3.) * 0.35 + sin_t(&t, 7.) * 0.15).view((1, -1))
    }

    #[test]
    fn test_interesting_entropy_gauss_01_matches_spectral_entropy_gaussian() -> Result<(), ThoError> {
        for i in 1..=8 {
            let signal = make_entropy_signal(2048, i);//3 frequencies
            // thoosa_common::utils::debug_plot_impl(&signal, &format!("entropy_gauss_signal_{i}"), 0)?;

            let eps = 1e-8;
            let entropy = super::interesting_spectral_entropy(&signal, eps)?;
            let h = f64_of_scalar(&entropy);
            assert!((0.0..=1.0).contains(&h), "expected normalized entropy, got {h}");

            let h0 = if h <= 0.5 { h + 0.17 } else { h - 0.17 };
            let sigma = 0.2;
            let score = super::interesting_entropy_gauss_01(&signal, h0, sigma, eps)?;
            let score_f64 = f64_of_scalar(&score); //it's always ~0.6 - sig

            debug!("i={i} score_f64={score_f64} h={h:?}");

            let z = (h - h0) / sigma;
            let expected = (-0.5 * z * z).exp();
            assert_close(score_f64, expected, 1e-6);
        }

        Ok(())
    }

    #[test]
    fn test_interesting_entropy_band_01_uses_spectral_entropy_and_sorts_bounds() -> Result<(), ThoError> {
        for i in 1..=8 {
            let signal = make_entropy_signal(2048, i);
            // thoosa_common::utils::debug_plot_impl(&signal, "entropy_band_signal.png", 0)?;

            let eps = 1e-8;
            let entropy = super::interesting_spectral_entropy(&signal, eps)?;
            let h = f64_of_scalar(&entropy);
            assert!((0.0..=1.0).contains(&h), "expected normalized entropy, got {h}");

            let inside = super::interesting_entropy_band_01(&signal, h + 0.01, h - 0.01, eps)?;
            let score_f64_inside = f64_of_scalar(&inside);
            assert_close(score_f64_inside, 1.0, 1e-12);

            let (outside_lo, outside_hi) = if h <= 0.5 { (h + 0.02, h + 0.10) } else { (h - 0.10, h - 0.02) };
            let outside = super::interesting_entropy_band_01(&signal, outside_lo, outside_hi, eps)?;
            let score_f64_outside = f64_of_scalar(&outside);
            assert_close(score_f64_outside, 0.0, 1e-12);

            debug!("i={i} score_f64_inside={score_f64_inside} score_f64_outside={score_f64_outside} h={h:?}");//always 1.0 - 0.0
        }
        Ok(())
    }

    #[test]
    fn test_interesting_turning_points_ratio_trapezoid_01() -> Result<(), ThoError> {
        let (a, b, c, d) = (2, 3, 5, 7);

        // Case 1: T=2 turns => score 0 (T <= a)
        // consensus: 0,1,0,1 => diffs: +,-,+ => 2 sign changes
        let x2 = make_batch_from_consensus(&[0.0, 1.0, 0.0, 1.0], 3);
        let s2 = interesting_turning_points_ratio_trapezoid_01(&x2, a, b, c, d)?;
        assert_close(f64_of_scalar(&s2), 0.0, 1e-6);

        // Case 2: T=4 turns => score 1 (b <= T <= c)
        // consensus: 0,1,0,1,0,1 => diffs: +,-,+,-,+ => 4 sign changes
        let x4 = make_batch_from_consensus(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 3);
        let s4 = interesting_turning_points_ratio_trapezoid_01(&x4, a, b, c, d)?;
        assert_close(f64_of_scalar(&s4), 1.0, 1e-6);

        // Case 3: T=6 turns => score (d - T)/(d - c) = (7-6)/(7-5) = 0.5
        // consensus: 0,1,0,1,0,1,0,1 => diffs alternating => 6 sign changes
        let x6 = make_batch_from_consensus(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], 3);
        let s6 = interesting_turning_points_ratio_trapezoid_01(&x6, a, b, c, d)?;
        assert_close(f64_of_scalar(&s6), 0.5, 1e-6);

        Ok(())
    }


    #[test]
    fn test_periodicity_acf_01_varies_with_periods_and_penalizes_noise() -> Result<(), ThoError> {
        let width: i64 = 2048;

        // Parameters: tuned for "reject repeating sub-patterns" but keep single global structure.
        let struct_lag_max = 32;
        let lag_min = 64;
        let lag_max = 1024;
        let struct0 = 0.20;
        let peak_good = 0.25;
        let peak_bad  = 0.65;
        let eps = 1e-8;

        // Evaluate periodic signals with different repeats.
        let mut scores = Vec::new();
        for periods in 1..=8 {
            let x = utils::make_periodic_signal(width, periods);
            // thoosa_common::utils::debug_plot_impl(&x, &format!("periodic_signal_{periods}"), 0)?;

            let s = periodicity_acf_01(
                &x,
                struct_lag_max,
                lag_min,
                lag_max,
                struct0,
                peak_good,
                peak_bad,
                eps,
            )?;
            let sf = f64_of_scalar(&s);
            debug!("test_periodicity_acf_01_varies_with_periods_and_penalizes_noise: periods={periods} sf={sf}");
            assert!((0.0..=1.0).contains(&sf), "score must be in [0,1], got {sf}");
            if periods == 1 {
                assert!((0.8..=1.0).contains(&sf), "score for aperiodic signal must be in [0.8,1], got {sf}");
            }
            if periods == 8 {
                assert!((0.0..=0.1).contains(&sf), "score for periodic signal must be close to 0, got {sf}");
            }
            scores.push(sf);
        }

        // Should not be (almost) constant.
        let min_s = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            (max_s - min_s) > 0.20,
            "expected noticeable spread across periods, got min={min_s:.4} max={max_s:.4} scores={scores:?}"
        );

        // Stronger repetition should look "more periodic" => lower score.
        assert!(
            scores[0] > scores[7],
            "expected periods=1 score > periods=8 score, got s1={:.4}, s8={:.4}, scores={scores:?}",
            scores[0],
            scores[7]
        );

        // Noise should score low (unstructured).
        tch::manual_seed(0);
        let noise = Tensor::randn(&[1, width], (Kind::Float, Device::Cpu));
        let s_noise = periodicity_acf_01(
            &noise,
            struct_lag_max,
            lag_min,
            lag_max,
            struct0,
            peak_good,
            peak_bad,
            eps,
        )?;
        let sn = f64_of_scalar(&s_noise);
        assert!(
            sn < 0.25,
            "expected noise to get low score, got {sn:.4}"
        );

        Ok(())
    }
}
