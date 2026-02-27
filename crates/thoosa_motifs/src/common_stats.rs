use tch::{Device, Kind, Tensor, TchError};

#[derive(Debug, Clone)]
pub struct Stats {
    // Similarity of random windows to the channel template (mean window).
    // euc_dist: average per-window RMS distance to template (>=0, smaller means "more similar").
    // pearson: average per-window Pearson correlation with template (in [-1,1], larger means "more similar").
    pub euc_dist: f32,
    pub pearson: f32,

    // “Half-saturation” scales for mapping raw interestingness into [0,1]
    // via q = s / (s + s0). Here s0 is chosen as a quantile of the per-window distribution.
    pub a0: f32,
    pub tv0: f32,
    pub curv0: f32,

    // Percentiles of robust per-window amplitude A_i = q_hi(row)-q_lo(row)
    pub amp_p10: f32,
    pub amp_p50: f32,
    pub amp_p90: f32,

    // Useful extra percentiles (often handy for picking tv0/curv0 later)
    pub tvn_p10: f32,
    pub tvn_p50: f32,
    pub tvn_p90: f32,

    pub curvn_p10: f32,
    pub curvn_p50: f32,
    pub curvn_p90: f32,

    // Optional: sanity info
    pub m: i64,
    pub l: i64,
}

fn f64_scalar(t: &Tensor) -> Result<f64, TchError> {
    // Ensure on CPU for scalar extraction.
    let tc = t.f_to_device(Device::Cpu)?;
    Ok(tc.double_value(&[]))
}

/// Quantile for 1D tensor v over dim=0 (v is [M]).
fn f_quantile_1d(v: &Tensor, q: f64) -> Result<Tensor, TchError> {
    let sz = v.size();
    if sz.len() != 1 {
        return Err(TchError::FileFormat(format!("expected [M], got {:?}", sz)));
    }
    let m = sz[0];
    if m < 1 {
        return Err(TchError::FileFormat("M must be >= 1".into()));
    }
    let (sorted, _idx) = v.f_sort(0, false)?;
    let mut i = ((m - 1) as f64 * q).round() as i64;
    if i < 0 { i = 0; }
    if i > m - 1 { i = m - 1; }
    sorted.f_select(0, i) // scalar
}

/// Per-row quantile for windows x: [M, L] along dim=1.
/// Returns [M].
fn f_quantile_dim1_per_row(x: &Tensor, q: f64) -> Result<Tensor, TchError> {
    let sz = x.size();
    if sz.len() != 2 {
        return Err(TchError::FileFormat(format!("expected [M,L], got {:?}", sz)));
    }
    let l = sz[1];
    if l < 1 {
        return Err(TchError::FileFormat("L must be >= 1".into()));
    }
    let (sorted, _idx) = x.f_sort(1, false)?; // [M,L]
    let mut i = ((l - 1) as f64 * q).round() as i64;
    if i < 0 { i = 0; }
    if i > l - 1 { i = l - 1; }
    sorted.f_select(1, i) // [M]
}

/// Compute per-window amplitude / TV / curvature (normalized) for windows [M,L].
/// Returns: (amp[M], tvn[M], curvn[M])
fn per_window_metrics(
    w: &Tensor, // [M,L]
    q_lo: f64,
    q_hi: f64,
    eps: f64,
) -> Result<(Tensor, Tensor, Tensor), TchError> {
    let sz = w.size();
    let m = sz[0];
    let l = sz[1];
    if l < 2 {
        return Err(TchError::FileFormat("need L>=2".into()));
    }

    // Robust amplitude per row: A_i = q_hi - q_lo
    let lo = f_quantile_dim1_per_row(w, q_lo)?;
    let hi = f_quantile_dim1_per_row(w, q_hi)?;
    let amp = hi.f_sub(&lo)?; // [M]

    // TV per row: sum |x[t+1]-x[t]|
    let x1 = w.f_narrow(1, 1, l - 1)?;
    let x0 = w.f_narrow(1, 0, l - 1)?;
    let tv = x1
        .f_sub(&x0)?
        .f_abs()?
        .f_sum_dim_intlist(1, false, w.kind())?; // [M]

    // Curvature per row: sum |x[t+1] - 2x[t] + x[t-1]|
    let curv = if l >= 3 {
        let xp = w.f_narrow(1, 2, l - 2)?;
        let xc = w.f_narrow(1, 1, l - 2)?;
        let xm = w.f_narrow(1, 0, l - 2)?;
        let two = Tensor::from(2.0).to_device(w.device()).to_kind(w.kind());
        xp.f_sub(&xc.f_mul(&two)?)?.f_add(&xm)?.f_abs()?
            .f_sum_dim_intlist(1, false, w.kind())? // [M]
    } else {
        Tensor::zeros(&[m], (w.kind(), w.device()))
    };

    let denom = amp.f_add_scalar(eps)?; // [M]
    let tvn = tv.f_div(&denom)?;        // [M]
    let curvn = curv.f_div(&denom)?;    // [M]
    Ok((amp, tvn, curvn))
}

/// Average RMS Euclidean distance of each window to template.
/// Returns scalar.
fn avg_rms_to_template(w: &Tensor, t: &Tensor) -> Result<Tensor, TchError> {
    // w: [M,L], t: [L]
    let sz = w.size();
    let l = sz[1];
    let diff = w.f_sub(t)?;                      // [M,L]
    let diff2 = diff.f_mul(&diff)?;              // [M,L]
    let mse = diff2.f_mean_dim(1, false, w.kind())?; // [M]
    let rms = mse.f_sqrt()?;                     // [M]
    rms.f_mean(w.kind())                         // scalar
}

/// Average Pearson correlation of each window with template.
/// Returns scalar.
fn avg_pearson_to_template(w: &Tensor, t: &Tensor, eps: f64) -> Result<Tensor, TchError> {
    // w: [M,L], t: [L]
    let t_mean = t.f_mean(w.kind())?;                  // scalar
    let t_c = t.f_sub(&t_mean)?;                       // [L]
    let t_var = t_c.f_mul(&t_c)?.f_mean(w.kind())?;    // scalar
    let t_std = t_var.f_add_scalar(eps)?.f_sqrt()?;    // scalar

    let w_mean = w.f_mean_dim(1, true, w.kind())?;  // [M,1]
    let w_c = w.f_sub(&w_mean)?;                       // [M,L]

    let cov = w_c.f_mul(&t_c)?.f_mean_dim(1, false, w.kind())?; // [M]
    let w_var = w_c.f_mul(&w_c)?.f_mean_dim(1, false, w.kind())?; // [M]
    let w_std = w_var.f_add_scalar(eps)?.f_sqrt()?;                 // [M]

    let denom = w_std.f_mul(&t_std)?.f_add_scalar(eps)?; // [M]
    let corr = cov.f_div(&denom)?;                        // [M]
    corr.f_mean(w.kind())                                 // scalar
}

/// Main function:
/// channels_windows: [C, M, L]
///
/// - Computes per-channel window template (mean over M)
/// - euc_dist: avg RMS distance to template
/// - pearson: avg correlation to template
/// - Computes per-window interestingness raw metrics (amp, tv_norm, curv_norm)
/// - Computes scales a0/tv0/curv0 as quantiles(scale_q) of those distributions
/// - Also returns p10/p50/p90 for amp/tvn/curvn
pub fn estimate_channel_stats_from_windows(
    channels_windows: &Tensor, // [C,M,L]
    q_lo: f64,                 // e.g. 0.05
    q_hi: f64,                 // e.g. 0.95
    scale_q: f64,              // e.g. 0.50 median, or 0.75/0.90 stricter
    eps: f64,
) -> Result<Vec<Stats>, TchError> {
    let sz = channels_windows.size();
    if sz.len() != 3 {
        return Err(TchError::FileFormat(format!("expected [C,M,L], got {:?}", sz)));
    }
    let c = sz[0];
    let m = sz[1];
    let l = sz[2];
    if m < 1 || l < 2 {
        return Err(TchError::FileFormat(format!("need M>=1 and L>=2, got M={m} L={l}")));
    }

    // For stats/scalar extraction, easiest is CPU float.
    let x = channels_windows.f_to_device(Device::Cpu)?.f_to_kind(Kind::Float)?;

    let mut out = Vec::with_capacity(c as usize);

    for ch in 0..c {
        let w = x.f_select(0, ch)?; // [M,L]

        // Template = mean over windows: [L]
        let t = w.f_mean_dim(0, false, w.kind())?; // [L]

        let euc = avg_rms_to_template(&w, &t)?;                 // scalar
        let pear = avg_pearson_to_template(&w, &t, eps)?;       // scalar

        let (amp, tvn, curvn) = per_window_metrics(&w, q_lo, q_hi, eps)?; // [M], [M], [M]

        // scales (quantile = scale_q)
        let a0 = f_quantile_1d(&amp, scale_q)?;
        let tv0 = f_quantile_1d(&tvn, scale_q)?;
        let curv0 = f_quantile_1d(&curvn, scale_q)?;

        // percentiles
        let amp_p10 = f_quantile_1d(&amp, 0.10)?;
        let amp_p50 = f_quantile_1d(&amp, 0.50)?;
        let amp_p90 = f_quantile_1d(&amp, 0.90)?;

        let tvn_p10 = f_quantile_1d(&tvn, 0.10)?;
        let tvn_p50 = f_quantile_1d(&tvn, 0.50)?;
        let tvn_p90 = f_quantile_1d(&tvn, 0.90)?;

        let curvn_p10 = f_quantile_1d(&curvn, 0.10)?;
        let curvn_p50 = f_quantile_1d(&curvn, 0.50)?;
        let curvn_p90 = f_quantile_1d(&curvn, 0.90)?;

        out.push(Stats {
            euc_dist: f64_scalar(&euc)? as f32,
            pearson:  f64_scalar(&pear)? as f32,

            a0:    f64_scalar(&a0)? as f32,
            tv0:   f64_scalar(&tv0)? as f32,
            curv0: f64_scalar(&curv0)? as f32,

            amp_p10: f64_scalar(&amp_p10)? as f32,
            amp_p50: f64_scalar(&amp_p50)? as f32,
            amp_p90: f64_scalar(&amp_p90)? as f32,

            tvn_p10: f64_scalar(&tvn_p10)? as f32,
            tvn_p50: f64_scalar(&tvn_p50)? as f32,
            tvn_p90: f64_scalar(&tvn_p90)? as f32,

            curvn_p10: f64_scalar(&curvn_p10)? as f32,
            curvn_p50: f64_scalar(&curvn_p50)? as f32,
            curvn_p90: f64_scalar(&curvn_p90)? as f32,

            m,
            l,
        });
    }

    Ok(out)
}