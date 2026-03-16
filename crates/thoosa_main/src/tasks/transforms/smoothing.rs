use tch::{Device, Kind, TchError, Tensor};
//TODO this doesn't work - signal is not changed
/// Rough heuristic:
/// convert a desired "number of local extrema per window" into lambda
/// for the discrete cubic smoothing spline / Whittaker smoother.
///
/// - `window`: local window size in samples
/// - `extrema_per_window`: rough number of local minima+maxima you want to preserve
///
/// Bigger lambda => smoother output.
/// This is only a heuristic, but it is often a useful starting point.
///
/// Derivation sketch:
/// for this smoother, attenuation starts when lambda * omega^4 ~ 1,
/// and if you want about `m` extrema per window W, then omega ~ pi * m / W,
/// so lambda ~ (W / (pi * m))^4.
pub fn spline_lambda_from_window_and_extrema(window: i64, extrema_per_window: f64) -> f64 {
    let w = window.max(1) as f64;
    let m = extrema_per_window.max(1e-6);
    (w / (std::f64::consts::PI * m)).powi(4)
}

/// Discrete cubic smoothing spline / Whittaker-Eilers smoother.
///
/// Accepts:
/// - [len]
/// - [channels, len]
///
/// Returns tensor with the same shape, device, and floating dtype.
///
/// Notes:
/// - Works on Float/Double inputs.
/// - For len < 3, returns a copy (second-difference penalty is not meaningful).
/// - This version is for preprocessing; gradients will NOT flow through it.
pub fn smoothing_spline_1d_channels(x: &Tensor, lambda: f64) -> Result<Tensor, TchError> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(TchError::Shape(format!(
            "lambda must be finite and >= 0, got {lambda}"
        )));
    }

    let orig_kind = x.kind();
    if orig_kind != Kind::Float && orig_kind != Kind::Double {
        return Err(TchError::Kind(format!(
            "expected Float or Double input, got {:?}",
            orig_kind
        )));
    }

    let orig_device = x.device();
    let orig_shape = x.size();

    let squeezed = match orig_shape.len() {
        1 => true,
        2 => false,
        other => {
            return Err(TchError::Shape(format!(
                "expected shape [len] or [channels, len], got {:?} (rank={other})",
                orig_shape
            )));
        }
    };

    let x2 = if squeezed {
        x.f_unsqueeze(0)?
    } else {
        x.shallow_clone()
    };

    let size = x2.size();
    let channels = size[0] as usize;
    let len = size[1] as usize;

    if len < 3 || lambda == 0.0 {
        return Ok(x.shallow_clone());
    }

    // Move to CPU + Double for stable solve.
    let x_cpu = x2
        .f_to_device(Device::Cpu)?
        .f_to_kind(Kind::Double)?
        .f_contiguous()?;

    let mut input = vec![0.0f64; channels * len];
    x_cpu.f_copy_data(&mut input, channels * len)?;

    let (off2, off1, diag) = make_spline_system_diagonals(len, lambda);

    let mut output = vec![0.0f64; channels * len];
    for ch in 0..channels {
        let start = ch * len;
        let end = start + len;
        let row_in = &input[start..end];
        let row_out = solve_symmetric_pentadiagonal(&off2, &off1, &diag, row_in)?;

        output[start..end].copy_from_slice(&row_out);
    }

    let y = Tensor::f_from_slice(&output)?
        .f_view([channels as i64, len as i64])?
        .f_to_kind(orig_kind)?
        .f_to_device(orig_device)?;

    if squeezed {
        y.f_squeeze_dim(0)
    } else {
        Ok(y)
    }
}

/// Build diagonals for:
///   A = I + lambda * D^T D
/// where D is the second-difference operator.
///
/// A is symmetric pentadiagonal.
///
/// Returns:
/// - off2: second off-diagonal, len n-2
/// - off1: first off-diagonal,  len n-1
/// - diag: main diagonal,       len n
fn make_spline_system_diagonals(n: usize, lambda: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut diag = vec![1.0f64; n];
    let mut off1 = vec![0.0f64; n.saturating_sub(1)];
    let mut off2 = vec![0.0f64; n.saturating_sub(2)];

    match n {
        0 => unreachable!("n comes from tensor length"),
        1 | 2 => {}
        3 => {
            // D^T D =
            // [ 1 -2  1]
            // [-2  4 -2]
            // [ 1 -2  1]
            diag[0] = 1.0 + lambda * 1.0;
            diag[1] = 1.0 + lambda * 4.0;
            diag[2] = 1.0 + lambda * 1.0;

            off1[0] = lambda * -2.0;
            off1[1] = lambda * -2.0;

            off2[0] = lambda * 1.0;
        }
        4 => {
            // D^T D =
            // [ 1 -2  1  0]
            // [-2  5 -4  1]
            // [ 1 -4  5 -2]
            // [ 0  1 -2  1]
            diag[0] = 1.0 + lambda * 1.0;
            diag[1] = 1.0 + lambda * 5.0;
            diag[2] = 1.0 + lambda * 5.0;
            diag[3] = 1.0 + lambda * 1.0;

            off1[0] = lambda * -2.0;
            off1[1] = lambda * -4.0;
            off1[2] = lambda * -2.0;

            off2[0] = lambda * 1.0;
            off2[1] = lambda * 1.0;
        }
        _ => {
            // General case:
            // diag: [1, 5, 6, ..., 6, 5, 1]
            // off1: [-2, -4, ..., -4, -2]
            // off2: [1, 1, ..., 1]
            diag[0] = 1.0 + lambda * 1.0;
            diag[1] = 1.0 + lambda * 5.0;
            for i in 2..(n - 2) {
                diag[i] = 1.0 + lambda * 6.0;
            }
            diag[n - 2] = 1.0 + lambda * 5.0;
            diag[n - 1] = 1.0 + lambda * 1.0;

            off1[0] = lambda * -2.0;
            for i in 1..(n - 2) {
                off1[i] = lambda * -4.0;
            }
            off1[n - 2] = lambda * -2.0;

            for v in &mut off2 {
                *v = lambda * 1.0;
            }
        }
    }

    (off2, off1, diag)
}

/// Solve symmetric pentadiagonal system:
///
/// off2[i] = A[i, i+2] = A[i+2, i]
/// off1[i] = A[i, i+1] = A[i+1, i]
/// diag[i] = A[i, i]
///
/// This is O(n) and much cheaper than dense inversion.
///
/// Returns x solving A x = rhs.
fn solve_symmetric_pentadiagonal(
    off2: &[f64],
    off1: &[f64],
    diag: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, TchError> {
    let n = diag.len();
    if rhs.len() != n {
        return Err(TchError::Shape(format!(
            "rhs len {} != diag len {}",
            rhs.len(),
            n
        )));
    }
    if off1.len() != n.saturating_sub(1) {
        return Err(TchError::Shape(format!(
            "off1 len {} != n-1 {}",
            off1.len(),
            n.saturating_sub(1)
        )));
    }
    if off2.len() != n.saturating_sub(2) {
        return Err(TchError::Shape(format!(
            "off2 len {} != n-2 {}",
            off2.len(),
            n.saturating_sub(2)
        )));
    }

    if n <= 2 {
        return Ok(rhs.to_vec());
    }

    let mut d = vec![0.0f64; n];
    let mut e = vec![0.0f64; n];
    let mut f = vec![0.0f64; n];

    const EPS: f64 = 1e-18;

    let c0 = diag[0];
    if !c0.is_finite() || c0.abs() < EPS {
        return Err(TchError::Shape("singular system at row 0".to_string()));
    }
    d[0] = off1[0] / c0;
    e[0] = if n > 2 { off2[0] / c0 } else { 0.0 };
    f[0] = rhs[0] / c0;

    let c1 = diag[1] - off1[0] * d[0];
    if !c1.is_finite() || c1.abs() < EPS {
        return Err(TchError::Shape("singular system at row 1".to_string()));
    }
    d[1] = if n > 2 {
        (off1[1] - off1[0] * e[0]) / c1
    } else {
        0.0
    };
    e[1] = if n > 3 { off2[1] / c1 } else { 0.0 };
    f[1] = (rhs[1] - off1[0] * f[0]) / c1;

    for i in 2..n {
        let a = off2[i - 2];
        let b = off1[i - 1];

        let b_eff = b - a * d[i - 2];
        let c = diag[i] - a * e[i - 2] - b_eff * d[i - 1];

        if !c.is_finite() || c.abs() < EPS {
            return Err(TchError::Shape(format!("singular system at row {i}")));
        }

        d[i] = if i < n - 1 {
            (off1[i] - b_eff * e[i - 1]) / c
        } else {
            0.0
        };

        e[i] = if i < n - 2 { off2[i] / c } else { 0.0 };

        f[i] = (rhs[i] - a * f[i - 2] - b_eff * f[i - 1]) / c;
    }

    let mut x = vec![0.0f64; n];
    x[n - 1] = f[n - 1];
    x[n - 2] = f[n - 2] - d[n - 2] * x[n - 1];

    for i in (0..=(n - 3)).rev() {
        x[i] = f[i] - d[i] * x[i + 1] - e[i] * x[i + 2];
    }

    Ok(x)
}

#[cfg(test)]
mod tests {
    use thoosa_common::utils::make_periodic_signal;
    use thoosa_shared::ThoError;
    use crate::tasks::transforms::smoothing::{smoothing_spline_1d_channels, spline_lambda_from_window_and_extrema};

    #[test]
    fn test_0() -> Result<(), ThoError> {
        for periods in 0..2 {
            let x = make_periodic_signal(1000, periods);

            let lambda = 0.5; //spline_lambda_from_window_and_extrema(x.size()[0], 1.0);
            let smooth_x = smoothing_spline_1d_channels(&x, lambda)?;


            thoosa_common::utils::debug_plot_impl(&x, &format!("x_{periods}"), 0)?;
            thoosa_common::utils::debug_plot_impl(&smooth_x, &format!("smooth_x{periods}"), 0)?;
        }

        Ok(())
    }

}