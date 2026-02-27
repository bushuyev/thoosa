use crate::ThoError;

/// Simple Gauss-Jordan inversion for small matrices.
/// Returns None if singular / numerically unstable.
pub fn invert_square(mat: &Vec<Vec<f64>>) -> Option<Vec<Vec<f64>>> {
    let n = mat.len();
    if n == 0 || mat.iter().any(|row| row.len() != n) {
        return None;
    }

    let mut a = mat.clone();
    let mut inv = vec![vec![0.0f64; n]; n];
    for i in 0..n {
        inv[i][i] = 1.0;
    }

    for col in 0..n {
        // Find pivot
        let mut pivot = col;
        let mut best = a[col][col].abs();
        for r in (col + 1)..n {
            let v = a[r][col].abs();
            if v > best {
                best = v;
                pivot = r;
            }
        }
        if best < 1e-12 {
            return None;
        }
        if pivot != col {
            a.swap(pivot, col);
            inv.swap(pivot, col);
        }

        // Normalize pivot row
        let diag = a[col][col];
        for c in 0..n {
            a[col][c] /= diag;
            inv[col][c] /= diag;
        }

        // Eliminate other rows
        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = a[r][col];
            if factor == 0.0 {
                continue;
            }
            for c in 0..n {
                a[r][c] -= factor * a[col][c];
                inv[r][c] -= factor * inv[col][c];
            }
        }
    }

    Some(inv)
}



/// Compute Savitzky–Golay smoothing (derivative=0) convolution coefficients.
/// Window length must be odd. poly_order must be < window_len.
/// Returns coefficients in left-to-right order over the window.
pub fn sg_smoothing_coeffs(window_len: usize, poly_order: usize) -> Result<Vec<f64>, ThoError> {
    if window_len < 3 || window_len % 2 == 0 {
        return Err(ThoError::new("window_len must be odd and >= 3"));
    }
    if poly_order == 0 {
        // With poly=0, SG is just a moving average.
        return Ok(vec![1.0 / window_len as f64; window_len]);
    }
    if poly_order >= window_len {
        return Err(ThoError::new("poly_order must be < window_len"));
    }

    let m = (window_len / 2) as i32; // window positions: -m..m
    let p = poly_order + 1;

    // Build A: [window_len x p], A[i,k] = x_i^k
    let mut a = vec![vec![0.0f64; p]; window_len];
    for i in 0..window_len {
        let x = (i as i32 - m) as f64;
        let mut xk = 1.0;
        for k in 0..p {
            a[i][k] = xk;
            xk *= x;
        }
    }

    // Compute ATA = A^T A  [p x p]
    let mut ata = vec![vec![0.0f64; p]; p];
    for r in 0..p {
        for c in 0..p {
            let mut s = 0.0;
            for i in 0..window_len {
                s += a[i][r] * a[i][c];
            }
            ata[r][c] = s;
        }
    }

    // Invert ATA via Gauss-Jordan: inv = (ATA)^-1  [p x p]
    let inv = invert_square(&ata).ok_or(ThoError::new("failed to invert (A^T A)"))?;

    // Pseudoinverse B = (A^T A)^-1 A^T  =>  [p x window_len]
    // We only need the first row (corresponding to value at x=0), i.e. row 0 of B:
    // coeffs[i] = sum_r inv[0,r] * A[i,r]
    let mut coeffs = vec![0.0f64; window_len];
    for i in 0..window_len {
        let mut s = 0.0;
        for r in 0..p {
            s += inv[0][r] * a[i][r];
        }
        coeffs[i] = s;
    }

    Ok(coeffs)
}