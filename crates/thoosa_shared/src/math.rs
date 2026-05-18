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
    let inv = invert_square(&ata).ok_or_else(||ThoError::new("failed to invert (A^T A)"))?;

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

pub fn unordered_sets_of_size<E>(elements: &[E], m: usize) -> Vec<Vec<E>>
where
 E:Copy+Clone
{
    assert!(m > 0);

    fn backtrack<E:Copy+Clone>(
        elements: &[E],
        m: usize,
        start: usize,
        current: &mut Vec<E>,
        result: &mut Vec<Vec<E>>,
    ) {
        if current.len() == m {
            result.push(current.clone());
            return;
        }

        let remaining_needed = m - current.len();

        // Not enough elements left.
        if elements.len() - start < remaining_needed {
            return;
        }

        for i in start..=elements.len() - remaining_needed {
            current.push(elements[i]);
            backtrack(elements, m, i + 1, current, result);
            current.pop();
        }
    }

    if m > elements.len() {
        return vec![elements.to_vec()];
    }

    let mut result = Vec::new();
    let mut current = Vec::with_capacity(m);

    backtrack(elements, m, 0, &mut current, &mut result);

    result
}


#[cfg(test)]
mod tests {
    use crate::math::unordered_sets_of_size;

    #[test]
    fn generates_unordered_sets_of_size_2() {
        let elements = vec![0, 1, 2, 3];

        let result = unordered_sets_of_size(&elements, 2);

        assert_eq!(
            result,
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![1, 2],
                vec![1, 3],
                vec![2, 3],
            ]
        );
    }

    #[test]
    fn returns_empty_when_m_is_larger_than_input() {
        let elements = vec![0, 1, 2];

        let result = unordered_sets_of_size(&elements, 4);

        assert_eq!(
            result,
            vec![
                vec![0, 1, 2]
            ]
        )
    }


    #[test]
    fn size_equal_to_input_returns_whole_input() {
        let elements = vec![10, 20, 30];

        let result = unordered_sets_of_size(&elements, 3);

        assert_eq!(result, vec![vec![10, 20, 30]]);
    }

}