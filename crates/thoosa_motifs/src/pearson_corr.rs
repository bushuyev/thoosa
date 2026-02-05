use tch::{Kind, Tensor, TchError};

/// Pairwise Pearson correlation distance in **[0,2]** range for each channel:
///   a: [C, Sa, W]
///   b: [C, Sb, W]
/// returns: [C, Sa, Sb]
///
/// d = 1 - r, where r is Pearson correlation along W for each (slice_i, slice_j).
pub fn pearson_corr_distance_matrix_0_2(a: &Tensor, b: &Tensor, eps: f64) -> Result<Tensor, TchError> {
    let sa = a.size();
    let sb = b.size();
    debug_assert_eq!(sa.len(), 3, "a must be [C, Sa, W]");
    debug_assert_eq!(sb.len(), 3, "b must be [C, Sb, W]");
    debug_assert_eq!(sa[0], sb[0], "channels mismatch");
    debug_assert_eq!(sa[2], sb[2], "window mismatch");

    // let device = a.device();
    // let a = a.to_device(device).to_kind(Kind::Float);
    // let b = b.to_device(device).to_kind(Kind::Float);

    let w = sa[2] as f64;

    // Means over window dim (dim=2): [C, Sa, 1], [C, Sb, 1]
    let a_mean = a.f_sum_dim_intlist(2, true, Kind::Float)?.f_divide_scalar(w)?;
    let b_mean = b.f_sum_dim_intlist(2, true, Kind::Float)?.f_divide_scalar(w)?;

    // Centered
    let ac = a.f_subtract(&a_mean)?;
    let bc = b.f_subtract(&b_mean)?;

    // Numerator: batched (ac @ bc^T) over window dimension:
    // ac: [C, Sa, W]
    // bc^T: [C, W, Sb]
    // num: [C, Sa, Sb]
    let bc_t = bc.f_transpose(1, 2)?;
    let num = ac.f_bmm(&bc_t)?;

    // Denominator: ||ac|| * ||bc|| (broadcast to [C, Sa, Sb])
    // a_ss: [C, Sa], b_ss: [C, Sb]
    let a_ss = ac.f_pow_tensor_scalar(2)?.f_sum_dim_intlist(2, false, Kind::Float)?;
    let b_ss = bc.f_pow_tensor_scalar(2)?.f_sum_dim_intlist(2, false, Kind::Float)?;

    let a_norm = a_ss.f_sqrt()?; // [C, Sa]
    let b_norm = b_ss.f_sqrt()?; // [C, Sb]

    let denom = a_norm
        .f_unsqueeze(2)?                 // [C, Sa, 1]
        .f_multiply(&b_norm.f_unsqueeze(1)?)? // [C, Sa, Sb]
        .f_add_scalar(eps)?;             // avoid /0

    let r = num.f_divide(&denom)?;      // [C, Sa, Sb]
    let r = r.clamp(-1.0, 1.0);         // numeric safety; keeps distance in [0,2]

    // distance = 1 - r
    let ones = Tensor::ones(&r.size(), (r.kind(), r.device()));
    let dist = ones.f_subtract(&r)?;

    Ok(dist)
}


/// Pairwise Pearson correlation distance in **[0,2]** range among slices of a single motif:
///   x: [Sa, W]
/// returns:
///   [Sa, Sa]
///
/// d = 1 - r, where r is Pearson correlation along W for each (slice_i, slice_j).
pub fn pearson_corr_distance_matrix_0_2_single(x: &Tensor, eps: f64) -> Result<Tensor, TchError> {
    let sx = x.size();
    debug_assert_eq!(sx.len(), 2, "x must be [Sa, W]");

    let device = x.device();
    // let x = x.to_device(device).to_kind(Kind::Float);

    let w = sx[1] as f64;

    // mean over W, keepdim: [Sa, 1]
    let mean = x.f_sum_dim_intlist(1, true, Kind::Float)?.f_divide_scalar(w)?;

    // centered: [Sa, W]
    let xc = x.f_subtract(&mean)?;

    // numerator: xc @ xc^T  => [Sa, Sa]
    let num = xc.f_mm(&xc.f_transpose(0, 1)?)?;

    // norms: sqrt(sum(xc^2)) => [Sa]
    let ss = xc.f_pow_tensor_scalar(2)?.f_sum_dim_intlist(1, false, Kind::Float)?;
    let n = ss.f_sqrt()?; // [Sa]

    // denom: n[:,None] * n[None,:] + eps => [Sa, Sa]
    let denom = n
        .f_unsqueeze(1)?                 // [Sa, 1]
        .f_multiply(&n.f_unsqueeze(0)?)?  // [Sa, Sa]
        .f_add_scalar(eps)?;

    let r = num.f_divide(&denom)?;       // [Sa, Sa]
    let r = r.clamp(-1.0, 1.0);          // numeric safety

    // distance = 1 - r  (range [0,2])
    let ones = Tensor::ones(&r.size(), (r.kind(), r.device()));
    let dist = ones.f_subtract(&r)?;

    Ok(dist)
}

#[cfg(test)]
mod tests {
    use log::debug;
    use tch::Kind::Float;
    use super::*;
    use tch::Tensor;
    use thoosa_common::utils::to_vec_f32;

    #[test]
    fn pairwise_matrix_gives_expected_values() -> Result<(), TchError> {
        // C=1, Sa=2, Sb=2, W=3
        //
        // slice0 = [1,2,3]
        // slice1 = [3,2,1]  => perfectly negatively correlated with slice0
        let a = Tensor::from_slice(&[
            1f32, 2., 3.,   // a0
            3f32, 2., 1.,   // a1
        ]).reshape(&[1, 2, 3]);

        let b = Tensor::from_slice(&[
            1f32, 2., 3.,   // b0
            3f32, 2., 1.,   // b1
        ]).reshape(&[1, 2, 3]);

        let d = pearson_corr_distance_matrix_0_2(&a, &b, 1e-12)?;
        assert_eq!(d.size(), vec![1, 2, 2]);

        // flatten order: [C=0][i][j] row-major => i=0,j=0 ; i=0,j=1 ; i=1,j=0 ; i=1,j=1
        let out = to_vec_f32(&d);

        // a0 vs b0 => r=1 => d=0
        assert!((out[0] - 0.0).abs() < 1e-5, "a0-b0 got {}", out[0]);
        // a0 vs b1 => r=-1 => d=2
        assert!((out[1] - 2.0).abs() < 1e-5, "a0-b1 got {}", out[1]);
        // a1 vs b0 => r=-1 => d=2
        assert!((out[2] - 2.0).abs() < 1e-5, "a1-b0 got {}", out[2]);
        // a1 vs b1 => r=1 => d=0
        assert!((out[3] - 0.0).abs() < 1e-5, "a1-b1 got {}", out[3]);

        Ok(())
    }

    #[test]
    fn single_motif_pairwise_matrix_basic() -> Result<(), TchError> {
        // Sa=2, W=3
        // slice0=[1,2,3], slice1=[3,2,1] => perfectly negative corr => dist=2 off-diagonal
        let x = Tensor::from_slice(&[
            1f32, 2., 3.,
            3f32, 2., 1.,
        ])
            .reshape(&[2, 3]);

        let d = pearson_corr_distance_matrix_0_2_single(&x, 1e-12)?;
        assert_eq!(d.size(), vec![2, 2]);

        let out = to_vec_f32(&d);
        // row-major: [0,0],[0,1],[1,0],[1,1]
        assert!((out[0] - 0.0).abs() < 1e-5, "d00 got {}", out[0]);
        assert!((out[3] - 0.0).abs() < 1e-5, "d11 got {}", out[3]);
        assert!((out[1] - 2.0).abs() < 1e-5, "d01 got {}", out[1]);
        assert!((out[2] - 2.0).abs() < 1e-5, "d10 got {}", out[2]);

        Ok(())
    }

    #[test]
    fn constant_slice_gives_about_one_against_anything() -> Result<(), TchError> {
        // slice0 varies, slice1 is constant => centered slice1 is zeros => r≈0 => d≈1
        let x = Tensor::from_slice(&[
            1f32, 2., 3.,
            7f32, 7., 7.,
        ])
            .reshape(&[2, 3]);

        let d = pearson_corr_distance_matrix_0_2_single(&x, 1e-6)?;
        let maan_corr = d.mean_dim(&[0,1][..], false, Float);
        debug!("maan_corr={:?}", maan_corr.double_value(&[]));
        let out = to_vec_f32(&d);

        // diagonal for constant slice: undefined correlation, but with eps our formula yields d=1 on diag too.
        // We mainly check the cross terms are ~1.
        assert!((out[1] - 1.0).abs() < 1e-3, "d01 got {}", out[1]);
        assert!((out[2] - 1.0).abs() < 1e-3, "d10 got {}", out[2]);

        Ok(())
    }
}
