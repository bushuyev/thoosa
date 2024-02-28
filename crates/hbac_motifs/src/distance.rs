
use tch::Tensor;
use tch::Kind;
use hbac_shared::HbacError;
use hbac_common::utils::z_norm_2;

pub fn z_normalized_euclidean(t_0:&Tensor, t_1:&Tensor, dim:i64) -> Result<Tensor, HbacError>{
    t_0.mean_dim(1, true, Kind::Float).print();
    t_0.std_dim(1, false, true).print();
    // z_norm_2(&t_0.copy(), 1)?.print();
    Ok(
        (
            z_norm_2(t_0, 1)? - z_norm_2(t_1, 1)?
        )
            .pow_tensor_scalar(2)
            .sum_dim_intlist(dim, true, Kind::Float)
            .pow_tensor_scalar(0.5)
    )
}

#[cfg(test)]
mod distance_tests {
    use tch::Tensor;
    use hbac_shared::HbacError;
    use crate::distance::z_normalized_euclidean;

    #[test]
    fn test_z_normalized_euclidean() -> Result<(), HbacError> {
        let d = z_normalized_euclidean(
            &Tensor::vstack(&[
                Tensor::from_slice(&[1.,  0., 1.,  0.,  1., ]),
                Tensor::from_slice(&[10., 0., 10., 0., 10., ]),
            ]),
            &Tensor::vstack(&[
                Tensor::from_slice(&[0., 1.,  0.,  1.,  0., ]),
                Tensor::from_slice(&[0.,  10., 0., 10., 0., ]),
            ]),
            1,
        )?;

        d.print();

        //mean - 0.6, std -
        //(1-2.2)/0.54772255750517 = −2,19089023



        Ok(())
    }
}