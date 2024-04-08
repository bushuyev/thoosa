
use tch::{IndexOp, Tensor};
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

pub fn dist_d2(dim: i64, slice_0: &Tensor, slice_1: &Tensor) -> Result<Tensor, HbacError> {
    //https://github.com/pytorch/pytorch/blob/96487d0d1f638ae9c665dcd3d4b79e7e2a976f59/aten/src/ATen/native/cuda/DistanceKernel.cu#L188
    //0 -use_mm_for_euclid_dist_if_necessary
    //1 -use_mm_for_euclid_dist
    //2 -donot_use_mm_for_euclid_dist
    assert_eq!(slice_0.size()[1], slice_1.size()[1]);
    Ok(Tensor::cdist(slice_0, slice_1, 2.0, 0))
}

pub fn slice_t(t_0: &Tensor, step: usize, window: i64) -> Tensor {

    Tensor::stack(&(0..=t_0.size()[1] - window).step_by(step).map(|i| t_0.i((.., i..i + window))).collect::<Vec<Tensor>>(), 0)
}


#[cfg(test)]
mod distance_tests {
    use futures::StreamExt;
    use log::debug;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::Tensor;
    use hbac_shared::HbacError;
    use crate::distance::z_normalized_euclidean;
    use tch::IndexOp;

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
        //TODO
        d.print();

        //mean - 0.6, std -
        //(1-2.2)/0.54772255750517 = −2,19089023



        Ok(())
    }

    #[test]
    fn test_dist_d2() -> Result<(), HbacError>{
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&[0.00, 0.01, 0.02, 0.02, 0.02]),
            // Tensor::from_slice(&[0.10, 0.10, 0.20, 0.20]),
        ]);

        let t_1 = Tensor::vstack(&[
            Tensor::from_slice(&[1.00, 1.01, 1.02]),
            // Tensor::from_slice(&[1.10, 1.10, 1.20]),
        ]);

        let t_2 = Tensor::vstack(&[
            Tensor::from_slice(&[2.00, 2.01, 2.02, 2.02]),
            // Tensor::from_slice(&[2.10, 2.10, 2.20]),
        ]);

        let t_3 = Tensor::vstack(&[
            Tensor::from_slice(&[3.00, 0.00, 0.01]),//[0,7] should be 0.0
            // Tensor::from_slice(&[3.10, 3.10, 3.20]),
        ]);

        let step = 1;
        let window = 2;
        println!("t_0.size()={:?}", t_0.size());
        let t_0_sliced = super::slice_t(&t_0, step, window);

        println!("t_0_sliced={:?}", t_0_sliced.size());
        // t_0_sliced.print();

        let spliced_t = [t_1, t_2, t_3].into_iter().map(|t| super::slice_t(&t, step, window)).collect::<Vec<Tensor>>();
        let sizes = spliced_t.iter().map(|t|t.size()[0]).collect::<Vec<_>>();

        let tensor_ids = vec!["111", "222", "333"];

        let tensor_ids = sizes.iter().zip(tensor_ids.into_iter()).flat_map(|(size, id)| std::iter::repeat(id).take(*size as usize)).collect::<Vec<&str>>();
        println!("sizes: {:?}", tensor_ids);
        let t_1_2_3_sliced = Tensor::vstack(&spliced_t);
        // t_1_2_3_sliced.print();

        println!("t_1_2_3_sliced={:?}", t_1_2_3_sliced.size());

        for c in  0..1 {
            let tensor = t_0_sliced.i((.., c, ..)).squeeze();
            let tensor1 = t_1_2_3_sliced.i((.., c, ..)).squeeze();
            println!("tensor={:?} tensor1.size()={:?}", tensor.size(), tensor1.size());
            let t = super::dist_d2(1, &tensor, &tensor1)?;
            t.print();
            
            println!("t={:?}", t.size());
            for i_0 in 0..t.size()[0] {
                 for i_1 in 0..t.size()[1] {
                     println!("t_1={:?} slice:{} v={}", tensor_ids[i_1 as usize], i_1 * window, t.double_value(&[i_0, i_1]));
                 }
            }

        }

        // let slice_0_b = Tensor::stack((0..3).map(|_|slice_0.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0);
        // let slice_1_b = Tensor::stack((0..3).map(|_|slice_1.shallow_clone()).collect::<Vec<Tensor>>().as_slice(), 0);
        //
        // let t = super::dist_d2(1, &slice_0_b, &slice_1_b)?;
        // println!("slice_0={:?}", slice_0.size());
        // println!("t={:?}", t.size());
        // t.print();
        //
        // println!("----------------------------------------");


        Ok(())
    }

    #[test]
    fn test_slice_t(){
        let tensor = super::slice_t(&Tensor::from_slice(&[1., 2., 3., 4., 5.]).view((1, -1)), 1, 2);

        assert_eq!(Tensor::from_slice(&[1., 2., 2., 3., 3., 4., 4., 5.,]).view((4, 1, 2)), tensor);
    }
}