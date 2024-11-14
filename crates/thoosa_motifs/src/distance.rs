use log::{debug, trace};
use tch::{IndexOp, TchError, Tensor};
use tch::Kind;

use thoosa_common::utils::z_norm_2;
use thoosa_shared::ThoosaError;

#[allow(dead_code)]
pub fn z_normalized_euclidean(t_0:&Tensor, t_1:&Tensor, dim:i64) -> Result<Tensor, ThoosaError>{


    Ok(
        (
            z_norm_2(t_0, 1)? - z_norm_2(t_1, 1)?
        )
            .pow_tensor_scalar(2)
            .sum_dim_intlist(dim, true, Kind::Float)
            .pow_tensor_scalar(0.5)
    )
}

pub fn dist_d2(slice_0: &Tensor, slice_1: &Tensor) -> Result<Tensor, ThoosaError> {
    //https://github.com/pytorch/pytorch/blob/96487d0d1f638ae9c665dcd3d4b79e7e2a976f59/aten/src/ATen/native/cuda/DistanceKernel.cu#L188
    //0 -use_mm_for_euclid_dist_if_necessary
    //1 -use_mm_for_euclid_dist
    //2 -donot_use_mm_for_euclid_dist
    assert_eq!(slice_0.size()[1], slice_1.size()[1]);//same slice size
    Ok(Tensor::cdist(slice_0, slice_1, 2.0, 0))
}

pub fn dist_d2_0(slice_0: &Tensor, slice_1: &Tensor) -> Result<Tensor, ThoosaError> {
    let slice_0_expanded = slice_0.unsqueeze(2);     // channels x motifs x 1 x motif_len
    let slice_1_expanded = slice_1.unsqueeze(1);         // channels x 1 x num_slices x motif_len

    // debug!("dist_d2_0: slice_0_expanded={slice_0_expanded}");
    debug!("dist_d2_0: slice_0_expanded.size={:?}", slice_0_expanded.size());

    // debug!("dist_d2_0: slice_1_expanded={slice_1_expanded}");
    debug!("dist_d2_0: slice_1_expanded.size={:?}", slice_1_expanded.size());

    // Compute the Euclidean distances
    let mut diff = match slice_0_expanded.f_subtract(&slice_1_expanded) {
        Ok(v) => {
            v
        }
        Err(e) => {
            panic!("e={e:?} {}", e.to_string().contains("CUDA out of memory"));
        }
    };
    // debug!("diff={diff}");

    let squared_diff = diff.f_pow_(2)?;
    let squared_distances = squared_diff.sum_dim_intlist(3, false, Kind::Float);
    let distances = squared_distances.sqrt();

    Ok(distances)
}

pub fn slice_t(t_0: &Tensor, step: u32, window: u32) -> Tensor {
    debug!("slice_t:  t_0.size()={:?} step={step} window={window}", t_0.size());

    let tensors_vec = (0..=t_0.size()[1] - window as i64).step_by(step as usize).map(|i| t_0.i((.., i..i + window as i64))).collect::<Vec<Tensor>>();
    trace!("slice_t: tensors_vec={tensors_vec:?}");
    Tensor::stack(&tensors_vec, 0)
}


#[cfg(test)]
mod distance_tests {
    use log::debug;
    use sci_rs::na::SimdBool;
    use tch::Device::Cpu;
    use tch::IndexOp;
    use tch::Kind::Float;
    use tch::Tensor;

    use thoosa_shared::ThoosaError;

    use crate::distance::{dist_d2_0, slice_t, z_normalized_euclidean};
    use crate::find::ChunkOfSlicedTensors;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::init_test_logging();
    }

    #[test]
    fn test_z_normalized_euclidean() -> Result<(), ThoosaError> {
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

        //TODO?
        assert!(
            Tensor::from_slice(&[4., 4.,]).totype(Float).allclose(&d, 0.01, 0.01, false)
        );

        Ok(())
    }
    #[test]
    fn test_0(){
        let t_0 = Tensor::from_slice(&[1,2]).view((1,2));
        let t_1 = Tensor::from_slice(&[3,4]).view((2,1));
        println!("t_0={t_0} .t_0 size={:?}", t_0.size());
        println!("t_1={t_1} .t_1 size={:?}", t_1.size());

        let d = t_1 - t_0;

        println!("d={d} size={:?}", d.size());

    }

    #[test]
    fn test_dist_d2_0_2ch() -> Result<(), ThoosaError>{
        let d_0_0 = [0.00f32, 0.01, 0.02, 0.03, 0.04, 0.05];
        let d_0_1 = [0.10f32, 0.11, 0.12, 0.13, 0.14, 0.15];

        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&d_0_0),
            Tensor::from_slice(&d_0_1),
        ]);



        let d_1_0 = [1.00f32, 1.01, 1.02, 1.03, 1.04, 1.05];
        let d_1_1 = [1.10f32, 1.11, 1.12, 1.13, 1.14, 1.15];



        let t_1 = Tensor::vstack(&[
            Tensor::from_slice(&d_1_0),
            Tensor::from_slice(&d_1_1),
        ]);


        let t_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(1, 3, &vec![t_0.into()], vec!["x"], Cpu).t.transpose(0,1);
        let t_1_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(1, 3, &vec![t_1.into()], vec!["x"], Cpu).t.transpose(0,1);

        let dists = dist_d2_0(&t_0_sliced, &t_1_sliced)?;

        // debug!("dists={dists}");


        assert_eq!(
            test_dist(&d_0_0[0..3], &d_1_0[0..3]),
            dists.i((0, 0, 0)).double_value(&[]) as f32
        );
        assert_eq!(
            test_dist(&d_0_0[0..3], &d_1_0[1..4]),
            dists.i((0, 0, 1)).double_value(&[]) as f32
        );
        assert_eq!(
            test_dist(&d_0_0[0..3], &d_1_0[2..5]),
            dists.i((0, 0, 2)).double_value(&[]) as f32
        );


        fn test_dist(p_0:&[f32], p_1:&[f32])->f32 {
            ((p_0[0] - p_1[0] ).powi(2)+ (p_0[1] - p_1[1] ).powi(2) + (p_0[2] - p_1[2] ).powi(2)).sqrt()
        }

        Ok(())
    }

    #[test]
    fn test_dist_d2() -> Result<(), ThoosaError>{
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&[0.00, 0.01, 0.02, 0.02, 0.02]),
        ]);

        let t_1 = Tensor::vstack(&[
            Tensor::from_slice(&[1.00, 1.01, 1.02]),
        ]);

        let t_2 = Tensor::vstack(&[
            Tensor::from_slice(&[2.00, 2.01, 2.02, 2.02]),
        ]);

        let t_3 = Tensor::vstack(&[
            Tensor::from_slice(&[3.00, 0.00, 0.01]),//[0,7] should be 0.0
        ]);

        let step = 1;
        let window = 2;
        debug!("t_0.size()={:?}", t_0.size());
        let t_0_sliced = super::slice_t(&t_0, step, window);

        debug!("t_0_sliced.size()={:?}", t_0_sliced.size());
        debug!("t_0_sliced=\n{}", t_0_sliced);


        let spliced_t = [t_1, t_2, t_3].into_iter().map(|t| slice_t(&t, step, window)).collect::<Vec<Tensor>>();
        let sizes = spliced_t.iter().map(|t|t.size()[0]).collect::<Vec<_>>();

        let tensor_ids = vec!["111", "222", "333"];

        let tensor_ids = sizes.iter().zip(tensor_ids.into_iter()).flat_map(|(size, id)| std::iter::repeat(id).take(*size as usize)).collect::<Vec<&str>>();
        debug!("sizes: {:?}", tensor_ids);
        let t_1_2_3_sliced = Tensor::vstack(&spliced_t);

        debug!("t_1_2_3_sliced={:?}", t_1_2_3_sliced.size());

        for c in  0..1 {
            let tensor = t_0_sliced.i((.., c, ..)).squeeze();
            let tensor1 = t_1_2_3_sliced.i((.., c, ..)).squeeze();
            debug!("tensor.size()={:?} tensor1.size()={:?}", tensor.size(), tensor1.size());
            debug!("tensor=\n{} tensor1=\n{}", tensor, tensor1);
            let t = super::dist_d2(&tensor, &tensor1)?;


            //first slice in t_0  & last slice in t_3 - 0.00, 0.01
            assert_eq!(0.0, t.double_value(&[0, 6]));
            
            debug!("t={:?}", t.size());
            debug!("t=\n{}", t);
            for i_0 in 0..t.size()[0] {
                 for i_1 in 0..t.size()[1] {
                     debug!("t_0={}/{} t_1={}/{} v={}", i_0, i_0 * window as i64, tensor_ids[i_1 as usize], i_1 * window as i64, t.double_value(&[i_0, i_1]));
                 }
            }
        }

        Ok(())
    }

    #[test]
    fn test_dist_d2_0(){

        let slice_0 = Tensor::from_slice(&[0., 0., 1., 1., 2., 2.,]).view((3,2));
        let slice_1 = Tensor::from_slice(&[3., 4., 4., 5., 5., 6.,]).view((3,2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        // let dist_t_0 = super::dist_d2_0(&slice_0, &slice_1).unwrap();
        debug!("dist_t={dist_t}");
        // debug!("dist_t_0={dist_t_0}");

        assert_eq!(5.0, dist_t.double_value(&[0,0]));
        assert_eq!(5.0, dist_t.double_value(&[1,1]));
        assert_eq!(5.0, dist_t.double_value(&[2,2]));

    }

    #[test]
    fn test_dist_d2_1(){//TODO remove
        let x = [
            0.9, 1.8,
            2.7, 3.6,
            4.5, 5.4,
            6.3, 7.2,
            8.1, 9.0,
        ];

        for (i_0, x_0) in x.windows(2).enumerate() {
            for (i_1, x_1) in x.windows(2).enumerate() {
                let dist:f64 = (x_0[0] - x_1[0]) * (x_0[0] - x_1[0]) + (x_0[1] - x_1[1]) * (x_0[1] - x_1[1]);
                let dist = dist.sqrt();
                debug!("[{i_0},{i_1}] x_0={x_0:?} x_1={x_1:?} dist={}", dist);
            }
        }
        //
        let slice_0 = Tensor::from_slice(&x).view((5, 2));
        let slice_1 = Tensor::from_slice(&x).view((5, 2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        debug!("dist_t={dist_t}");

        // // assert_eq!(5.0, dist_t.double_value(&[0,0]));
        // // assert_eq!(5.0, dist_t.double_value(&[1,1]));
        // // assert_eq!(5.0, dist_t.double_value(&[2,2]));



    }

    #[test]
    fn test_slice_t(){
        let tensor = super::slice_t(&Tensor::from_slice(&[1., 2., 3., 4., 5.]).view((1, -1)), 1, 2);

        assert_eq!(Tensor::from_slice(&[1., 2., 2., 3., 3., 4., 4., 5.,]).view((4, 1, 2)), tensor);
    }

}