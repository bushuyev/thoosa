use core::f32;
use log::{debug, trace};
use tch::{IndexOp, TchError, Tensor};
use tch::Device::Cpu;
use tch::Kind;
use thoosa_common::parzen::optimizer::Primitives::Float;
use thoosa_common::utils::{skewness_kurtosis_1_dim, tch_to_thoosa_err, z_norm_2};
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

/*
    t_0, t_1 - channels x slices x slice_len
 */
pub fn dist_d2_0(
    t_0: Tensor,
    t_1: Tensor,
    q: f64,
    with_distances: bool,
    exclude_diag: bool,
    // K: f64,
) -> Result<(Tensor, Option<Tensor>), ThoosaError> {
    //align dimensions for broadcasting
    let channels = t_0.size()[0];

    let t_0 = t_0.unsqueeze(2); // channels x motifs x 1 x motif_len
    let t_1 = t_1.unsqueeze(1); // channels x 1 x num_slices x motif_len

    debug!("dist_d2_0: t_0 size={:?}", t_0.size());
    debug!("dist_d2_0: t_1 size={:?}", t_1.size());

    let mut diff = t_0.f_subtract(&t_1).map_err(tch_to_thoosa_err)?;
    drop(t_0);
    drop(t_1);

    let squared_diff = diff.f_pow_(2).map_err(tch_to_thoosa_err)?;
    drop(diff);

    let squared_distances = squared_diff.sum_dim_intlist(3, false, Kind::Float);
    drop(squared_diff);

    let distances = squared_distances.sqrt();
    drop(squared_distances);

    debug!("dist_d2_0: distances size={:?}", distances.size());
    trace!("dist_d2_0: distances={:?}", distances);

    //distances to channels x (motifs * num_slices) for kurtosis calculation
    // let distances_flat = distances.view([distances.size()[0], -1]); // [channels x total_distances_per_channel]

    // let (_skewness, kurtosis) = skewness_kurtosis_1_dim(&distances_flat);

    // debug!("dist_d2_0:  kurtosis={kurtosis}");

    // Create a mask for channels with kurtosis <= K
    // let include_channels = kurtosis.gt(0.0); // [channels]
    // let include_channels_mask = include_channels.view([-1, 1, 1]); // [channels x 1 x 1]

    // Set distances of excluded channels to a high value (e.g., f64::MAX)
    // let max_value = f32::MAX;
    // let distances = distances.masked_fill(&include_channels_mask.logical_not(), max_value as f64);

    // Sample indices along the motifs dimension for quantile calculation
    let flat_distances = distances.view((channels, -1));

    let samples = if flat_distances.size()[1] > 1000 {
        debug!("dist_d2_0: flat_distances.size()={:?}", flat_distances.size());
        let indices = Tensor::randint(
            flat_distances.size()[1],
            &[(flat_distances.size()[1] as f64 * 0.1) as i64],
            (Kind::Int64, distances.device()),
        );

        let samples = flat_distances.index_select(1, &indices);
        drop(indices);
        samples
    } else {
        flat_distances
    };

    let mut q_t = samples.quantile_scalar(q, 1, true, "lower").view((channels, 1, 1));
    trace!("dist_d2_0: 1. q_t={q_t}");
    let _ = q_t.multiply_scalar_(0.5);

    debug!("dist_d2_0: samples.size()={:?} q_t={:?}", samples.size(), q_t.size());
    trace!("dist_d2_0: 2. q_t={q_t}");

    let mut le = distances.le_tensor(&q_t); //channels x motifs x num_slices

    // let le_sum = le.view((channels, -1)).sum_dim_intlist(1, false, Kind::Float);
    // trace!("dist_d2_0: le_sum={le_sum}");
    // exclude channels with kurtosis > K from the mask
    // le = le* &include_channels_mask;

    if exclude_diag {
        let mut le_diagonals = le.diagonal(0, 1, 2);
        _= le_diagonals.fill_(0);
    }

    trace!("dist_d2_0: le size={:?}", le.size());
    trace!("dist_d2_0: le={:?}", le);

    Ok((
        le,
        if with_distances {
            Some(distances)
        } else {
            None
        },
    ))
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
    use log::LevelFilter::Trace;
    use sci_rs::na::SimdBool;
    use tch::Device::Cpu;
    use tch::{Device, IndexOp, Kind};
    use tch::Kind::Float;
    use tch::Tensor;
    use thoosa_common::utils::{from, hist_1_dim, lin_t, plot_series, skewness_kurtosis_1_dim, three_sigs_tensor, vec_f32_from};
    use thoosa_shared::{sig_one_imp, sig_three_imp, sig_two_imp, ThoosaError};

    use crate::distance::{dist_d2_0, slice_t, z_normalized_euclidean};
    use crate::chunks::ChunkOfSlicedTensors;

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
    fn test_dist_d2_0_1ch_with_sig() -> Result<(), ThoosaError>{
        log::set_max_level(Trace);

        let n = 1000i64;
        let max_y = 30.0;

        let sig_0_v = sig_one_imp(n, max_y, None, None);
        let sig_1_v = sig_two_imp(n, max_y, None, None);
        let sig_2_v = sig_three_imp(n, max_y, None, None);

        //t_label_tensor
        //0 label - sig_0 in channel 0
        //1 label - sig_1 in channel 1
        //0,1 label - sig_2 in channel 2
        let (t_0_0, t_0_1, t_1_0, t_1_1) = three_sigs_tensor(n, &sig_0_v, &sig_1_v, &sig_2_v);

        let x = lin_t(n*3);
        plot_series(
            vec![
                (format!("t_0_0_0"), plotters::prelude::RED,   vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&t_0_0.i((0, ..)))).collect::<Vec<(f32, f32)>>()),
                (format!("t_0_1_0"), plotters::prelude::GREEN, vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&t_0_1.i((0, ..)))).collect::<Vec<(f32, f32)>>()),
                // (format!("ch2"), plotters::prelude::BLACK, vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&t_0_0.i((2, ..)))).collect::<Vec<(f32, f32)>>()),
            ], "t_0_0_0 & t_0_1_0", (2000, 1000), &format!("/tmp/plots/t_0_0{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;


        //t_label_tensor_channel
        let window = 100;
        let t_0_0_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(
            10, window,
            &vec![t_0_0.into()], //sig_0 at idx 0
            vec!["x"], Cpu
        ).t;
        let t_0_1_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(
            10, window,
            &vec![t_0_1.into()], //sig_0 at idx 2000
            vec!["x"], Cpu
        ).t;


        let (le, Some(distances)) = dist_d2_0(
            t_0_0_0_sliced.shallow_clone(),
            t_0_1_0_sliced.shallow_clone(),
            1.0,//all distances
            true, false

        )? else {
            panic!("no distances")
        };
        let x = lin_t(distances.size()[1]);
        plot_series(
            vec![
                (format!("0ch"), plotters::prelude::GREEN,   vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&distances.i((0, 0, ..)))).collect::<Vec<(f32, f32)>>()),//sig_0_v match
                (format!("1ch"), plotters::prelude::RED,     vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&distances.i((1, 0, ..)))).collect::<Vec<(f32, f32)>>()),//no matches
                (format!("2ch"), plotters::prelude::BLUE,    vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&distances.i((2, 100, ..)))).collect::<Vec<(f32, f32)>>()),//sig_2_v match
            ], "distance", (2000, 1000), &format!("/tmp/plots/distance_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;

        debug!("distances={distances:?}");

        let channels_dists_flat = distances.flatten(1,2);

        // let channels_hists = channels_dists_flat.hist
        debug!("channels_dists_flat={:?}", channels_dists_flat.size());
        let hists = hist_1_dim(&channels_dists_flat, window);
        let x = lin_t(window as i64);

        let (skewness, kurtosis) = skewness_kurtosis_1_dim(&hists.totype(Float));
        debug!("skewness={skewness}");
        debug!("skewness={kurtosis}");

        plot_series(
            vec![
                (format!("0ch"), plotters::prelude::ShapeStyle {
                    color: plotters::prelude::GREEN.into(),
                    filled: false,
                    stroke_width: 2,
                },   vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&hists.i(0))).collect::<Vec<(f32, f32)>>()),//sig_0_v match
                (format!("1ch"), plotters::prelude::RED.into(),     vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&hists.i(1))).collect::<Vec<(f32, f32)>>()),//no matches
                (format!("2ch"), plotters::prelude::BLUE.into(),    vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&hists.i(2))).collect::<Vec<(f32, f32)>>()),//sig_2_v match
            ], "hists", (2000, 1000), &format!("/tmp/plots/hists_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;

        // let distances_std_norm = channels_dists_flat.std_dim(1, true, true) / channels_dists_flat.mean_dim(1, true, channels_dists_flat.kind());
        //
        // let (skewness, kurtosis) = skewness_kurtosis(&channels_dists_flat);
        //
        // let kurtosis_gt = kurtosis.gt(0.0);
        //
        // debug!("distances_std_norm={distances_std_norm}");
        // debug!("skewness={skewness}");
        // debug!("kurtosis={kurtosis}");
        // debug!("kurtosis_gt={kurtosis_gt}");
        //
        // debug!("le={le}");




        Ok(())
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


        let t_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(1, 3, &vec![t_0.into()], vec!["x"], Cpu).t;
        let t_1_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(1, 3, &vec![t_1.into()], vec!["x"], Cpu).t;

        let (le, Some(dists)) = dist_d2_0(t_0_sliced.shallow_clone(), t_1_sliced.shallow_clone(), 0.1, true, false)? else {
            panic!("no distances")
        };

        debug!("le={le} size={:?}", le.size());
        debug!("dists={dists} size={:?}", dists.size());


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

        let slice_0 = Tensor::from_slice(&[0., 0., 1., 1., 2., 2.,]).view((1, 3,2));
        let slice_1 = Tensor::from_slice(&[3., 4., 4., 5., 5., 6.,]).view((1, 3,2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        // let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        let (le, Some(dist_t)) = super::dist_d2_0(slice_0, slice_1, 0.1, true, false).unwrap() else {
            panic!("no dists");
        };
        // debug!("dist_t={dist_t}");
        // debug!("dist_t_0={dist_t_0}");

        assert_eq!(5.0, dist_t.double_value(&[0,0,0]));
        assert_eq!(5.0, dist_t.double_value(&[0,1,1]));
        assert_eq!(5.0, dist_t.double_value(&[0,2,2]));

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


    #[test]
    fn test_q(){
        // let t = Tensor::from_slice((0..10).map(|v|v as f32).collect::<Vec<_>>().as_slice()).to_kind(Float);
        let t = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 4, 5, 6,4, 5, 6,4, 5, 6,4, 5, 6,4, 5, 6,]).to_kind(Float);

        let q = t.f_quantile_scalar(0.5, 0, true, "higher").unwrap();

        debug!("q={q}");

        /*
         let t = Tensor::vstack(&[
            Tensor::from_slice(&[0,1, 2, 3 , 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 3]),
            Tensor::from_slice(&[0,10,20,30,40,50,60,40,50,60,40,50,60,40,50,60,40,50,60,40,50,60,40,50,60,40,50,60, 90])
        ]).to_kind(Float);

        let q = t.quantile_scalar(0.1, 1, true, "higher");

        let le = t.le_tensor(&q);
        // let gt_0 = t.gt(0.);
        // let le_gt_0 = le.logical_and(&gt_0);
        //
        let masked = t.masked_select(&le);
        // let masked_gt0 = masked.masked_select(&masked.gt(0.));
        //
        debug!("size={:?} q={q}", q.size());
        debug!("le={le} count_nonzero={}", le.count_nonzero_dim_intlist([1]));
        // debug!("gt_0={gt_0}");
        // debug!("le_gt_0={le_gt_0}");
        debug!("masked={masked}");
        */
    }

}