use core::f32;
use log::{debug, log_enabled, trace};
use log::Level::Trace;
use tch::{IndexOp, TchError, Tensor};
use tch::Device::Cpu;
use tch::Kind;
use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::Kind::Bool;
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

//TODO use distance's histogram kurtosis and skew to scale down q_t? If histogram flat - prob. of useful patterns lower
#[derive(Debug)]
pub enum MaskMaker {
    Quantile{q:f32},
    Diagonal,
    Tril,
    LE {v:Tensor},//channels x thresholds
    GT {v:Tensor}, //channels x thresholds
    LocalMin
}

impl MaskMaker {
    pub fn make_mask(&self, perv_mask:Option<Tensor>, dists:&Tensor) -> Tensor {
        match self {
            MaskMaker::Quantile { q } => {
                Self::apply_prev(&perv_mask,  quantile_mask(*q as f64, dists))
            }
            MaskMaker::Diagonal => {
                let has_prev = perv_mask.is_some();
                if log_enabled!(Trace) && has_prev {
                    trace!("MaskMaker::make_mask Diagonal: perv_mask={}", perv_mask.as_ref().map(|v|v.shallow_clone()).unwrap());
                }

                let prev = perv_mask.unwrap_or(Tensor::ones(dists.size(), (Bool, dists.device())));
                _ = prev.diagonal(0, 1, 2).fill_(0);

                trace!("MaskMaker::make_mask Diagonal: 2. perv_mask.is_some()={has_prev} prev={prev}");

                prev
            }
            MaskMaker::Tril => {
                let prev = perv_mask.unwrap_or(Tensor::ones(dists.size(), (Bool, dists.device())));

                prev.triu(1)
            }
            MaskMaker::LE { v } => {
                Self::apply_prev(&perv_mask,  dists.le_tensor(v))
            }
            MaskMaker::GT { v } => {
                Self::apply_prev(&perv_mask,  dists.gt_tensor(v))
            }
            MaskMaker::LocalMin => {
                Self::apply_prev(&perv_mask,  local_min_mask(dists))
            }
        }
    }

    fn apply_prev(perv_mask: &Option<Tensor>, mask: Tensor) -> Tensor {
        match perv_mask {
            None => {
                mask
            }
            Some(prev) => {
                prev.logical_and(&mask)
            }
        }
    }
}
/*
    t_0, t_1 - channels x slices x slice_len

 */
pub fn clc_dist_d2_and_mask(
    t_0: Tensor,
    t_1: Tensor,
    with_distances: bool,
    mask_makers:&[MaskMaker],
) -> Result<(Tensor, Option<Tensor>), ThoosaError> {

    let distances = clc_dist_d2(t_0, t_1, 1)?;

    debug!("clc_dist_d2_and_mask: mask_makers={mask_makers:?} distances size={:?}", distances.size());
    trace!("clc_dist_d2_and_mask: distances=\n{distances}");

    Ok((
        mask_makers.into_iter().fold(None::<Tensor>, |mask, mask_maker|{
            debug!("clc_dist_d2_and_mask: mask={mask:?}");
            Some(mask_maker.make_mask(mask, &distances))
        }).unwrap_or(Tensor::ones(distances.size(), (Bool, Cpu))),

        if with_distances {
            Some(distances)
        } else {
            None
        },
    ))
}


/// distances - [channels x slices 0 x slices 1]
fn quantile_mask(q: f64, distances: &Tensor) -> Tensor {
    let channels= distances.size()[0];
    let flat_distances = distances.view((channels, -1));

    let samples = if flat_distances.size()[1] > 1000 {//TODO 1000 to config
        debug!("quantile_mask: flat_distances.size()={:?}", flat_distances.size());
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
    trace!("quantile_mask: 1. q_t=\n{q_t}");
    debug!("quantile_mask: samples.size={:?} q_t.size={:?}", samples.size(), q_t.size());

    let mut le = distances.le_tensor(&q_t); //channels x motifs x num_slices

    trace!("quantile_mask:  le=\n{le}");

    le
}

fn local_min_mask(distances: &Tensor) -> Tensor {
    debug!("local_min_mask: distances.size={:?}", distances.size());

    let distances_size = distances.size();
    let mid = distances.narrow(1, 1, distances_size[1]-2);
    let left = distances.narrow(1, 0, distances_size[1]-2);
    let right = distances.narrow(1, 2, distances_size[1]-2);

    let is_less_than_left = mid.lt_tensor(&left);
    let is_less_than_right = mid.lt_tensor(&right);

    let local_min_mask = is_less_than_left.logical_and(&is_less_than_right);


    let mut z = Tensor::zeros([distances_size[0], 1, distances_size[2]], (Bool, distances.device()));


    let full_mask = Tensor::cat(&[z.shallow_clone(), local_min_mask, z], 1);

    debug!("local_min_mask: full_mask.size={:?}", full_mask.size());

    full_mask
}

pub fn clc_dist_d2(t_0: Tensor, t_1: Tensor, dim: i64) -> Result<Tensor, ThoosaError> {


    let t_0 = t_0.unsqueeze(dim+1); // channels x motifs x 1 x motif_len
    let t_1 = t_1.unsqueeze(dim); // channels x 1 x num_slices x motif_len

    debug!("clc_dist_d2: t_0 size={:?}", t_0.size());
    debug!("clc_dist_d2: t_1 size={:?}", t_1.size());

    let mut diff = t_0.f_subtract(&t_1).map_err(tch_to_thoosa_err)?;
    drop(t_0);
    drop(t_1);

    let squared_diff = diff.f_pow_(2).map_err(tch_to_thoosa_err)?;
    drop(diff);
    debug!("clc_dist_d2: squared_diff size={:?}", squared_diff.size());

    let squared_distances = squared_diff.f_sum_dim_intlist(dim+2, false, Kind::Float).map_err(tch_to_thoosa_err)?;
    drop(squared_diff);

    Ok(squared_distances.sqrt())
}

pub fn slice_t(t_0: &Tensor, step: u32, window: u32) -> Tensor {
    debug!("slice_t:  t_0.size()={:?} step={step} window={window}", t_0.size());

    let tensors_vec = (0..=t_0.size()[1] - window as i64).step_by(step as usize).map(|i| t_0.i((.., i..i + window as i64))).collect::<Vec<Tensor>>();
    trace!("slice_t: tensors_vec={tensors_vec:?}");
    Tensor::stack(&tensors_vec, 0)
}


pub fn average_euclidean_distance(x: &Tensor) -> f32 {
    let n = x.size()[0] as i64;

    // Step 1: Compute the Gram matrix (dot products between vectors)
    let gram = x.mm(&x.transpose(0, 1)); // Shape: [n, n]

    // Step 2: Compute squared norms of each vector
    let squared_norms = x.pow_tensor_scalar(2).sum_dim_intlist(1, true, Kind::Float); // Shape: [n, 1]

    // Step 3: Compute squared distances matrix D
    let dists_squared:Tensor = &squared_norms + &squared_norms.transpose(0, 1) - 2.0 * &gram; // Shape: [n, n]
    let dists_squared = dists_squared.clamp_min(0.0); // Replace negative values with 0
    // Step 4: Extract upper triangle without diagonal
    let triu_indices = Tensor::triu_indices(n, n, 1, INT64_CPU); // Indices of upper triangle without diagonal
    let i = triu_indices.get(0);
    let j = triu_indices.get(1);

    let pairwise_squared_dists = dists_squared.index_select(0, &i).gather(1, &j.unsqueeze(1), false).squeeze();

    // Step 5: Compute square roots to get Euclidean distances
    let pairwise_dists = pairwise_squared_dists.sqrt();

    // Step 6: Compute the mean of the distances
    let mean_distance = pairwise_dists.mean(Kind::Float);

    trace!("average_euclidean_distance: pairwise_squared_dists=\n{pairwise_squared_dists} pairwise_dists=\n{pairwise_dists} mean_distance=\n{mean_distance}");

    mean_distance.double_value(&[]) as f32
}

#[cfg(test)]
mod distance_tests {
    use std::f32::consts::PI;
    use chrono::Utc;
    use log::__private_api::enabled;
    use log::{debug, set_max_level, trace};
    use log::LevelFilter::Trace;
    use plotters::prelude::{BLACK, GREEN, RED};
    use sci_rs::na::SimdBool;
    use tch::Device::Cpu;
    use tch::{Device, IndexOp, Kind};
    use tch::Kind::{Bool, Float};
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use thoosa_common::utils::{from, hist_1_dim, lin_t, plot_series, reset_print_options, skewness_kurtosis_1_dim, three_sigs_tensor, vec_f32_from};
    use thoosa_shared::{comp_sin_k_v, pulse_in_rand, saw, sig_one_imp, sig_three_imp, sig_two_imp, sin_v, ThoosaError};

    use crate::distance::{average_euclidean_distance, clc_dist_d2, clc_dist_d2_and_mask, slice_t, z_normalized_euclidean, MaskMaker};
    use crate::chunks::ChunkOfSlicedTensors;
    use crate::distance::MaskMaker::{Diagonal, LocalMin, Quantile, Tril, LE};

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

        let sig_0_v = sig_one_imp(n, max_y, None, None, 0.0);
        let sig_1_v = sig_two_imp(n, max_y, None, None, 0.1);
        let sig_2_v = sig_three_imp(n, max_y, None, None);

        //t_label_tensor
        //0 label - sig_0 in channel 0
        //1 label - sig_1 in channel 1
        //0,1 label - sig_2 in channel 2
        // let (t_0_0, t_0_1, t_1_0, t_1_1) = three_sigs_tensor(n, &sig_0_v, &sig_1_v, &sig_2_v);
        let t_0 = Tensor::vstack(&[
            Tensor::from_slice([sig_0_v.clone(), saw(n), saw(n)].concat().as_slice()),
            Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
            Tensor::from_slice([saw(n), sig_2_v.clone(), saw(n)].concat().as_slice()),
        ]);

        let t_1 =  Tensor::vstack(&[
            Tensor::from_slice([saw(n), sig_0_v.clone(), saw(n)].concat().as_slice()),
            Tensor::from_slice([saw(n), saw(n), saw(n)].concat().as_slice()),
            Tensor::from_slice([sig_2_v.clone(), saw(n), saw(n)].concat().as_slice()),
        ]);


        let x = lin_t(n*3);
        plot_series(
            vec![
                (format!("t_0_0_0"), plotters::prelude::RED,   vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&t_0.i((0, ..)))).collect::<Vec<(f32, f32)>>()),
                (format!("t_0_1_0"), plotters::prelude::GREEN, vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&t_1.i((0, ..)))).collect::<Vec<(f32, f32)>>()),
            ], "t_0_0_0 & t_0_1_0 channel 0", (2000, 1000), &format!("/tmp/plots/t_0_0{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;

        //t_label_tensor_channel
        let window = 1000;
        let t_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(
            10, window,
            &vec![t_0.into()], //sig_0 at idx 0
            vec![0], Cpu
        ).t;
        let t_1_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(
            10, window,
            &vec![t_1.into()], //sig_0 at idx 2000
            vec![0], Cpu
        ).t;

        let (le, Some(distances)) = clc_dist_d2_and_mask(
            t_0_sliced.shallow_clone(),
            t_1_sliced.shallow_clone(),

            true, &[Quantile {q:1.}]

        )? else {
            panic!("no distances")
        };

        let x = lin_t(distances.size()[1]);
        plot_series(
            vec![
                (format!("0ch t_0 slice 0"), plotters::prelude::GREEN,   vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&distances.i((0, 0, ..)))).collect::<Vec<(f32, f32)>>()),//sig_0_v match
                // (format!("1ch"), plotters::prelude::RED,     vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&distances.i((1, 0, ..)))).collect::<Vec<(f32, f32)>>()),//no matches
                // (format!("2ch"), plotters::prelude::BLUE,    vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&distances.i((2, 100, ..)))).collect::<Vec<(f32, f32)>>()),//sig_2_v match
            ], "distance", (2000, 1000), &format!("/tmp/plots/distance_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;

        debug!("distances={distances:?}");

        let channels_dists_flat = distances.flatten(1,2);

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


        let t_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(1, 3, &vec![t_0.into()], vec![0], Cpu).t;
        let t_1_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(1, 3, &vec![t_1.into()], vec![0], Cpu).t;

        let (le, Some(dists)) = clc_dist_d2_and_mask(t_0_sliced.shallow_clone(), t_1_sliced.shallow_clone(), true, &[Quantile {q:0.1}])? else {
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



        Ok(())
    }

    fn test_dist(p_0:&[f32], p_1:&[f32])->f32 {
        ((p_0[0] - p_1[0] ).powi(2)+ (p_0[1] - p_1[1] ).powi(2) + (p_0[2] - p_1[2] ).powi(2)).sqrt()
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
            let t = super::clc_dist_d2(tensor, tensor1, 0)?;


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
        let (mask, Some(dist_t)) = super::clc_dist_d2_and_mask(slice_0, slice_1, true, &[Quantile {q:0.0}]).unwrap() else {
            panic!("no dists");
        };
        // debug!("mask={mask}");
        // debug!("dist_t={dist_t}");

        assert_eq!(
            Tensor::from_slice(&[
                false, false, false,
                false, false, false,
                true,  false, false
            ]).view((1, 3,3)),
            mask
        );

        assert_eq!(5.0, dist_t.double_value(&[0,0,0]));
        assert_eq!(5.0, dist_t.double_value(&[0,1,1]));
        assert_eq!(5.0, dist_t.double_value(&[0,2,2]));


    }

    #[test]
    fn test_dist_d2_0_diag(){

        let slice_0 = Tensor::from_slice(&[0., 0., 1., 1., 2., 2.,]).view((1, 3,2));
        let slice_1 = Tensor::from_slice(&[3., 4., 4., 5., 5., 6.,]).view((1, 3,2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        // let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        //set_max_level(Trace);
        let (mask, Some(dist_t)) = clc_dist_d2_and_mask(slice_0, slice_1, true, &[Quantile {q:1.0}, Diagonal]).unwrap() else {
            panic!("no dists");
        };
        debug!("mask={mask}");
        // debug!("dist_t={dist_t}");

        assert_eq!(
            Tensor::from_slice(&[
                false, true,  true,
                true,  false, true,
                true,  true,  false
            ]).view((1, 3,3)),
            mask
        );
        //
        // assert_eq!(5.0, dist_t.double_value(&[0,0,0]));
        // assert_eq!(5.0, dist_t.double_value(&[0,1,1]));
        // assert_eq!(5.0, dist_t.double_value(&[0,2,2]));

    }

    #[test]
    fn test_dist_le_mask(){
        // let slices = Tensor::from_slice(&[0., 0., 1., 1., 2., 2., ]).view((1, 3, 2));
        // let motif_means = Tensor::from_slice(&[3., 4., 4., 5., 7., 8., 9., 10.,]).view((1, 4, 2));
        // let motif_avg_distances = Tensor::from_slice(&[3., 4., 5., 6., ]).view((1, 4));
        // debug!("slices={slices}");
        // debug!("motif_means={motif_means}");
        // debug!("motif_avg_distances={motif_avg_distances}");
        //
        //
        // // let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        // set_max_level(Trace);
        // let (mask, Some(dist_t)) = clc_dist_d2_and_mask(slices, motif_means, true, &[LE {v:motif_avg_distances}]).unwrap() else {
        //     panic!("no dists");
        // };
        // debug!("mask={mask}");
        // debug!("dist_t={dist_t}");

        //5 motifs 7 slices
        let dists_t = Tensor::from_slice(&[
            0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
            0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
            0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26,
            0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36,
            0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46,

            1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06,
            1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16,
            1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26,
            1.30, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36,
            1.40, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46,

            2.00, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06,
            2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16,
            2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26,
            2.30, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36,
            2.40, 2.41, 2.42, 2.43, 2.44, 2.45, 2.46,

        ]).view((3, 5, 7));

        let thresholds_t = Tensor::from_slice(&[
            0.01,
            0.11,
            0.21,
            0.31,
            0.41,

            1.01,
            1.11,
            1.21,
            1.31,
            1.41,

            2.01,
            2.11,
            2.21,
            2.31,
            2.41,

        ]).view((3, 5));

        let thresholds_t_broadcasted = thresholds_t.unsqueeze(2).broadcast_to(&[3,5,7]);

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
        let t = Tensor::from_slice((0..10).map(|v|v as f32).collect::<Vec<_>>().as_slice()).to_kind(Float);
        // let t = Tensor::from_slice(&[0, 1, 2, 3, 4, 5, 6, 4, 5, 6,4, 5, 6,4, 5, 6,4, 5, 6,4, 5, 6,]).to_kind(Float);

        let q = t.f_quantile_scalar(0.9, 0, true, "higher").unwrap();

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

    #[test]
    fn test_average_euclidean_distance(){
        let d = [
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.
        ];
        let ab = test_dist(&d[0..3], &d[3..6]);
        let ac = test_dist(&d[0..3], &d[6..9]);
        let bc = test_dist(&d[3..6], &d[6..9]);

        println!("ab={ab} ac={ac} bc={bc}");

        let test_avg = (
            ab + ac + bc
        ) / 3.;

        println!("test_avg={test_avg}");


        let x = Tensor::from_slice(&d).view((3, 3));
        println!("x={x}");
        let avg_dist = average_euclidean_distance(&x);
        assert_eq!(
            test_avg,
            avg_dist
        );


    }

    #[test]
    fn test_mask_maker_diag(){
        let mask = Diagonal.make_mask(None, &Tensor::ones(&[2, 3, 3], (Bool, Cpu)));

        assert_eq!(Tensor::from_slice(
            &[
                false, true , true ,
                true , false, true,
                true , true , false,

                false, true , true,
                true , false, true,
                true , true , false
            ]
        ).view((2, 3, 3)),
            mask
        );

        // debug!("mask=\n{mask}");
    }

    #[test]
    fn test_mask_maker_tril(){
        let mask = Tril.make_mask(None, &Tensor::ones(&[2, 3, 3], (Bool, Cpu)));
        debug!("mask=\n{mask}");

        assert_eq!(Tensor::from_slice(
            &[
                false, true , true ,
                false, false, true,
                false, false, false,

                false, true , true,
                false, false, true,
                false, false, false
            ]
        ).view((2, 3, 3)),
                   mask
        );

        debug!("mask=\n{mask}");
    }

    #[test]
    fn test_mask_maker_local_min() -> Result<(), ThoosaError>{
        let input_n =  1000;


        let ch_0 = (0..input_n).map(|x| (2. * PI * (x as f32 / input_n as f32)).sin()).collect::<Vec<f32>>(); //sin_v(input_n, 1.);
        let ch_1 = (0..input_n).map(|x| (4. * PI * (x as f32  / input_n as f32)).sin()).collect::<Vec<f32>>(); //sin_v(input_n, 0.5);

        /*let x = lin_t(input_n);
        plot_series(
            vec![
                (format!("ch_0"), GREEN, vec_f32_from(&x).into_iter().zip(ch_0.clone().into_iter()).collect::<Vec<(f32, f32)>>()),
                (format!("ch_1"), BLACK,  vec_f32_from(&x).into_iter().zip(ch_1.clone().into_iter()).collect::<Vec<(f32, f32)>>()),

            ], "Sigs", (2000, 1000), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;*/
        let min_ch0 = 3./4. * input_n as f32;
        let min_ch1 = [3./8. * input_n as f32, 7./8. * input_n as f32] ;
        debug!("min_ch0={min_ch0:?} min_ch1={min_ch1:?}");

        let distances = Tensor::stack(&[
            Tensor::from_slice(ch_0.as_slice()),
            Tensor::from_slice(ch_1.as_slice())
        ], 0).unsqueeze(2);//unsqueeze to make it [channels x t_0 slices x t_1 slices]

        let mask = LocalMin.make_mask(None, &distances);
        reset_print_options();
        trace!("mask=\n{mask}");
        debug!("mask.nonzero=\n{}", mask.nonzero());

        assert_eq!(
            Tensor::from_slice(&[
                0, 750, 0,
                1, 375, 0,
                1, 875, 0
            ]).view((3, 3)),
            mask.nonzero()
        );

        Ok(())
    }

    #[test]
    fn test_slice(){
        let t = Tensor::from_slice(&[0., 1., 2., 3., 4., 5., 6., 7.]);

        println!("s={}", t.slice(0, 3, 6, 2));
        println!("n={}", t.narrow(0, 0, 6));
    }



}