use core::f32;
use std::ops::Add;
use std::sync::Arc;
use std::sync::atomic::AtomicI64;
use log::{debug, log_enabled, trace};
use log::Level::Trace;
use parking_lot::{Mutex, RwLock};
use tch::{IndexOp, TchError, Tensor};
use tch::Device::Cpu;
use tch::Kind;
use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::Kind::Bool;
use thoosa_common::utils::{skewness_kurtosis_1_dim, z_norm_2};
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
pub enum MaskMaker<'a> {
    LeQuantile {q:f32, with_mask:bool},
    LeMean {k:f32, avg:&'a mut Mutex<Tensor>, total_samples:&'a mut Mutex<Tensor>},
    Diagonal,
    Tril,
    LE {v:Tensor},//channels x thresholds
    GT {v:Tensor}, //channels x thresholds
    LocalMin
}

impl <'a> MaskMaker<'a> {
    pub fn make_mask(&mut self, prev_mask:Option<Tensor>, dists:&Tensor) -> Tensor {
        match self {
            MaskMaker::LeQuantile { q, with_mask } => {
                Self::apply_prev(&prev_mask, quantile_mask(*q as f64, dists, if *with_mask { prev_mask.as_ref() } else {None}))
            }
            MaskMaker::Diagonal => {
                let has_prev = prev_mask.is_some();
                if log_enabled!(Trace) && has_prev {
                    trace!("MaskMaker::make_mask Diagonal: perv_mask={}", prev_mask.as_ref().map(|v|v.shallow_clone()).unwrap());
                }

                let prev = prev_mask.unwrap_or(Tensor::ones(dists.size(), (Bool, dists.device())));
                _ = prev.diagonal(0, 1, 2).fill_(0);

                trace!("MaskMaker::make_mask Diagonal: 2. perv_mask.is_some()={has_prev} prev={prev}");

                prev
            }
            MaskMaker::Tril => {
                let prev = prev_mask.unwrap_or(Tensor::ones(dists.size(), (Bool, dists.device())));
                prev.triu(1)
            }
            MaskMaker::LE { v } => {
                Self::apply_prev(&prev_mask, dists.le_tensor(v))
            }
            MaskMaker::GT { v } => {
                Self::apply_prev(&prev_mask, dists.gt_tensor(v))
            }
            MaskMaker::LocalMin => {
                Self::apply_prev(&prev_mask, local_min_mask(&dists))
            }
            MaskMaker::LeMean { k, avg, total_samples } => {
                Self::apply_prev(&prev_mask, mean_mask(*k as f64, dists, prev_mask.as_ref(), *avg, total_samples))
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
/// returns bool mask and if with_distances is true distances
pub fn clc_dist_d2_and_mask(
    t_0: Tensor,
    t_1: Tensor,
    with_distances: bool,
    mask_makers:&mut [MaskMaker],
) -> Result<(Tensor, Option<Tensor>), ThoosaError> {
    let de = t_0.device();

    let distances = clc_dist_d2(t_0, t_1, 1)?;

    debug!("clc_dist_d2_and_mask: mask_makers={mask_makers:?} distances size={:?}", distances.size());
    trace!("clc_dist_d2_and_mask: distances=\n{distances}");


    Ok((
        mask_makers.into_iter().fold(None::<Tensor>, |mask, mask_maker|{
            debug!("clc_dist_d2_and_mask: mask={mask:?}");
            Some(mask_maker.make_mask(mask, &distances))
        }).unwrap_or(Tensor::ones(distances.size(), (Bool, de))),

        if with_distances {
            Some(distances)
        } else {
            None
        },
    ))
}


/// distances - [channels x slices 0 x slices 1]
fn quantile_mask(q: f64, distances: &Tensor, pre_mask: Option<&Tensor>) -> Tensor {
    let distances_size = distances.size();

    trace!("quantile_mask: after pre_mask distances=\n{distances}");

    /*let samples = if distances.size()[1] > 1000 {//TODO 1000 to config
        debug!("quantile_mask: flat_distances.size()={:?}", distances.size());
        let indices = Tensor::randint(
            distances.size()[1],
            &[(distances.size()[1] as f64 * 0.1) as i64],
            (Kind::Int64, distances.device()),
        );

        let samples = distances.index_select(1, &indices);
        drop(indices);
        samples
    } else {
        distances.shallow_clone()
    };*/

    let q_t = if let Some(pm) = pre_mask {
        let mut quantiles = Vec::with_capacity((distances_size[0] * distances_size[2]) as usize);
        for channel in 0..distances_size[0] {
            for slice_1 in 0..distances_size[2] {
                let slice_1_t = distances.i((channel, .., slice_1)).masked_select(&pm.i((channel, .., slice_1)));
                let q_t = slice_1_t.quantile_scalar(q, 0, true, "lower");
                trace!("quantile_mask: channel={channel} slice_1={slice_1} slice_1_t=\n{slice_1_t} q_t={q_t}");
                quantiles.push(q_t);
            }
        }
        Tensor::stack(quantiles.as_slice(), 0).view((distances_size[0], 1, distances_size[2]))
    } else {
        distances.transpose(1,2).quantile_scalar(q, 2, true, "lower").transpose(1,2)
    };

    // let q_t = distances.quantile_scalar(q, 2, true, "lower");
    trace!("quantile_mask: full. q_t=\n{q_t}");
    debug!("quantile_mask: samples.size={:?} q_t.size={:?}", distances.size(), q_t.size());

    let le = distances.le_tensor(&q_t); //channels x motifs x num_slices

    trace!("quantile_mask:  le=\n{le}");

    le
}

///
/// * distances - [channels x slices 0 x slices 1]
/// * returns - Bool tensor of same size with true for local minimums for  slice 1 along slices 0.
fn local_min_mask(distances: &Tensor) -> Tensor {
    debug!("local_min_mask: distances.size={:?}", distances.size());
    let distances_size = distances.size();

    let first_min = distances
        .narrow(1, 0, 1)
        .lt_tensor(&distances.narrow(1, 1, 1));


    let last_min = distances
        .narrow(1, distances_size[1] - 1, 1)
        .lt_tensor(&distances.narrow(1, distances_size[1] - 2, 1));

    trace!("local_min_mask: first_min=\n{first_min}");
    trace!("local_min_mask: last_min=\n{first_min}");


    let mid = distances.narrow(1, 1, distances_size[1]-2);
    let left = distances.narrow(1, 0, distances_size[1]-2);
    let right = distances.narrow(1, 2, distances_size[1]-2);

    trace!("local_min_mask: mid=\n{mid}");
    trace!("local_min_mask: left=\n{left}");
    trace!("local_min_mask: right=\n{right}");

    let is_less_than_left = mid.lt_tensor(&left);
    let is_less_than_right = mid.le_tensor(&right);

    trace!("local_min_mask: is_less_than_left=\n{is_less_than_left}");
    trace!("local_min_mask: is_less_than_right=\n{is_less_than_right}");

    let local_min_mask = is_less_than_left.logical_and(&is_less_than_right);


    let full_mask = Tensor::cat(&[first_min, local_min_mask, last_min], 1);

    debug!("local_min_mask: full_mask.size={:?}", full_mask.size());

    full_mask
}

/// distances - [channels x slices 0 x slices 1]
fn mean_mask(k: f64, distances: &Tensor, pre_mask: Option<&Tensor>, means_acc:&mut Mutex<Tensor>, total_samples:&mut Mutex<Tensor>) -> Tensor {
    let distances_size = distances.size();

    trace!("mean_mask: after pre_mask distances=\n{distances}");


    let (mean_t, counts_t) = if let Some(pm) = pre_mask {
        let mut means = Vec::with_capacity(distances_size[1] as usize);
        let mut counts = Vec::with_capacity(distances_size[1] as usize);

        for channel in 0..distances_size[0] {
            let channel_dists_t = distances.i((channel, .., ..)).masked_select(&pm.i((channel, .., ..)));
            let mean = channel_dists_t.mean(Kind::Float);
            trace!("mean_mask: channel={channel} channel_dists_t=\n{channel_dists_t} mean={mean}");
            means.push(mean);
            counts.push(channel_dists_t.size()[0])
        }
        (
            Tensor::stack(means.as_slice(), 0).view((distances_size[0], 1)),
            Tensor::from_slice(counts.as_slice()).to_device(distances.device()).view((distances_size[0], 1)),
        )
    } else {
        (
            distances.mean_dim(&[1i64, 2i64][..],  false, Kind::Float),
            Tensor::from_slice(&[distances_size[1] * distances_size[2]]).to_device(distances.device()).broadcast_to(distances_size[0]) //  Tensor of size [channels x 1] with values= distances[1] * distances[2] in each channel
        )
    };

    trace!("mean_mask: full. mean_t=\n{mean_t}");

    debug!("mean_mask: samples.size={:?} mean_t.size={:?}", distances.size(), mean_t.size());


    debug!("mean_mask: total_samples={total_samples:?} counts={:?}", counts_t);

    //TODO update means_acc
    // total_means = (total_means * total_samples + current_means * batch_size_i)
    //     / (total_samples + batch_size_i);
    // total_samples += batch_size_i;

    let mut means_guard = means_acc.lock();
    let mut total_samples_guard = total_samples.lock();
    debug!("new_total_means: 1. means_guard=\n{means_guard} total_samples_guard=\n{total_samples_guard}");

    let means_device = total_samples_guard.device();

    *means_guard = ((means_guard.multiply(&total_samples_guard)  + mean_t * counts_t.shallow_clone())/counts_t.f_add(&total_samples_guard).expect("counts_t._add")).to_device(means_device);
    *total_samples_guard += counts_t.to_device(means_device);

    debug!("new_total_means: 2. means_guard=\n{means_guard} total_samples_guard=\n{total_samples_guard}");

    let le = distances.le_tensor(&means_guard.multiply_scalar(k).view((-1, 1, 1)).broadcast_to(distances_size)); //channels x motifs x num_slices

    trace!("mean_mask:  le=\n{le}");

    le
}


pub fn clc_dist_d2(t_0: Tensor, t_1: Tensor, dim: i64) -> Result<Tensor, ThoosaError> {


    let t_0 = t_0.unsqueeze(dim+1); // channels x motifs x 1 x motif_len
    let t_1 = t_1.unsqueeze(dim); // channels x 1 x num_slices x motif_len

    debug!("clc_dist_d2: t_0 size={:?}", t_0.size());
    debug!("clc_dist_d2: t_1 size={:?}", t_1.size());

    let mut diff = t_0.f_subtract(&t_1)?;
    drop(t_0);
    drop(t_1);

    let squared_diff = diff.f_pow_(2)?;
    drop(diff);
    debug!("clc_dist_d2: squared_diff size={:?}", squared_diff.size());

    let squared_distances = squared_diff.f_sum_dim_intlist(dim+2, false, Kind::Float)?;
    drop(squared_diff);

    Ok(squared_distances.sqrt())
}

pub fn slice_t(t_0: &Tensor, step: u32, window: u32) -> Tensor {
    debug!("slice_t:  t_0.size()={:?} step={step} window={window}", t_0.size());

    let tensors_vec = (0..=t_0.size()[1] - window as i64).step_by(step as usize).map(|i| t_0.i((.., i..i + window as i64))).collect::<Vec<Tensor>>();
    trace!("slice_t: tensors_vec={tensors_vec:?}");
    Tensor::stack(&tensors_vec, 0)
}


pub fn average_euclidean_distance(t: &Tensor) -> Result<f32, ThoosaError> {
    let n = t.size()[0];
    debug!("average_euclidean_distance: n={n}");

    // Step 2: Compute squared norms of each vector
    let squared_norms = t.f_pow_tensor_scalar(2)?
        .f_sum_dim_intlist(1, true, Kind::Float)?;

    // Step 3: Compute squared distances matrix D
    let squared_norms_sum = squared_norms.f_add(&squared_norms.transpose(0, 1))?;
    drop(squared_norms);

    // Step 1: Compute the Gram matrix (dot products between vectors)
    let gram = t.f_mm(&t.transpose(0, 1)).and_then(|v|v.f_multiply_scalar(2))?;
    let dists_squared:Tensor = squared_norms_sum.f_subtract(&gram)?;
    drop(gram);
    drop(squared_norms_sum);

    let dists_squared = dists_squared.clamp_min(0.0); // Replace negative values with 0
    // Step 4: Extract upper triangle without diagonal
    let triu_indices = Tensor::triu_indices(n, n, 1, (Kind::Int64, t.device())); // Indices of upper triangle without diagonal
    let i = triu_indices.get(0);
    let j = triu_indices.get(1);

    let pairwise_squared_dists = dists_squared.index_select(0, &i).gather(1, &j.unsqueeze(1), false).squeeze();
    drop(dists_squared);
    drop(i);
    drop(j);

    // Step 5: Compute square roots to get Euclidean distances
    let pairwise_dists = pairwise_squared_dists.sqrt();
    trace!("average_euclidean_distance: pairwise_squared_dists=\n{pairwise_squared_dists}");
    drop(pairwise_squared_dists);

    // Step 6: Compute the mean of the distances
    let mean_distance = pairwise_dists.mean(Kind::Float);

    trace!("average_euclidean_distance: pairwise_dists=\n{pairwise_dists} mean_distance=\n{mean_distance}");

    Ok(mean_distance.double_value(&[]) as f32)
}

#[cfg(test)]
mod distance_tests {
    use std::f32::consts::PI;
    use std::fmt::Display;
    use std::ops::Deref;
    use std::sync::atomic::AtomicI64;
    use chrono::Utc;
    use log::__private_api::enabled;
    use log::{debug, max_level, set_max_level, trace};
    use log::LevelFilter::{Debug, Trace};
    use parking_lot::Mutex;
    use plotters::prelude::{BLACK, GREEN, RED};
    use plotters::style::full_palette::ORANGE;
    use sci_rs::na::SimdBool;
    use tch::Device::{Cpu, Cuda};
    use tch::{Device, IndexOp, Kind};
    use tch::Kind::{Bool, Float, Int, Int64};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA, INT64_CPU};
    use tch::Tensor;
    use thoosa_common::builder::Node::Par;
    use thoosa_common::loader::SyncWrapper;
    use thoosa_common::panic_to_err;
    use thoosa_common::train_nets::{DummyFindMotifsRunConfigCtx, FindMotifsRunConfig};
    use thoosa_common::utils::{from, hist_1_dim, lin_iter, lin_t, plot_series, plot_series_with_points, reset_print_options, run_split_on_oom, skewness_kurtosis_1_dim, three_sigs_tensor, vec_f32_from, Reducer, SplitQueue, Splittable, SplittableTensor};
    use thoosa_shared::{comp_sin_k_v, pulse_in_rand, saw, sig_one_imp, sig_three_imp, sig_two_imp, sin_v, ThoosaError};
    use thoosa_shared::colors_iter::DistinctColorIter;
    use thoosa_shared::run_configs::Applicable;
    use crate::distance::{average_euclidean_distance, clc_dist_d2, clc_dist_d2_and_mask, local_min_mask, mean_mask, slice_t, z_normalized_euclidean, MaskMaker};
    use crate::chunks::{ChunkOfSlicedTensors};
    use crate::distance::MaskMaker::{Diagonal, LocalMin, LeQuantile, Tril, LE, LeMean};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
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

        // set_max_level(Trace);

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


        // let t_1 =  Tensor::vstack(&[
        //     Tensor::from_slice([sig_0_v.clone()].concat().as_slice()),
        //     Tensor::from_slice([saw(n)].concat().as_slice()),
        //     Tensor::from_slice([saw(n)].concat().as_slice()),
        // ]);


        let x = lin_t(n*3);
        plot_series(
            vec![
                (format!("t_0_0"), plotters::prelude::RGBColor(254,0,0),   vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&t_0.i((0, ..)))).collect::<Vec<(f32, f32)>>()),
                (format!("t_1_0"), GREEN, vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&t_1.i((0, ..)))).collect::<Vec<(f32, f32)>>()),
            ], "t_0_0_0 & t_0_1_0 channel 0", (2000, 1000), &format!("/tmp/plots/t_0_0{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        )?;

        //t_label_tensor_channel
        let window = 1000;
        let t_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(
            &FindMotifsRunConfig::new(None, 200, window, None, None).to_applicable(DummyFindMotifsRunConfigCtx, Cpu),
            &vec![t_0.into()], //sig_0 at idx 0
            vec![0], None
        ).t;
        let t_1_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(
            &FindMotifsRunConfig::new(None, 200, window, None, None).to_applicable(DummyFindMotifsRunConfigCtx, Cpu),
            &vec![t_1.into()], //sig_0 at idx 2000
            vec![0], None
        ).t;
        reset_print_options();
        let (mask, Some(distances)) = clc_dist_d2_and_mask(
            t_0_sliced.shallow_clone(),
            t_1_sliced.shallow_clone(),
            true, &mut [LocalMin, LeQuantile {q:0.01, with_mask:true}]

        )? else {
            panic!("no distances")
        };


        debug!("mask=\n{mask}");
        debug!("mask.0.nonzero=\n{}", mask.i((0,0..,0)).nonzero());

        plot_series_with_points(
            (0..distances.size()[2])
                .zip(DistinctColorIter::new())
                .map(|(slice, color)|(format!("0ch slice={}", slice), color,   lin_iter(distances.size()[1]).zip(from::<Vec<f32>>(&distances.i((0, .., slice)))).collect::<Vec<(f32, f32)>>()))
                .collect(),
            "distance", (2000, 1000), &format!("/tmp/plots/distance_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str(),
            &(0..distances.size()[2])
                .flat_map(|slice|
                    lin_iter(distances.size()[1])
                        .zip(from::<Vec<f32>>(&distances.i((0, .., slice))))
                        .filter(|(x, v)|&mask.i((0, *x as i64, slice)).double_value(&[]) == &1.)
                        .collect::<Vec<(f32, f32)>>())
                .collect::<Vec<(f32, f32)>>()
        )?;



        debug!("distances={distances}");

        let channels_dists_flat = distances.flatten(1,2);

        debug!("channels_dists_flat={:?}", channels_dists_flat.size());
        let hists = hist_1_dim(&channels_dists_flat, window);
        let x = lin_t(window as i64);

        let (skewness, kurtosis) = skewness_kurtosis_1_dim(&hists.totype(Float));
        debug!("skewness={skewness}");
        debug!("skewness={kurtosis}");

        // plot_series(
        //     vec![
        //         (format!("0ch"), plotters::prelude::ShapeStyle {
        //             color: GREEN.into(),
        //             filled: false,
        //             stroke_width: 2,
        //         },   vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&hists.i(0))).collect::<Vec<(f32, f32)>>()),//sig_0_v match
        //         (format!("1ch"), plotters::prelude::RED.into(),     vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&hists.i(1))).collect::<Vec<(f32, f32)>>()),//no matches
        //         (format!("2ch"), plotters::prelude::BLUE.into(),    vec_f32_from(&x).into_iter().zip(from::<Vec<f32>>(&hists.i(2))).collect::<Vec<(f32, f32)>>()),//sig_2_v match
        //     ], "hists", (2000, 1000), &format!("/tmp/plots/hists_{:?}.png", chrono::Utc::now().to_rfc3339()).as_str()
        // )?;

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

        let d_0_0 = [0.00_f32, 0.01, 0.02, 0.03, 0.04, 0.05];
        let d_0_1 = [0.10_f32, 0.11, 0.12, 0.13, 0.14, 0.15];

        let t_0 = Tensor::vstack(&[
            Tensor::from_slice(&d_0_0),
            Tensor::from_slice(&d_0_1),
        ]);

        let d_1_0 = [1.00_f32, 1.01, 1.02, 1.03, 1.04, 1.05];
        let d_1_1 = [1.10_f32, 1.11, 1.12, 1.13, 1.14, 1.15];

        let t_1 = Tensor::vstack(&[
            Tensor::from_slice(&d_1_0),
            Tensor::from_slice(&d_1_1),
        ]);


        let t_0_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(&FindMotifsRunConfig::new(None, 1, 3, None, None).to_applicable(DummyFindMotifsRunConfigCtx, Cpu), &vec![t_0.into()], vec![0], None).t;
        let t_1_sliced = ChunkOfSlicedTensors::slice_tensors_chunk(&FindMotifsRunConfig::new(None, 1, 3, None, None).to_applicable(DummyFindMotifsRunConfigCtx, Cpu), &vec![t_1.into()], vec![0], None).t;

        let (mask, Some(dists)) = clc_dist_d2_and_mask(t_0_sliced.shallow_clone(), t_1_sliced.shallow_clone(), true, &mut [])? else {
            panic!("no distances")
        };

        debug!("dists={dists} size={:?}", dists.size());


        assert_eq!(
            _dist_3(&d_0_0[0..3], &d_1_0[0..3]),
            dists.i((0, 0, 0)).double_value(&[]) as f32
        );
        assert_eq!(
            _dist_3(&d_0_0[0..3], &d_1_0[1..4]),
            dists.i((0, 0, 1)).double_value(&[]) as f32
        );
        assert_eq!(
            _dist_3(&d_0_0[0..3], &d_1_0[2..5]),
            dists.i((0, 0, 2)).double_value(&[]) as f32
        );



        Ok(())
    }

    fn _dist_2(p_0:&[f32], p_1:&[f32]) ->f32 {
        (
            (p_0[0] - p_1[0] ).powi(2)+
            (p_0[1] - p_1[1] ).powi(2)
        ).sqrt()
    }


    fn _dist_3(p_0:&[f32], p_1:&[f32]) ->f32 {
        (
            (p_0[0] - p_1[0] ).powi(2) +
            (p_0[1] - p_1[1] ).powi(2) +
            (p_0[2] - p_1[2] ).powi(2)
        ).sqrt()
    }


    #[test]
    fn test_clc_dist_d2_slices() -> Result<(), ThoosaError>{
        let a_00 = [0.00_f32, 0.01, 0.02, 0.02, 0.02];
        let t_00 = Tensor::from_slice(&a_00).view((1, -1));

        let a_01 = [1.00_f32, 1.01, 1.02];
        let t_01 = Tensor::from_slice(&a_01).view((1, -1));


        let a_10 = [2.00_f32, 2.01, 2.02, 2.02];
        let t_10 = Tensor::from_slice(&a_10).view((1, -1));

        let a_11 = [3.00_f32, 0.00, 0.01];
        let t_11 = Tensor::from_slice(&a_11).view((1, -1));

        let run_config = FindMotifsRunConfig::new(None, 1, 2, None, None);
        let run_config_applicable = run_config.to_applicable(DummyFindMotifsRunConfigCtx, Cpu);

        let  (tensor_0, sizes_0) =run_config_applicable.apply(&[
            t_00.into(), t_01.into()
        ][..]).expect("apply");

        let chunk_0 = ChunkOfSlicedTensors::new(
            vec![0,1],
            sizes_0,
            run_config.step(),
            tensor_0
        );

        let  (tensor_1, sizes_1) =run_config_applicable.apply(&[
            t_10.into(), t_11.into()
        ][..]).expect("apply");

        let chunk_1 = ChunkOfSlicedTensors::new(
            vec![2,3],
            sizes_1,
            run_config.step(),
            tensor_1
        );

        assert_eq!([1,6,2][..], chunk_0.t.size());
        assert_eq!([1,5,2][..], chunk_1.t.size());

        let distances = clc_dist_d2(
            chunk_0.t.deref().shallow_clone(),
            chunk_1.t.deref().shallow_clone(),
            1
        )?;

        debug!("distances=\n{distances}");

        assert_eq!(
             _dist_2(&a_00[0..2], &a_10[0..2]),
            distances.double_value(&[0, 0, 0]) as f32
        );

        assert_eq!(
            _dist_2(&a_00[0..2], &a_10[1..3]),
            distances.double_value(&[0, 0, 1]) as f32
        );

        assert_eq!(
            _dist_2(&a_00[0..2], &a_10[2..4]),
            distances.double_value(&[0, 0, 2]) as f32
        );

        assert_eq!(
            _dist_2(&a_00[0..2], &a_11[0..2]),
            distances.double_value(&[0, 0, 3]) as f32
        );

        assert_eq!(
            _dist_2(&a_00[0..2], &a_11[1..3]),
            distances.double_value(&[0, 0, 4]) as f32
        );



        assert_eq!(
            (_dist_2(&a_01[1..3], &a_10[0..2])* 100.).round(),
            (distances.double_value(&[0, 5, 0]) as f32* 100.).round(),
        );

        assert_eq!(
            (_dist_2(&a_01[1..3], &a_10[1..3])* 100.).round(),
            (distances.double_value(&[0, 5, 1]) as f32* 100.).round(),
        );

        assert_eq!(
            (_dist_2(&a_01[1..3], &a_10[2..4])* 100.).round(),
            (distances.double_value(&[0, 5, 2]) as f32* 100.).round(),
        );

        assert_eq!(
            (_dist_2(&a_01[1..3], &a_11[0..2])* 100.).round(),
            (distances.double_value(&[0, 5, 3]) as f32* 100.).round(),
        );

        assert_eq!(
            (_dist_2(&a_01[1..3], &a_11[1..3]) * 100.).round(),
            (distances.double_value(&[0, 5, 4]) as f32 * 100.).round()
        );

        Ok(())
    }

    #[test]
    fn test_clc_dist_d2_0() -> Result<(), ThoosaError>{
        let tensor_0 = Tensor::from_slice(&[0.0, 1.0, 2., 3.]).view((1, -1));
        let tensor_1 = Tensor::from_slice(&[0.0, 1.0, 2., 3.]).view((1, -1));

        let dist = clc_dist_d2(tensor_0, tensor_1, 0)?;

        debug!("dist={:?} dist={dist}", dist.size());

        assert_eq!(0.0, dist.double_value(&[0, 0]));

        Ok(())
    }


    #[test]
    fn test_dist_d2_with_mask_q_05(){

        let slice_0 = Tensor::from_slice(&[
            30., 30.,
            0., 0.,
            40., 40.,
        ]).view((1, 3, 2));
        let slice_1 = Tensor::from_slice(&[
            7., 8.,
            8., 9.,
            9., 10.,
        ]).view((1, 3, 2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        // let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        let (mask, Some(dist_t)) = super::clc_dist_d2_and_mask(slice_0, slice_1, true, &mut [LeQuantile {q:0.2, with_mask: false }]).unwrap() else {
            panic!("no dists");
        };
        // debug!("mask={mask}");
        debug!("dist_t={dist_t}");
        let expected_dists = Tensor::stack(&[
            Tensor::from_slice(&[
                _dist_2(&[ 30., 30.,], &[7., 8.,]),
                _dist_2(&[ 30., 30.,], &[8., 9.,]),
                _dist_2(&[ 30., 30.,], &[9., 10.])
            ]),
            Tensor::from_slice(&[
                _dist_2(&[ 0., 0.,], &[7., 8.,]),
                _dist_2(&[ 0., 0.,], &[8., 9.,]),
                _dist_2(&[ 0., 0.,], &[9., 10.])
            ]),
            Tensor::from_slice(&[
                _dist_2(&[ 40., 40.,], &[7., 8.,]),
                _dist_2(&[ 40., 40.,], &[8., 9.,]),
                _dist_2(&[ 40., 40.,], &[9., 10.])
            ])
        ], 0).unsqueeze(0);//unsqueeze for 0 channel

        assert!(expected_dists.allclose(&dist_t, 0.1, 0.1, false));

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                false, false, false,
                true, true, true,
                false, false, false
            ]).view((1, 3,3)),
            mask
        );
    }

    #[test]
    fn test_dist_d2_with_mask_local_min(){
        // set_max_level(Trace);
        let slice_0 = Tensor::from_slice(&[
            7., 8.,
            7., 8.,
            4., 5.,
            5., 6.,
            5., 6.,

        ]).view((1, 5, 2));
        let slice_1 = Tensor::from_slice(&[//slice_1 split if oom
            0., 0.,
            1., 1.,
            2., 2.,
        ]).view((1, 3, 2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        // let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        let (mask, Some(dist_t)) = super::clc_dist_d2_and_mask(slice_0, slice_1, true, &mut [LocalMin]).unwrap() else {
            panic!("no dists");
        };
        // debug!("mask={mask}");
        debug!("dist_t={dist_t}");
        let expected_dists = Tensor::stack(&[
            Tensor::from_slice(&[
                _dist_2(&[ 7., 8.,], &[0., 0.,]),
                _dist_2(&[ 7., 8.,], &[1., 1.,]),
                _dist_2(&[ 7., 8.,], &[2., 2.,]),

            ]),
            Tensor::from_slice(&[
                _dist_2(&[ 7., 8.,], &[0., 0.,]),
                _dist_2(&[ 7., 8.,], &[1., 1.,]),
                _dist_2(&[ 7., 8.,], &[2., 2.,]),
            ]),
            Tensor::from_slice(&[
                _dist_2(&[ 4., 5.,], &[0., 0.,]),
                _dist_2(&[ 4., 5.,], &[1., 1.,]),
                _dist_2(&[ 4., 5.,], &[2., 2.,]),
            ]),
            Tensor::from_slice(&[
                _dist_2(&[ 5.,6.], &[0., 0.,]),
                _dist_2(&[ 5.,6.], &[1., 1.,]),
                _dist_2(&[ 5.,6.], &[2., 2.,]),
            ]),
            Tensor::from_slice(&[
                _dist_2(&[ 5.,6.], &[0., 0.,]),
                _dist_2(&[ 5.,6.], &[1., 1.,]),
                _dist_2(&[ 5.,6.], &[2., 2.,]),
            ]),
        ], 0).unsqueeze(0);//unsqueeze for 0 channel

        debug!("expected_dists=\n{expected_dists}");

        assert!(expected_dists.allclose(&dist_t, 0.1, 0.1, false));

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
              // 0,0    1,1    2,2
                false, false, false,//7,8
                false, false, false,//7,8
                true , true , true ,//4,5 - min
                false, false, false,//5,6
                false, false, false //5,6
           ]).view((1, 5, 3)),
            mask
        );

    }


    #[test]
    fn test_dist_d2_0_diag(){

        let slice_0 = Tensor::from_slice(&[0., 0., 1., 1., 2., 2.,]).view((1, 3,2));
        let slice_1 = Tensor::from_slice(&[3., 4., 4., 5., 5., 6.,]).view((1, 3,2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        // let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        //set_max_level(Trace);
        let (mask, Some(dist_t)) = clc_dist_d2_and_mask(slice_0, slice_1, true, &mut [LeQuantile {q:1.0, with_mask: false }, Diagonal]).unwrap() else {
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

        debug!("dists_t=\n{dists_t}");

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

        debug!("thresholds_t_broadcasted=\n{thresholds_t_broadcasted}");

        let mask = LE {v:thresholds_t_broadcasted}.make_mask(None, &dists_t);

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false,
                 true, true, false, false, false, false, false
            ]).view((3,5,7)),
            mask
        );
    }

    #[test]
    fn test_acc_mean_mask(){
        set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();
        let mut means = Mutex::new(Tensor::zeros([3], FLOAT_CPU));

        let mask = LeMean {k:1.0, avg: &mut means, total_samples: &mut Mutex::new(Tensor::zeros([3], INT64_CPU)) }.make_mask(None, &dists_t);

        debug!("mask=\n{mask}");
        assert_eq!(
            Tensor::from_slice(&[
                true , true , true , false, false,
                true , true , true , false, false,
                true , true , true , false, false,
                true , true , true , false, false,
                true , true , false, false, false,
                true , true , false, false, false,
                true , true , false, false, false,

                true , true , true , false, false,
                true , true , true , false, false,
                true , true , true , false, false,
                true , true , true , false, false,
                true , true , false, false, false,
                true , true , false, false, false,
                true , true , false, false, false,

                true , true , true , false, false,
                true , true , true , false, false,
                true , true , true , false, false,
                true , true , true , false, false,
                true , true , false, false, false,
                true , true , false, false, false,
                true , true , false, false, false

            ]).view((3,7,5)),
            mask
        );
    }

    #[test]
    fn test_mean_mask_two_chunks(){
        set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let dists_t_0 = dists_t.i((.., .., ..3));
        let dists_t_1 = dists_t.i((.., .., 3..));

        let mut means = Mutex::new(Tensor::zeros([3], FLOAT_CPU));

        let mut total_samples = Mutex::new(Tensor::zeros([3], INT64_CPU));
        let mask = LeMean {k:1.0, avg: &mut means, total_samples: &mut total_samples }.make_mask(None, &dists_t_0);
        let mask = LeMean {k:1.0, avg: &mut means, total_samples: &mut total_samples }.make_mask(None, &dists_t_1);

        debug!("mask=\n{mask}");
        //
        // //means [0.2300, 1.2300, 2.2300]
        // assert_eq!(
        //     Tensor::from_slice(&[
        //         true , true , true , true , true ,
        //         true , true , true , true , true ,
        //         true , true , true , true , true ,
        //         true , true , true , true , true ,
        //         false, false, false, false, false,
        //         false, false, false, false, false,
        //         false, false, false, false, false,
        //
        //         true , true , true , true , true ,
        //         true , true , true , true , true ,
        //         true , true , true , true , true ,
        //         true , true , true , true , true ,
        //         false, false, false, false, false,
        //         false, false, false, false, false,
        //         false, false, false, false, false,
        //
        //         true , true , true , true , true,
        //         true , true , true , true , true,
        //         true , true , true , true , true,
        //         true , true , true , true , true,
        //         false, false, false, false, false,
        //         false, false, false, false, false,
        //         false, false, false, false, false
        //     ]).view((3,7,5)),
        //     mask
        // );
    }

    #[test]
    fn test_mean_mask(){
        set_max_level(Trace);

        const CHANNELS:i64 = 1;

        let mut means_acc = Mutex::new(Tensor::zeros([CHANNELS,1], INT64_CPU));
        let mut total_samples = Mutex::new(Tensor::zeros([CHANNELS,1], INT64_CPU));

        let mask = mean_mask(
            0.5,
            &Tensor::from_slice(&[
                // distances 3x2 slices
                0.2, 0.2,
                0.2, 0.2,
                0.2, 0.2
            ]).view((CHANNELS, 3, 2)),
            None,
            &mut means_acc,
            &mut total_samples
        );

        assert_eq!(
            Tensor::from_slice(&[
                false, false,
                false, false,
                false, false,
            ]).view((1,3,2)).totype(Bool),
            mask
        );

        assert_eq!(
            &Tensor::from_slice(&[0.2]).view((1,-1)).totype(Float),
            means_acc.lock().deref()
        );

        assert_eq!(
            &Tensor::from_slice(&[6]).view((1,-1)).totype(Int64),
            total_samples.lock().deref()
        );


        let mask = mean_mask(
            0.5,
            &Tensor::from_slice(&[
                // distances 3x2 slices
                0.4, 0.4,
                0.4, 0.4,
                0.4, 0.4
            ]).view((CHANNELS, 3, 2)),
            None,
            &mut means_acc,
            &mut total_samples
        );

        assert_eq!(
            Tensor::from_slice(&[
                false, false,
                false, false,
                false, false,
            ]).view((1,3,2)).totype(Bool),
            mask
        );

        assert_eq!(
            &Tensor::from_slice(&[0.3]).view((1,-1)).totype(Float),
            means_acc.lock().deref()
        );

        assert_eq!(
            &Tensor::from_slice(&[12]).view((1,-1)).totype(Int64),
            total_samples.lock().deref()
        );

        let mask = mean_mask(
            0.5,
            &Tensor::from_slice(&[
                // distances 3x2 slices
                0.2, 0.2,
                0.2, 0.1,
                0.1, 0.1
            ]).view((CHANNELS, 3, 2)),
            None,
            &mut means_acc,
            &mut total_samples
        );

        assert_eq!(
            Tensor::from_slice(&[
                false, false,//0.2 > 0.25 * 0.5
                false, true,
                true, true,  //0.1 < 0.25 * 0.5
            ]).view((1,3,2)).totype(Bool),
            mask
        );

        assert_eq!(
            &Tensor::from_slice(&[0.25]).view((1,-1)).totype(Float),
            means_acc.lock().deref()
        );

        assert_eq!(
            &Tensor::from_slice(&[18]).view((1,-1)).totype(Int64),
            total_samples.lock().deref()
        );

    }


    #[test]
    fn test_local_min_mask(){

        set_max_level(Trace);

        let distances = Tensor::from_slice(&[
            // distances 2x3 slices
            0.2, 0.2, 0.1, 0.3, 0.3,
            0.2, 0.2, 0.1, 0.3, 0.3
        ]).view((1, 5, 2));

        debug!("distances=\n{distances}");

        let mask = local_min_mask(&distances);

        debug!("mask=\n{mask}");
    }

    #[test]
    fn test_quantile_mask(){
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let mask = LeQuantile {q:0.5, with_mask: false }.make_mask(None, &dists_t);

        debug!("mask=\n{mask}");

        //means [0.2300, 1.2300, 2.2300]
        assert_eq!(
            Tensor::from_slice(&[
               true , true , true , true , true ,
               true , true , true , true , true ,
               true , true , true , true , true ,
               true , true , true , true , true ,
               false, false, false, false, false,
               false, false, false, false, false,
               false, false, false, false, false,

               true , true , true , true , true ,
               true , true , true , true , true ,
               true , true , true , true , true ,
               true , true , true , true , true ,
               false, false, false, false, false,
               false, false, false, false, false,
               false, false, false, false, false,

               true , true , true , true , true,
               true , true , true , true , true,
               true , true , true , true , true,
               true , true , true , true , true,
               false, false, false, false, false,
               false, false, false, false, false,
               false, false, false, false, false
            ]).view((3,7,5)),
            mask
        );
    }

    #[test]
    fn test_quantile_mask_with_empty_pre_mask(){
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let mask = LeQuantile {q:0.5, with_mask: true }.make_mask(
            Some(Tensor::from_slice(&[
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,

                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,

                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true
            ]).view((3,7,5))),
            &dists_t
        );

        debug!("mask=\n{mask}");


        assert_eq!(
            Tensor::from_slice(&[
                true , true , true , true , true ,
                true , true , true , true , true ,
                true , true , true , true , true ,
                true , true , true , true , true ,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false,

                true , true , true , true , true ,
                true , true , true , true , true ,
                true , true , true , true , true ,
                true , true , true , true , true ,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false,

                true , true , true , true , true,
                true , true , true , true , true,
                true , true , true , true , true,
                true , true , true , true , true,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false
            ]).view((3,7,5)),
            mask
        );
    }

    #[test]
    fn test_quantile_mask_with_pre_mask_1(){
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let mask = LeQuantile {q:0.5, with_mask: true }.make_mask(
            Some(Tensor::from_slice(&[
                false, false, false, false, false,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,

                false, false, false, false, false,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,

                false, false, false, false, false,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true
            ]).view((3,7,5))),
            &dists_t
        );

        debug!("mask=\n{mask}");


        assert_eq!(
            Tensor::from_slice(&[
                false, false, false, false, false,
                true , true , true , true , true ,
                true , true , true , true , true ,
                true , true , true , true , true ,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false,

                false, false, false, false, false,
                true , true , true , true , true ,
                true , true , true , true , true ,
                true , true , true , true , true ,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false,

                false, false, false, false, false,
                true , true , true , true , true,
                true , true , true , true , true,
                true , true , true , true , true,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false
            ]).view((3,7,5)),
            mask
        );
    }


    #[test]
    fn test_quantile_mask_with_pre_mask_2(){
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let mask = LeQuantile {q:0.5, with_mask: true }.make_mask(
            Some(Tensor::from_slice(&[

                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                false, false, false, false, false,


                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                false, false, false, false, false,


                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                true, true, true, true, true,
                false, false, false, false, false,
            ]).view((3,7,5))),
            &dists_t
        );

        debug!("mask=\n{mask}");


        assert_eq!(
            Tensor::from_slice(&[
                true , true , true , true , true ,
                true , true , true , true , true ,
                true , true , true , true , true ,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false,

                true , true , true , true , true ,
                true , true , true , true , true ,
                true , true , true , true , true ,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false,

                true , true , true , true , true ,
                true , true , true , true , true,
                true , true , true , true , true,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false,
                false, false, false, false, false
            ]).view((3,7,5)),
            mask
        );
    }

    // tensor = [
    //     0.0,  0.0,  2.0,  0.0,      10.0;
    //     0.01, 0.0,  2.0,  0.01,     9.0;
    //     0.02, 0.02, 2.0,  0.02,     8.0;
    //     0.03, 0.02, 2.0,  0.03,     7.0;
    //     0.04, 0.02, 2.0,  0.04,     6.0;
    //     0.05, 0.05, 2.1,  inf,      5.0;
    //     0.06, 0.1,  2.2,  inf,      4.0
    // ];
    //
    // % Initialize variables
    // [numRows, numCols] = size(tensor);
    // quantiles = zeros(1, numCols); % To store 0.5 quantiles
    // closestIndices = zeros(1, numCols); % To store the closest row indices
    //
    // % Calculate 0.5 quantiles and closest indices for each column
    // for col = 1:numCols
    //     colValues = tensor(:, col);
    //     finiteValues = colValues(isfinite(colValues)); % Ignore Inf values
    //     quantiles(col) = median(finiteValues); % Compute 0.5 quantile (median)
    //
    //     % Find the index of the value closest to the quantile
    //     [~, closestIndex] = min(abs(colValues - quantiles(col)));
    //     closestIndices(col) = closestIndex;
    // end
    //
    // % Display results
    // disp('Quantiles (0.5):');
    // disp(quantiles);
    //
    // disp('Closest Indices:');
    // disp(closestIndices);
    //
    #[test]
    fn test_quantile_mask_diff_lengths(){
        // set_max_level(Trace);

        let dists_t = Tensor::from_slice(&[
            0.0,  0.0,  2.0, 0.0,            10.0,
            0.01, 0.0,  2.0, 0.01,           9.0,
            0.02, 0.02, 2.0, 0.02,           8.0,
            0.03, 0.02, 2.0, 0.03,           7.0,
            0.04, 0.02, 2.0, 0.04,           6.0,
            0.05, 0.05, 2.1, f32::INFINITY,  5.0,
            0.06, 0.1,  2.2, f32::INFINITY,  4.0
        ]).view((1,7,5));


        let mask = LeQuantile {q:0.5, with_mask: true }.make_mask(None, &dists_t);

        debug!("mask=\n{mask}");
        //Quantiles (0.5):             (matlab ignores infs)
        //     0.0300    0.0200    2.0000    0.0200    7.0000
        //
        // Closest Indices:
        //       4       3     1      3      4

        assert_eq!(
            Tensor::from_slice(&[
                true , true , true , true , false,
                true , true , true , true , false,
                true , true , true , true , false,
                true , true , true , true , true ,
                false, true , true , false, true ,
                false, false, false, false, true ,
                false, false, false, false, true

            ]).view((1,7,5)),
            mask
        );
    }

    #[test]
    fn test_quantile_mask_diff_lengths_with_pre_mask(){
        // set_max_level(Trace);

        let dists_t = Tensor::from_slice(&[
            100., 100., 100., 100.,           100.,
            0.0,  0.0,  2.0,  0.0,            10.0,
            0.01, 0.0,  2.0,  0.01,           9.0,
            0.02, 0.02, 2.0,  0.02,           8.0,
            0.03, 0.02, 2.0,  0.03,           7.0,
            0.04, 0.02, 2.0,  0.04,           6.0,
            0.05, 0.05, 2.1,  f32::INFINITY,  5.0,
            0.06, 0.1,  2.2,  f32::INFINITY,  4.0,
            100., 100., 100., 100.,           100.
        ]).view((1,9,5));


        let mask = LeQuantile {q:0.5, with_mask: true }.make_mask(Some(
            Tensor::from_slice(&[
                false, false, false, false, false,
                true,  true,  true,  true,  true,
                true,  true,  true,  true,  true,
                true,  true,  true,  true,  true,
                true,  true,  true,  true,  true,
                true,  true,  true,  true,  true,
                true,  true,  true,  true,  true,
                true,  true,  true,  true,  true,
                false, false, false, false, false

            ]).view((1,9,5))
        ), &dists_t);

        debug!("mask=\n{mask}");


        assert_eq!(
            Tensor::from_slice(&[
                false, false, false, false, false,

                true , true , true , true , false,
                true , true , true , true , false,
                true , true , true , true , false,
                true , true , true , true , true ,
                false, true , true , false, true ,
                false, false, false, false, true ,
                false, false, false, false, true,

                false, false, false, false, false

            ]).view((1,9,5)),
            mask
        );
    }


    fn tensor_3_7_5_lin() -> Tensor {
        Tensor::from_slice(&[
    //channel 0
    //slice_1: 0       1       2       3       4
    //slice_0
    /*0*/   0.0000, 0.1000, 0.2000, 0.3000, 0.4000,
    /*1*/   0.0100, 0.1100, 0.2100, 0.3100, 0.4100,
    /*2*/   0.0200, 0.1200, 0.2200, 0.3200, 0.4200,
    /*3*/   0.0300, 0.1300, 0.2300, 0.3300, 0.4300,
    /*4*/   0.0400, 0.1400, 0.2400, 0.3400, 0.4400,
    /*5*/   0.0500, 0.1500, 0.2500, 0.3500, 0.4500,
    /*6*/   0.0600, 0.1600, 0.2600, 0.3600, 0.4600,
            //channel 1
            1.0000, 1.1000, 1.2000, 1.3000, 1.4000,
            1.0100, 1.1100, 1.2100, 1.3100, 1.4100,
            1.0200, 1.1200, 1.2200, 1.3200, 1.4200,
            1.0300, 1.1300, 1.2300, 1.3300, 1.4300,
            1.0400, 1.1400, 1.2400, 1.3400, 1.4400,
            1.0500, 1.1500, 1.2500, 1.3500, 1.4500,
            1.0600, 1.1600, 1.2600, 1.3600, 1.4600,
            //channel 2
            2.0000, 2.1000, 2.2000, 2.3000, 2.4000,
            2.0100, 2.1100, 2.2100, 2.3100, 2.4100,
            2.0200, 2.1200, 2.2200, 2.3200, 2.4200,
            2.0300, 2.1300, 2.2300, 2.3300, 2.4300,
            2.0400, 2.1400, 2.2400, 2.3400, 2.4400,
            2.0500, 2.1500, 2.2500, 2.3500, 2.4500,
            2.0600, 2.1600, 2.2600, 2.3600, 2.4600
        ]).view((3, 7, 5))
    }


    #[test]
    fn test_q(){
        let samples = Tensor::from_slice(&[0.0000, 0.0100, 0.0200, 0.0300, 0.0400]).view((1,1,-1));
        let q_t = samples.quantile_scalar(0.5, 2, true, "lower");

        debug!("full: q_t=\n{q_t}");

        let samples = Tensor::from_slice(&[0.0000, 0.0100, 0.0200, 0.0300, 0.0400, f64::INFINITY,    f64::INFINITY]).view((1,1,-1));

        let q_t = samples.quantile_scalar(0.5, 2, true, "lower");

        debug!("masked: q_t=\n{q_t}");
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
        let tensor = slice_t(&Tensor::from_slice(&[1., 2., 3., 4., 5.]).view((1, -1)), 1, 2);

        assert_eq!(Tensor::from_slice(&[1., 2., 2., 3., 3., 4., 4., 5.,]).view((4, 1, 2)), tensor);
    }

    #[test]
    fn test_average_euclidean_distance() -> Result<(), ThoosaError>{
        let d = [
            1., 2., 3.,
            4., 5., 6.,
            7., 8., 9.
        ];
        let ab = _dist_3(&d[0..3], &d[3..6]);
        let ac = _dist_3(&d[0..3], &d[6..9]);
        let bc = _dist_3(&d[3..6], &d[6..9]);

        println!("ab={ab} ac={ac} bc={bc}");

        let test_avg = (
            ab + ac + bc
        ) / 3.;

        println!("test_avg={test_avg}");


        let x = Tensor::from_slice(&d).view((3, 3));
        println!("x={x}");
        let avg_dist = average_euclidean_distance(&x)?;
        assert_eq!(
            test_avg,
            avg_dist
        );

        Ok(())
    }


    #[test]
    fn test_average_euclidean_distance_max_size() -> Result<(), ThoosaError> {
        set_max_level(Debug);
        use cudarc::driver::{CudaDevice};
        let _device = CudaDevice::new(0)?;

        let mut free_mem: usize = 0;
        let mut total_mem: usize = 0;

        unsafe {
            let lib = libloading::Library::new("libtorch_cuda.so").expect("no libtorch_cuda.so");

            let cudarc_lib = cudarc::driver::sys::lib();
            cudarc_lib.cuMemGetInfo_v2(&mut free_mem as *mut usize, &mut total_mem as *mut usize);
        }
        debug!("Device 0 free_mem: {free_mem}, total_mem: {total_mem}");

        let size = ((free_mem / size_of::<f32>()) as f64).sqrt() as i64 / 32;
        let mut big_t = Tensor::rand(size * size, FLOAT_CUDA).view((size, size));

        debug!("slices_t.size()={:?}", big_t.size());

        let mut avg_r: f32 = f32::NAN;

        run_split_on_oom(
            SplitQueue::<2, SplittableTensor::<0>>::new(SplittableTensor::<0>(big_t)),
            // Reducer::new(SplittableTensor::<0>(big_t)),
            |t| {
                debug!("average_euclidean_distance 0");
                let r = average_euclidean_distance(&t.as_ref());
                debug!("average_euclidean_distance r={r:?}");
                r
            },
            |v, t| {
                debug!("v={v:?}");
                avg_r = v;
            },
            2,
        )?;

        debug!("v={avg_r:?}");

        assert!(avg_r.is_normal());

        Ok(())
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
        // reset_print_options();
        // set_max_level(Trace);
        
        let input_n =  1000;


        let ch_0 = (0..input_n).map(|x| (2. * PI * (x as f32 / input_n as f32)).sin()).collect::<Vec<f32>>(); //sin_v(input_n, 1.);
        let ch_1 = (0..input_n).map(|x| (4. * PI * (x as f32  / input_n as f32)).sin()).collect::<Vec<f32>>(); //sin_v(input_n, 0.5);

        let x = lin_t(input_n);
        /*plot_series(
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
        ], 0).view((2, input_n));//unsqueeze to make it [channels x t_0 slices x t_1 slices]

        let mask = LocalMin.make_mask(None, &distances);

        trace!("mask=\n{mask}");
        debug!("mask.nonzero=\n{}", mask.nonzero());

        assert_eq!(
            Tensor::from_slice(&[
               0,   0,
               0, 750,
               1,   0,
               1, 375,
               1, 875,
            ]).view((5, 2)),
            mask.nonzero().totype(Int)
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