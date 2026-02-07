use core::f32;
use std::fmt::format;
use log::Level::Trace;
use log::{debug, error, log_enabled, trace, warn};
use parking_lot::{Mutex, RwLock};
use std::ops::{Add, ControlFlow};
use std::sync::Arc;
use std::sync::atomic::AtomicI64;
use tch::Device::{Cpu, Cuda};
use tch::Kind;
use tch::Kind::Bool;
use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::{IndexOp, TchError, Tensor};
use thoosa_common::utils::{skewness_kurtosis_1_dim, tensor_mem_size, z_norm_2, z_norm_dims};
use thoosa_shared::ThoError;
use crate::pearson_corr::pearson_corr_distance_matrix_0_2;

#[allow(dead_code)]
pub fn z_normalized_euclidean(t_0: &Tensor, t_1: &Tensor, dim: i64) -> Result<Tensor, ThoError> {
    Ok(z_norm_2(t_0, 1)?
        .f_sub(&z_norm_2(t_1, 1)?)?
        .f_pow_tensor_scalar(2)?
        .f_sum_dim_intlist(dim, true, Kind::Float)?
        .f_pow_tensor_scalar(0.5)?)
}

pub fn dist_d2(slice_0: &Tensor, slice_1: &Tensor) -> Result<Tensor, ThoError> {
    //https://github.com/pytorch/pytorch/blob/96487d0d1f638ae9c665dcd3d4b79e7e2a976f59/aten/src/ATen/native/cuda/DistanceKernel.cu#L188
    //0 -use_mm_for_euclid_dist_if_necessary
    //1 -use_mm_for_euclid_dist
    //2 -donot_use_mm_for_euclid_dist
    assert_eq!(slice_0.size()[1], slice_1.size()[1]); //same slice size
    Ok(Tensor::f_cdist(slice_0, slice_1, 2.0, 0)?)
}

//TODO use distance's histogram kurtosis and skew to scale down q_t? If histogram flat - prob. of useful patterns lower
#[derive(Debug)]
pub enum MaskMaker<'a> {
    LeQuantile {
        q: f32,
        with_mask: bool,
    },
    LeRunningMean {
        k: f32,
        avg: &'a Mutex<Tensor>,
        total_samples: &'a Mutex<Tensor>,
    },
    Diagonal,
    Tril,
    Le {
        v: Tensor,
    }, //channels x thresholds
    Gt {
        v: Tensor,
    }, //channels x thresholds
    LocalMin { dim: i64, radius: u64 },
}

impl<'a> MaskMaker<'a> {
    pub fn make_mask(&mut self, prev_mask: Option<Tensor>, dists: &Tensor) -> Result<Tensor, ThoError> {
        debug!("make_mask: prev_mask={prev_mask:?} dists={dists:?} self={self:?}");

        match self {
            MaskMaker::LeQuantile { q, with_mask } => Self::apply_prev(
                &prev_mask,
                quantile_mask(
                    *q as f64,
                    dists,
                    if *with_mask { prev_mask.as_ref() } else { None },
                )?,
            ),
            MaskMaker::Diagonal => {
                let has_prev = prev_mask.is_some();
                if log_enabled!(Trace) && has_prev {
                    trace!(
                        "MaskMaker::make_mask Diagonal: perv_mask={}",
                        prev_mask.as_ref().map(|v| v.shallow_clone()).unwrap()
                    );
                }

                let prev = match prev_mask {
                    Some(prev) => prev,
                    None => Tensor::f_ones(dists.size(), (Bool, dists.device()))?,
                };
                _ = prev.f_diagonal(0, 1, 2)?.f_fill_(0)?;

                trace!("MaskMaker::make_mask Diagonal: 2. perv_mask.is_some()={has_prev} prev={prev}");

                Ok(prev)
            }
            MaskMaker::Tril => {
                let prev = match prev_mask {
                    Some(prev) => prev,
                    None => Tensor::f_ones(dists.size(), (Bool, dists.device()))?,
                };
                prev.f_triu(1).map_err(|e|e.into())
            }
            MaskMaker::Le { v } => {
                let le = match dists.f_le_tensor(v) {
                    Ok(v) => v,
                    Err(e) => {
                        error!("MaskMaker::Le: e={e:?}");
                        panic!("MaskMaker::Le: e={e:?}");
                    }
                };
                Self::apply_prev(&prev_mask, le)
            }
            MaskMaker::Gt { v } => Self::apply_prev(&prev_mask, dists.f_gt_tensor(v)?),
            MaskMaker::LocalMin { dim, radius } => {
                Self::apply_prev(&prev_mask, local_min_mask_suppressed(&dists, *dim, *radius as i64)?)
            },
            MaskMaker::LeRunningMean {
                k,
                avg,
                total_samples,
            } => Self::apply_prev(
                &prev_mask,
                mean_mask(*k as f64, dists, prev_mask.as_ref(), *avg, total_samples)?,
            ),
        }
    }

    fn apply_prev(perv_mask: &Option<Tensor>, mask: Tensor) -> Result<Tensor, ThoError> {
        trace!("apply_prev: perv_mask={perv_mask:?} mask=\n{mask}");
        match perv_mask {
            None => Ok(mask),
            Some(prev) => match prev.f_logical_and(&mask) {
                Ok(v) => Ok(v),
                Err(e) => {
                    error!("prev.f_logical_and(&mask) failed: {:?}", e);
                    // panic!("prev.f_logical_and(&mask)");
                    Err(e.into())
                }
            },
        }
    }
}
/*
    t_0, t_1 - channels x slices x slice_len
*/
/// returns bool mask and if with_distances is true distances
#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="distance::clc_dist_d2_and_mask"))]
pub fn clc_dist_d2_and_mask(
    t_0: Tensor,
    t_1: Tensor,
    with_distances: bool,
    mask_makers: &mut [MaskMaker],
) -> Result<(Tensor, Option<Tensor>), ThoError> {
    let device = t_0.device();

    if t_0.size()[1] < 2 /*|| t_1.size()[1] < 2 TODO - when calculating distances between slices and motifs there can be single motif?*/ {
        return Err(ThoError::new(format!(
            "not enough slices in dimension 0 to calculate distances and mask: t_0.size={:?} t_1.size={:?}",
            t_0.size(),
            t_1.size()
        )));
    }
    if t_1.size()[1] < 2 {
        return Err(ThoError::new(format!(
            "not enough slices in dimension 1 to calculate distances and mask: t_0.size={:?} t_1.size={:?}",
            t_0.size(),
            t_1.size()
        )));
    }

    //TODO options for z_norm_dims and pearson_corr

    // let distances = Tensor::f_cdist(
    //     &z_norm_dims(&t_0, &[2], 1e-8)?,
    //     &z_norm_dims(&t_1, &[2], 1e-8)?,
    //     2.0, 0
    // )?;

    // let distances = pearson_corr_distance_matrix_0_2(&t_0, &t_1, 1e-8)?;

    let distances = Tensor::f_cdist(&t_0, &t_1, 2.0, 0)?;

    debug!("clc_dist_d2_and_mask: t_0.size={:?}, t_0.device={:?}, t_1.size={:?}, t_1.device={:?}",t_0.size(), t_0.device(), t_1.size(), t_1.device());
    debug!("clc_dist_d2_and_mask: mask_makers={mask_makers:?} distances size={:?}",distances.size());
    trace!("clc_dist_d2_and_mask: distances=\n{distances} device={:?}",distances.device());

    let mut mask_opt: Option<Tensor> = None;
    for maker in mask_makers.iter_mut() {
        debug!("clc_dist_d2_and_mask: mask={mask_opt:?}");

        let new_mask = maker.make_mask(mask_opt.take(), &distances)?;
        mask_opt = Some(new_mask);
    }

    let mask = match mask_opt {
        Some(mask) => mask,
        None => Tensor::f_ones(distances.size(), (Bool, device))?,
    };

    Ok((
        mask,
        if with_distances { Some(distances) } else { None },
    ))
}

/// distances - [channels x slices 0 x slices 1]
fn quantile_mask(q: f64, distances: &Tensor, pre_mask: Option<&Tensor>) -> Result<Tensor, ThoError> {
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
                let slice_1_t = distances.f_i((channel, .., slice_1))?.f_masked_select(&pm.f_i((
                    channel,
                    ..,
                    slice_1,
                ))?)?;
                let q_t = slice_1_t.f_quantile_scalar(q, 0, true, "lower")?;
                trace!(
                    "quantile_mask: channel={channel} slice_1={slice_1} slice_1_t=\n{slice_1_t} q_t={q_t}"
                );
                quantiles.push(q_t);
            }
        }
        Tensor::f_stack(quantiles.as_slice(), 0)?.f_view((distances_size[0], 1, distances_size[2]))?
    } else {
        distances
            .f_transpose(1, 2)?
            .f_quantile_scalar(q, 2, true, "lower")?
            .f_transpose(1, 2)?
    };

    // let q_t = distances.quantile_scalar(q, 2, true, "lower");
    trace!("quantile_mask: full. q_t=\n{q_t}");
    debug!(
        "quantile_mask: samples.size={:?} q_t.size={:?}",
        distances.size(),
        q_t.size()
    );

    let le = distances.f_le_tensor(&q_t)?; //channels x motifs x num_slices

    trace!("quantile_mask:  le=\n{le}");

    Ok(le)
}

///
/// * distances - [channels x slices 0 x slices 1]
/// * returns - Bool tensor of same size with true for local minimums for  slice 1 along slices 0.
fn local_min_mask(distances: &Tensor, dim: i64) -> Result<Tensor, ThoError> {
    debug!("local_min_mask: distances.size={:?} dim={dim}", distances.size());
    trace!("local_min_mask: distances=\n{distances}");
    let distances_size = distances.size();

    let first_min = distances
        .f_narrow(dim, 0, 1)
        .map_err(|e| {
            debug!("local_min_mask: f_narrow e={e:?}");
            error!("local_min_mask: f_narrow e={e:?}");
            e
        })?
        .f_lt_tensor(&distances.f_narrow(dim, 1, 1)?).map_err(|e| {
            debug!("local_min_mask: f_lt_tensor e={e:?}");
            error!("local_min_mask: f_lt_tensor e={e:?}");
            e
        })?;

    trace!("local_min_mask: first_min=\n{first_min}");

    let last_min = distances
        .f_narrow(dim, distances_size[dim as usize] - 1, 1)?
        .f_lt_tensor(&distances.f_narrow(dim, distances_size[dim as usize] - 2, 1)?)?;

    trace!("local_min_mask: last_min=\n{first_min}");

    let mid = distances.f_narrow(dim, 1, distances_size[dim as usize] - 2)?;
    let left = distances.f_narrow(dim, 0, distances_size[dim as usize] - 2)?;
    let right = distances.f_narrow(dim, 2, distances_size[dim as usize] - 2)?;

    trace!("local_min_mask: mid=\n{mid}");
    trace!("local_min_mask: left=\n{left}");
    trace!("local_min_mask: right=\n{right}");

    let is_less_than_left = mid.f_lt_tensor(&left)?;
    let is_less_than_right = mid.f_le_tensor(&right)?;

    trace!("local_min_mask: is_less_than_left=\n{is_less_than_left}");
    trace!("local_min_mask: is_less_than_right=\n{is_less_than_right}");

    let local_min_mask = is_less_than_left.f_logical_and(&is_less_than_right)?;

    let full_mask = Tensor::f_cat(&[first_min, local_min_mask, last_min], dim)?;

    debug!("local_min_mask: full_mask.size={:?}", full_mask.size());

    Ok(full_mask)
}

/// Minimum over a sliding window of size (2*radius+1) along `dim`.
/// Implemented as: minpool(x) = -maxpool(-x) using max_pool2d on a [B,1,1,L] view.
fn window_min_along_dim(x: &Tensor, dim: i64, radius: i64) -> Result<Tensor, ThoError> {
    if radius <= 0 {
        return Ok(x.shallow_clone());
    }

    let sizes = x.size();
    let ndim = sizes.len() as i64;
    let dim = if dim < 0 { dim + ndim } else { dim };
    if dim < 0 || dim >= ndim {
        return Err(ThoError::new(format!("window_min_along_dim: invalid dim={dim} for ndim={ndim}")));
    }

    // permute so that `dim` becomes the last dimension
    let mut perm: Vec<i64> = (0..ndim).filter(|&d| d != dim).collect();
    perm.push(dim);

    let x_perm = x.f_permute(&perm)?;

    let mut perm_sizes = x_perm.size();
    let l = perm_sizes.pop().unwrap(); // last dim length
    let b: i64 = perm_sizes.iter().product(); // flatten all leading dims

    // [B, 1, L] -> [B, 1, 1, L]
    let x_flat = x_perm.f_reshape(&[b, 1, l])?;
    let x_4d = x_flat.f_unsqueeze(2)?;

    let k = 2 * radius + 1;

    // minpool(x) = -maxpool(-x)
    let neg = x_4d.f_neg()?;
    let pooled = neg.f_max_pool2d(&[1, k], &[1, 1], &[0, radius], &[1, 1], false)?;
    let win_min_4d = pooled.f_neg()?;

    // back: [B, 1, 1, L] -> [.., L] -> unpermute to original layout
    let win_min_flat = win_min_4d.f_squeeze_dim(2)?;
    let win_min_perm = win_min_flat.f_reshape(&x_perm.size())?;

    // invert permutation
    let mut inv = vec![0i64; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p as usize] = i as i64;
    }

    Ok(win_min_perm.f_permute(&inv)?)
}

/// Keep only "well-separated" local minima:
/// a point must be a local minimum AND the minimum in its ±radius neighborhood.
fn local_min_mask_suppressed(distances: &Tensor, dim: i64, radius: i64) -> Result<Tensor, ThoError> {
    let local = local_min_mask(distances, dim)?;

    if radius <= 0 {
        return Ok(local);
    }

    let win_min = window_min_along_dim(distances, dim, radius)?;

    // `win_min` comes from the input values (via pooling), so exact equality is typically fine here.
    let is_best_in_window = distances.f_eq_tensor(&win_min)?;

    Ok(local.f_logical_and(&is_best_in_window)?)
}

/// distances - [channels x slices 0 x slices 1]
/// calculates accumulated means between distances
fn mean_mask(
    k: f64,
    distances: &Tensor,
    pre_mask: Option<&Tensor>,
    means_acc: &Mutex<Tensor>,
    total_samples: &Mutex<Tensor>,
) -> Result<Tensor, ThoError> {
    let distances_size = distances.size();

    debug!("mean_mask: k={k} distances={distances:?} pre_mask={pre_mask:?} means_acc={means_acc:?} total_samples={total_samples:?}");
    trace!("mean_mask: after pre_mask distances=\n{distances}");

    let (mean_t, counts_t) = if let Some(pm) = pre_mask {
        let mut means = Vec::with_capacity(distances_size[1] as usize);
        let mut counts = Vec::with_capacity(distances_size[1] as usize);

        for channel in 0..distances_size[0] {
            let channel_dists_t =
                distances
                    .f_i((channel, .., ..))?
                    .f_masked_select(&pm.f_i((channel, .., ..))?)?;
            let mean = channel_dists_t.f_mean(Kind::Float)?;
            trace!("mean_mask: channel={channel} channel_dists_t=\n{channel_dists_t} mean={mean}");
            means.push(mean);
            counts.push(channel_dists_t.size()[0])
        }
        (
            Tensor::f_stack(means.as_slice(), 0)?.f_view((distances_size[0], 1))?,
            Tensor::f_from_slice(counts.as_slice())?
                .f_to_device(distances.device())?
                .f_view((distances_size[0], 1))?,
        )
    } else {
        (
            distances.f_mean_dim(&[1i64, 2i64][..], false, Kind::Float)?,

            Tensor::f_from_slice(&[distances_size[1] * distances_size[2]])?
                .f_to_device(distances.device())?
                .f_broadcast_to(distances_size[0])?, //  Tensor of size [channels x 1] with values= distances[1] * distances[2] in each channel
        )
    };

    trace!("mean_mask: full. mean_t=\n{mean_t}");

    debug!("mean_mask: samples.size={:?} mean_t.size={:?}",distances.size(),mean_t.size());

    debug!("mean_mask: total_samples={total_samples:?} counts={counts_t:?}");
    debug!("mean_mask: total_samples.device={:?} counts.device={:?}",total_samples.lock().device(),counts_t.device());

    //TODO remove or use copy_ctx
    let (mean_t, counts_t) = (
        mean_t.f_to_device(distances.device())?,
        counts_t.f_to_device(distances.device())?,
    );

    //TODO update means_acc
    // total_means = (total_means * total_samples + current_means * batch_size_i)
    //     / (total_samples + batch_size_i);
    // total_samples += batch_size_i;

    let mut means_guard = means_acc.lock();
    debug!("new_total_means: 0. means_guard=\n{means_guard} ");

    let mut total_samples_guard = total_samples.lock();
    debug!("new_total_means: 1. total_samples_guard=\n{total_samples_guard} ");

    // let means_device = total_samples_guard.device();

    let count_t_added =  counts_t.f_add(&total_samples_guard)?;

    debug!("new_total_means: 1. count_t_added=\n{count_t_added} device={:?} thread name={:?}",count_t_added.device(),std::thread::current().name());

    let weighted = means_guard.f_multiply(&total_samples_guard)?;
    let mean_scaled = mean_t.f_multiply(&counts_t)?;
    *means_guard = weighted.f_add(&mean_scaled)?.f_div(&count_t_added)?; //.to_device(means_device);
    debug!("new_total_means: 2.0 means_guard=\n{means_guard}");

    let _ = total_samples_guard.f_add_(&counts_t)?; //.to_device(means_device);

    debug!("new_total_means: 2.1 total_samples_guard=\n{total_samples_guard}");

    let tensor_0 = means_guard.f_multiply_scalar(k)?;

    debug!("new_total_means: tensor_0=\n{tensor_0}");

    let tensor_1 = tensor_0.f_view((-1, 1, 1))?;
    debug!("new_total_means: tensor_1=\n{tensor_1}");

    let tensor_2 =  tensor_1.f_broadcast_to(distances_size)?;
    trace!("new_total_means: tensor_2=\n{tensor_2}");
    debug!("new_total_means: distances.dev={:?} tensor_1.device={:?}",distances.device(),tensor_1.device());

    let le = distances
        .f_le_tensor(&tensor_2)
        .map_err(|e| {
            error!("distances.f_le_tensor: e={e:?}");
            e
        })?; //channels x motifs x num_slices

    trace!("mean_mask:  le=\n{le}");

    Ok(le)
}

pub fn clc_dist_d2(t_0: Tensor, t_1: Tensor, dim: i64) -> Result<Tensor, ThoError> {
    let t_0 = t_0.f_unsqueeze(dim + 1)?; // channels x num_slices x 1 x slice_len
    let t_1 = t_1.f_unsqueeze(dim)?; // channels x 1 x num_slices x slice_len

    debug!(
        "clc_dist_d2: t_0.size()={:?} t_0.mem={:?}  t_1.size()={:?} t_1.mem={:?}",
        t_0.size(),
        tensor_mem_size(&t_0),
        t_1.size(),
        tensor_mem_size(&t_1)
    );

    let mut diff = t_0.f_subtract(&t_1)?;
    drop(t_0);
    drop(t_1);

    let d = diff.f_pow_(2)?;
    drop(diff);

    let s = d.f_sum_dim_intlist(dim + 2, false, Kind::Float)?;
    drop(d);

    Ok(s.f_sqrt()?)
}

pub fn slice_t(t_0: &Tensor, step: u32, window: u32) -> Tensor {
    debug!(
        "slice_t:  t_0.size()={:?} step={step} window={window}",
        t_0.size()
    );

    let tensors_vec = (0..=t_0.size()[1] - window as i64)
        .step_by(step as usize)
        .map(|i| t_0.i((.., i..i + window as i64)))
        .collect::<Vec<Tensor>>();
    trace!("slice_t: tensors_vec={tensors_vec:?}");
    Tensor::stack(&tensors_vec, 0)
}

pub fn average_euclidean_distance(t: &Tensor) -> Result<f32, ThoError> {
    let n = t.size()[0];
    debug!("average_euclidean_distance: n={n}");

    // Step 2: Compute squared norms of each vector
    let squared_norms = t
        .f_pow_tensor_scalar(2)?
        .f_sum_dim_intlist(1, true, Kind::Float)?;

    // Step 3: Compute squared distances matrix D
    let squared_norms_sum = squared_norms.f_add(&squared_norms.f_transpose(0, 1)?)?;
    drop(squared_norms);

    // Step 1: Compute the Gram matrix (dot products between vectors)
    let gram = t.f_mm(&t.f_transpose(0, 1)?)?;
    let gram = gram.f_add(&gram)?;

    let dists_squared: Tensor = squared_norms_sum.f_subtract(&gram)?;
    drop(gram);
    drop(squared_norms_sum);

    let dists_squared = dists_squared.f_clamp_min(0.0)?; // Replace negative values with 0
    // Step 4: Extract upper triangle without diagonal
    let triu_indices = Tensor::f_triu_indices(n, n, 1, (Kind::Int64, t.device()))?; // Indices of upper triangle without diagonal
    let i = triu_indices.f_get(0)?;
    let j = triu_indices.f_get(1)?;

    let pairwise_squared_dists = dists_squared
        .f_index_select(0, &i)?
        .f_gather(1, &j.f_unsqueeze(1)?, false)?
        .f_squeeze()?;
    drop(dists_squared);
    drop(i);
    drop(j);

    // Step 5: Compute square roots to get Euclidean distances
    trace!("average_euclidean_distance: pairwise_squared_dists=\n{pairwise_squared_dists}");
    let clc_dist_d2s = pairwise_squared_dists.f_sqrt()?;
    trace!("average_euclidean_distance: pairwise_squared_dists=\n{pairwise_squared_dists}");
    drop(pairwise_squared_dists);

    // Step 6: Compute the mean of the distances
    let mean_distance = clc_dist_d2s.f_mean(Kind::Float)?;

    debug!("average_euclidean_distance: mean_distance=\n{mean_distance}");
    trace!("average_euclidean_distance: clc_dist_d2s=\n{clc_dist_d2s}");

    Ok(mean_distance.f_double_value(&[])? as f32)
}

#[cfg(test)]
mod distance_tests {
    use crate::distance::MaskMaker::{Diagonal, Le, LeQuantile, LeRunningMean, LocalMin, Tril};
    use crate::distance::{
        MaskMaker, average_euclidean_distance, clc_dist_d2, clc_dist_d2_and_mask, local_min_mask,
        local_min_mask_suppressed, mean_mask, slice_t, z_normalized_euclidean,
    };
    use chrono::Utc;
    use log::__private_api::enabled;
    use log::LevelFilter::{Debug, Trace};
    use log::{debug, max_level, set_max_level, trace};
    use parking_lot::Mutex;
    use plotters::prelude::{BLACK, GREEN, RED};
    use plotters::style::full_palette::ORANGE;
    use sci_rs::na::SimdBool;
    use std::f32::consts::PI;
    use std::fmt::Display;
    use std::ops::Deref;
    use std::sync::atomic::AtomicI64;
    use tch::Device::{Cpu, Cuda};
    use tch::Kind::{Bool, Float, Int, Int64};
    use tch::Tensor;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA, INT64_CPU};
    use tch::{Device, IndexOp, Kind};
    use thoosa_common::builder::Node::Par;
    use thoosa_common::find_motifs_run_config::{FindMotifsRunConfig};
    use thoosa_common::loader::SyncWrapper;
    use thoosa_shared::panic_to_err;
    use thoosa_common::utils::{
        cuda_mem, from, hist_1_dim, lin_iter,
        lin_t, plot_series, plot_series_with_points, reset_print_options,
        skewness_kurtosis_1_dim, three_sigs_tensor, to_vec_f32,
    };
    use thoosa_common::ws::mock_ws_simp;
    use thoosa_shared::colors_iter::DistinctColorIter;
    use thoosa_shared::{
        ThoError, comp_sin_k_v, pulse_in_rand, saw, sig_one_imp, sig_three_imp, sig_two_imp,
        sin_v,
    };

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_z_normalized_euclidean() -> Result<(), ThoError> {
        let d = z_normalized_euclidean(
            &Tensor::vstack(&[
                Tensor::from_slice(&[1., 0., 1., 0., 1.]),
                Tensor::from_slice(&[10., 0., 10., 0., 10.]),
            ]),
            &Tensor::vstack(&[
                Tensor::from_slice(&[0., 1., 0., 1., 0.]),
                Tensor::from_slice(&[0., 10., 0., 10., 0.]),
            ]),
            1,
        )?;

        //TODO?
        assert!(
            Tensor::from_slice(&[4., 4.,])
                .totype(Float)
                .allclose(&d, 0.01, 0.01, false)
        );

        Ok(())
    }



    fn _dist_2(p_0: &[f32], p_1: &[f32]) -> f32 {
        ((p_0[0] - p_1[0]).powi(2) + (p_0[1] - p_1[1]).powi(2)).sqrt()
    }

    fn _dist_3(p_0: &[f32], p_1: &[f32]) -> f32 {
        ((p_0[0] - p_1[0]).powi(2) + (p_0[1] - p_1[1]).powi(2) + (p_0[2] - p_1[2]).powi(2)).sqrt()
    }

    #[test]
    fn test_clc_dist_d2_0() -> Result<(), ThoError> {
        let tensor_0 = Tensor::from_slice(&[0.0, 1.0, 2., 3.]).view((1, -1));
        let tensor_1 = Tensor::from_slice(&[0.0, 1.0, 2., 3.]).view((1, -1));

        let dist = clc_dist_d2(tensor_0, tensor_1, 0)?;

        debug!("dist={:?} dist={dist}", dist.size());

        assert_eq!(0.0, dist.double_value(&[0, 0]));

        Ok(())
    }

    #[test]
    fn test_single_channel() {
        // 1 channel, 2 slices, length 2
        // t0 slices: [0,0], [1,1]
        let t0 = Tensor::from_slice(&[0.0f32, 0.0, 1.0, 1.0])
            .totype(Float)
            .view([1, 2, 2]); // [channels=1, num_slices=2, slice_len=2]
        let t1 = t0.shallow_clone();

        let dist = clc_dist_d2(t0, t1, 1).unwrap();
        let expected = Tensor::from_slice(&[0.0f32, (2.0f32).sqrt(), (2.0f32).sqrt(), 0.0])
            .totype(Float)
            .view([1, 2, 2]);

        assert!(dist.allclose(&expected, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_two_channels() {
        // channel 0 slices: [0,0], [1,0]
        // channel 1 slices: [0,1], [0,2]
        let t0 = Tensor::from_slice(&[
            0.0, 0.0, 1.0, 0.0, // ch0
            0.0, 1.0, 0.0, 2.0, // ch1
        ])
        .totype(Float)
        .view([2, 2, 2]); // [channels=2, num_slices=2, slice_len=2]

        let t1 = t0.shallow_clone();
        let dist = clc_dist_d2(t0, t1, 1).unwrap();

        // both channels: pairwise distances of two 2D points
        let expected = Tensor::from_slice(&[
            0.0, 1.0, 1.0, 0.0, // ch0
            0.0, 1.0, 1.0, 0.0, // ch1
        ])
        .totype(Float)
        .view([2, 2, 2]);

        assert!(dist.allclose(&expected, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_three_channels() {
        // channel 0: [0], [1]
        // channel 1: [2], [5]
        // channel 2: [10], [13]
        let t0 = Tensor::from_slice(&[0.0, 1.0, 2.0, 5.0, 10.0, 13.0])
            .totype(Float)
            .view([3, 2, 1]); // [channels=3, num_slices=2, slice_len=1]
        let t1 = t0.shallow_clone();
        let dist = clc_dist_d2(t0, t1, 1).unwrap();

        let expected = Tensor::from_slice(&[
            0.0, 1.0, 1.0, 0.0, // ch0
            0.0, 3.0, 3.0, 0.0, // ch1
            0.0, 3.0, 3.0, 0.0, // ch2
        ])
        .totype(Float)
        .view([3, 2, 2]);

        assert!(dist.allclose(&expected, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_cross_distances() {
        let t0 = Tensor::from_slice(&[0.0, 0.0, 1.0, 0.0]).view([1, 2, 2]);

        let t1 = Tensor::from_slice(&[0.0, 1.0, 2.0, 0.0]).view([1, 2, 2]);

        let dist = clc_dist_d2(t0, t1, 1).unwrap();

        let expected = Tensor::from_slice(&[1.0, 2.0, (2.0f32).sqrt(), 1.0]).view([1, 2, 2]);

        assert!(dist.allclose(&expected, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_different_num_slices() {
        // t0: 2 slices, each of length 2
        let t0 = Tensor::from_slice(&[
            0.0, 0.0, // slice0
            1.0, 0.0, // slice1
        ])
        .view([1, 2, 2]);

        // t1: 3 slices, each of length 2
        let t1 = Tensor::from_slice(&[
            0.0, 1.0, // slice0
            1.0, 1.0, // slice1
            2.0, 3.0, // slice2
        ])
        .view([1, 3, 2]);

        let dist = clc_dist_d2(t0, t1, 1).unwrap();

        // Expected distances:
        // [0,0] vs [0,1] => sqrt((0-0)^2 + (0-1)^2) = 1
        // [0,0] vs [1,1] => sqrt((0-1)^2 + (0-1)^2) = sqrt(2)
        // [0,0] vs [2,3] => sqrt((0-2)^2 + (0-3)^2) = sqrt(13)
        //
        // [1,0] vs [0,1] => sqrt((1-0)^2 + (0-1)^2) = sqrt(2)
        // [1,0] vs [1,1] => sqrt((1-1)^2 + (0-1)^2) = 1
        // [1,0] vs [2,3] => sqrt((1-2)^2 + (0-3)^2) = sqrt(10)
        let expected = Tensor::from_slice(&[
            1.0,
            (2.0f32).sqrt(),
            (13.0f32).sqrt(),
            (2.0f32).sqrt(),
            1.0,
            (10.0f32).sqrt(),
        ])
        .view([1, 2, 3]);

        assert!(dist.allclose(&expected, 1e-6, 1e-6, false));
    }

    #[test]
    fn test_dist_d2_with_mask_q_05() {
        let slice_0 = Tensor::from_slice(&[30., 30., 0., 0., 40., 40.])
            .totype(Float)
            .view((1, 3, 2));
        let slice_1 = Tensor::from_slice(&[7., 8., 8., 9., 9., 10.])
            .totype(Float)
            .view((1, 3, 2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        // let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        let (mask, Some(dist_t)) = super::clc_dist_d2_and_mask(
            slice_0,
            slice_1,
            true,
            &mut [LeQuantile {
                q: 0.2,
                with_mask: false,
            }],
        )
        .unwrap() else {
            panic!("no dists");
        };
        // debug!("mask={mask}");
        debug!("dist_t={dist_t}");
        let expected_dists = Tensor::stack(
            &[
                Tensor::from_slice(&[
                    _dist_2(&[30., 30.], &[7., 8.]),
                    _dist_2(&[30., 30.], &[8., 9.]),
                    _dist_2(&[30., 30.], &[9., 10.]),
                ]),
                Tensor::from_slice(&[
                    _dist_2(&[0., 0.], &[7., 8.]),
                    _dist_2(&[0., 0.], &[8., 9.]),
                    _dist_2(&[0., 0.], &[9., 10.]),
                ]),
                Tensor::from_slice(&[
                    _dist_2(&[40., 40.], &[7., 8.]),
                    _dist_2(&[40., 40.], &[8., 9.]),
                    _dist_2(&[40., 40.], &[9., 10.]),
                ]),
            ],
            0,
        )
        .unsqueeze(0); //unsqueeze for 0 channel

        assert!(expected_dists.allclose(&dist_t, 0.1, 0.1, false));

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[false, false, false, true, true, true, false, false, false])
                .view((1, 3, 3)),
            mask
        );
    }

    #[test]
    fn test_dist_d2_with_mask_local_min() {
        // set_max_level(Trace);
        let slice_0 = Tensor::from_slice(&[7., 8., 7., 8., 4., 5., 5., 6., 5., 6.]).view((1, 5, 2));
        let slice_1 = Tensor::from_slice(&[
            //slice_1 split if oom
            0., 0., 1., 1., 2., 2.,
        ])
        .view((1, 3, 2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        // let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        let (mask, Some(dist_t)) =
            super::clc_dist_d2_and_mask(slice_0, slice_1, true, &mut [LocalMin { dim: 1, radius: 0 }]).unwrap()
        else {
            panic!("no dists");
        };
        // debug!("mask={mask}");
        debug!("dist_t=\n{dist_t}");
        let expected_dists = Tensor::stack(
            &[
                Tensor::from_slice(&[
                    _dist_2(&[7., 8.], &[0., 0.]),
                    _dist_2(&[7., 8.], &[1., 1.]),
                    _dist_2(&[7., 8.], &[2., 2.]),
                ]),
                Tensor::from_slice(&[
                    _dist_2(&[7., 8.], &[0., 0.]),
                    _dist_2(&[7., 8.], &[1., 1.]),
                    _dist_2(&[7., 8.], &[2., 2.]),
                ]),
                Tensor::from_slice(&[
                    _dist_2(&[4., 5.], &[0., 0.]),
                    _dist_2(&[4., 5.], &[1., 1.]),
                    _dist_2(&[4., 5.], &[2., 2.]),
                ]),
                Tensor::from_slice(&[
                    _dist_2(&[5., 6.], &[0., 0.]),
                    _dist_2(&[5., 6.], &[1., 1.]),
                    _dist_2(&[5., 6.], &[2., 2.]),
                ]),
                Tensor::from_slice(&[
                    _dist_2(&[5., 6.], &[0., 0.]),
                    _dist_2(&[5., 6.], &[1., 1.]),
                    _dist_2(&[5., 6.], &[2., 2.]),
                ]),
            ],
            0,
        )
        .unsqueeze(0); //unsqueeze for 0 channel

        debug!("expected_dists=\n{expected_dists}");

        assert!(expected_dists.allclose(&dist_t.totype(Float), 0.1, 0.1, false));

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                // 0,0    1,1    2,2
                false, false, false, //7,8
                false, false, false, //7,8
                true, true, true, //4,5 - min
                false, false, false, //5,6
                false, false, false //5,6
            ])
            .view((1, 5, 3)),
            mask
        );
    }

    #[test]
    fn test_dist_d2_0_diag() {
        let slice_0 = Tensor::from_slice(&[0., 0., 1., 1., 2., 2.]).view((1, 3, 2));
        let slice_1 = Tensor::from_slice(&[3., 4., 4., 5., 5., 6.]).view((1, 3, 2));
        debug!("slice_0={slice_0}");
        debug!("slice_1={slice_1}");

        // let dist_t = super::dist_d2(&slice_0, &slice_1).unwrap();
        //set_max_level(Trace);
        let (mask, Some(dist_t)) = clc_dist_d2_and_mask(
            slice_0,
            slice_1,
            true,
            &mut [
                LeQuantile {
                    q: 1.0,
                    with_mask: false,
                },
                Diagonal,
            ],
        )
        .unwrap() else {
            panic!("no dists");
        };
        debug!("mask={mask}");
        // debug!("dist_t={dist_t}");

        assert_eq!(
            Tensor::from_slice(&[false, true, true, true, false, true, true, true, false])
                .view((1, 3, 3)),
            mask
        );
        //
        // assert_eq!(5.0, dist_t.double_value(&[0,0,0]));
        // assert_eq!(5.0, dist_t.double_value(&[0,1,1]));
        // assert_eq!(5.0, dist_t.double_value(&[0,2,2]));
    }

    #[test]
    fn test_dist_le_mask() {
        //5 motifs 7 slices
        let dists_t = Tensor::from_slice(&[
            0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
            0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.30, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36,
            0.40, 0.41, 0.42, 0.43, 0.44, 0.45, 0.46, 1.00, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06,
            1.10, 1.11, 1.12, 1.13, 1.14, 1.15, 1.16, 1.20, 1.21, 1.22, 1.23, 1.24, 1.25, 1.26,
            1.30, 1.31, 1.32, 1.33, 1.34, 1.35, 1.36, 1.40, 1.41, 1.42, 1.43, 1.44, 1.45, 1.46,
            2.00, 2.01, 2.02, 2.03, 2.04, 2.05, 2.06, 2.10, 2.11, 2.12, 2.13, 2.14, 2.15, 2.16,
            2.20, 2.21, 2.22, 2.23, 2.24, 2.25, 2.26, 2.30, 2.31, 2.32, 2.33, 2.34, 2.35, 2.36,
            2.40, 2.41, 2.42, 2.43, 2.44, 2.45, 2.46,
        ])
        .view((3, 5, 7));

        debug!("dists_t=\n{dists_t}");

        let thresholds_t = Tensor::from_slice(&[
            0.01, 0.11, 0.21, 0.31, 0.41, 1.01, 1.11, 1.21, 1.31, 1.41, 2.01, 2.11, 2.21, 2.31,
            2.41,
        ])
        .view((3, 5));

        let thresholds_t_broadcasted = thresholds_t.unsqueeze(2).broadcast_to(&[3, 5, 7]);

        debug!("thresholds_t_broadcasted=\n{thresholds_t_broadcasted}");

        let mask = Le {
            v: thresholds_t_broadcasted,
        }
        .make_mask(None, &dists_t).unwrap();

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                true, true, false, false, false, false, false, true, true, false, false, false,
                false, false, true, true, false, false, false, false, false, true, true, false,
                false, false, false, false, true, true, false, false, false, false, false, true,
                true, false, false, false, false, false, true, true, false, false, false, false,
                false, true, true, false, false, false, false, false, true, true, false, false,
                false, false, false, true, true, false, false, false, false, false, true, true,
                false, false, false, false, false, true, true, false, false, false, false, false,
                true, true, false, false, false, false, false, true, true, false, false, false,
                false, false, true, true, false, false, false, false, false
            ])
            .view((3, 5, 7)),
            mask
        );
    }

    #[test]
    fn test_acc_mean_mask() {
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();
        let mut means = Mutex::new(Tensor::zeros([3], FLOAT_CPU));

        let mask = LeRunningMean {
            k: 1.0,
            avg: &mut means,
            total_samples: &mut Mutex::new(Tensor::zeros([3], INT64_CPU)),
        }
        .make_mask(None, &dists_t).unwrap();

        debug!("mask=\n{mask}");
        assert_eq!(
            Tensor::from_slice(&[
                true, true, true, false, false, true, true, true, false, false, true, true, true,
                false, false, true, true, true, false, false, true, true, false, false, false,
                true, true, false, false, false, true, true, false, false, false, true, true, true,
                false, false, true, true, true, false, false, true, true, true, false, false, true,
                true, true, false, false, true, true, false, false, false, true, true, false,
                false, false, true, true, false, false, false, true, true, true, false, false,
                true, true, true, false, false, true, true, true, false, false, true, true, true,
                false, false, true, true, false, false, false, true, true, false, false, false,
                true, true, false, false, false
            ])
            .view((3, 7, 5)),
            mask
        );
    }

    #[test]
    fn test_mean_mask_two_chunks() {
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let dists_t_0 = dists_t.i((.., .., ..3));
        let dists_t_1 = dists_t.i((.., .., 3..));

        let mut means = Mutex::new(Tensor::zeros([3], FLOAT_CPU));

        let mut total_samples = Mutex::new(Tensor::zeros([3], INT64_CPU));
        let mask = LeRunningMean {
            k: 1.0,
            avg: &mut means,
            total_samples: &mut total_samples,
        }
        .make_mask(None, &dists_t_0);
        let mask = LeRunningMean {
            k: 1.0,
            avg: &mut means,
            total_samples: &mut total_samples,
        }
        .make_mask(None, &dists_t_1).unwrap();

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
    fn test_mean_mask() {
        // set_max_level(Trace);

        const CHANNELS: i64 = 1;

        let mut means_acc = Mutex::new(Tensor::zeros([CHANNELS, 1], INT64_CPU));
        let mut total_samples = Mutex::new(Tensor::zeros([CHANNELS, 1], INT64_CPU));

        let mask = mean_mask(
            0.5,
            &Tensor::from_slice(&[
                // distances 3x2 slices
                0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
            ])
            .view((CHANNELS, 3, 2)),
            None,
            &mut means_acc,
            &mut total_samples,
        ).unwrap();

        assert_eq!(
            Tensor::from_slice(&[false, false, false, false, false, false,])
                .view((1, 3, 2))
                .totype(Bool),
            mask
        );

        assert_eq!(
            &Tensor::from_slice(&[0.2]).view((1, -1)).totype(Float),
            means_acc.lock().deref()
        );

        assert_eq!(
            &Tensor::from_slice(&[6]).view((1, -1)).totype(Int64),
            total_samples.lock().deref()
        );

        let mask = mean_mask(
            0.5,
            &Tensor::from_slice(&[
                // distances 3x2 slices
                0.4, 0.4, 0.4, 0.4, 0.4, 0.4,
            ])
            .view((CHANNELS, 3, 2)),
            None,
            &mut means_acc,
            &mut total_samples,
        ).unwrap();

        assert_eq!(
            Tensor::from_slice(&[false, false, false, false, false, false,])
                .view((1, 3, 2))
                .totype(Bool),
            mask
        );

        assert_eq!(
            &Tensor::from_slice(&[0.3]).view((1, -1)).totype(Float),
            means_acc.lock().deref()
        );

        assert_eq!(
            &Tensor::from_slice(&[12]).view((1, -1)).totype(Int64),
            total_samples.lock().deref()
        );

        let mask = mean_mask(
            0.5,
            &Tensor::from_slice(&[
                // distances 3x2 slices
                0.2, 0.2, 0.2, 0.1, 0.1, 0.1,
            ])
            .view((CHANNELS, 3, 2)),
            None,
            &mut means_acc,
            &mut total_samples,
        ).unwrap();

        assert_eq!(
            Tensor::from_slice(&[
                false, false, false,//0.2 > 0.25 * 0.5
                true, true, true,   //0.1 < 0.25 * 0.5
            ])
            .view((1, 3, 2))
            .totype(Bool),
            mask
        );

        assert_eq!(
            &Tensor::from_slice(&[0.25]).view((1, -1)).totype(Float),
            means_acc.lock().deref()
        );

        assert_eq!(
            &Tensor::from_slice(&[18]).view((1, -1)).totype(Int64),
            total_samples.lock().deref()
        );


        let mask = mean_mask(
            0.0,
            &Tensor::from_slice(&[
                // distances 3x2 slices
                0.2, 0.2, 0.2, 0.1, 0.1, 0.1,
            ])
                .view((CHANNELS, 3, 2)),
            None,
            &mut means_acc,
            &mut total_samples,
        ).unwrap();

        assert_eq!(
            Tensor::from_slice(&[
                false, false, false,
                false, false, false,
            ])
                .view((1, 3, 2))
                .totype(Bool),
            mask
        );
        debug!("mask=\n{mask}");
    }

    #[test]
    fn test_local_min_mask() {
        // set_max_level(Trace);

        let distances = Tensor::from_slice(&[
            // distances 2x3 slices
            0.2, 0.2, 0.1, 0.3, 0.3, 0.2, 0.2, 0.1, 0.3, 0.3,
        ])
        .view((1, 5, 2));

        debug!("distances=\n{distances}");

        let mask = local_min_mask(&distances, 1).unwrap();

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                false, true, true, false, false, false, true, true, false, false
            ])
            .view((1, 5, 2)),
            mask
        );
    }

    #[test]
    fn test_local_min_mask_suppressed() {
        let distances = Tensor::from_slice(&[5.0, 2.0, 3.0, 1.0, 4.0]).view((1, 5, 1));

        let mask = local_min_mask_suppressed(&distances, 1, 2).unwrap();

        assert_eq!(
            Tensor::from_slice(&[false, false, false, true, false]).view((1, 5, 1)),
            mask
        );
    }

    // #[test]TODO
    fn test_local_min_mask_0() {
        // set_max_level(Trace);

        let distances = Tensor::from_slice(&[0.]).view((1, 1, 1));

        debug!("distances=\n{distances}");

        let mask = local_min_mask(&distances, 1).unwrap();

        debug!("mask=\n{mask}");
    }

    #[test]
    fn test_quantile_mask() {
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let mask = LeQuantile {
            q: 0.5,
            with_mask: false,
        }
        .make_mask(None, &dists_t).unwrap();

        debug!("mask=\n{mask}");

        //means [0.2300, 1.2300, 2.2300]
        assert_eq!(
            Tensor::from_slice(&[
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true, true, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, true, true, true,
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                true, true, true, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, true, true, true, true, true, true,
                true, true, true, true, true, true, true, true, true, true, true, true, true,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false
            ])
            .view((3, 7, 5)),
            mask
        );
    }

    #[test]
    fn test_quantile_mask_with_empty_pre_mask() {
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let mask = LeQuantile {
            q: 0.5,
            with_mask: true,
        }
        .make_mask(
            Some(
                Tensor::from_slice(&[
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true,
                ])
                .view((3, 7, 5)),
            ),
            &dists_t,
        ).unwrap();

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, true, true, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, true, true, true,
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                true, true, true, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, true, true, true, true, true, true, true,
                true, true, true, true, true, true, true, true, true, true, true, true, true,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false
            ])
            .view((3, 7, 5)),
            mask
        );
    }

    #[test]
    fn test_quantile_mask_with_pre_mask_1() {
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let mask = LeQuantile {
            q: 0.5,
            with_mask: true,
        }
        .make_mask(
            Some(
                Tensor::from_slice(&[
                    false, false, false, false, false, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, false, false,
                    false, false, false, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, false, false, false, false,
                    false, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true,
                ])
                .view((3, 7, 5)),
            ),
            &dists_t,
        ).unwrap();

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                false, false, false, false, false, true, true, true, true, true, true, true, true,
                true, true, true, true, true, true, true, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, true, true, true, true, true, true, true, true, true, true, true,
                true, true, true, true, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false
            ])
            .view((3, 7, 5)),
            mask
        );
    }

    #[test]
    fn test_quantile_mask_with_pre_mask_2() {
        // set_max_level(Trace);

        let dists_t = tensor_3_7_5_lin();

        let mask = LeQuantile {
            q: 0.5,
            with_mask: true,
        }
        .make_mask(
            Some(
                Tensor::from_slice(&[
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, false, false, false, false, false, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, false, false, false, false, false, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, false, false,
                    false, false, false,
                ])
                .view((3, 7, 5)),
            ),
            &dists_t,
        ).unwrap();

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                true, true, true, true, true, true, true, true, true, true, true, true, true, true,
                true, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, false, false, true, true, true,
                true, true, true, true, true, true, true, true, true, true, true, true, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false, false, false, true, true, true, true, true,
                true, true, true, true, true, true, true, true, true, true, false, false, false,
                false, false, false, false, false, false, false, false, false, false, false, false,
                false, false, false, false, false
            ])
            .view((3, 7, 5)),
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
    fn test_quantile_mask_diff_lengths() {
        // set_max_level(Trace);

        let dists_t = Tensor::from_slice(&[
            0.0,
            0.0,
            2.0,
            0.0,
            10.0,
            0.01,
            0.0,
            2.0,
            0.01,
            9.0,
            0.02,
            0.02,
            2.0,
            0.02,
            8.0,
            0.03,
            0.02,
            2.0,
            0.03,
            7.0,
            0.04,
            0.02,
            2.0,
            0.04,
            6.0,
            0.05,
            0.05,
            2.1,
            f32::INFINITY,
            5.0,
            0.06,
            0.1,
            2.2,
            f32::INFINITY,
            4.0,
        ])
        .view((1, 7, 5));

        let mask = LeQuantile {
            q: 0.5,
            with_mask: true,
        }
        .make_mask(None, &dists_t).unwrap();

        debug!("mask=\n{mask}");
        //Quantiles (0.5):             (matlab ignores infs)
        //     0.0300    0.0200    2.0000    0.0200    7.0000
        //
        // Closest Indices:
        //       4       3     1      3      4

        assert_eq!(
            Tensor::from_slice(&[
                true, true, true, true, false, true, true, true, true, false, true, true, true,
                true, false, true, true, true, true, true, false, true, true, false, true, false,
                false, false, false, true, false, false, false, false, true
            ])
            .view((1, 7, 5)),
            mask
        );
    }

    #[test]
    fn test_quantile_mask_diff_lengths_with_pre_mask() {
        // set_max_level(Trace);

        let dists_t = Tensor::from_slice(&[
            100.,
            100.,
            100.,
            100.,
            100.,
            0.0,
            0.0,
            2.0,
            0.0,
            10.0,
            0.01,
            0.0,
            2.0,
            0.01,
            9.0,
            0.02,
            0.02,
            2.0,
            0.02,
            8.0,
            0.03,
            0.02,
            2.0,
            0.03,
            7.0,
            0.04,
            0.02,
            2.0,
            0.04,
            6.0,
            0.05,
            0.05,
            2.1,
            f32::INFINITY,
            5.0,
            0.06,
            0.1,
            2.2,
            f32::INFINITY,
            4.0,
            100.,
            100.,
            100.,
            100.,
            100.,
        ])
        .view((1, 9, 5));

        let mask = LeQuantile {
            q: 0.5,
            with_mask: true,
        }
        .make_mask(
            Some(
                Tensor::from_slice(&[
                    false, false, false, false, false, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, true, true, true, true, true, true, true, true, true, true, true,
                    true, true, false, false, false, false, false,
                ])
                .view((1, 9, 5)),
            ),
            &dists_t,
        ).unwrap();

        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                false, false, false, false, false, true, true, true, true, false, true, true, true,
                true, false, true, true, true, true, false, true, true, true, true, true, false,
                true, true, false, true, false, false, false, false, true, false, false, false,
                false, true, false, false, false, false, false
            ])
            .view((1, 9, 5)),
            mask
        );
    }

    fn tensor_3_7_5_lin() -> Tensor {
        Tensor::from_slice(&[
            //channel 0
            //slice_1: 0       1       2       3       4
            //slice_0
            /*0*/
            0.0000, 0.1000, 0.2000, 0.3000, 0.4000, /*1*/ 0.0100, 0.1100, 0.2100, 0.3100,
            0.4100, /*2*/ 0.0200, 0.1200, 0.2200, 0.3200, 0.4200, /*3*/ 0.0300, 0.1300,
            0.2300, 0.3300, 0.4300, /*4*/ 0.0400, 0.1400, 0.2400, 0.3400, 0.4400,
            /*5*/ 0.0500, 0.1500, 0.2500, 0.3500, 0.4500, /*6*/ 0.0600, 0.1600, 0.2600,
            0.3600, 0.4600, //channel 1
            1.0000, 1.1000, 1.2000, 1.3000, 1.4000, 1.0100, 1.1100, 1.2100, 1.3100, 1.4100, 1.0200,
            1.1200, 1.2200, 1.3200, 1.4200, 1.0300, 1.1300, 1.2300, 1.3300, 1.4300, 1.0400, 1.1400,
            1.2400, 1.3400, 1.4400, 1.0500, 1.1500, 1.2500, 1.3500, 1.4500, 1.0600, 1.1600, 1.2600,
            1.3600, 1.4600, //channel 2
            2.0000, 2.1000, 2.2000, 2.3000, 2.4000, 2.0100, 2.1100, 2.2100, 2.3100, 2.4100, 2.0200,
            2.1200, 2.2200, 2.3200, 2.4200, 2.0300, 2.1300, 2.2300, 2.3300, 2.4300, 2.0400, 2.1400,
            2.2400, 2.3400, 2.4400, 2.0500, 2.1500, 2.2500, 2.3500, 2.4500, 2.0600, 2.1600, 2.2600,
            2.3600, 2.4600,
        ])
        .view((3, 7, 5))
    }

    #[test]
    fn test_q() {
        let samples =
            Tensor::from_slice(&[0.0000, 0.0100, 0.0200, 0.0300, 0.0400]).view((1, 1, -1));
        let q_t = samples.quantile_scalar(0.5, 2, true, "lower");

        debug!("full: q_t=\n{q_t}");

        let samples = Tensor::from_slice(&[
            0.0000,
            0.0100,
            0.0200,
            0.0300,
            0.0400,
            f64::INFINITY,
            f64::INFINITY,
        ])
        .view((1, 1, -1));

        let q_t = samples.quantile_scalar(0.5, 2, true, "lower");

        debug!("masked: q_t=\n{q_t}");
    }

    #[test]
    fn test_dist_d2_1() {
        //TODO remove
        let x = [0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2, 8.1, 9.0];

        for (i_0, x_0) in x.windows(2).enumerate() {
            for (i_1, x_1) in x.windows(2).enumerate() {
                let dist: f64 =
                    (x_0[0] - x_1[0]) * (x_0[0] - x_1[0]) + (x_0[1] - x_1[1]) * (x_0[1] - x_1[1]);
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
    fn test_slice_t() {
        let tensor = slice_t(
            &Tensor::from_slice(&[1., 2., 3., 4., 5.]).view((1, -1)),
            1,
            2,
        );

        assert_eq!(
            Tensor::from_slice(&[1., 2., 2., 3., 3., 4., 4., 5.,]).view((4, 1, 2)),
            tensor
        );
    }

    #[test]
    fn test_average_euclidean_distance() -> Result<(), ThoError> {
        let d = [1., 2., 3., 4., 5., 6., 7., 8., 9.];
        let ab = _dist_3(&d[0..3], &d[3..6]);
        let ac = _dist_3(&d[0..3], &d[6..9]);
        let bc = _dist_3(&d[3..6], &d[6..9]);

        println!("ab={ab} ac={ac} bc={bc}");

        let test_avg = (ab + ac + bc) / 3.;

        println!("test_avg={test_avg}");

        let x = Tensor::from_slice(&d).view((3, 3));
        println!("x={x}");
        let avg_dist = average_euclidean_distance(&x)?;
        assert_eq!(test_avg, avg_dist);

        Ok(())
    }


    #[test]
    fn test_mask_maker_diag() {
        let mask = Diagonal.make_mask(None, &Tensor::ones(&[2, 3, 3], (Bool, Cpu))).unwrap();

        assert_eq!(
            Tensor::from_slice(&[
                false, true, true, true, false, true, true, true, false, false, true, true, true,
                false, true, true, true, false
            ])
            .view((2, 3, 3)),
            mask
        );

    }

    #[test]
    fn test_mask_maker_tril() {
        let mask = Tril.make_mask(None, &Tensor::ones(&[2, 3, 3], (Bool, Cpu))).unwrap();
        debug!("mask=\n{mask}");

        assert_eq!(
            Tensor::from_slice(&[
                false, true, true, false, false, true, false, false, false, false, true, true,
                false, false, true, false, false, false
            ])
            .view((2, 3, 3)),
            mask
        );

        debug!("mask=\n{mask}");
    }

    #[test]
    fn test_mask_maker_local_min() -> Result<(), ThoError> {
        // reset_print_options();
        // set_max_level(Trace);

        let input_n = 1000;

        let ch_0 = (0..input_n)
            .map(|x| (2. * PI * (x as f32 / input_n as f32)).sin())
            .collect::<Vec<f32>>(); //sin_v(input_n, 1.);
        let ch_1 = (0..input_n)
            .map(|x| (4. * PI * (x as f32 / input_n as f32)).sin())
            .collect::<Vec<f32>>(); //sin_v(input_n, 0.5);

        let x = lin_t(input_n);
        /*plot_series(
            vec![
                (format!("ch_0"), GREEN, vec_f32_from(&x).into_iter().zip(ch_0.clone().into_iter()).collect::<Vec<(f32, f32)>>()),
                (format!("ch_1"), BLACK,  vec_f32_from(&x).into_iter().zip(ch_1.clone().into_iter()).collect::<Vec<(f32, f32)>>()),

            ], "Sigs", (2000, 1000), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;*/
        let min_ch0 = 3. / 4. * input_n as f32;
        let min_ch1 = [3. / 8. * input_n as f32, 7. / 8. * input_n as f32];
        debug!("min_ch0={min_ch0:?} min_ch1={min_ch1:?}");

        let distances = Tensor::stack(
            &[
                Tensor::from_slice(ch_0.as_slice()),
                Tensor::from_slice(ch_1.as_slice()),
            ],
            0,
        )
        .view((2, input_n as i64)); //unsqueeze to make it [channels x t_0 slices x t_1 slices]

        debug!("distances={distances:?}");

        let mask = LocalMin { dim: 1, radius: 0 }.make_mask(None, &distances).unwrap();

        trace!("mask=\n{mask}");
        debug!("mask.nonzero=\n{}", mask.nonzero());

        assert_eq!(
            Tensor::from_slice(&[0, 0, 0, 750, 1, 0, 1, 375, 1, 875,]).view((5, 2)),
            mask.nonzero().totype(Int)
        );

        Ok(())
    }

    #[test]
    fn test_big() {
        let t_0 = Tensor::rand(&[20, 1661, 10], FLOAT_CPU);
        let t_1 = Tensor::rand(&[20, 1661, 10], FLOAT_CPU);

        let dist_1 = Tensor::cdist(&t_0, &t_1, 2.0, 0);
        let dist_0 = clc_dist_d2(t_0, t_1, 1).unwrap();

        println!("diff_0=\n{dist_0:?}");
        println!("dist_1=\n{dist_1:?}");
        let eq = dist_1.allclose(&dist_0, 0.01, 0.02, false);
        println!("eq={}", eq);

        assert!(eq);
    }
}
