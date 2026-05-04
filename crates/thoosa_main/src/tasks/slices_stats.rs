use std::fmt::Debug;
use tch::Kind;
use tch::Tensor;
pub use crate::tasks::find_motifs_run_config::{Aggr, CommonStatsMetric, Metrics};
use thoosa_common::utils::z_norm_dims;
use tracing::debug;
use thoosa_motifs::common_stats::Stats as CommonStatsValue;
use thoosa_motifs::criteria::{coherence, inlier_ratio, interesting_amp_01, interesting_band_energy_ratio_01, interesting_curvature_norm_01, interesting_entropy_band_01, interesting_entropy_gauss_01, interesting_turning_points_ratio_peaked_01, interesting_tv_norm_01, norm_thickness, periodicity_acf_01, periodicity_acf_01_frac};
use thoosa_motifs::pearson_corr::pearson_corr_distance_matrix_0_2_single as pearson_corr_distance_matrix_0_2_single_matrix;
use thoosa_shared::{panic_to_err, ThoError};

pub fn metric_scalar_for_motif<ID:Debug>(id:&ID, motif_slices: &Tensor, metric: &Metrics) -> Result<f32, ThoError> {
    let score_t = match metric {
        Metrics::EucDist(v) => {
            if v.z_normalize {
                return panic_to_err!(z_normalized_euclidean_distance(motif_slices, v.aggr));
            }
            return panic_to_err!(euclidean_distance(motif_slices, v.aggr));
        }
        Metrics::PearsonCorr(v) => return panic_to_err!(pearson_corr_distance_matrix_0_2_single(motif_slices, v.eps, v.aggr)),
        Metrics::DotProduct(v) => return panic_to_err!(dot_product_distance(motif_slices, v.aggr)),
        //TODO move Common* into single Metric, call common_stats_for_motif once
        Metrics::CommonEucDist(v) => return Ok(common_stats_for_motif(motif_slices, v)?.euc_dist),
        Metrics::CommonPearson(v) => return Ok(common_stats_for_motif(motif_slices, v)?.pearson),
        Metrics::CommonA0(v) => return Ok(common_stats_for_motif(motif_slices, v)?.a0),
        Metrics::CommonTv0(v) => return Ok(common_stats_for_motif(motif_slices, v)?.tv0),
        Metrics::CommonCurv0(v) => return Ok(common_stats_for_motif(motif_slices, v)?.curv0),
        Metrics::NormThickness(v) => norm_thickness(id, motif_slices, v.eps)?,
        Metrics::InlierRatio(v) => inlier_ratio(motif_slices, v.k, v.eps)?,
        Metrics::Coherence(v) => coherence(motif_slices, v.k.round().max(1.0) as i64, v.eps)?,
        Metrics::InterestingAmp01(v) => interesting_amp_01(motif_slices, v.q_lo, v.q_hi, v.a0, v.eps)?,
        Metrics::InterestingTvNorm01(v) => interesting_tv_norm_01(motif_slices, v.amp_q_lo, v.amp_q_hi, v.tv0, v.eps)?,
        Metrics::InterestingCurvatureNorm01(v) => interesting_curvature_norm_01(motif_slices, v.amp_q_lo, v.amp_q_hi, v.curv0, v.eps)?,
        Metrics::InterestingTurningPointsRatio(v) => interesting_turning_points_ratio_peaked_01(motif_slices, v.peak)?,
        Metrics::InterestingBandEnergyRatio01(v) => interesting_band_energy_ratio_01(motif_slices, v.fs, v.f_lo, v.f_hi, v.eps)?,
        Metrics::InterestingEntropyGauss01(v) => interesting_entropy_gauss_01(motif_slices, v.h0, v.sigma, v.eps)?,
        Metrics::InterestingEntropyBand01(v) => interesting_entropy_band_01(motif_slices, v.h_min, v.h_max, v.eps)?,
        Metrics::PeriodicityAcf(v) => periodicity_acf_01_frac(motif_slices, v.struct_lag_max, v.lag_min, v.lag_max, v.struct0, v.peak_good, v.peak_bad, v.eps)?,
    };

    let raw = score_t.f_double_value(&[])?;
    if raw.is_finite() { Ok(raw as f32) } else { Ok(0.0) }
}

pub fn metric_quantiles_for_motif(motif_slices: &Tensor, metric: &Metrics, quantiles: &[f32]) -> Result<Option<Vec<(f32, f32)>>, ThoError> {
    let quantiles = quantiles
        .iter()
        .copied()
        .filter(|q| q.is_finite())
        .map(|q| q.clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    if quantiles.is_empty() {
        return Ok(None);
    }

    match metric {
        Metrics::CommonA0(v) => Ok(Some(common_metric_quantiles_for_motif(motif_slices, v, &quantiles, CommonQuantileMetric::Amp)?)),
        Metrics::CommonTv0(v) => Ok(Some(common_metric_quantiles_for_motif(motif_slices, v, &quantiles, CommonQuantileMetric::Tvn)?)),
        Metrics::CommonCurv0(v) => Ok(Some(common_metric_quantiles_for_motif(motif_slices, v, &quantiles, CommonQuantileMetric::Curvn)?)),
        _ => Ok(None),
    }
}

fn common_stats_for_motif(motif_slices: &Tensor, params: &CommonStatsMetric) -> Result<CommonStatsValue, ThoError> {
    let common_stats = thoosa_motifs::common_stats::estimate_channel_stats_from_windows(
        &motif_slices.f_unsqueeze(0)?,
        params.q_lo,
        params.q_hi,
        params.scale_q,
        params.eps,
    )?;
    common_stats
        .into_iter()
        .next()
        .ok_or_else(|| ThoError::new("common_stats_for_motif: no common stats for motif"))
}

#[derive(Clone, Copy)]
enum CommonQuantileMetric {
    Amp,
    Tvn,
    Curvn,
}

fn common_metric_quantiles_for_motif(
    motif_slices: &Tensor,
    params: &CommonStatsMetric,
    quantiles: &[f32],
    metric: CommonQuantileMetric,
) -> Result<Vec<(f32, f32)>, ThoError> {
    let (amp, tvn, curvn) = per_window_metrics(motif_slices, params.q_lo, params.q_hi, params.eps)?;
    let src = match metric {
        CommonQuantileMetric::Amp => amp,
        CommonQuantileMetric::Tvn => tvn,
        CommonQuantileMetric::Curvn => curvn,
    };

    quantiles
        .iter()
        .map(|q| Ok((*q, f_quantile_1d(&src, *q as f64)?.f_double_value(&[])? as f32)))
        .collect()
}

fn f_quantile_1d(v: &Tensor, q: f64) -> Result<Tensor, ThoError> {
    let sz = v.size();
    if sz.len() != 1 {
        return Err(ThoError::new(format!("f_quantile_1d: expected [M], got {:?}", sz)));
    }
    let m = sz[0];
    if m < 1 {
        return Err(ThoError::new("f_quantile_1d: M must be >= 1"));
    }
    let (sorted, _idx) = v.f_sort(0, false)?;
    let mut i = ((m - 1) as f64 * q).round() as i64;
    if i < 0 { i = 0; }
    if i > m - 1 { i = m - 1; }
    Ok(sorted.f_select(0, i)?)
}

fn f_quantile_dim1_per_row(x: &Tensor, q: f64) -> Result<Tensor, ThoError> {
    let sz = x.size();
    if sz.len() != 2 {
        return Err(ThoError::new(format!("f_quantile_dim1_per_row: expected [M,L], got {:?}", sz)));
    }
    let l = sz[1];
    if l < 1 {
        return Err(ThoError::new("f_quantile_dim1_per_row: L must be >= 1"));
    }
    let (sorted, _idx) = x.f_sort(1, false)?;
    let mut i = ((l - 1) as f64 * q).round() as i64;
    if i < 0 { i = 0; }
    if i > l - 1 { i = l - 1; }
    Ok(sorted.f_select(1, i)?)
}

fn per_window_metrics(w: &Tensor, q_lo: f64, q_hi: f64, eps: f64) -> Result<(Tensor, Tensor, Tensor), ThoError> {
    let sz = w.size();
    if sz.len() != 2 {
        return Err(ThoError::new(format!("per_window_metrics: expected [M,L], got {:?}", sz)));
    }
    let m = sz[0];
    let l = sz[1];
    if l < 2 {
        return Err(ThoError::new("per_window_metrics: need L>=2"));
    }

    let lo = f_quantile_dim1_per_row(w, q_lo)?;
    let hi = f_quantile_dim1_per_row(w, q_hi)?;
    let amp = hi.f_sub(&lo)?;

    let x1 = w.f_narrow(1, 1, l - 1)?;
    let x0 = w.f_narrow(1, 0, l - 1)?;
    let tv = x1.f_sub(&x0)?.f_abs()?.f_sum_dim_intlist(1, false, w.kind())?;

    let curv = if l >= 3 {
        let xp = w.f_narrow(1, 2, l - 2)?;
        let xc = w.f_narrow(1, 1, l - 2)?;
        let xm = w.f_narrow(1, 0, l - 2)?;
        let two = Tensor::from(2.0).to_device(w.device()).to_kind(w.kind());
        xp.f_sub(&xc.f_mul(&two)?)?.f_add(&xm)?.f_abs()?.f_sum_dim_intlist(1, false, w.kind())?
    } else {
        Tensor::f_zeros([m], (w.kind(), w.device()))?
    };

    let denom = amp.f_add_scalar(eps)?;
    let tvn = tv.f_div(&denom)?;
    let curvn = curv.f_div(&denom)?;
    Ok((amp, tvn, curvn))
}

pub fn euclidean_distance(t: &Tensor, aggr: Aggr) -> Result<f32, ThoError> {
    let n = t.size()[0];
    if n < 2 {
        return Ok(0.0);
    }

    let squared_norms = t.f_pow_tensor_scalar(2)?.f_sum_dim_intlist(1, true, Kind::Float)?;
    let squared_norms_sum = squared_norms.f_add(&squared_norms.f_transpose(0, 1)?)?;
    let gram = t.f_mm(&t.f_transpose(0, 1)?)?;
    let gram = gram.f_add(&gram)?;
    let dists_squared = squared_norms_sum.f_subtract(&gram)?.f_clamp_min(0.0)?;
    let triu_indices = Tensor::f_triu_indices(n, n, 1, (Kind::Int64, t.device()))?;
    let i = triu_indices.f_get(0)?;
    let j = triu_indices.f_get(1)?;
    let pairwise_squared_dists = dists_squared.f_index_select(0, &i)?.f_gather(1, &j.f_unsqueeze(1)?, false)?.f_squeeze()?;
    let distances = pairwise_squared_dists.f_sqrt()?;
    aggregate_distances(&distances, aggr)
}

pub fn pearson_corr_distance_matrix_0_2_single(x: &Tensor, eps: f64, aggr: Aggr) -> Result<f32, ThoError> {
    let sx = x.size();
    if sx[0] < 2 {
        return Ok(0.0);
    }

    let matrix = pearson_corr_distance_matrix_0_2_single_matrix(x, eps)?;
    let pairwise = upper_triangle_without_diagonal(&matrix)?;
    aggregate_distances(&pairwise, aggr)
}

pub fn z_normalized_euclidean_distance(t: &Tensor, aggr: Aggr) -> Result<f32, ThoError> {
    let normalized = z_norm_dims(t, &[1], 1e-8)?;
    euclidean_distance(&normalized, aggr)
}

pub fn dot_product_distance(t: &Tensor, aggr: Aggr) -> Result<f32, ThoError> {
    let n = t.size()[0];
    if n < 2 {
        return Ok(0.0);
    }

    let gram = t.f_mm(&t.f_transpose(0, 1)?)?.f_neg()?;
    let pairwise = upper_triangle_without_diagonal(&gram)?;
    aggregate_distances(&pairwise, aggr)
}

fn upper_triangle_without_diagonal(matrix: &Tensor) -> Result<Tensor, ThoError> {
    let n = matrix.size()[0];
    let triu_indices = Tensor::f_triu_indices(n, n, 1, (Kind::Int64, matrix.device()))?;
    let i = triu_indices.f_get(0)?;
    let j = triu_indices.f_get(1)?;
    Ok(matrix.f_index_select(0, &i)?.f_gather(1, &j.f_unsqueeze(1)?, false)?.f_squeeze()?)
}

fn aggregate_distances(distances: &Tensor, aggr: Aggr) -> Result<f32, ThoError> {
    if distances.numel() == 0 {
        return Ok(0.0);
    }

    let scalar = match aggr {
        Aggr::Mean(k) => {
            let k = k.k as f64;
            let mean = distances.f_mean(Kind::Float)?;
            debug!("aggregate_distances, Mean: k={k} distances=\n{distances} mean=\n{mean}");
            mean.f_mul_scalar(k)?
        }
        Aggr::Min(k) => {
            let k = k.k as f64;
            let min = distances.f_min()?;
            debug!("aggregate_distances, Min: k={k} distances=\n{distances} min=\n{min}");
            min.f_mul_scalar(k)?
        }
        Aggr::Max(k) => {
            let k = k.k as f64;
            let max = distances.f_max()?;
            debug!("aggregate_distances, Max: k={k} distances=\n{distances} max=\n{max}");
            max.f_mul_scalar(k)?
        }
        Aggr::Quantile(q) => {
            debug!("aggregate_distances, Quantile: q={} distances=\n{distances}", q.q);
            let q = q.q.clamp(0.0, 1.0) as f64;
            let quantile = distances.f_quantile_scalar(q, 0, true, "lower")?;
            debug!("aggregate_distances, Quantile: distances=\n{distances}, quantile=\n{quantile}");
            let mask = distances.f_le_tensor(&quantile)?;
            let selected = distances.f_masked_select(&mask)?;
            if selected.numel() == 0 {
                debug!("aggregate_distances, Quantile: selected.numel==0, quantile=\n{quantile}");
                quantile
            } else {
                debug!("aggregate_distances, Quantile: selected.numel!=0, quantile=\n{quantile} selected.f_max={}", selected.max());
                selected.f_max()?
            }
        }
    };

    Ok(scalar.f_double_value(&[])? as f32)
}

#[cfg(test)]
mod tests {
    use tch::{Device, Kind, Tensor};

    use super::{aggregate_distances, metric_scalar_for_motif, Aggr, Metrics};
    use crate::tasks::find_motifs_run_config::{MeanAggr, MinAggr, MaxAggr, QuantileAggr, PeriodicityAcfRunConfig};

    #[test]
    fn test_aggregate_distances() {
        let distances = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let mean_k_half = aggregate_distances(&distances, Aggr::Mean(MeanAggr { k: 0.5 })).unwrap();
        assert!((mean_k_half - 1.25).abs() < 1e-6, "expected mean([1,2,3,4]) * 0.5 = 1.25, got {mean_k_half}");
        let min_k_2 = aggregate_distances(&distances, Aggr::Min(MinAggr { k: 2.0 })).unwrap();
        assert!((min_k_2 - 2.0).abs() < 1e-6, "expected min([1,2,3,4]) * 2.0 = 2.0, got {min_k_2}");
        let max_k_2 = aggregate_distances(&distances, Aggr::Max(MaxAggr { k: 2.0 })).unwrap();
        assert!((max_k_2 - 8.0).abs() < 1e-6, "expected max([1,2,3,4]) * 2.0 = 8.0, got {max_k_2}");
        let median_lower = aggregate_distances(&distances, Aggr::Quantile(QuantileAggr { q: 0.5 })).unwrap();
        assert!((median_lower - 2.0).abs() < 1e-6, "expected quantile(0.5, lower) for [1,2,3,4] to be 2.0, got {median_lower}");
    }

    #[test]
    fn test_aggregate_distances_with_zeros() {
        let distances = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let median_lower = aggregate_distances(&distances, Aggr::Quantile(QuantileAggr { q: 0.0 })).unwrap();
        assert!((median_lower - 1.0).abs() < 1e-6, "expected quantile(0.0, lower) for [1,2,3,4] to be 1.0, got {median_lower}");
    }

    #[test]
    fn test_metrics_all_includes_periodicity_acf_and_scalar_is_finite() {
        assert!(Metrics::all().iter().any(|metric| matches!(metric, Metrics::PeriodicityAcf(_))));

        let t = Tensor::arange(1024, (Kind::Float, Device::Cpu)) / 64;
        let motif = t.sin().view([1, -1]);
        let score = metric_scalar_for_motif(&"periodicity_test", &motif, &Metrics::PeriodicityAcf(PeriodicityAcfRunConfig::default())).unwrap();

        assert!(score.is_finite(), "expected finite periodicity score, got {score}");
        assert!((0.0..=1.0).contains(&score), "expected periodicity score in [0, 1], got {score}");
    }
}
