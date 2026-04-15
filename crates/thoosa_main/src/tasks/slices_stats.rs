use std::fmt::Debug;
use tch::Kind;
use tch::Tensor;
use thoosa_common::find_motifs_run_config::SimilarityMetric;
use thoosa_common::utils::z_norm_dims;
use thoosa_dummy_data_derive::DummyData;
use tracing::debug;
use thoosa_motifs::common_stats::Stats as CommonStatsValue;
use thoosa_motifs::criteria::{coherence, inlier_ratio, interesting_amp_01, interesting_band_energy_ratio_01, interesting_curvature_norm_01, interesting_entropy_band_01, interesting_entropy_gauss_01, interesting_turning_points_ratio_peaked_01, interesting_tv_norm_01, norm_thickness, periodicity_acf_01, periodicity_acf_01_frac};
use thoosa_motifs::pearson_corr::pearson_corr_distance_matrix_0_2_single as pearson_corr_distance_matrix_0_2_single_matrix;
use thoosa_shared::{panic_to_err, ThoError};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, DummyData)]
pub enum Metrics {
    EucDist,
    EucDistZNorm,
    PearsonCorr,
    DotProduct,
    CommonEucDist,
    CommonPearson,
    CommonA0,
    CommonTv0,
    CommonCurv0,
    CommonAmpP10,
    CommonAmpP50,
    CommonAmpP90,
    CommonTvnP10,
    CommonTvnP50,
    CommonTvnP90,
    CommonCurvnP10,
    CommonCurvnP50,
    CommonCurvnP90,
    NormThickness,
    InlierRatioRunConfig,
    CoherenceRunConfig,
    InterestingAmp01,
    InterestingTvNorm01,
    InterestingCurvatureNorm01,
    InterestingTurningPointsRatio,
    InterestingBandEnergyRatio01,
    InterestingEntropyGauss01,
    InterestingEntropyBand01,
    PeriodicityAcf,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::EucDist
    }
}

impl From<SimilarityMetric> for Metrics {
    fn from(value: SimilarityMetric) -> Self {
        match value {
            SimilarityMetric::EucDist(config) => {
                if config.z_normalize {
                    Self::EucDistZNorm
                } else {
                    Self::EucDist
                }
            }
            SimilarityMetric::PearsonCorr(_) => Self::PearsonCorr,
            SimilarityMetric::DotProduct(_) => Self::DotProduct,
        }
    }
}

impl Metrics {
    pub fn all() -> Vec<Self> {
        vec![
            Self::EucDist,
            Self::EucDistZNorm,
            Self::PearsonCorr,
            Self::DotProduct,
            Self::CommonEucDist,
            Self::CommonPearson,
            Self::CommonA0,
            Self::CommonTv0,
            Self::CommonCurv0,
            Self::CommonAmpP10,
            Self::CommonAmpP50,
            Self::CommonAmpP90,
            Self::CommonTvnP10,
            Self::CommonTvnP50,
            Self::CommonTvnP90,
            Self::CommonCurvnP10,
            Self::CommonCurvnP50,
            Self::CommonCurvnP90,
            Self::NormThickness,
            Self::InlierRatioRunConfig,
            Self::CoherenceRunConfig,
            Self::InterestingAmp01,
            Self::InterestingTvNorm01,
            Self::InterestingCurvatureNorm01,
            Self::InterestingTurningPointsRatio,
            Self::InterestingBandEnergyRatio01,
            Self::InterestingEntropyGauss01,
            Self::InterestingEntropyBand01,
            Self::PeriodicityAcf,
        ]
    }

    pub fn is_common_stats_metric(&self) -> bool {
        matches!(
            self,
            Metrics::CommonEucDist
                | Metrics::CommonPearson
                | Metrics::CommonA0
                | Metrics::CommonTv0
                | Metrics::CommonCurv0
                | Metrics::CommonAmpP10
                | Metrics::CommonAmpP50
                | Metrics::CommonAmpP90
                | Metrics::CommonTvnP10
                | Metrics::CommonTvnP50
                | Metrics::CommonTvnP90
                | Metrics::CommonCurvnP10
                | Metrics::CommonCurvnP50
                | Metrics::CommonCurvnP90
        )
    }

    pub fn common_stats_value(&self, stats: &CommonStatsValue) -> Result<f32, ThoError> {
        match self {
            Metrics::CommonEucDist => Ok(stats.euc_dist),
            Metrics::CommonPearson => Ok(stats.pearson),
            Metrics::CommonA0 => Ok(stats.a0),
            Metrics::CommonTv0 => Ok(stats.tv0),
            Metrics::CommonCurv0 => Ok(stats.curv0),
            Metrics::CommonAmpP10 => Ok(stats.amp_p10),
            Metrics::CommonAmpP50 => Ok(stats.amp_p50),
            Metrics::CommonAmpP90 => Ok(stats.amp_p90),
            Metrics::CommonTvnP10 => Ok(stats.tvn_p10),
            Metrics::CommonTvnP50 => Ok(stats.tvn_p50),
            Metrics::CommonTvnP90 => Ok(stats.tvn_p90),
            Metrics::CommonCurvnP10 => Ok(stats.curvn_p10),
            Metrics::CommonCurvnP50 => Ok(stats.curvn_p50),
            Metrics::CommonCurvnP90 => Ok(stats.curvn_p90),
            _ => Err(ThoError::new(format!("common_stats_value called with non-common metric: {self:?}"))),
        }
    }
}

#[derive(Debug, Copy, Clone, DummyData)]
pub enum Aggr {
    Mean(f32),
    Min(f32),
    Max(f32),
    Quantile(f32),
}

impl Default for Aggr {
    fn default() -> Self {
        Self::Mean(0.1)
    }
}

pub fn metric_scalar_for_motif<ID:Debug>(id:&ID, motif_slices: &Tensor, metric: Metrics, aggr: Aggr) -> Result<f32, ThoError> {
    let score_t = match metric {
        Metrics::EucDist => return panic_to_err!(euclidean_distance(motif_slices, aggr)),
        Metrics::EucDistZNorm => return panic_to_err!(z_normalized_euclidean_distance(motif_slices, aggr)),
        Metrics::PearsonCorr => return panic_to_err!(pearson_corr_distance_matrix_0_2_single(motif_slices, 1e-8, aggr)),
        Metrics::DotProduct => return panic_to_err!(dot_product_distance(motif_slices, aggr)),
        Metrics::CommonEucDist
        | Metrics::CommonPearson
        | Metrics::CommonA0
        | Metrics::CommonTv0
        | Metrics::CommonCurv0
        | Metrics::CommonAmpP10
        | Metrics::CommonAmpP50
        | Metrics::CommonAmpP90
        | Metrics::CommonTvnP10
        | Metrics::CommonTvnP50
        | Metrics::CommonTvnP90
        | Metrics::CommonCurvnP10
        | Metrics::CommonCurvnP50
        | Metrics::CommonCurvnP90 => {
            let common_stats = thoosa_motifs::common_stats::estimate_channel_stats_from_windows(&motif_slices.f_unsqueeze(0)?, 0.05, 0.95, 0.50, 1e-8)?;
            let stats = common_stats.first().ok_or_else(|| ThoError::new("metric_scalar_for_motif: no common stats for motif"))?;
            return metric.common_stats_value(stats);
        }
        Metrics::NormThickness => norm_thickness(id, motif_slices, 1e-8)?,
        Metrics::InlierRatioRunConfig => inlier_ratio(motif_slices, 4.5, 1e-8)?,
        Metrics::CoherenceRunConfig => coherence(motif_slices, 10, 1e-8)?,
        Metrics::InterestingAmp01 => interesting_amp_01(motif_slices, 0.05, 0.95, 1.0, 1e-8)?,
        Metrics::InterestingTvNorm01 => interesting_tv_norm_01(motif_slices, 0.05, 0.95, 1.0, 1e-8)?,
        Metrics::InterestingCurvatureNorm01 => interesting_curvature_norm_01(motif_slices, 0.05, 0.95, 1.0, 1e-8)?,
        Metrics::InterestingTurningPointsRatio => interesting_turning_points_ratio_peaked_01(motif_slices, 4)?,
        Metrics::InterestingBandEnergyRatio01 => interesting_band_energy_ratio_01(motif_slices, 250.0, 8.0, 30.0, 1e-8)?,
        Metrics::InterestingEntropyGauss01 => interesting_entropy_gauss_01(motif_slices, 0.6, 0.15, 1e-8)?,
        Metrics::InterestingEntropyBand01 => interesting_entropy_band_01(motif_slices, 0.3, 0.85, 1e-8)?,
        Metrics::PeriodicityAcf => periodicity_acf_01_frac(motif_slices, 1./64., 1./32., 0.5,  0.2, 0.25, 0.65, 1e-8)?,
    };

    let raw = score_t.f_double_value(&[])?;
    if raw.is_finite() { Ok(raw as f32) } else { Ok(0.0) }
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
            let mean = distances.f_mean(Kind::Float)?;
            debug!("aggregate_distances, Mean: k={k} distances=\n{distances} mean=\n{mean}");
            k * mean
        }
        Aggr::Min(k) => {
            let min = distances.f_min()?;
            debug!("aggregate_distances, Min: k={k} distances=\n{distances} min=\n{min}");
            k * min
        }
        Aggr::Max(k) => {
            let max = distances.f_max()?;
            debug!("aggregate_distances, Max: k={k} distances=\n{distances} max=\n{max}");
            k * max
        }
        Aggr::Quantile(q) => {
            debug!("aggregate_distances, Quantile: q={q} distances=\n{distances}");
            let q = q.clamp(0.0, 1.0) as f64;
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

    #[test]
    fn test_aggregate_distances() {
        let distances = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let mean_k_half = aggregate_distances(&distances, Aggr::Mean(0.5)).unwrap();
        assert!((mean_k_half - 1.25).abs() < 1e-6, "expected mean([1,2,3,4]) * 0.5 = 1.25, got {mean_k_half}");
        let min_k_2 = aggregate_distances(&distances, Aggr::Min(2.0)).unwrap();
        assert!((min_k_2 - 2.0).abs() < 1e-6, "expected min([1,2,3,4]) * 2.0 = 2.0, got {min_k_2}");
        let max_k_2 = aggregate_distances(&distances, Aggr::Max(2.0)).unwrap();
        assert!((max_k_2 - 8.0).abs() < 1e-6, "expected max([1,2,3,4]) * 2.0 = 8.0, got {max_k_2}");
        let median_lower = aggregate_distances(&distances, Aggr::Quantile(0.5)).unwrap();
        assert!((median_lower - 2.0).abs() < 1e-6, "expected quantile(0.5, lower) for [1,2,3,4] to be 2.0, got {median_lower}");
    }

    #[test]
    fn test_aggregate_distances_with_zeros() {
        let distances = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0]);
        let median_lower = aggregate_distances(&distances, Aggr::Quantile(0.0)).unwrap();
        assert!((median_lower - 1.0).abs() < 1e-6, "expected quantile(0.0, lower) for [1,2,3,4] to be 1.0, got {median_lower}");
    }

    #[test]
    fn test_metrics_all_includes_periodicity_acf_and_scalar_is_finite() {
        assert!(Metrics::all().contains(&Metrics::PeriodicityAcf));

        let t = Tensor::arange(1024, (Kind::Float, Device::Cpu)) / 64;
        let motif = t.sin().view([1, -1]);
        let score = metric_scalar_for_motif(&"periodicity_test", &motif, Metrics::PeriodicityAcf, Aggr::Mean(1.0)).unwrap();

        assert!(score.is_finite(), "expected finite periodicity score, got {score}");
        assert!((0.0..=1.0).contains(&score), "expected periodicity score in [0, 1], got {score}");
    }
}
