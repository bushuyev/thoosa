use crate::tch_error::tch_result;
use std::fmt::{Debug, Formatter};

use crate::parzen::optimizer::{Distribution, ParzenEstimator};
use crate::parzen::util::uniform_weight;
use log::trace;
use num_traits::*;
use rand::RngExt;
use rand::seq::IndexedRandom;
use tch::{Kind, Tensor};
use thoosa_shared::opt_rng::opt_rnd;
use thoosa_shared::val::Val;

#[allow(dead_code)]
pub enum BandwidthFactor<N>
where
    N: Num,
    f64: Into<N>,
    f64: From<N>,
{
    Min(N),
    MagicClipExponent(N, N),
}

//TODO Sheather and Jones method sheather91_kernel_density.pdf
#[allow(dead_code)]
pub enum BandwidthFn {
    // Only expose bandwidth estimators implemented by this backend.
    Simp,
}

impl BandwidthFn {
    ///returns means, bandwidth
    pub fn clc(&self, samples: Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        // Keep density estimation failures on the ordinary typed tensor error path.
        anyhow::ensure!(samples.dim() == 1 && samples.size()[0] > 0, "bandwidth requires a nonempty sample vector");
        let size = samples.size()[0];
        let std = tch_result(samples.f_std(size > 1))?;
        let q = tch_result(Tensor::f_from_slice(&[0.75, 0.25]))?;
        let q = tch_result(samples.f_quantile(&q, 0, true, "linear"))?;
        let iqr = tch_result(tch_result(q.f_select(0, 0))?.f_sub(&tch_result(q.f_select(0, 1))?))?;
        let scaled_iqr = tch_result(iqr.f_div_scalar(1.34))?;
        let min = tch_result(Tensor::f_cat(&[tch_result(std.f_view([1]))?, tch_result(scaled_iqr.f_view([1]))?], 0))?;
        let bandwidth = tch_result(tch_result(min.f_min())?.f_mul_scalar(1.059 * (size as f64).powf(-0.2)))?;
        Ok((samples, bandwidth))
    }
}

impl<N> BandwidthFactor<N>
where
    N: Num + PartialOrd + Copy,
    f64: Into<N>,
    f64: From<N>,
{
    pub fn value(&self, size: i64) -> N {
        match self {
            BandwidthFactor::Min(min) => *min,
            BandwidthFactor::MagicClipExponent(min, magic_clip_exponent) => {
                let magic_factor = (1.0 / size as f64).powf((*magic_clip_exponent).into());

                if *min < magic_factor.into() { magic_factor.into() } else { *min }
            }
        }
    }
}

impl<N> Default for BandwidthFactor<N>
where
    N: Num + tch::kind::Element + PartialOrd + Copy,
    f64: Into<N>,
    f64: From<N>,
{
    fn default() -> Self {
        BandwidthFactor::Min(1e-2.into())
    }
}
#[allow(dead_code)]
pub struct NumericalParzenEstimator<N>
where
    N: Num + tch::kind::Element + PartialOrd + Copy,
    f64: Into<N>,
    f64: From<N>,
    tch::Scalar: From<N>,
{
    name: Option<String>,
    lb: N,
    ub: N,
    bandwidth_clip: BandwidthFactor<N>,
    bandwidth: BandwidthFn,
    prior: bool,
    quant_step: Option<usize>,
}

impl<N> Debug for NumericalParzenEstimator<N>
where
    N: Num + tch::kind::Element + PartialOrd + Copy,
    f64: Into<N>,
    f64: From<N>,
    tch::Scalar: From<N>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Test").field("lb", &<N as Into<f64>>::into(self.lb)).field("ub", &<N as Into<f64>>::into(self.ub)).field("prior", &self.prior).finish()
    }
}

impl<N> ParzenEstimator for NumericalParzenEstimator<N>
where
    N: Num + tch::kind::Element + PartialOrd + Copy + Send,
    f64: Into<N>,
    f64: From<N>,
    tch::Scalar: From<N>,
{
    fn clc_distr(&self, samples: Tensor, _weights: Option<Tensor>) -> anyhow::Result<Box<dyn Distribution>> {
        Ok(Box::new(self.clc_distr(samples)?))
    }

    fn clc_distr_from_vec(&self, samples: &mut dyn Iterator<Item = Val>, _weights: Option<Tensor>) -> anyhow::Result<Box<dyn Distribution>> {
        let samples = samples.map(f32::try_from).collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Box::new(self.clc_distr(tch_result(Tensor::f_from_slice(&samples))?)?))
    }
}

const SQR2PI: f64 = 2.5066282746310002;

impl<N> NumericalParzenEstimator<N>
where
    N: Num + tch::kind::Element + PartialOrd + Copy,
    f64: Into<N>,
    f64: From<N>,
    tch::Scalar: From<N>,
{
    pub fn new(name: Option<String>, lb: N, ub: N, bandwidth_factor: BandwidthFactor<N>, prior: bool, quant_step: Option<usize>) -> Self {
        let i = Self {
            name,
            lb,
            ub,
            bandwidth_clip: bandwidth_factor,
            bandwidth: BandwidthFn::Simp,
            prior,
            quant_step,
        };

        i
    }

    fn clc_distr(&self, samples: Tensor) -> anyhow::Result<NumericalGaussian> {
        anyhow::ensure!(samples.dim() == 1, "numerical samples must be a vector");
        let device = samples.device();
        let samples = if self.prior {
            let midpoint: f64 = (<f64 as From<N>>::from(self.lb) + <f64 as From<N>>::from(self.ub)) * 0.5;
            let prior = tch_result(Tensor::f_from_slice(&[midpoint]))?;
            let prior = tch_result(prior.f_to_device(device))?;
            tch_result(Tensor::f_cat(&[samples, prior], 0))?
        } else {
            samples
        };
        let size = samples.size()[0];
        let (means, bandwidth) = self.bandwidth.clc(samples)?;
        let domain_range = self.ub - self.lb;
        let min_bandwidth = self.bandwidth_clip.value(size) * domain_range;
        let clipped = tch_result(bandwidth.f_clamp(min_bandwidth, 0.5.into() * domain_range))?;
        let clipped = tch_result(tch_result(clipped.f_view([1]))?.f_repeat([size]))?;
        if self.prior {
            let _ = tch_result(tch_result(clipped.f_select(0, size - 1))?.f_fill_(domain_range))?;
        }
        let (norm_consts, logpdf_consts) = Self::calculate_norm_consts(&self.lb, &self.ub, &means, &clipped)?;
        let weights = uniform_weight(size, device)?;
        NumericalGaussian::new(means, clipped, norm_consts, logpdf_consts, weights, self.lb.into(), self.ub.into(), self.quant_step)
    }

    fn calculate_norm_consts(lb: &N, ub: &N, means: &Tensor, stds: &Tensor) -> anyhow::Result<(Tensor, Tensor)> {
        let std_sqrt_2 = tch_result(stds.f_mul_scalar(std::f64::consts::SQRT_2))?;
        let negative = tch_result(means.f_neg())?;
        let zl = tch_result(tch_result(negative.f_add_scalar(*lb))?.f_div(&std_sqrt_2))?;
        let zu = tch_result(tch_result(negative.f_add_scalar(*ub))?.f_div(&std_sqrt_2))?;
        let diff = tch_result(tch_result(zu.f_erf())?.f_sub(&tch_result(zl.f_erf())?))?;
        let norm_consts = tch_result(tch_result(diff.f_reciprocal())?.f_mul_scalar(2))?;
        let scale = tch_result(stds.f_mul_scalar(SQR2PI))?;
        let logpdf_consts = tch_result(tch_result(norm_consts.f_div(&scale))?.f_log())?;
        Ok((norm_consts, logpdf_consts))
    }
}

#[derive(Debug)]
pub struct NumericalGaussian {
    means: Tensor,
    stds: Tensor,
    norm_consts: Tensor,
    logpdf_consts: Tensor,
    weights: Tensor,
    lb: f64,
    ub: f64,
    normal_dists: Vec<(f64, f64)>,
    quant_step: Option<usize>,
}

impl NumericalGaussian {
    pub fn new(means: Tensor, stds: Tensor, norm_consts: Tensor, logpdf_consts: Tensor, weights: Tensor, lb: f64, ub: f64, quant_step: Option<usize>) -> anyhow::Result<Self> {
        let means_vec = tch_result(Vec::<f64>::try_from(&means))?;
        let stds_vec = tch_result(Vec::<f64>::try_from(&stds))?;
        anyhow::ensure!(!means_vec.is_empty() && means_vec.len() == stds_vec.len(), "Gaussian means and deviations must be nonempty and aligned");
        Ok(Self {
            normal_dists: means_vec.into_iter().zip(stds_vec).collect(),

            means,
            stds,
            norm_consts,
            logpdf_consts,
            weights,
            lb,
            ub,
            quant_step,
        })
    }

    pub fn pdf(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let norm = tch_result(self.norm_consts.f_div(&tch_result(self.stds.f_mul_scalar(SQR2PI))?))?;
        let x = tch_result(tch_result(x.f_view([-1, 1]))?.f_repeat([1, self.means.size()[0]]))?;
        let delta = tch_result(x.f_sub(&tch_result(self.means.f_view([1, -1]))?))?;
        let mahalanobis = tch_result(tch_result(delta.f_div(&self.stds))?.f_pow_tensor_scalar(2))?;
        let exponential = tch_result(tch_result(mahalanobis.f_mul_scalar(-0.5))?.f_exp())?;
        let weighted = tch_result(tch_result(self.weights.f_mul(&norm))?.f_mul(&exponential))?;
        Ok(tch_result(weighted.f_sum_dim_intlist(1, false, None))?)
    }

    fn basis_loglikelihood(&self, x: &Tensor) -> anyhow::Result<Tensor> {
        let delta = tch_result(x.f_sub(&tch_result(self.means.f_view([-1, 1]))?))?;
        let ratio = tch_result(delta.f_div(&tch_result(self.stds.f_view([-1, 1]))?))?;
        let mahalanobis = tch_result(tch_result(ratio.f_pow_tensor_scalar(2))?.f_mul_scalar(0.5))?;
        Ok(tch_result(tch_result(self.logpdf_consts.f_view([-1, 1]))?.f_sub(&mahalanobis))?)
    }

    fn _sample(&self, n_samples: usize) -> Vec<f64> {
        trace!("NumericalGaussian::sample: components={}", self.normal_dists.len());

        (0..n_samples)
            .map(|_i| {
                let lb = self.lb;
                let ub = self.ub;
                let quant_step = self.quant_step;
                let distr = opt_rnd(|rng| self.normal_dists.choose(rng)).expect("distr");
                let mean = distr.0;
                let std = distr.1;

                if lb.is_nan() || ub.is_nan() {
                    return mean;
                }
                if lb == ub {
                    return lb;
                }
                if lb > ub {
                    return mean;
                }

                let quantize = |val: f64| -> f64 {
                    let clamped = val.clamp(lb, ub);
                    if let Some(q_val) = quant_step {
                        if q_val == 0 {
                            return clamped;
                        }
                        let q_val_f = q_val as f64;
                        let snapped = ((clamped - lb) / q_val_f).round() * q_val_f + lb;
                        snapped.clamp(lb, ub)
                    } else {
                        clamped
                    }
                };

                if !mean.is_finite() || !std.is_finite() || std <= 0.0 {
                    return quantize(mean);
                }

                let mut rng = rand::rng();
                const MAX_SAMPLE_ATTEMPTS: usize = 512;

                for _ in 0..MAX_SAMPLE_ATTEMPTS {
                    // Box-Muller transform using non-deterministic RNG (acceptable for sampling)
                    let u1: f64 = rng.random();
                    if u1 <= 0.0 {
                        continue;
                    }
                    let u2: f64 = rng.random();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    let val = mean + std * z;

                    if lb <= val && val <= ub {
                        return quantize(val);
                    }
                }

                quantize(mean)
            })
            .collect()
    }
}

impl Distribution for NumericalGaussian {
    fn sample(&self, n_samples: usize) -> anyhow::Result<Vec<Val>> {
        Ok(self._sample(n_samples).into_iter().map(|v| v.into()).collect())
    }

    fn basis_loglikelihood(&self, x: &mut dyn Iterator<Item = &Val>) -> anyhow::Result<Tensor> {
        let values = x.map(f64::try_from).collect::<anyhow::Result<Vec<_>>>()?;
        self.basis_loglikelihood(&tch_result(Tensor::f_from_slice(&values))?)
    }

    fn weights(&self) -> &Tensor {
        &self.weights
    }
}

///basis_loglikelihoods - params x samples.size + prior x candidates
pub fn compute_config_loglikelihoods(basis_loglikelihoods: &Tensor, weights: Option<&Tensor>) -> anyhow::Result<Tensor> {
    let sums = tch_result(basis_loglikelihoods.f_sum_dim_intlist(0, false, Kind::Float))?;
    let sums = if let Some(weights) = weights {
        let logs = tch_result(tch_result(weights.f_view([-1, 1]))?.f_log())?;
        tch_result(sums.f_add(&logs))?
    } else {
        sums
    };
    Ok(tch_result(sums.f_logsumexp(0, false))?)
}

#[cfg(test)]
mod tests {
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;

    use super::{BandwidthFactor, NumericalGaussian, NumericalParzenEstimator, compute_config_loglikelihoods};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_density_tensor_failures_are_returned() -> anyhow::Result<()> {
        let estimator = NumericalParzenEstimator::new(None, -3.0f64, 3.0f64, BandwidthFactor::default(), true, None);
        assert!(estimator.clc_distr(Tensor::f_zeros([2, 2], FLOAT_CPU)?).is_err());
        let density = estimator.clc_distr(Tensor::f_from_slice(&[0.0f64, 1.0])?)?;
        let error = density.basis_loglikelihood(&Tensor::f_zeros([2, 2, 2], FLOAT_CPU)?).unwrap_err();
        assert!(error.chain().any(|cause| cause.downcast_ref::<tch::TchError>().is_some()));
        Ok(())
    }

    #[test]
    fn test_npe() {
        StdRng::seed_from_u64(42);

        let npe = NumericalParzenEstimator::new(None, -3.0f64, 3.0f64, BandwidthFactor::default(), true, None);

        let gaussian = npe.clc_distr(Tensor::from_slice(&[-1.0, 0.0, 1.0])).unwrap();

        assert_eq!(Tensor::from_slice(&[-1., 0., 1., 0.]).round_decimals(4), gaussian.means.round_decimals(4));

        assert_eq!(Tensor::from_slice(&[0.29946714, 0.29946714, 0.29946714, 6.]).round_decimals(4), gaussian.stds.round_decimals(4));

        assert_eq!(Tensor::from_slice(&[1., 1., 1., 2.61147797]).round_decimals(4), gaussian.norm_consts.round_decimals(4));

        assert_eq!(Tensor::from_slice(&[0.28681207, 0.28681207, 0.28681207, -1.75078167]).round_decimals(4), gaussian.logpdf_consts.round_decimals(4));

        let pdf = gaussian.pdf(&Tensor::linspace(-3.0, 3.0, 100, FLOAT_CPU)).unwrap();

        assert_eq!(
            Tensor::from_slice(&[
                0.03830879, 0.03850079, 0.03868981, 0.03887579, 0.03905869, 0.03923846, 0.03941508, 0.0395886, 0.03975921, 0.03992746, 0.0400948, 0.04026458, 0.04044426, 0.04064949, 0.04091152, 0.04128953, 0.04188982, 0.04289303, 0.04458879, 0.04741365, 0.05198243, 0.0590974, 0.06971465,
                0.08484755, 0.10539555, 0.13190615, 0.16430442, 0.20165279, 0.24202132, 0.28254343, 0.31969902, 0.34980887, 0.36965828, 0.37711655, 0.37160411, 0.35428949, 0.32796568, 0.29663682, 0.26491237, 0.23733607, 0.21776788, 0.20889924, 0.21193992, 0.22648462, 0.25055861, 0.2808435,
                0.31308289, 0.34264849, 0.36520985, 0.37740817, 0.37740817, 0.36520985, 0.34264849, 0.31308289, 0.2808435, 0.25055861, 0.22648462, 0.21193992, 0.20889924, 0.21776788, 0.23733607, 0.26491237, 0.29663682, 0.32796568, 0.35428949, 0.37160411, 0.37711655, 0.36965828, 0.34980887,
                0.31969902, 0.28254343, 0.24202132, 0.20165279, 0.16430442, 0.13190615, 0.10539555, 0.08484755, 0.06971465, 0.0590974, 0.05198243, 0.04741365, 0.04458879, 0.04289303, 0.04188982, 0.04128953, 0.04091152, 0.04064949, 0.04044426, 0.04026458, 0.0400948, 0.03992746, 0.03975921,
                0.0395886, 0.03941508, 0.03923846, 0.03905869, 0.03887579, 0.03868981, 0.03850079, 0.03830879
            ])
            .round_decimals(4),
            pdf.round_decimals(4)
        );

        let basis_loglikelihood = gaussian.basis_loglikelihood(&Tensor::linspace(0., 10., 10, FLOAT_CPU)).unwrap();

        assert_eq!(
            Tensor::from_slice(&[
                -5.28853193e+00,
                -2.45613260e+01,
                -5.76004016e+01,
                -1.04405759e+02,
                -1.64977397e+02,
                -2.39315317e+02,
                -3.27419519e+02,
                -4.29290002e+02,
                -5.44926766e+02,
                -6.74329812e+02,
                2.86812067e-01,
                -6.59632868e+00,
                -2.72457509e+01,
                -6.16614546e+01,
                -1.09843440e+02,
                -1.71791706e+02,
                -2.47506255e+02,
                -3.36987084e+02,
                -4.40234195e+02,
                -5.57247588e+02,
                -5.28853193e+00,
                2.17980660e-01,
                -8.04178823e+00,
                -3.00678386e+01,
                -6.58601705e+01,
                -1.15418784e+02,
                -1.78743679e+02,
                -2.55834855e+02,
                -3.46692313e+02,
                -4.51316052e+02,
                -1.75078167e+00,
                -1.76792845e+00,
                -1.81936877e+00,
                -1.90510266e+00,
                -2.02513009e+00,
                -2.17945108e+00,
                -2.36806562e+00,
                -2.59097371e+00,
                -2.84817536e+00,
                -3.13967056e+00,
            ])
            .view((4, 10))
            .round_decimals(3),
            basis_loglikelihood.round_decimals(3)
        );

        // let samples = gaussian._sample(11);
        // debug!("samples={samples:?}");
    }

    #[test]
    fn test_compute_config_loglikelihoods() {
        assert_eq!(
            Tensor::from_slice(&[4.86626684, 5.97737795, 7.08848906, 8.19960018, 9.31071129]).round_decimals(3),
            compute_config_loglikelihoods(&Tensor::linspace(0., 10., 10, FLOAT_CPU).view((1, 2, 5)), Some(&Tensor::from_slice(&[0.5, 0.5]))).unwrap().round_decimals(3)
        );

        assert_eq!(
            Tensor::from_slice(&[14.82974904, 15.8642318, 16.89871456, 17.93319732, 18.96768008]).round_decimals(3),
            compute_config_loglikelihoods(&Tensor::linspace(0., 10., 30, FLOAT_CPU).view((3, 2, 5)), Some(&Tensor::from_slice(&[0.5, 0.5]))).unwrap().round_decimals(3)
        )
    }

    #[test]
    //std=0 and mean outside bounds
    fn test_numerical_gaussian_sample_degenerate_out_of_bounds() {
        let gaussian = NumericalGaussian::new(Tensor::from_slice(&[10.0]), Tensor::from_slice(&[0.0]), Tensor::from_slice(&[1.0]), Tensor::from_slice(&[0.0]), Tensor::from_slice(&[1.0]), 0.0, 1.0, None).unwrap();
        let samples = gaussian._sample(3);
        assert!(samples.iter().all(|v| (0.0..=1.0).contains(v)), "samples={samples:?}");
        assert_eq!(vec![1.0, 1.0, 1.0], samples);
    }
}
