use std::any::Any;
use std::fmt::{Debug, Formatter};

use log::{debug, trace};
use num_traits::*;
use rand::Rng;
use rand::seq::SliceRandom;
use statrs::distribution::Normal;
use tch::{IndexOp, Kind, Tensor};
use thoosa_shared::val::Val;
use thoosa_shared::opt_rng::opt_rnd;
use crate::parzen::optimizer::{Distribution, ParzenEstimator};
use crate::parzen::util::uniform_weight;

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
    Optuna,
    Hyperopt,
    Compress,
    Simp,
}

impl BandwidthFn {
    ///returns means, bandwidth
    pub fn clc(&self, samples: Tensor) -> (Tensor, Tensor) {
        match self {
            BandwidthFn::Optuna => {
                todo!()
            }
            BandwidthFn::Hyperopt => {
                todo!()
            }
            BandwidthFn::Compress => {
                todo!()
            }
            BandwidthFn::Simp => {
                // std = means.std(ddof=int(means.size > 1))
                // IQR = np.subtract.reduce(np.percentile(means, [75, 25]))
                // bandwidth = np.full_like(means, 1.059 * min(IQR / 1.34, std) * means.size ** (-0.2))
                let size = samples.size()[0];
                let std = samples.std(size > 1);
                // debug!("std={}", std);
                let q = samples.quantile(&Tensor::from_slice(&[0.75, 0.25]), 0, true, "linear");
                // debug!("q={}", q);
                let iqr = q.i((0, 0)) - q.i((1, 0));
                // debug!("IQR={}", iqr);
                let min = Tensor::cat(&[std.view(1), iqr.view(1) / 1.34], 0).min();
                // debug!("min={}", min);
                let b = 1.059 * min * (size as f64).powf(-0.2);
                // debug!("b={}", b);
                (samples, b)
            }
        }
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

                if *min < magic_factor.into() {
                    magic_factor.into()
                } else {
                    *min
                }
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
    name:Option<String>,
    lb: N,
    ub: N,
    bandwidth_clip: BandwidthFactor<N>,
    bandwidth: BandwidthFn,
    prior: bool,
    quant_step: Option<usize>
}

impl<N> Debug for NumericalParzenEstimator<N>
where
    N: Num + tch::kind::Element + PartialOrd + Copy,
    f64: Into<N>,
    f64: From<N>,

    tch::Scalar: From<N>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Test")
            // .field("lb", &<N as Into<f64>>::into(self.lb))
            // .field("ub", &<N as Into<f64>>::into(self.ub))
            .field("prior", &self.prior)
            .finish()
    }
}

impl<N> ParzenEstimator for NumericalParzenEstimator<N>
where
    N: Num + tch::kind::Element + PartialOrd + Copy,
    f64: Into<N>,
    f64: From<N>,

    tch::Scalar: From<N>,
{
    fn clc_distr(&self, samples: Tensor, _weights: Option<Tensor>) -> Box<dyn Distribution> {
        Box::new(self.clc_distr(samples))
    }

    fn clc_distr_from_vec(&self, samples:&mut dyn Iterator<Item=Val>, _weights: Option<Tensor>) -> Box<dyn Distribution>{
        Box::new(self.clc_distr(Tensor::from_slice(samples.map(|p| p.try_into().unwrap()).collect::<Vec<f32>>().as_slice())))
        // todo!()
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
            quant_step
        };

        i
    }

    fn clc_distr(&self, samples: Tensor) -> NumericalGaussian {
        trace!("clc_distr: samples={samples} kind={:?} ", samples.kind());
        let samples = if self.prior {
            //prior adds midpoint to 'stabilize the density estimation process' (gpt)
            Tensor::cat(
                &[
                    samples,
                    Tensor::from(0.5.into() * (self.lb + self.ub)).view(1),
                ],
                0,
            )
        } else {
            samples
        };

        let size = samples.size()[0];

        let (means, badwidth) = self.bandwidth.clc(samples);
        let domain_range = self.ub - self.lb;

        let min_bandwidth = self.bandwidth_clip.value(size) * domain_range;

        let clipped_bandwidth = badwidth
            .clip(min_bandwidth, 0.5.into() * domain_range)
            .view(1)
            .repeat(size);

        trace!("clipped_bandwidth={}", clipped_bandwidth);
        if self.prior {
            clipped_bandwidth
                .i(size - 1)
                .copy_(&Tensor::from(<f64>::into(domain_range.into())));
        }
        trace!("means={} clipped_bandwidth={}", means, clipped_bandwidth);
        let (norm_consts, logpdf_consts) =
            Self::calculate_norm_consts(&self.lb, &self.ub, &means, &clipped_bandwidth);

        let weights = uniform_weight(size); //TODO

        NumericalGaussian::new(
            means,
            clipped_bandwidth,
            norm_consts,
            logpdf_consts,
            weights,
            self.lb.into(),
            self.ub.into(),
            self.quant_step.clone()
        )
    }

    fn calculate_norm_consts(lb: &N, ub: &N, means: &Tensor, stds: &Tensor) -> (Tensor, Tensor) {
        let std_sqrt_2 = std::f64::consts::SQRT_2 * stds;
        let zl = (means.negative() + *lb) / &std_sqrt_2;
        let zu = (means.negative() + *ub) / &std_sqrt_2;
        let norm_consts = (zu.erf() - zl.erf()).reciprocal() * 2;
        let logpdf_consts = (&norm_consts / (SQR2PI * stds)).log();
        (norm_consts, logpdf_consts)
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct NumericalGaussian {
    means: Tensor,
    stds: Tensor,
    norm_consts: Tensor,
    logpdf_consts: Tensor,
    weights: Tensor,
    lb: f64,
    ub: f64,
    normal_dists: Vec<Normal>,
    quant_step: Option<usize>,
}


impl NumericalGaussian {
    pub fn new(
        means: Tensor,
        stds: Tensor,
        norm_consts: Tensor,
        logpdf_consts: Tensor,
        weights: Tensor,
        lb: f64,
        ub: f64,
        quant_step: Option<usize>,
    ) -> Self {
        Self {
            normal_dists: means
                .iter::<f64>()
                .unwrap()
                .zip(stds.iter::<f64>().unwrap())
                .map(|(mean, std)| {
                    // debug!("NumericalGaussian::new: mean={mean:} std={std}");
                    Normal::new(mean, std).unwrap()
                })
                .collect::<Vec<_>>(),

            means,
            stds,
            norm_consts,
            logpdf_consts,
            weights,
            lb,
            ub,
            quant_step
        }
    }

    #[allow(dead_code)]
    pub fn pdf(&self, x: &Tensor) -> Tensor {
        let norm_consts = &self.norm_consts / (SQR2PI * &self.stds);
        let mahalanobis = ((x.view((-1, 1)).repeat([1, self.means.size()[0]])
            - &self.means.view((1, -1)))
            / &self.stds)
            .pow_tensor_scalar(2);
        self.weights
            .multiply(&norm_consts)
            .multiply(&mahalanobis.multiply_scalar(-0.5).exp())
            .sum_dim_intlist(1, false, None)
    }

    fn _basis_loglikelihood(&self, x: &Tensor) -> Tensor {
        let mahalanobis =
            ((x - self.means.view((-1, 1))) / self.stds.view((-1, 1))).pow_tensor_scalar(2);

        self.logpdf_consts.view((-1, 1)) - 0.5 * mahalanobis
    }

    fn _sample(&self, n_samples: usize) -> Vec<f64> {
        trace!("self.means={:?} self.stds={:?}", self.means, self.stds);

        (0..n_samples)
            .map(|_i| {
                let distr = self.normal_dists.choose(&mut opt_rnd()).expect("distr");

                loop {
                    let val = opt_rnd().sample(distr);

                    if self.lb <= val && val <= self.ub {
                        if let Some(q_val) = self.quant_step.as_ref() {
                            return (((val - self.lb) / *q_val as f64).round() as usize * *q_val) as f64 + self.lb; //TODO hard bounds
                        } else {
                            return val;
                        }
                    }
                }
            })
            .collect()
    }
}



impl Distribution for NumericalGaussian {
    fn sample(&self, n_samples: usize) -> Vec<Val> {
        self._sample(n_samples).into_iter().map(|v|v.into()).collect()
    }

    fn basis_loglikelihood(&self, x:&mut dyn Iterator<Item=&Val>) -> Tensor {
        self._basis_loglikelihood(&Tensor::from_slice(x.map(|v|v.try_into().unwrap()).collect::<Vec<f64>>().as_slice()))
    }

    fn weights(&self) -> &Tensor {
        &self.weights
    }
}

///basis_loglikelihoods - params x samples.size + prior x candidates
pub fn compute_config_loglikelihoods(
    basis_loglikelihoods: &Tensor,
    weights: Option<&Tensor>,
) -> Tensor {
    // debug!("basis_loglikelihoods={basis_loglikelihoods} size={:?}", basis_loglikelihoods.size());
    // debug!("weights={weights:?}");

    let config_basis_loglikelihoods = basis_loglikelihoods.sum_dim_intlist(0, false, Kind::Float); //.view((-1, 1));//TODO kind

    // debug!("config_basis_loglikelihoods={config_basis_loglikelihoods}");
    // debug!("config_basis_loglikelihoods.size={:?} weights.size()={:?}", config_basis_loglikelihoods.size(), weights.expect("weights").size());

    let config_basis_loglikelihoods = if let Some(weights) = weights {
        config_basis_loglikelihoods + weights.view((-1, 1)).log()
    } else {
        config_basis_loglikelihoods
    };

    config_basis_loglikelihoods.logsumexp(0, false)
}

#[cfg(test)]
mod tests {
    use log::debug;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;

    use super::{BandwidthFactor, compute_config_loglikelihoods, NumericalParzenEstimator};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_npe() {
        StdRng::seed_from_u64(42);

        let npe = NumericalParzenEstimator::new(
            None,
            // Tensor::from_slice(&[-1.0, 0.0, 1.0]),
            -3.0f64,
            3.0f64,
            BandwidthFactor::default(),
            true,
            None,
        );

        // debug!("npe.weights={}", npe.weights);

        // assert_eq!(
        //     Tensor::from_slice(&[0.2500, 0.2500, 0.2500, 0.2500]).round_decimals(4),
        //     npe.weights.round_decimals(4)
        // );

        let gaussian = npe.clc_distr(Tensor::from_slice(&[-1.0, 0.0, 1.0]));
        debug!("x={:?}", gaussian.means);
        assert_eq!(
            Tensor::from_slice(&[-1., 0., 1., 0.]).round_decimals(4),
            gaussian.means.round_decimals(4)
        );

        debug!("x={:?}", gaussian.stds);
        assert_eq!(
            Tensor::from_slice(&[0.29946714, 0.29946714, 0.29946714, 6.]).round_decimals(4),
            gaussian.stds.round_decimals(4)
        );

        debug!("x={:?}", gaussian.norm_consts);
        assert_eq!(
            Tensor::from_slice(&[1., 1., 1., 2.61147797]).round_decimals(4),
            gaussian.norm_consts.round_decimals(4)
        );
        debug!("x={:?}", gaussian.logpdf_consts);
        assert_eq!(
            Tensor::from_slice(&[0.28681207, 0.28681207, 0.28681207, -1.75078167])
                .round_decimals(4),
            gaussian.logpdf_consts.round_decimals(4)
        );

        let pdf = gaussian.pdf(&Tensor::linspace(-3.0, 3.0, 100, FLOAT_CPU));
        debug!("pdf={pdf}");

        assert_eq!(
            Tensor::from_slice(&[
                0.03830879, 0.03850079, 0.03868981, 0.03887579, 0.03905869, 0.03923846, 0.03941508,
                0.0395886, 0.03975921, 0.03992746, 0.0400948, 0.04026458, 0.04044426, 0.04064949,
                0.04091152, 0.04128953, 0.04188982, 0.04289303, 0.04458879, 0.04741365, 0.05198243,
                0.0590974, 0.06971465, 0.08484755, 0.10539555, 0.13190615, 0.16430442, 0.20165279,
                0.24202132, 0.28254343, 0.31969902, 0.34980887, 0.36965828, 0.37711655, 0.37160411,
                0.35428949, 0.32796568, 0.29663682, 0.26491237, 0.23733607, 0.21776788, 0.20889924,
                0.21193992, 0.22648462, 0.25055861, 0.2808435, 0.31308289, 0.34264849, 0.36520985,
                0.37740817, 0.37740817, 0.36520985, 0.34264849, 0.31308289, 0.2808435, 0.25055861,
                0.22648462, 0.21193992, 0.20889924, 0.21776788, 0.23733607, 0.26491237, 0.29663682,
                0.32796568, 0.35428949, 0.37160411, 0.37711655, 0.36965828, 0.34980887, 0.31969902,
                0.28254343, 0.24202132, 0.20165279, 0.16430442, 0.13190615, 0.10539555, 0.08484755,
                0.06971465, 0.0590974, 0.05198243, 0.04741365, 0.04458879, 0.04289303, 0.04188982,
                0.04128953, 0.04091152, 0.04064949, 0.04044426, 0.04026458, 0.0400948, 0.03992746,
                0.03975921, 0.0395886, 0.03941508, 0.03923846, 0.03905869, 0.03887579, 0.03868981,
                0.03850079, 0.03830879
            ])
            .round_decimals(4),
            pdf.round_decimals(4)
        );

        let basis_loglikelihood =
            gaussian._basis_loglikelihood(&Tensor::linspace(0., 10., 10, FLOAT_CPU));
        debug!("basis_loglikelihood={basis_loglikelihood}");
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

        let samples = gaussian._sample(11);

        debug!("samples={samples:?}");
    }

    #[test]
    fn test_compute_config_loglikelihoods() {
        assert_eq!(
            Tensor::from_slice(&[4.86626684, 5.97737795, 7.08848906, 8.19960018, 9.31071129])
                .round_decimals(3),
            compute_config_loglikelihoods(
                &Tensor::linspace(0., 10., 10, FLOAT_CPU).view((1, 2, 5)),
                Some(&Tensor::from_slice(&[0.5, 0.5]))
            )
            .round_decimals(3)
        );

        assert_eq!(
            Tensor::from_slice(&[
                14.82974904,
                15.8642318,
                16.89871456,
                17.93319732,
                18.96768008
            ])
            .round_decimals(3),
            compute_config_loglikelihoods(
                &Tensor::linspace(0., 10., 30, FLOAT_CPU).view((3, 2, 5)),
                Some(&Tensor::from_slice(&[0.5, 0.5]))
            )
            .round_decimals(3)
        )
    }
}

