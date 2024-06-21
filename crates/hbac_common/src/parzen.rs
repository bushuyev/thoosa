use std::cmp::{max, max_by};
use std::ops::Div;
use num_traits::*;
use tch::kind::FLOAT_CPU;
use tch::{IndexOp, Tensor};

pub enum BandwidthFactor<N> where N: Float, f64: Into<N>  {
    Min(N),
    MagicClipExponent(N,N)
}

//TODO Sheather and Jones method sheather91_kernel_density.pdf
pub enum BandwidthFn {
    Optuna,
    Hyperopt,
    Compress,
    Simp
}

impl BandwidthFn {

    ///returns means, bandwidth
    pub fn clc(&self, samples:Tensor) -> (Tensor, Tensor){
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
                // println!("std={}", std);
                let q= samples.quantile(&Tensor::from_slice(&[0.75, 0.25]), 0, true, "linear");
                // println!("q={}", q);
                let iqr = q.i((0, 0)) - q.i((1, 0));
                // println!("IQR={}", iqr);
                let min = Tensor::cat(&[std.view(1), iqr.view(1)/1.34], 0).min();
                // println!("min={}", min);
                let b = 1.059 * min * (size as f64).powf(-0.2);
                // println!("b={}", b);
                (samples, b)
            }
        }
    }
}

impl <N> BandwidthFactor<N> where N: Float + PartialOrd, f64: Into<N>, f64: From<N> {
    pub fn value(&self, size:i64) -> N {
        match self {
            BandwidthFactor::Min(min) => {*min}
            BandwidthFactor::MagicClipExponent(min, magic_clip_exponent) => {
                let magic_factor = (1.0/size as f64).powf((*magic_clip_exponent).into());

                if *min < magic_factor.into() {magic_factor.into()} else {*min}
            }
        }
    }
}

impl <N> Default for BandwidthFactor<N> where N: Float, f64: Into<N>  {
    fn default() -> Self {
        BandwidthFactor::Min(1e-2.into())
    }
}

struct NumericalParzenEstimator<N> where N: Float + tch::kind::Element + PartialOrd, f64: Into<N>, f64: From<N>, tch::Scalar: From<N> {
    samples: Tensor,
    lb: N,
    ub: N,
    bandwidth_clip: BandwidthFactor<N>,
    bandwidth:BandwidthFn,
    weights:Tensor,
    prior:bool
}

const SQR2PI:f64 = 2.5066282746310002;

impl<N> NumericalParzenEstimator<N> where N: Float + tch::kind::Element + PartialOrd, f64: Into<N>, f64: From<N>, tch::Scalar: From<N>{

    pub fn new(samples: Tensor, lb: N, ub: N, weights:Option<Tensor>, bandwidth_factor: BandwidthFactor<N>, prior:bool) -> Self {
        assert_eq!(1, samples.size().len());

        let hard_lb = lb;
        let hard_ub = ub;

        let samples = if prior {
            Tensor::cat(&[
                samples,
                Tensor::from(0.5.into() * (hard_lb + hard_ub)).view(1)
            ], 0)
        } else {
            samples
        };

        let size = samples.size()[0];
        Self {
            samples,
            lb,
            ub,
            bandwidth_clip: bandwidth_factor,
            bandwidth: BandwidthFn::Simp,
            weights: weights.unwrap_or(Tensor::ones(&[size], FLOAT_CPU)/size),
            prior
        }
    }

    pub fn calculate(&self) -> PEGaussian {
        let (means, badwidth) = self.bandwidth.clc(self.samples.shallow_clone());
        let domain_range = self.ub - self.lb;

        let size = self.samples.size()[0];
        let min_bandwidth = self.bandwidth_clip.value(size) * domain_range;

        let clipped_bandwidth = badwidth.clip( min_bandwidth, 0.5.into() * domain_range).view(1).repeat(size);
        println!("clipped_bandwidth={}", clipped_bandwidth);
        if self.prior{
            clipped_bandwidth.i(size-1).copy_(&Tensor::from(<f64>::into(domain_range.into())));
        }
        println!("means={} clipped_bandwidth={}", means, clipped_bandwidth);
        let (norm_consts,  logpdf_consts) = Self::calculate_norm_consts(&self.lb, &self.ub, &means, &clipped_bandwidth);

        println!("norm_consts={} logpdf_consts={}", norm_consts, logpdf_consts);

        PEGaussian::new(means, clipped_bandwidth, norm_consts, logpdf_consts)
    }

    fn calculate_norm_consts(lb: &N, ub: &N, means: &Tensor, stds: &Tensor) -> (Tensor, Tensor){
        let std_sqrt_2 = std::f64::consts::SQRT_2 * stds;
        let zl = (means.negative() + *lb) / &std_sqrt_2;
        let zu = (means.negative() + *ub) / &std_sqrt_2;
        let norm_consts = (zu.erf() - zl.erf()).reciprocal() * 2;
        let logpdf_consts = (&norm_consts / (SQR2PI * stds)).log();
        return (norm_consts, logpdf_consts)
    }
}

#[derive(Debug)]
struct PEGaussian {
    means:Tensor,
    stds:Tensor,
    norm_consts:Tensor,
    logpdf_consts:Tensor
}

impl PEGaussian {
    pub fn new(means: Tensor, stds: Tensor, norm_consts: Tensor, logpdf_consts: Tensor) -> Self {
        Self { means, stds, norm_consts, logpdf_consts }
    }
}

#[cfg(test)]
mod tests {
    use tch::Tensor;
    use crate::parzen::{BandwidthFactor, NumericalParzenEstimator};

    #[test]
    fn test_0() {
        let npe = NumericalParzenEstimator::new(
            Tensor::from_slice(&[-1.0, 0.0, 1.0]),
            -3.0f64,
            3.0f64,
            None,
            BandwidthFactor::default(),
            true
        );

        println!("npe.weights={}", npe.weights);

        assert_eq!(
            Tensor::from_slice(&[0.2500, 0.2500, 0.2500, 0.2500]).round_decimals(4),
            npe.weights.round_decimals(4)
        );

        let x = npe.calculate();
        println!("x={:?}", x.means);
        assert_eq!(
            Tensor::from_slice(&[-1.,  0.,  1.,  0.]).round_decimals(4),
            x.means.round_decimals(4)
        );

        println!("x={:?}", x.stds);
        assert_eq!(
            Tensor::from_slice(&[0.29946714, 0.29946714, 0.29946714, 6.        ]).round_decimals(4),
            x.stds.round_decimals(4)
        );

        println!("x={:?}", x.norm_consts);
        assert_eq!(
            Tensor::from_slice(&[1.,         1.,         1.,         2.61147797]).round_decimals(4),
            x.norm_consts.round_decimals(4)
        );
        println!("x={:?}", x.logpdf_consts);
        assert_eq!(
            Tensor::from_slice(&[0.28681207,  0.28681207,  0.28681207, -1.75078167]).round_decimals(4),
            x.logpdf_consts.round_decimals(4)
        );
    }
}