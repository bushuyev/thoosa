use crate::tch_error::tch_result;
use crate::parzen::optimizer::{Distribution, ParzenEstimator};
use crate::parzen::util::uniform_weight;
use rand::distr::Distribution as RandDistribution;
use rand::distr::weighted::WeightedIndex;
use tch::Kind::Float;
use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::Tensor;
use thoosa_shared::opt_rng::opt_rnd;
use thoosa_shared::val::Val;

pub struct CategoricalParzenEstimator {
    #[allow(dead_code)]
    name: Option<String>,
    top: f32,
    bottom: f32,
    uniform: f32,
    n_choices: usize,
    prior: bool,
}

impl CategoricalParzenEstimator {
    pub fn new(name: Option<String>, n_choices: usize, prior: bool) -> Self {
        let top = 1.0;
        Self {
            name,
            top, //TODO The hyperparam of the categorical kernel. It defines the prob of the top category. ?
            bottom: (1. - top) / (n_choices - 1) as f32,
            uniform: 1. / n_choices as f32,
            n_choices,

            prior,
        }
    }

    fn _clc_distr(&self, samples: Tensor, weights: Option<Tensor>) -> anyhow::Result<CategoricalGaussian> {
        anyhow::ensure!(self.n_choices > 0 && samples.dim() == 1, "categorical distribution requires choices and vector samples");
        let count = samples.size()[0];
        let weights = match weights { Some(weights) => weights, None => uniform_weight(count + i64::from(self.prior), samples.device())? };
        let mut probs = if self.prior {
            let prior = tch_result(weights.f_double_value(&[count]))?;
            tch_result(Tensor::f_full([self.n_choices as i64], self.uniform as f64 * prior, FLOAT_CPU))?
        } else { tch_result(Tensor::f_zeros([self.n_choices as i64], FLOAT_CPU))? };
        let weights = if self.prior { tch_result(weights.f_narrow(0, 0, count))? } else { weights };
        let slicer = tch_result(Tensor::f_arange(self.n_choices as i64, INT64_CPU))?;
        let masks = tch_result(tch_result(slicer.f_view([-1, 1]))?.f_eq_tensor(&samples))?;
        for c in 0..self.n_choices {
            let mask = tch_result(masks.f_select(0, c as i64))?;
            let weight = tch_result(tch_result(weights.f_masked_select(&mask))?.f_sum(None))?;
            let weight = tch_result(weight.f_to_kind(Float))?;
            let not_c = tch_result(tch_result(slicer.f_ne(c as i64))?.f_to_kind(Float))?;
            let is_c = tch_result(tch_result(slicer.f_eq(c as i64))?.f_to_kind(Float))?;
            let other = tch_result(not_c.f_mul(&tch_result(weight.f_mul_scalar(self.bottom as f64))?))?;
            let selected = tch_result(is_c.f_mul(&tch_result(weight.f_mul_scalar(self.top as f64))?))?;
            probs = tch_result(tch_result(probs.f_add(&other))?.f_add(&selected))?;
        }
        let mut choices = tch_result(Tensor::f_full([self.n_choices as i64, self.n_choices as i64], self.bottom as f64, FLOAT_CPU))?;
        let _ = tch_result(choices.f_fill_diagonal_(self.top as f64, false))?;
        let likelihood = tch_result(choices.f_index_select(0, &samples))?;
        let blls = if self.prior {
            let prior = tch_result(Tensor::f_full([self.n_choices as i64], self.uniform as f64, FLOAT_CPU))?;
            tch_result(Tensor::f_vstack(&[likelihood, prior]))?
        } else { likelihood };
        let log_blls = tch_result(tch_result(blls.f_clamp_min(1e-12))?.f_log())?;
        CategoricalGaussian::new(probs, log_blls, weights)
    }
}

impl ParzenEstimator for CategoricalParzenEstimator {
    fn clc_distr(&self, samples: Tensor, weights: Option<Tensor>) -> anyhow::Result<Box<dyn Distribution>> {
        Ok(Box::new(self._clc_distr(samples, weights)?))
    }

    fn clc_distr_from_vec(&self, samples: &mut dyn Iterator<Item = Val>, weights: Option<Tensor>) -> anyhow::Result<Box<dyn Distribution>> {
        let samples = samples.map(i64::try_from).collect::<anyhow::Result<Vec<_>>>()?;
        Ok(Box::new(self._clc_distr(tch_result(Tensor::f_from_slice(&samples))?, weights)?))
    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct CategoricalGaussian {
    probs: Tensor,
    log_blls: Tensor,
    probs_vec: Vec<f32>,
    indexes: Vec<usize>,
    weights: Tensor,
}

impl CategoricalGaussian {
    pub fn new(probs: Tensor, log_blls: Tensor, weights: Tensor) -> anyhow::Result<Self> {
        let probs_vec = tch_result(Vec::<f32>::try_from(&probs))?;
        let probs_len = probs_vec.len();
        Ok(Self {
            probs_vec,
            probs, //TODO use it?
            indexes: (0..probs_len).collect(),
            log_blls,
            weights,
        })
    }
}

impl Distribution for CategoricalGaussian {
    fn sample(&self, n_samples: usize) -> anyhow::Result<Vec<Val>> {
        // Construct once and return invalid probability errors instead of panicking.
        let dist = WeightedIndex::new(self.probs_vec.iter().copied())?;
        Ok((0..n_samples).map(|_| Val::Int(opt_rnd(|rng| dist.sample(rng)) as i64)).collect())
    }

    fn basis_loglikelihood(&self, x: &mut dyn Iterator<Item = &Val>) -> anyhow::Result<Tensor> {
        let values = x.map(i64::try_from).collect::<anyhow::Result<Vec<_>>>()?;
        Ok(tch_result(self.log_blls.f_index_select(1, &tch_result(Tensor::f_from_slice(&values))?))?)
    }

    fn weights(&self) -> &Tensor {
        &self.weights
    }
}

#[cfg(test)]
mod categorical_parzen_estimator_tests {
    use crate::parzen::cpe::CategoricalParzenEstimator;
    use crate::parzen::optimizer::Distribution;
    use tch::Kind::Float;
    use tch::Tensor;
    use thoosa_shared::val::Val;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_basis_loglikelihood() {
        let cpe = CategoricalParzenEstimator::new(None, 3, true);

        let distr = cpe._clc_distr(
            Tensor::from_slice(&[0, 1, 2]),
            Some(Tensor::from_slice(&[0.3, 0.3, 0.3, 0.1])),
        ).unwrap();
        assert!(
            Tensor::from_slice(&[0.3333, 0.3333, 0.3333])
                .totype(Float)
                .allclose(&distr.probs, 0.01, 0.01, false)
        );
        // let samples = distr.sample(3);

        assert!(
            Tensor::from_slice(&[0.66666667, 0.16666667, 0.16666667])
                .totype(Float)
                .allclose(
                    &cpe._clc_distr(
                        Tensor::from_slice(&[0]),
                        Some(Tensor::from_slice(&[0.5, 0.5]))
                    ).unwrap()
                    .probs,
                    0.01,
                    0.01,
                    false
                )
        );

        let distr = cpe._clc_distr(
            Tensor::from_slice(&[0, 0, 0, 0]),
            Some(Tensor::from_slice(&[0.2, 0.2, 0.2, 0.2, 0.2])),
        ).unwrap();
        assert!(
            Tensor::from_slice(&[0.86666667, 0.06666667, 0.06666667])
                .totype(Float)
                .allclose(&distr.probs, 0.01, 0.01, false)
        );
        let samples = distr.sample(1000).unwrap();
        let p_0 = samples.iter().filter(|i| *i == &Val::Int(0)).count() as f32 / 1000.;
        let p_1 = samples.iter().filter(|i| *i == &Val::Int(1)).count() as f32 / 1000.;
        let p_2 = samples.iter().filter(|i| *i == &Val::Int(2)).count() as f32 / 1000.;

        assert!((0.86666667 - p_0).abs() < 0.1);
        assert!((0.06666667 - p_1).abs() < 0.1);
        assert!((0.06666667 - p_2).abs() < 0.1);

        // let bll = distr.basis_loglikelihood(&Tensor::from_slice(samples.into_iter().map(|v|v.into()).collect::<Vec<i64>>().as_slice()));

        let bll = distr.basis_loglikelihood(
            &mut vec![&Val::Int(0), &Val::Int(1), &Val::Int(2), &Val::Int(0)].into_iter(),
        ).unwrap();
        assert_eq!(
            Tensor::from_slice(&[
                0.,
                -27.63102112,
                -27.63102112,
                0.,
                0.,
                -27.63102112,
                -27.63102112,
                0.,
                0.,
                -27.63102112,
                -27.63102112,
                0.,
                0.,
                -27.63102112,
                -27.63102112,
                0.,
                -1.09861229,
                -1.09861229,
                -1.09861229,
                -1.09861229
            ])
            .totype(Float)
            .view((5, 4))
            .round_decimals(3),
            bll.round_decimals(3)
        );
        // debug!("bll={bll}");
    }
}
