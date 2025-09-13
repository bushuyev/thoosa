use log::{debug, trace};
use rand::seq::SliceRandom;
use rand::distr::{Distribution as RandDistribution};
use rand::distr::weighted::WeightedIndex;
use tch::{IndexOp, Tensor};
use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::Kind::Float;
use thoosa_shared::val::Val;
use thoosa_shared::opt_rng::opt_rnd;
use crate::parzen::optimizer::{Distribution, ParzenEstimator};
use crate::parzen::util::uniform_weight;
use crate::utils::from;

pub struct CategoricalParzenEstimator {
    name:Option<String>,
    top:f32,
    bottom:f32,
    uniform:f32,
    n_choices:usize,
    prior: bool
}

impl CategoricalParzenEstimator {
    pub fn new(name: Option<String>, n_choices:usize, prior: bool) -> Self {
        let top = 1.0;
        Self {
            name,
            top, //TODO The hyperparam of the cateogircal kernel. It defines the prob of the top category. ?
            bottom:  (1. - top) / (n_choices - 1) as f32,
            uniform: 1./n_choices as f32,
            n_choices,

            prior
        }
    }

    fn _clc_distr(&self, samples: Tensor, weights: Option<Tensor>) -> CategoricalGaussian {
        let weights = weights.unwrap_or(uniform_weight(samples.size()[0] + if self.prior {1} else {0}));
        debug!("samples={samples}");
        trace!("weights={weights} self.uniform={}", self.uniform);

        let mut probs = if self.prior {
            Tensor::full(self.n_choices as i64, self.uniform as f64 * weights.i(weights.size()[0]-1).double_value(&[]), FLOAT_CPU)//TODO last weight?
        } else {
            Tensor::zeros(self.n_choices as i64, FLOAT_CPU)
        }.view(-1); //from::<Vec<f64>>(&);

        let weights = if self.prior {weights.i(0..weights.size()[0]-1)} else {weights};
        trace!("weights={weights}");

        debug!("probs={probs} dim={:?}", probs.dim());

        let masks = Tensor::arange(self.n_choices as i64, INT64_CPU).view((-1,1)).eq_tensor(&samples);

        // debug!("masks={masks}");

        let slicer = Tensor::arange(self.n_choices as i64, INT64_CPU); //0..self.n_choices; //

        for c in 0..self.n_choices{
            let mask = masks.i(c as i64);
            let weight = weights.masked_select(&mask).sum(None).totype(Float);

            let c_t = Tensor::from_slice(&[c as i64]);
            let mask_not_c = slicer.ne_tensor(&c_t);
            let mask_is_c = slicer.eq_tensor(&c_t);
            // debug!("mask_not_c={mask_not_c} mask_is_c={mask_is_c}");
            // debug!("mask_not_c={} mask_is_c={}", mask_not_c.nonzero().view(-1).dim(), mask_is_c.nonzero().view(-1).dim());
            // Use the masks to index and update the probs tensor
            probs = probs + mask_not_c.to_kind(Float) * &weight.multiply_scalar(self.bottom as f64);
            probs = probs + mask_is_c.to_kind(Float) * &weight.multiply_scalar(self.top as f64);
            // debug!("probs={probs}");

            // debug!("c={c} mask={mask} weight={weight}");
        }
        // debug!("probs={probs}");

        let likelihood_choices = Tensor::full([self.n_choices as i64, self.n_choices as i64], self.bottom as f64, FLOAT_CPU)
            .fill_diagonal_(self.top as f64, false);

        debug!("likelihood_choices={likelihood_choices}");
        // debug!("likelihood_choices samples={}", likelihood_choices.index(&[Some(samples)]));

        let blls= if self.prior {
            Tensor::vstack(&[
                likelihood_choices.index(&[Some(samples)]),
                Tensor::full(self.n_choices as i64, self.uniform as f64, FLOAT_CPU)
            ])
        } else {
            likelihood_choices.index(&[Some(samples)])
        };
        let blls = blls.clamp_min(1e-12);


        CategoricalGaussian::new(probs, blls.log(), weights)

    }
}

impl ParzenEstimator for CategoricalParzenEstimator {

    fn clc_distr(&self, samples: Tensor, weights: Option<Tensor>) -> Box<dyn Distribution> {
        Box::new(self._clc_distr(samples, weights))
    }

    fn clc_distr_from_vec(&self, samples:&mut dyn Iterator<Item=Val>, weights: Option<Tensor>) -> Box<dyn Distribution>{
        Box::new(self._clc_distr(Tensor::from_slice(samples.map(|p| p.try_into().unwrap()).collect::<Vec<i64>>().as_slice()).totype(tch::Kind::Int64), weights))

    }
}

#[derive(Debug)]
#[allow(dead_code)]
struct CategoricalGaussian {
    probs:Tensor,
    log_blls:Tensor,
    probs_vec:Vec<f32>,
    indexes:Vec<usize>,
    weights:Tensor

}

impl CategoricalGaussian {

    pub fn new(probs:Tensor, log_blls:Tensor, weights:Tensor) -> Self {
        let probs_vec = from::<Vec<f32>>(&probs);
        let probs_len = probs_vec.len();
        Self {
            probs_vec,
            probs, //TODO use it?
            indexes: (0..probs_len).collect(),
            log_blls,
            weights
        }
    }
}

impl Distribution for CategoricalGaussian {

    fn sample(&self, n_samples: usize) -> Vec<Val> {

        (0..n_samples).map(|_|{
            let dist = WeightedIndex::new(self.indexes.iter().map(|i| self.probs_vec[*i]))
                .expect("WeightedIndex::new");
            let pos = dist.sample(opt_rnd());
            self.indexes[pos]
        })
        .map(|v|(v as i32).into())
        .collect()
    }

    fn basis_loglikelihood(&self, x:&mut dyn Iterator<Item=&Val>) -> Tensor {
        // debug!("basis_loglikelihood: x={x}");
        self.log_blls.index_select(1, &Tensor::from_slice(x.map(|v|v.try_into().unwrap()).collect::<Vec<i64>>().as_slice()))

    }

    fn weights(&self) -> &Tensor {
        &self.weights
    }
}

#[cfg(test)]
mod categorical_parzen_estimator_tests {
    use tch::Kind::Float;
    use tch::Tensor;
    use thoosa_shared::val::Val;
    use crate::parzen::cpe::CategoricalParzenEstimator;
    use crate::parzen::optimizer::{Distribution};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_0(){
        let cpe = CategoricalParzenEstimator::new(
            None,
            3,
            true
        );

        let distr = cpe._clc_distr(Tensor::from_slice(&[0, 1, 2]), Some(Tensor::from_slice(&[0.3, 0.3, 0.3, 0.1])));
        assert!(
            Tensor::from_slice(&[0.3333, 0.3333, 0.3333]).totype(Float).allclose(
                &distr.probs, 0.01, 0.01, false
            )
        );
        // let samples = distr.sample(3);

        assert!(
            Tensor::from_slice(&[0.66666667, 0.16666667, 0.16666667]).totype(Float).allclose(
                &cpe._clc_distr(Tensor::from_slice(&[0]), Some(Tensor::from_slice(&[0.5, 0.5]))).probs, 0.01, 0.01, false
            )
        );

        let distr = cpe._clc_distr(Tensor::from_slice(&[0, 0, 0, 0]), Some(Tensor::from_slice(&[0.2, 0.2, 0.2, 0.2, 0.2])));
        assert!(
            Tensor::from_slice(&[0.86666667, 0.06666667, 0.06666667]).totype(Float).allclose(
                &distr.probs, 0.01, 0.01, false
            )
        );
        let samples = distr.sample(1000);
        let p_0 = samples.iter().filter(|i|*i == &Val::Int(0)).count() as f32/1000.;
        let p_1 = samples.iter().filter(|i|*i == &Val::Int(1)).count() as f32/1000.;
        let p_2 = samples.iter().filter(|i|*i == &Val::Int(2)).count() as f32/1000.;


        assert!((0.86666667 - p_0).abs() < 0.1);
        assert!((0.06666667 - p_1).abs() < 0.1);
        assert!((0.06666667 - p_2).abs() < 0.1);


        // let bll = distr.basis_loglikelihood(&Tensor::from_slice(samples.into_iter().map(|v|v.into()).collect::<Vec<i64>>().as_slice()));

        let bll = distr.basis_loglikelihood(&mut vec![&Val::Int(0), &Val::Int(1), &Val::Int(2), &Val::Int(0)].into_iter());
        assert_eq!(
            Tensor::from_slice(
                &[
                    0., -27.63102112, -27.63102112, 0.,
                    0., -27.63102112, -27.63102112, 0.,
                    0., -27.63102112, -27.63102112, 0.,
                    0., -27.63102112, -27.63102112, 0.,
                    -1.09861229, -1.09861229, -1.09861229, -1.09861229
                ]).totype(Float).view((5, 4)).round_decimals(3),
            bll.round_decimals(3)
        );
        // debug!("bll={bll}");

    }
}
