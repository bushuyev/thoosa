use tch::kind::{FLOAT_CPU, INT64_CPU};
use tch::{IndexOp, Tensor};
use tch::Kind::Float;
use crate::parzen::optimizer::{Distribution, ParzenEstimator, Primitives};
use crate::parzen::util::uniform_weight;
use crate::utils::from;

pub struct CategoricalParzenEstimator {
    top:f32,
    bottom:f32,
    uniform:f32,
    n_choices:usize,
    prior: bool
}

impl CategoricalParzenEstimator {
    pub fn new(n_choices:usize, prior: bool) -> Self {
        let top = 1.0;
        Self {
            top, //TODO The hyperparam of the cateogircal kernel. It defines the prob of the top category. ?
            bottom:  (1. - top) / (n_choices - 1) as f32,
            uniform: 1./n_choices as f32,
            n_choices,

            prior
        }
    }

    fn _clc_dist(&self, samples: Tensor, weights: Option<Tensor>)-> CategoricalGaussian {
        let weights = weights.unwrap_or(uniform_weight(samples.size()[0] + if self.prior {1} else {0}));
        println!("weights={weights} self.uniform={}", self.uniform);

        let mut probs = if self.prior {
            Tensor::full(self.n_choices as i64, self.uniform as f64 * weights.i(weights.size()[0]-1).double_value(&[]), FLOAT_CPU)//TODO last weight?
        } else {
            Tensor::zeros(self.n_choices as i64, FLOAT_CPU)
        }.view(-1); //from::<Vec<f64>>(&);

        let weights = if self.prior {weights.i(0..weights.size()[0]-1)} else {weights};
        println!("weights={weights}");

        println!("probs={probs} dim={:?}", probs.dim());

        let masks = Tensor::arange(self.n_choices as i64, INT64_CPU).view((-1,1)).eq_tensor(&samples);

        println!("masks={masks}");

        let slicer = Tensor::arange(self.n_choices as i64, INT64_CPU); //0..self.n_choices; //

        for c in 0..self.n_choices{
            let mask = masks.i(c as i64);
            let weight = weights.masked_select(&mask).sum(None).totype(Float);

            let c_t = Tensor::from_slice(&[c as i64]);
            let mask_not_c = slicer.ne_tensor(&c_t);
            let mask_is_c = slicer.eq_tensor(&c_t);
            println!("mask_not_c={mask_not_c} mask_is_c={mask_is_c}");
            println!("mask_not_c={} mask_is_c={}", mask_not_c.nonzero().view(-1).dim(), mask_is_c.nonzero().view(-1).dim());
            // Use the masks to index and update the probs tensor
            probs = probs + mask_not_c.to_kind(Float) * &weight.multiply_scalar(self.bottom as f64);
            probs = probs + mask_is_c.to_kind(Float) * &weight.multiply_scalar(self.top as f64);
            println!("probs={probs}");
            // for s in 0..self.n_choices {
            //     if s == c {
            //
            //     } else {
            //
            //     }
            // }

            println!("c={c} mask={mask} weight={weight}");
        }
        println!("probs={probs}");

        CategoricalGaussian::new(probs)

    }
}

impl ParzenEstimator for CategoricalParzenEstimator {

    fn clc_dist(&self, samples: Tensor, weights: Option<Tensor>) -> Box<dyn Distribution> {
        Box::new(self._clc_dist(samples, weights))
    }
}

#[derive(Debug)]
struct CategoricalGaussian {
    probs:Tensor
}

impl CategoricalGaussian {

    pub fn new(probs:Tensor) -> Self {
        Self {
            probs
        }
    }
}

impl Distribution for CategoricalGaussian {

    fn sample(&self, n_samples: usize) -> Vec<Primitives> {
        vec![Primitives::Integer(0)]
    }

    fn basis_loglikelihood(&self, x: &Tensor) -> Tensor {
        Tensor::rand(x.size(), FLOAT_CPU)
    }

    fn weights(&self) -> &Tensor {
        todo!()
    }
}

mod categorical_parzen_estimator_tests {
    use tch::Kind::Float;
    use tch::Tensor;
    use crate::parzen::cpe::{CategoricalGaussian, CategoricalParzenEstimator};
    use crate::parzen::optimizer::ParzenEstimator;

    #[test]
    fn test_0(){
        let cpe = CategoricalParzenEstimator::new(
            3,
            true
        );

        assert!(
            Tensor::from_slice(&[0.3333, 0.3333, 0.3333]).totype(Float).allclose(
                &cpe._clc_dist(Tensor::from_slice(&[0,1,2]), Some(Tensor::from_slice(&[0.3, 0.3, 0.3, 0.1]))).probs, 0.01, 0.01, false
            )
        );

        assert!(
            Tensor::from_slice(&[0.66666667, 0.16666667, 0.16666667]).totype(Float).allclose(
                &cpe._clc_dist(Tensor::from_slice(&[0]), Some(Tensor::from_slice(&[0.5, 0.5]))).probs, 0.01, 0.01, false
            )
        );

        assert!(
            Tensor::from_slice(&[0.86666667, 0.06666667, 0.06666667]).totype(Float).allclose(
                &cpe._clc_dist(Tensor::from_slice(&[0,0,0,0]), Some(Tensor::from_slice(&[0.2, 0.2, 0.2, 0.2, 0.2]))).probs, 0.01, 0.01, false
            )
        );

    }
}