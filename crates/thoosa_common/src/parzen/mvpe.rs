use tch::Tensor;

use crate::parzen::npe;
use crate::parzen::optimizer::{Distribution, Primitives};

#[derive(Debug)]
pub struct MultiVariateParzenEstimator {
    parzen_estimators:Vec<Box<dyn Distribution>>
}

impl MultiVariateParzenEstimator {
    pub fn new(parzen_estimators:Vec<Box<dyn Distribution>>) -> Self {
        Self { parzen_estimators }
    }

    #[allow(dead_code)]
    pub fn len(&self)->usize {
        self.parzen_estimators.len()
    }

    pub fn sample(&self, dim_independent:bool, n_samples: usize) -> Vec<Vec<Primitives>>{
        if dim_independent {
            self.parzen_estimators.iter().map(|pe|pe.sample(n_samples)).collect()//TODO collect/collect

        } else {
            todo!()
        }
    }

    pub fn log_pdf(&self, configs:&Vec<Vec<Primitives>>) -> Tensor {
        let tensor = Tensor::stack(
            (&self.parzen_estimators).iter()
                .zip(configs.into_iter())
                .map(|(pe, config)| pe.basis_loglikelihood(&mut config.into_iter()))
                .collect::<Vec<Tensor>>().as_slice(),
            0
        );
        // debug!("log_pdf: tensor.size={:?}", tensor.size());
        npe::compute_config_loglikelihoods(
            &tensor, //.view((self.parzen_estimators.len() as i64, -1, configs.len() as i64)),
            // Some(self.parzen_estimators[0].weights.as_ref())
            Some(self.parzen_estimators[0].weights())
        )
    }

}
#[cfg(test)]
mod multi_variate_parzen_estimator_tests {
    use tch::Tensor;

    use crate::parzen::mvpe::MultiVariateParzenEstimator;
    use crate::parzen::npe::{BandwidthFactor, NumericalParzenEstimator};
    use crate::parzen::optimizer::ParzenEstimator;
    use crate::parzen::optimizer::Primitives::Float;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_log_pdf(){
        let samples = [-1.0, 0.0, 1.0];

        let mvpe = MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(
                None,
                // Tensor::from_slice(&samples),
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None
            ).clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(
                None,
                // Tensor::from_slice(&samples),
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None
            ).clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(
                None,
                // Tensor::from_slice(&samples),
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None
            ).clc_distr(Tensor::from_slice(&samples), None)

        ]);

        let v = mvpe.log_pdf(
            &vec![
                vec![Float(0.0)],//x0
                vec![Float(0.1)],//x1
                vec![Float(0.2)]//x2
            ]
        );

        // debug!("v={v}");
        assert_eq!(-8017,  (v.double_value(&[])*10000.0) as isize);


    }
}
