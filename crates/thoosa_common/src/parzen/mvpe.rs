use crate::tch_error::tch_result;
use crate::parzen::npe;
use crate::parzen::optimizer::Distribution;
use log::debug;
use tch::Tensor;
use thoosa_shared::val::Val;

#[derive(Debug)]
pub struct MultiVariateParzenEstimator {
    parzen_estimators: Vec<Box<dyn Distribution>>,
}

impl MultiVariateParzenEstimator {
    pub fn new(parzen_estimators: Vec<Box<dyn Distribution>>) -> anyhow::Result<Self> {
        debug!("MultiVariateParzenEstimator::new: dimensions={}", parzen_estimators.len());
        anyhow::ensure!(!parzen_estimators.is_empty(), "multivariate density requires at least one dimension");
        Ok(Self { parzen_estimators })
    }

    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.parzen_estimators.len()
    }

    /// Draw independent dimensions; correlated sampling is not implemented.
    pub fn sample_independent(&self, n_samples: usize) -> anyhow::Result<Vec<Vec<Val>>> {
        self.parzen_estimators.iter().map(|pe| pe.sample(n_samples)).collect()
    }

    pub fn log_pdf(&self, configs: &Vec<Vec<Val>>) -> anyhow::Result<Tensor> {
        anyhow::ensure!(configs.len() == self.parzen_estimators.len(), "candidate dimensions do not match estimators");
        let likelihoods = self.parzen_estimators.iter().zip(configs).map(|(pe, config)| pe.basis_loglikelihood(&mut config.iter())).collect::<anyhow::Result<Vec<_>>>()?;
        let tensor = tch_result(Tensor::f_stack(&likelihoods, 0))?;
        npe::compute_config_loglikelihoods(&tensor, Some(self.parzen_estimators[0].weights()))
    }
}
#[cfg(test)]
mod multi_variate_parzen_estimator_tests {
    use crate::parzen::mvpe::MultiVariateParzenEstimator;
    use crate::parzen::npe::{BandwidthFactor, NumericalParzenEstimator};
    use crate::parzen::optimizer::ParzenEstimator;
    use tch::Tensor;
    use thoosa_shared::val::Val;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_log_pdf() {
        let samples = [-1.0, 0.0, 1.0];

        let mvpe = MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(
                None,
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None,
            )
            .clc_distr(Tensor::from_slice(&samples), None).unwrap(),
            NumericalParzenEstimator::new(
                None,
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None,
            )
            .clc_distr(Tensor::from_slice(&samples), None).unwrap(),
            NumericalParzenEstimator::new(
                None,
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None,
            )
            .clc_distr(Tensor::from_slice(&samples), None).unwrap(),
        ]).unwrap();

        let v = mvpe.log_pdf(&vec![
            vec![Val::Real(0.0)], //x0
            vec![Val::Real(0.1)], //x1
            vec![Val::Real(0.2)], //x2
        ]).unwrap();

        assert_eq!(-8017, (v.double_value(&[]) * 10000.0) as isize);
    }
}
