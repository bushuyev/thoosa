use std::f64::consts::PI;
use rand::thread_rng;
use statrs::distribution::Normal;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use hbac_shared::HbacError;
use crate::opt::cholesky_with_jitter;
use crate::utils::vec_f32_from;
use rand::Rng;

pub struct Gaussian {
    μ:Tensor,
    σ:Tensor
}

impl Gaussian {

    pub fn new(μ: Tensor, σ: Tensor) -> Self {
        Self { μ, σ }
    }

    pub fn cdf(&self, value:&Tensor) -> Tensor {
        (
            Tensor::ones(&value.size(), FLOAT_CPU) + (value - &self.μ).matmul(&self.σ.reciprocal()).divide_scalar(std::f64::consts::SQRT_2).erf()
        ).multiply_scalar(0.5)
    }

    pub fn pdf(&self, value:&Tensor) -> Tensor {
        (value - &self.μ).matmul(&self.σ.reciprocal()).pow_tensor_scalar(2).multiply_scalar(-0.5).exp().multiply(&self.σ.multiply_scalar((2. * PI).sqrt()).reciprocal())
    }

    pub fn norm(n:i64) -> Tensor{
         let mut rng = thread_rng();

        Tensor::from_slice((0..n).map(|x| rng.sample(Normal::new(0., 1.).unwrap()) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None).view((1,-1))
    }

}

#[derive(Copy, Clone)]
pub struct GaussianProcess<F> where F: Fn(&Tensor, &Tensor) -> Tensor{
    kernel:F,
    sigma_y: f64,
}

pub struct GaussianProcessFit<'g, F> where F: Fn(&Tensor, &Tensor) -> Tensor {
    gp:&'g GaussianProcess<F>,
    x_train:Tensor,
    alpha:Tensor,
    chol_t:Tensor
}

impl <'g, F> GaussianProcessFit<'g, F> where F: Fn(&Tensor, &Tensor) -> Tensor{
    pub fn new(gp: &'g GaussianProcess<F>, x_train: Tensor, alpha: Tensor, chol_t:Tensor) -> Self {
        Self { gp, x_train, alpha,chol_t }
    }

    pub fn predict(&self, x_s: &Tensor) -> Result<Tensor, HbacError> {
        let (mu_s, cov_s) = self.mean_and_cov(x_s);

        let chol_or_cov = cholesky_with_jitter(&cov_s).unwrap();

        let sig_t_with_cov = chol_or_cov.matmul(&Gaussian::norm(x_s.size()[0]).transpose(0, 1));

        let sig_chol = (mu_s + sig_t_with_cov).transpose(0, 1);

        Ok(sig_chol)
    }

    pub fn mean_and_cov(&self, x_s: &Tensor) -> (Tensor, Tensor) {
        let k_trans = (self.gp.kernel)(x_s, &self.x_train);
        let y_mean = k_trans.matmul(&self.alpha);

        let k_trans_t = k_trans.transpose(0, 1);

        let v = k_trans_t.triangular_solve(&self.chol_t, false, false, false);

        let y_cov = (self.gp.kernel)(x_s, x_s) - v.0.transpose(0, 1).matmul(&v.0);


        //apply kernel to normal dst:

        let mu_s = y_mean.view((y_cov.size()[1], 1));
        let cov_s = y_cov;
        (mu_s, cov_s)
    }
}

impl <F> GaussianProcess<F> where F: Fn(&Tensor, &Tensor) -> Tensor {
    pub fn new(kernel: F, sigma_y: f64) -> Self {
        Self {
            kernel,
            sigma_y,
        }
    }

    //Gaussian_Processes_for_ML.pdf
    pub fn fit(&self, x_train:&Tensor, y_train:&Tensor) -> Result<GaussianProcessFit<F>, HbacError>{

        let k = (self.kernel)(&x_train, &x_train);
        let k_size = k.size();
        // println!("posterior_ch: k.size()={:?}", k.size());
        let k = k + Tensor::zeros(k_size, FLOAT_CPU).fill_diagonal_(self.sigma_y.powi(2) as f64, false);

        // println!("posterior_ch: k={:?}", vec_f32_from(&k));

        // let k_size = k.size();

        let chol_t = cholesky_with_jitter(&k)?;

        // println!("y_train.size()={:?}, chol_t={:?}", y_train.size(), chol_t.size());

        //from sklearn's GaussianProcessRegressor
        let alpha = y_train.view((y_train.size()[0], chol_t.size()[0], 1)).cholesky_solve(&chol_t, false);

        Ok(GaussianProcessFit::new(self, x_train.shallow_clone(), alpha, chol_t))
    }
}

#[cfg(test)]
mod tests {
    use tch::Kind::Float;
    use tch::Tensor;
    use crate::gaussian::Gaussian;

    #[test]
    fn test_cdf(){
        let g = Gaussian::new(Tensor::from_slice(&[0.]), Tensor::from_slice(&[3.]));
        let cdf = g.cdf(&Tensor::from_slice(&[0.]));
        assert_eq!(0.5, cdf.double_value(&[0]));

        let cdf = g.cdf(&Tensor::from_slice(&[-3]));
        assert!(Tensor::from_slice(&[0.1587]).totype(Float).allclose(&cdf, 1e-02, 1e-02, false));

        let cdf = g.cdf(&Tensor::from_slice(&[3]));
        assert!(Tensor::from_slice(&[0.8413]).totype(Float).allclose(&cdf, 1e-02, 1e-02, false));

        let g = Gaussian::new(Tensor::from_slice(&[0.,0.]), Tensor::from_slice(&[3., 3.]));
        let cdf = g.cdf(&Tensor::from_slice(&[0.,0.]));
        assert_eq!(0.5, cdf.double_value(&[0]));
        assert_eq!(0.5, cdf.double_value(&[1]));
        // cdf.print();
    }
}