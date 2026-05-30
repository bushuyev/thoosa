use std::f64::consts::PI;
#[cfg(test)]
use std::fmt;
#[cfg(test)]
use log::debug;
// Removed dependency on rand distributions here; use tch::Tensor::randn instead
use tch::Tensor;
use tch::kind::FLOAT_CPU;

#[cfg(test)]
use thoosa_shared::ThoError;
#[cfg(test)]
use crate::opt::cholesky_with_jitter;

#[allow(dead_code)]
pub struct Gaussian {
    μ: Tensor,
    σ: Tensor,
}

#[allow(dead_code)]
impl Gaussian {
    pub fn new(μ: Tensor, σ: Tensor) -> Self {
        Self { μ, σ }
    }

    pub fn cdf(&self, value: &Tensor) -> Tensor {
        (Tensor::ones(&value.size(), FLOAT_CPU)
            + (value - &self.μ)
                .matmul(&self.σ.reciprocal())
                .divide_scalar(std::f64::consts::SQRT_2)
                .erf())
        .multiply_scalar(0.5)
    }

    pub fn pdf(&self, value: &Tensor) -> Tensor {
        //TODO replace with Tensor::normal
        (value - &self.μ)
            .matmul(&self.σ.reciprocal())
            .pow_tensor_scalar(2)
            .multiply_scalar(-0.5)
            .exp()
            .multiply(&self.σ.multiply_scalar((2. * PI).sqrt()).reciprocal())
    }

    #[allow(dead_code)]
    pub fn norm(n: i64) -> Tensor {
        Tensor::randn([n], FLOAT_CPU).cumsum(0, None).view((1, -1))
    }
}

#[cfg(test)]
#[derive(Copy, Clone)]
pub struct GaussianProcess<F>
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    kernel: F,
    sigma_y: f64,
}

#[cfg(test)]
impl<F> fmt::Debug for GaussianProcess<F>
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaussianProcess")
            .field("kernel", &"<function>")
            .field("sigma_y", &self.sigma_y)
            .finish()
    }
}

#[cfg(test)]
pub struct GaussianProcessFit<'g, F>
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    gp: &'g GaussianProcess<F>,
    x_train: Tensor,
    alpha: Tensor,
    chol_t: Tensor,
}

#[cfg(test)]
impl<'g, F> fmt::Debug for GaussianProcessFit<'g, F>
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GaussianProcessFit")
            .field("gp", &self.gp)
            .field("x_train", &self.x_train)
            .field("alpha", &self.alpha)
            .field("chol_t", &self.chol_t)
            .finish()
    }
}

#[cfg(test)]
impl<'g, F> GaussianProcessFit<'g, F>
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    pub fn new(gp: &'g GaussianProcess<F>, x_train: Tensor, alpha: Tensor, chol_t: Tensor) -> Self {
        Self {
            gp,
            x_train,
            alpha,
            chol_t,
        }
    }

    #[allow(dead_code)]
    pub fn predict(&self, x_s: &Tensor) -> Result<Tensor, ThoError> {
        let (mu_s, cov_s) = self.mean_and_cov(x_s);

        let chol = cholesky_with_jitter(&cov_s.nan_to_num(0., None, None)).unwrap();

        let sig_t_with_cov = chol.matmul(&Gaussian::norm(x_s.size()[0]).transpose(0, 1));

        let sig_chol = (mu_s + sig_t_with_cov).transpose(0, 1);

        Ok(sig_chol)
    }

    pub fn mean_and_cov(&self, x_s: &Tensor) -> (Tensor, Tensor) {
        let k_trans = (self.gp.kernel)(x_s, &self.x_train);
        let y_mean = k_trans.matmul(&self.alpha);

        let k_trans_t = k_trans.transpose(0, 1);

        let v = k_trans_t.triangular_solve(&self.chol_t, false, false, false);

        let y_var = (self.gp.kernel)(x_s, x_s) - v.0.transpose(0, 1).matmul(&v.0);

        //apply kernel to normal dst:

        let cov_size = y_var.size();
        let mu_s = y_mean.view((cov_size[1], 1));
        let std = y_var.sqrt(); // +  Tensor::zeros(&[cov_size[1], cov_size[1]], FLOAT_CPU).fill_diagonal_(1e-8, false);;
        (mu_s, std)
    }
}
#[cfg(test)]
impl<F> GaussianProcess<F>
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{

    pub fn new(kernel: F, sigma_y: f64) -> Self {
        Self { kernel, sigma_y }
    }

    //Gaussian_Processes_for_ML.pdf

    pub fn fit(
        &self,
        x_train: &Tensor,
        y_train: &Tensor,
    ) -> Result<GaussianProcessFit<'_, F>, ThoError> {
        debug!("fit: x_train={}", x_train);
        debug!("fit: y_train={}", y_train);
        let k = (self.kernel)(&x_train, &x_train);
        let k_size = k.size();

        debug!("fit: k={}", k);
        let k =
            k + Tensor::zeros(k_size, FLOAT_CPU).fill_diagonal_(self.sigma_y.powi(2) as f64, false);
        debug!("fit: with sigma k={}", k);

        let chol_t = cholesky_with_jitter(&k)?;
        debug!("fit: chol_t k={}", chol_t);

        //from sklearn's GaussianProcessRegressor
        let alpha = y_train
            .view((y_train.size()[0], chol_t.size()[0], 1))
            .cholesky_solve(&chol_t, false);

        Ok(GaussianProcessFit::new(
            self,
            x_train.shallow_clone(),
            alpha,
            chol_t,
        ))
    }
}

#[cfg(test)]
mod tests {
    use tch::Kind::Float;
    use tch::{IndexOp, Tensor};

    use crate::gaussian::{Gaussian, GaussianProcess};
    use crate::opt::kernel_rbf;
    use crate::utils::sin3_pow2;

    #[test]
    fn test_cdf() {
        let g = Gaussian::new(Tensor::from_slice(&[0.]), Tensor::from_slice(&[3.]));
        let cdf = g.cdf(&Tensor::from_slice(&[0.]));
        assert_eq!(0.5, cdf.double_value(&[0]));

        let cdf = g.cdf(&Tensor::from_slice(&[-3]));
        assert!(
            Tensor::from_slice(&[0.1587])
                .totype(Float)
                .allclose(&cdf, 1e-02, 1e-02, false)
        );

        let cdf = g.cdf(&Tensor::from_slice(&[3]));
        assert!(
            Tensor::from_slice(&[0.8413])
                .totype(Float)
                .allclose(&cdf, 1e-02, 1e-02, false)
        );

        let g = Gaussian::new(Tensor::from_slice(&[0., 0.]), Tensor::from_slice(&[3., 3.]));
        let cdf = g.cdf(&Tensor::from_slice(&[0., 0.]));
        assert_eq!(0.5, cdf.double_value(&[0]));
        assert_eq!(0.5, cdf.double_value(&[1]));
        // cdf.print();
    }

    #[test]
    fn test_pdf() {
        let g = Gaussian::new(Tensor::from_slice(&[0.]), Tensor::from_slice(&[3.]));
        let pdf = g.pdf(&Tensor::from_slice(&[0.])).double_value(&[0]);
        let expected = 1.0 / (3.0 * (2.0 * std::f64::consts::PI).sqrt());
        assert!((pdf - expected).abs() < 1e-6, "pdf={pdf}, expected={expected}");
    }

    #[test]
    fn test_gp_fit() {
        let gp = GaussianProcess::new(
            move |t1, t2| {
                kernel_rbf(
                    &t1,
                    &t2,
                    &Tensor::from_slice(&[1.0]).totype(Float),
                    &Tensor::from_slice(&[1.0]).totype(Float),
                )
            },
            1e-8,
        );

        let (y_train_full, x_t, _) = sin3_pow2(10, -2., 2., None);
        let train_indexes = vec![0, 9];

        let x_train = x_t.i(train_indexes.as_slice());
        let y_train = y_train_full.i((.., train_indexes.as_slice()));

        let gpf = gp.fit(&x_train, &y_train).unwrap();

        let test_indexes = vec![2, 7];
        let x_test = x_t.i(test_indexes.as_slice());

        let (m, s) = gpf.mean_and_cov(&x_test);
        // debug!("alpha {}", gpf.alpha);
        // debug!("m {}", m);
        // debug!("s {}", s);
        //python/gp_test.py
        assert_eq!(
            Tensor::from_slice(&[2.3187, 5.6786,])
                .totype(Float)
                .view((1, 2, 1))
                .round_decimals(4),
            gpf.alpha.round_decimals(4)
        );

        assert_eq!(
            Tensor::from_slice(&[1.60687517, 3.84369169])
                .totype(Float)
                .view((2, 1))
                .round_decimals(4),
            m.round_decimals(4)
        );

        assert_eq!(
            Tensor::from_slice(&[0.73902109, 0.73902109])
                .totype(Float)
                .round_decimals(4),
            s.diagonal(0, 0, 1).round_decimals(4)
        );
    }

}
