use std::f64::consts::PI;
use tch::kind::FLOAT_CPU;
use tch::Tensor;

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