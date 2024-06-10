use tch::kind::FLOAT_CPU;
use tch::Tensor;
use crate::gaussian::{Gaussian, GaussianProcessFit};
use crate::utils::{from, vec_f32_from};



//http://arxiv.org/abs/1012.2599, (4)
pub fn expected_improvement<'a, F>(x:&Tensor, y_best:&Tensor, gpf:&GaussianProcessFit<'a, F>, ξ:f64 ) -> Tensor where F: Fn(&Tensor, &Tensor) -> Tensor {//y_best - f (x+)

    let (µ, σ) = gpf.mean_and_cov(&x);

    let impr = µ.subtract(
        &y_best.f_add_scalar(ξ).expect("y_best.f_add_scalar")
    );
    let z = impr.divide(&σ);


    let g = Gaussian::new(
        Tensor::zeros(µ.size(), FLOAT_CPU),
        Tensor::ones(µ.size(), FLOAT_CPU)
    );//TODO


    println!("{:?}", vec_f32_from(&z));
    let cdf = g.cdf(&z);
    println!("{:?}", vec_f32_from(&cdf));
    let pdf = g.pdf(&z);
    println!("{:?}", vec_f32_from(&pdf));

    let ei = impr * cdf + σ * pdf;

    println!("ei={ei}");
    ei
}

#[cfg(test)]
mod tests {
    use tch::Kind::Float;
    use tch::Tensor;
    use hbac_shared::HbacError;
    use crate::acquisitions::expected_improvement;
    use crate::argmin::{run_lbfgs, TensorProblem};
    use crate::opt::rbf_nll;

    #[test]
    fn test_expected_improvement(){
        // expected_improvement(
        //     &Tensor::from_slice(&[0.,0.]).totype(Float),
        //     &Tensor::from_slice(&[3.,3.]).totype(Float),
        //     &Tensor::from_slice(&[0.]).totype(Float),
        //     0.01
        // );

        // assert!(Tensor::from_slice(&[0.0507]).totype(Float).allclose(&expected_improvement(
        //     &Tensor::from_slice(&[-0.8003969]).totype(Float),
        //     &Tensor::from_slice(&[0.13932637]).totype(Float),
        //     &Tensor::from_slice(&[-0.80039698, -0.80039693]).max().totype(Float),
        //     0.01
        // ), 1e-02, 1e-02, false));
    }

    #[test]
    fn test_min_expected_improvement()-> Result<(), HbacError>{

        let res = run_lbfgs(
            TensorProblem::new(|theta| rbf_nll(
                &Tensor::from_slice(&[-4., -3., -2., -1., 1.]).view((-1, 1)),
                &Tensor::from_slice(&[0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.]).view((-1, 1)),
                theta, 1e-8
            ).expect("nll"))
        )?;

        Ok(())
    }
}