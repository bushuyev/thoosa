use tch::kind::FLOAT_CPU;
use tch::Tensor;
use crate::gaussian::Gaussian;
use crate::utils::{from, vec_f32_from};



//http://arxiv.org/abs/1012.2599, (4)
pub fn expected_improvement(µ:&Tensor, σ:&Tensor, y_best:&Tensor, ξ:f64 ){//y_best - f (x+)

    let impr = µ.subtract(
        &y_best.f_add_scalar(ξ).expect("y_best.f_add_scalar")
    );
    let z = impr.divide(σ);


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
}

#[cfg(test)]
mod tests {
    use tch::Kind::Float;
    use tch::Tensor;
    use crate::acquisitions::expected_improvement;

    #[test]
    fn test_expected_improvement(){
        expected_improvement(
            &Tensor::from_slice(&[0.,0.]).totype(Float),
            &Tensor::from_slice(&[3.,3.]).totype(Float),
            &Tensor::from_slice(&[0.]).totype(Float),
            0.01
        );
    }
}