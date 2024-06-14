use log::debug;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use crate::gaussian::{Gaussian, GaussianProcessFit};
use crate::utils::{from, vec_f32_from};



//http://arxiv.org/abs/1012.2599, (4)
pub fn expected_improvement<'a, F>(x:&Tensor, y_best:&Tensor, gpf:&GaussianProcessFit<'a, F>, ξ:f64 ) -> Tensor where F: Fn(&Tensor, &Tensor) -> Tensor {//y_best - f (x+)
    // let xxx = x.subtract_scalar(13.).pow_tensor_scalar(2).view(1);
    let x = x.view((-1,1));
    let (µ, σ) = gpf.mean_and_cov(&x);


    let impr = µ.subtract(&y_best.f_add_scalar(ξ).expect("y_best.f_add_scalar")); //max
    // let impr = y_best.subtract(&µ).subtract_scalar(ξ); //min
    debug!("expected_improvement: ξ={} x={:?} y_best={:?} impr={:?}", ξ, vec_f32_from(&x),  vec_f32_from(&y_best), vec_f32_from(&impr));

    let z = impr.divide(&σ);


    let g = Gaussian::new(
        Tensor::zeros(µ.size(), FLOAT_CPU),
        Tensor::ones(µ.size(), FLOAT_CPU)
    );//TODO


    let cdf = g.cdf(&z);
    let pdf = g.pdf(&z);
    debug!("expected_improvement: x={:?} µ={:?} σ={:?} z={:?} cdf={:?} pdf={:?}", vec_f32_from(&x), vec_f32_from(&µ), vec_f32_from(&σ), vec_f32_from(&z), vec_f32_from(&cdf), vec_f32_from(&pdf));

    let ei = impr * cdf + σ * pdf;

    debug!("expected_improvement: ei={} device={:?} requires_grad={:?}", ei, ei.device(), ei.requires_grad());
    ei.view(1) //+ xxx
    //

}


#[cfg(test)]
mod tests {
    use std::ops::Deref;
    use argmin::core::State;
    use chrono::Utc;
    use log::debug;
    use plotters::prelude::{BLACK, ShapeStyle};
    use plotters::prelude::full_palette::ORANGE;
    use plotters::style::RED;
    use rand::thread_rng;
    use rand::Rng;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::{IndexOp, Tensor};
    use hbac_shared::HbacError;
    use crate::acquisitions::{expected_improvement};
    use crate::argmin::{run_lbfgs, TensorProblem};
    use crate::gaussian::GaussianProcess;
    use crate::opt::{kernel_rbf, rbf_nll};
    use crate::utils::{from, plot_series_with_points, sin3_pow2, vec_f32_from};

    #[test]
    fn test_expected_improvement(){
        let l = Tensor::from_slice(&[1.0]).totype(Float);
        let sigma_f = Tensor::from_slice(&[1.0]).totype(Float);

        let mut gp = GaussianProcess::new (
            move |t1, t2| kernel_rbf(&t1, &t2, &l, &sigma_f),
            1e-8,
        );
       /* let x_train = Tensor::from_slice(&[0.0, 1.0]).view((-1, 1)).totype(Float);
        let y_train = Tensor::from_slice(&[10., 20.]).view((1, -1)).totype(Float);*/
        let n = 100i64;

        let (y_train_full, x_t) = sin3_pow2(n, -2., 2., true);
        // let train_indexes = vec![0, 99]; //(0..n).step_by(20).collect::<Vec<i64>>();//vec![0, n-1]; //
        let train_indexes = vec![0, ((2. +0.31)/4.*100.) as i64, 99]; //(0..n).step_by(20).collect::<Vec<i64>>();//vec![0, n-1]; //
        // let train_indexes = vec![0, ((2. -0.25)/4.*100.) as i64, ((2. +0.31)/4.*100.) as i64, 99]; //(0..n).step_by(20).collect::<Vec<i64>>();//vec![0, n-1]; //

        let x_train = x_t.i(train_indexes.as_slice());
        let y_train = y_train_full.i((.., train_indexes.as_slice()));

        let gpf = gp.fit(&x_train, &y_train).expect("gp.fit");

        // let x =  Tensor::from_slice(&[0.0]).view((-1, 1)).totype(Float).set_requires_grad(true);

        // let ei = expected_improvement(
        //     &x,
        //     &y_train.max(),
        //     &gpf,
        //     0.01
        // );
        //
        // ei.backward();
        // debug!("ei={} requires_grad={} x={} grad={}", ei, ei.requires_grad(), x, x.grad());
        let y_train_max = y_train.max();

        let next = run_lbfgs(

            TensorProblem::new( |theta| {

                expected_improvement(
                    &theta,
                    &y_train_max,
                    &gpf,
                    0.01
                ).negative()
                // rbf_nll(
                //     &Tensor::from_slice(&[-4., -3., -2., -1., 1.]).view((-1, 1)),
                //     &Tensor::from_slice(&[0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.]).view((-1, 1)),
                //     theta, 1e-8
                // ).expect("nll")
            }
            ),
            Tensor::from_slice(&[-2.]).totype(Float).to_device(Cpu).view(1),
            // Tensor::from_slice(&[1., 1.]).totype(Float).to_device(Cpu),
            f64::NEG_INFINITY,//0.,//
            100
        ).unwrap();

        debug!(">>>>>>>>>>>>>>>next={}", next.state().get_best_param().unwrap().deref());


        /*let x_vec = vec_f32_from(&x_t);
        let ei_for_all_x = x_vec.iter().map(|x_i|{
            expected_improvement(
                &Tensor::from(*x_i).totype(Float),
                &y_train_max,
                &gpf,
                0.01
            ).double_value(&[]) as f32
        }).collect::<Vec<f32>>();

        let pred_y = gpf.predict(&x_t).unwrap();

        plot_series_with_points(
            vec![
                (format!("ref"), ShapeStyle {
                    color: BLACK.into(),
                    filled: true,
                    stroke_width: 1,
                }, x_vec.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(n))).collect::<Vec<(f32, f32)>>()),

                (format!("ei"), ShapeStyle {
                    color: RED.into(),
                    filled: true,
                    stroke_width: 1,
                }, x_vec.clone().into_iter().zip(ei_for_all_x.clone().into_iter().map(|v|10.*v)).collect::<Vec<(f32, f32)>>()),

                (format!("pred_y"), ShapeStyle {
                    color: ORANGE.into(),
                    filled: true,
                    stroke_width: 1,
                }, x_vec.into_iter().zip(from::<Vec<f32>>(&(pred_y.i((0, ..))))).collect::<Vec<(f32, f32)>>()),
            ],
            "EEG", (2000, 500), &format!("/tmp/plots/1_{:?}.png", Utc::now().to_rfc3339()).as_str(),
            Some(&from::<Vec<f32>>(&x_train.view(-1)).into_iter().zip(vec_f32_from(&y_train)).collect::<Vec<(f32, f32)>>())
        );*/

    }

}