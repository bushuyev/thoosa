use std::any::Any;
use nalgebra::{DMatrix, DVector};
use tch::kind::FLOAT_CPU;
use tch::{IndexOp, Tensor};
use log::{warn, debug};
use tch::Kind::Float;
use hbac_shared::HbacError;
use crate::utils::vec_f32_from;

fn kernel(t_0:&Tensor, t_1:&Tensor, l:f64, sigma:f64) -> Tensor {
    Tensor::cdist(t_0, t_1, 2.0, 0).pow_tensor_scalar(2).multiply_scalar(-0.5 * l.powi(2)).exp().multiply_scalar(sigma.powi(2))
}

fn kernel_t(t_0:&Tensor, t_1:&Tensor, l:&Tensor, sigma:&Tensor) -> Tensor {
    let sqdist = Tensor::cdist(t_0, t_1, 2.0, 0).pow_tensor_scalar(2);

    let l_pow = (l + 1e-5).pow_tensor_scalar(2);

    let e_sqdist = sqdist.multiply(&(-0.5 / (l_pow)));

    e_sqdist.exp().multiply(&sigma.pow_tensor_scalar(2))
}


fn posterior(x_s:&Tensor, x_train:&Tensor, y_train:&Tensor, l:&Tensor, sigma_f:&Tensor, sigma_y:f64) -> Result<(Tensor, Tensor), HbacError>{
    // let k = kernel(&x_train, &x_train, l, sigma_f);
    // let l = Tensor::from_slice(&[l]).totype(Float);
    // let sigma_f = Tensor::from_slice(&[sigma_f]).totype(Float);

    let k = kernel_t(&x_train, &x_train, l, sigma_f);
    let k_size = k.size();
    println!("posterior_ch: k.size()={:?}", k.size());
    let k = k + Tensor::zeros(k_size, FLOAT_CPU).fill_diagonal_(sigma_y.powi(2) as f64, false);

    println!("posterior_ch: k={:?}", vec_f32_from(&k));

    let k_size = k.size();

    let chol_t = cholesky_with_jitter(&k)?;

    println!("y_train.size()={:?}, chol_t={:?}", y_train.size(), chol_t.size());

    let alpha = y_train.view((y_train.size()[0], chol_t.size()[0], 1)).cholesky_solve(&chol_t, false);

    // let k_trans = kernel(x_s, x_train, l, sigma_f);
    let k_trans = kernel_t(x_s, x_train, l, sigma_f);
    let y_mean = k_trans.matmul(&alpha);

    let k_trans_t = k_trans.transpose(0, 1);

    // println!("k_trans_T:");
    // k_trans_T.print();

    let v = k_trans_t.triangular_solve(&chol_t, false, false, false);
    // println!("v.0:");
    // v.0.print();

    // println!("v.v:");
    // v.1.print();

    // let y_cov = kernel(x_s, x_s, l, sigma_f) - v.0.transpose(0, 1).matmul(&v.0);
    let y_cov = kernel_t(x_s, x_s, l, sigma_f) - v.0.transpose(0, 1).matmul(&v.0);

    // y_cov.print();

    println!("y_mean={:?}", y_mean.size());

    Ok((y_mean.view((-1, y_cov.size()[1])).transpose(0,1), y_cov))
}

pub fn rbf_nll(x_train:&Tensor, y_train:&Tensor, theta:&Tensor, noise: f64) -> Result<Tensor, HbacError> {
    debug!("rbf_nll: theta.device={:?} theta.requires_grad={:?}", theta.device(), theta.requires_grad());
    let noise = noise.powi(2);
    // let l = theta.double_value(&[0]);
    // let sigma_f = theta.double_value(&[1]);
    let l = theta.i(0);
    let sigma_f = theta.i(1); //Tensor::from_slice(&[1.0]);//theta.i(1);
    debug!("rbf_nll: l={} sigma_f={}", l.double_value(&[]), sigma_f.double_value(&[]));

    let noise_t = Tensor::zeros(&[x_train.size()[0], x_train.size()[0]], FLOAT_CPU).fill_diagonal_(noise, false);
    // debug!("noise_t");
    // noise_t.print();

    let k = kernel_t(x_train, x_train, &l, &sigma_f) + noise_t;
    // debug!("K");
    // k.print();

    let l = cholesky_with_jitter(&k)?;
    // debug!("L");
    // l.print();

    let (s_1, _) = y_train.triangular_solve(&l, false, false, false);
    let l_t = l.transpose(0, 1);
    // debug!("L_T");
    // l_t.print();
    let (s_2, _) = s_1.triangular_solve(&l_t, true, false, false);

    // debug!("S_1");
    // s_1.print();

    // debug!("S_2");
    // s_2.print();

    let diag = l.diag(0);
    // debug!("DIAG");
    // diag.print();

    let log = diag.log();
    // debug!("LOG");
    // log.print();

    let np_sum = log.sum(Float); //.double_value(&[]);
    // debug!("NP_SUM={:?}", np_sum);

    let dot = y_train.transpose(0, 1).matmul(&s_2);//.double_value(&[]);
    // debug!("DOT={:?}", dot);

    let v:Tensor = np_sum + 0.5 * dot + 0.5 * x_train.size()[0] as f64 * (std::f64::consts::PI * 2.).ln();

    debug!("rbf_nll: v={:?} device={:?} grad={:?}", v, v.device(), v.requires_grad());

    // v = np.sum(np.log(np.diagonal(L))) + 0.5 * Y_train.dot(S2) + 0.5 * len(X_train) * np.log(2 * np.pi)

    Ok(v)
}

fn cholesky_rec(t:&Tensor, tries:usize, jitter:f64, e:i32, e_step:i32) -> Result<Tensor, HbacError>{

    match t.f_linalg_cholesky(false) {
        Ok(l) => Ok(l),
        Err(err) => {
            warn!("err={:?}", err);
            // debug!("err.type_id()={:?}", err.type_id()); TODO check type_id()
            if tries > 0 {
                cholesky_rec(
                    &(t + Tensor::zeros(&t.size(), FLOAT_CPU).fill_diagonal_(10_f64.powi(e) * jitter, false)),
                    tries - 1,
                    jitter,
                    e+ e_step,
                    e_step,
                )
            } else {
                Err(HbacError::new(format!("can't do cholesky: {:?}", err)))
            }
        }
    }
}

pub fn cholesky_with_jitter(t:&Tensor) -> Result<Tensor, HbacError> {
    cholesky_rec(t,  10, std::f64::EPSILON, 0, 2)
}



#[cfg(test)]
mod opt_tests {
    use std::sync::{Arc, Mutex};
    use chrono::Utc;
    use log::{debug, LevelFilter};
    use plotters::prelude::{BLACK, GREEN, IntoLinspace, RED};
    use plotters::style::Color;
    use plotters::prelude::ShapeStyle;
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::Kind::Float;
    use tch::Tensor;
    use hbac_shared::{comp_sin_k_v, join_with_offset, pulse_v, random_saw};
    use crate::utils::{cdt_for_dir, plot_series, plot_series_with_points, vec_f32_from};
    use crate::utils::lin_t;
    use hbac_shared::*;
    use statrs::distribution::{Uniform, Normal, MultivariateNormal};
    use rand::{Rng, thread_rng};
    use nalgebra::DMatrix;
    use plotters::style::BLUE;
    use plotters::prelude::YELLOW;
    use plotters::style::full_palette::*;
    use rand::prelude::ThreadRng;
    use crate::utils::from;
    use tch::IndexOp;
    use tch::kind::{DOUBLE_CPU, FLOAT_CPU};
    use std::iter::once;
    use argmin::core::{CostFunction, Error, Executor};
    use argmin::solver::brent::BrentOpt;
    use tch::Device::Cpu;
    use crate::opt::{kernel, rbf_nll};
    use argmin::core::State;
    use crate::acquisitions::expected_improvement;
    use crate::argmin::{run_lbfgs, TensorProblem};

    #[ctor::ctor]
    fn init() {
        CombinedLogger::init(
            vec![
                TermLogger::new(LevelFilter::Debug, Config::default(), TerminalMode::Mixed, ColorChoice::Auto)
            ]
        ).unwrap_or_else(|e|{
            println!("CombinedLogger::init failed: {:?}", e);
        });


        std::env::set_var("POLARS_FMT_MAX_COLS", "100");
        std::env::set_var("POLARS_FMT_MAX_ROWS", "100");
    }

    fn sig_1_norm( norm: Normal, mut rng: ThreadRng, n: i64) -> Tensor {

        Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None).view((1,-1))
    }


    fn sig_3_norm( norm: Normal, mut rng: ThreadRng, n: i64) -> Tensor {
        let sig_t_0 = Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None);
        let sig_t_1 = Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None);
        let sig_t_2 = Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None);

        let sig_t = Tensor::stack(&[sig_t_0, sig_t_1, sig_t_2], 0);
        sig_t
    }

    #[test]
    fn test_rbf()  -> Result<(), HbacError> {
        let x_train = Tensor::from_slice(&[1., 2., 3., 4.]).view((-1, 1));
        // let k = super::kernel(&x_train, &x_train, 1.0, 1.0);
        let k = super::kernel_t(&x_train, &x_train, &Tensor::from_slice(&[1.0]), &Tensor::from_slice(&[1.0]));

        let samp_k = Tensor::from_slice(&[
            1.,         0.60653066, 0.13533528, 0.011109,
            0.60653066, 1.,         0.60653066, 0.13533528,
            0.13533528, 0.60653066, 1.,         0.60653066,
            0.011109,   0.13533528, 0.60653066, 1.
        ]).view((4, 4));

        assert_eq!(
            samp_k.round_decimals(4),
            k.round_decimals(4)
        );

        // let chol_t = super::cholesky_with_jitter(&k)?;
        //
        // chol_t.print();

        Ok(())
    }

    #[test]
    fn test_rbf_10()  -> Result<(), HbacError> {
        let x_train = Tensor::from_slice(&[-4., -3., -2., -1., 0.]).view((-1, 1));

        let k = super::kernel_t(&x_train, &x_train, &Tensor::from_slice(&[-10.0]), &Tensor::from_slice(&[1.0]));

        let samp_k = Tensor::from_slice(&[
            1.        , 0.99501247, 0.98019863, 0.9559974 , 0.9231162,
            0.99501247, 1.        , 0.99501247, 0.98019863, 0.9559974,
            0.98019863, 0.99501247, 1.        , 0.99501247, 0.98019863,
            0.9559974 , 0.98019863, 0.99501247, 1.        , 0.99501247,
            0.9231162 , 0.9559974 , 0.98019863, 0.99501247, 1.,
        ]).view((5, 5));

        assert_eq!(
            samp_k.round_decimals(4),
            k.round_decimals(4)
        );

        Ok(())
    }



    #[test]
    fn test_rbf_chol_rand() -> Result<(), HbacError>{

        let N = 1000;
        let x_train = Tensor::rand([N, 1], (FLOAT_CPU));
        let k = super::kernel(&x_train, &x_train, 0.0, 0.0);

        let chol_t = super::cholesky_with_jitter(&k)?;

        assert_eq!(vec![1000, 1000], chol_t.size());

        Ok(())
    }

    #[test]
    fn test_posterior_ch_mult() -> Result<(), HbacError> {
        let x_train = Tensor::from_slice(&[
            1., 40.,
            2., 30.,
            3., 20.,
            4., 10.
        ]).view((-1, 2));

        x_train.print();

        let y_train = Tensor::from_slice(&[2., 20., 2., 20.]).view((1, -1));
        let x_pred = x_train.i((vec![1i64, 3i64], ..));

        x_pred.print();

        let Ok((mu_s, cov_s)) = super::posterior(&x_pred, &x_train, &y_train, &Tensor::from_slice(&[1.0]).totype(Float), &Tensor::from_slice(&[1.0]).totype(Float), 1e-8) else {
            panic!()
        };

        println!("mu_s={mu_s}");
        println!("cov_s={cov_s}");

        assert_eq!(
            Tensor::from_slice(&[
                20.,
                20.
            ]).view((2, 1)).totype(Float),
            mu_s.totype(Float)
        );

        assert_eq!(
            Tensor::from_slice(&[
                0.0,         0.0,
                0.0,         0.0,
            ]).view((2, 2)).totype(Float),
            cov_s.totype(Float)
        );

        Ok(())
    }

    #[test]
    fn test_posterior() -> Result<(), HbacError> {
        let x_train = Tensor::from_slice(&[1., 2., 3., 4.]).view((-1, 1));
        let y_train = Tensor::from_slice(&[2., 20., 2., 20.]).view((1, -1));

        let x_pred = Tensor::from_slice(&[1., 3.]).view((-1, 1));

        let Ok((mu_s, cov_s)) = super::posterior(
            &x_pred,
            &x_train,
            &y_train,
            &Tensor::from_slice(&[1.0]).totype(Float),
            &Tensor::from_slice(&[1.0]).totype(Float),
            1e-8
        ) else {
            panic!("posterior")
        };

        mu_s.print();
        cov_s.print();

        assert_eq!(
            Tensor::from_slice(&[
                2.0,         2.0,
            ]).view((2, 1)).totype(Float),
            mu_s.totype(Float)
        );

        assert_eq!(
            Tensor::from_slice(&[
                0.0,         0.0,
                0.0,         0.0,
            ]).view((2, 2)).totype(Float),
            cov_s.totype(Float)
        );


        Ok(())
    }


    #[test]
    fn test_rbf_nll() -> Result<(), HbacError> {
        let x_train = Tensor::from_slice(&[
            -4., -3., -2., -1., 1.
        ]).view((-1, 1));

        let y_train = Tensor::from_slice(&[
            0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.
        ]).view((-1, 1));

        for theta in [[0.1, 0.1, 48.0], [0.1, 1., 5.67], [1., 0.1, 41.00], [1., 1., 4.64]] {
            let v = super::rbf_nll(&x_train, &y_train, &Tensor::from_slice(&theta), 0.1_f64).expect("nll");

            assert_eq!(format!("{:.1}", theta[2]), format!("{:.1}", v.double_value(&[])));
        }


        // let v = super::nll(&x_train, &y_train, &Tensor::from_slice(&[-10., 1.])).expect("nll");
        //
        // println!("v={:.2}", v);
        //
        // let x = (-100..100).map(|x|x as f32/10.).collect::<Vec<f32>>();
        // plot_series(
        //     vec![
        //
        //         (format!("train"), ShapeStyle {
        //             color: GREEN.into(),
        //             filled: true,
        //             stroke_width: 1,
        //         }, x.clone().into_iter().zip(x.into_iter().map(|x|super::nll(&x_train, &y_train, &Tensor::from_slice(&[x])).expect("nll").double_value(&[]) as f32)).collect::<Vec<(f32, f32)>>()),
        //
        //
        //
        //     ], "EEG", (2000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        // )?;

        Ok(())
    }

    #[test]
    fn test_nll_brent() -> Result<(), HbacError> {

        let x_train = Tensor::from_slice(&[
            -4., -3., -2., -1., 1.
        ]).view((-1, 1));

        let y_train = Tensor::from_slice(&[
            0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.
        ]).view((-1, 1));

        debug!("x_train={:?}", x_train.size());
        y_train.print();

        struct MyProblem {
            x_train:Tensor,
            y_train:Tensor
        }

        impl CostFunction for MyProblem {
            type Param = f64;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
               Ok(super::rbf_nll(&self.x_train, &self.y_train, &Tensor::from_slice(&[*p, 0.74]), 0.1_f64).expect("nll").double_value(&[]))
            }
        }

        let cost = MyProblem {
            x_train,
            y_train
        };


        let solver = BrentOpt::new(-100., 100.);

        let res = Executor::new(cost, solver)
            .configure(|state|
            state
            .param(-100.)
            .max_iters(10)
            .target_cost(0.0)
            )
            .run()?;

        debug!("res={:?}", res.state().get_best_param().unwrap());

        Ok(())
    }


    #[test]
    fn test_nll_bfgs() -> Result<(), HbacError> {
        // X_train=[[-4.]
        //     [-3.]
        //     [-2.]
        //     [-1.]
        //     [ 0.]]

        let x_train = Tensor::from_slice(&[
            -4., -3., -2., -1., 1.
        ]).view((-1, 1));

        let y_train = Tensor::from_slice(&[
            0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.
        ]).view((-1, 1));

        debug!("x_train={:?}", x_train.size());
        y_train.print();

        // let theta = Tensor::ones([2], FLOAT_CPU);

        // Y_train=[[ 0.7568025 ]
        //     [-0.14112001]
        //     [-0.90929743]
        //     [-0.84147098]
        //     [ 0.        ]]

        let mut bfgs = crate::bfgs::BFGS::new(|theta: &mut Tensor| {
            theta.zero_grad();
            super::rbf_nll(&x_train, &y_train, &theta, 0.1_f64).expect("nll").to_device(Cpu).set_requires_grad(true)
            // let k = kernel(&x_train, &x_train, 1.0, 0.0) + Tensor::zeros(&x_train.size(), FLOAT_CPU).fill_diagonal_(0.1, false);

            // let v =
            //     theta.i(0).f_add_scalar(-11.).unwrap().pow_tensor_scalar(2) +
            //     theta.i(1).f_add_scalar(-22.).unwrap().pow_tensor_scalar(2);
            // debug!("func: v=");
            // v.print();
            // v
            // k.sum(Float).view((-1)) //+ theta.sum(Float).view((-1))
        });

        // let theta = Tensor::from_slice(&[20., 20.]).totype(Float).set_requires_grad(true);
        let theta = Tensor::from_slice(&[0.0, 0.74]).totype(Float).set_requires_grad(true);

        bfgs.opt(theta.shallow_clone(), Some(crate::bfgs::BFGSParams::default().with_lr(1.0).with_max_iter(20).with_with_wolf_line_search(true)));

        // debug!("opt: {:?} {:?}", theta.double_value(&[0]), theta.double_value(&[1]));
        debug!("opt: {:?}", theta.double_value(&[0]));

        Ok(())
    }

    #[test]
    fn test_opt_1d_with_ei() -> Result<(), HbacError> {
        let n = 100i64;
        let mut rng = rand::thread_rng();

        let x = (0..n).map(|i| -1. + (i as f32 / n as f32)  * 3. ).into_iter().collect::<Vec<f32>>();


        let y_train_full = Tensor::from_slice(x.iter().map(|x| (-(3. * *x).sin() - x.powi(2) + 0.7 * *x + 0.2 * rng.gen::<f32>()) as f64).collect::<Vec<f64>>().as_slice()).view((1, -1)).totype(Float); //.transpose(0,1);

        let x_t = Tensor::from_slice(x.as_slice()).totype(Float).view((-1, 1));

        let train_indexes = (0..n).step_by(20).collect::<Vec<i64>>();//vec![0, n-1]; //
        let train_indexes = train_indexes.as_slice();

        let x_train = x_t.i(train_indexes);
        let y_train = y_train_full.i((.., train_indexes));


        let Ok((mu_s, cov_s)) = super::posterior(&x_t, &x_train, &y_train, &Tensor::from_slice(&[1.0]).totype(Float), &Tensor::from_slice(&[1.0]).totype(Float), 1e-8) else {
            panic!()
        };


        let next = run_lbfgs(

            TensorProblem::new(|theta|

                expected_improvement(
                    &Tensor::from_slice(&[-0.8003969]).totype(Float),
                    &Tensor::from_slice(&[0.13932637]).totype(Float),
                    &Tensor::from_slice(&[-0.80039698, -0.80039693]).totype(Float),
                    0.01
                )
            )
        )?;


        let surrogate = apply_post(&mu_s, &cov_s, &sig_1_norm(Normal::new(0., 1.).unwrap(), thread_rng(), n).transpose(0, 1));

        plot_series_with_points(
            vec![
                (format!("train"), ShapeStyle {
                    color: GREEN_800.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(n))).collect::<Vec<(f32, f32)>>()),

                (format!("surrogate"), ShapeStyle {
                    color: GREEN.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&(surrogate.i((0, ..))))).collect::<Vec<(f32, f32)>>()),
            ],
            "EEG", (2000, 500), &format!("/tmp/plots/1_{:?}.png", Utc::now().to_rfc3339()).as_str(),
            Some(&from::<Vec<f32>>(&x_train.view(-1)).into_iter().zip(from::<Vec<f32>>(&y_train.view(-1)).into_iter()).collect::<Vec<(f32, f32)>>())
        );
        Ok(())
    }


    #[test]
    fn test_opt_1d() -> Result<(), HbacError>{

        let N = 100i64;

        // let sig =  comp_sin_k_v(N, vec![6., 3., 2.], 0.5);
        // let sig =  comp_sin(N).collect::<Vec<f32>>();
        let x = (0..N).map(|i| - std::f32::consts::PI + (i as f32 / N as f32)  * 2. * std::f32::consts::PI ).into_iter().collect::<Vec<f32>>();
        let X = Tensor::from_slice(x.as_slice()).totype(Float).view((-1, 1));


        let y_train_full = Tensor::stack(&[
            X.sin().multiply_scalar(10).view((-1, 1)).totype(Float),
            // X.f_add_scalar(std::f64::consts::PI/4.).unwrap().sin().multiply_scalar(10).view((-1, 1)).totype(Float),
            // X.f_add_scalar(std::f64::consts::PI/2.).unwrap().sin().multiply_scalar(10).view((-1, 1)).totype(Float)
        ], 0).view((1, -1)); //.transpose(0,1);

        println!("y_train_full={:?}", y_train_full.size());

        let train_indexes = (0..N).step_by(50).collect::<Vec<i64>>();
        let train_indexes = train_indexes.as_slice();
        let x_train = X.i(train_indexes);
        let y_train = y_train_full.i((.., train_indexes));

        println!("x_train={:?} y_train={:?}", x_train.size(), y_train.size());

        // let (mu_s, cov_s) = super::posterior(&X, &x_train, &y_train, 1., 1., 1e-8);
        // println!("1. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());

        println!("X={:?}", from::<Vec<f32>>(&X.view(-1)));
        println!("x_train={:?}", from::<Vec<f32>>(&x_train.view(-1)));
        y_train.print();
        println!("y_train={:?}", from::<Vec<f32>>(&y_train.view(-1)));


        let Ok((mu_s, cov_s)) = super::posterior(&X, &x_train, &y_train, &Tensor::from_slice(&[1.0]).totype(Float), &Tensor::from_slice(&[1.0]).totype(Float), 1e-8) else {
            panic!()
        };
        println!("2. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());
        let x_moved = x_train.view((-1, 1));
        let y_moved = y_train.view((-1, 1));
        let l_sigma = run_lbfgs(TensorProblem::new(move|theta| {
            rbf_nll(&x_moved, &y_moved, theta, 1e-8).expect("nll")
        }))?;

        let l_sigma = l_sigma.state().get_best_param().expect("best_param");

        println!("l_sigma={:?} - {} / {} ", l_sigma, l_sigma.i(0).double_value(&[]), l_sigma.i(1).double_value(&[]));

        // let Ok((mu_s_opt, cov_s_opt)) = super::posterior_ch(&X, &x_train, &y_train, 1.6409202082368064, 8.504394038804222, 1e-8) else {
        let Ok((mu_s_opt, cov_s_opt)) = super::posterior(&X, &x_train, &y_train, &l_sigma.i(0), &l_sigma.i(1), 1e-8) else {
            panic!()
        };
        println!("2. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());


        let norm = Normal::new(0., 1.).unwrap();
        let rng = thread_rng();
        let sig_t = sig_3_norm(norm, rng, N).transpose(0,1);
        println!("sig_t.size={:?}", sig_t.size());


        let sig_chol = apply_post(&mu_s, &cov_s, &sig_t);
        let sig_chol_opt = apply_post(&mu_s_opt, &cov_s_opt, &sig_t);

        plot_series_with_points(
            vec![

                (format!("train"), ShapeStyle {
                    color: GREEN_800.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(N))).collect::<Vec<(f32, f32)>>()),
                // (format!("train"), ShapeStyle {
                //     color: BLUE_800.into(),
                //     filled: true,
                //     stroke_width: 1,
                // }, x.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((1, ..)).view(N))).collect::<Vec<(f32, f32)>>()),
                // (format!("train"), ShapeStyle {
                //     color: RED_800.into(),
                //     filled: true,
                //     stroke_width: 1,
                // }, x.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((2, ..)).view(N))).collect::<Vec<(f32, f32)>>()),

                (format!("Sig0"), ShapeStyle {
                    color: GREEN.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&(sig_chol.i((0, ..))))).collect::<Vec<(f32, f32)>>()),
                // (format!("Sig0Chol"), ShapeStyle {
                //     color: BLUE.into(),
                //     filled: true,
                //     stroke_width: 2,
                // }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_chol.i(((1, ..))))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig0Chol"), ShapeStyle {
                    color: RED.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_chol_opt.i(((0, ..))))).collect::<Vec<(f32, f32)>>()),


            ], "EEG", (2000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str(),
            Some(&from::<Vec<f32>>(&x_train.view(-1)).into_iter().zip(from::<Vec<f32>>(&y_train.view(-1)).into_iter()).collect::<Vec<(f32, f32)>>())
        ).unwrap();

        Ok(())
    }

    fn apply_post(mu_s: &Tensor, cov_s: &Tensor, sig_t: &Tensor) -> Tensor {
        let chol_or_cov = super::cholesky_with_jitter(&cov_s).unwrap();
        println!("2. chol_or_cov={:?}", chol_or_cov.size());


        let sig_t_with_cov = chol_or_cov.matmul(&sig_t);
        println!("mu_s.size()={:?}, chol.size()={:?}", mu_s.size(), sig_t_with_cov.size());

        let sig_chol = (mu_s + sig_t_with_cov).transpose(0, 1);
        println!("sig_chol.size={:?}", sig_chol.size());
        sig_chol
    }

    #[test]
    fn test_opt_2d(){

        let N = 1000i64;
        let min_max = 100.; // std::f32::consts::PI;
        // let sig =  comp_sin_k_v(N, vec![6., 3., 2.], 0.5);
        // let sig =  comp_sin(N).collect::<Vec<f32>>();
        let x_01 = (0..N).map(|i| {
           (
                -min_max + (i as f32 / N as f32)  * 2. * min_max,
                min_max - (i as f32 / N as f32)  * 2. * min_max
           )
        }).collect::<Vec<(f32, f32)>>();

        // let y = x_01.iter().map(|x_01| (x_01.0 *  x_01.1).cos()).collect::<Vec<f32>>();
        let y = x_01.iter().map(|x_01| x_01.0.powi(2) +  x_01.1.powi(2)).collect::<Vec<f32>>();

        let x_0 = x_01.iter().map(|x_01| x_01.0).collect::<Vec<f32>>();
        let x_1 = x_01.iter().map(|x_01| x_01.1).collect::<Vec<f32>>();

        let X = Tensor::from_slice(x_01.into_iter().flat_map(|x_01|once(x_01.0).chain(once(x_01.1))).collect::<Vec<f32>>().as_slice()).view((-1, 2));
        print!("X={:?}", X.size());
        // X.print();

        let y_train_full = Tensor::from_slice(y.as_slice()).view((1, -1));
        print!("y_train_full={:?}", y_train_full.size());
        // y_train_full.print();

        let train_indexes = (0..N).step_by(10).collect::<Vec<i64>>();
        let train_indexes = train_indexes.as_slice();
        let x_train = X.i(train_indexes);
        let y_train = y_train_full.i((.., train_indexes));

        println!("x_train.size()={:?}, y_train.size()={:?}", x_train.size(), y_train.size());


        // let (mu_s, cov_s) = super::posterior(&X, &x_train, &y_train, 1., 1., 1e-8);
        // println!("1. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());
        let Ok((mu_s, cov_s)) = super::posterior(&X, &x_train, &y_train, &Tensor::from_slice(&[0.1]).totype(Float), &Tensor::from_slice(&[1.0]).totype(Float), 1e-8) else {
            panic!()
        };
        println!("2. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());
        // let cov_s = cov_s.round_decimals(2);
        // cov_s.print();


        let mut norm = Normal::new(0., 1.).unwrap();
        let mut rng = thread_rng();
        let sig_t = sig_3_norm(norm, rng, N).transpose(0,1);
        println!("sig_t.size={:?}", sig_t.size());


        let chol_or_cov = super::cholesky_with_jitter(&cov_s).unwrap();
        println!("2. chol_or_cov={:?}", chol_or_cov.size());


        let sig_t_with_cov = chol_or_cov.matmul(&sig_t);
        println!("mu_s.size()={:?}, chol.size()={:?}", mu_s.size(), sig_t_with_cov.size());

        let sig_chol = (&mu_s + sig_t_with_cov).transpose(0, 1);
        println!("sig_chol.size={:?}", sig_chol.size());

        plot_series(
            vec![

                (format!("train 0"), ShapeStyle {
                    color: GREEN_800.into(),
                    filled: false,
                    stroke_width: 1,
                }, x_0.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(N))).collect::<Vec<(f32, f32)>>()),
                (format!("train 1"), ShapeStyle {
                    color: ORANGE_800.into(),
                    filled: false,
                    stroke_width: 1,
                }, x_1.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(N))).collect::<Vec<(f32, f32)>>()),

                (format!("Sig0"), ShapeStyle {
                    color: GREEN.into(),
                    filled: true,
                    stroke_width: 2,
                }, x_0.clone().into_iter().zip(from::<Vec<f32>>(&(sig_chol.i((0, ..))))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1"), ShapeStyle {
                    color: ORANGE.into(),
                    filled: true,
                    stroke_width: 2,
                }, x_1.clone().into_iter().zip(from::<Vec<f32>>(&(sig_chol.i((0, ..))))).collect::<Vec<(f32, f32)>>()),


            ], "EEG", (2000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
    }


    #[test]
    fn test_opt(){
        let eeg_id_series:Vec<String> =  vec!["1111".into(), "2222".into(), "3333".into()];
        let store_path = format!("/tmp/test_store_{}.db3", cdt_for_dir());
        debug!("store_path={}", store_path);

        let pulse = pulse_v(50., 10., 100, 20.);
        let N = 1000i64;

        let sig_0=  join_with_offset::<f32>(0., 10.0, &[(100, pulse.clone())], random_saw(1000));
        let sig_1=  join_with_offset::<f32>(0., 10.0, &[(500, pulse.clone())], random_saw(1000));
        let sig_2=  join_with_offset::<f32>(0., 10.0, &[(800, pulse.clone())], random_saw(1000));


        let x = lin_t(N);

        plot_series(
            vec![
                (format!("Sig0"), RED, vec_f32_from(&x).into_iter().zip(sig_0.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1"), GREEN, vec_f32_from(&x).into_iter().zip(sig_1.clone()).collect::<Vec<(f32, f32)>>()),
                (format!("Sig2"), BLACK, vec_f32_from(&x).into_iter().zip(sig_2.clone()).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (1000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
    }

    #[test]
    fn test_test(){
        // let t = Tensor::from_slice(&[0.123, 1.123, 2.123]).round_decimals(2);

        let t = Tensor::zeros(&[3, 3], FLOAT_CPU).fill_diagonal_(1, false);

        t.print();
    }
}