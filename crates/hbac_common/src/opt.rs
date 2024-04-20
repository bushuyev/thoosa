
use nalgebra::{DMatrix, DVector};

fn opt(){
    println!("")
}

#[cfg(test)]
mod opt_tests {
    use std::sync::{Arc, Mutex};
    use chrono::Utc;
    use log::{debug, LevelFilter};
    use plotters::prelude::{BLACK, GREEN, RED};
    use plotters::style::Color;
    use plotters::prelude::ShapeStyle;
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::Kind::Float;
    use tch::Tensor;
    use hbac_shared::{comp_sin_k_v, join_with_offset, pulse_v, random_saw};
    use crate::utils::{cdt_for_dir, plot_series, vec_f32_from};
    use crate::utils::lin_t;
    use hbac_shared::*;
    use statrs::distribution::{Uniform, Normal, MultivariateNormal};
    use rand::{Rng, thread_rng};
    use nalgebra::DMatrix;
    use plotters::style::BLUE;
    use plotters::prelude::YELLOW;
    use rand::prelude::ThreadRng;
    use crate::utils::from;
    use tch::IndexOp;

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

    #[test]
    fn test_dist() -> Result<(), HbacError>{
        // let mut r = MultivariateNormal::new(vec![0.0], vec![]);

        // let mut norm = MultivariateNormal::new(vec![1.0, 2.0], vec![1., 2., 2., 1.])?;
        let mut norm = Normal::new(0., 100.)?;
        //
        let mut rng = thread_rng();
        // let x = rng.sample(norm);


        // x.lower_triangle() != cov.upper_triangle().transpose()

        // let x = DMatrix::from_vec(2, 2, vec![1., 2., 2., 1.]);
        // println!("lower_triangle={:?}", x.lower_triangle().data.as_slice());
        // println!("upper_triangle={:?}", x.upper_triangle().transpose().data.as_slice());
        // println!("x0={:?}", x.column(0).data.into_slice());
        // println!("x1={:?}", x.column(1).data.into_slice());

        let k = 0.9;
        let corr_t = Tensor::stack(&[
            Tensor::from_slice(&[1. , k, k]),
            Tensor::from_slice(&[k, 1. , k]),
            Tensor::from_slice(&[k, k, 1. ])
        ], 0);

        let chol = corr_t.totype(Float).cholesky(false);



        let n = 100;
        let x = (0..n).map(|x| x as f32).collect::<Vec<f32>>();

        let sig_t = sig_3_norm(norm, rng, n);
        let sig_t_chol = chol.matmul(&sig_t);

        plot_series(
            vec![
                (format!("Sig0"), ShapeStyle {
                    color: RED.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t.i((0, ..)))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig0Chol"), ShapeStyle {
                    color: YELLOW.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t_chol.i((0, ..)))).collect::<Vec<(f32, f32)>>()),

                (format!("Sig1"), ShapeStyle {
                    color: GREEN.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t.i((1, ..)))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1Chol"), ShapeStyle {
                    color: GREEN.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t_chol.i((1, ..)))).collect::<Vec<(f32, f32)>>()),


                (format!("Sig2"), ShapeStyle {
                    color: BLUE.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t.i((2, ..)))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig2Chol"), ShapeStyle {
                    color: BLUE.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t_chol.i((2, ..)))).collect::<Vec<(f32, f32)>>()),

            ], "EEG", (2000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();

        Ok(())
    }

    fn sig_3_norm(mut norm: Normal, mut rng: ThreadRng, n: i32) -> Tensor {
        let sig_t_0 = Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None);
        let sig_t_1 = Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None);
        let sig_t_2 = Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None);

        let sig_t = Tensor::stack(&[sig_t_0, sig_t_1, sig_t_2], 0);
        sig_t
    }


    #[test]
    fn test_opt_1(){

        fn kernel(t_0:&Tensor, t_1:&Tensor, l:f32, sigma_f:f32) -> Tensor {
            Tensor::cdist(t_0, t_1, 2.0, 0).pow_tensor_scalar(2).multiply_scalar(-0.5).exp()
        }

        fn posterior(x_s:&Tensor, x_train:&Tensor, y_train:&Tensor, l:f32, sigma_f:f32, sigma_y:f32) -> (Tensor, Tensor){
            println!("x_s={:?}", from::<Vec<f64>>(&x_s.flatten(0,1)));
            println!("x_train={:?}", from::<Vec<f64>>(&x_train.flatten(0,1)));
            println!("y_train={:?}", from::<Vec<f64>>(&y_train.flatten(0,1)));

            let k = kernel(&x_train, &x_train, l, sigma_f);
            println!("k={:?}", from::<Vec<f64>>(&k.flatten(0,1)));

            let k_s = kernel(x_train, x_s, l, sigma_f);

            println!("k_s={:?}", from::<Vec<f64>>(&k_s.flatten(0,1)));

            let k_s_T = k_s.transpose(0, 1);
            let k_ss = kernel(x_s, x_s, l, sigma_f); //TODO + 1e-8 * np.eye(len(X_s));
            println!("k_ss={:?}", from::<Vec<f64>>(&k_ss.flatten(0,1)));

            let k_inv = k.inverse();

            println!("k_inv={:?}", from::<Vec<f64>>(&k_inv.flatten(0,1)));

            let mu_s = k_s_T.matmul(&k_inv).matmul(y_train);
            let cov_s = k_ss - k_s_T.matmul(&k_inv).matmul(&k_s);
            /*
                K = kernel(X_train, X_train, l, sigma_f) + sigma_y**2 * np.eye(len(X_train))
                K_s = kernel(X_train, X_s, l, sigma_f)
                K_ss = kernel(X_s, X_s, l, sigma_f) + 1e-8 * np.eye(len(X_s))
                K_inv = inv(K)

                # Equation (7)
                mu_s = K_s.T.dot(K_inv).dot(Y_train)

                # Equation (8)
                cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

                return mu_s, cov_s
             */

            (mu_s, cov_s)
        }


        let N = 1000i64;

        // let sig =  comp_sin_k_v(N, vec![6., 3., 2.], 0.5);
        // let sig =  comp_sin(N).collect::<Vec<f32>>();
        let X = Tensor::from_slice((0..20).map(|i|-5. +i as f32 * 0.5).into_iter().collect::<Vec<f32>>().as_slice()).totype(Float).view((-1, 1));


        let y_train_full = X.sin().multiply_scalar(10).view((-1, 1)).totype(Float);
        let train_indexes = (0..20).step_by(4).collect::<Vec<i64>>();
        let train_indexes = train_indexes.as_slice();
        let x_train = X.i(train_indexes);
        let y_train = y_train_full.i(train_indexes);

        println!("train_indexes={:?} y_train={:?}", train_indexes, y_train.size());
        // Y_train.print();
        // x_train.print();
        // y_train.print();
        // kernel(&x_train, &y_train, 1.0, 1.0).print();


        let (mu_s, cov_s) = posterior(&X, &x_train, &y_train, 1., 1., 1e-8);


        let mu_v = from::<Vec<f64>>(&mu_s.flatten(0,1));
        println!("mu_v={:?}", mu_v);
        let mu_t = Tensor::from_slice(mu_v.as_slice()).view((20));
        // println!("cov_s={:?}", cov_s.size());
        let cov_v = from::<Vec<f64>>(&cov_s.flatten(0,1));
        for chunk in cov_v.chunks(4) {
            println!("cov_v={:?}", chunk);
        }


        // let mut norm = MultivariateNormal::new(mu_v, cov_v).unwrap();
        let mut norm = Normal::new(0., 1.).unwrap();
        let mut rng = thread_rng();
        let n = 20;
        let sig_t = sig_3_norm(norm, rng, n);

        // let chol = cov_s.totype(Float).cholesky(false);
        // let sig_t_chol = chol.matmul(&sig_t);

        let x = (0..n).map(|x| x as f32).collect::<Vec<f32>>();
        let sig_0_conv = cov_s.matmul(&sig_t.i((0, ..)));
        let sig_1_conv = cov_s.matmul(&sig_t.i((1, ..)));

        println!("mu_t.={:?} sig_0_conv={:?} sig_t.i={:?}", mu_t.size(), sig_0_conv.size(), sig_t.i((0, ..)).size());
        plot_series(
            vec![

                (format!("train"), ShapeStyle {
                    color: BLACK.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.view(20))).collect::<Vec<(f32, f32)>>()),

                (format!("Sig0"), ShapeStyle {
                    color: RED.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&(mu_t.shallow_clone() + sig_0_conv))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig0Chol"), ShapeStyle {
                    color: RED.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t.i((0, ..)))).collect::<Vec<(f32, f32)>>()),

                (format!("Sig1"), ShapeStyle {
                    color: GREEN.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&(mu_t.shallow_clone() + sig_1_conv))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig1Chol"), ShapeStyle {
                    color: GREEN.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t.i((1, ..)))).collect::<Vec<(f32, f32)>>()),

                //
                // (format!("Sig2"), ShapeStyle {
                //     color: BLUE.into(),
                //     filled: true,
                //     stroke_width: 1,
                // }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t.i((2, ..)))).collect::<Vec<(f32, f32)>>()),
                // (format!("Sig2Chol"), ShapeStyle {
                //     color: BLUE.into(),
                //     filled: true,
                //     stroke_width: 2,
                // }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_t.i((2, ..)))).collect::<Vec<(f32, f32)>>()),

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
}