use std::any::Any;
use nalgebra::{DMatrix, DVector};
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use log::{warn, debug};
use hbac_shared::HbacError;

fn kernel(t_0:&Tensor, t_1:&Tensor, l:f32, sigma_f:f32) -> Tensor {
    Tensor::cdist(t_0, t_1, 2.0, 0).pow_tensor_scalar(2).multiply_scalar(-0.5).exp()
}

fn posterior(x_s:&Tensor, x_train:&Tensor, y_train:&Tensor, l:f32, sigma_f:f32, sigma_y:f32) -> (Tensor, Tensor){
    println!("x_s={:?}", x_s.size());
    // println!("x_s={:?}", from::<Vec<f64>>(&x_s.flatten(0,1)));
    // println!("x_train={:?}", from::<Vec<f64>>(&x_train.flatten(0,1)));
    // println!("y_train={:?}", from::<Vec<f64>>(&y_train.flatten(0,1)));

    let k = kernel(&x_train, &x_train, l, sigma_f);
    let k_size = k.size();
    println!("k={:?}", k_size);
    // let k = k + Tensor::zeros(k_size, FLOAT_CPU).fill_diagonal_(sigma_y.powi(2) as f64, false);

    let k_s = kernel(&x_train, x_s, l, sigma_f);


    let k_s_T = k_s.transpose(0, 1);

    let k_ss = kernel(x_s, x_s, l, sigma_f);
    let k_ss_size = k_ss.size();
    let k_ss = k_ss  + Tensor::zeros(k_ss_size, FLOAT_CPU).fill_diagonal_(1, false); //TODO + 1e-8 * np.eye(len(X_s));

    let k_inv = k.inverse();


    let mu_s = k_s_T.matmul(&k_inv).matmul(y_train);
    let cov_s = (k_ss - k_s_T.matmul(&k_inv).matmul(&k_s)); //.fill_diagonal_(0, false);

    // let cov_s = cov_s.matmul(&cov_s.transpose(0, 1));

    (mu_s.squeeze().transpose(0,1), cov_s)
}

fn posterior_ch(x_s:&Tensor, x_train:&Tensor, y_train:&Tensor, l:f32, sigma_f:f32, sigma_y:f32) -> Result<(Tensor, Tensor), HbacError>{
    let k = kernel(&x_train, &x_train, l, sigma_f);

    let k_size = k.size();

    let chol_t = cholesky_with_jitter(&k)?;

    println!("y_train.size()={:?}, chol_t={:?}", y_train.size(), chol_t.size());

    let alpha = y_train.view((y_train.size()[0], chol_t.size()[0], 1)).cholesky_solve(&chol_t, false);

    let k_trans = kernel(x_s, x_train, l, sigma_f);
    let y_mean = k_trans.matmul(&alpha);

    let k_trans_T = k_trans.transpose(0, 1);

    // k_trans_T.print();

    let v = k_trans_T.triangular_solve(&chol_t, false, false, false);
    // v.0.print();

    // v.1.print();

    let y_cov = kernel(x_s, x_s, l, sigma_f) - v.0.transpose(0, 1).matmul(&v.0);

    // y_cov.print();

    println!("y_mean={:?}", y_mean.size());

    Ok((y_mean.view((-1, y_cov.size()[1])).transpose(0,1), y_cov))
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
    use plotters::style::full_palette::*;
    use rand::prelude::ThreadRng;
    use crate::utils::from;
    use tch::IndexOp;
    use tch::kind::{DOUBLE_CPU, FLOAT_CPU};
    use std::iter::once;

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


    fn sig_3_norm(mut norm: Normal, mut rng: ThreadRng, n: i64) -> Tensor {
        let sig_t_0 = Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None);
        let sig_t_1 = Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None);
        let sig_t_2 = Tensor::from_slice((0..n).map(|x| rng.sample(norm) as f32).collect::<Vec<f32>>().as_slice()).cumsum(0, None);

        let sig_t = Tensor::stack(&[sig_t_0, sig_t_1, sig_t_2], 0);
        sig_t
    }

    #[test]
    fn test_rbf()  -> Result<(), HbacError> {
        let x_train = Tensor::from_slice(&[1., 2., 3., 4.]).view((-1, 1));
        let k = super::kernel(&x_train, &x_train, 0.0, 0.0);

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

        let chol_t = super::cholesky_with_jitter(&k)?;

        chol_t.print();

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

        let y_train = Tensor::from_slice(&[2., 20., 2., 20.]).view((-1, 1));
        let x_pred = x_train.i((vec![1i64, 3i64], ..));

        x_pred.print();

        let Ok((mu_s, cov_s)) = super::posterior_ch(&x_pred, &x_train, &y_train, 1., 1., 1e-8) else {
            panic!()
        };

        mu_s.print();
        cov_s.print();

        Ok(())
    }

    #[test]
    fn test_posterior_ch() -> Result<(), HbacError> {
        let x_train = Tensor::from_slice(&[1., 2., 3., 4.]).view((-1, 1));
        let y_train = Tensor::from_slice(&[2., 20., 2., 20.]).view((-1, 1));

        let x_pred = Tensor::from_slice(&[1., 3.]).view((-1, 1));

        let Ok((mu_s, cov_s)) = super::posterior_ch(&x_pred, &x_train, &y_train, 1., 1., 1e-8) else {
            panic!()
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
    fn test_opt_1(){

        let N = 100i64;

        // let sig =  comp_sin_k_v(N, vec![6., 3., 2.], 0.5);
        // let sig =  comp_sin(N).collect::<Vec<f32>>();
        let x = (0..N).map(|i| - std::f32::consts::PI + (i as f32 / N as f32)  * 2. * std::f32::consts::PI ).into_iter().collect::<Vec<f32>>();
        let X = Tensor::from_slice(x.as_slice()).totype(Float).view((-1, 1));


        let y_train_full = Tensor::stack(&[
            X.sin().multiply_scalar(10).view((-1, 1)).totype(Float),
            X.f_add_scalar(std::f64::consts::PI/4.).unwrap().sin().multiply_scalar(10).view((-1, 1)).totype(Float),
            X.f_add_scalar(std::f64::consts::PI/2.).unwrap().sin().multiply_scalar(10).view((-1, 1)).totype(Float)
        ], 0).view((3, -1)); //.transpose(0,1);

        println!("y_train_full={:?}", y_train_full.size());

        let train_indexes = (0..N).step_by(20).collect::<Vec<i64>>();
        let train_indexes = train_indexes.as_slice();
        let x_train = X.i(train_indexes);
        let y_train = y_train_full.i((.., train_indexes));

        println!("y_train={:?}", y_train.size());

        // let (mu_s, cov_s) = super::posterior(&X, &x_train, &y_train, 1., 1., 1e-8);
        // println!("1. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());
        let Ok((mu_s, cov_s)) = super::posterior_ch(&X, &x_train, &y_train, 1., 1., 1e-8) else {
            panic!()
        };
        println!("2. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());
        // let cov_s = cov_s.round_decimals(2);
        cov_s.print();


        let mut norm = Normal::new(0., 1.).unwrap();
        let mut rng = thread_rng();
        let sig_t = sig_3_norm(norm, rng, N).transpose(0,1);
        println!("sig_t.size={:?}", sig_t.size());

        // let chol_or_cov = match cov_s.shallow_clone().totype(Float).f_cholesky(false) {
        //     Ok(r)=>r,
        //     Err(e)=>{
        //         println!("XXXXXXXXXXXXXXXXXXXXXXXXXXXe={:?}", e);
        //         cov_s.shallow_clone()
        //     }
        // };
        // let chol_or_cov = cov_s.totype(Float).f_cholesky(false).unwrap();
        // println!("1. chol_or_cov={:?}", chol_or_cov.size());
        let chol_or_cov = super::cholesky_with_jitter(&cov_s).unwrap();
        println!("2. chol_or_cov={:?}", chol_or_cov.size());

        // let x = X; // (0..N).map(|x| x as f32).collect::<Vec<f32>>();

        let sig_t_with_cov = chol_or_cov.matmul(&sig_t);
        println!("mu_s.size()={:?}, chol.size()={:?}", mu_s.size(), sig_t_with_cov.size());

        let sig_chol = (&mu_s + sig_t_with_cov).transpose(0, 1);
        println!("sig_chol.size={:?}", sig_chol.size());

        plot_series(
            vec![

                (format!("train"), ShapeStyle {
                    color: GREEN_800.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(N))).collect::<Vec<(f32, f32)>>()),
                (format!("train"), ShapeStyle {
                    color: BLUE_800.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((1, ..)).view(N))).collect::<Vec<(f32, f32)>>()),
                (format!("train"), ShapeStyle {
                    color: RED_800.into(),
                    filled: true,
                    stroke_width: 1,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((2, ..)).view(N))).collect::<Vec<(f32, f32)>>()),

                (format!("Sig0"), ShapeStyle {
                    color: GREEN.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&(sig_chol.i((0, ..))))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig0Chol"), ShapeStyle {
                    color: BLUE.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_chol.i(((1, ..))))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig0Chol"), ShapeStyle {
                    color: RED.into(),
                    filled: true,
                    stroke_width: 2,
                }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_chol.i(((2, ..))))).collect::<Vec<(f32, f32)>>()),


            ], "EEG", (2000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        ).unwrap();
    }

    #[test]
    fn test_opt_2(){

        let N = 100i64;
        let min_max = std::f32::consts::PI;
        // let sig =  comp_sin_k_v(N, vec![6., 3., 2.], 0.5);
        // let sig =  comp_sin(N).collect::<Vec<f32>>();
        let x_01 = (0..N).map(|i| {
           (
                -min_max + (i as f32 / N as f32)  * 2. * min_max,
                min_max - (i as f32 / N as f32)  * 2. * min_max
           )
        }).collect::<Vec<(f32, f32)>>();

        let y = x_01.iter().map(|x_01| x_01.0.sin() + x_01.1.cos()).collect::<Vec<f32>>();
        let x_0 = x_01.iter().map(|x_01| x_01.0).collect::<Vec<f32>>();
        let x_1 = x_01.iter().map(|x_01| x_01.1).collect::<Vec<f32>>();

        let X = Tensor::from_slice(x_01.into_iter().flat_map(|x_01|once(x_01.0).chain(once(x_01.1))).collect::<Vec<f32>>().as_slice()).view((-1, 2));
        print!("X={:?}", X.size());
        X.print();

        let y_train_full = Tensor::from_slice(y.as_slice()).view((1, -1));
        print!("y_train_full={:?}", y_train_full.size());
        y_train_full.print();

        let train_indexes = (0..N).step_by(5).collect::<Vec<i64>>();
        let train_indexes = train_indexes.as_slice();
        let x_train = X.i(train_indexes);
        let y_train = y_train_full.i((.., train_indexes));


        // let (mu_s, cov_s) = super::posterior(&X, &x_train, &y_train, 1., 1., 1e-8);
        // println!("1. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());
        let Ok((mu_s, cov_s)) = super::posterior_ch(&X, &x_train, &y_train, 1., 1., 1e-8) else {
            panic!()
        };
        println!("2. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());
        // let cov_s = cov_s.round_decimals(2);
        cov_s.print();


        let mut norm = Normal::new(0., 1.).unwrap();
        let mut rng = thread_rng();
        let sig_t = sig_3_norm(norm, rng, N).transpose(0,1);
        println!("sig_t.size={:?}", sig_t.size());

        // let chol_or_cov = match cov_s.shallow_clone().totype(Float).f_cholesky(false) {
        //     Ok(r)=>r,
        //     Err(e)=>{
        //         println!("XXXXXXXXXXXXXXXXXXXXXXXXXXXe={:?}", e);
        //         cov_s.shallow_clone()
        //     }
        // };
        // let chol_or_cov = cov_s.totype(Float).f_cholesky(false).unwrap();
        // println!("1. chol_or_cov={:?}", chol_or_cov.size());
        let chol_or_cov = super::cholesky_with_jitter(&cov_s).unwrap();
        println!("2. chol_or_cov={:?}", chol_or_cov.size());

        // let x = X; // (0..N).map(|x| x as f32).collect::<Vec<f32>>();

        let sig_t_with_cov = chol_or_cov.matmul(&sig_t);
        println!("mu_s.size()={:?}, chol.size()={:?}", mu_s.size(), sig_t_with_cov.size());

        let sig_chol = (&mu_s + sig_t_with_cov).transpose(0, 1);
        println!("sig_chol.size={:?}", sig_chol.size());

        plot_series(
            vec![

                (format!("train 0"), ShapeStyle {
                    color: GREEN_800.into(),
                    filled: true,
                    stroke_width: 1,
                }, x_0.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(N))).collect::<Vec<(f32, f32)>>()),
                (format!("train 1"), ShapeStyle {
                    color: ORANGE_800.into(),
                    filled: true,
                    stroke_width: 1,
                }, x_1.clone().into_iter().zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(N))).collect::<Vec<(f32, f32)>>()),
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
                }, x_0.clone().into_iter().zip(from::<Vec<f32>>(&(sig_chol.i((0, ..))))).collect::<Vec<(f32, f32)>>()),
                (format!("Sig0"), ShapeStyle {
                    color: ORANGE.into(),
                    filled: true,
                    stroke_width: 2,
                }, x_1.clone().into_iter().zip(from::<Vec<f32>>(&(sig_chol.i((0, ..))))).collect::<Vec<(f32, f32)>>()),
                // (format!("Sig0Chol"), ShapeStyle {
                //     color: BLUE.into(),
                //     filled: true,
                //     stroke_width: 2,
                // }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_chol.i(((1, ..))))).collect::<Vec<(f32, f32)>>()),
                // (format!("Sig0Chol"), ShapeStyle {
                //     color: RED.into(),
                //     filled: true,
                //     stroke_width: 2,
                // }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_chol.i(((2, ..))))).collect::<Vec<(f32, f32)>>()),
                //

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