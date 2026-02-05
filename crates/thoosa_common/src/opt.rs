use log::{debug, warn};
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tch::{IndexOp, Tensor};

use thoosa_shared::ThoError;

#[allow(dead_code)]
fn kernel_cdist(t_0: &Tensor, t_1: &Tensor, l: f64, sigma: f64) -> Tensor {
    Tensor::cdist(t_0, t_1, 2.0, 0)
        .pow_tensor_scalar(2)
        .multiply_scalar(-0.5 * l.powi(2))
        .exp()
        .multiply_scalar(sigma.powi(2))
}

#[allow(dead_code)]
pub fn kernel_rbf(t_0: &Tensor, t_1: &Tensor, l: &Tensor, sigma: &Tensor) -> Tensor {
    let sqdist = Tensor::cdist(t_0, t_1, 2.0, 0).pow_tensor_scalar(2);
    // debug!("sqdist={}", sqdist);
    // debug!("sqdist={:?}", vec_f32_from(&sqdist));
    // debug!("sqdist={:?} t_0.size={:?} t_0={:?} t_1.size={:?} t_1={:?}", vec_f32_from(&sqdist), t_0.size(), vec_f32_from(&t_0), t_1.size(), vec_f32_from(&t_1));

    let l_pow = (l + 1e-5).pow_tensor_scalar(2);

    let e_sqdist = sqdist.multiply(&(-0.5 / (l_pow)));

    e_sqdist.exp().multiply(&sigma.pow_tensor_scalar(2))
}

#[allow(dead_code)]
fn posterior(
    x_s: &Tensor,
    x_train: &Tensor,
    y_train: &Tensor,
    l: &Tensor,
    sigma_f: &Tensor,
    sigma_y: f64,
) -> Result<(Tensor, Tensor), ThoError> {
    // let k = kernel(&x_train, &x_train, l, sigma_f);
    // let l = Tensor::from_slice(&[l]).totype(Float);
    // let sigma_f = Tensor::from_slice(&[sigma_f]).totype(Float);

    let k = kernel_rbf(&x_train, &x_train, l, sigma_f);
    let k_size = k.size();
    // debug!("posterior_ch: k.size()={:?}", k.size());
    let k = k + Tensor::zeros(k_size, FLOAT_CPU).fill_diagonal_(sigma_y.powi(2) as f64, false);

    // debug!("posterior_ch: k={:?}", vec_f32_from(&k));

    // let k_size = k.size();

    let chol_t = cholesky_with_jitter(&k)?;

    // debug!("y_train.size()={:?}, chol_t={:?}", y_train.size(), chol_t.size());

    //from sklearn's GaussianProcessRegressor
    let alpha = y_train
        .view((y_train.size()[0], chol_t.size()[0], 1))
        .cholesky_solve(&chol_t, false);

    // let k_trans = kernel(x_s, x_train, l, sigma_f);
    let k_trans = kernel_rbf(x_s, x_train, l, sigma_f);
    let y_mean = k_trans.matmul(&alpha);

    let k_trans_t = k_trans.transpose(0, 1);

    // debug!("k_trans_T:");
    // k_trans_T.print();

    let v = k_trans_t.triangular_solve(&chol_t, false, false, false);
    // debug!("v.0:");
    // v.0.print();

    // debug!("v.v:");
    // v.1.print();

    // let y_cov = kernel(x_s, x_s, l, sigma_f) - v.0.transpose(0, 1).matmul(&v.0);
    let y_cov = kernel_rbf(x_s, x_s, l, sigma_f) - v.0.transpose(0, 1).matmul(&v.0);

    // y_cov.print();

    // debug!("y_mean={:?}", y_mean.size());

    Ok((y_mean.view((-1, y_cov.size()[1])).transpose(0, 1), y_cov))
}

#[allow(dead_code)]
pub fn rbf_nll(
    x_train: &Tensor,
    y_train: &Tensor,
    theta: &Tensor,
    noise: f64,
) -> Result<Tensor, ThoError> {
    debug!(
        "rbf_nll: theta.device={:?} theta.requires_grad={:?}",
        theta.device(),
        theta.requires_grad()
    );
    let noise = noise.powi(2);
    // let l = theta.double_value(&[0]);
    // let sigma_f = theta.double_value(&[1]);
    let l = theta.i(0);
    let sigma_f = theta.i(1); //Tensor::from_slice(&[1.0]);//theta.i(1);
    debug!(
        "rbf_nll: l={} sigma_f={}",
        l.double_value(&[]),
        sigma_f.double_value(&[])
    );

    let noise_t = Tensor::zeros(&[x_train.size()[0], x_train.size()[0]], FLOAT_CPU)
        .fill_diagonal_(noise, false);
    // debug!("noise_t");
    // noise_t.print();

    let k = kernel_rbf(x_train, x_train, &l, &sigma_f) + noise_t;
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

    let dot = y_train.transpose(0, 1).matmul(&s_2); //.double_value(&[]);
    // debug!("DOT={:?}", dot);

    let v: Tensor =
        np_sum + 0.5 * dot + 0.5 * x_train.size()[0] as f64 * (std::f64::consts::PI * 2.).ln();

    debug!(
        "rbf_nll: v={:?} device={:?} v.requires_grad={}",
        v,
        v.device(),
        v.requires_grad()
    );

    // v = np.sum(np.log(np.diagonal(L))) + 0.5 * Y_train.dot(S2) + 0.5 * len(X_train) * np.log(2 * np.pi)

    Ok(v)
}

#[allow(dead_code)]
fn cholesky_rec(
    t: &Tensor,
    tries: usize,
    jitter: f64,
    e: i32,
    e_step: i32,
) -> Result<Tensor, ThoError> {
    match t.f_linalg_cholesky(false) {
        Ok(l) => Ok(l),
        Err(err) => {
            warn!("err={:?}", err);
            // debug!("err.type_id()={:?}", err.type_id()); TODO check type_id()
            if tries > 0 {
                cholesky_rec(
                    &(t + Tensor::zeros(&t.size(), FLOAT_CPU)
                        .fill_diagonal_(10_f64.powi(e) * jitter, false)),
                    tries - 1,
                    jitter,
                    e + e_step,
                    e_step,
                )
            } else {
                Err(ThoError::from_error("can't do cholesky", err))
            }
        }
    }
}

pub fn cholesky_with_jitter(t: &Tensor) -> Result<Tensor, ThoError> {
    cholesky_rec(t, 10, std::f64::EPSILON, 0, 2)
}

#[cfg(test)]
#[allow(unused_imports, unused_variables)]
mod opt_tests {
    use std::iter::once;
    use std::ops::Deref;

    use argmin::core::State;
    use argmin::core::{CostFunction, Error, Executor};
    use argmin::solver::brent::BrentOpt;
    use chrono::Utc;
    use log::debug;
    use plotters::prelude::ShapeStyle;
    use plotters::prelude::{BLACK, GREEN, RED};
    use plotters::style::full_palette::*;
    use rand::Rng;
    use rand::rngs::ThreadRng;
    use statrs::distribution::Normal;
    use tch::Device::Cpu;
    use tch::IndexOp;
    use tch::Kind::Float;
    use tch::Tensor;
    use tch::kind::FLOAT_CPU;

    use thoosa_shared::*;
    use thoosa_shared::{join_with_offset, pulse_v, random_saw};

    use crate::acquisitions::expected_improvement;
    use crate::argmin::{TensorProblem, run_lbfgs};
    use crate::gaussian::GaussianProcess;
    use crate::opt::{kernel_rbf, rbf_nll};
    use crate::utils::from;
    use crate::utils::lin_t;
    use crate::utils::{plot_series, plot_series_with_points, sin3_pow2, to_vec_f32};

    #[derive(Debug)]
    struct ArgminCoreError {
        context: &'static str,
        source: Error,
    }

    impl ArgminCoreError {
        fn new(context: &'static str, source: Error) -> Self {
            Self { context, source }
        }
    }

    impl std::fmt::Display for ArgminCoreError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}: {}", self.context, self.source)
        }
    }

    impl std::error::Error for ArgminCoreError {}

    #[allow(dead_code)]
    fn sig_1_norm(_norm: Normal, _rng: ThreadRng, n: i64) -> Tensor {
        Tensor::randn([n], FLOAT_CPU).cumsum(0, None).view((1, -1))
    }

    fn sig_3_norm(_norm: Normal, _rng: ThreadRng, n: i64) -> Tensor {
        let sig_t_0 = Tensor::randn([n], FLOAT_CPU).cumsum(0, None);
        let sig_t_1 = Tensor::randn([n], FLOAT_CPU).cumsum(0, None);
        let sig_t_2 = Tensor::randn([n], FLOAT_CPU).cumsum(0, None);
        Tensor::stack(&[sig_t_0, sig_t_1, sig_t_2], 0)
    }

    #[test]
    fn test_rbf() -> Result<(), ThoError> {
        let x_train = Tensor::from_slice(&[1., 2., 3., 4.]).view((-1, 1));

        let k = kernel_rbf(
            &x_train,
            &x_train,
            &Tensor::from_slice(&[1.0]),
            &Tensor::from_slice(&[1.0]),
        );

        let samp_k = Tensor::from_slice(&[
            1., 0.60653066, 0.13533528, 0.011109, 0.60653066, 1., 0.60653066, 0.13533528,
            0.13533528, 0.60653066, 1., 0.60653066, 0.011109, 0.13533528, 0.60653066, 1.,
        ])
        .view((4, 4));

        assert_eq!(samp_k.round_decimals(4), k.round_decimals(4));

        Ok(())
    }

    //FROM sklearn.gaussian_process kernels.RBF()

    #[test]
    fn test_rbf_sim() -> Result<(), ThoError> {
        let x_train = Tensor::from_slice(&[-1.11111111, 1.11111111]).view((-1, 1));
        let k = kernel_rbf(
            &x_train,
            &x_train,
            &Tensor::from_slice(&[1.0]),
            &Tensor::from_slice(&[1.0]),
        );

        assert_eq!(
            Tensor::from_slice(&[1., 0.08465799, 0.08465799, 1.])
                .view((2, 2))
                .round_decimals(4),
            k.round_decimals(4)
        );

        let x_train = Tensor::from_slice(&[-2., 2.]).view((-1, 1));
        let k = kernel_rbf(
            &x_train,
            &x_train,
            &Tensor::from_slice(&[1.0]),
            &Tensor::from_slice(&[1.0]),
        );

        assert_eq!(
            Tensor::from_slice(&[
                1.00000000e+00,
                3.35462628e-04,
                3.35462628e-04,
                1.00000000e+00
            ])
            .view((2, 2))
            .round_decimals(4),
            k.round_decimals(4)
        );

        Ok(())
    }

    #[test]
    fn test_rbf_10() -> Result<(), ThoError> {
        let x_train = Tensor::from_slice(&[-4., -3., -2., -1., 0.]).view((-1, 1));

        let k = super::kernel_rbf(
            &x_train,
            &x_train,
            &Tensor::from_slice(&[-10.0]),
            &Tensor::from_slice(&[1.0]),
        );

        let samp_k = Tensor::from_slice(&[
            1., 0.99501247, 0.98019863, 0.9559974, 0.9231162, 0.99501247, 1., 0.99501247,
            0.98019863, 0.9559974, 0.98019863, 0.99501247, 1., 0.99501247, 0.98019863, 0.9559974,
            0.98019863, 0.99501247, 1., 0.99501247, 0.9231162, 0.9559974, 0.98019863, 0.99501247,
            1.,
        ])
        .view((5, 5));

        assert_eq!(samp_k.round_decimals(4), k.round_decimals(4));

        Ok(())
    }

    #[test]
    fn test_rbf_chol_rand() -> Result<(), ThoError> {
        let n = 1000;
        let x_train = Tensor::rand([n, 1], FLOAT_CPU);
        let k = super::kernel_cdist(&x_train, &x_train, 0.0, 0.0);

        let chol_t = super::cholesky_with_jitter(&k)?;

        assert_eq!(vec![1000, 1000], chol_t.size());

        Ok(())
    }

    #[test]
    fn test_posterior_ch_mult() -> Result<(), ThoError> {
        let x_train = Tensor::from_slice(&[1., 40., 2., 30., 3., 20., 4., 10.]).view((-1, 2));

        let y_train = Tensor::from_slice(&[2., 20., 2., 20.]).view((1, -1));
        let x_pred = x_train.i((vec![1i64, 3i64], ..));

        let Ok((mu_s, cov_s)) = super::posterior(
            &x_pred,
            &x_train,
            &y_train,
            &Tensor::from_slice(&[1.0]).totype(Float),
            &Tensor::from_slice(&[1.0]).totype(Float),
            1e-8,
        ) else {
            panic!()
        };

        debug!("mu_s={mu_s}");
        debug!("cov_s={cov_s}");

        assert_eq!(
            Tensor::from_slice(&[20., 20.]).view((2, 1)).totype(Float),
            mu_s.totype(Float)
        );

        assert_eq!(
            Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0,])
                .view((2, 2))
                .totype(Float),
            cov_s.totype(Float)
        );

        Ok(())
    }

    #[test]
    fn test_posterior() -> Result<(), ThoError> {
        let x_train = Tensor::from_slice(&[1., 2., 3., 4.]).view((-1, 1));
        let y_train = Tensor::from_slice(&[2., 20., 2., 20.]).view((1, -1));

        let x_pred = Tensor::from_slice(&[1., 3.]).view((-1, 1));

        let Ok((mu_s, cov_s)) = super::posterior(
            &x_pred,
            &x_train,
            &y_train,
            &Tensor::from_slice(&[1.0]).totype(Float),
            &Tensor::from_slice(&[1.0]).totype(Float),
            1e-8,
        ) else {
            panic!("posterior")
        };

        assert_eq!(
            Tensor::from_slice(&[2.0, 2.0,]).view((2, 1)).totype(Float),
            mu_s.totype(Float)
        );

        assert_eq!(
            Tensor::from_slice(&[0.0, 0.0, 0.0, 0.0,])
                .view((2, 2))
                .totype(Float),
            cov_s.totype(Float)
        );

        Ok(())
    }

    #[test]
    fn test_rbf_nll() -> Result<(), ThoError> {
        let x_train = Tensor::from_slice(&[-4., -3., -2., -1., 1.]).view((-1, 1));

        let y_train = Tensor::from_slice(&[0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.])
            .view((-1, 1));

        for theta in [
            [0.1, 0.1, 48.0],
            [0.1, 1., 5.67],
            [1., 0.1, 41.00],
            [1., 1., 4.64],
        ] {
            let v = super::rbf_nll(&x_train, &y_train, &Tensor::from_slice(&theta), 0.1_f64)
                .expect("nll");

            assert_eq!(
                format!("{:.1}", theta[2]),
                format!("{:.1}", v.double_value(&[]))
            );
        }

        // let v = super::nll(&x_train, &y_train, &Tensor::from_slice(&[-10., 1.])).expect("nll");
        //
        // debug!("v={:.2}", v);
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
    fn test_nll_brent() -> Result<(), ThoError> {
        let x_train = Tensor::from_slice(&[-4., -3., -2., -1., 1.]).view((-1, 1));

        let y_train = Tensor::from_slice(&[0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.])
            .view((-1, 1));

        debug!("x_train={:?}", x_train.size());

        struct MyProblem {
            x_train: Tensor,
            y_train: Tensor,
        }

        impl CostFunction for MyProblem {
            type Param = f64;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                Ok(super::rbf_nll(
                    &self.x_train,
                    &self.y_train,
                    &Tensor::from_slice(&[*p, 0.74]),
                    0.1_f64,
                )
                .expect("nll")
                .double_value(&[]))
            }
        }

        let cost = MyProblem { x_train, y_train };

        let solver = BrentOpt::new(-100., 100.);

        let res = Executor::new(cost, solver)
            .configure(|state| state.param(-100.).max_iters(10).target_cost(0.0))
            .run()
            .map_err(|e| ArgminCoreError::new("brent run", e))?;

        debug!("res={:?}", res.state().get_best_param().unwrap());

        Ok(())
    }

    //stack if first point is min
    #[test]
    fn test_opt_1d_with_ei() -> Result<(), ThoError> {
        let n = 100i64;

        let (y_train_full, x_t, x) = sin3_pow2(n, -2., 2., Some(0.02));

        let train_indexes = vec![0, 50, 70, 90];

        let x_train = x_t.i(train_indexes.as_slice());
        let y_train = y_train_full.i((.., train_indexes.as_slice()));

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

        let gpf = gp.fit(&x_train, &y_train)?;

        // debug!("gpf")
        let gpf_ref = &gpf;
        let y_train_min = y_train.min();
        let next = run_lbfgs(
            TensorProblem::new(move |theta| {
                expected_improvement(&theta, &y_train_min, gpf_ref, 0.01).negative()
            }),
            Tensor::from_slice(&[0.])
                .totype(Float)
                .to_device(Cpu)
                .view(1),
            // Tensor::from_slice(&[1., 1.]).totype(Float).to_device(Cpu),
            f64::NEG_INFINITY, //0.,//
            10,
        )?;

        let opt_result =next.state().get_best_param().unwrap().deref();


        assert!(opt_result.double_value(&[0]) < 1.);

        /*
        // debug!("opt_result=\n{opt_result}");
        // debug!("x_t={}", x_t);

        let pred_y = gpf.predict(&x_t)?;


        plot_series_with_points(
            vec![
                (
                    format!("train"),
                    ShapeStyle {
                        color: BLACK.into(),
                        filled: true,
                        stroke_width: 1,
                    },
                    x.clone()
                        .into_iter()
                        .zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(n)))
                        .collect::<Vec<(f32, f32)>>(),
                ),
                (
                    format!("pred_y"),
                    ShapeStyle {
                        color: ORANGE.into(),
                        filled: true,
                        stroke_width: 1,
                    },
                    x.clone()
                        .into_iter()
                        .zip(from::<Vec<f32>>(&(pred_y.i((0, ..)))))
                        .collect::<Vec<(f32, f32)>>(),
                ),
            ],
            "EEG",
            (2000, 500),
            &format!("/tmp/plots/1_{:?}.png", Utc::now().to_rfc3339()).as_str(),
            &from::<Vec<f32>>(&x_train.view(-1))
                .into_iter()
                .zip(from::<Vec<f32>>(&y_train.view(-1)).into_iter())
                .collect::<Vec<(f32, f32)>>(),
        )?;

         */
        Ok(())
    }

    #[test]
    fn test_opt_1d() -> Result<(), ThoError> {
        let n = 100i64;

        let x = (0..n)
            .map(|i| -std::f32::consts::PI + (i as f32 / n as f32) * 2. * std::f32::consts::PI)
            .into_iter()
            .collect::<Vec<f32>>();
        let x_t = Tensor::from_slice(x.as_slice()).totype(Float).view((-1, 1));

        let y_train_full = Tensor::stack(
            &[
                x_t.sin().multiply_scalar(10).view((-1, 1)).totype(Float),
            ],
            0,
        )
        .view((1, -1)); //.transpose(0,1);

        let train_indexes = (0..n).step_by(10).collect::<Vec<i64>>();
        let train_indexes = train_indexes.as_slice();
        let x_train = x_t.i(train_indexes);
        let y_train = y_train_full.i((.., train_indexes));

        // Fit GP on sparse samples and validate interpolation on the full grid.
        // let (mu_s, cov_s) = super::posterior(&X, &x_train, &y_train, 1., 1., 1e-8);
        // debug!("1. mu_s.size()={:?}, cov_s.size()={:?}", mu_s.size(), cov_s.size());

        let Ok((mu_s, cov_s)) = super::posterior(
            &x_t,
            &x_train,
            &y_train,
            &Tensor::from_slice(&[1.0]).totype(Float),
            &Tensor::from_slice(&[1.0]).totype(Float),
            1e-8,
        ) else {
            panic!()
        };
        debug!(
            "2. mu_s.size()={:?}, cov_s.size()={:?}",
            mu_s.size(),
            cov_s.size()
        );
        assert_eq!(vec![n, 1], mu_s.size());
        assert_eq!(vec![n, n], cov_s.size());

        let mu_train = mu_s.i(train_indexes);
        let y_train_t = y_train.transpose(0, 1);
        let interp_err = (mu_train - y_train_t).abs().max().double_value(&[]);
        assert!(
            interp_err < 1e-3,
            "posterior should interpolate training data, err={interp_err}"
        );
        let x_moved = x_train.view((-1, 1));
        let y_moved = y_train.view((-1, 1));
        let x_moved_eval = x_moved.shallow_clone();
        let y_moved_eval = y_moved.shallow_clone();
        let nll_start = rbf_nll(
            &x_moved_eval,
            &y_moved_eval,
            &Tensor::from_slice(&[1., 1.]).to_device(Cpu),
            1e-8,
        )?
        .double_value(&[]);
        // Optimize kernel hyperparameters by minimizing the negative log likelihood.
        let l_sigma = run_lbfgs(
            TensorProblem::new(move |theta| rbf_nll(&x_moved, &y_moved, theta, 1e-8).expect("nll")),
            Tensor::from_slice(&[1., 1.]).to_device(Cpu),
            0.0,
            100,
        )?;

        let l_sigma = l_sigma.state().get_best_param().expect("best_param");

        debug!("l_sigma={:?} - {} / {} ",l_sigma,l_sigma.i(0).double_value(&[]),l_sigma.i(1).double_value(&[]));

        let nll_opt = rbf_nll(&x_moved_eval, &y_moved_eval, &l_sigma, 1e-8)?
            .double_value(&[]);
        assert!(
            nll_opt <= nll_start + 1e-6,
            "expected NLL to improve, start={nll_start} opt={nll_opt}"
        );

        // let Ok((mu_s_opt, cov_s_opt)) = super::posterior_ch(&X, &x_train, &y_train, 1.6409202082368064, 8.504394038804222, 1e-8) else {
        let Ok((mu_s_opt, cov_s_opt)) =
            super::posterior(&x_t, &x_train, &y_train, &l_sigma.i(0), &l_sigma.i(1), 1e-8)
        else {
            panic!()
        };
        debug!("2. mu_s.size()={:?}, cov_s.size()={:?}",mu_s.size(),cov_s.size());
        assert_eq!(vec![n, 1], mu_s_opt.size());
        assert_eq!(vec![n, n], cov_s_opt.size());

        let norm = Normal::new(0., 1.).unwrap();
        let rng = rand::rng();
        // Draw a few posterior samples to validate shape propagation.
        let sig_t = sig_3_norm(norm, rng, n).transpose(0, 1);
        debug!("sig_t.size={:?}", sig_t.size());
        assert_eq!(vec![n, 3], sig_t.size());

        let sig_chol = apply_post(&mu_s, &cov_s, &sig_t);
        let sig_chol_opt = apply_post(&mu_s_opt, &cov_s_opt, &sig_t);
        assert_eq!(vec![3, n], sig_chol.size());
        assert_eq!(vec![3, n], sig_chol_opt.size());

/*        plot_series_with_points(
            vec![
                (
                    format!("train"),
                    ShapeStyle {
                        color: GREEN_800.into(),
                        filled: true,
                        stroke_width: 1,
                    },
                    x.clone()
                        .into_iter()
                        .zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(n)))
                        .collect::<Vec<(f32, f32)>>(),
                ),
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
                (
                    format!("Sig0"),
                    ShapeStyle {
                        color: GREEN.into(),
                        filled: true,
                        stroke_width: 2,
                    },
                    x.clone()
                        .into_iter()
                        .zip(from::<Vec<f32>>(&(sig_chol.i((0, ..)))))
                        .collect::<Vec<(f32, f32)>>(),
                ),
                // (format!("Sig0Chol"), ShapeStyle {
                //     color: BLUE.into(),
                //     filled: true,
                //     stroke_width: 2,
                // }, x.clone().into_iter().zip(from::<Vec<f32>>(&sig_chol.i(((1, ..))))).collect::<Vec<(f32, f32)>>()),
                (
                    format!("Sig0Chol"),
                    ShapeStyle {
                        color: RED.into(),
                        filled: true,
                        stroke_width: 2,
                    },
                    x.clone()
                        .into_iter()
                        .zip(from::<Vec<f32>>(&sig_chol_opt.i((0, ..))))
                        .collect::<Vec<(f32, f32)>>(),
                ),
            ],
            "EEG",
            (2000, 500),
            &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str(),
            &from::<Vec<f32>>(&x_train.view(-1))
                .into_iter()
                .zip(from::<Vec<f32>>(&y_train.view(-1)).into_iter())
                .collect::<Vec<(f32, f32)>>(),
        ).unwrap();
*/

        Ok(())
    }

    fn apply_post(mu_s: &Tensor, cov_s: &Tensor, sig_t: &Tensor) -> Tensor {
        let chol_or_cov = super::cholesky_with_jitter(&cov_s).unwrap();
        debug!("2. chol_or_cov={:?}", chol_or_cov.size());

        let sig_t_with_cov = chol_or_cov.matmul(&sig_t);
        debug!(
            "mu_s.size()={:?}, chol.size()={:?}",
            mu_s.size(),
            sig_t_with_cov.size()
        );

        let sig_chol = (mu_s + sig_t_with_cov).transpose(0, 1);
        debug!("sig_chol.size={:?}", sig_chol.size());
        sig_chol
    }

    #[test]
    fn test_opt_2d() {
        let n = 1000i64;
        let min_max = 100.; // std::f32::consts::PI;
        // Sample a 2D diagonal sweep so x0 increases while x1 decreases.
        let x_01 = (0..n)
            .map(|i| {
                (
                    -min_max + (i as f32 / n as f32) * 2. * min_max,
                    min_max - (i as f32 / n as f32) * 2. * min_max,
                )
            })
            .collect::<Vec<(f32, f32)>>();

        let y = x_01
            .iter()
            .map(|x_01| x_01.0.powi(2) + x_01.1.powi(2))
            .collect::<Vec<f32>>();

        let x_0 = x_01.iter().map(|x_01| x_01.0).collect::<Vec<f32>>();
        let x_1 = x_01.iter().map(|x_01| x_01.1).collect::<Vec<f32>>();

        let x = Tensor::from_slice(
            x_01.into_iter()
                .flat_map(|x_01| once(x_01.0).chain(once(x_01.1)))
                .collect::<Vec<f32>>()
                .as_slice(),
        )
        .view((-1, 2));
        assert_eq!(vec![n, 2], x.size());

        let y_train_full = Tensor::from_slice(y.as_slice()).view((1, -1));
        assert_eq!(vec![1, n], y_train_full.size());

        // Fit on a sparse subset and validate interpolation on training points.
        let train_indexes = (0..n).step_by(10).collect::<Vec<i64>>();
        let train_indexes = train_indexes.as_slice();
        let x_train = x.i(train_indexes);
        let y_train = y_train_full.i((.., train_indexes));
        let train_len = train_indexes.len() as i64;
        assert_eq!(vec![train_len, 2], x_train.size());
        assert_eq!(vec![1, train_len], y_train.size());

        debug!("x_train.size()={:?}, y_train.size()={:?}",x_train.size(),y_train.size());


        let Ok((mu_s, cov_s)) = super::posterior(
            &x,
            &x_train,
            &y_train,
            &Tensor::from_slice(&[0.1]).totype(Float),
            &Tensor::from_slice(&[1.0]).totype(Float),
            1e-8,
        ) else {
            panic!()
        };
        debug!("2. mu_s.size()={:?}, cov_s.size()={:?}",mu_s.size(),cov_s.size());
        assert_eq!(vec![n, 1], mu_s.size());
        assert_eq!(vec![n, n], cov_s.size());

        let mu_train = mu_s.i(train_indexes);
        let y_train_t = y_train.transpose(0, 1);
        let abs_err = (mu_train - &y_train_t).abs().max().double_value(&[]);
        let max_y = y_train_t.abs().max().double_value(&[]);
        let rel_err = abs_err / max_y.max(1.0);
        assert!(
            rel_err < 1e-4,
            "posterior should interpolate training data, rel_err={rel_err} abs_err={abs_err}"
        );

        let norm = Normal::new(0., 1.).unwrap();
        let rng = rand::rng();
        // Draw a few samples to ensure covariance application preserves expected shapes.
        let sig_t = sig_3_norm(norm, rng, n).transpose(0, 1);
        debug!("sig_t.size={:?}", sig_t.size());
        assert_eq!(vec![n, 3], sig_t.size());

        let chol_or_cov = super::cholesky_with_jitter(&cov_s).unwrap();
        debug!("2. chol_or_cov={:?}", chol_or_cov.size());
        assert_eq!(vec![n, n], chol_or_cov.size());

        let sig_t_with_cov = chol_or_cov.matmul(&sig_t);
        debug!(
            "mu_s.size()={:?}, chol.size()={:?}",
            mu_s.size(),
            sig_t_with_cov.size()
        );
        assert_eq!(vec![n, 3], sig_t_with_cov.size());

        let sig_chol = (&mu_s + sig_t_with_cov).transpose(0, 1);
        debug!("sig_chol.size={:?}", sig_chol.size());
        assert_eq!(vec![3, n], sig_chol.size());

/*        plot_series(
            vec![
                (
                    format!("train 0"),
                    ShapeStyle {
                        color: GREEN_800.into(),
                        filled: false,
                        stroke_width: 1,
                    },
                    x_0.clone()
                        .into_iter()
                        .zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(n)))
                        .collect::<Vec<(f32, f32)>>(),
                ),
                (
                    format!("train 1"),
                    ShapeStyle {
                        color: ORANGE_800.into(),
                        filled: false,
                        stroke_width: 1,
                    },
                    x_1.clone()
                        .into_iter()
                        .zip(from::<Vec<f32>>(&y_train_full.i((0, ..)).view(n)))
                        .collect::<Vec<(f32, f32)>>(),
                ),
                (
                    format!("Sig0"),
                    ShapeStyle {
                        color: GREEN.into(),
                        filled: true,
                        stroke_width: 2,
                    },
                    x_0.clone()
                        .into_iter()
                        .zip(from::<Vec<f32>>(&(sig_chol.i((0, ..)))))
                        .collect::<Vec<(f32, f32)>>(),
                ),
                (
                    format!("Sig1"),
                    ShapeStyle {
                        color: ORANGE.into(),
                        filled: true,
                        stroke_width: 2,
                    },
                    x_1.clone()
                        .into_iter()
                        .zip(from::<Vec<f32>>(&(sig_chol.i((0, ..)))))
                        .collect::<Vec<(f32, f32)>>(),
                ),
            ],
            "EEG",
            (2000, 500),
            &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str(),
        ) .unwrap();
*/
    }

}
