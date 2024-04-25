use tch::kind::FLOAT_CPU;
use tch::Tensor;
use tch::Kind::Float;
use std::cmp::min;
use log::debug;

struct BFGS {
    func:fn(x:&Tensor)->Tensor
}



impl BFGS {
    pub fn new(func: fn(&Tensor) -> Tensor) -> Self {
        Self {
            func        }
    }

    pub fn opt(&mut self){
        tch::no_grad(|| {
            self._opt()
        })
    }
    fn _opt(&mut self){
        let lr = 1.0;

        //tolerance_grad (float): termination tolerance on first order optimality
        //             (default: 1e-7).
        let tolerance_grad = 1e-7;

        //  tolerance_change (float): termination tolerance on function
        //             value/parameter changes (default: 1e-9).
        let mut tolerance_change = 1e-9;

        let mut x = Tensor::zeros([1], FLOAT_CPU).set_requires_grad(true);

        let orig_loss = tch::with_grad(|| {
            let orig_loss = (self.func)(&x);
            orig_loss.backward();
            orig_loss
        });

        let mut flat_grad = x.grad();
        flat_grad.print();

        let mut opt_cond = flat_grad.abs().max().double_value(&[]) <= tolerance_grad;

        // # optimal condition
        if opt_cond {
            return; // orig_loss;
        }

        let mut prev_flat_grad_op = Option::<Tensor>::None;

        let mut d = flat_grad.neg();



        let mut t = lr;
        let mut H_diag = 1f64;

        let history_size = 10;
        let mut old_dirs_stps_ro:Vec<(Tensor, Tensor, f64, )> = vec![];
        let line_search_fn:Option<fn()> = None;

        for n_iter in 1..6 {

            if n_iter == 1 {
                d = flat_grad.neg();
                H_diag = 1.
            } else {

                let prev_flat_grad_ref = prev_flat_grad_op.as_ref().unwrap();
                debug!("prev_flat_grad_ref print n_iter={}", n_iter);
                prev_flat_grad_ref.print();

                debug!("flat_grad print n_iter={}", n_iter);
                flat_grad.print();

                //y = flat_grad.sub(prev_flat_grad)
                let y = flat_grad.subtract(prev_flat_grad_ref);
                debug!("y print n_iter={}", n_iter);
                y.print();

                //s = d.mul(t)
                let s = d.multiply_scalar(t);
                debug!("s print n_iter={}", n_iter);
                s.print();

                // ys = y.dot(s)
                let ys = y.matmul(&s).double_value(&[]);
                debug!("ys print n_iter={} ys={}", n_iter, ys);

                if ys > 1e-10 {
                    if old_dirs_stps_ro.len() == history_size {
                        old_dirs_stps_ro.remove(0);
                    }
                    old_dirs_stps_ro.push((y.shallow_clone(), s, 1./ys));

                    //  H_diag = ys / y.dot(y)  # (y*y)
                    H_diag = ys / y.matmul(&y).double_value(&[]);
                }

                //compute the approximate (L-BFGS) inverse Hessian
                // multiplied by the gradient
                let num_old = old_dirs_stps_ro.len();

                let mut al:Vec<Option<f64>> = (0..history_size).map(|_|None).collect();
                let mut q = flat_grad.neg();

                for i in (0..num_old).rev() {
                    debug!("i={} num_old={}", i, num_old);
                    let old_dirs_stps_ro = old_dirs_stps_ro.get(i).expect(format!("{}", i).as_str());
                    //al[i] = old_stps[i].dot(q) * ro[i]
                    let ai = (old_dirs_stps_ro.1.matmul(&q).double_value(&[]) * old_dirs_stps_ro.2);
                    al[i] = Some(ai);
                    //q.add_(old_dirs[i], alpha=-al[i])
                    q.f_add_(&old_dirs_stps_ro.0.multiply_scalar(ai)).unwrap();
                }

                // # multiply by initial Hessian
                // # r/d is the final direction
                // d = r = torch.mul(q, H_diag)
                d = q.multiply_scalar(H_diag);
                // let mut r = d.shallow_clone();
                // for i in range(num_old):
                for i in 0..num_old {
                    //     be_i = old_dirs[i].dot(r) * ro[i]
                    let old_dirs_stps_ro = old_dirs_stps_ro.get(i).expect(format!("{}", i).as_str());
                    let be_i = old_dirs_stps_ro.0.matmul(&d).double_value(&[]) * old_dirs_stps_ro.2;

                    //r.add_(old_stps[i], alpha=al[i] - be_i)
                    d.f_add_(&old_dirs_stps_ro.1.multiply_scalar(al[i].expect(format!("al[{}]", i).as_str()) - be_i)).unwrap();
                }

            }

            //  if prev_flat_grad is None:
            //      prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            //  else:
            //      prev_flat_grad.copy_(flat_grad)
            //  prev_loss = loss
            match prev_flat_grad_op {
                None => {
                    let mut t = Tensor::zeros(flat_grad.size(), FLOAT_CPU);
                    flat_grad.clone(&t);

                    debug!("prev_flat_grad_op 1");
                    t.print();
                    prev_flat_grad_op.replace(t);

                }
                Some(ref prev_flat_grad) => {
                    let mut t = Tensor::zeros(flat_grad.size(), FLOAT_CPU);

                    t.copy_(&flat_grad);

                    debug!("prev_flat_grad_op 2");
                    t.print();
                    prev_flat_grad_op.replace(t);
                }
            }

            /*
            ############################################################
            # compute step length
            ############################################################
            # reset initial guess for step size
             */
            if n_iter == 1 {
                t = 1_f64.min(1. / flat_grad.abs().sum(Float).double_value(&[])) * lr;
            } else {
                t = lr;
            }

            // directional derivative
            let gtd = flat_grad.matmul(&d).double_value(&[]); //  # g * d

            // # directional derivative is below tolerance
            if gtd > -tolerance_change {
                break
            }

            // # optional line search: user function
            let ls_func_evals = 0;
            if line_search_fn.is_some() {

            } else {
                //# no line search, simply move with fixed-step
                Self::_add_grad(&mut x, t, &d);
                let orig_loss = (self.func)(&x);
                flat_grad = x.grad();

                debug!("new flat_grad print n_iter={}", n_iter);
                flat_grad.print();

                opt_cond = flat_grad.abs().max().double_value(&[]) <= tolerance_grad
            }




            debug!("end iter {}", n_iter);

            if opt_cond {
                break
            }
            //TODO
        }

        debug!("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX");
        x.print();
    }

    fn _add_grad(x:&mut Tensor, t:f64, d:&Tensor) {//TODO x - make 2 dim, 1 dim for all params
        debug!("end _add_grad t={}", t);
        x.print();

        // tch::no_grad(|| {
            x.f_add_(&d.multiply_scalar(t)).unwrap();
        // });

        debug!("--------");
        d.print();
        debug!("--------");
        x.print();
    }
}

#[cfg(test)]
mod opt_tests {
    use chrono::Utc;
    use log::{debug, LevelFilter};
    use plotters::prelude::full_palette::{GREEN_800, ORANGE, ORANGE_800};
    use plotters::prelude::{GREEN, ShapeStyle};
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::Tensor;
    use hbac_shared::HbacError;
    use crate::bfgs::BFGS;
    use crate::utils::{from, plot_series};

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
    fn test_bfgs() -> Result<(), HbacError>{
        let func = |x: &Tensor| x.f_add_scalar(-20.).unwrap().pow_tensor_scalar(2) + x.sin();

        let mut bfgs = BFGS::new(func);

        let x = (0..100).map(|x|-50. + x as f32).collect::<Vec<f32>>();
        let y = x.iter().map(|x|func(&Tensor::from_slice(vec![*x].as_slice())).double_value(&[]) as f32).collect::<Vec<f32>>();


        bfgs.opt();

        plot_series(
            vec![

                (format!("train 0"), ShapeStyle {
                    color: GREEN_800.into(),
                    filled: false,
                    stroke_width: 1,
                }, x.into_iter().zip(y.into_iter()).collect::<Vec<(f32, f32)>>()),


            ], "EEG", (1000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;

        Ok(())
    }
}