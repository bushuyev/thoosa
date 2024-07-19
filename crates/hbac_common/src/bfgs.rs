use log::debug;
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tch::Tensor;

//https://github.com/pytorch/pytorch/pull/8824
pub struct BFGS<F> where F:FnMut(&mut Tensor)->Tensor {
    func:Box<F>
}


pub struct BFGSParams {
    max_iter: i32,
    with_wolf_line_search:bool,
    lr: f64,

    //tolerance_grad (float): termination tolerance on first order optimality
    //             (default: 1e-7).
    tolerance_grad:f64,

    //  tolerance_change (float): termination tolerance on function
    //             value/parameter changes (default: 1e-9).
    tolerance_change:f64
}

impl BFGSParams {
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    pub fn with_max_iter(mut self, max_iter: i32) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_with_wolf_line_search(mut self, with_wolf_line_search: bool) -> Self {
        self.with_wolf_line_search = with_wolf_line_search;
        self
    }

    pub fn with_tolerance_grad(mut self, tolerance_grad: f64) -> Self {
        self.tolerance_grad = tolerance_grad;
        self
    }

    pub fn with_tolerance_change(mut self, tolerance_change: f64) -> Self {
        self.tolerance_change = tolerance_change;
        self
    }
}

impl Default for BFGSParams {

    fn default() -> Self {
        Self {
            lr: 1.0,
            max_iter:10,
            with_wolf_line_search: false,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9
        }
    }
}


impl <F> BFGS<F> where F:FnMut(&mut Tensor)->Tensor {

    pub fn new(func: F) -> Self {
        Self {
            func:Box::new(func)
        }
    }

    pub fn opt(&mut self, x: Tensor, params:Option<BFGSParams>){
        tch::no_grad(|| {
            self._opt(x, params.unwrap_or_default())
        })
    }

    fn _opt(&mut self, mut x: Tensor, params:BFGSParams){
        debug!("opt: x={:?}", crate::utils::from::<Vec<f64>>(&x));

        let orig_loss = tch::with_grad(|| {
            let orig_loss = (self.func)(&mut x);
            orig_loss.backward();
            orig_loss
        });
        let mut loss = orig_loss.double_value(&[]);

        let mut flat_grad  = Self::_gather_flat_grad(&x);

        debug!("flat_grad: {:?}", crate::utils::from::<Vec<f64>>(&flat_grad));


        let mut opt_cond = flat_grad.abs().max().double_value(&[]) <= params.tolerance_grad;

        // # optimal condition
        if opt_cond {
            return; // orig_loss;
        }

        let mut prev_flat_grad_op = Option::<Tensor>::None;

        let mut d = flat_grad.neg();



        let mut t = params.lr;
        let mut h_diag = 1f64;

        let history_size = 10;
        let mut old_dirs_stps_ro:Vec<(Tensor, Tensor, f64, )> = vec![];

        for n_iter in 1..params.max_iter {

            if n_iter == 1 {
                d = flat_grad.neg();
                h_diag = 1.
            } else {

                let prev_flat_grad_ref = prev_flat_grad_op.as_ref().unwrap();
                debug!("prev_flat_grad_ref print n_iter={}", n_iter);

                debug!("flat_grad print n_iter={}", n_iter);


                //y = flat_grad.sub(prev_flat_grad)
                let y = flat_grad.subtract(prev_flat_grad_ref);
                debug!("y print n_iter={}", n_iter);

                //s = d.mul(t)
                let s = d.multiply_scalar(t);
                debug!("s print n_iter={}", n_iter);

                // ys = y.dot(s)
                let ys = y.matmul(&s).double_value(&[]);
                debug!("ys print n_iter={} ys={}", n_iter, ys);

                if ys > 1e-10 {
                    if old_dirs_stps_ro.len() == history_size {
                        let _ = old_dirs_stps_ro.remove(0);
                    }
                    old_dirs_stps_ro.push((y.shallow_clone(), s, 1./ys));

                    //  h_diag = ys / y.dot(y)  # (y*y)
                    h_diag = ys / y.matmul(&y).double_value(&[]);
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
                    let ai = old_dirs_stps_ro.1.matmul(&q).double_value(&[]) * old_dirs_stps_ro.2;
                    al[i] = Some(ai);
                    //q.add_(old_dirs[i], alpha=-al[i])
                    let _ = q.f_add_(&old_dirs_stps_ro.0.multiply_scalar(ai)).unwrap();
                }

                // # multiply by initial Hessian
                // # r/d is the final direction
                // d = r = torch.mul(q, h_diag)
                d = q.multiply_scalar(h_diag);
                // let mut r = d.shallow_clone();
                // for i in range(num_old):
                for i in 0..num_old {
                    //     be_i = old_dirs[i].dot(r) * ro[i]
                    let old_dirs_stps_ro = old_dirs_stps_ro.get(i).expect(format!("{}", i).as_str());
                    let be_i = old_dirs_stps_ro.0.matmul(&d).double_value(&[]) * old_dirs_stps_ro.2;

                    //r.add_(old_stps[i], alpha=al[i] - be_i)
                    let _ = d.f_add_(&old_dirs_stps_ro.1.multiply_scalar(al[i].expect(format!("al[{}]", i).as_str()) - be_i)).unwrap();
                }

            }

            //  if prev_flat_grad is None:
            //      prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            //  else:
            //      prev_flat_grad.copy_(flat_grad)
            //  prev_loss = loss
            match prev_flat_grad_op {
                None => {

                    let t = crate::utils::clone(&flat_grad);

                    debug!("prev_flat_grad_op 1");

                    prev_flat_grad_op.replace(t);
                }

                Some(ref _prev_flat_grad) => {
                    let mut t = Tensor::zeros(flat_grad.size(), FLOAT_CPU);

                    t.copy_(&flat_grad);

                    debug!("prev_flat_grad_op 2");
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
                t = 1_f64.min(1. / flat_grad.abs().sum(Float).double_value(&[])) * params.lr;
            } else {
                t = params.lr;
            }

            // directional derivative
            let gtd = flat_grad.matmul(&d).double_value(&[]); //  # g * d

            // # directional derivative is below tolerance
            if gtd > - params.tolerance_change {
                break
            }

            // # optional line search: user function
            let mut _ls_func_evals = 0;
            debug!("params.with_wolf_line_search={}", params.with_wolf_line_search);
            if params.with_wolf_line_search {
                let x_init = Tensor::zeros(x.size(), FLOAT_CPU);//TODO clone
                let _ = x_init.set_requires_grad(true);
                let _ = x.clone(&x_init);

                (loss, flat_grad, t, _ls_func_evals)  = Self::_strong_wolfe(
                    |x:&mut Tensor, t:f64, d:&Tensor| Self::_directional_evaluate(self.func.as_mut(), x, t, d),
                    x_init, t, &d, loss, &flat_grad, gtd
                );
                Self::_add_grad(&mut x, t, &d);

            } else {
                //# no line search, simply move with fixed-step
                Self::_add_grad(&mut x, t, &d);
                debug!("x for new flat_grad print n_iter={}", n_iter);


                let orig_loss = tch::with_grad(|| {
                    let orig_loss = (self.func)(&mut x);
                    orig_loss.backward();
                    orig_loss
                });
                let _loss = orig_loss.double_value(&[]);
                flat_grad = Self::_gather_flat_grad(&x);

                debug!("new flat_grad print n_iter={}", n_iter);


                opt_cond = flat_grad.abs().max().double_value(&[]) <= params.tolerance_grad
            }

            debug!("end iter {}", n_iter);

            if opt_cond {
                break
            }

        }


    }

    fn _add_grad(x:&mut Tensor, t:f64, d:&Tensor) {//TODO x - make 2 dim, 1 dim for all params
        debug!("end _add_grad t={}", t);


        // tch::with_grad(|| {
        let _ = x.f_add_(&d.multiply_scalar(t)).unwrap();
        // });

    }

    fn _gather_flat_grad(v: &Tensor) -> Tensor {//TODO x - make 2 dim, 1 dim for all params
        let grad = v.grad();
        debug!("_gather_flat_grad: grad.size()={:?}", grad.size());
        grad
    }

    fn _directional_evaluate(closure:&mut F, x:&mut Tensor, t:f64, d:&Tensor) -> (f64, Tensor) {
        Self::_add_grad(x, t, d);
        debug!("_directional_evaluate: x={:?}", crate::utils::from::<Vec<f64>>(&x));

        let loss = tch::with_grad(|| {
            let loss = closure(x);
            loss.backward();
            loss
        });

        debug!("_directional_evaluate: loss.size()={:?} device={:?} grad.size={:?}", loss.size(), loss.device(), loss.grad().size());

        // debug!("_directional_evaluate: loss={:?} device={:?} grad.size={:?}", crate::utils::from::<Vec<f64>>(&loss), loss.device(), loss.grad().size());

        let flat_grad = Self::_gather_flat_grad(&x); //.to_device(Cpu);

        debug!("_directional_evaluate: flat_grad={:?}", crate::utils::from::<Vec<f64>>(&flat_grad));


        // self._set_param(x)
        // return loss, flat_grad
        (loss.double_value(&[]), flat_grad)
    }

    fn _cubic_interpolate(x1: f64, f1: f64, g1: f64, x2: f64, f2: f64, g2: f64, bounds: Option<(f64, f64)>) -> f64 {

        // # ported from https://github.com/torch/optim/blob/master/polyinterp.lua
        // # Compute bounds of interpolation area
        let  (xmin_bound, xmax_bound) = bounds.unwrap_or(
            if x1 <= x2 {
                (x1, x2)
            } else {
                (x2, x1)
            }
        );

        // # Code for most common case: cubic interpolation of 2 points
        // #   w/ function and derivative values for both
        // # Solution in this case (where x2 is the farthest point):
        // #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
        // #   d2 = sqrt(d1^2 - g1*g2);
        // #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
        // #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
        let d1 = g1 + g2 - 3. * (f1 - f2) / (x1 - x2);
        let d2_square = d1.powi(2) - g1 * g2;
        if d2_square >= 0. {
            let d2 = d2_square.sqrt();
            let min_pos = if x1 <= x2 {
                 x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2. * d2))
            } else {
                x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2. * d2))
            };

            return min_pos.max(xmin_bound).min(xmax_bound)
        } else {
            return (xmin_bound + xmax_bound) / 2.;
        }
    }

    fn _strong_wolfe<WF>(
        mut obj_func:WF,
        mut x:Tensor, mut t:f64, d:&Tensor, f:f64, g:&Tensor, gtd:f64
    ) -> (f64, Tensor, f64, i32) where WF:FnMut(&mut Tensor, f64, &Tensor)->(f64, Tensor){
        let c1=1e-4;
        let c2=0.9;
        let tolerance_change=1e-9;
        let max_ls = 25;

        debug!(
            "_strong_wolfe: x={:?} t={:?} d={:?} f={:?} g={:?} gtd={:?}",
            crate::utils::from::<Vec<f32>>(&x), t, crate::utils::from::<Vec<f32>>(&d), f, crate::utils::from::<Vec<f32>>(&g), gtd);
        //d_norm = d.abs().max()
        let d_norm = d.abs().max();

        //g = g.clone(memory_format=torch.contiguous_format)
        let _g = Tensor::zeros(g.size(), FLOAT_CPU);
        let _ = g.clone(&_g);
        let g = _g;

        // evaluate objective and gradient using initial step
        let (mut f_new, mut g_new) = obj_func(&mut x, t, d);

        let mut ls_func_evals = 1;

        debug!("_strong_wolfe: g_new={:?}", crate::utils::from::<Vec<f64>>(&g_new));
        debug!("_strong_wolfe: d={:?}", crate::utils::from::<Vec<f64>>(d));

        let mut gtd_new = g_new.matmul(d).double_value(&[]);
        //bracket an interval containing a point satisfying the Wolfe criteria
        let (mut t_prev, mut f_prev, mut g_prev, mut gtd_prev) = (0., f, g.shallow_clone(), gtd);
        let mut done = false;
        let mut ls_iter = 0;

        let mut bracket = vec![0f64, 0f64];
        let mut bracket_f = vec![0f64, 0f64];
        let mut bracket_g = vec![Tensor::new(), Tensor::new()];
        let mut bracket_gtd = vec![0f64, 0f64];
        //while ls_iter < max_ls:
        while ls_iter < max_ls {
            // # check conditions
            if f_new > (f + c1 * t * gtd) || (ls_iter > 1 && f_new >= f_prev){
                bracket = vec![t_prev, t];
                bracket_f = vec![f_prev, f_new];
                bracket_g = vec![g_prev, g_new.shallow_clone()];
                bracket_gtd = vec![gtd_prev, gtd_new];
                break
            }
            if f_new > (f + c1 * t * gtd) || (ls_iter > 1 && f_new >= f_prev) {
                bracket = vec![t_prev, t];
                bracket_f = vec![f_prev, f_new];
                bracket_g = vec![g_prev, g_new.shallow_clone()];
                bracket_gtd = vec![gtd_prev, gtd_new];
                break
            }

            if gtd_new.abs() <= -c2 * gtd {
                bracket = vec![t];
                bracket_f = vec![f_new];
                bracket_g = vec![g_new.shallow_clone(), Tensor::new()];
                done = true;
                break
            }

            if gtd_new >= 0. {
                bracket = vec![t_prev, t];
                bracket_f = vec![f_prev, f_new];
                bracket_g = vec![g_prev, g_new.shallow_clone()];
                bracket_gtd = vec![gtd_prev, gtd_new];
                break
            }

            // # interpolate
            let min_step = t + 0.01 * (t - t_prev);
            let max_step = t * 10.0;
            let tmp = t;
            t = Self::_cubic_interpolate(
                t_prev,
                f_prev,
                gtd_prev,
                t,
                f_new,
                gtd_new,
                Some((min_step, max_step)));

            // # next step
            t_prev = tmp;
            f_prev = f_new;
            g_prev = g_new.shallow_clone(); //clone(memory_format=torch.contiguous_format)
            gtd_prev = gtd_new;
            (f_new, g_new) = obj_func(&mut x, t, d);
            ls_func_evals += 1;
            gtd_new = g_new.matmul(&d).double_value(&[]);
            ls_iter += 1;
        }

        //# reached max number of iterations?
        if ls_iter == max_ls {
            bracket = vec![0.0, t];
            bracket_f = vec![f, f_new];
            bracket_g = vec![g.shallow_clone(), g_new.shallow_clone()];
        }

        // # zoom phase: we now have a point satisfying the criteria, or
        // # a bracket around it. We refine the bracket until we find the
        // # exact point satisfying the criteria
        let mut insuf_progress = false;
        // # find high and low points in bracket
        let (mut low_pos, mut high_pos) = if bracket_f[0] <= bracket_f[bracket_f.len()-1] {
            (0, 1)
        } else {
            (1, 0)
        };

        while !done && ls_iter < max_ls {
               // # line-search bracket is so small
               if (bracket[1] - bracket[0]).abs() * d_norm.double_value(&[]) < tolerance_change {
                   break
               }

               //compute new trial value
               t = Self::_cubic_interpolate(bracket[0], bracket_f[0], bracket_gtd[0],
                                      bracket[1], bracket_f[1], bracket_gtd[1], None);

               // # test that we are making sufficient progress:
               // # in case `t` is so close to boundary, we mark that we are making
               // # insufficient progress, and if
               // #   + we have made insufficient progress in the last step, or
               // #   + `t` is at one of the boundary,
               // # we will move `t` to a position which is `0.1 * len(bracket)`
               // # away from the nearest boundary point.
            let barcket_max = *bracket.iter().max_by(|a, b| a.total_cmp(b)).expect("bracket max");
            let bracket_min = *bracket.iter().min_by(|a, b| a.total_cmp(b)).expect("bracket_min");

            let eps = 0.1 * (barcket_max - bracket_min);
               if (barcket_max - t).min( t - bracket_min) < eps {
                   // # interpolation close to boundary

                   if insuf_progress || t >= barcket_max || t <= bracket_min {
                       // # evaluate at 0.1 away from boundary
                       if (t - barcket_max).abs() < (t - bracket_min).abs() {
                           t = barcket_max - eps;
                       } else {
                           t = bracket_min + eps;
                       }
                       insuf_progress = false;
                   } else {
                       insuf_progress = true;
                   }
               } else {
                   insuf_progress = false
               }

               // # Evaluate new point
                (f_new, g_new) = obj_func(&mut x, t, d);
               ls_func_evals += 1;
               gtd_new = g_new.matmul(&d).double_value(&[]);
               ls_iter += 1;

               if f_new > (f + c1 * t * gtd) || f_new >= bracket_f[low_pos] {
                   // # Armijo condition not satisfied or not lower than lowest point
                   bracket[high_pos] = t;
                   bracket_f[high_pos] = f_new;
                   bracket_g[high_pos] = g_new.shallow_clone();
                   bracket_gtd[high_pos] = gtd_new;
                    (low_pos, high_pos) =  if bracket_f[0] <= bracket_f[1] {(0, 1)} else {(1, 0)};
               } else {
                   if gtd_new.abs() <= -c2 * gtd {
                       // # Wolfe conditions satisfied
                       done = true;
                   } else if gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0. {
                       // # old high becomes new low
                       bracket[high_pos] = bracket[low_pos];
                       bracket_f[high_pos] = bracket_f[low_pos];
                       bracket_g[high_pos] = bracket_g[low_pos].shallow_clone();
                       bracket_gtd[high_pos] = bracket_gtd[low_pos];
                   }

                   // # new point becomes new low
                   bracket[low_pos] = t;
                   bracket_f[low_pos] = f_new;
                   bracket_g[low_pos] = g_new.shallow_clone();
                   bracket_gtd[low_pos] = gtd_new;
               }

       }


           // # return stuff
           t = bracket[low_pos];
           f_new = bracket_f[low_pos];
           g_new = bracket_g[low_pos].shallow_clone();
           return (f_new, g_new, t, ls_func_evals);



    }

}

#[cfg(test)]
mod opt_tests {
    use log::{debug, LevelFilter};
    use simplelog::{ColorChoice, CombinedLogger, Config, TerminalMode, TermLogger};
    use tch::{IndexOp, Tensor};
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;

    use hbac_shared::HbacError;

    use crate::bfgs::{BFGS, BFGSParams};




    #[test]
    fn test_bfgs_one_dim_no_line_search_3d() -> Result<(), HbacError>{
        let func = |x: &mut Tensor| {
            debug!("func: x=");

            x.zero_grad();

            let v =
                x.i(0).f_add_scalar(-11.).unwrap().pow_tensor_scalar(2) +
                x.i(1).f_add_scalar(-22.).unwrap().pow_tensor_scalar(2);
            debug!("func: v={v}");

            v
        };

        let mut bfgs = BFGS::new(func);


        let x = Tensor::zeros([2], FLOAT_CPU).set_requires_grad(true);
        debug!("init: x={:?}", x.size());

        bfgs.opt(x.shallow_clone(), Some(BFGSParams::default().with_max_iter(6)));

        debug!("opt: {:?} {:?}", x.double_value(&[0]), x.double_value(&[1]));
        assert_eq!("11.0", format!("{:.1}", x.double_value(&[0])));
        assert_eq!("22.0", format!("{:.1}", x.double_value(&[1])));

        Ok(())
    }


    #[test]
    fn test_bfgs_one_dim_no_line_search_4d() -> Result<(), HbacError>{
        let func = |x: &mut Tensor| {
            debug!("func: x={x}");

            x.zero_grad();

            let v =
                x.i(0).f_add_scalar(-1.).unwrap().pow_tensor_scalar(2) +
                x.i(1).f_add_scalar(-2.).unwrap().pow_tensor_scalar(2) +
                x.i(2).f_add_scalar(-3.).unwrap().pow_tensor_scalar(2);
            debug!("func: v=");

            v
        };

        let mut bfgs = BFGS::new(func);

        let x = Tensor::zeros([3], FLOAT_CPU).set_requires_grad(true);
        debug!("init: x={:?}", x.size());

        bfgs.opt(x.shallow_clone(), Some(BFGSParams::default().with_max_iter(6)));

        let x = x.round_decimals(2);
        debug!("opt: {:?} {:?} {:?}", x.double_value(&[0]), x.double_value(&[1]), x.double_value(&[2]));
        assert_eq!("1.00", format!("{:.2}", x.double_value(&[0])));
        assert_eq!("2.00", format!("{:.2}", x.double_value(&[1])));
        assert_eq!("3.00", format!("{:.2}", x.double_value(&[2])));

        Ok(())
    }

    #[test]
    fn test_bfgs_one_dim_no_line_search() -> Result<(), HbacError>{


        let func = |x: &mut Tensor| {
            debug!("func: x=");
            x.zero_grad();

            let v = x.f_add_scalar(-20.).unwrap().pow_tensor_scalar(2) + x.sin();
            debug!("func: v=");
            v
        };

        let mut bfgs = BFGS::new(func);


        let x = Tensor::zeros([1], FLOAT_CPU).set_requires_grad(true);
        // let x = Tensor::from_slice(&[0.]).totype(Float).set_requires_grad(true);
        bfgs.opt(x.shallow_clone(), Some(BFGSParams::default().with_max_iter(6)));

        assert_eq!("19.6520", format!("{:.4}", x.double_value(&[])));


      /*  let x = (0..100).map(|x|-50. + x as f32).collect::<Vec<f32>>();
        let y = x.iter().map(|x|func(&Tensor::from_slice(vec![*x].as_slice())).double_value(&[]) as f32).collect::<Vec<f32>>();


        plot_series(
            vec![

                (format!("train 0"), ShapeStyle {
                    color: GREEN_800.into(),
                    filled: false,
                    stroke_width: 1,
                }, x.into_iter().zip(y.into_iter()).collect::<Vec<(f32, f32)>>()),


            ], "EEG", (1000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
        )?;*/

        Ok(())
    }

    #[test]
    fn test_bfgs_one_dim_with_strong_wolfe() -> Result<(), HbacError>{
        let func = |x: &mut Tensor| {
            debug!("func: x={x}");

            x.zero_grad();

            let v = x.f_add_scalar(-20.).unwrap().pow_tensor_scalar(2) + x.sin();
            debug!("func: v={v}");

            v
        };

        let mut bfgs = BFGS::new(func);


        let x = Tensor::zeros([1], FLOAT_CPU).set_requires_grad(true);
        bfgs.opt(x.shallow_clone(), Some(BFGSParams::default().with_max_iter(6).with_with_wolf_line_search(true)));

        assert_eq!("19.6245", format!("{:.4}", x.double_value(&[])));


        /*  let x = (0..100).map(|x|-50. + x as f32).collect::<Vec<f32>>();
          let y = x.iter().map(|x|func(&Tensor::from_slice(vec![*x].as_slice())).double_value(&[]) as f32).collect::<Vec<f32>>();


          plot_series(
              vec![

                  (format!("train 0"), ShapeStyle {
                      color: GREEN_800.into(),
                      filled: false,
                      stroke_width: 1,
                  }, x.into_iter().zip(y.into_iter()).collect::<Vec<(f32, f32)>>()),


              ], "EEG", (1000, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str()
          )?;*/

        Ok(())
    }


    #[test]
    fn test_bfgs_one_dim_with_strong_wolfe_2() -> Result<(), HbacError> {
        let func = |x: &mut Tensor| {
            debug!("func: x={x}");
            x.zero_grad();

            let x_10 = x.f_add_scalar(-5.).unwrap().pow_tensor_scalar(2);
            let v = x_10.shallow_clone() + x_10.sin().multiply_scalar(2.);
            debug!("func: v={v}");

            v
        };

        let mut bfgs = BFGS::new(func);


        // let x_0 = Tensor::zeros([1], FLOAT_CPU).set_requires_grad(true);
        let x_0 = Tensor::from_slice(&[-10.]).totype(Float).set_requires_grad(true);
        bfgs.opt(x_0.shallow_clone(), Some(BFGSParams::default().with_max_iter(6).with_with_wolf_line_search(true)));

        // assert_eq!("19.6245", format!("{:.4}", x.double_value(&[])));


        /*let x_v = (0..200).map(|x| 0. + x as f32).collect::<Vec<f32>>();
        let y = x_v.iter().map(|x_1| func(&mut Tensor::from_slice(vec![*x_1/10.].as_slice())).double_value(&[]) as f32).collect::<Vec<f32>>();


        plot_series(
            vec![
                (format!("train 0"), ShapeStyle {
                    color: GREEN_800.into(),
                    filled: false,
                    stroke_width: 1,
                }, x_v.into_iter().zip(y.into_iter()).collect::<Vec<(f32, f32)>>()),
            ], "EEG", (500, 500), &format!("/tmp/plots/0_{:?}.png", Utc::now().to_rfc3339()).as_str(),
        )?;*/

        debug!("x: {:?}", x_0.double_value(&[]));

        Ok(())
    }



    // #[test]
    // fn test_bfgs_one_dim_no_grad() -> Result<(), HbacError>{
    //
    //
    //     let func = |x: &mut Tensor| {
    //         debug!("func: x=");
    //         x.print();
    //         x.zero_grad();
    //
    //         let v = x.f_add_scalar(-20.).unwrap().pow_tensor_scalar(2) + x.sin();
    //         debug!("func: v=");
    //         v.print();
    //         Tensor::from_slice(&[v.double_value(&[])]).to_device(Cpu).set_requires_grad(true)
    //     };
    //
    //     let mut bfgs = BFGS::new(func);
    //
    //
    //     let x = Tensor::zeros([1], FLOAT_CPU).set_requires_grad(true);
    //     // let x = Tensor::from_slice(&[0.]).totype(Float).set_requires_grad(true);
    //     bfgs.opt(x.shallow_clone(), Some(BFGSParams::default().with_max_iter(6)));
    //
    //     assert_eq!("19.6520", format!("{:.4}", x.double_value(&[])));
    //
    //
    //     Ok(())
    // }

}