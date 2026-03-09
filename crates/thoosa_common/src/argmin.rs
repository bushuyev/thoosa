use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

use argmin::core::{CostFunction, Error, Gradient};
#[cfg(test)]
use argmin::core::{Executor, IterState, OptimizationResult};
#[cfg(test)]
use argmin::solver::linesearch::MoreThuenteLineSearch;
#[cfg(test)]
use argmin::solver::quasinewton::LBFGS;
use argmin_math::{
    ArgminAdd, ArgminDot, ArgminL1Norm, ArgminL2Norm, ArgminMinMax, ArgminMul,
    ArgminSignum, ArgminSub, ArgminZeroLike,
};
use log::debug;
// Avoid importing rand::Rng to prevent trait version conflicts
use tch::Tensor;

#[cfg(test)]
use thoosa_shared::ThoError;
use thoosa_shared::traits::Named;
use crate::dims::Dims;
use crate::tho_tensor::ThoTensor;

#[cfg(test)]
#[derive(Debug)]
struct ArgminMathError {
    context: &'static str,
    source: argmin_math::Error,
}

#[cfg(test)]
impl ArgminMathError {
    fn new(context: &'static str, source: argmin_math::Error) -> Self {
        Self { context, source }
    }
}

#[cfg(test)]
impl std::fmt::Display for ArgminMathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.context, self.source)
    }
}

#[cfg(test)]
impl std::error::Error for ArgminMathError {}

//TODO remove
impl <D, Id> Clone for ThoTensor<D, Id>
where
 D: Dims,
 Id: Named + Debug + Clone
{
    fn clone(&self) -> Self {
        ThoTensor::new(self.id().clone(), self.t_ref().shallow_clone()).expect("can't clone")
    }
}

impl ThoTensor {
    pub fn force_requires_grad(&self) {
        unsafe {
            let const_ptr = self as *const Self;
            let mut_ptr = const_ptr as *mut Self;
            (*mut_ptr).t = self.t.shallow_clone().set_requires_grad(true);
        }
    }

}


impl Into<Tensor> for ThoTensor {
    fn into(self) -> Tensor {
        self.t
    }
}

impl Deref for ThoTensor {
    type Target = Tensor;

    fn deref(&self) -> &Self::Target {
        &self.t
    }
}

impl DerefMut for ThoTensor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.t
    }
}

impl ArgminSub<ThoTensor, ThoTensor> for ThoTensor {
    fn sub(&self, other: &ThoTensor) -> ThoTensor {
        (&self.t - &other.t).into()
    }
}

impl ArgminSub<f64, ThoTensor> for ThoTensor {
    fn sub(&self, other: &f64) -> ThoTensor {
        (self.t.subtract_scalar(*other)).into()
    }
}

impl ArgminAdd<ThoTensor, ThoTensor> for ThoTensor {
    fn add(&self, other: &ThoTensor) -> ThoTensor {
        (&self.t + &other.t).into()
    }
}

impl ArgminAdd<f64, ThoTensor> for ThoTensor {
    fn add(&self, other: &f64) -> ThoTensor {
        (self.t.f_add_scalar(*other)).unwrap().into()
    }
}

impl ArgminDot<ThoTensor, f64> for ThoTensor {
    fn dot(&self, other: &ThoTensor) -> f64 {
        debug!("ArgminDot::dot: size={:?} self.t={}", self.t.size(), self.t);
        debug!(
            "ArgminDot::dot: size={:?} other.t={}",
            other.t.size(),
            other.t
        );
        let v = self.t.dot(&other.t).double_value(&[]);
        debug!("ArgminDot::dot: v={}", v);
        v
    }
}

impl ArgminMul<f64, ThoTensor> for ThoTensor {
    fn mul(&self, other: &f64) -> ThoTensor {
        self.t.multiply_scalar(*other).into()
    }
}

impl ArgminMul<ThoTensor, ThoTensor> for ThoTensor {
    fn mul(&self, other: &ThoTensor) -> ThoTensor {
        self.t.multiply(other).into()
    }
}

impl ArgminL1Norm<f64> for ThoTensor {
    fn l1_norm(&self) -> f64 {
        self.t.linalg_norm(0, 0, true, None).double_value(&[])
    }
}

impl ArgminSignum for ThoTensor {
    fn signum(self) -> Self {
        self.t.sign().into()
    }
}

impl ArgminZeroLike for ThoTensor {
    fn zero_like(&self) -> Self {
        Tensor::zeros(self.t.size(), (self.t.kind(), self.t.device())).into()
    }
}

//TODO test
impl ArgminMinMax for ThoTensor {
    fn min(x: &Self, y: &Self) -> Self {
        Tensor::stack(&[&x.t, &y.t], 0).amin(1, false).into()
    }
    fn max(x: &Self, y: &Self) -> Self {
        Tensor::stack(&[&x.t, &y.t], 0).amax(1, false).into()
    }
}

impl ArgminL2Norm<f64> for ThoTensor {
    fn l2_norm(&self) -> f64 {
        self.t.linalg_norm(2, 0, true, None).double_value(&[])
    }
}

impl From<Tensor> for ThoTensor {
    fn from(value: Tensor) -> Self {
        // debug!("new: {:?} device={:?} requires_grad={}", crate::utils::from::<Vec<f32>>(&value), value.device(), value.requires_grad());
        ThoTensor::new((), value).unwrap()
    }
}

impl ArgminMul<ThoTensor, ThoTensor> for f64 {
    fn mul(&self, other: &ThoTensor) -> ThoTensor {
        other.t.multiply_scalar(*self).into()
    }
}

// Impl of ArgminRandom removed to avoid rand trait version conflicts (not required by our usage)

pub(crate) struct TensorProblem<F>
where
    F: Fn(&ThoTensor) -> Tensor,
{
    loss_fn: F,
}

impl<F> TensorProblem<F>
where
    F: Fn(&ThoTensor) -> Tensor,
{
    #[allow(dead_code)]
    pub fn new(loss_fn: F) -> Self {
        Self { loss_fn }
    }
}

impl<F> CostFunction for TensorProblem<F>
where
    F: Fn(&ThoTensor) -> Tensor,
{
    type Param = ThoTensor;
    type Output = f64;

    fn cost(&self, theta: &ThoTensor) -> Result<Self::Output, Error> {
        // debug!("cost: theta={:?} device={:?} requires_grad={:?}", theta, theta.device(), theta.requires_grad());

        Ok(tch::with_grad(|| {
            theta.force_requires_grad();

            let loss = (self.loss_fn)(theta).set_requires_grad(true);

            loss.backward();

            debug!(
                "cost:  theta.grad={:?} loss={} loss dv={}",
                theta.grad(),
                loss,
                loss.double_value(&[])
            );

            loss.double_value(&[])
        }))
    }
}

impl<F> Gradient for TensorProblem<F>
where
    F: Fn(&ThoTensor) -> Tensor,
{
    type Param = ThoTensor;
    type Gradient = ThoTensor;

    fn gradient(&self, p: &ThoTensor) -> Result<Self::Gradient, Error> {
        debug!(
            "gradient: p={:?} device={:?} requires_grad={:?}",
            p,
            p.device_idx(),
            p.requires_grad()
        );
        let grad = p.grad(); //.to_device(Cpu);
        debug!("gradient: grad={}", grad);
        Ok(grad.into())
    }
}
#[cfg(test)]
pub fn run_lbfgs<F>(
    cost: TensorProblem<F>,
    init: Tensor,
    target_cost: f64,
    max_iters: u64,
) -> Result<
    OptimizationResult<
        TensorProblem<F>,
        LBFGS<MoreThuenteLineSearch<ThoTensor, ThoTensor, f64>, ThoTensor, ThoTensor, f64>,
        IterState<ThoTensor, ThoTensor, (), (), (), f64>,
    >,
    ThoError,
>
where
    F: Fn(&ThoTensor) -> Tensor,
{
    let solver = LBFGS::new(
        MoreThuenteLineSearch::new(), //.with_c(1e-4, 0.9)?, //.with_bounds(1., 10.)?,
        10,
    )
    .with_tolerance_cost(1e-6)
    .map_err(|e| ArgminMathError::new("lbfgs with_tolerance_cost", e))?; //.with_l1_regularization(1.0)?;

    let executor = Executor::new(cost, solver).configure(|state| {
        state
            .param(init.set_requires_grad(true).into())
            .max_iters(max_iters)
            .target_cost(target_cost)
    });

    Ok(tch::no_grad(|| executor.run().unwrap()))
}

#[cfg(test)]
mod tests {
    use argmin::core::CostFunction;
    use argmin::core::Error;
    use argmin::core::Executor;
    use argmin::core::State;
    use argmin::core::*;
    use argmin::solver::linesearch::MoreThuenteLineSearch;
    use argmin::solver::quasinewton::LBFGS;
    use argmin_math::ArgminMul;
    use log::debug;
    use tch::Device::Cpu;
    use tch::Tensor;

    use thoosa_shared::ThoError;

    use crate::argmin::{ArgminMathError, ThoTensor, TensorProblem, run_lbfgs};
    use crate::opt::rbf_nll;

    #[test]
    fn test_min_x2() -> Result<(), ThoError> {
        struct MyProblem {}

        impl CostFunction for MyProblem {
            type Param = ThoTensor;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                debug!(
                    "cost: p={:?} device={:?} requires_grad={:?}",
                    p,
                    p.device_idx(),
                    p.requires_grad()
                );

                Ok(tch::with_grad(|| {
                    p.force_requires_grad();

                    let x_2 = p.f_add_scalar(-10.).unwrap().pow_tensor_scalar(2);
                    let x_2 = x_2.shallow_clone() + x_2.sin().multiply_scalar(1.01);

                    x_2.backward();

                    debug!("cost:  p.grad={:?}", p.grad());

                    x_2.double_value(&[])
                }))
            }
        }

        impl Gradient for MyProblem {
            type Param = ThoTensor;
            type Gradient = ThoTensor;

            fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
                debug!(
                    "gradient: p={:?} device={:?} requires_grad={:?}",
                    p,
                    p.device_idx(),
                    p.requires_grad()
                );
                let grad = p.grad(); //.to_device(Cpu);
                debug!("gradient: grad={:?}", grad);
                Ok(grad.into())
            }
        }

        let cost = MyProblem {};

        let linesearch = MoreThuenteLineSearch::new()
            .with_c(1e-4, 0.9)
            .map_err(|e| ArgminMathError::new("linesearch with_c", e))?;
        let solver = LBFGS::new(linesearch, 10);

        // let solver = BrentOpt::new(-100., 100.);
        // let solver = GaussNewton::new();
        // let solver = Landweber::new(0.001);
        // let solver = ParticleSwarm::new((MyTensor(Tensor::from_slice(&[-100.])), MyTensor(Tensor::from_slice(&[100.]))), 40);

        let executor = Executor::new(cost, solver).configure(|state| {
            state
                .param(
                    Tensor::from_slice(&[-100.])
                        .to_device(Cpu)
                        .set_requires_grad(true)
                        .into(),
                )
                .max_iters(10)
                .target_cost(0.0)
        });

        let res = tch::no_grad(|| executor.run().unwrap());

        debug!(
            "res={:?}",
            res.state().get_best_param().unwrap().double_value(&[])
        );

        assert_eq!(
            10.0,
            res.state().get_best_param().unwrap().double_value(&[])
        );

        Ok(())
    }

    #[test]
    fn test_min_rbf() -> Result<(), ThoError> {
        let res = run_lbfgs(
            TensorProblem::new(|theta| {
                rbf_nll(
                    &Tensor::from_slice(&[-4., -3., -2., -1., 1.]).view((-1, 1)),
                    &Tensor::from_slice(&[0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.])
                        .view((-1, 1)),
                    theta,
                    1e-8,
                )
                .expect("nll")
            }),
            Tensor::from_slice(&[1., 1.]).to_device(Cpu),
            0.0,
            100,
        )?;

        debug!("res={:?}", res.state().get_best_param());

        assert!(Tensor::from_slice(&[1.53, 0.74]).allclose(
            res.state().get_best_param().unwrap(),
            0.1,
            0.1,
            false
        ));

        Ok(())
    }

    #[test]
    fn test_linesearch() -> Result<(), ThoError> {
        struct MyProblem {}

        impl CostFunction for MyProblem {
            type Param = Vec<f64>;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                debug!("cost: p={:?}", p);
                let x_2 = (p[0] - 10.).powi(2);
                // Ok(x_2)
                Ok(x_2 + 2. * x_2.sin())

                // let x_10 = p.f_add_scalar(-5.).unwrap().pow_tensor_scalar(2);
                // let v = x_10.shallow_clone() + x_10.sin().multiply_scalar(2.);
                // Ok(v.double_value(&[]))
            }
        }

        impl Gradient for MyProblem {
            type Param = Vec<f64>;
            type Gradient = Vec<f64>;

            fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
                let grad = vec![2. * p[0] - 20., 0.];
                debug!("p={:?} grad={:?}", p, grad);
                Ok(grad)
            }
        }

        let cost = MyProblem {};

        // let solver = BrentOpt::new(vec![-100., 0.], vec![100., 0.]);
        // let solver = GaussNewton::new();
        // let solver = Landweber::new(0.001);

        let linesearch = MoreThuenteLineSearch::new()
            .with_c(1e-4, 0.9)
            .map_err(|e| ArgminMathError::new("linesearch with_c", e))?;

        // Set up solver
        let solver = LBFGS::new(linesearch, 7);

        let res = Executor::new(cost, solver)
            .configure(|state| state.param(vec![-100., 0.]).max_iters(10).target_cost(0.0))
            .run()
            .map_err(|e| ArgminMathError::new("linesearch run", e))?;

        let best_param =res.state().get_best_param().unwrap();

        assert_eq!(
            [10.0, 0.0],
            best_param.as_slice()
        );

        Ok(())
    }

    #[test]
    fn test_grad() -> Result<(), ThoError>{
        let p_1 = ThoTensor::new((), Tensor::from_slice(&[10.]).set_requires_grad(true))?;

        let t2 = ThoTensor::new((), p_1.pow_tensor_scalar(2))?;

        t2.backward();

        debug!("1. grad t1={:?} t1.grad={:?}", p_1, p_1.grad());

        let t2 = p_1.pow_tensor_scalar(2);

        t2.backward();

        debug!("2. grad t1={:?} t1.grad={:?}", p_1, p_1.grad());

        // let p_2:MyTensor = p_1.mul(&MyTensor(Tensor::from_slice(&[1., 2., 3.,]).set_requires_grad(true)));
        let p_2 = tch::no_grad(|| p_1.mul(&0.1));
        let _ = p_2.set_requires_grad(true);
        let t2 = p_2.pow_tensor_scalar(2);

        // p_2.zero_grad();
        t2.backward();

        debug!("3. grad t1={:?} t1.grad={:?}", p_2, p_2.grad());

        Ok(())
    }
}
