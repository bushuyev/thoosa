

#[cfg(test)]
mod tests {
    use std::ops::{Deref, DerefMut};
    use argmin::solver::brent::BrentOpt;
    use hbac_shared::HbacError;
    use argmin::core::{Executor, Operator};
    use argmin::core::CostFunction;
    use argmin::core::Error;
    use argmin::core::Jacobian;
    use log::debug;
    use argmin::core::State;
    use tch::Kind::Float;
    use tch::Tensor;
    use argmin::solver::gaussnewton::GaussNewton;
    use argmin::solver::landweber::Landweber;
    use argmin::solver::linesearch::MoreThuenteLineSearch;
    use argmin::solver::quasinewton::LBFGS;
    use argmin::solver::particleswarm::ParticleSwarm;
    use argmin::core::*;
    use argmin_math::{ArgminAdd, ArgminDot, ArgminL1Norm, ArgminL2Norm, ArgminMinMax, ArgminMul, ArgminRandom, ArgminSignum, ArgminSub, ArgminZeroLike};
    use tch::Device::Cpu;
    use argmin_math::ArgminScaledAdd;
    use rand::Rng;

    #[derive(Debug)]
    struct ArgmTensor(Tensor);

    impl ArgmTensor {
        pub fn force_requires_grad(&self) {
            unsafe  {
                let const_ptr = self as *const Self;
                let mut_ptr = const_ptr as *mut Self;
                (*mut_ptr).0 = self.0.shallow_clone().set_requires_grad(true);
            }
        }
    }

    impl Clone for ArgmTensor {
        fn clone(&self) -> Self{
            debug!("clone: self device={:?} requires_grad={:?} grad={:?}", self.device(), self.requires_grad(), self.grad());
            // let tensor = crate::utils::clone(&self.0); // self.shallow_clone();
            let tensor = self.shallow_clone();

            debug!("clone: tensor device={:?} requires_grad={:?} grad={:?}", tensor.device(), tensor.requires_grad(), tensor.grad());
            ArgmTensor(tensor)
        }
    }

    impl Deref for ArgmTensor {
        type Target = Tensor;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl DerefMut for ArgmTensor {

        fn deref_mut(&mut self) -> &mut Self::Target {
            &mut self.0
        }
    }

    impl ArgminSub<ArgmTensor, ArgmTensor> for ArgmTensor {
        fn sub(&self, other: &ArgmTensor) -> ArgmTensor {
            (&self.0 - &other.0).into()
        }
    }

    impl ArgminSub<f64, ArgmTensor> for ArgmTensor {
        fn sub(&self, other: &f64) -> ArgmTensor {
            (self.0.subtract_scalar(*other)).into()
        }
    }

    impl ArgminAdd<ArgmTensor, ArgmTensor> for ArgmTensor {
        fn add(&self, other: &ArgmTensor) -> ArgmTensor {
            (&self.0 + &other.0).into()
        }
    }

    impl ArgminAdd<f64, ArgmTensor> for ArgmTensor {
        fn add(&self, other: &f64) -> ArgmTensor {
            (self.0.f_add_scalar(*other)).unwrap().into()
        }
    }

    impl ArgminDot<ArgmTensor, f64> for ArgmTensor {
        fn dot(&self, other: &ArgmTensor) ->f64  {
            self.0.dot(&other.0).double_value(&[])
        }
    }

    impl ArgminMul<f64, ArgmTensor> for ArgmTensor {
        fn mul(&self, other: &f64) -> ArgmTensor {
            self.0.multiply_scalar(*other).into()
        }
    }

    impl ArgminMul<ArgmTensor, ArgmTensor> for ArgmTensor {
        fn mul(&self, other: &ArgmTensor) -> ArgmTensor {
            self.0.multiply(other).into()
        }
    }

    impl ArgminL1Norm<f64> for ArgmTensor {
        fn l1_norm(&self) -> f64 {
            self.0.linalg_norm(0, (0), true, None).double_value(&[])
        }
    }

    impl ArgminSignum for ArgmTensor {
        fn signum(self) -> Self {
            self.0.sign().into()
        }
    }

    impl ArgminZeroLike for ArgmTensor {
        fn zero_like(&self) -> Self {
            Tensor::zeros(self.0.size(), (self.0.kind(), self.0.device())).into()
        }
    }

    //TODO test
    impl ArgminMinMax for ArgmTensor {
        fn min(x: &Self, y: &Self) -> Self {
            Tensor::stack(&[&x.0, &y.0], 0).amin(1, false).into()
        }
        fn max(x: &Self, y: &Self) -> Self {
            Tensor::stack(&[&x.0, &y.0], 0).amax(1, false).into()
        }
    }

    impl ArgminL2Norm<f64> for ArgmTensor {
        fn l2_norm(&self) -> f64 {
            self.0.linalg_norm(2, (0), true, None).double_value(&[])
        }
    }


    impl From<Tensor> for ArgmTensor {
        fn from(value: Tensor) -> Self {
            debug!("new: {:?} device={:?} requires_grad={}", crate::utils::from::<Vec<f32>>(&value), value.device(), value.requires_grad());
            ArgmTensor(value)
        }
    }

    impl ArgminMul<ArgmTensor, ArgmTensor> for f64 {
        fn mul(&self, other: &ArgmTensor) -> ArgmTensor {
            other.0.multiply_scalar(*self).into()
        }
    }

    impl ArgminRandom for ArgmTensor {
        fn rand_from_range<R: Rng>(min: &Self, max: &Self, rng: &mut R) -> Self {
            ((&max.0 - &min.0).multiply(&Tensor::randn(min.size(), (min.kind(), min.device()))) + &max.0).into()
        }
    }


    #[test]
    fn test_min_x2() -> Result<(), HbacError>{
        struct MyProblem {}

        impl CostFunction for MyProblem {
            type Param = ArgmTensor;
            type Output = f64;

            fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
                debug!("cost: p={:?} device={:?} requires_grad={:?}", p, p.device(), p.requires_grad());

                Ok(tch::with_grad(|| {
                    p.force_requires_grad();

                    let x_2 = p.f_add_scalar(-10.).unwrap().pow_tensor_scalar(2);
                    let x_2 = x_2.shallow_clone() + x_2.sin().multiply_scalar(1.01);


                    x_2.backward();

                    debug!("cost:  p.grad={:?}",  p.grad());

                    x_2.double_value(&[])
                }))
            }
        }

        impl Gradient for MyProblem {
            type Param = ArgmTensor;
            type Gradient = ArgmTensor;

            fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
                debug!("gradient: p={:?} device={:?} requires_grad={:?}", p, p.device(), p.requires_grad());
                let grad = p.grad(); //.to_device(Cpu);
                debug!("gradient: grad={:?}", grad);
                Ok(grad.into())

            }
        }

        let cost = MyProblem {};



        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;
        let solver = LBFGS::new(linesearch, 10);


        // let solver = BrentOpt::new(-100., 100.);
        // let solver = GaussNewton::new();
        // let solver = Landweber::new(0.001);
        // let solver = ParticleSwarm::new((MyTensor(Tensor::from_slice(&[-100.])), MyTensor(Tensor::from_slice(&[100.]))), 40);


        let executor = Executor::new(cost, solver)
            .configure(|state|
                state
                    .param(Tensor::from_slice(&[-100.]).to_device(Cpu).set_requires_grad(true).into())
                    .max_iters(10)
                    .target_cost(0.0)
            );

        let res = tch::no_grad(|| {executor.run().unwrap()});

        debug!("res={:?}", res.state().get_best_param().unwrap().double_value(&[]));

        assert_eq!(10.0, res.state().get_best_param().unwrap().double_value(&[]));

        Ok(())
    }


    #[test]
    fn test_min_rbf() -> Result<(), HbacError>{
        struct RbfNllProblem {
            x_train:Tensor,
            y_train:Tensor
        }

        impl CostFunction for RbfNllProblem {
            type Param = ArgmTensor;
            type Output = f64;

            fn cost(&self, theta: &Self::Param) -> Result<Self::Output, Error> {
                debug!("cost: theta={:?} device={:?} requires_grad={:?}", theta, theta.device(), theta.requires_grad());

                Ok(tch::with_grad(|| {
                    theta.force_requires_grad();

                    let loss = crate::opt::rbf_nll(&self.x_train, &self.y_train, theta, 0.1_f64).expect("nll");

                    loss.backward();

                    debug!("cost:  theta.grad={:?} loss={:?}",  theta.grad(), loss);

                    loss.double_value(&[])
                }))
            }
        }

        impl Gradient for RbfNllProblem {
            type Param = ArgmTensor;
            type Gradient = ArgmTensor;

            fn gradient(&self, p: &Self::Param) -> Result<Self::Gradient, Error> {
                debug!("gradient: p={:?} device={:?} requires_grad={:?}", p, p.device(), p.requires_grad());
                let grad = p.grad(); //.to_device(Cpu);
                debug!("gradient: grad={:?}", grad);
                Ok(grad.into())

            }
        }

        let cost = RbfNllProblem {
            x_train: Tensor::from_slice(&[
                -4., -3., -2., -1., 1.
            ]).view((-1, 1)),
            y_train: Tensor::from_slice(&[
                0.7568025, -0.14112001, -0.90929743, -0.84147098, 0.
            ]).view((-1, 1))
        };



        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;
        let solver = LBFGS::new(linesearch, 10);


        let executor = Executor::new(cost, solver)
            .configure(|state|
                state
                    .param(Tensor::from_slice(&[10., 10.]).to_device(Cpu).set_requires_grad(true).into())
                    .max_iters(10)
                    .target_cost(0.0)
            );

        let res = tch::no_grad(|| {executor.run().unwrap()});

        debug!("res={:?}", res.state().get_best_param());

        assert!(Tensor::from_slice(&[1.53, 0.74]).allclose(res.state().get_best_param().unwrap(), 0.1, 0.1, false));

        Ok(())
    }


    #[test]
    fn test_test() -> Result<(), HbacError>{
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
                println!("p={:?} grad={:?}", p, grad);
                Ok(grad)
            }
        }

        let cost = MyProblem {

        };


        // let solver = BrentOpt::new(vec![-100., 0.], vec![100., 0.]);
        // let solver = GaussNewton::new();
        // let solver = Landweber::new(0.001);

        let linesearch = MoreThuenteLineSearch::new().with_c(1e-4, 0.9)?;

        // Set up solver
        let solver = LBFGS::new(linesearch, 7);


        let res = Executor::new(cost, solver)
            .configure(|state|
                state
                    .param(vec![-100., 0.])
                    .max_iters(10)
                    .target_cost(0.0)
            )
            .run()?;

        debug!("res={:?}", res.state().get_best_param().unwrap());

        Ok(())
    }

    #[test]
    fn test_grad(){
        let mut p_1 = ArgmTensor(Tensor::from_slice(&[10.]).set_requires_grad(true));

        let t2 = ArgmTensor(p_1.pow_tensor_scalar(2));

        t2.backward();

        debug!("1. grad t1={:?} t1.grad={:?}", p_1, p_1.grad());

        let t2 = p_1.pow_tensor_scalar(2);

        t2.backward();

        debug!("2. grad t1={:?} t1.grad={:?}", p_1, p_1.grad());


        // let p_2:MyTensor = p_1.mul(&MyTensor(Tensor::from_slice(&[1., 2., 3.,]).set_requires_grad(true)));
        let mut p_2 = tch::no_grad(|| { p_1.mul(&0.1)});
        p_2.set_requires_grad(true);
        let t2 = p_2.pow_tensor_scalar(2);

        // p_2.zero_grad();
        t2.backward();

        debug!("3. grad t1={:?} t1.grad={:?}", p_2, p_2.grad());
    }

}