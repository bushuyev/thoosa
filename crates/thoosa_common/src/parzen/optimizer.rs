use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use log::{debug, trace};
use rand::distributions::Uniform;
use rand::Rng;
use tch::{IndexOp, Tensor};
use thoosa_shared::{RangeOrValue, ThoosaError};
use thoosa_shared::hyperparameter::{Hyperparameter, FieldName};
use thoosa_shared::val::{NamedVal, Val};
use crate::argmin::ArgmTensor;
use crate::parzen::cpe::CategoricalParzenEstimator;
use crate::parzen::mvpe::MultiVariateParzenEstimator;
use crate::parzen::npe::{BandwidthFactor, NumericalParzenEstimator};
use thoosa_shared::opt_rng::opt_rnd;



#[allow(dead_code)]
#[derive(Debug, Clone)]
struct RunResult{
    loss:f32,
    eval_config:Vec<Val>
}

impl RunResult {
    pub fn new(loss: f32, eval_config: Vec<Val>) -> Self {
        Self { loss, eval_config }
    }
}

impl PartialEq for RunResult {
    fn eq(&self, other: &Self) -> bool {
        self.loss.eq(&other.loss)
    }
}
impl Eq for RunResult {}

impl PartialOrd for RunResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.loss.partial_cmp(&other.loss)
    }
}

impl Ord for RunResult {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(&other).unwrap()
    }
}

#[cfg(test)]
mod run_result_tests {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    use crate::parzen::optimizer::RunResult;

    #[test]
    fn test_0(){
        let mut observations = BinaryHeap::new();

        observations.push(Reverse(RunResult::new(10., vec![])));
        observations.push(Reverse(RunResult::new(1., vec![])));
        observations.push(Reverse(RunResult::new(20., vec![])));
        observations.push(Reverse(RunResult::new(2., vec![])));


        let mut observations_c = observations.clone();
        let mut observations_vec = vec![];
        while let Some(e) = observations_c.pop() {

            observations_vec.push(e);
        }

        // debug!("observations_vec={observations_vec:?}");
        assert_eq!(1., observations_vec[0].0.loss);
        assert_eq!(2., observations_vec[1].0.loss);
        assert_eq!(10., observations_vec[2].0.loss);
        assert_eq!(20., observations_vec[3].0.loss);
    }
}
#[allow(dead_code)]
pub trait ParzenEstimator:Send{
    fn clc_distr(&self, samples:Tensor, weights: Option<Tensor>) -> Box<dyn Distribution>;
    fn clc_distr_from_vec(&self, samples:&mut dyn Iterator<Item=Val>, weights: Option<Tensor>) -> Box<dyn Distribution>;
}

pub trait Distribution:Debug+Send {
    fn sample(&self, n_samples:usize) -> Vec<Val>;
    fn basis_loglikelihood(&self, x:&mut dyn Iterator<Item=&Val>) -> Tensor;
    fn weights(&self)->&Tensor;
}

struct TreeStructuredParzenEstimator {

    observations:BinaryHeap<Reverse<RunResult>>,
    mvpe_lower:Option<MultiVariateParzenEstimator>, //TODO move out
    mvpe_upper:Option<MultiVariateParzenEstimator>, //TODO move out
    percentile:f32, //TODO move out
    esitmators:Vec<Box<dyn ParzenEstimator>>
}

impl TreeStructuredParzenEstimator {

    pub fn new<T: FieldName>(cs:&ConfigurationSpace<T>) -> Self {
        Self {
            observations:BinaryHeap::new(),
            mvpe_lower:None,
            mvpe_upper:None,
            percentile: 0.,
            esitmators: cs.iter().map(|hp|
                match hp {

                    Hyperparameter::UniformFloat { name, lower, upper } => {
                        Box::new(NumericalParzenEstimator::new(
                            Some(name.to_string()),
                            // Tensor::zeros(1, FLOAT_CPU),
                            *lower,
                            *upper,
                            BandwidthFactor::default(),
                            true,
                            None
                        )) as Box<dyn ParzenEstimator>

                    }
                    Hyperparameter::UniformInt { name, lower, upper } => {
                        Box::new(NumericalParzenEstimator::new(
                            Some(name.to_string()),
                            // Tensor::zeros(1, FLOAT_CPU),
                            *lower as f64, //TODO
                            *upper as f64,
                            BandwidthFactor::default(),
                            true,
                            Some(1)//TODO step
                        )) as Box<dyn ParzenEstimator>
                    }

                    Hyperparameter::Categorical { name, values } => {
                        Box::new(CategoricalParzenEstimator::new(
                            Some(name.to_string()),
                            values.len(),
                            true
                        )) as Box<dyn ParzenEstimator>
                    }


                }
            ).collect()
        }
    }

    pub fn update_observations(&mut self, loss:f32, eval_config:&Vec<Val>) {
        self.observations.push(Reverse(RunResult::new(loss, eval_config.clone())));//TODO clone

        let n_lower = Self::percentile(self.observations.len()) as usize;

        debug!("n_lower={n_lower}");
        trace!("self.observations={:?}", self.observations);

        self.percentile = n_lower as f32 / self.observations.len() as f32; //[self._metric_name].size


        let (lower_vals, upper_vals) = self.pop_observations(n_lower);

        let (lower_esitmators, upper_esitmators) = self.esitmators.iter().enumerate().fold((vec![], vec![]), |mut lower_upper_esitmators, (param_i, pe)|{
            lower_upper_esitmators.0.push(
                pe.clc_distr_from_vec(&mut lower_vals.iter().map(|v|v[param_i].clone()), None)
            );
            lower_upper_esitmators.1.push(
                pe.clc_distr_from_vec(&mut upper_vals.iter().map(|v|v[param_i].clone()), None)
            );

            lower_upper_esitmators
        });

        self.mvpe_lower.replace(MultiVariateParzenEstimator::new(lower_esitmators));
        self.mvpe_upper.replace(MultiVariateParzenEstimator::new(upper_esitmators));

        trace!("self.mvpe_lower={:?}", self.mvpe_lower);
        trace!("self.self.mvpe_upper={:?}", self.mvpe_upper);
    }

    fn pop_observations(&mut self, n_lower: usize) -> (Vec<Vec<Val>>, Vec<Vec<Val>>) {
        let mut observations_c = self.observations.clone();
        let mut lower_vals = Vec::with_capacity(n_lower);
        let mut upper_vals = Vec::with_capacity(observations_c.len() - n_lower);

        while let Some(e) = observations_c.pop() {
            // debug!(">>>v.0.eval_config={:?}", o);
            if lower_vals.len() < n_lower {
                lower_vals.push(e.0.eval_config);
            } else {
                upper_vals.push(e.0.eval_config);
            }
        }
        (lower_vals, upper_vals)
    }

    fn percentile(size:usize) -> f32 {
        (0.25 * (size as f32).sqrt()).ceil()
    }

    /// variables/samples
    fn get_config_candidates(&self, n_samples: usize) -> Vec<Vec<Val>> {
        self.mvpe_lower.as_ref().map(|mvpe_lower|mvpe_lower.sample(true, n_samples)).unwrap_or(vec![])
    }

    fn compute_config_loglikelihoods(&self, config_cands:&Vec<Vec<Val>>) -> (Tensor, Tensor) {
        let config_ll_lower = self.mvpe_lower.as_ref().expect("mvpe_lower").log_pdf(config_cands);
        let config_ll_upper = self.mvpe_upper.as_ref().expect("mvpe_upper").log_pdf(config_cands);
        (config_ll_lower, config_ll_upper)
    }

    fn compute_probability_improvement(&self, config_cands:&Vec<Vec<Val>> ) -> Tensor{

        let eps = 1e-12;
        let (cll_lower, cll_upper) = self.compute_config_loglikelihoods(config_cands);
        let first_term = (self.percentile + eps).ln();
        let second_term = (1.0 - self.percentile + eps).ln() + cll_upper - cll_lower;
        second_term.logaddexp(&first_term.into()).negative()

    }
}

#[cfg(test)]
mod tree_structured_parzen_estimator_tests {
    use log::debug;
    use tch::{Kind, Tensor};
    use thoosa_shared::hyperparameter::Hyperparameter;
    use crate::parzen::mvpe::MultiVariateParzenEstimator;
    use crate::parzen::npe::{BandwidthFactor, NumericalParzenEstimator};
    use crate::parzen::optimizer::{ConfigurationSpace, ParzenEstimator, TreeStructuredParzenEstimator, Val};
    use crate::parzen::optimizer::Val::Real;

    #[test]
    fn test_percentile(){
        assert_eq!(1., TreeStructuredParzenEstimator::percentile(1));
        assert_eq!(1., TreeStructuredParzenEstimator::percentile(10));
        assert_eq!(3., TreeStructuredParzenEstimator::percentile(100));
    }


    #[test]
    fn test_configuration_space(){

        let cs:ConfigurationSpace<String> = ConfigurationSpace::new(vec![
            Hyperparameter::UniformFloat {name:"x_0".into(), lower: -1.0, upper: 1.0},
            Hyperparameter::UniformFloat {name:"x_1".into(), lower: -2.0, upper: 2.0},
        ]);

        let mut tree_structured_parzen_estimator = TreeStructuredParzenEstimator::new(&cs);

        tree_structured_parzen_estimator.update_observations(0.4, &vec![Val::Real(0.4), Val::Real(1.4)]);
        tree_structured_parzen_estimator.update_observations(0.1, &vec![Val::Real(0.1), Val::Real(1.1)]);
        tree_structured_parzen_estimator.update_observations(0.3, &vec![Val::Real(0.3), Val::Real(1.3)]);
        tree_structured_parzen_estimator.update_observations(0.2, &vec![Val::Real(0.2), Val::Real(1.2)]);

        // debug!("{:?}", tree_structured_parzen_estimator.observations);
        
        assert_eq!(0.1, tree_structured_parzen_estimator.observations.pop().expect("0.1").0.loss);
        assert_eq!(0.2, tree_structured_parzen_estimator.observations.pop().expect("0.2").0.loss);
        assert_eq!(0.3, tree_structured_parzen_estimator.observations.pop().expect("0.3").0.loss);
        assert_eq!(0.4, tree_structured_parzen_estimator.observations.pop().expect("0.4").0.loss);

        assert_eq!(2, tree_structured_parzen_estimator.mvpe_lower.expect("tree_structured_parzen_estimator").len())

    }

    #[test]
    fn test_compute_probability_improvement(){
        let cs:ConfigurationSpace<String> = ConfigurationSpace::new(vec![
            Hyperparameter::UniformFloat {name:"x_0".into(), lower: -3.0, upper: 3.0},
            Hyperparameter::UniformFloat {name:"x_1".into(), lower: -3.0, upper: 3.0},
            Hyperparameter::UniformFloat {name:"x_2".into(), lower: -3.0, upper: 3.0},
        ]);
        let mut tsp = TreeStructuredParzenEstimator::new(&cs);

        let samples = [-1.0, 0.0, 1.0];


        tsp.mvpe_lower = Some(MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(None, -3.0f64, 3.0f64, BandwidthFactor::default(), true, None).clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(None, -3.0f64, 3.0f64, BandwidthFactor::default(), true, None).clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(None, -3.0f64, 3.0f64, BandwidthFactor::default(), true, None).clc_distr(Tensor::from_slice(&samples), None)

        ]));
        let samples = [-2.0, 0.0, 2.0];

        tsp.mvpe_upper = Some(MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(None, -3.0f64, 3.0f64, BandwidthFactor::default(), true, None).clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(None, -3.0f64, 3.0f64, BandwidthFactor::default(), true, None).clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(None, -3.0f64, 3.0f64, BandwidthFactor::default(), true, None).clc_distr(Tensor::from_slice(&samples), None)

        ]));


        let pi = tsp.compute_probability_improvement(&vec![
            vec![Val::Real(0.0), Val::Real(1.0), Val::Real(2.0)], //x0
            vec![Val::Real(0.1), Val::Real(1.1), Val::Real(2.1)], //x1
            vec![Val::Real(0.2), Val::Real(1.2), Val::Real(2.2)]  //x2
        ]);

        debug!("pi={pi}");
        assert!(
            Tensor::from_slice(&[ 1.85448311,  4.59934198, -4.30722048]).totype(Kind::Float).allclose(&pi,  1e-02, 1e-02, false)
        );

    }
}
#[allow(dead_code)]
pub struct TPEOptimizer<F, T> where F: FnMut(&[NamedVal<T>]) -> f32, T: FieldName
{
    obj_func:F,
    cs:ConfigurationSpace<T>,
    n_init:u32,
    max_evals:u32,
    tpe_samplers:HashMap<String, TreeStructuredParzenEstimator>,
    _pd:PhantomData<T>
}

impl <F, T: FieldName> TPEOptimizer<F, T>
    where F: FnMut(&[NamedVal<T>]) -> f32 {

    pub fn new(obj_func:F, cs:ConfigurationSpace<T>, max_evals: u32, n_init: u32) -> Self {

        debug!("TPEOptimizer:new: cs={cs:?}");

        let estimator = TreeStructuredParzenEstimator::new(&cs);
        Self {
            obj_func,
            cs,
            n_init,
            max_evals,
            tpe_samplers: HashMap::from([
                ("loss".to_string(), estimator)
            ]),
            _pd:PhantomData::default()
        }
    }

    pub fn optimize(&mut self) -> Vec<Val> {
        let mut best_loss = f32::MAX;
        let mut best_config:Option<Vec<Val>> = None;

        (0..self.max_evals).for_each(|i|{

            let eval_config = if  i < self.n_init {
                debug!("TPEOptimizer::optimize: i={i} random sample");
                self.cs.random_sample()

            } else {
                debug!("TPEOptimizer::optimize: i={i} sample");
                self.sample()
            };

            let loss = (self.obj_func)(
                &eval_config.iter().zip(self.cs.hyperparameters.iter()).map(|(val, hp)|{
                    // (hp.name(), val.clone()).into()
                    // NamedVal::from((hp.name().into(), val.clone()))
                    NamedVal::from((hp.name().clone(), val.clone()))
                }).collect::<Vec<_>>().as_slice()
            );
            if loss < best_loss {
                best_loss = loss;
                best_config.replace(eval_config.clone());
            }
            debug!("TPEOptimizer::optimize: i={i} eval_config={:?} loss={loss}", eval_config);

            for (_key, tpe) in &mut self.tpe_samplers {
                tpe.update_observations(loss, &eval_config)
            }
        });

        best_config.unwrap_or(vec![])//TODO
    }

    fn sample(&self)->Vec<Val> {

        //TODO if always loss only - merge get_config_candidates & compute_probability_improvement
        let config_cands = stack_params(

            self.tpe_samplers.values().map(|tpe| {
                tpe.get_config_candidates(24)//TODO to param
            }).collect::<Vec<Vec<Vec<Val>>>>() //tpe(only loss)/params/candidates
        );

        //TODO weights? loss only
        let pi_config = Tensor::stack(
            self.tpe_samplers.values().map(|tpe| tpe.compute_probability_improvement(&config_cands)).collect::<Vec<Tensor>>().as_slice(),
            0,
        );

        trace!("sample: config_cands={config_cands:?}");
        trace!("sample: pi_config={pi_config}");

        debug!("sample: max={}", pi_config.max());

        let max_i = pi_config.argmax(1, true).i(0).double_value(&[]);

        config_cands.into_iter().map(|p|p[max_i as usize].clone()).collect()   // return weights @ pi_config_array

    }

}



#[allow(dead_code)]
pub struct AsyncTPEOptimizer<F, T> where F: AsyncFnMut(&[NamedVal<T>]) -> f32, T: FieldName
{
    obj_func:F,
    cs:ConfigurationSpace<T>,
    n_init:u32,
    max_evals:u32,
    tpe_samplers:HashMap<String, TreeStructuredParzenEstimator>,
    _pd:PhantomData<T>
}

impl <F, T: FieldName> AsyncTPEOptimizer<F, T>
where F: AsyncFnMut(&[NamedVal<T>]) -> f32 {

    pub fn new(obj_func:F, cs:ConfigurationSpace<T>, max_evals: u32, n_init: u32) -> Self {

        debug!("TPEOptimizer:new: cs={cs:?}");

        let estimator = TreeStructuredParzenEstimator::new(&cs);
        Self {
            obj_func,
            cs,
            n_init,
            max_evals,
            tpe_samplers: HashMap::from([
                ("loss".to_string(), estimator)
            ]),
            _pd:PhantomData::default()
        }
    }

    pub async fn optimize(&mut self) -> Vec<Val> {
        let mut best_loss = f32::MAX;
        let mut best_config:Option<Vec<Val>> = None;

        for i in (0..self.max_evals) {

            let eval_config = if  i < self.n_init {
                debug!("TPEOptimizer::optimize: i={i} random sample");
                self.cs.random_sample()

            } else {
                debug!("TPEOptimizer::optimize: i={i} sample");
                self.sample()
            };

            let loss = self.obj_func.async_call_mut((
                &eval_config.iter().zip(self.cs.hyperparameters.iter()).map(|(val, hp)|{
                    // (hp.name(), val.clone()).into()
                    // NamedVal::from((hp.name().into(), val.clone()))
                    NamedVal::from((hp.name().clone(), val.clone()))
                }).collect::<Vec<_>>().as_slice(),
            )).await;
            if loss < best_loss {
                best_loss = loss;
                best_config.replace(eval_config.clone());
            }
            debug!("TPEOptimizer::optimize: i={i} eval_config={:?} loss={loss}", eval_config);

            for (_key, tpe) in &mut self.tpe_samplers {
                tpe.update_observations(loss, &eval_config)
            }
        }

        best_config.unwrap_or(vec![])//TODO
    }

    fn sample(&self)->Vec<Val> {

        //TODO if always loss only - merge get_config_candidates & compute_probability_improvement
        let config_cands = stack_params(

            self.tpe_samplers.values().map(|tpe| {
                tpe.get_config_candidates(24)//TODO to param
            }).collect::<Vec<Vec<Vec<Val>>>>() //tpe(only loss)/params/candidates
        );

        //TODO weights? loss only
        let pi_config = Tensor::stack(
            self.tpe_samplers.values().map(|tpe| tpe.compute_probability_improvement(&config_cands)).collect::<Vec<Tensor>>().as_slice(),
            0,
        );

        trace!("sample: config_cands={config_cands:?}");
        trace!("sample: pi_config={pi_config}");

        debug!("sample: max={}", pi_config.max());

        let max_i = pi_config.argmax(1, true).i(0).double_value(&[]);

        config_cands.into_iter().map(|p|p[max_i as usize].clone()).collect()   // return weights @ pi_config_array

    }

}



fn stack_params<T:Clone>(input: Vec<Vec<Vec<T>>>) -> Vec<Vec<T>> {
    let num_rows = input[0].len();
    let mut result = vec![vec![]; num_rows];

    for params in input {
        for (i, sub_vec) in params.into_iter().enumerate() {
            result[i].extend(sub_vec);
        }
    }

    result
}

#[cfg(test)]
mod tpe_optimizer_tests {
    use log::debug;
    use tch::{IndexOp, Tensor};
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::val::NamedVal;
    use crate::argmin::ArgmTensor;
    use crate::parzen::optimizer::{stack_params, ConfigurationSpace, TPEOptimizer, Val};
    use crate::parzen::optimizer::Val::Int;


    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }


    #[test]
    fn test_params_stack(){
        let params_0 = vec![
            vec![1., 2., 3.],
            vec![10., 20., 30.],
        ];

        let params_1 = vec![
            vec![4., 5.,],
            vec![40., 50.],
        ];
        let r = stack_params(vec![params_0, params_1]);

        assert_eq!(
            vec![
                vec![1., 2., 3., 4., 5.],
                vec![10., 20., 30., 40., 50.],
            ],
            r
        );
        // debug!("r={r:?}");
    }

    #[test]
    fn test_sphere(){
        let sphere = sphere(30)(&vec![
            ("x".into(), Val::Real(0.1)).into(),
            ("x".into(), Val::Real(0.2)).into(),
            ("x".into(), Val::Real(0.3)).into(),
            ("x".into(), Val::Real(0.4)).into(),
            ("x".into(), Val::Real(0.5)).into(),
        ]);

        assert_eq!(4410.55, sphere);

    }

    #[test]
    fn test_optimize_numeric() {
        let mut opt = TPEOptimizer::new(
            sphere(300),
            ConfigurationSpace::new(
                (0..3).map(|i| Hyperparameter::UniformFloat { name: format!("x_{}", i).as_str().into(), lower: -500.0, upper: 500.0 }).collect()
            ),
            500,
            10,
        );

        let best_config = opt.optimize();

        debug!("best_config={best_config:?}");
        assert!((300. - <&Val as TryInto<f32>>::try_into((&best_config[0])).unwrap()).abs() < 3.);
        assert!((300. - <&Val as TryInto<f32>>::try_into((&best_config[1])).unwrap()).abs() < 3.);
        assert!((300. - <&Val as TryInto<f32>>::try_into((&best_config[2])).unwrap()).abs() < 3.);
    }


    #[test]
    fn test_optimize_numeric_int() {
        let mut opt = TPEOptimizer::new(
            sphere(300),
            ConfigurationSpace::new(
                (0..3).map(|i| Hyperparameter::UniformInt { name: format!("x_{}", i).as_str().into(), lower: -500, upper: 500 }).collect()
            ),
            500,
            10,
        );

        let best_config = opt.optimize();

        debug!("best_config={best_config:?}");
        // assert!((300. - <Primitives as Into<f32>>::into(best_config[0])).abs() < 3.);
        // assert!((300. - <Primitives as Into<f32>>::into(best_config[1])).abs() < 3.);
        // assert!((300. - <Primitives as Into<f32>>::into(best_config[2])).abs() < 3.);
    }


    #[test]
    fn test_optimize_ctaegoric(){

        let mut opt:TPEOptimizer<_, String> = TPEOptimizer::new(
            move |params| {
                debug!("params={params:?}");

                let x_0:f32 = (&params[0]).try_into().unwrap();
                let x_1:f32 = (&params[1]).try_into().unwrap();
                let x_2:u32 = (&params[2]).try_into().unwrap();

                let loss = (x_0 -30.).powi(2) + (x_1-30.).powi(2);
                let loss = match x_2 {
                    0 => loss * 3.,//TODO used to find min at cat 1 with 0=1.5 when upper tpe in update_observations was built from lower vals
                    1 => loss * 0.01,
                    2 => loss * 2.,
                    _ => {
                        panic!("unexpected category {x_2}")
                    }
                };

                // let loss = * 10_i32.pow(x_2) as f32;

                debug!("loss={loss:?}");

                loss
            },

            ConfigurationSpace::new(
                // (0..3).map(|i|Hyperparameter::UniformFloat {name:format!("x_{}", i), lower: -500.0, upper: 500.0}).collect()
                vec![
                    Hyperparameter::UniformFloat {name:"x_0".into(), lower: -50.0, upper: 50.0},
                    Hyperparameter::UniformFloat {name:"x_1".into(), lower: -50.0, upper: 50.0},
                    Hyperparameter::Categorical { name: "cat".into(), values: vec!["a".into(), "b".into(), "c".into()] }
                ]
            ),
            500,
            10
        );

        let best_config = opt.optimize();

        debug!("best_config={best_config:?}");
        assert!((30. - <&Val as TryInto<f32>>::try_into(&best_config[0]).unwrap()).abs() < 3.);
        assert!((30. - <&Val as TryInto<f32>>::try_into(&best_config[1]).unwrap()).abs() < 3.);
        assert_eq!(Int(1), best_config[2]);
    }


    fn sphere(d_x: i64) -> impl Fn(&[NamedVal<String>]) -> f32 {
        move |params| {
            debug!("params={params:?}");

            let t: Tensor = ArgmTensor(Tensor::from_slice(
                params.into_iter().map(|v|<&NamedVal<String> as TryInto<f64>>::try_into(v).unwrap()).collect::<Vec<f64>>().as_slice()
            )).into();
            let loss = t.subtract_scalar(d_x).pow_tensor_scalar(2).sum(None).double_value(&[]) as f32;

            debug!("loss={loss:?}");

            loss
        }
    }

    #[test]
    fn test_0(){
        let t = Tensor::stack(&[
                Tensor::from_slice(&[0.1, 1.1, 20.1, 3.1, 4.1]),
                // Tensor::from_slice(&[0.2, 1.2, 20.2, 3.2, 4.2])
            ], 0,
        );

        debug!("t={t} argmax={}", t.argmax(1, true).i(0).double_value(&[]));

    }
}

#[derive(Debug, Clone)]
pub struct ConfigurationSpace<T: FieldName> {
    hyperparameters:Vec<Hyperparameter<T>>
}

impl <T: FieldName> From<&[Hyperparameter<T>]> for ConfigurationSpace<T> {
    fn from(value: &[Hyperparameter<T>]) -> Self {
        ConfigurationSpace::new(value.to_vec())
    }
}

impl <T: FieldName> From<&ConfigurationSpace<T>> for HashMap<String, Val> {
    fn from(value: &ConfigurationSpace<T>) -> Self {
        let mut map = HashMap::new();
        map.insert("config_space".into(), Val::Int(1));

        for hp in &value.hyperparameters {
            match hp {
                Hyperparameter::UniformFloat { name, lower, upper } => {
                    map.insert(format!("lower_{}", name), (*lower).into());
                    map.insert(format!("upper_{}", name), (*upper).into());
                }
                Hyperparameter::UniformInt { name, lower, upper } => {
                    map.insert(format!("lower_{}", name), (*lower).into());
                    map.insert(format!("upper_{}", name), (*upper).into());
                }
                Hyperparameter::Categorical { name, values } => {
                    for (i, val) in values.iter().enumerate() {
                        map.insert(format!("{i}_{}", name), val.clone().into());
                    }
                }
            }
        }
        map
    }
}

#[allow(dead_code)]
impl <T: FieldName> ConfigurationSpace<T> {

    pub fn empty() -> Self {
        Self {
            hyperparameters: vec![]
        }
    }
    pub fn new(hyperparameters: Vec<Hyperparameter<T>>) -> Self {
        Self { hyperparameters }
    }

    pub fn random_sample(&self) ->Vec<Val> {
        self.hyperparameters.iter().map(|hp|hp.random_sample()).collect()
    }

    fn len(&self) ->usize {
        self.hyperparameters.len()
    }

    fn iter(&self) -> CsIter<T> {
        CsIter(self.hyperparameters.iter())
    }
}

pub struct CsIter<'a, T: FieldName>(std::slice::Iter<'a, Hyperparameter<T>>);

impl<'a, T: FieldName> Iterator for CsIter<'a, T> {
    type Item = &'a Hyperparameter<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a, T: FieldName> IntoIterator for &'a ConfigurationSpace<T> {
    type Item = &'a Hyperparameter<T>;
    type IntoIter = CsIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// impl TryFrom<&[LabelNodeType]> for Hyperparameter {
//     type Error = ();
//
//     fn try_from(value: &[LabelNodeType]) -> Result<Self, Self::Error> {
//         Ok(Hyperparameter::Categorical {
//             name: "LabelNodeType".to_string(),
//             values: value.into_iter().map(|v|v.to_string()).collect()
//         })
//     }
// }