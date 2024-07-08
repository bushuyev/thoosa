use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use log::{debug, trace};
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;
use tch::{IndexOp, Tensor};
use tch::kind::FLOAT_CPU;
use crate::argmin::ArgmTensor;
use crate::parzen::cpe::CategoricalParzenEstimator;
use crate::parzen::mvpe::MultiVariateParzenEstimator;
use crate::parzen::npe::{BandwidthFactor, NumericalParzenEstimator};
use super::opt_rng::opt_rnd;
// type TargetFn = dyn Fn(HashMap<String, Box<dyn num_traits::AsPrimitive<_>>>) -> f64;

#[derive(Debug, Copy, Clone)]
pub enum Primitives {
    Integer(i32),
    Long(i64),
    Float(f32),
    Double(f64)
}

impl From<f64> for Primitives {
    fn from(value: f64) -> Self {
        Primitives::Double(value)
    }
}

impl From<Primitives> for f32 {
    fn from(value: Primitives) -> Self {
        match value {
            Primitives::Integer(v) => {
                v as f32
            }
            Primitives::Long(v) => {
                v as f32
            }
            Primitives::Float(v) => {
                v as f32
            }
            Primitives::Double(v) => {
                v as f32
            }
        }
    }
}

impl From<Primitives> for f64 {
    fn from(value: Primitives) -> Self {
        match value {
            Primitives::Integer(v) => {
                v as f64
            }
            Primitives::Long(v) => {
                v as f64
            }
            Primitives::Float(v) => {
                v as f64
            }
            Primitives::Double(v) => {
                v as f64
            }
        }
    }
}


impl From<&Primitives> for f32 {
    fn from(value: &Primitives) -> Self {
        match value {
            Primitives::Integer(v) => {
                *v as f32
            }
            Primitives::Long(v) => {
                *v as f32
            }
            Primitives::Float(v) => {
                *v as f32
            }
            Primitives::Double(v) => {
                *v as f32
            }
        }
    }
}

impl From<Vec<Primitives>> for ArgmTensor {
    fn from(vals: Vec<Primitives>) -> Self {
        ArgmTensor(Tensor::from_slice(vals.into_iter().map(|v|v.into()).collect::<Vec<f64>>().as_slice()))
    }
}

// impl From<&Vec<Primitives>> for &ArgmTensor {
//     fn from(vals: &Vec<Primitives>) -> Self {
//         &ArgmTensor(Tensor::from_slice(vals.into_iter().map(|v|(*v).into()).collect::<Vec<f64>>().as_slice()))
//     }
// }


#[derive(Debug, Clone)]
struct RunResult{
    loss:f32,
    eval_config:Vec<Primitives>
}

impl RunResult {
    pub fn new(loss: f32, eval_config: Vec<Primitives>) -> Self {
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
        while let Some(o) = observations_c.pop() {

            observations_vec.push(o);
        }

        // println!("observations_vec={observations_vec:?}");
        assert_eq!(1., observations_vec[0].0.loss);
        assert_eq!(2., observations_vec[1].0.loss);
        assert_eq!(10., observations_vec[2].0.loss);
        assert_eq!(20., observations_vec[3].0.loss);
    }
}

pub trait ParzenEstimator {
    fn clc_dist(&self, samples:Tensor, weights: Option<Tensor>) -> Box<dyn Distribution>;
}

pub trait Distribution:Debug {
    fn sample(&self, n_samples:usize) -> Vec<Primitives>;
    fn basis_loglikelihood(&self, x:&Tensor) -> Tensor;
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

    pub fn new(cs:&ConfigurationSpace) -> Self {
        Self {
            observations:BinaryHeap::new(),
            mvpe_lower:None,
            mvpe_upper:None,
            percentile: 0.,
            esitmators: cs.iter().map(|hp|
                match hp {
                    Hyperparameter::UniformFloat { name, lower, upper } => {
                        Box::new(NumericalParzenEstimator::new(
                            // Tensor::zeros(1, FLOAT_CPU),
                            *lower,
                            *upper,
                            BandwidthFactor::default(),
                            true
                        )) as Box<dyn ParzenEstimator>

                    }
                    Hyperparameter::Categorical { name, values } => {
                        Box::new(CategoricalParzenEstimator::new(values.len(), true)) as Box<dyn ParzenEstimator>
                    }

                }
            ).collect()
        }
    }

    pub fn update_observations(&mut self, loss:f32, eval_config:&Vec<Primitives>) {
        self.observations.push(Reverse(RunResult::new(loss, eval_config.clone())));

        let n_lower = Self::percentile(self.observations.len()) as usize;

        debug!("n_lower={n_lower}");
        trace!("self.observations={:?}", self.observations);

        self.percentile = n_lower as f32 / self.observations.len() as f32; //[self._metric_name].size


        let (lower_vals, upper_vals) = self.pop_observations(n_lower);

        let (lower_esitmators, upper_esitmators) = self.esitmators.iter().enumerate().fold((vec![], vec![]), |mut lower_upper_esitmators, (param_i, e)|{
            lower_upper_esitmators.0.push(e.clc_dist(Tensor::from_slice(lower_vals.iter().map(|v|v[param_i].into()).collect::<Vec<f32>>().as_slice()), None));
            lower_upper_esitmators.1.push(e.clc_dist(Tensor::from_slice(upper_vals.iter().map(|v|v[param_i].into()).collect::<Vec<f32>>().as_slice()), None));

            lower_upper_esitmators
        });

        self.mvpe_lower.replace(MultiVariateParzenEstimator::new(lower_esitmators));
        self.mvpe_upper.replace(MultiVariateParzenEstimator::new(upper_esitmators));

        trace!("self.mvpe_lower={:?}", self.mvpe_lower);
        trace!("self.self.mvpe_upper={:?}", self.mvpe_upper);
    }

    fn pop_observations(&mut self, n_lower: usize) -> (Vec<Vec<Primitives>>, Vec<Vec<Primitives>>) {
        let mut observations_c = self.observations.clone();
        let mut lower_vals = Vec::with_capacity(n_lower);
        let mut upper_vals = Vec::with_capacity(observations_c.len() - n_lower);

        while let Some(o) = observations_c.pop() {
            // debug!(">>>v.0.eval_config={:?}", o);
            if lower_vals.len() < n_lower {
                lower_vals.push(o.0.eval_config);
            } else {
                upper_vals.push(o.0.eval_config);
            }
        }
        (lower_vals, upper_vals)
    }

    fn percentile(size:usize) -> f32 {
        (0.25 * (size as f32).sqrt()).ceil()
    }

    /// variables/samples
    fn get_config_candidates(&self, n_samples: usize) -> Vec<Vec<Primitives>> {
        self.mvpe_lower.as_ref().map(|mvpe_lower|mvpe_lower.sample(true, n_samples)).unwrap_or(vec![])
    }

    fn compute_config_loglikelihoods(&self, config_cands:&Vec<Vec<Primitives>>) -> (Tensor, Tensor) {
        let config_ll_lower = self.mvpe_lower.as_ref().expect("mvpe_lower").log_pdf(config_cands);
        let config_ll_upper = self.mvpe_upper.as_ref().expect("mvpe_upper").log_pdf(config_cands);
        (config_ll_lower, config_ll_upper)
    }

    fn compute_probability_improvement(&self, config_cands:&Vec<Vec<Primitives>> ) -> Tensor{

        let eps = 1e-12;
        let (cll_lower, cll_upper) = self.compute_config_loglikelihoods(config_cands);
        let first_term = (self.percentile + eps).ln();
        let second_term = (1.0 - self.percentile + eps).ln() + cll_upper - cll_lower;
        second_term.logaddexp(&first_term.into()).negative()

    }
}

#[cfg(test)]
mod tree_structured_parzen_estimator_tests {
    use tch::{Kind, Tensor};
    use crate::parzen::mvpe::MultiVariateParzenEstimator;
    use crate::parzen::npe::{BandwidthFactor, NumericalParzenEstimator};
    use crate::parzen::optimizer::{ConfigurationSpace, Hyperparameter, ParzenEstimator, Primitives, TreeStructuredParzenEstimator};
    use crate::parzen::optimizer::Primitives::Float;

    #[test]
    fn test_percentile(){
        assert_eq!(1., TreeStructuredParzenEstimator::percentile(1));
        assert_eq!(1., TreeStructuredParzenEstimator::percentile(10));
        assert_eq!(3., TreeStructuredParzenEstimator::percentile(100));
    }


    #[test]
    fn test_0(){

        let cs = ConfigurationSpace::new(vec![
            Hyperparameter::UniformFloat {name:"x_0".into(), lower: -1.0, upper: 1.0},
            Hyperparameter::UniformFloat {name:"x_1".into(), lower: -2.0, upper: 2.0},
        ]);

        let mut tree_structured_parzen_estimator = TreeStructuredParzenEstimator::new(&cs);

        tree_structured_parzen_estimator.update_observations(0.4, &vec![Primitives::Float(0.4), Primitives::Float(1.4)]);
        tree_structured_parzen_estimator.update_observations(0.1, &vec![Primitives::Float(0.1), Primitives::Float(1.1)]);
        tree_structured_parzen_estimator.update_observations(0.3, &vec![Primitives::Float(0.3), Primitives::Float(1.3)]);
        tree_structured_parzen_estimator.update_observations(0.2, &vec![Primitives::Float(0.2), Primitives::Float(1.2)]);

        // debug!("{:?}", tree_structured_parzen_estimator.observations);
        
        assert_eq!(0.1, tree_structured_parzen_estimator.observations.pop().expect("0.1").0.loss);
        assert_eq!(0.2, tree_structured_parzen_estimator.observations.pop().expect("0.2").0.loss);
        assert_eq!(0.3, tree_structured_parzen_estimator.observations.pop().expect("0.3").0.loss);
        assert_eq!(0.4, tree_structured_parzen_estimator.observations.pop().expect("0.4").0.loss);

        assert_eq!(2, tree_structured_parzen_estimator.mvpe_lower.expect("tree_structured_parzen_estimator").len())

    }

    #[test]
    fn test_compute_probability_improvement(){
        let cs = ConfigurationSpace::new(vec![
            Hyperparameter::UniformFloat {name:"x_0".into(), lower: -3.0, upper: 3.0},
            Hyperparameter::UniformFloat {name:"x_1".into(), lower: -3.0, upper: 3.0},
            Hyperparameter::UniformFloat {name:"x_2".into(), lower: -3.0, upper: 3.0},
        ]);
        let mut tsp = TreeStructuredParzenEstimator::new(&cs);

        let samples = [-1.0, 0.0, 1.0];


        tsp.mvpe_lower = Some(MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(-3.0f64, 3.0f64, BandwidthFactor::default(), true).clc_dist(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(-3.0f64, 3.0f64, BandwidthFactor::default(), true).clc_dist(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(-3.0f64, 3.0f64, BandwidthFactor::default(), true).clc_dist(Tensor::from_slice(&samples), None)

        ]));
        let samples = [-2.0, 0.0, 2.0];

        tsp.mvpe_upper = Some(MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(-3.0f64, 3.0f64, BandwidthFactor::default(), true).clc_dist(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(-3.0f64, 3.0f64, BandwidthFactor::default(), true).clc_dist(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(-3.0f64, 3.0f64, BandwidthFactor::default(), true).clc_dist(Tensor::from_slice(&samples), None)

        ]));


        let pi = tsp.compute_probability_improvement(&vec![
            vec![Float(0.0), Float(1.0), Float(2.0)], //x0
            vec![Float(0.1), Float(1.1), Float(2.1)], //x1
            vec![Float(0.2), Float(1.2), Float(2.2)]  //x2
        ]);

        println!("pi={pi}");
        assert!(
            Tensor::from_slice(&[ 1.85448311,  4.59934198, -4.30722048]).totype(Kind::Float).allclose(&pi,  1e-02, 1e-02, false)
        );

    }
}

struct TPEOptimizer<F> where F: Fn(&Vec<Primitives>) -> f32 {
    obj_func:F,
    cs:ConfigurationSpace,
    n_init:usize,
    max_evals:usize,
    tpe_samplers:HashMap<String, TreeStructuredParzenEstimator>,
}

impl <F> TPEOptimizer<F> where F: Fn(&Vec<Primitives>) -> f32{

    pub fn new(obj_func:F, cs:ConfigurationSpace, max_evals: usize, n_init: usize) -> Self {
        let estimator = TreeStructuredParzenEstimator::new(&cs);
        Self {
            obj_func,
            cs,
            n_init,
            max_evals,
            tpe_samplers: HashMap::from([
                ("loss".into(), estimator)
            ])
        }
    }

    pub fn optimize(&mut self) -> Vec<Primitives> {
        let mut best_loss = f32::MAX;
        let mut best_config:Option<Vec<Primitives>> = None;

        (0..self.max_evals).for_each(|i|{

            let eval_config = if  i < self.n_init {
                debug!("i={i} random sample");
                self.cs.random_sample()

            } else {
                debug!("---------------------------------------i={i} sample");
                self.sample()
            };

            let loss = (self.obj_func)(
                &eval_config
            );
            if loss < best_loss {
                best_loss = loss;
                best_config.replace(eval_config.clone());
            }
            debug!("i={i} eval_config={:?} loss={loss}", eval_config);

            for (_key, mut tpe) in &mut self.tpe_samplers {
                tpe.update_observations(loss, &eval_config)
            }
        });

        best_config.expect("best_config")
    }

    fn sample(&self)->Vec<Primitives> {

        //TODO if always loss only - merge get_config_candidates & compute_probability_improvement
        let config_cands = stack_params(

            self.tpe_samplers.values().map(|tpe| {
                tpe.get_config_candidates(24)
            }).collect::<Vec<Vec<Vec<Primitives>>>>() //tpe(only loss)/params/candidates
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

        config_cands.into_iter().map(|p|p[max_i as usize]).collect()   // return weights @ pi_config_array

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
    use rand::distributions::Uniform;
    use rand::prelude::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_mt::{Mt19937GenRand32, Mt19937GenRand64};

    use tch::{IndexOp, Tensor};
    use crate::argmin::ArgmTensor;
    use crate::parzen::optimizer::{ConfigurationSpace, Hyperparameter, Primitives, stack_params, TPEOptimizer};
    use crate::utils::sin3_pow2_f;

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
        let sphere = sphere(300)(&vec![
            Primitives::Float(0.1),
            Primitives::Float(0.2),
            Primitives::Float(0.3),
            Primitives::Float(0.4),
            Primitives::Float(0.5),
        ]);

        assert_eq!(0.55, sphere);

    }

    #[test]
    fn test_optimize_numeric(){

        let mut opt = TPEOptimizer::new(
            sphere(300),

            ConfigurationSpace::new(
                (0..3).map(|i|Hyperparameter::UniformFloat {name:format!("x_{}", i), lower: -500.0, upper: 500.0}).collect()
            ),
            500,
            10
        );

        let best_config = opt.optimize();

        println!("best_config={best_config:?}");
        assert!((300. - <Primitives as Into<f32>>::into(best_config[0])).abs() < 3.);
        assert!((300. - <Primitives as Into<f32>>::into(best_config[1])).abs() < 3.);
        assert!((300. - <Primitives as Into<f32>>::into(best_config[2])).abs() < 3.);
    }

    #[test]
    fn test_optimize_caegoric(){

        let mut opt = TPEOptimizer::new(
            move |params| {
                debug!("params={params:?}");
                let t: Tensor = Tensor::from_slice(
                    &[
                        0.0
                    ]
                );
                let loss = t.subtract_scalar(30.).pow_tensor_scalar(2).sum(None).double_value(&[]) as f32;

                debug!("loss={loss:?}");

                loss
            }
            ,

            ConfigurationSpace::new(
                // (0..3).map(|i|Hyperparameter::UniformFloat {name:format!("x_{}", i), lower: -500.0, upper: 500.0}).collect()
                vec![
                    Hyperparameter::UniformFloat {name:"x_0".into(), lower: -50.0, upper: 50.0},
                    Hyperparameter::UniformFloat {name:"x_1".into(), lower: -50.0, upper: 50.0},
                    Hyperparameter::Categorical { name: "".to_string(), values: vec!["a".into(), "b".into(), "c".into()] }
                ]
            ),
            500,
            10
        );

        let best_config = opt.optimize();

        println!("best_config={best_config:?}");
        assert!((300. - <Primitives as Into<f32>>::into(best_config[0])).abs() < 3.);
        assert!((300. - <Primitives as Into<f32>>::into(best_config[1])).abs() < 3.);
        assert!((300. - <Primitives as Into<f32>>::into(best_config[2])).abs() < 3.);
    }


    fn sphere(d_x: i64) -> impl Fn(&Vec<Primitives>) -> f32 {
        move |params| {
            debug!("params={params:?}");
            let t: Tensor = <Vec<Primitives> as Into<ArgmTensor>>::into(params.clone()).into();
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

        println!("t={t} argmax={}", t.argmax(1, true).i(0).double_value(&[]));

    }
}


enum Hyperparameter {
    UniformFloat{name:String, lower:f64, upper:f64},
    Categorical{name:String, values:Vec<String>}
}

impl Hyperparameter {
    fn name(&self)->String {
        match self {
            Hyperparameter::UniformFloat { name, .. } => {
                name.clone()
            }
            Hyperparameter::Categorical { name,.. } => {
                name.clone()
            }
        }
    }

    pub fn random_sample(&self) -> Primitives {
        match self {
            Hyperparameter::UniformFloat { name, lower, upper } => {
                let uniform = opt_rnd().sample(Uniform::new(0., 1.));
                let sample = lower + uniform * (upper - lower) ;
                // debug!("UniformFloat={uniform:?} lb={lower} ub={upper} sample={sample} ");
                Primitives::Double(sample)
            }
            Hyperparameter::Categorical { name, values } => {
                let uniform = opt_rnd().sample(Uniform::new(0., 1.));
                let sample = uniform * values.len() as f32;
                debug!("Categorical={sample:?} values={values:?} ");
                Primitives::Integer(sample as i32)
            }
        }
    }


}

struct ConfigurationSpace {
    hyperparameters:Vec<Hyperparameter>
}

impl ConfigurationSpace {
    pub fn empty() -> Self {
        Self {
            hyperparameters: vec![]
        }
    }
    pub fn new(hyperparameters: Vec<Hyperparameter>) -> Self {
        Self { hyperparameters }
    }


    pub fn add_hyperparameter(&mut self, hpe:Hyperparameter){
        self.hyperparameters.push(hpe);
    }

    pub fn random_sample(&self) ->Vec<Primitives> {
        self.hyperparameters.iter().map(|hp|hp.random_sample()).collect()
    }

    fn len(&self) ->usize {
        self.hyperparameters.len()
    }

    fn iter(&self) -> CsIter {
        CsIter(self.hyperparameters.iter())
    }
}

struct CsIter<'a>(std::slice::Iter<'a, Hyperparameter>);

impl<'a> Iterator for CsIter<'a> {
    type Item = &'a Hyperparameter;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next()
    }
}

impl<'a> IntoIterator for &'a ConfigurationSpace {
    type Item = &'a Hyperparameter;
    type IntoIter = CsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

#[cfg(test)]
mod tests {
    use log::debug;
    use rand::prelude::StdRng;
    use rand::SeedableRng;
    use crate::parzen::optimizer::{ConfigurationSpace, Hyperparameter, TPEOptimizer};
    use crate::utils::sin3_pow2_f;

    #[test]
    fn test_0(){

        let cs = ConfigurationSpace::new(vec![Hyperparameter::UniformFloat {name:format!("x_{}", 0), lower: -2.0, upper: 2.0}]);


        let mut opt = TPEOptimizer::new(
            |x| {
                debug!("x={x:?}");
                sin3_pow2_f(x.get(0).expect("x_0").into())
            },
            cs, 10, 5
        );

        opt.optimize();
    }
}