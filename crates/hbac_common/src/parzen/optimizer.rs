use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use log::debug;
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;
use tch::Tensor;
use crate::argmin::ArgmTensor;
use crate::parzen::npe::{BandwidthFactor, MultiVariateParzenEstimator, NumericalParzenEstimator};
use super::opt_rng::{opt_rnd};
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


#[derive(Debug)]
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


struct TreeStructuredParzenEstimator {
    observations:BinaryHeap<Reverse<RunResult>>,
    mvpe_lower:Option<MultiVariateParzenEstimator>, //TODO move out
    mvpe_upper:Option<MultiVariateParzenEstimator>, //TODO move out
    percentile:f32 //TODO move out
}

impl TreeStructuredParzenEstimator {

    pub fn new() -> Self {
        Self {
            observations:BinaryHeap::new(),
            mvpe_lower:None,
            mvpe_upper:None,
            percentile: 0.
        }
    }

    pub fn update_observations(&mut self, loss:f32, eval_config:&Vec<Primitives>, cs:&ConfigurationSpace) {


        self.observations.push(Reverse(RunResult::new(loss, eval_config.clone())));

        let n_lower = Self::percentile(self.observations.len()) as usize;

        self.percentile = n_lower as f32 / self.observations.len() as f32; //[self._metric_name].size

        debug!("loss={loss} eval_config={eval_config:?} n_lower={n_lower} self.percentile={} self.observations.len()={}", self.percentile, self.observations.len());


        let (lower_vals, upper_vals) = self.observations.iter().enumerate().fold((vec![], vec![]), |mut a, (i, v)|{
            if i < n_lower {
                a.0.push(v.0.eval_config.clone().clone());
            } else {
                a.1.push(v.0.eval_config.clone().clone());
            }
            a
        });
        debug!("lower_vals={lower_vals:?}");
        debug!("upper_vals={upper_vals:?}");

        let (lower_esitmators, upper_esitmators) = cs
            .into_iter()
            .enumerate()
            .fold((vec![], vec![]), |mut lower_upper_esitmators, (param_i, hp)| {

                lower_upper_esitmators.0.push(
                    NumericalParzenEstimator::new(
                       Tensor::from_slice(lower_vals.iter().map(|v|v[param_i].into()).collect::<Vec<f32>>().as_slice()),
                       hp.lower(),
                       hp.upper(),
                       None,
                       BandwidthFactor::default(),
                       true
                    )
                );

                lower_upper_esitmators.1.push(
                    NumericalParzenEstimator::new(
                        Tensor::from_slice(upper_vals.iter().map(|v|v[param_i].into()).collect::<Vec<f32>>().as_slice()),
                        hp.lower(),
                        hp.upper(),
                        None,
                        BandwidthFactor::default(),
                        true
                    )
                );

                lower_upper_esitmators
        });

        self.mvpe_lower.replace(MultiVariateParzenEstimator::new(lower_esitmators));
        self.mvpe_upper.replace(MultiVariateParzenEstimator::new(upper_esitmators));
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
    use tch::Tensor;
    use crate::parzen::npe::{BandwidthFactor, MultiVariateParzenEstimator, NumericalParzenEstimator};
    use crate::parzen::optimizer::{ConfigurationSpace, Hyperparameter, Primitives, TreeStructuredParzenEstimator};
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

        let mut tree_structured_parzen_estimator = TreeStructuredParzenEstimator::new();

        tree_structured_parzen_estimator.update_observations(0.4, &vec![Primitives::Float(0.4), Primitives::Float(1.4)], &cs);
        tree_structured_parzen_estimator.update_observations(0.1, &vec![Primitives::Float(0.1), Primitives::Float(1.1)], &cs);
        tree_structured_parzen_estimator.update_observations(0.3, &vec![Primitives::Float(0.3), Primitives::Float(1.3)], &cs);
        tree_structured_parzen_estimator.update_observations(0.2, &vec![Primitives::Float(0.2), Primitives::Float(1.2)], &cs);

        // debug!("{:?}", tree_structured_parzen_estimator.observations);
        
        assert_eq!(0.1, tree_structured_parzen_estimator.observations.pop().expect("0.1").0.loss);
        assert_eq!(0.2, tree_structured_parzen_estimator.observations.pop().expect("0.2").0.loss);
        assert_eq!(0.3, tree_structured_parzen_estimator.observations.pop().expect("0.3").0.loss);
        assert_eq!(0.4, tree_structured_parzen_estimator.observations.pop().expect("0.4").0.loss);

        assert_eq!(2, tree_structured_parzen_estimator.mvpe_lower.expect("tree_structured_parzen_estimator").len())

    }

    #[test]
    fn test_compute_probability_improvement(){
        let mut tsp = TreeStructuredParzenEstimator::new(

        );

        let samples = [-1.0, 0.0, 1.0];


        tsp.mvpe_lower = Some(MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(Tensor::from_slice(&samples), -3.0f64, 3.0f64, None, BandwidthFactor::default(), true),
            NumericalParzenEstimator::new(Tensor::from_slice(&samples), -3.0f64, 3.0f64, None, BandwidthFactor::default(), true),
            NumericalParzenEstimator::new(Tensor::from_slice(&samples), -3.0f64, 3.0f64, None, BandwidthFactor::default(), true)

        ]));
        let samples = [-2.0, 0.0, 2.0];

        tsp.mvpe_upper = Some(MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(Tensor::from_slice(&samples), -3.0f64, 3.0f64, None, BandwidthFactor::default(), true),
            NumericalParzenEstimator::new(Tensor::from_slice(&samples), -3.0f64, 3.0f64, None, BandwidthFactor::default(), true),
            NumericalParzenEstimator::new(Tensor::from_slice(&samples), -3.0f64, 3.0f64, None, BandwidthFactor::default(), true)

        ]));


        let pi = tsp.compute_probability_improvement(&vec![
            vec![Float(0.0), Float(0.1), Float(0.2)]
        ]);

        println!("pi={pi}");
        assert_eq!(18545, (pi.round_decimals(4).double_value(&[]) * 10000.0) as usize);
        // let pi = tree_structured_parzen_estimator.compute_probability_improvement(
        //
        // );
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
        Self {
            obj_func,
            cs,
            n_init,
            max_evals,
            tpe_samplers: HashMap::from([
                ("loss".into(), TreeStructuredParzenEstimator::new())
            ])
        }
    }

    pub fn optimize(&mut self) {

        (0..self.max_evals).for_each(|i|{

            let eval_config = if  i < self.n_init {
                debug!("i={i} random sample");
                self.cs.random_sample()

            } else {
                debug!("i={i} sample");
                self.sample()
            };

            let loss = (self.obj_func)(
                &eval_config
            );
            debug!("i={i} loss={loss}");

            for (_key, mut tpe) in &mut self.tpe_samplers {
                tpe.update_observations(loss, &eval_config, &self.cs)
            }
        });
    }

    fn sample(&self)->Vec<Primitives> {


        let config_cands = stack_params(
            self.tpe_samplers.values().map(|tpe| {
                tpe.get_config_candidates(21)
            }).collect::<Vec<Vec<Vec<Primitives>>>>() //tpe(only loss)/params/candidates
        );



        let res = Tensor::stack(
            self.tpe_samplers.values().map(|tpe| tpe.compute_probability_improvement(&config_cands)).collect::<Vec<Tensor>>().as_slice(),
            0,
        );

        // self.compute_config_loglikelihoods(stack_params(&config_cands, &self.parzen_estimators, 0))

        vec![]
        // return weights @ pi_config_array
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

    use tch::Tensor;
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
        let sphere = sphere()(&vec![
            Primitives::Float(0.1),
            Primitives::Float(0.2),
            Primitives::Float(0.3),
            Primitives::Float(0.4),
            Primitives::Float(0.5),
        ]);

        assert_eq!(0.55, sphere);

    }

    #[test]
    fn test_sample(){

        let cs = ConfigurationSpace::new(

            (0..10).map(|i|Hyperparameter::UniformFloat {name:format!("x_{}", i), lower: -5.0, upper: 5.0}).collect()
        );

        let mut opt = TPEOptimizer::new(
            sphere(),
            cs,
            5,
            5
        );

        opt.sample();


    }

    #[test]
    fn test_compute_probability_improvement(){
        let max_evals = 5;

        let cs = ConfigurationSpace::new(vec![Hyperparameter::UniformFloat {name:format!("x_{}", 0), lower: -2.0, upper: 2.0}]);


        let mut opt = TPEOptimizer::new(
            sphere(),
            cs, 10, 5
        );

        opt.optimize();
    }

    fn sphere() -> fn(&Vec<Primitives>) -> f32 {
        |params| {
            debug!("params={params:?}");
            let t: Tensor = <Vec<Primitives> as Into<ArgmTensor>>::into(params.clone()).into();
            let loss = t.pow_tensor_scalar(2).sum(None).double_value(&[]) as f32;

            debug!("loss={loss:?}");

            loss
        }
    }
}


enum Hyperparameter {
    UniformFloat{name:String, lower:f64, upper:f64}
}

impl Hyperparameter {
    fn name(&self)->String {
        match self {
            Hyperparameter::UniformFloat { name, .. } => {
                name.clone()
            }
        }
    }

    pub fn random_sample(&self) -> Primitives {
        match self {
            Hyperparameter::UniformFloat { name, lower, upper } => {
                let uniform = opt_rnd().sample(Uniform::new(0., 1.));
                let sample = lower + uniform * (upper - lower) ;
                // debug!("uniform={uniform:?} lb={lower} ub={upper} sample={sample} ");
                Primitives::Double(sample)
            }
        }
    }

    fn upper(&self) -> f64{
        match self {
            Hyperparameter::UniformFloat { name, lower, upper } => {
                *upper
            }
        }
    }

    fn lower(&self)-> f64 {
        match self {
            Hyperparameter::UniformFloat { name, lower, upper } => {
                *lower
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
    use crate::utils::{sin3_pow2_f};

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