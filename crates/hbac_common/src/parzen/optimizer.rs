use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use rand::{Rng, thread_rng};
use rand::distributions::Uniform;
use tch::Tensor;
use crate::parzen::npe::{BandwidthFactor, MultiVariateParzenEstimator, NumericalParzenEstimator};

// type TargetFn = dyn Fn(HashMap<String, Box<dyn num_traits::AsPrimitive<_>>>) -> f64;

#[derive(Debug, Copy, Clone)]
pub enum Primitives {
    Integer(i32),
    Long(i64),
    Float(f32),
    Double(f64)
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

impl From<&Primitives> for f32 {
    fn from(value: &Primitives) -> Self {
        2.2
    }
}

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
    mvpe_upper:Option<MultiVariateParzenEstimator> //TODO move out
}

impl TreeStructuredParzenEstimator {

    pub fn new() -> Self {
        Self {
            observations:BinaryHeap::new(),
            mvpe_lower:None,
            mvpe_upper:None
        }
    }

    pub fn update_observations(&mut self, loss:f32, eval_config:&Vec<Primitives>, cs:&ConfigurationSpace) {
        self.observations.push(Reverse(RunResult::new(loss, eval_config.clone())));

        let n_lower = Self::percentile(self.observations.len()) as usize;

        let lower_vals = self.observations.iter().take(n_lower).map(|r|r.0.eval_config.clone()).collect::<Vec<Vec<Primitives>>>();
        let upper_vals = self.observations.iter().skip(n_lower).map(|r|r.0.eval_config.clone()).collect::<Vec<Vec<Primitives>>>();

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
        (0.25 * (size as f32).sqrt()).ceil() / (size as f32)
    }
}

#[cfg(test)]
mod tree_structured_parzen_estimator_tests {
    use crate::parzen::optimizer::{ConfigurationSpace, Hyperparameter, Primitives, TreeStructuredParzenEstimator};

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

        // println!("{:?}", tree_structured_parzen_estimator.observations);
        
        assert_eq!(0.1, tree_structured_parzen_estimator.observations.pop().expect("0.1").0.loss);
        assert_eq!(0.2, tree_structured_parzen_estimator.observations.pop().expect("0.2").0.loss);
        assert_eq!(0.3, tree_structured_parzen_estimator.observations.pop().expect("0.3").0.loss);
        assert_eq!(0.4, tree_structured_parzen_estimator.observations.pop().expect("0.4").0.loss);

        assert_eq!(2, tree_structured_parzen_estimator.mvpe_lower.expect("tree_structured_parzen_estimator").len())

    }
}

struct TPEOptimizer<F> where F: Fn(&Vec<Primitives>) -> f32 {
    obj_func:F,
    cs:ConfigurationSpace,
    n_init:usize,
    max_evals:usize,
    tpe_samplers:HashMap<String, TreeStructuredParzenEstimator>
}

impl <F> TPEOptimizer<F> where F: Fn(&Vec<Primitives>) -> f32{

    pub fn new(obj_func:F, cs:ConfigurationSpace) -> Self {
        Self {
            obj_func,
            cs,
            n_init:3,
            max_evals:10,
            tpe_samplers: HashMap::from([
                ("loss".into(), TreeStructuredParzenEstimator::new())
            ])
        }
    }

    pub fn optimize(&mut self) {

        (0..self.n_init).for_each(|i|{
            let eval_config = self.cs.random_sample();

            let loss = (self.obj_func)(
                &eval_config
            );

            for mut key_tpe in &mut self.tpe_samplers {
                key_tpe.1.update_observations(loss, &eval_config, &self.cs)
            }
        });
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
        let mut rng = thread_rng();
        match self {
            Hyperparameter::UniformFloat { name, lower, upper } => {
                Primitives::Double(rng.sample(Uniform::new(lower, upper)))
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
    use crate::parzen::optimizer::{ConfigurationSpace, Hyperparameter, TPEOptimizer};
    use crate::utils::{sin3_pow2_f};

    #[test]
    fn test_0(){

        let cs = ConfigurationSpace::new(vec![Hyperparameter::UniformFloat {name:format!("x_{}", 0), lower: -2.0, upper: 2.0}]);


        let mut opt = TPEOptimizer::new(
            |x| {
                println!("x={x:?}");
                sin3_pow2_f(x.get(0).expect("x_0").into())
            },
            cs
        );

        opt.optimize();
    }
}