use log::{debug, trace};
use std::cmp::{Ordering, Reverse};
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::marker::PhantomData;
// use rand_distr::{Uniform, Distribution};
use crate::parzen::cpe::CategoricalParzenEstimator;
use crate::parzen::mvpe::MultiVariateParzenEstimator;
use crate::parzen::npe::{BandwidthFactor, NumericalParzenEstimator};
use tch::Tensor;
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::val::{NamedVal, Val};

#[derive(Debug, Clone)]
struct RunResult {
    loss: f32,
    eval_config: Vec<Val>,
    is_satisfied: Option<bool>,
}

impl RunResult {
    pub fn new(loss: f32, eval_config: Vec<Val>, is_satisfied: Option<bool>) -> Self {
        Self {
            loss,
            eval_config,
            is_satisfied,
        }
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
    fn test_run_result_in_binaryheap() {
        let mut observations = BinaryHeap::new();

        observations.push(Reverse(RunResult::new(10., vec![], None)));
        observations.push(Reverse(RunResult::new(1., vec![], None)));
        observations.push(Reverse(RunResult::new(20., vec![], None)));
        observations.push(Reverse(RunResult::new(2., vec![], None)));

        let mut observations_c = observations.clone();
        let mut observations_vec = vec![];
        while let Some(e) = observations_c.pop() {
            observations_vec.push(e);
        }

        assert_eq!(1., observations_vec[0].0.loss);
        assert_eq!(2., observations_vec[1].0.loss);
        assert_eq!(10., observations_vec[2].0.loss);
        assert_eq!(20., observations_vec[3].0.loss);
    }
}
pub trait ParzenEstimator: Send {
    fn clc_distr(&self, samples: Tensor, weights: Option<Tensor>) -> Box<dyn Distribution>;
    fn clc_distr_from_vec(
        &self,
        samples: &mut dyn Iterator<Item = Val>,
        weights: Option<Tensor>,
    ) -> Box<dyn Distribution>;
}

pub trait Distribution: Debug + Send {
    fn sample(&self, n_samples: usize) -> Vec<Val>;
    fn basis_loglikelihood(&self, x: &mut dyn Iterator<Item = &Val>) -> Tensor;
    fn weights(&self) -> &Tensor;
}

#[derive(Clone, Copy)]
enum PercentileStrategy {
    Default,
    Threshold { upper_bound: f32 },
}

struct TreeStructuredParzenEstimator {
    observations: BinaryHeap<Reverse<RunResult>>,
    mvpe_lower: Option<MultiVariateParzenEstimator>, //TODO move out
    mvpe_upper: Option<MultiVariateParzenEstimator>, //TODO move out
    percentile: f32,                                 //TODO move out
    esitmators: Vec<Box<dyn ParzenEstimator>>,
    percentile_strategy: PercentileStrategy,
}

impl TreeStructuredParzenEstimator {
    pub fn new<T: FieldName>(cs: &ConfigurationSpace<T>) -> Self {
        Self::new_with_percentile_strategy(cs, PercentileStrategy::Default)
    }

    pub fn new_with_threshold<T: FieldName>(cs: &ConfigurationSpace<T>, upper_bound: f32) -> Self {
        Self::new_with_percentile_strategy(cs, PercentileStrategy::Threshold { upper_bound })
    }

    fn new_with_percentile_strategy<T: FieldName>(
        cs: &ConfigurationSpace<T>,
        percentile_strategy: PercentileStrategy,
    ) -> Self {
        debug!("TreeStructuredParzenEstimator::new: cs={cs:?}");

        let esitmators:Vec<_> = cs
            .iter()
            .map(|hp| match hp {
                Hyperparameter::UniformFloat { name, lower, upper } => {
                    Box::new(NumericalParzenEstimator::new(
                        Some(name.to_string()),
                        *lower,
                        *upper,
                        BandwidthFactor::default(),
                        true,
                        None,
                    )) as Box<dyn ParzenEstimator>
                }
                Hyperparameter::UniformInt { name, lower, upper } => {
                    Box::new(NumericalParzenEstimator::new(
                        Some(name.to_string()),
                        *lower as f64,
                        *upper as f64,
                        BandwidthFactor::default(),
                        true,
                        Some(1), //TODO step
                    )) as Box<dyn ParzenEstimator>
                }

                Hyperparameter::Categorical { name, values } => Box::new(
                    CategoricalParzenEstimator::new(Some(name.to_string()), values.len(), true),
                )
                    as Box<dyn ParzenEstimator>,
            })
            .collect();
        assert!(!esitmators.is_empty(), "estimators are empty");


        Self {
            observations: BinaryHeap::new(),
            mvpe_lower: None,
            mvpe_upper: None,
            percentile: 0.,
            esitmators,
            percentile_strategy,
        }
    }

    pub fn update_observations(&mut self, loss: f32, eval_config: &Vec<Val>, is_satisfied: Option<bool>) {
        self.observations.push(Reverse(RunResult::new(
            loss,
            eval_config.clone(),
            is_satisfied,
        ))); //TODO clone

        let sorted_observations = self.sorted_observations();
        let n_lower = self.n_lower(&sorted_observations, is_satisfied.is_some());

        debug!("TreeStructuredParzenEstimator::update_observations: n_lower={n_lower} sorted_observations.len={}", sorted_observations.len());
        trace!("TreeStructuredParzenEstimator::update_observations: self.observations={:?}", self.observations);

        self.percentile = n_lower as f32 / self.observations.len() as f32; //[self._metric_name].size

        let (lower_vals, upper_vals) = self.split_observations(&sorted_observations, n_lower);

        let (lower_esitmators, upper_esitmators) = self.esitmators.iter().enumerate().fold(
            (vec![], vec![]),
            |mut lower_upper_esitmators, (param_i, pe)| {
                lower_upper_esitmators.0.push(
                    pe.clc_distr_from_vec(&mut lower_vals.iter().map(|v| v[param_i].clone()), None),
                );
                lower_upper_esitmators.1.push(
                    pe.clc_distr_from_vec(&mut upper_vals.iter().map(|v| v[param_i].clone()), None),
                );

                lower_upper_esitmators
            },
        );

        self.mvpe_lower.replace(MultiVariateParzenEstimator::new(lower_esitmators));
        self.mvpe_upper.replace(MultiVariateParzenEstimator::new(upper_esitmators));

        trace!("TreeStructuredParzenEstimator::update_observations: self.mvpe_lower={:?}", self.mvpe_lower);
        trace!("TreeStructuredParzenEstimator::update_observations: self.self.mvpe_upper={:?}", self.mvpe_upper);
    }

    fn sorted_observations(&self) -> Vec<RunResult> {
        let mut observations_c = self.observations.clone();
        let mut sorted_observations = Vec::with_capacity(observations_c.len());

        while let Some(e) = observations_c.pop() {
            sorted_observations.push(e.0);
        }
        sorted_observations
    }

    fn split_observations(&self, sorted_observations: &[RunResult], n_lower: usize) -> (Vec<Vec<Val>>, Vec<Vec<Val>>) {
        let mut lower_vals = Vec::with_capacity(n_lower);
        let mut upper_vals = Vec::with_capacity(sorted_observations.len() - n_lower);

        for e in sorted_observations {
            if lower_vals.len() < n_lower {
                lower_vals.push(e.eval_config.clone());
            } else {
                upper_vals.push(e.eval_config.clone());
            }
        }

        (lower_vals, upper_vals)
    }

    fn percentile(size: usize) -> f32 {
        (0.25 * (size as f32).sqrt()).ceil()
    }

    fn n_lower(&self, sorted_observations: &[RunResult], enforce_feasible_min: bool) -> usize {
        if sorted_observations.is_empty() {
            return 0;
        }

        let min_num = if enforce_feasible_min {
            let base = Self::percentile(sorted_observations.len()) as usize;
            let mut n_satisfied = 0usize;
            let mut min_num = sorted_observations.len();
            for (i, observation) in sorted_observations.iter().enumerate() {
                if observation.is_satisfied.unwrap_or(false) {
                    n_satisfied += 1;
                }
                if n_satisfied >= base {
                    min_num = i + 1;
                    break;
                }
            }
            min_num
        } else {
            1usize
        };

        match self.percentile_strategy {
            PercentileStrategy::Default => {
                (Self::percentile(sorted_observations.len()) as usize).max(min_num).min(sorted_observations.len())
            }
            PercentileStrategy::Threshold { upper_bound } => {
                let threshold_count = sorted_observations
                    .iter()
                    .take_while(|observation| observation.loss <= upper_bound)
                    .count();
                threshold_count.max(min_num).min(sorted_observations.len())
            }
        }
    }

    /// variables/samples
    fn get_config_candidates(&self, n_samples: usize) -> Vec<Vec<Val>> {
        self.mvpe_lower
            .as_ref()
            .map(|mvpe_lower| {
                debug!("TPEOptimizer::sample: mvpe_lower sample mvpe_lower={mvpe_lower:?} n_samples={n_samples}");

                let r = mvpe_lower.sample(true, n_samples);
                debug!("TPEOptimizer::sample: mvpe_lower sample r={r:?}");
                r
            })
            .unwrap_or(vec![])
    }

    fn compute_config_loglikelihoods(&self, config_cands: &Vec<Vec<Val>>) -> (Tensor, Tensor) {
        let config_ll_lower = self
            .mvpe_lower
            .as_ref()
            .expect("mvpe_lower")
            .log_pdf(config_cands);
        let config_ll_upper = self
            .mvpe_upper
            .as_ref()
            .expect("mvpe_upper")
            .log_pdf(config_cands);
        (config_ll_lower, config_ll_upper)
    }

    fn compute_probability_improvement(&self, config_cands: &Vec<Vec<Val>>) -> Tensor {
        let eps = 1e-12;
        let (cll_lower, cll_upper) = self.compute_config_loglikelihoods(config_cands);
        let first_term = (self.percentile + eps).ln();
        let second_term = (1.0 - self.percentile + eps).ln() + cll_upper - cll_lower;
        second_term.logaddexp(&first_term.into()).negative()
    }
}

#[cfg(test)]
mod tree_structured_parzen_estimator_tests {
    use crate::parzen::mvpe::MultiVariateParzenEstimator;
    use crate::parzen::npe::{BandwidthFactor, NumericalParzenEstimator};
    use crate::parzen::optimizer::{
        ConfigurationSpace, ParzenEstimator, TreeStructuredParzenEstimator, Val,
    };
    use log::debug;
    use tch::{Kind, Tensor};
    use thoosa_shared::hyperparameter::Hyperparameter;

    #[test]
    fn test_percentile() {
        assert_eq!(1., TreeStructuredParzenEstimator::percentile(1));
        assert_eq!(1., TreeStructuredParzenEstimator::percentile(10));
        assert_eq!(3., TreeStructuredParzenEstimator::percentile(100));
    }

    #[test]
    fn test_configuration_space() {
        let cs: ConfigurationSpace<String> = ConfigurationSpace::new(vec![
            Hyperparameter::UniformFloat {
                name: "x_0".into(),
                lower: -1.0,
                upper: 1.0,
            },
            Hyperparameter::UniformFloat {
                name: "x_1".into(),
                lower: -2.0,
                upper: 2.0,
            },
        ]);

        let mut tree_structured_parzen_estimator = TreeStructuredParzenEstimator::new(&cs);

        tree_structured_parzen_estimator
            .update_observations(0.4, &vec![Val::Real(0.4), Val::Real(1.4)], None);
        tree_structured_parzen_estimator
            .update_observations(0.1, &vec![Val::Real(0.1), Val::Real(1.1)], None);
        tree_structured_parzen_estimator
            .update_observations(0.3, &vec![Val::Real(0.3), Val::Real(1.3)], None);
        tree_structured_parzen_estimator
            .update_observations(0.2, &vec![Val::Real(0.2), Val::Real(1.2)], None);

        // debug!("{:?}", tree_structured_parzen_estimator.observations);

        assert_eq!(
            0.1,
            tree_structured_parzen_estimator
                .observations
                .pop()
                .expect("0.1")
                .0
                .loss
        );
        assert_eq!(
            0.2,
            tree_structured_parzen_estimator
                .observations
                .pop()
                .expect("0.2")
                .0
                .loss
        );
        assert_eq!(
            0.3,
            tree_structured_parzen_estimator
                .observations
                .pop()
                .expect("0.3")
                .0
                .loss
        );
        assert_eq!(
            0.4,
            tree_structured_parzen_estimator
                .observations
                .pop()
                .expect("0.4")
                .0
                .loss
        );

        assert_eq!(
            2,
            tree_structured_parzen_estimator
                .mvpe_lower
                .expect("tree_structured_parzen_estimator")
                .len()
        )
    }

    #[test]
    fn test_compute_probability_improvement() {
        let cs: ConfigurationSpace<String> = ConfigurationSpace::new(vec![
            Hyperparameter::UniformFloat {
                name: "x_0".into(),
                lower: -3.0,
                upper: 3.0,
            },
            Hyperparameter::UniformFloat {
                name: "x_1".into(),
                lower: -3.0,
                upper: 3.0,
            },
            Hyperparameter::UniformFloat {
                name: "x_2".into(),
                lower: -3.0,
                upper: 3.0,
            },
        ]);
        let mut tsp = TreeStructuredParzenEstimator::new(&cs);

        let samples = [-1.0, 0.0, 1.0];

        tsp.mvpe_lower = Some(MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(
                None,
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None,
            )
            .clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(
                None,
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None,
            )
            .clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(
                None,
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None,
            )
            .clc_distr(Tensor::from_slice(&samples), None),
        ]));
        let samples = [-2.0, 0.0, 2.0];

        tsp.mvpe_upper = Some(MultiVariateParzenEstimator::new(vec![
            NumericalParzenEstimator::new(
                None,
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None,
            )
            .clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(
                None,
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None,
            )
            .clc_distr(Tensor::from_slice(&samples), None),
            NumericalParzenEstimator::new(
                None,
                -3.0f64,
                3.0f64,
                BandwidthFactor::default(),
                true,
                None,
            )
            .clc_distr(Tensor::from_slice(&samples), None),
        ]));

        let pi = tsp.compute_probability_improvement(&vec![
            vec![Val::Real(0.0), Val::Real(1.0), Val::Real(2.0)], //x0
            vec![Val::Real(0.1), Val::Real(1.1), Val::Real(2.1)], //x1
            vec![Val::Real(0.2), Val::Real(1.2), Val::Real(2.2)], //x2
        ]);

        debug!("pi={pi}");
        assert!(
            Tensor::from_slice(&[1.85448311, 4.59934198, -4.30722048])
                .totype(Kind::Float)
                .allclose(&pi, 1e-02, 1e-02, false)
        );
    }
}

const TPE_LOSS_METRIC: &str = "loss";

trait TPEObjectiveResult: Debug {
    fn loss(&self) -> f32;
    fn constraint_loss(&self, constraint_name: &str) -> Option<f32>;
}

impl TPEObjectiveResult for f32 {
    fn loss(&self) -> f32 {
        *self
    }

    fn constraint_loss(&self, _constraint_name: &str) -> Option<f32> {
        None
    }
}

impl TPEObjectiveResult for HashMap<String, f32> {
    fn loss(&self) -> f32 {
        self
            .get(TPE_LOSS_METRIC)
            .copied()
            .unwrap_or(f32::MAX)
    }

    fn constraint_loss(&self, constraint_name: &str) -> Option<f32> {
        self.get(constraint_name).copied()
    }
}

struct AsyncTPEOptimizerImpl<F, T, R>
where
    F: AsyncFnMut(&[NamedVal<T>]) -> R,
    T: FieldName,
    R: TPEObjectiveResult,
{
    obj_func: F,
    cs: ConfigurationSpace<T>,
    constraints: HashMap<String, f32>,
    constraint_keys: Vec<String>,
    n_init: u32,
    max_evals: u32,
    tpe_samplers: HashMap<String, TreeStructuredParzenEstimator>,
    _pd: PhantomData<(T, R)>,
    descr: String
}

impl<F, T: FieldName, R> AsyncTPEOptimizerImpl<F, T, R>
where
    F: AsyncFnMut(&[NamedVal<T>]) -> R,
    R: TPEObjectiveResult,
{
    const LOSS_METRIC: &str = TPE_LOSS_METRIC;
    const N_EI_CANDIDATES: usize = 24;

    fn new(
        obj_func: F,
        cs: ConfigurationSpace<T>,
        constraints: HashMap<String, f32>,
        max_evals: u32,
        n_init: u32,
        descr: &str,
    ) -> Self {
        debug!("AsyncTPEOptimizerImpl:new: cs={cs:?} constraints={constraints:?}");

        let mut tpe_samplers = HashMap::from([(
            Self::LOSS_METRIC.to_string(),
            TreeStructuredParzenEstimator::new(&cs),
        )]);

        let mut constraint_keys: Vec<String> = constraints.keys().cloned().collect();
        constraint_keys.sort();

        for key in &constraint_keys {
            let upper_bound = constraints[key];
            tpe_samplers.insert(
                key.clone(),
                TreeStructuredParzenEstimator::new_with_threshold(&cs, upper_bound),
            );
        }

        Self {
            obj_func,
            cs,
            constraints,
            constraint_keys,
            n_init,
            max_evals,
            tpe_samplers,
            descr: descr.to_string(),
            _pd: PhantomData::default(),
        }
    }

    async fn optimize(&mut self) -> Vec<Val> {
        let mut best_loss = f32::MAX;
        let mut best_config: Option<Vec<Val>> = None;

        let grids_eta_guard = CollectEtaInvocationGuard::new(&format!("opt_{}", self.descr), self.max_evals as usize, 1);
        for i in 0..self.max_evals {
            let eval_config = if i < self.n_init {
                debug!("AsyncTPEOptimizerImpl::optimize: i={i} random sample");
                self.cs.random_sample()
            } else {
                debug!("AsyncTPEOptimizerImpl::optimize: i={i} sample");
                self.sample()
            };

            let eval_config_named = eval_config
                .iter()
                .zip(self.cs.hyperparameters.iter())
                .map(|(val, hp)| NamedVal::from((hp.name().clone(), val.clone())))
                .collect::<Vec<_>>();
            let results = self.obj_func.async_call_mut((&eval_config_named.as_slice(),)).await;
            let loss = results.loss();
            let is_satisfied = self.constraint_keys.iter().all(|constraint_name| {
                let upper_bound = self.constraints[constraint_name];
                upper_bound.is_infinite()
                    || results
                        .constraint_loss(constraint_name)
                        .map(|constraint_value| constraint_value <= upper_bound)
                        .unwrap_or(false)
            });
            let is_satisfied_opt = (!self.constraint_keys.is_empty()).then_some(is_satisfied);

            debug!("AsyncTPEOptimizerImpl::optimize: i={i} eval_config={eval_config:?} results={results:?} is_satisfied={is_satisfied}");

            if is_satisfied && loss < best_loss {
                best_loss = loss;
                best_config.replace(eval_config.clone());
            }

            self.tpe_samplers
                .get_mut(Self::LOSS_METRIC)
                .expect("loss sampler missing")
                .update_observations(loss, &eval_config, is_satisfied_opt);

            for constraint_name in &self.constraint_keys {
                let constraint_loss = results
                    .constraint_loss(constraint_name)
                    .unwrap_or(f32::INFINITY);
                self.tpe_samplers
                    .get_mut(constraint_name)
                    .expect("constraint sampler missing")
                    .update_observations(constraint_loss, &eval_config, None);
            }
            collect_eta_record_completion(grids_eta_guard.id());
        }

        best_config.unwrap_or(vec![]) //TODO
    }

    fn sample(&self) -> Vec<Val> {
        let mut candidate_sets = vec![
            self.tpe_samplers[Self::LOSS_METRIC].get_config_candidates(Self::N_EI_CANDIDATES),
        ];
        for constraint_name in &self.constraint_keys {
            if self.constraints[constraint_name].is_infinite() {
                continue;
            }

            candidate_sets.push(
                self.tpe_samplers[constraint_name].get_config_candidates(Self::N_EI_CANDIDATES),
            );
        }

        let config_cands = stack_params(candidate_sets);

        let mut pi_config = self.tpe_samplers[Self::LOSS_METRIC]
            .compute_probability_improvement(&config_cands);
        for constraint_name in &self.constraint_keys {
            if self.constraints[constraint_name].is_infinite() {
                continue;
            }

            pi_config += self.tpe_samplers[constraint_name].compute_probability_improvement(&config_cands);
        }

        trace!("AsyncTPEOptimizerImpl::sample: config_cands={config_cands:?}");
        trace!("AsyncTPEOptimizerImpl::sample: pi_config={pi_config}");

        let max_i = pi_config.argmax(0, false).double_value(&[]);
        config_cands
            .into_iter()
            .map(|p| p[max_i as usize].clone())
            .collect()
    }
}

#[allow(dead_code)]
pub struct AsyncTPEOptimizer<F, T>
where
    F: AsyncFnMut(&[NamedVal<T>]) -> f32,
    T: FieldName,
{
    inner: AsyncTPEOptimizerImpl<F, T, f32>,
}

impl<F, T: FieldName> AsyncTPEOptimizer<F, T>
where
    F: AsyncFnMut(&[NamedVal<T>]) -> f32,
{
    pub fn new(obj_func: F, cs: ConfigurationSpace<T>, max_evals: u32, n_init: u32, descr: &str) -> Self {
        debug!("AsyncTPEOptimizer:new: cs={cs:?}");

        Self {
            inner: AsyncTPEOptimizerImpl::new(
                obj_func,
                cs,
                HashMap::new(),
                max_evals,
                n_init,
                descr,
            ),
        }
    }

    pub async fn optimize(&mut self) -> Vec<Val> {
        self.inner.optimize().await
    }
}

#[allow(dead_code)]
pub struct AsyncCTPEOptimizer<F, T>
where
    F: AsyncFnMut(&[NamedVal<T>]) -> HashMap<String, f32>,
    T: FieldName,
{
    inner: AsyncTPEOptimizerImpl<F, T, HashMap<String, f32>>,
}

impl<F, T: FieldName> AsyncCTPEOptimizer<F, T>
where
    F: AsyncFnMut(&[NamedVal<T>]) -> HashMap<String, f32>,
{
    pub fn new(
        obj_func: F,
        cs: ConfigurationSpace<T>,
        constraints: HashMap<String, f32>,
        max_evals: u32,
        n_init: u32,
        descr: &str,
    ) -> Self {
        debug!("AsyncCTPEOptimizer:new: cs={cs:?} constraints={constraints:?}");

        Self {
            inner: AsyncTPEOptimizerImpl::new(
                obj_func,
                cs,
                constraints,
                max_evals,
                n_init,
                descr,
            ),
        }
    }

    pub async fn optimize(&mut self) -> Vec<Val> {
        self.inner.optimize().await
    }
}
fn stack_params<T: Clone>(input: Vec<Vec<Vec<T>>>) -> Vec<Vec<T>> {
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
    use crate::parzen::optimizer::Val::Int;
    use crate::parzen::optimizer::{AsyncCTPEOptimizer, AsyncTPEOptimizer, ConfigurationSpace, Val, stack_params};
    use log::debug;
    use std::collections::HashMap;
    use tch::Tensor;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::val::NamedVal;
    use crate::tho_tensor::ThoTensor;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_params_stack() {
        let params_0 = vec![vec![1., 2., 3.], vec![10., 20., 30.]];

        let params_1 = vec![vec![4., 5.], vec![40., 50.]];
        let r = stack_params(vec![params_0, params_1]);

        assert_eq!(
            vec![vec![1., 2., 3., 4., 5.], vec![10., 20., 30., 40., 50.],],
            r
        );
        // debug!("r={r:?}");
    }

    #[tokio::test]
    async fn test_sphere() {
        let sphere = sphere(30)(&vec![
            ("x".into(), Val::Real(0.1)).into(),
            ("x".into(), Val::Real(0.2)).into(),
            ("x".into(), Val::Real(0.3)).into(),
            ("x".into(), Val::Real(0.4)).into(),
            ("x".into(), Val::Real(0.5)).into(),
        ]);

        assert_eq!(4410.55, sphere);
    }

    #[tokio::test]
    async fn test_optimize_numeric() {
        let sphere = sphere(300);
        let mut opt = AsyncTPEOptimizer::new(
            async move |params| sphere(params),
            ConfigurationSpace::new(
                (0..3)
                    .map(|i| Hyperparameter::UniformFloat {
                        name: format!("x_{}", i).as_str().into(),
                        lower: -500.0,
                        upper: 500.0,
                    })
                    .collect(),
            ),
            500,
            10,
            "n/a",
        );

        let best_config = opt.optimize().await;

        debug!("best_config={best_config:?}");
        assert!((300. - <&Val as TryInto<f32>>::try_into(&best_config[0]).unwrap()).abs() < 3.);
        assert!((300. - <&Val as TryInto<f32>>::try_into(&best_config[1]).unwrap()).abs() < 3.);
        assert!((300. - <&Val as TryInto<f32>>::try_into(&best_config[2]).unwrap()).abs() < 3.);
    }

    #[tokio::test]
    async fn test_optimize_numeric_int() {
        let sphere = sphere(300);
        let mut opt = AsyncTPEOptimizer::new(
            async move |params| sphere(params),
            ConfigurationSpace::new(
                (0..3)
                    .map(|i| Hyperparameter::UniformInt {
                        name: format!("x_{}", i).as_str().into(),
                        lower: -500,
                        upper: 500,
                    })
                    .collect(),
            ),
            500,
            10,
            "n/a",
        );

        let best_config = opt.optimize().await;

        debug!("best_config={best_config:?}");
        assert!((300. - <&Val as TryInto<f32>>::try_into(&best_config[0]).unwrap()).abs() < 3.);
        assert!((300. - <&Val as TryInto<f32>>::try_into(&best_config[1]).unwrap()).abs() < 3.);
        assert!((300. - <&Val as TryInto<f32>>::try_into(&best_config[2]).unwrap()).abs() < 3.);
    }

    #[tokio::test]
    async fn test_optimize_categoric() {
        let mut opt: AsyncTPEOptimizer<_, String> = AsyncTPEOptimizer::new(
            async move |params| {
                debug!("params={params:?}");

                let x_0: f32 = (&params[0]).try_into().unwrap();
                let x_1: f32 = (&params[1]).try_into().unwrap();
                let x_2: u32 = (&params[2]).try_into().unwrap();

                let loss = (x_0 - 30.).powi(2) + (x_1 - 30.).powi(2);
                let loss = match x_2 {
                    0 => loss * 3., //TODO used to find min at cat 1 with 0=1.5 when upper tpe in update_observations was built from lower vals
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
                    Hyperparameter::UniformFloat {
                        name: "x_0".into(),
                        lower: -50.0,
                        upper: 50.0,
                    },
                    Hyperparameter::UniformFloat {
                        name: "x_1".into(),
                        lower: -50.0,
                        upper: 50.0,
                    },
                    Hyperparameter::Categorical {
                        name: "cat".into(),
                        values: vec!["a".into(), "b".into(), "c".into()],
                    },
                ],
            ),
            500,
            10,
            "n/a",
        );

        let best_config = opt.optimize().await;

        debug!("best_config={best_config:?}");
        assert!((30. - <&Val as TryInto<f32>>::try_into(&best_config[0]).unwrap()).abs() < 3.);
        assert!((30. - <&Val as TryInto<f32>>::try_into(&best_config[1]).unwrap()).abs() < 3.);
        assert_eq!(Int(1), best_config[2]);
    }

    #[tokio::test]
    async fn test_optimize_constrained() {
        let mut opt: AsyncCTPEOptimizer<_, String> = AsyncCTPEOptimizer::new(
            async move |params| {
                let x: f32 = (&params[0]).try_into().unwrap();
                let y: f32 = (&params[1]).try_into().unwrap();
                let loss = (x - 3.).powi(2) + (y - 3.).powi(2);
                HashMap::from([
                    ("loss".to_string(), loss),
                    ("c1".to_string(), x),
                    ("c2".to_string(), y),
                ])
            },
            ConfigurationSpace::new(vec![
                Hyperparameter::UniformFloat {
                    name: "x".into(),
                    lower: -5.0,
                    upper: 5.0,
                },
                Hyperparameter::UniformFloat {
                    name: "y".into(),
                    lower: -5.0,
                    upper: 5.0,
                },
            ]),
            HashMap::from([("c1".to_string(), 0.0), ("c2".to_string(), 0.0)]),
            300,
            10,
            "n/a",
        );

        let best_config = opt.optimize().await;

        assert!(!best_config.is_empty());
        assert!((<&Val as TryInto<f32>>::try_into(&best_config[0]).unwrap()) <= 1.0);
        assert!((<&Val as TryInto<f32>>::try_into(&best_config[1]).unwrap()) <= 1.0);
    }

    fn sphere(d_x: i64) -> impl Fn(&[NamedVal<String>]) -> f32 {
        move |params| {
            debug!("params={params:?}");

            let t: Tensor = Tensor::from_slice(
                params
                    .into_iter()
                    .map(|v| <&NamedVal<String> as TryInto<f64>>::try_into(v).unwrap())
                    .collect::<Vec<f64>>()
                    .as_slice(),
            );
            let loss = t
                .subtract_scalar(d_x)
                .pow_tensor_scalar(2)
                .sum(None)
                .double_value(&[]) as f32;

            debug!("loss={loss:?}");

            loss
        }
    }

}

#[derive(Debug, Clone)]
pub struct ConfigurationSpace<T: FieldName> {
    hyperparameters: Vec<Hyperparameter<T>>,
}

impl<T: FieldName> From<&[Hyperparameter<T>]> for ConfigurationSpace<T> {
    fn from(value: &[Hyperparameter<T>]) -> Self {
        ConfigurationSpace::new(value.to_vec())
    }
}

impl<T: FieldName> From<&ConfigurationSpace<T>> for HashMap<String, Val> {
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
                        map.insert(format!("{i}_{}", name), (*val).into());
                    }
                }
            }
        }
        map
    }
}

#[allow(dead_code)]
impl<T: FieldName> ConfigurationSpace<T> {
    pub fn empty() -> Self {
        Self {
            hyperparameters: vec![],
        }
    }
    pub fn new(hyperparameters: Vec<Hyperparameter<T>>) -> Self {
        Self { hyperparameters }
    }

    pub fn random_sample(&self) -> Vec<Val> {
        self.hyperparameters
            .iter()
            .map(|hp| hp.random_sample())
            .collect()
    }

    fn len(&self) -> usize {
        self.hyperparameters.len()
    }

    fn iter(&self) -> CsIter<'_, T> {
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
