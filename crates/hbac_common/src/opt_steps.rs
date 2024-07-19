use std::collections::HashMap;
use std::ops::Range;

use tch::nn::Module;
use tch::Tensor;

use hbac_shared::val::Val;
use log::debug;

struct EnvParams {

}

impl EnvParams {
    pub fn new() -> Self {
        Self {}
    }
}

struct OptParams {

}

impl OptParams {
    pub fn new() -> Self {
        Self {}
    }
}




trait Apply {
    fn apply(&self, t:Box<Tensor>) -> Box<Tensor>;
}


struct ApplyComp<'a> {
    applies:Vec<&'a Box<dyn Apply>>
}

impl <'a> ApplyComp<'a> {
    pub fn new(applies: Vec<&'a Box<dyn Apply>>) -> Self {
        Self { applies }
    }

    pub fn from_map(value: Option<&'a RunIdsStack>, applies_map:&'a mut HashMap<usize, Box<dyn Apply>>) -> Self {
        Self {
            applies: value.map(|run_ids_stack|run_ids_stack.run_ids.iter().map(|run_id|applies_map.get(run_id).expect("applies_map.get(run_id)")).collect::<Vec<_>>()).unwrap_or(vec![])
        }
    }
}
#[allow(dead_code)]
impl Apply for ApplyComp<'_> {
    fn apply(&self, t:Box<Tensor>) -> Box<Tensor> {

        let mut t = t;
        for apply in &self.applies {
            t = apply.apply(t)
        }

        t
    }
}
#[allow(dead_code)]
struct ApplyNet {
    net: Box<dyn Module>,
}

impl ApplyNet {
    pub fn new(net: Box<dyn Module>) -> Self {
        Self { net }
    }
}
#[allow(dead_code)]
impl Apply for ApplyNet {

    fn apply(&self, t: Box<Tensor>) -> Box<Tensor> {
        let out = self.net.forward(&t);

        debug!("apply 1 in size={:?}  out size()={:?}", t.size(), out.size());

        Box::new(out)
    }
}

#[allow(dead_code)]
struct OptStep {
    name:String,
    build:Box<dyn Fn(&HashMap<String, Val>, &Vec<String>, &dyn Apply)->usize>,//returns run_ids
    exec:fn(&Box<Tensor>, &HashMap<String, Val>)->Box<Tensor>,
    apply_fn: Option<Box<dyn Fn(&HashMap<String, Val>)->Box<dyn Apply>>>,
    data_keys:Vec<String>,

}
#[derive(Debug)]
struct RunIdsStack {
    run_ids:Vec<usize>
}

impl RunIdsStack {
    pub fn new(run_id: usize, prev:Option<&RunIdsStack>) -> Self {
        let mut run_ids = vec![];

        if let Some(prev_run_ids) = prev {
            run_ids.extend(prev_run_ids.run_ids.clone());
        }

        run_ids.push(run_id);

        Self {
            run_ids
        }
    }
}
#[allow(dead_code)]
impl OptStep {

    fn new(
        name:impl Into<String>,
        // build_params:HashMap<String, Range<Val>>,
        build:Box<dyn Fn(&HashMap<String, Val>, &Vec<String>,&dyn Apply)->usize>,
        exec:fn(&Box<Tensor>, &HashMap<String, Val>)->Box<Tensor>,
        apply_fn: Option<Box<dyn Fn(&HashMap<String, Val>)->Box<dyn Apply>>>,
        data_keys:Vec<String>
    )->Self {
        OptStep {
            name: name.into(),
            // build_params,
            build,
            exec,
            apply_fn,
            data_keys
        }
    }

    fn search(&self, opt_props:&HashMap<String, Range<Val>>, prev_run_ids_op: Option<Vec<RunIdsStack>>, _env_params: &EnvParams, applies_map:&mut HashMap<usize, Box<dyn Apply>>, tries: usize) -> Vec<RunIdsStack> {
        debug!("search: prev_run_ids_op={:?}", prev_run_ids_op);


        let mut best_run_ids = vec![];


        for t in 0..tries {
            let best_prev_run_ids = prev_run_ids_op.as_ref().and_then(|prev_run_ids|Some(&prev_run_ids[t]));


            let try_props = opt_props.iter().fold(HashMap::new(), |mut a:HashMap<String, Val>, prop_valrange|{
                // a.insert(prop_valrange.0.clone(), prop_valrange.1.start.clone());
                a.insert(prop_valrange.0.clone(), (10 * (1 + t)).into());

                a
            });

            let best_run_id = (self.build)(&try_props, &self.data_keys, &ApplyComp::from_map(best_prev_run_ids, applies_map) as &dyn Apply);

            debug!("applies_map.insert: best_run_id={}  best_prev_run_ids={:?} try_props={:?}", best_run_id, best_prev_run_ids, try_props);
            applies_map.insert(best_run_id, self.apply_fn.as_ref().map(|a|a(&try_props)).expect("apply_fn"));

            best_run_ids.push(RunIdsStack::new(best_run_id, best_prev_run_ids))
        }


        best_run_ids
    }

}


#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::rc::Rc;

    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::nn::{Module, VarStore};
    use tch::nn::AdamW;
    use tch::nn::OptimizerConfig;
    use tch::Reduction;
    use tch::Tensor;

    use hbac_shared::val::Val;

    use crate::builder::SeqBuilder;
    use crate::opt_steps::{Apply, ApplyComp, EnvParams};
    use crate::opt_steps::OptStep;
    use crate::opt_steps::RunIdsStack;
    use crate::utils::DataBatch;
    use crate::utils::train;
    use log::debug;

    #[test]
    fn test_two_nets(){



        let _props_store:HashMap<usize, String> = HashMap::new();

        let run_id = Rc::new(RefCell::new(0usize));
        let run_id_build_1 = Rc::clone(&run_id);
        let run_id_build_2 = Rc::clone(&run_id);


        let steps: Vec<OptStep> = vec![
            OptStep::new(
                "net1",
                Box::new(move |params: &HashMap<String, Val>, data_keys: &Vec<String>, _apply: &dyn Apply| {
                    debug!("params={:?} data_keys={:?}", params, data_keys);
                    let mut vs = VarStore::new(Cpu);
                    let out_features = params.get("window_1").expect("window_1").as_i64().expect("window_1 as i64");

                    let node = build_net(params, &mut vs, 100, out_features);

                    let _error = train(
                        &Cpu,
                        &mut crate::simp::DummyDs::new(
                            Tensor::rand(100, FLOAT_CPU),
                            Tensor::rand(100, FLOAT_CPU),
                        ) as &mut dyn DataBatch,
                        &node,
                        |pred: &Tensor, target: &Tensor| pred.sigmoid().binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean),
                        1,
                        AdamW::default().build(&vs, 1.0e-4).unwrap(),
                        |t| t,
                    ).expect("train");


                    *run_id_build_1.borrow_mut() += 1;
                    *run_id_build_1.borrow()
                }),
                |t: &Box<Tensor>, _params: &HashMap<String, Val>| {
                    Box::new(t.shallow_clone())
                },
                Some(Box::new(|params: &HashMap<String, Val>| {
                    debug!("new apply 1, params={:?}", params);

                    let mut vs = VarStore::new(Cpu);
                    let out_features = params.get("window_1").expect("window_1").as_i64().expect("window_1 as i64");
                    Box::new(super::ApplyNet::new(build_net(params, &mut vs, 100, out_features)))
                })),
                vec!["aaa".to_string(), "bbb".to_string()],
            ),
            OptStep::new(
                "net2",
                Box::new(move |params: &HashMap<String, Val>, data_keys: &Vec<String>, apply: &dyn Apply| {
                    debug!("params={:?} data_keys={:?}", params, data_keys);
                    let mut vs = VarStore::new(Cpu);
                    let in_features = params.get("window_1").expect("window_1").as_i64().expect("window_1 as i64");
                    let out_features = params.get("window_2").expect("window_2").as_i64().expect("window_2 as i64");

                    let node = build_net(params, &mut vs, in_features, out_features);

                    let _error = train(
                        &Cpu,
                        &mut crate::simp::DummyDs::new(
                            *apply.apply(Box::new(Tensor::rand(100, FLOAT_CPU))),
                            Tensor::rand(100, FLOAT_CPU),
                        ) as &mut dyn DataBatch,
                        &node,
                        |pred: &Tensor, target: &Tensor| pred.sigmoid().binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean),
                        1,
                        AdamW::default().build(&vs, 1.0e-4).unwrap(),
                        |t| t,
                    ).expect("train");


                    *run_id_build_2.borrow_mut() += 1;
                    *run_id_build_2.borrow()
                }),
                |t: &Box<Tensor>, _params: &HashMap<String, Val>| {
                    Box::new(t.shallow_clone())
                },
                Some(Box::new(|params: &HashMap<String, Val>| {
                    debug!("new apply2, params={:?}", params);

                    let mut vs = VarStore::new(Cpu);
                    let in_features = params.get("window_1").expect("window_1").as_i64().expect("window_1 as i64");
                    let out_features = params.get("window_2").expect("window_2").as_i64().expect("window_2 as i64");

                    Box::new(super::ApplyNet::new(build_net(params, &mut vs, in_features, out_features)))
                })),
                vec!["aaa".to_string(), "bbb".to_string()],
            )
        ];

        let env_params = EnvParams::new();

        let mut run_ids :Option<Vec<RunIdsStack>> = None;

        let mut applies_map:HashMap<usize, Box<dyn Apply>> = HashMap::new();

        let opt_props = HashMap::from([
            ("window_1".to_string(), Val::Int(10)..Val::Int(100)),
            ("window_2".to_string(), Val::Int(10)..Val::Int(100))
        ]);

        for (step_i, step) in steps.into_iter().enumerate() {
            debug!("step_i={}", step_i);
            let next_run_ids = step.search(&opt_props, run_ids.take(), &env_params, &mut applies_map, 3);
            run_ids.replace(next_run_ids);

        }

        debug!("test: run_ids={:?}", run_ids);

        let pred_0 = ApplyComp::from_map(run_ids.as_ref().and_then(|v|v.get(0)), &mut applies_map).apply(Box::new(Tensor::rand(100, FLOAT_CPU)));
        let pred_1 = ApplyComp::from_map(run_ids.as_ref().and_then(|v|v.get(1)), &mut applies_map).apply(Box::new(Tensor::rand(100, FLOAT_CPU)));
        let pred_2 = ApplyComp::from_map(run_ids.as_ref().and_then(|v|v.get(2)), &mut applies_map).apply(Box::new(Tensor::rand(100, FLOAT_CPU)));

        debug!("test: pred_0.size()={:?}  pred_1.size()={:?}, pred_2.size()={:?}", pred_0.size(), pred_1.size(), pred_2.size());

        assert_eq!(vec![10], pred_0.size());
        assert_eq!(vec![20], pred_1.size());
        assert_eq!(vec![30], pred_2.size());
    }

    fn build_net(_params: &HashMap<String, Val>, vs: &mut VarStore, in_features: i64, out_features: i64) -> Box<dyn Module> {

        debug!("build_net_1: in_features={} out_features={}", in_features, out_features);

        SeqBuilder::new(in_features)
            .linear_s(out_features)
            .end_seq()
            .build_tch_module(&vs.root()).expect("build_tch_module")
    }


}

