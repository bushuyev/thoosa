use std::collections::HashMap;
use std::ops::Range;
use tch::Tensor;
use hbac_shared::val::Val;
use tch::nn::VarStore;
enum Data {
    Tensors,
    TensorsSlice,
    Model
}



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

trait BuildParams {

}

impl BuildParams for Box<dyn BuildParams> {

}


trait Apply {
    fn apply(&self, t:&Box<Tensor>) -> Box<Tensor>;
}


struct CompApply<'a> {
    applies:Vec<&'a Box<dyn Apply>>
}

impl <'a> CompApply<'a> {
    pub fn new(applies: Vec<&'a Box<dyn Apply>>) -> Self {
        println!("CompApply::new applies.len()={}", applies.len());
        Self { applies }
    }
}

impl Apply for CompApply<'_> {
    fn apply(&self, t:&Box<Tensor>) -> Box<Tensor> {
        // Box::new(t.shallow_clone())
        let mut t = Box::new(t.shallow_clone());
        for apply in &self.applies {
            println!("Apply for CompApply apply *************");
            t = apply.apply(&t)
        }

        t
    }
}


struct OptStep {
    name:String,
    build_params:HashMap<String, Range<Val>>,
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
        let mut run_ids = vec![run_id];

        if let Some(prev_run_ids) = prev {
            run_ids.extend(prev_run_ids.run_ids.clone());
        }

        Self {
            run_ids
        }
    }
}

impl OptStep {

    fn new(
        name:impl Into<String>,
        build_params:HashMap<String, Range<Val>>,
        build:Box<dyn Fn(&HashMap<String, Val>, &Vec<String>,&dyn Apply)->usize>,
        exec:fn(&Box<Tensor>, &HashMap<String, Val>)->Box<Tensor>,
        apply_fn: Option<Box<dyn Fn(&HashMap<String, Val>)->Box<dyn Apply>>>,
        data_keys:Vec<String>
    )->Self {
        OptStep {
            name: name.into(),
            build_params,
            build,
            exec,
            apply_fn,
            data_keys
        }
    }

    fn search(&self, prev_run_ids_op: Option<Vec<RunIdsStack>>, env_params: &EnvParams, applies_map:&mut HashMap<usize, Box<dyn Apply>>) -> Vec<RunIdsStack> {
        println!("search: prev_run_ids_op={:?}", prev_run_ids_op);
        // (self.build)(&self.build_params);
        let tries = 3;

        let mut best_run_ids = vec![];


        for t in 0..tries {
            let best_prev_run_ids = prev_run_ids_op.as_ref().and_then(|prev_run_ids|Some(&prev_run_ids[t]));


            let try_props = self.build_params.iter().fold(HashMap::new(), |mut a:HashMap<String, Val>, prop_valrange|{
                // a.insert(prop_valrange.0.clone(), prop_valrange.1.start.clone());
                a.insert(prop_valrange.0.clone(), (10 * (1 + t)).into());

                a
            });



            let best_run_id = (self.build)(&try_props, &self.data_keys, &CompApply::new(
                best_prev_run_ids.map(|run_ids_stack|run_ids_stack.run_ids.iter().map(|run_id|applies_map.get(run_id).expect("applies_map.get(run_id)")).collect::<Vec<_>>()).unwrap_or(vec![])
            ) as &dyn Apply);

            println!("applies_map.insert: best_run_id={}  best_prev_run_ids={:?} try_props={:?}", best_run_id, best_prev_run_ids, try_props);
            applies_map.insert(best_run_id, self.apply_fn.as_ref().map(|a|a(&try_props)).expect("apply_fn"));

            best_run_ids.push(RunIdsStack::new(best_run_id, best_prev_run_ids))
        }


        best_run_ids
    }

    fn exec(&self, env_params: EnvParams, opt_params: OptParams) -> Option<f64>{
        None
    }
}

fn xxx(){
    println!("xxx")
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ops::Range;
    use tch::Device::Cpu;
    use tch::Kind::Float;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use hbac_shared::val::Val;
    use crate::builder::SeqBuilder;
    use crate::opt_steps::{Apply, BuildParams, EnvParams, OptParams, xxx};
    use crate::opt_steps::OptStep;
    use crate::opt_steps::RunIdsStack;
    use tch::nn::{Module, VarStore};
    use crate::utils::train;
    use crate::utils::DataBatch;
    use tch::nn::AdamW;
    use tch::nn::OptimizerConfig;
    use tch::Reduction;
    use std::rc::Rc;
    use std::cell::RefCell;

    #[test]
    fn test_xxx(){

        struct TestBulidParams {
            window:Range<i64>
        }

        impl BuildParams for TestBulidParams {

        }

        struct TestBulidParams2 {

        }

        impl BuildParams for TestBulidParams2 {

        }

        struct TestBulidParams3 {

        }

        impl BuildParams for TestBulidParams3 {

        }

        let mut props_store:HashMap<usize, String> = HashMap::new();

        let mut run_id = Rc::new(RefCell::new(0usize));
        let mut run_id_build_1 = Rc::clone(&run_id);
        let mut run_id_build_2 = Rc::clone(&run_id);

        let mut steps:Vec<OptStep>  = vec![
            OptStep::new(
                "net1",
                HashMap::from([
                    ("window".to_string(), Val::Int(10)..Val::Int(100))
                ]),
                Box::new(move |params:&HashMap<String, Val>, data_keys:&Vec<String>, apply:&dyn Apply| {
                    println!("params={:?} data_keys={:?}", params, data_keys);
                    let mut vs = VarStore::new(Cpu);

                    let node = build_net_1(params, &mut vs);

                    train(
                        &Cpu,
                        &mut crate::simp::DummyDs::new(
                            Tensor::rand((100), FLOAT_CPU),
                            Tensor::rand((100), FLOAT_CPU),
                        ) as &mut dyn DataBatch,
                        &node,
                        |pred: &Tensor, target: &Tensor|  pred.sigmoid().binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean),
                        1,
                        AdamW::default().build(&vs, 1.0e-4).unwrap(),
                        |t|t
                    );


                    *run_id_build_1.borrow_mut()+=1;
                    *run_id_build_1.borrow()
                }),
                |t: &Box<Tensor>, params:&HashMap<String, Val>| {
                    Box::new(t.shallow_clone())
                },
                Some(Box::new(|params:&HashMap<String, Val>|{
                    println!("new apply 1, params={:?}", params);

                    struct Net1Apply {
                        net:Box<dyn Module>
                    }

                    impl Apply for Net1Apply {
                        fn apply(&self, t:&Box<Tensor>) -> Box<Tensor> {

                            let out = self.net.forward(&t);

                            println!(">>>>>>>>>>>>>>>> apply 1  t.size()={:?}  out.size()={:?}", t.size(), out.size());

                            Box::new(out)
                        }
                    }

                    let mut vs = VarStore::new(Cpu);

                    Box::new(Net1Apply {
                        net: build_net_1(params, &mut vs)
                    })
                })),
                vec!["aaa".to_string(), "bbb".to_string()]
            ),
            OptStep::new(
                "net2",
                HashMap::from([
                    ("window".to_string(), Val::Int(10)..Val::Int(100))
                ]),
                Box::new(move |params:&HashMap<String, Val>, data_keys:&Vec<String>, apply:&dyn Apply| {
                    println!("params={:?} data_keys={:?}", params, data_keys);
                    let mut vs = VarStore::new(Cpu);

                    let node = build_net_2(params, &mut vs);

                    train(
                        &Cpu,
                        &mut crate::simp::DummyDs::new(
                            *apply.apply(&Box::new(Tensor::rand((100), FLOAT_CPU))),
                            Tensor::rand((100), FLOAT_CPU),
                        ) as &mut dyn DataBatch,
                        &node,
                        |pred: &Tensor, target: &Tensor|  pred.sigmoid().binary_cross_entropy::<Tensor>(&target, None, Reduction::Mean),
                        1,
                        AdamW::default().build(&vs, 1.0e-4).unwrap(),
                        |t|t
                    );


                    *run_id_build_2.borrow_mut()+=1;
                    *run_id_build_2.borrow()
                }),
                |t: &Box<Tensor>, params:&HashMap<String, Val>| {
                    Box::new(t.shallow_clone())
                },
                Some(Box::new(|params:&HashMap<String, Val>|{
                    println!("new apply 1, params={:?}", params);

                    struct Net1Apply {
                        net:Box<dyn Module>
                    }

                    impl Apply for Net1Apply {
                        fn apply(&self, t:&Box<Tensor>) -> Box<Tensor> {
                            println!(">>>>>>>>>>>>>>>> apply 2");
                            Box::new(self.net.forward(&t))
                        }
                    }

                    let mut vs = VarStore::new(Cpu);

                    Box::new(Net1Apply {
                        net: build_net_1(params, &mut vs)
                    })
                })),
                vec!["aaa".to_string(), "bbb".to_string()]
            )        ];

        let env_params = EnvParams::new();

        let mut run_ids :Option<Vec<RunIdsStack>> = None;

        let mut applies_map:HashMap<usize, Box<dyn Apply>> = HashMap::new();
        for (step_i, step) in steps.into_iter().enumerate() {
            println!("step_i={}", step_i);
            let next_run_ids = step.search(run_ids.take(), &env_params, &mut applies_map);
            run_ids.replace(next_run_ids);

        }

        println!("run_ids={:?}", run_ids);

    }

    fn build_net_1(params: &HashMap<String, Val>, vs: &mut VarStore) -> Box<dyn Module> {
        let features = params.get("window").expect("window").as_i64().expect("window as i64");
        println!("build_net_1: features={}", features);

        SeqBuilder::new(100)
            .linear_s(features)
            .end_seq()
            .build_tch_module(&vs.root()).expect("build_tch_module")
    }

    fn build_net_2(params: &HashMap<String, Val>, vs: &mut VarStore) -> Box<dyn Module> {
        let features = params.get("window").expect("window").as_i64().expect("window as i64");
        println!("build_net_1: features={}", features);

        SeqBuilder::new(features)
            .linear_s(10)
            .end_seq()
            .build_tch_module(&vs.root()).expect("build_tch_module")
    }

}

