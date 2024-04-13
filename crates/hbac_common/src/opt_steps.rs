use std::collections::HashMap;
use std::ops::Range;
use tch::Tensor;
use hbac_shared::val::Val;

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



struct OptStep {
    name:String,
    build_params:HashMap<String, Range<Val>>,
    build:Box<dyn Fn(&HashMap<String, Val>, &Vec<String>)>,
    exec:Box<dyn Fn(&Box<Tensor>, &HashMap<String, Val>)->Box<Tensor>>,
    data_keys:Vec<String>
}



impl OptStep {

    fn new(
        name:impl Into<String>,
        build_params:HashMap<String, Range<Val>>,
        build:Box<dyn Fn(&HashMap<String, Val>, &Vec<String>)>,
        exec:Box<dyn Fn(&Box<Tensor>, &HashMap<String, Val>)->Box<Tensor>>,
        data_keys:Vec<String>
    )->Self {
        OptStep {
            name: name.into(),
            build_params,
            build,
            exec,
            data_keys
        }
    }

    fn search(&self, run_id:usize, env_params: &EnvParams) -> Option<f64>{
        // (self.build)(&self.build_params);
        let tries = 10;

        for t in 0..tries {
            let try_props = self.build_params.iter().fold(HashMap::new(), |mut a:HashMap<String, Val>, prop_valrange|{
                a.insert(prop_valrange.0.clone(), prop_valrange.1.start.clone());

                a
            });

            (self.build)(&try_props, &self.data_keys);
        }

        Some(0.1)
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
    use tch::Tensor;
    use hbac_shared::val::Val;
    use crate::opt_steps::{BuildParams, EnvParams, OptParams, xxx};
    use crate::opt_steps::OptStep;

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

        let mut steps:Vec<OptStep>  = vec![
            OptStep::new(
                "find_motifs",
                HashMap::from([
                    ("window".to_string(), Val::Int(100)..Val::Int(1000))
                ]),
                Box::new(|params:&HashMap<String, Val>, data_keys:&Vec<String>| {
                    println!("params={:?} data_keys={:?}", params, data_keys);
                }),
                Box::new(|t: &Box<Tensor>, params:&HashMap<String, Val>| {
                    Box::new(t.shallow_clone().multiply(&Tensor::from_slice(&[1.])))
                }),
                vec!["aaa".to_string(), "bbb".to_string()]
            ),
            OptStep::new(
                "select_motifs",
                HashMap::from([
                    ("min_dist".to_string(), Val::Real(100.)..Val::Real(1000.0))
                ]),
                Box::new(|params:&HashMap<String, Val>, data_keys:&Vec<String>| {
                    println!("params={:?}", params);
                }),
                Box::new(|t: &Box<Tensor>, params:&HashMap<String, Val>| {
                    Box::new(t.shallow_clone().multiply(&Tensor::from_slice(&[1.])))
                }),
                vec!["aaa".to_string(), "bbb".to_string()]
            )
        ];

        let env_params = EnvParams::new();

        for step in steps {

            step.search(1, &env_params);
        }
        // find_motifs_step.build(1, EnvParams::new());
        // find_motifs_step.go(EnvParams::new(), OptParams::new());
    }
}

