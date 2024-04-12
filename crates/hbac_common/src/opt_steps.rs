use tch::Tensor;

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

struct OptStep<BP:BuildParams> {
    name:String,
    build_params:Box<BP>,
    build:Box<dyn Fn(&BP)>,
    exec:Box<dyn Fn(&Box<Tensor>)->Box<Tensor>>,
    next:Option<Box<OptStep<Box<dyn BuildParams>>>>
}

impl <BP:BuildParams> OptStep <BP> {

    fn new(name:impl Into<String>, build_params:Box<BP>, build:Box<dyn Fn(&BP)>,  exec:Box<dyn Fn(&Box<Tensor>)->Box<Tensor>>, next:Option<Box<OptStep<Box<dyn BuildParams>>>>)->Self {
        OptStep {
            name: name.into(),
            build_params,
            next,
            build,
            exec
        }
    }

    fn build(&self, env_params: EnvParams) -> Option<f64>{
        (self.build)(&self.build_params);

        Some(0.1)
    }

    fn go(&self, env_params: EnvParams, opt_params: OptParams) -> Option<f64>{
        None
    }
}

fn xxx(){
    println!("xxx")
}

#[cfg(test)]
mod tests {
    use tch::Tensor;
    use crate::opt_steps::{BuildParams, EnvParams, OptParams, xxx};
    use crate::opt_steps::OptStep;

    #[test]
    fn test_xxx(){

        struct TestBulidParams {

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

        let mut find_motifs_step = OptStep::new(
            "find_motifs",
            Box::new(TestBulidParams{}),
            Box::new(|bp|{

            }),
            Box::new(|t:&Box<Tensor>|{
                Box::new(t.shallow_clone().multiply(&Tensor::from_slice(&[1.])))
            }),
            Some(Box::new(
                OptStep::new(
                    "select_motifs",
                    Box::new(Box::new(TestBulidParams2 {

                    })),
                    Box::new(|bp|{

                    }),
                    Box::new(|t:&Box<Tensor>|{
                        Box::new(t.shallow_clone())
                    }),
                    Some(Box::new(
                        OptStep::new(
                            "select_motifs3",
                            Box::new(Box::new(TestBulidParams3 {

                            })),
                            Box::new(|bp|{

                            }),
                            Box::new(|t:&Box<Tensor>|{
                                Box::new(t.shallow_clone())
                            }),
                            None
                        )
                    ))

                )
            ))
        );

        find_motifs_step.build(EnvParams::new());
        // find_motifs_step.go(EnvParams::new(), OptParams::new());
    }
}