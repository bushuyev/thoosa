use crate::parzen::optimizer::ConfigurationSpace;

pub trait Params {
    fn to_run_configs(&self) ->Vec<impl RunConfig>;
}

pub trait RunConfig {
    fn config_space(&self) -> Option<&ConfigurationSpace> {
        None
    }
}
