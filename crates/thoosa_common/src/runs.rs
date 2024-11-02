use crate::parzen::optimizer::ConfigurationSpace;

pub trait Params {
    fn to_run_configs(&self) ->Vec<impl RunConfig>;
}

pub trait RunConfig {
    fn to_config_space(&self) -> Option<ConfigurationSpace> {
        None
    }
}
