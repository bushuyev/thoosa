use std::fmt::Debug;
use std::collections::HashMap;
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::val::{NamedRange, NamedVal, Val};


pub trait RunConfigRangeOrValue {

}
pub trait OptRunConfigSource {
    type P:FieldName;
    fn opt_run_configs(&self) -> thoosa_shared::run_configs::OptRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>, Vec<Hyperparameter<Self::P>>, Self::P>;
}


pub trait GridRunConfigSource {
    type P:FieldName;

    fn grid_run_configs(&self) -> thoosa_shared::run_configs::GridRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>>;
}

//TODO move to thoosa_run_config_args::
pub trait RunConfig:Debug {

    fn to_properties_map_(&self) -> HashMap<String, Val> {
        HashMap::new()//TODO
    }

    fn from_properties_map_(run_id: RunId, properties: HashMap<String, Val>) -> Self;

    fn set_id(&mut self, id: RunId);

    fn type_name()->&'static str;
}