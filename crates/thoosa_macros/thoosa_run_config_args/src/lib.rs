use std::fmt::Debug;
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::resource::ResourcesConfig;
use thoosa_shared::run_configs::{GridRunConfigs, OptRunConfigs};
use thoosa_shared::val::{NamedRange, NamedVal};

pub trait OptRunConfigSource {
    type P:FieldName;
    fn opt_run_configs(&self) -> OptRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>, Vec<Hyperparameter<Self::P>>, Self::P>;
}


pub trait GridRunConfigSource {
    type P:FieldName;

    fn grid_run_configs(&self) -> GridRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>>;
}