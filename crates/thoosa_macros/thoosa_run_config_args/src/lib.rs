use std::sync::Arc;
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::run_configs::{GridRunConfigs, OptRunConfigs};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::ThoError;

pub trait RunConfigRangeOrValue {}
pub trait OptRunConfigSource {
    type P: FieldName;
    fn opt_run_configs(
        &self,
    ) -> Result<thoosa_shared::run_configs::OptRunConfigs<
        Vec<NamedVal<Self::P>>,
        Vec<NamedRange<Self::P>>,
        Vec<Hyperparameter<Self::P>>,
        Self::P,
    >, ThoError>;
}

impl <G:OptRunConfigSource> OptRunConfigSource for Arc<G> {
    type P = G::P;

    fn opt_run_configs(&self) -> Result<OptRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>, Vec<Hyperparameter<Self::P>>, Self::P>, ThoError> {
        self.as_ref().opt_run_configs()
    }
}

pub trait GridRunConfigSource {
    type P: FieldName;

    fn grid_run_configs(
        &self,
    ) -> Result<thoosa_shared::run_configs::GridRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>>, ThoError>;
}


impl <G:GridRunConfigSource> GridRunConfigSource for Arc<G> {
    type P = G::P;

    fn grid_run_configs(&self) -> Result<GridRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>>, ThoError> {
        self.as_ref().grid_run_configs()
    }
}