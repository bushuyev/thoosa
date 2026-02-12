use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
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

pub trait GridRunConfigSource {
    type P: FieldName;

    fn grid_run_configs(
        &self,
    ) -> Result<thoosa_shared::run_configs::GridRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>>, ThoError>;
}
