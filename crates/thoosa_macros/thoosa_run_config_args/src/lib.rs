use std::sync::Arc;
use thoosa_shared::hyperparameter::FieldName;
use thoosa_shared::run_configs::GridRunConfigs;
use thoosa_shared::val::{NamedRange, NamedVal};

pub trait RunConfigRangeOrValue {}
pub trait OptRunConfigSource {
    type P: FieldName;
    fn opt_run_configs(&self) -> anyhow::Result<thoosa_shared::run_configs::TrainingRunConfigs<Self::P>>;
}

impl<G: OptRunConfigSource> OptRunConfigSource for Arc<G> {
    type P = G::P;

    fn opt_run_configs(&self) -> anyhow::Result<thoosa_shared::run_configs::TrainingRunConfigs<Self::P>> {
        self.as_ref().opt_run_configs()
    }
}

pub trait GridRunConfigSource {
    type P: FieldName;

    fn grid_run_configs(&self) -> anyhow::Result<thoosa_shared::run_configs::GridRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>>>;
}

impl<G: GridRunConfigSource> GridRunConfigSource for Arc<G> {
    type P = G::P;

    fn grid_run_configs(&self) -> anyhow::Result<GridRunConfigs<Vec<NamedVal<Self::P>>, Vec<NamedRange<Self::P>>>> {
        self.as_ref().grid_run_configs()
    }
}
