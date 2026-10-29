use crate::hyperparameter::{FieldName, Hyperparameter};
use crate::traits::Named;
use crate::val::{GridValueSet, NamedRange, NamedVal};
use crate::make_combos;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::OnceLock;

#[derive(Debug, Clone)]
pub struct ConstConfigs<CP> {
    const_params: CP,
}

impl<T: FieldName> ConstConfigs<Vec<NamedVal<T>>> {
    pub fn new(const_params: Vec<NamedVal<T>>) -> Self {
        Self { const_params }
    }
}

#[derive(Debug, Clone)]
pub struct GridRunConfigs<CP, GP> {
    const_config_source: ConstConfigs<CP>,
    grid_params: GP,
}

impl<T: FieldName + 'static> GridRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>> {
    pub fn new(
        const_config_source: ConstConfigs<Vec<NamedVal<T>>>,
        grid_params: Vec<NamedRange<T>>,
    ) -> Self {
        Self {
            const_config_source,
            grid_params,
        }
    }

    pub fn const_params(&self) -> &Vec<NamedVal<T>> {
        &self.const_config_source.const_params
    }

    pub fn grid_value_sets(&self) -> anyhow::Result<Vec<GridValueSet<T>>> {
        // Propagate invalid static or dynamically resolved ranges instead of silently dropping grid branches.
        let grids = self.grid_params.iter().map(|gp| gp.grid_value_sets()).collect::<anyhow::Result<Vec<_>>>()?;
        let mut resolved = Vec::new();
        for combo in make_combos(grids.into_iter().map(Vec::into_iter)) {
            let mut values = Vec::new();
            let mut opt_params = Vec::new();
            let mut pending_fields = Vec::new();
            for value_set in combo {
                let (value_set_values, value_set_opt_params, value_set_pending_fields) = value_set.into_parts_with_pending();
                values.extend(value_set_values);
                opt_params.extend(value_set_opt_params);
                pending_fields.extend(value_set_pending_fields);
            }
            resolved.extend(GridValueSet::with_pending_fields(values, opt_params, pending_fields).resolve_pending_fields()?);
        }
        Ok(resolved)
    }

    pub fn grid_to_vals(&self) -> anyhow::Result<Vec<Vec<NamedVal<T>>>> {
        Ok(self.grid_value_sets()?
            .into_iter()
            .map(|value_set| {
                let (values, opt_params) = value_set.into_parts();
                assert!(opt_params.is_empty(), "grid_to_vals encountered opt params after resolving grid values: {opt_params:?}");
                values
            })
            .collect())
    }

    pub fn to_run_configs<R>(&self) -> anyhow::Result<Vec<R>>
    where
        R: From<std::vec::IntoIter<NamedVal<T>>>,
    {
        Ok(self.grid_to_vals()?
            .into_iter()
            .map(|grid_vals| {
                [
                    self.const_config_source.const_params.as_slice(),
                    grid_vals.as_slice(),
                ]
                .concat()
                .into_iter()
                .into()
            })
            .collect())
    }
}

/// Standard owned configuration containers used by training orchestration.
pub type TrainingRunConfigs<F> = OptRunConfigs<Vec<NamedVal<F>>, Vec<NamedRange<F>>, Vec<Hyperparameter<F>>, F>;

#[derive(Debug, Clone)]
pub struct OptRunConfigs<CP, GP, OP, T> {
    grid_run_config_source: GridRunConfigs<CP, GP>,
    opt_params: OP,
    _pd: PhantomData<T>,
}

impl<T: FieldName + 'static> OptRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>, Vec<Hyperparameter<T>>, T> {
    pub fn new(
        grid_run_config_source: GridRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>>,
        opt_params: Vec<Hyperparameter<T>>,
    ) -> Self {
        Self {
            grid_run_config_source,
            opt_params,
            _pd: PhantomData,
        }
    }

    pub fn opt_params(&self) -> &Vec<Hyperparameter<T>> {
        &self.opt_params
    }

    pub fn grid_to_vals(&self) -> anyhow::Result<Vec<Vec<NamedVal<T>>>> {
        self.grid_run_config_source.grid_to_vals()
    }

    pub fn grid_to_vals_and_opt_params(&self) -> anyhow::Result<Vec<(Vec<NamedVal<T>>, Vec<Hyperparameter<T>>)>> {
        Ok(self.grid_run_config_source
            .grid_value_sets()?
            .into_iter()
            .map(|value_set| {
                let (values, specific_opt_params) = value_set.into_parts();
                let mut opt_params = self.opt_params.clone();
                opt_params.extend(specific_opt_params);
                (values, opt_params)
            })
            .collect())
    }

    pub fn grid_run_config_source(&self) -> &GridRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>> {
        &self.grid_run_config_source
    }

    pub fn to_run_configs<R>(&self) -> anyhow::Result<Vec<R>>
    where
        R: From<std::vec::IntoIter<NamedVal<T>>>,
    {
        Ok(self.grid_run_config_source
            .grid_to_vals()?
            .into_iter()
            .map(|grid_vals| {
                [
                    self.grid_run_config_source
                        .const_config_source
                        .const_params
                        .as_slice(),
                    grid_vals.as_slice(),
                ]
                .concat()
                .into_iter()
                .into()
            })
            .collect())
    }
}

pub type RunConfigId = u32;
pub type RunId = u32;

impl Named for RunConfigId {}
impl Named for PathBuf {}//TODO it's for LoadNet - add type for NetId

pub trait RunConfig: Debug {
    fn set_id(&mut self, id: RunConfigId);
    fn get_id(&self) ->RunConfigId;

    fn type_name() -> &'static str;//TODO rename
}

///to group several related run configuration under single id. When invoking a command its run configuration can either be saved under new RUN_ID or with one from previous run. See run_configs and runs tables in the thoosa_store::create_tables.
pub static RUN_ID: OnceLock<RunId> = OnceLock::new();


pub fn run_id() -> RunId {
    RUN_ID.get().map(|v|*v).unwrap_or(0)
}