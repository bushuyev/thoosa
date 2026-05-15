use crate::hyperparameter::{FieldName, Hyperparameter};
use crate::traits::Named;
use crate::val::{GridValueSet, NamedRange, NamedVal};
use crate::make_combos;
use std::fmt::Debug;
use std::marker::PhantomData;


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

    pub fn grid_value_sets(&self) -> Vec<GridValueSet<T>> {
        make_combos(self.grid_params.iter().map(|gp| gp.grid_value_sets().into_iter()))
            .into_iter()
            .flat_map(|combo| {
                let mut values = Vec::new();
                let mut opt_params = Vec::new();
                let mut pending_fields = Vec::new();

                for value_set in combo {
                    let (value_set_values, value_set_opt_params, value_set_pending_fields) = value_set.into_parts_with_pending();
                    values.extend(value_set_values);
                    opt_params.extend(value_set_opt_params);
                    pending_fields.extend(value_set_pending_fields);
                }

                GridValueSet::with_pending_fields(values, opt_params, pending_fields).resolve_pending_fields()
            })
            .collect()
    }

    pub fn grid_to_vals(&self) -> Vec<Vec<NamedVal<T>>> {
        self.grid_value_sets()
            .into_iter()
            .map(|value_set| {
                let (values, opt_params) = value_set.into_parts();
                assert!(opt_params.is_empty(), "grid_to_vals encountered opt params after resolving grid values: {opt_params:?}");
                values
            })
            .collect()
    }

    pub fn to_run_configs<R>(&self) -> Vec<R>
    where
        R: From<std::vec::IntoIter<NamedVal<T>>>,
    {
        self.grid_to_vals()
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
            .collect()
    }
}

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

    pub fn grid_to_vals(&self) -> Vec<Vec<NamedVal<T>>> {
        self.grid_run_config_source.grid_to_vals()
    }

    pub fn grid_to_vals_and_opt_params(&self) -> Vec<(Vec<NamedVal<T>>, Vec<Hyperparameter<T>>)> {
        self.grid_run_config_source
            .grid_value_sets()
            .into_iter()
            .map(|value_set| {
                let (values, specific_opt_params) = value_set.into_parts();
                let mut opt_params = self.opt_params.clone();
                opt_params.extend(specific_opt_params);
                (values, opt_params)
            })
            .collect()
    }

    pub fn grid_run_config_source(&self) -> &GridRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>> {
        &self.grid_run_config_source
    }

    pub fn to_run_configs<R>(&self) -> Vec<R>
    where
        R: From<std::vec::IntoIter<NamedVal<T>>>,
    {
        self.grid_run_config_source
            .grid_to_vals()
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
            .collect()
    }
}

pub type RunId = u32;

impl Named for RunId {}

pub trait RunConfig: Debug {
    fn set_id(&mut self, id: RunId);

    fn type_name() -> &'static str;
}
