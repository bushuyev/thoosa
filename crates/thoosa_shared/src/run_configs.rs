use crate::hyperparameter::{FieldName, Hyperparameter};
use crate::traits::Named;
use crate::val::{NamedRange, NamedVal};
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

impl<T: FieldName> GridRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>> {
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

    pub fn grid_to_vals(&self) -> Vec<Vec<NamedVal<T>>> {
        make_combos(self.grid_params.iter().map(|gp| gp.clone().into_iter()))
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

//TODO make VecOptRunConfigs with single T param?
#[derive(Debug, Clone)]
pub struct OptRunConfigs<CP, GP, OP, T> {
    grid_run_config_source: GridRunConfigs<CP, GP>,
    opt_params: OP,
    _pd: PhantomData<T>,
}

impl<T: FieldName> OptRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>, Vec<Hyperparameter<T>>, T> {
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

//TODO move from_properties_map__ into here + all run config meta data id, description, date, running time
#[allow(dead_code)]
pub struct RunConfigHolder<R: RunConfig> {
    id: RunId,
    run_config: R,
    descr: String,
}

impl<R: RunConfig> RunConfigHolder<R> {
    pub fn new(id: RunId, run_config: R, _descr: &str) -> Self {
        Self {
            id,
            run_config,
            descr: "".to_string(),
            // ran_on,
        }
    }
}
