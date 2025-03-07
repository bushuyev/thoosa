use std::marker::PhantomData;
use crate::hyperparameter::{FieldName, Hyperparameter};
use crate::make_combos;
use crate::val::{NamedRange, NamedVal};

////
#[derive(Debug)]
pub struct ConstConfigSource<CP> {
    const_params:CP,
}

impl <T: FieldName> ConstConfigSource<Vec<NamedVal<T>>> {

    pub fn new(const_params:Vec<NamedVal<T>>) -> Self {
        Self {
            const_params,
        }
    }

}

#[derive(Debug)]
pub struct GridRunConfigSource<CP, GP> {
    const_config_source: ConstConfigSource<CP>,
    grid_params:GP
}

impl <T: FieldName> GridRunConfigSource<Vec<NamedVal<T>>, Vec<NamedRange<T>>> {

    pub fn new(const_config_source:ConstConfigSource<Vec<NamedVal<T>>>, grid_params:Vec<NamedRange<T>>) -> Self {
        Self {
            const_config_source,
            grid_params
        }
    }

    pub fn const_params(&self) -> &Vec<NamedVal<T>> {
        &self.const_config_source.const_params
    }

    pub fn grid_to_vals(&self) -> Vec<Vec<NamedVal<T>>> {
        make_combos(
            self.grid_params.iter().map(|gp| {
                gp.clone().into_iter()
            })
        )
    }

}

#[derive(Debug)]
pub struct OptRunConfigSource<CP, GP, OP, T> {
    grid_run_config_source: GridRunConfigSource<CP, GP>,
    opt_params:OP,
    _pd: PhantomData<T>,
}

impl <T: FieldName> OptRunConfigSource<Vec<NamedVal<T>>, Vec<NamedRange<T>>, Vec<Hyperparameter<T>>, T>  {

    pub fn new(grid_run_config_source: GridRunConfigSource<Vec<NamedVal<T>>, Vec<NamedRange<T>>>, opt_params:Vec<Hyperparameter<T>>) -> Self {
        Self {
            grid_run_config_source,
            opt_params,
            _pd: PhantomData
        }
    }

    pub fn opt_params(&self) -> &Vec<Hyperparameter<T>> {
        &self.opt_params
    }

    pub fn grid_to_vals(&self) -> Vec<Vec<NamedVal<T>>> {
        self.grid_run_config_source.grid_to_vals()
    }

    pub fn grid_run_config_source(&self) -> &GridRunConfigSource<Vec<NamedVal<T>>, Vec<NamedRange<T>>> {
        &self.grid_run_config_source
    }
}