use std::marker::PhantomData;
use std::fmt::Debug;
use std::collections::HashMap;
use crate::hyperparameter::{FieldName, Hyperparameter};
use crate::make_combos;
use crate::val::{NamedRange, NamedVal, Val};

////
#[derive(Debug)]
pub struct ConstConfigs<CP> {
    const_params:CP,
}

impl <T: FieldName> ConstConfigs<Vec<NamedVal<T>>> {

    pub fn new(const_params:Vec<NamedVal<T>>) -> Self {
        Self {
            const_params,
        }
    }

}

#[derive(Debug)]
pub struct GridRunConfigs<CP, GP> {
    const_config_source: ConstConfigs<CP>,
    grid_params:GP
}

impl <T: FieldName> GridRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>> {

    pub fn new(const_config_source: ConstConfigs<Vec<NamedVal<T>>>, grid_params:Vec<NamedRange<T>>) -> Self {
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

    pub fn to_run_configs<R>(&self) ->Vec<R>
    where R:From<std::vec::IntoIter<NamedVal<T>>>{

        self.grid_to_vals().into_iter().map(|grid_vals|{
            [
                self.const_config_source.const_params.as_slice(),
                grid_vals.as_slice(),
            ]
            .concat()
            .into_iter()
            .into()
        }).collect()

    }

}

//TODO make VecOptRunConfigs with single T param?
#[derive(Debug)]
pub struct OptRunConfigs<CP, GP, OP, T> {
    grid_run_config_source: GridRunConfigs<CP, GP>,
    opt_params:OP,
    _pd: PhantomData<T>,
}

impl <T: FieldName> OptRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>, Vec<Hyperparameter<T>>, T>  {

    pub fn new(grid_run_config_source: GridRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>>, opt_params:Vec<Hyperparameter<T>>) -> Self {
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

    pub fn grid_run_config_source(&self) -> &GridRunConfigs<Vec<NamedVal<T>>, Vec<NamedRange<T>>> {
        &self.grid_run_config_source
    }

    pub fn to_run_configs<R>(&self) ->Vec<R>
    where
        R: From<std::vec::IntoIter<NamedVal<T>>>,
    {

        self.grid_run_config_source.grid_to_vals().into_iter().map(|grid_vals|{

            [
                self.grid_run_config_source.const_config_source.const_params.as_slice(),
                grid_vals.as_slice(),
            ]
                .concat()
                .into_iter()
                .into()
        }).collect()

    }
}

pub type RunId = u32;

pub trait RunConfig:Debug {

    fn set_id(&mut self, id: RunId);

    fn type_name()->&'static str;
}

pub struct RunConfigHolder<R:RunConfig> {
    id:RunId,
    run_config: R,
    descr:String,
    // ran_on:std::time::SystemTime TODO for each run X resource
}

impl<R:RunConfig> RunConfigHolder<R> {

    pub fn new(id:RunId, run_config: R, descr:&str) -> Self {
        Self {
            id,
            run_config,
            descr: "".to_string(),
            // ran_on,
        }
    }
}

