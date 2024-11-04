use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use thoosa_shared::val::Val;
use crate::parzen::optimizer::ConfigurationSpace;

#[derive(Debug)]
pub enum Resources {
    List(Vec<String>),
    Set(String),
    Group(Vec<String>)
}

type Label = i64;//TODO

#[derive(Debug)]
pub enum ResourcesConfig {
    Simple(Resources),
    ML{train:Vec<(Label, Resources)>, valid:Vec<(Label, Resources)>},
}

impl Display for  ResourcesConfig {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "xxx")
    }
}


pub trait Params<C:RunConfig>:Debug {
    fn to_run_configs(&self) ->Vec<C>;
    fn resource_keys(&self) -> Option<&ResourcesConfig> {
        None
    }
}

pub trait RunConfig:Debug {
    fn config_space(&self) -> Option<&ConfigurationSpace> {
        None
    }

    fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::new()//TODO
    }

    fn from_properties_map(properties: HashMap<String, Val>) -> Self;

    fn set_id(&mut self, id:usize) {
        todo!()
    }
}
