use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use thoosa_shared::Resources;
use thoosa_shared::val::Val;
use crate::parzen::optimizer::ConfigurationSpace;


type Label = i64;//TODO



pub trait ResourcesConfig:Display {

}

pub struct Simple {
    resources: Resources
}

impl Simple {
    pub fn new(resources: Resources) -> Self {
        Self { resources }
    }

    pub fn resources(&self) -> &Resources {
        &self.resources
    }
}

impl ResourcesConfig for Simple {

}

impl Display for Simple{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "xxx")
    }
}

pub struct NodeResources {//TODO rename Node to Net
    resources: Resources
}

impl ResourcesConfig for NodeResources {

}

impl Display for NodeResources {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "xxx")
    }
}


pub trait Params<C:RunConfig, R:ResourcesConfig, CommonConfig>:Debug {
    fn to_run_configs(&self) ->Vec<C>;
    fn resource_config(&self) -> R;
    fn common_config(&self) -> CommonConfig;
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
