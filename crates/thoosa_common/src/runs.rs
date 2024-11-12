use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use parking_lot::Mutex;
use thoosa_shared::resource::Resource;
use thoosa_shared::ResourceType;
use thoosa_shared::val::Val;
use crate::parzen::optimizer::ConfigurationSpace;


type Label = i64;//TODO

pub trait ResourcesConfig:Display {
    // fn resource_config_to_keys(store:&Arc<Mutex<Box<dyn Store>>>, resource_config:&crate::runs::Simple) -> Vec<Resource>;
}

pub struct Simple {
    resources: ResourceType
}

impl Simple {
    pub fn new(resources: ResourceType) -> Self {
        Self { resources }
    }

    pub fn resources(&self) -> &ResourceType {
        &self.resources
    }
}

impl ResourcesConfig for Simple {

}


impl Display for Simple{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Simple")
    }
}

pub struct Labeled {
    resources: HashMap<String, ResourceType>
}

impl Labeled {
    pub fn from_group(resources: ResourceType) -> Self {
       match resources {
         
           ResourceType::Group(_) => {
               Self {
                   resources: HashMap::new(),//TODO
               }
           }
           _=> {
               panic!("wrong resources")
           }
       }
    }
}

impl ResourcesConfig for Labeled {

}

impl Display for Labeled{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Labeled")
    }
}


pub struct NodeResources {//TODO rename Node to Net
    resources: ResourceType
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
