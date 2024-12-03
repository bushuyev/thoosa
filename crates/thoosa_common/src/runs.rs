use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use parking_lot::Mutex;
use thoosa_shared::resource::{Resource, ResourceStore};
use thoosa_shared::{ResourceType, ThoosaError};
use thoosa_shared::val::Val;
use crate::parzen::optimizer::ConfigurationSpace;


type Label = i64;//TODO


pub trait ResourcesConfig:Display {
    type Result;
    fn resource_config_to_keys<S:ResourceStore+ ?Sized>(&self, store:&Arc<Mutex<Box<S>>>) -> Result<Self::Result, ThoosaError>;
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
    type Result = Vec<Resource>;

    fn resource_config_to_keys<S:ResourceStore + ?Sized>(&self, store:&Arc<Mutex<Box<S>>>) -> Result<Vec<Resource>, ThoosaError>{
        Ok(match self.resources() {
            ResourceType::List(list) => {
                store.lock().list_resources_by_file_stems(&list.into_iter().map(|v|v.as_str()).collect()).expect("list_resources_by_file_stems")
            }
            ResourceType::Set(set) => {
                store.lock().list_resource_set_resources_by_descr(set).expect("list_resource_set_resources")
            }
            ResourceType::Group(group) => {
                let mut store_guard = store.lock();
                store_guard.list_resource_set_group_sets(group).expect("list_resource_set_group_sets").into_iter().flat_map(|v|
                    store_guard.list_resource_set_resources_by_descr(&v.1).expect("list_resource_set_resources_by_descr")
                ).collect()
            }
        })
    }
}

impl Display for Simple{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Simple")
    }
}

pub struct Labeled {
    group: String
}

impl Labeled {

    pub fn new(group: String) -> Self {
        Self{
            group
        }
    }
}

impl ResourcesConfig for Labeled {
    // (label, resources)
    type Result = Vec<(String,Vec<Resource>)>;

    fn resource_config_to_keys<S: ResourceStore + ?Sized>(&self, store: &Arc<Mutex<Box<S>>>) -> Result<Vec<(String,Vec<Resource>)>, ThoosaError> {
        let mut store_guard = store.lock();

        store_guard.list_group_labeled_resources_sets(&self.group)?.into_iter().map(|(label, resource_set)| {
            store_guard.list_resource_set_resources_by_descr(&resource_set)
                .map(|data|(label, data))
        }).collect()
    }
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
    type Result = (Vec<(String,Vec<Resource>)>, Vec<(String,Vec<Resource>)>);

    fn resource_config_to_keys<S: ResourceStore + ?Sized>(&self, store: &Arc<Mutex<Box<S>>>) -> Result<(Vec<(String,Vec<Resource>)>, Vec<(String,Vec<Resource>)>), ThoosaError> {
        todo!()
    }
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

    fn from_properties_map(run_id: usize, properties: HashMap<String, Val>) -> Self;

    fn set_id(&mut self, id:usize);
}
