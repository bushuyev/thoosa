use std::collections::HashMap;
use std::fmt::{Debug, Display, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use log::debug;
use parking_lot::{Mutex, RwLock};
use tch::nn::VarStore;
use thoosa_shared::resource::{Resource, ResourceStore, ResourcesConfig};
use thoosa_shared::{make_combos, ResourceType, ThoosaError};
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::run_configs::RunId;
use thoosa_shared::val::{NamedRange, NamedVal, Val};
use thoosa_store::Store;
use crate::parzen::optimizer::ConfigurationSpace;
use crate::ws::Workspace;

type Label = i64;//TODO


#[derive(Debug)]
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

    fn to_resources<S:ResourceStore + ?Sized>(&self, store:&Arc<Mutex<Box<S>>>) -> Result<Vec<Resource>, ThoosaError>{
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

pub struct  Empty  {}

impl ResourcesConfig for Empty {
    type Result = ();

    fn to_resources<S: ResourceStore + ?Sized>(&self, store: &Arc<Mutex<Box<S>>>) -> Result<Self::Result, ThoosaError> {
        Ok(())
    }
}

impl Display for Simple{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Simple")
    }
}

#[derive(Debug)]
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

    fn to_resources<S: ResourceStore + ?Sized>(&self, store: &Arc<Mutex<Box<S>>>) -> Result<Vec<(String, Vec<Resource>)>, ThoosaError> {
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

#[derive(Debug)]
pub struct Logits {
    group: String
}

impl Logits {

    pub fn new(group: String) -> Self {
        Self{
            group
        }
    }
}

impl ResourcesConfig for Logits {
    // (label, resources)
    type Result = Vec<(Resource,Vec<f32>)>;

    fn to_resources<S: ResourceStore + ?Sized>(&self, store: &Arc<Mutex<Box<S>>>) -> Result<Vec<(Resource,Vec<f32>)>, ThoosaError> {
        store.lock().list_resource_random_in_group(&self.group)?
            .into_iter()
            .map(move |(resourceId, labels_p)| {
                debug!("ResourcesConfig::to_resources resourceId={resourceId:?} labels_p={:?}", labels_p);
                Ok((
                    resourceId,
                    labels_p, //Box::new(Tensor::from_slice(labels_p.as_slice()).to_device(*device))
                ))
            })
            .collect()

    }
}

impl Display for Logits {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Logits")//TODO
    }
}



#[derive(Debug)]
pub struct NetResources<RC:ResourcesConfig> {//TODO rename Node to Net
    train_resources: RC,
    valid_resources: RC
}

impl <RC:ResourcesConfig> NetResources<RC> {
    pub fn new(train_resources:RC, valid_resources:RC) -> Self {
        Self {
            train_resources,
            valid_resources
        }
    }
}

impl <RC:ResourcesConfig> ResourcesConfig for NetResources<RC> {
    type Result = (RC::Result, RC::Result);

    fn to_resources<S: ResourceStore + ?Sized>(&self, store: &Arc<Mutex<Box<S>>>) -> Result<(RC::Result, RC::Result), ThoosaError> {
        self.train_resources.to_resources(store).and_then(|train|self.valid_resources.to_resources(store).map(|valid|(train, valid)))
    }
}

impl <R:ResourcesConfig> Display for NetResources<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "xxx")//TODO
    }
}

pub trait Params<C:RunConfig, R:ResourcesConfig, CommonConfig>:Debug {
    fn to_run_configs(&self) ->Vec<C>;
    fn resource_config(&self) -> R;
    fn common_config(&self) -> CommonConfig;
}

pub trait RunConfig:Debug {
    fn config_space(&self) -> Option<&ConfigurationSpace<String>> {
        None
    }

    fn to_properties_map(&self) -> HashMap<String, Val> {
        HashMap::new()//TODO
    }

    fn from_properties_map(run_id: RunId, properties: HashMap<String, Val>) -> Self;

    fn set_id(&mut self, id: RunId);

    fn type_name()->&'static str;
}

pub trait SavableOptResult:Sync+Send+Debug{
    fn value(&self)->f32;
    fn save(&mut self, store:&Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError>;
}

#[derive(Debug)]
pub struct DummySavableOptResult;

impl DummySavableOptResult {
    pub fn new() -> Box<dyn SavableOptResult> {
       Box::new(Self)
    }
}

impl SavableOptResult for DummySavableOptResult {
    fn value(&self) -> f32 {
        0.
    }

    fn save(&mut self, store: &Arc<Mutex<Box<dyn Store>>>, ws: &Arc<RwLock<Box<dyn Workspace>>>) -> Result<u32, ThoosaError> {
        Ok(0)
    }
}


