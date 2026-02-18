use crate::ws::Workspace;
use log::debug;
use parking_lot::RwLock;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use thoosa_shared::resource::{Resource, ResourceStore, ResourcesConfig};
use thoosa_shared::{ResourceType, ThoError};
use thoosa_store::Store;


#[derive(Debug)]
pub struct Simple {
    resources: ResourceType,
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
    type R = Vec<Resource>;

    fn to_resources<S: ResourceStore + ?Sized>(
        &self,
        store: &Arc<Box<S>>,
    ) -> Result<Vec<Resource>, ThoError> {
        Ok(match self.resources() {
            ResourceType::List(list) => store
                .list_resources_by_file_stems(
                    &list.into_iter().map(|v| v.as_str()).collect::<Vec<_>>(),
                )
                .expect("list_resources_by_file_stems"),
            ResourceType::Set(set) => store
                .list_resource_set_resources_by_descr(set)
                .expect("list_resource_set_resources"),
            ResourceType::Group(group) => {
                let store_guard = store;
                store_guard
                    .list_resource_set_group_sets(group)
                    .expect("list_resource_set_group_sets")
                    .into_iter()
                    .flat_map(|v| {
                        store_guard
                            .list_resource_set_resources_by_descr(&v.1)
                            .expect("list_resource_set_resources_by_descr")
                    })
                    .collect()
            }
            ResourceType::NotInGroup(descr) => store
                .list_resources_not_in_group(descr, 1000)//TODO limit
                .expect("list_resource_set_resources"),

            })
    }

    fn check<S: ResourceStore + ?Sized>(&self, store: &Arc<Box<S>>) -> Result<(), ThoError> {
        if let ResourceType::List(list) = self.resources() {
            if list.is_empty() {
                return Err(ThoError::new("resource list should not be empty"));
            }
        }

        let resources = self.to_resources(store)?;
        if resources.is_empty() {
            Err(ThoError::new(format!(
                "resource config {self:?} resolved to no resources"
            )))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct Empty {}

impl ResourcesConfig for Empty {
    type R = ();

    fn to_resources<S: ResourceStore + ?Sized>(
        &self,
        _store: &Arc<Box<S>>,
    ) -> Result<Self::R, ThoError> {
        Ok(())
    }

    fn check<S: ResourceStore + ?Sized>(&self, _store: &Arc<Box<S>>) -> Result<(), ThoError> {
        Ok(())
    }
}

impl Display for Simple {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Simple")
    }
}

#[derive(Debug)]
pub struct Labeled {
    group: String,
}

impl Labeled {
    pub fn new(group: String) -> Self {
        Self { group }
    }
}

impl ResourcesConfig for Labeled {
    // (label, resources)
    type R = Vec<(String, Vec<Resource>)>;

    fn to_resources<S: ResourceStore + ?Sized>(
        &self,
        store: &Arc<Box<S>>,
    ) -> Result<Vec<(String, Vec<Resource>)>, ThoError> {
        let store_guard = store;

        store_guard
            .list_group_labeled_resources_sets(&self.group)?
            .into_iter()
            .map(|(label, resource_set)| {
                store_guard
                    .list_resource_set_resources_by_descr(&resource_set)
                    .map(|data| (label, data))
            })
            .collect()
    }

    fn check<S: ResourceStore + ?Sized>(&self, store: &Arc<Box<S>>) -> Result<(), ThoError> {
        let labeled_resources = self.to_resources(store)?;
        if labeled_resources.is_empty() {
            return Err(ThoError::new(format!(
                "resource config {self:?} resolved to no labeled resource sets"
            )));
        }

        for (label, resources) in &labeled_resources {
            if resources.is_empty() {
                return Err(ThoError::new(format!(
                    "label {label} in resource config {self:?} has no resources"
                )));
            }
        }

        Ok(())
    }
}

impl Display for Labeled {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Labeled")
    }
}

#[derive(Debug)]
pub struct Logits {
    group: String,
}

impl Logits {
    pub fn new(group: String) -> Self {
        Self { group }
    }
}

impl ResourcesConfig for Logits {
    // (label, resources)
    type R = Vec<(Resource, Vec<f32>)>;

    fn to_resources<S: ResourceStore + ?Sized>(
        &self,
        store: &Arc<Box<S>>,
    ) -> Result<Vec<(Resource, Vec<f32>)>, ThoError> {
        store
            .list_resource_random_in_group(&self.group)?
            .into_iter()
            .map(move |(resource_id, labels_p)| {
                debug!(
                    "ResourcesConfig::to_resources resource_id={resource_id:?} labels_p={:?}",
                    labels_p
                );
                Ok((
                    resource_id,
                    labels_p, //Box::new(Tensor::from_slice(labels_p.as_slice()).to_device(*device))
                ))
            })
            .collect()
    }

    fn check<S: ResourceStore + ?Sized>(&self, store: &Arc<Box<S>>) -> Result<(), ThoError> {
        
        if store
            .list_resource_random_in_group(&self.group)?.is_empty() {
            Err(ThoError::new(format!(
                "resource config {self:?} resolved to no logits resources"
            )))
        } else {
            Ok(())
        }
    }
}

impl Display for Logits {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Logits") //TODO
    }
}

#[derive(Debug)]
pub struct NetResources<RC: ResourcesConfig> {
    //TODO rename Node to Net
    train_resources: RC,
    valid_resources: RC,
}

impl<RC: ResourcesConfig> NetResources<RC> {
    pub fn new(train_resources: RC, valid_resources: RC) -> Self {
        Self {
            train_resources,
            valid_resources,
        }
    }
}

impl<RC: ResourcesConfig> ResourcesConfig for NetResources<RC> {
    type R = (RC::R, RC::R);

    fn to_resources<S: ResourceStore + ?Sized>(
        &self,
        store: &Arc<Box<S>>,
    ) -> Result<(RC::R, RC::R), ThoError> {
        self.train_resources.to_resources(store).and_then(|train| {
            self.valid_resources
                .to_resources(store)
                .map(|valid| (train, valid))
        })
    }

    fn check<S: ResourceStore + ?Sized>(&self, store: &Arc<Box<S>>) -> Result<(), ThoError> {
        self.train_resources.check(store)?;
        self.valid_resources.check(store)?;
        Ok(())
    }
}

impl<R: ResourcesConfig> Display for NetResources<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "xxx") //TODO
    }
}

pub trait SavableResult: Sync + Send + Debug {
    fn value(&self) -> f32;//TODO remove
    fn save(
        &mut self,//TODO consume
        store: &Arc<Box<dyn Store>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoError>;
}

#[derive(Debug)]
pub struct DummySavableOptResult;

impl DummySavableOptResult {
    pub fn new() -> Box<dyn SavableResult> {
        Box::new(Self)
    }
}

impl SavableResult for DummySavableOptResult {
    fn value(&self) -> f32 {
        0.
    }

    fn save(
        &mut self,
        _store: &Arc<Box<dyn Store>>,
        _ws: &Arc<RwLock<Box<dyn Workspace>>>,
    ) -> Result<u32, ThoError> {
        Ok(0)
    }
}
