use crate::ws::Workspace;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use std::fmt::{Debug, Display, Formatter};
use std::sync::Arc;
use thoosa_shared::resource::{Resource, ResourceStore, ResourcesConfig};
use thoosa_shared::{ResourceType, ThoError};
use thoosa_store::Store;


#[derive(Debug, Clone)]
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

#[async_trait]
impl ResourcesConfig for Simple {
    type R = Vec<Resource>;

    async fn to_resources<S: ResourceStore + ?Sized + Send + Sync>(
        &self,
        store: &Arc<Box<S>>,
    ) -> Result<Vec<Resource>, ThoError> {
        Ok(match self.resources() {
            ResourceType::List(list) => store
                .list_resources_by_file_stems(&list.into_iter().map(|v| v.as_str()).collect::<Vec<_>>())
                .await?,
            ResourceType::Set(set) => store
                .list_resource_set_resources_by_descr(set)
                .await?,
            ResourceType::Group(group) => {
                let store_guard = store;
                let mut resources = Vec::new();
                for (_, resource_set) in store_guard
                    .list_resource_set_group_sets(group)
                    .await?
                {
                    resources.extend(
                        store_guard
                            .list_resource_set_resources_by_descr(&resource_set)
                            .await
                            .expect("list_resource_set_resources_by_descr"),
                    );
                }
                resources
            }
            ResourceType::NotInGroup(descr) => store
                .list_resources_not_in_group(descr, 1000)
                .await?, //TODO limit
        })
    }

    async fn check<S: ResourceStore + ?Sized + Send + Sync>(&self, store: &Arc<Box<S>>) -> Result<(), ThoError> {
        if let ResourceType::List(list) = self.resources() {
            if list.is_empty() {
                return Err(ThoError::new("resource list should not be empty"));
            }
        }

        let resources = self.to_resources(store).await?;
        if resources.is_empty() {
            Err(ThoError::new(format!(
                "resource config {self:?} resolved to no resources"
            )))
        } else {
            Ok(())
        }
    }
}

#[derive(Debug, Clone)]
pub struct Empty {}

#[async_trait]
impl ResourcesConfig for Empty {
    type R = ();

    async fn to_resources<S: ResourceStore + ?Sized + Send + Sync>(
        &self,
        _store: &Arc<Box<S>>,
    ) -> Result<Self::R, ThoError> {
        Ok(())
    }

    async fn check<S: ResourceStore + ?Sized + Send + Sync>(&self, _store: &Arc<Box<S>>) -> Result<(), ThoError> {
        Ok(())
    }
}

impl Display for Simple {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "Simple")
    }
}

#[derive(Debug, Clone)]
pub struct Labeled {
    group: String,
}

impl Labeled {
    pub fn new(group: String) -> Self {
        Self { group }
    }
}

#[async_trait]
impl ResourcesConfig for Labeled {
    // (label, resources)
    type R = Vec<(String, Vec<Resource>)>;

    async fn to_resources<S: ResourceStore + ?Sized + Send + Sync>(
        &self,
        store: &Arc<Box<S>>,
    ) -> Result<Vec<(String, Vec<Resource>)>, ThoError> {
        let store_guard = store;
        let mut out = Vec::new();
        for (label, resource_set) in store_guard.list_group_labeled_resources_sets(&self.group).await? {
            out.push((label, store_guard.list_resource_set_resources_by_descr(&resource_set).await?));
        }
        Ok(out)
    }

    async fn check<S: ResourceStore + ?Sized + Send + Sync>(&self, store: &Arc<Box<S>>) -> Result<(), ThoError> {
        let labeled_resources = self.to_resources(store).await?;
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

#[derive(Debug, Clone)]
pub struct Logits {
    group: String,
}

impl Logits {
    pub fn new(group: String) -> Self {
        Self { group }
    }
}

#[async_trait]
impl ResourcesConfig for Logits {
    // (label, resources)
    type R = Vec<(Resource, Vec<f32>)>;

    async fn to_resources<S: ResourceStore + ?Sized + Send + Sync>(
        &self,
        store: &Arc<Box<S>>,
    ) -> Result<Vec<(Resource, Vec<f32>)>, ThoError> {
        store
            .list_resource_random_in_group(&self.group)
            .await?
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

    async fn check<S: ResourceStore + ?Sized + Send + Sync>(&self, store: &Arc<Box<S>>) -> Result<(), ThoError> {
        if store.list_resource_random_in_group(&self.group).await?.is_empty() {
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

#[derive(Debug, Clone)]
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

#[async_trait]
impl<RC: ResourcesConfig + Sync> ResourcesConfig for NetResources<RC>
where
    RC::R: Send,
{
    type R = (RC::R, RC::R);

    async fn to_resources<S: ResourceStore + ?Sized + Send + Sync>(
        &self,
        store: &Arc<Box<S>>,
    ) -> Result<(RC::R, RC::R), ThoError> {
        Ok((self.train_resources.to_resources(store).await?, self.valid_resources.to_resources(store).await?))
    }

    async fn check<S: ResourceStore + ?Sized + Send + Sync>(&self, store: &Arc<Box<S>>) -> Result<(), ThoError> {
        self.train_resources.check(store).await?;
        self.valid_resources.check(store).await?;
        Ok(())
    }
}

impl<R: ResourcesConfig> Display for NetResources<R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "xxx") //TODO
    }
}

#[async_trait]
pub trait SavableResult: Sync + Send + Debug {
    type P<'a>:Default = ();

    async fn save(
        &mut self, //TODO consume
        store: &Arc<Box<dyn Store>>,
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        p: Self::P<'_>
    ) -> Result<u32, ThoError>;
}
