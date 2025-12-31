use crate::ThoosaError;
use crate::applicables_runner::Input;
use crate::traits::{Named, WithId};
use log::trace;
use mockall::automock;
use parking_lot::RwLock;
use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use thoosa_dummy_data_derive::DummyData;

#[automock]
pub trait ResourceStore {
    fn list_resources_by_file_stems<'a>(
        &self,
        file_stems: &'a [&'a str],
    ) -> Result<Vec<Resource>, ThoosaError>;

    fn list_resource_set_resources_by_descr(
        &self,
        descr: &str,
    ) -> Result<Vec<Resource>, ThoosaError>;

    fn list_resource_set_group_sets(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(usize, String)>, ThoosaError>;

    fn list_group_labeled_resources_sets(
        &self,
        group_decr: &str,
    ) -> Result<Vec<(String, String)>, ThoosaError>;

    fn list_resource_set_resources(&self, set_id: usize) -> Result<Vec<Resource>, ThoosaError>;

    fn list_resource_random_in_group(
        &self,
        group_descr: &str,
    ) -> Result<Vec<(Resource, Vec<f32>)>, ThoosaError>;

    fn list_resources_not_in_group(
        &self,
        group_descr: &str,
        limit: u32,
    ) -> Result<Vec<Resource>, ThoosaError>;


}

#[derive(Debug, Clone, DummyData)]
#[dummy_data(clone)]
pub struct Resource {
    id: ResourceId,
    import_id: u32,
    file_stem: String,
    length: u32,//TODO rename to data_len
    // applied_id: CachableId, TODO CompositeKey id + import_id
}

impl Resource {
    pub fn new<I:Into<String>+Display>(id: u32, import_id: u32, file_stem: I, length: u32) -> Self {
        trace!("Resource::new: id={id} import_id={import_id} file_stem={file_stem}");
        // let applied_id = format!("resource_{}_{}", id, import_id).into();
        Self {
            id,
            import_id,
            file_stem:file_stem.into(),
            length,
            // applied_id,
        }
    }

    pub fn id(&self) -> ResourceId {
        self.id
    }

    pub fn import_id(&self) -> u32 {
        self.import_id
    }

    pub fn file_stem(&self) -> &str {
        self.file_stem.as_str()
    }

    pub fn length(&self) -> u32 {
        self.length
    }

    // pub fn applied_id(&self) -> &CachableId {
    //     &self.applied_id
    // }
}


impl Named for Resource {}

impl WithId for Resource {
    type ID = Resource;

    fn id(&self) -> &Self::ID {
        &self
    }//return composite id+import_id?
}

impl Input for Resource {
    
}

impl From<(u32, u32, &str, u32)> for Resource {
    fn from(value: (u32, u32, &str, u32)) -> Self {
        Resource::new(value.0, value.1, value.2.to_string(), value.3)
    }
}

impl Hash for Resource {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state)
    }
}

impl PartialEq<Self> for Resource {
    fn eq(&self, other: &Self) -> bool {
        self.id.eq(&other.id)
    }
}

impl Eq for Resource {}

impl PartialOrd<Self> for Resource {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.id().partial_cmp(&other.id)
    }
}

impl Ord for Resource {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

pub type ResourceId = u32;

impl AsRef<ResourceId> for Resource {
    fn as_ref(&self) -> &ResourceId {
        &self.id
    }
}

//TODO mean, std per channel
pub struct ResourceStats {
    size: Vec<u32>,
    length_dim: usize,
}

impl ResourceStats {
    pub fn new(size: Vec<u32>) -> Self {
        assert!(
            size.len() > 1,
            "should be two dimensions at least: channels x values, was {size:?}"
        );
        Self {
            size,
            length_dim: 1, //TODO param
        }
    }

    pub fn length(&self) -> u32 {
        self.size[self.length_dim]
    }
}

pub trait ResourcesConfig {
    type R;
    fn to_resources<S: ResourceStore + ?Sized>(
        &self,
        store: &Arc<RwLock<Box<S>>>,
    ) -> Result<Self::R, ThoosaError>;

    fn check<S: ResourceStore + ?Sized>(&self, _store: &Arc<RwLock<Box<S>>>) -> Result<(), ThoosaError> {
        Err(ThoosaError::new("resource config check is not implemented"))
    }
}
