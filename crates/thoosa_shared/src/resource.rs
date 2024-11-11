use std::cmp::Ordering;
use std::fmt::Display;
use std::hash::{Hash, Hasher};
use clap::builder::Str;
use log::{debug, trace};

#[derive(Debug, Clone)]
pub struct Resource {
    id:ResourceId,
    import_id:u32,
    file_stem:String,
}

impl Resource {
    pub fn new(id: u32, import_id: u32, file_stem: String) -> Self {
        trace!("Resource::new: id={id} import_id={import_id} file_stem={file_stem}");
        Self { id, import_id, file_stem }
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

}

impl From<(u32, u32, &str)> for Resource {
    fn from(value: (u32, u32, &str)) -> Self {
        Resource::new(value.0, value.1, value.2.to_string())
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

impl Eq for Resource {

}

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
        &self.import_id
    }
}

