#[derive(Debug, PartialEq)]
pub struct Resource {
    id:ResourceId,
    import_id:u32,
    file_stem:String,
}

impl Resource {
    pub fn new(id: u32, import_id: u32, file_stem: String) -> Self {
        Self { id, import_id, file_stem }
    }

    pub fn id(&self) -> ResourceId {
        self.id
    }

    pub fn import_id(&self) -> u32 {
        self.import_id
    }

    pub fn file_stem(&self) -> &str {
        &self.file_stem
    }
}

pub type ResourceId = u32;

impl AsRef<ResourceId> for Resource {
    fn as_ref(&self) -> &ResourceId {
        &self.import_id
    }
}