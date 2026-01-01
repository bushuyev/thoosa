use std::collections::HashMap;

pub trait ToHashmap {
    fn to_properties_iter(
        &self,
        parent: Option<String>,
    ) -> Box<dyn Iterator<Item = (String, thoosa_shared::val::Val)>>;
}

pub trait FromHashmap {
    fn from_properties_map(
        parent: Option<String>,
        properties: &HashMap<String, thoosa_shared::val::Val>,
    ) -> Result<Self, thoosa_shared::ThoError>
    where
        Self: Sized;
}
