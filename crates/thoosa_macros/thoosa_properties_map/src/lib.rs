use std::collections::HashMap;

pub trait ToHashmap {
    fn to_properties_iter(&self, parent: Option<String>) -> Box<dyn Iterator<Item = (String, thoosa_shared::val::Val)>>;
}

pub trait FromHashmap {
    fn from_properties_map(parent: Option<String>, properties: &HashMap<String, thoosa_shared::val::Val>) -> anyhow::Result<Self>
    where
        Self: Sized;
}

/// Reject sparse or noncanonical indices before generated readers walk a vector.
/// Legacy maps without a length marker still receive the same validation.
pub fn validate_indices(properties: &HashMap<String, thoosa_shared::val::Val>, prefix: &str) -> anyhow::Result<()> {
    let mut indices = std::collections::BTreeSet::new();
    for key in properties.keys() {
        if let Some(suffix) = key.strip_prefix(prefix) {
            if suffix == "__len" {
                continue;
            }
            let index_text = suffix.split('.').next().unwrap_or_default();
            let index: usize = index_text.parse()?;
            anyhow::ensure!(index.to_string() == index_text, "noncanonical vector index in {key}");
            indices.insert(index);
        }
    }
    for (expected, actual) in indices.into_iter().enumerate() {
        anyhow::ensure!(actual == expected, "missing vector index {prefix}{expected} before {actual}");
    }
    Ok(())
}
