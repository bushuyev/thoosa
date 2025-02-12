use std::fmt::Debug;
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::ResourcesConfig;
use thoosa_shared::val::{NamedRange, NamedVal};

//TODO remove
pub trait XConstConfig {
    fn consts(&self) -> &[NamedVal<String>];
}

pub trait XGridConfig {
    fn grid(&self) -> &[NamedRange<String>];
}

pub trait XOptConfig {
    fn opt(&self) -> &[Hyperparameter<String>];
}

pub trait XRunConfigSource {
    type CC: XConstConfig;
    type GC: XGridConfig;
    type OC: XOptConfig;

    fn to_run_configs(self) -> XRunConfigs<Self::CC, Self::GC, Self::OC>;
}

pub trait XResourceConfigSource<R:ResourcesConfig>:Debug {
    fn resource_config(&self) -> R;
}


impl <V> From<Vec<V>> for XVecConfig<V> {
    fn from(value: Vec<V>) -> Self {
        XVecConfig::new(value)
    }
}

#[derive(Debug)]
pub struct XRunConfigs<CC: XConstConfig, GC: XGridConfig, OC: XOptConfig> {

    pub const_config: CC,//TODO pub
    pub grid_config: Option<GC>,
    pub opt_config: Option<OC>,
}

impl <CC: XConstConfig, GC: XGridConfig, OC: XOptConfig> XRunConfigs<CC, GC, OC> {

    pub fn new(const_config: CC, grid_config: Option<GC>, opt_config: Option<OC>) -> Self {
        Self { const_config, grid_config, opt_config }
    }
}

#[derive(Debug)]
pub struct XVecConfig<V> {
    pub configs: Vec<V> //TODO pub
}

impl <V> XVecConfig<V> {
    pub fn new(const_configs: Vec<V>) -> Self {
        Self { configs: const_configs }
    }
}

impl XConstConfig for XVecConfig<NamedVal<String>> {
    fn consts(&self) -> &[NamedVal<String>] {
        self.configs.as_slice()
    }
}

impl XGridConfig for XVecConfig<NamedRange<String>> {
    fn grid(&self) -> &[NamedRange<String>] {
        self.configs.as_slice()
    }
}

impl XOptConfig for XVecConfig<Hyperparameter<String>> {
    fn opt(&self) -> &[Hyperparameter<String>] {
        self.configs.as_slice()
    }
}

#[derive(Debug, Clone)]
pub struct FieldMeta {
    pub name: String,
    pub ty_str: String,
}

pub trait FlattenAllFields {
    fn all_fields() -> &'static [FieldMeta];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it() {

    }
}

