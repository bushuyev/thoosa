use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::resource::ResourcesConfig;
use thoosa_shared::val::{NamedRange, NamedVal};

pub trait ConstConfig {
    fn consts(&self) -> &[NamedVal];
}

pub trait GridConfig {
    fn grid(&self) -> &[NamedRange];
}

pub trait OptConfig {
    fn opt(&self) -> &[Hyperparameter];
}

pub trait RunConfigSource {
    type CC: ConstConfig;
    type GC: GridConfig;
    type OC: OptConfig;

    fn to_run_configs(self) -> RunConfigs<Self::CC, Self::GC, Self::OC>;
}

impl <V> From<Vec<V>> for VecConfig<V> {
    fn from(value: Vec<V>) -> Self {
        VecConfig::new(value)
    }
}

#[derive(Debug)]
pub struct RunConfigs<CC: ConstConfig, GC: GridConfig, OC: OptConfig> {

    pub const_config: CC,//TODO pub
    pub grid_config: Option<GC>,
    pub opt_config: Option<OC>,
}

impl <CC: ConstConfig, GC: GridConfig, OC: OptConfig> RunConfigs<CC, GC, OC> {

    pub fn new(const_config: CC, grid_config: Option<GC>, opt_config: Option<OC>) -> Self {
        Self { const_config, grid_config, opt_config }
    }
}

#[derive(Debug)]
pub struct VecConfig<V> {
    pub configs: Vec<V> //TODO pub
}

impl <V> VecConfig<V> {
    pub fn new(const_configs: Vec<V>) -> Self {
        Self { configs: const_configs }
    }
}

impl ConstConfig for VecConfig<NamedVal> {
    fn consts(&self) -> &[NamedVal] {
        self.configs.as_slice()
    }
}

impl GridConfig for VecConfig<NamedRange> {
    fn grid(&self) -> &[NamedRange] {
        self.configs.as_slice()
    }
}

impl OptConfig for VecConfig<Hyperparameter> {
    fn opt(&self) -> &[Hyperparameter] {
        self.configs.as_slice()
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_it() {

    }
}
