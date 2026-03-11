use log::debug;
use serde::{Deserialize, Serialize};
use thoosa_shared::hyperparameter::Hyperparameter;
use thoosa_shared::RangeOrValue::Range;
use thoosa_shared::test_utils::init_test_logging;
use thoosa_shared::ThoError;

#[ctor::ctor]
fn init() {
    init_test_logging();
}

#[test]
fn test_expand_run_config() {
    macrotest::expand("src/expand/*.rs");
}

#[test]
fn test_cat_range() -> Result<(), ThoError>{
    use thoosa_properties_map_derive::ThoosaPropertiesMap;
    use thoosa_run_config_args_derive::ThoosaRunConfig;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::RangeOrValue;

    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
    struct One {

    }
    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
    struct Two{

    }

    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
    enum CatEnum {
        One(One),
        Two(Two)
    }

    pub enum CatEnumRangeOrValue_X {
        One(OneRangeOrValue),
        Two(TwoRangeOrValue),
        Vec(Vec<CatEnumRangeOrValue_X>)
    }

    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
    #[run_config_opt]
    struct TestRunConfig {
        #[in_map]
        pub id: Option<u32>,

        #[in_map]
        #[run_config]
        pub cat_field: CatEnum,

        #[in_map]
        #[run_config]
        pub simple_field: u32,

    }

    impl TestRunConfig {
        pub fn new(id: Option<u32>, num_layers: u32) -> Self {
            Self { id, simple_field: num_layers, cat_field: CatEnum::One(One{}) }
        }
    }

    let test_config = TestRunConfigRangeOrValue {
        id: Some(1),
        simple_field: RangeOrValue::Range{ range: 0..10, step: None },

        cat_field: CatEnumRangeOrValue::One(OneRangeOrValue {})
    };


    let opt_configs = test_config.opt_run_configs()?;
    let grid_vals = opt_configs.grid_to_vals();

    let opt_params = opt_configs.opt_params();

    assert_eq!(opt_params[0], Hyperparameter::UniformInt { name: TestRunConfigFields::SimpleField, lower: 0, upper: 10 });
    


    debug!("opt_params={opt_params:?}");

    Ok(())
}