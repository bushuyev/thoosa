use log::debug;
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
fn test_categories() -> Result<(), ThoError>{
    use thoosa_properties_map_derive::ThoosaPropertiesMap;
    use thoosa_run_config_args_derive::ThoosaRunConfig;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::RangeOrValue;

    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
    #[run_config_opt]
    struct TestRunConfig {
        #[in_map]
        pub id: Option<u32>,

        #[in_map]
        #[run_config]
        pub num_layers: u32,

    }

    impl TestRunConfig {
        pub fn new(id: Option<u32>, num_layers: u32) -> Self {
            Self { id, num_layers }
        }
    }

    let test_config = TestRunConfigRangeOrValue{
        id: Some(1),
        num_layers: RangeOrValue::Range{ range: 0..10, step: None },
    };


    let opt_configs = test_config.opt_run_configs()?;

    let opt_params = opt_configs.opt_params();

    debug!("opt_params={opt_params:?}");

    Ok(())
}