use log::debug;
use serde::{Deserialize, Serialize};
use thoosa_shared::hyperparameter::Hyperparameter;
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
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_shared::RangeOrValue;

    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
    struct One {
        #[in_map]
        #[run_config]
        pub inner_field_one: u32,
    }
    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
    struct Two{
        #[in_map]
        #[run_config]
        pub inner_field_two: u32,
    }

    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
    enum CatEnum {
        One(One),
        Two(Two)
    }

    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Serialize)]
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
            Self { id, simple_field: num_layers, cat_field: CatEnum::One(One{ inner_field_one: 3 }) }
        }
    }

    let test_config = TestRunConfigRangeOrValue {
        id: Some(1),
        simple_field: RangeOrValue::Range{ range: 0..10, step: None },

        // cat_field: CatEnumRangeOrValue::One(OneRangeOrValue {})
        cat_field: CatEnumRangeOrValue::Vec(vec![
            CatEnumRangeOrValue::One(OneRangeOrValue {
                inner_field_one: RangeOrValue::Range{ range: 0..3, step: None }
            }),
            CatEnumRangeOrValue::Two(TwoRangeOrValue {
                inner_field_two: RangeOrValue::Range{ range: 0..3, step: None }
            }),
        ])
    };

    debug!("conf={}", serde_json5::to_string(&test_config)?);


    let opt_configs = test_config.opt_run_configs()?;
    let const_vals = opt_configs.grid_run_config_source().const_params().clone();
    let grid_runs = opt_configs.grid_to_vals_and_opt_params();
    debug!("grid_runs={grid_runs:?}");
    assert_eq!(grid_runs.len(), 2);
    let (one_grid_vals, one_opt_params) = &grid_runs[0];
    let (two_grid_vals, two_opt_params) = &grid_runs[1];

    assert_eq!(
        one_opt_params,
        &vec![
            Hyperparameter::UniformInt { name: TestRunConfigFields::SimpleField, lower: 0, upper: 10 },
            Hyperparameter::UniformInt { name: TestRunConfigFields::CatField(CatEnumFields::One(OneFields::InnerFieldOne)), lower: 0, upper: 3 },
        ]
    );
    assert_eq!(
        two_opt_params,
        &vec![
            Hyperparameter::UniformInt { name: TestRunConfigFields::SimpleField, lower: 0, upper: 10 },
            Hyperparameter::UniformInt { name: TestRunConfigFields::CatField(CatEnumFields::Two(TwoFields::InnerFieldTwo)), lower: 0, upper: 3 },
        ]
    );

    let one_run_config: TestRunConfig = [
        const_vals.as_slice(),
        one_grid_vals.as_slice(),
        &[
            NamedVal::new(TestRunConfigFields::SimpleField, Val::Int(7)),
            NamedVal::new(TestRunConfigFields::CatField(CatEnumFields::One(OneFields::InnerFieldOne)), Val::Int(2)),
        ],
    ]
    .concat()
    .into_iter()
    .into();
    assert_eq!(one_run_config.id, Some(1));
    assert_eq!(one_run_config.simple_field, 7);
    assert_eq!(one_run_config.cat_field, CatEnum::One(One { inner_field_one: 2 }));

    let two_run_config: TestRunConfig = [
        const_vals.as_slice(),
        two_grid_vals.as_slice(),
        &[
            NamedVal::new(TestRunConfigFields::SimpleField, Val::Int(5)),
            NamedVal::new(TestRunConfigFields::CatField(CatEnumFields::Two(TwoFields::InnerFieldTwo)), Val::Int(1)),
        ],
    ]
    .concat()
    .into_iter()
    .into();
    assert_eq!(two_run_config.id, Some(1));
    assert_eq!(two_run_config.simple_field, 5);
    assert_eq!(two_run_config.cat_field, CatEnum::Two(Two { inner_field_two: 1 }));

    Ok(())
}
