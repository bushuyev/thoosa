use log::debug;
use serde::{Deserialize, Serialize};
use thoosa_properties_map_derive::ThoosaPropertiesMap;
use thoosa_run_config_args_derive::ThoosaRunConfig;
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


#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(crate) struct One {
    #[in_map]
    #[run_config]
    pub inner_field_one: u32,
}
#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(crate)  struct Two{
    #[in_map]
    #[run_config]
    pub inner_field_two: u32,
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(crate)  enum CatEnum {
    One(One),
    Two(Two)
}

#[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug, Deserialize, Serialize)]
#[run_config_opt]
pub(crate)  struct TestRunConfig {
    #[in_map]
    pub id: Option<u32>,

    #[in_map]
    #[run_config]
    pub cat_field: CatEnum,

    #[in_map]
    #[run_config]
    pub simple_field: u32,

}

#[test]
fn test_grid_param() -> Result<(), ThoError> {
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::val::NamedVal;
    use thoosa_shared::RangeOrValue;

    let test_config = TestRunConfigRangeOrValue {
        id: Some(1),
        simple_field: RangeOrValue::Range { range: 0..10, step: Some((1,0)) },

        cat_field: CatEnumRangeOrValue::Vec(vec![
            CatEnumRangeOrValue::One(OneRangeOrValue {
                inner_field_one: RangeOrValue::dynamic(|named_vals: &[NamedVal<TestRunConfigFields>]| {
                    match named_vals.iter().find(|f|f.path().eq(&TestRunConfigFields::SimpleField)){
                        None => RangeOrValue::Value(0),
                        Some(_v) => {
                            debug!("val={:?}", _v.val().to_f64());
                            RangeOrValue::Range{ range: 0 .._v.val().to_i64(1) as u32, step: None }
                        },
                    }
                })
            })
        ])
    };

    let opt_configs = test_config.opt_run_configs()?;
    let grid_runs = opt_configs.grid_to_vals_and_opt_params();
    assert_eq!(grid_runs.len(), 10);
    assert!(grid_runs.iter().enumerate().all(|(i, (_, opt_params))| {
        opt_params == &vec![Hyperparameter::UniformInt {
            name: TestRunConfigFields::CatField(CatEnumFields::One(OneFields::InnerFieldOne)),
            lower: 0,
            upper: i as i64,
        }]
    }));

    Ok(())
}

#[test]
fn test_dynamic_grid_param() -> Result<(), ThoError> {
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_shared::RangeOrValue;

    let test_config = TestRunConfigRangeOrValue {
        id: Some(1),
        simple_field: RangeOrValue::Range { range: 0..3, step: Some((1,0)) },

        cat_field: CatEnumRangeOrValue::Vec(vec![
            CatEnumRangeOrValue::One(OneRangeOrValue {
                inner_field_one: RangeOrValue::dynamic(|named_vals: &[NamedVal<TestRunConfigFields>]| {
                    let simple_field = named_vals
                        .iter()
                        .find(|field| field.path() == &TestRunConfigFields::SimpleField)
                        .expect("simple_field should be available for dynamic grid resolution");

                    match simple_field.val() {
                        Val::UInt(1) => RangeOrValue::Range { range: 10..12, step: Some((1, 0)) },
                        _ => RangeOrValue::Value(7),
                    }
                }),
            })
        ])
    };

    let opt_configs = test_config.opt_run_configs()?;
    let grid_runs = opt_configs.grid_to_vals_and_opt_params();
    assert_eq!(grid_runs.len(), 4);
    assert!(grid_runs.iter().all(|(_, opt_params)| opt_params.is_empty()));

    let actual = grid_runs
        .into_iter()
        .map(|(grid_vals, _)| {
            let simple = grid_vals
                .iter()
                .find(|field| field.path() == &TestRunConfigFields::SimpleField)
                .expect("simple_field should be in grid values")
                .val()
                .clone();
            let inner = grid_vals
                .iter()
                .find(|field| field.path() == &TestRunConfigFields::CatField(CatEnumFields::One(OneFields::InnerFieldOne)))
                .expect("inner_field_one should be in grid values")
                .val()
                .clone();
            (simple, inner)
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual,
        vec![
            (Val::UInt(0), Val::UInt(7)),
            (Val::UInt(1), Val::UInt(10)),
            (Val::UInt(1), Val::UInt(11)),
            (Val::UInt(2), Val::UInt(7)),
        ]
    );

    Ok(())
}


#[test]
fn test_cat_range() -> Result<(), ThoError>{
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_shared::val::{NamedVal, Val};
    use thoosa_shared::RangeOrValue;


    let test_config = TestRunConfigRangeOrValue {
        id: Some(1),
        simple_field: RangeOrValue::Range{ range: 0..10, step: None },

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
