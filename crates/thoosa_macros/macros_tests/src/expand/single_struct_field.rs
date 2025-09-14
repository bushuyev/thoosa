use thoosa_run_config_args_derive::ThoosaRunConfig;


#[derive(ThoosaRunConfig)]
#[run_config_opt]
pub struct SingleField {
    #[run_config]
    pub val:i32,
}


#[derive(ThoosaRunConfig)]
#[run_config_opt]
pub struct Outer {
    #[run_config]
    pub inner:SingleField,
}