pub mod composite_input;
pub mod load_and_clc;
#[cfg(test)]
pub(crate) mod model_test;
mod sync_stage_init_oom;

#[ctor::ctor]
fn init() {
    thoosa_shared::test_utils::init_test_logging();
}

// impl Adjustable for ThoTensor2<LoadedTensor, SimpleId> {}
