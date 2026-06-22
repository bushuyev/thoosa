

#[cfg(test)]
mod model_test;
pub mod load_and_clc;

#[ctor::ctor]
fn init() {
    thoosa_shared::test_utils::init_test_logging();
}

// impl Adjustable for ThoTensor2<LoadedTensor, SimpleId> {}
