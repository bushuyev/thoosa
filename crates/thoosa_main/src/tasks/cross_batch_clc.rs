use std::fmt::Debug;
use std::sync::Arc;
use log::debug;
use parking_lot::RwLock;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use caching_runner::common::cachable::DataCacheWrapper;
use caching_runner::common::runnable::Stage;
use thoosa_motifs::distance::clc_dist_d2;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsSlicesData2};
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
pub struct BatchUnfoldedInput<'a, R:AsRef<Resource>> {
    tensors:&'a [&'a ThoTensor2<ChannelsData2, SimpleId>],
    b_0:&'a [R],
    b_1:&'a [R]
}


impl  <'a, R:AsRef<Resource>> BatchUnfoldedInput<'a, R> {
    pub fn new(tensors: &'a [&'a ThoTensor2<ChannelsData2, SimpleId>], b_0: &'a [R], b_1: &'a [R]) -> Self {
        Self { tensors, b_0, b_1 }
    }
}

impl <'a, R:AsRef<Resource> + Debug + Send + Sync> Input for BatchUnfoldedInput<'a, R> {}

#[derive(Debug)]
pub struct CrossBatchClc;

impl CrossBatchClc {
    pub fn new() -> Self {
        Self
    }
}

impl Stage for CrossBatchClc {
    type In<'s> = BatchUnfoldedInput<'s, &'s Arc<Resource>>;
    type Out = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type Ctx = RunCtx2;

    fn do_run<'s>(
        self: Arc<Self>,
        input: &Self::In<'s>,
        _ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized,
    {
        let batch_0_t = Tensor::concat(
            input.tensors[0..input.b_0.len()]
                .iter()
                .map(|t| t.t().unfold(1, 2, 1))
                .collect::<Vec<_>>()
                .as_slice(),
            1,
        );

        let batch_1_t = Tensor::concat(
            input.tensors[input.b_0.len()..]
                .iter()
                .map(|t| t.t().unfold(1, 2, 1))
                .collect::<Vec<_>>()
                .as_slice(),
            1,
        );

        debug!("CrossBatchClc::do_run:  batch_0_t={batch_0_t:?}  batch_1_t={batch_1_t:?}");

        let dists = clc_dist_d2(batch_0_t.shallow_clone(), batch_1_t, 1)?;

        debug!("CrossBatchClc::do_run:  dists={dists:?}");

        Ok(ThoTensor2::new("".into(), Tensor::rand(&[0], FLOAT_CPU)))
    }
}
