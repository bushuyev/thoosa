use std::fmt::{Debug, Formatter};
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use caching_runner::common::cachable::DataCacheWrapper;
use caching_runner::common::runnable::Stage;
use thoosa_motifs::distance::{clc_dist_d2, clc_dist_d2_and_mask};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsSlicesData2};
use crate::tasks::dists_tracker::DistWithMask;
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
pub struct BatchInput<'a, R:AsRef<Resource>> {
    tensors:&'a [&'a ThoTensor2<ChannelsData2, SimpleId>],
    b_0:&'a [R],
    b_1:&'a [R]
}


impl  <'a, R:AsRef<Resource>> BatchInput<'a, R> {
    pub fn new(tensors: &'a [&'a ThoTensor2<ChannelsData2, SimpleId>], b_0: &'a [R], b_1: &'a [R]) -> Self {
        Self { tensors, b_0, b_1 }
    }
}

impl <'a, R:AsRef<Resource> + Debug + Send + Sync> Input for BatchInput<'a, R> {}

pub struct CrossBatchClc<F, R>
where
    F:Fn(Tensor, Tensor)->Result<R, ThoosaError>
{
    batch_clc:F
}

impl <F, R>  Debug for CrossBatchClc<F, R>
where
    F:Fn(Tensor, Tensor)->Result<R, ThoosaError>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CrossBatchClc").finish()
    }
}

impl <F, R> CrossBatchClc<F, R>
where
    F:Fn(Tensor, Tensor)->Result<R, ThoosaError>
{
    pub fn new(batch_clc:F) -> Self {
        Self {
            batch_clc
        }
    }
}

#[async_trait]
impl <F,R> Stage for CrossBatchClc<F,R>
where
    F:Fn(Tensor, Tensor)->Result<R, ThoosaError> +'static,
    R:Debug+Output + 'static
{
    type In<'a> = BatchInput<'a, &'a Arc<Resource>>; //where Self:'a, R:'a, F: 'a;
    type Out = R;
    type Ctx = RunCtx2;

    fn do_run<'s>(
        self: Arc<Self>,
        input: &Self::In<'s>,
        _ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized
    {
        // let mut mask_makers = vec![];

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

        (self.batch_clc)(batch_0_t.shallow_clone(), batch_1_t)
    }
}
