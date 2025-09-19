use std::marker::PhantomData;
use std::sync::Arc;
use caching_runner::common::cachable::DataCacheWrapper;
use caching_runner::common::runnable::Stage;
use thoosa_shared::applicables_runner::Input;
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::{ChannelsData2, ChannelsSlicesData2};
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
pub struct BatchUnfoldedInput<'a> {
    tensors:&'a [&'a ThoTensor2<ChannelsData2, SimpleId>],
    b_0:Vec<Resource>,
    b_1:Vec<Resource>
}


impl  <'a> BatchUnfoldedInput<'a> {
    // pub fn new( b_0: Vec<Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>>, b_1: Vec<Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>>) -> Self {
    //     Self { b_0, b_1 }
    // }
}

impl <'a> Input for BatchUnfoldedInput<'a> {}

#[derive(Debug)]
pub struct CrossBatchClc<'a> {
    pd:PhantomData<&'a ()>
}

impl <'a> CrossBatchClc<'a> {
    pub fn new() -> Self {
        Self { pd: Default::default() }
    }
}

impl <'a> Stage for CrossBatchClc<'a> {
    type In = BatchUnfoldedInput<'a>;
    type Out = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type Ctx = RunCtx2;
}