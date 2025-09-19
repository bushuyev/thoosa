use std::fmt::Debug;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use caching_runner::common::cachable::DataCacheWrapper;
use caching_runner::common::Data;
use caching_runner::common::runnable::Stage;
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::ChannelsSlicesData2;
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
pub struct TensorsPair {
    id: SimpleId,
    t_0:Arc<DataCacheWrapper<ThoTensor2<ChannelsSlicesData2, SimpleId>>>,
    t_1:Arc<DataCacheWrapper<ThoTensor2<ChannelsSlicesData2, SimpleId>>>
}
impl Input for TensorsPair {}

impl WithId for TensorsPair {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl TensorsPair {
    pub fn new(id: SimpleId, t_0: Arc<DataCacheWrapper<ThoTensor2<ChannelsSlicesData2, SimpleId>>>, t_1: Arc<DataCacheWrapper<ThoTensor2<ChannelsSlicesData2, SimpleId>>>) -> Self {
        Self { id, t_0, t_1 }
    }
}

#[derive(Debug)]
pub struct CrossUnfold {

}

impl CrossUnfold {
    pub fn new() -> Self {
        Self {}
    }
}
#[async_trait]
impl Stage for CrossUnfold {
    type In = TensorsPair;
    type Out = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type Ctx = RunCtx2;

    fn do_run(self: Arc<Self>, _input: &Self::In, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized
    {
        Ok(ThoTensor2::new("".into(), Tensor::rand(&[10, 1], FLOAT_CPU)))
    }
}


#[cfg(test)]
mod tests {
    use thoosa_shared::ThoosaError;
    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError> {
        Ok(())
    }
}