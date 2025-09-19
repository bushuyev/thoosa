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
pub struct DistWithMask {
}


#[async_trait]
impl CopyToDevice for DistWithMask {
    async fn copy_to_device(&self, dst_device: usize) ->Self {
        todo!()
    }
}
impl Data for DistWithMask{
    type Metrics = ();
    type Sub = DistWithMask;
    type SubSelect = ();

    fn size(&self) -> Self::Metrics {
        todo!()
    }

    fn device(&self) -> usize {
        todo!()
    }
}

impl Output for DistWithMask {

}
impl Input for DistWithMask {}

impl WithId for DistWithMask {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
       todo!()
    }
}

impl DistWithMask {
    pub fn new() -> Self {
        Self {}
    }
}

#[derive(Debug)]
pub struct DistsTracker {

}

impl DistsTracker {
    pub fn new() -> Self {
        Self {}
    }
}
#[async_trait]
impl Stage for DistsTracker {
    type In<'a> = DistWithMask;
    type Out = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type Ctx = RunCtx2;

    fn do_run<'a>(
        self: Arc<Self>,
        _input: &Self::In<'a>,
        _ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
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
