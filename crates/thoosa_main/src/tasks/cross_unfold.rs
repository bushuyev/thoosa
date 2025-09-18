use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use caching_runner::common::cachable::DataCacheWrapper;
use caching_runner::common::Data;
use caching_runner::common::runnable::Stage;
use thoosa_common::dims::{ChannelsData, ChannelsSlicesData};
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;
use crate::tasks::cross_loader::Slice;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::RawChannelsData;

#[derive(Debug)]
pub struct TensorsPair {
    id: SimpleId,
    t_0:Arc<DataCacheWrapper<RawChannelsData>>,
    t_1:Arc<DataCacheWrapper<RawChannelsData>>
}
impl Input for TensorsPair {}

impl WithId for TensorsPair {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl TensorsPair {
    pub fn new(id: SimpleId, t_0: Arc<DataCacheWrapper<RawChannelsData>>, t_1: Arc<DataCacheWrapper<RawChannelsData>>) -> Self {
        Self { id, t_0, t_1 }
    }
}

#[derive(Debug)]
pub struct CrossUnfoldOutput {
    id:SimpleId,
    t:ThoTensor2<ChannelsSlicesData, SimpleId>,
}

impl WithId for CrossUnfoldOutput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}


impl Output for CrossUnfoldOutput {

}

impl CrossUnfoldOutput {
    pub fn new(id: SimpleId, t: ThoTensor2<ChannelsSlicesData, SimpleId>) -> Self {
        Self { id, t }
    }
}

impl Data for CrossUnfoldOutput {
    type Metrics = TwoDimsMetrics;
    type Sub = Self;
    type SubSelect = Range<TwoDimsMetrics>;

    fn size(&self) -> Self::Metrics {
        self.t.size().try_into().expect("Data for CrossUnfoldOutput::size failed")
    }

    fn device(&self) -> usize {
        self.t.device()
    }
}

#[async_trait]
impl CopyToDevice for CrossUnfoldOutput {
    async fn copy_to_device(&self, dst_device: usize) ->Self {
        CrossUnfoldOutput::new(self.id.clone(), self.t.copy_to_device(dst_device).await)
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
    type Out = CrossUnfoldOutput;
    type Ctx = RunCtx2;

    fn do_run(self: Arc<Self>, _input: &Self::In, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized
    {
        Ok(CrossUnfoldOutput::new("".into(), ThoTensor2::new("".into(), Tensor::rand(&[10, 1], FLOAT_CPU))))
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use log::debug;
    use tch::{Device, Tensor};
    use caching_runner::common::cachable::Cached;
    use caching_runner::common::runnable::Stage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::ThoosaError;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::cross_loader::{CrossLoader, ResourcesVec};
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{RawChannelsData, TensorLoader};

    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError> {
        Ok(())
    }
}