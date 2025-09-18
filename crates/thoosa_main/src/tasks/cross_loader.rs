use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::RwLock;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use caching_runner::common::{Data, OutId};
use caching_runner::common::runnable::Stage;
use thoosa_common::dims::{ChannelsData, ChannelsSlicesData, Dims};
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;
use crate::tasks::run_ctx_2::RunCtx2;
use crate::tasks::tensor_loader::RawChannelsData;

#[derive(Debug)]
pub struct Slice {
    id:SimpleId,
    t:ThoTensor2<ChannelsSlicesData, SimpleId>,
}

impl Slice {
    pub fn new(id: SimpleId, t: ThoTensor2<ChannelsSlicesData, SimpleId>) -> Self {
        Self { id, t }
    }
}
impl WithId for Slice {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for Slice {}

#[async_trait]
impl CopyToDevice for Slice {
    async fn copy_to_device(&self, dst_device: usize) ->Self {
        Slice::new(self.id.clone(), self.t.copy_to_device(dst_device).await)
    }
}

impl  Data for Slice {
    type Metrics = TwoDimsMetrics;
    type Sub = Self;
    type SubSelect = std::ops::Range<Self::Metrics>;

    fn size(&self) -> Self::Metrics {
        self.t.size().try_into().expect("Data for CrossSlices::size failed")
    }

    fn device(&self) -> usize {
        self.t.device()
    }
}

#[derive(Debug, cached_derive::Cached)]
#[cache_name = "LoadedTensorId"]
pub struct CrossLoader {
}


impl CrossLoader {
    pub fn new() -> Self {
        Self {}
    }
}

impl OutId for CrossLoader {
    fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("cross_{}", input_id))
    }
}
#[derive(Debug)]
pub struct ResourcesVec {
    id:SimpleId,
    resources:Vec<Resource>
}

impl ResourcesVec {
    pub fn new(id: SimpleId, resources: Vec<Resource>) -> Self {
        Self { id, resources }
    }


}

impl WithId for ResourcesVec {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Input for ResourcesVec{}

#[async_trait]
impl  Stage for CrossLoader {
    type In = ResourcesVec;
    type Out = Slice;
    type Ctx = RunCtx2;

    async fn run(self: Arc<Self>, _input: Arc<Self::In>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    {
        // unimplemented!()
        for resource_0 in &_input.resources {

            for resource_1 in &_input.resources {
                debug!("resource_0={resource_0:?} resource_1={resource_1:?}");
            }

        }

        Ok(Slice::new("".into(), ThoTensor2::new("id".into(), Tensor::zeros(&[10,0], FLOAT_CPU))))
    }


    // fn do_run(self: Arc<Self>, _input: &Self::In, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    // where
    //     Self: Sized
    // {
    //     todo!()
    // }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use caching_runner::common::cachable::Cached;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::ThoosaError;
    use crate::tasks::cross_loader::{CrossLoader, ResourcesVec};
    use crate::tasks::run_ctx_2::RunCtx2;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);
        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let  cl = Arc::new(CrossLoader::new());

        let r = cl.run_cached(
            Arc::new(ResourcesVec::new("".into(), vec![
                Resource::new(0, 0, "aaa".to_string(), 20),
                Resource::new(0, 0, "bbb".to_string(), 20)
            ])),
            ctx
        ).await?;

        Ok(())
    }
}