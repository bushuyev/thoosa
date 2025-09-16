use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::sink::With;
use parking_lot::RwLock;
use tch::Device;
use caching_runner::common::{CopyToDevice, Data, OutId};
use caching_runner::common::runnable::Stage;
use thoosa_common::applicables::{TCDLoader, ThoDevice};
use thoosa_common::dims::{ChannelsData, Dims, ThoTensor, TCD};
use thoosa_shared::applicables_runner::{Input, Output, Runner, ToOrdinal};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
pub struct RawChannels {
    t:ThoTensor<ChannelsData, SimpleId>
}

impl RawChannels {
    pub fn new(t: ThoTensor<ChannelsData, SimpleId>) -> Self {
        Self { t }
    }
}

impl WithId for RawChannels {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        WithId::id(&self.t)
    }
}

impl Input for RawChannels {

}

impl Output for RawChannels {}

#[derive(cached_derive::Cached)]
#[cache_name = "LoadedTensorId"]
pub struct TensorLoader{
    device: ThoDevice,
    loader_fn: Box<dyn Fn(&Resource, &Device) -> RawChannels + Sync + Send>,
}

impl Debug for TensorLoader {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorLoader")
            .field("device", &self.device)
            .finish()
    }
}


#[async_trait]
impl  Stage for TensorLoader {
    type In = Resource;
    type Out = RawChannels;
    type Ctx = RunCtx2;

    async fn run<I>(self: Arc<Self>, _input: Arc<I>, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
        I: WithId<ID=<<Self as Stage>::In as WithId>::ID> + Send + Sync + 'static
    {
        Ok((self.loader_fn)(_input.deref().id(), &self.device))
    }
}




impl Data for RawChannels {
    type Metrics = ();
    type Sub = RawChannels;
    type SubSelect = ();

    fn size(&self) -> Self::Metrics {
        todo!()
    }

    fn device(&self) -> usize {
        todo!()
    }
}

impl OutId for TensorLoader {
    fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("loaded_{}", input_id.id()))
    }
}
#[async_trait]
impl CopyToDevice for RawChannels {
    async fn copy_to_device(&self, dst_device: usize) ->Self {
        todo!()
    }
}


impl  TensorLoader {
    pub fn new<F>(loader_fn:F, device: ThoDevice) -> Self
    where
        F: Fn(&Resource, &Device) -> RawChannels + Sync + Send + 'static
    {
        Self {
            device,
            loader_fn: Box::new(loader_fn),
        }
    }
}
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use tch::kind::FLOAT_CPU;
    use tch::{Device, Tensor};
    use caching_runner::common::cachable::Cached;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::dims::ThoTensor;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::ThoosaError;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{RawChannels, TensorLoader};

    #[tokio::test]
    async fn test_load() -> Result<(), ThoosaError>{
        GlobalCtx::init_devices(2);
        let loader = Arc::new(TensorLoader::new(
            move |key, _| {
                RawChannels::new(ThoTensor::new(Input::id(key).clone(), Tensor::rand([100, 3], FLOAT_CPU)))
            },
            ThoDevice::from(Device::Cpu)
        ));

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        loader.run_cached(
            Arc::new(Resource::new(0, 0, "aaa".to_string(), 20)), ctx
        ).await.map_err(|e|ThoosaError::new("zzz"))?;



        Ok(())
    }

}