use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use async_trait::async_trait;
use parking_lot::RwLock;
use tch::{Device, Tensor};
use tch::kind::FLOAT_CPU;
use caching_runner::common::{Data, OutId};
use caching_runner::common::runnable::Stage;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::dims::{ChannelsData, Dims};
use thoosa_common::tho_tensor::{CopyToDevice, ThoTensor};
use thoosa_shared::applicables_runner::{Input, Output, Runner, ToOrdinal};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use torch_runner::metrics::TwoDimsMetrics;
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
pub struct RawChannelsData {
    t:ThoTensor2<ChannelsData, SimpleId>
}

impl RawChannelsData {
    pub fn new(t: ThoTensor2<ChannelsData, SimpleId>) -> Self {
        Self { t }
    }
}

impl WithId for RawChannelsData {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        WithId::id(&self.t)
    }
}

impl Input for RawChannelsData {

}

impl Output for RawChannelsData {}


#[derive(cached_derive::Cached)]
#[cache_name = "LoadedTensorId"]
pub struct TensorLoader{
    device: ThoDevice,
    loader_fn: Box<dyn Fn(&Resource, &Device) -> RawChannelsData + Sync + Send>,
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
    type Out = RawChannelsData;
    type Ctx = RunCtx2;

    async fn run<I>(self: Arc<Self>, _input: Arc<I>, _ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError>
    where
        I: WithId<ID=<<Self as Stage>::In as WithId>::ID> + Send + Sync + 'static
    {
        Ok((self.loader_fn)(_input.deref().id(), &self.device))
    }
}

impl Data for RawChannelsData {
    type Metrics = TwoDimsMetrics;
    type Sub = RawChannelsData;
    type SubSelect = std::ops::Range<Self::Metrics>;

    fn size(&self) -> Self::Metrics {
        self.t.size().try_into().expect("can't get size")
    }

    fn device(&self) -> usize {
        ThoDevice::from(self.t.device()).ordinal()
    }
}

impl OutId for TensorLoader {
    fn out_id(&self, input_id: &<Self::In as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("loaded_{}", input_id.id()))
    }
}

#[async_trait]
impl CopyToDevice for RawChannelsData {
    async fn copy_to_device(&self, dst_device: usize) ->Self {
        RawChannelsData::new(self.t.copy_to_device(dst_device).await)
    }
}


impl  TensorLoader {
    pub fn new<F>(loader_fn:F, device: ThoDevice) -> Self
    where
        F: Fn(&Resource, &Device) -> RawChannelsData + Sync + Send + 'static
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
    use rand::Rng;
    use tch::kind::FLOAT_CPU;
    use tch::{Device, Tensor};
    use caching_runner::common::cachable::Cached;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::applicables::ThoDevice;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::utils::lin_for_size;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::ThoosaError;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{RawChannelsData, TensorLoader};

    #[tokio::test]
    async fn test_load() -> Result<(), ThoosaError>{
        GlobalCtx::init_devices(2);


        let mut data = vec![0f32; 16];
        rand::rng().fill(&mut data[..]);
        let data_0 = data.clone();

        let loader = Arc::new(TensorLoader::new(
            move |key, _| {

                RawChannelsData::new(ThoTensor2::new(Input::id(key).clone(), Tensor::from_slice(&data_0).view((-1, 1))))
            },
            ThoDevice::from(Device::Cpu)
        ));

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let loaded = loader.run_cached(
            Arc::new(Resource::new(0, 0, "aaa".to_string(), 20)), ctx
        ).await.map_err(|e|ThoosaError::new("zzz"))?;


        let result = loaded.with_data(0, None, None, |v|{
            println!("loaded: v={v:?}");
            v.t.run_inner(
                |t|{
                    Ok(Tensor::from_slice(&data).view((-1,1)).allclose(t, 0.01, 0.01, false))
                }
            )
        }).await?;


        assert!(result);

        Ok(())
    }

}