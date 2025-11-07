use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::{MotifsCtx, RunCtx2};
use async_trait::async_trait;
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::global_ctx::GlobalCtx;
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use tch::Device;
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_common::applicables::ThoDevice;
use thoosa_common::dims::Dims;
use thoosa_shared::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output, Runner, ToOrdinal};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use thoosa_common::tho_tensor::ThoTensor;
use crate::tasks::transforms::freq_filter::FreqFilter;
// use crate::tasks::stages_seq::StagesSeq;

type LoadFn = dyn Fn(&Resource, &Device) -> ThoTensor<ChannelsData2, SimpleId> + Sync + Send;

#[derive(cached_derive::Cached)]
#[cache_name = "LoadedTensorId"]
pub struct RawTensorLoader<C>
where
    C: RunCtx + MotifsCtx + 'static,
{
    device: ThoDevice,
    loader_fn: Box<LoadFn>,
    _ctx: PhantomData<C>,
}

impl<C> Debug for RawTensorLoader<C>
where
    C: RunCtx + MotifsCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorLoader")
            .field("device", &self.device)
            .finish()
    }
}


impl<C> Stage for RawTensorLoader<C>
where
    C: RunCtx + MotifsCtx + 'static,
{
    type In<'a> = Resource;
    type Out = ThoTensor<ChannelsData2, SimpleId>;
    type Ctx = C;
}

#[async_trait]
impl<C> AsyncStage for RawTensorLoader<C>
where
    C: RunCtx + MotifsCtx + 'static,
{

    async fn run(
        self: Arc<Self>,
        input: Arc<DevicesWrap<Self::In<'static>>>,
        ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let cache_area = ctx.read().cache_area();

        let num_devices = GlobalCtx::global().device_count();
        let target_device = self.device.ordinal();

        input
            .with_data_ref(0, None, |resource: &Resource| {
                let tensor = (self.loader_fn)(resource, &self.device);
                Ok(Arc::new(DevicesWrap::wrap(
                    tensor,
                    cache_area,
                )))
            })
            .await
    }

    fn on_cached<'a>(&self, input: &impl WithId<ID=<<Self as Stage>::In<'a> as WithId>::ID>, ctx: &Arc<RwLock<Self::Ctx>>)
    where
        Self: 'a
    {
        
    }
}

impl<C> OutId for RawTensorLoader<C>
where
    C: RunCtx + MotifsCtx + 'static,
{
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("loaded_{}", input_id.id()))
    }
}

impl<C> RawTensorLoader<C>
where
    C: RunCtx + MotifsCtx + 'static,
{
    pub fn new<F>(loader_fn: F, device: ThoDevice) -> Self
    where
        F: Fn(&Resource, &Device) -> ThoTensor<ChannelsData2, SimpleId> + Sync + Send + 'static,
    {
        Self {
            device,
            loader_fn: Box::new(loader_fn),
            _ctx: PhantomData,
        }
    }
}

#[cfg(test)]
pub(crate) mod mock_tensor_loader {
    use std::collections::HashMap;
    use async_trait::async_trait;
    use futures_util::future::{BoxFuture, Shared};
    use mockall::mock;
    use std::sync::Arc;
    use caching_runner::common::cachable::{Cached, DevicesWrap};
    use caching_runner::common::{MockRunCtx, OutId};
    use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use thoosa_shared::{CopyToDevice, ThoosaError};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{Named, SimpleId};
    use thoosa_shared::traits::WithId;
    use thoosa_common::tho_tensor::ThoTensor;
    use crate::tasks::cross_dims::ChannelsData2;
    use dashmap::DashMap;
    use parking_lot::RwLock;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use tokio::runtime::Handle;
    use tokio::task::block_in_place;
    use caching_runner::global_ctx::GlobalCtx;

    mock! {
        #[derive(Debug)]
        pub TensorLoader {}

        impl Stage for TensorLoader {
            type Ctx = MockRunCtx;

            type In<'a>  = Resource;

            type Out = ThoTensor<ChannelsData2, SimpleId>;
        }

        impl OutId for TensorLoader {
            fn out_id<'a>(&self,input_id: &<<Self as Stage>::In<'a> as WithId>::ID,) -> <<MockTensorLoader as Stage>::Out as WithId>::ID;
        }

        #[async_trait]
        impl AsyncStage for TensorLoader {}

        #[async_trait]
        impl Cached for TensorLoader {

            async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<ThoTensor<ChannelsData2, SimpleId>>>, ThoosaError>>>>;
            async fn run_cached(
                    self: Arc<Self>,
                    input: Arc<DevicesWrap<<MockTensorLoader as Stage>::In<'static>>>,
                    ctx: Arc<RwLock<<MockTensorLoader as Stage>::Ctx>>,
                ) -> Result<Arc<DevicesWrap<<MockTensorLoader as Stage>::Out>>, ThoosaError>;
        }
    }

    pub struct  MockTensorLoaderBuilder {
        mock:MockTensorLoader
    }

    impl MockTensorLoaderBuilder {

        pub fn with_out_id(mut self)->Self {
            self.mock.expect_out_id().returning(|input_id|{
                SimpleId::from(format!("loaded_{}", input_id.id()))
            });
            self
        }

        pub fn with_run_cached<H>(mut self, h: H) -> Self
        where
            H: IntoIterator<Item=(Resource, Tensor)> ,
        {
            let data_map: Arc<HashMap<Resource, Arc<DevicesWrap<ThoTensor<ChannelsData2, SimpleId>>>>> = Arc::new(h.into_iter().map(|(k, v)| {
                let id = format!("{}", k.file_stem());
                (k, Arc::new(DevicesWrap::wrap(ThoTensor::new(id.into(), v), 0)))
            }).collect());

            self.mock.expect_run_cached().returning(move |input, ctx|{
                Ok(Arc::clone(data_map
                    .get(&input.id())
                    .expect(&format!("no mock data for {input:?} in MockTensorLoader"))))
            });

            self
        }

        pub fn finish(self) -> MockTensorLoader{
            self.mock
        }
    }

    pub fn build()->MockTensorLoaderBuilder {
        MockTensorLoaderBuilder {
            mock: Default::default(),
        }
    }

}


#[cfg(test)]
mod tests {
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
    use caching_runner::global_ctx::GlobalCtx;
    use rand::Rng;
    use std::sync::Arc;
    use tch::{Device, Tensor};
    use crate::tasks::cross_dims::ChannelsData2;
    use thoosa_common::applicables::ThoDevice;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::SimpleId;
    use thoosa_common::tho_tensor::ThoTensor;

    #[tokio::test]
    async fn test_load() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);

        let mut data = vec![0f32; 16];
        rand::rng().fill(&mut data[..]);
        let data_0 = data.clone();

        let loader = Arc::new(RawTensorLoader::<RunCtx2>::new(
            move |key, _| {
                ThoTensor::new(
                    Input::id(key).clone(),
                    Tensor::from_slice(&data_0).view((-1, 1)),
                )
            },
            ThoDevice::from(Device::Cpu),
        ));

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let loaded = loader
            .run_cached(Arc::new(DevicesWrap::wrap(Resource::new(0, 0, "aaa", 20), 0)), ctx)
            .await
            .map_err(|e| ThoosaError::new("zzz"))?;

        let result = loaded
            .with_data_ref(0, None, |v: &ThoTensor<ChannelsData2, SimpleId>| {
                println!("loaded: v={v:?}");
                v.run_inner(|t| {
                    Ok(Tensor::from_slice(&data)
                        .view((-1, 1))
                        .allclose(t, 0.01, 0.01, false))
                })
            })
            .await?;

        assert!(result);

        Ok(())
    }
}
