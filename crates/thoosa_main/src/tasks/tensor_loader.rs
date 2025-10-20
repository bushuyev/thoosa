use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::RunCtx2;
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
use caching_runner::common::cachable::{Cached, DevicesWrap};
use thoosa_common::applicables::ThoDevice;
use thoosa_common::dims::Dims;
use thoosa_shared::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output, Runner, ToOrdinal};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use crate::tasks::transforms::freq_filter::FreqFilter;
// use crate::tasks::stages_seq::StagesSeq;

type LoadFn = dyn Fn(&Resource, &Device) -> ThoTensor2<ChannelsData2, SimpleId> + Sync + Send;

#[derive(cached_derive::Cached)]
#[cache_name = "LoadedTensorId"]
pub struct RawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    device: ThoDevice,
    loader_fn: Box<LoadFn>,
    _ctx: PhantomData<C>,
}

impl<C> Debug for RawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorLoader")
            .field("device", &self.device)
            .finish()
    }
}


impl<C> Stage for RawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    type In<'a> = Resource;
    type Out = ThoTensor2<ChannelsData2, SimpleId>;
    type Ctx = C;
}

#[async_trait]
impl<C> AsyncStage for RawTensorLoader<C>
where
    C: RunCtx + 'static,
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
            .with_data_ref(0, None, |resource| {
                let tensor = (self.loader_fn)(resource, &self.device);
                Ok(Arc::new(DevicesWrap::wrap(
                    tensor,
                    cache_area,
                )))
            })
            .await
    }
}

impl<C> OutId for RawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("loaded_{}", input_id.id()))
    }
}

impl<C> RawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    pub fn new<F>(loader_fn: F, device: ThoDevice) -> Self
    where
        F: Fn(&Resource, &Device) -> ThoTensor2<ChannelsData2, SimpleId> + Sync + Send + 'static,
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
    use crate::tasks::cross_dims::ChannelsData2;
    use async_trait::async_trait;
    use parking_lot::RwLock;
    use std::future::Future;
    use std::marker::PhantomData;
    use std::pin::Pin;
    use std::sync::Arc;
    use tch::{Device, Tensor};
    use tch::Device::Cpu;
    use tokio::runtime::Handle;
    use tokio::task::block_in_place;
    use cached_derive::Cached;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::{MockRunCtx, OutId, RunCtx};
    use caching_runner::common::runnable::{AsyncStage, Stage};
    use thoosa_shared::{CopyToDevice, ThoosaError};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{SimpleId, WithId};
    use torch_runner::data::ThoTensor2;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{mock_tensor_loader, LoadFn};


    //TODO use mockall
    #[derive(Clone, Cached)]
    pub struct MockTensorLoader<C>
    where
        C: RunCtx + 'static,
    {
        load_fn: Arc<LoadFn>,
        _ctx: PhantomData<C>,
    }

    impl<C> OutId for MockTensorLoader<C>
    where
        C: RunCtx + 'static,
    {
        fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
            SimpleId("mock")
        }
    }

    impl<C> std::fmt::Debug for MockTensorLoader<C>
    where
        C: RunCtx + 'static,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockCrossLoaderTensorSource").finish()
        }
    }
    impl<C> Stage for MockTensorLoader<C>
    where
        C: RunCtx + 'static,
    {
        type Ctx = C;
        type In<'a>
        where
            Self: 'a
        = Resource;
        type Out = ThoTensor2<ChannelsData2, SimpleId>;
    }

    #[async_trait]
    impl<C> AsyncStage for MockTensorLoader<C>
    where
        C: RunCtx + 'static,
    {

        async fn run(self: Arc<Self>, resource: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
            resource.with_data_ref(0, None, |d|{
                Ok(Arc::new(DevicesWrap::wrap(
                    (self.load_fn)(d, &Cpu), 0
                )))
            }).await
        }
    }

    pub struct MockTensorLoaderBuilder<C>
    where
        C: RunCtx + 'static,
    {
        load_fn: Option<Arc<LoadFn>>,
        _ctx: PhantomData<C>,
    }

    pub fn make_wrapper(
        id: &'static str,
        data: &[f32],
    ) -> Arc<DevicesWrap<ThoTensor2<ChannelsData2, SimpleId>>> {

        make_wrapper_t(id, Tensor::from_slice(data).view((1, data.len() as i64)))
    }

    pub fn make_wrapper_t(id: &'static str, tensor: Tensor) -> Arc<DevicesWrap<ThoTensor2<ChannelsData2, SimpleId>>> {
        let tho_tensor = ThoTensor2::new(SimpleId::from(id), tensor);
        Arc::new(DevicesWrap::wrap(tho_tensor, 0))
    }

    impl<C> MockTensorLoaderBuilder<C>
    where
        C: RunCtx + 'static,
    {

        pub fn with_data<H>(mut self, h: H) -> Self
        where
            H: Into<HashMap<u32, Arc<DevicesWrap<ThoTensor2<ChannelsData2, SimpleId>>>>> + Send + Sync,
        {
            let map: Arc<
                HashMap<u32, Arc<DevicesWrap<ThoTensor2<ChannelsData2, SimpleId>>>>,
            > = Arc::new(h.into());

            self.with_load_fn(move |resource, _ctx| {
                let map = Arc::clone(&map);

                let wrapper = map
                    .get(&resource.id())
                    .cloned()
                    .expect("mock loader data should be present");

                let tensor_result = block_in_place(|| Handle::current().block_on(async {
                    wrapper.with_data_ref(0, None, |t| {
                        Ok(ThoTensor2::new(
                            t.id().clone(),
                            t.t_ref().shallow_clone(),
                        ))
                    })
                    .await
                }));

                tensor_result.expect("mock loader data access failed")
            })
        }

        pub fn with_load_fn<F>(mut self, f: F) -> Self
        where
            F:Fn(&Resource, &Device) -> ThoTensor2<ChannelsData2, SimpleId> + Send + Sync + 'static,
        {
            self.load_fn = Some(Arc::new(f));
            self
        }

        pub fn build(self) -> MockTensorLoader<C> {
            let load_fn = self
                .load_fn
                .expect("MockCrossLoaderTensorSourceBuilder requires load_fn to be set");
            MockTensorLoader {
                load_fn,
                _ctx: PhantomData,
            }
        }
    }

    pub fn new<C>() -> MockTensorLoaderBuilder<C>
    where
        C: RunCtx + 'static,
    {
        MockTensorLoaderBuilder {
            load_fn: None,
            _ctx: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use caching_runner::common::cachable::{Cached, DevicesWrap};
    use caching_runner::global_ctx::GlobalCtx;
    use rand::Rng;
    use std::sync::Arc;
    use tch::{Device, Tensor};
    use thoosa_common::applicables::ThoDevice;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::applicables_runner::Input;
    use thoosa_shared::resource::Resource;
    use torch_runner::data::ThoTensor2;

    #[tokio::test]
    async fn test_load() -> Result<(), ThoosaError> {
        GlobalCtx::init_devices(2);

        let mut data = vec![0f32; 16];
        rand::rng().fill(&mut data[..]);
        let data_0 = data.clone();

        let loader = Arc::new(RawTensorLoader::<RunCtx2>::new(
            move |key, _| {
                ThoTensor2::new(
                    Input::id(key).clone(),
                    Tensor::from_slice(&data_0).view((-1, 1)),
                )
            },
            ThoDevice::from(Device::Cpu),
        ));

        let ctx = Arc::new(parking_lot::RwLock::new(RunCtx2::new(0, 0, "aa")));

        let loaded = loader
            .run_cached(Arc::new(DevicesWrap::wrap(Resource::new(0, 0, "aaa".to_string(), 20), 0)), ctx)
            .await
            .map_err(|e| ThoosaError::new("zzz"))?;

        let result = loaded
            .with_data_ref(0, None, |v| {
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
