use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::RunCtx2;
use async_trait::async_trait;
use caching_runner::common::runnable::{AsyncStage, Stage};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::global_ctx::GlobalCtx;
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use tch::Device;
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use thoosa_common::applicables::ThoDevice;
use thoosa_common::dims::Dims;
use thoosa_shared::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output, Runner, ToOrdinal};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use torch_runner::data::ThoTensor2;
use crate::tasks::filters::band_pass::BandPassFilter;
// use crate::tasks::stages_seq::StagesSeq;

type LoadFn = dyn Fn(&Resource, &Device) -> ThoTensor2<ChannelsData2, SimpleId> + Sync + Send;

#[derive(cached_derive::Cached)]
#[cache_name = "LoadedTensorId"]
pub struct RawTensorLoader {
    device: ThoDevice,
    loader_fn: Box<LoadFn>,
}

impl Debug for RawTensorLoader {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorLoader")
            .field("device", &self.device)
            .finish()
    }
}


impl Stage for RawTensorLoader {
    type In<'a> = Resource;
    type Out = ThoTensor2<ChannelsData2, SimpleId>;
    type Ctx = RunCtx2;
}

#[async_trait]
impl AsyncStage for RawTensorLoader {

    async fn run(
        self: Arc<Self>,
        input: Arc<DataCacheWrapper<Self::In<'static>>>,
        ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
    ) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {
        let cache_area = ctx.read().cache_area();

        let num_devices = GlobalCtx::global().device_count();
        let target_device = self.device.ordinal();

        input
            .with_data_ref(0, None, |resource| {
                let tensor = (self.loader_fn)(resource, &self.device);
                Ok(Arc::new(DataCacheWrapper::new(
                    tensor,
                    target_device,
                    num_devices,
                    cache_area,
                )))
            })
            .await
    }
}

impl OutId for RawTensorLoader {
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        SimpleId::from(format!("loaded_{}", input_id.id()))
    }
}

impl RawTensorLoader {
    pub fn new<F>(loader_fn: F, device: ThoDevice) -> Self
    where
        F: Fn(&Resource, &Device) -> ThoTensor2<ChannelsData2, SimpleId> + Sync + Send + 'static,
    {
        Self {
            device,
            loader_fn: Box::new(loader_fn),
        }
    }
}

pub fn wrap_resource_for_cache(
    resource: Resource,
    num_devices: usize,
    cache_area: u64,
) -> Arc<DataCacheWrapper<Resource>> {
    Arc::new(DataCacheWrapper::new(
        resource,
        0,
        num_devices,
        cache_area,
    ))
}


#[cfg(test)]
pub(crate) mod mock_tensor_loader {
    use std::collections::HashMap;
    use crate::tasks::cross_dims::ChannelsData2;
    use async_trait::async_trait;
    use parking_lot::RwLock;
    use std::future::Future;
    use std::pin::Pin;
    use std::sync::Arc;
    use tch::{Device, Tensor};
    use tch::Device::Cpu;
    use tokio::runtime::Handle;
    use tokio::task::block_in_place;
    use caching_runner::common::cachable::DataCacheWrapper;
    use caching_runner::common::MockRunCtx;
    use caching_runner::common::runnable::{AsyncStage, Stage};
    use thoosa_shared::{CopyToDevice, ThoosaError};
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::SimpleId;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{mock_tensor_loader, LoadFn};




    #[derive(Clone)]
    pub struct MockTensorLoader {
        load_fn: Arc<LoadFn>,
    }

    impl std::fmt::Debug for MockTensorLoader {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockCrossLoaderTensorSource").finish()
        }
    }
    impl Stage for MockTensorLoader {
        type Ctx = RunCtx2;
        type In<'a>
        where
            Self: 'a
        = Resource;
        type Out = ThoTensor2<ChannelsData2, SimpleId>;
    }

    #[async_trait]
    impl AsyncStage for MockTensorLoader {



        async fn run(self: Arc<Self>, resource: Arc<DataCacheWrapper<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {
            resource.with_data_ref(0, None, |d|{
                Ok(Arc::new(DataCacheWrapper::new(
                    (self.load_fn)(d, &Cpu), 0, 2, 0
                )))
            }).await
        }
    }

    pub struct MockTensorLoaderBuilder {
        load_fn: Option<Arc<LoadFn>>,
    }

    pub fn make_wrapper(
        id: &'static str,
        data: &[f32],
    ) -> Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>> {

        make_wrapper_t(id, Tensor::from_slice(data).view((1, data.len() as i64)))
    }

    pub fn make_wrapper_t(id: &'static str, tensor: Tensor) -> Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>> {
        let tho_tensor = ThoTensor2::new(SimpleId::from(id), tensor);
        Arc::new(DataCacheWrapper::new(tho_tensor, 0, 2, 0))
    }

    impl MockTensorLoaderBuilder {

        pub fn with_data<H>(mut self, h: H) -> Self
        where
            H: Into<HashMap<u32, Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>>> + Send + Sync,
        {
            let map: Arc<
                HashMap<u32, Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>>,
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

        pub fn build(self) -> MockTensorLoader {
            let load_fn = self
                .load_fn
                .expect("MockCrossLoaderTensorSourceBuilder requires load_fn to be set");
            MockTensorLoader { load_fn }
        }
    }

    pub fn new() -> MockTensorLoaderBuilder {
        MockTensorLoaderBuilder { load_fn: None }
    }
}

#[cfg(test)]
mod tests {
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use caching_runner::common::cachable::{Cached, DataCacheWrapper};
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

        let loader = Arc::new(RawTensorLoader::new(
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
            .run_cached(Arc::new(DataCacheWrapper::new(Resource::new(0, 0, "aaa".to_string(), 20), 0, 2, 0)), ctx)
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
