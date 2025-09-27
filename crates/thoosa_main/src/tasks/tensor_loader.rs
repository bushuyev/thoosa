use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::RunCtx2;
use async_trait::async_trait;
use caching_runner::common::runnable::Stage;
use caching_runner::common::{Data, OutId};
use parking_lot::RwLock;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use tch::Device;
use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use thoosa_common::applicables::ThoDevice;
use thoosa_common::dims::Dims;
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output, Runner, ToOrdinal};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Named, SimpleId, WithId};
use torch_runner::data::ThoTensor2;

#[derive(cached_derive::Cached)]
#[cache_name = "LoadedTensorId"]
pub struct RawTensorLoader {
    device: ThoDevice,
    loader_fn: Box<dyn Fn(&Resource, &Device) -> ThoTensor2<ChannelsData2, SimpleId> + Sync + Send>,
}

impl Debug for RawTensorLoader {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorLoader")
            .field("device", &self.device)
            .finish()
    }
}

#[async_trait]
impl Stage for RawTensorLoader {
    type In<'a> = Resource;
    type Out = ThoTensor2<ChannelsData2, SimpleId>;
    type Ctx = RunCtx2;

    async fn run(
        self: Arc<Self>,
        _input: Arc<Self::In<'static>>,
        _ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError> {
        Ok((self.loader_fn)(&_input, &self.device))
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

#[async_trait]
pub trait TensorLoader: Send + Sync + Debug {
    async fn load(
        self: Arc<Self>,
        resource: Arc<Resource>,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>, ThoosaError>;
}

#[async_trait]
impl TensorLoader for RawTensorLoader {
    async fn load(
        self: Arc<Self>,
        resource: Arc<Resource>,
        ctx: Arc<parking_lot::RwLock<RunCtx2>>,
    ) -> Result<Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>, ThoosaError> {
        <Self as Cached>::run_cached(self, resource, ctx).await
    }
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
    use tch::Tensor;
    use caching_runner::common::cachable::DataCacheWrapper;
    use thoosa_shared::ThoosaError;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::SimpleId;
    use torch_runner::data::ThoTensor2;
    use crate::tasks::run_ctx_2::RunCtx2;
    use crate::tasks::tensor_loader::{mock_tensor_loader, TensorLoader};

    type LoadFuture = Pin<
        Box<
            dyn Future<
                    Output = Result<
                        Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>,
                        ThoosaError,
                    >,
                > + Send,
        >,
    >;

    type LoadFn = dyn Fn(Arc<Resource>, Arc<RwLock<RunCtx2>>) -> LoadFuture + Send + Sync;

    #[derive(Clone)]
    pub struct MockTensorLoader {
        load_fn: Arc<LoadFn>,
    }

    impl std::fmt::Debug for MockTensorLoader {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("MockCrossLoaderTensorSource").finish()
        }
    }

    #[async_trait]
    impl TensorLoader for MockTensorLoader {
        async fn load(
            self: Arc<Self>,
            resource: Arc<Resource>,
            ctx: Arc<RwLock<RunCtx2>>,
        ) -> Result<Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>, ThoosaError>
        {
            (self.load_fn)(resource, ctx).await
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
        Arc::new(DataCacheWrapper::new(tho_tensor, 0, 1, 0))
    }

    impl MockTensorLoaderBuilder {

        pub fn with_data<H>(mut self, h:H ) -> Self
        where
            H:Into<HashMap<u32, Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>>> + Send + Sync{

            let map:Arc<HashMap<u32, Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>>> = Arc::new(h.into());

            self.with_load_fn(move |resource, _ctx| {
                    let map = Arc::clone(&map);
                    async move {
                        Ok(map
                            .get(&resource.id())
                            .cloned()
                            .expect("mock loader data should be present"))
                    }
            })
        }

        pub fn with_load_fn<F, Fut>(mut self, f: F) -> Self
        where
            F: Fn(Arc<Resource>, Arc<RwLock<RunCtx2>>) -> Fut + Send + Sync + 'static,
            Fut: Future<
                    Output = Result<
                        Arc<DataCacheWrapper<ThoTensor2<ChannelsData2, SimpleId>>>,
                        ThoosaError,
                    >,
                > + Send
                + 'static,
        {
            self.load_fn = Some(Arc::new(move |resource, ctx| Box::pin(f(resource, ctx))));
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
    use caching_runner::common::cachable::Cached;
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
            .run_cached(Arc::new(Resource::new(0, 0, "aaa".to_string(), 20)), ctx)
            .await
            .map_err(|e| ThoosaError::new("zzz"))?;

        let result = loaded
            .with_data(0, None, None, |v| {
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
