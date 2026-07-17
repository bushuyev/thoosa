use std::any::Any;
use thoosa_common::dims::ChannelsData;
use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::runner::common::{OutId, RunCtx};

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use log::debug;
use tch::{Device, Tensor};
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap};
use thoosa_common::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::dims::Dims;
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_named_derive::Named;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::tracked_await::track_await;
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_proxy_derive::Proxy;
use crate::tasks::stage_ids::RAW_TENSOR_LOADER;
// use crate::tasks::stages_seq::StagesSeq;

pub const SYNC_RAW_TENSOR_LOADER_ENDPOINT: &str = "SyncRawTensorLoader";

type LoadFn =
    dyn Fn(&Resource, &Device) -> Result<ThoTensor<ChannelsData, SynKey>, ThoError> + Sync + Send;

#[derive(Clone)]
struct DebugLoadFn(Arc<LoadFn>);

impl Debug for DebugLoadFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Loader closures are opaque, so expose only the stable field name in proxy debug output.
        f.write_str("LoadFn")
    }
}

fn raw_tensor_ref_from_copy(input: ThoTensor, id: SynKey) -> Result<ThoTensor<ChannelsData, SynKey>, ThoError> {
    match input.inner {
        thoosa_common::tho_tensor::ThoTensorInner::Ref { key, size, kind, selector, tho_device_idx } => Ok(ThoTensor::new_ref(id, key, size, kind, selector, tho_device_idx)),
        thoosa_common::tho_tensor::ThoTensorInner::Val { .. } => Err(ThoError::new("SyncRawTensorLoader IPC copy must return a worker tensor reference")),
    }
}

#[derive(Debug)]
struct RawTensorLoaderInputWrapper {
    resource: Arc<Resource>,
}

impl RawTensorLoaderInputWrapper {
    fn new(resource: Arc<Resource>) -> Self {
        Self { resource }
    }
}

impl WithId for RawTensorLoaderInputWrapper {
    type ID = Resource;

    fn id(&self) -> &Self::ID {
        self.resource.id()
    }
}

impl BorrowRef for RawTensorLoaderInputWrapper {
    type Out<'a> = Resource where Self: 'a;

    fn borrow_to<'a>(&'a self, _to_device_idx: usize) -> impl std::future::Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        std::future::ready(Ok(self.resource.as_ref().clone()))
    }
}

#[derive(Debug)]
enum RawTensorLoaderDynSyncStage<C>
where
    C: RunCtx + 'static,
{
    Local(Arc<SyncRawTensorLoader<C>>),
    Proxy(Arc<ProxySyncRawTensorLoader>),
}

#[derive(Debug)]
struct SyncRawTensorLoaderIpcWrapper {
    // The derive initializes every wrapper with its target stage; this endpoint only validates
    // and returns the worker reference created by the proxy before invoking the endpoint.
    stage: Arc<SyncRawTensorLoader<RunCtxImpl>>,
}

impl SyncStageIpcWrapper for SyncRawTensorLoaderIpcWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
        let input_ref = JsonIpcCodec::decode::<ThoTensor<ChannelsData, SynKey>>(input)?;
        let input_key = input_ref.ref_key().ok_or_else(|| ThoError::new("SyncRawTensorLoader IPC input must be a worker tensor reference"))?;
        if !all_cuda_tensors.lock().contains_key(&input_key) {
            return Err(ThoError::new(format!("SyncRawTensorLoader IPC tensor key={input_key} was not found")));
        }
        JsonIpcCodec::encode(&input_ref)
    }
}

#[derive(Debug, Proxy)]
#[proxy(
    common = "thoosa_common",
    name = "ProxySyncRawTensorLoader",
    endpoint = "SYNC_RAW_TENSOR_LOADER_ENDPOINT",
    new_args = "loader_fn: Arc<LoadFn>, device: ThoDevice, tho_device_idx: usize",
    fields = "loader_fn: DebugLoadFn = DebugLoadFn(loader_fn), device: ThoDevice = device",
    out_id = "SynKey::new([mono_key!(input_id.id(), format!(\"resource_id={}\", input_id.id()))])",
    pre_run = "let loaded = (self.loader_fn.0)(input, &self.device)?; let input_id = loaded.id().clone(); let input: ThoTensor = ThoTensor::new((), loaded.t())?; let input = tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(async { GlobalCtx::global().cuda_actor(ctx.tho_device_idx()).copy(input, ctx.tho_device_idx().into()).await }))?; let input = raw_tensor_ref_from_copy(input, input_id)?;",
    run_input = "input",
    endpoint_init = "sync_raw_tensor_loader_endpoint_init",
    wrapper = "SyncRawTensorLoaderIpcWrapper"
)]
struct SyncRawTensorLoaderProxyTarget {
    base: SyncRawTensorLoader<RunCtxImpl>,
}

pub struct SyncRawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    device: ThoDevice,
    loader_fn: Arc<LoadFn>,
    pd: PhantomData<C>,
}

impl<C> Debug for SyncRawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncRawTensorLoader").field("device", &self.device).finish()
    }
}

impl<C> SyncRawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    async fn new(loader_fn: Arc<LoadFn>, device: ThoDevice, tho_device_idx: usize) -> Result<RawTensorLoaderDynSyncStage<C>, ThoError> {
        // IPC execution must return a worker-owned tensor reference for downstream proxy stages.
        if GlobalCtx::global().is_ipc() {
            Ok(RawTensorLoaderDynSyncStage::Proxy(Arc::new(ProxySyncRawTensorLoader::new(loader_fn, device, tho_device_idx).await?)))
        } else {
            Ok(RawTensorLoaderDynSyncStage::Local(Arc::new(Self::new_local(loader_fn, device))))
        }
    }

    fn new_local(loader_fn: Arc<LoadFn>, device: ThoDevice) -> Self {
        Self { device, loader_fn, pd: PhantomData }
    }
}

impl SyncRawTensorLoaderProxyTarget {
    fn new() -> SyncRawTensorLoader<RunCtxImpl> {
        // The custom endpoint wrapper returns an already-copied worker reference, so its target
        // stage is a derive-required placeholder and must never execute the loader closure.
        SyncRawTensorLoader::new_local(
            Arc::new(|_, _| Err(ThoError::new("SyncRawTensorLoader IPC placeholder stage must not run"))),
            ThoDevice::from(Device::Cpu),
        )
    }
}

impl<C> Adjustable for RawTensorLoaderDynSyncStage<C> where C: RunCtx + 'static {}

impl<C> OutId for RawTensorLoaderDynSyncStage<C>
where
    C: RunCtx + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        match self {
            RawTensorLoaderDynSyncStage::Local(_) => SynKey::new([mono_key!(input_id.id(), format!("resource_id={}", input_id.id()))]),
            RawTensorLoaderDynSyncStage::Proxy(proxy) => proxy.out_id(input_id),
        }
    }
}

impl<C> Stage for RawTensorLoaderDynSyncStage<C>
where
    C: RunCtx + 'static,
{
    type In<'a> = Resource;
    type Out = ThoTensor<ChannelsData, SynKey>;
    type Ctx = C;
}

impl<C> SyncStage for RawTensorLoaderDynSyncStage<C>
where
    C: RunCtx + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            RawTensorLoaderDynSyncStage::Local(local) => Arc::clone(local).do_run(input, ctx),
            RawTensorLoaderDynSyncStage::Proxy(proxy) => {
                let ctx = (ctx as &dyn Any).downcast_ref::<RunCtxImpl>().ok_or_else(|| ThoError::new("ProxySyncRawTensorLoader requires RunCtxImpl"))?;
                Arc::clone(proxy).do_run(input, ctx)
            }
        }
    }
}

impl<C> Adjustable for SyncRawTensorLoader<C> where C: RunCtx + 'static {}

impl<C> Stage for SyncRawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    type In<'a> = Resource;
    type Out = ThoTensor<ChannelsData, SynKey>;
    type Ctx = C;
}

impl<C> SyncStage for SyncRawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        (self.loader_fn)(input, &self.device)
    }
}

#[derive(cached_derive::Cached, Named)]
#[cache_name = "RawTensors"]
pub struct RawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    id: SynKey,
    device: ThoDevice,
    loader_fn: Arc<LoadFn>,
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
    type In<'a> = Arc<Resource>;
    type Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;
    type Ctx = C;
}

// #[async_trait]
impl<C> AsyncStage for RawTensorLoader<C>
where
    C: RunCtx + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="RawTensorLoader::run"))]
    async fn run(
        self: Arc<Self>,
        input: Self::In<'static>,
        ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        // TODO(perf): add a shared raw-resource tensor cache/prefetch layer here; traced runs repeatedly reload tens of thousands of raw tensors through this stage.
        let stage = Arc::new(SyncRawTensorLoader::<C>::new(Arc::clone(&self.loader_fn), self.device, ctx.tho_device_idx()).await?);
        let input = Arc::new(RawTensorLoaderInputWrapper::new(input));
        let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()> = OOMHandler::new(vec![], None);
        let tensor = track_await(oom_handler.run(&stage, input, ctx, NoContexts)).await?;

        debug!("RawTensorLoader::run: tensor={tensor:?}");

        Ok(Arc::new(DevicesWrap::wrap(tensor)))
    }

}

impl<C> OutId for RawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + mono_key!(input_id.id(), format!("loaded_{}", input_id.id()))
    }
}

impl<C> RawTensorLoader<C>
where
    C: RunCtx + 'static,
{
    pub fn new<F>(loader_fn: F, device: ThoDevice) -> Self
    where
        F: Fn(&Resource, &Device) -> Result<ThoTensor<ChannelsData, SynKey>, ThoError>
            + Sync
            + Send
            + 'static,
    {
        Self {
            id: (RAW_TENSOR_LOADER, "RawTensorLoader").into(),
            device,
            loader_fn: Arc::new(loader_fn),
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
    use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::{MockRunCtx, OutId};
    use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use thoosa_shared::ThoError;
    use thoosa_shared::resource::Resource;
    use thoosa_named_derive::Named;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::traits::WithId;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::dims::ChannelsData;
    use dashmap::DashMap;
    use parking_lot::RwLock;
    use tch::Tensor;
    mock! {
        #[derive(Debug, Named)]
        pub TensorLoader {}

        impl Stage for TensorLoader {
            type Ctx = MockRunCtx;

            type In<'a>  = Arc<Resource>;

            type Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;
        }

        impl OutId for TensorLoader {
            fn out_id<'a>(&self,input_id: &<<Self as Stage>::In<'a> as WithId>::ID,) -> <<MockTensorLoader as Stage>::Out as WithId>::ID;
        }

        // #[async_trait]
        impl AsyncStage for TensorLoader {
        }

        #[async_trait]
        impl Cached for TensorLoader {

            async fn cache() -> &'static DashMap<<<Self as Stage>::Out as WithId>::ID, Shared<BoxFuture<'static, Result<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey >>>, ThoError >>>>;
            async fn run_cached(
                    self: Arc<Self>,
                    input: <MockTensorLoader as Stage>::In<'static>,
                    ctx: <MockTensorLoader as Stage>::Ctx,
                ) -> Result<<MockTensorLoader as Stage>::Out, ThoError>;
        }
    }

    impl thoosa_shared::traits::Named for MockTensorLoader {
        fn name() -> &'static str { "MockTensorLoader" }
    }

    pub struct  MockTensorLoaderBuilder {
        mock:MockTensorLoader
    }

    impl MockTensorLoaderBuilder {

        pub fn with_out_id(mut self)->Self {
            self.mock.expect_out_id().returning(|input_id|{
                SynKey::new([mono_key!(input_id.id(), format!("loaded_{}", input_id.id()))])
            });
            self
        }

        pub fn with_run_cached<H>(mut self, h: H) -> Self
        where
            H: IntoIterator<Item=(Resource, Tensor)> ,
        {
            let data_map: Arc<HashMap<Resource, Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>> = Arc::new(h.into_iter().map(|(k, v)| {
                let id = SynKey::new([mono_key!(k.file_stem(), format!("file_stem={}", k.file_stem()))]);
                let tensor = ThoTensor::new(id, v)
                    .expect("ThoTensor::new failed in MockTensorLoaderBuilder");
                (k, Arc::new(DevicesWrap::wrap(tensor)))
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
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use crate::tasks::tensor_loader::RawTensorLoader;
    use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::common::RunCtx;
    use rand::{Fill, Rng, RngExt};
    use parking_lot::RwLock;
    use std::sync::Arc;
    use tch::{Device, Tensor};
    use thoosa_common::dims::ChannelsData;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_shared::ThoError;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_common::tho_tensor::ThoTensor;

    //TODO
    // #[tokio::test]
    // async fn test_load() -> Result<(), ThoError> {
    //     GlobalCtx::init_devices(2);
    //
    //     let mut data = vec![0f32; 16];
    //     rand::rng().fill(&mut data[..]);
    //     let data_0 = data.clone();
    //
    //     let loader = Arc::new(RawTensorLoader::<RunCtxImpl>::new(
    //         move |key, _| {
    //             Ok(ThoTensor::new(
    //                 SynKey::new([mono_key!(key.id(), format!("resource_id={}", key.id()))]),
    //                 Tensor::from_slice(&data_0).view((-1, 1)),
    //             )?)
    //         },
    //         ThoDevice::from(Device::Cpu),
    //     ));
    //
    //     let ctx = Arc::new(parking_lot::RwLock::new(RunCtxImpl::new(0, "test_load")));
    //
    //     let load_ctx = ctx.write().branch(false);
    //     let loaded = loader
    //         .run_cached(Arc::new(DevicesWrap::wrap(Resource::new(0, 0, "aaa", 20), CacheArea::default())), load_ctx)
    //         .await
    //         ?;
    //
    //     let mut data_ctx = ctx.write().branch(false);
    //     let result = loaded
    //         .with_data_ref(0,
    //         &mut data_ctx, None, |v: &ThoTensor<ChannelsData2, SynKey>, _ctx| {
    //            println!("loaded: v={v:?}");
    //            v.run_inner(|t| {
    //                Ok(Tensor::from_slice(&data)
    //                    .view((-1, 1))
    //                    .allclose(t, 0.01, 0.01, false))
    //            })
    //        })
    //         .await?;
    //
    //     assert!(result);
    //
    //     Ok(())
    // }
}
