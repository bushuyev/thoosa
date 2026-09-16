use std::any::Any;
use thoosa_common::dims::ChannelsData;
use thoosa_common::ipc::serialization::json::{IpcCodec, JsonIpcCodec};
use thoosa_common::ipc::wraps::SyncStageIpcWrapper;

use mockall_double::double;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use crate::stages::stage_ids::RAW_TENSOR_LOADER;
use log::debug;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tch::Device::Cpu;
use tch::{Device, Tensor};
use thoosa_common::dims::Dims;
use thoosa_common::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use thoosa_common::ipc::worker_msg::{WorkerInput, WorkerOutput};
use thoosa_common::ipc::wraps::WorkerWrapper;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap};
use thoosa_common::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoStageAdjuster, OOMHandler};
use thoosa_common::runner::run_ctx_2::RunCtxImpl;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::resource::Resource;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, AsyncStage, ConsumeResult, MaxParallelCtx, OutId, RunCtx, Stage, SyncStage, WithId};
use thoosa_shared::{ThoError, mono_key};
// use crate::stages::stages_seq::StagesSeq;


type LoadFn =
dyn Fn(&Resource, &Device) -> Result<ThoTensor<ChannelsData, SynKey>, ThoError> + Sync + Send;

#[derive(Clone)]
struct DebugLoadFn(Arc<LoadFn>);

impl Debug for DebugLoadFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // Loader closures are opaque, so expose only their stable role in debug output.
        f.write_str("LoadFn")
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

impl BorrowRef for RawTensorLoaderInputWrapper {
    type Out<'a> = Resource where Self: 'a;

    fn borrow_ref<'a>(&'a self, _to_device_idx: usize) -> impl std::future::Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        std::future::ready(Ok(self.resource.as_ref().clone()))
    }

}


///it always loads data into CPU tensors which are copied to gpu cells of its DevicesWrap output when BorrowRef::borrow_to is invoked from thoosa_common::runner::oom_handler::run_stage_borrowed
/// it needs to be on main side to retain CPU tensors when worker is restarted
#[derive(cached_derive::Cached, Named)]
#[cache_name = "RawTensors"]
pub struct RawTensorLoader<C>
where
    C: RunCtx + MaxParallelCtx + 'static,
{
    id: SynKey,
    device: ThoDevice,
    loader_fn: Arc<LoadFn>,
    _ctx: PhantomData<C>,
}

impl<C> Debug for RawTensorLoader<C>
where
    C: RunCtx + MaxParallelCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TensorLoader")
            .field("device", &self.device)
            .finish()
    }
}


impl<C> Stage for RawTensorLoader<C>
where
    C: RunCtx + MaxParallelCtx + 'static,
{
    type In<'a> = Arc<Resource>;
    type Out = Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>;
    type Ctx = C;
}

impl<C> AsyncStage for RawTensorLoader<C>
where
    C: RunCtx + MaxParallelCtx + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="RawTensorLoader::run"))]
    async fn run(
        self: Arc<Self>,
        input: Self::In<'static>,
        ctx: Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        debug!("RawTensorLoader::run: input={input:?} ctx={ctx:?}");

        let cpu_tensor = (self.loader_fn)(&input, &Cpu)?;
        debug!("RawTensorLoader::run: cpu_tensor={cpu_tensor:?}");

        let tensor = if ctx.tho_device_idx() != 0 {
            GlobalCtx::global().cuda_actor(ctx.tho_device_idx()).copy(cpu_tensor.strip_types(), ctx.tho_device_idx().into()).await?.to_types(self.out_id(input.id()))?
        } else {
            cpu_tensor
        };

        debug!("RawTensorLoader::run: tensor={tensor:?}");

        Ok(Arc::new(DevicesWrap::wrap(tensor)))
    }

}

impl<C> OutId for RawTensorLoader<C>
where
    C: RunCtx + MaxParallelCtx + 'static,
{
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + mono_key!(input_id.id(), format!("loaded_{}", input_id.id()))
    }
}

impl<C> RawTensorLoader<C>
where
    C: RunCtx + MaxParallelCtx + 'static,
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
    use async_trait::async_trait;
    use dashmap::DashMap;
    use futures_util::future::{BoxFuture, Shared};
    use mockall::mock;
    use std::collections::HashMap;
    use std::sync::Arc;
    use tch::Tensor;
    use thoosa_common::dims::ChannelsData;
    use thoosa_common::runner::common::MockRunCtx;
    use thoosa_common::runner::common::cachable::{Cached, DevicesWrap};
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_shared::ThoError;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::traits::{AsyncStage, OutId, Stage, SyncStage, WithId};
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
