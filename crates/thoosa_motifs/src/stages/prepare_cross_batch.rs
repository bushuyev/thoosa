use super::cross_batch_clc::{CrossInput, UnfoldParams, unfold_and_cat};
use futures_util::{FutureExt, future::BoxFuture};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, sync::{Arc, atomic::{AtomicU32, Ordering}}};
use thoosa_common::dims::AnyDims;
use thoosa_common::instance_cache::InstanceCache;
use thoosa_common::ipc::cuda_wokrer::{AllTensors, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use thoosa_common::ipc::serialization::json::JsonIpcCodec;
use thoosa_common::ipc::worker_msg::WorkerInput;
use thoosa_common::ipc::wraps::{SyncStageIpcWrapper, WorkerWrapper};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::sync_stage::SyncStage;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::applicables_runner::Output;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::ipc_types::WorkerOutput;
use thoosa_shared::resource::Resource;
use thoosa_shared::traits::{Adjustable, Data, OutId, RunCtx, Stage, WithId};
use thoosa_shared::{CopyToDevice, ThoError};

pub(crate) const PREPARE_CROSS_BATCH_ENDPOINT: &str = "PrepareCrossBatch";

// The concatenated windows and their resource boundaries travel together when
// OOM handling offloads the outer batch or IPC materializes its worker reference.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct PreparedCrossBatch {
    pub tensor: ThoTensor<AnyDims, SynKey>,
    pub sizes: Vec<i64>,
    pub params: UnfoldParams,
}

impl PreparedCrossBatch {
    pub fn clone_refs_for_worker(&self) -> Result<Self, ThoError> {
        Ok(Self { tensor: self.tensor.clone_ref()?, sizes: self.sizes.clone(), params: self.params })
    }
}

impl Data for PreparedCrossBatch {
    type Metrics = ();
    fn size(&self) {}
    fn tho_device_idx(&self) -> usize { self.tensor.tho_device_idx() }
    fn mem_size(&self) -> u64 { self.tensor.mem_size() }
}

#[async_trait::async_trait]
impl CopyToDevice for PreparedCrossBatch {
    async fn copy_to_device(&self, device: usize) -> Result<Self, ThoError> {
        Ok(Self { tensor: self.tensor.copy_to_device(device).await?, sizes: self.sizes.clone(), params: self.params })
    }
}

impl WithId for PreparedCrossBatch {
    type ID = SynKey;
    fn id(&self) -> &Self::ID { self.tensor.id() }
}
impl Output for PreparedCrossBatch {}
#[typetag::serde(name = "PreparedCrossBatch")]
impl WorkerOutput for PreparedCrossBatch {}

impl WorkerStageInput for PreparedCrossBatch {
    fn ref_to_tensor(&self, tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        Ok(Self { tensor: self.tensor.ref_to_tensor(tensors, remove)?, sizes: self.sizes.clone(), params: self.params })
    }
}

impl WorkerStageOutput for PreparedCrossBatch {
    fn encode_from_worker(self, tensors: &AllTensors, device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(Self { tensor: self.tensor.encode_from_worker(tensors, device)?, sizes: self.sizes, params: self.params })
    }
}

impl WorkerStageInput for UnfoldParams {}
#[typetag::serde(name = "PrepareCrossBatchInit")]
impl WorkerInput for UnfoldParams {}

#[derive(Debug)]
pub(crate) struct PrepareCrossBatch<C: RunCtx + 'static> {
    params: UnfoldParams,
    ctx: PhantomData<C>,
}

static PREPARE_CROSS_BATCH_INSTANCES: InstanceCache = InstanceCache::new();

impl<C: RunCtx + 'static> PrepareCrossBatch<C> {
    // One endpoint per discovery configuration, reused across all outer batches.
    pub async fn new(params: UnfoldParams, tho_device_idx: usize) -> Result<Arc<DynSyncStageEnum<Self>>, ThoError> {
        let id = params.new_id().to_string();

        PREPARE_CROSS_BATCH_INSTANCES.get_or_try_insert_recreatable_with::<C, _, _, _, _>(tho_device_idx, GlobalCtx::global().cuda_activity_gate(tho_device_idx), GlobalCtx::global().is_ipc(), id.clone(), move || async move {
            Ok(Arc::new(if GlobalCtx::global().is_ipc() {

                DynSyncStageEnum::Proxy {
                    stage: Arc::new(ProxyStage::proxy_recreatable(PREPARE_CROSS_BATCH_ENDPOINT, Some(id), tho_device_idx, Box::new(params)).await?),
                    init: (),
                }
            } else {
                DynSyncStageEnum::Local(Arc::new(Self { params, ctx: PhantomData }))
            }))

        }).await
    }
}

impl<C: RunCtx + 'static> Adjustable for PrepareCrossBatch<C> {}
impl<C: RunCtx + 'static> Stage for PrepareCrossBatch<C> {
    type Ctx = C;
    type In<'a> = CrossInput<'a, Arc<Resource>>;
    type Out = PreparedCrossBatch;
}
impl<C: RunCtx + 'static> OutId for PrepareCrossBatch<C> {
    fn out_id<'a>(&self, id: &SynKey) -> SynKey { id.clone() }
}
impl<C: RunCtx + 'static> SyncStage for PrepareCrossBatch<C> {
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &C) -> Result<Self::Out, ThoError> {
        let (tensor, sizes) = unfold_and_cat(&input.tensors(), &self.params)?;
        Ok(PreparedCrossBatch { tensor: ThoTensor::new(input.id().clone(), tensor)?, sizes, params: self.params })
    }
}

pub(crate) fn prepare_cross_batch_endpoint_init<'a>(_id: Option<String>, init: Box<dyn WorkerInput>, _ws: Arc<RwLock<Box<dyn crate::motifs_ws::MotifsWorkspace>>>, _tensors: &'a AllTensors, _device: &'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    async move {
        let init: Box<dyn std::any::Any> = init;
        let Ok(params) = init.downcast::<UnfoldParams>() else {
            return Err(ThoError::new("invalid outer batch preparation initializer"));
        };
        let stage = PrepareCrossBatch::<WokerRunCtx> { params: *params, ctx: PhantomData };
        Ok(Box::new(WorkerWrapper::<JsonIpcCodec, _, _, _>::new(Arc::new(stage), PhantomData)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}
