use super::cross_batch_clc::{CrossInput, UnfoldParams, unfold_and_cat};
use super::resource_batch::ResourceWindows;
use super::slices_iter::ResourceSlices;
use futures_util::{FutureExt, future::BoxFuture};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{marker::PhantomData, sync::Arc};
use thoosa_common::dims::AnyDims;
use thoosa_common::instance_cache::InstanceCache;
use thoosa_common::ipc::cuda_worker::{AllTensors, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
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
#[serde(try_from = "PreparedCrossBatchWire")]
pub(crate) struct PreparedCrossBatch {
    tensor: ThoTensor<AnyDims, SynKey>,
    layout: Vec<ResourceWindows<u32>>,
    params: UnfoldParams,
}

#[derive(Deserialize)]
struct PreparedCrossBatchWire {
    tensor: ThoTensor<AnyDims, SynKey>,
    layout: Vec<ResourceWindows<u32>>,
    params: UnfoldParams,
}

impl TryFrom<PreparedCrossBatchWire> for PreparedCrossBatch {
    type Error = ThoError;
    fn try_from(wire: PreparedCrossBatchWire) -> Result<Self, ThoError> {
        Self::new(wire.tensor, wire.layout, wire.params)
    }
}

impl PreparedCrossBatch {
    fn new(tensor: ThoTensor<AnyDims, SynKey>, layout: Vec<ResourceWindows<u32>>, params: UnfoldParams) -> Result<Self, ThoError> {
        params.validate()?;
        let mut total = 0i64;
        for entry in &layout {
            if entry.windows != params.windows(entry.samples)? {
                return Err(ThoError::new("prepared resource window count does not match its sample length"));
            }
            total = total.checked_add(entry.windows).ok_or_else(|| ThoError::new("prepared batch window count overflow"))?;
        }
        let shape = tensor.size();
        if layout.is_empty() || total == 0 || shape.len() != 3 || shape[0] <= 0 || shape[1] != total || shape[2] != params.size() {
            return Err(ThoError::new("prepared batch tensor does not match its resource window layout"));
        }
        Ok(Self { tensor, layout, params })
    }

    pub fn validate_resources<R: AsRef<Resource>>(&self, resources: &[R]) -> Result<(), ThoError> {
        if resources.len() != self.layout.len() || resources.iter().zip(&self.layout).any(|(resource, entry)| resource.as_ref().id() != entry.resource) {
            return Err(ThoError::new("prepared batch resource identities do not match"));
        }
        Ok(())
    }

    pub fn validate_lengths(&self, lengths: &[i64]) -> Result<(), ThoError> {
        if lengths.len() != self.layout.len() || lengths.iter().zip(&self.layout).any(|(length, entry)| *length != entry.samples) {
            return Err(ThoError::new("prepared batch sample lengths do not match"));
        }
        Ok(())
    }

    pub fn tensor(&self) -> &ThoTensor<AnyDims, SynKey> { &self.tensor }
    pub fn params(&self) -> UnfoldParams { self.params }
    pub fn slices(&self) -> Vec<ResourceSlices> {
        self.layout.iter().filter(|entry| entry.windows > 0).map(|entry| ResourceSlices::new(entry.resource, entry.windows as usize)).collect()
    }

    pub fn clone_refs_for_worker(&self) -> Result<Self, ThoError> {
        Self::new(self.tensor.clone_ref()?, self.layout.clone(), self.params)
    }
}

impl Data for PreparedCrossBatch {
    type Metrics = ();
    fn size(&self) {}
    fn tho_device_idx(&self) -> usize {
        self.tensor.tho_device_idx()
    }
    fn mem_size(&self) -> u64 {
        self.tensor.mem_size()
    }
}

#[async_trait::async_trait]
impl CopyToDevice for PreparedCrossBatch {
    async fn copy_to_device(&self, device: usize) -> Result<Self, ThoError> {
        Self::new(self.tensor.copy_to_device(device).await?, self.layout.clone(), self.params)
    }
}

impl WithId for PreparedCrossBatch {
    type ID = SynKey;
    fn id(&self) -> &Self::ID {
        self.tensor.id()
    }
}
impl Output for PreparedCrossBatch {}
#[typetag::serde(name = "PreparedCrossBatch")]
impl WorkerOutput for PreparedCrossBatch {}

impl WorkerStageInput for PreparedCrossBatch {
    fn ref_to_tensor(&self, tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        Self::new(self.tensor.ref_to_tensor(tensors, remove)?, self.layout.clone(), self.params)
    }
}

impl WorkerStageOutput for PreparedCrossBatch {
    fn encode_from_worker(self, tensors: &AllTensors, device: &ThoDevice) -> Result<Self, ThoError> {
        Self::new(self.tensor.encode_from_worker(tensors, device)?, self.layout, self.params)
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

        PREPARE_CROSS_BATCH_INSTANCES
            .get_or_try_insert_recreatable_with::<C, _, _, _, _>(tho_device_idx, GlobalCtx::global().cuda_activity_gate(tho_device_idx), GlobalCtx::global().is_ipc(), id.clone(), move || async move {
                Ok(Arc::new(if GlobalCtx::global().is_ipc() {
                    DynSyncStageEnum::Proxy {
                        stage: Arc::new(ProxyStage::proxy_recreatable(PREPARE_CROSS_BATCH_ENDPOINT, Some(id), tho_device_idx, Box::new(params)).await?),
                        init: (),
                    }
                } else {
                    DynSyncStageEnum::Local(Arc::new(Self { params, ctx: PhantomData }))
                }))
            })
            .await
    }
}

impl<C: RunCtx + 'static> Adjustable for PrepareCrossBatch<C> {}
impl<C: RunCtx + 'static> Stage for PrepareCrossBatch<C> {
    type Ctx = C;
    type In<'a> = CrossInput<'a, Arc<Resource>>;
    type Out = PreparedCrossBatch;
}
impl<C: RunCtx + 'static> OutId for PrepareCrossBatch<C> {
    fn out_id<'a>(&self, id: &SynKey) -> SynKey {
        id.clone()
    }
}
impl<C: RunCtx + 'static> SyncStage for PrepareCrossBatch<C> {
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, _ctx: &C) -> Result<Self::Out, ThoError> {
        if !input.batch_1().is_empty() {
            return Err(ThoError::new("outer preparation requires an empty inner batch"));
        }
        let (tensor, layout) = unfold_and_cat(input.raw_outer()?, &self.params)?;
        let layout = layout.into_iter().map(|entry| ResourceWindows { resource: entry.resource.as_ref().id(), samples: entry.samples, windows: entry.windows }).collect();
        PreparedCrossBatch::new(ThoTensor::new(input.id().clone(), tensor)?, layout, self.params)
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
    }
    .boxed()
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Kind;
    use thoosa_common::ipc::worker_msg::CudaWorkerId;

    #[test]
    fn test_prepared_batch_validates_layout_and_wire() -> anyhow::Result<()> {
        let tensor = ThoTensor::new_ref("prepared".into(), 15, vec![1, 5, 3], Kind::Float, None, CudaWorkerId { device_idx: 1, generation: thoosa_shared::CudaDeviceGeneration(7) });
        let layout = vec![ResourceWindows { resource: 10, samples: 5, windows: 2 }, ResourceWindows { resource: 11, samples: 1, windows: 0 }, ResourceWindows { resource: 12, samples: 7, windows: 3 }];
        let prepared = PreparedCrossBatch::new(tensor, layout, UnfoldParams::new(1, 3, 2))?;
        let resources: Vec<Arc<Resource>> = (10..13).map(|id| Arc::new(Resource::new(id, 1, "resource", 100))).collect();
        prepared.validate_resources(&resources)?;
        prepared.validate_lengths(&[5, 1, 7])?;
        assert!(prepared.validate_resources(&[resources[2].clone(), resources[1].clone(), resources[0].clone()]).is_err());
        assert!(prepared.validate_lengths(&[5, 2, 7]).is_err());
        let input = CrossInput::from_prepared(thoosa_common::runner::common::cachable::ThoDataRef::Borrowed(&prepared), resources, super::super::resource_batch::ResourceBatch::default(), "prepared_input".into())?;
        let mut input_wire = serde_json::to_value(&input)?;
        input_wire["outer"]["Prepared"]["resources"].as_array_mut().expect("resource array").reverse();
        assert!(serde_json::from_value::<CrossInput<'static, Arc<Resource>>>(input_wire).is_err());
        let wire = serde_json::to_value(&prepared)?;
        let decoded: PreparedCrossBatch = serde_json::from_value(wire.clone())?;
        assert_eq!(decoded.slices(), vec![ResourceSlices::new(10, 2), ResourceSlices::new(12, 3)]);
        let mut malformed = wire.clone();
        malformed["layout"][1]["windows"] = serde_json::json!(1);
        assert!(serde_json::from_value::<PreparedCrossBatch>(malformed).is_err());
        let mut malformed = wire.clone();
        malformed["tensor"]["size"] = serde_json::json!([1, 6, 3]);
        assert!(serde_json::from_value::<PreparedCrossBatch>(malformed).is_err());
        let mut malformed = wire;
        malformed["params"]["step"] = serde_json::json!(0);
        assert!(serde_json::from_value::<PreparedCrossBatch>(malformed).is_err());
        Ok(())
    }
}
