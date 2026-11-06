use super::*;
use crate::cuda_utils::tensor_mem_size;
use thoosa_common::ipc::tensor_info::WorkerTensorInfoPayload;
use thoosa_common::tch_error::tch_result;
use thoosa_shared::ipc_types::TensorKey;

/// An immutable CPU recovery point, retained independently of live/offloaded state.
/// Metadata is encoded so every restoration gets its own owned copy.
#[derive(Debug)]
pub(crate) struct CpuStateCheckpoint {
    info: Vec<u8>,
    pub(crate) tensors: HashMap<String, ThoTensor>,
}

impl CpuStateCheckpoint {
    pub(crate) fn new(info: StageStateInfo, tensors: HashMap<String, ThoTensor>) -> Result<Self, ThoError> {
        Ok(Self { info: serde_json::to_vec(&info)?, tensors })
    }

    pub(crate) fn info(&self) -> Result<StageStateInfo, ThoError> {
        Ok(serde_json::from_slice(&self.info)?)
    }

    fn from_dump(mut dump: StageStateDump) -> Result<Self, ThoError> {
        let tensors = dump
            .tensors
            .iter()
            .map(|(key, tensor)| WorkerTensorInfoPayload::new(TensorKey::stage_state("", "", key), tensor.kind().into(), tensor.size(), tensor_mem_size(tensor.kind(), &tensor.size())))
            .collect();
        Self::new(StageStateInfo::new(tensors, dump.scalars.take()), dump.tensors)
    }

    fn restore_dump(&self) -> Result<StageStateDump, ThoError> {
        let tensors = self.tensors.iter().map(|(key, tensor)| Ok((key.clone(), ThoTensor::new((), tch_result(tensor.local()?.f_alias())?)?))).collect::<Result<_, ThoError>>()?;
        // Deferred/offloaded checkpoints must retain optimizer and VarStore metadata.
        Ok(StageStateDump::new(self.info()?.scalars, tensors))
    }
}

// Retain state owners and CUDA admission through cancellation and copy completion.
pub(crate) async fn checkpoint_operation<T: Send + 'static>(tho_device_idx: usize, future: impl Future<Output = Result<T, ThoError>> + Send + 'static) -> Result<T, ThoError> {
    checkpoint_operation_on(GlobalCtx::global().cuda_activity_gate(tho_device_idx), future).await
}

async fn checkpoint_operation_on<T: Send + 'static>(gate: Option<Arc<thoosa_shared::CudaActivityGate>>, future: impl Future<Output = Result<T, ThoError>> + Send + 'static) -> Result<T, ThoError> {
    let activity = gate.map(|gate| thoosa_shared::CudaActivityScope::for_operation(&gate)).transpose()?;
    tokio::spawn(async move {
        match activity {
            Some(activity) => activity.run(future).await,
            None => future.await,
        }
    })
    .await
    .context("checkpoint task failed")?
}

pub(crate) async fn copy_state_tensor(tensor: Tensor, device: usize) -> Result<Tensor, ThoError> {
    let source = crate::tho_device::ThoDevice::try_from(tensor.device())?.tho_device_idx();
    if source == device {
        // CPU training must not mutate the saved CPU allocation through an alias.
        return Ok(tch_result(tensor.f_to_device_(tensor.device(), tensor.kind(), false, true))?);
    }
    let actor = GlobalCtx::global().cuda_actor(source.max(device));
    Ok(actor.copy(ThoTensor::new((), tensor)?, device.into()).await?.into_local()?)
}

impl<St> ProxyStage<St>
where
    St: Stage + Send + Sync + 'static,
    for<'a> St::In<'a>: WorkerStageInput + Serialize,
    St::Out: WorkerStageOutput + DeserializeOwned + 'static,
{
    pub(crate) async fn save_cpu_checkpoint(self: Arc<Self>) -> Result<CpuStateCheckpoint, ThoError> {
        checkpoint_operation_on(Some(self.actor.activity_gate()), async move {
            let _execution = self.execution.write().await;
            let state = self.state.lock().await;
            // CPU-only saves still belong to this training session's worker generation.
            self.worker_id.check(self.actor.worker_id()?)?;
            if let Some(state) = state.as_ref() {
                // Snapshot CPU-offloaded state directly; saving must not allocate GPU memory.
                let scalars = serde_json::from_slice(&serde_json::to_vec(&state.scalars)?)?;
                let tensors = state
                    .tensors
                    .iter()
                    .map(|(key, tensor)| {
                        let source = tensor.local()?;
                        Ok((key.clone(), ThoTensor::new((), tch_result(source.f_to_device_(tch::Device::Cpu, source.kind(), false, true))?)?))
                    })
                    .collect::<Result<_, ThoError>>()?;
                return CpuStateCheckpoint::from_dump(StageStateDump::new(scalars, tensors));
            }
            // Training checkpoints belong to the owner-managed session, never a replacement endpoint.
            let actor = &self.actor;
            let mut dump = actor.purge_stage_state(self.stage_id.clone(), self.endpoint_name.to_string(), self.worker_id).await?;
            let export = dump.export.take().ok_or_else(|| ThoError::new("checkpoint export lacks acknowledgement"))?;
            let checkpoint = CpuStateCheckpoint::from_dump(dump)?;
            // commit=false releases export bookkeeping while preserving live GPU state.
            export.keep_live().await?;
            Ok(checkpoint)
        })
        .await
    }

    pub(crate) async fn schedule_cpu_restore(self: Arc<Self>, checkpoint: Arc<CpuStateCheckpoint>) -> Result<(), ThoError> {
        checkpoint_operation_on(Some(self.actor.activity_gate()), async move {
            let _execution = self.execution.write().await;
            let mut state = self.state.lock().await;
            let actor = &self.actor;
            let dump = checkpoint.restore_dump()?;
            // Discard the failed module and any newer offload; restore matching
            // metadata before uploading tensors from the persistent checkpoint.
            actor.reset_stage_state(self.worker_id, self.endpoint_name.to_string(), self.stage_id.clone(), checkpoint.info()?).await?;
            *state = Some(Box::new(dump));
            // Upload and worker pre_run reconstruction belong to the next OOM-managed invocation.
            Ok(())
        })
        .await
    }
}
