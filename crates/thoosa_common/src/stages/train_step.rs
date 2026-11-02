use thoosa_common::tch_error::tch_result;
use async_trait::async_trait;
use futures_util::{FutureExt, StreamExt};
use futures_util::future::BoxFuture;
use futures_util::stream::BoxStream;
use log::{debug, info, warn};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use tch::Tensor;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::ipc::proxy::{DynSyncStageWithStateEnum, CpuStateCheckpoint, checkpoint_operation, copy_state_tensor};
use thoosa_common::ipc::cuda_wokrer::{AllTensors, WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::worker_msg::{WorkerInput};
use thoosa_common::nets::batches::BatchLengths;
use thoosa_common::nets::nodes::LossFp;
use thoosa_common::nets::tho_varstore::ThoTrainModule;
use thoosa_common::runner::common::cachable::{BorrowRef, SelectRef, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContext, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::oom_handler::{Adjusters, NoInputAdjuster, NoStageAdjuster, OOMHandler, WithRunCtxContext, run_ctx_adjuster};
use thoosa_common::runner::purge::CachePurger;
use thoosa_common::runner::tmp_cache::{TmpCached, TmpCachedEntry};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::{ThoTensor, ThoTensorInner};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, MaxParallelCtx, RunCtx, Stage, ThoSelector, WithId};
use thoosa_common::runner::sync_stage::{ErasedStageWithState, StageWithState, SyncStage};
use thoosa_shared::{CopyToDevice, MoveToDevice, ThoError, mono_key, panic_to_err};
use tokio::sync::MutexGuard;
use tokio::task::coop::cooperative;
use thoosa_shared::ipc_types::WorkerOutput;

/// Training uses mean loss with equally weighted samples and one optimizer update.
/// Deterministic sample-independent models match an unsplit batch up to rounding.
/// Batch-dependent layers use microbatch statistics. Failed epochs are discarded
/// and restored from a complete CPU checkpoint; RNG streams are not rolled back.
/// Cross-sample objectives and external mutable state are unsupported.
/// No rollback tensors are allocated per batch. Callers await acknowledgement
/// before submitting another batch or capturing a checkpoint.
pub struct TrainStep<F, DI, DO, C>
where
    DI: Dims, DO: Dims, C: RunCtx
{
    id:SynKey,
    #[expect(dead_code, reason = "Retains stored metadata and its existing construction and diagnostic representation.")]
    device: ThoDevice,
    // Keep the module, optimizer, and VarStore under one state owner so none of
    // them crosses the local/IPC boundary independently.
    train_module: tokio::sync::Mutex<TrainStepState<DI, DO>>,
    x_f: Arc<F>,
    loss_fn: LossFp,
    best_checkpoint_path: PathBuf,
    // A retry after reply/stream completion failure must not apply a committed batch twice.
    last_training_batch: std::sync::Mutex<Option<(u64, f32)>>,
    pd: PhantomData<C>,
}

struct TrainStepState<DI: Dims, DO: Dims> {
    module: ThoTrainModule<DI, DO>,
    // Retain the CPU source until preparation succeeds, including partial-copy OOMs.
    pending_checkpoint: Option<Arc<CpuStateCheckpoint>>,
}

impl<DI: Dims, DO: Dims> TrainStepState<DI, DO> {
    async fn checkpoint(&mut self) -> Result<Arc<CpuStateCheckpoint>, ThoError> {
        if let Some(checkpoint) = &self.pending_checkpoint { return Ok(Arc::clone(checkpoint)); }
        let info = self.module.state_info()?;
        let keys = info.tensors.iter().map(|tensor| tensor.tensor_key.to_tensor_key()).collect::<Vec<_>>();
        let sources = keys.iter().map(|key| self.module.state_tensor(key)).collect::<Result<Vec<_>, _>>()?;
        // Inspect the actual source: an OOM purge may already have offloaded it.
        let source_device = sources.first().map(|tensor| ThoDevice::try_from(tensor.device())).transpose()?.map_or(0, |device| device.tho_device_idx());
        let copied = if source_device == 0 {
            // CPU snapshots also need independent storage; training mutates live parameters.
            sources.into_iter().map(|source| Ok(tch_result(source.f_to_device_(tch::Device::Cpu, source.kind(), false, true))?)).collect::<Result<Vec<_>, ThoError>>()?
        } else {
            GlobalCtx::global().cuda_actor(source_device).copy_batch_to_cpu(sources).await?
        };
        let tensors = keys.into_iter().zip(copied).map(|(key, tensor)| Ok((key, ThoTensor::new((), tensor)?))).collect::<Result<_, ThoError>>()?;
        Ok(Arc::new(CpuStateCheckpoint::new(info, tensors)?))
    }
}

impl <F, DI, DO, C> WithId for TrainStep<F, DI, DO, C>
where
    DI: Dims, DO: Dims, C: RunCtx
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

// Auto traits also check the captured input transform, rather than bypassing its bounds.
#[async_trait]
impl <F, DI, DO, C> CachePurger for TrainStep<F, DI, DO, C>
where
    DI:Dims,
    DO:Dims,
    C:RunCtx,
    Self:Send + Sync
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        self.train_module.lock().await.module.has_gpu_data(device_idx)
    }

    async fn purge_gpu(&self, _ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, _name: &str) -> Result<(), ThoError> {
        info!("TrainStep::purge_gpu");
        // This also handles partially restored state: its retained CPU checkpoint
        // is authoritative, so no live optimizer is needed to free GPU tensors.
        let mut guard = self.train_module.lock().await;
        if guard.module.has_gpu_data(device_idx) {
            guard.pending_checkpoint = Some(guard.checkpoint().await?);
            guard.module.drop_gpu(device_idx)?;
        }
        Ok(())
    }

    async fn drop_gpu(&self, device_idx: usize, _name: &str) -> Result<(), ThoError> {
        debug!("TrainStep::drop_gpu");
        self.train_module.lock().await.module.drop_gpu(device_idx)
    }

    async fn purge_cpu(&self, _purge_until: u64) -> usize {
        debug!("TrainStep::purge_cpu");
        0
    }
}

impl<F, DI: Dims, DO: Dims, C: RunCtx> Debug for TrainStep<F, DI, DO, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainStep").field("best_checkpoint_path", &self.best_checkpoint_path).finish()
    }
}

impl<F, DI, DO, C> TrainStep<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{

    pub async fn new_local<NP, TMB, TMBF, RC>(device: ThoDevice, train_module_builder: TMB, x_f: Arc<F>, loss_fn: LossFp, run_config: RC, best_checkpoint_path: PathBuf) -> Result<(Self, NP), ThoError>
    where
        TMB: FnOnce(RC, ThoDevice) -> TMBF,
        TMBF: Future<Output = Result<(ThoTrainModule<DI, DO>, NP), ThoError>>,
    {
        let (train_module, node_params) = train_module_builder(run_config, device).await?;
        debug!("TrainStep::new_local: best_checkpoint_path={best_checkpoint_path:?}");

        Ok((Self {
            id: Self::next_id(),
            device,
            train_module: tokio::sync::Mutex::new(TrainStepState { module: train_module, pending_checkpoint: None }),
            x_f,
            loss_fn,
            best_checkpoint_path,
            last_training_batch: std::sync::Mutex::new(None),
            pd: PhantomData,
        }, node_params))
    }

    pub fn next_id() ->SynKey {
        static STEP_ID: AtomicU32 = AtomicU32::new(0);
        let stage_id = STEP_ID.fetch_add(1, Ordering::AcqRel);
        mono_key!(stage_id, "TrainStep").into()
    }

}

impl<F, DI, DO, C> DynSyncStageWithStateEnum<TrainStep<F, DI, DO, C>>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    pub(crate) async fn run_once(self: &Arc<Self>, input: TmpCachedEntry<TrainStepInput<DI, DO>>, mut ctx: C, max_other_parallel: usize) -> Result<TrainStepOutput, ThoError>
    where C: MaxParallelCtx + Adjustable {
        let input = Arc::new(input);
        let mut required_inputs = HashMap::new();
        input.add_required_ids(&mut required_inputs);
        let required_ids = required_inputs.iter().map(|(key, ids)| (key.clone(), ids)).collect();
        let mut parallel = MaxParallelAdjustContextImpl::new();
        *parallel.max_other_parallel_mut() = max_other_parallel.try_into().map_err(|_| ThoError::new("training parallelism exceeds u8"))?;
        ctx.set_max_other_parallel(parallel.max_other_parallel());
        // Retry preparation with the same request and protected input. A failed
        // training transaction remains non-retryable and aborts the whole epoch.
        OOMHandler::<'_, '_, NoInputAdjuster, NoStageAdjuster, _, ()>::new(vec![
            Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::FAST))),
            Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("train_step"))),
        ], None).run(self, Arc::clone(&input), ctx, WithRunCtxContext::new(
            MaxParallelAndDumpToCpuAdjustableContex::new(parallel, DumpToCpuContextImpl::new(&required_ids)),
        )).await
    }

    pub(crate) async fn run_batches(
        self: &Arc<Self>,
        batches: BoxStream<'_, Result<(ThoBatch<DI>, Option<BatchLengths>, ThoBatch<DO>), ThoError>>,
        mut ctx: C,
        train: bool,
        max_other_parallel: usize,
        prefetch: bool,
        mut on_completed: impl FnMut(f32) + Send,
    ) -> Result<(), ThoError>
    where C: MaxParallelCtx + Adjustable {
        let cache = Arc::new(TmpCached::new("prepared_training_batches").await);
        // Cache synchronously when preparation finishes: a ready future must
        // never hold raw tensors while the other half of the join is pending.
        let mut batches = batches.map(|batch| batch.map(|(x, lengths, expected)| {
            cache.insert_guarded(TrainStepInput::new(x, lengths, expected, train))
        }));
        let mut current = cooperative(batches.next()).await;
        while let Some(input) = current {
            let step = self.run_once(input?, ctx.branch(false), max_other_parallel);
            // Exactly one preparation overlaps the current step. join! drains
            // both on failure before TrainNet can restore its epoch checkpoint;
            // cancellation drops both futures rather than detaching a producer.
            let (output, next) = if prefetch {
                tokio::join!(step, cooperative(batches.next()))
            } else {
                (step.await, None)
            };
            // A reset during preparation invalidates the worker generation even
            // if training reported an ordinary OOM first. Do not roll it back.
            if let Some(Err(error)) = &next {
                if error.needs_cuda_recovery() || error.is_cuda_recovery_failed() {
                    return Err(error.clone());
                }
            }
            on_completed(output?.loss);
            current = if prefetch { next } else { cooperative(batches.next()).await };
        }
        Ok(())
    }

    pub(crate) async fn save_epoch_checkpoint(self: &Arc<Self>, tho_device_idx: usize/*it's current device, not a destination*/) -> Result<Arc<CpuStateCheckpoint>, ThoError> {
        match self.as_ref() {
            Self::Proxy { stage, .. } => Ok(Arc::new(Arc::clone(stage).save_cpu_checkpoint().await?)),
            Self::Local(stage) => {
                let stage = Arc::clone(stage);
                checkpoint_operation(tho_device_idx, async move {
                    stage.train_module.lock().await.checkpoint().await
                }).await
            }
        }
    }

    pub(crate) async fn schedule_epoch_restore(self: &Arc<Self>, checkpoint: Arc<CpuStateCheckpoint>, device: usize) -> Result<(), ThoError> {
        match self.as_ref() {
            Self::Proxy { stage, .. } => Arc::clone(stage).schedule_cpu_restore(checkpoint).await,
            Self::Local(stage) => {
                let stage = Arc::clone(stage);
                checkpoint_operation(device, async move {
                    let mut state = stage.train_module.lock().await;
                    state.module.reset_state(checkpoint.info()?)?;
                    state.pending_checkpoint = Some(checkpoint);
                    *stage.last_training_batch.lock().or_else(|_| Err(ThoError::new("training acknowledgement state poisoned")))? = None;
                    // GPU copies and reconstruction run only inside the next OOM attempt.
                    Ok(())
                }).await
            }
        }
    }

    /// Acknowledgement means the state owner has published the checkpoint in either mode.
    pub async fn checkpoint(self: &Arc<Self>, ctx: C, max_other_parallel: usize) -> Result<(), ThoError>
    where C: MaxParallelCtx + Adjustable {
        let cache = Arc::new(TmpCached::new("training_checkpoint_command").await);
        self.run_once(cache.insert_guarded(TrainStepInput::checkpoint()?), ctx, max_other_parallel).await?;
        Ok(())
    }
}

impl<F, DI, DO, C> Stage for TrainStep<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    type Ctx = C;
    type In<'a> = TrainStepInput<DI, DO>;
    type Out = TrainStepOutput;
}

impl<F, DI, DO, C> SyncStage for TrainStep<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        // The common runner acquires and restores the whole mutable training state.
        Some(self)
    }
}

impl<F, DI, DO, C> StageWithState for TrainStep<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    type S = ThoTrainModule<DI, DO>;

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep::pre_run"))]
    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<Option<tokio::sync::MappedMutexGuard<'_, Self::S>>, ThoError>> {
        async move {
            let mut guard = self.train_module.lock().await;
            if let Some(checkpoint) = guard.pending_checkpoint.clone() {
                // Restart partial imports from immutable CPU storage on every attempt.
                guard.module.reset_state(checkpoint.info()?)?;
                for (key, tensor) in &checkpoint.tensors {
                    let source = tch_result(tensor.as_ref().f_alias())?;
                    let restored = copy_state_tensor(source, to_device_idx).await?;
                    guard.module.set_state_tensor(key, restored, to_device_idx)?;
                }
                guard.module.prepare_state(to_device_idx)?;
                guard.pending_checkpoint = None;
            } else {
                guard.module.prepare_state(to_device_idx)?;
            }
            guard.module.ensure_training_usable()?;
            guard.module.move_to_device(to_device_idx).await?;
            Ok(Some(MutexGuard::map(guard, |state| &mut state.module)))
        }.boxed()
    }

    fn get_state_info(&self, _device: usize) -> BoxFuture<'_, Result<thoosa_common::runner::sync_stage::StageStateInfo, ThoError>> {
        async move { self.train_module.lock().await.module.state_info() }.boxed()
    }

    fn reset_state(&self, info: thoosa_common::runner::sync_stage::StageStateInfo) -> BoxFuture<'_, Result<(), ThoError>> {
        async move {
            let mut state = self.train_module.lock().await;
            state.module.reset_state(info)?;
            state.pending_checkpoint = None;
            *self.last_training_batch.lock().or_else(|_| Err(ThoError::new("training acknowledgement state poisoned")))? = None;
            Ok(())
        }.boxed()
    }

    fn get_state_tensor(&self, _device: usize, tensor_id: String) -> BoxFuture<'_, Result<Tensor, ThoError>> {
        async move { self.train_module.lock().await.module.state_tensor(&tensor_id) }.boxed()
    }

    fn set_state_tensor(&self, device: usize, tensor_id: String, tensor: Tensor) -> BoxFuture<'_, Result<(), ThoError>> {
        async move { self.train_module.lock().await.module.set_state_tensor(&tensor_id, tensor, device) }.boxed()
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep::do_run"))]
    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, train_module: &mut Self::S, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        train_module.ensure_training_usable()?;
        if input.command == TrainStepCommand::Checkpoint {
            // Publish only a completed checkpoint; failed writes cannot replace the previous best.
            let pending = self.best_checkpoint_path.with_extension("pending.safetensors");
            let result = train_module.save_var_store(&pending).and_then(|()| Ok(std::fs::rename(&pending, &self.best_checkpoint_path)?));
            if result.is_err() { let _ = std::fs::remove_file(&pending); }
            result?;
            return Ok(TrainStepOutput::new(0.0));//TODO add enum in TrainStepOutput for each input command to avoid sending dummy 0.0
        }
        let train = input.train();
        let batch_len = *input.x_ref()?.size().first().ok_or_else(|| ThoError::new("training input has no batch dimension"))?;
        let chunk_len = input.microbatch_size.unwrap_or(batch_len);
        if batch_len <= 0 || chunk_len <= 0 { return Err(ThoError::new("training requires a nonempty batch and positive microbatch size")); }
        let calculate = |module: &mut ThoTrainModule<DI, DO>| -> Result<TrainStepOutput, ThoError> {
            let mut weighted_loss = 0.0;
            for start in (0..batch_len).step_by(chunk_len as usize) {
                let end = (start + chunk_len).min(batch_len);
                let chunk = input.select_batch((start..end).into())?;
                let x = (self.x_f)(chunk.x()?);
                let mut lengths = chunk.lengths.clone();
                let predicted = module.module().tho_forward(&x, lengths.as_mut(), train).map_err(|e|{
                    warn!("TrainStep::do_run_with_state: error={e:?}");
                    e
                })?;

                let loss = (self.loss_fn)(&predicted.as_ref(), &chunk.expected_y_ref()?.as_ref())?;
                let weight = (end - start) as f64 / batch_len as f64;
                weighted_loss += tch_result(loss.f_double_value(&[]))? * weight;
                if train {
                    // Loss is a mean over equally weighted samples. Backward accumulates;
                    // only training_transaction may update the optimizer, after every chunk.
                    tch_result(tch_result(loss.f_mul_scalar(weight))?.f_backward())?;
                }
            }
            Ok(TrainStepOutput::new(weighted_loss as f32))
        };
        if train {
            let mut committed = self.last_training_batch.lock().or_else(|_| Err(ThoError::new("training acknowledgement state poisoned")))?;
            if let Some((id, loss)) = *committed {
                if id == input.request_id { return Ok(TrainStepOutput::new(loss)); }
            }
            let output = train_module.training_transaction(calculate)?;//<-opt.step and zero_grad are here
            *committed = Some((input.request_id, output.loss));
            Ok(output)
        } else {
            panic_to_err!(tch::no_grad(|| calculate(train_module)))
        }
    }
}


impl<F, DI, DO, C> Adjustable for TrainStep<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
}

pub type TrainStepXFn<D> = Box<dyn Fn(ThoBatch<D>) -> ThoBatch<D> + Send + Sync + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
enum TrainStepCommand { Train, Validate, Checkpoint }

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct TrainStepInput<DI: Dims, DO: Dims> {
    id: SynKey,
    x: Option<ThoBatch<DI>>,
    pub lengths: Option<BatchLengths>,
    expected_y: Option<ThoBatch<DO>>,
    command: TrainStepCommand,
    request_id: u64,
    microbatch_size: Option<i64>,
}

impl<DI: Dims, DO: Dims> TrainStepInput<DI, DO> {
    pub fn new(x: ThoBatch<DI>, lengths: Option<BatchLengths>, expected_y: ThoBatch<DO>, train: bool) -> Self {
        let request_id: u64 = rand::random();
        Self {
            // Active and prefetched batches coexist in one temporary cache.
            id: SynKey::new([mono_key!(request_id.to_string(), "train_step")]),
            x: Some(x),
            lengths,
            expected_y: Some(expected_y),
            command: if train { TrainStepCommand::Train } else { TrainStepCommand::Validate },
            request_id,
            microbatch_size: None,
        }
    }

    fn checkpoint() -> Result<Self, ThoError> {
        // Control commands carry no tensors and never run a forward or backward pass.
        Ok(Self {
            id: SynKey::new([mono_key!("train_checkpoint", "stage")]),
            x: None,
            expected_y: None,
            lengths: None,
            command: TrainStepCommand::Checkpoint,
            request_id: rand::random(),
            microbatch_size: None,
        })
    }

    fn x(&self) -> Result<ThoBatch<DI>, ThoError> {
        // The alias stays within the synchronous forward/backward transaction.
        Ok(self.x_ref()?.shallow_clone::<Batch<DI>>())
    }

    pub fn x_ref(&self) -> Result<&ThoBatch<DI>, ThoError> {
        self.x.as_ref().ok_or_else(|| ThoError::new("training command has no input batch"))
    }

    pub fn lengths_ref(&self) -> Option<&BatchLengths> {
        self.lengths.as_ref()
    }

    pub fn expected_y_ref(&self) -> Result<&ThoBatch<DO>, ThoError> {
        self.expected_y.as_ref().ok_or_else(|| ThoError::new("training command has no target batch"))
    }

    pub fn train(&self) -> bool {
        self.command == TrainStepCommand::Train
    }

    pub fn clone_refs_for_worker(&self) -> Result<Self, ThoError> {
        if self.command == TrainStepCommand::Checkpoint { return Self::checkpoint(); }
        // Training requests keep only worker tensor references; the worker owns the materialized tensors.
        Ok(Self {
            id: self.id.clone(),
            x: Some(self.x_ref()?.clone_ref()?),
            lengths: self.lengths.clone(),
            expected_y: Some(self.expected_y_ref()?.clone_ref()?),
            command: self.command,
            request_id: self.request_id,
            microbatch_size: self.microbatch_size,
        })
    }

    fn select_batch(&self, selector: TrainStepInputSelector) -> Result<Self, ThoError> {
        let batch_range = selector.0;
        let lengths = self.lengths.as_ref().map(|lengths| {
            let range = checked_batch_range(lengths.as_slice().len() as i64, &batch_range)?;
            Ok::<BatchLengths, ThoError>(BatchLengths::new(lengths.as_slice()[range].to_vec()))
        }).transpose()?;

        Ok(Self {
            id: self.id.clone(),
            x: Some(select_batch_tensor(self.x_ref()?, &batch_range)?),
            lengths,
            expected_y: Some(select_batch_tensor(self.expected_y_ref()?, &batch_range)?),
            command: self.command,
            request_id: self.request_id,
            microbatch_size: None,
        })
    }
}

fn checked_batch_range(batch_len: i64, range: &Range<i64>) -> Result<Range<usize>, ThoError> {
    if range.start < 0 || range.end < range.start || range.end > batch_len {
        return Err(ThoError::new(format!("TrainStep batch selector range={range:?} is invalid for batch_len={batch_len}")));
    }
    Ok(range.start as usize..range.end as usize)
}

fn select_batch_tensor<D: Dims>(tensor: &ThoBatch<D>, batch_range: &Range<i64>) -> Result<ThoBatch<D>, ThoError> {
    let size = tensor.size();
    let batch_len = *size.first().ok_or_else(|| ThoError::new("TrainStep batch tensor has no batch dimension"))?;
    let range = checked_batch_range(batch_len, batch_range)?;
    let selected_len = (range.end - range.start) as i64;

    match &tensor.inner {
        ThoTensorInner::Val { t } => ThoTensor::new(tensor.id().clone(), tch_result(t.lock().f_narrow(0, range.start as i64, selected_len))?),
        ThoTensorInner::Ref { key, size, kind, selector } => {
            let mut ranges = match selector {
                Some(selector) => selector.tensor_ranges()?,
                None => vec![0..batch_len],
            };
            let current_batch_range = ranges.first_mut().ok_or_else(|| ThoError::new("TrainStep tensor selector has no batch range"))?;
            let current_start = current_batch_range.start;
            *current_batch_range = current_start + batch_range.start..current_start + batch_range.end;
            let mut selected_size = size.clone();
            selected_size[0] = selected_len;
            Ok(ThoTensor::new_ref(tensor.id().clone(), key.key, selected_size, *kind, Some(Box::new(TrainStepTensorSelector(ranges))), key.worker_id))
        }
    }
}

/// Selects the same batch range from every tensor carried by a training step.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrainStepInputSelector(Range<i64>);

impl From<Range<i64>> for TrainStepInputSelector {
    fn from(value: Range<i64>) -> Self {
        Self(value)
    }
}

impl Adjustable for TrainStepInputSelector {}

#[typetag::serde(name = "TrainStepInputSelector")]
impl ThoSelector for TrainStepInputSelector {
    fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(self.clone())
    }

    fn tensor_ranges(&self) -> Result<Vec<Range<i64>>, ThoError> {
        Ok(vec![self.0.clone()])
    }
}

/// Preserves any pre-existing tensor ranges while narrowing the batch dimension.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct TrainStepTensorSelector(Vec<Range<i64>>);

#[typetag::serde(name = "TrainStepTensorSelector")]
impl ThoSelector for TrainStepTensorSelector {
    fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(self.clone())
    }

    fn tensor_ranges(&self) -> Result<Vec<Range<i64>>, ThoError> {
        Ok(self.0.clone())
    }
}

impl<DI: Dims, DO: Dims> Input for TrainStepInput<DI, DO> {

}

// Batch preparation produces these cacheable inputs for the training consumer.
impl<DI: Dims, DO: Dims> Output for TrainStepInput<DI, DO> {}

impl<DI: Dims, DO: Dims> WithId for TrainStepInput<DI, DO> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<DI: Dims, DO: Dims> Debug for TrainStepInput<DI, DO> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainStepInput").finish()
    }
}

impl<DI: Dims, DO: Dims> Data for TrainStepInput<DI, DO> {
    type Metrics = <ThoBatch<DI> as Data>::Metrics;


    fn size(&self) -> Self::Metrics {
        self.x.as_ref().map_or_else(Default::default, |x| <ThoBatch<DI> as Data>::size(x))
    }

    fn tho_device_idx(&self) -> usize {
        self.x.as_ref().map_or(0, |x| x.tho_device_idx())
    }

    fn mem_size(&self) -> u64 {
        self.x.as_ref().map_or(0, |x| x.mem_size()) + self.expected_y.as_ref().map_or(0, |y| y.mem_size())
    }
}

#[async_trait]
impl<DI: Dims, DO: Dims> CopyToDevice for TrainStepInput<DI, DO> {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError>
    where
        Self: Sized,
    {
        if self.command == TrainStepCommand::Checkpoint { return Self::checkpoint(); }
        Ok(TrainStepInput {
            id: self.id.clone(),
            x: Some(track_await!(self.x_ref()?.copy_to_device(dst_device)).await?),
            lengths: self.lengths.clone(),
            expected_y: Some(track_await!(self.expected_y_ref()?.copy_to_device(dst_device)).await?),
            command: self.command,
            request_id: self.request_id,
            microbatch_size: self.microbatch_size,
        })
    }
}

impl<DI: Dims, DO: Dims> BorrowRef for TrainStepInput<DI, DO> {
    type Out<'a> = ThoDataRef<'a, Self> where Self: 'a;

    fn borrow_ref<'a>(&'a self, to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            if self.tho_device_idx() == to_tho_device_idx {
                Ok(ThoDataRef::Borrowed(self))
            } else {
                Ok(ThoDataRef::Owned(track_await!(self.copy_to_device(to_tho_device_idx)).await?))
            }
        }
    }

}

#[derive(Debug)]
pub struct TrainStepInputWrapper<DI: Dims, DO: Dims> {
    input: TrainStepInput<DI, DO>,
}

impl<DI: Dims, DO: Dims> TrainStepInputWrapper<DI, DO> {
    pub fn new(input: TrainStepInput<DI, DO>) -> Self {
        Self { input }
    }

    pub fn batch_len(&self) -> Result<i64, ThoError> {
        self.input.x_ref()?.size().first().copied().ok_or_else(|| ThoError::new("TrainStep input has no batch dimension"))
    }
}

impl<DI: Dims, DO: Dims> WithId for TrainStepInputWrapper<DI, DO> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        self.input.id()
    }
}

impl<DI: Dims, DO: Dims> BorrowRef for TrainStepInputWrapper<DI, DO> {
    type Out<'a> = ThoDataRef<'a, TrainStepInput<DI, DO>> where Self: 'a;

    fn borrow_ref<'a>(&'a self, to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        self.input.borrow_ref(to_tho_device_idx)
    }

}

impl<DI: Dims, DO: Dims> SelectRef<TrainStepInputSelector> for TrainStepInputWrapper<DI, DO> {
    fn select_ref<'a>(&'a self, to_tho_device_idx: usize, selector: Option<TrainStepInputSelector>) -> impl Future<Output = Result<<Self as BorrowRef>::Out<'a>, ThoError>> + Send
    where
        TrainStepInputSelector: 'a,
        Self: 'a,
    {
        async move {
            let Some(selector) = selector else {
                return BorrowRef::borrow_ref(self, to_tho_device_idx).await;
            };
            let selected = if self.input.train() {
                // The selector changes execution granularity, never the logical training batch.
                let len = self.batch_len()?;
                checked_batch_range(len, &selector.0)?;
                let mut full = self.input.select_batch((0..len).into())?;
                full.microbatch_size = Some(selector.0.end - selector.0.start);
                full
            } else {
                self.input.select_batch(selector)?
            };
            if selected.tho_device_idx() == to_tho_device_idx {
                Ok(ThoDataRef::Owned(selected))
            } else {
                Ok(ThoDataRef::Owned(track_await!(selected.copy_to_device(to_tho_device_idx)).await?))
            }
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TrainStepOutput {
    pub loss: f32,
}

impl TrainStepOutput {
    fn new(loss: f32) -> Self {
        Self { loss }
    }
}

impl WithId for TrainStepOutput {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Output for TrainStepOutput {}

#[typetag::serde]
impl WorkerOutput for TrainStepOutput {

}


impl<DI, DO> WorkerStageInput for TrainStepInput<DI, DO>
where
    DI: Dims,
    DO: Dims,
    Self: serde::de::DeserializeOwned,
{
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        // Generic training inputs need the registered adapter supplied by TrainNet.
        Err(ThoError::new("TrainStepInput requires its typed worker input adapter; use clone_refs_for_worker"))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        if self.command == TrainStepCommand::Checkpoint { return Self::checkpoint(); }
        // Materialize both batch references together immediately before the worker-side run.
        Ok(Self {
            id: self.id.clone(),
            x: Some(WorkerStageInput::ref_to_tensor(self.x_ref()?, all_cuda_tensors, remove)?),
            lengths: self.lengths.clone(),
            expected_y: Some(WorkerStageInput::ref_to_tensor(self.expected_y_ref()?, all_cuda_tensors, remove)?),
            command: self.command,
            request_id: self.request_id,
            microbatch_size: self.microbatch_size,
        })
    }

}

impl WorkerStageOutput for TrainStepOutput {
    fn encode_from_worker(self, _all_cuda_tensors: &AllTensors, _tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(self)
    }
}

#[cfg(test)]
#[path = "train_step_state_tests.rs"]
mod state_tests;
