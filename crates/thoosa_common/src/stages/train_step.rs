use async_trait::async_trait;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::{debug, info, trace};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tch::Tensor;
use thoosa_common::cuda_utils::maybe_cuda_mem;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::ipc::cuda_wokrer::{AllTensors, WorkerStageInput, WorkerStageOutput};
use thoosa_common::ipc::serialization::json::IpcCodec;
use thoosa_common::ipc::worker_msg::{WorkerInput};
use thoosa_common::nets::batches::BatchLengths;
use thoosa_common::nets::nodes::LossFp;
use thoosa_common::nets::tho_varstore::ThoTrainModule;
use thoosa_common::runner::common::cachable::{BorrowRef, SelectRef, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::{ThoTensor, ThoTensorInner};
use thoosa_common::utils::optimizer_mem_size;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, ErasedStageWithState, OutId, RunCtx, Stage, StageWithState, SyncStage, ThoSelector, WithId};
use thoosa_shared::{CopyToDevice, MoveToDevice, ThoError, mono_key, os_thread_id, panic_to_err};
use tokio::sync::MutexGuard;
use thoosa_shared::ipc_types::WorkerOutput;

pub struct TrainStep<F, DI: Dims, DO: Dims, C: RunCtx> {
    device: ThoDevice,
    // Keep the module, optimizer, and VarStore under one state owner so none of
    // them crosses the local/IPC boundary independently.
    train_module: tokio::sync::Mutex<ThoTrainModule<DI, DO>>,
    x_f: Arc<F>,
    loss_fn: LossFp,
    best_checkpoint_path: PathBuf,
    pd: PhantomData<C>,
}

unsafe impl <F, DI, DO, C> Send for TrainStep<F, DI, DO, C>
where
    DI:Dims,
    DO:Dims,
    C:RunCtx
{}

unsafe impl <F, DI, DO, C> Sync for TrainStep<F, DI, DO, C>
where
    DI:Dims,
    DO:Dims,
    C:RunCtx
{}

#[async_trait]
impl <F, DI, DO, C> FuturesRunnerMapPurger for TrainStep<F, DI, DO, C>
where
    DI:Dims,
    DO:Dims,
    C:RunCtx,
    Self:Send + Sync
{
    async fn has_gpu_data(&self, _device_idx: usize) -> bool {
        true
    }

    async fn purge_gpu(&self, _ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, _device_idx: usize, _name: &str) -> Result<(), ThoError> {
        info!("TrainStep::purge_gpu");
        // Moving the complete state together preserves optimizer snapshots when
        // the OOM handler evicts this stage from a GPU.
        let mut guard = self.train_module.lock().await;
        guard.opt().zero_grad();
        guard.move_to_device(0).await
    }

    async fn drop_gpu(&self, _device_idx: usize, _name: &str) -> Result<(), ThoError> {
        debug!("TrainStep::drop_gpu");
        todo!()
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
            device,
            train_module: tokio::sync::Mutex::new(train_module),
            x_f,
            loss_fn,
            best_checkpoint_path,
            pd: PhantomData,
        }, node_params))
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
    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<tokio::sync::MappedMutexGuard<'_, Self::S>, ThoError>> {
        async move {
            let mut guard = self.train_module.lock().await;
            guard.move_to_device(to_device_idx).await?;
            Ok(MutexGuard::map(guard, |v|v))
        }.boxed()
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep::do_run"))]
    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, mut train_module: tokio::sync::MappedMutexGuard<'_, Self::S>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("TrainStep::do_run: input={input:?}");
        // Save inside the state owner so IPC training never exports a VarStore.
        if input.save_var_store {
            debug!("TrainStep::do_run: saved to={:?}", self.best_checkpoint_path);
            train_module.save_var_store(&self.best_checkpoint_path)?;
        }

        let x_t_0 = input.x();
        let expected_y = input.expected_y();

        debug!("TrainStep::do_run: device={:?} x_t_0.size={:?} expected_y.size={:?}", self.device, x_t_0.size(), expected_y.size());
        debug!("TrainStep::do_run: 0. cuda_mem={:?}",maybe_cuda_mem(&self.device));

        let x_t = (self.x_f)(x_t_0);

        debug!("TrainStep::do_run: device={:?}  x_t.size={:?}", self.device, x_t.size());

        let mut lengths = input.lengths.clone();
        let mut calculate_loss = || {


            let predicted_y = train_module.module().tho_forward(&x_t, lengths.as_mut(), input.train)?;

            debug!("TrainStep::do_run: x_t={x_t:?} predicted_y={predicted_y:?} expected_y={expected_y:?} self.loss_fn={:?}", self.loss_fn);


            let loss = (self.loss_fn)(predicted_y.as_ref(), &expected_y.as_ref())?;

            trace!("TrainStep::do_run: x_t=\n{x_t} predicted_y=\n{predicted_y} expected_y=\n{expected_y} loss=\n{loss}");

            Ok::<_, ThoError>((predicted_y, loss))

        };
        let (predicted_y, loss) = if input.train {
            panic_to_err!(calculate_loss())?
        } else {
            panic_to_err!(tch::no_grad(calculate_loss))?
        };
        debug!("TrainStep::do_run: 1. cuda_mem={:?}",maybe_cuda_mem(&self.device));


        x_t.drop_inner();
        let loss_value = panic_to_err!(Ok::<_, ThoError>(loss.double_value(&[])))? as f32;

        if input.train {

            let opt = train_module.opt();
            debug!("TrainStep::do_run: opt={opt:?} optimizer_size={}", optimizer_mem_size(&opt));

            debug!("TrainStep::do_run: loss={loss} cuda_mem={:?}", maybe_cuda_mem(&self.device));
            trace!("TrainStep::do_run: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}",optimizer_mem_size(&opt));

            predicted_y.drop_inner();
            drop(expected_y);
            debug!("TrainStep::do_run: 2.1 cuda_mem={:?}",maybe_cuda_mem(&self.device));

            let t1 = Instant::now();

            panic_to_err!({
                opt.backward_step(&loss.set_requires_grad(true));
                opt.zero_grad();
                Ok::<_, ThoError>(())
            })?;


            debug!("TrainStep::do_run: 2.2 cuda_mem={:?}  optimizer_size={}", maybe_cuda_mem(&self.device),optimizer_mem_size(&opt));
            debug!("TrainStep::do_run: device={:?}  backward_step done in {}", self.device, t1.elapsed().as_millis());
        } else {
            predicted_y.drop_inner();
            drop(expected_y);
        }

        Ok(TrainStepOutput::new(loss_value))
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

#[derive(Serialize, Deserialize)]
#[serde(bound(serialize = "", deserialize = ""))]
pub struct TrainStepInput<DI: Dims, DO: Dims> {
    id: SynKey,
    pub x: ThoBatch<DI>,
    pub lengths: Option<BatchLengths>,
    pub expected_y: ThoBatch<DO>,
    train: bool,
    save_var_store: bool,
}

impl<DI: Dims, DO: Dims> TrainStepInput<DI, DO> {
    pub fn new(x: ThoBatch<DI>, lengths: Option<BatchLengths>, expected_y: ThoBatch<DO>, train: bool, save_var_store: bool) -> Self {
        Self {
            id: SynKey::new([mono_key!("train_step", "stage")]),
            x,
            lengths,
            expected_y,
            train,
            save_var_store,
        }
    }

    pub fn x(&self) -> ThoBatch<DI> {
        self.x.shallow_clone::<Batch<DI>>()//called form sync TrainStep::do_run
    }

    pub fn expected_y(&self) -> ThoBatch<DO> {
        self.expected_y.shallow_clone::<Batch<DO>>()
    }

    pub fn x_ref(&self) -> &ThoBatch<DI> {
        &self.x
    }

    pub fn lengths_ref(&self) -> Option<&BatchLengths> {
        self.lengths.as_ref()
    }

    pub fn expected_y_ref(&self) -> &ThoBatch<DO> {
        &self.expected_y
    }

    pub fn train(&self) -> bool {
        self.train
    }

    pub fn clone_refs_for_worker(&self) -> Result<Self, ThoError> {
        // Training requests keep only worker tensor references; the worker owns the materialized tensors.
        Ok(Self {
            id: self.id.clone(),
            x: self.x.clone_ref()?,
            lengths: self.lengths.clone(),
            expected_y: self.expected_y.clone_ref()?,
            train: self.train,
            save_var_store: self.save_var_store,
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
            x: select_batch_tensor(&self.x, &batch_range)?,
            lengths,
            expected_y: select_batch_tensor(&self.expected_y, &batch_range)?,
            train: self.train,
            // Saving on every retry chunk would checkpoint parameters after earlier chunks updated them.
            save_var_store: self.save_var_store && batch_range.start == 0,
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
        ThoTensorInner::Val { t } => ThoTensor::new(tensor.id().clone(), t.f_narrow(0, range.start as i64, selected_len)?),
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
        <ThoBatch<DI> as Data>::size(&self.x)
    }

    fn tho_device_idx(&self) -> usize {
        self.x.tho_device_idx()
    }

    fn mem_size(&self) -> u64 {
        self.x.mem_size() + self.expected_y.mem_size()
    }
}

#[async_trait]
impl<DI: Dims, DO: Dims> CopyToDevice for TrainStepInput<DI, DO> {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError>
    where
        Self: Sized,
    {
        Ok(TrainStepInput {
            id: self.id.clone(),
            x: track_await(self.x.copy_to_device(dst_device)).await?,
            lengths: self.lengths.clone(),
            expected_y: track_await(self.expected_y.copy_to_device(dst_device)).await?,
            train: self.train,
            save_var_store: self.save_var_store,
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
                Ok(ThoDataRef::Owned(track_await(self.copy_to_device(to_tho_device_idx)).await?))
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
        self.input.x.size().first().copied().ok_or_else(|| ThoError::new("TrainStep input has no batch dimension"))
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
            let selected = self.input.select_batch(selector)?;
            if selected.tho_device_idx() == to_tho_device_idx {
                Ok(ThoDataRef::Owned(selected))
            } else {
                Ok(ThoDataRef::Owned(track_await(selected.copy_to_device(to_tho_device_idx)).await?))
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
        todo!()
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        // Materialize both batch references together immediately before the worker-side run.
        Ok(Self {
            id: self.id.clone(),
            x: WorkerStageInput::ref_to_tensor(&self.x, all_cuda_tensors, remove)?,
            lengths: self.lengths.clone(),
            expected_y: WorkerStageInput::ref_to_tensor(&self.expected_y, all_cuda_tensors, remove)?,
            train: self.train,
            save_var_store: self.save_var_store,
        })
    }

}

impl WorkerStageOutput for TrainStepOutput {
    fn encode_from_worker(self, _all_cuda_tensors: &AllTensors, _tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(self)
    }
}
