use crate::tasks::train_net::maybe_cuda_mem;
use async_trait::async_trait;
use futures_util::FutureExt;
use futures_util::future::BoxFuture;
use log::{debug, trace};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tch::Tensor;
use thoosa_common::batches::BatchLengths;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::ipc::codec::IpcCodec;
use thoosa_common::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use thoosa_common::my_tch::LossFp;
use thoosa_common::runner::common::cachable::{BorrowRef, ThoDataRef};
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::tho_varstore::ThoTrainModule;
use thoosa_common::utils::optimizer_mem_size;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, ErasedStageWithState, OutId, RunCtx, Stage, StageWithState, SyncStage, WithId};
use thoosa_shared::{CopyToDevice, MoveToDevice, ThoError, mono_key, os_thread_id, panic_to_err};
use tokio::sync::MutexGuard;
use thoosa_common::ipc::worker_msg::WorkerOutput;

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
        debug!("TrainStep::purge_gpu");
        // Moving the complete state together preserves optimizer snapshots when
        // the OOM handler evicts this stage from a GPU.
        self.train_module.lock().await.move_to_device(0).await
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

    pub(crate) async fn new_local<NP, TMB, TMBF, RC>(device: ThoDevice, train_module_builder: TMB, x_f: Arc<F>, loss_fn: LossFp, run_config: RC, best_checkpoint_path: PathBuf) -> Result<(Self, NP), ThoError>
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

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
        async move {
            let mut guard = self.train_module.lock().await;
            guard.move_to_device(to_device_idx).await?;
            Ok(guard)
        }.boxed()
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep::do_run"))]
    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, mut train_module: MutexGuard<'_, Self::S>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
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

            debug!("TrainStep::do_run: predicted_y={predicted_y:?} expected_y={expected_y:?} self.loss_fn={:?}", self.loss_fn);

            let loss = (self.loss_fn)(predicted_y.as_ref(), &expected_y.as_ref())?;
            Ok::<_, ThoError>((predicted_y, loss))

        };
        let (predicted_y, loss) = if input.train { calculate_loss()? } else { tch::no_grad(calculate_loss)? };
        debug!("TrainStep::do_run: 1. cuda_mem={:?}",maybe_cuda_mem(&self.device));
        trace!("TrainStep::do_run: x_t=\n{x_t}");

        x_t.drop();
        let loss_value = panic_to_err!(Ok::<_, ThoError>(loss.double_value(&[])))? as f32;

        if input.train {

            let opt = train_module.opt();
            debug!("TrainStep::do_run: opt={opt:?} optimizer_size={}", optimizer_mem_size(&opt));

            debug!("TrainStep::do_run: loss={loss} cuda_mem={:?}", maybe_cuda_mem(&self.device));
            trace!("TrainStep::do_run: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}",optimizer_mem_size(&opt));

            predicted_y.drop();
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
            predicted_y.drop();
            drop(expected_y);
        }

        Ok(TrainStepOutput::new(loss_value))
    }
}

impl<F, DI, DO, C> OutId for TrainStep<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    fn out_id<'a>(&self, _input_id: &SynKey) -> () {
        ()
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

pub(crate) type TrainStepXFn<D> = Box<dyn Fn(ThoBatch<D>) -> ThoBatch<D> + Send + Sync + 'static>;

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
    pub(crate) fn new(x: ThoBatch<DI>, lengths: Option<BatchLengths>, expected_y: ThoBatch<DO>, train: bool, save_var_store: bool) -> Self {
        Self {
            id: SynKey::new([mono_key!("train_step", "stage")]),
            x,
            lengths,
            expected_y,
            train,
            save_var_store,
        }
    }

    pub(crate) fn x(&self) -> ThoBatch<DI> {
        self.x.shallow_clone::<Batch<DI>>()//called form sync TrainStep::do_run
    }

    pub(crate) fn expected_y(&self) -> ThoBatch<DO> {
        self.expected_y.shallow_clone::<Batch<DO>>()
    }

    pub(crate) fn x_ref(&self) -> &ThoBatch<DI> {
        &self.x
    }

    pub(crate) fn lengths_ref(&self) -> Option<&BatchLengths> {
        self.lengths.as_ref()
    }

    pub(crate) fn expected_y_ref(&self) -> &ThoBatch<DO> {
        &self.expected_y
    }

    pub(crate) fn train(&self) -> bool {
        self.train
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
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        <ThoBatch<DI> as Data>::size(&self.x)
    }

    fn device(&self) -> usize {
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

    fn borrow_to<'a>(&'a self, to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            if self.device() == to_tho_device_idx {
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
}

impl<DI: Dims, DO: Dims> WithId for TrainStepInputWrapper<DI, DO> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        self.input.id()
    }
}

impl<DI: Dims, DO: Dims> BorrowRef for TrainStepInputWrapper<DI, DO> {
    type Out<'a> = ThoDataRef<'a, TrainStepInput<DI, DO>> where Self: 'a;

    fn borrow_to<'a>(&'a self, to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        self.input.borrow_to(to_tho_device_idx)
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
    fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<parking_lot::Mutex<std::collections::HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
        let input = C::decode::<Self>(input)?;
        // Shared wrappers decode worker references before the stateful stage runs.
        Ok(Self::new(
            ThoTensor::ref_to_tensor(all_cuda_tensors, input.x)?,
            input.lengths,
            ThoTensor::ref_to_tensor(all_cuda_tensors, input.expected_y)?,
            input.train,
            input.save_var_store,
        ))
    }
}

impl WorkerStageOutput for TrainStepOutput {
    fn encode_from_worker(self, _all_cuda_tensors: &Arc<parking_lot::Mutex<std::collections::HashMap<u64, Tensor>>>, _tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(self)
    }
}
