use crate::tasks::train_net::maybe_cuda_mem;
use async_trait::async_trait;
use log::{debug, trace};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::io::Write;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::time::Instant;
use tch::Device;
use tch::nn::VarStore;
use tokio::sync::{MutexGuard, RwLockReadGuard};
use thoosa_common::batches::BatchLengths;
use thoosa_common::copy_ctx::CudaActorHandle;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::my_tch::{LossFp, ThoModule};
use thoosa_common::runner::common::cachable::{BorrowRef, ThoDataRef};
use thoosa_common::runner::common::runnable::{DynSyncStageBox, ErasedStageWithState, Stage, StageWithState, SyncStage};
use thoosa_common::runner::common::{MaxParallelCtx, OutId, RunCtx};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::tho_varstore::ThoTrainModule;
use thoosa_common::utils::optimizer_mem_size;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, WithId};
use thoosa_shared::{mono_key, os_thread_id, panic_to_err, perf_log, sync_mutex_lock, CopyToDevice, MoveToDevice, ThoError};
use futures::FutureExt;
use futures_util::future::BoxFuture;
use futures_util::StreamExt;

// The proxy purge path and the test worker registration must use the same
// endpoint name in both normal and cfg(test) builds.
pub(crate) const TRAIN_TEST_STEP2_ENDPOINT: &str = "TrainTestStep2";

pub struct TrainStep2<F, DI: Dims, DO: Dims, C: RunCtx> {
    device: ThoDevice,
    // Keep the module, optimizer, and their VarStore under one owner so their
    // lifetimes cannot diverge while training state is copied or exported.
    train_module: tokio::sync::Mutex<ThoTrainModule<DI, DO>>,

    x_f: Arc<F>,
    loss_fn: LossFp,
    pd: PhantomData<C>,
}

unsafe impl <F, DI, DO, C> Send for TrainStep2<F, DI, DO, C>
where
    DI:Dims,
    DO:Dims,
    C:RunCtx
{}

unsafe impl <F, DI, DO, C> Sync for TrainStep2<F, DI, DO, C>
where
    DI:Dims,
    DO:Dims,
    C:RunCtx
{}

#[async_trait]
impl <F, DI, DO, C> FuturesRunnerMapPurger for TrainStep2<F, DI, DO, C>
where
    DI:Dims,
    DO:Dims,
    C:RunCtx,
    Self:Send + Sync
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        true
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        debug!("TrainStep2::purge_gpu");
        let guard = self.train_module.lock().await.move_to_device(0).await?;

        Ok(())
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        debug!("TrainStep2::drop_gpu");
        todo!()
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        debug!("TrainStep2::purge_cpu");
        0
    }
}

impl<F, DI: Dims, DO: Dims, C: RunCtx> Debug for TrainStep2<F, DI, DO, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainStep2").finish()
    }
}

impl<F, DI, DO, C> TrainStep2<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep2::new"))]
    pub(crate) async fn new<NP, TMB, TMBF, RC>(tho_device: ThoDevice, train_module_builder: TMB, x_f: Arc<F>, loss_fn: LossFp, proxy_factory: Option<&Arc<dyn TrainStep2ProxyFactory<DI, DO, C, NP, RC>>>, run_config: &RC) -> Result<(TrainStep2DynSyncStage<F, DI, DO, C>, NP), ThoError>
    where
        DI: 'static,
        DO: 'static,
        C: 'static,
        NP: Debug + Send + Sync + 'static,
        RC: Debug + Send + Sync + 'static,
        TMB: FnOnce(&RC, ThoDevice) -> TMBF,
        TMBF: Future<Output = Result<(ThoTrainModule<DI, DO>, NP), ThoError>>,
    {
        if GlobalCtx::global().is_ipc() && let Some(proxy_factory) = proxy_factory
        {
            let TrainStep2ProxyParts { stage, weights_sync, node_params } = proxy_factory.new_proxy(run_config, tho_device.tho_device_idx()).await?;
            return Ok((TrainStep2DynSyncStage::Proxy { stage }, node_params));
        }

        let (stage, node_params) = Self::new_local(tho_device, train_module_builder, x_f, loss_fn, run_config).await?;
        Ok((TrainStep2DynSyncStage::Local(Arc::new(stage)), node_params))
    }

    pub(crate) async fn new_local<NP, TMB, TMBF, RC>(device: ThoDevice, train_module_builder: TMB, x_f: Arc<F>, loss_fn: LossFp, run_config: RC) -> Result<(Self, NP), ThoError>
    where
        TMB: FnOnce(RC, ThoDevice) -> TMBF,
        TMBF: Future<Output = Result<(ThoTrainModule<DI, DO>, NP), ThoError>>,
    {
        let (train_module, node_params) = train_module_builder(run_config, device).await?;
        Ok((Self {
            device,
            train_module: tokio::sync::Mutex::new(train_module),
            x_f,
            loss_fn,
            pd: PhantomData,
        }, node_params))
    }

}

impl<F, DI, DO, C> Stage for TrainStep2<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    type Ctx = C;
    type In<'a> = TrainStep2Input<DI, DO>;
    type Out = TrainStep2Output;
}

impl<F, DI, DO, C> SyncStage for TrainStep2<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        // Advertise typed state execution without exposing ThoTrainModule to the common runner.
        Some(self)
    }
}

impl<F, DI, DO, C> StageWithState for TrainStep2<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    type S = ThoTrainModule<DI, DO>;

    fn pre_run(&self) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
        async {
            let mut guard = self.train_module.lock().await;
            guard.move_to_device(1).await?;
            Ok(guard)
        }.boxed()
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep2::do_run_with_state"))]
    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, mut train_module: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {

        debug!("TrainStep2::do_run: os_thread_id={}", os_thread_id());
        let x_t_0 = input.x();
        let expected_y = input.expected_y();

        debug!("TrainStep2::do_run: device={:?} x_t_0.size={:?} expected_y.size={:?}", self.device, x_t_0.size(), expected_y.size());
        debug!("TrainStep2::do_run: 0. cuda_mem={:?}",maybe_cuda_mem(&self.device));

        let x_t = (self.x_f)(x_t_0);

        debug!("TrainStep2::do_run: device={:?}  x_t.size={:?}", self.device, x_t.size());

        let mut lengths = input.lengths.clone();
        let mut calculate_loss = || {
            let predicted_y = train_module.module().tho_forward(&x_t, lengths.as_mut(), input.train)?;

            debug!("TrainStep2::do_run: predicted_y={predicted_y:?} expected_y={expected_y:?} self.loss_fn={:?}", self.loss_fn);

            let loss = (self.loss_fn)(predicted_y.as_ref(), &expected_y.as_ref())?;
            Ok::<_, ThoError>((predicted_y, loss))

        };
        let (predicted_y, loss) = if input.train { calculate_loss()? } else { tch::no_grad(calculate_loss)? };
        debug!("TrainStep2::do_run: 1. cuda_mem={:?}",maybe_cuda_mem(&self.device));
        trace!("TrainStep2::do_run: x_t=\n{x_t}");

        x_t.drop();
        let loss_value = panic_to_err!(Ok::<_, ThoError>(loss.double_value(&[])))? as f32;

        if input.train {
            let opt = train_module.opt();
            debug!("TrainStep2::do_run: opt={opt:?} optimizer_size={}", optimizer_mem_size(&opt));

            debug!("TrainStep2::do_run: loss={loss} cuda_mem={:?}", maybe_cuda_mem(&self.device));
            trace!("TrainStep2::do_run: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}",optimizer_mem_size(&opt));

            predicted_y.drop();
            drop(expected_y);
            debug!("TrainStep2::do_run: 2.1 cuda_mem={:?}",maybe_cuda_mem(&self.device));

            let t1 = Instant::now();

            panic_to_err!({
                opt.backward_step(&loss.set_requires_grad(true));
                opt.zero_grad();//to let
                Ok::<_, ThoError>(())
            })?;


            debug!("TrainStep2::do_run: 2.2 cuda_mem={:?}  optimizer_size={}", maybe_cuda_mem(&self.device),optimizer_mem_size(&opt));
            debug!("TrainStep2::do_run: device={:?}  backward_step done in {}", self.device, t1.elapsed().as_millis());
        } else {
            predicted_y.drop();
            drop(expected_y);
        }

        Ok(TrainStep2Output::new(loss_value))
    }
}

impl<F, DI, DO, C> OutId for TrainStep2<F, DI, DO, C>
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

impl<F, DI, DO, C> Adjustable for TrainStep2<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
}

pub(crate) type TrainStep2XFn<D> = Box<dyn Fn(ThoBatch<D>) -> ThoBatch<D> + Send + Sync + 'static>;

#[async_trait]
pub(crate) trait TrainStep2WeightsSync: Debug + Send + Sync {
    async fn into_var_store(&self) -> Result<VarStore, ThoError>;
}

#[async_trait]
pub(crate) trait TrainStep2ProxyFactory<DX, DY, C, NP, RC>: Debug + Send + Sync
where
    DX: Dims + 'static,
    DY: Dims + 'static,
    C: RunCtx + 'static,
    NP: Debug + Send + Sync + 'static,
    RC: Debug + Send + Sync + 'static,
{
    async fn new_proxy(&self, run_config: &RC, tho_device_idx: usize) -> Result<TrainStep2ProxyParts<DX, DY, C, NP>, ThoError>;
}

#[derive(Serialize, Deserialize)]
pub struct TrainStep2Input<DI: Dims, DO: Dims> {
    id: SynKey,
    pub x: ThoBatch<DI>,
    pub lengths: Option<BatchLengths>,
    pub expected_y: ThoBatch<DO>,
    train: bool,
}

impl<DI: Dims, DO: Dims> TrainStep2Input<DI, DO> {
    pub(crate) fn new(x: ThoBatch<DI>, lengths: Option<BatchLengths>, expected_y: ThoBatch<DO>, train: bool) -> Self {
        Self {
            id: SynKey::new([mono_key!("train_step", "stage")]),
            x,
            lengths,
            expected_y,
            train,
        }
    }

    pub(crate) fn x(&self) -> ThoBatch<DI> {
        self.x.shallow_clone::<Batch<DI>>()//called form sync TrainStep2::do_run
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

impl<DI: Dims, DO: Dims> Input for TrainStep2Input<DI, DO> {

}

impl<DI: Dims, DO: Dims> WithId for TrainStep2Input<DI, DO> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<DI: Dims, DO: Dims> Debug for TrainStep2Input<DI, DO> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainStep2Input")
            .field("x", &self.x)
            .field("expected_y", &self.expected_y)
            .finish()
    }
}

impl<DI: Dims, DO: Dims> Data for TrainStep2Input<DI, DO> {
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
impl<DI: Dims, DO: Dims> CopyToDevice for TrainStep2Input<DI, DO> {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError>
    where
        Self: Sized,
    {
        Ok(TrainStep2Input {
            id: self.id.clone(),
            x: track_await(self.x.copy_to_device(dst_device)).await?,
            lengths: self.lengths.clone(),
            expected_y: track_await(self.expected_y.copy_to_device(dst_device)).await?,
            train: self.train,
        })
    }
}

impl<DI: Dims, DO: Dims> BorrowRef for TrainStep2Input<DI, DO> {
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
pub struct TrainStep2InputWrapper<DI: Dims, DO: Dims> {
    input: TrainStep2Input<DI, DO>,
}

impl<DI: Dims, DO: Dims> TrainStep2InputWrapper<DI, DO> {
    pub fn new(input: TrainStep2Input<DI, DO>) -> Self {
        Self { input }
    }
}

impl<DI: Dims, DO: Dims> WithId for TrainStep2InputWrapper<DI, DO> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        self.input.id()
    }
}

impl<DI: Dims, DO: Dims> BorrowRef for TrainStep2InputWrapper<DI, DO> {
    type Out<'a> = ThoDataRef<'a, TrainStep2Input<DI, DO>> where Self: 'a;

    fn borrow_to<'a>(&'a self, to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        self.input.borrow_to(to_tho_device_idx)
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct TrainStep2Output {
    pub loss: f32,
}

impl TrainStep2Output {
    fn new(loss: f32) -> Self {
        Self { loss}
    }
}

impl WithId for TrainStep2Output {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Output for TrainStep2Output {}

pub(crate) enum TrainStep2DynSyncStage<F, DI: Dims + 'static, DO: Dims + 'static, C: RunCtx + 'static> {
    Local(Arc<TrainStep2<F, DI, DO, C>>),
    Proxy {
        stage: Arc<DynSyncStageBox<TrainStep2Input<DI, DO>, TrainStep2Output, C>>,
    },
}

#[async_trait]
impl <F, DI, DO, C> FuturesRunnerMapPurger for TrainStep2DynSyncStage<F, DI, DO, C>
where
    DI:Dims,
    DO:Dims,
    C:RunCtx,
    Self:Send + Sync
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        match self {
            TrainStep2DynSyncStage::Local(local) => {
                local.has_gpu_data(device_idx).await
            }
            TrainStep2DynSyncStage::Proxy { .. } => {
                true
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            TrainStep2DynSyncStage::Local(local) => {
                local.purge_gpu(ids_to_keep, device_idx, name).await
            }
            TrainStep2DynSyncStage::Proxy { stage,  .. } => {
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(async move {
                        thoosa_common::ipc::codec::purge_ipc_sync_stage::<thoosa_common::ipc::codec::JsonIpcCodec>(
                            GlobalCtx::global().cuda_actor(device_idx).as_ref(),
                            TRAIN_TEST_STEP2_ENDPOINT,
                        ).await
                    })
                })?;
                Ok(())
            }
        }
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            TrainStep2DynSyncStage::Local(local) => {
                local.drop_gpu(device_idx, name).await
            }
            TrainStep2DynSyncStage::Proxy { .. } => {
                Ok(())
            }
        }

    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        match self {
            TrainStep2DynSyncStage::Local(local) => {
                local.purge_cpu(purge_until).await
            }
            TrainStep2DynSyncStage::Proxy { .. } => {
                0
            }
        }

    }
}

impl<F, DI, DO, C> Debug for TrainStep2DynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainStep2DynSyncStage").finish()
    }
}

impl<F, DI, DO, C> Stage for TrainStep2DynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    type Ctx = C;
    type In<'a> = TrainStep2Input<DI, DO>;
    type Out = TrainStep2Output;
}

impl<F, DI, DO, C> SyncStage for TrainStep2DynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
        // Proxy execution is stateless locally; only the local variant owns a training-module guard.
        match self {
            Self::Local(_) => Some(self),
            Self::Proxy { .. } => None,
        }
    }

    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(input, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, ctx),
        }
    }

}

impl<F, DI, DO, C> StageWithState for TrainStep2DynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    type S = <TrainStep2<F, DI, DO, C> as StageWithState>::S;

    fn pre_run(&self) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
        match self {
            Self::Local(stage) => StageWithState::pre_run(stage.as_ref()),
            Self::Proxy { .. } => async {
                Err(ThoError::new("proxy TrainStep2 does not have local state"))
            }.boxed(),
        }
    }

    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, state: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => StageWithState::do_run_with_state(Arc::clone(stage), input, state, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, ctx),
        }
    }
}

impl<F, DI, DO, C> OutId for TrainStep2DynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    fn out_id<'a>(&self, _input_id: &SynKey) -> () {
        ()
    }
}

impl<F, DI, DO, C> Adjustable for TrainStep2DynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{}

// Train-step proxies pass worker tensor references instead of serializing batch tensors through JSON.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TrainStep2TensorRef {
    pub(crate) ref_key: u64,
    size: Vec<i64>,
    kind: String,
    tho_device_idx: usize,
    pub(crate) remove_after_run: bool,
}

impl TrainStep2TensorRef {
    fn from_ref<D, Id>(tensor: &ThoTensor<D, Id>) -> Result<Self, ThoError>
    where
        D: Dims,
        Id: thoosa_shared::traits::Named + Debug + Clone,
    {
        Ok(Self {
            ref_key: tensor.ref_key().ok_or_else(|| ThoError::new(format!("TrainStep2 expected worker tensor ref, got {tensor:?}")))?,
            size: tensor.size(),
            kind: thoosa_common::ipc::json_ser::tho_tensor::kind_to_str(&tensor.kind()).map_err(ThoError::new)?.to_string(),
            tho_device_idx: tensor.tho_device().tho_device_idx(),
            remove_after_run: false,
        })
    }

    async fn from_tensor<D>(tensor: &ThoBatch<D>, tho_device_idx: usize) -> Result<Self, ThoError>
    where
        D: Dims,
    {
        if tensor.ref_key().is_some() {
            return Self::from_ref(tensor);
        }

        // CPU-built label/final batches must be copied into the worker before invoking the train step.
        let cpu_tensor = tensor.t_ref().f_to_device(Device::Cpu)?;
        let worker_ref = track_await(GlobalCtx::global().cuda_actor(tho_device_idx).copy(cpu_tensor.into(), ThoDevice::from(tho_device_idx))).await?;
        let mut tensor_ref = Self::from_ref(&worker_ref)?;
        tensor_ref.remove_after_run = true;
        Ok(tensor_ref)
    }

    pub(crate) fn into_tensor<D>(self) -> Result<ThoBatch<D>, ThoError>
    where
        D: Dims,
    {
        let kind = thoosa_common::ipc::json_ser::tho_tensor::str_to_kind(&self.kind).ok_or_else(|| ThoError::new(format!("TrainStep2 received unsupported tensor kind {:?}", self.kind)))?;
        Ok(ThoTensor::new_ref(thoosa_shared::simple_id!("train_step_ipc"), self.ref_key, self.size, kind, None, self.tho_device_idx))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum TrainStep2IpcRequest {
    Run {
        x: TrainStep2TensorRef,
        lengths: Option<Vec<i64>>,
        expected_y: TrainStep2TensorRef,
        train: bool,
        max_other_parallel: u8,
    },
    ExportWeights,
}

impl TrainStep2IpcRequest {
    pub(crate) async fn run<DI, DO, C>(input: &TrainStep2Input<DI, DO>, ctx: &C) -> Result<Self, ThoError>
    where
        DI: Dims,
        DO: Dims,
        C: RunCtx,
    {
        Ok(Self::Run {
            x: track_await(TrainStep2TensorRef::from_tensor(input.x_ref(), ctx.tho_device_idx())).await?,
            lengths: input.lengths_ref().map(|lengths| lengths.as_slice().to_vec()),
            expected_y: track_await(TrainStep2TensorRef::from_tensor(input.expected_y_ref(), ctx.tho_device_idx())).await?,
            train: input.train(),
            max_other_parallel: ctx.max_other_parallel(),
        })
    }
}

pub(crate) struct TrainStep2ProxyParts<DX, DY, C, NP>
where
    DX: Dims + 'static,
    DY: Dims + 'static,
    C: RunCtx + 'static,
{
    stage: Arc<DynSyncStageBox<TrainStep2Input<DX, DY>, TrainStep2Output, C>>,
    weights_sync: Arc<dyn TrainStep2WeightsSync>,
    node_params: NP,
}

impl<DX, DY, C, NP> TrainStep2ProxyParts<DX, DY, C, NP>
where
    DX: Dims + 'static,
    DY: Dims + 'static,
    C: RunCtx + 'static,
{
    pub(crate) fn new(stage: Arc<DynSyncStageBox<TrainStep2Input<DX, DY>, TrainStep2Output, C>>, weights_sync: Arc<dyn TrainStep2WeightsSync>, node_params: NP) -> Self {
        Self { stage, weights_sync, node_params }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::fmt::{Debug, Formatter};
    use std::future::ready;
    use std::marker::PhantomData;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize};
    use std::sync::atomic::Ordering::Relaxed;
    use async_trait::async_trait;
    use futures_util::future::BoxFuture;
    use futures_util::{stream, StreamExt};
    use log::debug;
    use mockall::PredicateBoxExt;
    use parking_lot::Mutex;
    use serde::{Deserialize, Serialize};
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::nn::{Adam, AdamW, OptimizerConfig, VarStore};
    use tch::{nn, Device, Kind, Reduction, Tensor};
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tokio::task::coop::cooperative;
    use tracing::field::debug;
    use thoosa_common::batches::{mock_async_data_batch, AsyncDataBatch, BatchLengths};
    use thoosa_common::builder::{Node, SeqBuilder};
    use thoosa_common::copy_ctx::CudaActorHandle;
    use thoosa_common::dims::{AnyDims, AnyMetrics, Batch, Dims, ThoBatch};
    use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
    use thoosa_common::ipc::cuda_wokrer::CudaWorker;
    use thoosa_common::ipc::json_ser::{ErasedWithState, SyncStageIpcResponse, SyncStageIpcWrapper, SyncStageIpcWrapperWithState};
    use thoosa_common::ipc::tests::run_ipc_test;
    use thoosa_common::runner::common::{MockRunCtx, RunCtx};
    use thoosa_common::runner::common::runnable::{DynSyncStage, DynSyncStageBox, Stage, StageWithState, SyncStage};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::purge::FuturesRunnerMapPurger;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::{ThoTensor, ThoTensorInner};
    use thoosa_common::tho_varstore::{ModConfig, ThoTrainModule};
    use thoosa_proxy_derive::Proxy;
    use thoosa_shared::{simple_id, ThoError, CopyToDevice};
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::traits::{MonoMetrics, SimpleId};
    use crate::tasks::train_motif_nets::TRAIN_MOTIF_STEP_ENDPOINT;
    use crate::tasks::train_step::{TrainStep, TrainStepInput, TrainStepInputWrapper};
    use crate::tasks::train_step_2::{TrainStep2, TrainStep2DynSyncStage, TrainStep2Input, TrainStep2InputWrapper, TrainStep2Output, TrainStep2WeightsSync, TrainStep2XFn, TRAIN_TEST_STEP2_ENDPOINT, TrainStep2ProxyFactory};
    use futures_util::FutureExt;
    use futures_util::stream::BoxStream;
    use tokio::sync::MutexGuard;

    ////
    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct InDimsX {}
    impl Dims for InDimsX { type Metrics = MonoMetrics<0, 1>; }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct OutDimsX {}
    impl Dims for OutDimsX { type Metrics = MonoMetrics<0, 1>; }

    #[derive(Debug, Clone)]
    struct TestNetXParams {
        in_features:u32,
        layers:usize
    }

    #[derive(Debug, Clone)]
    struct TestRunConfig {
        pub lr:f64,
        pub in_features:u32,
        pub layers:usize
    }

    impl TestRunConfig {
        pub fn new(lr: f64, in_features: u32, layers: usize) -> Self {
            Self { lr, in_features, layers }
        }
    }

    #[derive(Debug)]
    struct TrainStepXFactory {//X for Motifs, Label, Final in TrainMotifNet, TrainLabelNets, TrainFinalNet
    }

    trait TrainStepFactory {
        type NP:Send;
        type R:Send + Debug+Clone; //RunConfig
        type InDims:Dims + 'static;
        type OutDims:Dims + 'static;

        type C:RunCtx + 'static;



        fn new(run_config:Self::R, ctx:Self::C) -> impl Future<Output = Result<(TrainStep2DynSyncStage<Box<dyn Fn(ThoBatch<Self::InDims>)-> ThoBatch<Self::InDims>+ Send + Sync + 'static>, Self::InDims, Self::OutDims, Self::C>, Self::NP), ThoError>> + Send;
    }

    impl TrainStepFactory for TrainStepXFactory {
        type NP = TestNetXParams;
        type R = TestRunConfig;
        type InDims = InDimsX;
        type OutDims = OutDimsX;
        type C = RunCtxImpl;

        fn new(run_config:TestRunConfig, ctx:Self::C) -> impl Future<Output = Result<(TrainStep2DynSyncStage<Box<dyn Fn(ThoBatch<Self::InDims>)-> ThoBatch<Self::InDims>+ Send + Sync + 'static>, Self::InDims, Self::OutDims, Self::C>, Self::NP), ThoError>> + Send{

            async move {
                if GlobalCtx::global().is_ipc() {
                    Ok((
                        TrainStep2DynSyncStage::Proxy {
                            stage: Arc::new(DynSyncStageBox::new(Arc::new(ProxyTrainTestStep2::new(
                                TrainTestStep2Init{},
                                1
                            ).await?) as Arc<dyn DynSyncStage<TrainStep2Input<InDimsX, OutDimsX>, TrainStep2Output,RunCtxImpl>>)),
                        },
                        TestNetXParams { in_features: 100, layers: 2 }
                    ))
                } else {
                    let (stage, params) = TrainStep2::new_local(
                        ctx.tho_device_idx().into(),
                        |rc, device| {
                            let lr = rc.lr;
                            async move {
                                let params = TestNetXParams { in_features: rc.in_features, layers: rc.layers };
                                let params_for_mod_config = params.clone();
                                /*
                                 * Recreate the same module topology for every
                                 * ThoTrainModule device restoration.
                                 */
                                let config = ModConfig::new(
                                    Adam::default(),
                                    move |var_store| test_net(&params_for_mod_config)?.build_tch_module::<Batch<InDimsX>, Batch<OutDimsX>>(&var_store.root()),
                                    lr,
                                );
                                Ok((ThoTrainModule::new_on_device(config, device)?, params))
                            }
                        },
                        Arc::new(Box::new(|t| {
                            t
                        }) as Box<dyn Fn(ThoBatch<InDimsX>) -> ThoBatch<InDimsX> + Send + Sync + 'static>),
                        |pred_y: &Tensor, target: &Tensor| {
                            Ok(pred_y.f_binary_cross_entropy_with_logits::<Tensor>(&target.f_totype(Float)?, None, None, Reduction::Mean)?)
                        },
                        &run_config,
                    ).await?;
                    Ok((TrainStep2DynSyncStage::Local(Arc::new(stage)), params))

                }
            }
        }
    }

    pub async fn opt_train_2<R, F, N>(//corresponds to opt_train crates/thoosa_common/src/train_nets.rs:2037
        run_config_source:R,//opt_train functionality should remain as is, the single run config here is for simplicity
        mut train: F
    ) ->Result<(VarStore, f32, N), ThoError>
    where
        R:Clone,
        F:AsyncFnMut(R)->Result<(VarStore, f32, N), ThoError> //+run_config for debug
    {

        //runs few times getting run configurations from run_config_source and searching for optimal run_config
        train(
            run_config_source.clone(),
        ).await
    }


    struct TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {
        pd:PhantomData<SF>,
        run_config_source:Box<dyn Fn()->SF::R>,
        batch:B
    }

    impl <SF, B> Debug for TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("TrainNet2")
                .field("batch", &self.batch)
                .finish()
        }
    }
    #[derive(Debug)]
    struct TrainNet2Input;
    impl Input for TrainNet2Input{}
    impl <SF, B> Stage for TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {
        // Use the factory context here so the stage, train step, and batch stream share one context type.
        type Ctx = SF::C;
        type In<'a> = TrainNet2Input;
        type Out = ();
    }

    //-->for test only to avoid errors because of run_config_source which is not Sync and Send
    unsafe impl <SF, B> Sync for TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {}
    unsafe impl <SF, B> Send for TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {}//<--

    impl <SF, B> TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {
        fn run(self: Arc<Self>, input: <Self as Stage>::In<'static>, mut ctx: <Self as Stage>::Ctx) -> impl Future<Output=Result<()/*Self::Out*/, ThoError>> + Send {
            let run_config = (self.run_config_source)();
            let self_for_train = Arc::clone(&self);
            async move {


                opt_train_2(
                    run_config,
                    move |run_config| {
                        let self_arc = Arc::clone(&self_for_train);

                        self_arc.run_inner(
                            run_config,
                            ctx.branch(false),
                        )
                    }).await?;


                Ok(())
            }
        }

        fn run_inner(self:Arc<Self>, run_config:SF::R, mut ctx: <Self as Stage>::Ctx)->BoxFuture<'static, Result<(VarStore, f32, SF::NP), ThoError>>{
            debug!("TrainNet2::run_inner: run_config={run_config:?}");

            let self_arc = Arc::clone(&self);
            let  ctx_for_run_stage = ctx.branch(false);

            Box::pin(async move {
                let (train_step, net_params) = SF::new(run_config, ctx.branch(false)).await?;
                let train_step= Arc::new(train_step);
                let mut best_output = None;
                for epoch in 0..3 {
                    let self_arc = Arc::clone(&self_arc);
                    let mut batches = self_arc.batch.train_batch_stream(ctx.branch(false));


                    while let Some(Ok((x_t_0, lengths, expected_y))) = cooperative(batches.next()).await {
                        let input = TrainStep2Input::new(x_t_0, lengths, expected_y, true);

                        debug!("TrainNet2::run_inner: epoch={epoch} input={input:?}");

                        // TrainStep2 owns mutable module state, so use its stateful execution contract.
                        let r = GlobalCtx::global().cuda_actor(ctx.tho_device_idx()).run_stage(Arc::clone(&train_step), input, ctx.tho_device_idx(), &ctx_for_run_stage).await?;

                        debug!("TrainNet2::run_inner: epoch={epoch} r={r:?}");

                        best_output.replace(r);
                    }

                }

                let vs =VarStore::new(Cuda(0));
                Ok((vs,0.1, net_params))
            })
        }

        pub fn new(batch: B, run_config_source:Box<dyn Fn()->SF::R>) -> Self {
            Self {
                batch,
                run_config_source,
                pd: PhantomData
            }
        }
    }



    pub fn test_net(params: &TestNetXParams) -> Result<Node, ThoError> {
        debug!("final_node: params={:?}", params);

        let mut builder = SeqBuilder::new(params.in_features, None)?;
        let mut net = builder
            .closure(
                Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                    debug!("final_net: input={t:?}");
                    Ok(t.shallow_clone())
                })),
                None,
            )?
            .linear_s(
                2 * params.in_features,
                "final_net_first_layer".to_string(),
            )?
            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?;

        for i in 0..params.layers {
            net = net
                .linear_s(
                    params.in_features,
                    format!("final_net_{i}th_layer"),
                )?
                .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?;
        }

        net
            // .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))
            .linear_s(
                1,
                "final_net_last_layer",
            )?
            .func(|t|Tensor::f_relu(t).map_err(|e|e.into()))?
            .end_seq()
    }


    #[derive(Debug, Serialize, Deserialize)]
    struct TrainTestStep2Init {

    }

    #[derive(Debug)]
    struct TrainTestStep2IpcWrapper {
        stage: Arc<TrainTestStep2ProxyTarget>,
    }

    impl SyncStageIpcWrapper for TrainTestStep2IpcWrapper {
        fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
            // Keep this endpoint's concrete training-module state behind the erased worker capability.
            Some(self)
        }

        fn purge(&self, tho_device: ThoDevice) -> BoxFuture<Result<(), ThoError>>
        where
            Self: Sync
        {
            // Purge uses the application's CPU-first device index, not tch::Device.
            self.stage.base.purge_gpu(None, tho_device.tho_device_idx(), "zzz")
        }
    }

    impl SyncStageIpcWrapperWithState for TrainTestStep2IpcWrapper {
        type S = ThoTrainModule<InDimsX, OutDimsX>;

        fn pre_run<'a>(&'a self, _tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<MutexGuard<'a, Self::S>, ThoError>> {
            StageWithState::pre_run(self.stage.base.as_ref())
        }

        fn do_run_with_state(&self, input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, state: MutexGuard<'_, Self::S>, _tho_device: &ThoDevice) -> Result<SyncStageIpcResponse, ThoError> {
            tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(async move {
                    debug!("TrainTestStep2IpcWrapper::do_run:input={:?}", String::from_utf8_lossy(input));
                    let input = JsonIpcCodec::decode::<thoosa_common::ipc::codec::IpcSyncStageRunInput>(&input)?;
                    let input_dec = JsonIpcCodec::decode::<TrainStep2Input<InDimsX, OutDimsX>>(&input.into_parts().0)?;

                    let x = ThoTensor::ref_to_tensor(all_cuda_tensors, input_dec.x)?;
                    let expected_y = ThoTensor::ref_to_tensor(all_cuda_tensors, input_dec.expected_y)?;

                    let input = TrainStep2Input::new(x, input_dec.lengths, expected_y, true);
                    let stage = Arc::clone(&self.stage.base);

                    Ok(JsonIpcCodec::encode(&StageWithState::do_run_with_state(Arc::clone(&stage), &input, state, &RunCtxImpl::new(1, TRAIN_MOTIF_STEP_ENDPOINT))?)?)
                })
            }).map(SyncStageIpcResponse::Encoded)
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct  TrainTestStep2IpcRequest {

    }

    impl TrainTestStep2IpcRequest {
        fn new ()->TrainTestStep2IpcRequest {
            TrainTestStep2IpcRequest{

            }
        }
    }

    #[derive(Debug, Proxy)]
    #[proxy(
        common = "thoosa_common",
        name = "ProxyTrainTestStep2",
        endpoint = "TRAIN_TEST_STEP2_ENDPOINT",
        new_args = "init: TrainTestStep2Init, tho_device_idx: usize",
        init = "init",
        out_id = "()",
        endpoint_init = "train_test_step_2_endpoint_init",
        wrapper = "TrainTestStep2IpcWrapper"

    )]
    struct TrainTestStep2ProxyTarget {
        base: Arc<TrainStep2<Box<dyn Fn(ThoBatch<InDimsX>) -> ThoBatch<InDimsX> + Send + Sync + 'static>, InDimsX, OutDimsX, RunCtxImpl>>,
    }

    impl TrainTestStep2ProxyTarget {
        pub fn new()->Self {
            let run_config = TestRunConfig::new(0.001, 1000, 3);

            let (train_step, params) : (TrainStep2<Box<dyn Fn<_, Output = _> + std::marker::Send + Sync>, _, _, _>, TestNetXParams)=  futures::executor::block_on(TrainStep2::new_local(
                Cuda(0).into(),
                |rc, device|  {
                    let lr = rc.lr;
                    async move {
                        let params = TestNetXParams { in_features: 100, layers: 2 };
                        /*
                         * Recreate the same module topology for every
                         * ThoTrainModule device restoration.
                         */
                        let config = ModConfig::new(
                            Adam::default(),
                            |var_store| test_net(&TestNetXParams { in_features: 100, layers: 2 })?.build_tch_module::<Batch<InDimsX>, Batch<OutDimsX>>(&var_store.root()),
                            lr,
                        );
                        Ok((ThoTrainModule::new_on_device(config, device)?, params))
                    }
                },
                Arc::new(Box::new(|t|{
                    t
                }) as Box<dyn Fn(ThoBatch<InDimsX>) -> ThoBatch<InDimsX> + Send + Sync + 'static>),
                |pred_y: &Tensor, target: &Tensor|{
                    Ok(pred_y.f_binary_cross_entropy_with_logits::<Tensor>(&target.f_totype(Float)?, None, None, Reduction::Mean)?)
                },
                &run_config,
            )).unwrap();

            Self {
                base: Arc::new(train_step),
            }
        }
    }

    #[test]
    fn test_train_step_2_ipc() -> Result<(), ThoError> {


        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(async {
            let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
            let test_name = format!("{module}::test_train_step_2_ipc");
            GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
            let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, true);
            let cuda_actor = GlobalCtx::global().cuda_actor(1);

            let ipc_ref = cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[100], FLOAT_CPU))?, 1.into()).await?;
            let ThoTensorInner::Ref{ key, size, kind, selector, tho_device_idx } = ipc_ref.inner  else {
                panic!("should be ref")
            };


            let (train_step, params): (TrainStep2DynSyncStage<TrainStep2XFn<InDimsX>, InDimsX, OutDimsX, RunCtxImpl>, TestNetXParams) = (
                TrainStep2DynSyncStage::Proxy {
                    stage: Arc::new(DynSyncStageBox::new(Arc::new(ProxyTrainTestStep2::new(
                        TrainTestStep2Init{},
                        1
                    ).await?) as Arc<dyn DynSyncStage<TrainStep2Input<InDimsX, OutDimsX>, TrainStep2Output,RunCtxImpl>>)),
                },
                TestNetXParams { in_features: 100, layers: 2 }
            );

            do_test_train_step_2(train_step, async ||Ok(TrainStep2Input::<InDimsX, OutDimsX>::new(
                to_tho_tensor(cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[3, 100], FLOAT_CPU))?, 1.into()).await?.inner, "2".into()),
                None,
                to_tho_tensor(cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[3, 1], FLOAT_CPU))?, 1.into()).await?.inner, "3".into()),
                false
            ))).await?;


            Ok(())
        }, async |sock_path| {
            CudaWorker::run_worker(
                sock_path,
                HashMap::from([
                    (
                        TRAIN_TEST_STEP2_ENDPOINT.to_string(),
                        Box::new(train_test_step_2_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
                    ),
                ]),
                ThoDevice::from(Cuda(0)),
            ).await
        })

    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn test_train_step_2_local() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let (train_step, np) = TrainStepXFactory::new(TestRunConfig::new(0.001, 100, 3), RunCtxImpl::new(1, "test")).await?;


        do_test_train_step_2(train_step, || ready(Ok(TrainStep2Input::<InDimsX, OutDimsX>::new(
            tch::Tensor::rand(&[3, 100], FLOAT_CPU).into(),
            None,
            tch::Tensor::rand(&[3, 1], FLOAT_CPU).into(),
            false
        )))).await?;

        Ok(())
    }

    async fn do_test_train_step_2<FI>(train_step: TrainStep2DynSyncStage<TrainStep2XFn<InDimsX>, InDimsX, OutDimsX, RunCtxImpl>, input: FI) -> Result<(), ThoError>
    where
        FI:AsyncFn()->Result<TrainStep2Input<InDimsX, OutDimsX>, ThoError>
    {
        let train_step = Arc::new(train_step);
        GlobalCtx::global().register_cache(("train_step", Arc::clone(&train_step) as Arc<dyn FuturesRunnerMapPurger>));

        let cuda_actor = GlobalCtx::global().cuda_actor(1);

        let ctx = RunCtxImpl::new(1, "x");

        // let r = Arc::clone(&train_step).do_run(&input, &ctx)?;
        let r = GlobalCtx::global().cuda_actor(1).run_stage(Arc::clone(&train_step), input().await?, 1, &ctx).await?;

        GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;


        let r = GlobalCtx::global().cuda_actor(1).run_stage(Arc::clone(&train_step), input().await?, 1, &ctx).await?;

        debug!("do_test_train_step_2: r={r:?}");
        Ok(())
    }

    pub fn to_tho_tensor<D>(inner: ThoTensorInner, id: String) -> ThoTensor<D, SimpleId>
    where
        D:Dims
    {
        let ThoTensorInner::Ref{ key, size, kind, selector, tho_device_idx } = inner  else {
            panic!("should be ref")
        };
        ThoTensor::new_ref(simple_id!(id), key, size.clone(), kind, None, tho_device_idx)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_net_2_local()->Result<(), ThoError>{
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let train_stream_calls = Arc::new(AtomicUsize::new(0));
        let valid_stream_calls = Arc::new(AtomicUsize::new(0));

        let mock_test_batch = Arc::new(
            mock_async_data_batch::build::<InDimsX, OutDimsX, RunCtxImpl>()
                .with_train_stream_calls(Arc::clone(&train_stream_calls), |_ctx| make_stream(false))
                .with_valid_stream_calls(Arc::clone(&valid_stream_calls), |_ctx| make_stream(false))
                .finish(),
        );
        let train_net = Arc::new(TrainNet2::<TrainStepXFactory, _>::new(mock_test_batch, Box::new(||TestRunConfig::new(0.001, 100, 3))));

        let r = train_net.run(TrainNet2Input{}, RunCtxImpl::new(1, "test")).await?;

        Ok(())

    }

    #[test]
    fn test_train_net_2_ipc()->Result<(), ThoError>{

        run_ipc_test(async {
            let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
            let test_name = format!("{module}::test_train_net_2_ipc");
            GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
            let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, true);
            let cuda_actor = GlobalCtx::global().cuda_actor(1);

            let train_stream_calls = Arc::new(AtomicUsize::new(0));
            let valid_stream_calls = Arc::new(AtomicUsize::new(0));


            let mock_test_batch = Arc::new(
                mock_async_data_batch::build::<InDimsX, OutDimsX, RunCtxImpl>()
                    .with_train_stream_calls(Arc::clone(&train_stream_calls), |_ctx| make_stream(true))
                    .with_valid_stream_calls(Arc::clone(&valid_stream_calls), |_ctx| make_stream(true))
                    .finish(),
            );
            let train_net = Arc::new(TrainNet2::<TrainStepXFactory, _>::new(mock_test_batch, Box::new(||TestRunConfig::new(0.001, 100, 3))));

            let r = train_net.run(TrainNet2Input{}, RunCtxImpl::new(1, "test")).await?;

            Ok(())
        }, async |sock_path| {
            CudaWorker::run_worker(
                sock_path,
                HashMap::from([
                    (
                        TRAIN_TEST_STEP2_ENDPOINT.to_string(),
                        Box::new(train_test_step_2_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
                    ),
                ]),
                ThoDevice::from(Cuda(0)),
            ).await
        })

    }



    fn dummy_batch(make_refs:bool) -> Result<(ThoBatch<InDimsX>, ThoBatch<OutDimsX>), ThoError> {
        // Match the test network's 100 input features so this fixture exercises the training path.
        if make_refs {
            let cuda_actor = GlobalCtx::global().cuda_actor(1);

            futures::executor::block_on(async {
                let x = cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[1, 1, 100], FLOAT_CPU))?, 1.into()).await?;
                let ThoTensorInner::Ref{ key:key_x, size:size_x, kind, selector, tho_device_idx } = x.inner  else {
                    panic!("should be ref")
                };
                let y = cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[1, 1, 1], FLOAT_CPU))?, 1.into()).await?;
                let ThoTensorInner::Ref{ key:key_y, size:size_y, kind, selector, tho_device_idx } = y.inner  else {
                    panic!("should be ref")
                };
                Ok(
                    (
                        ThoTensor::new_ref(simple_id!(format!("x_{}", key_x)), key_x, vec![1, 1, 100], Float, None, 1),
                        ThoTensor::new_ref(simple_id!(format!("y_{}", key_y)), key_y, vec![1, 1, 1], Float, None, 1),

                    )
                )
            })

        } else {
            let x = Tensor::zeros([1, 1, 100], FLOAT_CUDA);
            let y = Tensor::zeros([1, 1, 1], FLOAT_CUDA);
            Ok((x.into(), y.into()))
        }
    }

    fn make_stream(make_refs:bool) -> BoxStream<'static, Result<(ThoBatch<InDimsX>, Option<BatchLengths>, ThoBatch<OutDimsX>), ThoError>> {
        let (x, y) = dummy_batch(make_refs).unwrap();
        stream::iter(vec![Ok((x, None, y))]).boxed()
    }
}
