use crate::tasks::train_net::maybe_cuda_mem;
use async_trait::async_trait;
use log::{debug, trace};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;
use tch::Device;
use tch::nn::{Optimizer, VarStore};
use thoosa_common::batches::BatchLengths;
use thoosa_common::copy_ctx::CudaActorHandle;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::my_tch::{LossFp, ThoModule};
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_common::runner::common::cachable::{BorrowRef, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::optimizer_mem_size;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, DynSyncStageBox, OutId, RunCtx, Stage, SyncStage, WithId};
use thoosa_shared::{CopyToDevice, ThoError, mono_key, os_thread_id, panic_to_err, perf_log, sync_mutex_lock};

pub struct TrainStep<F, DI: Dims, DO: Dims, C: RunCtx> {
    device: ThoDevice,
    net: Mutex<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>>,

    opt: Mutex<Optimizer>,
    var_store: Arc<Mutex<VarStore>>,

    x_f: Arc<F>,
    loss_fn: LossFp,
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
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        true
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        debug!("TrainStep::purge_gpu");
        todo!()
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        debug!("TrainStep::drop_gpu");
        todo!()
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        debug!("TrainStep::purge_cpu");
        0
    }
}

impl<F, DI: Dims, DO: Dims, C: RunCtx> Debug for TrainStep<F, DI, DO, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainStep").finish()
    }
}

impl<F, DI, DO, C> TrainStep<F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep::new"))]
    pub(crate) async fn new<NP, NB, NBF, OF, RC>(tho_device: ThoDevice, net_builder: NB, optimizer_builder: OF, x_f: Arc<F>, loss_fn: LossFp, proxy_factory: Option<&Arc<dyn TrainStepProxyFactory<DI, DO, C, NP, RC>>>, run_config: &RC) -> Result<(TrainStepDynSyncStage<F, DI, DO, C>, NP), ThoError>
    where
        DI: 'static,
        DO: 'static,
        C: 'static,
        NP: Debug + Send + Sync + 'static,
        RC: Debug + Send + Sync + 'static,
        NB: FnOnce(Arc<Mutex<VarStore>>) -> NBF,
        NBF: Future<Output = Result<(Box<dyn ThoModule<Batch<DI>, Batch<DO>>>, NP), ThoError>>,
        OF: FnOnce(&RC, &VarStore) -> Result<Optimizer, ThoError>,
    {
        if GlobalCtx::global().is_ipc()
            && let Some(proxy_factory) = proxy_factory
        {
            let TrainStepProxyParts { stage, weights_sync, node_params } = proxy_factory.new_proxy(run_config, tho_device.tho_device_idx()).await?;
            return Ok((TrainStepDynSyncStage::Proxy { stage, weights_sync }, node_params));
        }

        let (stage, node_params) = Self::new_local(tho_device, net_builder, optimizer_builder, x_f, loss_fn, run_config).await?;
        Ok((TrainStepDynSyncStage::Local(Arc::new(stage)), node_params))
    }

    pub(crate) async fn  new_local<NP, NB, NBF, OF, RC>(device: ThoDevice, net_builder: NB, optimizer_builder: OF, x_f: Arc<F>, loss_fn: LossFp, run_config: &RC) -> Result<(Self, NP), ThoError>
    where
        NB: FnOnce(Arc<Mutex<VarStore>>) -> NBF,
        NBF: Future<Output = Result<(Box<dyn ThoModule<Batch<DI>, Batch<DO>>>, NP), ThoError>>,
        OF: FnOnce(&RC, &VarStore) -> Result<Optimizer, ThoError>,
    {
        let var_store = Arc::new(Mutex::new(VarStore::new(device.clone().into())));
        let (net, node_params) = net_builder(Arc::clone(&var_store)).await?;
        let opt = {
            let var_store = perf_log!(sync_mutex_lock!(var_store));
            optimizer_builder(run_config, &var_store)?
        };
        Ok((Self {
            device,
            net: Mutex::new(net),
            opt: Mutex::new(opt),
            var_store,
            x_f,
            loss_fn,
            pd: PhantomData,
        }, node_params))
    }

    pub fn into_var_store(self) -> Result<VarStore, ThoError> {
        let Self { net, opt, var_store, .. } = self;
        drop(net);
        drop(opt);
        Arc::try_unwrap(var_store).map(Mutex::into_inner).map_err(|_| ThoError::new("TrainStep::into_var_store: VarStore still shared after training"))
    }

    pub(crate) fn save_var_store_to_stream(&self, output: &mut dyn Write) -> Result<(), ThoError> {
        let var_store = perf_log!(sync_mutex_lock!(self.var_store));
        debug!("save_var_store_to_stream: var_store={var_store:?}");
        var_store.save_to_stream(output).map_err(ThoError::from)
    }

    fn set_lr(&self, lr: f64) {
        let mut opt = perf_log!(sync_mutex_lock!(self.opt));
        opt.set_lr(lr);
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
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep::do_run"))]
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("TrainStep::do_run: os_thread_id={}", os_thread_id());
        let x_t_0 = input.x();
        let expected_y = input.expected_y();

        debug!("TrainStep::do_run: device={:?} x_t_0.size={:?} expected_y.size={:?}", self.device, x_t_0.size(), expected_y.size());
        debug!("TrainStep::do_run: 0. cuda_mem={:?}",maybe_cuda_mem(&self.device));

        let x_t = (self.x_f)(x_t_0);

        debug!("TrainStep::do_run: device={:?}  x_t.size={:?}", self.device, x_t.size());

        let mut lengths = input.lengths.clone();
        let mut calculate_loss = || {


            let net = perf_log!(sync_mutex_lock!(self.net));
            let predicted_y = net.tho_forward(&x_t, lengths.as_mut(), input.train)?;
            drop(net);

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
            // TODO(perf): lock_opt is per batch; keep optimizer ownership local to the training task or queue optimizer steps to avoid mutex traffic in TrainStep::do_run.
            let mut opt = thoosa_shared::perf_log!(thoosa_shared::sync_mutex_lock!(self.opt));
            debug!("TrainStep::do_run: opt={opt:?} optimizer_size={}", optimizer_mem_size(&opt));

            debug!("TrainStep::do_run: loss={loss} cuda_mem={:?}", maybe_cuda_mem(&self.device));
            trace!("TrainStep::do_run: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}",optimizer_mem_size(&opt));

            predicted_y.drop();
            drop(expected_y);
            debug!("TrainStep::do_run: 2.1 cuda_mem={:?}",maybe_cuda_mem(&self.device));

            let t1 = Instant::now();

            panic_to_err!({
                opt.backward_step(&loss.set_requires_grad(true));
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

#[async_trait]
pub(crate) trait TrainStepWeightsSync: Debug + Send + Sync {
    async fn into_var_store(&self) -> Result<VarStore, ThoError>;
}

#[async_trait]
pub(crate) trait TrainStepProxyFactory<DX, DY, C, NP, RC>: Debug + Send + Sync
where
    DX: Dims + 'static,
    DY: Dims + 'static,
    C: RunCtx + 'static,
    NP: Debug + Send + Sync + 'static,
    RC: Debug + Send + Sync + 'static,
{
    async fn new_proxy(&self, run_config: &RC, tho_device_idx: usize) -> Result<TrainStepProxyParts<DX, DY, C, NP>, ThoError>;
}

#[derive(Serialize, Deserialize)]
pub struct TrainStepInput<DI: Dims, DO: Dims> {
    id: SynKey,
    pub x: ThoBatch<DI>,
    pub lengths: Option<BatchLengths>,
    pub expected_y: ThoBatch<DO>,
    train: bool,
}

impl<DI: Dims, DO: Dims> TrainStepInput<DI, DO> {
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
    pub vs_ref: String,
}

impl TrainStepOutput {
    fn new(loss: f32) -> Self {
        Self { loss, vs_ref: "todo".to_string() }
    }
}

impl WithId for TrainStepOutput {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Output for TrainStepOutput {}

pub(crate) enum TrainStepDynSyncStage<F, DI: Dims + 'static, DO: Dims + 'static, C: RunCtx + 'static> {
    Local(Arc<TrainStep<F, DI, DO, C>>),
    Proxy {
        stage: Arc<DynSyncStageBox<TrainStepInput<DI, DO>, TrainStepOutput, C>>,
        weights_sync: Arc<dyn TrainStepWeightsSync>,
    },
}

#[async_trait]
impl <F, DI, DO, C> FuturesRunnerMapPurger for TrainStepDynSyncStage<F, DI, DO, C>
where
    DI:Dims,
    DO:Dims,
    C:RunCtx,
    Self:Send + Sync
{
    async fn has_gpu_data(&self, device_idx: usize) -> bool {
        match self {
            TrainStepDynSyncStage::Local(local) => {
                local.has_gpu_data(device_idx).await
            }
            TrainStepDynSyncStage::Proxy { .. } => {
                true
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            TrainStepDynSyncStage::Local(local) => {
                local.purge_gpu(ids_to_keep, device_idx, name).await
            }
            TrainStepDynSyncStage::Proxy {  stage, .. } => {
                Ok(())
            }
        }

    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            TrainStepDynSyncStage::Local(local) => {
                local.drop_gpu(device_idx, name).await
            }
            TrainStepDynSyncStage::Proxy { .. } => {
                Ok(())
            }
        }

    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        match self {
            TrainStepDynSyncStage::Local(local) => {
                local.purge_cpu(purge_until).await
            }
            TrainStepDynSyncStage::Proxy { .. } => {
                0
            }
        }

    }
}

impl<F, DI, DO, C> Debug for TrainStepDynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainStepDynSyncStage").finish()
    }
}

impl<F, DI, DO, C> TrainStepDynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStepDynSyncStage::into_var_store"))]
    pub(crate) async fn into_var_store(self) -> Result<VarStore, ThoError> {
        match self {
            Self::Local(stage) => Arc::try_unwrap(stage).map_err(|_| ThoError::new("TrainStepDynSyncStage::into_var_store: local TrainStep still shared"))?.into_var_store(),
            Self::Proxy { weights_sync, .. } => weights_sync.into_var_store().await,
        }
    }
}

impl<F, DI, DO, C> Stage for TrainStepDynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    type Ctx = C;
    type In<'a> = TrainStepInput<DI, DO>;
    type Out = TrainStepOutput;
}

impl<F, DI, DO, C> SyncStage for TrainStepDynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(input, ctx),
            Self::Proxy { stage, .. } => Arc::clone(stage).do_run(input, ctx),
        }
    }
}

impl<F, DI, DO, C> OutId for TrainStepDynSyncStage<F, DI, DO, C>
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

impl<F, DI, DO, C> Adjustable for TrainStepDynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{}

// Train-step proxies pass worker tensor references instead of serializing batch tensors through JSON.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TrainStepTensorRef {
    pub(crate) ref_key: u64,
    size: Vec<i64>,
    kind: String,
    tho_device_idx: usize,
    pub(crate) remove_after_run: bool,
}

impl TrainStepTensorRef {
    fn from_ref<D, Id>(tensor: &ThoTensor<D, Id>) -> Result<Self, ThoError>
    where
        D: Dims,
        Id: thoosa_shared::traits::Named + Debug + Clone,
    {
        Ok(Self {
            ref_key: tensor.ref_key().ok_or_else(|| ThoError::new(format!("TrainStep expected worker tensor ref, got {tensor:?}")))?,
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
        let kind = thoosa_common::ipc::json_ser::tho_tensor::str_to_kind(&self.kind).ok_or_else(|| ThoError::new(format!("TrainStep received unsupported tensor kind {:?}", self.kind)))?;
        Ok(ThoTensor::new_ref(thoosa_shared::simple_id!("train_step_ipc"), self.ref_key, self.size, kind, None, self.tho_device_idx))
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) enum TrainStepIpcRequest {
    Run {
        x: TrainStepTensorRef,
        lengths: Option<Vec<i64>>,
        expected_y: TrainStepTensorRef,
        train: bool,
        max_other_parallel: u8,
    },
    ExportWeights,
}

impl TrainStepIpcRequest {
    pub(crate) async fn run<DI, DO, C>(input: &TrainStepInput<DI, DO>, ctx: &C) -> Result<Self, ThoError>
    where
        DI: Dims,
        DO: Dims,
        C: RunCtx,
    {
        Ok(Self::Run {
            x: track_await(TrainStepTensorRef::from_tensor(input.x_ref(), ctx.tho_device_idx())).await?,
            lengths: input.lengths_ref().map(|lengths| lengths.as_slice().to_vec()),
            expected_y: track_await(TrainStepTensorRef::from_tensor(input.expected_y_ref(), ctx.tho_device_idx())).await?,
            train: input.train(),
            max_other_parallel: ctx.max_other_parallel(),
        })
    }
}

pub(crate) struct TrainStepProxyParts<DX, DY, C, NP>
where
    DX: Dims + 'static,
    DY: Dims + 'static,
    C: RunCtx + 'static,
{
    stage: Arc<DynSyncStageBox<TrainStepInput<DX, DY>, TrainStepOutput, C>>,
    weights_sync: Arc<dyn TrainStepWeightsSync>,
    node_params: NP,
}

impl<DX, DY, C, NP> TrainStepProxyParts<DX, DY, C, NP>
where
    DX: Dims + 'static,
    DY: Dims + 'static,
    C: RunCtx + 'static,
{
    pub(crate) fn new(stage: Arc<DynSyncStageBox<TrainStepInput<DX, DY>, TrainStepOutput, C>>, weights_sync: Arc<dyn TrainStepWeightsSync>, node_params: NP) -> Self {
        Self { stage, weights_sync, node_params }
    }
}