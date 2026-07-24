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
use thoosa_common::runner::common::cachable::{BorrowRef, ThoDataRef};
use thoosa_common::runner::common::runnable::{DynSyncStageBox, Stage, SyncStage};
use thoosa_common::runner::common::{MaxParallelCtx, OutId, RunCtx};
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::optimizer_mem_size;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, WithId};
use thoosa_shared::{CopyToDevice, ThoError, mono_key, os_thread_id, panic_to_err, perf_log, sync_mutex_lock};

pub struct TrainStep2<F, DI: Dims, DO: Dims, C: RunCtx> {
    device: ThoDevice,
    net: Mutex<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>>,

    opt: Mutex<Optimizer>,
    var_store: Arc<Mutex<VarStore>>,

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
        todo!()
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
    pub(crate) async fn new<NP, NB, NBF, OF, RC>(tho_device: ThoDevice, net_builder: NB, optimizer_builder: OF, x_f: Arc<F>, loss_fn: LossFp, proxy_factory: Option<&Arc<dyn TrainStep2ProxyFactory<DI, DO, C, NP, RC>>>, run_config: &RC) -> Result<(TrainStep2DynSyncStage<F, DI, DO, C>, NP), ThoError>
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
            let TrainStep2ProxyParts { stage, weights_sync, node_params } = proxy_factory.new_proxy(run_config, tho_device.tho_device_idx()).await?;
            return Ok((TrainStep2DynSyncStage::Proxy { stage, weights_sync }, node_params));
        }

        let (stage, node_params) = Self::new_local(tho_device, net_builder, optimizer_builder, x_f, loss_fn, run_config).await?;
        Ok((TrainStep2DynSyncStage::Local(Arc::new(stage)), node_params))
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
        Arc::try_unwrap(var_store).map(Mutex::into_inner).map_err(|_| ThoError::new("TrainStep2::into_var_store: VarStore still shared after training"))
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
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep2::do_run"))]
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        debug!("TrainStep2::do_run: os_thread_id={}", os_thread_id());
        let x_t_0 = input.x();
        let expected_y = input.expected_y();

        debug!("TrainStep2::do_run: device={:?} x_t_0.size={:?} expected_y.size={:?}", self.device, x_t_0.size(), expected_y.size());
        debug!("TrainStep2::do_run: 0. cuda_mem={:?}",maybe_cuda_mem(&self.device));

        let x_t = (self.x_f)(x_t_0);

        debug!("TrainStep2::do_run: device={:?}  x_t.size={:?}", self.device, x_t.size());

        let mut lengths = input.lengths.clone();
        let mut calculate_loss = || {


            let net = perf_log!(sync_mutex_lock!(self.net));
            let predicted_y = net.tho_forward(&x_t, lengths.as_mut(), input.train)?;
            drop(net);

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
            // TODO(perf): lock_opt is per batch; keep optimizer ownership local to the training task or queue optimizer steps to avoid mutex traffic in TrainStep2::do_run.
            let mut opt = thoosa_shared::perf_log!(thoosa_shared::sync_mutex_lock!(self.opt));
            debug!("TrainStep2::do_run: opt={opt:?} optimizer_size={}", optimizer_mem_size(&opt));

            debug!("TrainStep2::do_run: loss={loss} cuda_mem={:?}", maybe_cuda_mem(&self.device));
            trace!("TrainStep2::do_run: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}",optimizer_mem_size(&opt));

            predicted_y.drop();
            drop(expected_y);
            debug!("TrainStep2::do_run: 2.1 cuda_mem={:?}",maybe_cuda_mem(&self.device));

            let t1 = Instant::now();

            panic_to_err!({
                opt.backward_step(&loss.set_requires_grad(true));
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
        f.debug_struct("TrainStep2Input").finish()
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
    pub vs_ref: String,
}

impl TrainStep2Output {
    fn new(loss: f32) -> Self {
        Self { loss, vs_ref: "todo".to_string() }
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
        weights_sync: Arc<dyn TrainStep2WeightsSync>,
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
            TrainStep2DynSyncStage::Proxy {  .. } => {
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

impl<F, DI, DO, C> TrainStep2DynSyncStage<F, DI, DO, C>
where
    DI: Dims + 'static,
    DO: Dims + 'static,
    C: RunCtx + 'static,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep2DynSyncStage::into_var_store"))]
    pub(crate) async fn into_var_store(self) -> Result<VarStore, ThoError> {
        match self {
            Self::Local(stage) => Arc::try_unwrap(stage).map_err(|_| ThoError::new("TrainStep2DynSyncStage::into_var_store: local TrainStep2 still shared"))?.into_var_store(),
            Self::Proxy { weights_sync, .. } => weights_sync.into_var_store().await,
        }
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
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => Arc::clone(stage).do_run(input, ctx),
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
    use std::sync::Arc;
    use async_trait::async_trait;
    use log::debug;
    use parking_lot::Mutex;
    use serde::{Deserialize, Serialize};
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::nn::{AdamW, OptimizerConfig, VarStore};
    use tch::{Reduction, Tensor};
    use tch::kind::FLOAT_CPU;
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::builder::{Node, SeqBuilder};
    use thoosa_common::dims::{Batch, Dims, ThoBatch};
    use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
    use thoosa_common::ipc::cuda_wokrer::CudaWorker;
    use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
    use thoosa_common::ipc::tests::run_ipc_test;
    use thoosa_common::runner::common::runnable::{DynSyncStage, DynSyncStageBox, SyncStage};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::purge::FuturesRunnerMapPurger;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::{ThoTensor, ThoTensorInner};
    use thoosa_proxy_derive::Proxy;
    use thoosa_shared::{simple_id, ThoError};
    use thoosa_shared::traits::{MonoMetrics, SimpleId};
    use crate::tasks::train_motif_nets::TRAIN_MOTIF_STEP_ENDPOINT;
    use crate::tasks::train_step_2::{TrainStep2, TrainStep2DynSyncStage, TrainStep2Input, TrainStep2Output, TrainStep2WeightsSync, TrainStep2XFn};

    #[test]
    fn test_train_step_ipc_2() -> Result<(), ThoError> {
        pub(crate) const TRAIN_TEST_STEP2_ENDPOINT: &str = "TrainTestStep2";

        #[derive(Debug, Serialize, Deserialize)]
        struct TrainTestStep2Init {

        }
        #[derive(Debug)]
        struct TrainTestStep2IpcWrapper {
            stage: Arc<TrainTestStep2ProxyTarget>,
        }
        impl SyncStageIpcWrapper for TrainTestStep2IpcWrapper {
            fn do_run(&self, input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
                debug!("TrainTestStep2IpcWrapper::do_run:input={:?}", String::from_utf8_lossy(input));
                let input_dec = JsonIpcCodec::decode::<TrainStep2Input<InDims, OutDims>>(input)?;

                let x = ThoTensor::ref_to_tensor(all_cuda_tensors, input_dec.x)?;
                let expected_y = ThoTensor::ref_to_tensor(all_cuda_tensors, input_dec.expected_y)?;

                let input = TrainStep2Input::new(x, input_dec.lengths, expected_y, true);

                Ok(JsonIpcCodec::encode(&Arc::clone(&self.stage.base).do_run(&input, &RunCtxImpl::new(1, TRAIN_MOTIF_STEP_ENDPOINT))?)?)
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

        #[derive(Debug, Serialize, Deserialize, Clone)]
        pub struct InDims {}
        impl Dims for InDims{ type Metrics = MonoMetrics<0, 1>; }

        #[derive(Debug, Serialize, Deserialize, Clone)]
        pub struct OutDims {}
        impl Dims for OutDims{ type Metrics = MonoMetrics<0, 1>; }

        #[derive(Debug)]
        struct TestNetParams {
            in_features:u32,
            layers:usize
        }

        #[derive(Debug)]
        struct TestRunConfig {
            lr:f64
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
            base: Arc<TrainStep2<Box<dyn Fn(ThoBatch<InDims>) -> ThoBatch<InDims> + Send + Sync + 'static>, InDims, OutDims, RunCtxImpl>>,
        }

        impl TrainTestStep2ProxyTarget {
            pub fn new()->Self {
                let run_config = TestRunConfig{ lr: 0.001 };

                let (train_step, params) : (TrainStep2<Box<dyn Fn<_, Output = _> + std::marker::Send + Sync>, _, _, _>, TestNetParams)=  futures::executor::block_on(TrainStep2::new_local(
                    Cuda(0).into(),
                    |vs|  {
                        async move {
                            let params = TestNetParams { in_features: 100, layers: 2 };
                            let net = test_net(&params)?.build_tch_module::<Batch<InDims>, Batch<OutDims>>(&vs.lock().root())?;

                            Ok((net, params))
                        }
                    },
                    |rc, vs|  {
                        AdamW::default().build(&vs, rc.lr).map_err(|e|e.into())
                    },
                    Arc::new(Box::new(|t|{
                        t
                    }) as Box<dyn Fn(ThoBatch<InDims>) -> ThoBatch<InDims> + Send + Sync + 'static>),
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

        pub fn test_net(params: &TestNetParams) -> Result<Node, ThoError> {
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


        // Use the shared harness so the main and CUDA worker processes get identical runtime setup.
        run_ipc_test(async {
            let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
            let test_name = format!("{module}::test_train_step_ipc_2");
            GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
            let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, true);
            let cuda_actor = GlobalCtx::global().cuda_actor(1);

            let ipc_ref = cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[100], FLOAT_CPU))?, 1.into()).await?;
            let ThoTensorInner::Ref{ key, size, kind, selector, tho_device_idx } = ipc_ref.inner  else {
                panic!("should be ref")
            };

            let proxy = Arc::new(ProxyTrainTestStep2::new(
                TrainTestStep2Init{},
                1
            ).await?);

            #[derive(Debug)]
            struct TestWeightSync{}
            #[async_trait]
            impl TrainStep2WeightsSync for TestWeightSync {
                async fn into_var_store(&self) -> Result<VarStore, ThoError> {
                    Ok(VarStore::new(Cuda(0)))
                }
            }
            let (train_step, params):
                (TrainStep2DynSyncStage<TrainStep2XFn<InDims>, InDims, OutDims, RunCtxImpl>, TestNetParams)
                = (
                TrainStep2DynSyncStage::Proxy {
                    stage: Arc::new(DynSyncStageBox::new(proxy as Arc<dyn DynSyncStage<TrainStep2Input<InDims, OutDims>, TrainStep2Output,RunCtxImpl>>)),
                    weights_sync: Arc::new(TestWeightSync{}) as Arc<dyn TrainStep2WeightsSync>
                },
                TestNetParams { in_features: 100, layers: 2 }
            );

            let train_step = Arc::new(train_step);
            GlobalCtx::global().register_cache(("train_step", Arc::clone(&train_step) as Arc<dyn FuturesRunnerMapPurger>));


            let input = TrainStep2Input::<InDims, OutDims>::new(
                to_tho_tensor(cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[3, 100], FLOAT_CPU))?, 1.into()).await?.inner, "2".into()),
                None,
                to_tho_tensor(cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[3, 1], FLOAT_CPU))?, 1.into()).await?.inner, "3".into()),
                false
            );
            let ctx = RunCtxImpl::new(1, "x");

            let r = Arc::clone(&train_step).do_run(&input, &ctx)?;

            GlobalCtx::global().dump_all_to_cpu(&HashMap::new(), 1, "test").await?;


            let r = train_step.do_run(&input, &ctx)?;

            debug!("test_train_step_ipc: r={r:?}");


            Ok(())
        }, async |sock_path| {
            CudaWorker::run_worker(
                sock_path,
                HashMap::from([
                    (
                        TRAIN_TEST_STEP2_ENDPOINT.to_string(),
                        Box::new(train_test_step_2_endpoint_init) as Box<dyn Fn(Vec<u8>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError>>,
                    ),

                ])
                ,
                ThoDevice::from(Cuda(0)),
            ).await
        })


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

}