use crate::tasks::train_net::maybe_cuda_mem;
use async_trait::async_trait;
use futures::FutureExt;
use futures_util::StreamExt;
use futures_util::future::BoxFuture;
use log::{debug, trace};
use parking_lot::{Mutex, RwLock};
use rand::random;
use serde::{Deserialize, Serialize};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::future::ready;
use std::io::Write;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Deref, DerefMut};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use tch::Device;
use tch::nn::VarStore;
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
use thoosa_common::tho_varstore::ThoTrainModule;
use thoosa_common::utils::optimizer_mem_size;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::tmp_file::TmpFile;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::ErasedStageWithState;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, OutId, RunCtx, Stage, StageWithState, SyncStage, WithId};
use thoosa_shared::{CopyToDevice, MoveToDevice, ThoError, mono_key, now_ms, os_thread_id, panic_to_err, perf_log, sync_mutex_lock};
use tokio::sync::{MutexGuard, RwLockReadGuard};

// The proxy purge path and the test worker registration must use the same
// endpoint name in both normal and cfg(test) builds.
pub(crate) const TRAIN_TEST_STEP2_ENDPOINT: &str = "TRAIN_TEST_STEP2_ENDPOINT";
pub(crate) const APPLY_NET2_X_ENDPOINT: &str = "APPLY_NET2_X_ENDPOINT";

pub struct TrainStep2<F, DI: Dims, DO: Dims, C: RunCtx> {
    device: ThoDevice,
    // Keep the module, optimizer, and their VarStore under one owner so their
    // lifetimes cannot diverge while training state is copied or exported.
    train_module: tokio::sync::Mutex<ThoTrainModule<DI, DO>>,

    x_f: Arc<F>,
    loss_fn: LossFp,
    best_checkpoint_path:PathBuf,
    pd: PhantomData<C>,

}
//for FuturesRunnerMapPurger
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

    //it's called when local GPU reset is requested
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
        f.debug_struct("TrainStep2")
            .field("best_checkpoint_path", &self.best_checkpoint_path)
            .finish()
    }
}

impl<F, DI, DO, C> TrainStep2<F, DI, DO, C>
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


        debug!("TrainStep2::new_local: best_checkpoint_path={best_checkpoint_path:?}");
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

    fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
        async {
            let mut guard = self.train_module.lock().await;
            guard.move_to_device(1).await?;//TODO config
            Ok(guard)
        }.boxed()
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainStep2::do_run_with_state"))]
    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, mut train_module: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {

        debug!("TrainStep2::do_run: os_thread_id={} input={input:?}", os_thread_id());
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
            if loss_value < input.best_valid_loss {
                train_module.save_var_store(&self.best_checkpoint_path)?;
            }
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



#[derive(Serialize, Deserialize)]
pub struct TrainStep2Input<DI: Dims, DO: Dims> {
    id: SynKey,
    pub x: ThoBatch<DI>,
    pub lengths: Option<BatchLengths>,
    pub expected_y: ThoBatch<DO>,
    train: bool,
    best_valid_loss:f32

}

impl<DI: Dims, DO: Dims> TrainStep2Input<DI, DO> {
    pub(crate) fn new(x: ThoBatch<DI>, lengths: Option<BatchLengths>, expected_y: ThoBatch<DO>, train: bool, best_valid_loss:f32) -> Self {
        Self {
            id: SynKey::new([mono_key!("train_step", "stage")]),
            x,
            lengths,
            expected_y,
            train,
            best_valid_loss
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
            .field("id", &self.id)
            .field("x", &self.x)
            .field("expected_y", &self.expected_y)
            .field("best_valid_loss", &self.best_valid_loss)
            .field("train", &self.train)
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
            best_valid_loss: self.best_valid_loss,
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
        Self {
            loss
        }
    }
}

impl WithId for TrainStep2Output {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Output for TrainStep2Output {}

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
        C: RunCtx + MaxParallelCtx,
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


#[cfg(test)]
mod tests {
    use crate::tasks::load_motif_net::{LoadMotifNet, MotifNetMeta};
    use crate::tasks::train_step::{TrainStep, TrainStepInput, TrainStepInputWrapper};
    use crate::tasks::train_step_2::APPLY_NET2_X_ENDPOINT;
    use crate::tasks::train_step_2::{TRAIN_TEST_STEP2_ENDPOINT, TrainStep2, TrainStep2Input, TrainStep2InputWrapper, TrainStep2Output, TrainStep2XFn};
    use async_trait::async_trait;
    use futures_util::FutureExt;
    use futures_util::future::BoxFuture;
    use futures_util::stream::BoxStream;
    use futures_util::{StreamExt, stream};
    use log::debug;
    use mockall::PredicateBoxExt;
    use parking_lot::{Mutex, RwLock};
    use rand::random;
    use serde::{Deserialize, Serialize, Serializer};
    use std::cell::RefCell;
    use std::collections::{HashMap, HashSet};
    use std::fmt::{Debug, Formatter};
    use std::fs;
    use std::future::ready;
    use std::marker::PhantomData;
    use std::path::{Path, PathBuf};
    use std::rc::Rc;
    use std::sync::Arc;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize};
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tch::nn::{Adam, AdamW, OptimizerConfig, VarStore};
    use tch::{Device, Kind, Reduction, Tensor, nn};
    use thoosa_common::batches::{AsyncDataBatch, BatchLengths, mock_async_data_batch};
    use thoosa_common::builder::{AsNode, Node, SeqBuilder};
    use thoosa_common::copy_ctx::CudaActorHandle;
    use thoosa_common::dims::{AnyDims, AnyMetrics, Batch, Dims, MotifProbability, Slice, ThoBatch};
    use thoosa_common::ipc::codec::{IpcCodec, JsonIpcCodec};
    use thoosa_common::ipc::cuda_wokrer::{CudaWorker, InitFn, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
    use thoosa_common::ipc::json_ser::SyncStageIpcWrapper;
    use thoosa_common::ipc::tests::run_ipc_test;
    use thoosa_common::ipc::wraps::{DynSyncStageWithStateEnum, ProxyStage, WorkerWrapperWithState};
    use thoosa_common::my_tch::{LossFp, ThoModule};
    use thoosa_common::runner::common::MockRunCtx;
    use thoosa_common::runner::common::cachable::{BorrowRef, DevicesWrap, ThoDataRef};
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
    use thoosa_common::runner::purge::FuturesRunnerMapPurger;
    use thoosa_common::runner::run_ctx_2::RunCtxImpl;
    use thoosa_common::runs::SavableResult;
    use thoosa_common::tho_device::ThoDevice;
    use thoosa_common::tho_tensor::{ThoTensor, ThoTensorInner};
    use thoosa_common::tho_varstore::{ModConfig, ThoTrainModule};
    use thoosa_common::train_nets::{FinalNetParams, MotifNetParams, ThoModuleWithStore, TrainMotifNetRunConfig};
    use thoosa_common::ws::{FsWorkspace, MockWorkspace, Workspace, WsConfigs, mock_ws_simp, net_files_from_base};
    use thoosa_properties_map::ToHashmap;
    use thoosa_properties_map_derive::ThoosaPropertiesMap;
    use thoosa_run_config_args_derive::ThoosaRunConfig;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::copy_mode::CpuOrGpu;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::run_configs::{RUN_ID, RunConfig, RunConfigId};
    use thoosa_shared::tmp_file::TmpFile;
    use thoosa_shared::tracked_await::track_await;
    use thoosa_shared::traits::ErasedStageWithState;
    use thoosa_shared::traits::{AsyncStage, CachableIdTr, Data, FromJsonFile, MaxParallelCtx, MonoMetrics, RunCtx, SimpleId, Stage, StageWithState, SyncStage, ToJsonFile, WithId};
    use thoosa_shared::{CopyToDevice, MoveToDevice, ThoError, cdt_for_dir, mono_key, now_ms, panic_to_err, simple_id};
    use thoosa_store::{MockStoreBuilder, SqlStore, Store};
    use tokio::sync::MutexGuard;
    use tokio::task::coop::cooperative;
    use tracing::field::debug;

    ////
    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct InDimsX {}
    impl Dims for InDimsX { type Metrics = MonoMetrics<0, 1>; }

    #[derive(Debug, Serialize, Deserialize, Clone)]
    pub struct OutDimsX {}
    impl Dims for OutDimsX { type Metrics = MonoMetrics<0, 1>; }

    #[derive(Debug, Clone, Serialize, Deserialize)]
    struct TestNetXParams {
        in_features:u32,
        layers:usize
    }

    impl AsNode for TestNetXParams {
        fn as_node(&self, init_data: Self::InitData) -> Result<Node, ThoError> {
            test_net(&self)
        }
    }

    impl FromJsonFile for TestNetXParams {
        fn from_file(path: &PathBuf) -> Result<Self, ThoError>
        where
            Self: Sized
        {
            serde_json5::from_str::<TestNetXParams>(fs::read_to_string(&path)?.as_str()).map_err(
                |serde_err| ThoError::from_error(format!("read net params from {path:?}"), serde_err),
            )
        }
    }

    impl ToJsonFile for TestNetXParams{
        fn to_file(&self, path: &PathBuf) -> Result<(), ThoError>
        where
            Self: Sized
        {
            fs::write(
                path,
                serde_json5::to_string(self)?,
            )?;

            Ok(())
        }
    }

    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Debug, Clone, Serialize, Deserialize)]
    #[run_config_grid]
    struct TestXRunConfig {
        #[in_map]
        pub id: Option<RunConfigId>,


        #[in_map]
        pub lr:f64,
        #[in_map]
        pub in_features:u32,
        #[in_map]
        pub layers:usize
    }

    impl TestXRunConfig {
        pub fn new(id: RunConfigId, lr: f64, in_features: u32, layers: usize) -> Self {
            Self {id:Some(id), lr, in_features, layers }
        }
    }

    #[derive(Debug)]
    struct TrainStepXFactory {//X for Motifs, Label, Final in TrainMotifNet, TrainLabelNets, TrainFinalNet
    }

    trait TrainStepFactory {
        type NP:FromJsonFile+ToJsonFile + Debug +Sync  + Send;
        type R:RunConfig + Debug + Sync + Send + Clone;
        type InDims:Dims + Serialize + serde::de::DeserializeOwned + 'static;
        type OutDims:Dims + Serialize + serde::de::DeserializeOwned + 'static;

        type C:RunCtx + 'static;



        fn new(run_config:Self::R, ctx:Self::C, vs_path: PathBuf) -> impl Future<Output = Result<(DynSyncStageWithStateEnum<TrainStep2<Box<dyn Fn(ThoBatch<Self::InDims>)-> ThoBatch<Self::InDims>+ Send + Sync + 'static>, Self::InDims, Self::OutDims, Self::C>>, Self::NP), ThoError>> + Send;
    }

    impl TrainStepFactory for TrainStepXFactory {
        type NP = TestNetXParams;
        type R = TestXRunConfig;
        type InDims = InDimsX;
        type OutDims = OutDimsX;
        type C = RunCtxImpl;

        fn new(run_config: Self::R, ctx:Self::C, vs_path: PathBuf) -> impl Future<Output = Result<(DynSyncStageWithStateEnum<TrainStep2<Box<dyn Fn(ThoBatch<Self::InDims>)-> ThoBatch<Self::InDims>+ Send + Sync + 'static>, Self::InDims, Self::OutDims, Self::C>>, Self::NP), ThoError>> + Send{

            async move {
                if GlobalCtx::global().is_ipc() {
                    let init = TrainTestStep2Init {
                        tho_device_idx: ctx.tho_device_idx(),
                        run_config: run_config.clone(),
                        vs_path,
                    };
                    // Use the shared stateful proxy path so client dispatch and worker state handling stay in thoosa_common.
                    thoosa_common::ipc::codec::install_ipc_endpoint::<JsonIpcCodec, TrainTestStep2Init>(GlobalCtx::global().cuda_actor(ctx.tho_device_idx()).as_ref(), TRAIN_TEST_STEP2_ENDPOINT, &init).await?;
                    Ok((
                        DynSyncStageWithStateEnum::Proxy {
                            stage: Arc::new(ProxyStage::proxy(TRAIN_TEST_STEP2_ENDPOINT)),
                        },
                        TestNetXParams { in_features: run_config.in_features, layers: run_config.layers }
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
                        vs_path,
                    ).await?;
                    Ok((DynSyncStageWithStateEnum::Local(Arc::new(stage)), params))

                }
            }
        }
    }

    #[derive(Debug)]
    pub struct TrainNetResult<P, R>
    where 
        P:ToJsonFile + Debug + Send + Sync +'static,
        R:RunConfig + Debug + Send + Sync +'static,
    {
        pub loss: f32,

        pub net_var_store: NetVarStorePath<P>,
        run_config: R,
    }

    impl <P, R> Output for TrainNetResult<P,R>
    where
        P:ToJsonFile + Debug + Send + Sync +'static,
        R:RunConfig + Debug + Send + Sync +'static,
    {}

    impl <P, R> TrainNetResult<P, R>
    where
        P:ToJsonFile+Debug + Send + Sync +'static,
        R:RunConfig + Debug + Send + Sync +'static,

    {
        pub fn new(loss: f32, net_var_store: NetVarStorePath<P>, run_config: R) -> Self {
            Self { loss, net_var_store, run_config }
        }
    }

    #[async_trait]
    impl <P, R> SavableResult for TrainNetResult<P, R>
    where
        P:ToJsonFile + Debug + Send + Sync +'static,
        R:RunConfig + Debug + Send + Sync +'static,
    {
        type R = u32;
        async fn save(&mut self, store: &Arc<Box<dyn Store>>, ws: &Arc<RwLock<Box<dyn Workspace>>>, p: Self::P<'_>) -> Result<Self::R, ThoError> {
            let run_config_id = self.run_config.get_id();

            //store.save_run_config skipped here, should be preserved in main app
            let path = panic_to_err!(ws.read().save_net(//save_final_net and FinalNetParams are used here to avoid adding save_X_net to production code
                    run_config_id,
                (&self.net_var_store.path.path().to_path_buf()).into(),
                R::type_name(),
                &self.net_var_store.net_params, //self.params,
            ))?;

            debug!("TrainNetResult::save: run_config_id={run_config_id}, path={path:?} self.loss={}",  self.loss);


            track_await(store.save_trained_net(run_config_id, path, self.loss, None)).await
        }
    }


    pub async fn opt_train_2<R, F, N>(//corresponds to opt_train crates/thoosa_common/src/train_nets.rs:2033
        run_config_source:Arc<Box<dyn Fn()->R + Send + Sync>>,//opt_train functionality should remain as is, the single run config here is for simplicity
        mut train: F
    ) ->Result<(TmpFile, N, R, f32), ThoError>
    where
        R:Clone,
        F:AsyncFnMut(R)->Result<(Option<TmpFile>, f32, N), ThoError>
    {

        //runs few times getting run configurations from run_config_source and searching for optimal run_config
        let run_config = run_config_source();
        let (vs_path, loss, params) = train( //corresponds to thoosa_common/src/train_nets.rs:2095
            run_config.clone(),

        ).await?;


        vs_path.map(|vs_path|(vs_path, params, run_config, loss)).ok_or(ThoError::new("no min_var_store"))
    }


    struct TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        SF::C: MaxParallelCtx,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {
        pd:PhantomData<SF>,
        run_config_source:Arc<Box<dyn Fn()->SF::R + Send + Sync>>,
        batch:B
    }

    impl <SF, B> Debug for TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        SF::C: MaxParallelCtx,
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



    #[derive(Debug)]
    struct NetVarStorePath<P>
    where
    P:ToJsonFile + Debug
    {
        path:TmpFile,
        net_params:P
    }

    impl <P> NetVarStorePath<P>
    where
        P:ToJsonFile+Debug
    {
        pub fn new(path: TmpFile, net_params: P) -> Self {
            Self { path, net_params }
        }
    }


    impl <SF, B> Stage for TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        SF::C: MaxParallelCtx,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {
        // Use the factory context here so the stage, train step, and batch stream share one context type.
        type Ctx = SF::C;
        type In<'a> = TrainNet2Input;
        type Out = TrainNetResult<SF::NP, SF::R>;
    }

    //-->for test only to avoid errors because of run_config_source which is not Sync and Send
    unsafe impl <SF, B> Sync for TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        SF::C: MaxParallelCtx,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {}
    unsafe impl <SF, B> Send for TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        SF::C: MaxParallelCtx,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {}//<--

    impl <SF, B> AsyncStage for TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        SF::C: MaxParallelCtx,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,

    {
        fn run(self: Arc<Self>, input: <Self as Stage>::In<'static>, mut ctx: <Self as Stage>::Ctx) -> impl Future<Output=Result<<Self as Stage>::Out, ThoError>> + Send {
            let run_config_source = Arc::clone(&self.run_config_source);
            let self_for_train = Arc::clone(&self);
            async move {

                let (var_store_path, node_params, run_config, loss) = opt_train_2(
                    run_config_source,
                    move |run_config| {
                        let self_arc = Arc::clone(&self_for_train);

                        self_arc.run_inner(
                            run_config,
                            ctx.branch(false),
                        )
                    }).await?;


                // Ok(TrainNet2Output::new(NetVarStorePath::new(var_store_path, node_params))) - in the original application TrainMotifNetResult is built from TrainNet output in the model-specific structs like thoosa_main/src/tasks/train_motif_nets.rs:975. In the refactored app TrainNet output can implement SavableResult and be used directly
                Ok(TrainNetResult::new(loss, NetVarStorePath::new(var_store_path, node_params), run_config))//run config on which the result was achieved

            }
        }
    }

    impl <SF, B> TrainNet2<SF, B>
    where
        SF:TrainStepFactory + Debug + Send + Sync  +'static,
        SF::C: MaxParallelCtx,
        B:AsyncDataBatch<SF::InDims, SF::OutDims, SF::C>+ Debug + 'static,
    {

        fn run_inner(self:Arc<Self>, run_config:SF::R, mut ctx: <Self as Stage>::Ctx)->BoxFuture<'static, Result<(Option<TmpFile>, f32, SF::NP), ThoError>>{
            debug!("TrainNet2::run_inner: run_config={run_config:?}");

            let self_arc = Arc::clone(&self);
            let  ctx_for_run_stage = ctx.branch(false);

            Box::pin(async move {
                std::fs::create_dir_all("/tmp/thoosa")?;
                let vs_path = PathBuf::from("/tmp/thoosa").join(format!("thoosa_train_net_best_{}_{}.safetensors", now_ms(), random::<u64>()));


                let (train_step, net_params) = SF::new(run_config, ctx.branch(false), vs_path.clone()).await?;

                let train_step= Arc::new(train_step);
                let mut best_valid_loss = f32::MAX;

                for epoch in 0..3 {
                    let self_arc = Arc::clone(&self_arc);
                    let mut batches = self_arc.batch.train_batch_stream(ctx.branch(false));


                    while let Some(Ok((x_t_0, lengths, expected_y))) = cooperative(batches.next()).await {
                        let test_step_input = TrainStep2Input::new(x_t_0, lengths, expected_y, true, best_valid_loss);

                        debug!("TrainNet2::run_inner: epoch={epoch} test_step_input={test_step_input:?}");

                        // TrainStep2 owns mutable module state, so use its stateful execution contract.
                        let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()> = OOMHandler::new(vec![], None);
                        let r = track_await(oom_handler.run(&Arc::clone(&train_step), Arc::new(test_step_input), ctx.branch(false), NoContexts)).await?;

                        debug!("TrainNet2::run_inner: epoch={epoch} r={r:?}");

                        let TrainStep2Output { loss } = r;


                        if loss < best_valid_loss {
                            best_valid_loss = best_valid_loss;
                            debug!("TrainNet2::run_inner: epoch={epoch} best_valid_loss={best_valid_loss:?}");
                        }
                    }

                }

                debug!("TrainNet2::run_inner: best_valid_loss={best_valid_loss:?}");

                Ok((Some(TmpFile::new(vs_path)),best_valid_loss, net_params))
            })
        }

        pub fn new(batch: B, run_config_source:Box<dyn Fn()->SF::R + Send + Sync>) -> Self {//here run_config_source is a closure producing the only possible run_config, in the real app run_config_source may produce many different run configs
            Self {
                batch,
                run_config_source: Arc::new(run_config_source),
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
        tho_device_idx:usize,
        run_config: TestXRunConfig,
        vs_path:PathBuf
    }

    impl WorkerStageInput for TrainTestStep2Init {
        fn decode_for_worker<C: IpcCodec>(input: &[u8], _all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
            C::decode(input)
        }
    }

    impl<DI, DO> WorkerStageInput for TrainStep2Input<DI, DO>
    where
        DI: Dims,
        DO: Dims,
        Self: serde::de::DeserializeOwned,
    {
        fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
            let input = C::decode::<Self>(input)?;
            // Worker wrappers resolve tensor references before the sync stage receives its borrowed input.
            Ok(TrainStep2Input::new(
                ThoTensor::ref_to_tensor(all_cuda_tensors, input.x)?,
                input.lengths,
                ThoTensor::ref_to_tensor(all_cuda_tensors, input.expected_y)?,
                input.train,
                input.best_valid_loss,
            ))
        }
    }

    impl WorkerStageOutput for TrainStep2Output {
        fn encode_from_worker<C: IpcCodec>(self, _all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
            C::encode(&self)
        }
    }

    fn train_test_step_2_endpoint_init(init_msg: Vec<u8>, _ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError> {
        let init = TrainTestStep2Init::decode_for_worker::<JsonIpcCodec>(&init_msg, all_cuda_tensors)?;
        debug!("train_test_step_2_endpoint_init: init={init:?}");

        let (train_step, _params): (TrainStep2<TrainStep2XFn<InDimsX>, InDimsX, OutDimsX, WokerRunCtx>, TestNetXParams) = futures::executor::block_on(TrainStep2::new_local(
                init.tho_device_idx.into(), //TODO assert it matches actual worker's device or pass the latter into init function
                |rc, device|  {
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
                Arc::new(Box::new(|t|{
                    t
                }) as Box<dyn Fn(ThoBatch<InDimsX>) -> ThoBatch<InDimsX> + Send + Sync + 'static>),
                |pred_y: &Tensor, target: &Tensor|{
                    Ok(pred_y.f_binary_cross_entropy_with_logits::<Tensor>(&target.f_totype(Float)?, None, None, Reduction::Mean)?)
                },
                &init.run_config,
                init.vs_path,
            ))?;

        // The common wrapper owns state preparation and encoded worker dispatch for this existing stage.
        Ok(Box::new(WorkerWrapperWithState::new(Arc::new(train_step), PhantomData::<JsonIpcCodec>)))
    }
    ///////////////////////
    pub struct ApplyNetStage2<DI, DO, C>
    where
        DI:Dims + 'static,
        DO:Dims + 'static,
        C:RunCtx
    {
        device: ThoDevice,
        train_module: tokio::sync::Mutex<ThoModuleWithStore<Batch<DI>, Batch<DO>>>,

        pd: PhantomData<C>,
    }

    //for FuturesRunnerMapPurger
    unsafe impl <DI, DO, C> Send for ApplyNetStage2<DI, DO, C>
    where
        DI:Dims,
        DO:Dims,
        C:RunCtx
    {}

    unsafe impl <DI, DO, C> Sync for ApplyNetStage2<DI, DO, C>
    where
        DI:Dims,
        DO:Dims,
        C:RunCtx
    {}

    #[async_trait]
    impl <DI, DO, C> FuturesRunnerMapPurger for ApplyNetStage2<DI, DO, C>
    where
        DI:Dims,
        DO:Dims,
        C:RunCtx,
        Self:Send + Sync
    {
        async fn has_gpu_data(&self, device_idx: usize) -> bool {
            self.train_module.lock().await.device() == device_idx
        }

        async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
            debug!("ApplyNetStep2::purge_gpu");
            let mut mutex_guard = self.train_module.lock().await;
            *mutex_guard = mutex_guard.copy_to_device(0).await?;

            Ok(())
        }

        //it's called when local GPU reset is requested
        async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
            debug!("TrainStep2::drop_gpu");
            todo!()
        }

        async fn purge_cpu(&self, purge_until: u64) -> usize {
            debug!("TrainStep2::purge_cpu");
            0
        }
    }


    impl<DI: Dims, DO: Dims, C: RunCtx> Debug for ApplyNetStage2<DI, DO, C> {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ApplyNetStep2").finish()
        }
    }


    impl<DI, DO, C> ApplyNetStage2<DI, DO, C>
    where
        DI: Dims,
        DO: Dims,
        C: RunCtx,
    {

        pub(crate) async fn new_local<NP, IDT>(device: ThoDevice, base_path:PathBuf, init_data:IDT) -> Result<Self, ThoError>
        where
            NP: FromJsonFile+AsNode<InitData=IDT> + Send + Sync +'static,
            IDT: Clone + Send + Sync  +'static,
        {
            let (net_path, params_path) = net_files_from_base(&base_path);

            let params = NP::from_file(&params_path)?;
            let train_module = ThoModuleWithStore::load(
                mono_key!(0/*TODO static count?*/, net_path.to_string_lossy().to_string()).into(),
                move |path: &nn::Path| params.as_node(init_data.clone())?.build_tch_module(path),//needs to be cloneable because ThoModuleWithStore re-uses it when reloading
                net_path,
                *device,
                false,
            )?;

            Ok(Self {
                device,
                train_module: tokio::sync::Mutex::new(train_module),

                pd: PhantomData,
            })
        }

    }

    #[derive(Debug, Serialize, Deserialize)]
    pub struct ApplyNetStage2Input<DI>
    where
        DI:Dims,
    {
        id: SynKey,
        pub x: ThoBatch<DI>,
        pub lengths: Option<BatchLengths>,//real lengths without padding
    }

    impl<DI: Dims> Data for ApplyNetStage2Input<DI> {
        type Metrics = <ThoBatch<DI> as Data>::Metrics;
        type M = CpuOrGpu<Self>;

        fn size(&self) -> Self::Metrics {
            <ThoBatch<DI> as Data>::size(&self.x)
        }

        fn device(&self) -> usize {
            self.x.tho_device_idx()
        }

        fn mem_size(&self) -> u64 {
            self.x.mem_size()
        }
    }


    impl <DI> Input for ApplyNetStage2Input<DI>
    where
        DI:Dims
    {}

    impl <DI> WithId for ApplyNetStage2Input<DI>
    where
        DI:Dims,
    {
        type ID = SynKey;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl WorkerStageInput for ApplyNetStage2Input<InDimsX> {
        fn decode_for_worker<C: IpcCodec>(input: &[u8], all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
            let input = C::decode::<Self>(input)?;
            // Resolve the existing tensor reference at the shared worker-wrapper boundary.
            Ok(Self {
                id: input.id,
                x: ThoTensor::ref_to_tensor(all_cuda_tensors, input.x)?,
                lengths: input.lengths,
            })
        }
    }

    #[async_trait]
    impl<DI: Dims> CopyToDevice for ApplyNetStage2Input<DI> {
        async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError>
        where
            Self: Sized,
        {
            Ok(ApplyNetStage2Input {
                id: self.id.clone(),
                x: track_await(self.x.copy_to_device(dst_device)).await?,
                lengths: self.lengths.clone(),
            })
        }
    }


    impl <DI> BorrowRef for ApplyNetStage2Input<DI>
    where
        DI:Dims
    {
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

    #[derive(Debug, Serialize, Deserialize)]
    #[serde(bound(serialize = ""))]
    pub struct ApplyNetStage2Output<DO>
    where
        DO:Dims,
    {
        id: SynKey,
        pub y: ThoBatch<DO>,
        pub lengths: Option<BatchLengths>,
    }

    impl <DI> Output for ApplyNetStage2Output<DI>
    where
        DI:Dims
    {}


    impl <DO> WithId for ApplyNetStage2Output<DO>
    where
        DO:Dims,
    {
        type ID = SynKey;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }


    impl <DO> WorkerStageOutput for ApplyNetStage2Output<DO>
    where
        DO:Dims,
    {
        fn encode_from_worker<C: IpcCodec>(self, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, tho_device: &ThoDevice) -> Result<Vec<u8>, ThoError> {
            let output = ApplyNetStage2Output {
                id: self.id,
                y: self.y.worker_tensor_to_ref(all_cuda_tensors, tho_device)?,
                lengths: self.lengths,
            };
            C::encode(&output)
        }
    }


    impl <DI, DO, C> Stage for ApplyNetStage2<DI, DO, C>
    where
        DI: Dims,
        DO: Dims,
        C: RunCtx,
    {
        type Ctx = C;
        type In<'a> = ApplyNetStage2Input<DI>;
        type Out = ApplyNetStage2Output<DO>;
    }

    impl<DI, DO, C> SyncStage for ApplyNetStage2<DI, DO, C>
    where
        DI: Dims,
        DO: Dims,
        C: RunCtx,
    {
        fn as_with_state(&self) -> Option<&dyn ErasedStageWithState<Self>> {
            // Advertise typed state execution without exposing ThoTrainModule to the common runner.
            Some(self)
        }
    }

    impl<DI, DO, C> StageWithState for ApplyNetStage2<DI, DO, C>
    where
        DI: Dims,
        DO: Dims,
        C: RunCtx,
    {
        type S = ThoModuleWithStore<Batch<DI>, Batch<DO>>;

        fn pre_run(&self, to_device_idx: usize) -> BoxFuture<'_, Result<MutexGuard<'_, Self::S>, ThoError>> {
            async {
                let mut guard = self.train_module.lock().await;
                if guard.device() != 1 { //TODO pass device to pre_run
                    *guard = guard.copy_to_device(1).await?;
                }
                Ok(guard)
            }.boxed()
        }

        #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "ApplyNetStep2::do_run_with_state"))]
        fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, mut train_module: MutexGuard<'_, Self::S>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
            let y = train_module.tho_forward(&input.x, None, false)?;


            Ok(ApplyNetStage2Output {
                id: mono_key!(0, "todo").into(),//TODO
                y,
                lengths: None,
            })
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct ApplyNetStage2Init {
        id: SynKey,
        tho_device_idx: usize,
        base_path:PathBuf
    }

    impl WorkerStageInput for ApplyNetStage2Init {
        fn decode_for_worker<C: IpcCodec>(input: &[u8], _all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Self, ThoError> {
            C::decode(input)
        }
    }

    async fn new_apply_net_x_stage(id: SynKey, device: ThoDevice, base_path: PathBuf, init_data: ()) -> Result<DynSyncStageWithStateEnum<ApplyNetStage2<InDimsX, OutDimsX, RunCtxImpl>>, ThoError> {
        if GlobalCtx::global().is_ipc() {
            let init = ApplyNetStage2Init {
                id,
                tho_device_idx: device.tho_device_idx(),
                base_path,
            };
            debug!("new_apply_net_x_stage: init={init:?}");
            // Install the endpoint before returning the shared proxy stage used by the common dispatch enum.
            thoosa_common::ipc::codec::install_ipc_endpoint::<JsonIpcCodec, ApplyNetStage2Init>(GlobalCtx::global().cuda_actor(device.tho_device_idx()).as_ref(), APPLY_NET2_X_ENDPOINT, &init).await?;
            Ok(DynSyncStageWithStateEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy(APPLY_NET2_X_ENDPOINT)),
            })
        } else {
            Ok(DynSyncStageWithStateEnum::Local(Arc::new(ApplyNetStage2::new_local::<TestNetXParams, ()>(device, base_path, init_data).await?)))
        }
    }

    fn apply_net_stage_2_endpoint_init(init_msg: Vec<u8>, _ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>) -> Result<Box<dyn SyncStageIpcWrapper>, ThoError> {
        let init = ApplyNetStage2Init::decode_for_worker::<JsonIpcCodec>(&init_msg, all_cuda_tensors)?;
        debug!("apply_net_stage_2_endpoint_init: init={init:?}");
        let stage: ApplyNetStage2<InDimsX, OutDimsX, WokerRunCtx> = futures::executor::block_on(ApplyNetStage2::new_local::<TestNetXParams, ()>(init.tho_device_idx.into(), init.base_path, ()))?;

        // The common stateful wrapper performs state preparation and converts worker outputs back to references.
        Ok(Box::new(WorkerWrapperWithState::new(Arc::new(stage), PhantomData::<JsonIpcCodec>)))
    }

    /////////////////

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

            let run_config = TestXRunConfig::new(0, 0.01, 100, 3);
            let x_params = TestNetXParams { in_features: run_config.in_features, layers: run_config.layers };
            let init = TrainTestStep2Init { tho_device_idx, run_config, vs_path: PathBuf::new() };
            thoosa_common::ipc::codec::install_ipc_endpoint::<JsonIpcCodec, TrainTestStep2Init>(cuda_actor.as_ref(), TRAIN_TEST_STEP2_ENDPOINT, &init).await?;
            let (train_step, params): (DynSyncStageWithStateEnum<TrainStep2<TrainStep2XFn<InDimsX>, InDimsX, OutDimsX, RunCtxImpl>>, TestNetXParams) = (
                DynSyncStageWithStateEnum::Proxy {
                    stage: Arc::new(ProxyStage::proxy(TRAIN_TEST_STEP2_ENDPOINT)),
                },
                x_params
            );

            do_test_train_step_2(train_step, async ||Ok(TrainStep2Input::<InDimsX, OutDimsX>::new(
                to_tho_tensor(cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[3, 100], FLOAT_CPU))?, 1.into()).await?.inner, "2".into()),
                None,
                to_tho_tensor(cuda_actor.copy(ThoTensor::new((), tch::Tensor::rand(&[3, 1], FLOAT_CPU))?, 1.into()).await?.inner, "3".into()),
                false,
                0.1
            ))).await?;


            Ok(())
        }, async |sock_path| {
            let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
            CudaWorker::run_worker(sock_path, &ws,
                                   HashMap::from([
                    (
                        TRAIN_TEST_STEP2_ENDPOINT.to_string(),
                        Box::new(train_test_step_2_endpoint_init) as InitFn,
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

        let (train_step, np) = TrainStepXFactory::new(TestXRunConfig::new(0,0.001, 100, 3), RunCtxImpl::new(1, "test"), PathBuf::new() ).await?;


        do_test_train_step_2(train_step, || ready(Ok(TrainStep2Input::<InDimsX, OutDimsX>::new(
            tch::Tensor::rand(&[3, 100], FLOAT_CPU).into(),
            None,
            tch::Tensor::rand(&[3, 1], FLOAT_CPU).into(),
            false,
            0.1
        )))).await?;

        Ok(())
    }

    async fn do_test_train_step_2<FI>(train_step: DynSyncStageWithStateEnum<TrainStep2<TrainStep2XFn<InDimsX>, InDimsX, OutDimsX, RunCtxImpl>>, input: FI) -> Result<(), ThoError>
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
        // let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1000)));
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, WsConfigs::new(2, 123, 123))?));
        do_test_train_net_2(false, &ws).await?;

        Ok(())

    }

    async fn do_test_train_net_2(make_refs:bool, ws:&Arc<RwLock<Box<dyn Workspace>>>) -> Result<(), ThoError> {
        let store = Arc::new(SqlStore::in_mem().await?);
        RUN_ID.set(store.save_run("test".into()).await?).expect("can't set run id");


        let train_stream_calls = Arc::new(AtomicUsize::new(0));
        let valid_stream_calls = Arc::new(AtomicUsize::new(0));

        let mock_test_batch = Arc::new(
            mock_async_data_batch::build::<InDimsX, OutDimsX, RunCtxImpl>()
                .with_train_stream_calls(Arc::clone(&train_stream_calls), move |_ctx| make_stream(make_refs))
                .with_valid_stream_calls(Arc::clone(&valid_stream_calls), move |_ctx| make_stream(make_refs))
                .finish(),
        );


        let mut run_config = TestXRunConfig::new(0, 0.001, 100, 3);
        let run_config_id = store.save_run_config(None, "TestXRunConfig", "test", run_config.to_properties_iter(None).collect()).await?;
        run_config.set_id(run_config_id);
        debug!("do_test_train_net_2: run_config_id={run_config_id:?}");

        let train_net = Arc::new(TrainNet2::<TrainStepXFactory, _>::new(mock_test_batch, Box::new(move || {
            run_config.clone()
        })));

        let mut ctx = RunCtxImpl::new(1, "test");
        let mut r = train_net.run(TrainNet2Input {}, ctx.branch(false)).await?;
        let saved = r.save(&store, ws, ()).await?;
        debug!("do_test_train_net_2: r={r:?}");

        // let rr = TrainNetXResult::new()
        let x_nets = track_await(store.list_trained_nets(vec![run_config_id], None, None)).await?;
        debug!("do_test_train_net_2: x_nets={x_nets:?}");
        for x_net in x_nets {

            let apply_net = Arc::new(new_apply_net_x_stage(mono_key!(0, "").into(), 1.into(), x_net.net_path, ()).await?);
            debug!("do_test_train_net_2: apply_net={apply_net:?}");
            let input = make_input_batch(make_refs).await?;
            let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ()> = OOMHandler::new(vec![], None);
            let apply_result = track_await(oom_handler.run(&Arc::clone(&apply_net), Arc::new(input), ctx.branch(false).branch(false), NoContexts)).await?;
            debug!("do_test_train_net_2: apply_result={apply_result:?}");
        }

        Ok(())
    }

    async fn make_input_batch(make_refs: bool) -> Result<ApplyNetStage2Input<InDimsX>, ThoError> {
        if make_refs {
            let cuda_actor = GlobalCtx::global().cuda_actor(1);
            let x = create_worker_tensor(&cuda_actor, &[1, 1, 100], FLOAT_CPU).await?;
            let key_x = extract_key(&x);
            debug!("make_input_batch: key_x={key_x}");
            let input: ApplyNetStage2Input<InDimsX> = ApplyNetStage2Input {
                id: mono_key!(0, "").into(),
                x: ThoTensor::new_ref(simple_id!(format!("x_{}", key_x)), key_x, vec![1, 1, 100], Float, None, 1),
                lengths: None,
            };
            Ok(input)
        } else {
            let input: ApplyNetStage2Input<InDimsX> = ApplyNetStage2Input {
                id: mono_key!(0, "").into(),
                x: ThoTensor::new(simple_id!(format!("x_{}", 0)), tch::Tensor::rand(&[1, 1, 100], FLOAT_CPU).f_contiguous()?)?,
                lengths: None,
            };
            Ok(input)
        }
    }

    #[test]
    fn test_train_net_2_ipc()->Result<(), ThoError>{
        let test_dir = Path::new("/tmp").join(cdt_for_dir());
        let ws = Arc::new(RwLock::new(FsWorkspace::new(test_dir, WsConfigs::new(2, 123, 123))?));
        run_ipc_test(async {
            let module = module_path!().split_once("::").map(|(_, module)| module).unwrap_or("");
            let test_name = format!("{module}::test_train_net_2_ipc");
            GlobalCtx::init_devices_ipc(2, vec![test_name, "--exact".to_string(), "--nocapture".to_string()]).await?;
            let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, true);
            let cuda_actor = GlobalCtx::global().cuda_actor(1);

            do_test_train_net_2(true, &ws).await?;

            Ok(())
        }, async |sock_path| {

            CudaWorker::run_worker(
                sock_path,
                &ws,
                HashMap::from([
                    (
                        TRAIN_TEST_STEP2_ENDPOINT.to_string(),
                        Box::new(train_test_step_2_endpoint_init) as InitFn,
                    ),
                    (
                        APPLY_NET2_X_ENDPOINT.to_string(),
                        Box::new(apply_net_stage_2_endpoint_init) as InitFn,
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
                let x = create_worker_tensor(&cuda_actor, &[1, 1, 100], FLOAT_CPU).await?;
                let y = create_worker_tensor(&cuda_actor, &[1, 1, 1], FLOAT_CPU).await?;

                let key_x = extract_key(&x);
                let key_y = extract_key(&y);

                Ok::<_, ThoError>((
                    ThoTensor::new_ref(simple_id!(format!("x_{}", key_x)), key_x, vec![1, 1, 100], Float, None, 1),
                    ThoTensor::new_ref(simple_id!(format!("y_{}", key_y)), key_y, vec![1, 1, 1], Float, None, 1),
                ))
            }).map_err(Into::into)

        } else {
            let x = Tensor::zeros([1, 1, 100], FLOAT_CUDA);
            let y = Tensor::zeros([1, 1, 1], FLOAT_CUDA);
            Ok((x.into(), y.into()))
        }
    }

    async fn create_worker_tensor(cuda_actor: &Box<dyn CudaActorHandle>, shape: &[i64], kind: (tch::Kind, Device)) -> Result<ThoTensor, ThoError> {
        let tensor = tch::Tensor::rand(shape, kind).f_contiguous()?;
        cuda_actor.copy(ThoTensor::new((), tensor)?, 1.into()).await
    }

    fn extract_key(tho_tensor: &ThoTensor) -> u64 {
        match &tho_tensor.inner {
            ThoTensorInner::Ref { key, .. } => *key,
            _ => panic!("should be ref"),
        }
    }

    fn make_stream(make_refs:bool) -> BoxStream<'static, Result<(ThoBatch<InDimsX>, Option<BatchLengths>, ThoBatch<OutDimsX>), ThoError>> {
        let (x, y) = dummy_batch(make_refs).unwrap();
        stream::iter(vec![Ok((x, None, y))]).boxed()
    }
    
    
}
