use thoosa_common::dims::ChannelsData;
use async_trait::async_trait;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MaxParallelAndDumpToCpuAdjustableContex, MAX_PARALLEL};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, ThoDataRef};
use thoosa_common::runner::common::runnable::{AsyncStage, DynSyncStageBox, Stage, SyncStage};
use thoosa_common::runner::common::{MaxParallelCtx, OutId, RunCtx};
use thoosa_common::runner::global_ctx::GlobalCtx;

use thoosa_common::runner::oom_handler::{NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, run_ctx_adjuster, Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler, WithRunCtxContext};
use thoosa_shared::{cdt_for_dir, mono_key, now_ms, os_thread_id, perf_log, sync_mutex_lock};
use futures_util::{future::BoxFuture, StreamExt};
use log::{debug, info, trace, warn};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::io::Write;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use rand::random;
use serde::{Deserialize, Serialize};
use tch::nn::{Optimizer, VarStore};
use tch::{nn, Device};
use tch::Kind::Float;
use tokio::task::coop::cooperative;
use tracing::Instrument;
use thoosa_common::parzen::optimizer::CompletedTrial;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::my_tch::{LossFp, ThoModule};
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{opt_train, CommonTrainingRunConfig, ConvNetParams, LabelNetTypeArgs, LabelNetTypeParams, LinNetArgs, OptConfig, TrainLabelNetRunConfig, TrainLabelNetRunConfigResult, TrainMotifNetRunConfig};
use thoosa_common::utils::{optimizer_mem_size, TrainListener};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::CopyToDevice;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{panic_to_err, ThoError};
use thoosa_shared::collect_futures_eta::{collect_eta_record_completion, CollectEtaInvocationGuard};
use thoosa_shared::comp_id::SynKey;
use thoosa_properties_map::ToHashmap;
use thoosa_shared::tmp_file::TmpFile;
use thoosa_shared::tracked_await::track_await;

pub(crate) type TrainStepXFn<D> = Box<dyn Fn(ThoBatch<D>) -> ThoBatch<D> + Send + Sync + 'static>;

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

#[async_trait]
pub(crate) trait TrainStepWeightsSync: Debug + Send + Sync {
    async fn into_var_store(&self) -> Result<VarStore, ThoError>;
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

fn maybe_cuda_mem(device: &Device) -> Option<(usize, usize)> {
    match device {
        Device::Cuda(_) if  Device::cuda_if_available().is_cuda() => Some(cuda_mem()),
        _ => None,
    }
}

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainNetInput{
    id: SynKey
}

impl TrainNetInput {
    pub fn new(id: SynKey) -> Self {
        Self { id }
    }
}
impl Input for TrainNetInput {

}

impl WithId for TrainNetInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainNetOutput<N, R>
where
    N:Debug + Send + Sync + 'static,
    R:Debug + Send + Sync + 'static,
{
    id: SynKey,
    pub var_store: VarStore,
    pub node_params:N,
    pub run_config:R,
    pub min_loss:f32
}

impl<N, R> TrainNetOutput<N, R>
where
    N:Debug + Send + Sync + 'static,
    R:Debug + Send + Sync + 'static,
{
    pub fn new(id: SynKey, var_store: VarStore, node_params: N, run_config: R, min_loss:f32) -> Self {
        Self { id, var_store, node_params, run_config, min_loss }
    }
}

impl<N, R> WithId for TrainNetOutput<N, R>
where
    N:Debug + Send + Sync + 'static,
    R:Debug + Send + Sync + 'static,
{
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<N, R> Output for TrainNetOutput<N, R>
where
    N:Debug + Send + Sync + 'static,
    R:Debug + Send + Sync + 'static,
{

}

pub struct TrainNet<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY, C> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    for<'a> NF:Fn(C, &'a RC,Arc<Mutex<VarStore>>)->BoxFuture<'a, Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError>> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    batch:BF,
    net:NF,
    opt_config:Arc<OptConfig>,
    run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >,
    loss_fn:LossFp,
    optimizer:FO,
    x_f:XF,
    epochs:FC,
    validation_step: u32,
    descr:String,
    reusable_completed_trials: Vec<CompletedTrial>,
    train_step_proxy_factory: Option<Arc<dyn TrainStepProxyFactory<DX, DY, C, NP, RC>>>,
    pd:PhantomData<(C, DX, DY, BF, P, RC)>
}

impl<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> Debug for TrainNet<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY, C> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    for<'a> NF:Fn(C, &'a RC,Arc<Mutex<VarStore>>)->BoxFuture<'a, Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError>> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainNet").finish()
    }
}

impl<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> TrainNet<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B: AsyncDataBatch<DX, DY, C> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NF: Fn(C, &RC, Arc<Mutex<VarStore>>) -> BoxFuture<Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError>> + Send + Sync + 'static,
    FO: Fn(&RC,&VarStore)->Result<Optimizer, ThoError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    pub fn new(batch: BF, net: NF, opt_config: Arc<OptConfig>, run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P>>, loss_fn: LossFp, optimizer: FO, x_f: XF, epochs: FC, validation_step: u32, descr: String) -> Self {
        Self { batch, net, pd: PhantomData, opt_config, run_config_source, loss_fn, optimizer, x_f, epochs, validation_step, descr, reusable_completed_trials: vec![], train_step_proxy_factory: None }
    }

    pub fn with_reusable_completed_trials(mut self, reusable_completed_trials: Vec<CompletedTrial>) -> Self {
        self.reusable_completed_trials = reusable_completed_trials;
        self
    }

    pub(crate) fn with_train_step_proxy_factory(mut self, train_step_proxy_factory: Arc<dyn TrainStepProxyFactory<DX, DY, C, NP, RC>>) -> Self {
        self.train_step_proxy_factory = Some(train_step_proxy_factory);
        self
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainNet::run_inner"))]
    fn run_inner(
        self: Arc<Self>,
        mut ctx: C,
        tho_device_idx: usize,
        run_config: RC,
        validation_step: u32,
        descr:String
    ) -> BoxFuture<'static, Result<(VarStore, f32, NP), ThoError>> {
        Box::pin(async move {
            debug!("TrainNet::run_inner: descr={descr} validation_step={validation_step}");

            let device = Device::from(ThoDevice::from(ctx.tho_device_idx()));
            let epochs = (self.epochs)(&run_config);

            let self_x_f = Arc::clone(&self);
            let x_f: Arc<TrainStepXFn<DX>> = Arc::new(Box::new(move |batch: ThoBatch<DX>| (self_x_f.x_f)(batch)));
            let (train_step, params) = track_await(TrainStep::new(
                ThoDevice::from(ctx.tho_device_idx()),
                |var_store| (self.net)(ctx.branch(false), &run_config, var_store),
                |run_config, var_store| (self.optimizer)(&run_config, var_store),
                x_f,
                self.loss_fn,
                self.train_step_proxy_factory.as_ref(),
                &run_config,
            )).await?;
            let train_step = Arc::new(train_step);
            GlobalCtx::global().register_cache(("train_step", Arc::clone(&train_step) as Arc<dyn FuturesRunnerMapPurger>));
            let dump_to_cpu_exclude: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
            let out_id = Arc::new(()) as Arc<dyn CachableIdTr>;

            let mut bad_epochs = 0u32;
            let patience = run_config.as_ref().patience;
            let min_delta = run_config.as_ref().min_delta;



            let batch = (self.batch)(ctx.branch(false), &run_config)
                .instrument(tracing::debug_span!("run_inner/self.batch", file = file!(),line = line!()))
                .await?;

            debug!("TrainNet::run_inner: descr={descr} device={device:?} epochs={epochs}");

            let oom_handler = OOMHandler::<'_, '_, NoInputAdjuster, NoTaskAdjuster, _, ()>::new(
                vec![
                    Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(
                        MAX_PARALLEL,
                        *MaxParallelStep::FIFTY,
                    ))),
                    Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("train_step"))),
                ],
                None,
            );

            //TODO configure in ws
            std::fs::create_dir_all("/tmp/thoosa")?;
            let best_checkpoint_path = TmpFile::new(PathBuf::from("/tmp/thoosa").join(format!("thoosa_train_net_best_{}_{}.safetensors", now_ms(), random::<u64>())));

            let mut best_valid_loss = f32::MAX;

            let mut saved_best = false;
            let epochs_eta_guard = CollectEtaInvocationGuard::new(&format!("train_{descr}"), epochs as usize, 1);
            for idx in 0..epochs {

                debug!("TrainNet::run_inner, train: descr={descr}   0. cuda_mem={:?} epoch={idx} tokio::task_id={:?}",maybe_cuda_mem(&device), tokio::task::try_id());

                // TODO(perf): rebuilding the batch stream every epoch duplicates batch-index/resource setup; reuse per-epoch planning state when stochastic inputs allow it.
                let mut batches = batch.train_batch_stream(ctx.branch(false));
                debug!("TrainNet::run_inner, train: descr={descr}  1. cuda_mem={:?} tokio::task_id={:?}",maybe_cuda_mem(&device), tokio::task::try_id());

                let batch_eta_guard = CollectEtaInvocationGuard::new(&format!("epoch_{idx}_batches_in_{descr}"), batch.num_batches(), 1);

                // debug!("TrainNet::run_inner, train: descr={descr}  next={:?}", batches.next().await);

                while let Some(Ok((x_t_0, lengths, expected_y))) = cooperative(batches.next())
                    .instrument(tracing::debug_span!("run_inner/batches.next", file = file!(),line = line!()))
                    .await {
                    let input = Arc::new(TrainStepInputWrapper::new(TrainStepInput::new(x_t_0, lengths, expected_y, true)));

                    debug!("TrainNet::run_inner, train: descr={descr}  before oom input.id={:?} tokio::task_id={:?}", input.id(), tokio::task::try_id());

                    oom_handler.run(
                        &train_step,
                        Arc::clone(&input),
                        ctx.branch(false),
                        WithRunCtxContext::new(
                            MaxParallelAndDumpToCpuAdjustableContex::new(
                                MaxParallelAdjustContextImpl::new(),
                                DumpToCpuContextImpl::new(&dump_to_cpu_exclude),
                            ),
                        ),
                    )
                        .instrument(tracing::debug_span!("run_inner/oom_handler.run", file = file!(),line = line!()))
                        .await?;

                    debug!("TrainNet::run_inner, train: descr={descr}  after oom input.id={:?} tokio::task_id={:?}", input.id(), tokio::task::try_id());

                    collect_eta_record_completion(batch_eta_guard.id());

                }

                if idx % validation_step == 0 || idx == epochs - 1 {
                    debug!("TrainNet::run_inner, valid: descr={descr}  0. cuda_mem={:?} tokio::task_id={:?}", maybe_cuda_mem(&device), tokio::task::try_id());

                    let mut batches = batch.valid_batch_stream(ctx.branch(false));
                    debug!("TrainNet::run_inner, valid: descr={descr}  1. cuda_mem={:?}",maybe_cuda_mem(&device));
                    let mut valid_loss_sum = 0f32;
                    let mut valid_loss_count = 0u32;

                    while let Some(Ok((x_t_0, lengths, y_t))) = batches.next()
                        .instrument(tracing::debug_span!("run_inner/validation_step/batches.next", file = file!(),line = line!()))
                        .await {
                        let input = Arc::new(TrainStepInputWrapper::new(TrainStepInput::new(x_t_0, lengths, y_t, false)));
                        let output = oom_handler.run(
                            &train_step,
                            input,
                            ctx.branch(false),
                            WithRunCtxContext::new(
                                MaxParallelAndDumpToCpuAdjustableContex::new(
                                    MaxParallelAdjustContextImpl::new(),
                                    DumpToCpuContextImpl::new(&dump_to_cpu_exclude),
                                ),
                            ),
                        )
                            .instrument(tracing::debug_span!("run_inner/validation_step/oom_handler.run", file = file!(),line = line!()))
                            .await?;
                        debug!("TrainNet::run_inner: descr={descr} valid_loss={}", output.loss);
                        valid_loss_sum += output.loss;
                        valid_loss_count += 1;
                    }
                    if valid_loss_count > 0 {
                        let avg_valid_loss = valid_loss_sum / valid_loss_count as f32;
                        debug!("TrainNet::run_inner: descr={descr}  avg_valid_loss={avg_valid_loss}");

                        if avg_valid_loss < best_valid_loss - min_delta {
                            best_valid_loss = avg_valid_loss;
                            bad_epochs = 0;
                            {

                                // TODO(perf): best-checkpoint saves block the training loop; save on a background CPU task or only after a minimum score delta when configs produce many small improvements.
                                // let mut vs_guard = thoosa_shared::perf_log!(thoosa_shared::sync_mutex_lock!(vs)); TODO - download vs from Cuda/IPC
                                // panic_to_err!(vs_guard.save(&best_checkpoint_path))?;//panics at safetensors.rs:97 self.tensor.f_copy_data_u8(&mut data, numel).unwrap();
                            }
                            saved_best = true;
                        } else {
                            bad_epochs += 1;
                            if bad_epochs >= patience {
                                debug!("TrainNet::run_inner: descr={descr}  early stopping triggered valid_loss={avg_valid_loss} best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta}");
                                break;
                            }
                        }
                        debug!("TrainNet::run_inner: descr={descr}  valid_loss={avg_valid_loss} best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta}");
                    } else {
                        debug!("TrainNet::run_inner: descr={descr}  valid_loss skipped, no validation batches");
                    }

                }


                collect_eta_record_completion(epochs_eta_guard.id());
            }
            if saved_best {
                // let mut vs_guard = thoosa_shared::perf_log!(thoosa_shared::sync_mutex_lock!(vs)); TODO
                // vs_guard.load(&best_checkpoint_path)?;
                debug!("TrainNet::run_inner: descr={descr}  final best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta} best_checkpoint_path={best_checkpoint_path:?}");
            } else {
                warn!("TrainNet::run_inner: descr={descr}  done without saving best for run_config={run_config:?}");
            }

            let train_step = Arc::try_unwrap(train_step).map_err(|_| ThoError::new("TrainNet::run_inner: TrainStep still shared after training"))?;
            let var_store =  track_await(train_step.into_var_store()).await?;//nn::VarStore::new(device); //

            Ok((var_store, best_valid_loss, params))
        })


    }
}

impl<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> Stage for TrainNet<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY, C>,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    for<'a> NF:Fn(C, &'a RC,Arc<Mutex<VarStore>>)->BoxFuture<'a, Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError>> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<TrainNetInput>;
    type Out = Arc<DevicesWrap<TrainNetOutput<NP, RC>>>;
}


impl<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> AsyncStage for TrainNet<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY, C> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig> + ToHashmap,
    for<'a> NF:Fn(C, &'a RC,Arc<Mutex<VarStore>>)->BoxFuture<'a, Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError>> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainNet::run"))]
    fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> impl Future<Output=Result<Self::Out, ThoError>> + Send {

        let run_config_source = Arc::clone(&self.run_config_source);
        let device_idx = ctx.tho_device_idx();
        let reusable_completed_trials = self.reusable_completed_trials.clone();

        debug!("TrainNet::run: self.descr={:?}", self.descr);

        async move {
            let self_for_train = Arc::clone(&self);
            let device_idx_for_train = device_idx;

            let descr = self.descr.clone();
            let validation_step = self.validation_step;
            let (var_store, node_params, run_config, loss) = track_await(opt_train::<_, _, _, NP>(
                Arc::clone(&self.opt_config),
                move |run_config: RC, _listener: &mut (dyn TrainListener + '_)| {
                    debug!("TrainNet::run, about to call run_inner: descr={descr:?} run_config={:?}", run_config);

                    let self_arc = Arc::clone(&self_for_train);

                    self_arc.run_inner(
                        ctx.branch(false),
                        device_idx_for_train,
                        run_config,
                        validation_step,
                        descr.clone()
                    )
                },
                run_config_source,
                &self.descr,
                reusable_completed_trials,
            )).await?;

            info!("TrainNet::run: descr={:?} \x1b[38;5;208mmin_loss={loss} \x1b[0m run_config={run_config:?}", self.descr);

            Ok(Arc::new(DevicesWrap::wrap_plain(TrainNetOutput::new(SynKey::new([mono_key!("TrainNetOutput", "stage")])/*TODO id*/, var_store, node_params, run_config, loss))))
        }
    }
}

pub struct TrainStepInput<DI: Dims, DO: Dims> {
    id: SynKey,
    x: ThoBatch<DI>,
    lengths: Option<BatchLengths>,
    expected_y: ThoBatch<DO>,
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

    fn x(&self) -> ThoBatch<DI> {
        self.x.shallow_clone::<Batch<DI>>()//called form sync TrainStep::do_run
    }

    fn expected_y(&self) -> ThoBatch<DO> {
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
struct TrainStepInputWrapper<DI: Dims, DO: Dims> {
    input: TrainStepInput<DI, DO>,
}

impl<DI: Dims, DO: Dims> TrainStepInputWrapper<DI, DO> {
    fn new(input: TrainStepInput<DI, DO>) -> Self {
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
        todo!()
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        todo!()
    }

    async fn drop_gpu(&self, device_idx: usize, name: &str) -> Result<(), ThoError> {
        todo!()
    }

    async fn purge_cpu(&self, purge_until: u64) -> usize {
        todo!()
    }
}

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
                false
            }
        }
    }

    async fn purge_gpu(&self, ids_to_keep: Option<&HashSet<&dyn CachableIdTr>>, device_idx: usize, name: &str) -> Result<(), ThoError> {
        match self {
            TrainStepDynSyncStage::Local(local) => {
                local.purge_gpu(ids_to_keep, device_idx, name).await
            }
            TrainStepDynSyncStage::Proxy { .. } => {
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

    fn into_var_store(self) -> Result<VarStore, ThoError> {
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



#[cfg(test)]
mod tests {
    #![allow(unused_assignments)]
    use std::future::ready;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::common::runnable::AsyncStage;
    use futures_util::StreamExt;
    use futures::stream::{self, BoxStream};
    use tch::nn::{AdamW, LinearConfig, OptimizerConfig, VarStore};
    use tch::kind::FLOAT_CUDA;
    use tch::Tensor;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_common::runner::common::MockRunCtx;
    use thoosa_common::batches::mock_async_data_batch;
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::dims::{Batch, Dims, ThoBatch};
    use thoosa_common::my_tch::{my_linear, ThoModule};
    #[allow(unused_imports)]
    use thoosa_common::train_nets::{CommonTrainingRunConfig, CommonTrainingRunConfigBuilder, CommonTrainingRunConfigFields, CommonTrainingRunConfigRangeOrValue, OptConfig, OptimizersRangeOrValue};
    use thoosa_properties_map_derive::ThoosaPropertiesMap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_run_config_args_derive::ThoosaRunConfig;
    use thoosa_shared::{RangeOrValue, ThoError};
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::traits::MonoMetrics;
    use crate::tasks::train_net::{TrainNet, TrainNetInput};
    use futures::FutureExt;
    use log::debug;
    use mockall::PredicateBoxExt;
    use parking_lot::Mutex;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::train_nets::AdamWRangeOrValue;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use futures_util::future::BoxFuture;

    #[derive(Clone, Debug)]
    struct DummyX;
    impl Dims for DummyX { type Metrics = MonoMetrics<1, 2>; }


    #[derive(Clone, Debug)]
    struct DummyY;
    impl Dims for DummyY { type Metrics = MonoMetrics<1, 2>; }

    static X_F_CALLS: AtomicUsize = AtomicUsize::new(0);
    static LOSS_FN_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn dummy_x_f(t: ThoBatch<DummyX>) -> ThoBatch<DummyX> {
        debug!("dummy_x_f: X_F_CALLS={X_F_CALLS:?}");
        X_F_CALLS.fetch_add(1, Ordering::SeqCst);
        t
    }

    fn dummy_loss_fn(pred: &Tensor, target: &Tensor) -> Result<Tensor, ThoError> {
        LOSS_FN_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(pred.f_sub(target)?.pow_tensor_scalar(2.).mean(tch::Kind::Float))
    }

    fn dummy_batch() -> (ThoBatch<DummyX>, ThoBatch<DummyY>) {
        let x = Tensor::zeros([1, 1, 1], FLOAT_CUDA);
        let y = Tensor::zeros([1, 1, 1], FLOAT_CUDA);
        (x.into(), y.into())
    }

    fn make_stream() -> BoxStream<'static, Result<(ThoBatch<DummyX>, Option<BatchLengths>,  ThoBatch<DummyY>), ThoError>> {
        let (x, y) = dummy_batch();
        stream::iter(vec![Ok((x, None, y))]).boxed()
    }

    #[allow(unused_assignments)]
    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
    #[run_config_opt]
    struct DummyRunConfig {
        #[in_map]
        pub id: Option<RunConfigId>,

        #[in_map]
        #[run_config]
        pub num_layers: u32,

        #[in_map]
        #[run_config]
        common_config:CommonTrainingRunConfig
    }

    impl DummyRunConfig {
        pub fn new(id: Option<RunConfigId>, num_layers: u32) -> Self {
            Self { id, num_layers, common_config: CommonTrainingRunConfig::new(10, 10, 0.0001) }
        }
    }

    impl AsRef<CommonTrainingRunConfig> for DummyRunConfig {
        fn as_ref(&self) -> &CommonTrainingRunConfig {
            &self.common_config
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_train_net() -> Result<(), ThoError> {
        X_F_CALLS.store(0, Ordering::SeqCst);
        LOSS_FN_CALLS.store(0, Ordering::SeqCst);


        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let opt_config = Arc::new(OptConfig::new(10, 3));
        let run_config = Arc::new(
            DummyRunConfigRangeOrValue {
                id: Some(1),
                num_layers: RangeOrValue::Range { range: 1..3, step: None },
                common_config: CommonTrainingRunConfigRangeOrValue{
                    epochs: RangeOrValue::Value(10),
                    batch_size: RangeOrValue::Value(10),
                    lr: RangeOrValue::Value(0.0001),
                    patience: RangeOrValue::Value(5),
                    min_delta: RangeOrValue::Value(0.0001),
                    optimizer: Some(OptimizersRangeOrValue::AdamW(AdamWRangeOrValue{
                        beta1: RangeOrValue::Value(0.9),
                        beta2: RangeOrValue::Value(0.999),
                        wd: RangeOrValue::Value(0.01),
                        amsgrad: RangeOrValue::Value(false),
                    })),
                },
            }
            .opt_run_configs().expect("DummyRunConfigRangeOrValue::opt_run_configs should not fail"),
        );
        let optimizer = |_rc: &DummyRunConfig, vs: &VarStore| {
            AdamW::default()
                .amsgrad(true)
                .build(&vs, 0.001)
                .map_err(ThoError::from)
        };
        let epochs_count = 10u32;
        let epochs = move |_rc: &DummyRunConfig| epochs_count;

        let train_stream_calls = Arc::new(AtomicUsize::new(0));
        let valid_stream_calls = Arc::new(AtomicUsize::new(0));
        let mock_test_batch = Arc::new(
            mock_async_data_batch::build::<DummyX, DummyY, MockRunCtx>()
                .with_train_stream_calls(Arc::clone(&train_stream_calls), |_ctx| make_stream())
                .with_valid_stream_calls(Arc::clone(&valid_stream_calls), |_ctx| make_stream())
                .finish(),
        );
        let mut ctx = mock_run_ctx::default(1);

        let validation_step = 10;
        let train_net: Arc<TrainNet<MockRunCtx, _, _, _, _, _, _, _, (), _, _, _>> = Arc::new(
            TrainNet::new(
                move |c:MockRunCtx, _rc: &DummyRunConfig| ready(Ok(Arc::clone(&mock_test_batch))).boxed(),
                |_:MockRunCtx, _rc: &DummyRunConfig, vs:Arc<Mutex<VarStore>>| {
                    async move {
                        let net: Box<dyn ThoModule<Batch<DummyX>, Batch<DummyY>>> =
                            Box::new(my_linear::<_, Batch<DummyX>, Batch<DummyY>>(
                                vs.lock().root() / "linear",
                                1,
                                1,
                                LinearConfig::default(),
                                #[cfg(any(debug_assertions, tho_debug))]
                                "test_train_net".to_string(),
                            )?);
                        Ok((net, ())) // Net params, dummy
                    }.boxed()
                },
                Arc::clone(&opt_config),
                Arc::clone(&run_config),
                dummy_loss_fn,
                optimizer,
                dummy_x_f,
                epochs,
                validation_step,
                "n/a".to_string(),
            ),
        );


        let output = train_net
            .run(
                Arc::new(TrainNetInput::new(SynKey::new([mono_key!("test", "test")]))),
                ctx,
            )
            .await?;
        let _ = Arc::into_inner(output)
            .expect("train_net output still shared")
            .take_data(1)
            .await?;

        // Train stream runs once per epoch; valid stream runs every 10th and the final epoch.
        let grid_runs = run_config.grid_to_vals().len();
        let train_iters = epochs_count;
        let valid_iters = (train_iters + 9) / 10;

        let expected_train_streams = grid_runs * opt_config.opt_max_evals as usize * epochs_count as usize;
        let expected_valid_streams = expected_train_streams / validation_step as usize + epochs_count as usize;

        // let expected_batch_calls = expected_train_streams + expected_valid_streams;

        assert_eq!(
            expected_train_streams,
            train_stream_calls.load(Ordering::SeqCst)
        );
        assert_eq!(
            expected_valid_streams,
            valid_stream_calls.load(Ordering::SeqCst)
        );
        debug!("X_F_CALLS={X_F_CALLS:?} LOSS_FN_CALLS={LOSS_FN_CALLS:?}");
        assert_eq!(120, X_F_CALLS.load(Ordering::SeqCst));
        assert_eq!(120, LOSS_FN_CALLS.load(Ordering::SeqCst));

        Ok(())
    }

}
