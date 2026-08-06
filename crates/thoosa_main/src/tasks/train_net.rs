use async_trait::async_trait;
use thoosa_common::dims::ChannelsData;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_shared::traits::MaxParallelCtx;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, ThoDataRef};
use thoosa_common::runner::global_ctx::GlobalCtx;

use crate::tasks::train_step::{TrainStep, TrainStepInput, TrainStepInputWrapper, TrainStepOutput, TrainStepProxyFactory, TrainStepWeightsSync, TrainStepXFn};
use futures_util::{StreamExt, future::BoxFuture};
use log::{debug, info, trace, warn};
use parking_lot::Mutex;
use rand::random;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::io::Write;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use std::time::{Instant, SystemTime};
use tch::Kind::Float;
use tch::nn::{Optimizer, VarStore};
use tch::{Device, nn};
use thoosa_common::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::my_tch::{LossFp, ThoModule};
use thoosa_common::parzen::optimizer::CompletedTrial;
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, NoopInputContext, NoopRunnableContext, OOMHandler, WithRunCtxContext, run_ctx_adjuster};
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{CommonTrainingRunConfig, ConvNetParams, LabelNetTypeArgs, LabelNetTypeParams, LinNetArgs, OptConfig, TrainLabelNetRunConfig, TrainLabelNetRunConfigResult, TrainMotifNetRunConfig, opt_train};
use thoosa_common::utils::{TrainListener, optimizer_mem_size};
use thoosa_dummy_data_derive::DummyData;
use thoosa_properties_map::ToHashmap;
use thoosa_shared::CopyToDevice;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::collect_futures_eta::{CollectEtaInvocationGuard, collect_eta_record_completion};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig};
use thoosa_shared::tmp_file::TmpFile;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, AsyncStage, CachableIdTr, Data, DynSyncStageBox, OutId, RunCtx, Stage, SyncStage, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{ThoError, panic_to_err};
use thoosa_shared::{cdt_for_dir, mono_key, now_ms, os_thread_id, perf_log, sync_mutex_lock};
use tokio::task::coop::cooperative;
use tracing::Instrument;

pub fn maybe_cuda_mem(device: &Device) -> Option<(usize, usize)> {
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
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
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
            static STEP_COUNT:AtomicU32 = AtomicU32::new(0);//TODO
            let step_id = format!("train_step_{}", STEP_COUNT.fetch_add(1, Relaxed)).leak();
            GlobalCtx::global().register_cache((step_id, Arc::clone(&train_step) as Arc<dyn FuturesRunnerMapPurger>));

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

            GlobalCtx::global().remove_cache(step_id);
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
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
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


#[cfg(test)]
pub mod tests {
    #![allow(unused_assignments)]
    use futures::stream::{self, BoxStream};
    use futures_util::StreamExt;
    use std::future::ready;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tch::Tensor;
    use tch::kind::FLOAT_CUDA;
    use tch::nn::{AdamW, LinearConfig, OptimizerConfig, VarStore};
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::batches::mock_async_data_batch;
    use thoosa_common::dims::{Batch, Dims, ThoBatch};
    use thoosa_common::my_tch::{ThoModule, my_linear};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::MockRunCtx;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_shared::traits::AsyncStage;
    #[allow(unused_imports)]
    use thoosa_common::train_nets::{CommonTrainingRunConfig, CommonTrainingRunConfigBuilder, CommonTrainingRunConfigFields, CommonTrainingRunConfigRangeOrValue, OptConfig, OptimizersRangeOrValue};
    use crate::tasks::train_net::{TrainNet, TrainNetInput};
    use futures::FutureExt;
    use futures_util::future::BoxFuture;
    use log::debug;
    use mockall::PredicateBoxExt;
    use parking_lot::Mutex;
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::train_nets::AdamWRangeOrValue;
    use thoosa_properties_map_derive::ThoosaPropertiesMap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_run_config_args_derive::ThoosaRunConfig;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::traits::MonoMetrics;
    use thoosa_shared::{RangeOrValue, ThoError};

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
