use async_trait::async_trait;
use thoosa_common::dims::ChannelsData;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use thoosa_common::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContext};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, ThoDataRef};
use thoosa_common::runner::global_ctx::{GlobalCtx, RegisteredCacheHandle};
use thoosa_shared::traits::MaxParallelCtx;

use futures_util::{StreamExt, future::BoxFuture};
use log::{debug, error, info, trace, warn};
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
use tch::Device;
use tch::Kind::Float;
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::cuda_utils::{cuda_mem, maybe_cuda_mem};
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::ipc::proxy::DynSyncStageWithStateEnum;
use thoosa_common::nets::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::nets::common_nets::{CommonTrainingRunConfig, CommonTrainingRunConfigBuilder, CommonTrainingRunConfigFields, CommonTrainingRunConfigRangeOrValue, ConvNetArgs, ConvNetParams, LinLenNetParams, OptConfig, OptimizersRangeOrValue, SelfAttentionNetArgs, SelfAttentionNetParams, WithCommonTrainingRunConfig, opt_train};
use thoosa_common::parzen::optimizer::CompletedTrial;
use thoosa_common::runner::oom_handler::{Adjusters, Contexts, InputContext, NoRunCtxAdjuster, NoStageAdjuster, NoopRunnableContext, OOMHandler, input_adjuster, run_ctx_adjuster};
use thoosa_common::runner::purge::FuturesRunnerMapPurger;
use thoosa_common::stages::train_step::{TrainStep, TrainStepInput, TrainStepInputSelector, TrainStepInputWrapper, TrainStepOutput, TrainStepXFn};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::utils::{TrainListener, optimizer_mem_size};
use thoosa_cpu_data_derive::CpuData;
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
use thoosa_shared::traits::{Adjustable, AsyncStage, CachableIdTr, Data, OutId, RunCtx, Stage, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::{ThoError, panic_to_err};
use thoosa_shared::{cdt_for_dir, mono_key, now_ms, os_thread_id, perf_log, sync_mutex_lock};
use tokio::task::coop::cooperative;
use tracing::Instrument;

/// Tracks batch chunks and combines their mean losses without biasing the final short chunk.
#[derive(Debug)]
struct TrainStepSplitsContext {
    input_len: i64,
    split_len: i64,
    done_to: i64,
    weighted_loss_sum: f64,
    samples: i64,
}

impl TrainStepSplitsContext {
    fn new(input_len: i64) -> Self {
        Self { input_len, split_len: input_len, done_to: 0, weighted_loss_sum: 0.0, samples: 0 }
    }
}

impl SplitsAdjusterContext for TrainStepSplitsContext {
    type Metrics = i64;

    fn input_len(&self) -> &Self::Metrics {
        &self.input_len
    }

    fn split_len(&self) -> &Self::Metrics {
        &self.split_len
    }

    fn done_to(&self) -> &Self::Metrics {
        &self.done_to
    }

    fn split_len_mut(&mut self) -> &mut Self::Metrics {
        &mut self.split_len
    }
}

impl<'a, R> InputContext<'a, R> for TrainStepSplitsContext
where
    R: Stage<Out = TrainStepOutput> + 'a,
{
    async fn add_partial_output(&mut self, output: R::Out) {
        let chunk_len = self.split_len.min(self.input_len - self.done_to);
        self.weighted_loss_sum += f64::from(output.loss) * chunk_len as f64;
        self.samples += chunk_len;
        self.done_to += chunk_len;
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }

    async fn build_final_output(self) -> Result<R::Out, ThoError> {
        if self.samples == 0 {
            return Err(ThoError::new("TrainStep input adjuster produced no samples"));
        }
        Ok(TrainStepOutput { loss: (self.weighted_loss_sum / self.samples as f64) as f32 })
    }
}

#[derive(Debug, CpuData)]
#[dummy_data(no_clone)]
pub struct TrainNetInput{
}

impl TrainNetInput {
    pub fn new() -> Self {
        Self {  }
    }
}
impl Input for TrainNetInput {

}



#[derive(Debug, CpuData)]
#[dummy_data(no_clone)]
pub struct TrainNetOutput<N, R>
where
    N:Debug + Send + Sync + 'static,
    R:Debug + Send + Sync + 'static,
{
    // The worker saves weights directly; only the temporary path crosses back
    // into the async orchestration and result-saving layers.
    pub var_store_path: TmpFile,
    pub node_params:N,
    pub run_config:R,
    pub min_loss:f32
}

impl<N, R> TrainNetOutput<N, R>
where
    N:Debug + Send + Sync + 'static,
    R:Debug + Send + Sync + 'static,
{
    pub fn new(var_store_path: TmpFile, node_params: N, run_config: R, min_loss:f32) -> Self {
        Self {  var_store_path, node_params, run_config, min_loss }
    }
}


impl<N, R> Output for TrainNetOutput<N, R>
where
    N:Debug + Send + Sync + 'static,
    R:Debug + Send + Sync + 'static,
{

}

#[async_trait]
pub trait TrainStepFactory<DX, DY, C, NP, RC>: Debug + Send + Sync
where
    DX: Dims + 'static,
    DY: Dims + 'static,
    C: RunCtx + 'static,
    NP: Debug + Send + Sync + 'static,
    RC: Debug + Send + Sync + 'static,
{
    async fn new(&self, run_config: RC, ctx: C, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<TrainStep<TrainStepXFn<DX>, DX, DY, C>>, NP), ThoError>;
}

// A trial must release the registry's strong stage reference even when it errors or is cancelled.
struct TrainStepCacheGuard(RegisteredCacheHandle);

impl Drop for TrainStepCacheGuard {
    fn drop(&mut self) {
        GlobalCtx::global().remove_cache(self.0.name());
    }
}

pub struct TrainNet<C, B, BF, DX, DY, P, RC, NP, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims + 'static,
    DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY, C> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    batch:BF,
    train_step_factory: Arc<dyn TrainStepFactory<DX, DY, C, NP, RC>>,
    opt_config:Arc<OptConfig>,
    run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >,
    epochs:FC,
    validation_step: u32,
    descr:String,
    reusable_completed_trials: Vec<CompletedTrial>,
    pd:PhantomData<(C, DX, DY, BF, P, RC)>
}

impl<C, B, BF, DX, DY, P, RC, NP, FC> Debug for TrainNet<C, B, BF, DX, DY, P, RC, NP, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims + 'static,
    DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY, C> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainNet").finish()
    }
}

impl<C, B, BF, DX, DY, P, RC, NP, FC> TrainNet<C, B, BF, DX, DY, P, RC, NP, FC>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    DX: Dims + 'static,
    DY: Dims + 'static,
    B: AsyncDataBatch<DX, DY, C> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + WithCommonTrainingRunConfig+ From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    pub fn new(
        batch: BF,
        train_step_factory: Arc<dyn TrainStepFactory<DX, DY, C, NP, RC>>,
        opt_config: Arc<OptConfig>,
        run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P>>,
        epochs: FC,
        validation_step: u32,
        descr: String
    ) -> Self {
        Self { batch, train_step_factory, pd: PhantomData, opt_config, run_config_source, epochs, validation_step, descr, reusable_completed_trials: vec![] }
    }

    pub fn with_reusable_completed_trials(mut self, reusable_completed_trials: Vec<CompletedTrial>) -> Self {
        self.reusable_completed_trials = reusable_completed_trials;
        self
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainNet::run_inner"))]
    fn run_inner(
        self: Arc<Self>,
        mut ctx: C,
        _tho_device_idx: usize,
        run_config: RC,
        validation_step: u32,
        descr:String
    ) -> BoxFuture<'static, Result<(TmpFile, f32, NP), ThoError>> {
        Box::pin(async move {
            debug!("TrainNet::run_inner: descr={descr} validation_step={validation_step}");

            let device = Device::from(ThoDevice::from(ctx.tho_device_idx()));
            let epochs = (self.epochs)(&run_config);

            // The path is part of worker initialization so the state owner can
            // checkpoint without exporting its VarStore.
            std::fs::create_dir_all("/tmp/thoosa")?;
            let best_checkpoint_path = PathBuf::from("/tmp/thoosa").join(format!("thoosa_train_net_best_{}_{}.safetensors", now_ms(), random::<u64>()));
            let (train_step, params) = track_await(self.train_step_factory.new(run_config.clone(), ctx.branch(false), best_checkpoint_path.clone())).await?;
            let train_step = Arc::new(train_step);
            static STEP_COUNT:AtomicU32 = AtomicU32::new(0);//TODO
            let step_id = format!("train_step_{}", STEP_COUNT.fetch_add(1, Relaxed)).leak();
            let _cache_guard = TrainStepCacheGuard(GlobalCtx::global().register_cache((step_id, Arc::clone(&train_step) as Arc<dyn FuturesRunnerMapPurger>)));

            let dump_to_cpu_exclude: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
            let out_id = Arc::new(()) as Arc<dyn CachableIdTr>;

            let mut bad_epochs = 0u32;
            let patience = run_config.as_ref().patience;
            let min_delta = run_config.as_ref().min_delta;


            let batch = (self.batch)(ctx.branch(false), &run_config)
                .instrument(tracing::debug_span!("run_inner/self.batch", file = file!(),line = line!()))
                .await?;

            debug!("TrainNet::run_inner: descr={descr} device={device:?} epochs={epochs}");

            let oom_handler: OOMHandler<'_, '_, _, NoStageAdjuster, _, TrainStepInputSelector> = OOMHandler::new(
                vec![
                    Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(
                        MAX_PARALLEL,
                        *MaxParallelStep::FAST,
                    ))),
                    Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("train_step"))),
                    Adjusters::Input(input_adjuster(SplitsAdjuster::<TrainStepInputSelector, i64>::new(1))),
                ],
                None,
            );

            let mut best_valid_loss = f32::MAX;

            let mut save_var_store = false;//the var store is saved in TrainStep stage to keep it on worker side in IPC mode
            let mut best_saved = false;
            let epochs_eta_guard = CollectEtaInvocationGuard::new(&format!("train_{descr}"), epochs as usize, 1);
            for idx in 0..epochs {

                debug!("TrainNet::run_inner, train: descr={descr} 0. cuda_mem={:?} epoch={idx} tokio::task_id={:?}",maybe_cuda_mem(&device), tokio::task::try_id());

                let mut batches = batch.train_batch_stream(ctx.branch(false));
                debug!("TrainNet::run_inner, train: descr={descr} 1. cuda_mem={:?} tokio::task_id={:?}",maybe_cuda_mem(&device), tokio::task::try_id());

                let batch_eta_guard = CollectEtaInvocationGuard::new(&format!("epoch_{idx}_batches_in_{descr}"), batch.num_batches(), 1);

                while let Some(batch_result) = cooperative(batches.next())
                    .instrument(tracing::debug_span!("run_inner/batches.next", file = file!(),line = line!()))
                    .await {
                    // A failed batch invalidates the trial instead of ending this epoch early.
                    let (x_t_0, lengths, expected_y) = batch_result?;
                    let input = Arc::new(TrainStepInputWrapper::new(TrainStepInput::new(x_t_0, lengths, expected_y, true, save_var_store)));
                    save_var_store = false;

                    debug!("TrainNet::run_inner, train: descr={descr}  before oom input.id={:?} tokio::task_id={:?}", input.id(), tokio::task::try_id());

                    let out_result = oom_handler.run(
                        &train_step,
                        Arc::clone(&input),
                        ctx.branch(false),
                        Contexts::<DynSyncStageWithStateEnum<TrainStep<TrainStepXFn<DX>, DX, DY, C>>, _, _, _>::new(
                            TrainStepSplitsContext::new(input.batch_len()?),
                            NoopRunnableContext {},
                            MaxParallelAndDumpToCpuAdjustableContex::new(
                                MaxParallelAdjustContextImpl::new(),
                                DumpToCpuContextImpl::new(&dump_to_cpu_exclude),
                            ),
                        ),
                    )
                        .instrument(tracing::debug_span!("run_inner/oom_handler.run", file = file!(),line = line!()))
                        .await;

                    if let Err(e) = &out_result {
                        error!("TrainNet::run_inner, train: descr={descr} e={e:?}");
                    }

                    debug!("TrainNet::run_inner, train: descr={descr}  after oom input.id={:?} tokio::task_id={:?} out={out_result:?}", input.id(), tokio::task::try_id());

                    collect_eta_record_completion(batch_eta_guard.id());

                }

                if idx % validation_step == 0 || idx == epochs - 1 {
                    debug!("TrainNet::run_inner, valid: descr={descr} 0. cuda_mem={:?} tokio::task_id={:?}", maybe_cuda_mem(&device), tokio::task::try_id());

                    let mut batches = batch.valid_batch_stream(ctx.branch(false));
                    debug!("TrainNet::run_inner, valid: descr={descr} 1. cuda_mem={:?}",maybe_cuda_mem(&device));
                    let mut valid_loss_sum = 0f32;
                    let mut valid_loss_count = 0u32;

                    while let Some(batch_result) = batches.next()
                        .instrument(tracing::debug_span!("run_inner/validation_step/batches.next", file = file!(),line = line!()))
                        .await {
                        // Never score a configuration using only the validation batches before an error.
                        let (x_t_0, lengths, y_t) = batch_result?;
                        let input = Arc::new(TrainStepInputWrapper::new(TrainStepInput::new(x_t_0, lengths, y_t, false, false)));
                        let input_context = TrainStepSplitsContext::new(input.batch_len()?);
                        let output = oom_handler.run(
                            &train_step,
                            input,
                            ctx.branch(false),
                            Contexts::<DynSyncStageWithStateEnum<TrainStep<TrainStepXFn<DX>, DX, DY, C>>, _, _, _>::new(
                                input_context,
                                NoopRunnableContext {},
                                MaxParallelAndDumpToCpuAdjustableContex::new(
                                    MaxParallelAdjustContextImpl::new(),
                                    DumpToCpuContextImpl::new(&dump_to_cpu_exclude),
                                ),
                            ),
                        )
                            .instrument(tracing::debug_span!("run_inner/validation_step/oom_handler.run", file = file!(),line = line!()))
                            .await?;
                        debug!("TrainNet::run_inner, valid: descr={descr} valid_loss={}", output.loss);
                        valid_loss_sum += output.loss;
                        valid_loss_count += 1;
                    }
                    if valid_loss_count > 0 {
                        let avg_valid_loss = valid_loss_sum / valid_loss_count as f32;
                        debug!("TrainNet::run_inner, valid: descr={descr}  avg_valid_loss={avg_valid_loss}");

                        if avg_valid_loss < best_valid_loss - min_delta {
                            best_valid_loss = avg_valid_loss;
                            bad_epochs = 0;

                            save_var_store = true;
                            best_saved = true;
                        } else {
                            bad_epochs += 1;
                            if bad_epochs >= patience {
                                info!("TrainNet::run_inner, valid: descr={descr}  early stopping triggered valid_loss={avg_valid_loss} best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta}");
                                break;
                            }
                        }
                        debug!("TrainNet::run_inner, valid: descr={descr}  valid_loss={avg_valid_loss} best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta}");
                    } else {
                        debug!("TrainNet::run_inner, valid: descr={descr}  valid_loss skipped, no validation batches");
                    }

                }

                collect_eta_record_completion(epochs_eta_guard.id());
            }

            if best_saved {
                debug!("TrainNet::run_inner: descr={descr} final best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta} best_checkpoint_path={best_checkpoint_path:?}");
            } else {
                warn!("TrainNet::run_inner: descr={descr}  done without saving best for run_config={run_config:?}");
            }

            Ok((TmpFile::new(best_checkpoint_path), best_valid_loss, params))
        })

    }
}

impl<C, B, BF, DX, DY, P, RC, NP, FC> Stage for TrainNet<C, B, BF, DX, DY, P, RC, NP, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims + 'static,
    DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY, C>,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + WithCommonTrainingRunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    type Ctx = C;
    type In<'a> = Arc<TrainNetInput>;
    type Out = TrainNetOutput<NP, RC>;
}


impl<C, B, BF, DX, DY, P, RC, NP, FC> AsyncStage for TrainNet<C, B, BF, DX, DY, P, RC, NP, FC>
where
    C: RunCtx + MaxParallelCtx + Adjustable + 'static,
    DX: Dims + 'static,
    DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY, C> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + WithCommonTrainingRunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig> + ToHashmap,
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
            let (var_store_path, node_params, run_config, loss) = track_await(opt_train::<_, _, _, NP, TmpFile>(
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

            Ok(TrainNetOutput::new(var_store_path, node_params, run_config, loss))
        }
    }
}


#[cfg(test)]
pub mod tests {
    #![allow(unused_assignments)]
    use futures::stream::{self, BoxStream};
    use futures_util::StreamExt;
    use std::future::ready;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tch::Tensor;
    use tch::kind::FLOAT_CUDA;
    use tch::nn::{AdamW, LinearConfig, OptimizerConfig, VarStore};
    use thoosa_common::dims::{Batch, Dims, ThoBatch};
    use thoosa_common::nets::batches::BatchLengths;
    use thoosa_common::nets::batches::mock_async_data_batch;
    use thoosa_common::nets::nodes::{ThoModule, my_linear};
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::MockRunCtx;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_shared::traits::{AsyncStage, RunCtx};
    #[allow(unused_imports)]
    use thoosa_common::stages::train_net::{TrainNet, TrainNetInput, TrainStepFactory};
    use async_trait::async_trait;
    use futures::FutureExt;
    use futures_util::future::BoxFuture;
    use log::debug;
    use mockall::PredicateBoxExt;
    use parking_lot::Mutex;
    use thoosa_common::stages::train_step::{TrainStep, TrainStepXFn};
    #[allow(unused_imports)]
    use thoosa_common::nets::common_nets::CommonTrainingRunConfig;
    #[allow(unused_imports)]
    use thoosa_common::nets::common_nets::{AdamWRangeOrValue, WithCommonTrainingRunConfig};
    use thoosa_common::ipc::proxy::DynSyncStageWithStateEnum;
    use thoosa_common::nets::common_nets::{CommonTrainingRunConfigBuilder, CommonTrainingRunConfigFields, CommonTrainingRunConfigRangeOrValue, ConvNetArgs, ConvNetParams, LinLenNetParams, OptConfig, OptimizersRangeOrValue, SelfAttentionNetArgs, SelfAttentionNetParams, opt_train};
    use thoosa_common::nets::tho_varstore::{ModConfig, ThoTrainModule};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
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

    fn dummy_loss_fn(pred: &Tensor, target: &Tensor) -> anyhow::Result<Tensor> {
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

    impl WithCommonTrainingRunConfig for DummyRunConfig {
        fn config(&self) -> &CommonTrainingRunConfig {
            &self.common_config
        }
    }

    #[derive(Debug)]
    struct DummyTrainStepFactory;

    #[async_trait]
    impl TrainStepFactory<DummyX, DummyY, MockRunCtx, (), DummyRunConfig> for DummyTrainStepFactory {
        async fn new(&self, run_config: DummyRunConfig, ctx: MockRunCtx, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<TrainStep<TrainStepXFn<DummyX>, DummyX, DummyY, MockRunCtx>>, ()), ThoError> {
            let x_f: Arc<TrainStepXFn<DummyX>> = Arc::new(Box::new(dummy_x_f));
            let (stage, ()) = TrainStep::new_local(
                ctx.tho_device_idx().into(),
                move |_run_config, device| async move {
                    let config = ModConfig::new(
                        AdamW::default().amsgrad(true).into(),
                        |var_store| Ok(Box::new(my_linear::<_, Batch<DummyX>, Batch<DummyY>>(var_store.root() / "linear", 1, 1, LinearConfig::default(), #[cfg(any(debug_assertions, tho_debug))] "test_train_net".to_string())?) as Box<dyn ThoModule<Batch<DummyX>, Batch<DummyY>>>),
                        0.001,
                    );
                    Ok((ThoTrainModule::new_on_device(config, device)?, ()))
                },
                x_f,
                dummy_loss_fn,
                run_config,
                vs_path,
            ).await?;
            Ok((DynSyncStageWithStateEnum::Local(Arc::new(stage)), ()))
        }
    }

    type TestTrainStep = TrainStep<TrainStepXFn<DummyX>, DummyX, DummyY, MockRunCtx>;
    type TestBatch = mock_async_data_batch::MockTestBatch<DummyX, DummyY, MockRunCtx>;
    type TestBatchFactory = Box<dyn for<'a> Fn(MockRunCtx, &'a DummyRunConfig) -> BoxFuture<'a, Result<Arc<TestBatch>, ThoError>> + Send + Sync>;
    type TestTrial = BoxFuture<'static, Result<(thoosa_shared::tmp_file::TmpFile, f32, ()), ThoError>>;

    mod mock_train_step_factory {
        use super::*;

        mockall::mock! {
            #[derive(Debug)]
            pub TrainStepFactory {}

            #[async_trait]
            impl TrainStepFactory<DummyX, DummyY, MockRunCtx, (), DummyRunConfig> for TrainStepFactory {
                async fn new(&self, run_config: DummyRunConfig, ctx: MockRunCtx, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<TestTrainStep>, ()), ThoError>;
            }
        }

        pub struct MockTrainStepFactoryBuilder {
            mock: MockTrainStepFactory,
        }

        impl MockTrainStepFactoryBuilder {
            pub fn new(stage: DynSyncStageWithStateEnum<TestTrainStep>) -> Self {
                let mut mock = MockTrainStepFactory::default();
                // Transfer the only strong stage reference to the trial so a leaked cache keeps the weak probe alive.
                mock.expect_new().times(1).return_once(move |_, _, _| Ok((stage, ())));
                Self { mock }
            }

            pub fn finish(self) -> MockTrainStepFactory {
                self.mock
            }
        }
    }

    async fn trial_with_batch_factory(batch: TestBatchFactory) -> Result<(TestTrial, std::sync::Weak<TestTrainStep>), ThoError> {
        let run_config = DummyRunConfig::new(None, 1);
        let (stage, ()) = DummyTrainStepFactory.new(run_config.clone(), mock_run_ctx::default(1), PathBuf::from("/tmp/thoosa_train_net_cleanup.safetensors")).await?;
        let DynSyncStageWithStateEnum::Local(local_stage) = &stage else {
            panic!("test factory must return a local stage");
        };
        let stage_ref = Arc::downgrade(local_stage);
        let factory = mock_train_step_factory::MockTrainStepFactoryBuilder::new(stage).finish();
        let run_configs = thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(Vec::<thoosa_shared::val::NamedVal<DummyRunConfigFields>>::new()),
                vec![],
            ),
            vec![],
        );
        let train_net = Arc::new(TrainNet::new(batch, Arc::new(factory), Arc::new(OptConfig::new(1, 1)), Arc::new(run_configs), |_rc: &DummyRunConfig| 1, 1, "cleanup regression".to_string()));
        // Exercise the trial directly: opt_train intentionally converts failed trials to a penalty score.
        Ok((train_net.run_inner(mock_run_ctx::default(1), 1, run_config, 1, "cleanup regression".to_string()), stage_ref))
    }

    fn batch_factory_with_stream_errors(train_error: Option<&'static str>, valid_error: Option<&'static str>) -> TestBatchFactory {
        let make_batches = |error: Option<&'static str>| {
            // Include a successful batch first so an error cannot silently return a partial epoch or validation score.
            make_stream().chain(stream::iter(error.into_iter().map(|msg| Err(ThoError::new(msg))))).boxed()
        };
        let batch = Arc::new(mock_async_data_batch::build::<DummyX, DummyY, MockRunCtx>()
            .with_train_stream_calls(Arc::new(AtomicUsize::new(0)), move |_ctx| make_batches(train_error))
            .with_valid_stream_calls(Arc::new(AtomicUsize::new(0)), move |_ctx| make_batches(valid_error))
            .finish());
        Box::new(move |_ctx, _rc| ready(Ok(Arc::clone(&batch))).boxed())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_net_releases_stage_after_success() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let (trial, stage_ref) = trial_with_batch_factory(batch_factory_with_stream_errors(None, None)).await?;
        trial.await?;
        assert!(stage_ref.upgrade().is_none(), "successful trial retained its training stage");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_net_releases_stage_after_batch_factory_error() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let (trial, stage_ref) = trial_with_batch_factory(Box::new(|_ctx, _rc| ready(Err(ThoError::new("batch factory failed"))).boxed())).await?;
        assert_eq!(trial.await.unwrap_err().to_string(), "batch factory failed");
        assert!(stage_ref.upgrade().is_none(), "failed batch factory retained its training stage");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_net_propagates_training_batch_error() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let (trial, stage_ref) = trial_with_batch_factory(batch_factory_with_stream_errors(Some("training batch failed"), None)).await?;
        assert_eq!(trial.await.unwrap_err().to_string(), "training batch failed");
        assert!(stage_ref.upgrade().is_none(), "training batch error retained its training stage");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_net_propagates_validation_batch_error() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let (trial, stage_ref) = trial_with_batch_factory(batch_factory_with_stream_errors(None, Some("validation batch failed"))).await?;
        assert_eq!(trial.await.unwrap_err().to_string(), "validation batch failed");
        assert!(stage_ref.upgrade().is_none(), "validation batch error retained its training stage");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_net_releases_stage_after_cancellation() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let started = Arc::new(tokio::sync::Notify::new());
        let started_for_batch = Arc::clone(&started);
        let (mut trial, stage_ref) = trial_with_batch_factory(Box::new(move |_ctx, _rc| {
            let started = Arc::clone(&started_for_batch);
            async move {
                started.notify_one();
                std::future::pending().await
            }.boxed()
        })).await?;
        // Cancel only after registration, while the batch factory is suspended at an await.
        tokio::time::timeout(std::time::Duration::from_secs(10), async {
            tokio::select! {
                _ = started.notified() => {},
                result = &mut trial => panic!("trial finished before cancellation: {result:?}"),
            }
        }).await.expect("batch factory was not reached");
        assert!(stage_ref.upgrade().is_some(), "trial must still own its stage before cancellation");
        drop(trial);
        assert!(stage_ref.upgrade().is_none(), "cancelled trial retained its training stage");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_train_net() -> Result<(), ThoError> {
        X_F_CALLS.store(0, Ordering::SeqCst);
        LOSS_FN_CALLS.store(0, Ordering::SeqCst);


        GlobalCtx::init_devices(2)?;
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
        let train_net: Arc<TrainNet<MockRunCtx, _, _, _, _, _, _, (), _>> = Arc::new(
            TrainNet::new(
                move |c:MockRunCtx, _rc: &DummyRunConfig| ready(Ok(Arc::clone(&mock_test_batch))).boxed(),
                Arc::new(DummyTrainStepFactory),
                Arc::clone(&opt_config),
                Arc::clone(&run_config),
                epochs,
                validation_step,
                "n/a".to_string(),
            ),
        );


        let output = train_net
            .run(
                Arc::new(TrainNetInput::new()),
                ctx,
            )
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
