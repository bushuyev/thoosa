use std::cmp::min;
use async_trait::async_trait;
use thoosa_common::runner::adjusters::max_parallel::MAX_PARALLEL;
use thoosa_common::runner::global_ctx::{GlobalCtx, RegisteredCacheHandle};
use thoosa_shared::traits::MaxParallelCtx;

use futures_util::future::BoxFuture;
use log::{debug, info, warn};
use rand::random;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use tch::Device;
use thoosa_common::dims::Dims;
use thoosa_common::ipc::proxy::DynSyncStageWithStateEnum;
use thoosa_common::nets::batches::AsyncDataBatch;
use thoosa_common::nets::common_nets::{CommonTrainingRunConfig, OptConfig, WithCommonTrainingRunConfig, opt_train};
use thoosa_common::parzen::optimizer::CompletedTrial;
use thoosa_common::runner::purge::CachePurger;
use thoosa_common::stages::train_step::{TrainStep, TrainStepXFn};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::utils::TrainListener;
use thoosa_cpu_data_derive::CpuData;
use thoosa_properties_map::ToHashmap;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::collect_futures_eta::{CollectEtaInvocationGuard, collect_eta_record_completion};
use thoosa_shared::hyperparameter::FieldName;
use thoosa_shared::run_configs::RunConfig;
use thoosa_shared::tmp_file::TmpFile;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, AsyncStage, RunCtx, Stage};
use thoosa_shared::val::NamedVal;
use thoosa_shared::ThoError;
use thoosa_shared::now_ms;
use tracing::Instrument;

#[derive(Debug, CpuData)]
#[cpu_data(shallow)]
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
#[cpu_data(shallow)]
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
    async fn new_step(&self, run_config: RC, ctx: C, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<TrainStep<TrainStepXFn<DX>, DX, DY, C>>, NP), ThoError>;
}

// A trial must release the registry's strong stage reference even when it errors or is cancelled.
struct TrainStepCacheGuard(RegisteredCacheHandle);

impl Drop for TrainStepCacheGuard {
    fn drop(&mut self) {
        GlobalCtx::global().remove_cache(self.0.key());
    }
}

pub async fn retry_training_after_cuda_reset<T, F, Fut>(gate: Option<Arc<thoosa_shared::CudaActivityGate>>, mut train: F) -> Result<T, ThoError>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ThoError>>,
{
    let Some(gate) = gate else { return train().await; };
    // An admitted ancestor must unwind before recovery can finish. Only the
    // outer async trial, which owns no CUDA activity, can wait and retry safely.
    if thoosa_shared::CudaActivityScope::current().is_some() { return train().await; }
    loop {
        gate.wait_if_paused().await?;
        let generation = gate.device_generation();
        // Prepared batches and their producers can be suspended outside runner
        // admission when reset clears their GPU data. Cancel the whole attempt
        // at that boundary, without holding an activity lease that blocks reset.
        let result = tokio::select! {
            biased;
            error = gate.wait_for_generation_end(generation) => Err(error),
            result = train() => {
                // Reset can also finish during this poll, before its notification
                // is observed. Do not publish a stale loss or a cache NotFound.
                gate.check_generation(generation).and(result)
            }
        };
        match result {
            Err(error) if matches!(error.what(), ThoError::CudaBeingRecovered) || error.needs_cuda_restart() => {
                // The attempt has returned and released its stage/cache/batches.
                // A reset is not a bad hyperparameter trial: rebuild all state
                // on the replacement worker before publishing any loss to TPE.
                gate.wait_if_paused().await?;
                if gate.device_generation() == generation { return Err(error); }
                warn!("Retrying training trial after CUDA device {} recovered from generation {}: {error:#}", gate.device_idx(), generation.0);
            }
            result => return result,
        }
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
    RC: RunConfig + TryFrom<std::vec::IntoIter<NamedVal<P>>, Error = anyhow::Error> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    batch:BF,
    train_step_factory: Arc<dyn TrainStepFactory<DX, DY, C, NP, RC>>,
    opt_config:Arc<OptConfig>,
    run_config_source: Arc<thoosa_shared::run_configs::TrainingRunConfigs<P> >,
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
    RC: RunConfig + TryFrom<std::vec::IntoIter<NamedVal<P>>, Error = anyhow::Error> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
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
    RC: RunConfig + WithCommonTrainingRunConfig+ TryFrom<std::vec::IntoIter<NamedVal<P>>, Error = anyhow::Error> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    pub fn new(
        batch: BF,
        train_step_factory: Arc<dyn TrainStepFactory<DX, DY, C, NP, RC>>,
        opt_config: Arc<OptConfig>,
        run_config_source: Arc<thoosa_shared::run_configs::TrainingRunConfigs<P>>,
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
            let best_checkpoint = TmpFile::new(best_checkpoint_path.clone());
            let (train_step, params) = track_await!(self.train_step_factory.new_step(run_config.clone(), ctx.branch(false), best_checkpoint_path.clone())).await?;
            let train_step = Arc::new(train_step);

            let _cache_guard = TrainStepCacheGuard(GlobalCtx::global().register_cache((train_step.stage_id(), Arc::clone(&train_step) as Arc<dyn CachePurger>)));

            let mut bad_epochs = 0u32;
            let patience = run_config.as_ref().patience;
            let min_delta = run_config.as_ref().min_delta;


            let batch = (self.batch)(ctx.branch(false), &run_config)
                .instrument(tracing::debug_span!("run_inner/self.batch", file = file!(),line = line!()))
                .await?;

            debug!("TrainNet::run_inner: descr={descr} device={device:?} epochs={epochs}");

            // This immutable CPU recovery point survives both GPU offload and
            // restoration. Even an OOM in epoch zero has a complete initial state.
            let mut recovery_checkpoint = train_step.save_epoch_checkpoint(ctx.tho_device_idx()).await?;
            let mut best_valid_loss = f32::MAX;
            let mut best_saved = false;
            let epochs_eta_guard = CollectEtaInvocationGuard::new(&format!("train_{descr}"), epochs as usize, 1);
            let mut max_other_parallel = MAX_PARALLEL as usize;
            let mut prefetch = true;
            let mut completed_epochs = 0;
            let mut aborted_epochs = 0;
            for idx in 0..epochs {
                // An epoch commits only after training, scheduled validation, and
                // checkpoint publication all succeed. No batch is retried in place.
                let epoch_result = async {
                    let batches = batch.train_batch_stream(ctx.branch(false));
                    let batch_eta_guard = CollectEtaInvocationGuard::new(&format!("epoch_{idx}_batches_in_{descr}"), batch.num_batches(), 1);
                    train_step.run_batches(batches, ctx.branch(false), true, max_other_parallel, prefetch, |_| {
                        collect_eta_record_completion(batch_eta_guard.id());
                    }).await?;
                    let mut valid_loss = None;
                    if idx % validation_step == 0 || idx == epochs - 1 {
                        let batches = batch.valid_batch_stream(ctx.branch(false));
                        let mut sum = 0.0;
                        let mut count = 0;
                        train_step.run_batches(batches, ctx.branch(false), false, max_other_parallel, prefetch, |loss| {
                            sum += loss;
                            count += 1;
                        }).await?;
                        if count > 0 { valid_loss = Some(sum / count as f32); }
                    }
                    let checkpoint = train_step.save_epoch_checkpoint(ctx.tho_device_idx()).await?;
                    if valid_loss.is_some_and(|loss| loss < best_valid_loss - min_delta) {
                        // Best-model publication remains explicit, including the last epoch.
                        train_step.checkpoint(&ctx).await?;
                    }
                    Ok::<_, ThoError>((checkpoint, valid_loss))
                }.await;



                collect_eta_record_completion(epochs_eta_guard.id());
                match epoch_result {
                    Ok((checkpoint, valid_loss)) => {
                        if max_other_parallel < MAX_PARALLEL as usize  {
                            max_other_parallel = min(max_other_parallel * 2, MAX_PARALLEL as usize);
                        }
                        recovery_checkpoint = checkpoint;
                        completed_epochs += 1;
                        if let Some(loss) = valid_loss {
                            if loss < best_valid_loss - min_delta {
                                best_valid_loss = loss;
                                bad_epochs = 0;
                                best_saved = true;
                            } else {
                                bad_epochs += 1;
                                if bad_epochs >= patience { break; }
                            }
                        }
                    }
                    Err(error) if (error.is_oom() || error.is_retryable_oom()) && !error.needs_cuda_restart() && !error.is_cuda_recovery_failed() => {
                        aborted_epochs += 1;
                        // Retrying a memory-constrained trial with the same extra
                        // resident batch can OOM even with exclusive GPU access.
                        prefetch = false;
                        // Restoration replaces the entire module/optimizer and its
                        // acknowledgement state. Failed epochs never affect patience.
                        train_step.restore_epoch_checkpoint(Arc::clone(&recovery_checkpoint), ctx.tho_device_idx()).await?;
                        warn!("TrainNet: discarded epoch={idx} after OOM; completed={completed_epochs} aborted={aborted_epochs} next_max_other_parallel={max_other_parallel} error={error:#}");
                        // Apply the user's per-epoch reduction on success and failure alike.
                        if max_other_parallel == 0 {
                            return Err(error.context("training epoch ran out of memory with exclusive runner access"));
                        }
                        max_other_parallel /= 2;

                    }
                    Err(error) => return Err(error),
                }
            }
            info!("TrainNet: completed_epochs={completed_epochs} aborted_epochs={aborted_epochs} descr={descr}");

            if best_saved {
                debug!("TrainNet::run_inner: descr={descr} final best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta} best_checkpoint_path={best_checkpoint_path:?}");
            } else {
                return Err(ThoError::new("training completed without a valid checkpoint"));
            }

            Ok((best_checkpoint, best_valid_loss, params))
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
    RC: RunConfig + WithCommonTrainingRunConfig + TryFrom<std::vec::IntoIter<NamedVal<P>>, Error = anyhow::Error> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
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
    RC: RunConfig + WithCommonTrainingRunConfig + TryFrom<std::vec::IntoIter<NamedVal<P>>, Error = anyhow::Error> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig> + ToHashmap,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainNet::run"))]
    fn run(self: Arc<Self>, _input: Self::In<'static>, mut ctx: Self::Ctx) -> impl Future<Output=Result<Self::Out, ThoError>> + Send {

        let run_config_source = Arc::clone(&self.run_config_source);
        let tho_device_idx = ctx.tho_device_idx();
        let reusable_completed_trials = self.reusable_completed_trials.clone();

        debug!("TrainNet::run: self.descr={:?}", self.descr);

        async move {
            let self_for_train = Arc::clone(&self);
            let device_idx_for_train = tho_device_idx;

            let descr = self.descr.clone();
            let validation_step = self.validation_step;
            let (var_store_path, node_params, run_config, loss) = track_await!(opt_train::<_, _, _, NP, TmpFile>(
                Arc::clone(&self.opt_config),
                move |run_config: RC, _listener: &mut (dyn TrainListener + '_)| {
                    debug!("TrainNet::run, about to call run_inner: descr={descr:?} run_config={:?}", run_config);

                    let self_arc = Arc::clone(&self_for_train);
                    let mut trial_ctx = ctx.branch(false);
                    let trial_descr = descr.clone();

                    retry_training_after_cuda_reset(GlobalCtx::global().cuda_activity_gate(device_idx_for_train), move || {
                        Arc::clone(&self_arc).run_inner(
                            trial_ctx.branch(false),
                            device_idx_for_train,
                            run_config.clone(),
                            validation_step,
                            trial_descr.clone()
                        )
                    })
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
use crate::tch_error::tch_result;
    use futures::stream::{self, BoxStream};
    use futures_util::StreamExt;
    use std::future::ready;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tch::Tensor;
    use tch::kind::FLOAT_CUDA;
    use tch::nn::{AdamW, LinearConfig};
    use thoosa_common::dims::{Batch, Dims, ThoBatch};
    use thoosa_common::nets::batches::BatchLengths;
    use thoosa_common::nets::batches::mock_async_data_batch;
    use thoosa_common::nets::nodes::{ThoModule, my_linear};
    use thoosa_common::runner::common::MockRunCtx;
    use thoosa_common::runner::common::mock_run_ctx;
    use thoosa_common::runner::global_ctx::GlobalCtx;
    use thoosa_shared::traits::{AsyncStage, RunCtx};
    #[allow(unused_imports)]
    use thoosa_common::stages::train_net::{TrainNet, TrainNetInput, TrainStepFactory};
    use async_trait::async_trait;
    use futures::FutureExt;
    use futures_util::future::BoxFuture;
    use log::debug;
    use thoosa_common::stages::train_step::{TrainStep, TrainStepXFn};
    #[allow(unused_imports)]
    use thoosa_common::nets::common_nets::CommonTrainingRunConfig;
    #[allow(unused_imports)]
    use thoosa_common::nets::common_nets::{AdamWRangeOrValue, WithCommonTrainingRunConfig};
    use thoosa_common::ipc::proxy::DynSyncStageWithStateEnum;
    use thoosa_common::nets::common_nets::{CommonTrainingRunConfigBuilder, CommonTrainingRunConfigFields, CommonTrainingRunConfigRangeOrValue, OptConfig, OptimizersRangeOrValue, opt_train};
    use thoosa_common::nets::tho_varstore::{ModConfig, ThoTrainModule};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_properties_map_derive::ThoosaPropertiesMap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_run_config_args_derive::ThoosaRunConfig;
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
        if target.f_double_value(&[0, 0, 0])? == -999.0 { return Err(ThoError::oom().into()); }
        Ok(pred.f_sub(target)?.f_pow_tensor_scalar(2.)?.f_mean(tch::Kind::Float)?)
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
        #[config_scalar]
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
        async fn new_step(&self, run_config: DummyRunConfig, ctx: MockRunCtx, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<TrainStep<TrainStepXFn<DummyX>, DummyX, DummyY, MockRunCtx>>, ()), ThoError> {
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
                async fn new_step(&self, run_config: DummyRunConfig, ctx: MockRunCtx, vs_path: PathBuf) -> Result<(DynSyncStageWithStateEnum<TestTrainStep>, ()), ThoError>;
            }
        }

        pub struct MockTrainStepFactoryBuilder {
            mock: MockTrainStepFactory,
        }

        impl MockTrainStepFactoryBuilder {
            pub fn new(stage: DynSyncStageWithStateEnum<TestTrainStep>) -> Self {
                let mut mock = MockTrainStepFactory::default();
                // Transfer the only strong stage reference to the trial so a leaked cache keeps the weak probe alive.
                mock.expect_new_step().times(1).return_once(move |_, _, _| Ok((stage, ())));
                Self { mock }
            }

            pub fn finish(self) -> MockTrainStepFactory {
                self.mock
            }
        }
    }

    #[cfg(test)]
    async fn trial_with_batch_factory(batch: TestBatchFactory) -> Result<(TestTrial, std::sync::Weak<TestTrainStep>), ThoError> {
        let run_config = DummyRunConfig::new(None, 1);
        let (stage, ()) = DummyTrainStepFactory.new_step(run_config.clone(), mock_run_ctx::default(1), PathBuf::from("/tmp/thoosa_train_net_cleanup.safetensors")).await?;
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

    fn oom_stream() -> BoxStream<'static, Result<(ThoBatch<DummyX>, Option<BatchLengths>, ThoBatch<DummyY>), ThoError>> {
        // Failure originates inside TrainStep after forward, not in the loader.
        let input = (|| {
            let x = tch_result(Tensor::f_zeros([1, 1, 1], (tch::Kind::Float, tch::Device::Cpu)))?;
            let y = tch_result(Tensor::f_full([1, 1, 1], -999.0, (tch::Kind::Float, tch::Device::Cpu)))?;
            Ok((x.into(), None, y.into()))
        })();
        stream::iter([input]).boxed()
    }

    fn recovery_run_configs() -> Arc<thoosa_shared::run_configs::TrainingRunConfigs<DummyRunConfigFields>> {
        Arc::new(DummyRunConfigRangeOrValue {
            id: None,
            num_layers: RangeOrValue::Value(1),
            common_config: CommonTrainingRunConfigRangeOrValue {
                epochs: RangeOrValue::Value(1), batch_size: RangeOrValue::Value(1), lr: RangeOrValue::Range { range: 0.001..0.002, step: None },
                patience: RangeOrValue::Value(1), min_delta: RangeOrValue::Value(0.0), optimizer: None,
            },
        }.opt_run_configs().unwrap())
    }

    #[tokio::test]
    async fn test_training_reset_waits_and_retries_without_spending_trial_budget() -> Result<(), ThoError> {
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let attempts = Arc::new(AtomicUsize::new(0));
        let (paused, mut pause) = tokio::sync::oneshot::channel();
        let mut paused = Some(paused);
        let train_gate = Arc::clone(&gate);
        let calls = Arc::clone(&attempts);
        // Use the actual optimization callback boundary: one evaluation must
        // survive an interruption without becoming MAX_LOSS/no min_var_store.
        let mut run = Box::pin(opt_train(
            Arc::new(OptConfig::new(1, 1)),
            async move |_config: DummyRunConfig, _listener: &mut (dyn crate::utils::TrainListener + '_)| {
                super::retry_training_after_cuda_reset(Some(Arc::clone(&train_gate)), || {
                    let attempt = calls.fetch_add(1, Ordering::SeqCst);
                    let activity = thoosa_shared::CudaActivityScope::try_enter(&train_gate).unwrap().unwrap();
                    if attempt == 0 {
                        paused.take().unwrap().send(train_gate.request_pause(ThoError::CudaBeingRecovered).unwrap()).unwrap();
                    }
                    async move {
                        let _activity = activity;
                        if attempt == 0 { Err(ThoError::CudaBeingRecovered) } else { Ok((42u32, 0.5, ())) }
                    }
                }).await
            },
            recovery_run_configs(), "reset trial", vec![],
        ));
        assert!(futures::poll!(run.as_mut()).is_pending(), "optimization must wait for recovery instead of exhausting its trials");
        let ticket = pause.try_recv().unwrap();
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        tokio::time::timeout(std::time::Duration::from_secs(1), gate.wait_until_idle()).await.unwrap();
        gate.finish_pause(ticket);
        let (artifact, (), _, loss) = run.await?;
        assert_eq!((artifact, loss), (42, 0.5));
        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_training_reset_cancels_suspended_batch_producer() -> Result<(), ThoError> {
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let calls = AtomicUsize::new(0);
        let mut owner = Some(Arc::new(()));
        let producer = Arc::downgrade(owner.as_ref().unwrap());
        let mut run = Box::pin(super::retry_training_after_cuda_reset(Some(Arc::clone(&gate)), || {
            let first = calls.fetch_add(1, Ordering::SeqCst) == 0;
            let owner = owner.take();
            async move {
                // A batch producer need not be awaiting a CUDA runner when
                // another trial triggers reset. It must still be discarded.
                let _owner = owner;
                if first { std::future::pending::<()>().await; }
                Ok(42)
            }
        }));
        assert!(futures::poll!(run.as_mut()).is_pending());
        let ticket = gate.request_pause(ThoError::CudaBeingRecovered)?;
        assert!(futures::poll!(run.as_mut()).is_pending());
        assert!(producer.upgrade().is_none(), "reset retained the suspended producer");
        gate.finish_pause(ticket);
        assert_eq!(run.await?, 42);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        Ok(())
    }

    #[tokio::test]
    async fn test_training_reset_rejects_outcomes_from_retired_generation() -> Result<(), ThoError> {
        for stale_result in [Ok(7), Err(ThoError::NotFound(Some("prepared batch".into())))] {
            let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
            let calls = AtomicUsize::new(0);
            let result = super::retry_training_after_cuda_reset(Some(Arc::clone(&gate)), || {
                let first = calls.fetch_add(1, Ordering::SeqCst) == 0;
                let gate = Arc::clone(&gate);
                let stale_result = stale_result.clone();
                async move {
                    if first {
                        // Complete recovery in the same poll as the old result,
                        // so the notification cannot win select before it returns.
                        let ticket = gate.request_pause(ThoError::CudaBeingRecovered)?;
                        gate.finish_pause(ticket);
                        stale_result
                    } else { Ok(42) }
                }
            }).await?;
            assert_eq!(result, 42, "published an outcome from the retired attempt");
            assert_eq!(calls.load(Ordering::SeqCst), 2);
        }
        Ok(())
    }

    #[tokio::test]
    async fn test_training_reset_does_not_wait_inside_cuda_activity_or_retry_without_recovery() {
        let gate = Arc::new(thoosa_shared::CudaActivityGate::new(1));
        let calls = AtomicUsize::new(0);
        let error = super::retry_training_after_cuda_reset(Some(Arc::clone(&gate)), || {
            calls.fetch_add(1, Ordering::SeqCst);
            ready(Err::<(), _>(ThoError::CudaBeingRecovered))
        }).await.unwrap_err();
        assert!(matches!(error.what(), ThoError::CudaBeingRecovered));
        assert_eq!(calls.load(Ordering::SeqCst), 1, "a bare reset error must not spin without a replacement generation");
        let activity = thoosa_shared::CudaActivityScope::try_enter(&gate).unwrap().unwrap();
        let ticket = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        let result = tokio::time::timeout(std::time::Duration::from_secs(1), activity.run(super::retry_training_after_cuda_reset(Some(Arc::clone(&gate)), || ready(Err::<(), _>(ThoError::CudaBeingRecovered))))).await.unwrap();
        assert!(matches!(result.unwrap_err().what(), ThoError::CudaBeingRecovered), "an admitted ancestor must unwind for recovery to drain");
        gate.finish_pause(ticket);
    }

    #[tokio::test]
    async fn test_opt_train_preserves_recovery_failure_even_after_success() {
        for successful_first in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let attempts = Arc::clone(&calls);
            let failure = ThoError::CudaRecoveryFailed(Arc::new(thoosa_shared::CudaRecoveryFailure {
                // Use the constructor so injected CUDA errors capture their backtrace.
                original: thoosa_shared::tho_error::cuda_error(thoosa_shared::CudaAction::NeedsRestart, "original worker failure".into()),
                recovery: Some(ThoError::new("replacement failed")),
            }));
            let error: ThoError = opt_train(
                Arc::new(OptConfig::new(3, 3)),
                async move |_config: DummyRunConfig, _listener: &mut (dyn crate::utils::TrainListener + '_)| {
                    let attempt = attempts.fetch_add(1, Ordering::SeqCst);
                    if successful_first && attempt == 0 { Ok((42u32, 0.5, ())) } else { Err(failure.clone()) }
                },
                recovery_run_configs(), "failed recovery", vec![],
            ).await.unwrap_err().into();
            assert!(error.is_cuda_recovery_failed(), "optimization lost the typed failure: {error:#}");
            assert!(error.to_string().contains("replacement failed"));
            assert_eq!(calls.load(Ordering::SeqCst), if successful_first { 2 } else { 1 }, "terminal recovery must stop further training");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_net_discards_oom_epochs_and_continues() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        for fail_validation in [false, true] {
            for failed_epoch in [0, 1] {
                let train_calls = Arc::new(AtomicUsize::new(0));
                let valid_calls = Arc::new(AtomicUsize::new(0));
                let epoch = Arc::new(AtomicUsize::new(0));
                let train_epoch = Arc::clone(&epoch);
                let valid_epoch = Arc::clone(&epoch);
                let batch = Arc::new(mock_async_data_batch::build::<DummyX, DummyY, MockRunCtx>()
                    .with_train_stream_calls(Arc::clone(&train_calls), move |_| {
                        let current = train_epoch.fetch_add(1, Ordering::SeqCst);
                        if !fail_validation && current == failed_epoch { make_stream().chain(oom_stream()).boxed() } else { make_stream() }
                    })
                    .with_valid_stream_calls(Arc::clone(&valid_calls), move |_| {
                        if fail_validation && valid_epoch.load(Ordering::SeqCst) - 1 == failed_epoch { oom_stream() } else { make_stream() }
                    }).finish());
                let configs = thoosa_shared::run_configs::OptRunConfigs::new(thoosa_shared::run_configs::GridRunConfigs::new(thoosa_shared::run_configs::ConstConfigs::new(Vec::<thoosa_shared::val::NamedVal<DummyRunConfigFields>>::new()), vec![]), vec![]);
                let factory: TestBatchFactory = Box::new(move |_, _| ready(Ok(Arc::clone(&batch))).boxed());
                let train_net = Arc::new(TrainNet::new(factory, Arc::new(DummyTrainStepFactory), Arc::new(OptConfig::new(1, 1)), Arc::new(configs), |_: &DummyRunConfig| 3, 1, "epoch rollback".into()));
                let (_checkpoint, loss, ()) = train_net.run_inner(mock_run_ctx::default(1), 1, DummyRunConfig::new(None, 1), 1, "epoch rollback".into()).await?;
                assert!(loss.is_finite());
                assert_eq!(train_calls.load(Ordering::SeqCst), 3, "an OOM skips the failed epoch without replaying its batches");
                assert_eq!(valid_calls.load(Ordering::SeqCst), if fail_validation { 3 } else { 2 }, "aborted training must not enter validation");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_net_stops_after_oom_at_exclusive_access() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let calls = Arc::new(AtomicUsize::new(0));
        let batch = Arc::new(mock_async_data_batch::build::<DummyX, DummyY, MockRunCtx>()
            .with_train_stream_calls(Arc::clone(&calls), |_| oom_stream()).finish());

        let configs = thoosa_shared::run_configs::OptRunConfigs::new(thoosa_shared::run_configs::GridRunConfigs::new(thoosa_shared::run_configs::ConstConfigs::new(Vec::<thoosa_shared::val::NamedVal<DummyRunConfigFields>>::new()), vec![]), vec![]);
        let factory: TestBatchFactory = Box::new(move |_, _| ready(Ok(Arc::clone(&batch))).boxed());

        let train_net = Arc::new(TrainNet::new(factory, Arc::new(DummyTrainStepFactory), Arc::new(OptConfig::new(1, 1)), Arc::new(configs), |_: &DummyRunConfig| 10, 1, "persistent OOM".into()));
        let error = train_net.run_inner(mock_run_ctx::default(1), 1, DummyRunConfig::new(None, 1), 1, "persistent OOM".into()).await.unwrap_err();
        assert!(error.is_oom());
        assert!(error.to_string().contains("exclusive runner access"));
        assert_eq!(calls.load(Ordering::SeqCst), 7, "epochs run with limits 32, 16, 8, 4, 2, 1, 0");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_train_net_final_epoch_checkpoint() -> Result<(), ThoError> {
        GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let configs = thoosa_shared::run_configs::OptRunConfigs::new(
            thoosa_shared::run_configs::GridRunConfigs::new(
                thoosa_shared::run_configs::ConstConfigs::new(Vec::<thoosa_shared::val::NamedVal<DummyRunConfigFields>>::new()),
                vec![],
            ),
            vec![],
        );
        // The only validation is in the final epoch. The old deferred save never ran.
        let train_net = Arc::new(TrainNet::new(batch_factory_with_stream_errors(None, None), Arc::new(DummyTrainStepFactory), Arc::new(OptConfig::new(1, 1)), Arc::new(configs), |_rc: &DummyRunConfig| 1, 1, "final checkpoint regression".to_string()));
        let (checkpoint, loss, ()) = train_net.run_inner(mock_run_ctx::default(1), 1, DummyRunConfig::new(None, 1), 1, "final checkpoint regression".to_string()).await?;
        let tensors = tch_result(Tensor::read_safetensors(checkpoint.path()))?;
        let bias = tch_result(tensors.iter().find(|(name, _)| name.ends_with("bias")).ok_or_else(|| ThoError::new("checkpoint has no bias"))?.1.f_double_value(&[0]))?;
        assert!((loss as f64 - bias * bias).abs() < 1e-6, "best loss must describe the completed checkpoint");
        Ok(())
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
        let ctx = mock_run_ctx::default(1);

        let validation_step = 10;
        let train_net: Arc<TrainNet<MockRunCtx, _, _, _, _, _, _, (), _>> = Arc::new(
            TrainNet::new(
                move |_c:MockRunCtx, _rc: &DummyRunConfig| ready(Ok(Arc::clone(&mock_test_batch))).boxed(),
                Arc::new(DummyTrainStepFactory),
                Arc::clone(&opt_config),
                Arc::clone(&run_config),
                epochs,
                validation_step,
                "n/a".to_string(),
            ),
        );


        let _output = train_net
            .run(
                Arc::new(TrainNetInput::new()),
                ctx,
            )
            .await?;

        // Train stream runs once per epoch; valid stream runs every 10th and the final epoch.
        let grid_runs = run_config.grid_to_vals()?.len();
        let train_iters = epochs_count;
        let _valid_iters = (train_iters + 9) / 10;

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
