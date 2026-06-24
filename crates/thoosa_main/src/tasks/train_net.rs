use thoosa_common::dims::ChannelsData;
use async_trait::async_trait;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MaxParallelAndDumpToCpuAdjustableContex, MAX_PARALLEL};
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_common::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::runner::common::{MaxParallelCtx, OutId, RunCtx};

use thoosa_common::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::{cdt_for_dir, mono_key, now_ms, os_thread_id, perf_log, sync_mutex_lock};
use futures_util::{future::BoxFuture, StreamExt};
use log::{debug, info, trace, warn};
use parking_lot::Mutex;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::io::Write;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use rand::random;
use tch::nn::{Optimizer, VarStore};
use tch::Device;
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
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{opt_train, CommonTrainingRunConfig, ConvNetParams, LabelNetTypeArgs, LabelNetTypeParams, LinNetArgs, OptConfig, TrainLabelNetRunConfig, TrainLabelNetRunConfigResult};
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

fn maybe_cuda_mem(device: &Device) -> Option<(usize, usize)> {
    match device {
        Device::Cuda(_) => Some(cuda_mem()),
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
        Self { batch, net, pd: PhantomData, opt_config, run_config_source, loss_fn, optimizer, x_f, epochs, validation_step, descr, reusable_completed_trials: vec![] }
    }

    pub fn with_reusable_completed_trials(mut self, reusable_completed_trials: Vec<CompletedTrial>) -> Self {
        self.reusable_completed_trials = reusable_completed_trials;
        self
    }

    fn run_inner(
        self: Arc<Self>,
        mut ctx: C,
        tho_device_idx: usize,
        vs: Arc<Mutex<VarStore>>,
        run_config: RC,
        validation_step: u32,
        descr:String
    ) -> BoxFuture<'static, Result<(f32, NP), ThoError>> {
        let optimizer = {
            let vs_guard = perf_log!(sync_mutex_lock!(vs));
            match (self.optimizer)(&run_config, &vs_guard) {
                Ok(v) => v,
                Err(e) => return Box::pin(async { Err(e) }),
            }
        };

        Box::pin(async move {
            debug!("TrainNet::run_inner: descr={descr} validation_step={validation_step}");
            let (node, params) =  track_await((self.net)(ctx.branch(false), &run_config, Arc::clone(&vs)))
                .await?;

            let device = Device::from(ThoDevice::from(tho_device_idx));
            let epochs = (self.epochs)(&run_config);

            let self_x_f = Arc::clone(&self);
            let train_step = Arc::new(TrainStep::new(
                device,
                node,
                optimizer,
                Arc::new(move |batch: ThoBatch<DX>| (self_x_f.x_f)(batch)),
                self.loss_fn,
            ));
            let dump_to_cpu_exclude: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
            let out_id = Arc::new(()) as Arc<dyn CachableIdTr>;

            let mut bad_epochs = 0u32;
            let patience = run_config.as_ref().patience;
            let min_delta = run_config.as_ref().min_delta;



            let batch = (self.batch)(ctx.branch(false), &run_config)
                .instrument(tracing::debug_span!("run_inner/self.batch", file = file!(),line = line!()))
                .await?;

            debug!("TrainNet::run_inner: descr={descr} device={:?} epochs={}", train_step.as_ref().device(), epochs);

            let oom_handler = OOMHandler::new(
                vec![
                    Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(
                        MAX_PARALLEL,
                        *MaxParallelStep::FIFTY,
                    ))),
                    Adjusters::RunContext(Box::new(DumpToCpu::new("train_step"))),
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
                    let input = Arc::new(TrainStepInput::new(x_t_0, lengths, expected_y));

                    debug!("TrainNet::run_inner, train: descr={descr}  before oom input.id={:?} tokio::task_id={:?}", input.id(), tokio::task::try_id());

                    oom_handler.run(
                        &train_step,
                        Arc::clone(&input),
                        ctx.branch(false),
                        Contexts::new(
                            NoopInputContext(None),
                            NoopRunnableContext {},
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

                    while let Some(Ok((x_t_0, mut lengths, y_t))) = batches.next()
                        .instrument(tracing::debug_span!("run_inner/validation_step/batches.next", file = file!(),line = line!()))
                        .await {

                        tch::no_grad(||{

                            let x_t = (train_step.as_ref().x_f())(x_t_0);


                            let pred_y = train_step.as_ref().forward(&x_t, lengths.as_mut(), false)?;

                            let valid_loss = (train_step.as_ref().loss_fn())(&pred_y.as_ref(), &y_t.as_ref())?;
                            debug!("TrainNet::run_inner: descr={descr}  valid_loss={valid_loss}");

                            let valid_loss_f32 = panic_to_err!(Ok::<_, ThoError>(valid_loss.double_value(&[])))? as f32;
                            valid_loss_sum += valid_loss_f32;
                            valid_loss_count += 1;

                            Ok::<_, ThoError>(())
                        })?;
                    }
                    if valid_loss_count > 0 {
                        let avg_valid_loss = valid_loss_sum / valid_loss_count as f32;
                        debug!("TrainNet::run_inner: descr={descr}  avg_valid_loss={avg_valid_loss}");

                        if avg_valid_loss < best_valid_loss - min_delta {
                            best_valid_loss = avg_valid_loss;
                            bad_epochs = 0;
                            {
                                // TODO(perf): best-checkpoint saves block the training loop; save on a background CPU task or only after a minimum score delta when configs produce many small improvements.
                                let mut vs_guard = thoosa_shared::perf_log!(thoosa_shared::sync_mutex_lock!(vs));
                                panic_to_err!(vs_guard.save(&best_checkpoint_path))?;//panics at safetensors.rs:97 self.tensor.f_copy_data_u8(&mut data, numel).unwrap();
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
                let mut vs_guard = thoosa_shared::perf_log!(thoosa_shared::sync_mutex_lock!(vs));
                vs_guard.load(&best_checkpoint_path)?;
                debug!("TrainNet::run_inner: descr={descr}  final best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta} best_checkpoint_path={best_checkpoint_path:?}");
            } else {
                warn!("TrainNet::run_inner: descr={descr}  done without saving best for run_config={run_config:?}");
            }


            Ok((best_valid_loss, params))
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
    type In<'a> = TrainNetInput;
    type Out = TrainNetOutput<NP, RC>;
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
    type RunInput = Arc<DevicesWrap<Self::In<'static>>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        thoosa_shared::traits::WithId::id(input)
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainNet::run"))]
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoError>> + Send {

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
                move |vs: Arc<Mutex<VarStore>>, run_config: RC, _listener: &mut (dyn TrainListener + '_)| {
                    debug!("TrainNet::run, about to call run_inner: descr={descr:?} run_config={:?}", run_config);

                    let self_arc = Arc::clone(&self_for_train);

                    self_arc.run_inner(
                        ctx.branch(false),
                        device_idx_for_train,
                        vs,
                        run_config,
                        validation_step,
                        descr.clone()
                    )
                },
                run_config_source,
                ThoDevice::from(device_idx).into(),
                &self.descr,
                reusable_completed_trials,
            )).await?;

            info!("TrainNet::run: descr={:?} \x1b[38;5;208mmin_loss={loss} \x1b[0m run_config={run_config:?}", self.descr);

            Ok(Arc::new(DevicesWrap::wrap(TrainNetOutput::new(SynKey::new([mono_key!("TrainNetOutput", "stage")])/*TODO id*/, var_store, node_params, run_config, loss))))
        }
    }
}

pub struct TrainStepInput<DI: Dims, DO: Dims> {
    id: SynKey,
    x: ThoBatch<DI>,
    lengths: Option<BatchLengths>,
    expected_y: ThoBatch<DO>,
    cache_area: CacheArea,
}

impl<DI: Dims, DO: Dims> TrainStepInput<DI, DO> {
    fn new(x: ThoBatch<DI>, lengths: Option<BatchLengths>, expected_y: ThoBatch<DO>) -> Self {
        Self {
            id: SynKey::new([mono_key!("train_step", "stage")]),
            x,
            lengths,
            expected_y,
            cache_area: CacheArea::default(),
        }
    }

    fn x(&self) -> ThoBatch<DI> {
        self.x.shallow_clone::<Batch<DI>>()//called form sync TrainStep::do_run
    }

    fn expected_y(&self) -> ThoBatch<DO> {
        self.expected_y.shallow_clone::<Batch<DO>>()
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
        self.x.device_idx()
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
            cache_area: self.cache_area.clone(),
        })
    }
}

impl<DI: Dims, DO: Dims> WithDataRef for TrainStepInput<DI, DO> {
    type O<'o> = TrainStepInput<DI, DO>;

    fn with_data_ref<'a, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, permits_ctx: P, _trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move {
            if self.x.device_idx() == to_device_idx {
                do_run(self, &permits_ctx)

            } else {

                let (x, lengths) = (
                    track_await(self.x.copy_to_device(to_device_idx)).await?,
                    self.lengths.clone(),
                );

                let expected_y = track_await(self.expected_y.copy_to_device(to_device_idx)).await?;

                let input = TrainStepInput::new(x, lengths, expected_y);
                do_run(&input, &permits_ctx)
            }
        }

    }
}

impl<DI: Dims, DO: Dims> WithSelectDataRef<()> for TrainStepInput<DI, DO> {
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(&'a self, _to_device_idx: usize, _permits_ctx: P, _selector: (), _trace_ctx: Option<&'static str>, _do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
       async move { unimplemented!() }//TODO fix OOMHandler to accept inputs without WithSelectDataRef impl
    }
}

pub struct TrainStep<F, DI: Dims, DO: Dims, C: RunCtx> {
    device: Device,
    net: Mutex<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>>,
    opt: Mutex<Optimizer>,
    x_f: Arc<F>,
    loss_fn: LossFp,
    pd: PhantomData<C>,
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
    fn new(device: Device, net: Box<dyn ThoModule<Batch<DI>, Batch<DO>>>, opt: Optimizer, x_f: Arc<F>, loss_fn: LossFp) -> Self {
        Self {
            device,
            net: Mutex::new(net),
            opt: Mutex::new(opt),
            x_f,
            loss_fn,
            pd: PhantomData,
        }
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn forward(&self, x_t: &ThoBatch<DI>, lengths: Option<&mut BatchLengths>, train: bool) -> Result<ThoBatch<DO>, ThoError> {
        // TODO(perf): lock_net is taken for every forward call; if TrainStep is not shared concurrently, split ownership so the hot path can call the module without a mutex.
        let net = perf_log!(sync_mutex_lock!(self.net));
        net.tho_forward(x_t, lengths, train)
    }

    fn x_f(&self) -> Arc<F> {
        Arc::clone(&self.x_f)
    }

    fn loss_fn(&self) -> LossFp {
        self.loss_fn
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
    type Out = ();
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
        let predicted_y = self.forward(&x_t, lengths.as_mut(), true)?;
        debug!("TrainStep::do_run: 1. cuda_mem={:?}",maybe_cuda_mem(&self.device));
        trace!("TrainStep::do_run: x_t=\n{x_t}");

        x_t.drop();

        let loss = (self.loss_fn)(predicted_y.as_ref(), &expected_y.as_ref())?;

        {
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
        }

        Ok(())
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
                Arc::new(DevicesWrap::wrap(TrainNetInput::new(SynKey::new([mono_key!("test", "test")])))),
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
