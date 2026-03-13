use crate::tasks::cross_dims::ChannelsData2;
use async_trait::async_trait;
use thoosa_shared::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_shared::runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::common::{OutId, PermitsCtx, RunCtx};
use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_shared::runner::permits::{PermitStep, MAX_PERMITS};
use thoosa_shared::{mono_key, now_ms};
use futures_util::{future::BoxFuture, StreamExt};
use log::{debug, info, trace, warn};
use parking_lot::Mutex;
use rand::random;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::io::Write;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime};
use tch::nn::{Optimizer, VarStore};
use tch::Device;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::batches::{AsyncDataBatch, BatchLengths};
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::my_tch::{LossFp, ThoModule};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{opt_train, LinNetArgs, ConvNetParams, LabelNetTypeArgs, LabelNetTypeParams, OptConfig, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigResult, CommonTrainingRunConfig};
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
//TODO too many type params
pub struct TrainNet<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError> +  Send + Sync + 'static,
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
    pd:PhantomData<(C, DX, DY, BF, P, RC)>
}

impl<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> Debug for TrainNet<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    C: RunCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError> +  Send + Sync + 'static,
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
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    pub fn new(batch:BF, net:NF, opt_config:Arc<OptConfig>, run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >, loss_fn:LossFp, optimizer:FO, x_f:XF, cycles:FC, validation_step: u32, descr: String) -> Self {
        Self { batch, net, pd: PhantomData, opt_config, run_config_source, loss_fn, optimizer, x_f, epochs: cycles, validation_step, descr }
    }

    fn run_inner(
        self: Arc<Self>,
        mut ctx: C,
        device_idx: usize,
        vs: Arc<Mutex<VarStore>>,
        run_config: RC,
        validation_step: u32,
        descr:String
    ) -> BoxFuture<'static, Result<(f32, NP), ThoError>> {
        debug!("TrainNet::run_inner: descr={descr}");
        let (node, params) = {
            let vs_guard = thoosa_shared::with_perf_log(
                || vs.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                "TrainNet/run_inner/lock_vs",
            );
            match (self.net)(&run_config, &vs_guard) {
                Ok(v) => v,
                Err(e) => return Box::pin(async { Err(e) }),
            }
        };
        let optimizer = {
            let vs_guard = thoosa_shared::with_perf_log(
                || vs.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                "TrainNet/run_inner/lock_vs",
            );
            match (self.optimizer)(&run_config, &vs_guard) {
                Ok(v) => v,
                Err(e) => return Box::pin(async { Err(e) }),
            }
        };

        let device = Device::from(ThoDevice::from(device_idx));
        let epochs = (self.epochs)(&run_config);
        let x_f = {
            let this = Arc::clone(&self);
            Arc::new(move |batch: ThoBatch<DX>| (this.x_f)(batch))
        };

        let train_step = Arc::new(TrainStep::new(
            device,
            node,
            optimizer,
            x_f,
            self.loss_fn,
        ));

        let dump_to_cpu_exclude: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
        let out_id = Arc::new(()) as Arc<dyn CachableIdTr>;



        let mut bad_epochs = 0u32;
        let patience = run_config.as_ref().patience;
        let min_delta = run_config.as_ref().min_delta;

        Box::pin(async move {

            let batch = (self.batch)(ctx.branch(Some("train_batch"), false, false), &run_config).await?;
            debug!("TrainNet::run_inner: descr={descr} device={:?} cycles={}", train_step.device(), epochs);

            let oom_handler = OOMHandler::new(
                vec![
                    Adjusters::RunContext(Box::new(PermitsAdjuster::new(
                        MAX_PERMITS,
                        *PermitStep::TWENTY,
                    ))),
                    Adjusters::RunContext(Box::new(DumpToCpu::new())),
                ],
                None,
            );

            //TODO configure in ws
            std::fs::create_dir_all("/tmp/thoosa")?;
            let best_checkpoint_path = PathBuf::from("/tmp/thoosa").join(format!("thoosa_train_net_best_{}_{}.safetensors", now_ms(), random::<u64>()));
            let mut best_valid_loss = f32::MAX;


            let mut saved_best = false;
            let epochs_eta_guard = CollectEtaInvocationGuard::new(&format!("train_{descr}"), epochs as usize, 1);
            for idx in 1..epochs {

                debug!("TrainNet::run_inner, train: descr={descr}   0. cuda_mem={:?}",maybe_cuda_mem(&device));

                let mut batches = batch.train_batch_stream().await;
                debug!("TrainNet::run_inner, train: descr={descr}  1. cuda_mem={:?} ",maybe_cuda_mem(&device));

                let epoch_eta_guard = CollectEtaInvocationGuard::new(&format!("epoch_{idx}_in_{descr}"), 10/*TODO size hint*/, 1);
                while let Some(Ok((x_t_0, lengths, expected_y))) = batches.next().await {
                    let input = Arc::new(TrainStepInput::new(x_t_0, lengths, expected_y));

                    debug!("TrainNet::run_inner, train: descr={descr}  before oom input.id={:?}", input.id());

                    oom_handler.run(
                        &train_step,
                        Arc::clone(&input),
                        ctx.branch(Some("train_step"), false, false),
                        Contexts::new(
                            NoopInputContext(None),
                            NoopRunnableContext {},
                            PermitsAndDumpToCpuAdjustableContex::new(
                                PermitsAdjustContextImpl::new(),
                                DumpToCpuContextImpl::new(&dump_to_cpu_exclude),
                            ),
                        ),
                    ).await?;

                    debug!("TrainNet::run_inner, train: descr={descr}  after oom input.id={:?}", input.id());

                    collect_eta_record_completion(epoch_eta_guard.id());

                }

                //TODO config validation_step

                if idx % validation_step == 0 || idx == epochs - 1 {

                    debug!("TrainNet::run_inner, valid: descr={descr}  0. cuda_mem={:?}",maybe_cuda_mem(&device));

                    let mut batches = batch.valid_batch_stream().await;
                    debug!("TrainNet::run_inner, valid: descr={descr}  1. cuda_mem={:?}",maybe_cuda_mem(&device));
                    let mut valid_loss_sum = 0f32;
                    let mut valid_loss_count = 0u32;

                    while let Some(Ok((x_t_0, mut lengths, y_t))) = batches.next().await {
                        tch::no_grad(||{

                            let x_t = (train_step.x_f())(x_t_0);


                            let pred_y = train_step.forward(&x_t, lengths.as_mut(), false)?;

                            let valid_loss = (train_step.loss_fn())(&pred_y.as_ref(), &y_t.as_ref())?;
                            let valid_loss_f32 = valid_loss.double_value(&[]) as f32;
                            valid_loss_sum += valid_loss_f32;
                            valid_loss_count += 1;

                            Ok::<_, ThoError>(())
                        })?;
                    }
                    if valid_loss_count > 0 {
                        let valid_loss = valid_loss_sum / valid_loss_count as f32;
                        if valid_loss < best_valid_loss - min_delta {//TODO min_delta to * coefficient
                            best_valid_loss = valid_loss;
                            bad_epochs = 0;
                            {
                                let mut vs_guard = thoosa_shared::with_perf_log(|| vs.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"), "TrainNet/run_inner/lock_vs");
                                vs_guard.save(&best_checkpoint_path)?;
                            }
                            saved_best = true;
                        } else {
                            bad_epochs += 1;
                            if bad_epochs >= patience {
                                debug!("TrainNet::run_inner: descr={descr}  early stopping triggered valid_loss={valid_loss} best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta}");
                                break;
                            }
                        }
                        debug!("TrainNet::run_inner: descr={descr}  valid_loss={valid_loss} best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta}");
                    } else {
                        debug!("TrainNet::run_inner: descr={descr}  valid_loss skipped, no validation batches");
                    }

                }


                collect_eta_record_completion(epochs_eta_guard.id());
            }
            if saved_best {
                let mut vs_guard = thoosa_shared::with_perf_log(
                    || vs.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                    "TrainNet/run_inner/lock_vs",
                );
                vs_guard.load(&best_checkpoint_path)?;
                debug!("TrainNet::run_inner: descr={descr}  final best_valid_loss={best_valid_loss} bad_epochs={bad_epochs} patience={patience} min_delta={min_delta} best_checkpoint_path={best_checkpoint_path:?}");
                std::fs::remove_file(best_checkpoint_path)?;
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
    B:AsyncDataBatch<DX, DY>,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError> +  Send + Sync + 'static,
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
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(C, &RC)->BoxFuture<Result<B, ThoError>> +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static + AsRef<CommonTrainingRunConfig>,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainNet::run"))]
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoError>> + Send {

        let run_config_source = Arc::clone(&self.run_config_source);
        let device_idx = ctx.device_idx();

        debug!("TrainNet::run: self.descr={:?}", self.descr);

        async move {
            let self_for_train = Arc::clone(&self);
            let device_idx_for_train = device_idx;

            let descr = self.descr.clone();
            let validation_step = self.validation_step;
            let (var_store, node_params, run_config, loss) = opt_train::<_, _, _, NP>(
                Arc::clone(&self.opt_config),
                move |vs: Arc<Mutex<VarStore>>, run_config: RC, _listener: &mut (dyn TrainListener + '_)| {
                    debug!("TrainNet::run, about to call run_inner: descr={descr:?} run_config={:?}", run_config);

                    let self_arc = Arc::clone(&self_for_train);

                    self_arc.run_inner(
                        ctx.branch(/*TODO impl WithId for RunConfigs*/Some(&format!("train on run_config={run_config:?}")), false, false),
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
            ).await?;

            info!("TrainNet::run: descr={:?} \x1b[38;5;208mmin_loss={loss} \x1b[0m run_config={run_config:?}", self.descr);

            Ok(Arc::new(DevicesWrap::wrap(TrainNetOutput::new(SynKey::new([mono_key!("TrainNetOutput", "stage")]), var_store, node_params, run_config, loss), CacheArea::default())))
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
            x: self.x.copy_to_device(dst_device).await?,
            lengths: self.lengths.clone(),
            expected_y: self.expected_y.copy_to_device(dst_device).await?,
            cache_area: self.cache_area.clone(),
        })
    }
}

impl<DI: Dims, DO: Dims> WithDataRef for TrainStepInput<DI, DO> {
    type O<'o> = TrainStepInput<DI, DO>;

    async fn with_data_ref<R, F, P>(&self, to_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send,
        P: RunCtx,
    {


        if self.x.device_idx() == to_device_idx {
            do_run(self, &permits_ctx)

        } else {

            let (x, lengths) = (
                self.x.copy_to_device(to_device_idx).await?,
                self.lengths.clone(),
            );

            let expected_y = self.expected_y.copy_to_device(to_device_idx).await?;

            let input = TrainStepInput::new(x, lengths, expected_y);

            do_run(&input, &permits_ctx)
        }
    }

    fn cache_area(&self) -> &CacheArea {
        &self.cache_area
    }
}

impl<DI: Dims, DO: Dims> WithSelectDataRef<()> for TrainStepInput<DI, DO> {
    async fn with_selected_data_ref<'o, R, F, P>(&self, to_device_idx: usize, permits_ctx: P, _selector: (), trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError>,
        P: RunCtx,
    {
       unimplemented!()//TODO fix OOMHandler to accept inputs without WithSelectDataRef impl
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
        let net = thoosa_shared::with_perf_log(
            || self.net.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "TrainStep/forward/lock_net",
        );
        net.tho_forward(x_t, lengths, train)
    }

    fn x_f(&self) -> Arc<F> {
        Arc::clone(&self.x_f)
    }

    fn loss_fn(&self) -> LossFp {
        self.loss_fn
    }

    fn set_lr(&self, lr: f64) {
        let mut opt = thoosa_shared::with_perf_log(
            || self.opt.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
            "TrainStep/set_lr/lock_opt",
        );
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
            let mut opt = thoosa_shared::with_perf_log(
                || self.opt.try_lock_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout"),
                "TrainStep/do_run/lock_opt",
            );
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
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::{mock_run_ctx, RunCtx};
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use futures_util::StreamExt;
    use futures::stream::{self, BoxStream};
    use parking_lot::RwLock;
    use tch::nn::{AdamW, LinearConfig, OptimizerConfig, VarStore};
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_shared::runner::common::MockRunCtx;
    use thoosa_common::batches::mock_async_data_batch;
    use thoosa_common::batches::BatchLengths;
    use thoosa_common::dims::{Batch, Dims, ThoBatch};
    use thoosa_common::my_tch::{my_linear, ThoModule};
    #[allow(unused_imports)]
    use thoosa_common::train_nets::{CommonTrainingRunConfig, CommonTrainingRunConfigBuilder, CommonTrainingRunConfigFields, CommonTrainingRunConfigRangeOrValue, OptConfig};
    use thoosa_properties_map_derive::ThoosaPropertiesMap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_run_config_args_derive::ThoosaRunConfig;
    use thoosa_shared::{RangeOrValue, ThoError};
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::traits::MonoMetrics;
    use crate::tasks::train_net::{TrainNet, TrainNetInput};
    use futures::FutureExt;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;

    #[derive(Clone, Debug)]
    struct DummyX;
    impl Dims for DummyX { type Metrics = MonoMetrics<1, 2>; }


    #[derive(Clone, Debug)]
    struct DummyY;
    impl Dims for DummyY { type Metrics = MonoMetrics<1, 2>; }

    static X_F_CALLS: AtomicUsize = AtomicUsize::new(0);
    static LOSS_FN_CALLS: AtomicUsize = AtomicUsize::new(0);

    fn mock_x_f(t: ThoBatch<DummyX>) -> ThoBatch<DummyX> {
        X_F_CALLS.fetch_add(1, Ordering::SeqCst);
        t
    }

    fn mock_loss_fn(pred: &Tensor, target: &Tensor) -> Result<Tensor, ThoError> {
        LOSS_FN_CALLS.fetch_add(1, Ordering::SeqCst);
        Ok(pred.f_sub(target)?.pow_tensor_scalar(2.).mean(tch::Kind::Float))
    }

    fn make_batch() -> (ThoBatch<DummyX>, ThoBatch<DummyY>) {
        let x = Tensor::zeros([1, 1, 1], FLOAT_CPU);
        let y = Tensor::zeros([1, 1, 1], FLOAT_CPU);
        (x.into(), y.into())
    }

    fn make_stream() -> BoxStream<'static, Result<(ThoBatch<DummyX>, Option<BatchLengths>,  ThoBatch<DummyY>), ThoError>> {
        let (x, y) = make_batch();
        stream::iter(vec![Ok((x, None, y))]).boxed()
    }

    #[allow(unused_assignments)]
    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
    #[run_config_opt]
    struct DummyRunConfig {
        #[in_map]
        pub id: Option<RunId>,

        #[in_map]
        #[run_config]
        pub num_layers: u32,

        #[in_map]
        #[run_config]
        common_config:CommonTrainingRunConfig
    }

    impl DummyRunConfig {
        pub fn new(id: Option<RunId>, num_layers: u32) -> Self {
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

        let net_fn = |_rc: &DummyRunConfig, vs: &VarStore| {
            let net: Box<dyn ThoModule<Batch<DummyX>, Batch<DummyY>>> =
                Box::new(my_linear::<_, Batch<DummyX>, Batch<DummyY>>(
                    vs.root() / "linear",
                    1,
                    1,
                    LinearConfig::default(),
                    #[cfg(debug_assertions)]
                    "test_train_net".to_string(),
                )?);
            Ok((net, ())) // Net params, dummy
        };
        let opt_config = Arc::new(OptConfig::new(10, 3));
        let run_config = Arc::new(
            DummyRunConfigRangeOrValue {
                id: Some(1),
                num_layers: RangeOrValue::Range { range: 1..3, step: None },
                common_config: CommonTrainingRunConfigRangeOrValue{
                    cycles: RangeOrValue::Value(10),
                    batch_size: RangeOrValue::Value(10),
                    lr: RangeOrValue::Value(0.0001),
                    patience: RangeOrValue::Value(5),
                    min_delta: RangeOrValue::Value(0.0001),
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
        let cycles_count = 10u32;
        let cycles = move |_rc: &DummyRunConfig| cycles_count;

        let train_stream_calls = Arc::new(AtomicUsize::new(0));
        let valid_stream_calls = Arc::new(AtomicUsize::new(0));
        let mock_test_batch = Arc::new(
            mock_async_data_batch::build::<DummyX, DummyY>()
                .with_train_stream_calls(Arc::clone(&train_stream_calls), make_stream)
                .with_valid_stream_calls(Arc::clone(&valid_stream_calls), make_stream)
                .finish(),
        );
        let mut ctx = mock_run_ctx::default(0);

        let train_net: Arc<TrainNet<MockRunCtx, _, _, _, _, _, _, _, (), _, _, _>> = Arc::new(
            TrainNet::new(
                move |c:MockRunCtx, _rc: &DummyRunConfig| ready(Ok(Arc::clone(&mock_test_batch))).boxed(),
                net_fn,
                Arc::clone(&opt_config),
                Arc::clone(&run_config),
                mock_loss_fn,
                optimizer,
                mock_x_f,
                cycles,
                10,
                "n/a".to_string(),
            ),
        );


        let output = train_net
            .run(
                Arc::new(DevicesWrap::wrap(TrainNetInput::new(SynKey::new([mono_key!("test", "test")])), CacheArea::default())),
                ctx,
            )
            .await?;
        let _ = Arc::into_inner(output)
            .expect("train_net output still shared")
            .take_data(0)
            .await?;

        // Train stream runs once per epoch; valid stream runs every 10th and the final epoch.
        let grid_runs = run_config.grid_to_vals().len();
        let train_iters = cycles_count - 1;
        let valid_iters = (train_iters + 9) / 10;
        let expected_train_streams =
            grid_runs * opt_config.opt_max_evals as usize * train_iters as usize;
        let expected_valid_streams =
            grid_runs * opt_config.opt_max_evals as usize * valid_iters as usize;
        let expected_batch_calls = expected_train_streams + expected_valid_streams;

        assert_eq!(
            expected_train_streams,
            train_stream_calls.load(Ordering::SeqCst)
        );
        assert_eq!(
            expected_valid_streams,
            valid_stream_calls.load(Ordering::SeqCst)
        );
        assert_eq!(expected_batch_calls, X_F_CALLS.load(Ordering::SeqCst));
        assert_eq!(expected_batch_calls, LOSS_FN_CALLS.load(Ordering::SeqCst));

        Ok(())
    }

}
