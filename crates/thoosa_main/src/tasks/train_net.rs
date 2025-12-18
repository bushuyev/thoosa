use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::MotifsCtx;
use caching_runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use caching_runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use caching_runner::common::{OutId, PermitsCtx, RunCtx};
use caching_runner::global_ctx::GlobalCtx;
use caching_runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use caching_runner::permits::{PermitStep, MAX_PERMITS};
use futures_util::{future::BoxFuture, StreamExt};
use log::{debug, info, trace};
use parking_lot::{Mutex, RwLock};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;
use tch::nn::{Optimizer, VarStore};
use tch::Device;
use thoosa_common::applicables::ThoDevice;
use thoosa_common::batches::AsyncDataBatch;
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::my_tch::{LossFp, ThoModule};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{opt_train_2, FixedNumSlicesNetArgs, FixedNumSlicesNetParams, LabelNetTypeArgs, LabelNetTypeParams, OptConfig, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigResult};
use thoosa_common::utils::{optimizer_mem_size, TrainListener};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, SimpleId, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use thoosa_shared::CopyToDevice;
use thoosa_shared::{panic_to_err, ThoosaError};

#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct TrainNetInput{
    id:SimpleId
}

impl TrainNetInput {
    pub fn new(id: SimpleId) -> Self {
        Self { id }
    }
}
impl Input for TrainNetInput {

}

impl WithId for TrainNetInput {
    type ID = SimpleId;

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
    id:SimpleId,
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
    pub fn new(id: SimpleId, var_store: VarStore, node_params: N, run_config: R, min_loss:f32) -> Self {
        Self { id, var_store, node_params, run_config, min_loss }
    }
}

impl<N, R> WithId for TrainNetOutput<N, R>
where
    N:Debug + Send + Sync + 'static,
    R:Debug + Send + Sync + 'static,
{
    type ID = SimpleId;

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
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
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
    cycles:FC,
    pd:PhantomData<(C, DX, DY, BF, P, RC)>
}

impl<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> Debug for TrainNet<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
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
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{
    pub fn new(batch:BF, net:NF,opt_config:Arc<OptConfig>,run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >, loss_fn:LossFp, optimizer:FO, x_f:XF, cycles:FC) -> Self {
        Self { batch, net, pd: PhantomData, opt_config, run_config_source, loss_fn, optimizer, x_f, cycles }
    }

    fn run_inner(
        self: Arc<Self>,
        ctx: Arc<RwLock<C>>,
        device_idx: usize,
        vs: &VarStore,
        run_config: RC,
    ) -> BoxFuture<'static, Result<(f32, NP), ThoosaError>> {
        let batch = (self.batch)(&run_config);
        let (node, params) = match (self.net)(&run_config, vs) {
            Ok(v) => v,
            Err(e) => return Box::pin(async { Err(e) }),
        };
        let optimizer = match (self.optimizer)(&run_config, vs) {
            Ok(v) => v,
            Err(e) => return Box::pin(async { Err(e) }),
        };

        let device = Device::from(ThoDevice::from(device_idx));
        let cycles = (self.cycles)(&run_config);
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

        let ctx_name = ctx.read().name();
        let dump_to_cpu_exclude: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> = HashMap::new();
        let out_id = Arc::new(()) as Arc<dyn CachableIdTr>;

        Box::pin(async move {
            debug!("train: device={:?} cycles={}", train_step.device(), cycles);
            {
                let opt = train_step.opt.lock();
                debug!("train: opt={opt:?} optimizer_size={}", optimizer_mem_size(&opt));
            }

            let oom_handler = OOMHandler::new(
                vec![
                    Adjusters::RunContext(Box::new(PermitsAdjuster::new(
                        MAX_PERMITS,
                        *PermitStep::TWENTY,
                        {
                            let out_id = Arc::clone(&out_id);
                            move |n| GlobalCtx::global().acquire_many(
                                device_idx,
                                n,
                                Arc::clone(&out_id),
                            )
                        },
                    ))),
                    Adjusters::RunContext(Box::new(DumpToCpu::new())),
                ],
                None,
            );

            let mut last_valid_loss = [0f32; 3];

            for idx in 1..cycles {
                debug!("train:  0. cuda_mem={:?}", cuda_mem());
                let mut batches = batch.train_batch_stream().await;
                debug!("train:  1. cuda_mem={:?}", cuda_mem());
                while let Some(Ok((x_t_0, expected_y))) = batches.next().await {
                    let input = Arc::new(TrainStepInput::new(x_t_0, expected_y));

                    oom_handler.run(
                        &train_step,
                        Arc::clone(&input),
                        Arc::clone(&ctx),
                        Contexts::new(
                            NoopInputContext(None),
                            NoopRunnableContext {},
                            PermitsAndDumpToCpuAdjustableContex::new(
                                PermitsAdjustContextImpl::new(ctx_name),
                                DumpToCpuContextImpl::new(&dump_to_cpu_exclude),
                            ),
                        ),
                    ).await?;

                }

                if idx % 10 == 0 || idx == cycles - 1 {

                    debug!("train:  0. cuda_mem={:?}", cuda_mem());
                    let mut batches = batch.valid_batch_stream().await;
                    debug!("train:  1. cuda_mem={:?}", cuda_mem());
                    while let Some(Ok((x_t_0,y_t))) = batches.next().await {
                        tch::no_grad(||{

                            let x_t = (train_step.x_f())(x_t_0);

                            let pred_y = train_step.forward(&x_t, false)?;

                            let valid_loss = (train_step.loss_fn())(&pred_y.as_ref(), &y_t.as_ref())?;
                            let valid_loss_f32 = valid_loss.double_value(&[]) as f32;

                            last_valid_loss[(idx % last_valid_loss.len() as u32) as usize] = valid_loss_f32;
                            debug!("train: last_valid_loss={last_valid_loss:?}  valid loss={valid_loss:?} avg loss={}", last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32);

                            Ok::<_, ThoosaError>(())
                        })?;
                    }

                }
                info!(".");
                std::io::stdout().flush()?;
            }
            debug!("train: final valid last_valid_loss={last_valid_loss:?} avgvalid_loss={}",last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32);

            Ok((last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32, params))
        })
    }
}

impl<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> Stage for TrainNet<C, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY>,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
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
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<(Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, NP), ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
    NP: Default + Debug + Send + Sync + 'static,
    FC: Fn(&RC) -> u32 + Send + Sync + 'static,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TrainNet::run"))]
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>> + Send {
        let run_config_source = Arc::clone(&self.run_config_source);
        let device_idx = ctx.read().device_idx();

        async move {
            // let ctx_for_train = Arc::clone(&ctx);
            let self_for_train = Arc::clone(&self);
            let device_idx_for_train = device_idx;

            let (var_store, node_params, run_config, loss) = opt_train_2::<_, _, _, NP>(
                Arc::clone(&self.opt_config),
                move |vs: &VarStore, run_config: RC, _listener: &mut (dyn TrainListener + '_)| {
                    let ctx = Arc::clone(&ctx);
                    let self_arc = Arc::clone(&self_for_train);

                    self_arc.run_inner(ctx, device_idx_for_train, vs, run_config)
                },
                run_config_source,
                ThoDevice::from(device_idx).into(),
                "labels",
            ).await?;

            Ok(Arc::new(DevicesWrap::wrap_cpu(TrainNetOutput::new("".into(), var_store, node_params, run_config, loss))))
        }
    }
}

pub struct TrainStepInput<DI: Dims, DO: Dims> {
    id: SimpleId,
    x: ThoBatch<DI>,
    expected_y: ThoBatch<DO>,
}

impl<DI: Dims, DO: Dims> TrainStepInput<DI, DO> {
    fn new(x: ThoBatch<DI>, expected_y: ThoBatch<DO>) -> Self {
        Self {
            id: "train_step".into(),
            x,
            expected_y,
        }
    }

    fn x(&self) -> ThoBatch<DI> {
        self.x.shallow_clone::<Batch<DI>>()
    }

    fn expected_y(&self) -> ThoBatch<DO> {
        self.expected_y.shallow_clone::<Batch<DO>>()
    }
}

impl<DI: Dims, DO: Dims> Input for TrainStepInput<DI, DO> {
    fn id(&self) -> &SimpleId {
        &self.id
    }
}

impl<DI: Dims, DO: Dims> WithId for TrainStepInput<DI, DO> {
    type ID = SimpleId;

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

    fn size(&self) -> Self::Metrics {
        <ThoBatch<DI> as Data>::size(&self.x)
    }

    fn device(&self) -> usize {
        self.x.device()
    }

    fn mem_size(&self) -> u64 {
        self.x.mem_size() + self.expected_y.mem_size()
    }
}

impl<DI: Dims, DO: Dims> WithDataRef for TrainStepInput<DI, DO> {
    type O<'o> = TrainStepInput<DI, DO>;

    async fn with_data_ref<R, F, P>(&self, to_device_idx: usize, permits_ctx: Arc<RwLock<P>>, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
        for<'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>,
        P: PermitsCtx,
    {
        let _ = permits_ctx;
        let _ = trace_ctx;

        let x = if self.x.device() == to_device_idx {
            self.x()
        } else {
            self.x.copy_to_device(to_device_idx).await?
        };

        let expected_y = if self.expected_y.device() == to_device_idx {
            self.expected_y()
        } else {
            self.expected_y.copy_to_device(to_device_idx).await?
        };

        let input = TrainStepInput::new(x, expected_y);

        do_run(&input)
    }
}

impl<DI: Dims, DO: Dims> WithSelectDataRef<()> for TrainStepInput<DI, DO> {
    async fn with_selected_data_ref<'o, R, F, P>(&self, to_device_idx: usize, permits_ctx: Arc<RwLock<P>>, _selector: (), trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
        for<'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>,
        P: PermitsCtx,
    {
        self.with_data_ref(to_device_idx, permits_ctx, trace_ctx, do_run).await
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

    fn forward(&self, x_t: &ThoBatch<DI>, train: bool) -> Result<ThoBatch<DO>, ThoosaError> {
        let net = self.net.lock();
        net.tho_forward(x_t, train)
    }

    fn x_f(&self) -> Arc<F> {
        Arc::clone(&self.x_f)
    }

    fn loss_fn(&self) -> LossFp {
        self.loss_fn
    }

    fn set_lr(&self, lr: f64) {
        let mut opt = self.opt.lock();
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
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {
        let _ = ctx;
        let x_t_0 = input.x();
        let expected_y = input.expected_y();

        debug!("train: device={:?} x_t_0.size={:?} expected_y.size={:?}", self.device, x_t_0.size(), expected_y.size());
        debug!("train: 0. cuda_mem={:?}", cuda_mem());

        let x_t = (self.x_f)(x_t_0);

        debug!("train: device={:?}  x_t.size={:?}", self.device, x_t.size());

        let predicted_y = self.forward(&x_t, true)?;
        debug!("train: 1. cuda_mem={:?}", cuda_mem());
        trace!("train: x_t=\n{x_t}");
        x_t.drop();

        let loss = (self.loss_fn)(predicted_y.as_ref(), &expected_y.as_ref())?;

        {
            let mut opt = self.opt.lock();

            debug!("train: 2. cuda_mem={:?}", cuda_mem());
            debug!("train: loss={loss}");
            trace!("train: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}",optimizer_mem_size(&opt));

            predicted_y.drop();
            drop(expected_y);
            debug!("train: 2.1 cuda_mem={:?}", cuda_mem());
            let t1 = Instant::now();

            panic_to_err!({
                opt.backward_step(&loss);
                Ok::<_, ThoosaError>(())//TODO make panic_to_err to handle void functions
            })?;


            debug!("train: 3. cuda_mem={:?}  optimizer_size={}",cuda_mem(),optimizer_mem_size(&opt));
            debug!("train: device={:?}  backward_step done in {}",self.device,t1.elapsed().as_millis());
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
    fn out_id<'a>(&self, _input_id: &SimpleId) -> () {
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
    use crate::tasks::cross_dims::ChannelsData2;
    use std::sync::Arc;
    use tch::nn::{AdamW, OptimizerConfig, VarStore};
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use caching_runner::common::MockRunCtx;
    use thoosa_common::batches::mock_async_data_batch;
    use thoosa_common::batches::mock_async_data_batch::MockTestBatch;
    use thoosa_common::dims::{Batch, Dims, MotifProbability, ThoBatch};
    use thoosa_common::train_nets::{dummy_label_net, OptConfig};
    use thoosa_properties_map_derive::ThoosaPropertiesMap;
    use thoosa_run_config_args::OptRunConfigSource;
    use thoosa_run_config_args_derive::ThoosaRunConfig;
    use thoosa_shared::hyperparameter::Hyperparameter;
    use thoosa_shared::{RangeOrValue, ThoosaError};
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::traits::MonoMetrics;
    use crate::tasks::train_net::TrainNet;

    #[derive(Clone, Debug)]
    struct DummyX;
    impl Dims for DummyX { type Metrics = MonoMetrics<1, 2>; }


    #[derive(Clone, Debug)]
    struct DummyY;
    impl Dims for DummyY { type Metrics = MonoMetrics<1, 2>; }

    #[derive(ThoosaRunConfig, ThoosaPropertiesMap, Clone, PartialEq, Debug)]
    #[run_config_opt]
    struct DummyRunConfig {
        #[in_map]
        pub id: Option<RunId>,

        #[in_map]
        #[run_config]
        pub num_layers: u32,
    }

    impl DummyRunConfig {
        pub fn new(id: Option<RunId>, num_layers: u32) -> Self {
            Self { id, num_layers }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_0(){

        let batch = |rc:&DummyRunConfig| mock_async_data_batch::build().finish();
        let net_fn = |rc:&DummyRunConfig, vs:&VarStore|{
            Ok((
                dummy_label_net::<Batch<DummyX>, Batch<DummyY>>(10, 1., Cpu, "test")?,
                ()//Net params, dummy
            ))
        };
        let opt_config = Arc::new(OptConfig::new(10, 3));
        let run_config = Arc::new(DummyRunConfigRangeOrValue {
            id: Some(1),
            num_layers: RangeOrValue::Value(3),
        }.opt_run_configs());
        let loss_fn = |pred:&Tensor, target:&Tensor| {
            Ok::<_, ThoosaError>(Tensor::zeros([1], FLOAT_CPU))
        };
        let optimizer = |rc:&DummyRunConfig, vs: &VarStore|{
            AdamW::default().amsgrad(true).build(&vs, 0.001).map_err(|e|ThoosaError::from(e))
        };
        let x_f = |t:ThoBatch<DummyX>| {
            t
        };
        let cycles = |rc:&DummyRunConfig|{
            10
        };

        let train_net:TrainNet<MockRunCtx, _, _, _, _, _, _, _, (), _, _, _> = TrainNet::new(
            batch,
            net_fn,
            opt_config,
            run_config,
            loss_fn,
            optimizer,
            x_f,
            cycles
        );

        //TODO OOM
    }

}
