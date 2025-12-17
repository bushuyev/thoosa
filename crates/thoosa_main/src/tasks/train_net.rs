use std::fmt::{Debug, Formatter};
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;
use futures_util::StreamExt;
use log::{debug, info, trace};
use parking_lot::{Mutex, RwLock};
use tch::Device;
use tch::nn::{Optimizer, VarStore};
use caching_runner::common::cachable::{Cached, DevicesWrap};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::applicables::ThoDevice;
use thoosa_common::batches::AsyncDataBatch;
use thoosa_common::cuda_utils::cuda_mem;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::my_tch::{LossFp, ThoModule};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{opt_train_2, CommonTrainingRunConfig, FixedNumSlicesNetArgs, FixedNumSlicesNetParams, LabelNetTypeArgs, LabelNetTypeParams, OptConfig, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigResult};
use thoosa_common::utils::{optimizer_mem_size, ListenerResponse, TrainListener};
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::hyperparameter::{FieldName, Hyperparameter};
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::{OptRunConfigs, RunConfig};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, SimpleId, WithId};
use thoosa_shared::val::{NamedRange, NamedVal};
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::MotifsCtx;

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
pub struct TrainNet<C, L, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
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
    batch:BF,
    net:NF,
    opt_config:Arc<OptConfig>,
    run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >,
    loss_fn:LossFp,
    optimizer:FO,
    x_f:XF,
    cycles:FC,
    pd:PhantomData<(C, L, DX, DY, BF, P, RC)>
}

impl<C, L, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> Debug for TrainNet<C, L, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
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
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainNet").finish()
    }
}

impl<C, L, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> TrainNet<C, L, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
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
    pub fn new(batch:BF, net:NF,opt_config:Arc<OptConfig>,run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >, loss_fn:LossFp, optimizer:FO, x_f:XF, cycles:FC) -> Self {
        Self { batch, net, pd: PhantomData, opt_config, run_config_source, loss_fn, optimizer, x_f, cycles }
    }
}

impl<C, L, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> Stage for TrainNet<C, L, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
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


impl<C, L, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC> AsyncStage for TrainNet<C, L, B, BF, DX, DY, P, RC, NF, NP, FO, XF, FC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
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

            let (var_store, node_params, run_config, loss) = opt_train_2::<_, _, _, NP>(
                Arc::clone(&self.opt_config),
                async move |vs: &VarStore, run_config: RC, _listener| {

                    let batch = (self.batch)(&run_config);
                    let (node, params) = (self.net)(&run_config, vs)?;
                    let optimizer = (self.optimizer)(&run_config, vs)?;

                    let device = Device::from(ThoDevice::from(device_idx));
                    let cycles = (self.cycles)(&run_config);

                    let train_step = Arc::new(TrainStep::new(
                        device,
                        node,
                        optimizer,
                        &self.x_f,
                        self.loss_fn,
                    ));

                    debug!("train: device={:?} cycles={}", train_step.device(), cycles);
                    {
                        let opt = train_step.opt.lock();
                        debug!("train: opt={opt:?} optimizer_size={}", optimizer_mem_size(&opt));
                    }

                    let mut last_valid_loss = [0f32; 3];
                    let mut listener: Option<&mut dyn TrainListener> = None;

                    for idx in 1..cycles {
                        debug!("train:  0. cuda_mem={:?}", cuda_mem());
                        let mut batches = batch.train_batch_stream().await;
                        debug!("train:  1. cuda_mem={:?}", cuda_mem());
                        while let Some(Ok((x_t_0, expected_y))) = batches.next().await {
                            let input = TrainStepInput::new(x_t_0, expected_y);
                            train_step.clone().do_run(&input, Arc::clone(&ctx))?;

                            if let Some(listener) = listener.as_mut() {
                                listener.on_cycle(0, 0.0).await;
                            }
                        }

                        if idx % 10 == 0 || idx == cycles - 1 {

                            debug!("train:  0. cuda_mem={:?}", cuda_mem());
                            let mut batches = batch.valid_batch_stream().await;
                            debug!("train:  1. cuda_mem={:?}", cuda_mem());
                            while let Some(Ok((x_t_0,y_t))) = batches.next().await {
                                if let Some(rs) = tch::no_grad(||{

                                    let x_t = (train_step.x_f())(x_t_0);

                                    let pred_y = train_step.forward(&x_t, false);

                                    let valid_loss = (train_step.loss_fn())(&pred_y.as_ref(), &y_t.as_ref());
                                    let valid_loss_f32 = valid_loss.double_value(&[]) as f32;

                                    last_valid_loss[(idx % last_valid_loss.len() as u32) as usize] = valid_loss_f32;
                                    debug!("train: last_valid_loss={last_valid_loss:?}  valid loss={valid_loss:?} avg loss={}", last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32);

                                    listener.as_mut().map(|l|l.on_valid_cycle(idx, valid_loss_f32))

                                }).take() {
                                    match rs {
                                        ListenerResponse::Stop => {
                                            break;
                                        }
                                        ListenerResponse::UpdateLR(lr) => {
                                            train_step.set_lr(lr);
                                        }
                                        ListenerResponse::Nope => {}
                                    }
                                }
                            }

                        }
                        info!(".");
                        std::io::stdout().flush()?;
                    }
                    debug!("train: final valid last_valid_loss={last_valid_loss:?} avgvalid_loss={}",last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32);

                    Ok((last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32, params))

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
    x: Mutex<Option<ThoBatch<DI>>>,
    expected_y: Mutex<Option<ThoBatch<DO>>>,
}

impl<DI: Dims, DO: Dims> TrainStepInput<DI, DO> {
    fn new(x: ThoBatch<DI>, expected_y: ThoBatch<DO>) -> Self {
        Self {
            id: "train_step".into(),
            x: Mutex::new(Some(x)),
            expected_y: Mutex::new(Some(expected_y)),
        }
    }

    fn take(&self) -> Option<(ThoBatch<DI>, ThoBatch<DO>)> {
        let x = self.x.lock().take();
        let expected_y = self.expected_y.lock().take();

        match (x, expected_y) {
            (Some(x), Some(expected_y)) => Some((x, expected_y)),
            _ => None,
        }
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

pub struct TrainStep<'xf, F, DI: Dims, DO: Dims, C: RunCtx> {
    device: Device,
    net: Mutex<Box<dyn ThoModule<Batch<DI>, Batch<DO>>>>,
    opt: Mutex<Optimizer>,
    x_f: &'xf F,
    loss_fn: LossFp,
    pd: PhantomData<C>,
}

impl<'xf, F, DI: Dims, DO: Dims, C: RunCtx> Debug for TrainStep<'xf, F, DI, DO, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainStep").finish()
    }
}

impl<'xf, F, DI, DO, C> TrainStep<'xf, F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    fn new(device: Device, net: Box<dyn ThoModule<Batch<DI>, Batch<DO>>>, opt: Optimizer, x_f: &'xf F, loss_fn: LossFp) -> Self {
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

    fn forward(&self, x_t: &ThoBatch<DI>, train: bool) -> ThoBatch<DO> {
        let net = self.net.lock();
        net.tho_forward(x_t, train)
    }

    fn x_f(&self) -> &F {
        self.x_f
    }

    fn loss_fn(&self) -> LossFp {
        self.loss_fn
    }

    fn set_lr(&self, lr: f64) {
        let mut opt = self.opt.lock();
        opt.set_lr(lr);
    }
}

impl<'xf, F, DI, DO, C> Stage for TrainStep<'xf, F, DI, DO, C>
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

impl<'xf, F, DI, DO, C> SyncStage for TrainStep<'xf, F, DI, DO, C>
where
    DI: Dims,
    DO: Dims,
    C: RunCtx,
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI> + Send + Sync + 'static,
{
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: Arc<RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {
        let _ = ctx;
        let (x_t_0, expected_y) = input.take().ok_or_else(|| ThoosaError::new("TrainStepInput already consumed"))?;

        debug!("train: device={:?} x_t_0.size={:?} expected_y.size={:?}", self.device, x_t_0.size(), expected_y.size());
        debug!("train: 0. cuda_mem={:?}", cuda_mem());

        let x_t = (self.x_f)(x_t_0);

        debug!("train: device={:?}  x_t.size={:?}", self.device, x_t.size());

        let predicted_y = self.forward(&x_t, true);
        debug!("train: 1. cuda_mem={:?}", cuda_mem());
        trace!("train: x_t=\n{x_t}");
        x_t.drop();

        let loss = (self.loss_fn)(predicted_y.as_ref(), &expected_y.as_ref());

        {
            let mut opt = self.opt.lock();

            debug!("train: 2. cuda_mem={:?}", cuda_mem());
            debug!("train: loss={loss}");
            trace!("train: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}",optimizer_mem_size(&opt));

            predicted_y.drop();
            drop(expected_y);
            debug!("train: 2.1 cuda_mem={:?}", cuda_mem());
            let t1 = Instant::now();

            opt.backward_step(&loss);

            debug!("train: 3. cuda_mem={:?}  optimizer_size={}",cuda_mem(),optimizer_mem_size(&opt));
            debug!("train: device={:?}  backward_step done in {}",self.device,t1.elapsed().as_millis());
        }

        Ok(())
    }
}



#[cfg(test)]
mod tests {

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_0(){

    }

}
