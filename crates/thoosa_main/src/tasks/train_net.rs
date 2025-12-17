use std::fmt::{Debug, Formatter};
use std::io::Write;
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;
use futures_util::StreamExt;
use log::{debug, info, trace};
use parking_lot::RwLock;
use tch::Device;
use tch::Device::{Cpu, Cuda};
use tch::nn::{AdamW, Optimizer, VarStore};
use caching_runner::common::cachable::{Cached, DevicesWrap};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
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

                    let r = train_3(
                        &ThoDevice::from(device_idx),
                        &batch,
                        node,
                        self.loss_fn,
                        (self.cycles)(&run_config), //TODO self.opt_config.,
                        optimizer,
                        &self.x_f,
                        None
                    ).await?;

                    Ok((r, params))

                },
                run_config_source,
                ThoDevice::from(device_idx).into(),
                "labels",
            ).await?;

            Ok(Arc::new(DevicesWrap::wrap_cpu(TrainNetOutput::new("".into(), var_store, node_params, run_config, loss))))
        }
    }
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="train_3"))]
pub async fn train_3<F, DI: Dims, DO: Dims>(
    device: &Device,
    data: &dyn AsyncDataBatch<DI, DO>,
    net: Box<dyn ThoModule<Batch<DI>, Batch<DO>>>,
    loss_fn: LossFp,
    cycles: u32,
    mut opt: Optimizer,
    x_f: &F,
    mut listener: Option<&mut dyn TrainListener>,
) -> Result<f32, ThoosaError>
where
    F: Fn(ThoBatch<DI>) -> ThoBatch<DI>,
{
    debug!("train: device={:?} cycles={}", device, cycles);
    debug!("train: opt={opt:?} optimizer_size={}",optimizer_mem_size(&opt));

    let mut last_valid_loss = [0f32; 3];

    for idx in 1..cycles {
        debug!("train:  0. cuda_mem={:?}", cuda_mem());
        let mut batches = data.train_batch_stream().await;
        debug!("train:  1. cuda_mem={:?}", cuda_mem());
        while let Some(Ok((x_t_0,expected_y))) = batches.next().await {
            debug!("train: device={:?} x_t_0.size={:?} expected_y.size={:?}", device, x_t_0.size(), expected_y.size());
            debug!("train: 0. cuda_mem={:?}", cuda_mem());

            let x_t = x_f(x_t_0);

            debug!("train: device={:?}  x_t.size={:?}",device,x_t.size());

            let predicted_y = net.tho_forward(&x_t, true);
            debug!("train: 1. cuda_mem={:?}", cuda_mem());
            trace!("train: x_t=\n{x_t}");
            x_t.drop();
            // unsafe { at_autocast_clear_cache(); }

            let loss = loss_fn(predicted_y.as_ref(), &expected_y.as_ref());

            debug!("train: 2. cuda_mem={:?}", cuda_mem());
            debug!("train: loss={loss}");
            trace!("train: loss={loss} predicted_y=\n{predicted_y} y_t=\n{expected_y} optimizer_size={}",optimizer_mem_size(&opt));

            predicted_y.drop();
            drop(expected_y);
            debug!("train: 2.1 cuda_mem={:?}", cuda_mem());
            let t1 = Instant::now();

            opt.backward_step(&loss);

            debug!("train: 3. cuda_mem={:?}  optimizer_size={}",cuda_mem(),optimizer_mem_size(&opt));
            debug!("train: device={:?}  backward_step done in {}",device,t1.elapsed().as_millis());

            if let  Some(listener) = listener.as_mut() {
                listener.on_cycle(0, 0.0).await;
            }
        }




        if idx % 10 == 0 || idx == cycles - 1 {

            debug!("train:  0. cuda_mem={:?}", cuda_mem());
            let mut batches = data.valid_batch_stream().await;
            debug!("train:  1. cuda_mem={:?}", cuda_mem());
            while let Some(Ok((x_t_0,y_t))) = batches.next().await {
                if let Some(rs) = tch::no_grad(||{

                    let x_t = x_f(x_t_0);

                    let pred_y = net.tho_forward(&x_t, false);

                    let valid_loss = loss_fn(&pred_y.as_ref(), &y_t.as_ref());
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
                            opt.set_lr(lr);
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

    Ok(last_valid_loss.into_iter().sum::<f32>() / last_valid_loss.len() as f32)
}



#[cfg(test)]
mod tests {

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_0(){

    }

}
