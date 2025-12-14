use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use log::debug;
use parking_lot::RwLock;
use tch::Device::{Cpu, Cuda};
use tch::nn::{AdamW, Optimizer, VarStore};
use caching_runner::common::cachable::{Cached, DevicesWrap};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::applicables::ThoDevice;
use thoosa_common::batches::AsyncDataBatch;
use thoosa_common::dims::{Batch, Dims, ThoBatch};
use thoosa_common::my_tch::{LossFp, ThoModule};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{opt_train_2, CommonTrainingRunConfig, FixedNumSlicesNetArgs, FixedNumSlicesNetParams, LabelNetParams, LabelNetTypeArgs, LabelNetTypeParams, OptConfig, TrainLabelNetsRunConfig, TrainLabelNetsRunConfigResult};
use thoosa_common::utils::train_3;
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
pub struct TrainNetOutput {
    id:SimpleId,
    pub r:TrainLabelNetsRunConfigResult
}

impl TrainNetOutput {
    pub fn new(id: SimpleId, result: TrainLabelNetsRunConfigResult) -> Self {
        Self {
            id,
            r: result
        }
    }
}

impl WithId for TrainNetOutput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for TrainNetOutput{

}

pub struct TrainNet<C, L, B, BF, DX, DY, P, RC, NF, FO, XF>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY>,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
     XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
{
    batch:BF,
    net:NF,
    opt_config:Arc<OptConfig>,
    run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >,
    loss_fn:LossFp,
    optimizer:FO,
    x_f:XF,
    pd:PhantomData<(C, L, DX, DY, BF, P, RC)>
}

impl<C, L, B, BF, DX, DY, P, RC, NF, FO, XF> Debug for TrainNet<C, L, B, BF, DX, DY, P, RC, NF, FO, XF>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY>,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
     XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainNet").finish()
    }
}

impl<C, L, B, BF, DX, DY, P, RC, NF, FO, XF> TrainNet<C, L, B, BF, DX, DY, P, RC, NF, FO, XF>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY>,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
     XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
{
    pub fn new(batch:BF, net:NF,opt_config:Arc<OptConfig>,run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >, loss_fn:LossFp, optimizer:FO, x_f:XF) -> Self {
        Self { batch, net, pd: PhantomData, opt_config, run_config_source, loss_fn, optimizer, x_f }
    }
}

impl<C, L, B, BF, DX, DY, P, RC, NF, FO, XF> Stage for TrainNet<C, L, B, BF, DX, DY, P, RC, NF, FO, XF>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY>,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
     XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
{
    type Ctx = C;
    type In<'a> = TrainNetInput;
    type Out = TrainNetOutput;
}


impl<C, L, B, BF, DX, DY, P, RC, NF, FO, XF> AsyncStage for TrainNet<C, L, B, BF, DX, DY, P, RC, NF, FO, XF>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
    NF:Fn(&RC,&VarStore)->Result<Box<dyn ThoModule<Batch<DX>, Batch<DY>>>, ThoosaError> +  Send + Sync + 'static,
    FO:Fn(&RC,&VarStore)->Result<Optimizer, ThoosaError> +  Send + Sync + 'static,
    XF: Fn(ThoBatch<DX>) -> ThoBatch<DX>+  Send + Sync + 'static,
{
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>> + Send {
        let run_config_source = Arc::clone(&self.run_config_source);
        let device_idx = ctx.read().device_idx();

        async move {


            let (VarStore, N, R, f32) = opt_train_2::<_, _, _, LabelNetParams>(
                Arc::clone(&self.opt_config),
                async move |vs: &VarStore, run_config: RC, _listener| {



                    let batch = (self.batch)(&run_config);
                    let node = (self.net)(&run_config, vs)?;
                    let optimizer = (self.optimizer)(&run_config, vs)?;

                    let r = train_3(
                        &Cuda(0), //TODO
                        &batch,
                        node,
                        self.loss_fn,
                        100, //TODO self.opt_config.,
                        optimizer,
                        &self.x_f,
                        None
                    ).await?;

                    Ok((r, LabelNetParams::new(0, 0, 0, 0, LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new()))))

                    // let apply_motif_nets = Arc::clone(&apply_motif_nets);
                    // let label = label.clone();
                    // let resources_batch = (self.batch)(100);
                    //
                    // //TODO build proper LabelNetParams
                    // let num_layers = match run_config.net {
                    //     LabelNetTypeArgs::MaxPadding(MaxPaddingNetArgs{num_layers}) => {
                    //         num_layers
                    //     }
                    //     LabelNetTypeArgs::FixedNumSlices(FixedNumSlicesNetArgs{num_layers, ..}) => {
                    //         num_layers
                    //     }
                    // };
                    //
                    // let params = LabelNetParams::new(channels as u32, num_nets, num_layers, slices_len, LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams::new()));
                    // let loss = train_label_net_2(&run_config, &resources_batch, &ThoDevice::from(device_idx), vs, params.clone(), Arc::clone(&ctx), &label).await?;
                    //
                    // Ok((loss, params))
                    // todo!()
                },
                run_config_source,
                ThoDevice::from(device_idx).into(),
                "labels",
            ).await?;

            Ok(Arc::new(DevicesWrap::wrap_cpu(TrainNetOutput::new("".into(), TrainLabelNetsRunConfigResult::new(
                vec![],
                "xxx".to_string(),
                0.0,
                LabelNetParams::new(0, 0, 0, 0,  LabelNetTypeParams::FixedNumSlices(FixedNumSlicesNetParams{})),
                TrainLabelNetsRunConfig::new(None, CommonTrainingRunConfig::new(0,0, 0.0, 0, 0), LabelNetTypeArgs::FixedNumSlices(FixedNumSlicesNetArgs{ num_layers: 0, slices_len: 0 })),
                VarStore::new(Cpu)
            )))))
        }
    }
}


#[cfg(test)]
mod tests {

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_0(){

    }

}
