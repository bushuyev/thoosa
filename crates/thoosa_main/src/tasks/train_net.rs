use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use log::debug;
use parking_lot::RwLock;
use tch::nn::VarStore;
use caching_runner::common::cachable::{Cached, DevicesWrap};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage};
use thoosa_common::applicables::ThoDevice;
use thoosa_common::batches::AsyncDataBatch;
use thoosa_common::dims::Dims;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{opt_train_2, OptConfig};
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
}

impl TrainNetOutput {
    pub fn new(id: SimpleId) -> Self {
        Self { id }
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

pub struct TrainNet<C, L, B, BF, DX, DY, P, RC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY>,

    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
{
    batch:BF,
    opt_config:Arc<OptConfig>,
    run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >,
    pd:PhantomData<(C, L, DX, DY, BF, P, RC)>
}

impl<C, L, B, BF, DX, DY, P, RC> Debug for TrainNet<C, L, B, BF, DX, DY, P, RC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY>,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TrainNet").finish()
    }
}

impl<C, L, B, BF, DX, DY, P, RC> TrainNet<C, L, B, BF, DX, DY, P, RC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY>,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
{
    pub fn new(batch:BF, opt_config:Arc<OptConfig>,run_config_source: Arc<OptRunConfigs<Vec<NamedVal<P>>, Vec<NamedRange<P>>, Vec<Hyperparameter<P>>, P> >) -> Self {
        Self { batch, pd: PhantomData, opt_config, run_config_source }
    }
}

impl<C, L, B, BF, DX, DY, P, RC> Stage for TrainNet<C, L, B, BF, DX, DY, P, RC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims, DY: Dims,
    B:AsyncDataBatch<DX, DY>,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
{
    type Ctx = C;
    type In<'a> = TrainNetInput;
    type Out = TrainNetOutput;
}


impl<C, L, B, BF, DX, DY, P, RC> AsyncStage for TrainNet<C, L, B, BF, DX, DY, P, RC>
where
    L: for<'i> Cached<In<'i> = Resource, Out = ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static,
    DX: Dims + 'static, DY: Dims + 'static,
    B:AsyncDataBatch<DX, DY> + 'static,
    BF:Fn(&RC)->B +  Send + Sync + 'static,
    P: FieldName + Send + Sync + 'static,
    RC: RunConfig + From<std::vec::IntoIter<NamedVal<P>>> + Clone + Send + Sync + 'static,
{
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<RwLock<Self::Ctx>>) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>> + Send {
        let run_config_source = Arc::clone(&self.run_config_source);
        let device_idx = ctx.read().device_idx();
        async move {


            opt_train_2::<_, _, _, ()>(
                Arc::clone(&self.opt_config),
                async move |vs: &VarStore, run_config: RC, _listener| {





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
                    todo!()
                },
                run_config_source,
                ThoDevice::from(device_idx).into(),
                "labels",
            ).await.map(|(var_store, label_node_params, train_motifs_nets_run_config, min_loss)| {
                debug!("TrainLabelNet::run: label_node_params={label_node_params:?} train_motifs_nets_run_config={train_motifs_nets_run_config:?} min_loss={min_loss}");
                // TrainLabelNetsRunConfigResult::new(
                //     select_motifs_run_ids,//TODO
                //     label_,
                //     min_loss,
                //     label_node_params,
                //     train_motifs_nets_run_config,
                //     var_store,
                //
                // )
                Arc::new(DevicesWrap::wrap(TrainNetOutput::new(SimpleId("zzzz")), device_idx as u64))
            })
        }
    }
}


#[cfg(test)]
mod tests {

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_0(){

    }

}
