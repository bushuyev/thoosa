use std::marker::PhantomData;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::debug;
use parking_lot::lock_api::RwLock;
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::TrainMotifsNetsResult;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Adjustable, CachableIdTr, SimpleId, WithId};
use thoosa_store::Store;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::run_ctx_2::MotifsCtx;

#[derive(Debug, Clone, DummyData)]
pub struct ApplyMotifNetsInput {
    id:SimpleId,
    train_motif_nets_run_id: Vec<RunId>,
}

impl ApplyMotifNetsInput {
    pub fn new(id: SimpleId, train_motif_nets_run_id: Vec<RunId>) -> Self {
        Self { id, train_motif_nets_run_id }
    }
}
impl Input for ApplyMotifNetsInput {}


#[derive(Debug)]
pub struct ApplyMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    id:SimpleId,
    loader: Arc<L>,
    ws:Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
    store:Arc<parking_lot::RwLock<Box<dyn Store>>>,
    pd:PhantomData<C>
}

impl <L, C> ApplyMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    pub fn new(
        id: SimpleId,
        ws: &Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
        store: &Arc<parking_lot::RwLock<Box<dyn Store>>>,
        loader: Arc<L>
    ) -> Self {

        // ws.read().load_motif_nets()

        Self {
            id,
            loader,
            ws: Arc::clone(&ws),
            store:Arc::clone(&store),
            pd: PhantomData,
        }
    }
}

impl  WithId for ApplyMotifNetsInput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug, Clone, DummyData)]
pub struct ApplyMotifNetsOutput {
    id:SimpleId,
}

impl ApplyMotifNetsOutput {
    pub fn new(id: SimpleId) -> Self {
        Self { id }
    }
}

impl  WithId for ApplyMotifNetsOutput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for ApplyMotifNetsOutput {}

impl <L, C> OutId for ApplyMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        format!("ApplyMotifNets_{}_{}", self.id, input_id).into()
    }
}

impl <L, C> Stage for ApplyMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ApplyMotifNetsInput;
    type Out = ApplyMotifNetsOutput;
}

#[async_trait]
impl <L, C> AsyncStage for ApplyMotifNets<L, C>
where
    L: for <'i> AsyncStage<In<'i>=Resource, Out=ThoTensor<ChannelsData2, SimpleId>, Ctx=C> + Cached + OutId + 'static,
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area, ctx_name) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area(), guard.name())
        };

        let input = input.clone_from_cpu().await?;

        futures::stream::iter(

            input.train_motif_nets_run_id
                .iter()
                .flat_map(|train_motif_nets_run_id|
                     self.store.read()
                        .get_input_run_ids(*train_motif_nets_run_id, None).expect("motif_selection_id failed")
                )
                .flat_map(|select_motifs_run_id| {
                    debug!("ApplyMotifNets::run: select_motifs_run_id={select_motifs_run_id}");

                    self.store.read().list_selected_motifs(select_motifs_run_id, self.ws.read().channels()).expect("list_selected_motifs")
                        .into_iter()
                        .enumerate()
                        .flat_map(|(channel, channels_motifs)| {
                            channels_motifs
                                .into_iter()
                                .map(move |(motif_id, _)| (channel, motif_id))
                        })
                })

        ).map( |(channel, motif_id)| {
            async move {
                debug!("ApplyMotifNets::run: channel={channel} channels_motifs={motif_id}");

                Ok::<_, ThoosaError>(0)
            }
        })
        .buffer_unordered(10)
        .try_collect::<Vec<_>>().await?;

        let channels = self.ws.read().channels();

        let out_key = {//TODO move to with_data!
            let out_id = self.out_id(WithId::id(&input));
            Arc::new(out_id) as Arc<dyn CachableIdTr>
        };


        Ok(Arc::new(DevicesWrap::wrap_cpu(ApplyMotifNetsOutput::new("".into()))))

    }
}


#[derive(Debug)]
pub struct ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable
{
    pd:PhantomData<C>
}

impl <C>  Stage for ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData2, SimpleId>;
    type Out = ThoTensor<ChannelsData2, SimpleId>;
}

impl <C> SyncStage for ApplyMotifNet<C> where
    C: RunCtx + MotifsCtx + Adjustable
{
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use log::LevelFilter::Trace;
    use parking_lot::RwLock;
    use tch::Device::Cpu;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::mock_run_ctx;
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::utils::{lin_t, motif_test_net, reset_print_options};
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::{TensorIdSliceIdx, ThoosaError};
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::apply_motif_nets::{ApplyMotifNets, ApplyMotifNetsInput};
    use crate::tasks::tensor_loader::mock_tensor_loader;

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_0() -> Result<(), ThoosaError>{
        log::set_max_level(Trace);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let input_n_0 = 1000;
        let motif_width = 120;
        let train_motifs_run_id:RunId = 1;
        let find_motifs_run_id:RunId = 2;
        let select_motifs_run_id:RunId = 3;
        let motif_id = 10;

        let resource_a = Resource::new(1, 1, "tensor_a", input_n_0);
        let resource_b = Resource::new(2, 1, "tensor_b", input_n_0);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, 100, input_n_0 as u32)
                .with_motif_nets(vec![
                    vec![
                        //channel 0
                        motif_test_net(motif_width, 1., Cpu),
                        motif_test_net(motif_width, 1., Cpu),
                    ],
                    vec![
                        //channel 1
                        motif_test_net(motif_width, 0., Cpu),
                        motif_test_net(motif_width, 0., Cpu),
                        motif_test_net(motif_width, 10., Cpu),
                    ],
                    vec![],
                ])
                .build(),
        ));

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new(Some(find_motifs_run_id), 50, 100, None, None).to_properties_iter(None).collect()),
                )

                .with_get_input_run_ids([
                    (train_motifs_run_id, Ok(vec![select_motifs_run_id])),
                    (select_motifs_run_id, Ok(vec![find_motifs_run_id]))
                ])
                .with_get_motif_meta( [(motif_id, Ok((0usize, find_motifs_run_id)))])
                .with_list_selected_motifs([
                    (select_motifs_run_id, Ok(vec![vec![(motif_id, 1.0f64)]]))
                ])
                .build()));


        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            "1".into(),
            &ws,
            &store,
            Arc::new(
                mock_tensor_loader::build()
                    .with_out_id()
                    .with_run_cached([
                        (resource_a, lin_t(1000).view((1,-1))),
                        (resource_b, lin_t(1000).view((1,-1)))
                    ])
                    .finish(),
            ),
        ));

        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ApplyMotifNetsInput::new("".into(), vec![train_motifs_run_id]), 0)),
            Arc::new(RwLock::new(mock_run_ctx::default(1)))
        ).await?;


        Ok(())
    }
}