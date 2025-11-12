use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::debug;
use parking_lot::lock_api::RwLock;
use caching_runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use caching_runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use caching_runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use caching_runner::global_ctx::GlobalCtx;
use caching_runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use caching_runner::permits::{PermitStep, MAX_PERMITS};
use thoosa_common::dims::{Batch, MotifProbability, Slice};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore, TrainMotifsNetsResult};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, MonoMetrics, Select, SimpleId, WithId};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput};
use crate::tasks::resources_motifs_clc_dist::ClcRresourcesMotifsDistOutput;
use crate::tasks::run_ctx_2::MotifsCtx;

#[derive(Debug, Clone, DummyData)]
pub struct ApplyMotifNetsInput {
    id:SimpleId,
    select_motifs_run_id: Vec<RunId>,

}

impl ApplyMotifNetsInput {
    pub fn new(id: SimpleId, select_motifs_run_id: Vec<RunId>) -> Self {
        Self { id, select_motifs_run_id }
    }
}
impl Input for ApplyMotifNetsInput {}


#[derive(Debug)]
pub struct ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    id:SimpleId,
    ws:Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
    store:Arc<parking_lot::RwLock<Box<dyn Store>>>,
    pd:PhantomData<C>,
    load_motif_net:Arc<LoadMotifNet<C>>,//TODO not needed?
    loaded_motif_nets:Vec<Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Slice, MotifProbability>, MotifNetParams>>>>,
    apply_motif_net: Arc<ApplyMotifNet<C>>
}

impl <C>ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    pub async fn new<R>(
        id: SimpleId,
        ws: &Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
        store: &Arc<parking_lot::RwLock<Box<dyn Store>>>,
        select_motifs_run_id: Vec<RunId>,
        stage_ctx:Arc<parking_lot::RwLock<R>>
    ) -> Result<Self, ThoosaError>
    where
        R:RunCtx + MotifsCtx + Adjustable + 'static
    {
        let load_motif_net = Arc::new(LoadMotifNet::<R>::new(&ws));

        let loaded_motif_nets = futures::stream::iter(

            select_motifs_run_id
                .iter()

                .flat_map(|select_motifs_run_id| {
                    debug!("ApplyMotifNets::run: select_motifs_run_id={select_motifs_run_id}");

                    store.read().list_selected_motifs(*select_motifs_run_id, ws.read().channels()).expect("list_selected_motifs")
                        .into_iter()
                        .enumerate()
                        .flat_map(|(channel, channels_motifs)| {
                            channels_motifs
                                .into_iter()
                                .map(move |(motif_id, _)| (channel, motif_id))
                        })
                })

        ).map( |(channel, motif_id)| {
            let ws = Arc::clone(&ws);
            let load_motif_net = Arc::clone(&load_motif_net);
            let stage_ctx = Arc::clone(&stage_ctx);
            async move {

                debug!("ApplyMotifNets::run: channel={channel} channels_motifs={motif_id}");

                load_motif_net.run_cached(Arc::new(DevicesWrap::wrap_cpu(LoadMotifNetInput::new(motif_id, channel as u32))), stage_ctx).await
            }
        })
            .buffer_unordered(10)
            .try_collect::<Vec<_>>().await?;



        Ok(Self {
            id,
            ws: Arc::clone(&ws),
            store:Arc::clone(&store),
            pd: PhantomData,
            load_motif_net: Arc::new(LoadMotifNet::new(&ws)),
            loaded_motif_nets,
            apply_motif_net: Arc::new(ApplyMotifNet::new())
        })
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

impl <C>OutId for ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        format!("ApplyMotifNets_{}_{}", self.id, input_id).into()
    }
}

impl <C>Stage for ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ThoTensor<Batch<ChannelsData2>, SimpleId>;
    type Out = ApplyMotifNetsOutput;
}

#[derive(Debug)]
struct MotifsProbabilitiesOutputBuilder {

}

impl MotifsProbabilitiesOutputBuilder {
    pub fn new() -> Self {
        Self {}
    }
}

impl OutputBuilder for MotifsProbabilitiesOutputBuilder {
    type O = ThoTensor<ChannelsData2, SimpleId>;

    fn add_partial_output(&mut self, o: Self::O) {
        todo!()
    }

    fn build(self: Box<Self>) -> Self::O {
        todo!()
    }
}

#[async_trait]
impl <C>AsyncStage for ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {
        let (device_idx, cache_area, ctx_name) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area(), guard.name())
        };

        let out_key = {
            let out_id = self.out_id(WithId::id(&input));
            Arc::new(out_id) as Arc<dyn CachableIdTr>
        };
        let metrics = input.size().clone();

        let wrapper = Arc::new(ApplyMotifNetInputDeviceWrapper::new(input, self.loaded_motif_nets.clone()));
        let output_builder = Box::new(MotifsProbabilitiesOutputBuilder::new());


        let map = HashMap::new();

        let oom_handler = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(PermitsAdjuster::new(MAX_PERMITS, *PermitStep::TWENTY, move |n| GlobalCtx::global().acquire_many(device_idx, n, out_key.clone())))),
                Adjusters::RunContext(Box::new(DumpToCpu::new())),
                Adjusters::Input(Box::new(SplitsAdjuster::<ApplyMotifNetInputDeviceWrapper, std::ops::Range<MonoMetrics<0, 3>>>::new(metrics))),
            ], None).run(
            &self.apply_motif_net,
            Arc::clone(&wrapper),
            ctx,
            Contexts::new(
                SplitsAdjusterContextImpl::new(wrapper.size().into(), output_builder),
                NoopRunnableContext {},
                PermitsAndDumpToCpuAdjustableContex::new(
                    PermitsAdjustContextImpl::new(ctx_name),
                    DumpToCpuContextImpl::new(&map),
                ),
            ),
        ).await?;



        Ok(Arc::new(DevicesWrap::wrap_cpu(ApplyMotifNetsOutput::new("".into()))))

    }
}

#[derive(Debug)]
struct ApplyMotifNetInputDeviceWrapper {
    batch: Arc<DevicesWrap<ThoTensor<Batch<ChannelsData2>, SimpleId>>>,
    motif_nets: Vec<Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Slice, MotifProbability>, MotifNetParams>>>>
}

impl ApplyMotifNetInputDeviceWrapper {
    pub fn new(batch: Arc<DevicesWrap<ThoTensor<Batch<ChannelsData2>, SimpleId>>>, motif_nets: Vec<Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Slice, MotifProbability>, MotifNetParams>>>>) -> Self {
        Self { batch, motif_nets }
    }
}

impl WithId for ApplyMotifNetInputDeviceWrapper {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl WithDataRef for ApplyMotifNetInputDeviceWrapper {
    type O<'o> = ApplyMotifNetInput<'o>;

    async fn with_data_ref<R, F>(&self, to_device_idx: usize, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
            for<'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>
    {
        let t_1_batch= Arc::clone(&self.batch);
        let motif_nets = self.motif_nets.clone();
        with_data!(to_device_idx, t_1_batch, [motif_nets];
            |
             batch_v:&ThoTensor<Batch<ChannelsData2>, SimpleId>,
             batched_motifs_output:&[&DataWithExtra<ThoModuleWithStore<Slice, MotifProbability>, MotifNetParams>]
            |{

                do_run(&ApplyMotifNetInput::new(
                    batch_v,
                    batched_motifs_output,
                ))

        })
    }

    async fn with_selected_data_ref<'o, S, R, F>(&self, to_device_idx: usize, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
        S: Clone + Debug + 'static,
        Self::O<'o>: Select<S, Output=Self::O<'o>>,
        for<'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>
    {
        todo!()
    }
}

#[async_trait]
impl CopyToDevice for ApplyMotifNetInputDeviceWrapper {
    async fn copy_to_device(&self, dst_device: usize) -> Self {
        todo!()
    }
}

impl Data for ApplyMotifNetInputDeviceWrapper {
    type Metrics = MonoMetrics<0, 3>;

    fn size(&self) -> Self::Metrics {
        self.batch.size().clone()
    }

    fn device(&self) -> usize {
        // 0 //TODOself.batch.device
        todo!()
    }
}


impl Adjustable for ApplyMotifNetInputDeviceWrapper {}

#[derive(Debug)]
struct ApplyMotifNetInput<'a> {
    batch: &'a ThoTensor<Batch<ChannelsData2>, SimpleId>,
    motif_nets: &'a [&'a DataWithExtra<ThoModuleWithStore<Slice, MotifProbability>, MotifNetParams>]
}

#[async_trait]
impl CopyToDevice for ApplyMotifNetInput<'_> {
    async fn copy_to_device(&self, dst_device: usize) -> Self {
        todo!()
    }
}

impl Data for ApplyMotifNetInput<'_> {
    type Metrics = MonoMetrics<0, 3>;

    fn size(&self) -> Self::Metrics {
        todo!()
    }

    fn device(&self) -> usize {
        todo!()
    }
}

impl <'a> ApplyMotifNetInput<'a> {
    pub fn new(batch: &'a ThoTensor<Batch<ChannelsData2>, SimpleId>, motif_nets: &'a [&'a DataWithExtra<ThoModuleWithStore<Slice, MotifProbability>, MotifNetParams>]) -> Self {
        Self {  batch, motif_nets }
    }
}

impl <'a> Input for ApplyMotifNetInput<'a> {}
impl <'a> WithId for ApplyMotifNetInput<'a> {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl Select<std::ops::Range<MonoMetrics<0, 3>>> for ApplyMotifNetInput<'_> {
    type Output = Self;

    fn subview(&self, selector: Range<MonoMetrics<0, 3>>) -> Result<Self::Output, ThoosaError> {
        todo!()
    }
}


#[derive(Debug)]
struct ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable
{
    pd:PhantomData<C>
}
impl <C> ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable
{
    pub fn new() -> Self {
        Self { pd:PhantomData }
    }
}

impl <C> Adjustable for ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable
{

}

impl <C> OutId for ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        format!("ApplyMotifNet: {:?}", input_id).into()
    }
}


impl <C>  Stage for ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable{
    type Ctx = C;
    type In<'a> = ApplyMotifNetInput<'a>;
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
    use tch::kind::FLOAT_CPU;
    use tch::Tensor;
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::mock_run_ctx;
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::tho_tensor::ThoTensor;
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

        let find_motifs_run_id:RunId = 1;
        let select_motifs_run_id:RunId = 2;

        let train_motifs_run_id:RunId = 3;
        let motif_id = 1;

        let resource_a = Resource::new(1, 1, "tensor_a", input_n_0);
        let resource_b = Resource::new(2, 1, "tensor_b", input_n_0);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, 100, input_n_0 as u32)
                .with_load_motif_nets(vec![
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
            // Arc::new(
            //     mock_tensor_loader::build()
            //         .with_out_id()
            //         .with_run_cached([
            //             (resource_a, lin_t(1000).view((1,-1))),
            //             (resource_b, lin_t(1000).view((1,-1)))
            //         ])
            //         .finish(),
            // ),
            vec![select_motifs_run_id],
            Arc::new(RwLock::new(mock_run_ctx::default(1)))
        ).await?);

        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new("test_batch".into(), Tensor::rand(&[11, 5, 1000], FLOAT_CPU)), 0)),
            Arc::new(RwLock::new(mock_run_ctx::default(1)))
        ).await?;


        Ok(())
    }
}