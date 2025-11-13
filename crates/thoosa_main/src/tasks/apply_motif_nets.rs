use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::debug;
use parking_lot::lock_api::RwLock;
use tch::IndexOp;
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
use thoosa_common::my_tch::ThoModule;
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
use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput, MotifNetMeta};
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
    loaded_motif_nets:Vec<Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>>,
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
    type O = ThoTensor<MotifProbability, SimpleId>;

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

        let out_id = Arc::new(self.out_id(WithId::id(&input))) as Arc<dyn CachableIdTr>;


        futures::stream::iter(self.loaded_motif_nets.iter())
            .map(|motif_net|{
                let motif_net = Arc::clone(motif_net);
                let metrics = input.size().clone();
                let input = Arc::clone(&input);
                let out_id = Arc::clone(&out_id);
                let ctx = Arc::clone(&ctx);
                let apply_motif_net = Arc::clone(&self.apply_motif_net);

                async move {
                    let wrapper = Arc::new(ApplyMotifNetInputDeviceWrapper::new(input, motif_net));
                    let output_builder = Box::new(MotifsProbabilitiesOutputBuilder::new());


                    let map = HashMap::new();

                    OOMHandler::new(
                        vec![
                            Adjusters::RunContext(Box::new(PermitsAdjuster::new(MAX_PERMITS, *PermitStep::TWENTY, move |n| GlobalCtx::global().acquire_many(device_idx, n, Arc::clone(&out_id))))),
                            Adjusters::RunContext(Box::new(DumpToCpu::new())),
                            Adjusters::Input(Box::new(SplitsAdjuster::<ApplyMotifNetInputDeviceWrapper, std::ops::Range<MonoMetrics<0, 3>>>::new(metrics))),
                        ], None).run(
                        &apply_motif_net,
                        Arc::clone(&wrapper),
                        ctx,
                        Contexts::new(
                            SplitsAdjusterContextImpl::new(wrapper._size().into(), output_builder),
                            NoopRunnableContext {},
                            PermitsAndDumpToCpuAdjustableContex::new(
                                PermitsAdjustContextImpl::new(ctx_name),
                                DumpToCpuContextImpl::new(&map),
                            ),
                        ),
                    ).await
                }
            })
            .buffer_unordered(10)
            .try_collect::<Vec<_>>().await?;


        Ok(Arc::new(DevicesWrap::wrap_cpu(ApplyMotifNetsOutput::new("".into()))))

    }
}

#[derive(Debug)]
struct ApplyMotifNetInputDeviceWrapper {
    batch: Arc<DevicesWrap<ThoTensor<Batch<ChannelsData2>, SimpleId>>>,
    motif_net: Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>
}

impl ApplyMotifNetInputDeviceWrapper {
    pub fn new(batch: Arc<DevicesWrap<ThoTensor<Batch<ChannelsData2>, SimpleId>>>, motif_net: Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>) -> Self {
        Self { batch, motif_net }
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
        let motif_net = Arc::clone(&self.motif_net);

        with_data!(to_device_idx, t_1_batch, motif_net;
            |
             batch_v:&ThoTensor<Batch<ChannelsData2>, SimpleId>,
             batched_motifs_output:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>
            |{

                do_run(&ApplyMotifNetInput::new(
                    batch_v,
                    batched_motifs_output,
                    100//TODO slice step
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
    slice_step:u32,
    batch: &'a ThoTensor<Batch<ChannelsData2>, SimpleId>,
    motif_net: &'a DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>
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
    pub fn new(batch: &'a ThoTensor<Batch<ChannelsData2>, SimpleId>, motif_net: &'a DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>, slice_step: u32) -> Self {
        Self {  batch, motif_net, slice_step }
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
    type Out = ThoTensor<MotifProbability, SimpleId>;
}

impl <C> SyncStage for ApplyMotifNet<C> where
    C: RunCtx + MotifsCtx + Adjustable
{
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, _ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {

        let params = input.motif_net.extra();

        let slices = input.batch.as_ref().i((.., params.channel() as i64, ..)).unfold(1, params.motif_net_params().num_features() as i64, input.slice_step as i64);
        Ok(input.motif_net.data().tho_forward(&ThoTensor::new("".into(), slices), false))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use log::debug;
    use log::LevelFilter::Trace;
    use parking_lot::RwLock;
    use plotters::prelude::{BLACK, RED};
    use plotters::style::{BLUE, GREEN};
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::{nn, Device, IndexOp, Kind, Tensor};
    use tch::nn::{LinearConfig, VarStore};
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::mock_run_ctx;
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::builder::SeqBuilder;
    use thoosa_common::dims::{AnyDims, Batch, Dims, MotifProbability, Slice};
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::my_tch::ThoModule;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore};
    use thoosa_common::utils::{lin_t, plot_series_with_lanes, reset_print_options, test_net, vec_f32_from};
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::{CopyToDevice, MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::traits::SimpleId;
    use thoosa_store::MockStoreBuilder;
    use crate::tasks::apply_motif_nets::{ApplyMotifNets, ApplyMotifNetsInput};
    use crate::tasks::select_motifs::tests::{motifs_tensor, resource_tensor};
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

        let motif_t = Arc::new(ThoTensor::new("".into(), motifs_tensor(100, 3, MotifKind::Pulse)));
        let resource_a = Resource::new(1, 1, "tensor_a", input_n_0);
        let resource_b = Resource::new(2, 1, "tensor_b", input_n_0);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, 100, input_n_0 as u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        motif_test_net_(Cpu, Arc::clone(&motif_t)),
                        motif_test_net_(Cpu, Arc::clone(&motif_t)),
                    ],
                    vec![
                        //channel 1
                        motif_test_net_(Cpu, Arc::clone(&motif_t)),
                        motif_test_net_(Cpu, Arc::clone(&motif_t)),
                        motif_test_net_(Cpu, Arc::clone(&motif_t)),
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

            vec![select_motifs_run_id],
            Arc::new(RwLock::new(mock_run_ctx::default(1)))
        ).await?);

        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new("test_batch".into(), Tensor::rand(&[11, 5, 1000], FLOAT_CPU)), 0)),
            Arc::new(RwLock::new(mock_run_ctx::default(1)))
        ).await?;


        Ok(())
    }


    pub fn test_motif_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
        num_features: u32,
        k: f64,
        device: Device,
        descr: &str,
        template:&Tensor,
    ) -> Result<Box<ThoModuleWithStore<DI, DO>>, ThoosaError> {
        assert_eq!(template.size().len(), 1, "template must be 1-D");
        let l = template.size()[0];

        // mean-center and unit-norm the template (helps with DC/scale variations)
        let t = template
            .to_device(device)
            .to_kind(Kind::Float);
        let t = &t - t.mean(Kind::Float);
        let t = &t / (t.pow_tensor_scalar(2).sum(Kind::Float).sqrt() + 1e-12);

        let vs = VarStore::new(device);
        let mut cfg = nn::ConvConfig { bias: false, ..Default::default() };


        let tho_tensor:Arc<ThoTensor<AnyDims, SimpleId>> = Arc::new(ThoTensor::new("".into(), t));

        let module = SeqBuilder::new(num_features, None)

            .split(vec![
                SeqBuilder::new(num_features, None).conv_1d("conv", l as u32, 1, 1, cfg, Some(tho_tensor)).end_seq(),
                SeqBuilder::new(num_features, None)
                    .closure(
                        Arc::new(Box::new(move |t: &Tensor| {
                            t.pow_tensor_scalar(2)
                        })),
                        None,
                    )
                    .conv_1d("energy", l as u32, 1, 1, cfg, Some(Arc::new(ThoTensor::new("".into(), Tensor::ones([1, 1, l], FLOAT_CPU) ))))
                    .closure(
                        Arc::new(Box::new(move |t: &Tensor| {
                            t.clamp_min(1e-12)//.sqrt()
                        })),
                        None,
                    )
                    .end_seq()
            ],
                   Box::new(move |t: &Tensor|{
                       vec![
                           t.shallow_clone(),
                           t.shallow_clone()
                       ]
                   }),
                   Box::new(move |num_denom: Vec<Tensor>| {
                       let [num, denom] = <[Tensor; 2]>::try_from(num_denom).unwrap();

                       (num / denom).squeeze_dim(1)
                   })
            ).end_seq()
            .build_tch_module(&vs.root())?;

        Ok(Box::new(ThoModuleWithStore::with_factory(
            "test_net".into(),
            module,
            vs,
            Some(Arc::new( move |path:&nn::Path| {
                //for ws mocks & copy_to_device
                SeqBuilder::new(num_features, None)
                    .conv_1d("conv", l as u32, 1, 1, cfg, None)
                    .end_seq()
                    .build_tch_module(path)
            }))
        )))

    }


    #[test]
    fn test_test_net() -> Result<(), ThoosaError> {
        let motif_t = motifs_tensor(100, 3, MotifKind::Pulse);
        let (resource_a_t, resource_a_intervals) = resource_tensor(2000, &[vec![(3, 100, MotifKind::Pulse)]], 200)?;

        debug!("resource_a_t.size={:?}", resource_a_t.size());

        let test_dir = Path::new("/tmp/test_test_net"); //.join(cdt_for_dir());


        let x = test_motif_net::<AnyDims, AnyDims>(100, 1., Cpu, "test", &motif_t.i((0, ..)))?;

        let out = x.tho_forward(&ThoTensor::new("".into(), resource_a_t.shallow_clone()), false).t;//.i((0, .., 0));//channel[1], slices, vals[1]

        /*plot_series_with_lanes(
            vec![
                (format!("Tensor"), BLACK, vec_f32_from(&resource_a_t.i((0, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>()),
                (format!("Motif_0"), BLUE, vec_f32_from(&motif_t.i((0, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>()),
                // (format!("Results"), RED, vec_f32_from(&out.transpose(1,2).upsample_linear1d(&[1000], false, None)).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>()),
            ],
            format!("sig").as_str(),
            (2000, 1000),
            &test_dir.join(&format!("input_{}.png", 1)),
            &[(0, 1000)],
        )?;*/

        debug!("out.size={:?}", out.size());

        plot_series_with_lanes(
            vec![
                (format!("Results"), RED, vec_f32_from(&out.i((0, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>()),
            ],
            format!("sig").as_str(),
            (2000, 1000),
            &test_dir.join(&format!("matches_{}.png", 1)),
            &[(0, 1000)],
        )?;

        debug!("resource_a_intervals={resource_a_intervals:?}");
        let (topk_0, topk_1) = out.topk(3 as i64, 1, true, true);
        debug!("topk=\n{topk_0} \n{topk_1}");

        //0 chanel/0 motif
        assert!(
            Tensor::from_slice(resource_a_intervals[0][0].iter().map(|i|i.start as i64).collect::<Vec<i64>>().as_slice())
                .allclose(&topk_1.i(0), 10., 10., false)
        );

        Ok(())
    }

    //TODO merge with old one
    pub fn motif_test_net_(
        device: Device,
        motif_t: Arc<ThoTensor<Batch<Slice>, SimpleId>>,
    ) -> (
        Box<dyn Fn() -> MotifNetParams + Sync + Send>,
        Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send>,
    ) {
        let motif_width = motif_t.as_ref().as_ref().size()[1] as u32;
        (
            Box::new(move || MotifNetParams::new(motif_width, 1)),
            Box::new(move || {
                *test_motif_net::<Batch<Slice>, MotifProbability>(motif_width, 1., device, "test", &motif_t.as_ref().as_ref().i((0, ..))).unwrap()//first slice
            }),
        )
    }

    #[tokio::test]
    async fn test_copy_to_device(){
        let cpu_net = *test_motif_net::<Batch<Slice>, MotifProbability>(123, 1., Cpu, "test", &Tensor::rand([123], FLOAT_CPU)).unwrap();//first slice

        let cuda_net = cpu_net.copy_to_device(1).await;
    }

}