use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, Range};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;
use std::time::Instant;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, info};
use parking_lot::lock_api::RwLock;
use tch::{IndexOp, Tensor};
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tokio::task::JoinError;
use cached_derive::Cached;
use caching_runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use caching_runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use caching_runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
use caching_runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use caching_runner::common::{DropIf, OutId, PermitsCtx, RunCtx};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use caching_runner::global_ctx::GlobalCtx;
use caching_runner::oom_handler::{Adjusters, AsyncOOMHandler, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use caching_runner::permits::{PermitStep, MAX_PERMITS};
use thoosa_common::applicables::ThoDevice;
use thoosa_common::dims::{Batch, MotifProbability, Slice};
use thoosa_common::my_tch::ThoModule;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore, TrainMotifsNetsResult};
use thoosa_common::utils::pad_to;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{CachableId, Input, Output};
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{CopyToDevice, ThoosaError};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Metrics, MonoMetrics, Select, SimpleId, WithId};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use crate::tasks::cross_dims::{BatchChannelsData2Selector, ChannelsData2, ChannelsData2Selector};
use crate::tasks::load_motif_and_clc_dist::MotifSlicesSelector;
use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput, MotifNetMeta};
use crate::tasks::resources_motifs_clc_dist::ClcResourcesMotifsDistOutput;
use crate::tasks::run_ctx_2::MotifsCtx;


#[derive(Debug, Cached)]
pub struct ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    id:SimpleId,
    ws:Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
    store:Arc<parking_lot::RwLock<Box<dyn Store>>>,
    pd:PhantomData<C>,
    load_motif_net:Arc<LoadMotifNet<C>>,//TODO not needed?
    // loaded_motif_nets:Vec<Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>>,
    apply_motif_nets: Vec<Arc<ApplyMotifNet<C>>>,
    pub slice_step: u32,//should be in the input
    pub motif_width:u32,
}

impl <C> ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    pub async fn new<R>(
        id: SimpleId,
        ws: &Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
        store: &Arc<parking_lot::RwLock<Box<dyn Store>>>,
        select_motifs_run_id: Vec<RunId>,
        stage_ctx:Arc<parking_lot::RwLock<R>>,
        #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Result<Self, ThoosaError>
    where
        R:RunCtx + MotifsCtx + Adjustable + 'static
    {
        let device_idx = stage_ctx.read().device_idx();
        let load_motif_net = Arc::new(LoadMotifNet::<R>::new(&ws));

        let loaded_motif_nets = futures::stream::iter(

            select_motifs_run_id
                .iter()
                .flat_map(|select_motifs_run_id| {
                    debug!("ApplyMotifNets::run: select_motifs_run_id={select_motifs_run_id}");

                    store.read()
                        .list_selected_motifs(*select_motifs_run_id, ws.read().channels()).expect("list_selected_motifs")
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
        .buffered(10)//TODO config order is important to keep output tensor same
        .try_collect::<Vec<_>>().await?;


        //TODO all motif nets should have same width/step, handle different ones?
        let mut motif_width = HashSet::new();
        let mut slice_step = HashSet::new();
        for v in &loaded_motif_nets {
            with_data!(stage_ctx, device_idx, v; |v:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>|{
                motif_width.insert(v.extra().motif_net_params.num_features);
                slice_step.insert(v.extra().motif_net_params.stride);

                Ok::<_, ThoosaError>(())
            })?;
        };
        assert_eq!(motif_width.len(), 1);//TODO  group motif nets by width. all motifs should be of same width so that tensors can be batched for label training
        assert_eq!(slice_step.len(), 1);

        let motif_width = motif_width.into_iter().next().expect("no motif_width");
        let slice_step = slice_step.into_iter().next().expect("no motif_width");
        debug!("ApplyMotifNets::run: motif_width={motif_width:?}");

        Ok(Self {
            id,
            ws: Arc::clone(&ws),
            store:Arc::clone(&store),
            pd: PhantomData,
            load_motif_net: Arc::new(LoadMotifNet::new(&ws)),
            // loaded_motif_nets,
            apply_motif_nets: loaded_motif_nets
                .into_iter()
                .map(|net| Arc::new(ApplyMotifNet::new(
                    net,
                    slice_step,//TODO shouldn't be same as one used when training motif net
                    #[cfg(test)]
                    fail_if_size_bigger_than
                ))).collect(),
            slice_step,
            motif_width
        })
    }

    pub fn num_nets(&self) -> usize {
        self.apply_motif_nets.len()
    }
}

impl <C>Stage for ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData2, SimpleId>;
    type Out = ThoTensor<MotifProbability, SimpleId>;
}


// #[async_trait]
impl <C>AsyncStage for ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{

    // #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyMotifNets::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Arc<DevicesWrap<Self::Out>>, ThoosaError> {

        let (device_idx, cache_area, ctx_name) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area(), guard.name())
        };

        let out_id = Arc::new(self.out_id(WithId::id(&input))) as Arc<dyn CachableIdTr>;

        debug!("ApplyMotifNets::run: self.loaded_motif_nets.len={:?}", self.apply_motif_nets.len());



        let apply_motif_nets = self.apply_motif_nets.clone();
        let min_split_len = self.motif_width as i64;
        let overlap_len = (self.motif_width  - self.slice_step) as i64;
        let matches = futures::stream::iter(apply_motif_nets.into_iter())
            .map(|apply_motif_net|{
                let input = Arc::clone(&input);
                let out_id = Arc::clone(&out_id);
                let ctx = Arc::clone(&ctx);
                let slice_step = self.slice_step;

                let slices = ((input.size().dim::<1>() as u32 - self.motif_width)/slice_step) as i64 + 1;

                debug!("ApplyMotifNets::run: input.size={:?} self.motif_width={:?} slice_step={slice_step} slices={slices} min_split_len={min_split_len} overlap_len={overlap_len}", input.size(), self.motif_width);

                tokio::spawn(async move {
                    let map = HashMap::new();

                     AsyncOOMHandler::new(
                         vec![
                             Adjusters::RunContext(Box::new(PermitsAdjuster::new(MAX_PERMITS, *PermitStep::TWENTY, move |n| GlobalCtx::global().acquire_many(device_idx, n, Arc::clone(&out_id))))),
                             Adjusters::RunContext(Box::new(DumpToCpu::new())),
                             Adjusters::Input(Box::new(SplitsAdjuster::<ChannelsData2Selector, MonoMetrics<1,2>>::new_with_overlap(
                                 [-1, min_split_len].into(),
                                 [-1, overlap_len].into()
                             ))), //MonoMetrics<0, 3> - 0 dim of 3-dimensional  Batch<ChannelsData>
                         ], None).run(
                         &apply_motif_net,
                         Arc::clone(&input),
                         ctx,
                         Contexts::new(
                             SplitsAdjusterContextImpl::new(input._size().into(), SingleDimTensorOutputBuilder::new("".into(), 0, &[slices+1/*TODO +1 for padded slice, ApplyMotifNet::run   */,1], (Float, ThoDevice::from(device_idx).into()), BuildAs::Append)),
                             NoopRunnableContext {},
                             PermitsAndDumpToCpuAdjustableContex::new(
                                 PermitsAdjustContextImpl::new(ctx_name),
                                 DumpToCpuContextImpl::new(&map),
                             ),
                         ),
                     ).await
                 })

            })
            .buffered(10)//TODO config, TODO - use buffer_unordered and then sort by motif_id?
            .try_collect::<Vec<_>>().await?
            .into_iter().
            collect::<Result<Vec<_>, ThoosaError>>()?;

        debug!("ApplyMotifNets::run: matches={matches:?}");


        let tensor = ThoTensor::new(
            "".into(),
            Tensor::f_concat(
                &matches.into_iter().map(|v| v.t).collect::<Vec<_>>(), 1
            )?
            .f_transpose(0, 1)?,
        )?;

        Ok(Arc::new(DevicesWrap::wrap(tensor, cache_area)))

    }

    fn on_cached<'a>(&self, input: &impl WithId<ID=<<Self as Stage>::In<'a> as WithId>::ID>, ctx: &Arc<parking_lot::RwLock<Self::Ctx>>)
    where
        Self: 'a
    {
        //TODO
    }
}


impl <C>OutId for ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        format!("ApplyMotifNets_{}_{}", self.id, input_id).into()
    }
}


#[derive(Debug)]
struct ApplyMotifNetInputDeviceWrapper {
    batch: Arc<DevicesWrap<ThoTensor<Batch<ChannelsData2>, SimpleId>>>,
    motif_net: Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>,
    slice_step: u32
}

impl ApplyMotifNetInputDeviceWrapper {
    pub fn new(batch: Arc<DevicesWrap<ThoTensor<Batch<ChannelsData2>, SimpleId>>>, motif_net: Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>, slice_step: u32) -> Self {
        Self { batch, motif_net, slice_step }
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

    async fn with_data_ref<R, F, P>(&self, to_device_idx: usize, permits_ctx: Arc<parking_lot::RwLock<P>>, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
        for<'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>,
        P:PermitsCtx
    {
        let t_1_batch= Arc::clone(&self.batch);
        let motif_net = Arc::clone(&self.motif_net);

        with_data!(permits_ctx, to_device_idx, t_1_batch, motif_net;
            |
             batch_v:&ThoTensor<Batch<ChannelsData2>, SimpleId>,
             batched_motifs_output:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>
            |{

                do_run(&ApplyMotifNetInput::new(
                    batch_v,
                    batched_motifs_output,
                ))

        })
    }
}

#[derive(Debug, Clone, Default)]
struct ApplyMotifNetInputDeviceWrapperSelector {
    batch_slices: BatchChannelsData2Selector,
}


impl Adjustable for ApplyMotifNetInputDeviceWrapperSelector{}


impl From<Range<MonoMetrics<0, 3>>> for ApplyMotifNetInputDeviceWrapperSelector {
    fn from(value: Range<MonoMetrics<0, 3>>) -> Self {
        ApplyMotifNetInputDeviceWrapperSelector { batch_slices: BatchChannelsData2Selector(value) }
    }
}

impl WithSelectDataRef<ApplyMotifNetInputDeviceWrapperSelector> for ApplyMotifNetInputDeviceWrapper {
    async fn with_selected_data_ref<'o, R, F, P>(&self, to_device_idx: usize, permits_ctx: Arc<parking_lot::RwLock<P>>, selector: ApplyMotifNetInputDeviceWrapperSelector, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
        P:PermitsCtx,
        for<'f> F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>,
    {


        let name = permits_ctx.read().name();
        if permits_ctx.read().acquired_permits() == 0 {
            let result = GlobalCtx::global()
                .acquire_many(to_device_idx, 1, Arc::new(SimpleId(name)))
                .await?;
            permits_ctx.write().add_permit(result);//permits can be dropped only from OOM handler or when context is dropped
        }

        let t_1_batch = Arc::clone(&self.batch);
        let motif_net = Arc::clone(&self.motif_net);

        let selected_batch = t_1_batch.with_selected_data(to_device_idx, selector.batch_slices, |v|{

            Ok(v)
        }).await?;

        let __with_data_device = to_device_idx;

        let __device_idx = __with_data_device;
        // let __with_data_var_0 = ::std::sync::Arc::clone(&(t_1_batch));
        // let __with_data_var_1 = ::std::sync::Arc::clone(&(motif_net));

        // let __with_data_handle_0 = ::tokio::spawn({
        //     let __wrap = ::std::sync::Arc::clone(&__with_data_var_0);
        //     let __device_idx = __device_idx;
        //     async move {
        //         __wrap.load_from_any_device(__device_idx).await?;
        //         Ok::<(), ::thoosa_shared::ThoosaError>(())
        //     }
        // });

        // let __with_data_handle_1 = ::tokio::spawn({
        //     let __wrap = ::std::sync::Arc::clone(&__with_data_var_1);
        //     let __device_idx = __device_idx;
        //     async move {
        //         __wrap.load_from_any_device(__device_idx).await?;
        //         Ok::<(), ::thoosa_shared::ThoosaError>(())
        //     }
        // });

        // let __with_data_handle_res_0 = __with_data_handle_0.await?;
        // __with_data_handle_res_0?;
        // let __with_data_handle_res_1 = __with_data_handle_1.await?;
        // let __with_data_handle_res_1 = __with_data_handle_res_1?;

        // let __with_data_guard_0 = __with_data_var_0.get_copied(__device_idx).await?;
        // let __with_data_ref_0 = &*__with_data_guard_0;
        let __with_data_guard_1 = motif_net.load_from_any_device_(__device_idx).await?;

        // let __with_data_guard_1 = motif_net.get_copied(__device_idx).await?;
        let __with_data_ref_1 = &*__with_data_guard_1;

        // (|batch_v: &ThoTensor<Batch<ChannelsData2>, SimpleId>,
        //   batched_motifs_output: &DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>| {
        //
        // })(__with_data_ref_0, __with_data_ref_1)

        permits_ctx.write().drop_permits(DropIf::Always);
        do_run(&ApplyMotifNetInput::new(
            &selected_batch,
            __with_data_ref_1,
        ))


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
    motif_net: &'a DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>
}


impl Data for ApplyMotifNetInput<'_> {
    type Metrics = MonoMetrics<0, 3>;

    fn size(&self) -> Self::Metrics {
        MonoMetrics::<0, 3>::try_from(self.batch.size()).unwrap_or_default()
    }

    fn device(&self) -> usize {
        self.batch.device()
    }

    fn mem_size(&self) -> u64 {
        self.batch.mem_size()
    }
}

impl <'a> ApplyMotifNetInput<'a> {
    pub fn new(batch: &'a ThoTensor<Batch<ChannelsData2>, SimpleId>, motif_net: &'a DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>) -> Self {
        Self {  batch, motif_net }
    }
}

impl <'a> Input for ApplyMotifNetInput<'a> {}
impl <'a> WithId for ApplyMotifNetInput<'a> {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

#[derive(Debug)]
struct ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable
{
    net_wrapper:Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>,
    slice_step:u32,
    pd:PhantomData<C>,
    #[cfg(test)]
    fail_if_size_bigger_than:Option<usize>,
}
impl <C> ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable
{
    pub fn new(
        net_wrapper:Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>,
        slice_step:u32,
        #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Self {
        Self {
            net_wrapper,
            slice_step,
            pd: PhantomData ,
            #[cfg(test)]
            fail_if_size_bigger_than
        }
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
    type In<'a> = ThoTensor<ChannelsData2, SimpleId>;
    type Out = ThoTensor<MotifProbability, SimpleId>;
}

impl <C> AsyncStage for ApplyMotifNet<C> where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyMotifNet::run"))]
    fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> impl Future<Output=Result<Arc<DevicesWrap<Self::Out>>, ThoosaError>> + Send {


        let net_wrapper = Arc::clone(&self.net_wrapper);
        let (device_idx, cache_area) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.cache_area())
        };
        async move {
            with_data!(ctx, device_idx, net_wrapper, input; |net:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>, input:&ThoTensor<ChannelsData2, SimpleId>|{
                let motif_id = net.extra().motif_id();
                 #[cfg(test)]
                if self.fail_if_size_bigger_than.map(|v| v < input.size()[1] as usize).unwrap_or(false) {
                    debug!("AsyncStage for ApplyMotifNet::run, triggering OOM: motif_id={motif_id} input.size={:?} self.fail_if_size_bigger_than={:?}", input.size(), self.fail_if_size_bigger_than);
                    return Err(ThoosaError::oom());
                }

                let c = net.extra().channel() as i64;

                let motif_width = net.extra().motif_net_params.num_features() as i64;

                debug!("ApplyMotifNet::run, ok: motif_id={motif_id} c={c} motif_width={motif_width} input.t.size={:?}", input.t.size());
                let t = input.t.i((c, ..));
                let t  = if t.size()[0] >= motif_width {
                    t
                } else {
                    debug!("ApplyMotifNet::run, ok: motif_id={motif_id} padding t.size={:?} to motif_width={motif_width}", t.size());
                    pad_to(&t, &[motif_width])?
                };

                let slices = t.f_unfold(0, motif_width, self.slice_step as i64)?; //.reshape([-1, motif_width]);

                debug!("ApplyMotifNet::run, ok: motif_id={motif_id} slices.size={:?}",  slices.size());

                let slices_tensor = ThoTensor::new("".into(), slices)?;
                let out = net.data().tho_forward(&slices_tensor, false)?;

                debug!("ApplyMotifNet::run, ok: motif_id={motif_id} out.size={:?}", out.size());

                Ok(Arc::new(DevicesWrap::wrap(out, cache_area)))
            })
        }
    }
}

impl<C> ApplyMotifNet<C>
where
    C: Adjustable + MotifsCtx + RunCtx,
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyMotifNet::run_forward"))]
    fn run_forward(input: &ApplyMotifNetInput, slices: Tensor) -> Result<ThoTensor<MotifProbability, CachableId>, ThoosaError> {
        let slices = ThoTensor::new("".into(), slices)?;
        Ok(input.motif_net.data().tho_forward(&slices, false)?)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::path::Path;
    use std::sync::Arc;
    use log::{debug, trace};
    use log::LevelFilter::{Debug, Trace};
    use parking_lot::RwLock;
    use plotters::prelude::{BLACK, RED};
    use plotters::style::{BLUE, GREEN};
    use tch::Device::Cpu;
    use tch::kind::FLOAT_CPU;
    use tch::{nn, Device, IndexOp, Kind, Tensor};
    use tch::nn::{LinearConfig, VarStore};
    use caching_runner::common::cachable::DevicesWrap;
    use caching_runner::common::mock_run_ctx;
    use caching_runner::common::PermitsCtx;
    use caching_runner::common::runnable::AsyncStage;
    use caching_runner::global_ctx::GlobalCtx;
    use thoosa_common::builder::SeqBuilder;
    use thoosa_common::dims::{AnyDims, Batch, Dims, MotifProbability, Slice};
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::my_tch::ThoModule;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore};
    use thoosa_common::utils::{assert_close, lin_t, plot_series_with_lanes, reset_print_options, test_net, vec_f32_from};
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_motifs::generators::noise::resource_tensor;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::{CopyToDevice, MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::perf_logging::PERF;
    use thoosa_shared::traits::Named;
    use thoosa_shared::traits::SimpleId;
    use thoosa_store::MockStoreBuilder;
    use thoosa_with_data::with_data;
    use crate::tasks::apply_motif_nets::ApplyMotifNets;
    use crate::tasks::apply_motif_nets_to_batch::tests::motif_test_net_;
    use crate::tasks::select_motifs::tests::motifs_tensor;
    use crate::tasks::tensor_loader::mock_tensor_loader;


    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_without_oom() -> Result<(), ThoosaError> {
        test_apply_motif_nets_3_channels_2_motifs(None, 4000).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_without_oom_uneven() -> Result<(), ThoosaError> {
        test_apply_motif_nets_3_channels_2_motifs(None, 4321).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_with_oom() -> Result<(), ThoosaError> {
        test_apply_motif_nets_3_channels_2_motifs(Some(1000), 4000).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_with_oom_uneven() -> Result<(), ThoosaError> {
        test_apply_motif_nets_3_channels_2_motifs(Some(987), 4321).await
    }

    async fn test_apply_motif_nets_3_channels_2_motifs(trigger_oom_len: Option<usize>, input_n_0: usize) -> Result<(), ThoosaError> {
        thoosa_shared::test_utils::maybe_init_perf_tracing_stdout("debug");
        log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let step = 3;
        let find_motifs_run_id:RunId = 1;
        let select_motifs_run_id:RunId = 2;

        let train_motifs_run_id:RunId = 3;
        let motif_0_0_id = 0; //channel 0 motif 0
        let motif_0_1_id = 1; //channel 0 motif 1
        let motif_1_0_id = 0; //channel 1 motif 0

        let motif_0_0 = Arc::new(ThoTensor::new("motif_0_0".into(), motifs_tensor(100, 3, MotifKind::Pulse))?);
        let motif_0_1 = Arc::new(ThoTensor::new("motif_0_1".into(), motifs_tensor(100, 3, MotifKind::Triangle))?);
        let motif_1_0 = Arc::new(ThoTensor::new("motif_1_0".into(), motifs_tensor(100, 3, MotifKind::Sin))?);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, 100, input_n_0 as u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        motif_test_net_(Cpu, Arc::clone(&motif_0_0), step),
                        motif_test_net_(Cpu, Arc::clone(&motif_0_1), step),
                    ],
                    vec![
                        //channel 1
                        motif_test_net_(Cpu, Arc::clone(&motif_1_0), step),
                    ],
                    vec![],
                ])
                .build(),
        ));

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new(Some(find_motifs_run_id), step, 100, None, None).to_properties_iter(None).collect()),
                )

                .with_get_input_run_ids([
                    (train_motifs_run_id, Ok(vec![select_motifs_run_id])),
                    (select_motifs_run_id, Ok(vec![find_motifs_run_id]))
                ])
                .with_get_motif_meta( [(motif_0_0_id, Ok((0usize, find_motifs_run_id)))])
                .with_list_selected_motifs([
                    (select_motifs_run_id, Ok(
                        vec![
                            vec![(motif_0_0_id, 1.0f64), (motif_0_1_id, 1.0f64)], //channel 0
                            vec![(motif_1_0_id, 1.0f64)], //channel 1
                            vec![]//channel 2
                        ]
                    ))
                ])
                .build()
        ));


        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            "1".into(),
            &ws,
            &store,
            vec![select_motifs_run_id],
            Arc::new(RwLock::new(mock_run_ctx::default(1))),
            trigger_oom_len
        ).await?);

        let (resource_a_t, resource_a_intervals) = resource_tensor(input_n_0, &[
            vec![(1, 100, MotifKind::Pulse), (1, 100, MotifKind::Triangle)],
            vec![(1, 100, MotifKind::Sin)],
            vec![]
        ], 200)?;


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_t={resource_a_t:?}");// resource_b_t={resource_b_t:?} resource_c_t={resource_c_t:?}

        let run_ctx = Arc::new(RwLock::new(mock_run_ctx::default(1)));

        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new("test_batch".into(), resource_a_t)?, 0)),
            Arc::clone(&run_ctx)
        ).await?;

        debug!("test_apply_motif_nets_3_channels_1_motif: PERF={:?}", PERF);


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_intervals={resource_a_intervals:?}");

        let pulse_interval_starts_in_resources_a = resource_a_intervals[0][0][0].start;

        let triangle_interval_starts_in_resources_a = resource_a_intervals[0][1][0].start;

        let sin_interval_starts_in_resources_a = resource_a_intervals[1][0][0].start;

        with_data!(run_ctx, 0, out; |r:&ThoTensor<MotifProbability, SimpleId>|{
            debug!("test_apply_motif_nets_3_channels_1_motif:  matches={:?}", r);

            let (pulse_matches_vals, pulse_matches_idx) = &r.t.i((0, ..)).topk(1i64, 0, true, true);
            let (triangle_matches_vals, triangle_matches_idx) = &r.t.i((1, ..)).topk(1i64, 0, true, true);
            let (sin_matches_vals, sin_matches_idx) = &r.t.i((2, ..)).topk(1i64, 0, true, true);

            assert_close(pulse_interval_starts_in_resources_a as f64, pulse_matches_idx.double_value(&[0]), 2.);
            assert_close(triangle_interval_starts_in_resources_a as f64, triangle_matches_idx.double_value(&[0]), 2.);
            assert_close(sin_interval_starts_in_resources_a as f64, sin_matches_idx.double_value(&[0]), 2.);

            Ok::<_, ThoosaError>(())
        })?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_single_net_1_channel_1_motif_without_oom() -> Result<(), ThoosaError> {
        test_apply_motif_single_net_1_channel_1_motif(None, 4000, 3).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_single_net_1_channel_1_motif_with_oom() -> Result<(), ThoosaError> {
        test_apply_motif_single_net_1_channel_1_motif(Some(1000), 4000, 3).await
    }


    async fn test_apply_motif_single_net_1_channel_1_motif(trigger_oom_len: Option<usize>, input_n_0: usize, step: u32) -> Result<(), ThoosaError> {
        thoosa_shared::test_utils::maybe_init_perf_tracing_stdout("debug");
        log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let step = step;
        let find_motifs_run_id:RunId = 1;
        let select_motifs_run_id:RunId = 2;

        let train_motifs_run_id:RunId = 3;
        let motif_0_0_id = 0; //channel 0 motif 0

        let motif_0_0 = Arc::new(ThoTensor::new("motif_0_0".into(), motifs_tensor(100, 3, MotifKind::Pulse))?);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 1, 100, input_n_0 as u32)
                .with_load_motif_nets(vec![
                    vec![
                        //channel 0
                        motif_test_net_(Cpu, Arc::clone(&motif_0_0), step),
                    ],
                ])
                .build(),
        ));

        let mut store = Arc::new(RwLock::new(
            MockStoreBuilder::new()
            .with_list_run_config_values(
                find_motifs_run_id,
                Ok(FindMotifsRunConfig::new(Some(find_motifs_run_id), step, 100, None, None).to_properties_iter(None).collect()),
            )

            .with_get_input_run_ids([
                (train_motifs_run_id, Ok(vec![select_motifs_run_id])),
                (select_motifs_run_id, Ok(vec![find_motifs_run_id]))
            ])
            .with_get_motif_meta( [(motif_0_0_id, Ok((0usize, find_motifs_run_id)))])
            .with_list_selected_motifs([
                (select_motifs_run_id, Ok(
                    vec![
                        vec![(motif_0_0_id, 1.0f64)], //channel 0
                    ]
                ))
            ])
            .build()
        ));


        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            "1".into(),
            &ws,
            &store,
            vec![select_motifs_run_id],
            Arc::new(RwLock::new(mock_run_ctx::default(1))),
            trigger_oom_len//Some(1000)
        ).await?);

        let (resource_a_t, resource_a_intervals) = resource_tensor(input_n_0, &[
            vec![(1, 100, MotifKind::Pulse)],
        ], 200)?;


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_t={resource_a_t:?}");// resource_b_t={resource_b_t:?} resource_c_t={resource_c_t:?}

        let run_ctx = Arc::new(RwLock::new(mock_run_ctx::default(1)));

        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new("test_batch".into(), resource_a_t)?, 0)),
            Arc::clone(&run_ctx)
        ).await?;

        debug!("test_apply_motif_nets_3_channels_1_motif: PERF={:?}", PERF);


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_intervals={resource_a_intervals:?}");

        let pulse_interval_starts_in_resources_a = resource_a_intervals[0][0][0].start;


        with_data!(run_ctx, 0, out; |r:&ThoTensor<MotifProbability, SimpleId>|{
            debug!("test_apply_motif_nets_3_channels_1_motif:  matches={:?}", r);

            let (pulse_matches_vals, pulse_matches_idx) = &r.t.i((0, ..)).topk(1i64, 0, true, true);

            assert_close(pulse_interval_starts_in_resources_a as f64, step as f64 * pulse_matches_idx.double_value(&[0]), step as f64 * 2.);

            Ok::<_, ThoosaError>(())
        })?;

        Ok(())
    }


}
