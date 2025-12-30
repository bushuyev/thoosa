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
use log::{debug, error, info};
use parking_lot::lock_api::RwLock;
use tch::{nn, Device, IndexOp, Kind, Tensor};
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tch::nn::VarStore;
use tokio::task::JoinError;
use tracing::warn;
use cached_derive::Cached;

use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::oom_handler::{Adjusters, AsyncOOMHandler, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::builder::SeqBuilder;
use thoosa_common::dims::{AnyDims, Batch, Dims, MotifProbability, Slice};
use thoosa_common::my_tch::ThoModule;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore, TrainMotifsNetsResult};
use thoosa_common::utils::pad_to;
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::{mono_key, ThoosaError};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Metrics, MonoMetrics, Select, SimpleId, WithId};
use thoosa_shared::comp_id::{SynKey, MonoKey};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use thoosa_common::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use thoosa_shared::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_shared::runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use thoosa_shared::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef};
use thoosa_shared::runner::common::{OutId, RunCtx};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage};
use thoosa_shared::runner::permits::{PermitStep, MAX_PERMITS};
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
    id: SynKey,
    ws:Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
    store:Arc<parking_lot::RwLock<Box<dyn Store>>>,
    pd:PhantomData<C>,
    apply_motif_nets: Vec<Arc<ApplyMotifNet<C>>>,
    pub slice_step: u32,//should be in the input
    pub motif_width:u32,
}

impl <C> ApplyMotifNets<C>
where
    C: RunCtx + MotifsCtx + Adjustable + 'static
{
    pub async fn new<R>(
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

                debug!("ApplyMotifNets::new: channel={channel} motif_id={motif_id} load_motif_net={load_motif_net:?}");

                load_motif_net.run_cached(Arc::new(DevicesWrap::wrap(LoadMotifNetInput::new(motif_id, channel as u32), 0)), stage_ctx).await
            }
        })
        .buffered(10)//TODO config order is important to keep output tensor same
            .filter_map(|r|async {
                match r {
                    Ok(net) => {
                        debug!("ApplyMotifNets::new: loaded net={net:?}");

                        Some(net)
                    },
                    Err(e) => {
                        error!("ApplyMotifNets::new: can't load net for motif; there must have been an error while training the net: e={e:?}");
                        None
                    }
                }
            })
        .collect::<Vec<_>>().await;


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
        debug!("ApplyMotifNets::new: motif_width={motif_width:?} slice_step={slice_step:?}");

        let mut select_motifs_run_id_sorted = select_motifs_run_id.clone();//TODO move to util CompKey::from Vec
        select_motifs_run_id_sorted.sort();
        Ok(Self {
            id: SynKey::new(select_motifs_run_id_sorted.into_iter().map(|run_id| mono_key!(run_id, format!("select_motifs_run_id_{run_id}"))).collect::<Vec<_>>()),
            ws: Arc::clone(&ws),
            store:Arc::clone(&store),
            pd: PhantomData,
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
    type In<'a> = ThoTensor<ChannelsData2, SynKey>;
    type Out = ThoTensor<MotifProbability, SynKey>;
}



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

        debug!("ApplyMotifNets::run: self.loaded_motif_nets.len={:?} input={input:?}", self.apply_motif_nets.len());


        let apply_motif_nets = self.apply_motif_nets.clone();
        let min_split_len = self.motif_width as i64;
        let overlap_len = (self.motif_width  - self.slice_step) as i64;
        let matches = futures::stream::iter(apply_motif_nets.into_iter())
            .map(|apply_motif_net|{
                let input = Arc::clone(&input);
                let out_id = apply_motif_net.out_id(&input.id());
                let apply_motif_net_ctx = ctx.write().branch(format!("apply_motif_net_ctx_{out_id:?}").leak());

                let slice_step = self.slice_step;

                let slices = ((input.size().dim::<1>() as u32 - self.motif_width)/slice_step) as i64 + 1;

                debug!("ApplyMotifNets::run: input.size={:?} self.motif_width={:?} slice_step={slice_step} slices={slices} min_split_len={min_split_len} overlap_len={overlap_len}", input.size(), self.motif_width);
                let ctx_key = Arc::new(SynKey::from(format!("ApplyMotifNets_{:?}", input.id()).as_str())) as Arc<dyn CachableIdTr>;//TODO format


                async move {
                    let map = HashMap::new();

                     let r= AsyncOOMHandler::new(
                         vec![
                             Adjusters::RunContext(Box::new(PermitsAdjuster::new(MAX_PERMITS, *PermitStep::TWENTY, move |n| GlobalCtx::global().acquire_many(device_idx, n, Arc::clone(&ctx_key) as Arc<dyn CachableIdTr>)))),
                             Adjusters::RunContext(Box::new(DumpToCpu::new())),
                             Adjusters::Input(Box::new(SplitsAdjuster::<ChannelsData2Selector, MonoMetrics<1,2>>::new_with_overlap(
                                 [-1, min_split_len].into(),
                                 [-1, overlap_len].into()
                             ))), //MonoMetrics<0, 3> - 0 dim of 3-dimensional  Batch<ChannelsData>
                         ], None)
                     .run(
                         &apply_motif_net,
                         Arc::clone(&input),
                         apply_motif_net_ctx,
                         Contexts::new(
                             SplitsAdjusterContextImpl::new(input._size().into(), SingleDimTensorOutputBuilder::new(out_id, 0, &[slices+1/*TODO +1 for padded slice, ApplyMotifNet::run   */,1], (Float, ThoDevice::from(device_idx).into()), BuildAs::Append)),
                             NoopRunnableContext {},
                             PermitsAndDumpToCpuAdjustableContex::new(
                                 PermitsAdjustContextImpl::new(ctx_name),
                                 DumpToCpuContextImpl::new(&map),
                             ),
                         ),
                     ).await;
                    debug!("ApplyMotifNets::run: r={r:?}");
                    r
                 }

            })
            .buffered(10)//TODO config, TODO - use buffer_unordered and then sort by motif_id?
            .try_collect::<Vec<_>>().await?;

        debug!("ApplyMotifNets::run: matches={matches:?}");


        let tensor = ThoTensor::new(
            self.out_id(input.id()),
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
        &self.id + input_id
    }
}


#[derive(Debug)]
struct ApplyMotifNetInput<'a> {
    batch: &'a ThoTensor<Batch<ChannelsData2>, SynKey>,
    motif_net: &'a DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>
}


impl Data for ApplyMotifNetInput<'_> {
    type Metrics = MonoMetrics<0, 3>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        MonoMetrics::<0, 3>::try_from(self.batch.size()).unwrap_or_default()
    }

    fn device(&self) -> usize {
        self.batch.device_idx()
    }

    fn mem_size(&self) -> u64 {
        self.batch.mem_size()
    }
}

impl <'a> ApplyMotifNetInput<'a> {
    pub fn new(batch: &'a ThoTensor<Batch<ChannelsData2>, SynKey>, motif_net: &'a DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>) -> Self {
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
    id: SynKey,
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
            id: mono_key!("ApplyMotifNet", "ApplyMotifNet") + net_wrapper.id(),
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
        let stage_key = SynKey::new([mono_key!("ApplyMotifNet", "stage")]);
        &stage_key + input_id
    }
}


impl <C>  Stage for ApplyMotifNet<C>
where
    C: RunCtx + MotifsCtx + Adjustable{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData2, SynKey>;
    type Out = ThoTensor<MotifProbability, SynKey>;
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
            with_data!(ctx, device_idx, net_wrapper, input; |net:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>, input:&ThoTensor<ChannelsData2, SynKey>|{
                let motif_id = net.extra().motif_id();
                 #[cfg(test)]
                if self.fail_if_size_bigger_than.map(|v| v < input.size()[1] as usize).unwrap_or(false) {
                    debug!("AsyncStage for ApplyMotifNet::run, triggering OOM: motif_id={motif_id} input.size={:?} self.fail_if_size_bigger_than={:?}", input.size(), self.fail_if_size_bigger_than);
                    return Err(ThoosaError::oom());
                }

                let c = net.extra().channel() as i64;

                let motif_width = net.extra().motif_net_params.num_features() as i64;

                debug!("ApplyMotifNet::run, ok: motif_id={motif_id} c={c} motif_width={motif_width} input.t.size={:?} self.slice_step={}", input.t.size(), self.slice_step);
                let t = input.t.i((c, ..));
                let t  = if t.size()[0] >= motif_width {
                    t
                } else {
                    debug!("ApplyMotifNet::run, ok: motif_id={motif_id} padding t.size={:?} to motif_width={motif_width}", t.size());
                    pad_to(&t, &[motif_width])?
                };

                let slices = t.f_unfold(0, motif_width, self.slice_step as i64)?; //.reshape([-1, motif_width]);

                debug!("ApplyMotifNet::run, ok: motif_id={motif_id} slices.size={:?}",  slices.size());

                let out_id = self.out_id(input.id());
                let slices_tensor = ThoTensor::<Batch<Slice>, SimpleId>::from(slices);
                let out = net.data().tho_forward(&slices_tensor, false)?;
                let out = ThoTensor::new(out_id, out.t())?;

                debug!("ApplyMotifNet::run, ok: motif_id={motif_id} out.size={:?}", out.size());

                Ok(Arc::new(DevicesWrap::wrap(out, cache_area)))
            })
        }
    }
}


#[cfg(test)]
pub fn motif_test_net_(
    device: Device,
    motif_t: Arc<ThoTensor<Batch<Slice>, SynKey>>,
    step: u32,
) -> (
    Box<dyn Fn() -> MotifNetParams + Sync + Send>,
    Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send>,
) {
    let motif_width = motif_t.as_ref().t_ref().size()[1] as u32;
    (
        Box::new(move || MotifNetParams::new(motif_width, 1, step)),
        Box::new(move || {
            *dummy_motif_net::<Batch<Slice>, MotifProbability>(motif_width, 1., device, "test", &motif_t.as_ref().t_ref().i((0, ..)), step as i64).unwrap()//first slice
        }),
    )
}

#[cfg(test)]
///builds net initialized to detect provided pattern
pub fn dummy_motif_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static>(
    num_features: u32,
    k: f64,
    device: Device,
    descr: &str,
    pattern:&Tensor,
    stride: i64,
) -> Result<Box<ThoModuleWithStore<DI, DO>>, ThoosaError> {
    assert_eq!(pattern.size().len(), 1, "template must be 1-D");
    let l = pattern.size()[0];

    // mean-center and unit-norm the template (helps with DC/scale variations)
    let t = pattern
        .to_device(device)
        .to_kind(Kind::Float);
    let t = &t - t.mean(Kind::Float);
    let t = &t / (t.pow_tensor_scalar(2).sum(Kind::Float).sqrt() + 1e-12);

    let vs = VarStore::new(device);
    let mut cfg = nn::ConvConfig { bias: false, stride, ..Default::default() };


    let tho_tensor: Arc<ThoTensor<AnyDims, SimpleId>> = Arc::new(ThoTensor::from(t));
    let ones: Arc<ThoTensor<AnyDims, SimpleId>> = Arc::new(ThoTensor::from(Tensor::ones([1, 1, l], FLOAT_CPU)));

    let node = SeqBuilder::new(num_features, None)
        .closure(
            Arc::new(Box::new(move |t: &Tensor| {

                let r = t.unsqueeze(1);//dummy channel

                debug!("motif_node_2: 0. input.size={:?}", r.size());

                Ok(r)
            })),
            None,
        )
        .split(
            vec![
                SeqBuilder::new(num_features, None).conv_1d("conv", l as u32, 1, 1, cfg, Some(Arc::clone(&tho_tensor))).end_seq(),
                SeqBuilder::new(num_features, None)
                    .closure(
                        Arc::new(Box::new(move |t: &Tensor| {
                            Ok(t.f_pow_tensor_scalar(2)?)
                        })),
                        None,
                    )
                    .conv_1d("energy", l as u32, 1, 1, cfg, Some(Arc::clone(&ones)))
                    .closure(
                        Arc::new(Box::new(move |t: &Tensor| {
                            Ok(t.f_clamp_min(1e-12)?) //.sqrt()
                        })),
                        None,
                    )
                    .end_seq()
            ],
            Box::new(move |t: &Tensor| {
                Ok(vec![
                    t.shallow_clone(),
                    t.shallow_clone()
                ])
            }),
            Box::new(move |num_denom: Vec<Tensor>| {
                let [num, denom] = <[Tensor; 2]>::try_from(num_denom).unwrap();

                Ok((num / denom).squeeze_dim(1))
            }),
        ).end_seq();



    let module = node.clone().build_tch_module(&vs.root())?;
    Ok(Box::new(ThoModuleWithStore::with_factory(
        "test_net",
        module,
        vs,
        Some(Arc::new( move |path:&nn::Path| {
            node.clone().build_tch_module(path)
        }))
    )))

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
    use thoosa_shared::runner::common::cachable::DevicesWrap;
    use thoosa_shared::runner::common::mock_run_ctx;
    use thoosa_shared::runner::common::PermitsCtx;
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
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
    use thoosa_shared::{MotifKind, TensorIdSliceIdx, ThoosaError};
    use thoosa_shared::perf_logging::PERF;
    use thoosa_shared::traits::{Named, SimpleId};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_store::MockStoreBuilder;
    use thoosa_with_data::with_data;
    use crate::tasks::apply_motif_nets::{dummy_motif_net, motif_test_net_, ApplyMotifNets};
    use crate::tasks::select_motifs::tests::motifs_tensor;
    use crate::tasks::tensor_loader::mock_tensor_loader;


    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_without_oom() -> Result<(), ThoosaError> {
        test_apply_motif_nets_3_channels_2_motifs(None, 4000, 2).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_without_oom_uneven() -> Result<(), ThoosaError> {
        test_apply_motif_nets_3_channels_2_motifs(None, 4321, 2).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_step_1_with_oom() -> Result<(), ThoosaError> {
        test_apply_motif_nets_3_channels_2_motifs(Some(1000), 4000, 1).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_step_2_with_oom() -> Result<(), ThoosaError> {
        test_apply_motif_nets_3_channels_2_motifs(Some(1000), 4000, 2).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_with_oom_uneven() -> Result<(), ThoosaError> {
        test_apply_motif_nets_3_channels_2_motifs(Some(987), 4321, 2).await
    }

    async fn test_apply_motif_nets_3_channels_2_motifs(trigger_oom_len: Option<usize>, input_n_0: usize, step: u32) -> Result<(), ThoosaError> {
        thoosa_shared::test_utils::maybe_init_perf_tracing_stdout("debug");
        log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let find_motifs_run_id:RunId = 1;
        let select_motifs_run_id:RunId = 2;

        let train_motifs_run_id:RunId = 3;
        let motif_0_0_id = 0; //channel 0 motif 0
        let motif_0_1_id = 1; //channel 0 motif 1
        let motif_1_0_id = 0; //channel 1 motif 0

        let motif_0_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_0", "motif_id")]), motifs_tensor(100, 3, MotifKind::Pulse))?);
        let motif_0_1 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_1", "motif_id")]), motifs_tensor(100, 3, MotifKind::Triangle))?);
        let motif_1_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_1_0", "motif_id")]), motifs_tensor(100, 3, MotifKind::Sin))?);

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
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test_batch", "test_batch")]), resource_a_t)?, 0)),
            Arc::clone(&run_ctx)
        ).await?;

        debug!("test_apply_motif_nets_3_channels_1_motif: PERF={:?}", PERF);


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_intervals={resource_a_intervals:?}");

        let pulse_interval_starts_in_resources_a = resource_a_intervals[0][0][0].start;

        let triangle_interval_starts_in_resources_a = resource_a_intervals[0][1][0].start;

        let sin_interval_starts_in_resources_a = resource_a_intervals[1][0][0].start;

        with_data!(run_ctx, 0, out; |r:&ThoTensor<MotifProbability, SynKey>|{
            debug!("test_apply_motif_nets_3_channels_1_motif:  matches={:?}", r);

            let (pulse_matches_vals, pulse_matches_idx) = &r.t.i((0, ..)).topk(1i64, 0, true, true);
            let (triangle_matches_vals, triangle_matches_idx) = &r.t.i((1, ..)).topk(1i64, 0, true, true);
            let (sin_matches_vals, sin_matches_idx) = &r.t.i((2, ..)).topk(1i64, 0, true, true);

            assert_close(pulse_interval_starts_in_resources_a as f64, step as f64 * pulse_matches_idx.double_value(&[0]), step as f64 * 3.);
            assert_close(triangle_interval_starts_in_resources_a as f64, step as f64 * triangle_matches_idx.double_value(&[0]), step as f64 * 3.);
            assert_close(sin_interval_starts_in_resources_a as f64, step as f64 * sin_matches_idx.double_value(&[0]), step as f64 * 3.);

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

        let motif_0_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_0", "motif_id")]), motifs_tensor(100, 3, MotifKind::Pulse))?);

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
            &ws,
            &store,
            vec![select_motifs_run_id],
            Arc::new(RwLock::new(mock_run_ctx::default(1))),
            trigger_oom_len //Some(1000)
        ).await?);

        let (resource_a_t, resource_a_intervals) = resource_tensor(input_n_0, &[
            vec![(1, 100, MotifKind::Pulse)],
        ], 200)?;


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_t={resource_a_t:?}");// resource_b_t={resource_b_t:?} resource_c_t={resource_c_t:?}

        let run_ctx = Arc::new(RwLock::new(mock_run_ctx::default(1)));

        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test_batch", "test_batch")]), resource_a_t)?, 0)),
            Arc::clone(&run_ctx)
        ).await?;

        debug!("test_apply_motif_nets_3_channels_1_motif: PERF={:?}", PERF);


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_intervals={resource_a_intervals:?}");

        let pulse_interval_starts_in_resources_a = resource_a_intervals[0][0][0].start;


        with_data!(run_ctx, 0, out; |r:&ThoTensor<MotifProbability, SynKey>|{
            debug!("test_apply_motif_nets_3_channels_1_motif:  matches={:?}", r);

            let (pulse_matches_vals, pulse_matches_idx) = &r.t.i((0, ..)).topk(1i64, 0, true, true);

            assert_close(pulse_interval_starts_in_resources_a as f64, step as f64 * pulse_matches_idx.double_value(&[0]), step as f64 * 2.5);

            Ok::<_, ThoosaError>(())
        })?;

        Ok(())
    }

    #[test]
    fn test_test_net() -> Result<(), ThoosaError> {
        let motif_t = motifs_tensor(100, 3, MotifKind::Pulse);
        let (resource_a_t, resource_a_intervals) = resource_tensor(2000, &[vec![(3, 100, MotifKind::Pulse)]], 200)?;

        debug!("test_test_net: resource_a_t.size={:?}", resource_a_t.size());

        let test_dir = Path::new("/tmp/test_test_net"); //.join(cdt_for_dir());


        let test_net = dummy_motif_net::<AnyDims, AnyDims>(100, 1., Cpu, "test", &motif_t.i((0, ..)), 1)?;

        let resource: ThoTensor<AnyDims, SimpleId> = resource_a_t.shallow_clone().into();
        let out = test_net.tho_forward(&resource, false)?.t;//.i((0, .., 0));//channel[1], slices, vals[1]

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

        debug!("test_test_net: out.size={:?}", out.size());

        /*plot_series_with_lanes(
             vec![
                 (format!("Results"), RED, vec_f32_from(&out.i((0, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>()),
             ],
             format!("sig").as_str(),
             (2000, 1000),
             &test_dir.join(&format!("matches_{}.png", 1)),
             &[(0, 1000)],
         )?;*/

        debug!("test_test_net: resource_a_intervals={resource_a_intervals:?}");

        let (topk_0, topk_1) = out.topk(3i64, 1, true, true);

        trace!("test_test_net: topk=\n{topk_0} \n{topk_1}");

        //0 chanel/0 motif
        assert!(
            Tensor::from_slice(resource_a_intervals[0][0].iter().map(|i|i.start as i64).collect::<Vec<i64>>().as_slice())
                .allclose(&topk_1.i(0), 10., 10., false)
        );

        Ok(())
    }


}
