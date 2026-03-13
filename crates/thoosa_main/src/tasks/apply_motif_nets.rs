#![allow(unexpected_cfgs)]

use std::collections::{HashMap, HashSet};
use std::fmt::{format, Debug};
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, Range};
use std::sync::Arc;
use std::time::Instant;
use async_trait::async_trait;
use log::{debug, error, info, trace};
use parking_lot::RwLock;
use tch::{nn, Device, IndexOp, Kind, Tensor};
use tch::Kind::Float;
use tch::kind::FLOAT_CPU;
use tch::nn::VarStore;
use tracing::warn;
use cached_derive::Cached;
use thoosa_named_derive::Named;

use thoosa_shared::runner::global_ctx::GlobalCtx;
use thoosa_shared::runner::oom_handler::{Adjusters, Contexts, NoopRunnableContext, OOMHandler};
use thoosa_common::debug_plot;
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::builder::SeqBuilder;
use thoosa_common::dims::{AnyDims, Batch, Dims, MotifProbability, Slice};
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
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
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Metrics, MonoMetrics, Select, SimpleId, WithId};
use thoosa_shared::comp_id::{ElvVal, MonoKey, SynKey};
use thoosa_store::Store;
use thoosa_with_data::with_data;
use thoosa_common::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use thoosa_shared::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_shared::runner::adjusters::permits::{PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex};
use thoosa_shared::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use thoosa_shared::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_shared::runner::cache_areas::CacheArea;
use thoosa_shared::runner::common::{DropIf, OutId, RunCtx};
use thoosa_shared::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use thoosa_shared::runner::permits::{PermitStep, MAX_PERMITS};
use crate::tasks::cross_dims::{ChannelsData2, ChannelsData2Selector};
use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput, MotifNetMeta};
use crate::tasks::resources_motifs_clc_dist::ClcResourcesMotifsDistOutput;
use crate::tasks::transforms::freq_filter::FreqFilter;


#[derive(Debug, Cached, Named)]
pub struct ApplyMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    id: SynKey,
    ws:Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
    store:Arc<Box<dyn Store>>,
    pd:PhantomData<C>,
    apply_motif_nets: Vec<Arc<ApplyMotifNet<C>>>,
    pub slice_step: u32,//should be in the input
    pub motif_width:u32,
}

impl <C> ApplyMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    pub async fn new<R>(
        ws: &Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        train_motif_nets_run_ids: Vec<RunId>,
        mut stage_ctx:R,
        max_loss: Option<f32>,
        selected_nets_limit: Option<usize>,
        slice_step: u32,
        #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Result<Self, ThoError>
    where
        R:RunCtx + Adjustable + 'static
    {
        let load_motif_net = Arc::new(LoadMotifNet::<R>::new(&ws));

        //gpt Load all selected motif nets for the provided train_motif_nets_run_ids.
        let mut selected_nets = Vec::new();
        for train_motif_nets_run_id in &train_motif_nets_run_ids {
            selected_nets.extend(store.list_selected_nets(*train_motif_nets_run_id, max_loss, selected_nets_limit).await?);
        }
        let selected_nets_len = selected_nets.len();
        let futures = selected_nets
            .into_iter()
            .map(|(path, _losss)|{
                let ws = Arc::clone(&ws);
                let load_motif_net = Arc::clone(&load_motif_net);
                let load_ctx = stage_ctx.branch(Some("apply_motif_nets_load_motif_net"), false, false);
                async move {

                    // debug!("ApplyMotifNets::new: channel={channel} motif_id={motif_id} load_motif_net={load_motif_net:?}");


                    match load_motif_net.run_cached(Arc::new(DevicesWrap::wrap(LoadMotifNetInput::new(path), CacheArea::default())), load_ctx).await {
                        Ok(net) => {
                            debug!("ApplyMotifNets::new: loaded net={net:?}");
                            Ok::<_, ThoError>(Some(net))
                        }
                        Err(e) => {
                            error!("ApplyMotifNets::new: can't load net for motif; there must have been an error while training the net: e={e:?}");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let loaded_motif_nets = collect_futures_with_progress(
            futures,
            selected_nets_len,
            crate::buffer_limits::APPLY_MOTIF_NETS_NEW_BUFFERED_LIMIT,
            StreamBuffering::Ordered,
            "ApplyMotifNets::new",
        )
        .await
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();


        let mut motif_widths = HashSet::new();
        let mut freq_filters_by_find_motifs_run_id: HashMap<RunId, Option<Arc<FreqFilter<C>>>> = HashMap::new();
        let mut loaded_motif_nets_with_filters = Vec::with_capacity(loaded_motif_nets.len());
        for motif_net in loaded_motif_nets {
            let (motif_id, motif_width) = with_data!(stage_ctx.branch(Some("apply_motif_nets_meta"), false, false), motif_net; |_ctx, v:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>|{
                let motif_width = v.extra().motif_net_params.num_features();
                motif_widths.insert(motif_width);
                Ok::<_, ThoError>((v.extra().motif_id(), motif_width))
            })?;

            let (_channel, find_motifs_run_id) = store.get_motif_meta(motif_id).await?;
            let freq_filter = if let Some(freq_filter) = freq_filters_by_find_motifs_run_id.get(&find_motifs_run_id) {
                freq_filter.clone()
            } else {
                let find_motifs_run_config = FindMotifsRunConfig::from_properties_map_with_id(
                    find_motifs_run_id,
                    &store.list_run_config_values(find_motifs_run_id).await?,
                )?;
                let freq_filter = find_motifs_run_config
                    .freq_filter
                    .as_ref()
                    .map(|freq_filter| FreqFilter::<C>::new(freq_filter.as_slice(), ws, find_motifs_run_id))
                    .transpose()?
                    .map(Arc::new);
                freq_filters_by_find_motifs_run_id.insert(find_motifs_run_id, freq_filter.clone());
                freq_filter
            };

            let freq_filter_find_motifs_run_id = if freq_filter.is_some() { Some(find_motifs_run_id) } else { None };
            loaded_motif_nets_with_filters.push((motif_net, motif_width, freq_filter_find_motifs_run_id, freq_filter));
        }

        if motif_widths.len() > 1 {
            warn!("ApplyMotifNets::new: mixed motif widths in selected nets; widths={motif_widths:?} run_ids={train_motif_nets_run_ids:?}. Will run per-net and pad outputs for concat.");
        }

        //gpt TODO!! if no motif nets load, these expects will panic; consider returning a structured error.
        let motif_width = motif_widths.iter().copied().max().expect("no motif_width");
        // let slice_step = slice_step.into_iter().next().expect("no slice_step");//TODO shouldn't necessarily be what was used during motif training
        debug!("ApplyMotifNets::new: motif_width(max)={motif_width:?} motif_widths={motif_widths:?} slice_step={slice_step:?} loaded_motif_nets.len={}", loaded_motif_nets_with_filters.len());

        let mut train_motif_nets_run_ids_sorted = train_motif_nets_run_ids.clone();//TODO move to util CompKey::from Vec
        train_motif_nets_run_ids_sorted.sort();
        Ok(Self {
            id: SynKey::new(train_motif_nets_run_ids_sorted.into_iter().map(|run_id| mono_key!(run_id, format!("train_motif_nets_run_ids_{run_id}"))).collect::<Vec<_>>()),
            ws: Arc::clone(&ws),
            store:Arc::clone(&store),
            pd: PhantomData,
            apply_motif_nets: loaded_motif_nets_with_filters
                .into_iter()
                .map(|(net, motif_width, freq_filter_find_motifs_run_id, freq_filter)| Arc::new(ApplyMotifNet::new(
                    net,
                    motif_width,
                    freq_filter_find_motifs_run_id,
                    freq_filter,
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
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData2, SynKey>;
    type Out = ThoTensor<MotifProbability, SynKey>;
}



impl <C>AsyncStage for ApplyMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyMotifNets::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {

        let device_idx = ctx.device_idx();
        let cache_area = ctx.cache_area();

        debug!("ApplyMotifNets::run: self.loaded_motif_nets.len={:?} input={input:?}", self.apply_motif_nets.len());


        let apply_motif_nets = self.apply_motif_nets.clone();
        let apply_motif_nets_len = apply_motif_nets.len();
        let mut inputs_by_filter_group: HashMap<Option<RunId>, Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>> = HashMap::new();
        inputs_by_filter_group.insert(None, Arc::clone(&input));

        for apply_motif_net in &apply_motif_nets {
            let group_key = apply_motif_net.freq_filter_find_motifs_run_id;
            if inputs_by_filter_group.contains_key(&group_key) {
                continue;
            }
            let filtered_input = match &apply_motif_net.freq_filter {
                Some(freq_filter) => {
                    let freq_filter_ctx = ctx.branch(Some(&format!("apply_motif_nets_prefilter_{group_key:?}")), true, false);
                    freq_filter.clone().run_cached(Arc::clone(&input), freq_filter_ctx).await?
                }
                None => Arc::clone(&input),
            };
            inputs_by_filter_group.insert(group_key, filtered_input);
        }

        let inputs_by_filter_group = Arc::new(inputs_by_filter_group);
        let futures = apply_motif_nets
            .into_iter()
            .map(|apply_motif_net|{
                let inputs_by_filter_group = Arc::clone(&inputs_by_filter_group);
                let group_key = apply_motif_net.freq_filter_find_motifs_run_id;
                let input = inputs_by_filter_group.get(&group_key).cloned().expect("filtered input for group must exist");
                let motif_width = apply_motif_net.motif_width;
                let motif_width_i64 = motif_width as i64;
                let min_split_len = motif_width_i64;
                let overlap_len = motif_width.saturating_sub(self.slice_step) as i64;
                let input_len = input.size().dim::<1>() as i64;
                let slices = if input_len >= motif_width_i64 {
                    ((input_len - motif_width_i64) / self.slice_step as i64) + 1
                } else {
                    1
                };
                let out_id = apply_motif_net.out_id(&input.id());
                let apply_motif_net_ctx = ctx.branch(Some(&format!("apply_motif_nets_{out_id:?}")), true, false);
                let apply_motif_net_input = Arc::new(ApplyMotifNetInputWrapper::new(Arc::clone(&input), Arc::clone(&apply_motif_net.net_wrapper)));

                let slice_step = self.slice_step;

                debug!("ApplyMotifNets::run: input.size={:?} motif_width={motif_width:?} self.motif_width(max)={:?} slice_step={slice_step} slices={slices} min_split_len={min_split_len} overlap_len={overlap_len}", input.size(), self.motif_width);

                let out_id_arc = Arc::new(out_id.clone()) as Arc<dyn CachableIdTr>;
                async move {
                    let map = HashMap::new();

                    let r = OOMHandler::new(
                        vec![
                            Adjusters::RunContext(Box::new(PermitsAdjuster::new(MAX_PERMITS, *PermitStep::TWENTY))),
                            Adjusters::RunContext(Box::new(DumpToCpu::new())),
                            Adjusters::Input(Box::new(SplitsAdjuster::<ChannelsData2Selector, MonoMetrics<1, 2>>::new_with_overlap(
                                [-1, min_split_len].into(),
                                [-1, overlap_len].into()
                            ))),
                        ], None)
                        .run(
                            &apply_motif_net,
                            Arc::clone(&apply_motif_net_input),
                            apply_motif_net_ctx,
                            Contexts::new(
                                SplitsAdjusterContextImpl::new(input._size(), SingleDimTensorOutputBuilder::new(out_id.clone(), 0, &[slices + 1/*TODO +1 for padded slice, ApplyMotifNet::run   */, 1], (Float, ThoDevice::from(device_idx).into()), BuildAs::Append)?),
                                NoopRunnableContext {},
                                PermitsAndDumpToCpuAdjustableContex::new(
                                    PermitsAdjustContextImpl::new(),
                                    DumpToCpuContextImpl::new(&map),
                                ),
                            ),
                        ).await;
                    debug!("ApplyMotifNets::run: r={r:?}");
                    r
                }

            });

        let matches = collect_futures_with_progress(
            futures,
            apply_motif_nets_len,
            crate::buffer_limits::APPLY_MOTIF_NETS_RUN_BUFFERED_LIMIT,
            StreamBuffering::Ordered,
            "ApplyMotifNets::run",
        )
        .await
        .into_iter()
        .collect::<Vec<_>>();

        debug!("ApplyMotifNets::run: matches={matches:?}");

        let match_lens = matches.iter().map(|v| v.t.size()[0]).collect::<Vec<_>>();
        let unique_match_lens = match_lens.iter().copied().collect::<HashSet<_>>();
        let max_match_len = match_lens.iter().copied().max().unwrap_or(0);
        let matches = if unique_match_lens.len() > 1 {
            warn!("ApplyMotifNets::run: mixed motif widths/filters produced different output lengths; lengths={match_lens:?}. Padding to max len={max_match_len} for concat.");
            matches.into_iter().map(|v| {
                let size = v.t.size();
                if size[0] == max_match_len {
                    return Ok(v);
                }
                let mut to_size = size.clone();
                to_size[0] = max_match_len;
                let t = pad_to(&v.t, to_size.as_slice(), 0.0)?;
                ThoTensor::new(v.id().clone(), t)
            }).collect::<Result<Vec<_>, ThoError>>()?
        } else {
            matches
        };

        debug!("ApplyMotifNets::run: matches_2={matches:?}");


        let res_t = Tensor::f_vstack(
            &matches.into_iter().map(|v| v.t).collect::<Vec<_>>(),
        )?;

        debug!("ApplyMotifNets::run: res_t.size={:?}", res_t.size());

        let motif_nets_applied = ThoTensor::new(
            self.out_id(input.id()),
            res_t, //.f_transpose(0, 1)?,
        )?;

        Ok(Arc::new(DevicesWrap::wrap(motif_nets_applied, cache_area.into())))

    }

    fn on_cached<'a>(&self, input: &impl WithId<ID=<<Self as Stage>::In<'a> as WithId>::ID>, ctx: &mut Self::Ctx)
    where
        Self: 'a
    {
        //TODO
    }
}


impl <C>OutId for ApplyMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}


#[derive(Debug)]
struct ApplyMotifNetInput<'a> {
    batch: &'a ThoTensor<ChannelsData2, SynKey>,
    motif_net: &'a DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>
}


impl Data for ApplyMotifNetInput<'_> {
    type Metrics = MonoMetrics<1, 2>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        MonoMetrics::<1, 2>::try_from(self.batch.size()).unwrap_or_default()
    }

    fn device(&self) -> usize {
        self.batch.device_idx()
    }

    fn mem_size(&self) -> u64 {
        self.batch.mem_size()
    }
}

impl <'a> ApplyMotifNetInput<'a> {
    pub fn new(batch: &'a ThoTensor<ChannelsData2, SynKey>, motif_net: &'a DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>) -> Self {
        Self {  batch, motif_net }
    }
}

impl <'a> Input for ApplyMotifNetInput<'a> {}
impl <'a> WithId for ApplyMotifNetInput<'a> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        self.batch.id()
    }
}

#[derive(Debug)]
struct ApplyMotifNetInputWrapper {
    batch: Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>,
    motif_net: Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>,
}

impl ApplyMotifNetInputWrapper {
    fn new(
        batch: Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>,
        motif_net: Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>,
    ) -> Self {
        Self { batch, motif_net }
    }
}

impl WithId for ApplyMotifNetInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        self.batch.id()
    }
}

impl WithDataRef for ApplyMotifNetInputWrapper {
    type O<'o> = ApplyMotifNetInput<'o>;

    async fn with_data_ref<R, F, P>(
        &self,
        _to_device_idx: usize,
        permits_ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<R, ThoError>
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send,
        P: RunCtx,
    {
        debug!("ApplyMotifNetInputWrapper::with_data_ref: 00. ctx.id={:?}", permits_ctx.id());

        with_data!(permits_ctx, self.batch, self.motif_net; move |ctx:P, batch:&ThoTensor<ChannelsData2, SynKey>, motif_net:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>|{
            let input = ApplyMotifNetInput::new(batch, motif_net);

            debug!("ApplyMotifNetInputWrapper::with_data_ref: 0. ctx.id={:?}", ctx.id());

            let r = do_run(&input, &ctx);

            let id = ctx.id().clone();
            drop(ctx);

            debug!("ApplyMotifNetInputWrapper::with_data_ref: 1. ctx.id={id:?}");

            r
        })
    }

    fn _size<'f>(&self) -> <Self::O<'f> as Data>::Metrics {
        self.batch.size().clone()
    }

    fn mem_size(&self) -> u64 {
        self.batch.mem_size() + self.motif_net.mem_size()
    }

    fn cache_area(&self) -> &CacheArea {
        self.batch.cache_area()
    }
}

impl WithSelectDataRef<ChannelsData2Selector> for ApplyMotifNetInputWrapper {
    async fn with_selected_data_ref<'o, R, F, P>(
        &self,
        to_device_idx: usize,
        mut permits_ctx: P,
        selector: ChannelsData2Selector,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> Result<R, ThoError>
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError>,
        P: RunCtx,
    {
        let selected_batch = self
            .batch
            .with_selected_data(to_device_idx, permits_ctx.branch(Some("apply_motif_net_input_selected_batch"), false, false), selector, |selected_batch, _ctx| Ok(selected_batch))
            .await?;

        with_data!(permits_ctx, self.motif_net; |mut ctx:P, motif_net:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>|{
            let input = ApplyMotifNetInput::new(&selected_batch, motif_net);
            debug!("ApplyMotifNetInputWrapper::with_selected_data_ref: 0. ctx.id={:?}", ctx.id());
            let r = do_run(&input, &ctx);

            let id = ctx.id().clone();
            drop(ctx);
            debug!("ApplyMotifNetInputWrapper::with_selected_data_ref: 1. ctx.id={id:?}");

            r
        })
    }
}

#[derive(Debug)]
struct ApplyMotifNet<C>
where
    C: RunCtx + Adjustable + 'static
{
    id: SynKey,
    net_wrapper:Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>,
    motif_width: u32,
    freq_filter_find_motifs_run_id: Option<RunId>,
    freq_filter: Option<Arc<FreqFilter<C>>>,
    slice_step:u32,
    pd:PhantomData<C>,
    #[cfg(test)]
    fail_if_size_bigger_than:Option<usize>,
}
impl <C> ApplyMotifNet<C>
where
    C: RunCtx + Adjustable + 'static
{
    pub fn new(
        net_wrapper:Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>,
        motif_width: u32,
        freq_filter_find_motifs_run_id: Option<RunId>,
        freq_filter: Option<Arc<FreqFilter<C>>>,
        slice_step:u32,
        #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Self {

        Self {
            id: mono_key!("ApplyMotifNet", "ApplyMotifNet") + net_wrapper.id(),
            net_wrapper,
            motif_width,
            freq_filter_find_motifs_run_id,
            freq_filter,
            slice_step,
            pd: PhantomData ,
            #[cfg(test)]
            fail_if_size_bigger_than
        }
    }
}

impl <C> Adjustable for ApplyMotifNet<C>
where
    C: RunCtx + Adjustable + 'static
{

}

impl <C> OutId for ApplyMotifNet<C>
where
    C: RunCtx + Adjustable + 'static
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}


impl <C>  Stage for ApplyMotifNet<C>
where
    C: RunCtx + Adjustable + 'static{
    type Ctx = C;
    type In<'a> = ApplyMotifNetInput<'a>;
    type Out = ThoTensor<MotifProbability, SynKey>;
}

impl <C> SyncStage for ApplyMotifNet<C> where
    C: RunCtx + Adjustable + 'static
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyMotifNet::run"))]
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        let motif_id = input.motif_net.extra().motif_id();

        trace!("ApplyMotifNet::do_run, self.id={:?} motif_id={motif_id} input.batch.t=\n{}", self.id, input.batch.t);

        #[cfg(test)]
        if self.fail_if_size_bigger_than.map(|v| v < input.batch.size()[1] as usize).unwrap_or(false) {
            debug!("SyncStage for ApplyMotifNet::do_run, triggering OOM: motif_id={motif_id} input.size={:?} self.fail_if_size_bigger_than={:?}", input.batch.size(), self.fail_if_size_bigger_than);
            return Err(ThoError::oom());
        }

        let out_id = self.out_id(input.id());
        let out = apply_motif_net_forward(out_id, input.batch, input.motif_net, self.slice_step, input.id())?;

        trace!("ApplyMotifNet::do_run, self.id={:?} out=\n{}", self.id, out.t);
        debug!("ApplyMotifNet::do_run, ok: motif_id={motif_id} input.id={:?} out.size={:?}", input.id(), out.size());

        Ok(out)
    }
}

pub fn apply_motif_net_forward(
    out_id: SynKey,
    batch: &ThoTensor<ChannelsData2, SynKey>,
    motif_net: &DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>,
    slice_step: u32,
    input_id: &SynKey,
) -> Result<ThoTensor<MotifProbability, SynKey>, ThoError> {
    let motif_id = motif_net.extra().motif_id();
    let c = motif_net.extra().channel() as i64;
    let motif_width = motif_net.extra().motif_net_params.num_features() as i64;

    debug!("ApplyMotifNet::do_run: 0. motif_id={motif_id} c={c} motif_width={motif_width} input.id={input_id:?} input.t.size={:?} slice_step={slice_step}", batch.t.size());
    let t = batch.t.f_i((c, ..))?;
    let t = if t.size()[0] >= motif_width {
        t
    } else {
        debug!("ApplyMotifNet::do_run: 1. motif_id={motif_id} padding t.size={:?} to motif_width={motif_width}", t.size());
        pad_to(&t, &[motif_width], 0.0)?
    };

    let slices = t.f_unfold(0, motif_width, slice_step as i64)?;
    debug!("ApplyMotifNet::do_run: 2_1 motif_id={motif_id} slices={slices:?}");

    let slices_tensor = ThoTensor::<Batch<Slice>, SimpleId>::from(slices);
    let out = motif_net.data().tho_forward(&slices_tensor, None, false)?;
    debug!("ApplyMotifNet::do_run: 2_2 motif_id={motif_id} out.size={:?}", out.size());

    let out_t = out.as_ref().f_squeeze()?.f_sigmoid()?;
    debug!("ApplyMotifNet::do_run: 2_2 motif_id={motif_id} out_t.size={:?}", out_t.size());

    let t_plot = t.f_view((1, -1))?;
    let out_plot = out_t.f_view((1, -1))?;
    debug_plot!(&t_plot, "apply_motif_net_run_input_{input_id}")?;
    debug_plot!(&out_plot, "apply_motif_net_run_input_motif_applied_{input_id}_{motif_id}")?;

    ThoTensor::new(out_id, out_t)
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
        Box::new(move || MotifNetParams::new(motif_width, 1, 1, step, 32, 10)),
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
) -> Result<Box<ThoModuleWithStore<DI, DO>>, ThoError> {
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

    let node = SeqBuilder::new(num_features, None)?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&[i64]>| {

                let r = t.unsqueeze(1);//dummy channel

                // debug!("dummy_motif_net: t=\n{t}");

                Ok(r)
            })),
            None,
        )?
        .split(
            vec![
                SeqBuilder::new(num_features, None)?.conv_1d("conv", l as u32, 1, 1, cfg, Some(Arc::clone(&tho_tensor)))?.end_seq()?,
                SeqBuilder::new(num_features, None)?
                    .closure(
                        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&[i64]>| {
                            Ok(t.f_pow_tensor_scalar(2)?)
                        })),
                        None,
                    )?
                    .conv_1d("energy", l as u32, 1, 1, cfg, Some(Arc::clone(&ones)))?
                    .closure(
                        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&[i64]>| {
                            Ok(t.f_clamp_min(1e-12)?) //.sqrt()
                        })),
                        None,
                    )?
                    .end_seq()?
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
        )?
        .closure(
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&[i64]>| {

                // debug!("dummy_motif_net: out=\n{t}");

                Ok(t.shallow_clone())
            })),
            None,
        )?
        .end_seq()?;



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
    use std::path::{Path, PathBuf};
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
    use thoosa_shared::runner::cache_areas::CacheArea;
    use thoosa_shared::runner::common::mock_run_ctx;
    use thoosa_shared::runner::common::{PermitsCtx, RunCtx};
    use thoosa_shared::runner::common::runnable::AsyncStage;
    use thoosa_shared::runner::global_ctx::GlobalCtx;
    use thoosa_common::builder::SeqBuilder;
    use thoosa_common::dims::{AnyDims, Batch, Dims, MotifProbability, Slice};
    use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
    use thoosa_common::my_tch::ThoModule;
    use thoosa_common::tch_test_utils;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{MotifNetParams, ThoModuleWithStore};
    use thoosa_common::debug_plot;
    use thoosa_common::utils::{assert_close, lin_t, plot_series_with_lanes, reset_print_options, test_net, to_vec_f32};
    use thoosa_common::ws::WsMockBuilder;
    use thoosa_common::tch_test_utils::Interval;
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunId;
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_shared::perf_logging::PERF;
    use thoosa_shared::traits::{Named, SimpleId};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::mono_key;
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use thoosa_store::MockStoreBuilder;
    use thoosa_with_data::with_data;
    use crate::tasks::apply_motif_nets::{dummy_motif_net, motif_test_net_, ApplyMotifNets};
    use crate::tasks::tensor_loader::mock_tensor_loader;


    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_without_oom() -> Result<(), ThoError> {
        test_apply_motif_nets_3_channels_2_motifs(None, 4000, 10, 10).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_without_oom_uneven() -> Result<(), ThoError> {
        test_apply_motif_nets_3_channels_2_motifs(None, 4321, 10, 10).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_step_1_with_oom() -> Result<(), ThoError> {
        test_apply_motif_nets_3_channels_2_motifs(Some(1000), 4000, 10, 10).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_step_2_with_oom() -> Result<(), ThoError> {
        test_apply_motif_nets_3_channels_2_motifs(Some(1000), 4000, 10, 10).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_with_oom_uneven() -> Result<(), ThoError> {
        test_apply_motif_nets_3_channels_2_motifs(Some(987), 4321, 10, 10).await
    }

    async fn test_apply_motif_nets_3_channels_2_motifs(trigger_oom_len: Option<usize>, input_n_0: usize, step: u32, slice_step: u32) -> Result<(), ThoError> {
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

        let motif_0_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_0", "motif_id")]), tch_test_utils::motifs_tensor_fixed(100, 3, MotifKind::Pulse).f_mul_scalar(0.2)?)?);
        let motif_0_1 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_1", "motif_id")]), tch_test_utils::motifs_tensor_fixed(100, 3, MotifKind::Triangle).f_mul_scalar(0.2)?)?);
        let motif_1_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_1_0", "motif_id")]), tch_test_utils::motifs_tensor_fixed(100, 3, MotifKind::Sin).f_mul_scalar(0.2)?)?);

        let net_id_0_0 = PathBuf::from(format!("0/{}_{}", motif_0_0_id, train_motifs_run_id));
        let net_id_0_1 = PathBuf::from(format!("0/{}_{}", motif_0_1_id, train_motifs_run_id));
        let net_id_1_0 = PathBuf::from(format!("1/{}_{}", motif_1_0_id, train_motifs_run_id));
        let (params_0_0, module_0_0) = motif_test_net_(Cpu, Arc::clone(&motif_0_0), step);
        let (params_0_1, module_0_1) = motif_test_net_(Cpu, Arc::clone(&motif_0_1), step);
        let (params_1_0, module_1_0) = motif_test_net_(Cpu, Arc::clone(&motif_1_0), step);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, 100, input_n_0 as u32)
                .with_load_motif_net(vec![
                    (net_id_0_0.clone(), params_0_0, module_0_0),
                    (net_id_0_1.clone(), params_0_1, module_0_1),
                    (net_id_1_0.clone(), params_1_0, module_1_0),
                ])
                .build(),
        ));

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_list_run_config_values(
                    find_motifs_run_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), step, 100, None, None).to_properties_iter(None).collect()),
                )

                .with_get_input_run_ids([
                    (train_motifs_run_id, Ok(vec![select_motifs_run_id])),
                    (select_motifs_run_id, Ok(vec![find_motifs_run_id]))
                ])
                .with_get_motif_meta([
                    (motif_0_0_id, Ok((0usize, find_motifs_run_id))),
                    (motif_0_1_id, Ok((0usize, find_motifs_run_id))),
                    (motif_1_0_id, Ok((1usize, find_motifs_run_id))),
                ])
                .with_list_selected_motifs([
                    (select_motifs_run_id, Ok(
                        vec![
                            vec![(motif_0_0_id, 1.0f64), (motif_0_1_id, 1.0f64)], //channel 0
                            vec![(motif_1_0_id, 1.0f64)], //channel 1
                            vec![]//channel 2
                        ]
                    ))
                ])
                .with_list_selected_nets([
                    (select_motifs_run_id, Some(0.3), None, Ok(vec![
                        (net_id_0_0.clone(), 0.1f32),
                        (net_id_0_1.clone(), 0.2f32),
                        (net_id_1_0.clone(), 0.25f32),
                    ]))
                ])
                .build()
        );

        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            &ws,
            &store,
            vec![select_motifs_run_id],
            mock_run_ctx::default(1), Some(0.3),
            None,
            slice_step,
            trigger_oom_len
        ).await?);

        let channels_patterns = vec![
            vec![motif_0_0.t_ref().i((0, ..)), motif_0_1.t_ref().i((0, ..))],
            vec![motif_1_0.t_ref().i((0, ..))],
            Vec::new(),
        ];
        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor_from_patterns(input_n_0, channels_patterns.as_slice(), 200, step as usize)?;


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_t={resource_a_t:?}");// resource_b_t={resource_b_t:?} resource_c_t={resource_c_t:?}


        let mut run_ctx_branch = mock_run_ctx::default(1);
        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test_batch", "test_batch")]), resource_a_t)?, CacheArea::default())),
            run_ctx_branch
        ).await?;

        debug!("test_apply_motif_nets_3_channels_1_motif: PERF={:?}", PERF);


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_intervals={resource_a_intervals:?}");

        let pulse_interval_starts_in_resources_a = resource_a_intervals[0][0][0].start;

        let triangle_interval_starts_in_resources_a = resource_a_intervals[0][1][0].start;

        let sin_interval_starts_in_resources_a = resource_a_intervals[1][0][0].start;

        let motif_width = motif_0_0.t_ref().size()[1] as i64;
        let slices = ((input_n_0 as i64 - motif_width) / slice_step as i64) + 1;
        let ctx_for_data = mock_run_ctx::default(1);
        with_data!(ctx_for_data, out; |_ctx, r:&ThoTensor<MotifProbability, SynKey>|{
            debug!("test_apply_motif_nets_3_channels_1_motif:  matches={:?}", r);

            let (pulse_matches_vals, pulse_matches_idx) = &r.t.i((0, 0..slices)).topk(1i64, 0, true, true);
            let (triangle_matches_vals, triangle_matches_idx) = &r.t.i((1, 0..slices)).topk(1i64, 0, true, true);
            let (sin_matches_vals, sin_matches_idx) = &r.t.i((2, 0..slices)).topk(1i64, 0, true, true);

            assert_close(pulse_interval_starts_in_resources_a as f64, step as f64 * pulse_matches_idx.double_value(&[0]), step as f64 * 3.);
            assert_close(triangle_interval_starts_in_resources_a as f64, step as f64 * triangle_matches_idx.double_value(&[0]), step as f64 * 3.);
            assert_close(sin_interval_starts_in_resources_a as f64, step as f64 * sin_matches_idx.double_value(&[0]), step as f64 * 3.);

            Ok::<_, ThoError>(())
        })?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_single_net_1_channel_1_motif_without_oom_step_3() -> Result<(), ThoError> {
        test_apply_motif_single_net_1_channel_1_motif(None, 4000, 3).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_single_net_1_channel_1_motif_without_oom_step_1() -> Result<(), ThoError> {
        test_apply_motif_single_net_1_channel_1_motif(None, 4000, 1).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_single_net_1_channel_1_motif_with_oom() -> Result<(), ThoError> {
        test_apply_motif_single_net_1_channel_1_motif(Some(1000), 4000, 3).await
    }


    async fn test_apply_motif_single_net_1_channel_1_motif(trigger_oom_len: Option<usize>, input_n_0: usize, step: u32) -> Result<(), ThoError> {
        thoosa_shared::test_utils::maybe_init_perf_tracing_stdout("debug");
        log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);

        let find_motifs_run_id:RunId = 1;
        let select_motifs_run_id:RunId = 2;

        let train_motifs_run_id:RunId = 3;
        let motif_0_0_id = 0; //channel 0 motif 0

        let motif_0_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_0", "motif_id")]), tch_test_utils::motifs_tensor_fixed(100, 3, MotifKind::Pulse).f_mul_scalar(0.2)?)?);

        let net_id_0_0 = PathBuf::from(format!("0/{}_{}", motif_0_0_id, train_motifs_run_id));
        let (params_0_0, module_0_0) = motif_test_net_(Cpu, Arc::clone(&motif_0_0), step);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 1, 100, input_n_0 as u32)
                .with_load_motif_net(vec![
                    (net_id_0_0.clone(), params_0_0, module_0_0),
                ])
                .build(),
        ));

        let mut store = Arc::new(
            MockStoreBuilder::new()
            .with_list_run_config_values(
                find_motifs_run_id,
                Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_id), step, 100, None, None).to_properties_iter(None).collect()),
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
            .with_list_selected_nets([
                (select_motifs_run_id, Some(0.3), None, Ok(vec![
                    (net_id_0_0.clone(), 0.1f32),
                ]))
            ])
            .build()
        );


        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            &ws,
            &store,
            vec![select_motifs_run_id],
            mock_run_ctx::default(1), Some(0.3) //Some(1000)
            ,
            None,
            step,
            trigger_oom_len
        ).await?);

        debug_plot!(motif_0_0.t_ref(), "motif_test")?;

        let channels_patterns = vec![vec![motif_0_0.t_ref().i((0, ..))]];
        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor_from_patterns(input_n_0, channels_patterns.as_slice(), 200, step as usize)?;


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_t={resource_a_t:?}");// resource_b_t={resource_b_t:?} resource_c_t={resource_c_t:?}

        let run_ctx = Arc::new(RwLock::new(mock_run_ctx::default(1)));

        let run_ctx_branch = run_ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout").branch(Some("apply_motif_nets_run"), false, false);

        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test_batch", "test_batch")]), resource_a_t)?, CacheArea::default())),
            run_ctx_branch
        ).await?;

        debug!("test_apply_motif_nets_3_channels_1_motif: PERF={:?}", PERF);


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_intervals={resource_a_intervals:?}");

        let pulse_interval_starts_in_resources_a = resource_a_intervals[0][0][0].start;


        let motif_width = motif_0_0.t_ref().size()[1] as i64;
        let slices = ((input_n_0 as i64 - motif_width) / step as i64) + 1;

        let ctx_for_data = run_ctx.try_write_for(thoosa_shared::LOCK_TIMEOUT).expect("lock timeout").branch(Some("apply_motif_nets_assert"), false, false);

        with_data!(ctx_for_data, out; |_ctx, r:&ThoTensor<MotifProbability, SynKey>|{
            debug!("test_apply_motif_nets_3_channels_1_motif:  matches={:?}", r);

            let (pulse_matches_vals, pulse_matches_idx) = &r.t.i((0, 0..slices)).topk(1i64, 0, true, true);

            assert_close(pulse_interval_starts_in_resources_a as f64, step as f64 * pulse_matches_idx.double_value(&[0]), step as f64 * 3.);

            Ok::<_, ThoError>(())
        })?;

        Ok(())
    }

    #[test]
    fn test_test_net() -> Result<(), ThoError> {
        let motif_t = tch_test_utils::motifs_tensor_fixed(100, 3, MotifKind::Pulse).f_mul_scalar(0.2)?;
        let channels_patterns = vec![vec![motif_t.i((0, ..))]];
        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor_from_patterns(2000, channels_patterns.as_slice(), 200, 1)?;

        debug!("test_test_net: resource_a_t.size={:?}", resource_a_t.size());

        let test_dir = Path::new("/tmp/test_test_net"); //.join(cdt_for_dir());


        let test_net = dummy_motif_net::<AnyDims, AnyDims>(100, 1., Cpu, "test", &motif_t.i((0, ..)), 1)?;

        let resource: ThoTensor<AnyDims, SimpleId> = resource_a_t.shallow_clone().into();
        let out = test_net.tho_forward(&resource, None, false)?.t;//.i((0, .., 0));//channel[1], slices, vals[1]

        plot_series_with_lanes(
            vec![
                (format!("Tensor"), BLACK, to_vec_f32(&resource_a_t.i((0, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>()),
                (format!("Motif_0"), BLUE, to_vec_f32(&motif_t.i((0, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>()),
                // (format!("Results"), RED, vec_f32_from(&out.transpose(1,2).upsample_linear1d(&[1000], false, None)).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>()),
            ],
            format!("sig").as_str(),
            (2000, 1000),
            &test_dir.join(&format!("input_{}.png", 1)),
            &[(0, 1000)],
        )?;

        debug!("test_test_net: out.size={:?}", out.size());

        plot_series_with_lanes(
             vec![
                 (format!("Results"), RED, to_vec_f32(&out.i((0, ..))).into_iter().enumerate().map(|(i, v)| (i as f32, v)).collect::<Vec<(f32, f32)>>()),
             ],
             format!("sig").as_str(),
             (2000, 1000),
             &test_dir.join(&format!("matches_{}.png", 1)),
             &[(0, 1000)],
         )?;

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
