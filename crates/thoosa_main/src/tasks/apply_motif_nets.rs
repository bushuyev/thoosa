#![allow(unexpected_cfgs)]

use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
use async_trait::async_trait;
use log::{debug, error, info, trace, warn};
use mockall_double::double;
use parking_lot::{Mutex, RwLock};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::fmt::{format, Debug};
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, Range};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;
use tch::kind::FLOAT_CPU;
use tch::nn::VarStore;
use tch::Kind::Float;
use tch::{nn, Device, IndexOp, Kind, Tensor};
use thoosa_common::batches::BatchLengths;
use thoosa_common::builder::SeqBuilder;
use thoosa_common::debug_plot;
use thoosa_common::dims::{AnyDims, Batch, ChannelsData, Dims, MotifProbability, Slice};
use thoosa_common::my_tch::ThoModule;
use thoosa_common::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use thoosa_common::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep, MAX_PARALLEL};
use thoosa_common::runner::adjusters::splits::SplitsAdjusterContext;
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::cachable::{Cached, ConsumeData, DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_common::runner::common::runnable::{AsyncStage, ConsumeResult, Stage, SyncStage};
use thoosa_common::runner::common::{DropIf, OutId, RunCtx};
use thoosa_common::runner::oom_handler::{Adjuster, AdjusterError, Adjusters, Contexts, InputContext, NoopInputContext, NoopRunnableContext, OOMHandler};
use thoosa_common::tho_device::ThoDevice;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_common::train_nets::{MotifNetParams, StftRunConfig, ThoModuleWithStore, TrainMotifsNetsResult};
use thoosa_common::utils::{load_run_config, pad_to, stretch};
use thoosa_common::ws::Workspace;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::{ElvVal, MonoKey, SynKey};
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::data_with_extra::DataWithExtra;
use thoosa_shared::resource::Resource;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, DataId, Metrics, MonoMetrics, Select, SimpleId, WithId};
use thoosa_shared::{collect_futures_with_progress, mono_key, StreamBuffering, ThoError};
use thoosa_store::{Store, TrainedNet};
use thoosa_with_data::with_data;

#[cfg(any(test, feature = "mock_global_ctx"))]
use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;

use crate::tasks::cross_dims::ChannelsData2Selector;
use crate::tasks::load_motif_net::{LoadMotifNet, LoadMotifNetInput, MotifNetMeta};
use crate::tasks::resources_motifs_clc_dist::ClcResourcesMotifsDistOutput;
use crate::tasks::stage_ids::{APPLY_MOTIF_NET, APPLY_MOTIF_NETS};
use crate::tasks::stft::SyncStft;
use crate::tasks::transforms::freq_filter::FreqFilter;
use thoosa_common::runner::tmp_cache::TmpCached;
use tracing::Instrument;
use thoosa_shared::tracked_await::track_await;


#[derive(Debug, cached_derive::Cached, Named)]
pub struct ApplyMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    id: SynKey,
    ws:Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
    store:Arc<Box<dyn Store>>,
    apply_motif_nets: Vec<(u32/*net_id*/, Arc<ApplyMotifNet<C>>)>,
    pub slice_step: u32,//should be in the input
    pub motif_width:u32,
    batch_applied_motif_nets:Arc<BatchAppliedMotifNets<C>>

}

#[derive(Debug)]
struct StepAlignedSplitsAdjuster {
    min_split_len: MonoMetrics<1, 2>,
    overlap_len: Option<MonoMetrics<1, 2>>,
    slice_step: i64,
}

impl StepAlignedSplitsAdjuster {
    fn new_with_overlap(min_split_len: MonoMetrics<1, 2>, overlap_len: MonoMetrics<1, 2>, slice_step: i64) -> Self {
        Self {
            min_split_len,
            overlap_len: Some(overlap_len),
            slice_step,
        }
    }

    fn align_split_len(&self, split_len: &mut MonoMetrics<1, 2>) {
        let dims: [i64; 2] = (&*split_len).into();
        let min_len = self.min_split_len.dim::<1>();
        let mut aligned_len = dims[1];
        if aligned_len > min_len {
            aligned_len = (aligned_len / self.slice_step) * self.slice_step;
            if aligned_len < min_len {
                aligned_len = min_len;
            }
        }
        if aligned_len != dims[1] {
            *split_len = [dims[0], aligned_len].into();
        }
    }
}

/// Computes how many motif scores from an overlapped split still belong to that split's non-overlapped core.
fn valid_partial_output_len(
    input_len: i64,
    split_start: i64,
    split_len: i64,
    motif_width: i64,
    slice_step: i64,
) -> i64 {
    if input_len < motif_width {
        return if split_start == 0 { 1 } else { 0 };
    }

    let last_valid_start = input_len - motif_width;
    if split_start > last_valid_start || split_len <= 0 {
        return 0;
    }

    let last_owned_start = (split_start + split_len - 1).min(last_valid_start);
    ((last_owned_start - split_start) / slice_step) + 1
}

#[derive(Debug)]
/// Reassembles split motif-net outputs while dropping scores that only come from overlap or impossible tail windows.
struct ApplyMotifNetSplitsContext {
    done_to: MonoMetrics<1, 2>,
    split_len: MonoMetrics<1, 2>,
    input_len: MonoMetrics<1, 2>,
    motif_width: i64,
    slice_step: i64,
    output_id: SynKey,
    output: Tensor,
    added_partial_output_len: i64,
}

// Matches the existing `ThoTensor` contract in this codebase: the context is only mutated via `&mut self`.
unsafe impl Sync for ApplyMotifNetSplitsContext {}

impl ApplyMotifNetSplitsContext {
    /// Preallocates the exact final output shape for one motif net so OOM split chunks can be appended safely.
    fn new(
        input_len: MonoMetrics<1, 2>,
        output_id: SynKey,
        output_size: i64,
        options: (Kind, Device),
        motif_width: i64,
        slice_step: i64,
    ) -> Result<Self, ThoError> {
        Ok(Self {
            done_to: input_len.default_end(),
            split_len: input_len.clone(),
            input_len,
            motif_width,
            slice_step,
            output_id,
            output: Tensor::f_zeros([output_size], options)?,
            added_partial_output_len: 0,
        })
    }
}

impl<'a, C> InputContext<'a, ApplyMotifNet<C>> for ApplyMotifNetSplitsContext
where
    C: RunCtx + Adjustable + 'static,
{
    async fn add_partial_output(&mut self, partial: ThoTensor<MotifProbability, SynKey>) {
        let input_len = self.input_len.dim::<1>();
        let split_start = self.done_to.dim::<1>();
        let split_len = self.split_len.dim::<1>();
        let valid_len = valid_partial_output_len(input_len, split_start, split_len, self.motif_width, self.slice_step);
        let partial_len = partial.size()[0];

        debug!("ApplyMotifNetSplitsContext::add_partial_output: input_len={input_len} split_start={split_start} split_len={split_len} motif_width={} slice_step={} partial_len={partial_len} valid_len={valid_len}", self.motif_width, self.slice_step);

        if valid_len > 0 {
            self.output
                .narrow(0, self.added_partial_output_len, valid_len)
                .copy_(&partial.t_ref().narrow(0, 0, valid_len));
            self.added_partial_output_len += valid_len;
        }

        self.done_to += self.split_len.clone();

        debug!("ApplyMotifNetSplitsContext::add_partial_output: done_to={:?} split_len={:?}", self.done_to, self.split_len);
    }

    fn all_done(&self) -> bool {
        self.done_to >= self.input_len
    }

    async fn build_final_output(self) -> Result<ThoTensor<MotifProbability, SynKey>, ThoError> {
        ThoTensor::new(self.output_id, self.output)
    }
}

impl SplitsAdjusterContext for ApplyMotifNetSplitsContext {
    type Metrics = MonoMetrics<1, 2>;

    fn input_len(&self) -> &Self::Metrics {
        &self.input_len
    }

    fn split_len(&self) -> &Self::Metrics {
        &self.split_len
    }

    fn done_to(&self) -> &Self::Metrics {
        &self.done_to
    }

    fn split_len_mut(&mut self) -> &mut Self::Metrics {
        &mut self.split_len
    }
}

#[async_trait]
impl<AC> Adjuster<AC> for StepAlignedSplitsAdjuster
where
    AC: SplitsAdjusterContext<Metrics = MonoMetrics<1, 2>> + Send + Sync,
{
    type A = ChannelsData2Selector;

    async fn adjust(
        &self,
        _adjustable: Self::A,
        adjust_context: &mut AC,
    ) -> Result<Self::A, AdjusterError<Self::A>> {
        let split_start = adjust_context.done_to();
        let mut split_end = split_start.clone();
        split_end += adjust_context.split_len().clone();

        if let Some(overlap_len) = self.overlap_len.as_ref() {
            split_end += overlap_len.clone();
        }
        adjust_context.input_len().cap(&mut split_end);

        debug!("StepAlignedSplitsAdjuster::adjust: split_start={split_start:?} split_end={split_end:?} input_len={:?}", adjust_context.input_len());

        Ok((split_start.clone()..split_end).into())
    }

    fn degrade(&self, adjust_context: &mut AC) -> bool {
        let prev_len = adjust_context.split_len().dim::<1>();
        (*adjust_context.split_len_mut()) /= 2;
        self.align_split_len(adjust_context.split_len_mut());
        let next_len = adjust_context.split_len().dim::<1>();
        let can_degrade = next_len >= self.min_split_len.dim::<1>() && next_len < prev_len;

        debug!("StepAlignedSplitsAdjuster::degrade: prev_len={prev_len} next_len={next_len} min_split_len={:?} can_degrade={can_degrade}", self.min_split_len);

        can_degrade
    }
}

impl <C> ApplyMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    pub async fn new<R>(
        ws: &Arc<parking_lot::RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>,
        train_motif_net_run_config_ids: Vec<RunConfigId>,
        mut stage_ctx:R,
        max_loss: Option<f32>,
        trained_nets_limit: Option<usize>,
        slice_step: u32,
        // #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>,
        quantize: bool
    ) -> Result<Self, ThoError>
    where
        R:RunCtx + Adjustable + 'static
    {

        // Load all motif nets matching max_loss and trained_nets_limit for the provided train_motif_nets_run_config_ids.
        let motif_nets = track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::new/1", store.list_trained_nets(train_motif_net_run_config_ids.clone(), max_loss, trained_nets_limit)).instrument(tracing::debug_span!("new/1", file = file!(), line = line!())).await?;

        if motif_nets.is_empty() {
            return  Err(ThoError::new(format!("No selected nets for train_motif_nets_run_config_ids={train_motif_net_run_config_ids:?}")));
        };

        track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::new/2", Self::from_trained_nets(&ws, &store, stage_ctx, slice_step, fail_if_size_bigger_than, motif_nets, quantize)).instrument(tracing::debug_span!("new/2", file = file!(), line = line!())).await
    }

    pub async fn from_trained_nets<R>(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<Box<dyn Store>>, //for logging only
        mut stage_ctx: R,
        slice_step: u32,
        fail_if_size_bigger_than: Option<usize>,
        trained_nets: Vec<TrainedNet>,
        quantize: bool
    ) -> Result<ApplyMotifNets<C>, ThoError>
    where
        R: RunCtx + Adjustable + 'static
    {
        let load_motif_net = Arc::new(LoadMotifNet::<R>::new(&ws, quantize));//TODO move to new
        let trained_net_ids = trained_nets.iter().map(|v|v.id).collect::<Vec<_>>();
        let trained_nets_len = trained_nets.len();
        let id = mono_key!(APPLY_MOTIF_NETS, "ApplyMotifNets") + &SynKey::new(trained_nets.iter().map(|net| mono_key!(net.id, format!("trained_net_id_{}", net.id))).collect::<Vec<_>>());

        //this defines order in which motif nets should be applied to a resource before running label net
        let futures = trained_nets
            .into_iter()
            .map(|trained_net| {
                let id = trained_net.id;
                let path = trained_net.net_path;
                let ws = Arc::clone(&ws);
                let load_motif_net = Arc::clone(&load_motif_net);
                let load_ctx = stage_ctx.branch(false);
                async move {
                    match track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::from_trained_nets/1", load_motif_net.run_cached(Arc::new(DevicesWrap::wrap(LoadMotifNetInput::new(path, false))), load_ctx)).instrument(tracing::debug_span!("from_trained_nets/1", file = file!(), line = line!())).await {
                        Ok(net) => {
                            debug!("ApplyMotifNets::new: loaded net={net:?}");
                            Ok::<_, ThoError>(Some((id, net)))
                        }
                        Err(e) => {
                            error!("ApplyMotifNets::new: can't load net for motif; there must have been an error while training the net: e={e:?}");
                            Ok::<_, ThoError>(None)
                        }
                    }
                }
            });

        let loaded_motif_nets: Vec<(u32, Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>)> = track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::from_trained_nets/2", collect_futures_with_progress(
            futures,
            trained_nets_len,
            crate::buffer_limits::APPLY_MOTIF_NETS_NEW_BUFFERED_LIMIT,
            StreamBuffering::Ordered, //to keep same order as in trained_nets_ids
            "ApplyMotifNets::new/load motif nets",
        ))
            .instrument(tracing::debug_span!("from_trained_nets/2", file = file!(), line = line!())).await
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        if loaded_motif_nets.is_empty() {
            return Err(ThoError::new(format!("No motif nets for trained_net_ids={trained_net_ids:?}")));
        };

        let mut motif_widths = HashSet::new();
        let mut freq_filters_by_find_motifs_run_config_id: HashMap<RunConfigId, Option<Arc<FreqFilter<C>>>> = HashMap::new();
        let mut loaded_motif_nets_with_filters = Vec::with_capacity(loaded_motif_nets.len());
        for (net_id, motif_net) in loaded_motif_nets {
            let extra = track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::from_trained_nets/3", motif_net.extra()).instrument(tracing::debug_span!("from_trained_nets/3", file = file!(), line = line!())).await?;

            let motif_id = extra.motif_id();
            let motif_width = extra.motif_net_params.num_features();
            motif_widths.insert(motif_width);

            let (_channel, find_motifs_run_config_id) = track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::from_trained_nets/4", store.get_motif_meta(motif_id)).instrument(tracing::debug_span!("from_trained_nets/4", file = file!(), line = line!())).await?;
            let find_motifs_run_config: FindMotifsRunConfig = track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::from_trained_nets/5", load_run_config(store, find_motifs_run_config_id)).instrument(tracing::debug_span!("from_trained_nets/5", file = file!(), line = line!())).await?;

            let freq_filter = if let Some(freq_filter) = freq_filters_by_find_motifs_run_config_id.get(&find_motifs_run_config_id) {
                freq_filter.clone()
            } else {
                let freq_filter = find_motifs_run_config
                    .freq_filter
                    .as_ref()
                    .map(|freq_filter| FreqFilter::<C>::new(freq_filter.as_slice(), ws, find_motifs_run_config_id))
                    .transpose()?
                    .map(Arc::new);
                freq_filters_by_find_motifs_run_config_id.insert(find_motifs_run_config_id, freq_filter.clone());
                freq_filter
            };

            let freq_filter_find_motifs_run_config_id = if freq_filter.is_some() { Some(find_motifs_run_config_id) } else { None };
            loaded_motif_nets_with_filters.push((motif_id, net_id, motif_net, motif_width, freq_filter_find_motifs_run_config_id, freq_filter));
        }

        if motif_widths.len() > 1 {
            warn!("ApplyMotifNets::new: mixed motif widths in selected nets; widths={motif_widths:?} trained_net_ids={trained_net_ids:?}. Will run per-net and pad outputs for concat.");
        }

        let Some(motif_width) = motif_widths.iter().copied().max() else {
            return Err(ThoError::new("no motif_width"));
        };

        debug!("ApplyMotifNets::new: motif_width(max)={motif_width:?} motif_widths={motif_widths:?} slice_step={slice_step:?} loaded_motif_nets.len={}", loaded_motif_nets_with_filters.len());

        let batch_applied_motif_nets = Arc::new(BatchAppliedMotifNets::<C>::new(id.clone()));
        Ok(Self {
            id,
            ws: Arc::clone(&ws),
            store: Arc::clone(&store),
            apply_motif_nets: loaded_motif_nets_with_filters
                .into_iter()
                .map(|(motif_id, net_id, net, motif_width, freq_filter_find_motifs_run_config_id, freq_filter)| (net_id, Arc::new(ApplyMotifNet::new(
                    motif_id,
                    net,
                    motif_width,
                    freq_filter_find_motifs_run_config_id,
                    freq_filter,
                    slice_step, //TODO shouldn't be same as one used when training motif net
                    #[cfg(test)]
                    fail_if_size_bigger_than
                )))).collect(),
            slice_step,
            motif_width,
            batch_applied_motif_nets
        })
    }

    pub fn num_nets(&self) -> usize {
        self.apply_motif_nets.len()
    }

    pub fn motif_nets_ids(&self) -> Vec<u32> {
        self.apply_motif_nets.iter().map(|(id, ..)| *id).collect()
    }
}

impl <C>Stage for ApplyMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{
    type Ctx = C;
    type In<'a> = ThoTensor<ChannelsData, SynKey>;
    type Out = DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>;// extra carries pre-stretch original lengths per motif net; TODO it should be MotifsProbabilities - for many motifs
}



impl <C>AsyncStage for ApplyMotifNets<C>
where
    C: RunCtx + Adjustable + 'static
{

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ApplyMotifNets::run"))]
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {

        let device_idx = ctx.tho_device_idx();

        debug!("ApplyMotifNets::run: self.loaded_motif_nets.len={:?} input={input:?}", self.apply_motif_nets.len());

        let apply_motif_nets = self.apply_motif_nets.clone();
        let apply_motif_nets_len = apply_motif_nets.len();
        let mut inputs_by_filter_group: HashMap<Option<RunConfigId>, Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>> = HashMap::new();
        inputs_by_filter_group.insert(None, Arc::clone(&input));

        for (net_id, apply_motif_net) in &apply_motif_nets {
            let group_key = apply_motif_net.freq_filter_find_motifs_run_config_id;
            if inputs_by_filter_group.contains_key(&group_key) {
                continue;
            }
            let filtered_input: Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>> = match &apply_motif_net.freq_filter {
                Some(freq_filter) => {
                    debug!("ApplyMotifNets::run: freq_filter={freq_filter:?} motif_id={:?} apply_motif_net.freq_filter_find_motifs_run_config_id={:?}", apply_motif_net.motif_id, apply_motif_net.freq_filter_find_motifs_run_config_id);

                    track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::AsyncStage::run/1", freq_filter.clone().run_cached(Arc::clone(&input), ctx.branch(true))).instrument(tracing::debug_span!("run/1", file = file!(), line = line!())).await?
                }
                None => {
                    debug!("ApplyMotifNets::run: no freq_filter for  motif_id={:?}", apply_motif_net.motif_id);
                    Arc::clone(&input)
                },
            };
            inputs_by_filter_group.insert(group_key, filtered_input);
        }

        let tmp_cache = track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::AsyncStage::run/2", TmpCached::new("tmp_apply_motif_nets")).instrument(tracing::debug_span!("run/2", file = file!(), line = line!())).await;
        let tmp_cache_name = tmp_cache.registered_name();
        let tmp_cache:Arc<Mutex<TmpCached<ThoTensor<MotifProbability, SynKey>>>> = Arc::new(Mutex::new(tmp_cache));

        let inputs_by_filter_group = Arc::new(inputs_by_filter_group);
        // TODO(perf): group nets by filter/run id, channel, and motif width so the same input slice can feed a batched multi-net pass instead of thousands of tiny ApplyMotifNet/OOMHandler runs.
        let futures = apply_motif_nets
            .into_iter()
            .map(|(net_id, apply_motif_net)|{
                let inputs_by_filter_group = Arc::clone(&inputs_by_filter_group);
                let group_key = apply_motif_net.freq_filter_find_motifs_run_config_id;

                let input = inputs_by_filter_group.get(&group_key).cloned().expect("filtered input for group must exist");
                let motif_width = apply_motif_net.motif_width;
                let motif_width_i64 = motif_width as i64;
                let step_aligned_min_split_len = ((motif_width_i64 + self.slice_step as i64 - 1) / self.slice_step as i64) * self.slice_step as i64;
                let min_split_len = step_aligned_min_split_len.max(motif_width_i64);
                let overlap_len = motif_width.saturating_sub(self.slice_step) as i64;
                let input_len = input.size().dim::<1>();

                let slices = if input_len >= motif_width_i64 {
                    ((input_len - motif_width_i64) / self.slice_step as i64) + 1
                } else {
                    1
                };
                let out_id = apply_motif_net.out_id(&input.id());
                let apply_motif_net_ctx = ctx.branch(true);
                let apply_motif_net_input = Arc::new(ApplyMotifNetInputWrapper::new(Arc::clone(&input), Arc::clone(&apply_motif_net.net_wrapper)));//The reason to copy apply_motif_net.net_wrapper into apply_motif_net input is to get it out of Wrapper in sync do_run method

                let slice_step = self.slice_step;

                debug!("ApplyMotifNets::run: input.size={:?} motif_width={motif_width:?} self.motif_width(max)={:?} slice_step={slice_step} slices={slices} min_split_len={min_split_len} overlap_len={overlap_len}", input.size(), self.motif_width);

                let out_id_arc = Arc::new(out_id.clone()) as Arc<dyn CachableIdTr>;
                let tmp_cache = Arc::clone(&tmp_cache);

                async move {
                    let set = HashSet::<&dyn CachableIdTr>::new();

                    let required_ids: HashMap<&'static str, &HashSet<&dyn CachableIdTr>> =HashMap::from([
                        (tmp_cache_name, &set)
                    ]);

                    let r = track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::AsyncStage::run/3", OOMHandler::new(
                        vec![
                            Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::TWENTY))),
                            Adjusters::RunContext(Box::new(DumpToCpu::new("apply_moitf_net"))),
                            Adjusters::Input(Box::new(StepAlignedSplitsAdjuster::new_with_overlap(
                                [-1, min_split_len].into(),
                                [-1, overlap_len].into(),
                                slice_step as i64,
                            ))),
                        ], None)
                        .run(
                            &apply_motif_net,
                            Arc::clone(&apply_motif_net_input),
                            apply_motif_net_ctx,
                            Contexts::new(
                                ApplyMotifNetSplitsContext::new(
                                    input._size(),
                                    out_id.clone(),
                                    slices,
                                    (Float, ThoDevice::from(device_idx).into()),
                                    motif_width_i64,
                                    slice_step as i64,
                                )?,
                                NoopRunnableContext {},
                                MaxParallelAndDumpToCpuAdjustableContex::new(
                                    MaxParallelAdjustContextImpl::new(),
                                    DumpToCpuContextImpl::new(&required_ids),
                                ),
                            ),
                        )).instrument(tracing::debug_span!("run/3", file = file!(), line = line!())).await;

                    debug!("ApplyMotifNets::run: after OOMHandler r={r:?}");
                    if let Ok(r)= r {
                        tmp_cache.lock().insert(r);
                    }

                    Ok(())
                }

            });

        let motif_nets_applied = track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::AsyncStage::run/4", collect_futures_with_progress(
            futures,
            apply_motif_nets_len,
            crate::buffer_limits::APPLY_MOTIF_NETS_RUN_BUFFERED_LIMIT,
            StreamBuffering::Unordered, //it's sorted in BatchAppliedMotifNets
            "ApplyMotifNets::run",
        ))
        .instrument(tracing::debug_span!("run/4", file = file!(), line = line!())).await
        .into_iter()
        .collect::<Vec<_>>();



        let batch_input = BatchAppliedMotifNetsInputWrapper::new(Arc::into_inner(tmp_cache).unwrap().into_inner());
        let required_tmp_cache_ids = HashSet::<&dyn CachableIdTr>::new();
        let required_ids = HashMap::from([(tmp_cache_name, &required_tmp_cache_ids)]);
        let oom_handler: OOMHandler<'_, '_, _, _, _, _, ()> = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FIFTY))),
                Adjusters::RunContext(Box::new(DumpToCpu::new("batch_apply_motif_nets"))),
            ],
            None,
        );
        let motif_nets_applied = track_await("thoosa_main::tasks::apply_motif_nets::ApplyMotifNets::AsyncStage::run/5", oom_handler
        .consume(
            &self.batch_applied_motif_nets,
            batch_input,
            ctx.branch(true),
            Contexts::new(
                NoopInputContext::<BatchAppliedMotifNets<C>>(None),
                NoopRunnableContext {},
                MaxParallelAndDumpToCpuAdjustableContex::new(
                    MaxParallelAdjustContextImpl::new(),
                    DumpToCpuContextImpl::new(&required_ids),
                ),
            ),
        ))
        .instrument(tracing::debug_span!("run/5", file = file!(), line = line!())).await?;

        Ok(Arc::new(DevicesWrap::wrap(motif_nets_applied)))

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
pub struct BatchAppliedMotifNetsInput {
    id: SynKey,
    device_idx: usize,
    motif_nets_applied: Vec<ThoTensor<MotifProbability, SynKey>>,

}

impl BatchAppliedMotifNetsInput {
    pub fn new(id: SynKey, motif_nets_applied: Vec<ThoTensor<MotifProbability, SynKey>>, device_idx: usize) -> Self {
        Self {
            id,
            device_idx,
            motif_nets_applied,
        }
    }
}

impl WithId for BatchAppliedMotifNetsInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Data for BatchAppliedMotifNetsInput {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.motif_nets_applied.len().into()
    }

    fn device(&self) -> usize {
        self.device_idx
    }

    fn mem_size(&self) -> u64 {
        self.motif_nets_applied.iter().map(|vv| vv.mem_size()).sum()
    }
}

impl Input for BatchAppliedMotifNetsInput {}

// impl Select<()> for BatchAppliedMotifNetsInput {
//     type Output = Self;
//
//     fn subview(&self, _selector: ()) -> Result<Self::Output, ThoError> {
//         Ok(self.clone())
//     }
// }

impl Adjustable for BatchAppliedMotifNetsInput {}

//to avoid Only traits defined in the current crate can be implemented for arbitrary types [E0117]
#[derive(Debug)]
struct BatchAppliedMotifNetsInputWrapper {
    id:SynKey,
    tmp_cache: TmpCached<ThoTensor<MotifProbability, SynKey>>
}

impl BatchAppliedMotifNetsInputWrapper {
    pub fn new(tmp_cache: TmpCached<ThoTensor<MotifProbability, SynKey>>) -> Self {
        Self { id: SynKey::new(vec![/*todo*/]), tmp_cache }
    }
}

impl WithId for BatchAppliedMotifNetsInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl ConsumeData for BatchAppliedMotifNetsInputWrapper {
    type O = BatchAppliedMotifNetsInput;

    fn consume<R: Send, F, P>(
        self,
        to_device_idx: usize,
        ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl std::future::Future<Output = ConsumeResult<Self::O, R>> + Send
    where
        F: FnOnce(Self::O, P) -> ConsumeResult<Self::O, R> + Send,
        P: RunCtx,
    {
        async move {
            let BatchAppliedMotifNetsInputWrapper { id, tmp_cache } = self;
            let device_idx = ctx.tho_device_idx();

            match track_await("thoosa_main::tasks::apply_motif_nets::BatchAppliedMotifNetsInputWrapper::ConsumeData::consume/1", tmp_cache.take_all_recovering(device_idx)).instrument(tracing::debug_span!("consume/1", file = file!(), line = line!())).await {
                Ok(motif_nets_applied) => {
                    let len = motif_nets_applied.len();
                    do_run(BatchAppliedMotifNetsInput::new(id, motif_nets_applied, device_idx), ctx)
                }
                Err(err) => {
                    let len = err.data.len();
                    let input = BatchAppliedMotifNetsInput::new(id, err.data, device_idx);
                    ConsumeResult::Failed {
                        input,
                        error: err.error,
                    }
                }
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="TmpCacheWrapper::burp"))]
    fn burp(o: Self::O) -> impl Future<Output=Self> + Send {
        debug!("TmpCacheWrapper::burp: o.motif_nets_applied.len={}", o.motif_nets_applied.len());
        async {
            let cache = track_await("thoosa_main::tasks::apply_motif_nets::BatchAppliedMotifNetsInputWrapper::ConsumeData::burp/1", TmpCached::new("tmp tmp")).instrument(tracing::debug_span!("burp/1", file = file!(), line = line!())).await;
            for x in o.motif_nets_applied {
                cache.insert(x);
            }
            BatchAppliedMotifNetsInputWrapper::new(cache)
        }
    }
}


#[derive(Debug, Named)]
pub struct BatchAppliedMotifNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    id: SynKey,
    pd: PhantomData<C>,
}

impl<C> BatchAppliedMotifNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(id: SynKey) -> Self {
        Self {
            id,
            pd: PhantomData,
        }
    }
}

impl<C> Adjustable for BatchAppliedMotifNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
}

impl<C> OutId for BatchAppliedMotifNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl<C> Stage for BatchAppliedMotifNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = BatchAppliedMotifNetsInput;
    type Out = DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>;
}

impl<C> SyncStage for BatchAppliedMotifNets<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="BatchAppliedMotifNets::consume"))]
    fn consume<'f>(self: Arc<Self>, mut input: Self::In<'f>, _ctx: Self::Ctx) -> ConsumeResult<Self::In<'f>, Self::Out> {
        let input_id = input.id().clone();
        let mut motif_nets_applied = std::mem::take(&mut input.motif_nets_applied);

        debug!("BatchAppliedMotifNets::consume: motif_nets_applied.len={} input.id={:?} motif_nets_applied={motif_nets_applied:?}", motif_nets_applied.len(), input_id);

        motif_nets_applied.sort_by(|a,b|a.id().partial_cmp(&b.id()).unwrap_or(Ordering::Equal));

        let motif_nets_applied_lens = motif_nets_applied.iter().map(|v| v.t_ref().size()[0]).collect::<Vec<_>>();

        let unique_motif_nets_applied_lens = motif_nets_applied_lens.iter().copied().collect::<HashSet<_>>();
        let max_match_len = motif_nets_applied_lens.iter().copied().max().unwrap_or(0);
        let output_options = motif_nets_applied
            .first()
            .map(|v| (v.t_ref().kind(), v.t_ref().device()))
            .unwrap_or((Float, ThoDevice::from(input.device_idx).into()));


        if unique_motif_nets_applied_lens.len() > 1 {
            debug!("BatchAppliedMotifNets::consume: mixed motif widths/filters produced different output lengths; lengths={motif_nets_applied_lens:?}. Padding to max len={max_match_len} for concat.");
        }

        // TODO(perf): this pads and copies each motif-net output row by row; batch padding/stretching into a stack/cat operation would reduce tensor indexing overhead for large net counts.
        let res_t = match Tensor::f_zeros([motif_nets_applied.len() as i64, max_match_len], output_options) {
            Ok(res_t) => res_t,
            Err(error) => {
                input.motif_nets_applied = motif_nets_applied;
                return ConsumeResult::Failed { input, error: error.into() };
            }
        };

        for (row_i, motif_net_applied) in motif_nets_applied.iter().enumerate() {
            let size = motif_net_applied.t_ref().size();
            let stretched;
            let src = if size[0] == max_match_len {
                motif_net_applied.t_ref()
            } else {
                stretched = match stretch(motif_net_applied.t_ref(), max_match_len as usize, 0) {
                    Ok(stretched) => stretched,
                    Err(error) => {
                        input.motif_nets_applied = motif_nets_applied;
                        return ConsumeResult::Failed { input, error };
                    }
                };
                &stretched
            };

            let mut dst = match res_t.f_get(row_i as i64) {
                Ok(dst) => dst,
                Err(error) => {//TODO it's not rally recoverable, returning input only wastes memory
                    input.motif_nets_applied = motif_nets_applied;
                    return ConsumeResult::Failed { input, error: error.into() };
                }
            };
            if let Err(error) = dst.f_copy_(src) {
                input.motif_nets_applied = motif_nets_applied;
                return ConsumeResult::Failed { input, error: error.into() };
            }
        }

        debug!("BatchAppliedMotifNets::consume: res_t.size={:?}", res_t.size());

        let motif_nets_applied = match ThoTensor::new(self.out_id(&input_id), res_t) {
            Ok(motif_nets_applied) => motif_nets_applied,
            Err(error) => {
                input.motif_nets_applied = motif_nets_applied;
                return ConsumeResult::Failed { input, error };
            }
        };

        ConsumeResult::Ok(DataWithExtra::new(motif_nets_applied, motif_nets_applied_lens))
    }
}


#[derive(Debug)]
struct ApplyMotifNetInput<'a> {
    batch: &'a ThoTensor<ChannelsData, SynKey>,
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
    pub fn new(batch: &'a ThoTensor<ChannelsData, SynKey>, motif_net: &'a DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>) -> Self {
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
    batch: Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>,
    motif_net: Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>,
}

impl ApplyMotifNetInputWrapper {
    fn new(
        batch: Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>,
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

    fn with_data_ref<'a, R: Send + 'a, F, P>(
        &'a self,
        _to_device_idx: usize,
        mut permits_ctx: P,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move {
            debug!("ApplyMotifNetInputWrapper::with_data_ref: 00. ctx.id={:?}", permits_ctx.id());
            // let x_ctx = permits_ctx.branch(Some("xxx"), false);

            with_data!(permits_ctx, self.batch, self.motif_net; move |mut ctx:P, batch:&ThoTensor<ChannelsData, SynKey>, motif_net:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>|{
                let input = ApplyMotifNetInput::new(batch, motif_net);

                debug!("ApplyMotifNetInputWrapper::with_data_ref: 0. ctx.id={:?}", ctx.id());

                let r = do_run(&input, &ctx);

                let id = ctx.id().clone();
                // ctx.drop_permits(DropIf::Always);
                drop(ctx);

                debug!("ApplyMotifNetInputWrapper::with_data_ref: 1. ctx.id={id:?}");

                r
            })
        }
    }

    fn _size<'f>(&self) -> <Self::O<'f> as Data>::Metrics {
        self.batch.size().clone()
    }

    fn mem_size(&self) -> u64 {
        self.batch.mem_size() + self.motif_net.mem_size()
    }

}

impl WithSelectDataRef<ChannelsData2Selector> for ApplyMotifNetInputWrapper {
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(
        &'a self,
        _to_device_idx: usize,
        mut permits_ctx: P,
        selector: ChannelsData2Selector,
        _trace_ctx: Option<&'static str>,
        do_run: F,
    ) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
        for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
        P: RunCtx + 'a,
    {
        async move {

            with_data!(permits_ctx, self.motif_net, self.batch; |mut ctx:P, motif_net:&DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>, batch:&ThoTensor<ChannelsData, SynKey>|{
                let b = batch.subview(selector)?;
                let input = ApplyMotifNetInput::new(&b, motif_net);
                debug!("ApplyMotifNetInputWrapper::with_selected_data_ref: 0. ctx.id={:?}", ctx.id());
                let r = do_run(&input, &ctx);

                let id = ctx.id().clone();
                drop(ctx);
                debug!("ApplyMotifNetInputWrapper::with_selected_data_ref: 1. ctx.id={id:?}");

                r
            })
        }
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
    freq_filter_find_motifs_run_config_id: Option<RunConfigId>,
    freq_filter: Option<Arc<FreqFilter<C>>>,
    slice_step:u32,
    pd:PhantomData<C>,
    #[cfg(test)]
    fail_if_size_bigger_than:Option<usize>,
    motif_id:MonoKey,
}
impl <C> ApplyMotifNet<C>
where
    C: RunCtx + Adjustable + 'static
{
    pub fn new(
        motif_id:u32,
        net_wrapper:Arc<DevicesWrap<DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>>>,
        motif_width: u32,
        freq_filter_find_motifs_run_config_id: Option<RunConfigId>,
        freq_filter: Option<Arc<FreqFilter<C>>>,
        slice_step:u32,
        #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Self {

        debug!("ApplyMotifNet::new: net_wrapper.id={:?}", net_wrapper.id());


        Self {
            id: mono_key!(APPLY_MOTIF_NET, "ApplyMotifNet") + net_wrapper.id(),
            net_wrapper,
            motif_width,
            freq_filter_find_motifs_run_config_id,
            freq_filter,
            slice_step,
            pd: PhantomData ,
            #[cfg(test)]
            fail_if_size_bigger_than,
            motif_id: mono_key!(motif_id, "motif_id")
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
        &self.motif_id + input_id//it's assumed that there can be only one net for a motif
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

        debug!("ApplyMotifNet::do_run, self.id={:?} motif_id={motif_id} input.batch.t={:?}", self.id, input.batch.t);
        trace!("ApplyMotifNet::do_run, self.id={:?} motif_id={motif_id} input.batch.t=\n{}", self.id, input.batch.t);

        #[cfg(test)]
        if self.fail_if_size_bigger_than.map(|v| v < input.batch.size()[1] as usize).unwrap_or(false) {
            debug!("SyncStage for ApplyMotifNet::do_run, triggering OOM: motif_id={motif_id} input.size={:?} self.fail_if_size_bigger_than={:?}", input.batch.size(), self.fail_if_size_bigger_than);
            return Err(ThoError::oom());
        }

        let out_id = self.out_id(&input.id());

        let out = apply_motif_net_forward(out_id, input.batch, input.motif_net, self.slice_step, input.id())?;

        trace!("ApplyMotifNet::do_run, self.id={:?} out=\n{}", self.id, out.t);
        debug!("ApplyMotifNet::do_run, ok: motif_id={motif_id} input.id={:?} out.id()={:?} out.size={:?}", input.id(), out.id(), out.size());

        Ok(out)
    }
}

pub fn apply_motif_net_forward(
    out_id: SynKey,
    input: &ThoTensor<ChannelsData, SynKey>,
    motif_net: &DataWithExtra<ThoModuleWithStore<Batch<Slice>, MotifProbability>, MotifNetMeta>,
    slice_step: u32,
    input_id: &SynKey,
) -> Result<ThoTensor<MotifProbability, SynKey>, ThoError>
{
    let motif_id = motif_net.extra().motif_id();
    let c = motif_net.extra().channel() as i64;
    let motif_width = motif_net.extra().motif_net_params.num_features() as i64;

    debug!("ApplyMotifNet::do_run: 0. motif_id={motif_id} c={c} motif_width={motif_width} input.id={input_id:?} input.t.size={:?} slice_step={slice_step}", input.t.size());
    trace!("ApplyMotifNet::do_run: batch.t=\n{}", input.t);
    let t = input.t.f_i((c, ..))?;
    let t = if t.size()[0] >= motif_width {
        t
    } else {
        warn!("ApplyMotifNet::do_run: 1. motif_id={motif_id} padding t.size={:?} to motif_width={motif_width} out_id={out_id:?}", t.size());
        pad_to(&t, &[motif_width], 0.0)?
    };

    // TODO(perf): cache unfolded channel windows by (resource, channel, motif_width, slice_step) across motif nets; tracing shows this stage is invoked tens of thousands of times.
    let slices = t.f_unfold(0, motif_width, slice_step as i64)?;
    debug!("ApplyMotifNet::do_run: 2_1 motif_id={motif_id} slices={slices:?}");

    // let stft_slices = stft.unwrap().build_spectrogram(&slices.totype(Float))?;
    //
    // debug!("ApplyMotifNet::do_run: 2_1 motif_id={motif_id} stft_slices={stft_slices:?}");
    //
    // let slices_tensor = ThoTensor::<Batch<Slice>, SimpleId>::from(stft_slices); //.f_totype(Float)?
    let slices_tensor = ThoTensor::<Batch<Slice>, SimpleId>::from(slices);
    let out = motif_net.data().tho_forward(&slices_tensor, None, false)?;
    debug!("ApplyMotifNet::do_run: 2_2 motif_id={motif_id} out.size={:?}", out.size());

    let out_t = out.as_ref().f_view([-1])?.f_sigmoid()?;
    debug!("ApplyMotifNet::do_run: 2_2 motif_id={motif_id} out_t.size={:?}", out_t.size());
    trace!("ApplyMotifNet::do_run: out_t=\n{out_t}");

    debug_plot!(&t.f_view((1, -1))?, "apply_motif_net_run_input_{input_id}")?;
    debug_plot!(&out_t.f_view((1, -1))?, "apply_motif_net_run_input_motif_applied_{input_id}_{motif_id}")?;

    ThoTensor::new(out_id, out_t.totype(Float))
}


#[cfg(test)]
pub fn motif_test_net_(
    device: Device,
    motif_t: Arc<ThoTensor<Batch<Slice>, SynKey>>,
    step: u32, kernel: u32,
) -> (
    Box<dyn Fn() -> MotifNetParams + Sync + Send>,
    Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send>,
) {
    let id = motif_t.id().clone();
    let motif_width = motif_t.as_ref().t_ref().size()[1] as u32;
    (
        Box::new(move || MotifNetParams::new(motif_width, 1, 1, step, 32, kernel)),
        Box::new(move || {
            *dummy_motif_net::<Batch<Slice>, MotifProbability, SynKey>(id.clone(), motif_width, 1., device, "test", &motif_t.as_ref().t_ref().i((0, ..)), step as i64).unwrap()//first slice
        }),
    )
}

#[cfg(test)]
///builds net initialized to detect provided pattern
pub fn dummy_motif_net<DI: Dims + Send + 'static, DO: Dims + Send + 'static, I:Into<SynKey>>(
    id: I,
    num_features: u32,
    k: f64,
    device: Device,
    descr: &str,
    pattern:&Tensor,
    stride: i64
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
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {

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
                        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
                            Ok(t.f_pow_tensor_scalar(2)?)
                        })),
                        None,
                    )?
                    .conv_1d("energy", l as u32, 1, 1, cfg, Some(Arc::clone(&ones)))?
                    .closure(
                        Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {
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
            Arc::new(Box::new(move |t: &Tensor, _lengths: Option<&BatchLengths>| {

                // debug!("dummy_motif_net: out=\n{t}");

                Ok(t.shallow_clone())
            })),
            None,
        )?
        .end_seq()?;



    let module = node.clone().build_tch_module(&vs.root())?;
    Ok(Box::new(ThoModuleWithStore::with_factory(
        id,
        module,
        vs,
        Some(Arc::new( move |path:&nn::Path| {
            node.clone().build_tch_module(path)
        }))
    )))

}


#[cfg(test)]
pub(crate) mod tests {
    use crate::tasks::apply_motif_nets::{dummy_motif_net, motif_test_net_, valid_partial_output_len, ApplyMotifNets};
    use crate::tasks::find_motifs_run_config::FindMotifsRunConfig;
    use crate::tasks::tensor_loader::mock_tensor_loader;
    use log::LevelFilter::{Debug, Trace};
    use log::{debug, trace};
    use parking_lot::{RawRwLock, RwLock};
    use plotters::prelude::{BLACK, RED};
    use plotters::style::{BLUE, GREEN};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tch::kind::FLOAT_CPU;
    use tch::nn::{LinearConfig, VarStore};
    use tch::Device::Cpu;
    use tch::{nn, Device, IndexOp, Kind, Tensor};
    use thoosa_common::builder::SeqBuilder;
    use thoosa_common::debug_plot;
    use thoosa_common::dims::{AnyDims, Batch, ChannelsData, Dims, MotifProbability, Slice};
    use thoosa_common::my_tch::ThoModule;
    use thoosa_common::runner::cache_areas::CacheArea;
    use thoosa_common::runner::common::cachable::DevicesWrap;
    use thoosa_common::runner::common::{mock_run_ctx, MockRunCtx};
    use thoosa_common::runner::common::runnable::AsyncStage;
    use thoosa_common::runner::common::{MaxParallelCtx, RunCtx};
    use thoosa_common::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_common::runner::global_ctx::{mock_global_ctx, GlobalCtx};
    use thoosa_common::tch_test_utils;
    use thoosa_common::tch_test_utils::Interval;
    use thoosa_common::tho_tensor::ThoTensor;
    use thoosa_common::train_nets::{CommonTrainingRunConfig, MotifCosine2DNetArgs, MotifNetParams, MotifNetTypeArgs, StftRunConfig, ThoModuleWithStore, TrainMotifNetRunConfig};
    use thoosa_common::utils::{assert_close, lin_t, plot_series_with_lanes, reset_print_options, test_net, to_vec_f32};
    use thoosa_common::ws::{Workspace, WsMockBuilder};
    use thoosa_properties_map::ToHashmap;
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::data_with_extra::DataWithExtra;
    use thoosa_shared::mono_key;
    use thoosa_shared::perf_logging::PERF;
    use thoosa_shared::resource::Resource;
    use thoosa_shared::run_configs::RunConfigId;
    use thoosa_shared::test_utils::{gen_motif, MotifKind};
    use thoosa_shared::traits::{Adjustable, Named, SimpleId};
    use thoosa_shared::{TensorIdSliceIdx, ThoError};
    use thoosa_store::{MockStoreBuilder, RunConfigRow, Store, TrainedNet};
    use thoosa_with_data::with_data;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[test]
    fn test_valid_partial_output_len_trims_last_chunk_and_short_input() {
        assert_eq!(valid_partial_output_len(4000, 0, 999, 100, 3), 333);
        assert_eq!(valid_partial_output_len(4000, 2997, 999, 100, 3), 302);
        assert_eq!(valid_partial_output_len(4000, 3996, 999, 100, 3), 0);
        assert_eq!(valid_partial_output_len(50, 0, 25, 100, 3), 1);
        assert_eq!(valid_partial_output_len(50, 25, 25, 100, 3), 0);
    }


    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_without_oom() -> Result<(), ThoError> {
        apply_motif_nets_3_channels_2_motifs(None, 4000, 10, new_apply_motif_nets, |device, motif, |motif_test_net_(device, motif, 10, 100), Default::default(), 100, 100).await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_without_oom_uneven() -> Result<(), ThoError> {
        apply_motif_nets_3_channels_2_motifs(None, 4321, 10, new_apply_motif_nets, |device, motif, |motif_test_net_(device, motif, 10, 100), Default::default(), 100, 100).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_step_1_with_oom() -> Result<(), ThoError> {
        apply_motif_nets_3_channels_2_motifs(Some(1000), 4000, 10, new_apply_motif_nets, |device, motif, |motif_test_net_(device, motif, 10, 100), Default::default(), 100, 100).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_step_2_with_oom() -> Result<(), ThoError> {
        apply_motif_nets_3_channels_2_motifs(Some(1000), 4000, 10, new_apply_motif_nets, |device, motif, |motif_test_net_(device, motif, 10, 100), Default::default(), 100, 100).await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_3_channels_2_motifs_with_oom_uneven() -> Result<(), ThoError> {
        apply_motif_nets_3_channels_2_motifs(Some(987), 4321, 10, new_apply_motif_nets, |device, motif, |motif_test_net_(device, motif, 10, 100), Default::default(), 100, 100).await
    }

    pub async fn apply_motif_nets_3_channels_2_motifs<F,A, MF>(trigger_oom_len: Option<usize>, input_n_0: usize, apply_motif_net_step: u32, apply_motif_net_fn: F, motif_net_fn_: MF, stft_config: StftRunConfig/*shouldn't be here, needed only for cosine 2d*/, motif_len: usize, sample_rate: u32) -> Result<(), ThoError>
    where
        for <'a> A:AsyncStage<In<'a>=ThoTensor<ChannelsData, SynKey>, Out=DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>, Ctx=MockRunCtx>,
        F:AsyncFn(Option<usize>, u32, RunConfigId, &Arc<RwLock<Box<dyn Workspace>>>, &mut Arc<Box<dyn Store>>) -> Result<Arc<A>, ThoError>,
        MF:Fn(Device, Arc<ThoTensor<Batch<Slice>, SynKey>>) -> (Box<dyn Fn() -> MotifNetParams + Sync + Send>, Box<dyn Fn() -> ThoModuleWithStore<Batch<Slice>, MotifProbability> + Sync + Send>)
    {
        thoosa_shared::test_utils::maybe_init_perf_tracing_stdout("debug");
        log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let find_motifs_run_config_id: RunConfigId = 1;
        let select_motifs_run_config_id: RunConfigId = 2;

        let train_motifs_run_config_id: RunConfigId = 3;
        let motif_0_0_id = 0; //channel 0 motif 0
        let motif_0_1_id = 1; //channel 0 motif 1
        let motif_1_0_id = 2; //channel 1 motif 0

        let motif_0_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_0", "motif_id")]), tch_test_utils::motifs_tensor_fixed(motif_len, 3, MotifKind::Pulse).f_mul_scalar(0.2)?)?);
        let motif_0_1 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_1", "motif_id")]), tch_test_utils::motifs_tensor_fixed(motif_len, 3, MotifKind::Triangle).f_mul_scalar(0.2)?)?);
        let motif_1_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_1_0", "motif_id")]), tch_test_utils::motifs_tensor_fixed(motif_len, 3, MotifKind::Sin).f_mul_scalar(0.2)?)?);

        let net_id_0_0 = PathBuf::from(format!("0/{}_{}", motif_0_0_id, train_motifs_run_config_id));
        let net_id_0_1 = PathBuf::from(format!("0/{}_{}", motif_0_1_id, train_motifs_run_config_id));
        let net_id_1_0 = PathBuf::from(format!("1/{}_{}", motif_1_0_id, train_motifs_run_config_id));
        let (params_0_0, module_0_0) = motif_net_fn_(Cpu, Arc::clone(&motif_0_0));
        let (params_0_1, module_0_1) = motif_net_fn_(Cpu, Arc::clone(&motif_0_1));
        let (params_1_0, module_1_0) = motif_net_fn_(Cpu, Arc::clone(&motif_1_0));

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 3, sample_rate, input_n_0 as u32)
                .with_load_motif_net(vec![
                    (net_id_0_0.clone(), params_0_0, module_0_0),
                    (net_id_0_1.clone(), params_0_1, module_0_1),
                    (net_id_1_0.clone(), params_1_0, module_1_0),
                ])
                .build(),
        ));

        let mut store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([(
                    find_motifs_run_config_id,
                    Ok(RunConfigRow {
                        id: find_motifs_run_config_id,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find motifs".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_list_run_config_values([
                    (find_motifs_run_config_id, Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), 0, motif_len as u32, None, None).to_properties_iter(None).collect())),
                    (train_motifs_run_config_id, Ok(TrainMotifNetRunConfig::new(Some(train_motifs_run_config_id), CommonTrainingRunConfig::new(10, 2, 0.1), MotifNetTypeArgs::Cosine2DNet(MotifCosine2DNetArgs{
                        a: 10.0,
                        b: 0.0,
                        init_mode: Default::default(),
                        stft_config,
                    })).to_properties_iter(None).collect())),
                ])
                .with_get_input_run_config_ids([
                    (train_motifs_run_config_id, Ok(vec![select_motifs_run_config_id])),
                    (select_motifs_run_config_id, Ok(vec![find_motifs_run_config_id]))
                ])
                .with_get_motif_meta([
                    (motif_0_0_id, Ok((0usize, find_motifs_run_config_id))),
                    (motif_0_1_id, Ok((0usize, find_motifs_run_config_id))),
                    (motif_1_0_id, Ok((1usize, find_motifs_run_config_id))),
                ])
                .with_list_selected_motifs([
                    (select_motifs_run_config_id, Ok(
                        vec![
                            vec![(motif_0_0_id, 1.0f64), (motif_0_1_id, 1.0f64)], //channel 0
                            vec![(motif_1_0_id, 1.0f64)], //channel 1
                            vec![]//channel 2
                        ]
                    ))
                ])
                .with_list_trained_nets([
                    (vec![select_motifs_run_config_id], Some(0.3), None, Ok(vec![
                        TrainedNet { id: 0, run_config_id: train_motifs_run_config_id, net_path: net_id_0_0.clone(), loss: 0.1f32 },
                        TrainedNet { id: 1, run_config_id: train_motifs_run_config_id, net_path: net_id_0_1.clone(), loss: 0.2f32 },
                        TrainedNet { id: 2, run_config_id: train_motifs_run_config_id, net_path: net_id_1_0.clone(), loss: 0.25f32 },
                    ]))
                ])
                .build()
        );

        let apply_motif_nets = apply_motif_net_fn(trigger_oom_len, apply_motif_net_step, select_motifs_run_config_id, &ws, &mut store).await?;

        let channels_patterns = vec![
            vec![motif_0_0.t_ref().i((0, ..)), motif_0_1.t_ref().i((0, ..))],
            vec![motif_1_0.t_ref().i((0, ..))],
            Vec::new(),
        ];
        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor_from_patterns(input_n_0, channels_patterns.as_slice(), 300, apply_motif_net_step as usize)?;


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_t={resource_a_t:?}");// resource_b_t={resource_b_t:?} resource_c_t={resource_c_t:?}


        let mut run_ctx_branch = mock_run_ctx::default(1);
        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test_batch", "test_batch")]), resource_a_t)?)),
            run_ctx_branch
        ).await?;

        debug!("test_apply_motif_nets_3_channels_1_motif: PERF={:?}", PERF);


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_intervals={resource_a_intervals:?}");

        let pulse_interval_starts_in_resources_a = resource_a_intervals[0][0][0].start;

        let triangle_interval_starts_in_resources_a = resource_a_intervals[0][1][0].start;

        let sin_interval_starts_in_resources_a = resource_a_intervals[1][0][0].start;

        let motif_width = motif_0_0.t_ref().size()[1] as i64;
        let slices = ((input_n_0 as i64 - motif_width) / apply_motif_net_step as i64) + 1;
        let ctx_for_data = mock_run_ctx::default(1);
        with_data!(ctx_for_data, out; |_ctx, r:&DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>|{
            debug!("test_apply_motif_nets_3_channels_1_motif:  matches={:?}", r);
            assert_eq!(r.extra().as_slice(), &[slices, slices, slices]);
            assert_eq!(r.data().t_ref().size(), vec![3, slices]);

            let (pulse_matches_vals, pulse_matches_idx) = &r.data().t_ref().i((0, 0..slices)).topk(1i64, 0, true, true);
            let (triangle_matches_vals, triangle_matches_idx) = &r.data().t_ref().i((1, 0..slices)).topk(1i64, 0, true, true);
            let (sin_matches_vals, sin_matches_idx) = &r.data().t_ref().i((2, 0..slices)).topk(1i64, 0, true, true);

            assert_close(pulse_interval_starts_in_resources_a as f64, apply_motif_net_step as f64 * pulse_matches_idx.double_value(&[0]), apply_motif_net_step as f64 * 3.);
            assert_close(triangle_interval_starts_in_resources_a as f64, apply_motif_net_step as f64 * triangle_matches_idx.double_value(&[0]), apply_motif_net_step as f64 * 3.);
            assert_close(sin_interval_starts_in_resources_a as f64, apply_motif_net_step as f64 * sin_matches_idx.double_value(&[0]), apply_motif_net_step as f64 * 3.);

            Ok::<_, ThoError>(())
        })?;

        Ok(())
    }

    async fn new_apply_motif_nets(trigger_oom_len: Option<usize>, slice_step: u32, select_motifs_run_config_id: RunConfigId, ws: &Arc<RwLock<Box<dyn Workspace>>>, store: &mut Arc<Box<dyn Store>>) -> Result<Arc<ApplyMotifNets<MockRunCtx>>, ThoError> {
        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            &ws,
            &store,
            vec![select_motifs_run_config_id],
            mock_run_ctx::default(1),
            Some(0.3),
            None,
            slice_step,
            trigger_oom_len,
            false,
        ).await?);
        Ok(apply_motif_nets)
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_apply_motif_nets_mixed_widths_return_real_lengths() -> Result<(), ThoError> {
        thoosa_shared::test_utils::maybe_init_perf_tracing_stdout("debug");
        // log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let find_motifs_run_config_id: RunConfigId = 1;
        let select_motifs_run_config_id: RunConfigId = 2;
        let train_motifs_run_config_id: RunConfigId = 3;
        let step = 3;
        let input_len = 400usize;

        let motif_0_0_id = 0;
        let motif_0_1_id = 1;
        let motif_0_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("mixed_motif_0_0", "motif_id")]), tch_test_utils::motifs_tensor_fixed(100, 3, MotifKind::Pulse).f_mul_scalar(0.2)?)?);
        let motif_0_1 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("mixed_motif_0_1", "motif_id")]), tch_test_utils::motifs_tensor_fixed(60, 3, MotifKind::Triangle).f_mul_scalar(0.2)?)?);

        let net_id_0_0 = PathBuf::from(format!("0/{}_{}", motif_0_0_id, train_motifs_run_config_id));
        let net_id_0_1 = PathBuf::from(format!("0/{}_{}", motif_0_1_id, train_motifs_run_config_id));
        let (params_0_0, module_0_0) = motif_test_net_(Cpu, Arc::clone(&motif_0_0), step, 50);
        let (params_0_1, module_0_1) = motif_test_net_(Cpu, Arc::clone(&motif_0_1), step, 50);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 1, 100, input_len as u32)
                .with_load_motif_net(vec![
                    (net_id_0_0.clone(), params_0_0, module_0_0),
                    (net_id_0_1.clone(), params_0_1, module_0_1),
                ])
                .build(),
        ));

        let store = Arc::new(
            MockStoreBuilder::new()
                .with_run_config_descr([(
                    find_motifs_run_config_id,
                    Ok(RunConfigRow {
                        id: find_motifs_run_config_id,
                        r#type: "FindMotifsRunConfig".to_string(),
                        descr: "test find motifs".to_string(),
                        run_on: "2026-01-01 00:00:00".to_string(),
                    }),
                )])
                .with_list_run_config_values([(
                    find_motifs_run_config_id,
                    Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), step, 100, None, None).to_properties_iter(None).collect()),
                )])
                .with_get_input_run_config_ids([
                    (train_motifs_run_config_id, Ok(vec![select_motifs_run_config_id])),
                    (select_motifs_run_config_id, Ok(vec![find_motifs_run_config_id])),
                ])
                .with_get_motif_meta([
                    (motif_0_0_id, Ok((0usize, find_motifs_run_config_id))),
                    (motif_0_1_id, Ok((0usize, find_motifs_run_config_id))),
                ])
                .with_list_selected_motifs([
                    (select_motifs_run_config_id, Ok(vec![vec![(motif_0_0_id, 1.0f64), (motif_0_1_id, 1.0f64)]])),
                ])
                .with_list_trained_nets([
                    (vec![select_motifs_run_config_id], Some(0.3), None, Ok(vec![
                        TrainedNet { id: 0, run_config_id: select_motifs_run_config_id, net_path: net_id_0_0.clone(), loss: 0.1f32 },
                        TrainedNet { id: 0, run_config_id: select_motifs_run_config_id, net_path: net_id_0_1.clone(), loss: 0.2f32 },
                    ])),
                ])
                .build(),
        );

        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            &ws,
            &store,
            vec![select_motifs_run_config_id],
            mock_run_ctx::default(1),
            Some(0.3),
            None,
            step,
            None,
            false,
        ).await?);

        let input = ThoTensor::new(
            SynKey::new([mono_key!("mixed_motif_input", "test_batch")]),
            Tensor::f_zeros([1, input_len as i64], FLOAT_CPU)?,
        )?;
        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(input)),
            mock_run_ctx::default(1),
        ).await?;

        let expected_lengths = vec![
            ((input_len as i64 - motif_0_0.t_ref().size()[1]) / step as i64) + 1,
            ((input_len as i64 - motif_0_1.t_ref().size()[1]) / step as i64) + 1,
        ];
        let max_len = *expected_lengths.iter().max().expect("no max len");

        with_data!(mock_run_ctx::default(1), out; |_ctx, out:&DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>| {
            assert_eq!(out.extra().as_slice(), expected_lengths.as_slice());
            assert_eq!(out.data().t_ref().size(), vec![2, max_len]);
            Ok::<_, ThoError>(())
        })?;

        Ok(())
    }


    async fn test_apply_motif_single_net_1_channel_1_motif(trigger_oom_len: Option<usize>, input_n_0: usize, step: u32) -> Result<(), ThoError> {
        thoosa_shared::test_utils::maybe_init_perf_tracing_stdout("debug");
        log::set_max_level(Debug);
        reset_print_options();
        GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let find_motifs_run_config_id: RunConfigId = 1;
        let select_motifs_run_config_id: RunConfigId = 2;

        let train_motifs_run_config_id: RunConfigId = 3;
        let motif_0_0_id = 0; //channel 0 motif 0

        let motif_0_0 = Arc::new(ThoTensor::new(SynKey::new([mono_key!("motif_0_0", "motif_id")]), tch_test_utils::motifs_tensor_fixed(100, 3, MotifKind::Pulse).f_mul_scalar(0.2)?)?);

        let net_id_0_0 = PathBuf::from(format!("0/{}_{}", motif_0_0_id, train_motifs_run_config_id));
        let (params_0_0, module_0_0) = motif_test_net_(Cpu, Arc::clone(&motif_0_0), step, 100);

        let ws = Arc::new(RwLock::new(
            WsMockBuilder::new(None, 1, 100, input_n_0 as u32)
                .with_load_motif_net(vec![
                    (net_id_0_0.clone(), params_0_0, module_0_0),
                ])
                .build(),
        ));

        let mut store = Arc::new(
            MockStoreBuilder::new()
            .with_run_config_descr([(
                find_motifs_run_config_id,
                Ok(RunConfigRow {
                    id: find_motifs_run_config_id,
                    r#type: "FindMotifsRunConfig".to_string(),
                    descr: "test find motifs".to_string(),
                    run_on: "2026-01-01 00:00:00".to_string(),
                }),
            )])
            .with_list_run_config_values([(
                find_motifs_run_config_id,
                Ok(FindMotifsRunConfig::new_with_running_mean(Some(find_motifs_run_config_id), step, 100, None, None).to_properties_iter(None).collect()),
            )])

            .with_get_input_run_config_ids([
                (train_motifs_run_config_id, Ok(vec![select_motifs_run_config_id])),
                (select_motifs_run_config_id, Ok(vec![find_motifs_run_config_id]))
            ])
            .with_get_motif_meta( [(motif_0_0_id, Ok((0usize, find_motifs_run_config_id)))])
            .with_list_selected_motifs([
                (select_motifs_run_config_id, Ok(
                    vec![
                        vec![(motif_0_0_id, 1.0f64)], //channel 0
                    ]
                ))
            ])
            .with_list_trained_nets([
                (vec![select_motifs_run_config_id], Some(0.3), None, Ok(vec![
                    TrainedNet { id: 0, run_config_id: select_motifs_run_config_id, net_path: net_id_0_0.clone(), loss: 0.1f32 },
                ]))
            ])
            .build()
        );


        let apply_motif_nets = Arc::new(ApplyMotifNets::new(
            &ws,
            &store,
            vec![select_motifs_run_config_id],
            mock_run_ctx::default(1), Some(0.3) //Some(1000)
            ,
            None,
            step,
            trigger_oom_len,
            false
        ).await?);

        debug_plot!(motif_0_0.t_ref(), "motif_test")?;

        let channels_patterns = vec![vec![motif_0_0.t_ref().i((0, ..))]];
        let (resource_a_t, resource_a_intervals) = tch_test_utils::resource_tensor_from_patterns(input_n_0, channels_patterns.as_slice(), 200, step as usize)?;


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_t={resource_a_t:?}");// resource_b_t={resource_b_t:?} resource_c_t={resource_c_t:?}

        let run_ctx = Arc::new(RwLock::new(mock_run_ctx::default(1)));

        let run_ctx_branch = thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(run_ctx)).branch(false);

        let out = apply_motif_nets.run(
            Arc::new(DevicesWrap::wrap(ThoTensor::new(SynKey::new([mono_key!("test_batch", "test_batch")]), resource_a_t)?)),
            run_ctx_branch
        ).await?;

        debug!("test_apply_motif_nets_3_channels_1_motif: PERF={:?}", PERF);


        debug!("test_apply_motif_nets_3_channels_1_motif: resource_a_intervals={resource_a_intervals:?}");

        let pulse_interval_starts_in_resources_a = resource_a_intervals[0][0][0].start;


        let motif_width = motif_0_0.t_ref().size()[1] as i64;
        let slices = ((input_n_0 as i64 - motif_width) / step as i64) + 1;

        let ctx_for_data = thoosa_shared::perf_log!(thoosa_shared::sync_write_lock!(run_ctx)).branch(false);

        with_data!(ctx_for_data, out; |_ctx, r:&DataWithExtra<ThoTensor<MotifProbability, SynKey>, Vec<i64>>|{
            debug!("test_apply_motif_nets_3_channels_1_motif:  matches={:?}", r);
            assert_eq!(r.extra().as_slice(), &[slices]);
            assert_eq!(r.data().t_ref().size(), vec![1, slices]);

            let (pulse_matches_vals, pulse_matches_idx) = &r.data().t_ref().i((0, 0..slices)).topk(1i64, 0, true, true);

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


        let test_net = dummy_motif_net::<AnyDims, AnyDims, &str>("test", 100, 1., Cpu, "test", &motif_t.i((0, ..)), 1)?;

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
