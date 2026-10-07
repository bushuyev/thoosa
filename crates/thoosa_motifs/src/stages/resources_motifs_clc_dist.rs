use crate::distance::MaskMaker::{Le, LocalMin};
use async_trait::async_trait;
use log::{debug, trace};
use parking_lot::RwLock;
use std::any::Any;
use std::cmp::min;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::marker::PhantomData;
use std::ops::{DivAssign, Range};
use std::sync::Arc;
use tch::Device::Cpu;
use tch::{IndexOp, Tensor};
use thoosa_common::dims::ChannelsData;
use crate::dims::ChannelsMotifsDists;
use thoosa_common::runner::adjusters::max_parallel::MaxParallel;
use thoosa_common::runner::cache_areas::CacheArea;
use thoosa_common::runner::common::DropIf;
use thoosa_common::runner::common::cachable::{BorrowRef, DevicesWrap, SelectRef, ThoDataRef};
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_cpu_data_derive::CpuData;
use crate::stages::cross_batch_clc::{UnfoldParams, unfold_and_cat};
use crate::stages::dists_tracker::{DistsMask, DistsMaskType};
use crate::stages::slices_iter::{ResourceSlices, SlicesIter};
use crate::stages::stacked_moitfs_loader::StackedMotifsOutput;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::resource::ResourceId;
use thoosa_shared::run_configs::RunConfigId;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, Data, DataId, Metrics, MonoMetrics, Named, OutId, RunCtx, Select, Stage, SyncStage, ThoSelector, WithId};
use thoosa_shared::{ThoError, mono_key};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
pub struct ClcResourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    num_channels: usize,
    #[cfg(test)]
    fail_if_size_bigger_than:Option<usize>,
    _ctx: PhantomData<C>,
}

impl <C> Debug for ClcResourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClcResourcesMotifsDist").finish()
    }
}

impl <C> Adjustable for ClcResourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{}

impl <C>  OutId for  ClcResourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        let stage_key = SynKey::new([mono_key!("ClcResourcesMotifsDist", "stage")]);
        &stage_key + input_id
    }
}

impl<C> ClcResourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(
        num_channels: usize,
        #[cfg(test)]
        fail_if_size_bigger_than:Option<usize>
    ) -> Self {
        Self {
            num_channels,
            _ctx: PhantomData,
            #[cfg(test)]
            fail_if_size_bigger_than
        }
    }
}

impl<C> Stage for ClcResourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ClcResourcesMotifsDistInput<'a>;
    type Out = ClcResourcesMotifsDistOutput;
}

impl<C> SyncStage for ClcResourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="ClcResourcesMotifsDist::do_run"))]
    fn do_run<'f>(
        self: Arc<Self>,
        input: &Self::In<'f>,
        ctx: &Self::Ctx,
    ) -> Result<Self::Out, ThoError> {
        debug!("ClcResourcesMotifsDist::do_run: input={input:?}");

        let device_idx = ctx.tho_device_idx();

        let unfold_params = UnfoldParams::new(
            1,
            input.stacked_motifs_output.as_ref().motifs.extra().1.window as i64,
            input.stacked_motifs_output.as_ref().motifs.extra().1.step as i64,
        );

        let batch = input.batch.iter().map(ThoDataRef::as_ref).collect::<Vec<_>>();
        let (batch_1_t, batch_1_sizes) = unfold_and_cat(batch.as_slice(), &unfold_params)?;

        debug!("ClcResourcesMotifsDist::do_run: self.num_channels={} batch_1_t.size={:?} batch_1_t.device={:?} motifs.size={:?} motifs_stats.size={:?} input.resource_ids={:?}",self.num_channels, batch_1_t.size(), batch_1_t.device(), input.stacked_motifs_output.as_ref().motifs.data().t_ref().size(), input.motifs_stats.t_ref().size(), input.resource_ids);

        #[cfg(test)]
        if self.fail_if_size_bigger_than.map(|v| v < batch_1_t.size()[1] as usize).unwrap_or(false) {
            debug!("ClcResourcesMotifsDist::do_run: triggering OOM: v={:?} batch_1_t.size()[1]={}", self.fail_if_size_bigger_than, batch_1_t.size()[1]);
            return Err(ThoError::oom());
        }

        let resource_ids_sizes = input
            .resource_ids
            .iter()
            .zip(batch_1_sizes.into_iter())
            .zip(input.resource_offsets.iter().copied())
            .map(|((resource_id, size), offset)| ResourceSlices::new(*resource_id, size as usize))
            .collect::<Vec<_>>();

        let resources_iter = SlicesIter::new_with_offsets(unfold_params.step() as usize, resource_ids_sizes);

        let tensor = input
            .motifs_stats
            .t_ref()
            .f_unsqueeze(-1)?
            .f_expand(
                [-1, -1, batch_1_t.data().size()[1]],
                false,
            )?;
        let mut mask_makers = vec![
            // dim 1 are motifs - local minimums would be differences between motifs, we're interested in miniums within motifs
            LocalMin { dim: 2, radius: unfold_params.size() as u64  /2 },
            Le {
                v: &tensor,
            },
        ];

        let channel_masks_dists_ids = {


            let find_motifs_run_config = input.stacked_motifs_output.as_ref().motifs.extra().1.as_ref();
            let cross_loader_run_config = find_motifs_run_config.cross_loader_run_config().ok_or_else(|| ThoError::new(format!("ClcResourcesMotifsDist::run: find_motifs_run_config_id={:?} is not a CrossLoader config", find_motifs_run_config.id)))?;

            let (mask, dists)  = crate::distance::clc_d2_similarity_and_mask(
                &input
                        .stacked_motifs_output.as_ref()
                        .motifs
                        .data()
                        .t_ref(),
                &batch_1_t,
                cross_loader_run_config.similarity_metric,
                true,
                &mut mask_makers,
                )?;

            (0..self.num_channels)
                .map(|c| {
                    let ids = &input.stacked_motifs_output.as_ref().motifs.extra().0[c];
                    if ids.is_empty() {
                        Ok(None) //no motifs in channel c
                    } else {
                        let mask = fold_mask(mask.f_i((c as i64, .., ..))?, ids, 0)?;
                        let dists = match dists.as_ref() {
                            Some(d) => Some(fold_dists(d.f_i((c as i64, .., ..))?, ids, 0)?),
                            None => None,
                        };
                        let mask_type = if mask.device().eq(&Cpu) {
                            DistsMaskType::AsIs
                        } else {
                            DistsMaskType::MaskPacked(mask.size(), dists.as_ref().map(|t| t.size()))
                        };

                        Ok(Some((
                            //unsqueeze is to put back dimension for channel removed before folding
                            DistsMask::new(mask.f_unsqueeze(0)?, dists.map(|t| t.f_unsqueeze(0)).transpose()?, mask_type)?,
                            ids.iter()
                                .fold((vec![], HashSet::new()), |mut acc, d| {
                                    if !acc.1.contains(d) {
                                        acc.1.insert(d);
                                        acc.0.push(d.clone());
                                    }
                                    acc
                                })
                                .0,
                        )))
                    }
                })
                .collect::<Result<Vec<_>, ThoError>>()?
        };

        let select_motifs_run_config_id = input
            .stacked_motifs_output.as_ref()
            .motifs
            .extra()
            .1
            .id
            .ok_or_else(|| ThoError::new("no select_motifs_run_config_id"))?;

        Ok(ClcResourcesMotifsDistOutput::new(
            select_motifs_run_config_id,
            channel_masks_dists_ids,
            resources_iter,
        ))
    }
}


#[derive(Debug)]
pub struct ClcResourcesMotifsDistInputDeviceWrapper {
    id: SynKey,
    batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
    resource_ids: Vec<ResourceId>,
    stacked_motifs: Arc<DevicesWrap<StackedMotifsOutput>>,
    motifs_stats: Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>,
}

impl Adjustable for ClcResourcesMotifsDistInputDeviceWrapper {

}



//implemented so that ClcResourcesMotifsDistInputDeviceWrapper's resources batch can be adjusted.
//stacked_motifs_output can't be adjusted since moitfs' ranges do not match in different channels
impl Data for ClcResourcesMotifsDistInputDeviceWrapper {
    type Metrics = i64;


    fn size(&self) -> Self::Metrics {
        self.batch.len() as i64
    }

    fn tho_device_idx(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        self.batch.iter().map(|batch| batch.mem_size()).sum::<u64>()
            + self.stacked_motifs.mem_size()
            + self.motifs_stats.mem_size()
    }
}

impl ClcResourcesMotifsDistInputDeviceWrapper {
    pub fn new(
        id: SynKey,
        batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData, SynKey>>>>,
        resource_ids: Vec<ResourceId>,
        stacked_motifs: Arc<DevicesWrap<StackedMotifsOutput>>,
        motifs_stats: Arc<DevicesWrap<ThoTensor<ChannelsMotifsDists, SynKey>>>,
    ) -> Self {
        Self { id, batch, resource_ids, stacked_motifs, motifs_stats }
    }
}

impl BorrowRef for ClcResourcesMotifsDistInputDeviceWrapper {
    type Out<'a> = ClcResourcesMotifsDistInput<'a> where Self: 'a;

    fn borrow_ref<'a>(&'a self, to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let mut batch = Vec::with_capacity(self.batch.len());
            for tensor in &self.batch {
                batch.push(ThoDataRef::ReadLocked(track_await(tensor.load_from_any_device_(to_device_idx)).await?));
            }
            let stacked_motifs = track_await(self.stacked_motifs.load_from_any_device_(to_device_idx)).await?;
            let motifs_stats = track_await(self.motifs_stats.load_from_any_device_(to_device_idx)).await?;
            Ok(ClcResourcesMotifsDistInput::from_data_refs(batch, self.resource_ids.clone(), vec![0; self.batch.len()], ThoDataRef::ReadLocked(stacked_motifs), ThoDataRef::ReadLocked(motifs_stats)))
        }
    }

}

impl SelectRef<ClcResourcesMotifsDistInputDeviceWrapperSelector> for ClcResourcesMotifsDistInputDeviceWrapper {
    fn select_ref<'a>(&'a self, to_device_idx: usize, selector: Option<ClcResourcesMotifsDistInputDeviceWrapperSelector>) -> impl Future<Output = Result<<Self as BorrowRef>::Out<'a>, ThoError>> + Send
    where
        ClcResourcesMotifsDistInputDeviceWrapperSelector: 'a,
        Self: 'a,
    {
        async move {
            let Some(selector) = selector else {
                return BorrowRef::borrow_ref(self, to_device_idx).await;
            };
            let range = selector.resources_batch_range.start as usize..selector.resources_batch_range.end as usize;
            let mut batch = Vec::with_capacity(range.len());
            for tensor in &self.batch[range.clone()] {
                batch.push(ThoDataRef::ReadLocked(track_await(tensor.load_from_any_device_(to_device_idx)).await?));
            }
            let stacked_motifs = track_await(self.stacked_motifs.load_from_any_device_(to_device_idx)).await?;
            let motifs_stats = track_await(self.motifs_stats.load_from_any_device_(to_device_idx)).await?;
            Ok(ClcResourcesMotifsDistInput::from_data_refs(batch, self.resource_ids[range.clone()].to_vec(), vec![0; range.len()], ThoDataRef::ReadLocked(stacked_motifs), ThoDataRef::ReadLocked(motifs_stats)))
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ClcResourcesMotifsDistInputDeviceWrapperSelector {
    resources_batch_range:Range<i64>
}

#[typetag::serde]
impl ThoSelector for ClcResourcesMotifsDistInputDeviceWrapperSelector {
    fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(self.clone())
    }
}

impl ClcResourcesMotifsDistInputDeviceWrapperSelector {
    pub fn new() -> Self {
        Self { resources_batch_range: Default::default() }
    }
}

impl From<Range<i64>> for ClcResourcesMotifsDistInputDeviceWrapperSelector {
    fn from(value: Range<i64>) -> Self {
        Self { resources_batch_range: value }
    }
}


impl Adjustable for ClcResourcesMotifsDistInputDeviceWrapperSelector {}




#[derive(Debug)]
pub struct ClcResourcesMotifsDistInput<'a> {//TODO remove unused
    id: SynKey,
    batch: Vec<ThoDataRef<'a, ThoTensor<ChannelsData, SynKey>>>,
    resource_ids: Vec<ResourceId>,
    resource_offsets: Vec<u32>,
    stacked_motifs_output: ThoDataRef<'a, StackedMotifsOutput>,
    motifs_stats: ThoDataRef<'a, ThoTensor<ChannelsMotifsDists, SynKey>>,
}


impl<'a> ClcResourcesMotifsDistInput<'a> {
    pub fn new(
        batch: &'a [&'a ThoTensor<ChannelsData, SynKey>],
        resource_ids: Vec<ResourceId>,
        batched_motifs_output: &'a StackedMotifsOutput,
        motifs_stats: &'a ThoTensor<ChannelsMotifsDists, SynKey>,
    ) -> ClcResourcesMotifsDistInput<'a> {
        let resource_offsets = vec![0; resource_ids.len()];
        Self::new_with_offsets(batch, resource_ids, resource_offsets, batched_motifs_output, motifs_stats)
    }

    pub fn new_with_offsets(
        batch: &'a [&'a ThoTensor<ChannelsData, SynKey>],
        resource_ids: Vec<ResourceId>,
        resource_offsets: Vec<u32>,
        batched_motifs_output: &'a StackedMotifsOutput,
        motifs_stats: &'a ThoTensor<ChannelsMotifsDists, SynKey>,
    ) -> ClcResourcesMotifsDistInput<'a> {
        assert_eq!(resource_ids.len(), resource_offsets.len());
        Self {
            id: SynKey::new([mono_key!("ClcResourcesMotifsDistInput", "stage")]),
            batch: batch.iter().map(|tensor| ThoDataRef::Borrowed(*tensor)).collect(),
            resource_ids,
            resource_offsets,
            stacked_motifs_output: ThoDataRef::Borrowed(batched_motifs_output),
            motifs_stats: ThoDataRef::Borrowed(motifs_stats),
        }
    }

    pub fn from_data_refs(batch: Vec<ThoDataRef<'a, ThoTensor<ChannelsData, SynKey>>>, resource_ids: Vec<ResourceId>, resource_offsets: Vec<u32>, stacked_motifs_output: ThoDataRef<'a, StackedMotifsOutput>, motifs_stats: ThoDataRef<'a, ThoTensor<ChannelsMotifsDists, SynKey>>) -> Self {
        assert_eq!(resource_ids.len(), resource_offsets.len());
        Self { id: SynKey::new([mono_key!("ClcResourcesMotifsDistInput", "stage")]), batch, resource_ids, resource_offsets, stacked_motifs_output, motifs_stats }
    }

}


impl <'a> Data for ClcResourcesMotifsDistInput<'a> {
    type Metrics = i64;


    fn size(&self) -> Self::Metrics {
        self.batch.len() as i64
    }

    fn tho_device_idx(&self) -> usize {
        self.batch[0].tho_device_idx()
    }

    fn mem_size(&self) -> u64 {

        let batch_mem_size = self.batch.iter().map(|t|t.mem_size()).sum::<u64>();
        let motifs_output_mem_size =  self.stacked_motifs_output.as_ref().motifs.mem_size();
        let motifs_stats_mem_size = self.motifs_stats.mem_size();

        debug!("ClcResourcesMotifsDistInput::mem_size: batch_mem_size={} motifs_output_mem_size={} motifs_stats_mem_size={}", batch_mem_size, motifs_output_mem_size, motifs_stats_mem_size);

        batch_mem_size + motifs_output_mem_size + motifs_stats_mem_size
    }
}


impl<'a> WithId for ClcResourcesMotifsDistInput<'a> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<'a> Input for ClcResourcesMotifsDistInput<'a> {}

#[derive(Debug)]
pub struct ClcResourcesMotifsDistOutput {
    pub id: SynKey,
    pub find_motifs_run_config_id: RunConfigId,
    pub channel_masks_dists_ids: Vec<Option<(DistsMask, Vec<SynKey>)>>,
    pub resources_iter: SlicesIter,
}

impl ClcResourcesMotifsDistOutput {
    pub fn new(
        find_motifs_run_config_id: RunConfigId,
        channel_masks_dists_ids: Vec<Option<(DistsMask, Vec<SynKey>)>>,
        resources_iter: SlicesIter,
    ) -> Self {
        debug!("ClcResourcesMotifsDistOutput::new: select_motifs_run_config_id={find_motifs_run_config_id} channel_masks_dists_ids={channel_masks_dists_ids:?} resources_iter={resources_iter:?}");

        Self {
            id: SynKey::new([mono_key!("ClcResourcesMotifsDistOutput", "stage")]),
            find_motifs_run_config_id,
            channel_masks_dists_ids,
            resources_iter,
        }
    }

    pub async fn to_cpu(self) -> Result<ClcResourcesMotifsDistOutput, ThoError> {
        let channel_masks_dists_ids = track_await(futures::future::try_join_all(
            self.channel_masks_dists_ids.into_iter().map(|op| async move {
                match op {
                    None => Ok::<_, ThoError>(None),
                    Some((dists_mask, data_ids)) => Ok(Some((track_await(dists_mask.to_cpu()).await?, data_ids))),
                }
            }),
        )).await?;

        Ok(ClcResourcesMotifsDistOutput {
            id: self.id,
            find_motifs_run_config_id: self.find_motifs_run_config_id,
            channel_masks_dists_ids,
            resources_iter: self.resources_iter,
        })
    }
}

impl Data for ClcResourcesMotifsDistOutput {
    type Metrics = ();


    fn size(&self) -> Self::Metrics {
        ()
    }

    fn tho_device_idx(&self) -> usize {
        0
    }

    fn mem_size(&self) -> u64 {
        self.channel_masks_dists_ids
            .iter()
            .map(|channel_output| channel_output.as_ref().map(|(dists_mask, _)| dists_mask.mem_size()).unwrap_or(0))
            .sum()
    }
}

impl WithId for ClcResourcesMotifsDistOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for ClcResourcesMotifsDistOutput {}

///returns true if any motif slices x resource slice contains true (close enough)
pub fn fold_mask<I>(mask: Tensor, ids: &Vec<I>, dim: usize) -> Result<Tensor, ThoError>
where
    I: PartialEq + Debug,
{
    reduce_by_groups(mask, |chunk| chunk.any_dim(dim as i64, true), ids, dim)
}

///returns mean dist between all motif slices x resource slice
pub fn fold_dists<I>(mask: Tensor, ids: &Vec<I>, dim: usize) -> Result<Tensor, ThoError>
where
    I: PartialEq + Debug,
{
    reduce_by_groups(
        mask,
        |chunk| chunk.mean_dim(&[dim as i64][..], true, chunk.kind()),
        ids,
        dim,
    )
}

pub fn reduce_by_groups<I, F>(
    full: Tensor,
    reduce: F,
    ids: &Vec<I>,
    dim: usize,
) -> Result<Tensor, ThoError>
where
    I: PartialEq + Debug,
    F: Fn(Tensor) -> Tensor,
{
    debug!("fold_mask: mask={full:?} dim={dim}");
    trace!("fold_mask: ids={ids:?} ");
    trace!("fold_mask: mask=\n{full}");

    let sizes = full.size();
    let nd = sizes.len();
    assert!(dim < nd, "dim out of range");

    let mut out: Vec<Tensor> = Vec::new();

    let mut push_group = |start: usize, len: usize| {
        let s = start as i64;
        let l = len as i64;
        debug!("fold_mask: s={s} l={l:?}");
        let chunk = full.f_narrow(dim as i64, s, l)?;
        let reduced = reduce(chunk);
        out.push(reduced);
        Ok::<(), ThoError>(())
    };

    let mut start = 0usize;
    let mut prev = &ids[0];
    for (i, id) in ids.iter().enumerate().skip(1) {
        if id != prev {
            push_group(start, i - start)?;
            start = i;
            prev = id;
        }
    }
    push_group(start, ids.len() - start)?;

    Ok(Tensor::f_cat(&out, dim as i64)?)
}
