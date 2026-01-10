use std::any::Any;
use std::cmp::min;
use std::collections::HashSet;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{DivAssign, Range};
use std::sync::Arc;
use async_trait::async_trait;
use futures_util::{StreamExt, TryStreamExt};
use log::{debug, trace};
use parking_lot::RwLock;
use tch::{IndexOp, Tensor};
use tch::Device::Cpu;

use thoosa_shared::runner::common::runnable::{Stage, SyncStage};
use thoosa_shared::runner::common::{DropIf, OutId, PermitsCtx, RunCtx};
use thoosa_shared::runner::common::cachable::{DevicesWrap, WithDataRef, WithSelectDataRef};
use thoosa_common::cuda_utils::with_cuda_stream;
use thoosa_dummy_data_derive::DummyData;
use thoosa_motifs::distance::MaskMaker::{Le, LocalMin};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::resource::ResourceId;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::copy_mode::CpuOnly;
use thoosa_shared::traits::{Adjustable, Data, DataId, Metrics, MonoMetrics, Named, Select, WithId};
use thoosa_shared::{mono_key, ThoError};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use thoosa_with_data::with_data;
use crate::tasks::stacked_moitfs_loader::StackedMotifsOutput;
use crate::tasks::cross_batch_clc::{unfold_and_cat, UnfoldParams};
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::dists_tracker::{DistsMask, DistsMaskType};
use crate::tasks::slices_iter::SlicesIter;
use thoosa_common::tho_tensor::ThoTensor;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::runner::permits::{NumPermits, RunnerPerm};

#[derive(Clone, Copy, Debug)]
struct DeviceCtx {
    device_idx: usize,
}

impl DeviceCtx {
    fn new(device_idx: usize) -> Self {
        Self { device_idx }
    }

    fn device_idx(&self) -> usize {
        self.device_idx
    }
}

impl Named for DeviceCtx {

}

impl WithId for DeviceCtx {
    type ID = ();

    fn id(&self) -> &Self::ID {
        &()
    }
}

impl PermitsCtx for DeviceCtx {
    fn add_permit(&mut self, permit: RunnerPerm) {

    }

    fn drop_permits(&mut self, drop_if: DropIf) {

    }

    fn acquired_permits(&self) -> NumPermits {
        0
    }
}

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

        let device_idx = ctx.device_idx();

        let unfold_params = UnfoldParams::new(
            1,
            input.stacked_motifs_output.motifs.extra().1.window as i64,
            input.stacked_motifs_output.motifs.extra().1.step as i64,
        );

        let (batch_1_t, batch_1_sizes) = unfold_and_cat(input.batch, &unfold_params)?;

        debug!("ClcResourcesMotifsDist::do_run: self.num_channels={} batch_1_t.size={:?} batch_1_t.device={:?} motifs.size={:?} avg_dists.size={:?} input.resource_ids={:?}",self.num_channels, batch_1_t.size(), batch_1_t.device(), input.stacked_motifs_output.motifs.data().t_ref().size(), input.stacked_motifs_output.motifs_avg_dists.t_ref().size(), input.resource_ids);

        #[cfg(test)]
        if self.fail_if_size_bigger_than.map(|v| v < batch_1_t.size()[1] as usize).unwrap_or(false) {
            debug!("ClcResourcesMotifsDist::do_run: triggering OOM: v={:?} batch_1_t.size()[1]={}", self.fail_if_size_bigger_than, batch_1_t.size()[1]);
            return Err(ThoError::oom());
        }

        let resource_ids_sizes = input
            .resource_ids
            .iter()
            .zip(batch_1_sizes.into_iter())
            .map(|(resource_id, size)| (*resource_id, size as usize))
            .collect::<Vec<_>>();

        let resources_iter =
            SlicesIter::new(unfold_params.step() as usize, resource_ids_sizes);

        let mut mask_makers = vec![
            LocalMin { dim: 2 },
            Le {
                v: input
                    .stacked_motifs_output
                    .motifs_avg_dists
                    .t_ref()
                    .unsqueeze(-1)
                    .expand(
                        [-1, -1, batch_1_t.data().size()[1]],
                        false,
                    ),
            },
        ];

        let channel_masks_dists_ids = {
            if device_idx > 0 {
                let motifs_stream = ThoCudaStream::new((device_idx - 1) as i32);
                let _guard = ThoStreamGuard::set(&motifs_stream);
            }

            let (mask_dists_result, motifs_stream) = with_cuda_stream(device_idx, || {
                thoosa_motifs::distance::clc_dist_d2_and_mask(
                    input
                        .stacked_motifs_output
                        .motifs
                        .data()
                        .t_ref()
                        .shallow_clone(),
                    batch_1_t.shallow_clone(),
                    true,
                    &mut mask_makers,
                )
            });

            let (mask, dists) = mask_dists_result?;

            //motif ids are channel-specific so have to build DistsMask from folded
            (0..self.num_channels)
                .map(|c| {
                    let ids = &input.stacked_motifs_output.motifs.extra().0[c];
                    if ids.is_empty() {
                        Ok(None) //no motifs in channel c
                    } else {
                        let mask = fold_mask(mask.i((c as i64, .., ..)), ids, 0);
                        let dists = dists.as_ref().map(|d| fold_dists(d.i((c as i64, .., ..)), ids, 0));
                        let mask_type = if mask.device().eq(&Cpu) {
                            DistsMaskType::AsIs
                        } else {
                            DistsMaskType::MaskPacked(mask.size(), dists.as_ref().map(|t| t.size()))
                        };
                        let stream = if mask.device().eq(&Cpu) {
                            None
                        } else {
                            motifs_stream.clone()
                        };
                        Ok(Some((
                            //unsqueeze is to put back dimension for channel removed before folding
                            DistsMask::new(mask.unsqueeze(0), dists.map(|t|t.unsqueeze(0)), mask_type, stream)?,
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

        Ok(ClcResourcesMotifsDistOutput::new(
            input
                .stacked_motifs_output
                .motifs
                .extra()
                .1
                .id
                .expect("no select_motifs_run_config_id"),
            channel_masks_dists_ids,
            resources_iter,
        ))
    }
}


#[derive(Debug)]
pub struct ClcResourcesMotifsDistInputDeviceWrapper {
    id: SynKey,
    batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>,
    resource_ids: Vec<ResourceId>,
    stacked_motifs: Arc<DevicesWrap<StackedMotifsOutput>>,
}

impl Adjustable for ClcResourcesMotifsDistInputDeviceWrapper {

}



//implemented so that ClcResourcesMotifsDistInputDeviceWrapper's resources batch can be adjusted.
//stacked_motifs_output can't be adjusted since moitfs' ranges do not match in different channels
impl Data for ClcResourcesMotifsDistInputDeviceWrapper {
    type Metrics = i64;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.batch.len() as i64
    }

    fn device(&self) -> usize {
        0
    }
}

impl ClcResourcesMotifsDistInputDeviceWrapper {
    pub fn new(id: SynKey, batch: Vec<Arc<DevicesWrap<ThoTensor<ChannelsData2, SynKey>>>>, resource_ids: Vec<ResourceId>, stacked_motifs: Arc<DevicesWrap<StackedMotifsOutput>>) -> Self {
        Self { id, batch, resource_ids, stacked_motifs }
    }
}

impl WithId for ClcResourcesMotifsDistInputDeviceWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl WithDataRef for ClcResourcesMotifsDistInputDeviceWrapper {
    type O<'o> = ClcResourcesMotifsDistInput<'o>;

    async fn with_data_ref<R, F, P>(&self, to_device_idx: usize, _permits_ctx: &mut P, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        F: for <'f> FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError>,
        P:PermitsCtx
    {
        let t_1_batch= self.batch.clone();
        let stacked_motifs = Arc::clone(&self.stacked_motifs);
        let ctx = DeviceCtx::new(to_device_idx);
        with_data!(ctx, [t_1_batch], stacked_motifs;
            |
             batch_v:&[&ThoTensor<ChannelsData2, SynKey>],
             stacked_motifs:&StackedMotifsOutput
            |{

                do_run(&ClcResourcesMotifsDistInput::new(
                    batch_v,
                    self.resource_ids.clone(),
                    stacked_motifs,
                ), _permits_ctx)

        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct ClcResourcesMotifsDistInputDeviceWrapperSelector {
    resources_batch_range:Range<i64>
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


impl WithSelectDataRef<ClcResourcesMotifsDistInputDeviceWrapperSelector> for ClcResourcesMotifsDistInputDeviceWrapper {
    async fn with_selected_data_ref<'o, R, F, P>(&self, to_device_idx: usize, _permits_ctx: &mut P, selector: ClcResourcesMotifsDistInputDeviceWrapperSelector, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoError>
    where
        P:PermitsCtx,
        for <'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError>
    {

        debug!("ClcResourcesMotifsDistInputDeviceWrapper::with_selected_data_ref: selector={selector:?}");


        let range = selector.resources_batch_range.start as usize..selector.resources_batch_range.end as usize;
        let t_1_batch= self.batch[range.clone()].to_vec();
        let stacked_motifs_output = Arc::clone(&self.stacked_motifs);

        let ctx = DeviceCtx::new(to_device_idx);
        with_data!(ctx, [t_1_batch], stacked_motifs_output;
            |
             batch_v:&[&ThoTensor<ChannelsData2, SynKey>],
             stacked_motifs_output:&StackedMotifsOutput
            |{
                let inpt = ClcResourcesMotifsDistInput::new(
                    &batch_v,
                    self.resource_ids[range].to_vec(),
                    stacked_motifs_output,
                );

                do_run(&inpt, _permits_ctx)
        })
    }
}


#[derive(Debug)]
pub struct ClcResourcesMotifsDistInput<'a> {
    id: SynKey,
    batch: &'a [&'a ThoTensor<ChannelsData2, SynKey>],
    resource_ids: Vec<ResourceId>,
    stacked_motifs_output: &'a StackedMotifsOutput,
}


impl<'a> ClcResourcesMotifsDistInput<'a> {
    pub fn new(
        batch: &'a [&'a ThoTensor<ChannelsData2, SynKey>],
        resource_ids: Vec<ResourceId>,
        batched_motifs_output: &'a StackedMotifsOutput,
    ) -> ClcResourcesMotifsDistInput<'a> {
        Self {
            id: SynKey::new([mono_key!("ClcResourcesMotifsDistInput", "stage")]),
            batch,
            resource_ids,
            stacked_motifs_output: batched_motifs_output,
        }
    }

}


impl <'a> Data for ClcResourcesMotifsDistInput<'a> {
    type Metrics = i64;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        self.batch.len() as i64
    }

    fn device(&self) -> usize {
        self.batch[0].device_idx()
    }

    fn mem_size(&self) -> u64 {

        let batch_mem_size = self.batch.iter().map(|t|t.mem_size()).sum::<u64>();
        let motifs_output_mem_size =  self.stacked_motifs_output.motifs.mem_size();
        let motifs_avg_dists_mem_size = self.stacked_motifs_output.motifs_avg_dists.mem_size();

        debug!("ClcResourcesMotifsDistInput::mem_size: batch_mem_size={} motifs_output_mem_size={} motifs_avg_dists_mem_size={}", batch_mem_size, motifs_output_mem_size, motifs_avg_dists_mem_size);

        batch_mem_size + motifs_output_mem_size + motifs_avg_dists_mem_size
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
    pub find_motifs_run_id: RunId,
    pub channel_masks_dists_ids: Vec<Option<(DistsMask, Vec<DataId>)>>,
    pub resources_iter: SlicesIter,
}

impl ClcResourcesMotifsDistOutput {
    pub fn new(
        find_motifs_run_id: RunId,
        channel_masks_dists_ids: Vec<Option<(DistsMask, Vec<DataId>)>>,
        resources_iter: SlicesIter,
    ) -> Self {
        debug!("ClcResourcesMotifsDistOutput::new: select_motifs_run_id={find_motifs_run_id} channel_masks_dists_ids={channel_masks_dists_ids:?} resources_iter={resources_iter:?}");

        Self {
            id: SynKey::new([mono_key!("ClcResourcesMotifsDistOutput", "stage")]),
            find_motifs_run_id,
            channel_masks_dists_ids,
            resources_iter,
        }
    }

    pub async fn to_cpu(self) -> ClcResourcesMotifsDistOutput {
        ClcResourcesMotifsDistOutput {
            id: self.id,
            find_motifs_run_id: self.find_motifs_run_id,
            channel_masks_dists_ids: futures::stream::iter(self.channel_masks_dists_ids.into_iter())
                .map( |op|{
                    async {
                        match op {
                            None => None,
                            Some((dists_mask, data_ids)) => {
                                Some((dists_mask.to_cpu().await.unwrap(), data_ids))//TODO unwrap
                            }
                        }
                    }
                })
                .buffered(crate::buffer_limits::CLC_RESOURCES_MOTIFS_DIST_OUTPUT_TO_CPU_BUFFERED_LIMIT)
                .collect::<Vec<_>>()
                .await,
            resources_iter: self.resources_iter,
        }
    }
}

impl Data for ClcResourcesMotifsDistOutput {
    type Metrics = ();
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        ()
    }

    fn device(&self) -> usize {
        0
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
pub fn fold_mask<I>(mask: Tensor, ids: &Vec<I>, dim: usize) -> Tensor
where
    I: PartialEq + Debug,
{
    reduce_by_groups(mask, |chunk| chunk.any_dim(dim as i64, true), ids, dim)
}

///returns mean dist between all motif slices x resource slice
pub fn fold_dists<I>(mask: Tensor, ids: &Vec<I>, dim: usize) -> Tensor
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
) -> Tensor
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
        let chunk = full.narrow(dim as i64, s, l);
        let reduced = reduce(chunk);
        out.push(reduced);
    };

    let mut start = 0usize;
    let mut prev = &ids[0];
    for (i, id) in ids.iter().enumerate().skip(1) {
        if id != prev {
            push_group(start, i - start);
            start = i;
            prev = id;
        }
    }
    push_group(start, ids.len() - start);

    Tensor::cat(&out, dim as i64)
}
