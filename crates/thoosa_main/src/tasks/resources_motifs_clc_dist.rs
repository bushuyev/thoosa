use std::collections::HashSet;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

use log::{debug, trace};
use parking_lot::RwLock;
use tch::{IndexOp, Tensor};

use caching_runner::common::runnable::{Stage, SyncStage};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::common::cachable::{DevicesWrap, WithDataRef};
use thoosa_common::applicables::with_cuda_stream;
use thoosa_dummy_data_derive::DummyData;
use thoosa_motifs::distance::MaskMaker::{Le, LocalMin};
use thoosa_shared::resource::ResourceId;
use thoosa_shared::run_configs::RunId;
use thoosa_shared::traits::{Adjustable, DataId, Select, SimpleId, WithId};
use thoosa_shared::ThoosaError;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use thoosa_with_data::with_data;
use crate::tasks::batched_moitfs_loader::BatchedMotifsOutput;
use crate::tasks::cross_batch_clc::{unfold_and_cat, UnfoldParams};
use crate::tasks::cross_dims::ChannelsData2;
use crate::tasks::dists_tracker::DistsMask;
use crate::tasks::slices_iter::SlicesIter;
use torch_runner::data::ThoTensor2;

#[derive(Debug)]
pub struct ClcRresourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    num_channels: usize,
    _ctx: PhantomData<C>,
}

impl <C> Adjustable for ClcRresourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{}

impl <C>  OutId for  ClcRresourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        format!("ClcRresourcesMotifsDist_{}", input_id).into()
    }
}

impl<C> ClcRresourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    pub fn new(num_channels: usize) -> Self {
        Self {
            num_channels,
            _ctx: PhantomData,
        }
    }
}

impl<C> Stage for ClcRresourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    type Ctx = C;
    type In<'a> = ClcResourcesMotifsDistInput<'a>;
    type Out = ClcRresourcesMotifsDistOutput;
}

impl<C> SyncStage for ClcRresourcesMotifsDist<C>
where
    C: RunCtx + Adjustable + 'static,
{
    fn do_run<'f>(
        self: Arc<Self>,
        input: &Self::In<'f>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError> {
        let device_idx = ctx.read().device_idx();

        let unfold_params = UnfoldParams::new(
            1,
            input.batched_motifs_output.motifs.extra().1.window as i64,
            input.batched_motifs_output.motifs.extra().1.step as i64,
        );

        let (batch_1_t, batch_1_sizes) = unfold_and_cat(input.batch, &unfold_params)?;
        debug!("ClcRresourcesMotifsDist::do_run: batch_1_t.size()={:?} motifs.size={:?} avg_dists.size={:?}",batch_1_t.size(),input.batched_motifs_output.motifs.data().t_ref().size(),input.batched_motifs_output.motifs_avg_dists.t_ref().size());

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
                    .batched_motifs_output
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
                let motifs_stream =
                    ThoCudaStream::new((device_idx - 1) as i32);
                let _guard = ThoStreamGuard::set(&motifs_stream);
            }

            let ((mask, dists), motifs_stream) = with_cuda_stream(device_idx, || {
                thoosa_motifs::distance::clc_dist_d2_and_mask(
                    input
                        .batched_motifs_output
                        .motifs
                        .data()
                        .t_ref()
                        .shallow_clone(),
                    batch_1_t.shallow_clone(),
                    true,
                    &mut mask_makers,
                )
                .unwrap()
            });

            (0..self.num_channels)
                .map(|c| {
                    let ids =
                        &input.batched_motifs_output.motifs.extra().0[c];
                    if ids.is_empty() {
                        None
                    } else {
                        Some((
                            DistsMask::new(
                                fold_mask(
                                    mask.i((c as i64, .., ..)),
                                    ids,
                                    0,
                                ),
                                dists.as_ref().map(|d| {
                                    fold_dists(
                                        d.i((c as i64, .., ..)),
                                        ids,
                                        0,
                                    )
                                }),
                                true,
                                motifs_stream.clone(),
                            ),
                            ids.iter()
                                .fold((vec![], HashSet::new()), |mut acc, d| {
                                    if !acc.1.contains(d) {
                                        acc.1.insert(d);
                                        acc.0.push(d.clone());
                                    }
                                    acc
                                })
                                .0,
                        ))
                    }
                })
                .collect::<Vec<Option<(DistsMask, Vec<DataId>)>>>()
        };

        Ok(ClcRresourcesMotifsDistOutput::new(
            input
                .batched_motifs_output
                .motifs
                .extra()
                .2
                .id
                .expect("no select_motifs_run_config_id"),
            channel_masks_dists_ids,
            resources_iter,
        ))
    }
}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct ClcResourcesMotifsDistInputDeviceWrapper {
    id: SimpleId,
    batch: Vec<Arc<DevicesWrap<ThoTensor2<ChannelsData2, SimpleId>>>>,
    resource_ids: Vec<ResourceId>,
    batched_motifs_output: Arc<DevicesWrap<BatchedMotifsOutput>>,
}

impl WithId for ClcResourcesMotifsDistInputDeviceWrapper {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl WithDataRef for ClcResourcesMotifsDistInputDeviceWrapper {
    type O<'o> = ClcResourcesMotifsDistInput<'o>;

    async fn with_data_ref<'f, R, F>(&self, to_device_idx: usize, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
        F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>
    {
        let t_1_batch= self.batch.clone();
        let batched_motifs_output = Arc::clone(&self.batched_motifs_output);
        with_data!(to_device_idx, [t_1_batch], batched_motifs_output;
            |
             batch_v:&'f [&ThoTensor2<ChannelsData2, SimpleId>],
             batched_motifs_output:&'f BatchedMotifsOutput
            |{

                do_run(&ClcResourcesMotifsDistInput::new(
                    batch_v,
                    self.resource_ids.clone(),
                    batched_motifs_output,
                ))

        })
        // todo!()
    }

    async fn with_selected_data_ref<'f, S, R, F>(&self, to_device_idx: usize, selector: S, trace_ctx: Option<&'static str>, do_run: F) -> Result<R, ThoosaError>
    where
        S: Clone,
        Self::O<'f>: Select<S, Output=Self::O<'f>>,
        F: FnOnce(&Self::O<'f>) -> Result<R, ThoosaError>
    {
        todo!()
    }
}


#[derive(Debug, DummyData)]
#[dummy_data(no_clone)]
pub struct ClcResourcesMotifsDistInput<'a> {
    id: SimpleId,
    batch: &'a [&'a ThoTensor2<ChannelsData2, SimpleId>],
    resource_ids: Vec<ResourceId>,
    batched_motifs_output: &'a BatchedMotifsOutput,
}


impl<'a> ClcResourcesMotifsDistInput<'a> {
    pub fn new(
        batch: &'a [&'a ThoTensor2<ChannelsData2, SimpleId>],
        resource_ids: Vec<ResourceId>,
        batched_motifs_output: &'a BatchedMotifsOutput,
    ) -> ClcResourcesMotifsDistInput<'a>{
        Self {
            id: SimpleId("ClcRresourcesMotifsDistInput"),
            batch,
            resource_ids,
            batched_motifs_output,
        }
    }

}

impl <'a> Select<()> for ClcResourcesMotifsDistInput<'a> {
    type Output =  ClcResourcesMotifsDistInput<'a>;

    fn subview(&self, selector: ()) -> Result<Self::Output, ThoosaError> {
        unimplemented!()
    }
}

impl<'a> WithId for ClcResourcesMotifsDistInput<'a> {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl<'a> thoosa_shared::applicables_runner::Input for ClcResourcesMotifsDistInput<'a> {}

#[derive(Debug)]
pub struct ClcRresourcesMotifsDistOutput {
    id: SimpleId,
    pub select_motifs_run_config_id: RunId,
    pub channel_masks_dists_ids: Vec<Option<(DistsMask, Vec<DataId>)>>,
    pub resources_iter: SlicesIter,
}

impl ClcRresourcesMotifsDistOutput {
    fn new(
        select_motifs_run_config_id: RunId,
        channel_masks_dists_ids: Vec<Option<(DistsMask, Vec<DataId>)>>,
        resources_iter: SlicesIter,
    ) -> Self {
        Self {
            id: SimpleId("ClcRresourcesMotifsDistOutput"),
            select_motifs_run_config_id,
            channel_masks_dists_ids,
            resources_iter,
        }
    }
}

impl WithId for ClcRresourcesMotifsDistOutput {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl thoosa_shared::applicables_runner::Output for ClcRresourcesMotifsDistOutput {}

pub fn fold_mask<I>(mask: Tensor, ids: &Vec<I>, dim: usize) -> Tensor
where
    I: PartialEq + Debug,
{
    reduce_by_groups(mask, |chunk| chunk.any_dim(dim as i64, true), ids, dim)
}

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
    debug!(
        "fold_mask: mask={full:?} ids={ids:?} dim={dim}"
    );
    trace!("fold_mask: mask=\n{full}");

    let sizes = full.size();
    let nd = sizes.len();
    assert!(dim < nd, "dim out of range");

    let mut out: Vec<Tensor> = Vec::new();

    let mut push_group = |start: usize, len: usize| {
        let s = start as i64;
        let l = len as i64;
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
