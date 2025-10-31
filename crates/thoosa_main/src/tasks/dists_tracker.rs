use crate::tasks::cross_batch_clc::CrossUnfoldClcOutput;
use crate::tasks::run_ctx_2::RunCtx2;
use async_trait::async_trait;
use caching_runner::common::cachable::DevicesWrap;
use caching_runner::common::runnable::{Stage, SyncStage};
use caching_runner::common::RunCtx;
use log::{debug, error, info, trace, warn};
use parking_lot::{Mutex, RwLock};
use std::fmt::{Debug, Formatter};
use std::iter::{Repeat, Zip};
use std::mem;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::{IndexOp, Kind, Tensor};
use tokio::runtime::Handle;
use futures::{stream, StreamExt};
use tch::Device::Cpu;
use tch::Kind::{Bool, Float, Uint8};
use thoosa_common::bool_iter::BoolTensorIterator;
use thoosa_common::copy_ctx::CopyCtx;
use thoosa_common::cuda_utils::{pack_bits_bool, unpack_bits_u8};
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_shared::CopyToDevice;
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::{FixedFindMotifsConfig, SimpleMotifTracker};
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{SimpleId, WithId};
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use thoosa_store::fs_store::FileMotifTrackerStore;
use thoosa_store::{SqlMotifTrackerStore, Store};
use torch_runner::data::ThoTensor2;

#[derive(Debug)]
pub struct DistsMask {
    mask: Tensor,
    dists: Option<Tensor>,
    sizes: Option<(Vec<i64>, Option<Vec<i64>>)>,//if packed
    stream_to_sync: Option<ThoCudaStream>,
}


unsafe impl Sync for DistsMask {}

impl Output for DistsMask {}
impl Input for DistsMask {}

impl WithId for DistsMask {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl DistsMask {

    pub fn new(mask: Tensor, dists: Option<Tensor>, pack: bool, stream_to_sync: Option<ThoCudaStream>) -> Self {
        assert_eq!(mask.kind(), Bool);
        Self {
            sizes: if pack {
                Some((mask.size(), dists.as_ref().map(|t|t.size())))
            } else {
                None
            },

            mask: if pack {
                pack_bits_bool(&mask)
            } else {
                mask
            },

            dists,
            stream_to_sync
        }
    }


    pub async fn to_cpu(self) -> Self  {

        if self.mask.device().eq(&Cpu) {
            self
        } else {
            if let Some(stream) = self.stream_to_sync {
                stream.synchronize();
            }


            Self {

                mask: if self.sizes.is_some() {
                    CopyCtx::instance().move_t(self.mask, 0, Uint8).await
                } else {
                    CopyCtx::instance().move_t(self.mask, 0, Bool).await
                },

                dists: None,
                sizes: self.sizes,
                stream_to_sync:None
            }
        }
    }

    pub fn unpack(self) -> (Tensor, Option<Tensor>) {
        debug!("DistsMask::unpack: self.sizes={:?}", self.sizes);
        match &self.sizes {//was packed
            Some(size) => {
                (
                    unpack_bits_u8(
                        &self.mask,
                        Some(size.0.iter().fold(1, |mut a, v| {a*=v; a}))
                    )
                    .to_kind(Bool)
                    .view(size.0.as_slice()),
                    self.dists
                )
            },
            None => {
                (self.mask, self.dists)
            }
        }
    }

}

pub struct DistsTracker<C>
where
    C: RunCtx + 'static,
{
    channel_trackers: Vec<Mutex<SimpleMotifTracker<'static>>>,
    store: Arc<RwLock<Box<dyn Store>>>,
    _ctx: PhantomData<C>,
}

impl<C> Debug for DistsTracker<C>
where
    C: RunCtx + 'static,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistsTracker")
            .field("channel_trackers", &self.channel_trackers)
            .finish()
    }
}

impl<C> DistsTracker<C>
where
    C: RunCtx + 'static,
{
    pub fn new(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
        store: &Arc<RwLock<Box<dyn Store>>>,
        run_config: &FindMotifsRunConfig,
        fixed_config: &FixedFindMotifsConfig,
        max_slices_counts: Option<u32>,
    ) -> Self {
        let channel_trackers = (0..ws.read().channels())
            .map(|channel| {
                debug!("DistsTracker:new: channel={channel}");

                Mutex::new(SimpleMotifTracker::new(
                    if fixed_config.keep_slice_matches {
                        SqlMotifTrackerStore::new(
                            ws.read().tmp_run_store_path(
                                run_config.id.expect("id") as u32,
                                channel as u32,
                            ),
                            run_config.id.expect("id"),
                            channel,
                        )
                        .expect("can't create tmp store")
                    } else {
                        FileMotifTrackerStore::new(
                            ws.read().tmp_path(),
                            run_config.id.expect("id"),
                            channel,
                        )
                        .expect("can't create tmp store")
                    },
                    fixed_config.min_tracker_step,
                    max_slices_counts,
                ))
            })
            .collect::<Vec<Mutex<SimpleMotifTracker>>>();

        debug!("DistsTracker: channel_trackers={:?}", channel_trackers);
        Self {
            channel_trackers,
            store: Arc::clone(store),
            _ctx: PhantomData,
        }
    }


    pub async fn build_groups(&self) {
        let mut trackers_done = AtomicU32::new(0);

        for mt in &self.channel_trackers {
            mt.lock().build_groups(&self.store).expect("build_groups");
            info!(" {} trackers done", trackers_done.fetch_add(1, Relaxed) + 1);
        }
    }
}

pub fn dist_iter(dists_flattened: Option<Tensor>) -> Box<dyn Iterator<Item = f64>> {
    //TODO use Either::
    match dists_flattened.as_ref().map(|v| v.iter::<f64>().unwrap()) {
        None => Box::new(std::iter::repeat(0f64)) as Box<dyn Iterator<Item = f64>>,
        Some(v) => Box::new(v) as Box<dyn Iterator<Item = f64>>,
    }
}


impl<C> Stage for DistsTracker<C>
where
    C: RunCtx + 'static,
{
    type Ctx = C;

    type In<'a> = CrossUnfoldClcOutput<DistsMask>;
    type Out = ();

}

impl<C> SyncStage for DistsTracker<C>
where
    C: RunCtx + 'static,
{

    fn consume<'a>(
        self: Arc<Self>,
        input: Self::In<'a>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError> {


        //TODO clone to resources ref
        let iter_0 = input.iter_0();
        let iter_1 = input.iter_1();

        let (ref mask_cpu, ref dists_cpu)= input.clc_result().unpack();

        tokio::task::block_in_place(|| {
            Handle::current().block_on(async {
                stream::iter(self.channel_trackers.iter().enumerate()).map(|(channel, mt)| {
                    let iter_0 = iter_0.clone();
                    let iter_1 = iter_1.clone();
                    async move {
                        let mask_cpu_flattened = mask_cpu.i((channel as i64, .., ..)).flatten(0, 1);
                        let dists_flattened = dists_cpu
                            .as_ref()
                            .map(|d| d.i((channel as i64, .., ..)).flatten(0, 1));


                        debug!("find_motifs_by_euc_dist: dists_flattened.size={:?}",dists_flattened.as_ref().map(|v| v.size()));
                        trace!("find_motifs_by_euc_dist: mask_cpu_flattened=\n{mask_cpu_flattened}");
                        trace!("find_motifs_by_euc_dist: dists_flattened=\n{:?}",dists_flattened);

                        let mut channel_mask: BoolTensorIterator = (&mask_cpu_flattened).into();
                        let mut dists_and_mask_iter = channel_mask.zip(dist_iter(dists_flattened));//

                        debug!("find_motifs_by_euc_dist: channel={channel} mask_cpu_flattened.size={:?}",mask_cpu_flattened.size());
                        trace!("find_motifs_by_euc_dist: mask_cpu_flattened={mask_cpu_flattened}");

                        for slice_0 in iter_0 {
                            let iter_1 = iter_1.clone();

                            for slice_1 in iter_1 {
                                if let Some((mask, dist)) = dists_and_mask_iter.next() {
                                    trace!("find_motifs_by_euc_dist: slice_0={slice_0:?}  slice_1={slice_1:?} mask={mask:?} dist={dist}");

                                    if mask
                                        && (slice_0.0 != slice_1.0 || slice_0.1 != slice_0.1
                                    ) {
                                        mt.lock().track(
                                            slice_0.clone(),
                                            slice_1.clone(),
                                            dist as f32, //TODO actual distances not needed?
                                        );
                                    }
                                } else {
                                    error!("find_motifs_by_euc_dist: not enough dists");
                                    panic!("find_motifs_by_euc_dist: not enough dists");
                                }
                            }
                        }
                        mt.lock().dump();
                    }
                }).buffer_unordered(self.channel_trackers.len()) // <- bound parallelism
                    .collect::<Vec<()>>()
                    .await
            })
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use thoosa_shared::ThoosaError;
    #[ctor::ctor]
    pub fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    #[tokio::test]
    async fn test_0() -> Result<(), ThoosaError> {
        Ok(())
    }
}
