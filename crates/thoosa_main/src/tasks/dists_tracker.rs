use crate::tasks::cross_batch_clc::CrossBatchClcOutput;
use crate::tasks::cross_dims::ChannelsSlicesData2;
use crate::tasks::run_ctx_2::RunCtx2;
use async_trait::async_trait;
use caching_runner::common::Data;
use caching_runner::common::cachable::DataCacheWrapper;
use caching_runner::common::runnable::Stage;
use log::{debug, trace};
use parking_lot::{Mutex, RwLock};
use std::fmt::Debug;
use std::iter::Repeat;
use std::sync::Arc;
use tch::{IndexOp, Tensor};
use tch::kind::FLOAT_CPU;
use thoosa_common::bool_iter::BoolTensorIterator;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::{FixedFindMotifsConfig, SimpleMotifTracker};
use thoosa_shared::ThoosaError;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{SimpleId, WithId};
use thoosa_store::SqlMotifTrackerStore;
use thoosa_store::fs_store::FileMotifTrackerStore;
use torch_runner::data::ThoTensor2;

#[derive(Debug)]
pub struct DistWithMask {
    mask: Tensor,
    dists: Option<Tensor>,
}

impl Output for DistWithMask {}
impl Input for DistWithMask {}

impl WithId for DistWithMask {
    type ID = SimpleId;

    fn id(&self) -> &Self::ID {
        todo!()
    }
}

impl DistWithMask {
    pub fn new(mask: Tensor, dists: Option<Tensor>) -> Self {
        Self { mask, dists }
    }
}

#[derive(Debug)]
pub struct DistsTracker {
    channel_trackers: Vec<Mutex<SimpleMotifTracker<'static>>>,
}

impl DistsTracker {
    pub fn new() -> Self {
        Self {
            channel_trackers: vec![],
        }
    }

    pub fn new_2(
        ws: &Arc<RwLock<Box<dyn Workspace>>>,
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
        Self { channel_trackers }
    }

    fn dist_iter(dists_flattened: Option<Tensor>) -> Box<dyn Iterator<Item=f64>> {//TODO use Either::
        match dists_flattened.as_ref().map(|v| v.iter::<f64>().unwrap()) {
            None => Box::new(std::iter::repeat(0f64)) as Box<dyn Iterator<Item=f64>>,
            Some(v) => Box::new(v) as  Box<dyn Iterator<Item=f64>>
        }
    }
}
#[async_trait]
impl Stage for DistsTracker {
    type In<'a> = CrossBatchClcOutput<DistWithMask>;
    type Out = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type Ctx = RunCtx2;

    fn do_run<'a>(
        self: Arc<Self>,
        input: &Self::In<'a>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized,
    {
        let DistWithMask {
            mask:mask_cpu,
            dists,
        } = input.clc_result();



        for (channel, mt) in self.channel_trackers.iter().enumerate()/*.par_bridge()*/ {
            let mask_cpu_flattened = mask_cpu.i((channel as i64, .., ..)).flatten(0, 1);
            let dists_flattened = dists.as_ref().map(|d|d.i((channel as i64, .., ..)).flatten(0, 1));


            trace!("find_motifs_by_euc_dist: mask_cpu_flattened=\n{mask_cpu_flattened}");
            trace!("find_motifs_by_euc_dist: dists_flattened=\n{:?}", dists_flattened);

            let mut channel_mask: BoolTensorIterator = (&mask_cpu_flattened).into();
            let mut dists_and_mask_iter = channel_mask.zip(Self::dist_iter(dists_flattened));
            debug!("find_motifs_by_euc_dist: channel={channel} mask_cpu_flattened.size={:?}", mask_cpu_flattened.size());
            trace!("find_motifs_by_euc_dist: mask_cpu_flattened={mask_cpu_flattened}");


            input.iter().zip(dists_and_mask_iter).for_each(|v|{
                debug!("find_motifs_by_euc_dist: v={v:?}");
            });


        };


       Ok(ThoTensor2::new(
            "".into(),
            Tensor::rand(&[10, 1], FLOAT_CPU),
        ))
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
