use std::fmt::Debug;
use std::sync::Arc;
use async_trait::async_trait;
use log::debug;
use parking_lot::{Mutex, RwLock};
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use caching_runner::common::cachable::DataCacheWrapper;
use caching_runner::common::Data;
use caching_runner::common::runnable::Stage;
use thoosa_common::find_motifs_run_config::FindMotifsRunConfig;
use thoosa_common::tho_tensor::CopyToDevice;
use thoosa_common::ws::Workspace;
use thoosa_motifs::find::{FixedFindMotifsConfig, SimpleMotifTracker};
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{SimpleId, WithId};
use thoosa_store::fs_store::FileMotifTrackerStore;
use thoosa_store::SqlMotifTrackerStore;
use torch_runner::data::ThoTensor2;
use crate::tasks::cross_dims::ChannelsSlicesData2;
use crate::tasks::run_ctx_2::RunCtx2;

#[derive(Debug)]
pub struct DistWithMask {
    mask:Tensor,
    dists:Option<Tensor>
}


impl Output for DistWithMask {

}
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
    channel_trackers: Vec<Mutex<SimpleMotifTracker<'static>>>
}

impl DistsTracker {
    pub fn new() -> Self {
        Self { channel_trackers: vec![] }
    }

    pub fn new_2(ws:&Box<dyn Workspace>, run_config:&FindMotifsRunConfig, fixed_config: &FixedFindMotifsConfig, max_slices_counts:Option<u32>) ->Self {
        let channel_trackers = (0..ws.channels()).map(|channel| {
            debug!("DistsTracker:new: channel={channel}");
            Mutex::new(SimpleMotifTracker::new(
                if fixed_config.keep_slice_matches {
                    SqlMotifTrackerStore::new(
                        ws.tmp_run_store_path(run_config.id.expect("id") as u32, channel as u32),
                        run_config.id.expect("id"),
                        channel
                    ).expect("can't create tmp store")
                } else {
                    debug!("DistsTracker:new: ws.tmp_path()={:?}", ws.tmp_path());
                    FileMotifTrackerStore::new(
                        ws.tmp_path(),
                        run_config.id.expect("id"),
                        channel
                    ).expect("can't create tmp store")
                },
                fixed_config.min_tracker_step,
                max_slices_counts
            ))
        }).collect::<Vec<Mutex<SimpleMotifTracker>>>();

        debug!("DistsTracker: channel_trackers={:?}", channel_trackers);
        Self {
            channel_trackers
        }
    }
}
#[async_trait]
impl Stage for DistsTracker {
    type In<'a> = DistWithMask;
    type Out = ThoTensor2<ChannelsSlicesData2, SimpleId>;
    type Ctx = RunCtx2;

    fn do_run<'a>(
        self: Arc<Self>,
        input: &Self::In<'a>,
        ctx: Arc<RwLock<Self::Ctx>>,
    ) -> Result<Self::Out, ThoosaError>
    where
        Self: Sized
    {


        Ok(ThoTensor2::new("".into(), Tensor::rand(&[10, 1], FLOAT_CPU)))
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
