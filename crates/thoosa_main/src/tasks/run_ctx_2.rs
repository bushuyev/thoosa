use std::mem;
use std::sync::Arc;
use ambassador::Delegate;
use async_trait::async_trait;
use futures_util::future::BoxFuture;
use log::{debug, info};
use parking_lot::RwLock;
use caching_runner::common::{ambassador_impl_PermitsCtx, DropIf, MockRunCtx, PermitsCtx, RunCtx};
use caching_runner::common::cachable::Cached;
use thoosa_shared::traits::Adjustable;
use caching_runner::permits::{NumPermits, RunnerPerm};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{Named, SimpleId};
use torch_runner::TorchPermsCtx;
use crate::tasks::stacked_moitfs_loader::{StackedMotifsLoader, StackedMotifsOutputId};
use crate::tasks::motif_loader::MotifLoader;
use crate::tasks::tensor_loader::RawTensorLoader;

#[derive(Debug, Delegate)]
#[delegate(PermitsCtx, target = "perms")]
pub struct RunCtx2 {
    device_idx: usize,
    cache_area: u64,
    name: &'static str,
    perms: TorchPermsCtx,

    loaded_raw_tensors:Vec<SimpleId>,
    batched_motifs_id:Vec<StackedMotifsOutputId>
}

impl Adjustable for RunCtx2 {}

impl Named for RunCtx2 {
    fn name(&self) -> &'static str {
        self.name
    }
}
impl RunCtx2 {
    pub fn new(device_idx: usize, cache_area: u64, name: &'static str) -> Self {
        Self {
            device_idx,
            cache_area,
            name,
            perms: TorchPermsCtx::new(name),
            loaded_raw_tensors:vec![],
            batched_motifs_id: vec![],
        }
    }
}

#[async_trait]
impl RunCtx for RunCtx2 {
    fn device_idx(&self) -> usize {
        self.device_idx
    }

    fn purge_gpu_caches(ctx: Arc<RwLock<Self>>) -> BoxFuture<'static, Result<(), ThoosaError>>
    where
        Self: Sized
    {
        Box::pin(async move {
            let (device_idx, ctx_name) = {
                let guard = ctx.read();
                (guard.device_idx, guard.name)
            };
            assert_ne!(device_idx, 0);

            debug!("TorchRunCtx::purge_gpu_caches: 0 ctx_name={ctx_name} device_idx={device_idx} ");

            let (cache_area, loaded_raw_tensors, batched_motifs_ids) = {
                let mut guard = ctx.write();

                (
                    guard.cache_area,
                    mem::replace(&mut guard.loaded_raw_tensors, Vec::new()),
                    mem::replace(&mut guard.batched_motifs_id, Vec::new()),
                )
            };
            debug!("purge_gpu_caches: 1 ctx_name={ctx_name} device_idx={device_idx} batched_motifs_ids={batched_motifs_ids:?} loaded_raw_tensors={loaded_raw_tensors:?}");
            debug!("dumping {} batched motifs", batched_motifs_ids.len());

            tokio::try_join!(
                    RawTensorLoader::<RunCtx2>::purge_gpu_cache_area(
                        loaded_raw_tensors,
                        cache_area,
                        device_idx
                    ),
                    StackedMotifsLoader::<MotifLoader<RawTensorLoader<RunCtx2>, RunCtx2>, RunCtx2>::purge_gpu_cache_area(
                        batched_motifs_ids,
                        cache_area,
                        device_idx
                    ),
                )?;

            debug!("purge_gpu_caches: 2 ctx_name={ctx_name} device_idx={device_idx}");

            Ok(())
        })

    }

    fn cache_area(&self) -> u64 {
        self.cache_area
    }

    fn branch(&self, new_name: &str) -> Arc<RwLock<Self>> {
        Arc::new(parking_lot::RwLock::new(RunCtx2::new(self.device_idx, self.cache_area, new_name.to_string().leak())))//TODO
    }
}


pub trait MotifsCtx {
    fn add_stacked_motifs_id(&mut self, id:StackedMotifsOutputId);
    fn add_loaded_raw_tensor_id(&mut self, id:SimpleId);
}


impl MotifsCtx for RunCtx2 {
    fn add_stacked_motifs_id(&mut self, id: StackedMotifsOutputId) {
        self.batched_motifs_id.push(id);
    }

    fn add_loaded_raw_tensor_id(&mut self, id: SimpleId) {
        self.loaded_raw_tensors.push(id);
    }
}

//TODO move to real mock
impl MotifsCtx for MockRunCtx {
    fn add_stacked_motifs_id(&mut self, id: StackedMotifsOutputId) {

    }

    fn add_loaded_raw_tensor_id(&mut self, id: SimpleId) {

    }
}