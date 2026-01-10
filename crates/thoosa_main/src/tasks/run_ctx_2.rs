use std::fmt::{Debug, Formatter};
use std::mem;
use std::sync::Arc;
use ambassador::Delegate;
use async_trait::async_trait;
use futures_util::future::BoxFuture;
use log::{debug, info, warn};
use parking_lot::{Mutex, RwLock};
use tracing::error;
use thoosa_shared::runner::common::{ambassador_impl_PermitsCtx, DropIf, MockRunCtx, PermitsCtx, RunCtx};
use thoosa_shared::runner::common::cachable::Cached;
use thoosa_shared::traits::{Adjustable, WithId};
use thoosa_named_derive::Named;
use thoosa_shared::runner::permits::{NumPermits, RunnerPerm};
use thoosa_shared::resource::Resource;
use thoosa_shared::ThoError;
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_common::torch_runner::TorchPermsCtx;
use thoosa_shared::runner::global_ctx::GlobalCtx;
use crate::tasks::stacked_moitfs_loader::{StackedMotifsLoader, StackedMotifsOutputId};
use crate::tasks::motif_loader::MotifLoader;
use crate::tasks::tensor_loader::RawTensorLoader;

#[derive(Delegate, Named)]
#[delegate(PermitsCtx, target = "perms")]
pub struct RunCtx2 {
    id: SynKey,
    device_idx: usize,
    cache_area: u64,
    pub(crate) perms: TorchPermsCtx,//TODO pub

    loaded_raw_tensors:Vec<SynKey>,
    batched_motifs_id:Vec<StackedMotifsOutputId>,

    purge_gpu_closures:Option<Arc<Mutex<Vec<Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>> + Send + Sync>>>>>
}

impl Debug for RunCtx2 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunCtx2")
            .field("device_idx", &self.device_idx)
            .field("cache_area", &self.cache_area)
            .finish()
    }
}

impl Adjustable for RunCtx2 {}

impl WithId for RunCtx2 {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}


impl RunCtx2 {
    pub fn new<I:Into<SynKey>>(device_idx: usize, name: I) -> Self {
        Self::with_cache_area(device_idx, name, GlobalCtx::global().next_cache_area().expect("no more cache areas"))
    }

    fn with_cache_area<I: Into<SynKey>>(device_idx: usize, name: I, cache_area: u64) -> Self {

        let key = name.into();


        Self {
            id: key,
            device_idx,
            cache_area: GlobalCtx::global().next_cache_area().expect("no more cache areas"),
            perms: TorchPermsCtx::new("TorchPermsCtx"),
            loaded_raw_tensors: vec![],
            batched_motifs_id: vec![],
            purge_gpu_closures: Some(Arc::new(Mutex::new(vec![])))
        }
    }
}



#[async_trait]
impl RunCtx for RunCtx2 {
    fn device_idx(&self) -> usize {
        self.device_idx
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="RunCtx2::purge_gpu_caches"))]
    fn purge_gpu_caches(&mut self) -> BoxFuture<'static, Result<(), ThoError>>
    where
        Self: Sized
    {

        let (device_idx, ctx_id, cache_area, loaded_raw_tensors, batched_motifs_ids) = {

            (
                self.device_idx,
                self.id().clone(),
                self.cache_area,
                mem::replace(&mut self.loaded_raw_tensors, Vec::new()),
                mem::replace(&mut self.batched_motifs_id, Vec::new()),
            )
        };
        debug!("TorchRunCtx::purge_gpu_caches: 0 ctx_name={ctx_id:?} device_idx={device_idx} ");

        Box::pin(async move {

            debug!("purge_gpu_caches: 1 ctx_name={ctx_id:?} device_idx={device_idx} batched_motifs_ids={batched_motifs_ids:?} loaded_raw_tensors={loaded_raw_tensors:?}");
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

            debug!("purge_gpu_caches: 2 ctx_name={ctx_id:?} device_idx={device_idx}");

            Ok(())
        })

    }

    fn cache_area(&self) -> u64 {
        self.cache_area
    }

    fn branch<I:Into<SynKey>>(&mut self, branch_name: I, with_new_cache_area: bool) -> Self {

        let mut ctx2 = RunCtx2::with_cache_area(self.device_idx, &self.id + branch_name.into(), 0);
        ctx2.perms.permits = mem::take(&mut self.perms.permits);
        ctx2.device_idx = self.device_idx;
        if with_new_cache_area {
            let new_cache_area = GlobalCtx::global().next_cache_area().expect("ran out of cache areas");

            debug!("RunCtx2::with_cache_area: branch_name={:?}  new_cache_area={new_cache_area:064b}", ctx2.id);

            ctx2.cache_area = new_cache_area;
            ctx2.purge_gpu_closures = Some(Arc::new(Mutex::new(vec![])));//this causes it to purge gpu caches on Drop
        } else {
            ctx2.cache_area = self.cache_area;
            ctx2.purge_gpu_closures = self.purge_gpu_closures.as_ref().map(|v| Arc::clone(v));
        }
        ctx2
    }

    fn add_purge_gpu_closure(&mut self, cl:Box<dyn Fn()->BoxFuture<'static, Result<(), ThoError>>+ Send + Sync>) {
        if let Some(purge_gpu_closures) = self.purge_gpu_closures.as_ref() {
            purge_gpu_closures.lock().push(cl);
        }
    }

}

impl Drop for RunCtx2 {
    fn drop(&mut self) {

        if let Some(spurge_gpu_closures) = self.purge_gpu_closures.take() {
            if Arc::strong_count(&spurge_gpu_closures) == 1 {// it's topmost ctx
                match Arc::into_inner(spurge_gpu_closures) {
                    None => {
                        warn!("can't get inner from Arc with 1 strong ref in Drop for RunCtx2");
                    }
                    Some(m) => {

                        if let Err(e) = GlobalCtx::global().run_purge_gpu(m.into_inner(), self.cache_area) {
                            error!("run_purge_gpu failed: {e:?} ");
                        }
                    }
                }
            }
        }


        debug!("Drop for RunCtx2::drop: id={:?} self.perms.permits.len={}", self.id, self.perms.permits.len());
    }
}


pub trait MotifsCtx {
    fn add_stacked_motifs_id(&mut self, id:StackedMotifsOutputId);
    fn add_loaded_raw_tensor_id(&mut self, id: SynKey);
}


impl MotifsCtx for RunCtx2 {
    fn add_stacked_motifs_id(&mut self, id: StackedMotifsOutputId) {
        self.batched_motifs_id.push(id);
    }

    fn add_loaded_raw_tensor_id(&mut self, id: SynKey) {
        self.loaded_raw_tensors.push(id);
    }
}

//TODO move to real mock
impl MotifsCtx for MockRunCtx {
    fn add_stacked_motifs_id(&mut self, id: StackedMotifsOutputId) {

    }

    fn add_loaded_raw_tensor_id(&mut self, id: SynKey) {

    }
}
