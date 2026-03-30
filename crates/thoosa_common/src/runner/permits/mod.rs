use std::any::Any;
use async_trait::async_trait;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use log::debug;
use thoosa_shared::ThoError;
use thoosa_shared::traits::CachableIdTr;
use tokio::sync::{Notify, OwnedSemaphorePermit, mpsc};
use thoosa_shared::comp_id::SynKey;
use crate::runner::global_ctx::GlobalCtx;

pub mod runner_queue;

#[derive(Debug)]
pub enum PermitStep {
    TEN,
    TWENTY,
    FIFTY,
}

impl Deref for PermitStep {
    type Target = NumPermits;

    fn deref(&self) -> &Self::Target {
        match self {
            PermitStep::TEN => &2,
            PermitStep::TWENTY => &4,
            PermitStep::FIFTY => &16,
        }
    }
}

pub const MAX_PERMITS: NumPermits = 32;

#[async_trait]
pub trait PermitSource: Debug + Send + Sync {
    async fn acquire(
        &self,
        num_permits: NumPermits,
        out_id: std::sync::Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoError>;

    async fn wait_for_permits_hint(&self);

    #[cfg(tho_debug)]
    fn move_permits(&self, from:&SynKey, to:&SynKey);

    #[cfg(tho_debug)]
    fn remove_granted_counts(&self, requester_id:Arc<dyn CachableIdTr>, count:u8);

    async fn shutdown(&self) {

    }

    fn as_any<'a>(&'a self) -> &'a dyn Any {
        todo!()
    }
}

pub type NumPermits = u8;

pub struct RunnerPerm {
    inner: Option<OwnedSemaphorePermit>,
    drop_signal: Option<DropSignal>,
    num_permits: NumPermits,
}

#[derive(Debug)]
pub struct DropSignal {
    worker_notify: Arc<Notify>,//Arc to Arbiter worker notify, for triggering next loop at crates/thoosa_shared/src/runner/permits/runner_queue.rs:404
    hint_notify: Arc<Notify>,//Arc to Arbiter hint notify, for waking wait_for_permits_hint without stealing worker wakeups
    device_idx: usize,
    requester_id: Arc<dyn CachableIdTr>,
}

impl DropSignal {
    pub fn new(
        worker_notify: Arc<Notify>,
        hint_notify: Arc<Notify>,
        device_idx:usize,
        requester_id: Arc<dyn CachableIdTr>,
    ) -> Self {
        Self {
            worker_notify,
            hint_notify,
            device_idx,
            requester_id,
        }
    }

}

impl Debug for RunnerPerm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnerPerm")
            .field("num_permits", &self.num_permits)
            .field("v", &self.inner)
            .finish()
    }
}

impl RunnerPerm {
    #[inline]
    pub fn num_permits(&self) -> NumPermits {
        self.num_permits
    }

    #[cfg(test)]
    pub fn new(inner: OwnedSemaphorePermit) -> Self {
        let num_permits = inner.num_permits() as NumPermits;
        Self {
            inner: Some(inner),
            drop_signal: None,
            num_permits,
        }
    }

    pub fn with_drop_signal(inner: OwnedSemaphorePermit, drop_signal: DropSignal) -> Self {
        let num_permits = inner.num_permits() as NumPermits;
        Self {
            inner: Some(inner),
            drop_signal: Some(drop_signal),
            num_permits,
        }
    }

    pub fn dummy(num_permits: NumPermits) -> Self {
        Self {
            inner: None,
            drop_signal: None,
            num_permits,
        }
    }

    pub fn update_requester_id(&mut self, requester_id:Arc<dyn CachableIdTr>){
        if let Some(drop_signal) = self.drop_signal.as_mut() {
            drop_signal.requester_id = requester_id;
        }
    }
}

impl Drop for RunnerPerm {
    fn drop(&mut self) {


        #[cfg(tho_debug)]
        if let Some(drop_signal) = self.drop_signal.take() {
            let permits_to_drop = self.num_permits() as u8;

            debug!("PermitsDebug: Drop for RunnerPerm::drop: drop_signal={drop_signal:?}");

            GlobalCtx::global().debug_semaphore_remove_perm(drop_signal.device_idx, drop_signal.requester_id, permits_to_drop);
        }
        drop(self.inner.take());
        // Releasing permits is done by tokio::OwnedSemaphorePermit’s Drop.
        //  wake the arbiter to re-check the head of queue
        if let Some(drop_signal) = self.drop_signal.take() {
            debug!("PermitsDebug: Drop for RunnerPerm::drop: drop_signal={drop_signal:?}");


            drop_signal.worker_notify.notify_one();
            drop_signal.hint_notify.notify_one();
        }
    }
}
