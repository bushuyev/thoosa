use async_trait::async_trait;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use log::debug;
use crate::ThoError;
use crate::traits::CachableIdTr;
use tokio::sync::{Notify, OwnedSemaphorePermit, mpsc};
use crate::comp_id::SynKey;

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
            PermitStep::TEN => &10,
            PermitStep::TWENTY => &20,
            PermitStep::FIFTY => &51,
        }
    }
}

pub const MAX_PERMITS: NumPermits = 100;

#[async_trait]
pub trait PermitSource: Debug + Send + Sync {
    async fn acquire(
        &self,
        num_permits: NumPermits,
        out_id: std::sync::Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, ThoError>;

    async fn wait_for_permits_hint(&self);

    //TODO #[cfg(debug_assertions)]
    fn move_permits(&self, from:&SynKey, to:&SynKey);

    async fn shutdown(&self) {

    }
}

pub type NumPermits = u8;

pub struct RunnerPerm {
    inner: Option<OwnedSemaphorePermit>,
    drop_signal: Option<DropSignal>,
}

pub struct DropSignal {
    notify: Arc<Notify>,//Arc to Arbiter::notify
    drop_tx: mpsc::UnboundedSender<(Arc<dyn CachableIdTr>, u8)>,
    requester_id: Arc<dyn CachableIdTr>,
}

impl DropSignal {
    pub fn new(
        notify: Arc<Notify>,
        drop_tx: mpsc::UnboundedSender<(Arc<dyn CachableIdTr>, u8)>,
        requester_id: Arc<dyn CachableIdTr>,
    ) -> Self {
        Self {
            notify,
            drop_tx,
            requester_id,
        }
    }

}

impl Debug for RunnerPerm {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunnerPerm")
            .field("v", &self.inner)
            .finish()
    }
}

impl RunnerPerm {
    #[inline]
    pub fn num_permits(&self) -> NumPermits {
        self.inner.as_ref().map(|v|v.num_permits()).unwrap_or(0) as NumPermits
    }

    #[cfg(test)]
    pub fn new(inner: OwnedSemaphorePermit) -> Self {
        Self {
            inner: Some(inner),
            drop_signal: None,
        }
    }

    pub fn with_drop_signal(inner: OwnedSemaphorePermit, drop_signal: DropSignal) -> Self {
        Self {
            inner: Some(inner),
            drop_signal: Some(drop_signal),
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

        let permits_to_drop = self.num_permits() as u8;
        drop(self.inner.take());
        // Releasing permits is done by tokio::OwnedSemaphorePermit’s Drop.
        //  wake the arbiter to re-check the head of queue
        if let Some(drop_signal) = self.drop_signal.take() {
            debug!("PermitsDebug: Drop for RunnerPerm::drop: drop_signal.requester_id={:?} num_permits={permits_to_drop:?}", drop_signal.requester_id);

            let _ = drop_signal.drop_tx.send((drop_signal.requester_id, permits_to_drop));
            drop_signal.notify.notify_one(); //requesters also wait on same notify
        }
    }
}
