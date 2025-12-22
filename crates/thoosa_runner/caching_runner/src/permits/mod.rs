use async_trait::async_trait;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use log::trace;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::CachableIdTr;
use tokio::sync::{Notify, OwnedSemaphorePermit, mpsc};

pub mod runner_queue;
pub mod runner_semaphore;

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
            PermitStep::FIFTY => &50,
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
    ) -> Result<RunnerPerm, ThoosaError>;
}

pub type NumPermits = u8;

pub struct RunnerPerm {
    inner: OwnedSemaphorePermit,
    drop_signal: Option<DropSignal>,
}

pub struct DropSignal {
    notify: Arc<Notify>,
    drop_tx: mpsc::UnboundedSender<Arc<dyn CachableIdTr>>,
    requester_id: Arc<dyn CachableIdTr>,
}

impl DropSignal {
    pub fn new(
        notify: Arc<Notify>,
        drop_tx: mpsc::UnboundedSender<Arc<dyn CachableIdTr>>,
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
            .field("v", &self.inner.num_permits())
            .finish()
    }
}

impl RunnerPerm {
    #[inline]
    pub fn num_permits(&self) -> NumPermits {
        self.inner.num_permits() as NumPermits
    }

    pub fn new(inner: OwnedSemaphorePermit) -> Self {
        Self {
            inner,
            drop_signal: None,
        }
    }

    pub fn with_drop_signal(inner: OwnedSemaphorePermit, drop_signal: DropSignal) -> Self {
        Self {
            inner,
            drop_signal: Some(drop_signal),
        }
    }
}

impl Drop for RunnerPerm {
    fn drop(&mut self) {
        trace!("Drop for RunnerPerm::drop");
        // Releasing permits is done by tokio::OwnedSemaphorePermit’s Drop.
        // We only need to wake the arbiter to re-check the head of queue.
        if let Some(drop_signal) = self.drop_signal.take() {
            let _ = drop_signal.drop_tx.send(drop_signal.requester_id);
            drop_signal.notify.notify_one();
        }
    }
}
