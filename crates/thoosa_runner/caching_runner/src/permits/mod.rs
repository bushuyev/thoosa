use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::{Notify, OwnedSemaphorePermit};
use thoosa_common::traits::CachableIdTr;
use crate::common::MyError;

pub mod runner_queue;
pub mod runner_semaphore;


#[derive(Debug)]
pub enum PermitStep {
    TEN,
    TWENTY,
    FIFTY
}

impl Deref for PermitStep {
    type Target = NumPermits;

    fn deref(&self) -> &Self::Target {
        match self {
            PermitStep::TEN => &10,
            PermitStep::TWENTY => &20,
            PermitStep::FIFTY => &50
        }
    }
}

pub const MAX_PERMITS:NumPermits = 100;

#[async_trait]
pub trait PermitSource:Debug + Send + Sync {
    async fn acquire(
        &self,
        n: NumPermits,
        out_id: std::sync::Arc<dyn CachableIdTr>,
    ) -> Result<RunnerPerm, MyError>;
}

pub type NumPermits = u8;

pub struct RunnerPerm {
    inner: OwnedSemaphorePermit,
    notify: Option<Arc<Notify>>,
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

    pub fn new(inner: OwnedSemaphorePermit) -> Self { Self { inner, notify: None } }


    pub fn with_notify(inner: OwnedSemaphorePermit, notify: Arc<Notify>) -> Self {
        Self { inner, notify: Some(notify) }
    }
}

impl Drop for RunnerPerm {
    fn drop(&mut self) {
        // Releasing permits is done by tokio::OwnedSemaphorePermit’s Drop.
        // We only need to wake the arbiter to re-check the head of queue.
        if let Some(notify) = self.notify.as_ref() {
            notify.notify_one();
        }
    }
}
