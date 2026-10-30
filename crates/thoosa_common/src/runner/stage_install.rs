//! Typed installation shared by feature stages. Endpoint definitions and input
//! adapters describe a stage; backend selection and endpoint ownership live here.
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use std::fmt::Debug;
use std::future::Future;
use serde::{Serialize, de::DeserializeOwned};
use thoosa_shared::{ThoError, traits::Stage};
use crate::ipc::{cuda_wokrer::{WorkerStageInput, WorkerStageOutput}, worker_msg::WorkerInput, endpoints::StageRecovery, proxy::{ProxyStage, DynSyncStageWithStateEnum}};
use crate::runner::{global_ctx::DeviceExecutionCtx, sync_stage::{SyncStage, StageWithState}};

pub type InputCloner<S> = Arc<dyn for<'a> Fn(&<S as Stage>::In<'a>) -> Result<Box<dyn WorkerInput>, ThoError> + Send + Sync>;

impl DeviceExecutionCtx {
    pub async fn install_stateful<S, I, L, LF, R, RF>(
        &self, endpoint: &'static str, recovery: StageRecovery,
        local: L, remote: R, cloner: Option<InputCloner<S>>,
    ) -> Result<DynSyncStageWithStateEnum<S, I>, ThoError>
    where
        S: SyncStage + StageWithState + Send + Sync + 'static,
        for<'a> S::In<'a>: WorkerStageInput + Serialize,
        S::Out: WorkerStageOutput + DeserializeOwned + 'static,
        I: Debug + Send + Sync + 'static,
        L: FnOnce() -> LF, LF: Future<Output = Result<Arc<S>, ThoError>>,
        R: FnOnce() -> RF, RF: Future<Output = Result<(Box<dyn WorkerInput>, I), ThoError>>,
    {
        if let Some(actor) = self.ipc_backend() {
            // IDs belong to installations, not feature models. Every retry gets
            // a distinct ID; ProxyStage retains cleanup and recreation ownership.
            static NEXT_INSTALLATION: AtomicU64 = AtomicU64::new(1);
            let id = format!("installed-{}", NEXT_INSTALLATION.fetch_add(1, Ordering::Relaxed));
            let (wire, init) = remote().await?;
            let stage = ProxyStage::install_on(actor, endpoint, Some(id), self.device_idx(), wire, cloner, recovery).await?;
            Ok(DynSyncStageWithStateEnum::Proxy { stage: Arc::new(stage), init })
        } else {
            Ok(DynSyncStageWithStateEnum::Local(local().await?))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runner::{global_ctx::Runtime, run_ctx::RunCtxImpl};
    use crate::stages::freq_filter::{SyncFreqFilter, SyncFreqFilterIpcInit, SYNC_FREQ_FILTER_ENDPOINT};

    #[tokio::test]
    async fn test_typed_installation_selects_only_local_factory() -> Result<(), ThoError> {
        let runtime = Runtime::local(1)?;
        let installed = runtime.execution(0).install_stateful::<SyncFreqFilter<RunCtxImpl>, (), _, _, _, _>(
            SYNC_FREQ_FILTER_ENDPOINT, StageRecovery::RecreateFromInit,
            || async { Ok(Arc::new(SyncFreqFilter::new_local(SyncFreqFilterIpcInit::new(vec![], 1, 100., 1, 0))?)) },
            || async { panic!("local installation must not construct transport inputs") }, None,
        ).await?;
        assert!(matches!(installed, DynSyncStageWithStateEnum::Local(_)));
        runtime.shutdown().await;
        Ok(())
    }
}
