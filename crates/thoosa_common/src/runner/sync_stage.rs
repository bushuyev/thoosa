//! Kernel capabilities. Async scheduling and remote dispatch live in execution;
//! tensor serialization belongs to the backend, not to these stage contracts.
use futures_util::future::BoxFuture;
use std::{fmt::Debug, sync::Arc};
use thoosa_shared::{ThoError, traits::Stage};

/// A borrowed synchronous kernel must implement execution; there is no fallback.
/// Remote handles cannot be called through the synchronous kernel API:
///
/// ```compile_fail
/// use thoosa_common::{ipc::proxy::ProxyStage, runner::{run_ctx::RunCtxImpl, sync_stage::SyncStage}, stages::freq_filter::SyncFreqFilter};
/// fn kernel<S: SyncStage>() {}
/// kernel::<ProxyStage<SyncFreqFilter<RunCtxImpl>>>();
/// ```
pub trait SyncStage: Stage + Debug {
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError>;
}

/// Consuming execution is independent: failure must return the owned input.
pub trait ConsumingStage: Stage + Debug {
    fn consume<'f>(self: Arc<Self>, input: Self::In<'f>, ctx: Self::Ctx) -> ConsumeResult<Self::In<'f>, Self::Out>;
}

/// Preparation always yields a usable, exclusively held state. Remote proxies
/// do not implement this capability: restoration belongs to their endpoint lease.
pub trait PrepareStage: Stage + Sync {
    type S: Debug + Send;
    fn pre_run(&self, device: usize) -> BoxFuture<'_, Result<tokio::sync::MappedMutexGuard<'_, Self::S>, ThoError>>;
}

pub trait StageWithState: PrepareStage {
    // The execution owner retains this guard through CUDA synchronization.
    fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, state: &mut Self::S, ctx: &Self::Ctx) -> Result<Self::Out, ThoError>;
}

#[derive(Debug)]
pub enum ConsumeResult<I, O> {
    Failed { input: I, error: ThoError },
    Ok(O),
}
