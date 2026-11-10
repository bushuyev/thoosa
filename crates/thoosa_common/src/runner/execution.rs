//! The common async execution handle owns preparation and stream lifetimes.
//! Kernel traits never advertise remote or consuming operations they cannot run.
use crate::cuda_copier::{HandleCtx, LocalCudaExecution, finish_cuda_error, finish_cuda_panic, finish_cuda_run, run_cuda_stage};
use crate::runner::{
    common::cachable::{BorrowRef, ConsumeData, SelectRef},
    global_ctx::{CudaActorHandleEnum, DeviceExecutionCtx},
    sync_stage::{ConsumeResult, ConsumingStage, StageWithState, SyncStage},
};
use std::{borrow::Borrow, fmt::Debug, future::Future, sync::Arc};
use thoosa_shared::{
    ThoError,
    traits::{Stage, ThoSelector},
};

pub trait StageExecution: Stage + Send + Sync {
    fn execute<'a, 'i: 'a, I, IFut>(self: Arc<Self>, input: IFut, execution: &'a ExecutionContext<'_>, ctx: &'a Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<Self::Out, ThoError>>
    where
        Self: 'a,
        Self::In<'i>: Sync,
        Self::Out: Send,
        I: Borrow<Self::In<'i>> + Send + 'a,
        IFut: Future<Output = Result<I, ThoError>> + Send + 'a;
}

pub trait ConsumingExecution: Stage + Send + Sync {
    fn execute_consume<'a, 'i: 'a>(self: Arc<Self>, input: Self::In<'i>, execution: &'a ExecutionContext<'_>, ctx: Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<ConsumeResult<Self::In<'i>, Self::Out>, ThoError>>
    where
        Self: 'a,
        Self::In<'i>: Send + 'a,
        Self::Out: Send,
        Self::Ctx: 'a;
}

/// IPC has no local stream capability. The CPU context is an actual no-op
/// execution boundary; a local kernel submitted to IPC is a wiring error.
pub enum ExecutionContext<'a> {
    Cpu,
    Local(&'a dyn LocalCudaExecution),
    Remote,
}

struct CpuHandleCtx;
impl HandleCtx for CpuHandleCtx {}

// An admitted operation can outlive cancellation of its outer retry future.
// Its own input lease must survive until tensor execution and synchronization end.
impl ExecutionContext<'_> {
    async fn local_context(&self) -> Result<Box<dyn HandleCtx>, ThoError> {
        match self {
            Self::Cpu => Ok(Box::new(CpuHandleCtx)),
            Self::Local(actor) => actor.init_handle_ctx().await,
            Self::Remote => Err(ThoError::new("local kernel requires a local execution backend")),
        }
    }

    pub async fn run_stage<S, W>(&self, stage: Arc<S>, input: W, device: usize, ctx: &S::Ctx) -> Result<S::Out, ThoError>
    where
        S: StageExecution,
        S::Out: Send,
        for<'a> S::In<'a>: Sync,
        W: BorrowRef + Debug + Sync,
        for<'a> W::Out<'a>: Borrow<S::In<'a>> + Send,
    {
        // The input future is lazy: a stateful implementation prepares state
        // before materializing inputs, under the same admitted operation.
        let _inputs = input.protect(device);
        stage.execute(BorrowRef::borrow_ref(&input, device), self, ctx).await
    }

    pub async fn run_stage_with_select<S, W, SL>(&self, stage: Arc<S>, input: W, selector: Option<SL>, device: usize, ctx: &S::Ctx) -> Result<S::Out, ThoError>
    where
        S: StageExecution,
        S::Out: Send,
        for<'a> S::In<'a>: Sync,
        SL: ThoSelector + Clone + Debug + Send + 'static,
        W: SelectRef<SL> + Debug + Sync,
        for<'a> W::Out<'a>: Borrow<S::In<'a>> + Send,
    {
        let _inputs = input.protect(device);
        // Keep materialization lazy so state preparation still precedes input borrowing.
        let prepared = async {
            match selector {
                Some(selector) => SelectRef::select_ref(&input, device, selector).await,
                None => BorrowRef::borrow_ref(&input, device).await,
            }
        };
        stage.execute(prepared, self, ctx).await
    }

    pub async fn consume_stage<'i, S, W>(&self, stage: Arc<S>, input: W, device: usize, ctx: S::Ctx) -> Result<ConsumeResult<S::In<'i>, S::Out>, ThoError>
    where
        S: ConsumingExecution,
        S::Out: Send,
        W: ConsumeData<O = S::In<'i>> + Debug,
        S::In<'i>: Send + 'static,
    {
        let _inputs = input.protect(device);
        let (input, ctx) = match input.consume(device, ctx, |input, ctx| ConsumeResult::Ok((input, ctx))).await {
            ConsumeResult::Ok(prepared) => prepared,
            ConsumeResult::Failed { input, error } => return Ok(ConsumeResult::Failed { input, error }),
        };
        stage.execute_consume(input, self, ctx).await
    }
}

impl DeviceExecutionCtx {
    pub async fn run_stage<S, W>(&self, stage: Arc<S>, input: W, device: usize, ctx: &S::Ctx) -> Result<S::Out, ThoError>
    where
        S: StageExecution,
        S::Out: Send,
        for<'a> S::In<'a>: Sync,
        W: BorrowRef + Debug + Sync,
        for<'a> W::Out<'a>: Borrow<S::In<'a>> + Send,
    {
        self.execution_context().run_stage(stage, input, device, ctx).await
    }

    pub async fn run_stage_with_select<S, W, SL>(&self, stage: Arc<S>, input: W, selector: Option<SL>, device: usize, ctx: &S::Ctx) -> Result<S::Out, ThoError>
    where
        S: StageExecution,
        S::Out: Send,
        for<'a> S::In<'a>: Sync,
        SL: ThoSelector + Clone + Debug + Send + 'static,
        W: SelectRef<SL> + Debug + Sync,
        for<'a> W::Out<'a>: Borrow<S::In<'a>> + Send,
    {
        self.execution_context().run_stage_with_select(stage, input, selector, device, ctx).await
    }

    pub async fn consume_stage<'i, S, W>(&self, stage: Arc<S>, input: W, device: usize, ctx: S::Ctx) -> Result<ConsumeResult<S::In<'i>, S::Out>, ThoError>
    where
        S: ConsumingExecution,
        S::Out: Send,
        W: ConsumeData<O = S::In<'i>> + Debug,
        S::In<'i>: Send + 'static,
    {
        self.execution_context().consume_stage(stage, input, device, ctx).await
    }

    pub fn execution_context(&self) -> ExecutionContext<'_> {
        match &self.actor_handle {
            CudaActorHandleEnum::None => ExecutionContext::Cpu,
            CudaActorHandleEnum::Local(actor) => ExecutionContext::Local(actor.as_ref()),
            CudaActorHandleEnum::Ipc(_) => ExecutionContext::Remote,
        }
    }
}

impl<S: SyncStage + Send + Sync> StageExecution for S {
    fn execute<'a, 'i: 'a, I, IFut>(self: Arc<Self>, input: IFut, execution: &'a ExecutionContext<'_>, ctx: &'a Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<Self::Out, ThoError>>
    where
        Self: 'a,
        Self::In<'i>: Sync,
        Self::Out: Send,
        I: Borrow<Self::In<'i>> + Send + 'a,
        IFut: Future<Output = Result<I, ThoError>> + Send + 'a,
    {
        Box::pin(async move {
            let input = input.await?;
            let mut handle = execution.local_context().await?;
            let _guard = handle.sync_ctx()?;
            run_cuda_stage(|| self.do_run(input.borrow(), ctx), || handle.on_after_run())
        })
    }
}

// This helper is shared by concrete stateful stages, without inventing a
// synchronous entry point that could bypass their state lock.
pub async fn execute_stateful<'a, 'i: 'a, S, I, IFut>(stage: Arc<S>, input: IFut, execution: &'a ExecutionContext<'_>, ctx: &'a S::Ctx) -> Result<S::Out, ThoError>
where
    S: StageWithState + Send + Sync + 'a,
    S::In<'i>: Sync,
    S::Out: Send,
    I: Borrow<S::In<'i>> + Send + 'a,
    IFut: Future<Output = Result<I, ThoError>> + Send + 'a,
{
    use thoosa_shared::traits::RunCtx;
    let mut state = stage.pre_run(ctx.tho_device_idx()).await?;
    let input = input.await?;
    let mut handle = execution.local_context().await?;
    let _guard = handle.sync_ctx()?;
    // Keep both state and input alive until synchronization, including unwind.
    run_cuda_stage(|| Arc::clone(&stage).do_run_with_state(input.borrow(), &mut *state, ctx), || handle.on_after_run())
}

impl<S: ConsumingStage + Send + Sync> ConsumingExecution for S {
    fn execute_consume<'a, 'i: 'a>(self: Arc<Self>, input: Self::In<'i>, execution: &'a ExecutionContext<'_>, ctx: Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<ConsumeResult<Self::In<'i>, Self::Out>, ThoError>>
    where
        Self: 'a,
        Self::In<'i>: Send + 'a,
        Self::Out: Send,
        Self::Ctx: 'a,
    {
        Box::pin(async move {
            let mut handle = match execution.local_context().await {
                Ok(handle) => handle,
                Err(error) => return Ok(ConsumeResult::Failed { input, error }),
            };
            let guard = match handle.sync_ctx() {
                Ok(guard) => guard,
                Err(error) => return Ok(ConsumeResult::Failed { input, error }),
            };
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| self.consume(input, ctx)));
            let sync = handle.on_after_run();
            drop(guard);
            match finish_cuda_panic(result, &sync)? {
                ConsumeResult::Ok(output) => finish_cuda_run(Ok(output), sync).map(ConsumeResult::Ok),
                ConsumeResult::Failed { input, error } => Ok(ConsumeResult::Failed { input, error: finish_cuda_error(error, sync) }),
            }
        })
    }
}
