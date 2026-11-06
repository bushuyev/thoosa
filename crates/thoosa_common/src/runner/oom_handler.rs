pub use super::syn_stage_runner::RetryToken;
use crate as thoosa_common;
use crate::runner::common::cachable::{BorrowRef, ConsumeData, SelectRef};
use crate::ws::Workspace;
use mockall_double::double;
use parking_lot::RwLock;
use std::borrow::Borrow;
use thoosa_common::runner::execution::StageExecution;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;
use thoosa_common::runner::sync_stage::ConsumeResult;
use thoosa_shared::collect_futures_eta::{collect_eta_record_oom_all_failed, collect_eta_record_oom_success};
use thoosa_shared::traits::MaxParallelCtx;
#[cfg(test)]
use thoosa_shared::traits::OutId;
use thoosa_shared::traits::Stage;

use crate::runner::statistics::Statistics;
use async_trait::async_trait;
use log::{debug, trace, warn};
use mockall::automock;
use std::fmt::Debug;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_shared::ThoError;
use thoosa_shared::tracked_await::t;
use thoosa_shared::traits::{Adjustable, Data, RunCtx, ThoSelector, WithId};
use thoosa_shared::{os_thread_id, panic_to_err_async};

#[derive(Debug)]
pub enum AdjusterError<A: Adjustable> {
    AdjustFailed(A),
    SomethingElse(ThoError),
}
#[async_trait]
pub trait Adjuster<AC>: Debug + Send + Sync {
    type A: Adjustable;

    async fn adjust(&self, adjustable: Self::A, adjust_context: &mut AC) -> Result<Self::A, AdjusterError<Self::A>>;

    fn degrade(&self, _adjust_context: &mut AC) -> bool {
        false
    }
}

#[allow(async_fn_in_trait)]
pub trait InputContext<'a, R>: Debug + Send + Sync
where
    R: Stage + 'a,
{
    async fn add_partial_output(&mut self, out: R::Out) -> Result<(), ThoError>;
    fn all_done(&self) -> bool;
    async fn build_final_output(self) -> Result<R::Out, ThoError>;
}

#[cfg(test)]
pub mod mock_input_context {
    use crate::runner::oom_handler::mock_runnable::MockRunnable;
    use crate::runner::oom_handler::test_types::TestOutput;
    use mockall::mock;
    use thoosa_shared::ThoError;

    mock! {
        #[derive(Debug)]
        pub InputContext {

        }
        impl super::InputContext<'static, MockRunnable> for InputContext{
            async fn add_partial_output(&mut self, out: TestOutput) -> Result<(), ThoError>;
            fn all_done(&self) -> bool;
            // fn input_len(&self) -> &[usize; 1];
            async fn build_final_output(self) -> Result<TestOutput, ThoError>;
        }
    }

    pub struct InputContextBuilder {
        mock: MockInputContext,
    }

    impl InputContextBuilder {
        pub fn with_partial_output(mut self, inp: TestOutput) -> Self {
            self.mock.expect_add_partial_output().returning(|_| Ok(())).call(inp).expect("mock partial output");

            self
        }

        pub fn with_build_final_output(mut self, inp: TestOutput) -> Self {
            self.mock.expect_build_final_output().return_once(move || Ok(inp));
            self
        }

        pub fn with_all_done(mut self, out: &[bool]) -> Self {
            for v in out {
                self.mock.expect_all_done().return_const(*v);
            }

            self
        }

        pub fn finish(self) -> MockInputContext {
            self.mock
        }
    }

    pub fn build() -> InputContextBuilder {
        InputContextBuilder { mock: MockInputContext::new() }
    }
}

#[automock]
pub trait RunnableContext: Debug {}

#[automock]
pub trait RunCtxContext: Debug {}

pub trait HasInputContext<IC> {
    fn input_context(&mut self) -> &mut IC;
    fn into_input_context(self) -> IC;
}

pub trait HasRunnableContext<RC> {
    fn runnable_context(&mut self) -> &mut RC;
}

pub trait HasRunCtxContext<RCC> {
    fn run_ctx_context(&mut self) -> &mut RCC;
}

#[derive(Debug)]
pub struct NoContexts;

#[derive(Debug)]
pub struct WithInputContext<IC, Rest = NoContexts> {
    input: IC,
    rest: Rest,
}

impl<IC> WithInputContext<IC> {
    pub fn new(input: IC) -> Self {
        Self { input, rest: NoContexts }
    }
}

impl<IC, Rest> WithInputContext<IC, Rest> {
    pub fn with_rest(input: IC, rest: Rest) -> Self {
        Self { input, rest }
    }
}

#[derive(Debug)]
pub struct WithRunnableContext<RC, Rest = NoContexts> {
    runnable: RC,
    rest: Rest,
}

impl<RC> WithRunnableContext<RC> {
    pub fn new(runnable: RC) -> Self {
        Self { runnable, rest: NoContexts }
    }
}

impl<RC, Rest> WithRunnableContext<RC, Rest> {
    pub fn with_rest(runnable: RC, rest: Rest) -> Self {
        Self { runnable, rest }
    }
}

#[derive(Debug)]
pub struct WithRunCtxContext<RCC, Rest = NoContexts> {
    run_ctx: RCC,
    rest: Rest,
}

impl<RCC> WithRunCtxContext<RCC> {
    pub fn new(run_ctx: RCC) -> Self {
        Self { run_ctx, rest: NoContexts }
    }
}

impl<RCC, Rest> WithRunCtxContext<RCC, Rest> {
    pub fn with_rest(run_ctx: RCC, rest: Rest) -> Self {
        Self { run_ctx, rest }
    }
}

#[derive(Debug)]
pub struct Contexts<'a, R, IC, RC, RCC>
where
    R: Stage + 'a,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
{
    input: IC,
    runnable: RC,
    run_ctx: RCC,
    pd: PhantomData<&'a R>,
}

impl<'a, R, IC, RC, RCC> Contexts<'a, R, IC, RC, RCC>
where
    R: Stage + 'a,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
{
    pub fn new(input: IC, runnable: RC, run_ctx: RCC) -> Self {
        Self { input, runnable, run_ctx, pd: PhantomData }
    }
}

#[derive(Debug)]
pub struct NoopInputContext<R: Stage + 'static>(pub Option<R::Out>);

impl<'a, R> InputContext<'a, R> for NoopInputContext<R>
where
    <R as Stage>::In<'a>: Data,
    <R as Stage>::Out: Send + Sync,
    R: Stage + 'a,
{
    async fn add_partial_output(&mut self, o: R::Out) -> Result<(), ThoError> {
        self.0.replace(o);
        Ok(())
    }

    fn all_done(&self) -> bool {
        self.0.is_some()
    }

    async fn build_final_output(self) -> Result<R::Out, ThoError> {
        self.0.ok_or(ThoError::new("no final output"))
    }
}

#[derive(Debug)]
pub struct NoopRunnableContext {}
impl RunnableContext for NoopRunnableContext {}

#[derive(Debug)]
pub struct NoopRunCtxContext {}

impl RunCtxContext for NoopRunCtxContext {}

#[derive(Debug)]
pub enum NoInputAdjuster {}

#[derive(Debug)]
pub enum NoStageAdjuster {}

#[derive(Debug)]
pub enum NoRunCtxAdjuster {}

#[derive(Debug)]
pub enum Adjusters<IA = NoInputAdjuster, SA = NoStageAdjuster, CA = NoRunCtxAdjuster> {
    Input(IA), //it adjusts Selector used to fetch chunk of input
    Stage(SA),
    RunContext(CA),
}

pub fn input_adjuster<'a, IC, S, A>(adjuster: A) -> Box<dyn Adjuster<IC, A = S> + 'a>
where
    S: Adjustable,
    A: Adjuster<IC, A = S> + 'a,
{
    Box::new(adjuster)
}

pub fn task_adjuster<'a, R, RC, A>(adjuster: A) -> Box<dyn Adjuster<RC, A = Arc<R>> + 'a>
where
    R: Adjustable,
    A: Adjuster<RC, A = Arc<R>> + 'a,
{
    Box::new(adjuster)
}

pub fn run_ctx_adjuster<'a, Ctx, RCC, A>(adjuster: A) -> Box<dyn Adjuster<RCC, A = Ctx> + 'a>
where
    Ctx: Adjustable,
    A: Adjuster<RCC, A = Ctx> + 'a,
{
    Box::new(adjuster)
}

/// Constructs one concrete sync-stage wrapper from retryable initialization data.
///
/// The workspace is passed by `Arc` because construction may run in the same dynamic executor
/// used for synchronous CUDA work and therefore must own everything held across the await.
#[async_trait]
pub trait SyncStageInit<RC, St, W: Workspace + ?Sized = dyn Workspace>: Clone + Debug + Send + Sync + 'static
where
    RC: RunCtx,
    // Construction must also support consuming-only stages; it does not execute a kernel.
    St: Stage + Send + Sync + 'static,
{
    async fn init(self, run_context: RC, ws: Arc<RwLock<Box<W>>>) -> Result<Arc<St>, ThoError>;
}

#[doc(hidden)]
pub enum InputAdjustedRun<C, O> {
    Done(O),
    Continue(C),
}

#[doc(hidden)]
#[allow(async_fn_in_trait)]
pub trait InputAdjustment<'a, R, C, S>: Debug
where
    R: Stage + 'a,
{
    async fn adjust_input(&self, input_select_adjusted: &mut Option<S>, contexts: &mut C) -> Result<(), ThoError>;
    fn degrade_input(&self, contexts: &mut C) -> bool;
    async fn handle_success(out: R::Out, contexts: C) -> Result<InputAdjustedRun<C, R::Out>, ThoError>;
}

impl<'a, R, C, S> InputAdjustment<'a, R, C, S> for NoInputAdjuster
where
    R: Stage + 'a,
{
    async fn adjust_input(&self, _input_select_adjusted: &mut Option<S>, _contexts: &mut C) -> Result<(), ThoError> {
        match *self {}
    }

    fn degrade_input(&self, _contexts: &mut C) -> bool {
        match *self {}
    }

    async fn handle_success(out: R::Out, contexts: C) -> Result<InputAdjustedRun<C, R::Out>, ThoError> {
        let _ = contexts;
        Ok(InputAdjustedRun::Done(out))
    }
}

impl<'stage, 'adjuster, R, C, IC, S> InputAdjustment<'stage, R, C, S> for Box<dyn Adjuster<IC, A = S> + 'adjuster>
where
    R: Stage + 'stage,
    IC: InputContext<'stage, R>,
    C: HasInputContext<IC>,
    S: Adjustable + Debug + Clone + Default + 'static,
{
    async fn adjust_input(&self, input_select_adjusted: &mut Option<S>, contexts: &mut C) -> Result<(), ThoError> {
        let base_selector = input_select_adjusted.clone().unwrap_or_default();

        match t!(self.adjust(base_selector, contexts.input_context())).await {
            Ok(v) => {
                debug!("InputAdjustment::adjust_input: input adjuster ok: {v:?}");
                *input_select_adjusted = Some(v);
                Ok(())
            }
            Err(adjuster_err) => {
                let msg = format!("adjuster failed: {adjuster_err:?}");
                debug!("InputAdjustment::adjust_input: input adjuster failed: {msg:?}");

                match adjuster_err {
                    AdjusterError::AdjustFailed(v) => {
                        *input_select_adjusted = Some(v);
                        Err(ThoError::new_with_cause(msg, ThoError::AdjusterFailed))
                    }
                    AdjusterError::SomethingElse(error) => Err(error.context("input adjuster failed")),
                }
            }
        }
    }

    fn degrade_input(&self, contexts: &mut C) -> bool {
        self.degrade(contexts.input_context())
    }

    async fn handle_success(out: R::Out, mut contexts: C) -> Result<InputAdjustedRun<C, R::Out>, ThoError> {
        // Assembly failures retain their recovery category and must stop this split.
        t!(contexts.input_context().add_partial_output(out)).await?;

        let all_done = contexts.input_context().all_done();

        if all_done {
            Ok(InputAdjustedRun::Done(t!(contexts.into_input_context().build_final_output()).await?))
        } else {
            Ok(InputAdjustedRun::Continue(contexts))
        }
    }
}

#[doc(hidden)]
#[allow(async_fn_in_trait)]
pub trait StageAdjustment<'a, R, C>: Debug
where
    R: Stage + 'a,
{
    async fn adjust_task(&self, task: &mut Option<Arc<R>>, contexts: &mut C) -> Result<(), ThoError>;
    fn degrade_task(&self, contexts: &mut C) -> bool;
}

impl<'a, R, C> StageAdjustment<'a, R, C> for NoStageAdjuster
where
    R: Stage + 'a,
{
    async fn adjust_task(&self, _task: &mut Option<Arc<R>>, _contexts: &mut C) -> Result<(), ThoError> {
        match *self {}
    }

    fn degrade_task(&self, _contexts: &mut C) -> bool {
        match *self {}
    }
}

impl<'stage, 'adjuster, R, C, RC> StageAdjustment<'stage, R, C> for Box<dyn Adjuster<RC, A = Arc<R>> + 'adjuster>
where
    R: Stage + Adjustable + 'stage,
    RC: RunnableContext,
    C: HasRunnableContext<RC>,
{
    async fn adjust_task(&self, task: &mut Option<Arc<R>>, contexts: &mut C) -> Result<(), ThoError> {
        match t!(self.adjust(task.take().expect("adjust_task: missing task"), contexts.runnable_context())).await {
            Ok(v) => {
                debug!("OOMHandler::run, error, task adjuster ok: {v:?}");
                *task = Some(v);
                Ok(())
            }
            Err(adjuster_err) => {
                let msg = format!("adjuster failed: {adjuster_err:?}");
                debug!("OOMHandler::run, error, task adjuster failed: {msg:?}");

                match adjuster_err {
                    AdjusterError::AdjustFailed(v) => {
                        *task = Some(v);
                        Err(ThoError::new_with_cause(msg, ThoError::AdjusterFailed))
                    }
                    AdjusterError::SomethingElse(error) => Err(error.context("task adjuster failed")),
                }
            }
        }
    }

    fn degrade_task(&self, contexts: &mut C) -> bool {
        self.degrade(contexts.runnable_context())
    }
}

#[doc(hidden)]
#[allow(async_fn_in_trait)]
pub trait RunContextAdjustment<Ctx, C>: Debug {
    async fn adjust_run_ctx(&self, run_context: &mut Option<Ctx>, contexts: &mut C) -> Result<(), ThoError>;
    fn degrade_run_ctx(&self, contexts: &mut C) -> bool;
}

impl<Ctx, C> RunContextAdjustment<Ctx, C> for NoRunCtxAdjuster {
    async fn adjust_run_ctx(&self, _run_context: &mut Option<Ctx>, _contexts: &mut C) -> Result<(), ThoError> {
        match *self {}
    }

    fn degrade_run_ctx(&self, _contexts: &mut C) -> bool {
        match *self {}
    }
}

impl<'adjuster, Ctx, C, RCC> RunContextAdjustment<Ctx, C> for Box<dyn Adjuster<RCC, A = Ctx> + 'adjuster>
where
    Ctx: Adjustable + Debug,
    RCC: RunCtxContext,
    C: HasRunCtxContext<RCC>,
{
    async fn adjust_run_ctx(&self, run_context: &mut Option<Ctx>, contexts: &mut C) -> Result<(), ThoError> {
        match t!(self.adjust(run_context.take().expect("adjust_run_ctx: missing run context"), contexts.run_ctx_context())).await {
            Ok(v) => {
                debug!("OOMHandler::run, error, run context adjuster ok: {v:?}");
                *run_context = Some(v);
                Ok(())
            }
            Err(adjuster_err) => {
                let msg = format!("adjuster failed: {adjuster_err:?}");
                debug!("OOMHandler::run, error, run context adjuster failed: {msg:?}");

                match adjuster_err {
                    AdjusterError::AdjustFailed(v) => {
                        *run_context = Some(v);
                        Err(ThoError::new_with_cause(msg, ThoError::AdjusterFailed))
                    }
                    AdjusterError::SomethingElse(error) => Err(error.context("run context adjuster failed")),
                }
            }
        }
    }

    fn degrade_run_ctx(&self, contexts: &mut C) -> bool {
        self.degrade(contexts.run_ctx_context())
    }
}

impl<IA, TA, CA> Adjusters<IA, TA, CA> {
    async fn apply<'a, R, C, S>(&self, task: &mut Option<Arc<R>>, run_context: &mut Option<R::Ctx>, input_select_adjusted: &mut Option<S>, contexts: &mut C) -> Result<(), ThoError>
    where
        R: Stage + 'a,
        IA: InputAdjustment<'a, R, C, S>,
        TA: StageAdjustment<'a, R, C>,
        CA: RunContextAdjustment<R::Ctx, C>,
    {
        match self {
            Adjusters::Input(v) => t!(v.adjust_input(input_select_adjusted, contexts)).await,
            Adjusters::Stage(v) => t!(v.adjust_task(task, contexts)).await,
            Adjusters::RunContext(v) => t!(v.adjust_run_ctx(run_context, contexts)).await,
        }
    }

    fn degrade<'a, R, C, S>(&self, contexts: &mut C) -> bool
    where
        R: Stage + 'a,
        IA: InputAdjustment<'a, R, C, S>,
        TA: StageAdjustment<'a, R, C>,
        CA: RunContextAdjustment<R::Ctx, C>,
    {
        match self {
            Adjusters::Input(v) => v.degrade_input(contexts),
            Adjusters::Stage(v) => v.degrade_task(contexts),
            Adjusters::RunContext(v) => v.degrade_run_ctx(contexts),
        }
    }
}

impl<'a, R, IC, RC, RCC> HasInputContext<IC> for Contexts<'a, R, IC, RC, RCC>
where
    R: Stage + 'a,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
{
    fn input_context(&mut self) -> &mut IC {
        &mut self.input
    }

    fn into_input_context(self) -> IC {
        self.input
    }
}

impl<'a, R, IC, RC, RCC> HasRunnableContext<RC> for Contexts<'a, R, IC, RC, RCC>
where
    R: Stage + 'a,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
{
    fn runnable_context(&mut self) -> &mut RC {
        &mut self.runnable
    }
}

impl<'a, R, IC, RC, RCC> HasRunCtxContext<RCC> for Contexts<'a, R, IC, RC, RCC>
where
    R: Stage + 'a,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
{
    fn run_ctx_context(&mut self) -> &mut RCC {
        &mut self.run_ctx
    }
}

impl<IC, Rest> HasInputContext<IC> for WithInputContext<IC, Rest> {
    fn input_context(&mut self) -> &mut IC {
        &mut self.input
    }

    fn into_input_context(self) -> IC {
        self.input
    }
}

impl<IC, RC, Rest> HasInputContext<IC> for WithRunnableContext<RC, Rest>
where
    Rest: HasInputContext<IC>,
{
    fn input_context(&mut self) -> &mut IC {
        self.rest.input_context()
    }

    fn into_input_context(self) -> IC {
        self.rest.into_input_context()
    }
}

impl<IC, RCC, Rest> HasInputContext<IC> for WithRunCtxContext<RCC, Rest>
where
    Rest: HasInputContext<IC>,
{
    fn input_context(&mut self) -> &mut IC {
        self.rest.input_context()
    }

    fn into_input_context(self) -> IC {
        self.rest.into_input_context()
    }
}

impl<RC, Rest> HasRunnableContext<RC> for WithRunnableContext<RC, Rest> {
    fn runnable_context(&mut self) -> &mut RC {
        &mut self.runnable
    }
}

impl<RC, IC, Rest> HasRunnableContext<RC> for WithInputContext<IC, Rest>
where
    Rest: HasRunnableContext<RC>,
{
    fn runnable_context(&mut self) -> &mut RC {
        self.rest.runnable_context()
    }
}

impl<RC, RCC, Rest> HasRunnableContext<RC> for WithRunCtxContext<RCC, Rest>
where
    Rest: HasRunnableContext<RC>,
{
    fn runnable_context(&mut self) -> &mut RC {
        self.rest.runnable_context()
    }
}

impl<RCC, Rest> HasRunCtxContext<RCC> for WithRunCtxContext<RCC, Rest> {
    fn run_ctx_context(&mut self) -> &mut RCC {
        &mut self.run_ctx
    }
}

impl<RCC, IC, Rest> HasRunCtxContext<RCC> for WithInputContext<IC, Rest>
where
    Rest: HasRunCtxContext<RCC>,
{
    fn run_ctx_context(&mut self) -> &mut RCC {
        self.rest.run_ctx_context()
    }
}

impl<RCC, RC, Rest> HasRunCtxContext<RCC> for WithRunnableContext<RC, Rest>
where
    Rest: HasRunCtxContext<RCC>,
{
    fn run_ctx_context(&mut self) -> &mut RCC {
        self.rest.run_ctx_context()
    }
}

#[derive(Debug)]
pub struct OOMHandler<'a, 'z, IA = NoInputAdjuster, TA = NoStageAdjuster, CA = NoRunCtxAdjuster, S = ()> {
    adjusters: Vec<Adjusters<IA, TA, CA>>,
    #[allow(dead_code)]
    statistics: Option<&'a Arc<Statistics>>,
    pd_rc: PhantomData<(&'z (), S)>,
}

// These private adapter types let stage construction reuse the established OOM retry loop without
// pretending that an input or an already-created task can be adjusted before construction succeeds.
#[derive(Debug)]
struct SyncStageInitInput;

impl thoosa_shared::applicables_runner::Input for SyncStageInitInput {}

#[derive(Debug)]
struct SyncStageInitOutput<St>(Arc<St>);

impl<St> thoosa_shared::applicables_runner::Output for SyncStageInitOutput<St> where St: Debug {}

struct SyncStageInitImpl<I, RC, St, W: ?Sized> {
    init: I,
    ws: Arc<RwLock<Box<W>>>,
    pd: PhantomData<fn(RC) -> St>,
}

impl<I, RC, St, W: ?Sized> Debug for SyncStageInitImpl<I, RC, St, W>
where
    I: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyncStageInitTask").field("init", &self.init).finish_non_exhaustive()
    }
}

impl<I, RC, St, W> Stage for SyncStageInitImpl<I, RC, St, W>
where
    I: Debug,
    RC: RunCtx,
    St: Stage + Send + Sync + 'static,
    W: Workspace + ?Sized,
{
    type Ctx = RC;
    type In<'a> = SyncStageInitInput;
    type Out = SyncStageInitOutput<St>;
}

fn oom_executor(tho_device_idx: usize) -> &'static crate::runner::syn_stage_runner::SyncStageRunner {
    GlobalCtx::global().get_runner(tho_device_idx)
}

async fn run_stage_borrowed<R, W>(task: Arc<R>, input: Arc<W>, tho_device_idx: usize, run_ctx: R::Ctx, retry_token: Option<RetryToken>) -> Result<R::Out, ThoError>
where
    R: StageExecution + Send + Sync + 'static,
    R::Ctx: MaxParallelCtx + Send + Sync + 'static,
    for<'i> <R as Stage>::In<'i>: Send + Sync + Debug,
    R::Out: Send + Sync + 'static,
    W: BorrowRef + Debug + Send + Sync + 'static,
    for<'f> <W as BorrowRef>::Out<'f>: Borrow<<R as Stage>::In<'f>> + Send + Debug,
{
    if tho_device_idx == 0 {
        return crate::runner::execution::ExecutionContext::Cpu.run_stage(task, input, 0, &run_ctx).await;
    }

    t!(oom_executor(tho_device_idx).submit(
        run_ctx.max_other_parallel().into(),
        async move {
            let execution = GlobalCtx::global().execution(tho_device_idx);
            execution.run_stage(task, input, tho_device_idx, &run_ctx).await
        },
        retry_token
    ))
    .await
}

async fn consume_stage_owned<'z, R, W>(task: Arc<R>, input: W, tho_device_idx: usize, run_ctx: R::Ctx, retry: Option<RetryToken>) -> Result<ConsumeResult<R::In<'z>, R::Out>, ThoError>
where
    R: crate::runner::execution::ConsumingExecution + Send + Sync + 'static,
    R::Ctx: MaxParallelCtx + Send + Sync + 'static,
    for<'i> <R as Stage>::In<'i>: Send + Sync,
    R::Out: Send + Sync + 'static + Debug,
    W: ConsumeData<O = R::In<'z>> + Debug + Send + Sync + 'static,
    R::In<'z>: 'static,
{
    debug!("consume_stage_owned: task={task:?} input={input:?} tho_device_idx={tho_device_idx}");

    if tho_device_idx == 0 {
        return crate::runner::execution::ExecutionContext::Cpu.consume_stage(task, input, 0, run_ctx).await;
    }

    t!(oom_executor(tho_device_idx).submit_consume(
        run_ctx.max_other_parallel().into(),
        async move {
            let execution = GlobalCtx::global().execution(tho_device_idx);
            // A sync failure after consumption has no recoverable input; let
            // submit_consume classify both that outer error and owned failures.
            execution.consume_stage(task, input, tho_device_idx, run_ctx).await
        },
        retry,
    ))
    .await
}

async fn run_stage_selected<R, W, S>(task: Arc<R>, input: Arc<W>, selector: Option<S>, tho_device_idx: usize, run_ctx: R::Ctx, retry: Option<RetryToken>) -> Result<R::Out, ThoError>
where
    R: StageExecution + Send + Sync + 'static,
    R::Ctx: MaxParallelCtx + Send + Sync + 'static,
    for<'i> <R as Stage>::In<'i>: Send + Sync + Debug,
    R::Out: Send + Sync + 'static,
    S: ThoSelector + Debug + Clone + Default + Send + 'static,
    W: SelectRef<S> + Debug + Send + Sync + 'static,
    for<'f> <W as BorrowRef>::Out<'f>: Borrow<<R as Stage>::In<'f>> + Send + Debug,
{
    if tho_device_idx == 0 {
        return crate::runner::execution::ExecutionContext::Cpu.run_stage_with_select(task, input, selector, 0, &run_ctx).await;
    }

    t!(oom_executor(tho_device_idx).submit(
        run_ctx.max_other_parallel().into(),
        async move {
            let execution = GlobalCtx::global().execution(tho_device_idx);
            execution.run_stage_with_select(task, input, selector, tho_device_idx, &run_ctx).await
        },
        retry
    ))
    .await
}

#[doc(hidden)]
#[allow(async_fn_in_trait)]
pub trait InputRun<R, W, S>: Debug
where
    R: StageExecution,
{
    async fn run_stage(task: Arc<R>, input: Arc<W>, selector: Option<S>, tho_device_idx: usize, run_ctx: R::Ctx, retry: Option<RetryToken>) -> Result<R::Out, ThoError>;
}

impl<R, W, S> InputRun<R, W, S> for NoInputAdjuster
where
    R: StageExecution + Send + Sync + 'static,
    R::Ctx: MaxParallelCtx + Send + Sync + 'static,
    for<'i> <R as Stage>::In<'i>: Send + Sync + Debug,
    R::Out: Send + Sync + 'static,
    W: BorrowRef + Debug + Send + Sync + 'static,
    for<'f> <W as BorrowRef>::Out<'f>: Borrow<<R as Stage>::In<'f>> + Send + Debug,
{
    async fn run_stage(task: Arc<R>, input: Arc<W>, selector: Option<S>, tho_device_idx: usize, run_ctx: R::Ctx, retry: Option<RetryToken>) -> Result<R::Out, ThoError> {
        let _ = selector;
        t!(run_stage_borrowed(task, input, tho_device_idx, run_ctx, retry)).await
    }
}

impl<'adjuster, R, W, S, IC> InputRun<R, W, S> for Box<dyn Adjuster<IC, A = S> + 'adjuster>
where
    R: StageExecution + Send + Sync + 'static,
    R::Ctx: MaxParallelCtx + Send + Sync + 'static,
    for<'i> <R as Stage>::In<'i>: Send + Sync + Debug,
    R::Out: Send + Sync + 'static,
    S: ThoSelector + Debug + Clone + Default + Send + 'static,
    W: SelectRef<S> + Debug + Send + Sync + 'static,
    for<'f> <W as BorrowRef>::Out<'f>: Borrow<<R as Stage>::In<'f>> + Send + Debug,
{
    async fn run_stage(task: Arc<R>, input: Arc<W>, selector: Option<S>, tho_device_idx: usize, run_ctx: R::Ctx, retry: Option<RetryToken>) -> Result<R::Out, ThoError> {
        t!(run_stage_selected(task, input, selector, tho_device_idx, run_ctx, retry)).await
    }
}

impl<'a, 'z, IA, TA, CA, S> OOMHandler<'a, 'z, IA, TA, CA, S>
where
    IA: Debug,
    TA: Debug,
    CA: Debug,
{
    pub fn new(adjusters: Vec<Adjusters<IA, TA, CA>>, statistics: Option<&'a Arc<Statistics>>) -> Self {
        Self { adjusters, statistics, pd_rc: Default::default() }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "OOMHandler::run"))]
    pub async fn run<R, W, C>(&self, task: &Arc<R>, input: Arc<W>, run_context: R::Ctx, contexts: C) -> Result<R::Out, ThoError>
    where
        R: StageExecution + Send + Sync + 'static + 'a,
        R::Ctx: MaxParallelCtx + Send + Sync + 'static,
        IA: InputAdjustment<'a, R, C, S>,
        IA: InputRun<R, W, S>,
        TA: StageAdjustment<'a, R, C>,
        CA: RunContextAdjustment<R::Ctx, C>,
        C: Debug,
        for<'i> <R as Stage>::In<'i>: Send + Sync + Debug,
        R::Out: Send + Sync + 'static,
        S: Debug + Clone + Default + Send + 'static,
        W: BorrowRef + Debug + Send + Sync + 'static,
        for<'f> <W as BorrowRef>::Out<'f>: Borrow<<R as Stage>::In<'f>> + Send + Debug,
    {
        // Retain the input's owned exclusions through every retry and adjustment.
        // Cache aliases, removed cache entries, and temporary inputs behave alike.
        let _inputs = input.protect(run_context.tho_device_idx());
        if run_context.tho_device_idx() == 0 {
            return t!(<IA as InputRun<R, W, S>>::run_stage(Arc::clone(task), input, None, 0, run_context, None)).await;
        }

        let _tho_device_idx = run_context.tho_device_idx();

        t!(run_with_runner(&self.adjusters, Arc::clone(task), input, run_context, contexts, |task, input, selector, tho_device_idx, run_ctx, retry| async move {
            debug!("OOMHandler::run: task={task:?} input={input:?} run_ctx={run_ctx:?} max_other_parallel={}", run_ctx.max_other_parallel());

            /*
            SyncStage::do_run – executes operations on tensors synchronously, receives tensors or composite tensor types as references
            OOMHandler::run - runs syn_stage.do_run/consume until it succeeds decreasing allowed number of stages executed in parallel and applying stage- and  stage input-specific adjusters as configured at call site
            StageExecutor::submit – ensures that no more than requested max_other_parallel sync stages are being executed at a given moment.
            selector – when OOMHandler tries to process input in chunks the selector is used to select current chunk.
            Then – input.with_data_ref/with_selected_data_ref calls
                        load_from_any_device_→ensure_copied→
                        if tensor is absent in current (from RunContext)  memory it’s copied with
                        ThoTensor::copy_to_device-> GlobalCtx::global().cuda_actor(ctx_device)/copy→copy_h2d_from_ptr or ipc_copy_h2d_from_ptr

             */
            t!(<IA as InputRun<R, W, S>>::run_stage(task, input, selector, tho_device_idx, run_ctx, retry)).await
        },))
        .await
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "OOMHandler::consume"))]
    pub async fn consume<R, W, C>(&self, task: &Arc<R>, input: W, run_context: R::Ctx, contexts: C) -> Result<R::Out, ThoError>
    where
        R: crate::runner::execution::ConsumingExecution + Send + Sync + 'static + 'a,
        R::Ctx: MaxParallelCtx + Send + Sync + 'static,
        IA: InputAdjustment<'a, R, C, S>,
        TA: StageAdjustment<'a, R, C>,
        CA: RunContextAdjustment<R::Ctx, C>,
        C: Debug,
        for<'i> <R as Stage>::In<'i>: Send + Sync,
        R::Out: Send + Sync + 'static,
        S: Debug + Clone + Default + Send + 'static,
        W: ConsumeData<O = <R as Stage>::In<'z>>,
        W: Debug + Send + Sync + 'static,
        <R as Stage>::In<'z>: 'static,
    {
        t!(run_with_runner_consume(
            &self.adjusters,
            Arc::clone(task),
            input,
            run_context,
            contexts,
            |task: Arc<R>, input: W, _selector: Option<S>, tho_device_idx: usize, run_ctx: R::Ctx, retry| async move {
                debug!("OOMHandler::consume: task={task:?} run_ctx={run_ctx:?} max_other_parallel={}", run_ctx.max_other_parallel());
                t!(consume_stage_owned(task, input, tho_device_idx, run_ctx, retry)).await
            },
        ))
        .await
    }
}

impl<'a, 'z, CA> OOMHandler<'a, 'z, NoInputAdjuster, NoStageAdjuster, CA, ()>
where
    CA: Debug,
{
    /// Constructs a sync stage with the same run-context adjustment and retry machinery used by execution.
    /// Input and task adjusters are intentionally unavailable because neither exists before construction succeeds.
    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "OOMHandler::new_stage"))]
    pub async fn new_stage<I, RC, St, C, W>(&self, sync_init: I, run_context: RC, contexts: C, ws: &Arc<RwLock<Box<W>>>) -> Result<Arc<St>, ThoError>
    where
        I: SyncStageInit<RC, St, W>,
        RC: RunCtx + MaxParallelCtx + Send + Sync + 'static,
        St: Stage + Send + Sync + 'static,
        CA: RunContextAdjustment<RC, C>,
        C: Debug,
        W: Workspace + ?Sized + 'static,
    {
        if run_context.tho_device_idx() == 0 {
            // Match do_run behavior: CPU construction does not enter CUDA OOM retry handling.
            return panic_to_err_async!(sync_init.init(run_context, Arc::clone(ws)));
        }
        #[cfg(tho_debug)]
        let input_id_dbg = format!("sync_init={:?}", sync_init);

        let stage = Arc::new(SyncStageInitImpl { init: sync_init, ws: Arc::clone(ws), pd: PhantomData });

        let output = t!(run_with_runner_impl(
            &self.adjusters,
            stage,
            #[cfg(tho_debug)]
            input_id_dbg,
            run_context,
            contexts,
            SyncStageInitRetryAdapter,
        ))
        .await?;
        Ok(output.0)
    }
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf = "pref_oom_count"))]
fn perf_oom_count() {}

#[allow(async_fn_in_trait)]
trait RunnerRetryAdapter<'z, R, S>
where
    R: Stage,
{
    async fn run(&mut self, task: Arc<R>, selector: Option<S>, device_idx: usize, run_ctx: R::Ctx, retry: Option<RetryToken>) -> Result<R::Out, ThoError>;

    fn can_retry(&self) -> bool {
        true
    }
}

struct SyncStageInitRetryAdapter;

impl<'z, I, RC, St, W> RunnerRetryAdapter<'z, SyncStageInitImpl<I, RC, St, W>, ()> for SyncStageInitRetryAdapter
where
    I: SyncStageInit<RC, St, W>,
    RC: RunCtx + MaxParallelCtx + Send + Sync + 'static,
    St: Stage + Send + Sync + 'static,
    W: Workspace + ?Sized + 'static,
{
    async fn run(&mut self, task: Arc<SyncStageInitImpl<I, RC, St, W>>, _selector: Option<()>, device_idx: usize, run_ctx: RC, retry: Option<RetryToken>) -> Result<SyncStageInitOutput<St>, ThoError> {
        let init = task.init.clone();
        let ws = Arc::clone(&task.ws);
        let max_other_parallel = run_ctx.max_other_parallel();

        // Constructor work enters the same executor as do_run/consume so OOM retries retain job identity and can become exclusive.
        t!(oom_executor(device_idx).submit(
            max_other_parallel.into(),
            async move {
                let stage = panic_to_err_async!(init.init(run_ctx, ws))?;
                Ok(SyncStageInitOutput(stage))
            },
            retry,
        ))
        .await
    }
}

struct SharedRunnerRetryAdapter<W, Run> {
    input: Arc<W>,
    runner: Run,
}

impl<'z, R, S, W, Run, Fut> RunnerRetryAdapter<'z, R, S> for SharedRunnerRetryAdapter<W, Run>
where
    R: Stage,
    Run: Fn(Arc<R>, Arc<W>, Option<S>, usize, R::Ctx, Option<RetryToken>) -> Fut,
    Fut: Future<Output = Result<R::Out, ThoError>>,
{
    async fn run(&mut self, task: Arc<R>, selector: Option<S>, device_idx: usize, run_ctx: R::Ctx, retry: Option<RetryToken>) -> Result<R::Out, ThoError> {
        let runner = &self.runner;

        panic_to_err_async!(runner(task, Arc::clone(&self.input), selector, device_idx, run_ctx, retry))
    }
}

struct ConsumeRunnerRetryAdapter<W, Run> {
    input: Option<W>,
    leases: crate::runner::residency::ResidencyLease,
    runner: Run,
}

impl<'z, R, S, W, Run, Fut> RunnerRetryAdapter<'z, R, S> for ConsumeRunnerRetryAdapter<W, Run>
where
    R: Stage,
    W: ConsumeData<O = R::In<'z>>,
    Run: Fn(Arc<R>, W, Option<S>, usize, R::Ctx, Option<RetryToken>) -> Fut,
    Fut: Future<Output = Result<ConsumeResult<R::In<'z>, R::Out>, ThoError>>,
{
    async fn run(&mut self, task: Arc<R>, selector: Option<S>, device_idx: usize, run_ctx: R::Ctx, retry: Option<RetryToken>) -> Result<R::Out, ThoError> {
        let input = self.input.take().expect("ConsumeRunnerRetryAdapter::run: missing input");
        let runner = &self.runner;

        let run_result = match t!(futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe({
            //TODO use panic_to_err_async?
            runner(task, input, selector, device_idx, run_ctx, retry)
        })))
        .await
        {
            Ok(run_result) => run_result?,
            Err(panic) => std::panic::resume_unwind(panic),
        };

        match run_result {
            ConsumeResult::Ok(output) => Ok(output),
            ConsumeResult::Failed { input, error } => {
                let restored = t!(W::burp(input)).await;
                // Consumption rebuilds registrations. Reacquire from the new
                // handles before the next OOM adjustment can offload anything.
                self.leases = restored.protect(device_idx);
                self.input = Some(restored);
                Err(error)
            }
        }
    }

    fn can_retry(&self) -> bool {
        // A synchronization error after successful consumption has no input to replay.
        self.input.is_some()
    }
}

async fn run_with_runner_impl<'a, 'z, R, S, A, C, IA, TA, CA>(adjusters: &[Adjusters<IA, TA, CA>], stage: Arc<R>, #[cfg(tho_debug)] input_id_dbg: String, run_context: R::Ctx, mut contexts: C, mut runner: A) -> Result<R::Out, ThoError>
where
    R: Stage + 'a,
    S: Debug + Clone + Default + 'static,
    C: Debug,
    IA: InputAdjustment<'a, R, C, S>,
    TA: StageAdjustment<'a, R, C>,
    CA: RunContextAdjustment<R::Ctx, C>,
    for<'i> <R as Stage>::In<'i>: Send + Sync + Debug,
    R::Out: Send + Sync + 'static,
    R::Ctx: MaxParallelCtx,
    A: RunnerRetryAdapter<'z, R, S>,
{
    let (device_idx, ctx_id) = (run_context.tho_device_idx(), run_context.id().clone());
    // Ownership starts before the first attempt and survives every adapter/adjuster
    // await, so cancellation never leaves a detached job owning an exclusive retry.
    let retry_scope = oom_executor(device_idx).retry_scope();
    retry_scope
        .run(async {
            let mut task = Some(stage);
            let mut run_context = Some(run_context);

            debug!(
                "OOMHandler::run: 1.  ctx_id={ctx_id:?} task={:?} os_thread_id={}  max_other_parallel={:?}",
                task.as_ref().expect("run_with_runner_impl: missing task"),
                os_thread_id(),
                run_context.as_ref().expect("run_with_runner_impl: missing run context").max_other_parallel(),
            );

            let mut i = 0;
            let mut input_select_adjusted: Option<S> = None;

            match t!(runner.run(
                Arc::clone(task.as_ref().expect("run_with_runner_impl: missing task")),
                None,
                device_idx,
                run_context.as_mut().expect("run_with_runner_impl: missing run context").branch(false),
                Some(retry_scope.token()),
            ))
            .await
            {
                Err(te) => {
                    if !runner.can_retry() {
                        warn!("OOMHandler::run, error: 0. ctx_id={ctx_id:?} task={:?} os_thread_id={}  te={te:?} runner can't retry", task.as_ref().expect("run_with_runner_impl: missing task"), os_thread_id(),);

                        return Err(te);
                    }

                    debug!("OOMHandler::run, error: 0. ctx_id={ctx_id:?} task={:?} os_thread_id={}  te={te:?}", task.as_ref().expect("run_with_runner_impl: missing task"), os_thread_id(),);

                    if te.is_retryable_oom() {
                        perf_oom_count();
                        debug!("OOMHandler::run, error: 1. ctx_id={ctx_id:?} task={:?} os_thread_id={} before adjusters", task.as_ref().expect("run_with_runner_impl: missing task"), os_thread_id(),);

                        for adjuster in adjusters {
                            debug!(
                                "OOMHandler::run, error: 2. ctx_id={ctx_id:?} task={:?} adjuster={adjuster:?} os_thread_id={}  starting adjuster",
                                task.as_ref().expect("run_with_runner_impl: missing task"),
                                os_thread_id(),
                            );

                            loop {
                                //TODO while adjuster.can_degrade
                                i += 1;
                                if i > 100 {
                                    //TODO Config
                                    return Err(ThoError::new(format!("OOMHandler: ctx_id={ctx_id:?}  i={i}")));
                                }

                                debug!("OOMHandler::run, error: 3. ctx_id={ctx_id:?} task={:?} os_thread_id={} oom i={i} in loop", task.as_ref().expect("run_with_runner_impl: missing task"), os_thread_id(),);

                                let adjust_result = t!(adjuster.apply(&mut task, &mut run_context, &mut input_select_adjusted, &mut contexts,)).await;

                                debug!("OOMHandler::run, error: on adjust_result ={adjust_result:?} os_thread_id={}", os_thread_id());

                                if let Err(te) = adjust_result {
                                    debug!(
                                        "OOMHandler::run, error: on ctx_id={ctx_id:?} task={:?} adjuster={adjuster:?} os_thread_id={} adjuster error te={te:?}",
                                        task.as_ref().expect("run_with_runner_impl: missing task"),
                                        os_thread_id(),
                                    );

                                    if te.is_retryable_oom() {
                                        debug!(
                                            "OOMHandler::run, error: on adjust_result ctx_id={ctx_id:?} task={:?} os_thread_id={} adjust_context={contexts:?}",
                                            task.as_ref().expect("run_with_runner_impl: missing task"),
                                            os_thread_id(),
                                        );

                                        if !adjuster.degrade(&mut contexts) {
                                            debug!(
                                                "OOMHandler::run, error: on adjust_result moving to next adjuster adjust_result ctx_id={ctx_id:?} task={:?} os_thread_id={} adjust_context={contexts:?}",
                                                task.as_ref().expect("run_with_runner_impl: missing task"),
                                                os_thread_id(),
                                            );

                                            break;
                                        }

                                        continue;
                                    }

                                    return Err(te);
                                }

                                debug!(
                                    "OOMHandler::run, error: ctx_id={ctx_id:?} task={:?} os_thread_id={}  1. input_select_adjusted={input_select_adjusted:?} max_other_parallel={:?}",
                                    task.as_ref().expect("run_with_runner_impl: missing task"),
                                    os_thread_id(),
                                    run_context.as_ref().expect("run_with_runner_impl: missing run context").max_other_parallel(),
                                );

                                let run_result = t!(runner.run(
                                    Arc::clone(task.as_ref().expect("run_with_runner_impl: missing task")),
                                    input_select_adjusted.clone(),
                                    device_idx,
                                    run_context.as_mut().expect("run_with_runner_impl: missing run context").branch(false),
                                    Some(retry_scope.token()),
                                ))
                                .await;

                                match run_result {
                                    Ok(v) => {
                                        debug!(
                                            "OOMHandler::run, error: got ok ctx_id={ctx_id:?} task={:?} os_thread_id={} adjuster={adjuster:?} adjust_context={contexts:?}",
                                            task.as_ref().expect("run_with_runner_impl: missing task"),
                                            os_thread_id(),
                                        );
                                        trace!("OOMHandler::run, error: got ok ctx_id={ctx_id:?} v={v:?}");

                                        match t!(<IA as InputAdjustment<'a, R, C, S>>::handle_success(v, contexts)).await? {
                                            InputAdjustedRun::Done(out) => {
                                                collect_eta_record_oom_success();

                                                debug!("done in {i} steps");
                                                return Ok(out);
                                            }
                                            InputAdjustedRun::Continue(next_contexts) => {
                                                contexts = next_contexts;
                                            }
                                        }
                                    }
                                    Err(te) => {
                                        if !runner.can_retry() {
                                            return Err(te);
                                        }

                                        debug!("OOMHandler::run, error: err is_retryable_oom={} te={te:?}", te.is_retryable_oom());

                                        if te.is_retryable_oom() {
                                            perf_oom_count();
                                            debug!("OOMHandler::run, error: on oom ctx_id={ctx_id:?} task={:?} os_thread_id={} adjust_context={contexts:?}", task.as_ref().expect("run_with_runner_impl: missing task"), os_thread_id(),);

                                            if !adjuster.degrade(&mut contexts) {
                                                debug!(
                                                    "OOMHandler::run, inner error: on oom moving to next adjuster ctx_id={ctx_id:?} task={:?} os_thread_id={} adjust_context={contexts:?}",
                                                    task.as_ref().expect("run_with_runner_impl: missing task"),
                                                    os_thread_id(),
                                                );

                                                break;
                                            }
                                        } else {
                                            warn!("OOMHandler::run, inner error: task={:?} os_thread_id={} \nunexpected error: {te:?}", task.as_ref().expect("run_with_runner_impl: missing task"), os_thread_id(),);
                                            return Err(te);
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        warn!("OOMHandler::run, error: ctx_id={ctx_id:?}  task={:?}  os_thread_id={} \nunexpected error: {te:?}", task.as_ref().expect("run_with_runner_impl: missing task"), os_thread_id(),);
                        return Err(te);
                    }

                    collect_eta_record_oom_all_failed();

                    warn!("OOMHandler::run, error: ctx_id={ctx_id:?} task={:?} os_thread_id={} all failed", task.as_ref().expect("run_with_runner_impl: missing task"), os_thread_id());
                    #[cfg(tho_debug)]
                    t!(GlobalCtx::global().debug_caches(device_idx, &format!("all_failed for {}", input_id_dbg))).await;

                    #[cfg(tho_debug)]
                    GlobalCtx::global().execution(device_idx).debug_worker().await?;

                    // Exhausted adjustments may exclude this data item, without making it retryable again. TODO : it's not ThoError::Data - make ThoError::OOMHandlingFailed
                    Err(ThoError::Data {
                        kind: thoosa_shared::DataErrorKind::OOMHandlingFailed,
                        msg: format!("all_failed: task={:?}", task.as_ref().expect("run_with_runner_impl: missing task")),
                    })
                }
                Ok(o) => {
                    debug!("OOMHandler::run: ctx_id={ctx_id:?} os_thread_id={} done without oom ", os_thread_id());

                    Ok(o)
                }
            }
        })
        .await
}

async fn run_with_runner<'a, 'z, R, S, W, Run, Fut, C, IA, TA, CA>(adjusters: &[Adjusters<IA, TA, CA>], stage: Arc<R>, input: Arc<W>, run_context: R::Ctx, contexts: C, runner: Run) -> Result<R::Out, ThoError>
where
    R: Stage + 'a,
    S: Debug + Clone + Default + 'static,
    C: Debug,
    IA: InputAdjustment<'a, R, C, S>,
    TA: StageAdjustment<'a, R, C>,
    CA: RunContextAdjustment<R::Ctx, C>,
    for<'i> <R as Stage>::In<'i>: Send + Sync + Debug,
    R::Out: Send + Sync + 'static,
    R::Ctx: MaxParallelCtx,
    W: Debug,
    Run: Fn(Arc<R>, Arc<W>, Option<S>, usize, R::Ctx, Option<RetryToken>) -> Fut,
    Fut: Future<Output = Result<R::Out, ThoError>>,
{
    #[cfg(tho_debug)]
    let input_id_dbg = format!("input={:?}", input);

    t!(run_with_runner_impl(
        adjusters,
        stage,
        #[cfg(tho_debug)]
        input_id_dbg,
        run_context,
        contexts,
        SharedRunnerRetryAdapter { input, runner },
    ))
    .await
}

async fn run_with_runner_consume<'a, 'z, R, S, W, Run, Fut, C, IA, TA, CA>(adjusters: &[Adjusters<IA, TA, CA>], stage: Arc<R>, input: W, run_context: R::Ctx, contexts: C, runner: Run) -> Result<R::Out, ThoError>
where
    R: Stage + 'a,
    S: Debug + Clone + Default + 'static,
    C: Debug,
    IA: InputAdjustment<'a, R, C, S>,
    TA: StageAdjustment<'a, R, C>,
    CA: RunContextAdjustment<R::Ctx, C>,
    for<'i> <R as Stage>::In<'i>: Send + Sync,
    R::Out: Send + Sync + 'static,
    R::Ctx: MaxParallelCtx,
    W: ConsumeData<O = <R as Stage>::In<'z>> + Debug, // + WithId<ID = <R::In<'z> as WithId>::ID>,
    Run: Fn(Arc<R>, W, Option<S>, usize, R::Ctx, Option<RetryToken>) -> Fut,
    Fut: Future<Output = Result<ConsumeResult<R::In<'z>, R::Out>, ThoError>>,
{
    #[cfg(tho_debug)]
    let input_id_dbg = format!("input={:?}", input);

    let leases = input.protect(run_context.tho_device_idx());
    t!(run_with_runner_impl(
        adjusters,
        stage,
        #[cfg(tho_debug)]
        input_id_dbg,
        run_context,
        contexts,
        ConsumeRunnerRetryAdapter { leases, input: Some(input), runner },
    ))
    .await
}

#[cfg(test)]
pub mod test_types {
    use super::*;
    // use crate::common::Data;
    use crate::runner::test_utils::DummyTensor;
    use thoosa_named_derive::Named;
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::traits::{RangeSelector, Select, WithId};

    #[derive(Debug, Clone, Hash, Eq, PartialEq, Named)]
    pub struct TestId(&'static str);

    impl From<&'static str> for TestId {
        fn from(value: &'static str) -> Self {
            TestId(value)
        }
    }

    #[derive(Clone, Debug)]
    pub struct TestInput {
        pub id: TestId,
        pub data: DummyTensor,
    }

    impl Select<RangeSelector> for TestInput {
        type Output = TestInput;

        fn subview(&self, selector: RangeSelector) -> Result<Self::Output, ThoError> {
            Ok(TestInput {
                id: self.id.clone(),
                data: DummyTensor(self.data.0[selector.start[0] as usize..selector.end[0] as usize].to_vec()), //.0[selector.start..selector.end])
            })
        }
    }

    impl Input for TestInput {}

    impl Adjustable for TestInput {}

    #[async_trait]
    impl Data for TestInput {
        type Metrics = [usize; 1];

        fn size(&self) -> Self::Metrics {
            [self.data.0.len()]
        }

        fn tho_device_idx(&self) -> usize {
            0
        }

        fn mem_size(&self) -> u64 {
            0
        }
    }

    #[async_trait]
    impl CopyToDevice for TestInput {
        async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
            Ok(TestInput {
                id: self.id.clone(),
                data: DummyTensor(self.data.0.clone()),
            })
        }
    }

    impl WithId for TestInput {
        type ID = TestId;
        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl TestInput {
        pub fn new(n: u32) -> Self {
            Self {
                id: "test".into(),
                data: DummyTensor((0..n).map(|i| i as f32).collect()),
            }
        }
    }

    #[derive(Debug, Clone)]
    pub struct TestOutput {
        pub id: TestId,
        pub v: TestData,
    }

    impl TestOutput {
        pub fn new(v: f32) -> Self {
            Self { id: "test".into(), v: v.into() }
        }
    }

    #[derive(Debug, Clone, Copy)]
    pub struct TestData(pub f32);

    impl From<f32> for TestData {
        fn from(value: f32) -> Self {
            TestData(value)
        }
    }

    #[async_trait]
    impl Data for TestOutput {
        type Metrics = [usize; 1];

        fn size(&self) -> Self::Metrics {
            [1]
        }

        fn tho_device_idx(&self) -> usize {
            0
        }

        fn mem_size(&self) -> u64 {
            0
        }
    }

    #[async_trait]
    impl CopyToDevice for TestData {
        async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
            Ok(*self)
        }
    }

    #[async_trait]
    impl CopyToDevice for TestOutput {
        async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
            Ok(TestOutput { id: self.id.clone(), v: self.v.clone() })
        }
    }

    impl WithId for TestOutput {
        type ID = TestId;
        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl Output for TestOutput {}
}

#[cfg(test)]
pub mod mock_runnable {
    use super::test_types::{TestInput, TestOutput};
    use super::*;
    use crate::runner::common::mock_run_ctx::MockRunCtx;
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use thoosa_common::runner::sync_stage::SyncStage;
    use thoosa_shared::traits::{Stage, WithId};

    mock! {
        #[derive(Clone, Debug)]
        pub Runnable {
            pub fn do_run(&self, input: &TestInput, ctx: &MockRunCtx) -> Result<TestOutput, ThoError>;
            pub fn consume(&self, input: TestInput, ctx: MockRunCtx) -> ConsumeResult<TestInput, TestOutput>;
        }

        impl Stage for Runnable {
            type In<'a> = TestInput;
            type Out = TestOutput;
            type Ctx = MockRunCtx;
        }

    }

    // Forward the GAT-based stage interface to concrete mock methods; mockall
    // cannot erase an owned GAT input's lifetime in consume expectations.
    impl SyncStage for MockRunnable {
        fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
            MockRunnable::do_run(self.as_ref(), input, ctx)
        }
    }

    impl crate::runner::sync_stage::ConsumingStage for MockRunnable {
        fn consume<'f>(self: Arc<Self>, input: Self::In<'f>, ctx: Self::Ctx) -> ConsumeResult<Self::In<'f>, Self::Out> {
            MockRunnable::consume(self.as_ref(), input, ctx)
        }
    }

    impl Adjustable for MockRunnable {}

    impl OutId for MockRunnable {
        fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
            input_id.clone()
        }
    }

    pub fn build<I>(results: I) -> (MockRunnable, Arc<AtomicUsize>)
    where
        I: IntoIterator<Item = Result<TestOutput, ThoError>>,
    {
        let mut task = MockRunnable::new();
        let calls = Arc::new(AtomicUsize::new(0));

        for r in results {
            let c = calls.clone();
            match r {
                Ok(o) => {
                    task.expect_do_run().times(1).return_once(move |_, _| {
                        c.fetch_add(1, Ordering::SeqCst);
                        Ok(o)
                    });
                }
                Err(e) => {
                    task.expect_do_run().times(1).return_once(move |_, _| {
                        c.fetch_add(1, Ordering::SeqCst);
                        Err(e)
                    });
                }
            }
        }
        (task, calls)
    }
}

// removed unused TestInputContext test scaffolding

#[cfg(test)]
pub mod mock_input_adjuster {
    use super::*;
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use thoosa_shared::traits::RangeSelector;
    // use crate::common::mock_input_output::MockInput;

    mock! {
        #[derive(Debug)]
        pub InputAdjuster {}
        #[async_trait]
        impl Adjuster<super::mock_input_context::MockInputContext> for InputAdjuster {
            type A = RangeSelector; //std::ops::Range<<TestInput as Data>::Metrics>;
            async fn adjust(&self, adjustable: <Self as Adjuster<super::mock_input_context::MockInputContext>>::A, adjust_context: &mut super::mock_input_context::MockInputContext) -> Result<<Self as Adjuster<super::mock_input_context::MockInputContext>>::A, AdjusterError<<Self as Adjuster<super::mock_input_context::MockInputContext>>::A>>;
            fn degrade(&self, adjust_context: &mut super::mock_input_context::MockInputContext) -> bool;
        }
    }

    pub fn build<I>(inputs: I) -> (MockInputAdjuster, Arc<AtomicUsize>)
    where
        I: IntoIterator<Item = RangeSelector>,
    {
        let mut adj = MockInputAdjuster::new();
        let calls = Arc::new(AtomicUsize::new(0));
        for input in inputs {
            let c = calls.clone();
            adj.expect_adjust().times(1).return_once(move |_, _| {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(input)
            });
        }
        adj.expect_degrade().return_const(false);
        (adj, calls)
    }
}

#[cfg(test)]
pub mod mock_task_adjuster {
    use super::mock_runnable::MockRunnable;

    use super::*;
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    mock! {
        #[derive(Debug)]
        pub StageAdjuster {}
        #[async_trait]
        impl Adjuster<MockRunnableContext> for StageAdjuster {
            type A = Arc<MockRunnable>;
            async fn adjust(&self, adjustable: <Self as Adjuster<MockRunnableContext>>::A, adjust_context: &mut MockRunnableContext) -> Result<<Self as Adjuster<MockRunnableContext>>::A, AdjusterError<<Self as Adjuster<MockRunnableContext>>::A>>;
            fn degrade(&self, adjust_context: &mut MockRunnableContext) -> bool;
        }
    }

    pub fn build<I>(tasks: I) -> (MockStageAdjuster, Arc<AtomicUsize>)
    where
        I: IntoIterator<Item = MockRunnable>,
    {
        let mut adj = MockStageAdjuster::new();
        let calls = Arc::new(AtomicUsize::new(0));

        for task in tasks.into_iter().map(Some) {
            let c = calls.clone();
            adj.expect_adjust().return_once(move |_, _| {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(Arc::new(task.unwrap()))
            });
        }
        adj.expect_degrade().return_const(false);
        (adj, calls)
    }
}

#[cfg(test)]
pub mod mock_context_adjuster {
    use super::*;
    use crate::runner::common::mock_run_ctx::MockRunCtx;
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    mock! {
        #[derive(Debug)]
        pub ContextAdjuster {}
        #[async_trait]
        impl Adjuster<MockRunCtxContext> for ContextAdjuster {
            type A = MockRunCtx;
            async fn adjust(&self, adjustable: <Self as Adjuster<MockRunCtxContext>>::A, adjust_context: &mut MockRunCtxContext) -> Result<<Self as Adjuster<MockRunCtxContext>>::A, AdjusterError<<Self as Adjuster<MockRunCtxContext>>::A>>;
            fn degrade(&self, adjust_context: &mut MockRunCtxContext) -> bool;
        }
    }

    pub fn build(times: usize) -> (MockContextAdjuster, Arc<AtomicUsize>) {
        let mut adj = MockContextAdjuster::new();
        let calls = Arc::new(AtomicUsize::new(0));
        for _ in 0..times {
            let c = calls.clone();
            adj.expect_adjust().returning(move |rc, _ctx| {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(rc)
            });
        }
        adj.expect_degrade().return_const(false);
        (adj, calls)
    }
}

#[cfg(test)]
mod tests {
    use super::mock_context_adjuster::build as mock_context_adjuster;
    use super::mock_input_adjuster::{MockInputAdjuster, build as mock_input_adjuster};
    use super::mock_runnable::build as mock_runnable;
    use super::mock_task_adjuster::build as mock_task_adjuster;
    use super::test_types::{TestInput, TestOutput};
    use super::{Adjusters, Contexts, OOMHandler};
    use super::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoStageAdjuster, WithInputContext, WithRunCtxContext, WithRunnableContext, input_adjuster, run_ctx_adjuster, task_adjuster};
    // use crate::common::runnable::Runnable;
    use crate::runner::common::cachable::{BorrowRef, DevicesWrap};
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use thoosa_shared::ThoError;
    use thoosa_shared::traits::RangeSelector;
    // type Selector = Range<[usize; 1]>;

    #[derive(Debug)]
    struct BorrowOnlyInput {
        inner: TestInput,
    }

    impl BorrowOnlyInput {
        fn new(inner: TestInput) -> Self {
            Self { inner }
        }
    }

    impl BorrowRef for BorrowOnlyInput {
        fn protect(&self, _device: usize) -> thoosa_common::runner::residency::ResidencyLease {
            thoosa_common::runner::residency::ResidencyLease::default()
        }

        type Out<'a>
            = &'a TestInput
        where
            Self: 'a;

        fn borrow_ref<'o>(&'o self, _to_tho_device_idx: usize) -> impl Future<Output = Result<Self::Out<'o>, ThoError>> + Send
        where
            Self: 'o,
        {
            async move { Ok(&self.inner) }
        }
    }

    #[tokio::test]
    async fn test_consume_without_recoverable_input_stops_retry_and_releases_runner() {
        use crate::runner::global_ctx::{MockGlobalCtx, mock_global_ctx};
        use crate::runner::syn_stage_runner::SyncStageRunner;
        use thoosa_common::runner::sync_stage::ConsumeResult;
        let global_context = MockGlobalCtx::global_context();
        let mock = Box::leak(Box::new(mock_global_ctx::build().with_runner(SyncStageRunner::new_with_reset_hook(8, 1, Arc::new(|_| Box::pin(async { Ok(()) })))).finish()));
        global_context.expect().return_const(&*mock);
        let (task, _) = mock_runnable([]);
        let run_ctx = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(0).with_branch(1).finish();
        let adjusters: Vec<Adjusters<NoInputAdjuster, NoStageAdjuster, NoRunCtxAdjuster>> = Vec::new();
        // An outer synchronization error means consume succeeded and its input
        // cannot be replayed. Preserve that error instead of exhausting adjusters.
        let error = super::run_with_runner_consume(&adjusters, Arc::new(task), TestInput::new(1), run_ctx, NoContexts, |_, _, _: Option<()>, device_idx, _, retry| async move {
            super::oom_executor(device_idx).submit_consume(0, async { Err::<ConsumeResult<TestInput, TestOutput>, _>(ThoError::oom().context("consume synchronization")) }, retry).await
        })
        .await
        .unwrap_err();
        assert!(error.is_oom());
        assert!(format!("{error:?}").contains("consume synchronization"));
        // Leaving the retry chain must release the exclusive OOM reservation.
        tokio::time::timeout(std::time::Duration::from_secs(2), super::oom_executor(1).submit(0, async { Ok(()) }, None)).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_consume_retry_adapter_preserves_failed_input() {
        use super::RunnerRetryAdapter;
        use thoosa_common::runner::sync_stage::ConsumeResult;
        let (task, _) = mock_runnable([]);
        let input = TestInput::new(3);
        let original_allocation = input.data.0.as_ptr();
        let mut adapter = super::ConsumeRunnerRetryAdapter {
            leases: Default::default(),
            input: Some(input),
            runner: |_: Arc<super::mock_runnable::MockRunnable>, input, _: Option<()>, _, _, _| async move { Ok(ConsumeResult::Failed { input, error: ThoError::oom() }) },
        };
        let run_ctx = crate::runner::common::mock_run_ctx::build().finish();
        let error = adapter.run(Arc::new(task), None, 1, run_ctx, None).await.unwrap_err();
        assert!(error.is_oom());
        assert!(adapter.can_retry());
        assert_eq!(adapter.input.unwrap().data.0.as_ptr(), original_allocation);
    }

    #[tokio::test]
    async fn test_recovery_interrupts_oom_adapter_between_attempts() {
        use crate::runner::global_ctx::{MockGlobalCtx, mock_global_ctx};
        use crate::runner::syn_stage_runner::SyncStageRunner;
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            let global_context = MockGlobalCtx::global_context();
            let mock = Box::leak(Box::new(mock_global_ctx::build().with_runner(SyncStageRunner::new_with_reset_hook(8, 1, Arc::new(|_| Box::pin(async { Ok(()) })))).finish()));
            global_context.expect().return_const(&*mock);
            let (task, _) = mock_runnable([]);
            let run_ctx = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(0).with_branch(1).finish();
            let (returned, received) = tokio::sync::oneshot::channel();
            let returned = Arc::new(Mutex::new(Some(returned)));
            let adjusters: Vec<Adjusters<NoInputAdjuster, NoStageAdjuster, NoRunCtxAdjuster>> = Vec::new();
            // Exercise the actual OOM runner while its adapter is waiting after
            // a completed exclusive OOM attempt (like asynchronous input burping).
            let mut run = Box::pin(super::run_with_runner(&adjusters, Arc::new(task), Arc::new(()), run_ctx, NoContexts, move |_, _, _: Option<()>, device_idx, _, retry| {
                let returned = Arc::clone(&returned);
                async move {
                    let _ = super::oom_executor(device_idx).submit(0, async { Err::<TestOutput, _>(ThoError::oom()) }, retry).await;
                    returned.lock().unwrap().take().unwrap().send(()).unwrap();
                    std::future::pending::<Result<TestOutput, ThoError>>().await
                }
            }));
            assert!(futures::poll!(run.as_mut()).is_pending());
            // Poll the adapter to its post-attempt await without a timing sleep.
            tokio::select! {
                _ = &mut run => panic!("adapter unexpectedly completed"),
                _ = received => {}
            }
            let executor = super::oom_executor(1);
            let ticket = executor.request_cuda_reset(ThoError::CudaBeingRecovered).unwrap().unwrap();
            assert!(matches!(run.await.unwrap_err().what(), ThoError::CudaBeingRecovered));
            executor.cuda_gate().unwrap().wait_for_pause_to_finish(ticket).await.unwrap();
            executor.submit(0, async { Ok(()) }, None).await.unwrap();
        })
        .await
        .expect("an old OOM chain must unwind instead of surviving recovery");
    }

    #[tokio::test]
    async fn test_cancelling_retry_runner_before_oom_releases_executor() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let (task, _) = mock_runnable([]);
        let run_ctx = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(0).with_branch(1).finish();
        let started = Arc::new(tokio::sync::Notify::new());
        let release = Arc::new(tokio::sync::Notify::new());
        let adjusters: Vec<Adjusters<NoInputAdjuster, NoStageAdjuster, NoRunCtxAdjuster>> = Vec::new();
        // Exercise the actual retry runner's ownership, rather than creating a
        // scope in the test: the first error has not reached the OOM handler yet.
        let mut run = Box::pin(super::run_with_runner(&adjusters, Arc::new(task), Arc::new(()), run_ctx, NoContexts, {
            let started = Arc::clone(&started);
            let release = Arc::clone(&release);
            move |_, _, _: Option<()>, device_idx, _, retry| {
                let started = Arc::clone(&started);
                let release = Arc::clone(&release);
                async move {
                    super::oom_executor(device_idx)
                        .submit(
                            0,
                            async move {
                                started.notify_one();
                                release.notified().await;
                                Err::<TestOutput, _>(ThoError::oom())
                            },
                            retry,
                        )
                        .await
                }
            }
        }));
        assert!(futures::poll!(run.as_mut()).is_pending());
        started.notified().await;
        drop(run);
        release.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(2), super::oom_executor(1).submit(0, async { Ok(()) }, None)).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_exhausted_oom_is_skippable_and_releases_runner() {
        GlobalCtx::init_devices(2).unwrap();
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);
        let (task, task_calls) = mock_runnable([Err(ThoError::oom()), Err(ThoError::oom())]);
        let (adjuster, adjust_calls) = mock_context_adjuster(1);
        let handler = OOMHandler::<NoInputAdjuster, NoStageAdjuster, _, ()>::new(vec![Adjusters::RunContext(run_ctx_adjuster(adjuster))], None);
        let ctx = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(1).with_branch(1).finish();
        let error = handler
            .run(&Arc::new(task), Arc::new(BorrowOnlyInput::new(TestInput::new(1))), ctx, WithRunCtxContext::new(super::MockRunCtxContext::new()))
            .await
            .unwrap_err()
            .context("motif training");
        assert_eq!(error.data_error_kind(), Some(thoosa_shared::DataErrorKind::OOMHandlingFailed));
        assert!(!error.is_retryable_oom());
        assert_eq!(task_calls.load(Ordering::SeqCst), 2);
        assert_eq!(adjust_calls.load(Ordering::SeqCst), 1);
        // Skipping an oversized item must also let the next item's GPU work proceed.
        tokio::time::timeout(std::time::Duration::from_secs(2), super::oom_executor(1).submit(0, async { Ok(()) }, None)).await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn test_terminal_training_oom_does_not_retry() {
        let error = ThoError::TrainingSessionFailed(Arc::new(thoosa_shared::TrainingSessionFailure { original: ThoError::oom(), recovery: None })).context("optimizer commit failed");
        let (task, task_calls) = mock_runnable([Err(error)]);
        let (adjuster, adjust_calls) = mock_context_adjuster(1);
        let handler = OOMHandler::<NoInputAdjuster, NoStageAdjuster, _, ()>::new(vec![Adjusters::RunContext(run_ctx_adjuster(adjuster))], None);
        let ctx = crate::runner::common::mock_run_ctx::build().with_device_idx(0).finish();
        let result = handler.run(&Arc::new(task), Arc::new(BorrowOnlyInput::new(TestInput::new(1))), ctx, WithRunCtxContext::new(super::MockRunCtxContext::new())).await;
        let error = result.unwrap_err();
        assert!(error.is_oom(), "OOM cause must remain visible to the handler's caller");
        assert!(!error.is_retryable_oom());
        assert_eq!(task_calls.load(Ordering::SeqCst), 1, "a partial update must never be replayed");
        assert_eq!(adjust_calls.load(Ordering::SeqCst), 0, "terminal errors must bypass recovery adjusters");
    }

    #[tokio::test]
    async fn test_no_input_adjuster_accepts_borrow_ref_only_input() {
        let (task, task_calls) = mock_runnable([Ok(TestOutput::new(12.))]);
        let oom_handler = OOMHandler::<NoInputAdjuster, NoStageAdjuster, NoRunCtxAdjuster, ()>::new(vec![], None);
        let ctx_mock = crate::runner::common::mock_run_ctx::build().with_device_idx(0).finish();

        let out: TestOutput = oom_handler.run(&Arc::new(task), Arc::new(BorrowOnlyInput::new(TestInput::new(1))), ctx_mock, NoContexts).await.unwrap();

        assert_eq!(12., out.v.0);
        assert_eq!(1, task_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_input_task_context_once() {
        GlobalCtx::init_devices(2).unwrap();
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let input_adj_calls = Arc::new(AtomicUsize::new(0));
        let mut input_adj = MockInputAdjuster::new();
        {
            let c = input_adj_calls.clone();

            input_adj.expect_adjust().returning(move |i, _| {
                c.fetch_add(1, Ordering::SeqCst);
                Ok(i.clone())
            });
        }
        input_adj.expect_degrade().times(1).return_const(false);

        let (ctx_adj, ctx_adj_calls) = mock_context_adjuster(1);

        let (init_task, init_calls) = mock_runnable([
            Err(ThoError::oom()), //no adjusters
            Err(ThoError::oom()), //input adjuster
        ]);

        let (adjusted_task, adjusted_calls) = mock_runnable([
            Err(ThoError::oom()),      //task adjuster
            Ok(TestOutput::new(111.)), //ctx adjuster
        ]);

        let (task_adj, task_adj_calls) = mock_task_adjuster([adjusted_task]);

        let ctx_mock = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(1).with_branch(4).finish();

        let oom_handler = OOMHandler::new(vec![Adjusters::Input(input_adjuster(input_adj)), Adjusters::Stage(task_adjuster(task_adj)), Adjusters::RunContext(run_ctx_adjuster(ctx_adj))], None);

        let out: TestOutput = oom_handler
            .run(
                &Arc::new(init_task),
                Arc::new(DevicesWrap::wrap_cpu(TestInput::new(100))),
                ctx_mock,
                Contexts::new(
                    super::mock_input_context::build().with_partial_output(TestOutput::new(111.)).with_all_done(&[true]).with_build_final_output(TestOutput::new(111.)).finish(),
                    super::MockRunnableContext::new(),
                    super::MockRunCtxContext::new(),
                ),
            )
            .await
            .unwrap();

        assert_eq!(111., out.v.0);

        assert_eq!(1, input_adj_calls.load(Ordering::SeqCst));
        assert_eq!(1, task_adj_calls.load(Ordering::SeqCst));
        assert_eq!(1, ctx_adj_calls.load(Ordering::SeqCst));

        assert_eq!(2, init_calls.load(Ordering::SeqCst));
        assert_eq!(2, adjusted_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_only_task_adjuster() {
        GlobalCtx::init_devices(2).unwrap();
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let (init_task, init_calls) = mock_runnable([Err(ThoError::oom())]);
        let (final_task, final_calls) = mock_runnable([Ok(TestOutput::new(10.))]);
        let (task_adj, task_adj_calls) = mock_task_adjuster([final_task]);
        let adjusters: Vec<Adjusters<NoInputAdjuster, _, NoRunCtxAdjuster>> = vec![Adjusters::Stage(task_adjuster(task_adj))];
        let oom_handler: OOMHandler<'_, 'static, _, _, _, RangeSelector> = OOMHandler::new(adjusters, None);

        let ctx_mock = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(1).with_branch(4).finish();

        let out: TestOutput = oom_handler.run(&Arc::new(init_task), Arc::new(DevicesWrap::wrap_cpu(TestInput::new(1))), ctx_mock, WithRunnableContext::new(super::MockRunnableContext::new())).await.unwrap();
        assert_eq!(10., out.v.0);
        assert_eq!(1, task_adj_calls.load(Ordering::SeqCst));
        assert_eq!(1, init_calls.load(Ordering::SeqCst));
        assert_eq!(1, final_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_only_input_adjuster() {
        GlobalCtx::init_devices(2).unwrap();
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let (task, task_calls) = mock_runnable([Err(ThoError::oom()), Ok(TestOutput::new(10.))]);
        let (input_adj, input_calls) = mock_input_adjuster([RangeSelector([0]..[1])]);
        let adjusters: Vec<Adjusters<_, NoStageAdjuster, NoRunCtxAdjuster>> = vec![Adjusters::Input(input_adjuster(input_adj))];
        let oom_handler = OOMHandler::new(adjusters, None);
        let ctx_mock = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(1).with_branch(4).finish();

        let out = oom_handler
            .run(
                &Arc::new(task),
                Arc::new(DevicesWrap::wrap_cpu(TestInput::new(1))),
                ctx_mock,
                WithInputContext::new(super::mock_input_context::build().with_partial_output(TestOutput::new(10.)).with_all_done(&[true]).with_build_final_output(TestOutput::new(10.)).finish()),
            )
            .await
            .unwrap();
        assert_eq!(10., out.v.0);
        assert_eq!(1, input_calls.load(Ordering::SeqCst));
        assert_eq!(2, task_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_only_run_ctx_adjuster() {
        GlobalCtx::init_devices(2).unwrap();
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let (task, task_calls) = mock_runnable([Err(ThoError::oom()), Ok(TestOutput::new(10.))]);
        let (ctx_adj, ctx_adj_calls) = mock_context_adjuster(1);

        let adjusters: Vec<Adjusters<NoInputAdjuster, NoStageAdjuster, _>> = vec![Adjusters::RunContext(run_ctx_adjuster(ctx_adj))];
        let oom_handler: OOMHandler<'_, 'static, _, _, _, RangeSelector> = OOMHandler::new(adjusters, None);

        let ctx_mock = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(1).with_branch(1).finish();

        let out = oom_handler.run(&Arc::new(task), Arc::new(DevicesWrap::wrap_cpu(TestInput::new(1))), ctx_mock, WithRunCtxContext::new(super::MockRunCtxContext::new())).await.unwrap();
        assert_eq!(10., out.v.0);
        assert_eq!(1, ctx_adj_calls.load(Ordering::SeqCst));
        assert_eq!(2, task_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_parallel_tasks() {
        GlobalCtx::init_devices(2).unwrap();
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let (init_task1, init_calls1) = mock_runnable([Err(ThoError::oom())]);
        let (final_task1, final_calls1) = mock_runnable([Ok(TestOutput::new(5.))]);

        let (task_adj1, task_adj_calls1) = mock_task_adjuster([final_task1]);

        let (init_task2, init_calls2) = mock_runnable([Err(ThoError::oom())]);
        let (final_task2, final_calls2) = mock_runnable([Ok(TestOutput::new(7.))]);

        let (task_adj2, task_adj_calls2) = mock_task_adjuster([final_task2]);

        let init_task1_arc = Arc::new(init_task1);
        let input_arc = Arc::new(DevicesWrap::wrap_cpu(TestInput::new(1)));

        let ctx_mock_1 = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(1).with_branch(4).finish();
        let ctx_mock_2 = crate::runner::common::mock_run_ctx::build().with_id(0).with_device_idx(1).with_max_other_parallel(1).with_branch(4).finish();

        let adjusters1: Vec<Adjusters<NoInputAdjuster, _, NoRunCtxAdjuster>> = vec![Adjusters::Stage(task_adjuster(task_adj1))];
        let oom1: OOMHandler<'_, 'static, _, _, _, RangeSelector> = OOMHandler::new(adjusters1, None);
        let h1 = oom1.run(&init_task1_arc, Arc::clone(&input_arc), ctx_mock_1, WithRunnableContext::new(super::MockRunnableContext::new()));
        let init_task2_arc = Arc::new(init_task2);

        let adjusters2: Vec<Adjusters<NoInputAdjuster, _, NoRunCtxAdjuster>> = vec![Adjusters::Stage(task_adjuster(task_adj2))];
        let oom2: OOMHandler<'_, 'static, _, _, _, RangeSelector> = OOMHandler::new(adjusters2, None);
        let h2 = oom2.run(&init_task2_arc, Arc::clone(&input_arc), ctx_mock_2, WithRunnableContext::new(super::MockRunnableContext::new()));
        let r1: Result<TestOutput, ThoError> = h1.await;
        let r2: Result<TestOutput, ThoError> = h2.await;
        assert_eq!(5., r1.unwrap().v.0);
        assert_eq!(7., r2.unwrap().v.0);

        assert_eq!(1, task_adj_calls1.load(Ordering::SeqCst));
        assert_eq!(1, task_adj_calls2.load(Ordering::SeqCst));

        assert_eq!(1, init_calls1.load(Ordering::SeqCst));
        assert_eq!(1, init_calls2.load(Ordering::SeqCst));

        assert_eq!(1, final_calls1.load(Ordering::SeqCst));
        assert_eq!(1, final_calls2.load(Ordering::SeqCst));
    }
}
