use std::borrow::Borrow;
use thoosa_shared::collect_futures_eta::{collect_eta_record_oom_all_failed, collect_eta_record_oom_success};
use crate::runner::common::{MaxParallelCtx, RunCtx};
use crate::runner::common::cachable::{ConsumeData, DevicesWrap, DevicesWrapCopyMode, WithDataRef, WithSelectDataRefSend};
use crate::runner::common::runnable::{AsyncStage, ConsumeResult, Stage, SyncStage};
use crate::runner::common::{DropIf, OutId};
use crate as thoosa_common;
#[double]
use thoosa_common::runner::global_ctx::GlobalCtx;

use crate::runner::statistics::Statistics;
use async_trait::async_trait;
use log::{debug, warn};
use mockall::automock;
use mockall_double::double;
use std::fmt::Debug;
use std::future::Future;
use std::marker::PhantomData;
use std::sync::Arc;
use plotters::coord::ranged1d::ReversibleRanged;
use thoosa_shared::{os_thread_id, panic_to_err_async, CopyToDevice};
use thoosa_shared::traits::{Adjustable, Data, Select, WithId};
use thoosa_shared::{Cause, ThoError};
use crate::runner::adjusters::max_parallel::MAX_PARALLEL;

#[derive(Debug)]
pub enum AdjusterError<A: Adjustable> {
    AdjustFailed(A),
    SomethingElse

}
#[async_trait]
pub trait Adjuster<AC>: Debug + Send + Sync {
    type A: Adjustable;

    async fn adjust(
        &self,
        adjustable: Self::A,
        adjust_context: &mut AC,
    ) -> Result<Self::A, AdjusterError<Self::A>>;

    fn degrade(&self, _adjust_context: &mut AC) -> bool {
        false
    }
}

#[allow(async_fn_in_trait)]
pub trait InputContext<'a, R>: Debug + Send + Sync
where
    R: Stage + 'a,
    // <R as Stage>::In<'a>: Data
{
    async fn add_partial_output(&mut self, out: R::Out);
    fn all_done(&self) -> bool;
    // fn input_len(&self) -> &<<R as Stage>::In<'a> as Data>::Metrics;
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
            async fn add_partial_output(&mut self, out: TestOutput);
            fn all_done(&self) -> bool;
            // fn input_len(&self) -> &[usize; 1];
            async fn build_final_output(self) -> Result<TestOutput, ThoError>;
        }
    }

    pub struct InputContextBuilder {
        mock: MockInputContext,
    }

    impl InputContextBuilder {
        pub fn with_partial_output(mut self, inp: super::test_types::TestOutput) -> Self {
            self.mock.expect_add_partial_output().call(inp);

            self
        }

        pub fn with_build_final_output(mut self, inp: super::test_types::TestOutput) -> Self {
            self.mock
                .expect_build_final_output()
                .return_once(move || Ok(inp));
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
        InputContextBuilder {
            mock: MockInputContext::new(),
        }
    }
}

#[automock]
pub trait RunnableContext: Debug {}

#[automock]
pub trait RunCtxContext: Debug {}

#[derive(Debug)]
pub struct Contexts<'a, R, IC, RC, RCC>
where
    R: Stage + 'a,
    // <R as Stage>::In<'a>: Data,
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
    // <R as Stage>::In<'a>: Data,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
{
    pub fn new(input: IC, runnable: RC, run_ctx: RCC) -> Self {
        Self {
            input,
            runnable,
            run_ctx,
            pd: PhantomData,
        }
    }
}

#[derive(Debug)]
pub enum Adjusters<'a, R, IC, RC, RCC, S>
where
    R: Stage + 'a,
    <R as Stage>::In<'a>: /*Data + Select<S> +*/ Debug,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    // <R as Stage>::In<'a>: Select<S>,
    S: Adjustable + Debug + Default + Clone
{
    Input(Box<dyn Adjuster<IC, A=S> + 'a>),//it adjusts Selector used to fetch chunk of input
    Task(Box<dyn Adjuster<RC, A = Arc<R>> + 'a>),
    RunContext(Box<dyn Adjuster<RCC, A = R::Ctx> + 'a>),
}

impl<'a, R, IC, RC, RCC, S> Adjusters<'a, R, IC, RC, RCC, S>
where
    R: Stage + 'a,
    IC: InputContext<'a, R>,
    R: Adjustable,
    <R as Stage>::In<'a>: /*Data + Select<S> +*/ Debug,
    RC: RunnableContext,
    RCC: RunCtxContext,
    <R as Stage>::Ctx: Adjustable,
    // <R as Stage>::In<'a>: Select<S>,
    S: Adjustable + Debug + Default + Clone
{
    fn degrade(&self, adjust_context: &mut Contexts<'a, R, IC, RC, RCC>) -> bool {
        match self {
            Adjusters::Input(v) => v.degrade(&mut adjust_context.input),
            Adjusters::Task(v) => v.degrade(&mut adjust_context.runnable),
            Adjusters::RunContext(v) => v.degrade(&mut adjust_context.run_ctx),
        }
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
    async fn add_partial_output(&mut self, o: R::Out) {
        self.0.replace(o);
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
pub struct OOMHandler<'a, 'z, R, IC, RC, RCC, S>
where
    R: SyncStage + Adjustable + OutId,
    // for<'i> <R as Stage>::In<'i>: Data,
    R::Ctx: Adjustable,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    S: Adjustable + Debug + Clone + Default + 'static,
    // <R as Stage>::In<'a>: Select<S, Output=<R as Stage>::In<'a>> + Send + Sync ,

    for<'i> <R as Stage>::In<'i>: thoosa_shared::traits::WithId
{
    adjusters: Vec<Adjusters<'a, R, IC, RC, RCC, S>>,
    #[allow(dead_code)]
    statistics: Option<&'a Arc<Statistics>>,
    pd_rc: PhantomData<(RC, &'z RCC)>,
}

#[cfg(any(test, feature = "mock_global_ctx"))]
fn oom_executor(_device_idx: usize) -> &'static crate::runner::dyn_mpsc::DynamicTaskExecutor {
    static EXECUTOR: std::sync::OnceLock<crate::runner::dyn_mpsc::DynamicTaskExecutor> = std::sync::OnceLock::new();
    EXECUTOR.get_or_init(|| crate::runner::dyn_mpsc::DynamicTaskExecutor::new(16, std::num::NonZeroUsize::new(1).expect("non-zero")))
}

#[cfg(not(any(test, feature = "mock_global_ctx")))]
fn oom_executor(device_idx: usize) -> &'static crate::runner::dyn_mpsc::DynamicTaskExecutor {
    GlobalCtx::global().get_executor(device_idx)
}

impl<'a, 'z, R, IC, RC, RCC, S> OOMHandler<'a, 'z, R, IC, RC, RCC, S>
where
    R: SyncStage + Adjustable + OutId + Send + Sync + 'static + 'a,
    R::Ctx: Adjustable + Send + Sync + 'static,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    S: Adjustable + Debug + Clone + Default + Send + 'static,//S - for Select<Selector>, 'static is for Any in WithSelectDataRef for ClcResourcesMotifsDistInputDeviceWrapper::with_selected_data_ref
    // <R as Stage>::In<'a>: Select<S, Output=<R as Stage>::In<'a>> + Send + Sync ,

    for<'i> <R as Stage>::In<'i>: thoosa_shared::traits::WithId + Data + Send + Sync,
    R::Out: WithId + Send + Sync + 'static,
{
    pub fn new(
        adjusters: Vec<Adjusters<'a, R, IC, RC, RCC, S>>,
        statistics: Option<&'a Arc<Statistics>>,
    ) -> Self {
        Self {
            adjusters,
            statistics,
            pd_rc: Default::default(),
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="OOMHandler::run"))]
    pub async fn run<W>(
        &self,
        task: &Arc<R>,
        input: Arc<W>,
        run_context: R::Ctx,
        contexts: Contexts<'a, R, IC, RC, RCC>,
    ) -> Result<R::Out, ThoError>
    where
        W: WithSelectDataRefSend<S> + Send + Sync + 'static,
        W: WithId<ID = <R::In<'z> as WithId>::ID>,
        for<'f> <W as WithDataRef>::O<'f>: Borrow<<R as Stage>::In<'f>>,
    {
        run_with_runner(
            &self.adjusters,
            Arc::clone(task),
            input,
            run_context,
            contexts,
            |task, input, selector, device_idx, run_ctx, job_id_to_rerun| async move {
                debug!("OOMHandler::run: task={task:?} run_ctx={run_ctx:?} acquired_permits={}", run_ctx.max_other_parallel());

                oom_executor(device_idx).submit(run_ctx.max_other_parallel().into(), async move {
                    if let Some(selector) = selector {
                        input
                            .with_selected_data_ref_send(
                                device_idx,
                                run_ctx,
                                selector,
                                None,
                                |data, ctx| Arc::clone(&task).do_run(data.borrow(), ctx),
                            )
                            .await
                    } else {
                        input
                            .with_data_ref_send(
                                device_idx,
                                run_ctx,
                                None,
                                |data, ctx| Arc::clone(&task).do_run(data.borrow(), ctx),
                            )
                            .await
                    }
                }, job_id_to_rerun)
                .await
            },
        )
        .await
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="OOMHandler::consume"))]
    pub async fn consume<W>(
        &self,
        task: &Arc<R>,
        input: W,
        run_context: R::Ctx,
        contexts: Contexts<'a, R, IC, RC, RCC>,
    ) -> Result<R::Out, ThoError>
    where
        W:ConsumeData<O = <R as Stage>::In<'z>>,
        W:WithId<ID = <R::In<'z> as WithId>::ID> + Debug + Send + Sync + 'static,
        <R as Stage>::In<'z>: 'static,
    {


        run_with_runner_consume(
            &self.adjusters,
            Arc::clone(task),
            input,
            run_context,
            contexts,
            |task: Arc<R>, input: W, _selector: Option<S>, device_idx: usize, run_ctx: R::Ctx, job_id_to_rerun| async move {
                debug!("OOMHandler::run: task={task:?} run_ctx={run_ctx:?} acquired_permits={}", run_ctx.max_other_parallel());

                oom_executor(device_idx).submit(run_ctx.max_other_parallel().into(), async move {

                    Ok(input.consume(device_idx, run_ctx, None, |input, run_ctx| {
                        Arc::clone(&task).consume(input, run_ctx)

                    }).await)


                }, job_id_to_rerun).await.unwrap()
            },
        )
        .await
    }

}

#[derive(Debug)]
pub struct AsyncOOMHandler<'a, 'z, R, IC, RC, RCC, S>
where
    R: AsyncStage + Adjustable,
    R::Ctx: Adjustable,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    S: Adjustable + Debug + Clone + Default + 'static,
    for<'i> <R as Stage>::In<'i>: thoosa_shared::traits::WithId + Data + CopyToDevice + Send + Sync + 'static,
    for<'i> <<R as Stage>::In<'i> as Data>::M: DevicesWrapCopyMode<R::In<'i>>,
    <R as Stage>::In<'static>: Select<S, Output = <R as Stage>::In<'static>>,
    R::Out: Data + CopyToDevice + WithId + Send + Sync + 'static,
    <R::Out as Data>::M: DevicesWrapCopyMode<R::Out>,
{
    adjusters: Vec<Adjusters<'a, R, IC, RC, RCC, S>>,
    #[allow(dead_code)]
    statistics: Option<&'a Arc<Statistics>>,
    pd_rc: PhantomData<(RC, &'z RCC)>,
}

#[tracing::instrument(level = "debug", skip_all, fields(agg_perf="pref_oom_count"))]
fn pref_oom_count(){

}

fn oom_job_id(error: &ThoError) -> Option<u64> {
    match error.what() {
        Cause::JobError { job_id, inner } if inner.is_oom() => Some(*job_id),
        _ => None,
    }
}

fn is_retryable_oom(error: &ThoError) -> bool {
    matches!(error.what(), Cause::OutOfMemory | Cause::AdjusterFailed) || oom_job_id(error).is_some()
}

struct ExclusiveJobGuard {
    device_idx: usize,
    job_id: Option<u64>,
}

impl ExclusiveJobGuard {
    fn new(device_idx: usize) -> Self {
        Self { device_idx, job_id: None }
    }

    fn update_from_error(&mut self, error: &ThoError) {
        if let Some(job_id) = oom_job_id(error) {
            debug!("ExclusiveJobGuard::update_from_error: job_id={job_id}");

            self.job_id.replace(job_id);
        }
    }

    fn job_id(&self) -> Option<u64> {
        self.job_id
    }
}

impl Drop for ExclusiveJobGuard {
    fn drop(&mut self) {
        if let Some(job_id) = self.job_id {
            oom_executor(self.device_idx).clear_exclusive_job_id(job_id);
        }
    }
}


#[allow(async_fn_in_trait)]
trait RunnerRetryAdapter<'z, R, S>
where
    R: Stage,
{
    async fn run(
        &mut self,
        task: Arc<R>,
        selector: Option<S>,
        device_idx: usize,
        run_ctx: R::Ctx,
        job_id_to_rerun: Option<u64>,
    ) -> Result<R::Out, ThoError>;

    fn requires_all_done(&self) -> bool {
        false
    }
}

struct SharedRunnerRetryAdapter<W, Run> {
    input: Arc<W>,
    runner: Run,
}

impl<'z, R, S, W, Run, Fut> RunnerRetryAdapter<'z, R, S> for SharedRunnerRetryAdapter<W, Run>
where
    R: Stage,
    Run: Fn(Arc<R>, Arc<W>, Option<S>, usize, R::Ctx, Option<u64>) -> Fut,
    Fut: Future<Output = Result<R::Out, ThoError>>,
{
    async fn run(
        &mut self,
        task: Arc<R>,
        selector: Option<S>,
        device_idx: usize,
        run_ctx: R::Ctx,
        job_id_to_rerun: Option<u64>,
    ) -> Result<R::Out, ThoError> {
        let runner = &self.runner;

        panic_to_err_async!({
            runner(task, Arc::clone(&self.input), selector, device_idx, run_ctx, job_id_to_rerun)
        })
    }
}

struct ConsumeRunnerRetryAdapter<W, Run> {
    input: Option<W>,
    runner: Run,
}

impl<'z, R, S, W, Run, Fut> RunnerRetryAdapter<'z, R, S> for ConsumeRunnerRetryAdapter<W, Run>
where
    R: Stage,
    W: ConsumeData<O = R::In<'z>>,
    Run: Fn(Arc<R>, W, Option<S>, usize, R::Ctx, Option<u64>) -> Fut,
    Fut: Future<Output = ConsumeResult<R::In<'z>, R::Out>>,
{
    async fn run(
        &mut self,
        task: Arc<R>,
        selector: Option<S>,
        device_idx: usize,
        run_ctx: R::Ctx,
        job_id_to_rerun: Option<u64>,
    ) -> Result<R::Out, ThoError> {
        let input = self.input.take().expect("ConsumeRunnerRetryAdapter::run: missing input");
        let runner = &self.runner;

        let run_result = match futures::FutureExt::catch_unwind(std::panic::AssertUnwindSafe({
            runner(task, input, selector, device_idx, run_ctx, job_id_to_rerun)
        }))
        .await
        {
            Ok(run_result) => run_result,
            Err(panic) => std::panic::resume_unwind(panic),
        };

        match run_result {
            ConsumeResult::Ok(output) => Ok(output),
            ConsumeResult::Failed { input, error } => {
                self.input = Some(W::burp(input).await);
                Err(error)
            }
        }
    }

    fn requires_all_done(&self) -> bool {
        true
    }
}

async fn apply_adjuster<'a, R, IC, RC, RCC, S>(
    adjuster: &Adjusters<'a, R, IC, RC, RCC, S>,
    task: &mut Option<Arc<R>>,
    run_context: &mut Option<R::Ctx>,
    input_select_adjusted: &mut Option<S>,
    contexts: &mut Contexts<'a, R, IC, RC, RCC>,
) -> Result<(), ThoError>
where
    R: Adjustable + Stage + 'a,
    R::Ctx: Adjustable,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    S: Adjustable + Debug + Clone + Default + 'static,
    for<'i> <R as Stage>::In<'i>: WithId + Send + Sync,
    R::Out: WithId + Send + Sync + 'static,
{
    match adjuster {
        Adjusters::Input(input_ad) => {
            let base_selector = input_select_adjusted.clone().unwrap_or_default();

            match input_ad.adjust(base_selector, &mut contexts.input).await {
                Ok(v) => {
                    debug!("OOMHandler::run, error, input adjuster ok: {v:?}");
                    *input_select_adjusted = Some(v);
                    Ok(())
                }
                Err(adjuster_err) => {
                    let msg = format!("adjuster failed: {adjuster_err:?}");
                    debug!("OOMHandler::run, error, input adjuster failed: {msg:?}");

                    match adjuster_err {
                        AdjusterError::AdjustFailed(v) => {
                            *input_select_adjusted = Some(v);
                            Err(ThoError::new_with_error(msg, Cause::AdjusterFailed))
                        }
                        AdjusterError::SomethingElse => Err(ThoError::new_with_error(msg, Cause::SomethingElse("zz".to_string()))),
                    }
                }
            }
        }
        Adjusters::Task(task_ad) => match task_ad.adjust(task.take().expect("apply_adjuster: missing task"), &mut contexts.runnable).await {
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
                        Err(ThoError::new_with_error(msg, Cause::AdjusterFailed))
                    }
                    AdjusterError::SomethingElse => Err(ThoError::new_with_error(msg, Cause::SomethingElse("zz".to_string()))),
                }
            }
        },
        Adjusters::RunContext(context_ad) => {
            match context_ad
                .adjust(run_context.take().expect("apply_adjuster: missing run context"), &mut contexts.run_ctx)
                .await
            {
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
                            Err(ThoError::new_with_error(msg, Cause::AdjusterFailed))
                        }
                        AdjusterError::SomethingElse => Err(ThoError::new_with_error(msg, Cause::SomethingElse("zz".to_string()))),
                    }
                }
            }
        }
    }
}

async fn run_with_runner_impl<'a, 'z, R, IC, RC, RCC, S, A>(
    adjusters: &[Adjusters<'a, R, IC, RC, RCC, S>],
    task: Arc<R>,
    input_id_dbg: String,
    out_key: String,
    run_context: R::Ctx,
    mut contexts: Contexts<'a, R, IC, RC, RCC>,
    mut runner: A,
) -> Result<R::Out, ThoError>
where
    R: Adjustable + Stage + 'a,
    R::Ctx: Adjustable,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    S: Adjustable + Debug + Clone + Default + 'static,
    for<'i> <R as Stage>::In<'i>: WithId + Send + Sync,
    R::Out: WithId + Send + Sync + 'static,
    A: RunnerRetryAdapter<'z, R, S>,
{
    let (device_idx, ctx_id) = (run_context.device_idx(), run_context.id().clone());
    let mut exclusive_job = ExclusiveJobGuard::new(device_idx);
    let mut task = Some(task);
    let mut run_context = Some(run_context);

    debug!(
        "OOMHandler::run: 0.  ctx_id={ctx_id:?} task={:?} os_thread_id={} input.id={input_id_dbg}",
        task.as_ref().expect("run_with_runner_impl: missing task"),
        os_thread_id(),
    );

    debug!(
        "OOMHandler::run: 1. ctx_id={ctx_id:?} task={:?} os_thread_id={} acquired_permits={:?}",
        task.as_ref().expect("run_with_runner_impl: missing task"),
        os_thread_id(),
        run_context.as_ref().expect("run_with_runner_impl: missing run context").max_other_parallel(),
    );

    let mut i = 0;
    let mut input_select_adjusted: Option<S> = None;

    match runner
        .run(
            Arc::clone(task.as_ref().expect("run_with_runner_impl: missing task")),
            None,
            device_idx,
            run_context
                .as_mut()
                .expect("run_with_runner_impl: missing run context")
                .branch(Some("oom_handler_run_initial"), false),
            None,
        )
        .await
    {
        Err(te) => {
            exclusive_job.update_from_error(&te);
            debug!(
                "OOMHandler::run, error: 0. ctx_id={ctx_id:?} task={:?} os_thread_id={}  te={te:?}",
                task.as_ref().expect("run_with_runner_impl: missing task"),
                os_thread_id(),
            );

            if is_retryable_oom(&te) {
                pref_oom_count();
                debug!(
                    "OOMHandler::run, error: 1. ctx_id={ctx_id:?} task={:?} os_thread_id={} before adjusters",
                    task.as_ref().expect("run_with_runner_impl: missing task"),
                    os_thread_id(),
                );

                for adjuster in adjusters {
                    debug!(
                        "OOMHandler::run, error: 2. ctx_id={ctx_id:?} task={:?} adjuster={adjuster:?} os_thread_id={}  starting adjuster",
                        task.as_ref().expect("run_with_runner_impl: missing task"),
                        os_thread_id(),
                    );

                    loop {
                        i += 1;
                        if i > 10000 {
                            panic!("OOMHandler: ctx_id={ctx_id:?}  i={i}");
                        }

                        debug!(
                            "OOMHandler::run, error: 3. ctx_id={ctx_id:?} task={:?} os_thread_id={} oom i={i} in loop",
                            task.as_ref().expect("run_with_runner_impl: missing task"),
                            os_thread_id(),
                        );

                        let adjust_result = apply_adjuster(
                            adjuster,
                            &mut task,
                            &mut run_context,
                            &mut input_select_adjusted,
                            &mut contexts,
                        )
                        .await;

                        debug!(
                            "OOMHandler::run, error: on adjust_result ={adjust_result:?} os_thread_id={} adjuster error ",
                            os_thread_id()
                        );

                        if let Err(te) = adjust_result {
                            exclusive_job.update_from_error(&te);
                            debug!(
                                "OOMHandler::run, error: on ctx_id={ctx_id:?} task={:?} adjuster={adjuster:?} os_thread_id={} adjuster error te={te:?}",
                                task.as_ref().expect("run_with_runner_impl: missing task"),
                                os_thread_id(),
                            );

                            if is_retryable_oom(&te) {
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
                            "OOMHandler::run, error: ctx_id={ctx_id:?} task={:?} os_thread_id={}  1. input_select_adjusted={input_select_adjusted:?} acquired_permits={:?}",
                            task.as_ref().expect("run_with_runner_impl: missing task"),
                            os_thread_id(),
                            run_context.as_ref().expect("run_with_runner_impl: missing run context").max_other_parallel(),
                        );

                        let run_result = runner
                            .run(
                                Arc::clone(task.as_ref().expect("run_with_runner_impl: missing task")),
                                input_select_adjusted.clone(),
                                device_idx,
                                run_context
                                    .as_mut()
                                    .expect("run_with_runner_impl: missing run context")
                                    .branch(None, false),
                                exclusive_job.job_id(),
                            )
                            .await;

                        match run_result {
                            Ok(v) => {
                                contexts.input.add_partial_output(v).await;

                                let all_done = contexts.input.all_done();
                                if runner.requires_all_done() {
                                    assert!(all_done, "not yet all_done");
                                }

                                debug!(
                                    "OOMHandler::run, error: ctx_id={ctx_id:?} task={:?} os_thread_id={}  ok adjust_context={contexts:?} all_done={all_done}",
                                    task.as_ref().expect("run_with_runner_impl: missing task"),
                                    os_thread_id(),
                                );

                                if all_done {
                                    collect_eta_record_oom_success();

                                    debug!("done in {i} steps");
                                    return contexts.input.build_final_output().await;
                                }
                            }
                            Err(te) => {
                                exclusive_job.update_from_error(&te);
                                debug!("OOMHandler::run, error: err te={te:?}");

                                if is_retryable_oom(&te) {
                                    pref_oom_count();
                                    debug!(
                                        "OOMHandler::run, error: on oom ctx_id={ctx_id:?} task={:?} os_thread_id={} adjust_context={contexts:?}",
                                        task.as_ref().expect("run_with_runner_impl: missing task"),
                                        os_thread_id(),
                                    );

                                    if !adjuster.degrade(&mut contexts) {
                                        debug!(
                                            "OOMHandler::run, inner error: on oom moving to next adjuster ctx_id={ctx_id:?} task={:?} os_thread_id={} adjust_context={contexts:?}",
                                            task.as_ref().expect("run_with_runner_impl: missing task"),
                                            os_thread_id(),
                                        );

                                        break;
                                    }
                                } else {
                                    warn!(
                                        "OOMHandler::run, inner error: ctx_id={ctx_id:?} task={:?} os_thread_id={} unexpected error",
                                        task.as_ref().expect("run_with_runner_impl: missing task"),
                                        os_thread_id(),
                                    );
                                    return Err(te);
                                }
                            }
                        }
                    }
                }
            } else {
                warn!(
                    "OOMHandler::run, error: ctx_id={ctx_id:?} task={:?} os_thread_id={} unexpected error: {te:?}",
                    task.as_ref().expect("run_with_runner_impl: missing task"),
                    os_thread_id(),
                );
                return Err(te);
            }

            collect_eta_record_oom_all_failed();

            warn!("OOMHandler::run, error: ctx_id={ctx_id:?} out_key={out_key:?} os_thread_id={} all failed", os_thread_id());
            #[cfg(tho_debug)]
            GlobalCtx::global().debug_caches(device_idx, "all failed").await;

            Err(Cause::SomethingElse(format!("all failed: task.out_key={out_key:?}")).into())
        }
        Ok(o) => {
            debug!(
                "OOMHandler::run: ctx_id={ctx_id:?} task={:?} os_thread_id={} done without oom ",
                task.as_ref().expect("run_with_runner_impl: missing task"),
                os_thread_id(),
            );

            Ok(o)
        }
    }
}

async fn run_with_runner<'a, 'z, R, IC, RC, RCC, S, W, Run, Fut>(
    adjusters: &[Adjusters<'a, R, IC, RC, RCC, S>],
    task: Arc<R>,
    input: Arc<W>,
    run_context: R::Ctx,
    contexts: Contexts<'a, R, IC, RC, RCC>,
    runner: Run,
) -> Result<R::Out, ThoError>
where
    R: Adjustable + Stage + 'a,
    R::Ctx: Adjustable,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    S: Adjustable + Debug + Clone + Default + 'static,
    for<'i> <R as Stage>::In<'i>: thoosa_shared::traits::WithId + Send + Sync,
    R::Out: WithId + Send + Sync + 'static,
    W: WithId<ID = <R::In<'z> as WithId>::ID>,
    Run: Fn(Arc<R>, Arc<W>, Option<S>, usize, R::Ctx, Option<u64>) -> Fut,
    Fut: Future<Output = Result<R::Out, ThoError>>,
{
    let input_id_dbg = format!("{:?}", input.id());
    let out_key = format!("stage_id={task:?} input_id={:?}", input.id());

    run_with_runner_impl(
        adjusters,
        task,
        input_id_dbg,
        out_key,
        run_context,
        contexts,
        SharedRunnerRetryAdapter { input, runner },
    )
    .await
}


async fn run_with_runner_consume<'a, 'z, R, IC, RC, RCC, S, W, Run, Fut>(
    adjusters: &[Adjusters<'a, R, IC, RC, RCC, S>],
    task: Arc<R>,
    input: W,
    run_context: R::Ctx,
    contexts: Contexts<'a, R, IC, RC, RCC>,
    runner: Run,
) -> Result<R::Out, ThoError>
where
    R: Adjustable + Stage + 'a,
    R::Ctx: Adjustable,
    IC: InputContext<'a, R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    S: Adjustable + Debug + Clone + Default + 'static,
    for<'i> <R as Stage>::In<'i>: WithId + Send + Sync,
    R::Out: WithId + Send + Sync + 'static,
    W: ConsumeData<O = <R as Stage>::In<'z>> + WithId<ID = <R::In<'z> as WithId>::ID>,
    Run: Fn(Arc<R>, W, Option<S>, usize, R::Ctx, Option<u64>) -> Fut,
    Fut: Future<Output = ConsumeResult<R::In<'z>, R::Out>>,
{
    let input_id_dbg = format!("{:?}", input.id());
    let out_key = format!("stage_id={task:?} input_id={:?}", input.id());

    run_with_runner_impl(
        adjusters,
        task,
        input_id_dbg,
        out_key,
        run_context,
        contexts,
        ConsumeRunnerRetryAdapter {
            input: Some(input),
            runner,
        },
    )
    .await
}

#[cfg(test)]
pub mod test_types {
    use super::*;
    use thoosa_named_derive::Named;
    // use crate::common::Data;
    use crate::runner::test_utils::DummyTensor;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::copy_mode::CpuOrGpu;
    use thoosa_shared::traits::WithId;

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

    impl Select<std::ops::Range<[usize;1]>> for TestInput {
        type Output = TestInput;

        fn subview(&self, selector: std::ops::Range<[usize; 1]>) -> Result<Self::Output, ThoError> {
            Ok(TestInput {
                id: self.id.clone(),
                data: DummyTensor(self.data.0[selector.start[0]..selector.end[0]].to_vec())//.0[selector.start..selector.end])
            })
        }
    }

    impl Input for TestInput {  }

    impl Adjustable for TestInput {}

    #[async_trait]
    impl Data for TestInput {
        type Metrics = [usize; 1];
        type M = CpuOrGpu<Self>;

        fn size(&self) -> Self::Metrics {
            [self.data.0.len()]
        }

        fn device(&self) -> usize {
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

    // impl Input for TestInput {}

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
            Self {
                id: "test".into(),
                v: v.into(),
            }
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
        type M = CpuOrGpu<Self>;

        fn size(&self) -> Self::Metrics {
            [1]
        }

        fn device(&self) -> usize {
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
            Ok(TestOutput {
                id: self.id.clone(),
                v: self.v.clone(),
            })
        }
    }

    impl WithId for TestOutput {
        type ID = TestId;
        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl Output for TestOutput {  }
}

#[cfg(test)]
pub mod mock_runnable {
    use super::test_types::{TestInput, TestOutput};
    use super::*;
    use crate::runner::common::mock_run_ctx::MockRunCtx;
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use thoosa_shared::traits::WithId;
    use crate::runner::common::runnable::{Stage, SyncStage};

    mock! {
        #[derive(Clone, Debug)]
        pub Runnable {}

        impl Stage for Runnable {
            type In<'a> = TestInput;
            type Out = TestOutput;
            type Ctx = MockRunCtx;
        }

        impl SyncStage for Runnable {
            fn do_run<'f>(self: Arc<Self>, _input: &<Self as Stage>::In<'f>, _ctx: &<Self as Stage>::Ctx) -> Result<<Self as Stage>::Out, ThoError>;
        }

    }

    impl Adjustable for MockRunnable {}

    impl OutId for MockRunnable {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
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
    use super::test_types::TestInput;
    use super::*;
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    // use crate::common::mock_input_output::MockInput;

    mock! {
        #[derive(Debug)]
        pub InputAdjuster {}
        #[async_trait]
        impl Adjuster<super::mock_input_context::MockInputContext> for InputAdjuster {
            type A = std::ops::Range<<TestInput as Data>::Metrics>;
            async fn adjust(&self, adjustable: <Self as Adjuster<super::mock_input_context::MockInputContext>>::A, adjust_context: &mut super::mock_input_context::MockInputContext) -> Result<<Self as Adjuster<super::mock_input_context::MockInputContext>>::A, AdjusterError<<Self as Adjuster<super::mock_input_context::MockInputContext>>::A>>;
            fn degrade(&self, adjust_context: &mut super::mock_input_context::MockInputContext) -> bool;
        }
    }

    pub fn build<I>(inputs: I) -> (MockInputAdjuster, Arc<AtomicUsize>)
    where
        I: IntoIterator<Item = std::ops::Range<[usize; 1]>>,
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
        pub TaskAdjuster {}
        #[async_trait]
        impl Adjuster<MockRunnableContext> for TaskAdjuster {
            type A = Arc<MockRunnable>;
            async fn adjust(&self, adjustable: <Self as Adjuster<MockRunnableContext>>::A, adjust_context: &mut MockRunnableContext) -> Result<<Self as Adjuster<MockRunnableContext>>::A, AdjusterError<<Self as Adjuster<MockRunnableContext>>::A>>;
            fn degrade(&self, adjust_context: &mut MockRunnableContext) -> bool;
        }
    }

    pub fn build<I>(tasks: I) -> (MockTaskAdjuster, Arc<AtomicUsize>)
    where
        I: IntoIterator<Item = MockRunnable>,
    {
        let mut adj = MockTaskAdjuster::new();
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
    use super::mock_input_adjuster::{build as mock_input_adjuster, MockInputAdjuster};
    use super::mock_runnable::build as mock_runnable;
    use super::mock_task_adjuster::build as mock_task_adjuster;
    use super::test_types::{TestInput, TestOutput};
    use super::{AdjusterError, Adjusters, AsyncOOMHandler, Contexts, OOMHandler};
    use crate::runner::common::cachable::DevicesWrap;
    // use crate::common::runnable::Runnable;
    use crate::runner::common::{DropIf, MockRunCtx, OutId};
    use crate::runner::common::runnable::{AsyncStage, Stage};
    use std::collections::VecDeque;
    use std::ops::Range;
    use std::sync::{Arc, Mutex};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use async_trait::async_trait;
    use thoosa_shared::{Cause, ThoError};
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use thoosa_shared::traits::{Adjustable, WithId};
    use super::{Adjuster, InputContext};
    type Selector = Range<[usize; 1]>;

    #[derive(Debug)]
    struct AsyncTestTask {
        results: Arc<Mutex<VecDeque<Result<TestOutput, ThoError>>>>,
    }

    impl AsyncTestTask {
        fn new(results: VecDeque<Result<TestOutput, ThoError>>) -> Self {
            Self {
                results: Arc::new(Mutex::new(results)),
            }
        }
    }

    impl Adjustable for AsyncTestTask {}

    impl Stage for AsyncTestTask {
        type In<'a> = TestInput;
        type Out = TestOutput;
        type Ctx = MockRunCtx;
    }

    impl AsyncStage for AsyncTestTask {
        async fn run(
            self: Arc<Self>,
            input: Arc<DevicesWrap<Self::In<'static>>>,
            _ctx: Self::Ctx,
        ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
            let next = self
                .results
                .lock()
                .unwrap()
                .pop_front()
                .unwrap_or_else(|| Err(Cause::SomethingElse("no more results".into()).into()));

            match next {
                Ok(o) => Ok(Arc::new(DevicesWrap::wrap(o))),
                Err(e) => Err(e),
            }
        }
    }

    impl OutId for AsyncTestTask {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            input_id.clone()
        }
    }

    #[derive(Debug)]
    struct AsyncInputCtx {
        outputs: Vec<TestOutput>,
    }

    impl InputContext<'static, AsyncTestTask> for AsyncInputCtx {
        async fn add_partial_output(&mut self, o: TestOutput) {
            self.outputs.push(o);
        }

        fn all_done(&self) -> bool {
            true
        }

        async fn build_final_output(self) -> Result<TestOutput, ThoError> {
            self.outputs.into_iter().next().ok_or_else(|| {
                ThoError::new("no output in AsyncInputCtx")
            })
        }
    }


    #[tokio::test]
    async fn test_input_task_context_once() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

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
            Err(ThoError::from(Cause::OutOfMemory)), //no adjusters
            Err(ThoError::from(Cause::OutOfMemory)), //input adjuster
        ]);

        let (adjusted_task, adjusted_calls) = mock_runnable([
            Err(ThoError::from(Cause::OutOfMemory)), //task adjuster
            Ok(TestOutput::new(111.)),                    //ctx adjuster
        ]);

        let (task_adj, task_adj_calls) = mock_task_adjuster([adjusted_task]);

        let ctx_mock = crate::runner::common::mock_run_ctx::build()
            .with_id(0)
            .with_device_idx(1)
            .with_max_other_parallel(1)
            .with_branch(4)
            .finish();

        let oom_handler = OOMHandler::new(
            vec![
                Adjusters::Input(Box::new(input_adj)),
                Adjusters::Task(Box::new(task_adj)),
                Adjusters::RunContext(Box::new(ctx_adj)),
            ],
            None,
        );

        let out: TestOutput = oom_handler
            .run(
                &Arc::new(init_task),
                Arc::new(DevicesWrap::wrap(TestInput::new(100))),
                ctx_mock,
                Contexts::new(
                    super::mock_input_context::build()
                        .with_partial_output(TestOutput::new(111.))
                        .with_all_done(&[true]) 
                        .with_build_final_output(TestOutput::new(111.))
                        .finish(),
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
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let (init_task, init_calls) = mock_runnable([Err(ThoError::from(Cause::OutOfMemory))]);
        let (final_task, final_calls) = mock_runnable([Ok(TestOutput::new(10.))]);
        let (task_adj, task_adj_calls) = mock_task_adjuster([final_task]);
        let oom_handler: OOMHandler<'_, 'static, _, _, _, _, Selector> = OOMHandler::new(vec![Adjusters::Task(Box::new(task_adj))], None);

        let ctx_mock = crate::runner::common::mock_run_ctx::build()
            .with_id(0)
            .with_device_idx(1)
            .with_max_other_parallel(1)
            .with_branch(4)
            .finish();

        let out: TestOutput = oom_handler
            .run(
                &Arc::new(init_task),
                Arc::new(DevicesWrap::wrap(TestInput::new(1))),
                ctx_mock,
                Contexts::new(
                    super::mock_input_context::build()
                        .with_partial_output(TestOutput::new(10.))
                        .with_all_done(&[true]) //TODO should
                        .with_build_final_output(TestOutput::new(10.))
                        .finish(),
                    super::MockRunnableContext::new(),
                    super::MockRunCtxContext::new(),
                ),
            )
            .await
            .unwrap();
        assert_eq!(10., out.v.0);
        assert_eq!(1, task_adj_calls.load(Ordering::SeqCst));
        assert_eq!(1, init_calls.load(Ordering::SeqCst));
        assert_eq!(1, final_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_only_input_adjuster() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let (task, task_calls) = mock_runnable([
            Err(ThoError::from(Cause::OutOfMemory)),
            Ok(TestOutput::new(10.)),
        ]);
        let (input_adj, input_calls) = mock_input_adjuster([[0]..[1]]);
        let oom_handler = OOMHandler::new(vec![Adjusters::Input(Box::new(input_adj))], None);
        let ctx_mock = crate::runner::common::mock_run_ctx::build()
            .with_id(0)
            .with_device_idx(1)
            .with_max_other_parallel(1)
            .with_branch(4)
            .finish();

        let out = oom_handler
            .run(
                &Arc::new(task),
                Arc::new(DevicesWrap::wrap(TestInput::new(1))),
                ctx_mock,
                Contexts::new(
                    super::mock_input_context::build()
                        .with_partial_output(TestOutput::new(10.))
                        .with_all_done(&[true])
                        .with_build_final_output(TestOutput::new(10.))
                        .finish(),
                    super::MockRunnableContext::new(),
                    super::MockRunCtxContext::new(),
                ),
            )
            .await
            .unwrap();
        assert_eq!(10., out.v.0);
        assert_eq!(1, input_calls.load(Ordering::SeqCst));
        assert_eq!(2, task_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_only_run_ctx_adjuster() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let (task, task_calls) = mock_runnable([
            Err(ThoError::from(Cause::OutOfMemory)),
            Ok(TestOutput::new(10.)),
        ]);
        let (ctx_adj, ctx_adj_calls) = mock_context_adjuster(1);

        let oom_handler: OOMHandler<'_, 'static, _, _, _, _, Selector> = OOMHandler::new(vec![Adjusters::RunContext(Box::new(ctx_adj))], None);

        let ctx_mock = crate::runner::common::mock_run_ctx::build()
            .with_id(0)
            .with_device_idx(1)
            .with_max_other_parallel(1)
            .with_branch(1)
            .finish();

        let out = oom_handler
            .run(
                &Arc::new(task),
                Arc::new(DevicesWrap::wrap(TestInput::new(1))),
                ctx_mock,
                Contexts::new(
                    super::mock_input_context::build()
                        .with_partial_output(TestOutput::new(10.))
                        .with_all_done(&[true]) //TODO should
                        .with_build_final_output(TestOutput::new(10.))
                        .finish(),
                    super::MockRunnableContext::new(),
                    super::MockRunCtxContext::new(),
                ),
            )
            .await
            .unwrap();
        assert_eq!(10., out.v.0);
        assert_eq!(1, ctx_adj_calls.load(Ordering::SeqCst));
        assert_eq!(2, task_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_parallel_tasks() {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

        let (init_task1, init_calls1) =
            mock_runnable([Err(ThoError::from(Cause::OutOfMemory))]);
        let (final_task1, final_calls1) = mock_runnable([Ok(TestOutput::new(5.))]);

        let (task_adj1, task_adj_calls1) = mock_task_adjuster([final_task1]);

        let (init_task2, init_calls2) =
            mock_runnable([Err(ThoError::from(Cause::OutOfMemory))]);
        let (final_task2, final_calls2) = mock_runnable([Ok(TestOutput::new(7.))]);

        let (task_adj2, task_adj_calls2) = mock_task_adjuster([final_task2]);

        let init_task1_arc = Arc::new(init_task1);
        let input_arc = Arc::new(DevicesWrap::wrap(TestInput::new(1)));

        let ctx_mock_1 = crate::runner::common::mock_run_ctx::build()
            .with_id(0)
            .with_device_idx(1)
            .with_max_other_parallel(1)
            .with_branch(4)
            .finish();
        let ctx_mock_2 = crate::runner::common::mock_run_ctx::build()
            .with_id(0)
            .with_device_idx(1)
            .with_max_other_parallel(1)
            .with_branch(4)
            .finish();

        let oom1 = OOMHandler::new(vec![Adjusters::Task(Box::new(task_adj1))], None);
        let h1 = oom1.run(
            &init_task1_arc,
            Arc::clone(&input_arc),
            ctx_mock_1,
            Contexts::new(
                super::mock_input_context::build()
                    .with_partial_output(TestOutput::new(5.))
                    .with_all_done(&[true]) //TODO should
                    .with_build_final_output(TestOutput::new(5.))
                    .finish(),
                super::MockRunnableContext::new(),
                super::MockRunCtxContext::new(),
            ),
        );
        let init_task2_arc = Arc::new(init_task2);

        let oom2 = OOMHandler::new(vec![Adjusters::Task(Box::new(task_adj2))], None);
        let h2 = oom2.run(
            &init_task2_arc,
            Arc::clone(&input_arc),
            ctx_mock_2,
            Contexts::new(
                super::mock_input_context::build()
                    .with_partial_output(TestOutput::new(7.))
                    .with_all_done(&[true]) //TODO should
                    .with_build_final_output(TestOutput::new(7.))
                    .finish(),
                super::MockRunnableContext::new(),
                super::MockRunCtxContext::new(),
            ),
        );
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
