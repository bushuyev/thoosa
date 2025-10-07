use crate::common::{PermitsCtx, RunCtx, Select};
use crate::common::cachable::DataCacheWrapper;
use crate::common::runnable::Stage;
use crate::common::{Data, DropIf, OutId};
#[double]
use crate::global_ctx::GlobalCtx;
use crate::permits::MAX_PERMITS;
use crate::statistics::Statistics;
use async_trait::async_trait;
use log::debug;
use mockall::automock;
use mockall_double::double;
use parking_lot::RwLock;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::Range;
use std::process::Output;
use std::sync::Arc;
use thoosa_shared::panic_to_err_async;
use thoosa_shared::CopyToDevice;
use thoosa_shared::traits::{CachableIdTr, Named, WithId};
use thoosa_shared::{MyError, ThoosaError};
use tracing::instrument;

pub trait Adjustable: Send + Sync {}

impl<A> Adjustable for Arc<RwLock<A>> where A: Adjustable {}
impl<T: Send + Sync> Adjustable for std::ops::Range<T> {}

#[async_trait]
pub trait Adjuster<AC>: Debug + Send + Sync {
    type A: Adjustable;

    async fn adjust(
        &self,
        adjustable: &Self::A,
        adjust_context: &mut AC,
    ) -> Result<Self::A, ThoosaError>;

    fn degrade(&self, _adjust_context: &mut AC) -> bool {
        false
    }
}

pub trait InputContext<R>: Debug + Send + Sync
where
    R: Stage + 'static,
    <R as Stage>::In<'static>: Data,
{
    fn add_partial_output(&mut self, o: R::Out);
    fn all_done(&self) -> bool;
    fn input_len(&self) -> &<<R as Stage>::In<'static> as Data>::Metrics;
    fn build_final_output(self) -> R::Out;
}

#[cfg(test)]
pub mod mock_input_context {
    use crate::oom_handler::mock_runnable::MockRunnable;
    use crate::oom_handler::test_types::TestOutput;
    use mockall::mock;

    mock! {
        #[derive(Debug)]
        pub InputContext {

        }
        impl super::InputContext<MockRunnable> for InputContext{
            fn add_partial_output(&mut self, o: TestOutput);
            fn all_done(&self) -> bool;
            fn input_len(&self) -> &[usize; 1];
            fn build_final_output(self) -> TestOutput;
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
            self.mock.expect_build_final_output().return_once(|| inp);
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
pub struct Contexts<R, IC, RC, RCC>
where
    R: Stage + 'static,
    <R as Stage>::In<'static>: Data,
    IC: InputContext<R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
{
    input: IC,
    runnable: RC,
    run_ctx: RCC,
    pd: PhantomData<R>,
}

impl<R, IC, RC, RCC> Contexts<R, IC, RC, RCC>
where
    R: Stage + 'static,
    <R as Stage>::In<'static>: Data,
    IC: InputContext<R>,
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
    R: Stage + 'static,
    <R as Stage>::In<'static>: Data + Select<S> + Debug,
    IC: InputContext<R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    <R as Stage>::In<'static>: Select<S>,
    S: Adjustable + Debug + Default + Clone
{
    Input(Box<dyn Adjuster<IC, A =S> + 'a>),
    Task(Box<dyn Adjuster<RC, A = R> + 'a>),
    RunContext(Box<dyn Adjuster<RCC, A = Arc<RwLock<R::Ctx>>> + 'a>),
}

impl<'a, R, IC, RC, RCC, S> Adjusters<'a, R, IC, RC, RCC, S>
where
    R: Stage + 'static,
    IC: InputContext<R>,
    R: Adjustable,
    <R as Stage>::In<'static>: Data + Debug,
    RC: RunnableContext,
    RCC: RunCtxContext,
    <R as Stage>::Ctx: Adjustable,
    <R as Stage>::In<'static>: Select<S>,
    S: Adjustable + Debug + Default + Clone
{
    fn degrade(&self, adjust_context: &mut Contexts<R, IC, RC, RCC>) -> bool {
        match self {
            Adjusters::Input(v) => v.degrade(&mut adjust_context.input),
            Adjusters::Task(v) => v.degrade(&mut adjust_context.runnable),
            Adjusters::RunContext(v) => v.degrade(&mut adjust_context.run_ctx),
        }
    }
}

#[derive(Debug)]
pub struct NoopInputContext<R: Stage + 'static>(pub Option<R::Out>);

impl<R> InputContext<R> for NoopInputContext<R>
where
    <R as Stage>::In<'static>: Data,
    <R as Stage>::Out: Send + Sync,
    R: Stage + 'static,
{
    fn add_partial_output(&mut self, o: R::Out) {
        self.0.replace(o);
    }

    fn all_done(&self) -> bool {
        self.0.is_some()
    }

    fn input_len(&self) -> &<<R as Stage>::In<'static> as Data>::Metrics {
        unimplemented!()
    }

    fn build_final_output(self) -> R::Out {
        self.0.unwrap()
    }
}

#[derive(Debug)]
pub struct NoopRunnableContext {}
impl RunnableContext for NoopRunnableContext {}

#[derive(Debug)]
pub struct NoopRunCtxContext {}

impl RunCtxContext for NoopRunCtxContext {}

#[derive(Debug)]
pub struct OOMHandler<'a, R, IC, RC, RCC, S>
where
    R: Stage + Adjustable + OutId + 'static,
    <R as Stage>::In<'static>: Data + WithId,
    R::Ctx: Adjustable,
    IC: InputContext<R>,
    RC: RunnableContext,
    RCC: RunCtxContext,
    S: Adjustable + Debug + Clone + Default,
    <R as Stage>::In<'static>: Select<S, Output=<R as Stage>::In<'static>>
{
    adjusters: Vec<Adjusters<'a, R, IC, RC, RCC, S>>,
    #[allow(dead_code)]
    statistics: Option<&'a Arc<Statistics>>,
    pd_rc: PhantomData<RC>,
    pd_rcc: PhantomData<RCC>,
}

impl<'a, R, IC, RC, RCC, S> OOMHandler<'a, R, IC, RC, RCC, S>
where
    R: Stage + Adjustable + OutId + 'static,
    <R as Stage>::In<'static>: Data + WithId,
    <R as Stage>::In<'static>: Debug + Data + CopyToDevice,
    R::Ctx: Adjustable,
    IC: InputContext<R> + Debug,
    RC: RunnableContext + Debug,
    RCC: RunCtxContext + Debug,
    S: Adjustable + Debug + Clone + Default,//Selector (like range for a tensor)
    <R as Stage>::In<'static>: Select<S, Output=<R as Stage>::In<'static>> //Select<S... - something (ThoTensor) from what a part can be selected
{
    pub fn new(
        adjusters: Vec<Adjusters<'a, R, IC, RC, RCC, S>>,
        statistics: Option<&'a Arc<Statistics>>,
    ) -> Self {
        Self {
            adjusters,
            statistics,
            pd_rc: Default::default(),
            pd_rcc: Default::default(),
        }
    }

    #[instrument]
    pub async fn run(
        &self,
        task: &Arc<R>,
        input: Arc<DataCacheWrapper<<R as Stage>::In<'static>>>,
        run_context: Arc<RwLock<R::Ctx>>,
        mut contexts: Contexts<R, IC, RC, RCC>,
    ) -> Result<R::Out, ThoosaError>
    where
        <R as Stage>::In<'static>: WithId +  Select<S, Output=<R as Stage>::In<'static>> + Send + Sync,
    {
        let (device_idx, ctx_name) = {
            let guard = run_context.read();
            (guard.device_idx(), guard.name())
        };
        debug!(
            "OOMHandler::run: ctx_name={ctx_name} input.id={:?}",
            input.id()
        );

        // Identify this task by its output id for Arbiter fairness.
        let out_key = {
            let out_id = task.out_id(input.id());
            std::sync::Arc::new(out_id) as std::sync::Arc<dyn CachableIdTr>
        };

        if run_context.read().acquired_permits() == 0 {
            let result = GlobalCtx::global()
                .acquire_many(device_idx, 1, out_key.clone())
                .await?;
            run_context.write().add_permit(result);
        }

        debug!(
            "OOMHandler::run: ctx_name={ctx_name}  acquired_permits={:?}",
            run_context.read().acquired_permits()
        );

        let mut i = 0;

        let mut last_input_adjuster = None::<&Box<dyn Adjuster<IC, A = S>>>;
        let mut last_task_adjuster = None::<&Box<dyn Adjuster<RC, A = R>>>;
        let mut last_context_adjuster = None::<&Box<dyn Adjuster<RCC, A = Arc<RwLock<R::Ctx>>>>>;

        let mut input_select_adjusted = None;

        match panic_to_err_async!(
            input
                .with_data_ref(device_idx, None, |data| {
                    Arc::clone(task).do_run(data, Arc::clone(&run_context))
                })
                .await
        ) {
            Ok(o) => Ok(o),
            Err(te) => {
                match te.what() {
                    MyError::OutOfMemory => {
                        debug!("OOMHandler::run: ctx_name={ctx_name} oom 0");

                        let mut task_adjusted = None::<Arc<R>>;

                        for adjuster in &self.adjusters {
                            loop {
                                i += 1;
                                if i > 10000 {
                                    //TODO
                                    panic!("OOMHandler: ctx_name={ctx_name}  i={i}");
                                }
                                debug!("OOMHandler::run: ctx_name={ctx_name} oom i={i}");

                                match adjuster {
                                    Adjusters::Input(input_ad) => {
                                        if last_input_adjuster.is_some() {
                                            last_input_adjuster = None;
                                        }

                                        input_select_adjusted = Some(
                                            input_ad
                                                .adjust(
                                                    input_select_adjusted
                                                        .as_ref()
                                                        .unwrap_or(&S::default())/*.unwrap_or(input_guard_op.as_ref().unwrap())*/,
                                                    &mut contexts.input,
                                                )
                                                .await?,
                                        );
                                    }

                                    Adjusters::Task(task_ad) => {
                                        if last_task_adjuster.is_some() {
                                            last_task_adjuster = None;
                                        }
                                        task_adjusted.replace(Arc::new(
                                            task_ad.adjust(task, &mut contexts.runnable).await?,
                                        ));
                                    }

                                    Adjusters::RunContext(context_ad) => {
                                        if last_context_adjuster.is_some() {
                                            last_context_adjuster = None;
                                        }

                                        context_ad
                                            .adjust(&run_context, &mut contexts.run_ctx)
                                            .await?;
                                    }
                                };

                                let task_to_run = task_adjusted.as_ref().unwrap_or(task);

                                if run_context.read().acquired_permits() == 0 {
                                    let result = GlobalCtx::global()
                                        .acquire_many(device_idx, 1, out_key.clone())
                                        .await?;
                                    run_context.write().add_permit(result);
                                }

                                debug!(
                                    "OOMHandler::run: ctx_name={ctx_name} 1. input_select_adjusted={input_select_adjusted:?} acquired_permits={:?}",
                                    run_context.read().acquired_permits()
                                );

                                let run_result = if let Some(selector) =
                                    input_select_adjusted.as_ref()
                                {
                                    panic_to_err_async!(input.with_selected_data_ref(device_idx, selector.clone(), None, |data| {
                                        Arc::clone(task_to_run).do_run(&data, Arc::clone(&run_context))
                                    }).await)

                                } else {
                                    panic_to_err_async!(input.with_data_ref(device_idx, None, |data| {
                                        Arc::clone(task_to_run).do_run(data, Arc::clone(&run_context))
                                    }).await)

                                };

                                match run_result
                                {
                                    Ok(v) => {
                                        contexts.input.add_partial_output(v);

                                        let all_done = contexts.input.all_done();

                                        debug!(
                                            "OOMHandler::run: ctx_name={ctx_name}  ok adjust_context={contexts:?} all_done={all_done}"
                                        );

                                        if all_done {
                                            run_context.write().drop_permits(DropIf::Always);

                                            debug!("done in {i} steps");
                                            return Ok(contexts.input.build_final_output());
                                        }
                                    }
                                    Err(te) => {
                                        debug!("OOMHandler::run: err te={te:?}");

                                        if run_context.read().acquired_permits() == MAX_PERMITS {
                                            GlobalCtx::global()
                                                .dump_all_to_cpu(&HashMap::new(), device_idx)
                                                .await?;
                                        }

                                        run_context.write().drop_permits(DropIf::IfNotMax); //TODO config? - not needed for runner_semaphore

                                        match te.what() {
                                            MyError::OutOfMemory => {
                                                debug!("OOMHandler::run: err ctx_name={ctx_name}  adjust_context={contexts:?}");

                                                if !adjuster.degrade(&mut contexts) {
                                                    match adjuster {
                                                        Adjusters::Input(v) => {
                                                            last_input_adjuster.replace(v);
                                                        }
                                                        Adjusters::Task(v) => {
                                                            last_task_adjuster.replace(v);
                                                        }
                                                        Adjusters::RunContext(v) => {
                                                            last_context_adjuster.replace(v);
                                                        }
                                                    }

                                                    break;
                                                }
                                            }
                                            e => {
                                                debug!("OOMHandler::run: ctx_name={ctx_name} other error 1");
                                                return Err(te);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    _ => {
                        debug!("OOMHandler::run: ctx_name={ctx_name} other error 2");
                        return Err(te);
                    }
                }

                Err(MyError::SomethingElse(format!("all failed: task={task:?} input={input:?}")).into())
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test_types {
    use super::*;
    use crate::common::Data;
    use crate::test_utils::DummyTensor;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::traits::WithId;

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    pub struct TestId(&'static str);

    impl Named for TestId {
        fn name(&self) -> &'static str {
            "TestId"
        }
    }

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

    impl Select<Range<[usize;1]>> for TestInput {
        type Output = TestInput;

        fn subview(&self, selector: Range<[usize; 1]>) -> Result<Self::Output, ThoosaError> {
            Ok(TestInput {
                id: self.id.clone(),
                data: DummyTensor(self.data.0[selector.start[0]..selector.end[0]].to_vec())//.0[selector.start..selector.end])
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

        fn device(&self) -> usize {
            0
        }
    }

    #[async_trait]
    impl CopyToDevice for TestInput {
        async fn copy_to_device(&self, _dst_device: usize) -> Self {
            TestInput {
                id: self.id.clone(),
                data: DummyTensor(self.data.0.clone()),
            }
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

        fn size(&self) -> Self::Metrics {
            [1]
        }

        fn device(&self) -> usize {
            0
        }
    }

    #[async_trait]
    impl CopyToDevice for TestData {
        async fn copy_to_device(&self, _dst_device: usize) -> Self {
            *self
        }
    }

    #[async_trait]
    impl CopyToDevice for TestOutput {
        async fn copy_to_device(&self, _dst_device: usize) -> Self {
            TestOutput {
                id: self.id.clone(),
                v: self.v.clone(),
            }
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
pub(crate) mod mock_runnable {
    use super::test_types::{TestInput, TestOutput};
    use super::*;
    use crate::common::MockRunCtx;
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use thoosa_shared::traits::WithId;

    mock! {
        #[derive(Clone, Debug)]
        pub Runnable {}
        #[async_trait]
        impl Stage for Runnable {
            type In<'a> = TestInput;
            type Out = TestOutput;
            type Ctx = MockRunCtx;

            fn do_run<'a>(self: Arc<Self>, input:&<MockRunnable as Stage>::In<'a>, tx: Arc<RwLock<<MockRunnable as Stage>::Ctx>>) -> Result<<MockRunnable as Stage>::Out, ThoosaError> where Self:Sized;
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
        I: IntoIterator<Item = Result<TestOutput, ThoosaError>>,
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
pub(crate) mod mock_input_adjuster {
    use super::test_types::TestInput;
    use super::*;
    use crate::test_utils::MockDataWithId;
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
            async fn adjust(&self, adjustable: &<Self as Adjuster<super::mock_input_context::MockInputContext>>::A, adjust_context: &mut super::mock_input_context::MockInputContext) -> Result<<Self as Adjuster<super::mock_input_context::MockInputContext>>::A, ThoosaError>;
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
pub(crate) mod mock_task_adjuster {
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
            type A = MockRunnable;
            async fn adjust(&self, adjustable: &<Self as Adjuster<MockRunnableContext>>::A, adjust_context: &mut MockRunnableContext) -> Result<<Self as Adjuster<MockRunnableContext>>::A, ThoosaError>;
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
                Ok(task.unwrap())
            });
        }
        adj.expect_degrade().return_const(false);
        (adj, calls)
    }
}

#[cfg(test)]
pub(crate) mod mock_context_adjuster {
    use super::*;
    use crate::common::MockRunCtx;
    use mockall::mock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    mock! {
        #[derive(Debug)]
        pub ContextAdjuster {}
        #[async_trait]
        impl Adjuster<MockRunCtxContext> for ContextAdjuster {
            type A = Arc<RwLock<MockRunCtx>>;
            async fn adjust(&self, adjustable: &<Self as Adjuster<MockRunCtxContext>>::A, adjust_context: &mut MockRunCtxContext) -> Result<<Self as Adjuster<MockRunCtxContext>>::A, ThoosaError>;
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
                Ok(rc.clone())
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
    use super::{Adjustable, Adjusters, Contexts, OOMHandler};
    use crate::common::cachable::DataCacheWrapper;
    // use crate::common::runnable::Runnable;
    use crate::common::{DropIf, MockRunCtx};
    use crate::global_ctx;
    use crate::global_ctx::{MockGlobalCtx, mock_global_ctx};
    use parking_lot::RwLock;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use thoosa_shared::{MyError, ThoosaError};

    impl Adjustable for MockRunCtx {}

    #[tokio::test]
    async fn test_input_task_context_once() {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build()
                    .with_age(0)
                    .with_dump_all_to_cpu()
                    .with_reset_all_on_cpu()
                    .finish(),
            )) as &'static global_ctx::MockGlobalCtx
        });

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
            Err(ThoosaError::from(MyError::OutOfMemory)), //no adjusters
            Err(ThoosaError::from(MyError::OutOfMemory)), //input adjuster
        ]);

        let (adjusted_task, adjusted_calls) = mock_runnable([
            Err(ThoosaError::from(MyError::OutOfMemory)), //task adjuster
            Ok(TestOutput::new(111.)),                    //ctx adjuster
        ]);

        let (task_adj, task_adj_calls) = mock_task_adjuster([adjusted_task]);

        let ctx_mock = crate::common::mock_run_ctx::build()
            .with_device_idx(1)
            .with_acquired_permits(1)
            .with_drop_permits(vec![DropIf::IfNotMax])
            .finish();

        let oom_handler = OOMHandler::new(
            vec![
                Adjusters::Input(Box::new(input_adj)),
                Adjusters::Task(Box::new(task_adj)),
                Adjusters::RunContext(Box::new(ctx_adj)),
            ],
            None,
        );

        let out = oom_handler
            .run(
                &Arc::new(init_task),
                Arc::new(DataCacheWrapper::new(TestInput::new(100), 1, 2, 0)),
                Arc::new(RwLock::new(ctx_mock)),
                Contexts::new(
                    super::mock_input_context::build()
                        .with_partial_output(TestOutput::new(111.))
                        .with_all_done(&[true]) //TODO should
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
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build()
                    .with_age(0)
                    .with_dump_all_to_cpu()
                    .with_reset_all_on_cpu()
                    .finish(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let (init_task, init_calls) = mock_runnable([Err(ThoosaError::from(MyError::OutOfMemory))]);
        let (final_task, final_calls) = mock_runnable([Ok(TestOutput::new(10.))]);
        let (task_adj, task_adj_calls) = mock_task_adjuster([final_task]);
        let oom_handler = OOMHandler::new(vec![Adjusters::Task(Box::new(task_adj))], None);

        let ctx_mock = crate::common::mock_run_ctx::build()
            .with_device_idx(1)
            .with_acquired_permits(1)
            .with_drop_permits(vec![DropIf::IfNotMax])
            .finish();

        let out = oom_handler
            .run(
                &Arc::new(init_task),
                Arc::new(DataCacheWrapper::new(TestInput::new(1), 1, 2, 0)),
                Arc::new(RwLock::new(ctx_mock)),
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
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build()
                    .with_age(0)
                    .with_dump_all_to_cpu()
                    .with_reset_all_on_cpu()
                    .finish(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let (task, task_calls) = mock_runnable([
            Err(ThoosaError::from(MyError::OutOfMemory)),
            Ok(TestOutput::new(10.)),
        ]);
        let (input_adj, input_calls) = mock_input_adjuster([[0]..[1]]);
        let oom_handler = OOMHandler::new(vec![Adjusters::Input(Box::new(input_adj))], None);
        let ctx_mock = crate::common::mock_run_ctx::build()
            .with_device_idx(1)
            .with_acquired_permits(1)
            .with_drop_permits(vec![DropIf::IfNotMax])
            .finish();

        let out = oom_handler
            .run(
                &Arc::new(task),
                Arc::new(DataCacheWrapper::new(TestInput::new(1), 1, 2, 0)),
                Arc::new(RwLock::new(ctx_mock)),
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
        assert_eq!(1, input_calls.load(Ordering::SeqCst));
        assert_eq!(2, task_calls.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn test_only_run_ctx_adjuster() {
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build()
                    .with_age(0)
                    .with_dump_all_to_cpu()
                    .with_reset_all_on_cpu()
                    .finish(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let (task, task_calls) = mock_runnable([
            Err(ThoosaError::from(MyError::OutOfMemory)),
            Ok(TestOutput::new(10.)),
        ]);
        let (ctx_adj, ctx_adj_calls) = mock_context_adjuster(1);

        let oom_handler = OOMHandler::new(vec![Adjusters::RunContext(Box::new(ctx_adj))], None);

        let ctx_mock = crate::common::mock_run_ctx::build()
            .with_device_idx(1)
            .with_acquired_permits(1)
            .with_drop_permits(vec![DropIf::IfNotMax])
            .finish();

        let out = oom_handler
            .run(
                &Arc::new(task),
                Arc::new(DataCacheWrapper::new(TestInput::new(1), 1, 2, 0)),
                Arc::new(RwLock::new(ctx_mock)),
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
        let ctx = MockGlobalCtx::global_context();
        ctx.expect().returning(|| {
            Box::leak(Box::new(
                mock_global_ctx::build()
                    .with_age(0)
                    .with_dump_all_to_cpu()
                    .with_reset_all_on_cpu()
                    .finish(),
            )) as &'static global_ctx::MockGlobalCtx
        });

        let (init_task1, init_calls1) =
            mock_runnable([Err(ThoosaError::from(MyError::OutOfMemory))]);
        let (final_task1, final_calls1) = mock_runnable([Ok(TestOutput::new(5.))]);

        let (task_adj1, task_adj_calls1) = mock_task_adjuster([final_task1]);

        let (init_task2, init_calls2) =
            mock_runnable([Err(ThoosaError::from(MyError::OutOfMemory))]);
        let (final_task2, final_calls2) = mock_runnable([Ok(TestOutput::new(7.))]);

        let (task_adj2, task_adj_calls2) = mock_task_adjuster([final_task2]);

        let init_task1_arc = Arc::new(init_task1);
        let input_arc = Arc::new(DataCacheWrapper::new(TestInput::new(1), 1, 2, 0));

        let ctx_mock = Arc::new(RwLock::new(
            crate::common::mock_run_ctx::build()
                .with_device_idx(1)
                .with_acquired_permits(1)
                .with_drop_permits(vec![DropIf::IfNotMax])
                .finish(),
        ));

        let oom1 = OOMHandler::new(vec![Adjusters::Task(Box::new(task_adj1))], None);

        let h1 = oom1.run(
            &init_task1_arc,
            Arc::clone(&input_arc),
            Arc::clone(&ctx_mock),
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
            Arc::clone(&ctx_mock),
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
        let (r1, r2) = tokio::join!(h1, h2);
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
