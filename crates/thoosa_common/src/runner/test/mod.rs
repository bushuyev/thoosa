#![allow(dead_code)]

use log::debug;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, Range};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use thoosa_shared::{CopyToDevice, ThoError};
use thoosa_shared::traits::{Adjustable, CachableIdTr, Data, Select, WithId};
use tokio::task::JoinSet;
use crate::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use crate::runner::adjusters::max_parallel::{MaxParallel, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MaxParallelAndDumpToCpuAdjustableContex, MAX_PARALLEL};
use crate::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
use crate::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::common::runnable::{Stage, SyncStage};
use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
type StageIn<'a, S> = <S as Stage>::In<'a>;
type StageInStatic<S> = StageIn<'static, S>;

#[derive(Debug)]
pub enum FailUntil {
    MemLimit(u64),
    NaxParallel(MaxParallel),
    DumpedToCpu,
    Never,
}

//for tests in torch_runner
pub struct ClcOOMRunner<R, S, B>
where
    R: Stage + Adjustable + 'static,
    R::Ctx: Adjustable,
    for<'i> <R as Stage>::In<'i>: Adjustable,
    S:Debug+Default+Clone+Adjustable + 'static,
    B:OutputBuilder<O = <R as Stage>::Out>,
{
    pd: PhantomData<(R,S)>,
    config: OOMRunnerConfig,
    output_builder: Box<
        dyn Fn(<<R as Stage>::Out as WithId>::ID) -> B
            + Send
            + Sync,
    >,
}

impl<R, S, B> Debug for ClcOOMRunner<R, S, B>
where
    R: Stage + Adjustable + 'static,
    R::Ctx: Adjustable,
    for<'i> <R as Stage>::In<'i>: Adjustable,
    S:Debug+Default+Clone+Adjustable + 'static,
    B:OutputBuilder<O = <R as Stage>::Out>,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MulOOMRunner")
            .field("config", &self.config)
            .finish()
    }
}

impl<R, S, B> ClcOOMRunner<R, S, B>
where
    R: Stage + Adjustable + 'static,
    R::Ctx: Adjustable,
    for<'i> <R as Stage>::In<'i>: Adjustable,
    S:Debug+Default+Clone+Adjustable + 'static,
    B:OutputBuilder<O = <R as Stage>::Out>,
{
    pub fn new(
        config: OOMRunnerConfig,
        output_builder: Box<dyn Fn(<<R as Stage>::Out as WithId>::ID) -> B + Send + Sync >,
    ) -> Self {
        Self {
            pd: PhantomData,
            config,
            output_builder,
        }
    }

    pub async fn run<'z, W>(
        &self,
        runnable: &Arc<R>,
        input: Arc<W>,
        mut ctx: <R as Stage>::Ctx,
        required_ids: &'z HashMap<&'static str, &HashSet<&dyn CachableIdTr>>,
    ) -> Result<<R as Stage>::Out, ThoError>
    where
        for <'f> W: WithSelectDataRef<S, O<'f>=<R as Stage>::In<'f>>, // + WithId<ID=<R::In<'a> as WithId>::ID>,
        W: WithId<ID = <R::In<'z> as WithId>::ID> + Send + Sync + 'static,

        R: SyncStage + Adjustable + Send + Sync + 'static,
        for<'i> <R as Stage>::In<'i>: Debug + Adjustable + Data,
        for<'a> <R as Stage>::In<'a>: WithId,
        R::Out: Send + Sync + 'static,
        R::Ctx: Adjustable + Send + Sync + 'static,
        R: OutId,
        S: From<Range<<<R as Stage>::In<'z> as Data>::Metrics>>,
        for <'i> <R::In<'i> as Data>::Metrics:  AddAssign<<R::In<'i> as Data>::Metrics>,
        for <'i> <R::In<'i> as Data>::Metrics: DivAssign<i64>,
        for <'i> <R::In<'i> as Data>::Metrics: PartialOrd<i64>,
        for <'i> <R::In<'i> as Data>::Metrics: PartialOrd,
        <R as Stage>::In<'z>: Select<S, Output=<R as Stage>::In<'z>>
    {

        let size = input._size().clone();
        let mut min_split_len = input._size().clone();
        min_split_len /= 10;

        // debug!("ClcOOMRUnner::run: min_split_len={min_split_len:?} size={size:?}");
        let out_id = runnable.out_id(input.id()); //.clone();

        match self.config {
            OOMRunnerConfig::MaxParallel { max, step } => {
                let run_ctx = ctx.branch(Some("clc_oom_runner_permits"), false);
                OOMHandler::new(
                    vec![Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(
                        max,
                        step,
                    )))],
                    None,
                )
                .run(
                    &runnable,
                    input,
                    run_ctx,
                    Contexts::new(
                        NoopInputContext(None),
                        NoopRunnableContext {},
                        MaxParallelAdjustContextImpl::new(),
                    ),
                )
                .await
                // todo!()
            },
            OOMRunnerConfig::Splits => {
                let run_ctx = ctx.branch(Some("clc_oom_runner_splits"), false);
                OOMHandler::new(
                    vec![
                        Adjusters::Input(Box::new(
                           SplitsAdjuster::<S,  <<R as Stage>::In<'z> as Data>::Metrics>::new(min_split_len),
                        ))
                    ],
                    None,
                )
                .run(
                    &runnable,
                    input,
                    run_ctx,
                    Contexts::new(
                        SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id)),
                        NoopRunnableContext {},
                        NoopRunCtxContext {},
                    ),
                )
                .await
            },
            OOMRunnerConfig::MaxParallelAndSplits {
                max: _max,
                step: _step,
            } => {
                let run_ctx = ctx.branch(Some("clc_oom_runner_permits_splits"), false);
                OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(
                            MAX_PARALLEL,
                            *MaxParallelStep::TWENTY,
                        ))),
                        Adjusters::Input(Box::new(SplitsAdjuster::<S,  <<R as Stage>::In<'z> as Data>::Metrics>::new(
                            min_split_len,
                        ))),
                    ],
                    None,
                )
                .run(
                    &runnable,
                    input,
                    run_ctx,
                    Contexts::new(
                        SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id)),
                        NoopRunnableContext {},
                        MaxParallelAdjustContextImpl::new(),
                    ),
                )
                .await
            }
            OOMRunnerConfig::DumpedToCpu => {
                let run_ctx = ctx.branch(Some("clc_oom_runner_dumped_to_cpu"), false);
                OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(DumpToCpu::new("n/a"))),
                        Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(
                            MAX_PARALLEL,
                            *MaxParallelStep::FIFTY,
                        ))),
                        Adjusters::Input(Box::new(SplitsAdjuster::<S,  <<R as Stage>::In<'z> as Data>::Metrics>::new(
                            min_split_len,
                        ))),
                    ],
                    None,
                )
                .run(
                    &runnable,
                    input,
                    run_ctx,
                    Contexts::new(
                        SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id)),
                        NoopRunnableContext {},
                        MaxParallelAndDumpToCpuAdjustableContex::new(
                            MaxParallelAdjustContextImpl::new(),
                            DumpToCpuContextImpl::new(required_ids),
                        ),
                    ),
                )
                .await
            }
        }
    }
}

//just for tests, normally task has single config
#[derive(Debug)]
pub enum OOMRunnerConfig {
    MaxParallel { max: MaxParallel, step: MaxParallel },
    DumpedToCpu,
    Splits,
    MaxParallelAndSplits { max: MaxParallel, step: MaxParallel },
}

pub async fn run_mul_task_invocations<R, F, CF, I>(
    load_and_mul: Arc<R>,
    resources: &[&'static str],
    device_idx: usize,
    ctx_fac: F,
    check: CF,
) -> Result<(), ThoError>
where
    R: Cached + OutId + Send + Sync + 'static,
    for<'a> <R as Stage>::In<'a>: Data + WithId<ID = I> + Send + Sync,
    // for<'a> <<R as Stage>::In<'a> as Data>::M: DevicesWrapCopyMode<R::In<'a>>,
    DevicesWrap<StageInStatic<R>>: WithId<ID = I>,
    for<'a> DevicesWrap<StageInStatic<R>>: WithDataRef<O<'a> = R::In<'a>>,
    R::Out:  Send + Sync, //Data + CopyToDevice +
    // <R::Out as Data>::M: DevicesWrapCopyMode<R::Out>,
    StageInStatic<R>: From<&'static str>,
    F: Fn(usize/*device*/, u64/*cache_area*/, &'static str/*name*/) -> <R as Stage>::Ctx + Send + Sync + 'static,
    CF: Fn(&'static str, &<R as Stage>::Out) + Send + Sync + Clone + 'static,
{
    let check_arc = Arc::new(check);
    let ctx_fac_arc = Arc::new(ctx_fac);
    let thread_ids = Arc::new(parking_lot::RwLock::new(vec![]));
    let res_cnt = Arc::new(AtomicU32::new(0));
    let res_cnt_arc = Arc::clone(&res_cnt);

    let r = resources
        .iter()
        .enumerate()
        .map(|(idx, res)| {
            let load_and_mul = Arc::clone(&load_and_mul);
            let thread_ids = Arc::clone(&thread_ids);
            let res = *res;

            let check_arc = Arc::clone(&check_arc);
            let ctx_fac_arc = Arc::clone(&ctx_fac_arc);
            let res_cnt_arc = Arc::clone(&res_cnt_arc);
            async move {
                debug!("run_mul_task_invocations: task {idx}: device_idx={device_idx} thread id={:?}",std::thread::current().id());
                thread_ids.write().push(std::thread::current().id());

                let name: &'static str = Box::leak(format!("task_{idx}({res})").into_boxed_str());

                let mut ctx_device = ctx_fac_arc(device_idx, 0, name);

                let multed = load_and_mul
                    .run_cached_(
                        Arc::new(DevicesWrap::wrap(<&str as Into<StageInStatic<R>>>::into(res))),
                        ctx_device.branch(Some(&format!("run_mul_task_run_{idx}")), false),
                    )
                    .await?;

                multed.with_data_ref(device_idx, ctx_device.branch(Some("run_mul_task_data"), false), None, |guard: &<R as Stage>::Out, _ctx| {
                       // let guard = multed.load_from_any_device(device_idx, "in test".into()).await;

                       check_arc(res, &guard);

                       debug!("run_mul_task_invocations: task {idx}: device_idx={device_idx} thread id={:?} done",std::thread::current().id());
                       debug!("run_mul_task_invocations: task {idx}: multed=\n{:?}", guard);

                       res_cnt_arc.fetch_add(1, Relaxed);

                       Ok(())
                   })
                .await?;

                debug!("run_mul_task_invocations: task {idx}: all done");

                Ok::<(), ThoError>(())
            }
        })
        .collect::<JoinSet<_>>()
        .join_all()
        .await
        .into_iter()
        .collect::<Result<(), ThoError>>();

    debug!("test result={r:?} res_cnt={res_cnt:?}");
    assert!(r.is_ok());

    assert_eq!(resources.len() as u32, res_cnt.load(Relaxed));

    // assert_eq!(thread_ids.read().iter().collect::<HashSet<_>>().len(), thread_ids.read().len());
    assert!(thread_ids.read().iter().collect::<HashSet<_>>().len() > 1);

    Ok(())
}

#[cfg(test)]
// Shared fixtures for tests that must bypass the MockGlobalCtx from `tests`.
pub mod fixtures {
    use std::cmp::min;
    use std::ops::Range;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::Arc;

    use async_trait::async_trait;
    use thoosa_named_derive::Named;
    use thoosa_shared::{simple_id, CopyToDevice};
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::traits::{Data, Select, SimpleId, WithId};
    use thoosa_shared::ThoError;
    // use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
    use crate::runner::common::{OutId, RunCtx};
    use crate::runner::common::cachable::{DevicesWrap, WithDataRef};
    use crate::runner::common::runnable::{AsyncStage, Stage};
    use crate::runner::run_ctx_2::RunCtxImpl;
    use crate::runner::test_utils::{parse_numeric_part, DummyTensor};
    use thoosa_shared::traits::Adjustable;
    use crate as thoosa_common;

    #[derive(Debug, Clone, Hash, Eq, PartialEq, Named)]
    #[name = "MulTaskOutputId"]
    pub struct DMulTaskOutputId(pub &'static str);


    #[derive(Debug, cached_derive::Cached, Named)]
    #[cache_name = "LoadedTensorId"]
    pub struct DummyLoader {
        pub run_count: AtomicUsize,
        len: usize,
    }

    impl Adjustable for DLoadedTensor {}

    impl OutId for DummyLoader {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            simple_id!(format!("loaded_{}", input_id.0))
        }

    }

    pub struct DLoadedTensor {
        pub id: SimpleId,
        pub data: DummyTensor,
    }

    impl Select<std::ops::Range<<DLoadedTensor as Data>::Metrics>> for DLoadedTensor {
        type Output = DLoadedTensor;

        fn subview(&self, selector: Range<<DLoadedTensor as Data>::Metrics>) -> Result<Self::Output, ThoError> {
            Ok(DLoadedTensor {
                id: self.id.clone(),
                data: DummyTensor(self.data.0[selector.start as usize..min(self.data.0.len(), selector.end as usize)].to_vec()),
            })
        }
    }

    impl DLoadedTensor {}

    impl std::fmt::Debug for DLoadedTensor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("ALoadedTensor")
                .field("id", &self.id)
                .field("data", &self.data)
                .finish()
        }
    }

    #[async_trait]
    impl Data for DLoadedTensor {
        type Metrics = i64;
        // type M = CpuOrGpu<Self>;

        fn size(&self) -> Self::Metrics {
            self.data.0.len() as i64
        }

        fn device(&self) -> usize {
            0
        }

        fn mem_size(&self) -> u64 {
            0
        }
    }


    #[async_trait]
    impl CopyToDevice for DLoadedTensor {
        async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
            Ok(DLoadedTensor {
                id: self.id.clone(),
                data: self.data.copy_to_device(dst_device).await?,
            })
        }
    }

    impl WithId for DLoadedTensor {
        type ID = SimpleId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }
    impl Output for DLoadedTensor { }

    impl Input for DLoadedTensor {}

    impl DummyLoader {
        pub fn new(len: usize) -> DummyLoader {
            DummyLoader {
                run_count: Default::default(),
                len,
            }
        }

    }

    #[derive(Debug, Clone, Eq, PartialEq, Hash, Named)]
    pub struct DLoaderInputId(pub &'static str);

    #[derive(Debug, Clone)]
    pub struct DLoaderInput {
        pub id: DLoaderInputId,
    }
    impl Input for DLoaderInput {  }
    impl WithId for DLoaderInput {
        type ID = DLoaderInputId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }


    impl WithDataRef for DLoaderInput {
        type O<'o> = Self;

        fn with_data_ref<'a, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> impl Future<Output=Result<R, ThoError>> + Send + 'a
        where
                for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
                P: RunCtx + 'a
        {
            async move {
                do_run(self, &permits_ctx)
            }
        }
    }

    #[async_trait]
    impl Data for DLoaderInput {
        type Metrics = ();
        // type M = CpuOnly<Self>;

        fn size(&self) -> Self::Metrics {
            ()
        }

        fn device(&self) -> usize {
            0
        }

        fn mem_size(&self) -> u64 {
            0
        }
    }

    impl From<&'static str> for DLoaderInput {
        fn from(value: &'static str) -> Self {
            DLoaderInput {
                id: DLoaderInputId(value),
            }
        }
    }


    impl Stage for DummyLoader {
        type In<'a> = DLoaderInput;
        type Out = DevicesWrap<DLoadedTensor>;
        type Ctx = RunCtxImpl;
    }

    // #[async_trait]
    impl AsyncStage for DummyLoader {

        async fn run_<W>(self: Arc<Self>, input: Arc<W>, ctx: Self::Ctx) -> Result<Arc<Self::Out>, ThoError>
        where
            W: for<'a> WithDataRef<O<'a>=Self::In<'a>> + WithId<ID=<Self::In<'static> as WithId>::ID> + 'static
        {
            self.run_count.fetch_add(1, Relaxed);

            let k = parse_numeric_part(WithId::id(input.as_ref()).0)
                .unwrap_or(1.0) as f32;
            let out_id = self.out_id(WithId::id(input.as_ref()));


            Ok(Arc::new(DevicesWrap::wrap(DLoadedTensor {
                id: out_id,
                data: (0..self.len)
                    .map(|v| v as f32 * k)
                    .collect::<Vec<f32>>()
                    .into(),
            })))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::fmt::Debug;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::{Arc, LazyLock, OnceLock};
    use std::time::Duration;
    use super::fixtures::{DLoadedTensor, DLoaderInput, DLoaderInputId, DMulTaskOutputId};
    use super::{run_mul_task_invocations, ClcOOMRunner, FailUntil, OOMRunnerConfig};
    use async_trait::async_trait;
    use cached_derive::Cached;
    use dashmap::DashMap;
    use futures_util::future::{BoxFuture, Shared};
    use log::{debug, error};
    use thoosa_named_derive::Named;
    use mockall_double::double;
    use thoosa_shared::{simple_id, CopyToDevice};
    use thoosa_shared::applicables_runner::Output;
    // use thoosa_shared::copy_mode::CpuOrGpu;
    use thoosa_shared::traits::{Data, WithId};
    use thoosa_shared::{Cause, ThoError};
    use crate::runner::adjusters::splits::OutputBuilder;
    use crate::runner::common::{MaxParallelCtx, OutId, RunCtx};
    use crate::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
    use crate::runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use crate::runner::global_ctx::MockGlobalCtx;
    use crate::runner::purge::FuturesRunnerMapPurger;
    use crate::runner::run_ctx_2::RunCtxImpl;
    use crate::runner::test_utils::{parse_numeric_part, DummyTensor};
    use thoosa_shared::traits::Adjustable;
    #[double]
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx;
    use crate as thoosa_common;
    use crate::runner::adjusters::max_parallel::{MaxParallelStep, MAX_PARALLEL};
    use crate::runner::cache_areas::CacheArea;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    static TELEMETRY: std::sync::LazyLock<()> = LazyLock::new(|| {
        console_subscriber::ConsoleLayer::builder()
            .retention(std::time::Duration::from_secs(60))
            .init();
    });

    #[derive(Debug, Named)]
    pub struct DummyLoader {
        pub run_count: AtomicUsize,
        len: usize,
    }

    static DUMMY_LOADER_CACHE_PTR: OnceLock<Arc<DashMap<
        <<DummyLoader as Stage>::Out as WithId>::ID,
        Shared<BoxFuture<'static, Result<Arc<DevicesWrap<<DummyLoader as Stage>::Out>>, ThoError>>>,
    >>> = OnceLock::new();
    static DUMMY_LOADER_REGISTER: tokio::sync::OnceCell<()> = tokio::sync::OnceCell::const_new();

    #[async_trait::async_trait]
    impl Cached for DummyLoader {
        async fn cache() -> &'static DashMap<
            <<Self as Stage>::Out as WithId>::ID,
            Shared<BoxFuture<'static, Result<Arc<DevicesWrap<Self::Out>>, ThoError>>>,
        >
        where
            Self: Sized,
            Self::Out: Data + Sync + Send,
        {
            let map = DUMMY_LOADER_CACHE_PTR.get_or_init(|| Arc::new(Default::default()));

            let cache_name: &'static str = "LoadedTensorId";

            DUMMY_LOADER_REGISTER
                .get_or_init(|| async move {
                    GlobalCtx::global()
                        .register_cache((
                            cache_name,
                            Arc::new(crate::runner::purge::StaticMapPurger::new(map.as_ref())) as Arc<dyn FuturesRunnerMapPurger>,
                        ));
                })
                .await;

            map.as_ref()
        }
    }

    impl DummyLoader {
        pub fn new(len: usize) -> DummyLoader {
            DummyLoader {
                run_count: Default::default(),
                len,
            }
        }
    }

    impl OutId for DummyLoader {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            simple_id!(format!("loaded_{}", input_id.0))
        }
    }

    impl Stage for DummyLoader {
        type In<'a> = DLoaderInput;
        type Out = DevicesWrap<DLoadedTensor>;
        type Ctx = RunCtxImpl;
    }

    impl AsyncStage for DummyLoader {


        fn run_<W>(self: Arc<Self>, input: Arc<W>, _ctx: Self::Ctx) -> impl Future<Output=Result<Arc<Self::Out>, ThoError>> + Send
        where
            W: for<'a> WithDataRef<O<'a>=Self::In<'a>>  + WithId<ID= <Self::In<'static> as WithId>::ID> +'static
        {
            async move {
                self.run_count.fetch_add(1, Relaxed);

                let k = parse_numeric_part(WithId::id(input.as_ref()).0).unwrap_or(1.0) as f32;
                let out_id = self.out_id(WithId::id(input.as_ref()));


                Ok(Arc::new(DevicesWrap::wrap(DLoadedTensor {
                    id: out_id,
                    data: (0..self.len)
                        .map(|v| v as f32 * k)
                        .collect::<Vec<f32>>()
                        .into(),
                })))
            }
        }
    }

    ////////////
    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct DMulTaskId(&'static str);

    #[derive(Debug)]
    #[allow(dead_code)]
    struct DMulTask {
        id: DMulTaskId,
        k: f32,
        fail_until: FailUntil,
    }

    impl DMulTask {
        pub fn new(id: DMulTaskId, k: f32, fail_until: FailUntil) -> Self {
            Self { id, k, fail_until }
        }
    }

    impl Stage for DMulTask {
        type In<'a> = DLoadedTensor;
        type Out = DMulTaskOutput;
        type Ctx = RunCtxImpl;
    }

    impl SyncStage for DMulTask {

        fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
            // tokio::time::sleep(tokio::time::Duration::from_millis(random_range(0..10))).await;
            // #[cfg(any(test, feature = "mock_global_ctx"))]
            // let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

            let deive_idx = ctx.device_idx();

            match self.fail_until {
                FailUntil::MemLimit(mem_limit) => {
                    let _device_idx = deive_idx;
                    if input.data.0.len() > mem_limit as usize {
                        return Err(ThoError::oom());
                    }
                }
                FailUntil::NaxParallel(max_other_parallel_threshold) => {
                    let max_other_parallel = ctx.max_other_parallel();
                    debug!("MulTask::do_run: ctx={ctx:?} max_other_parallel={max_other_parallel} max_other_parallel_threshold={max_other_parallel_threshold}");

                    if max_other_parallel > max_other_parallel_threshold {
                        return Err(ThoError::oom());
                    }
                }

                FailUntil::DumpedToCpu => {
                    let device_idx = ctx.device_idx();
                    let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(device_idx);
                    debug!("MulTask::do_run: ctx.name={:?} device_idx={} is_all_on_cpu={is_all_on_cpu}","ctx.name()",device_idx);

                    if !is_all_on_cpu {
                        return Err(ThoError::oom());
                    }
                }
                FailUntil::Never => {}
            }

            debug!("MulTask::do_run: running ctx.name={:?} device_idx={} ","ctx.name()",ctx.device_idx());

            Ok(DMulTaskOutput {
                id: self.out_id(WithId::id(input)),
                data: input
                    .data
                    .0
                    .iter()
                    .map(|v| v * self.k)
                    .collect::<Vec<f32>>()
                    .into(),
            })
        }
    }

    impl Adjustable for DMulTask {}

    impl OutId for DMulTask {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            DMulTaskOutputId(format!("multiplied_{}", input_id).leak())
        }

    }

    #[derive(Debug)]
    struct DMulTaskOutput {
        id: DMulTaskOutputId,
        data: DummyTensor,
    }

    #[async_trait]
    impl Data for DMulTaskOutput {
        type Metrics = [usize; 1];
        // type M = CpuOrGpu<Self>;


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
    impl CopyToDevice for DMulTaskOutput {
        async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
            Ok(DMulTaskOutput {
                id: self.id.clone(),
                data: self.data.clone(),
            })
        }
    }

    #[derive(Debug)]
    struct DMulTaskOutputBuilder {
        id: DMulTaskOutputId,
        partial_output: Vec<DMulTaskOutput>,
    }

    impl DMulTaskOutputBuilder {
        pub fn new(id: DMulTaskOutputId) -> Self {
            Self {
                id,
                partial_output: vec![],
            }
        }
    }

    impl OutputBuilder for DMulTaskOutputBuilder {
        type O = DMulTaskOutput;

        async fn add_partial_output(&mut self, o: Self::O) {
            self.partial_output.push(o);
        }

        async fn build(self) -> Result<Self::O, ThoError> {
            Ok(DMulTaskOutput {
                id: self.id.clone(),

                data: DummyTensor(self.partial_output.into_iter().fold(vec![], |mut a, v| {
                    a.extend(v.data.0);
                    a
                })),
            })
        }
    }

    impl WithId for DMulTaskOutput {
        type ID = DMulTaskOutputId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl Output for DMulTaskOutput {}

    ////////////
    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct DLoadAndMulTaskId(&'static str);

    #[derive(Debug, Cached, Named)]
    #[cache_name = "MulTaskOutputId"]
    #[allow(dead_code)]
    struct DLoadAndMulTask {
        id: DLoadAndMulTaskId,

        loader: Arc<DummyLoader>,
        mul: Arc<DMulTask>,
        mul_oom_runner: ClcOOMRunner<DMulTask, std::ops::Range<<DLoadedTensor as Data>::Metrics>,DMulTaskOutputBuilder>,
    }

    impl OutId for DLoadAndMulTask {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            self.mul.out_id(&self.loader.out_id(input_id))
        }
    }

    impl DLoadAndMulTask {
        pub fn new(
            id: DLoadAndMulTaskId,
            k: f32,
            fail_until: FailUntil,
            len: usize,
            oom_runner: ClcOOMRunner<DMulTask, std::ops::Range<<DLoadedTensor as Data>::Metrics>, DMulTaskOutputBuilder>,
        ) -> Self {
            Self {
                id,
                loader: Arc::new(DummyLoader::new(len)),
                mul: Arc::new(DMulTask::new(DMulTaskId("mul_task_0"), k, fail_until)),
                mul_oom_runner: oom_runner,
            }
        }
    }


    impl Stage for DLoadAndMulTask {
        type In<'a> = DLoaderInput;
        type Out = DevicesWrap<DMulTaskOutput>;
        type Ctx = RunCtxImpl;
    }

    // #[async_trait]
    impl AsyncStage for DLoadAndMulTask {

        async fn run_<W>(self: Arc<Self>, input: Arc<W>, mut ctx: Self::Ctx) -> Result<Arc<Self::Out>, ThoError>
        where
            W: for<'a> WithDataRef<O<'a>=Self::In<'a>>  + WithId<ID= <Self::In<'static> as WithId>::ID> +'static, // + WithSelectDataRef<S>,
            // S: Clone + Debug + Send + 'static
        {
            let ctx_for_loader = ctx.branch(Some("load_and_mul_loader"), false);
            let loaded = Arc::clone(&self.loader)
                .run_(input, ctx_for_loader)
                .await?;

            debug!("LoadAndMulTask::run_2: loaded.id={:?}", loaded.id());


            async move {

                Ok(Arc::new(DevicesWrap::wrap(self.mul_oom_runner
                    .run(
                        &self.mul,
                        loaded,
                        ctx.branch(Some("load_and_mul_mul"), false),
                        &HashMap::from([("", &HashSet::new())]),
                    )
                    .await.map_err(|e|{
                    error!("DLoadAndMulTask::run: e={e:?}");
                    e
                })?)))
            }.await
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_dummy() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);

        let device_idx = 0;

        let loader = Arc::new(DummyLoader::new(3));

        let mut ctx_device_0 = RunCtxImpl::new(device_idx, "noname");
        let ctx_device_0_0 = ctx_device_0.branch(Some("test_load_dummy_0_0"), false);
        let ctx_device_0_1 = ctx_device_0.branch(Some("test_load_dummy_0_1"), false);

        let loaded_tensor_0_0 = Arc::clone(&loader)
            .run_cached_(
                Arc::new(DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                }),
                ctx_device_0_0,
            )
            .await;

        // debug!("{:?}", loaded_tensor_0_0);

        let loaded_tensor_0_1 = Arc::clone(&loader)
            .run_cached_(
                Arc::new(DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                }),
                ctx_device_0_1,
            )
            .await;

        assert!(loaded_tensor_0_0.is_ok());
        assert!(loaded_tensor_0_1.is_ok());

        assert!(Arc::ptr_eq(
            &loaded_tensor_0_0.unwrap(),
            loaded_tensor_0_1.as_ref().unwrap()
        ));

        assert_eq!(loader.run_count.load(Relaxed), 1);
        // let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();
        let mut ctx_device_1 = RunCtxImpl::new(1, "noname");
        let ctx_device_1_0 = ctx_device_1.branch(Some("test_load_dummy_1_0"), false);
        let ctx_device_1_data = ctx_device_1.branch(Some("test_load_dummy_1_0_data"), false);

        let loaded_tensor_1_0 = Arc::clone(&loader)
            .run_cached_(
                Arc::new(DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                }),
                ctx_device_1_0,
            )
            .await?;

        debug!("loaded_tensor_1_0={loaded_tensor_1_0:?}");
        loaded_tensor_1_0
            .with_data_ref(1,
           ctx_device_1_data, None, |data: &DLoadedTensor, _ctx| {
                           assert_eq!(data.data.0, [0.0, 1.0, 2.0]);

                           Ok(())
                       })
            .await?;

        // assert!(loaded_tensor_1_0.is_ok());
        assert!(Arc::ptr_eq(
            //cached wrapper same for all devices
            &loaded_tensor_1_0,
            &loaded_tensor_0_1?
        ));


        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_purge_cpu() -> Result<(), ThoError> {

        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false);

        let device_idx = 0;

        let loader = Arc::new(DummyLoader::new(3));

        let mut ctx_device_0 = RunCtxImpl::new(device_idx, "noname");
        let ctx_device_0_0 = ctx_device_0.branch(Some("test_load_dummy_0_0"), false);

        let loaded_tensor_0_0 = Arc::clone(&loader)
            .run_cached_(
                Arc::new(DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                }),
                ctx_device_0_0,
            )
            .await?;

        {
            let loaded_tensor_on_device_1 = loaded_tensor_0_0.load_from_any_device_(1).await?;
            debug!("loaded_tensor_on_device_1={loaded_tensor_on_device_1:?}");
        }

        debug!("loaded_tensor_0_0={loaded_tensor_0_0:?} is_all_on_cpu={}", loaded_tensor_0_0.is_all_on_cpu().await);

        assert!(!loaded_tensor_0_0.is_all_on_cpu().await);

        DummyLoader::cache().await.purge_gpu(Some(&HashSet::new()), 1, "test").await?;

        assert!(loaded_tensor_0_0.is_all_on_cpu().await);

        assert_eq!(1, DummyLoader::cache().await.len());
        let purged = DummyLoader::cache().await.purge_cpu(0).await;

        assert_eq!(0, purged);
        assert_eq!(1, DummyLoader::cache().await.len());

        drop(loaded_tensor_0_0);

        let purged = DummyLoader::cache().await.purge_cpu(0).await;

        assert_eq!(1, purged);
        assert_eq!(0, DummyLoader::cache().await.len());

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_no_oom_dummy() -> Result<(), ThoError> {
        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;

        RealGlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let _device_idx = 0;

        let load_and_mut = Arc::new(DLoadAndMulTask::new(
            DLoadAndMulTaskId("load_and_mul_id"),
            2.,
            FailUntil::Never,
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel {
                    max: MAX_PARALLEL,
                    step: 10,
                },
                Box::new(|id| DMulTaskOutputBuilder::new(id)),
            ),
        ));
        let mut ctx_device_0 = RunCtxImpl::new(1, "noname");
        let ctx_device_0_run = ctx_device_0.branch(Some("test_load_and_mul_no_oom_run"), false);
        let ctx_device_0_data = ctx_device_0.branch(Some("test_load_and_mul_no_oom_data"), false);

        let multed = load_and_mut.run_(//run_cached
            Arc::new(DLoaderInput {
                id: DLoaderInputId("resource_1"),
            }),
            ctx_device_0_run,
        )
        .await?;

        multed
        .with_data_ref(1,
           ctx_device_0_data, None, |data: &DMulTaskOutput, _ctx| {
               assert_eq!(
                   data.data.0,
                   (0..100).map(|v| v as f32 * 2.).collect::<Vec<_>>()
               );

               Ok(())
           })
            .await?;

        debug!("multed={multed:?}");

        Ok(())
    }

    fn mul_check(mul_k: f32, res: &str, out: &DMulTaskOutput) {
        let self_k = parse_numeric_part(res).unwrap_or(1.0) as f32;

        assert_eq!(
            *out.data.0,
            (0..100)
                .map(|v| v as f32 * self_k * mul_k)
                .collect::<Vec<_>>(),
        );

        debug!("mul_check: multed_{res}={out:?}");
    }

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_till_max_parallel_dummy() -> Result<(), ThoError> {
        let _global_ctx = GlobalCtx::init_devices(3);

        let device_idx = 1;
        const MUL_K: f32 = 2.;

        let load_and_mut = Arc::new(DLoadAndMulTask::new(
            DLoadAndMulTaskId("load_and_mul_id"),
            MUL_K,
            FailUntil::NaxParallel(20),
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel { max: 20, step: 10 },
                Box::new(|id| DMulTaskOutputBuilder::new(id)),
            ),
        ));

        super::run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            |device_idx, _cache_area, name| RunCtxImpl::new(device_idx, name),
            |res, out| {
                mul_check(MUL_K, res, out);
            },
        )
        .await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn test_load_and_mul_oom_compete_for_max_parallel_dummy() -> Result<(), ThoError> {
        crate::runner::global_ctx::GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let device_idx = 1;

        const MUL_K: f32 = 2.;
        let load_and_mut = Arc::new(DLoadAndMulTask::new(
            DLoadAndMulTaskId("load_and_mul_id"),
            MUL_K,
            FailUntil::NaxParallel(MAX_PARALLEL),
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel {
                    max: MAX_PARALLEL,
                    step: *MaxParallelStep::TEN,
                },
                Box::new(|id| DMulTaskOutputBuilder::new(id)),
            ),
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            |device_idx, _cache_area, name| RunCtxImpl::new(device_idx, name),
            |res, out| {
                mul_check(MUL_K, res, out);
            },
        )
        .await?;

        Ok::<_, ThoError>(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_splits_dummy() -> Result<(), ThoError> {
        crate::runner::global_ctx::GlobalCtx::init_devices(2);
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        let device_idx = 1;

        const MUL_K: f32 = 2.;
        let load_and_mut = Arc::new(DLoadAndMulTask::new(
            DLoadAndMulTaskId("load_and_mul_id"),
            MUL_K,
            FailUntil::MemLimit(12),
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::Splits {},
                Box::new(|id| DMulTaskOutputBuilder::new(id)),
            ),
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &[
                "resource_1",
                "resource_2",
                "resource_3",
                "resource_4",
                "resource_5",
            ], //,
            device_idx,
            |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
            move |res, out| {
                mul_check(MUL_K, res, out);
            },
        )
        .await?;

        Ok::<_, ThoError>(())
    }

    //TODO #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_oom_till_dump_all_to_cpu_dummy() -> Result<(), ThoError> {
        // let _ = *TELEMETRY;

        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 1;
        const MUL_K: f32 = 2.;
        let load_and_mut = Arc::new(DLoadAndMulTask::new(
            DLoadAndMulTaskId("load_and_mul_id"),
            MUL_K,
            FailUntil::DumpedToCpu,
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::DumpedToCpu,
                Box::new(|id| DMulTaskOutputBuilder::new(id)),
            ),
        ));

        super::run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            |device_idx, _cache_area, name| RunCtxImpl::new(device_idx, name),
            |res, out| {
                mul_check(MUL_K, res, out);
            },
        )
        .await?;

        Ok(())
    }

}
