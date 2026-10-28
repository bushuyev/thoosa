#![allow(dead_code)]

use crate::runner::global_ctx::CacheKey;
use log::debug;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{AddAssign, DivAssign, Range};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use thoosa_shared::traits::{Adjustable, AsyncStage, CachableIdTr, Data, MaxParallelCtx, OutId, RunCtx, Select, Stage, ThoSelector, WithId};
use thoosa_common::runner::sync_stage::{SyncStage};
use thoosa_shared::{CopyToDevice, ThoError};
use tokio::task::JoinSet;

use crate::runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallel, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelAndDumpToCpuAdjustableContex, MaxParallelStep};
use crate::runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
use crate::runner::common::cachable::{BorrowRef, Cached, CachedPurgeOutput, DevicesWrap, SelectRef};
use crate::runner::oom_handler::{Adjusters, NoInputAdjuster, NoRunCtxAdjuster, NoStageAdjuster, OOMHandler, WithInputContext, WithRunCtxContext, input_adjuster, run_ctx_adjuster};
use crate::tho_tensor::ThoTensor;
use thoosa_shared::comp_id::SynKey;


type StageIn<'a, S> = <S as Stage>::In<'a>;
// type StageInStatic<S> = StageIn<'static, S>;

#[derive(Debug, Serialize, Deserialize, PartialEq)]
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
    <R as Stage>::Out:WithId
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
    <R as Stage>::Out:WithId
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
    S:ThoSelector + Debug+Default+Clone+Adjustable + 'static,
    B:OutputBuilder<O = <R as Stage>::Out>,
    <R as Stage>::Out:WithId
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
        required_ids: &'z HashMap<CacheKey, &HashSet<&dyn CachableIdTr>>,
        size: <R::In<'z> as Data>::Metrics,
    ) -> Result<<R as Stage>::Out, ThoError>
    where
        W: SelectRef<S> + Debug,
        for<'f> <W as BorrowRef>::Out<'f>: std::borrow::Borrow<<R as Stage>::In<'f>> + Send + Debug,

        W: WithId<ID = <R::In<'z> as WithId>::ID> + Send + Sync + 'static,

        R: SyncStage + Adjustable + Send + Sync + 'static,
        for<'i> <R as Stage>::In<'i>: Debug + Adjustable + Data,
        for<'a> <R as Stage>::In<'a>: WithId,
        R::Out: Send + Sync + 'static,
        R::Ctx: MaxParallelCtx + Adjustable + Send + Sync + 'static,
        R: OutId,
        S: From<Range<<<R as Stage>::In<'z> as Data>::Metrics>>,
        for <'i> <R::In<'i> as Data>::Metrics:  AddAssign<<R::In<'i> as Data>::Metrics>,
        for <'i> <R::In<'i> as Data>::Metrics: DivAssign<i64>,
        for <'i> <R::In<'i> as Data>::Metrics: PartialOrd<i64>,
        for <'i> <R::In<'i> as Data>::Metrics: PartialOrd,
        <R as Stage>::In<'z>: Select<S, Output=<R as Stage>::In<'z>>
    {

        let size = size.clone();
        let mut min_split_len = size.clone();
        min_split_len /= 10;

        debug!("ClcOOMRunner::run: min_split_len={min_split_len:?} size={size:?}");
        let out_id = runnable.out_id(input.id()); //.clone();

        match self.config {
            OOMRunnerConfig::MaxParallel { max, step } => {
                let run_ctx = ctx.branch(false);
                let adjusters: Vec<Adjusters<NoInputAdjuster, NoStageAdjuster, _>> = vec![Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(
                    max,
                    step,
                )))];
                OOMHandler::<NoInputAdjuster, NoStageAdjuster, _, S>::new(
                    adjusters,
                    None,
                )
                .run(
                    &runnable,
                    input,
                    run_ctx,
                    WithRunCtxContext::new(MaxParallelAdjustContextImpl::new()),
                )
                .await
            },
            OOMRunnerConfig::Splits => {
                let run_ctx = ctx.branch(false);
                let adjusters: Vec<Adjusters<_, NoStageAdjuster, NoRunCtxAdjuster>> = vec![
                    Adjusters::Input(input_adjuster(
                       SplitsAdjuster::<S,  <<R as Stage>::In<'z> as Data>::Metrics>::new(min_split_len),
                    ))
                ];
                OOMHandler::new(
                    adjusters,
                    None,
                )
                .run(
                    &runnable,
                    input,
                    run_ctx,
                    WithInputContext::new(SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id))),
                )
                .await
            },
            OOMRunnerConfig::MaxParallelAndSplits {
                max: _max,
                step: _step,
            } => {
                let run_ctx = ctx.branch(false);
                let adjusters: Vec<Adjusters<_, NoStageAdjuster, _>> = vec![
                    Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(
                        MAX_PARALLEL,
                        *MaxParallelStep::MID,
                    ))),
                    Adjusters::Input(input_adjuster(SplitsAdjuster::<S,  <<R as Stage>::In<'z> as Data>::Metrics>::new(
                        min_split_len,
                    ))),
                ];
                OOMHandler::new(
                    adjusters,
                    None,
                )
                .run(
                    &runnable,
                    input,
                    run_ctx,
                    WithRunCtxContext::with_rest(
                        MaxParallelAdjustContextImpl::new(),
                        WithInputContext::new(SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id))),
                    ),
                )
                .await
            }
            OOMRunnerConfig::DumpedToCpu => {
                let run_ctx = ctx.branch(false);
                let adjusters: Vec<Adjusters<_, NoStageAdjuster, _>> = vec![
                    Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(
                        MAX_PARALLEL,
                        *MaxParallelStep::FAST,
                    ))),
                    Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("n/a"))),
                    Adjusters::Input(input_adjuster(SplitsAdjuster::<S,  <<R as Stage>::In<'z> as Data>::Metrics>::new(
                        min_split_len,
                    ))),
                ];
                OOMHandler::new(
                    adjusters,
                    None,
                )
                .run(
                    &runnable,
                    input,
                    run_ctx,
                    WithRunCtxContext::with_rest(
                        MaxParallelAndDumpToCpuAdjustableContex::new(
                            MaxParallelAdjustContextImpl::new(),
                            DumpToCpuContextImpl::new(required_ids),
                        ),
                        WithInputContext::new(SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id))),
                    ),
                )
                .await
            }
            OOMRunnerConfig::DumpedToCpuAndSplits  => {
                let run_ctx = ctx.branch(false);
                let adjusters: Vec<Adjusters<_, NoStageAdjuster, _>> = vec![
                    Adjusters::RunContext(run_ctx_adjuster(DumpToCpu::new("n/a"))),
                    Adjusters::Input(input_adjuster(SplitsAdjuster::<S,  <<R as Stage>::In<'z> as Data>::Metrics>::new(
                        min_split_len,
                    ))),
                ];
                OOMHandler::new(
                    adjusters,
                    None,
                )
                    .run(
                        &runnable,
                        input,
                        run_ctx,
                        WithRunCtxContext::with_rest(
                            DumpToCpuContextImpl::new(required_ids),
                            WithInputContext::new(SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id))),
                        ),
                    )
                    .await
            }
        }
    }
}
//
//just for tests, normally task has single config
#[derive(Debug)]
pub enum OOMRunnerConfig {
    MaxParallel { max: MaxParallel, step: u8 },
    DumpedToCpu,
    Splits,
    MaxParallelAndSplits { max: MaxParallel, step: u8 },
    DumpedToCpuAndSplits,
}

pub async fn run_mul_task_invocations<R, F, CF>(
    load_and_clc: Arc<R>,
    resources: &[&'static str],
    device_idx: usize,
    check_device_idx: usize,
    ctx_fac: F,
    check: CF,
    check_threads: bool,
) -> Result<(), ThoError>
where
    // D:WithId + Data + Send + Sync,
    R: AsyncStage + Cached + OutId + Send + Sync + 'static,
    // R: crate::runner::common::runnable::AsyncStage<In<'static> = Arc<StageInStatic<R>>, Out = Arc<DevicesWrap<<R as Stage>::Out>>>,
    for<'a> <R as Stage>::In<'a>:  WithId + Send + Sync,
    R::Out: CachedPurgeOutput +  Clone + Send + Sync,
    <R as Stage>::In<'static>: From<&'static str>,
    F: Fn(usize/*device*/, u64/*cache_area*/, &'static str/*name*/) -> <R as Stage>::Ctx + Send + Sync + 'static,
    CF: Fn(&'static str, &<<R as Stage>::Out as BorrowRef>::Out<'_>)  + Send + Sync + Clone + 'static,
    for<'o> <R as Stage>::Out:WithId+ BorrowRef+Debug,
    for<'o> <<R as Stage>::Out as BorrowRef>::Out<'o>: Debug
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
            let load_and_clc = Arc::clone(&load_and_clc);
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

                let multed = load_and_clc
                    .run_cached(
                        <&str as Into<<R as Stage>::In<'static>>>::into(res),
                        ctx_device.branch(false),
                    )
                    .await?;

                let guard = BorrowRef::borrow_ref(&multed, check_device_idx).await?;


               check_arc(res, &guard);

               debug!("run_mul_task_invocations: task {idx}: device_idx={device_idx} thread id={:?} done",std::thread::current().id());
               debug!("run_mul_task_invocations: task {idx}: multed=\n{:?}", guard);

               res_cnt_arc.fetch_add(1, Relaxed);


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
    r?;

    assert_eq!(resources.len() as u32, res_cnt.load(Relaxed));

    // assert_eq!(thread_ids.read().iter().collect::<HashSet<_>>().len(), thread_ids.read().len());
    if check_threads {
        assert!(thread_ids.read().iter().collect::<HashSet<_>>().len() > 1);
    }

    Ok(())
}

#[cfg(test)]
// Shared fixtures for tests that must bypass the MockGlobalCtx from `tests`.
pub mod fixtures {
    use std::cmp::min;
    use std::ops::{Deref, Range};
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;

    use crate as thoosa_common;
    use crate::runner::common::cachable::DevicesWrap;
    use crate::runner::run_ctx::RunCtxImpl;
    use crate::runner::test_utils::{DummyTensor, parse_numeric_part};
    use async_trait::async_trait;
    use serde::{Deserialize, Serialize};
    use thoosa_named_derive::Named;
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::ThoError;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
    use thoosa_shared::traits::OutId;
    use thoosa_shared::traits::{Adjustable, AsyncStage, Stage};
    use thoosa_shared::traits::{Data, Metrics, Select, ThoSelector, WithId};

    #[derive(Debug, Clone, Hash, Eq, PartialEq, Named)]
    #[name = "MulTaskOutputId"]
    pub struct DMulStageOutputId(pub &'static str);

    #[derive(Debug, Clone, Hash, Eq, PartialEq, Named)]
    #[name = "LoadedTensorId"]
    pub struct DLoadedTensorId(pub &'static str);


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
            DLoadedTensorId(format!("loaded_{}", input_id.0).leak())
        }

    }

    pub struct DLoadedTensor {
        pub id: DLoadedTensorId,
        pub data: DummyTensor,
    }

    #[derive(Debug, Serialize, Deserialize, Clone, Default)]
    pub(crate) struct MetricsRangeSelector<M>(pub Range<M>)
    where
        M:Metrics + Send + Sync;//MetricsRangeSelector

    impl <M> Adjustable for MetricsRangeSelector<M>
    where
        M:Metrics + Send + Sync,
    {}

    impl <M:Metrics> From<Range<M>> for MetricsRangeSelector<M>
    where
        M:Metrics + Send + Sync
    {
        fn from(value: Range<M>) -> Self {
            Self(value)
        }
    }

    #[typetag::serde]
    impl ThoSelector for MetricsRangeSelector<<DLoadedTensor as Data>::Metrics> {
        fn clone_box(&self) -> Box<dyn ThoSelector> {
            Box::new(MetricsRangeSelector(self.0.clone()))
        }
    }

    impl <M> Deref for MetricsRangeSelector<M>
    where
        M:Metrics + Send + Sync
    {
        type Target = Range<M>;

        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }

    impl Select<MetricsRangeSelector<<DLoadedTensor as Data>::Metrics>> for DLoadedTensor
    {
        type Output = DLoadedTensor;

        fn subview(&self, selector: MetricsRangeSelector<<DLoadedTensor as Data>::Metrics>) -> Result<Self::Output, ThoError> {
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


        fn size(&self) -> Self::Metrics {
            self.data.0.len() as i64
        }

        fn tho_device_idx(&self) -> usize {
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
        type ID = DLoadedTensorId;

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

    #[async_trait]
    impl Data for DLoaderInput {
        type Metrics = ();


        fn size(&self) -> Self::Metrics {
            ()
        }

        fn tho_device_idx(&self) -> usize {
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
        type In<'a> =DLoaderInput;
        type Out = Arc<DevicesWrap<DLoadedTensor>>;
        type Ctx = RunCtxImpl;
    }


    impl AsyncStage for DummyLoader {
        // type RunInput = Arc<Self::In<'static>>;
        // type RunOutput = Arc<DevicesWrap<Self::Out>>;

        // fn run_input_id(input: &Self::In<'static>) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        //     WithId::id(input)
        // }

        async fn run(
            self: Arc<Self>,
            input: Self::In<'static>,
            ctx: Self::Ctx,
        ) -> Result<Self::Out, ThoError> {
            self.run_count.fetch_add(1, Relaxed);

            let k = parse_numeric_part(WithId::id(&input).0)
                .unwrap_or(1.0) as f32;
            let out_id = self.out_id(WithId::id(&input));


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
    use super::fixtures::{DLoadedTensor, DLoadedTensorId, DLoaderInput, DLoaderInputId, DMulStageOutputId, MetricsRangeSelector};
    use super::{ClcOOMRunner, FailUntil, OOMRunnerConfig, run_mul_task_invocations};
    use crate::runner::adjusters::splits::OutputBuilder;
    use crate::runner::common::cachable::{BorrowRef, Cached, DevicesWrap};
    use crate::runner::global_ctx::MockGlobalCtx;
    use crate::runner::purge::FuturesRunnerMapPurger;
    use crate::runner::run_ctx::RunCtxImpl;
    use crate::runner::test_utils::{DummyTensor, parse_numeric_part};
    use async_trait::async_trait;
    use cached_derive::Cached;
    use clap::{Args, Parser, Subcommand};
    use crossbeam::deque::Worker;
    use futures_util::future::{BoxFuture, Shared};
    use log::{debug, error};
    use mockall_double::double;
    use std::collections::{HashMap, HashSet};
    use std::process::Stdio;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::{Arc, LazyLock};
    use std::time::Duration;
    use thoosa_named_derive::Named;
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::applicables_runner::Output;
    use thoosa_shared::copy_mode::CpuOrGpu;
    use thoosa_shared::traits::MaxParallelCtx;
    use thoosa_shared::traits::{Adjustable, AsyncStage, Stage};
    use thoosa_common::runner::sync_stage::{SyncStage};
    use thoosa_shared::traits::{Data, OutId, RunCtx, WithId};
    use thoosa_shared::ThoError;
    use tokio::process::{Child, Command};
    #[double]
    use crate::runner::global_ctx::GlobalCtx;
    use crate as thoosa_common;
    use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelStep};
    use crate::runner::cache_areas::CacheArea;
    use crate::runner::global_ctx::mock_global_ctx;
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

    #[derive(Debug, Named, cached_derive::Cached)]
    pub struct DummyLoader {
        pub run_count: AtomicUsize,
        len: usize,
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
            DLoadedTensorId(format!("loaded_{}", input_id.0).leak())
        }
    }

    impl Stage for DummyLoader {
        type In<'a> = DLoaderInput;
        type Out = Arc<DevicesWrap<DLoadedTensor>>;
        type Ctx = RunCtxImpl;
    }

    impl AsyncStage for DummyLoader {
        // type RunInput = Arc<Self::In<'static>>;
        // type RunOutput = Arc<DevicesWrap<Self::Out>>;

        // fn run_input_id(input: &Self::In<'static>) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        //     WithId::id(input)
        // }
        async fn run(
            self: Arc<Self>,
            input: Self::In<'static>,
            ctx: Self::Ctx,
        ) -> Result<Self::Out, ThoError> {
            self.run_count.fetch_add(1, Relaxed);

            let k = parse_numeric_part(WithId::id(&input).0)
                .unwrap_or(1.0) as f32;
            let out_id = self.out_id(WithId::id(&input));


            Ok(Arc::new(DevicesWrap::wrap(DLoadedTensor {
                id: out_id,
                data: (0..self.len)
                    .map(|v| v as f32 * k)
                    .collect::<Vec<f32>>()
                    .into(),
            })))
        }

    }

    ////////////
    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct DMulStageId(&'static str);

    #[derive(Debug)]
    #[allow(dead_code)]
    struct DMulStage {
        id: DMulStageId,
        k: f32,
        fail_until: FailUntil,
    }

    impl DMulStage {
        pub fn new(id: DMulStageId, k: f32, fail_until: FailUntil) -> Self {
            Self { id, k, fail_until }
        }
    }

    impl Stage for DMulStage {
        type In<'a> = DLoadedTensor;
        type Out = DMulStageOutput;
        type Ctx = RunCtxImpl;
    }

    impl SyncStage for DMulStage {

        fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
            // tokio::time::sleep(tokio::time::Duration::from_millis(random_range(0..10))).await;
            // #[cfg(any(test, feature = "mock_global_ctx"))]
            // let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock();

            let deive_idx = ctx.tho_device_idx();

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
                    let device_idx = ctx.tho_device_idx();
                    let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(device_idx, false);
                    debug!("MulTask::do_run: ctx.name={:?} device_idx={} is_all_on_cpu={is_all_on_cpu}","ctx.name()",device_idx);

                    if !is_all_on_cpu {
                        return Err(ThoError::oom());
                    }
                }
                FailUntil::Never => {}
            }

            debug!("MulTask::do_run: running ctx.name={:?} device_idx={} ","ctx.name()",ctx.tho_device_idx());

            Ok(DMulStageOutput {
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

    impl Adjustable for DMulStage {}

    impl OutId for DMulStage {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            DMulStageOutputId(format!("multiplied_{}", input_id.0).leak())
        }

    }

    #[derive(Debug)]
    struct DMulStageOutput {
        id: DMulStageOutputId,
        data: DummyTensor,
    }

    #[async_trait]
    impl Data for DMulStageOutput {
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
    impl CopyToDevice for DMulStageOutput {
        async fn copy_to_device(&self, _dst_device: usize) -> Result<Self, ThoError> {
            Ok(DMulStageOutput {
                id: self.id.clone(),
                data: self.data.clone(),
            })
        }
    }

    #[derive(Debug)]
    struct DMulStageOutputBuilder {
        id: DMulStageOutputId,
        partial_output: Vec<DMulStageOutput>,
    }

    impl DMulStageOutputBuilder {
        pub fn new(id: DMulStageOutputId) -> Self {
            Self {
                id,
                partial_output: vec![],
            }
        }
    }

    impl OutputBuilder for DMulStageOutputBuilder {
        type O = DMulStageOutput;

        async fn add_partial_output(&mut self, o: Self::O) -> Result<(), ThoError> {
            self.partial_output.push(o);
            Ok(())
        }

        async fn build(self) -> Result<Self::O, ThoError> {
            Ok(DMulStageOutput {
                id: self.id.clone(),

                data: DummyTensor(self.partial_output.into_iter().fold(vec![], |mut a, v| {
                    a.extend(v.data.0);
                    a
                })),
            })
        }
    }

    impl WithId for DMulStageOutput {
        type ID = DMulStageOutputId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    impl Output for DMulStageOutput {}

    ////////////
    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct DLoadAndClcStageId(&'static str);

    #[derive(Debug, Cached, Named)]
    #[cache_name = "MulTaskOutputId"]
    #[allow(dead_code)]
    struct DLoadAndClcStage {
        id: DLoadAndClcStageId,

        loader: Arc<DummyLoader>,
        mul: Arc<DMulStage>,
        mul_oom_runner: ClcOOMRunner<DMulStage, MetricsRangeSelector<<DLoadedTensor as Data>::Metrics>, DMulStageOutputBuilder>,
    }

    impl OutId for DLoadAndClcStage {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            self.mul.out_id(&self.loader.out_id(input_id))
        }
    }

    impl DLoadAndClcStage {
        pub fn new(
            id: DLoadAndClcStageId,
            k: f32,
            fail_until: FailUntil,
            len: usize,
            oom_runner: ClcOOMRunner<DMulStage, MetricsRangeSelector<<DLoadedTensor as Data>::Metrics>, DMulStageOutputBuilder>,
        ) -> Self {
            Self {
                id,
                loader: Arc::new(DummyLoader::new(len)),
                mul: Arc::new(DMulStage::new(DMulStageId("mul_task_0"), k, fail_until)),
                mul_oom_runner: oom_runner,
            }
        }
    }


    impl Stage for DLoadAndClcStage {
        type In<'a> = DLoaderInput;
        type Out = Arc<DevicesWrap<DMulStageOutput>>;
        type Ctx = RunCtxImpl;
    }


    impl AsyncStage for DLoadAndClcStage {


        async fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
            let ctx_for_loader = ctx.branch(false);
            let loaded = Arc::clone(&self.loader)
                .run_cached(input, ctx_for_loader)
                .await?;

            debug!("LoadAndMulTask::run_2: loaded.id={:?}", loaded.id());


            async move {
                let size = *loaded.size();
                Ok(Arc::new(DevicesWrap::wrap(
                    self.mul_oom_runner
                        .run(
                            &self.mul,
                            loaded,
                            ctx.branch(false),
                            &HashMap::new(),
                            size,
                        )
                        .await?
                )))
            }
            .await
        }

    }

    #[tokio::test]
    async fn test_run_cached_evicts_failed_computations_for_retry() -> Result<(), ThoError> {
        use futures_util::FutureExt;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        let loader = Arc::new(DummyLoader::new(3));
        let input = DLoaderInput { id: DLoaderInputId("recovery_retry") };
        let key = loader.out_id(input.id());
        let cache = DummyLoader::cache().await?;
        // Ordinary failures need a fresh computation too; a device restart is
        // neither necessary nor appropriate to evict a failed shared future.
        for failure in [ThoError::ContextReset, ThoError::oom(), ThoError::new("ordinary stage error")] {
            cache.insert(key.clone(), futures_util::future::ready(Err(failure.clone())).boxed().shared().into());
            let error = Arc::clone(&loader).run_cached(input.clone(), RunCtxImpl::new(0, "failed")).await.unwrap_err();
            assert_eq!(error.to_string(), failure.to_string());
            assert!(!cache.contains_key(&key));
        }
        let output = Arc::clone(&loader).run_cached(input, RunCtxImpl::new(0, "retry")).await.unwrap();
        assert_eq!(output.id(), &key);
        assert_eq!(loader.run_count.load(Relaxed), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_dummy() -> Result<(), ThoError> {
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

        let device_idx = 0;

        let loader = Arc::new(DummyLoader::new(3));

        let mut ctx_device_0 = RunCtxImpl::new(device_idx, "noname");
        let ctx_device_0_0 = ctx_device_0.branch(false);
        let ctx_device_0_1 = ctx_device_0.branch(false);

        let loaded_tensor_0_0 = Arc::clone(&loader)
            .run_cached(
                DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                },
                ctx_device_0_0,
            )
            .await;

        // debug!("{:?}", loaded_tensor_0_0);

        let loaded_tensor_0_1 = Arc::clone(&loader)
            .run_cached(
                DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                },
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
        let ctx_device_1_0 = ctx_device_1.branch(false);
        let ctx_device_1_data = ctx_device_1.branch(false);

        let loaded_tensor_1_0 = Arc::clone(&loader)
            .run_cached(
                DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                },
                ctx_device_1_0,
            )
            .await?;

        debug!("loaded_tensor_1_0={loaded_tensor_1_0:?}");
        let data = BorrowRef::borrow_ref(&loaded_tensor_1_0, 1).await?;

       assert_eq!(data.as_ref().data.0, [0.0, 1.0, 2.0]);


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

        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);

        let device_idx = 0;

        let loader = Arc::new(DummyLoader::new(3));

        let mut ctx_device_0 = RunCtxImpl::new(device_idx, "noname");
        let ctx_device_0_0 = ctx_device_0.branch(false);

        let loaded_tensor_0_0 = Arc::clone(&loader)
            .run_cached(
                DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                },
                ctx_device_0_0,
            )
            .await?;

        {
            let loaded_tensor_on_device_1 = loaded_tensor_0_0.load_from_any_device_(1).await?;
            debug!("loaded_tensor_on_device_1={loaded_tensor_on_device_1:?}");
        }

        debug!("loaded_tensor_0_0={loaded_tensor_0_0:?} is_all_on_cpu={}", loaded_tensor_0_0.is_all_on_cpu().await);

        assert!(!loaded_tensor_0_0.is_all_on_cpu().await);

        DummyLoader::cache().await?.purge_gpu(Some(&HashSet::new()), 1, "test").await?;

        assert!(loaded_tensor_0_0.is_all_on_cpu().await);

        assert_eq!(1, DummyLoader::cache().await?.len());
        let purged = DummyLoader::cache().await?.purge_cpu(0).await;

        assert_eq!(0, purged);
        assert_eq!(1, DummyLoader::cache().await?.len());

        // Retention follows the cache area, not incidental output handles.
        drop(ctx_device_0);

        let purged = DummyLoader::cache().await?.purge_cpu(0).await;

        assert_eq!(1, purged);
        assert_eq!(0, DummyLoader::cache().await?.len());
        assert!(loaded_tensor_0_0.is_all_on_cpu().await);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_cached_cleanup_runs_after_last_area_with_live_output_handles() -> Result<(), ThoError> {
        use crate::runner::global_ctx::MockGlobalCtx;
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let global = Box::leak(Box::new(mock_global_ctx::build()
            .with_age(0).with_device_count(2).with_get_runner(2).with_register_cache()
            .with_run_purge_gpu_handler(move |closures| { tx.send(closures).unwrap(); Ok(()) })
            .finish()));
        let global_context = MockGlobalCtx::global_context();
        global_context.expect().return_const(&*global);
        let shutdown_context = MockGlobalCtx::is_shutting_down_context();
        shutdown_context.expect().return_const(false);
        let mut first_area = RunCtxImpl::new(1, "first area");
        let mut second_area = RunCtxImpl::new(1, "second area");
        let loader = Arc::new(DummyLoader::new(3));
        let first = Arc::clone(&loader).run_cached(DLoaderInput { id: DLoaderInputId("resource_1") }, first_area.branch(false)).await?;
        let second = Arc::clone(&loader).run_cached(DLoaderInput { id: DLoaderInputId("resource_1") }, second_area.branch(false)).await?;
        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(loader.run_count.load(Relaxed), 1);
        drop(first.load_from_any_device_(1).await?);

        // Exercise the actual Cached::run_cached registrations and shared-owner
        // Drop. Neither the cache nor the two returned handles should block eviction.
        drop(first_area);
        for cleanup in rx.recv().await.unwrap() { cleanup().await?; }
        assert!(!first.is_all_on_cpu().await, "the second area still uses this entry");
        drop(second_area);
        for cleanup in rx.recv().await.unwrap() { cleanup().await?; }
        assert!(first.is_all_on_cpu().await);
        assert!(second.is_on_device(0));
        assert_eq!(DummyLoader::cache().await?.len(), 1);
        assert!(rx.try_recv().is_err());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_clc_no_oom_dummy() -> Result<(), ThoError> {
        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;

        RealGlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let _device_idx = 0;

        let load_and_mut = Arc::new(DLoadAndClcStage::new(
            DLoadAndClcStageId("load_and_clc_id"),
            2.,
            FailUntil::Never,
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel {
                    max: MAX_PARALLEL,
                    step: 10,
                },
                Box::new(|id| DMulStageOutputBuilder::new(id)),
            ),
        ));
        let mut ctx_device_0 = RunCtxImpl::new(1, "noname");
        let ctx_device_0_run = ctx_device_0.branch(false);
        let ctx_device_0_data = ctx_device_0.branch(false);

        let multed = load_and_mut.run_cached(
            DLoaderInput {
                id: DLoaderInputId("resource_1"),
            },
            ctx_device_0_run,
            )
            .await?;

        let data = BorrowRef::borrow_ref(&multed, 1).await?;
           assert_eq!(
               data.as_ref().data.0,
               (0..100).map(|v| v as f32 * 2.).collect::<Vec<_>>()
           );



        debug!("multed={multed:?}");

        Ok(())
    }

    fn mul_check(mul_k: f32, res: &str, out: &DMulStageOutput) {
        let self_k = parse_numeric_part(res).unwrap_or(1.0) as f32;

        assert_eq!(
            *out.data.0,
            (0..100)
                .map(|v| v as f32 * self_k * mul_k)
                .collect::<Vec<_>>(),
        );

        debug!("mul_check: multed_{res}={out:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_clc_oom_till_max_parallel_dummy() -> Result<(), ThoError> {
        crate::runner::global_ctx::GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let device_idx = 1;
        const MUL_K: f32 = 2.;

        let load_and_mut = Arc::new(DLoadAndClcStage::new(
            DLoadAndClcStageId("load_and_clc_id"),
            MUL_K,
            FailUntil::NaxParallel(20),
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel { max: 20, step: 10 },
                Box::new(|id| DMulStageOutputBuilder::new(id)),
            ),
        ));

        super::run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            device_idx,
            |device_idx, _cache_area, name| RunCtxImpl::new(device_idx, name),
            |res, out| {
                mul_check(MUL_K, res, out.as_ref());
            },
            true,
        )
        .await?;

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
    async fn test_load_and_clc_oom_compete_for_max_parallel_dummy() -> Result<(), ThoError> {
        crate::runner::global_ctx::GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let device_idx = 1;

        const MUL_K: f32 = 2.;
        let load_and_mut = Arc::new(DLoadAndClcStage::new(
            DLoadAndClcStageId("load_and_clc_id"),
            MUL_K,
            FailUntil::NaxParallel(MAX_PARALLEL),
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel {
                    max: MAX_PARALLEL,
                    step: *MaxParallelStep::SLOW,
                },
                Box::new(|id| DMulStageOutputBuilder::new(id)),
            ),
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            device_idx,
            |device_idx, _cache_area, name| RunCtxImpl::new(device_idx, name),
            |res, out| {
                mul_check(MUL_K, res, out.as_ref());
            },
            true,
        )
        .await?;

        Ok::<_, ThoError>(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_clc_oom_splits_dummy() -> Result<(), ThoError> {
        crate::runner::global_ctx::GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);

        let device_idx = 1;

        const MUL_K: f32 = 2.;
        let load_and_mut = Arc::new(DLoadAndClcStage::new(
            DLoadAndClcStageId("load_and_clc_id"),
            MUL_K,
            FailUntil::MemLimit(12),
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::Splits {},
                Box::new(|id| DMulStageOutputBuilder::new(id)),
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
            device_idx,
            |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
            move |res, out| {
                mul_check(MUL_K, res, out.as_ref());
            },
            true,
        )
        .await?;

        Ok::<_, ThoError>(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_clc_oom_till_dump_all_to_cpu_dummy() -> Result<(), ThoError> {
        //TODO check if with_is_all_on_cpu mocked correctly

        crate::runner::global_ctx::GlobalCtx::init_devices(2)?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, false);


    let device_idx = 1;
        const MUL_K: f32 = 2.;
        let load_and_mut = Arc::new(DLoadAndClcStage::new(
            DLoadAndClcStageId("load_and_clc_id"),
            MUL_K,
            FailUntil::DumpedToCpu,
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::DumpedToCpu,
                Box::new(|id| DMulStageOutputBuilder::new(id)),
            ),
        ));

        super::run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            device_idx,
            |device_idx, _cache_area, name| RunCtxImpl::new(device_idx, name),
            |res, out| {
                mul_check(MUL_K, res, out.as_ref());
            },
            true,
        )
        .await?;

        Ok(())
    }

}
