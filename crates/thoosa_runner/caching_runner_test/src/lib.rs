use caching_runner::adjusters::dump_to_cpu::{DumpToCpu, DumpToCpuContextImpl};
use caching_runner::adjusters::permits::{
    PermitsAdjustContextImpl, PermitsAdjuster, PermitsAndDumpToCpuAdjustableContex,
};
use caching_runner::adjusters::splits::{OutputBuilder, SplitsAdjuster, SplitsAdjusterContextImpl};
use caching_runner::common::{OutId, RunCtx};
use caching_runner::global_ctx::GlobalCtx;
use caching_runner::oom_handler::{
    Adjustable, Adjusters, Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext,
    OOMHandler,
};
use caching_runner::permits::{MAX_PERMITS, NumPermits, PermitStep};

use caching_runner::common::cachable::{Cached, DataCacheWrapper};
use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
use log::debug;
use parking_lot::RwLock;
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, DivAssign, Range};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::sync::atomic::Ordering::Relaxed;
use thoosa_shared::ThoosaError;
use thoosa_shared::traits::{CachableIdTr, Data, Named, Select, WithId};
use tokio::task::JoinSet;
use caching_runner::common::stage_input::StageInput;

type StageIn<'a, S> = <S as Stage>::In<'a>;
type StageInStatic<S> = StageIn<'static, S>;
type StageInMetrics<S> = <StageInStatic<S> as Data>::Metrics;

#[derive(Debug)]
pub enum FailUntil {
    MemLimit(u64),
    MinPermits(NumPermits),
    DumpedToCpu,
    Never,
}

//for tests in torch_runner
pub struct ClcOOMRunner<R: Stage + 'static, S:Debug+Default+Clone+Adjustable> {
    pd: PhantomData<(R,S)>,
    config: OOMRunnerConfig,
    output_builder: Box<
        dyn Fn(<<R as Stage>::Out as WithId>::ID) -> Box<dyn OutputBuilder<O = <R as Stage>::Out>>
            + Send
            + Sync,
    >,
}

impl<R: Stage + Stage, S:Debug+Default+Clone+Adjustable> Debug for ClcOOMRunner<R, S> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MulOOMRunner")
            .field("config", &self.config)
            .finish()
    }
}

impl<R, S:Debug+Default+Clone+Adjustable> ClcOOMRunner<R, S>
where
    R: Stage + Adjustable + 'static,
    StageInStatic<R>: Adjustable,
    R::Ctx: Adjustable,
{
    pub fn new(
        config: OOMRunnerConfig,
        output_builder: Box<
            dyn Fn(
                    <<R as Stage>::Out as WithId>::ID,
                ) -> Box<dyn OutputBuilder<O = <R as Stage>::Out>>
                + Send
                + Sync,
        >,
    ) -> Self {
        Self {
            pd: PhantomData,
            config,
            output_builder,
        }
    }

    pub async fn run(
        &self,
        runnable: &Arc<R>,
        input: Arc<DataCacheWrapper<StageInStatic<R>>>,
        ctx: Arc<RwLock<<R as Stage>::Ctx>>,
        required_ids: &HashMap<&'static str, &HashSet<&dyn CachableIdTr>>,
    ) -> Result<<R as Stage>::Out, ThoosaError>
    where
        R: SyncStage + Adjustable + 'static,
        StageInStatic<R>: Debug + Adjustable + Data,
        for<'a> <R as Stage>::In<'a>: WithId,
        R::Out: Send + Sync,
        R::Ctx: Adjustable,
        R: OutId,
        StageInMetrics<R>: for<'a> AddAssign<&'a StageInMetrics<R>>,
        StageInMetrics<R>: DivAssign<i64>,
        StageInMetrics<R>: PartialOrd<i64>,
        for<'a, 'b> &'a StageInMetrics<R>: Add<&'b StageInMetrics<R>, Output = StageInMetrics<R>>,
        StageInMetrics<R>: PartialOrd,
        <R as Stage>::In<'static>: Select<std::ops::Range<<<R as Stage>::In<'static> as Data>::Metrics>, Output=<R as Stage>::In<'static>>
    {
        let (device_idx, ctx_name) = {
            let guard = ctx.read();
            (guard.device_idx(), guard.name())
        };

        let size = input.size().clone();
        let mut min_split_len = size.clone();
        min_split_len /= 10;

        debug!("ClcOOMRUnner::run: min_split_len={min_split_len:?} size={size:?}");
        let out_id = runnable.out_id(input.id()); //.clone();
        let out_key = Arc::new(out_id.clone()) as Arc<dyn CachableIdTr>;

        match self.config {
            OOMRunnerConfig::Permits { max, step } => {
                OOMHandler::new(
                    vec![Adjusters::RunContext(Box::new(PermitsAdjuster::new(
                        max,
                        step,
                        move |n| GlobalCtx::global().acquire_many(device_idx, n, out_key.clone()),
                    )))],
                    None,
                )
                .run(
                    &runnable,
                    input,
                    ctx,
                    Contexts::new(
                        NoopInputContext(None),
                        NoopRunnableContext {},
                        PermitsAdjustContextImpl::new(ctx_name),
                    ),
                )
                .await
            }

            OOMRunnerConfig::Splits => {
                OOMHandler::new(
                    vec![Adjusters::Input(Box::new(
                        SplitsAdjuster::<StageInStatic<R>, std::ops::Range<<<R as Stage>::In<'static> as Data>::Metrics>>::new(min_split_len),
                    ))],
                    None,
                )
                .run(
                    &runnable,
                    input,
                    ctx,
                    Contexts::new(
                        SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id)),
                        NoopRunnableContext {},
                        NoopRunCtxContext {},
                    ),
                )
                .await
            }
            OOMRunnerConfig::PermitsAndSplits {
                max: _max,
                step: _step,
            } => {
                OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(PermitsAdjuster::new(
                            MAX_PERMITS,
                            *PermitStep::TWENTY,
                            move |n| GlobalCtx::global().acquire_many(device_idx, n, out_key.clone()),
                        ))),
                        Adjusters::Input(Box::new(SplitsAdjuster::<StageInStatic<R>, std::ops::Range<<<R as Stage>::In<'static> as Data>::Metrics>>::new(
                            min_split_len,
                        ))),
                    ],
                    None,
                )
                .run(
                    &runnable,
                    input,
                    ctx,
                    Contexts::new(
                        SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id)),
                        NoopRunnableContext {},
                        PermitsAdjustContextImpl::new(ctx_name),
                    ),
                )
                .await
            }
            OOMRunnerConfig::DumpedToCpu => {
                OOMHandler::new(
                    vec![
                        Adjusters::RunContext(Box::new(PermitsAdjuster::new(
                            MAX_PERMITS,
                            *PermitStep::FIFTY,
                            move |n| GlobalCtx::global().acquire_many(device_idx, n, out_key.clone()),
                        ))),
                        Adjusters::RunContext(Box::new(DumpToCpu::new())),
                        Adjusters::Input(Box::new(SplitsAdjuster::<StageInStatic<R>, std::ops::Range<<<R as Stage>::In<'static> as Data>::Metrics>>::new(
                            min_split_len,
                        ))),
                    ],
                    None,
                )
                .run(
                    &runnable,
                    input,
                    ctx,
                    Contexts::new(
                        SplitsAdjusterContextImpl::new(size, (self.output_builder)(out_id)),
                        NoopRunnableContext {},
                        PermitsAndDumpToCpuAdjustableContex::new(
                            PermitsAdjustContextImpl::new(ctx_name),
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
    Permits { max: NumPermits, step: NumPermits },
    DumpedToCpu,
    Splits,
    PermitsAndSplits { max: NumPermits, step: NumPermits },
}

pub async fn run_mul_task_invocations<R, F, CF>(
    load_and_mul: Arc<R>,
    resources: &[&'static str],
    device_idx: usize,
    ctx_fac: F,
    check: CF,
) -> Result<(), ThoosaError>
where
    R: Cached + OutId + Send + Sync + 'static,
    for<'a> <R as Stage>::In<'a>: Data + WithId+ Send + Sync,
    R::Out: Data + Send + Sync,
    StageInStatic<R>: From<&'static str>,
    F: Fn(usize/*device*/, u64/*cache_area*/, &'static str/*name*/) -> <R as Stage>::Ctx + Send + Sync + 'static,
    CF: Fn(&'static str, &<R as Stage>::Out) + Send + Sync + Clone + 'static,
    <R as Stage>::Out: Data,
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

                let ctx_device = Arc::new(RwLock::new(ctx_fac_arc(device_idx, 0, name)));
                let multed = load_and_mul
                    .run_cached(
                        StageInput::from_plain(Arc::new(<&str as Into<StageInStatic<R>>>::into(res))),
                        ctx_device.clone(),
                    )
                    .await?;

                multed
                    .with_data_ref(device_idx, None, |guard: &<R as Stage>::Out| {
                        // let guard = multed.load_from_any_device(device_idx, "in test".into()).await;

                        check_arc(res, &guard);

                        debug!("run_mul_task_invocations: task {idx}: device_idx={device_idx} thread id={:?} done",std::thread::current().id());
                        debug!("run_mul_task_invocations: task {idx}: multed=\n{:?}", guard);

                        res_cnt_arc.fetch_add(1, Relaxed);

                        Ok(())
                    })
                    .await?;

                debug!("run_mul_task_invocations: task {idx}: all done");

                Ok::<(), ThoosaError>(())
            }
        })
        .collect::<JoinSet<_>>()
        .join_all()
        .await
        .into_iter()
        .collect::<Result<(), ThoosaError>>();

    debug!("r={r:?} res_cnt={res_cnt:?}");
    assert!(r.is_ok());

    assert_eq!(resources.len() as u32, res_cnt.load(Relaxed));

    // assert_eq!(thread_ids.read().iter().collect::<HashSet<_>>().len(), thread_ids.read().len());
    assert!(thread_ids.read().iter().collect::<HashSet<_>>().len() > 1);

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::cmp::{max, min};
    use std::collections::{HashMap, HashSet};
    use std::fmt::Debug;

    use super::{ClcOOMRunner, FailUntil, OOMRunnerConfig, run_mul_task_invocations};
    use ambassador::Delegate;
    use async_trait::async_trait;
    use caching_runner::adjusters::splits::OutputBuilder;
    use caching_runner::common::cachable::{Cached, DataCacheWrapper};
    use caching_runner::common::runnable::{AsyncStage, Stage, SyncStage};
    use caching_runner::common::{ambassador_impl_PermitsCtx, DropIf, OutId, PermitsCtx, RunCtx};
    use caching_runner::global_ctx::GlobalCtx;
    use caching_runner::oom_handler::Adjustable;
    use caching_runner::permits::{MAX_PERMITS, NumPermits, PermitStep, RunnerPerm};
    use caching_runner::test_utils::DummyTensor;
    use dashmap::DashMap;
    use futures_util::future::{BoxFuture, Shared};
    use log::{debug, error};
    use parking_lot::RwLock;
    use std::mem;
    use std::ops::Range;
    use std::pin::Pin;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::Relaxed;
    use std::sync::{Arc, LazyLock};
    use cached_derive::Cached;
    use caching_runner::common::stage_input::StageInput;
    use thoosa_shared::CopyToDevice;
    use thoosa_shared::applicables_runner::{Input, Output};
    use thoosa_shared::traits::{Data, Named, Select, SimpleId, WithId};
    use thoosa_shared::{MyError, ThoosaError};

    #[ctor::ctor]
    fn init() {
        thoosa_shared::test_utils::init_test_logging();
    }

    static TELEMETRY: std::sync::LazyLock<()> = LazyLock::new(|| {
        console_subscriber::ConsoleLayer::builder()
            .retention(std::time::Duration::from_secs(60))
            .init();
    });

    #[derive(Debug, Delegate)]
    #[delegate(PermitsCtx, target = "perms_ctx")]
    struct DummyRunCtx {
        device_idx: usize,
        loaded_tensors_cached: Vec<SimpleId>,
        multed_tensors_cached: Vec<DMulTaskOutputId>,
        cache_area: u64,
        name: &'static str,
        perms_ctx: DummyPermsCtx,
    }

    #[derive(Debug)]
    struct DummyPermsCtx {//TODO it's same as TorchPermsCtx, replace
        name: &'static str,
        permits: Vec<RunnerPerm>,
    }

    impl DummyPermsCtx {
        fn new(name: &'static str) -> Self {
            Self {
                name,
                permits: Vec::new(),
            }
        }
    }

    impl Named for DummyPermsCtx {
        fn name(&self) -> &'static str {
            self.name
        }
    }

    impl PermitsCtx for DummyPermsCtx {
        fn add_permit(&mut self, permit: RunnerPerm) {
            self.permits.push(permit);
        }

        fn drop_permits(&mut self, drop_if: DropIf) {

            let acquired_permits: NumPermits = self.acquired_permits();
            debug!("drop_permits::drop_permits acquired_permits={acquired_permits} ctx.name={} drop_if={drop_if:?}",self.name);

            match drop_if {
                DropIf::Always => {
                    debug!("TorchRunCtx: ctx.name={} dropping permits", self.name);
                    self.permits.clear();
                }
                DropIf::IfNotMax => {
                    if acquired_permits < MAX_PERMITS {
                        debug!("TorchRunCtx: ctx.name={} dropping permits", self.name);
                        self.permits.clear();
                    }
                }
            }
        }

        fn acquired_permits(&self) -> NumPermits {
            self.permits.iter().map(|p| p.num_permits()).sum()
        }
    }

    impl Adjustable for DummyRunCtx {}
    impl DummyRunCtx {
        pub fn new(device_idx: usize, cache_area: u64, name: &'static str) -> Self {
            Self {
                device_idx,
                loaded_tensors_cached: vec![],
                multed_tensors_cached: vec![],
                cache_area,
                name,
                perms_ctx: DummyPermsCtx::new(name),
            }
        }
    }

    impl RunCtx for DummyRunCtx {
        fn device_idx(&self) -> usize {
            self.device_idx
        }

        fn purge_gpu_caches(ctx: Arc<RwLock<Self>>) -> BoxFuture<'static, Result<(), ThoosaError>>
        where
            Self: Sized,
        {
            Box::pin(async move {
                let device_idx = ctx.read().device_idx;
                assert_ne!(device_idx, 0);

                let (cache_area, loaded_tensors_cached, mul_output_cached) = {
                    let mut guard = ctx.write();

                    (
                        guard.cache_area,
                        mem::replace(&mut guard.loaded_tensors_cached, Vec::new()),
                        mem::replace(&mut guard.multed_tensors_cached, Vec::new()),
                    )
                };
                debug!("purge_gpu_caches: device_idx={device_idx} loaded_tensors_cached={loaded_tensors_cached:?} mul_output_cached={mul_output_cached:?}");

                tokio::try_join!(
                    tokio::spawn(DummyLoader::purge_gpu_cache_area(loaded_tensors_cached, cache_area, device_idx)),
                    tokio::spawn(DLoadAndMulTask::purge_gpu_cache_area(mul_output_cached, cache_area, device_idx))
                )?;

                Ok(())
            })
        }

        fn cache_area(&self) -> u64 {
            self.cache_area
        }
    }

    impl Named for DummyRunCtx {
        fn name(&self) -> &'static str {
            self.name
        }
    }

    #[derive(Debug, cached_derive::Cached)]
    #[cache_name = "LoadedTensorId"]
    struct DummyLoader {
        run_count: AtomicUsize,
        len: usize,
    }

    impl Adjustable for DLoadedTensor {}

    impl OutId for DummyLoader {
        fn out_id(
            &self,
            input_id: &<Self::In<'static> as WithId>::ID,
        ) -> <Self::Out as WithId>::ID {
            SimpleId::from(format!("loaded_{}", input_id.0))
        }

    }

    struct DLoadedTensor {
        id: SimpleId,
        data: DummyTensor,
    }

    impl Select<std::ops::Range<<DLoadedTensor as Data>::Metrics>> for DLoadedTensor {
        type Output = DLoadedTensor;

        fn subview(&self, selector: Range<<DLoadedTensor as Data>::Metrics>) -> Result<Self::Output, ThoosaError> {
            Ok(DLoadedTensor {
                id: self.id.clone(),
                data: DummyTensor(self.data.0[selector.start as usize..min(self.data.0.len(), selector.end as usize)].to_vec()),
            })
        }
    }

    impl DLoadedTensor {}

    impl Debug for DLoadedTensor {
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

        fn device(&self) -> usize {
            0
        }
    }

    #[async_trait]
    impl CopyToDevice for DLoadedTensor {
        async fn copy_to_device(&self, dst_device: usize) -> Self {
            DLoadedTensor {
                id: self.id.clone(),
                data: self.data.copy_to_device(dst_device).await,
            }
        }
    }

    impl WithId for DLoadedTensor {
        type ID = SimpleId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }
    impl Output for DLoadedTensor {}

    impl Input for DLoadedTensor {}

    impl DummyLoader {
        fn new(len: usize) -> DummyLoader {
            DummyLoader {
                run_count: Default::default(),
                len,
            }
        }
    }

    #[derive(Debug, Clone, Eq, PartialEq, Hash)]
    pub(super) struct DLoaderInputId(&'static str);

    impl Named for DLoaderInputId {
        fn name(&self) -> &'static str {
            "DLoaderInputId"
        }
    }

    #[derive(Debug, Clone)]
    pub(super) struct DLoaderInput {
        id: DLoaderInputId,
    }
    impl Input for DLoaderInput {}
    impl WithId for DLoaderInput {
        type ID = DLoaderInputId;

        fn id(&self) -> &Self::ID {
            &self.id
        }
    }

    #[async_trait]
    impl CopyToDevice for DLoaderInput {
        async fn copy_to_device(&self, dst_device: usize) -> Self {
           self.clone()
        }
    }

    #[async_trait]
    impl Data for DLoaderInput {
        type Metrics = ();

        fn size(&self) -> Self::Metrics {
            ()
        }

        fn device(&self) -> usize {
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
        type Out = DLoadedTensor;
        type Ctx = DummyRunCtx;
    }

    #[async_trait]
    impl AsyncStage for DummyLoader {

        async fn run(
            self: Arc<Self>,
            input: StageInput<Self::In<'static>>,
            ctx: Arc<parking_lot::RwLock<Self::Ctx>>,
        ) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {
            self.run_count.fetch_add(1, Relaxed);

            let k = caching_runner::test_utils::parse_numeric_part(input.id().0)
                .unwrap_or(1.0) as f32;
            let out_id = self.out_id(input.id());

            let (device, cache_area) = {
                let guard = ctx.read();
                (guard.device_idx, guard.cache_area)
            };

            Ok(Arc::new(DataCacheWrapper::new(DLoadedTensor {
                id: out_id,
                data: (0..self.len)
                    .map(|v| v as f32 * k)
                    .collect::<Vec<f32>>()
                    .into(),
            }, device, 2, cache_area)))
        }

        fn on_cached<'a>(&self, input: &<<Self as Stage>::In<'static> as WithId>::ID, ctx: &Arc<RwLock<Self::Ctx>>)
        where
            Self:'a{
            ctx.write()
                .loaded_tensors_cached
                .push(self.out_id(input));
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
        type Ctx = DummyRunCtx;
    }

    impl SyncStage for DMulTask {

        fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Self::Out, ThoosaError> {
            // tokio::time::sleep(tokio::time::Duration::from_millis(random_range(0..10))).await;

            let deive_idx = ctx.read().device_idx;

            match self.fail_until {
                FailUntil::MemLimit(mem_limit) => {
                    let _device_idx = deive_idx;
                    if input.data.0.len() > mem_limit as usize {
                        return Err(MyError::OutOfMemory.into());
                    }
                }
                FailUntil::MinPermits(min_permits) => {
                    let acquired_permits = ctx.read().acquired_permits();
                    debug!("MulTask::do_run: ctx={ctx:?} acquired_permits={acquired_permits} min_permits={min_permits}");

                    if acquired_permits < min_permits {
                        return Err(MyError::OutOfMemory.into());
                    }
                }

                FailUntil::DumpedToCpu => {
                    let is_all_on_cpu = GlobalCtx::global().is_all_on_cpu(ctx.read().device_idx());
                    debug!("MulTask::do_run: ctx.name={:?} device_idx={} is_all_on_cpu={is_all_on_cpu}",ctx.read().name(),ctx.read().device_idx());

                    if !is_all_on_cpu {
                        return Err(MyError::OutOfMemory.into());
                    }
                }
                FailUntil::Never => {}
            }

            debug!("MulTask::do_run: running ctx.name={:?} device_idx={} ",ctx.read().name(),ctx.read().device_idx());

            Ok(DMulTaskOutput {
                id: self.out_id(Input::id(input)),
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

    #[derive(Debug, Clone, Hash, Eq, PartialEq)]
    struct DMulTaskOutputId(&'static str);

    impl Named for DMulTaskOutputId {
        fn name(&self) -> &'static str {
            "MulTaskOutputId"
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


        fn size(&self) -> Self::Metrics {
            [self.data.0.len()]
        }

        fn device(&self) -> usize {
            0
        }
    }

    #[async_trait]
    impl CopyToDevice for DMulTaskOutput {
        async fn copy_to_device(&self, _dst_device: usize) -> Self {
            DMulTaskOutput {
                id: self.id.clone(),
                data: self.data.clone(),
            }
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

        fn add_partial_output(&mut self, o: Self::O) {
            self.partial_output.push(o);
        }

        fn build(self: Box<Self>) -> Self::O {
            DMulTaskOutput {
                id: self.id.clone(),

                data: DummyTensor(self.partial_output.into_iter().fold(vec![], |mut a, v| {
                    a.extend(v.data.0);
                    a
                })),
            }
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

    #[derive(Debug, Cached)]
    #[cache_name = "MulTaskOutputId"]
    #[allow(dead_code)]
    struct DLoadAndMulTask {
        id: DLoadAndMulTaskId,

        loader: Arc<DummyLoader>,
        mul: Arc<DMulTask>,
        mul_oom_runner: ClcOOMRunner<DMulTask, std::ops::Range<<DLoadedTensor as Data>::Metrics>>,
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
            oom_runner: ClcOOMRunner<DMulTask, std::ops::Range<<DLoadedTensor as Data>::Metrics>>,
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
        type Out = DMulTaskOutput;
        type Ctx = DummyRunCtx;
    }

    #[async_trait]
    impl AsyncStage for DLoadAndMulTask {


        async fn run(self: Arc<Self>, input: StageInput<DLoaderInput>, ctx: Arc<parking_lot::RwLock<Self::Ctx>>) -> Result<Arc<DataCacheWrapper<Self::Out>>, ThoosaError> {
            let loaded = Arc::clone(&self.loader)
                .run_cached(input, Arc::clone(&ctx))
                .await?;

            debug!("LoadAndMulTask::run_2: loaded.id={:?}", loaded.id());

            let (device_idx, cache_area) = {
                let guard = ctx.read();
                (guard.device_idx(), guard.cache_area())
            };

            async move {

                Ok(Arc::new(DataCacheWrapper::new(self.mul_oom_runner
                    .run(
                        &self.mul,
                        loaded,
                        ctx,
                        &HashMap::from([("", &HashSet::new())]),
                    )
                    .await.map_err(|e|{
                        error!("DLoadAndMulTask::run: e={e:?}");
                        e
                     })?, device_idx, 2, cache_area)))
            }
            .await
        }

        fn on_cached<'a>(&self, input: &<<Self as Stage>::In<'static> as WithId>::ID, ctx: &Arc<RwLock<Self::Ctx>>)
        where
            Self:'a
        {
            ctx.write()
                .multed_tensors_cached
                .push(self.out_id(input));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_dummy() -> Result<(), ThoosaError> {
        let _global_ctx = GlobalCtx::init_devices(2);

        let device_idx = 0;

        let loader = Arc::new(DummyLoader::new(3));

        let ctx_device_0 = Arc::new(RwLock::new(DummyRunCtx::new(device_idx, 0, "noname")));

        let loaded_tensor_0_0 = Arc::clone(&loader)
            .run_cached(
                StageInput::from_plain(Arc::new(DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                })),
                Arc::clone(&ctx_device_0),
            )
            .await;

        debug!("{:?}", loaded_tensor_0_0);

        let loaded_tensor_0_1 = Arc::clone(&loader)
            .run_cached(
                StageInput::from_plain(Arc::new(DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                })),
                Arc::clone(&ctx_device_0),
            )
            .await;

        assert!(loaded_tensor_0_0.is_ok());
        assert!(loaded_tensor_0_1.is_ok());

        assert!(Arc::ptr_eq(
            &loaded_tensor_0_0.unwrap(),
            loaded_tensor_0_1.as_ref().unwrap()
        ));

        assert_eq!(loader.run_count.load(Relaxed), 1);

        let ctx_device_1 = Arc::new(RwLock::new(DummyRunCtx::new(1, 0, "noname")));

        let loaded_tensor_1_0 = Arc::clone(&loader)
            .run_cached(
                StageInput::from_plain(Arc::new(DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                })),
                Arc::clone(&ctx_device_1),
            )
            .await?;

        debug!("loaded_tensor_1_0={loaded_tensor_1_0:?}");
        loaded_tensor_1_0
            .with_data_ref(1, None, |data| {
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

        debug!("{:?}", _global_ctx);

        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
    async fn test_load_and_mul_no_oom_dummy() -> Result<(), ThoosaError> {
        let _global_ctx = GlobalCtx::init_devices(2);

        let _device_idx = 0;

        let load_and_mut = Arc::new(DLoadAndMulTask::new(
            DLoadAndMulTaskId("load_and_mul_id"),
            2.,
            FailUntil::Never,
            100,
            ClcOOMRunner::new(
                OOMRunnerConfig::Permits {
                    max: MAX_PERMITS,
                    step: 10,
                },
                Box::new(|id| Box::new(DMulTaskOutputBuilder::new(id))),
            ),
        ));
        let ctx_device_0 = Arc::new(RwLock::new(DummyRunCtx::new(1, 0, "noname")));

        let multed = load_and_mut
            .run_cached(
                StageInput::from_plain(Arc::new(DLoaderInput {
                    id: DLoaderInputId("resource_1"),
                })),
                ctx_device_0,
            )
            .await?;

        multed
            .with_data_ref(1, None, |data| {
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
        let self_k = caching_runner::test_utils::parse_numeric_part(res).unwrap_or(1.0) as f32;

        assert_eq!(
            *out.data.0,
            (0..100)
                .map(|v| v as f32 * self_k * mul_k)
                .collect::<Vec<_>>(),
        );

        debug!("mul_check: multed_{res}={out:?}");
    }

        #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
        async fn test_load_and_mul_oom_till_min_permits_dummy() -> Result<(), ThoosaError> {
            let _global_ctx = GlobalCtx::init_devices(3);

            let device_idx = 1;
            const MUL_K: f32 = 2.;

            let load_and_mut = Arc::new(DLoadAndMulTask::new(
                DLoadAndMulTaskId("load_and_mul_id"),
                MUL_K,
                FailUntil::MinPermits(20),
                100,
                ClcOOMRunner::new(
                    OOMRunnerConfig::Permits { max: 20, step: 10 },
                    Box::new(|id| Box::new(DMulTaskOutputBuilder::new(id))),
                ),
            ));

            super::run_mul_task_invocations(
                Arc::clone(&load_and_mut),
                &["resource_1", "resource_2", "resource_3"],
                device_idx,
                |device_idx, cache_area, name| DummyRunCtx::new(device_idx, cache_area, name),
                |res, out| {
                    mul_check(MUL_K, res, out);
                },
            )
            .await?;

            Ok(())
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 32)]
        async fn test_load_and_mul_oom_compete_for_permits_dummy() -> Result<(), ThoosaError> {
            let _global_ctx = GlobalCtx::init_devices(2);

            let device_idx = 1;

            const MUL_K: f32 = 2.;
            let load_and_mut = Arc::new(DLoadAndMulTask::new(
                DLoadAndMulTaskId("load_and_mul_id"),
                MUL_K,
                FailUntil::MinPermits(MAX_PERMITS),
                100,
                ClcOOMRunner::new(
                    OOMRunnerConfig::Permits {
                        max: MAX_PERMITS,
                        step: *PermitStep::TEN,
                    },
                    Box::new(|id| Box::new(DMulTaskOutputBuilder::new(id))),
                ),
            ));

            run_mul_task_invocations(
                Arc::clone(&load_and_mut),
                &["resource_1", "resource_2", "resource_3"],
                device_idx,
                |device_idx, cache_area, name| DummyRunCtx::new(device_idx, cache_area, name),
                |res, out| {
                    mul_check(MUL_K, res, out);
                },
            )
            .await?;

            Ok::<_, ThoosaError>(())
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
        async fn test_load_and_mul_oom_splits_dummy() -> Result<(), ThoosaError> {
            let _global_ctx = GlobalCtx::init_devices(2);

            let device_idx = 1;

            const MUL_K: f32 = 2.;
            let load_and_mut = Arc::new(DLoadAndMulTask::new(
                DLoadAndMulTaskId("load_and_mul_id"),
                MUL_K,
                FailUntil::MemLimit(12),
                100,
                ClcOOMRunner::new(
                    OOMRunnerConfig::Splits {},
                    Box::new(|id| Box::new(DMulTaskOutputBuilder::new(id))),
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
                |_device_idx, _cache_area, _name| DummyRunCtx::new(1, 0, "noname"),
                move |res, out| {
                    mul_check(MUL_K, res, out);
                },
            )
            .await?;

            Ok::<_, ThoosaError>(())
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
        async fn test_load_and_mul_oom_till_dump_all_to_cpu_dummy() -> Result<(), ThoosaError> {
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
                    Box::new(|id| Box::new(DMulTaskOutputBuilder::new(id))),
                ),
            ));

            super::run_mul_task_invocations(
                Arc::clone(&load_and_mut),
                &["resource_1", "resource_2", "resource_3"],
                device_idx,
                |device_idx, cache_area, name| DummyRunCtx::new(device_idx, cache_area, name),
                |res, out| {
                    mul_check(MUL_K, res, out);
                },
            )
            .await?;

            Ok(())
        }

        #[tokio::test(flavor = "multi_thread", worker_threads = 16)]
        async fn test_cache_areas() -> Result<(), ThoosaError> {
            let global_ctx = GlobalCtx::init_devices(2);

            let loader = Arc::new(DummyLoader::new(10));
            let resource_1_arc = DLoaderInput::from("resource_1");
            let loader_cache = <DummyLoader as Cached>::cache().await;

            let device_idx = 1;

            let cache_area_0 = global_ctx.next_cache_area()?;
            debug!("cache_area_0={:064b}", cache_area_0);

            let ctx_0 = Arc::new(parking_lot::RwLock::new(DummyRunCtx::new(
                device_idx,
                cache_area_0,
                "ctx_0",
            )));
            let resource_0_for_ctx_0 = Arc::clone(&loader)
                .run_cached(StageInput::from_plain(Arc::new(resource_1_arc.clone())), ctx_0.clone())
                .await?;

            let cache_area_1 = global_ctx.next_cache_area()?;
            debug!("cache_area_1={:064b}", cache_area_1);

            let ctx_1 = Arc::new(parking_lot::RwLock::new(DummyRunCtx::new(
                device_idx,
                cache_area_1,
                "ctx_1",
            )));
            let resource_0_for_ctx_1 = Arc::clone(&loader)
                .run_cached(StageInput::from_plain(Arc::new(resource_1_arc.clone())), ctx_1.clone())
                .await?;

            debug!(
                "cache_areas 0+1={:064b}",
                resource_0_for_ctx_1.cache_areas()
            );

            assert!(Arc::ptr_eq(&resource_0_for_ctx_0, &resource_0_for_ctx_1));
            assert_ne!(resource_0_for_ctx_1.cache_areas(), 0);

            DummyRunCtx::purge_gpu_caches(ctx_0.clone()).await?;
            debug!(
                "cache_areas   1={:064b}",
                resource_0_for_ctx_1.cache_areas()
            );
            assert_ne!(resource_0_for_ctx_1.cache_areas(), 0);

            DummyRunCtx::purge_gpu_caches(ctx_1.clone()).await?;
            debug!("cache_areas   ={:064b}", resource_0_for_ctx_1.cache_areas());
            assert_eq!(resource_0_for_ctx_1.cache_areas(), 0);

            assert_eq!(loader_cache.len(), 1);

            global_ctx.purge_cpu(0).await;
            debug!("loader_cache={loader_cache:?} len={}", loader_cache.len());
            assert_eq!(loader_cache.len(), 0);

            Ok(())
        }
}
