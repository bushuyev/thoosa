use std::ops::Range;
use std::sync::Arc;
use async_trait::async_trait;
use tch::Kind;
use cached_derive::Cached;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, Data, MonoMetrics, WithId};
use thoosa_with_data::with_data;
use crate::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use crate::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::tests::composite_input::dims::{ABDims, AStageOutputDims};
use crate::torch_runner::tests::composite_input::loader_a::{AStage, AStageInput, AStageOutput};
use crate as thoosa_common;
use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MAX_PARALLEL};
use crate::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use crate::tho_device::ThoDevice;
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use crate::torch_runner::tests::composite_input::loader_b::{BStage, BStageInput, BStageOutput};
use crate::torch_runner::tests::model_test::common::{ClsOut, ModelTestId};

#[derive(Debug, Named, Clone, DummyData)]
struct ABClcStageInput {
    id:SynKey,
    from_a:Arc<DevicesWrap<AStageInput>>,
    from_b:Arc<DevicesWrap<BStageInput>>,
}

impl ABClcStageInput {
    pub fn new(from_a: Arc<DevicesWrap<AStageInput>>, from_b: Arc<DevicesWrap<BStageInput>>) -> Self {
        Self {
            id: mono_key!(0, "zzz").into(),
            from_a,
            from_b
        }
    }
}

impl WithId for ABClcStageInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}



impl Input for ABClcStageInput {}

#[derive(Debug, Named, Clone, DummyData)]
struct ABClcStageOutput {
    id:SynKey,
    value:f32
}


impl WithId for ABClcStageOutput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

impl Output for ABClcStageOutput {}

#[async_trait]
impl thoosa_shared::CopyToDevice for ABClcStageOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            value: self.value,
        })
    }
}


#[derive(Debug, Named, Cached)]
struct ABClcStage {
    a_stage:Arc<AStage>,
    b_stage:Arc<BStage>,
    ab_clc_sync:Arc<ABClcSyncStage>
}

impl Stage for ABClcStage {
    type Ctx = RunCtxImpl;
    type In<'a> = ABClcStageInput;
    type Out = ABClcStageOutput;
}

impl ABClcStage {
    pub fn new() -> Self {
        Self {
            a_stage: Arc::new(AStage::new()),
            b_stage: Arc::new(BStage::new()),
            ab_clc_sync: Arc::new(ABClcSyncStage::new())
        }
    }
}

#[derive(Debug)]
struct ABClcSyncStageInputWrapper {
    id:SynKey,
    a_out:Arc<DevicesWrap<AStageOutput>>,
    b_out:Arc<DevicesWrap<BStageOutput>>
}

impl ABClcSyncStageInputWrapper {
    pub fn new(a_out: Arc<DevicesWrap<AStageOutput>>, b_out: Arc<DevicesWrap<BStageOutput>>) -> Self {
        Self {
            id: a_out.id() + b_out.id(),
            a_out,
            b_out
        }
    }
}

impl WithDataRef for ABClcSyncStageInputWrapper {
    type O<'o> = ABClcSyncStageInput<'o>;

    fn with_data_ref<'a, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, permits_ctx: P, trace_ctx: Option<&'static str>, do_run: F) -> impl Future<Output=Result<R, ThoError>> + Send + 'a
    where
            for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
            P: RunCtx + 'a
    {
        async {
            todo!()
        }
    }
}

impl WithSelectDataRef<ABClcSyncStageInputWrapperSelector> for ABClcSyncStageInputWrapper {
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, _permits_ctx: P, selector: ABClcSyncStageInputWrapperSelector, trace_ctx: Option<&'static str>, do_run: F) -> impl Future<Output=Result<R, ThoError>> + Send + 'a
    where
            for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
            P: RunCtx + 'a
    {
        async {
            todo!()
        }
    }
}

impl WithId for ABClcSyncStageInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
#[derive(Debug, Clone, Default)]
struct ABClcSyncStageInputWrapperSelector {

}

impl From<std::ops::Range<MonoMetrics<2, 1>>> for ABClcSyncStageInputWrapperSelector {
    fn from(value: Range<MonoMetrics<2, 1>>) -> Self {
        todo!()
    }
}

impl Adjustable for ABClcSyncStageInputWrapperSelector {}


impl AsyncStage for ABClcStage {
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, mut ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {

        let input = Arc::into_inner(input).unwrap().take_data(0).await?;

        let a_out = Arc::clone(&self.a_stage).run_cached(Arc::clone(&input.from_a), ctx.branch(false)).await?;
        let b_out = Arc::clone(&self.b_stage).run_cached(Arc::clone(&input.from_b), ctx.branch(false)).await?;

        let out_id = self.ab_clc_sync.out_id(input.id());


        let input = Arc::new(ABClcSyncStageInputWrapper::new(a_out, b_out));
        let result = OOMHandler::new(
            vec![],
            None,
        )
            .run(
                &self.ab_clc_sync,
                input,
                ctx,
                Contexts::new(
                    NoopInputContext(None),
                    NoopRunnableContext {},
                    NoopRunCtxContext {}
                ),
            )
            .await?;




        Ok(Arc::new(DevicesWrap::wrap( ABClcStageOutput {
            id: out_id,
            value: 0.0
        })))

    }
}


#[derive(Debug, Named)]
struct ABClcSyncStage {
    id:MonoKey
}

impl ABClcSyncStage {
    pub fn new() -> Self {
        Self {
            id: mono_key!(3, "ABClcSyncStage"),
        }
    }
}

impl Adjustable for ABClcSyncStage {}

impl WithId for ABClcSyncStageInput<'_> {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[derive(Debug)]
struct ABClcSyncStageInput<'a> {
    id:SynKey,
    from_a:&'a AStageOutput,
    from_b:&'a BStageOutput,
}

impl <'a> ABClcSyncStageInput<'a> {
    pub fn new(from_a: &'a AStageOutput, from_b: &'a BStageOutput) -> Self {

        Self {
            id: from_a.id() + from_b.id(),
            from_a,
            from_b
        }
    }
}

impl Data for ABClcSyncStageInput<'_> {
    type Metrics = MonoMetrics<1, 2>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        todo!()
    }

    fn device(&self) -> usize {
        todo!()
    }

    fn mem_size(&self) -> u64 {
        todo!()
    }
}

impl OutId for ABClcSyncStage {
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl Input for ABClcSyncStageInput<'_> {}

impl Stage for ABClcSyncStage {
    type Ctx = RunCtxImpl;
    type In<'a> = ABClcSyncStageInput<'a>;
    type Out = ABClcStageOutput;
}

impl SyncStage for ABClcSyncStage {
    fn do_run<'f>(self: Arc<Self>, _input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {

        todo!()
    }
}