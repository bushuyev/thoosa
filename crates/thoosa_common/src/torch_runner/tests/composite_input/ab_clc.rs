use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Acquire;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tch::Kind;
use tracing::debug;
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
use crate::copy_ctx::CudaActorHandle;
use crate::ipc::cuda_process::encode_endpoint_payload;
use crate::ipc::cuda_wokrer::WorkerStageInput;
use crate::ipc::json_ser::SyncStageJsonWrapper;
use crate::runner::common::cachable::{Cached, DevicesWrap, WithDataRef, WithSelectDataRef};
use crate::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::tests::composite_input::dims::{ABDims, AStageOutputDims, BStageOutputDims};
use crate::torch_runner::tests::composite_input::loader_a::{AStage, AStageInput, AStageOutput};
use crate as thoosa_common;
use crate::ipc::ipc_cuda_utils::IpcTensorKind::Float;
use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MAX_PARALLEL};
use crate::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunCtxContext, NoopRunnableContext, OOMHandler};
use crate::tho_device::ThoDevice;
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use crate::torch_runner::tests::composite_input::{AB_CLC_STAGE_ID, AB_SYNC_CLC_STAGE_ID};
use crate::torch_runner::tests::composite_input::loader_b::{BStage, BStageInput, BStageOutput};
use crate::torch_runner::tests::model_test::common::{ClsOut, ModelTestId};

pub(crate) const SYNC_AB_CLC_ENDPOINT: &str = "ABClcSyncStage";

#[derive(Debug, Named, Clone, DummyData)]
pub(super) struct ABClcStageInput {
    id:SynKey,
    from_a:Arc<DevicesWrap<AStageInput>>,
    from_b:Arc<DevicesWrap<BStageInput>>,
}

impl ABClcStageInput {
    pub fn new(from_a: Arc<DevicesWrap<AStageInput>>, from_b: Arc<DevicesWrap<BStageInput>>) -> Self {
        Self {
            id: from_a.id().clone() + from_b.id(),
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

#[derive(Debug, Named, Clone, DummyData, Serialize, Deserialize)]
pub(super) struct ABClcStageOutput {
    id:SynKey,
    pub(super) value:f32
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
pub(super) struct ABClcStage {
    id:MonoKey,
    a_stage:Arc<AStage>,
    b_stage:Arc<BStage>,
    ab_clc_sync:Arc<ABClcSyncStageRunner>
}

impl Stage for ABClcStage {
    type Ctx = RunCtxImpl;
    type In<'a> = ABClcStageInput;
    type Out = ABClcStageOutput;
}

impl ABClcStage {
    pub fn new() -> Self {
        Self {
            id: mono_key!(AB_CLC_STAGE_ID, "ABClcStage"),
            a_stage: Arc::new(AStage::new()),
            b_stage: Arc::new(BStage::new()),
            ab_clc_sync: Arc::new(ABClcSyncStageRunner::new())
        }
    }

    pub fn num_calls(&self) -> usize {
        self.ab_clc_sync.num_calls()
    }
}

impl OutId for ABClcStage {
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
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

    fn with_data_ref<'a, R: Send + 'a, F, P>(&'a self, _to_device_idx: usize, ctx: P, _trace_ctx: Option<&'static str>, do_run: F) -> impl std::future::Future<Output = Result<R, ThoError>> + Send + 'a
    where
            for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
            P: RunCtx + 'a,
    {
        async move {
            with_data!(ctx, self.a_out, self.b_out; |ctx, a, b|{
                do_run(&ABClcSyncStageInput::new(a, b), &ctx)
            })
        }
    }
}

//required only when SplitAdjuster is used
impl WithSelectDataRef<ABClcSyncStageInputWrapperSelector> for ABClcSyncStageInputWrapper {
    fn with_selected_data_ref<'a, 'o, R: Send + 'a, F, P>(&'a self, to_device_idx: usize, ctx: P, selector: ABClcSyncStageInputWrapperSelector, trace_ctx: Option<&'static str>, do_run: F) -> impl Future<Output=Result<R, ThoError>> + Send + 'a
    where
            for<'f> F: FnOnce(&Self::O<'f>, &P) -> Result<R, ThoError> + Send + 'a,
            P: RunCtx + 'a
    {
        async move {
            track_await(self.with_data_ref(to_device_idx, ctx, None, do_run)).await
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
struct ABClcSyncStageInputWrapperSelector {}


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

        Ok(Arc::new(DevicesWrap::wrap( result)))

    }
}


#[derive(Debug, Named)]
pub(super) struct ABClcSyncStage {
    id:MonoKey,
    num_calls:AtomicUsize,
}

impl ABClcSyncStage {
    pub fn new() -> Self {
        Self {
            id: mono_key!(AB_SYNC_CLC_STAGE_ID, "ABClcSyncStage"),
            num_calls: Default::default(),
        }
    }

    pub fn num_calls(&self) -> usize {
        self.num_calls.load(Acquire)
    }

    fn do_run_inner<'f>(&self, input: &ABClcSyncStageInput<'f>, _ctx: &RunCtxImpl) -> Result<ABClcStageOutput, ThoError> {
        self.do_run_values(input.id(), &input.from_a.value, &input.from_b.value)
    }

    fn do_run_values(&self, input_id: &SynKey, from_a: &ThoTensor<AStageOutputDims, ()>, from_b: &ThoTensor<BStageOutputDims, ()>) -> Result<ABClcStageOutput, ThoError> {
        let call_nr = self.num_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        debug!("ABClcSyncStage::do_run: from_a={from_a:?} from_b={from_b:?} call_nr={call_nr}");

        let value = (from_a.as_ref()+ from_b.as_ref()).mean_dim(0, false, None).double_value(&[]) as f32;

        debug!("ABClcSyncStage::do_run: value={value}");

        Ok(ABClcStageOutput{
            id: self.out_id(input_id),
            value,
        })
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
pub(super) struct ABClcSyncStageInput<'a> {
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
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        self.do_run_inner(input, _ctx)
    }

}

#[derive(Debug)]
enum ABClcSyncStageRunner {
    Local(ABClcSyncStage),
    Proxy(ProxyABClcSyncStage),
}

impl ABClcSyncStageRunner {
    fn new() -> Self {
        if GlobalCtx::global().is_ipc() {
            let proxy = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(ProxyABClcSyncStage::new())
            }).expect("ABClcSyncStageRunner::new: install IPC proxy sync stage");
            Self::Proxy(proxy)
        } else {
            Self::Local(ABClcSyncStage::new())
        }
    }

    fn num_calls(&self) -> usize {
        match self {
            Self::Local(stage) => stage.num_calls(),
            Self::Proxy(stage) => stage.num_calls(),
        }
    }
}

impl Adjustable for ABClcSyncStageRunner {}

impl OutId for ABClcSyncStageRunner {
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        match self {
            Self::Local(stage) => stage.out_id(input_id),
            Self::Proxy(stage) => stage.out_id(input_id),
        }
    }
}

impl Stage for ABClcSyncStageRunner {
    type Ctx = RunCtxImpl;
    type In<'a> = ABClcSyncStageInput<'a>;
    type Out = ABClcStageOutput;
}

impl SyncStage for ABClcSyncStageRunner {
    fn do_run<'f>(self: Arc<Self>, input: &Self::In<'f>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        match self.as_ref() {
            Self::Local(stage) => stage.do_run_inner(input, ctx),
            Self::Proxy(stage) => stage.do_run_inner(input, ctx),
        }
    }
}

#[derive(Debug)]
struct ProxyABClcSyncStage {
    id: MonoKey,
    num_calls: AtomicUsize,
}

impl ProxyABClcSyncStage {
    async fn new() -> Result<Self, ThoError> {
        GlobalCtx::global()
            .cuda_actor(1)
            .install_endpoint(encode_endpoint_payload(SYNC_AB_CLC_ENDPOINT, ""))
            .await?;

        Ok(Self {
            id: mono_key!(AB_SYNC_CLC_STAGE_ID, "ABClcSyncStage"),
            num_calls: Default::default(),
        })
    }

    fn num_calls(&self) -> usize {
        self.num_calls.load(Acquire)
    }

    fn do_run_inner<'f>(&self, input: &ABClcSyncStageInput<'f>, ctx: &RunCtxImpl) -> Result<ABClcStageOutput, ThoError> {
        self.num_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(async move {
            let response = GlobalCtx::global()
                .cuda_actor(ctx.tho_device_idx())
                .invoke_endpoint(encode_endpoint_payload(SYNC_AB_CLC_ENDPOINT, &serde_json::to_string(&ABClcSyncStageIpcInputRef::new(input))?))
                .await?;

            serde_json::from_str::<ABClcStageOutput>(&response).map_err(ThoError::from)
        }))
    }
}

impl OutId for ProxyABClcSyncStage {
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

impl Stage for ProxyABClcSyncStage {
    type Ctx = RunCtxImpl;
    type In<'a> = ABClcSyncStageInput<'a>;
    type Out = ABClcStageOutput;
}

#[derive(Serialize)]
struct ABClcSyncStageIpcInputRef<'a> {
    id: &'a SynKey,
    from_a: &'a ThoTensor<AStageOutputDims, ()>,
    from_b: &'a ThoTensor<BStageOutputDims, ()>,
}

impl<'a> ABClcSyncStageIpcInputRef<'a> {
    fn new(input: &'a ABClcSyncStageInput<'a>) -> Self {
        Self {
            id: input.id(),
            from_a: &input.from_a.value,
            from_b: &input.from_b.value,
        }
    }
}

#[derive(Deserialize)]
struct ABClcSyncStageIpcInput {
    id: SynKey,
    from_a: ThoTensor<AStageOutputDims, ()>,
    from_b: ThoTensor<BStageOutputDims, ()>,
}

#[derive(Debug)]
struct ABClcSyncStageJsonWrapper {
    stage: ABClcSyncStage,
}

impl SyncStageJsonWrapper for ABClcSyncStageJsonWrapper {
    fn do_run(&self, input: String, all_cuda_tensors: &Arc<tokio::sync::Mutex<std::collections::HashMap<u64, tch::Tensor>>>) -> Result<String, ThoError> {
        let input = serde_json::from_str::<ABClcSyncStageIpcInput>(&input)?;
        let from_a = ThoTensor::<AStageOutputDims, ()>::decode_for_worker(serde_json::to_string(&input.from_a)?, all_cuda_tensors)?;
        let from_b = ThoTensor::<BStageOutputDims, ()>::decode_for_worker(serde_json::to_string(&input.from_b)?, all_cuda_tensors)?;
        let out = self.stage.do_run_values(&input.id, &from_a, &from_b)?;

        serde_json::to_string(&out).map_err(ThoError::from)
    }
}

pub(crate) fn sync_ab_clc_endpoint_init(_init_msg: String) -> Result<Box<dyn SyncStageJsonWrapper>, ThoError> {
    Ok(Box::new(ABClcSyncStageJsonWrapper {
        stage: ABClcSyncStage::new(),
    }))
}
