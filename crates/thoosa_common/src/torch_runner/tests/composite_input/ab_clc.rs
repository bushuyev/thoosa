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
use thoosa_proxy_derive::Proxy;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, Data, MonoMetrics, RangeSelector, ThoSelector, WithId};
use crate::runner::common::cachable::{BorrowRef, SelectRef, Cached, DevicesWrap, ThoDataRef};
use crate::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::tests::composite_input::dims::{ABDims, AStageOutputDims, BStageOutputDims};
use crate::torch_runner::tests::composite_input::loader_a::{AStage, AStageInput, AStageOutput};
use crate as thoosa_common;
use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MAX_PARALLEL};
use crate::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoTaskAdjuster, OOMHandler};
use crate::tho_device::ThoDevice;
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use crate::torch_runner::tests::composite_input::{AB_CLC_STAGE_ID, AB_SYNC_CLC_STAGE_ID};
use crate::torch_runner::tests::composite_input::loader_b::{BStage, BStageInput, BStageOutput};

pub(crate) const SYNC_AB_CLC_ENDPOINT: &str = "ABClcSyncStage";

#[derive(Debug, Named, Clone)]
pub struct ABClcStageInput {
    id:SynKey,
    from_a:Arc<AStageInput>,
    from_b:Arc<BStageInput>,
}

impl ABClcStageInput {
    pub(super) fn new(from_a: Arc<AStageInput>, from_b: Arc<BStageInput>) -> Self {
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
pub struct ABClcStageOutput {
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
pub struct ABClcStage<S = ABClcSyncStage>
where
    S: SyncStage + Adjustable + OutId + Send + Sync + 'static,
    S::Ctx: Adjustable + Send + Sync + 'static,
    for<'f> S: Stage<Ctx = RunCtxImpl, In<'f> = ABClcSyncStageInput<'f>, Out = ABClcStageOutput>,
{
    id:MonoKey,
    a_stage:Arc<AStage>,
    b_stage:Arc<BStage>,
    pub ab_clc_sync:Arc<S>
}


impl ABClcStage {
    pub fn new() -> Self {
        Self::new_with_sync(Arc::new(ABClcSyncStage::new()))
    }

    pub fn num_calls(&self) -> usize {
        self.ab_clc_sync.num_calls()
    }
}

impl<S> ABClcStage<S>
where
    S: SyncStage + Adjustable + OutId + Send + Sync + 'static,
    S::Ctx: Adjustable + Send + Sync + 'static,
    for<'f> S: Stage<Ctx = RunCtxImpl, In<'f> = ABClcSyncStageInput<'f>, Out = ABClcStageOutput>,
{
    pub(crate) fn new_with_sync(ab_clc_sync: Arc<S>) -> Self {
        Self {
            id: mono_key!(AB_CLC_STAGE_ID, "ABClcStage"),
            a_stage: Arc::new(AStage::new()),
            b_stage: Arc::new(BStage::new()),
            ab_clc_sync,
        }
    }
}

impl<S> OutId for ABClcStage<S>
where
    S: SyncStage + Adjustable + OutId + Send + Sync + 'static,
    S::Ctx: Adjustable + Send + Sync + 'static,
    for<'f> S: Stage<Ctx = RunCtxImpl, In<'f> = ABClcSyncStageInput<'f>, Out = ABClcStageOutput>,
{
    fn out_id<'a>(&self, input_id: &<<Self as Stage>::In<'a> as WithId>::ID) -> <Self::Out as WithId>::ID {
        &self.id + input_id
    }
}

#[derive(Debug)]
pub struct ABClcSyncStageInputWrapper {
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


impl BorrowRef for ABClcSyncStageInputWrapper {
    type Out<'a> = ABClcSyncStageInput<'a> where Self: 'a;

    fn borrow_to<'a>(&'a self, to_device_idx: usize) -> impl Future<Output = Result<Self::Out<'a>, ThoError>> + Send
    where
        Self: 'a,
    {
        async move {
            let from_a = track_await(self.a_out.load_from_any_device_(to_device_idx)).await?;
            let from_b = track_await(self.b_out.load_from_any_device_(to_device_idx)).await?;
            Ok(ABClcSyncStageInput::from_data_refs(ThoDataRef::ReadLocked(from_a), ThoDataRef::ReadLocked(from_b)))
        }
    }
}

impl SelectRef<ABClcSyncStageInputWrapperSelector> for ABClcSyncStageInputWrapper {
    fn borrow_to<'a>(&'a self, to_device_idx: usize, selector: Option<ABClcSyncStageInputWrapperSelector>) -> impl Future<Output = Result<<Self as BorrowRef>::Out<'a>, ThoError>> + Send
    where
        ABClcSyncStageInputWrapperSelector: 'a,
        Self: 'a,
    {
        async move {
            match selector{
                None => {
                    BorrowRef::borrow_to(self, to_device_idx).await
                }
                Some(selector) => {
                    let from_a = SelectRef::borrow_to(&self.a_out, to_device_idx, Some(selector.a_select)).await?;
                    let from_b =  SelectRef::borrow_to(&self.b_out, to_device_idx, Some(selector.b_select)).await?;
                    Ok(ABClcSyncStageInput::from_data_refs(from_a, from_b))
                }
            }
        }
    }
}


impl WithId for ABClcSyncStageInputWrapper {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ABClcSyncStageInputWrapperSelector {
    a_select:RangeSelector,
    b_select:RangeSelector
}

impl ABClcSyncStageInputWrapperSelector {
    pub fn new(a_end:i64, b_end:i64) -> Self {
        Self {
            a_select: RangeSelector([0i64]..[a_end]),
            b_select: RangeSelector([0i64]..[b_end])
        }
    }
}

#[typetag::serde]
impl ThoSelector for ABClcSyncStageInputWrapperSelector {fn clone_box(&self) -> Box<dyn ThoSelector> {
        Box::new(self.clone())
    }

}
impl Adjustable for ABClcSyncStageInputWrapperSelector {}


impl<S> Stage for ABClcStage<S>
where
    S: SyncStage + Adjustable + OutId + Send + Sync + 'static,
    S::Ctx: Adjustable + Send + Sync + 'static,
    for<'f> S: Stage<Ctx = RunCtxImpl, In<'f> = ABClcSyncStageInput<'f>, Out = ABClcStageOutput>,
{
    type Ctx = RunCtxImpl;
    type In<'a> = ABClcStageInput;
    type Out = Arc<DevicesWrap<ABClcStageOutput>>;
}


impl<S> AsyncStage for ABClcStage<S>
where
    S: SyncStage + Adjustable + OutId + Send + Sync + 'static,
    S::Ctx: Adjustable + Send + Sync + 'static,
    for<'f> S: Stage<Ctx = RunCtxImpl, In<'f> = ABClcSyncStageInput<'f>, Out = ABClcStageOutput>,
{


    async fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        let a_out = Arc::clone(&self.a_stage).run_cached(Arc::clone(&input.from_a), ctx.branch(false)).await?;
        let b_out = Arc::clone(&self.b_stage).run_cached(Arc::clone(&input.from_b), ctx.branch(false)).await?;

        let out_id = self.ab_clc_sync.out_id(input.id());


        let input = Arc::new(ABClcSyncStageInputWrapper::new(a_out, b_out));
        let result = OOMHandler::<NoInputAdjuster, NoTaskAdjuster, NoRunCtxAdjuster, ABClcSyncStageInputWrapperSelector>::new(
            vec![],
            None,
        )
            .run(
                &self.ab_clc_sync,
                input,
                ctx,
                NoContexts,
            )
            .await?;

        Ok(Arc::new(DevicesWrap::wrap( result)))

    }
}


#[derive(Serialize)]
struct ABClcSyncStageIpcInputRef<'a> {
    id: &'a SynKey,
    from_a: &'a ThoTensor<AStageOutputDims, ()>,
    from_b: &'a ThoTensor<BStageOutputDims, ()>,
}

impl<'a> ABClcSyncStageIpcInputRef<'a> {
    fn new(input: &'a ABClcSyncStageInput<'_>) -> Self {
        Self {
            id: input.id(),
            from_a: input.from_a_value(),
            from_b: input.from_b_value(),
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
struct ABClcSyncStageIpcWrapper {
    stage: std::sync::Arc<ABClcSyncStage>,
}
impl crate::ipc::json_ser::SyncStageIpcWrapper for ABClcSyncStageIpcWrapper {
    fn do_run(&self, input: &[u8], all_cuda_tensors: &std::sync::Arc<parking_lot::Mutex<std::collections::HashMap<u64, tch::Tensor>>> ) -> Result<Vec<u8>, thoosa_shared::ThoError> {
        type __ProxyIpcCodec = crate::ipc::codec::JsonIpcCodec;
        let input = <__ProxyIpcCodec as crate::ipc::codec::IpcCodec>::decode::<ABClcSyncStageIpcInput>(input)?;
        let from_a = <ThoTensor<AStageOutputDims, ()> as crate::ipc::cuda_wokrer::WorkerStageInput>::decode_for_worker::<__ProxyIpcCodec>(&<__ProxyIpcCodec as crate::ipc::codec::IpcCodec>::encode(&input.from_a)?, all_cuda_tensors)?;
        let from_b = <ThoTensor<BStageOutputDims, ()> as crate::ipc::cuda_wokrer::WorkerStageInput>::decode_for_worker::<__ProxyIpcCodec>(&<__ProxyIpcCodec as crate::ipc::codec::IpcCodec>::encode(&input.from_b)?, all_cuda_tensors)?;
        let out = self.stage.do_run_values(&input.id, &from_a, &from_b)?;
        <__ProxyIpcCodec as crate::ipc::codec::IpcCodec>::encode(&out)
    }
}

#[derive(Debug, Named, Proxy)]
#[proxy(
    name = "ProxyABClcSyncStage",
    endpoint = "SYNC_AB_CLC_ENDPOINT",
    new_args = "tho_device_idx: usize",
    fields = "id: MonoKey = mono_key!(AB_SYNC_CLC_STAGE_ID, \"ABClcSyncStage\"), num_calls: AtomicUsize = Default::default()",
    out_id = "&self.id + input_id",
    pre_run = "self.num_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);",
    run_input = "ABClcSyncStageIpcInputRef::new(input)",
    counter = "num_calls",
    endpoint_init = "sync_ab_clc_endpoint_init",
    wrapper = "ABClcSyncStageIpcWrapper"
)]
pub struct ABClcSyncStage {
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

    pub(crate) fn do_run_inner<'f>(&self, input: &ABClcSyncStageInput<'f>, _ctx: &RunCtxImpl) -> Result<ABClcStageOutput, ThoError> {
        self.do_run_values(input.id(), &input.from_a.as_ref().value, &input.from_b.as_ref().value)
    }

    pub(crate) fn do_run_values(&self, input_id: &SynKey, from_a: &ThoTensor<AStageOutputDims, ()>, from_b: &ThoTensor<BStageOutputDims, ()>) -> Result<ABClcStageOutput, ThoError> {
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
pub struct ABClcSyncStageInput<'a> {
    id:SynKey,
    from_a:ThoDataRef<'a, AStageOutput>,
    from_b:ThoDataRef<'a, BStageOutput>,
}

impl <'a> ABClcSyncStageInput<'a> {
    pub(super) fn new(from_a: &'a AStageOutput, from_b: &'a BStageOutput) -> Self {

        Self {
            id: from_a.id() + from_b.id(),
            from_a: ThoDataRef::Borrowed(from_a),
            from_b: ThoDataRef::Borrowed(from_b)
        }
    }

    fn from_data_refs(from_a: ThoDataRef<'a, AStageOutput>, from_b: ThoDataRef<'a, BStageOutput>) -> Self {
        Self { id: from_a.as_ref().id() + from_b.as_ref().id(), from_a, from_b }
    }

    pub(crate) fn from_a_value(&self) -> &ThoTensor<AStageOutputDims, ()> {
        &self.from_a.as_ref().value
    }

    pub(crate) fn from_b_value(&self) -> &ThoTensor<BStageOutputDims, ()> {
        &self.from_b.as_ref().value
    }
}

impl Data for ABClcSyncStageInput<'_> {
    type Metrics = MonoMetrics<0, 1>;
    type M = CpuOnly<Self>;

    fn size(&self) -> Self::Metrics {
        &self.from_a.as_ref().size() + &self.from_b.as_ref().size()
    }

    fn device(&self) -> usize {
        assert_eq!(self.from_a.as_ref().device(), self.from_b.as_ref().device());
        self.from_a.as_ref().device()
    }

    fn mem_size(&self) -> u64 {
        self.from_a.as_ref().mem_size() + self.from_b.as_ref().mem_size()
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
