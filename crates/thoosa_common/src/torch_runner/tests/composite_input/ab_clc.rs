use crate as thoosa_common;
use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep};
use crate::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use crate::runner::common::cachable::{BorrowRef, Cached, DevicesWrap, SelectRef, ThoDataRef};
use crate::runner::oom_handler::{NoContexts, NoInputAdjuster, NoRunCtxAdjuster, NoStageAdjuster, OOMHandler};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::ipc::codec::{IpcCodec, JsonIpcCodec};
use crate::ipc::cuda_wokrer::{AllTensors, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::json_ser::SyncStageIpcWrapper;
use crate::ipc::wraps::WorkerWrapper;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use crate::torch_runner::tests::composite_input::dims::{ABDims, AStageOutputDims, BStageOutputDims};
use crate::torch_runner::tests::composite_input::loader_a::{AStage, AStageInput, AStageOutput};
use crate::torch_runner::tests::composite_input::loader_b::{BStage, BStageInput, BStageOutput};
use crate::torch_runner::tests::composite_input::{AB_CLC_STAGE_ID, AB_SYNC_CLC_STAGE_ID};
use async_trait::async_trait;
use cached_derive::Cached;
use serde::{Deserialize, Serialize, Serializer};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::Acquire;
use futures_util::future::BoxFuture;
use parking_lot::{Mutex, RwLock};
use tch::{Kind, Tensor};
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::{MonoKey, SynKey};
use thoosa_shared::copy_mode::{CpuOnly, CpuOrGpu};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Adjustable, AsyncStage, Data, MonoMetrics, OutId, RangeSelector, RunCtx, Stage, SyncStage, ThoSelector, WithId};
use thoosa_shared::{ThoError, mono_key};
use tracing::debug;
use crate::ipc::worker_msg::{WorkerInput, WorkerOutput};
use crate::ipc::proxy::{DynSyncStageEnum, ProxyStage};
use crate::ws::Workspace;

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

#[typetag::serde]
impl WorkerOutput for ABClcStageOutput {}

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
        let result = OOMHandler::<NoInputAdjuster, NoStageAdjuster, NoRunCtxAdjuster, ABClcSyncStageInputWrapperSelector>::new(
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


#[derive(Debug, Serialize)]
struct ABClcSyncStageIpcInputRef<'a> {
    id: &'a SynKey,
    from_a: &'a AStageOutput,
    from_b: &'a BStageOutput,
}

impl<'a> ABClcSyncStageIpcInputRef<'a> {
    fn new(input: &'a ABClcSyncStageInput<'_>) -> Self {
        Self {
            id: input.id(),
            // Serialize the complete nested values so postcard matches ThoDataRef's owned deserializer layout.
            from_a: input.from_a.as_ref(),
            from_b: input.from_b.as_ref(),
        }
    }
}

#[derive(Deserialize)]
struct ABClcSyncStageIpcInput {
    id: SynKey,
    from_a: AStageOutput,
    from_b: BStageOutput,
}

#[derive(Debug, Named)]
pub struct ABClcSyncStage {
    id:MonoKey,
    num_calls:AtomicUsize,
}

#[derive(Debug, Named)]
pub struct ProxyABClcSyncStage {
    id: MonoKey,
    num_calls: AtomicUsize,
    stage: Arc<DynSyncStageEnum<ABClcSyncStage>>,
}

impl ProxyABClcSyncStage {
    pub async fn new(tho_device_idx: usize) -> Result<Self, ThoError> {
        Ok(Self {
            id: mono_key!(AB_SYNC_CLC_STAGE_ID, "ABClcSyncStage"),
            num_calls: AtomicUsize::default(),
            stage: Arc::new(DynSyncStageEnum::Proxy {
                stage: Arc::new(ProxyStage::proxy(SYNC_AB_CLC_ENDPOINT, 0, tho_device_idx, Box::new(())).await?),
                init: ()
            }),
        })
    }

    pub fn num_calls(&self) -> usize {
        self.num_calls.load(Acquire)
    }
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

#[derive(Debug, Deserialize)]
pub struct ABClcSyncStageInput<'a> {
    id:SynKey,
    from_a:ThoDataRef<'a, AStageOutput>,
    from_b:ThoDataRef<'a, BStageOutput>,
}

#[typetag::serde]
impl WorkerInput for ABClcSyncStageInput<'static> {}

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

}

impl Serialize for ABClcSyncStageInput<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        ABClcSyncStageIpcInputRef::new(self).serialize(serializer)
    }
}

impl<'a> WorkerStageInput for ABClcSyncStageInput<'a> {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        // typetag can only erase 'static values; both ThoDataRef fields are owned in the IPC copy.
        Ok(Box::new(ABClcSyncStageInput::<'static> {
            id: self.id.clone(),
            from_a: ThoDataRef::Owned(
                AStageOutput{
                    id: self.from_a.as_ref().id.clone(),
                    value: self.from_a.as_ref().value.clone_ref()?,
                }
            ),
            from_b: ThoDataRef::Owned(
                BStageOutput{
                    id: self.from_b.as_ref().id.clone(),
                    value: self.from_b.as_ref().value.clone_ref()?,
                }
            ),
        }))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
        // Rebuild owned stage inputs so no worker tensor reference escapes this conversion.
        Ok(Self {
            id: self.id.clone(),
            from_a: ThoDataRef::Owned(self.from_a.as_ref().ref_to_tensor(all_cuda_tensors, remove)?),
            from_b: ThoDataRef::Owned(self.from_b.as_ref().ref_to_tensor(all_cuda_tensors, remove)?),
        })
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

impl WorkerStageOutput for ABClcStageOutput {
    fn encode_from_worker(self, _all_cuda_tensors: &AllTensors, _tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(self)
    }
}

impl Adjustable for ProxyABClcSyncStage {}

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

impl SyncStage for ProxyABClcSyncStage {
    fn do_run<'a>(self: Arc<Self>, input: &Self::In<'a>, ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        self.num_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Arc::clone(&self.stage).do_run(input, ctx)
    }
}

pub fn sync_ab_clc_endpoint_init<'a>(init: Box<dyn WorkerInput>, ws: Arc<RwLock<Box<dyn Workspace>>>, all_cuda_tensors: &'a AllTensors, tho_device:&'a ThoDevice) -> BoxFuture<'a, Result<Box<dyn SyncStageIpcWrapper>, ThoError>> {
    use futures_util::FutureExt;
    async move {
        // The shared wrapper decodes both borrowed tensor inputs into worker-owned values.
        Ok(Box::new(WorkerWrapper::new(Arc::new(ABClcSyncStage::new()), PhantomData::<JsonIpcCodec>)) as Box<dyn SyncStageIpcWrapper>)
    }.boxed()
}
