use std::sync::Arc;
use async_trait::async_trait;
use tch::kind::FLOAT_CPU;
use tch::Tensor;
use cached_derive::Cached;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::{ElvVal, MonoKey, SynKey};
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Data, WithId};
use thoosa_with_data::with_data;
use crate::dims::MotifSlices;
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::common::{OutId, RunCtx};
use crate::runner::common::runnable::{AsyncStage, Stage};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::tests::composite_input::dims::AStageOutputDims;
use crate as thoosa_common;
use crate::torch_runner::tests::composite_input::A_STAGE_ID;

#[derive(Debug, Clone, DummyData, Hash, Eq, PartialEq, Named)]
#[dummy_data(clone)]
pub(super) struct AStageInput {
    id:MonoKey,
    db_id:u32,
    token:String,
}

impl AStageInput {
    pub(super) fn new<S:Into<String>>(db_id: u32, token:S) -> Self {
        let token = token.into();
        Self {
            id: mono_key!(db_id, &token),
            db_id,
            token
        }
    }
}

impl From<&AStageInput> for ElvVal {
    fn from(value: &AStageInput) -> Self {
        ElvVal::Str(value.token.clone())
    }
}

impl WithId for AStageInput {
    type ID = MonoKey;

    fn id(&self) -> &MonoKey {
        &self.id
    }
}


impl Input for AStageInput{}


#[derive(Debug, Named)]
pub(super) struct AStageOutput {
    id: SynKey,
    pub(super) value:ThoTensor<AStageOutputDims, ()>,
}

impl WithId for AStageOutput {
    type ID = SynKey;

    fn id(&self) -> &SynKey {
        &self.id
    }
}

impl Data for AStageOutput {
    type Metrics = <ThoTensor<AStageOutputDims, ()> as Data>::Metrics;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        <ThoTensor<AStageOutputDims, ()> as Data>::size(&self.value)
    }

    fn device(&self) -> usize {
        self.value.device()
    }

    fn mem_size(&self) -> u64 {
        self.value.mem_size()
    }
}

#[async_trait]
impl thoosa_shared::CopyToDevice for AStageOutput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            value: track_await(self.value.copy_to_device(dst_device)).await?,
        })
    }
}


impl Output for AStageOutput{}


#[derive(Debug, Named, Cached)]
pub(super) struct AStage {
    id: MonoKey,
}


impl AStage {
    pub(super) fn new() -> Self {
        Self { id: mono_key!(A_STAGE_ID, "AStage") }
    }
}

impl Stage for AStage  {
    type Ctx = RunCtxImpl;
    type In<'a> = AStageInput;
    type Out = AStageOutput;
}

impl AsyncStage for AStage  {
    type RunInput = Arc<Self::In<'static>>;
    type RunOutput = Arc<DevicesWrap<Self::Out>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        WithId::id(input)
    }
    async fn run(self: Arc<Self>, input: Arc<Self::In<'static>>, _ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {


        let tensor = match input.token.as_str() {
            "a100" => Tensor::from_slice(&[100.1]),
            _ => Tensor::rand(&[1], FLOAT_CPU)
        };

        let id = self.out_id(input.id());

        Ok(Arc::new(DevicesWrap::wrap(
            AStageOutput{
                id,
                value: ThoTensor::new((), tensor)?,
            }
        )))

    }
}

impl OutId for AStage {
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        self.id.clone() + input_id
    }
}
