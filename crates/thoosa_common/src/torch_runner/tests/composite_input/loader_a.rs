use crate as thoosa_common;
use crate::ipc::cuda_wokrer::{AllTensors, WorkerStageInput};
use crate::ipc::worker_msg::WorkerInput;
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::run_ctx::RunCtxImpl;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::tests::composite_input::A_STAGE_ID;
use crate::torch_runner::tests::composite_input::dims::AStageOutputDims;
use async_trait::async_trait;
use cached_derive::Cached;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tch::kind::FLOAT_CPU;
use tch::{IndexOp, Tensor};
use thoosa_common::dims::LocalSelect;
use thoosa_cpu_data_derive::CpuData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::{ElvVal, MonoKey, SynKey};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{AsyncStage, Data, OutId, RangeSelector, Select, Stage, WithId};
use thoosa_shared::{ThoError, mono_key};

#[derive(Debug, Clone, CpuData, Hash, Eq, PartialEq, Named)]
#[cpu_data(shallow)]
pub(super) struct AStageInput {
    id: MonoKey,
    db_id: u32,
    token: String,
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


#[derive(Debug, Named, Serialize, Deserialize)]
pub(super) struct AStageOutput {
    pub id: SynKey,
    pub(super) value:ThoTensor<AStageOutputDims, ()>,
}

impl AStageOutput {
    pub(super) fn from_value(id: SynKey, value: ThoTensor<AStageOutputDims, ()>) -> Self {
        // Worker decoding already owns the tensor, so avoid rebuilding it through a shallow clone.
        Self { id, value }
    }

    pub(super) fn new(id: SynKey, tensor: Tensor) -> Result<Arc<DevicesWrap<Self>>, ThoError> {
        Ok(Arc::new(DevicesWrap::wrap(
            AStageOutput{
                id,
                value: ThoTensor::new((), tensor)?,
            }
        )))
    }
}

#[typetag::serde]
impl WorkerInput for AStageOutput {}

impl WorkerStageInput for AStageOutput {
    fn clone_refs(&self) -> Result<Box<dyn WorkerInput>, ThoError> {
        Ok(Box::new(AStageOutput{
            id: self.id.clone(),
            value: self.value.clone_ref()?,
        }))
    }

    fn ref_to_tensor(&self, all_cuda_tensors: &AllTensors, remove: bool) -> Result<Self, ThoError> {
      Ok(AStageOutput{
          id: self.id.clone(),
          value: self.value.ref_to_tensor(all_cuda_tensors, remove)?,
      })
    }

}

impl Select<RangeSelector> for AStageOutput {
    type Output = AStageOutput;

    fn subview(&self, selector: RangeSelector) -> Result<Self::Output, ThoError> {
        Ok(AStageOutput {
            id: self.id.clone(),
            // value: ThoTensor::new((), self.value.as_ref().i(selector.start[0] as i64..selector.end[0] as i64).copy().into())?
            value: self.value.subview(selector)?,
        })
    }
}

impl LocalSelect<(), RangeSelector> for ThoTensor<AStageOutputDims, ()> {
    type Output = ThoTensor<AStageOutputDims, ()>;

    fn local_subview(id: &(), t: &Tensor, selector: RangeSelector) -> Result<Self::Output, ThoError> {
        ThoTensor::new(
            (),
            t.i(selector.start[0]..selector.end[0]).copy().into()
        )
    }
}



impl WithId for AStageOutput {
    type ID = SynKey;

    fn id(&self) -> &SynKey {
        &self.id
    }
}

impl Data for AStageOutput {
    type Metrics = <ThoTensor<AStageOutputDims, ()> as Data>::Metrics;


    fn size(&self) -> Self::Metrics {
        <ThoTensor<AStageOutputDims, ()> as Data>::size(&self.value)
    }

    fn tho_device_idx(&self) -> usize {
        self.value.tho_device_idx()
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
            value: track_await!(self.value.copy_to_device(dst_device)).await?,
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
    type In<'a> = Arc<AStageInput>;
    type Out = Arc<DevicesWrap<AStageOutput>>;
}

impl AsyncStage for AStage  {
    // type RunInput = Arc<Self::In<'static>>;
    // type RunOutput = Arc<DevicesWrap<Self::Out>>;

    // fn run_input_id(input: &Self::In<'static>) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
    //     WithId::id(input)
    // }
    async fn run(self: Arc<Self>, input: Self::In<'static>, _ctx: Self::Ctx) -> Result<Self::Out, ThoError> {


        let tensor = match input.token.as_str() {
            "a100" => Tensor::from_slice(&[100.1]),
            _ => Tensor::rand(&[1], FLOAT_CPU)
        };

        let id = self.out_id(input.id());


        AStageOutput::new(id, tensor)

    }
}

impl OutId for AStage {
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        self.id.clone() + input_id
    }
}
