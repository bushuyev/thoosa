use std::sync::Arc;
use async_trait::async_trait;
use cached_derive::Cached;
use thoosa_dummy_data_derive::DummyData;
use thoosa_named_derive::Named;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::copy_mode::CpuOrGpu;
use thoosa_shared::ThoError;
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::{Data, MonoMetrics, WithId};
use thoosa_with_data::with_data;
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::common::runnable::{AsyncStage, Stage, SyncStage};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::tests::composite_input::dims::AStageOutputDims;
use crate::torch_runner::tests::composite_input::loader_a::AStageOutput;
use crate as thoosa_common;
use crate::runner::common::RunCtx;

#[derive(Debug, Named)]
struct ABClcStageInput {
    id:SynKey,
    from_a:AStageOutput,
    from_b:AStageOutput,
}

impl ABClcStageInput {
    pub fn new(from_a: AStageOutput, from_b: AStageOutput) -> Self {
        Self {
            id: from_a.id() + from_b.id(),
            from_a,
            from_b
        }
    }
}
impl Data for ABClcStageInput {
    type Metrics = MonoMetrics<0,1>;
    type M = CpuOrGpu<Self>;

    fn size(&self) -> Self::Metrics {
        &<ThoTensor<AStageOutputDims, ()> as Data>::size(&self.from_a.value) + &<ThoTensor<AStageOutputDims, ()> as Data>::size(&self.from_b.value)
    }

    fn device(&self) -> usize {
        self.from_a.value.device()
    }

    fn mem_size(&self) -> u64 {
        self.from_a.value.mem_size()
    }
}

impl WithId for ABClcStageInput {
    type ID = SynKey;

    fn id(&self) -> &Self::ID {
        &self.id
    }
}

#[async_trait]
impl thoosa_shared::CopyToDevice for ABClcStageInput {
    async fn copy_to_device(&self, dst_device: usize) -> Result<Self, ThoError> {
        Ok(Self {
            id: self.id.clone(),
            from_a: track_await(self.from_a.copy_to_device(dst_device)).await?,
            from_b: track_await(self.from_b .copy_to_device(dst_device)).await?,
        })
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
    ab_clc_sync:ABClcSyncStage
}

impl Stage for ABClcStage {
    type Ctx = RunCtxImpl;
    type In<'a> = ABClcStageInput;
    type Out = ABClcStageOutput;
}

impl ABClcStage {
    pub fn new() -> Self {
        Self { ab_clc_sync: ABClcSyncStage::new() }
    }
}

impl AsyncStage for ABClcStage {
    async fn run(self: Arc<Self>, input: Arc<DevicesWrap<Self::In<'static>>>, ctx: Self::Ctx) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {

       with_data!(ctx, input; |ctx, input:&ABClcStageInput| {

           let r = ABClcStageOutput {
               id: input.id.clone(),
               value: 0.0
           };

           Ok(Arc::new(DevicesWrap::wrap(r)))
       })
    }
}


#[derive(Debug, Named)]
struct ABClcSyncStage {

}

impl ABClcSyncStage {
    pub fn new() -> Self {
        Self {}
    }
}

impl Stage for ABClcSyncStage {
    type Ctx = RunCtxImpl;
    type In<'a> = ABClcStageInput;
    type Out = ABClcStageOutput;
}

impl SyncStage for ABClcSyncStage {
    fn do_run<'f>(self: Arc<Self>, _input: &Self::In<'f>, _ctx: &Self::Ctx) -> Result<Self::Out, ThoError> {
        todo!()
    }
}