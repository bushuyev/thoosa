use std::ops::Deref;
use std::sync::Arc;

use tch::Kind;
use thoosa_named_derive::Named;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{AsyncStage, CachableIdTr, OutId, RunCtx, Stage, WithId};

use crate as thoosa_common;
use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep};
use crate::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use crate::runner::common::cachable::{Cached, DevicesWrap};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{Adjusters, NoStageAdjuster, OOMHandler, WithInputContext, WithRunCtxContext, input_adjuster, run_ctx_adjuster};
use crate::runner::run_ctx::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};

use super::batch_builder::BatchBuilder;
use super::classifier::Classifier;
use super::common::{ABMetrics, ABRange, ClsOut, FileListInput, ModelTestId};
use super::loader::ABLoader;

#[derive(Debug, cached_derive::Cached, Named)]
#[cache_name = "ClsOutId"]
pub(crate) struct JoinAndClassifyStage {
    batch_builder: Arc<BatchBuilder>,
    clf: Arc<Classifier>,
}

impl JoinAndClassifyStage {
    pub(crate) fn new(loader: Arc<ABLoader>, clf: Arc<Classifier>, batch_size: usize) -> Self {
        Self {
            batch_builder: Arc::new(BatchBuilder::new(loader, batch_size)),
            clf,
        }
    }
}

impl OutId for JoinAndClassifyStage {
    fn out_id(&self, input_id: &<Self::In<'static> as WithId>::ID) -> <Self::Out as WithId>::ID {
        self.clf.out_id(input_id)
    }
}

impl Stage for JoinAndClassifyStage {
    type In<'a> = FileListInput;
    type Out = Arc<DevicesWrap<ThoTensor<ClsOut, ModelTestId>>>;
    type Ctx = RunCtxImpl;
}

impl AsyncStage for JoinAndClassifyStage {
    // type RunInput = Arc<Self::In<'static>>;
    // type RunOutput = Arc<DevicesWrap<Self::Out>>;

    // fn run_input_id(input: &Self::In<'static>) -> &<<Self as Stage>::In<'static> as WithId>::ID {
    //     WithId::id(input)
    // }
    async fn run(self: Arc<Self>, input: Self::In<'static>, mut ctx: Self::Ctx) -> Result<Self::Out, ThoError> {
        let device_idx = ctx.tho_device_idx();
        let batched = Arc::clone(&self.batch_builder).run_cached(input.clone(), ctx.branch(false)).await?;

        let out_id = self.clf.out_id(batched.id());
        let _out_key = Arc::new(out_id.clone()) as Arc<dyn CachableIdTr>;
        let size = *batched.size();
        let min_split_len = ABMetrics { dim_0: 32, dim_1: 0 };

        let output_builder = SingleDimTensorOutputBuilder::<ModelTestId, ClsOut>::new(out_id, 1, &[1, 1], if GlobalCtx::global().is_ipc() { (Kind::Float, tch::Device::Cpu) } else { (Kind::Float, ThoDevice::from(device_idx).deref().clone()) }, BuildAs::Sum)?;
        let output_builder = if GlobalCtx::global().is_ipc() { output_builder.copy_partial_outputs_to_device(0) } else { output_builder };

        let adjusters: Vec<Adjusters<_, NoStageAdjuster, _>> = vec![
            Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(*MaxParallelStep::SLOW))),
            Adjusters::Input(input_adjuster(SplitsAdjuster::<ABRange, ABMetrics>::new(min_split_len))),
        ];
        let result = OOMHandler::new(adjusters, None)
            .run(&self.clf, batched, ctx, WithRunCtxContext::with_rest(MaxParallelAdjustContextImpl::new(), WithInputContext::new(SplitsAdjusterContextImpl::new(size, output_builder))))
            .await?;

        Ok(Arc::new(DevicesWrap::wrap(result)))
    }
}
