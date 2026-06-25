use std::ops::Deref;
use std::sync::Arc;

use tch::Kind;
use thoosa_named_derive::Named;
use thoosa_shared::traits::{CachableIdTr, WithId};
use thoosa_shared::ThoError;

use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep, MAX_PARALLEL};
use crate::runner::adjusters::splits::{SplitsAdjuster, SplitsAdjusterContextImpl};
use crate as thoosa_common;
use crate::runner::common::cachable::{Cached, DevicesWrap};
use crate::runner::common::runnable::{AsyncStage, Stage};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::global_ctx::GlobalCtx;
use crate::runner::oom_handler::{Adjusters, Contexts, NoopRunnableContext, OOMHandler};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};

use super::batch_builder::BatchBuilder;
use super::classifier::Classifier;
use super::common::{ABData, ABMetrics, ABRange, ClsOut, FileListInput, ModelTestId};
use super::loader::ABLoader;

#[derive(Debug, cached_derive::Cached, Named)]
#[cache_name = "ClsOutId"]
pub(crate) struct JoinAndClassifyTask {
    batch_builder: Arc<BatchBuilder>,
    clf: Arc<Classifier>,
}

impl JoinAndClassifyTask {
    pub(crate) fn new(loader: Arc<ABLoader>, clf: Arc<Classifier>, batch_size: usize) -> Self {
        Self {
            batch_builder: Arc::new(BatchBuilder::new(loader, batch_size)),
            clf,
        }
    }
}

impl OutId for JoinAndClassifyTask {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        self.clf.out_id(input_id)
    }
}

impl Stage for JoinAndClassifyTask {
    type In<'a> = FileListInput;
    type Out = ThoTensor<ClsOut, ModelTestId>;
    type Ctx = RunCtxImpl;
}

impl AsyncStage for JoinAndClassifyTask {
    type RunInput = Arc<Self::In<'static>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        thoosa_shared::traits::WithId::id(input)
    }
    async fn run(
        self: Arc<Self>,
        input: Arc<Self::In<'static>>,
        mut ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let device_idx = ctx.tho_device_idx();
        let batched = Arc::clone(&self.batch_builder)
            .run_cached(Arc::clone(&input), ctx.branch(false))
            .await?;

        let out_id = self.clf.out_id(batched.id());
        let _out_key = Arc::new(out_id.clone()) as Arc<dyn CachableIdTr>;
        let size = *batched.size();
        let min_split_len = ABMetrics {
            dim_0: 32,
            dim_1: 0,
        };

        let output_builder = SingleDimTensorOutputBuilder::<ModelTestId, ClsOut>::new(
            out_id,
            1,
            &[1, 1],
            if GlobalCtx::global().is_ipc() {
                (Kind::Float, tch::Device::Cpu)
            } else {
                (Kind::Float, ThoDevice::from(device_idx).deref().clone())
            },
            BuildAs::Sum,
        )?;
        let output_builder = if GlobalCtx::global().is_ipc() {
            output_builder.copy_partial_outputs_to_device(0)
        } else {
            output_builder
        };

        let result = OOMHandler::new(
            vec![
                Adjusters::RunContext(Box::new(MaxParallelAdjuster::new(
                    MAX_PARALLEL,
                    *MaxParallelStep::TEN,
                ))),
                Adjusters::Input(Box::new(SplitsAdjuster::<ABRange, ABMetrics>::new(
                    min_split_len,
                ))),
            ],
            None,
        )
        .run(
            &self.clf,
            batched,
            ctx,
            Contexts::new(
                SplitsAdjusterContextImpl::new(size, output_builder),
                NoopRunnableContext {},
                MaxParallelAdjustContextImpl::new(),
            ),
        )
        .await?;

        Ok(Arc::new(DevicesWrap::wrap(result)))
    }
}
