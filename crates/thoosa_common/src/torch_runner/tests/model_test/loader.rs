use std::sync::Arc;

use tch::{Device, Kind, Tensor};
use thoosa_named_derive::Named;
use thoosa_shared::traits::WithId;
use thoosa_shared::ThoError;

use crate as thoosa_common;
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::common::runnable::{AsyncStage, Stage};
use crate::runner::common::{OutId, RunCtx};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::runner::test_utils::parse_numeric_part;
use crate::tho_tensor::ThoTensor;

use super::common::{ABData, FileInput, ModelTestId};

#[derive(Debug, cached_derive::Cached, Named)]
#[cache_name = "ABDataId"]
pub(crate) struct ABLoader {
    input_len: i64,
    feat: i64,
    kind: Kind,
}

impl ABLoader {
    pub(crate) fn new(input_len: i64, feat: i64, kind: Kind) -> Self {
        Self {
            input_len,
            feat,
            kind,
        }
    }
}

pub(crate) fn new_file_input_cache(value: impl Into<FileInput>) -> Arc<FileInput> {
    Arc::new(value.into())
}

impl OutId for ABLoader {
    fn out_id(
        &self,
        input_id: &<Self::In<'static> as WithId>::ID,
    ) -> <Self::Out as WithId>::ID {
        ModelTestId(format!("features_{}", input_id.0).leak())
    }
}

impl Stage for ABLoader {
    type In<'a> = FileInput;
    type Out = ThoTensor<ABData, ModelTestId>;
    type Ctx = RunCtxImpl;
}

impl AsyncStage for ABLoader {
    type RunInput = Arc<Self::In<'static>>;
    type RunOutput = Arc<DevicesWrap<Self::Out>>;

    fn run_input_id(input: &Self::RunInput) -> &<<Self as Stage>::In<'static> as thoosa_shared::traits::WithId>::ID {
        thoosa_shared::traits::WithId::id(input)
    }
    async fn run(
        self: Arc<Self>,
        input: Arc<Self::In<'static>>,
        _ctx: Self::Ctx,
    ) -> Result<Arc<DevicesWrap<Self::Out>>, ThoError> {
        let idx = parse_numeric_part(WithId::id(input.as_ref()).0).unwrap_or(0.0) as i64;
        let base = if idx % 2 == 0 { 1.0 } else { -1.0 };
        let x = Tensor::ones([self.input_len, self.feat], (self.kind, Device::Cpu)) * base;
        let id = self.out_id(WithId::id(input.as_ref()));

        Ok(Arc::new(DevicesWrap::wrap(ThoTensor::new(id, x)?)))
    }
}
