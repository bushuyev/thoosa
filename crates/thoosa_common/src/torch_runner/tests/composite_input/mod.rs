use std::sync::Arc;
use log::debug;
use thoosa_shared::ThoError;
use thoosa_with_data::with_data;
use crate::runner::common::cachable::{Cached, DevicesWrap};
use crate::runner::common::RunCtx;
use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::torch_runner::tests::composite_input::ab_clc::{ABClcStage, ABClcStageInput, ABClcStageOutput, ABClcSyncStage};
use crate::torch_runner::tests::composite_input::loader_a::AStageInput;
use crate::torch_runner::tests::composite_input::loader_b::BStageInput;
use crate::torch_runner::tests::load_and_clc::LoadAndClcStage;

pub mod loader_a;
pub mod loader_b;
pub mod ab_clc;
pub mod dims;

pub const AB_CLC_STAGE_ID:u32 = 0;
pub const A_STAGE_ID:u32 = 1;
pub const B_STAGE_ID:u32 = 2;
pub const AB_SYNC_CLC_STAGE_ID:u32 = 3;

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_ab_clc() -> Result<(), ThoError>{
    crate::runner::global_ctx::GlobalCtx::init_devices(2);
    let ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true, true);

    let ab_stage  = Arc::new(ABClcStage::new());
    let mut ctx = RunCtxImpl::new(1, "root");

    let out = ab_stage.run_cached(
        Arc::new(DevicesWrap::wrap(ABClcStageInput::new(
            Arc::new(DevicesWrap::wrap(AStageInput::new(0, "a100"))),
            Arc::new(DevicesWrap::wrap(BStageInput::new(1, "b100"))),
        ))),
        ctx.branch(false)
    ).await?;

    let value = with_data!(ctx, out; |ctx, out:&ABClcStageOutput|{

        Ok(out.value)
    })?;

    debug!("test_ab_clc: value={value:?}");

    assert_eq!(value, 200.3);

    Ok(())
}