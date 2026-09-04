use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use tch::Tensor;
use thoosa_shared::applicables_runner::{Input, Output};
use thoosa_shared::traits::{MaxParallelCtx, Stage, SyncStage};
use thoosa_shared::ThoError;

use crate::ipc::cuda_wokrer::{WorkerStageInput, WorkerStageOutput};
use crate::ipc::proxy::DynSyncStageEnum;
use crate::runner::adjusters::max_parallel::{MAX_PARALLEL, MaxParallelAdjustContextImpl, MaxParallelAdjuster, MaxParallelStep};
use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use crate::runner::oom_handler::{Adjusters, NoInputAdjuster, NoTaskAdjuster, OOMHandler, SyncStageInit, WithRunCtxContext, run_ctx_adjuster};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::ws::{Workspace, mock_ws_simp};

#[derive(Debug, Serialize)]
struct ConstructorInput;

impl Input for ConstructorInput {}
impl WorkerStageInput for ConstructorInput {}

#[derive(Debug, Deserialize, Serialize)]
struct ConstructorOutput;

impl Output for ConstructorOutput {}

impl WorkerStageOutput for ConstructorOutput {
    fn encode_from_worker(self, _all_cuda_tensors: &Arc<Mutex<HashMap<u64, Tensor>>>, _tho_device: &ThoDevice) -> Result<Self, ThoError> {
        Ok(self)
    }
}

#[derive(Debug)]
struct ConstructorStage {
    max_other_parallel: u8,
    workspace_max_input: u32,
}

impl Stage for ConstructorStage {
    type Ctx = RunCtxImpl;
    type In<'a> = ConstructorInput;
    type Out = ConstructorOutput;
}

impl SyncStage for ConstructorStage {}

#[derive(Clone, Debug)]
struct ConstructorInit {
    calls: Arc<AtomicUsize>,
}

#[async_trait]
impl SyncStageInit<RunCtxImpl, DynSyncStageEnum<ConstructorStage, ConstructorInit>> for ConstructorInit {
    async fn init(self, run_context: RunCtxImpl, ws: Arc<RwLock<Box<dyn Workspace>>>) -> Result<Arc<DynSyncStageEnum<ConstructorStage, ConstructorInit>>, ThoError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        // Fail until MaxParallelAdjuster lowers the constructor's scheduling context, proving that initialization uses the normal OOM retry loop.
        if run_context.max_other_parallel() > 16 {
            return Err(ThoError::oom());
        }
        let workspace_max_input = ws.read().max_input();
        Ok(Arc::new(DynSyncStageEnum::Local(Arc::new(ConstructorStage { max_other_parallel: run_context.max_other_parallel(), workspace_max_input }))))
    }
}

#[tokio::test]
async fn test_sync_stage_constructor_retries_after_run_context_adjustment() -> Result<(), ThoError> {
    let _global_ctx = MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
    let ws = Arc::new(RwLock::new(mock_ws_simp(None, 2, 100, 1234)));
    let calls = Arc::new(AtomicUsize::new(0));
    let sync_init = ConstructorInit { calls: Arc::clone(&calls) };
    let oom_handler: OOMHandler<'_, '_, NoInputAdjuster, NoTaskAdjuster, _, ()> = OOMHandler::new(
        vec![Adjusters::RunContext(run_ctx_adjuster(MaxParallelAdjuster::new(MAX_PARALLEL, *MaxParallelStep::FAST)))],
        None,
    );

    let stage: Arc<DynSyncStageEnum<ConstructorStage, ConstructorInit>> = oom_handler.new_stage(
        sync_init,
        RunCtxImpl::new(1, "sync_stage_init_oom"),
        WithRunCtxContext::new(MaxParallelAdjustContextImpl::new()),
        &ws,
    ).await?;

    let DynSyncStageEnum::Local(stage) = stage.as_ref() else {
        return Err(ThoError::new("constructor test unexpectedly produced a proxy stage"));
    };
    assert_eq!((16, 1234, 3), (stage.max_other_parallel, stage.workspace_max_input, calls.load(Ordering::SeqCst)));
    Ok(())
}
