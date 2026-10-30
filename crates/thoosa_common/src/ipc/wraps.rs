use crate::copy_ctx::{HandleCtx, run_cuda_stage};
use crate::ipc::cuda_wokrer::{AllTensors, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::serialization::json::IpcCodec;
use crate::ipc::worker_msg::WorkerInput;
use crate::runner::purge::FuturesRunnerMapPurger;
use crate::tho_device::ThoDevice;
use futures_util::{FutureExt};
use futures_util::future::BoxFuture;
use log::debug;
use std::any::Any;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use tch::Tensor;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{RunCtx, Stage};
use thoosa_common::runner::sync_stage::{ConsumeResult, StageStateInfo, StageWithState, SyncStage};
#[cfg(test)]
use tokio::sync::MappedMutexGuard;
#[cfg(test)]
use tokio::sync::MutexGuard;
use thoosa_shared::ipc_types::WorkerOutput;

///

pub trait WorkerWrapperCtx: RunCtx {
    fn from_worker_ctx(ctx: &WokerRunCtx) -> Self;
}

impl WorkerWrapperCtx for WokerRunCtx {
    fn from_worker_ctx(ctx: &WokerRunCtx) -> Self {
        WokerRunCtx::new(ctx.tho_device_idx())
    }
}

impl WorkerWrapperCtx for crate::runner::run_ctx::RunCtxImpl {
    fn from_worker_ctx(ctx: &WokerRunCtx) -> Self {
        let stage_ctx = Self::new(ctx.tho_device_idx(), "WorkerWrapper");

        stage_ctx
    }
}


pub type InpF<I>= Box<dyn for<'a> Fn(&'a dyn WorkerInput)->Box<&I> + Sync + Send>;

pub struct WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    base:Arc<S>,
    pd:PhantomData<Co>,
    inpf:InpF<I>
}

impl <Co, S, I, O> Debug for WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,

{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerWrapper")
            .field("base", &self.base)
            .finish()
    }
}

impl <Co, S, I, O> WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    pub fn new(base: Arc<S>, pd: PhantomData<Co>) -> Self {
        Self {
            base, pd,
            inpf: Box::new(|i| {
                let a: &dyn WorkerInput = i;
                let a: &dyn Any = a;

                Box::new(a.downcast_ref::<I>().expect(&format!("downcast_ref failed for input={i:?}")))
            })
        }
    }
}

impl <Co, S, I, O>SyncStageIpcWrapper for WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: SyncStage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput+ 'static,
{

    fn consume(&self, input: Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
        debug!("SyncStageIpcWrapper::consume: input={input:?}");

        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx());
        let stage_ctx = S::Ctx::from_worker_ctx(&worker_ctx);


        let a = (self.inpf)(input.as_ref());
        let worker_input = a.ref_to_tensor(all_cuda_tensors, false)?;
        match Arc::clone(&self.base).consume(worker_input, stage_ctx) {
            ConsumeResult::Ok(out) => {
                // Commit removal only after the stage accepts the input so a failed consume can be retried.
                drop(a.ref_to_tensor(all_cuda_tensors, true)?);
                debug!("SyncStageIpcWrapper::consume: out={out:?}");
                Ok(Box::new(out.encode_from_worker(all_cuda_tensors, tho_device)?))
            }
            ConsumeResult::Failed { error, .. } => Err(error),
        }
    }

    fn do_run(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>

    {
        debug!("SyncStageIpcWrapper::do_run: input={input:?}");

        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx());
        let stage_ctx = S::Ctx::from_worker_ctx(&worker_ctx);

        let input_ref = (self.inpf)(input.as_ref());
        let out = Arc::clone(&self.base).do_run(&input_ref.ref_to_tensor(all_cuda_tensors, false)?, &stage_ctx)?;

        debug!("SyncStageIpcWrapper::do_run: out={out:?}");

        Ok(Box::new(out.encode_from_worker(all_cuda_tensors, tho_device)?))
    }

}
///
// A downstream crate can register its own wire type and lend the shared stage its
// original input, without implementing WorkerInput for a foreign generic type.
fn borrowed_worker_input<I: 'static, W: Borrow<I> + 'static>(input: &dyn WorkerInput) -> Result<&I, ThoError> {
    let input: &dyn Any = input;
    input.downcast_ref::<W>().map(Borrow::borrow).ok_or_else(|| ThoError::new(format!("worker expected input type {}", std::any::type_name::<W>())))
}

#[derive(Debug)]
pub struct WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: SyncStage + StageWithState + FuturesRunnerMapPurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    base:Arc<St>,
    pd:PhantomData<Co>,
    input_ref: for<'a> fn(&'a dyn WorkerInput) -> Result<&'a I, ThoError>,
}

impl <Co, St, I, O> WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: SyncStage + StageWithState + FuturesRunnerMapPurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + 'static,
{
    pub fn new(base: Arc<St>, pd: PhantomData<Co>) -> Self {
        Self { base, pd, input_ref: borrowed_worker_input::<I, I> }
    }

    /// Accept a crate-owned WorkerInput wrapper that borrows the shared stage input.
    /// The proxy's input cloner must send W; only I materializes tensor references.
    pub fn new_with_input<W>(base: Arc<St>, pd: PhantomData<Co>) -> Self
    where
        W: WorkerInput + Borrow<I>,
    {
        Self { base, pd, input_ref: borrowed_worker_input::<I, W> }
    }
}

impl <Co, St, I, O>SyncStageIpcWrapper for WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
    St: SyncStage + StageWithState + FuturesRunnerMapPurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput + 'static,
{
    fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
        // Keep this endpoint's concrete training-module state behind the erased worker capability.
        Some(self as &dyn ErasedWithState)
    }

    fn purge(&self, tho_device: ThoDevice) -> BoxFuture<'_, Result<(), ThoError>>
    where
        Self: Sync
    {
        debug!("SyncStageIpcWrapper>::purge: self.base={:?}", self.base);

        let base = Arc::clone(&self.base);
        async move {

            base.drop_gpu(tho_device.tho_device_idx(), "worker").await?;

            Ok(())
        }.boxed()
    }
}

impl <Co, St, I, O> SyncStageIpcWrapperWithState for WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: SyncStage + StageWithState + FuturesRunnerMapPurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput +WorkerOutput + 'static,
{
    type S = <St as StageWithState>::S;

    fn pre_run<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<tokio::sync::MappedMutexGuard<'a, Self::S>, ThoError>> {
        async {
            Ok(StageWithState::pre_run(self.base.as_ref(), tho_device.tho_device_idx()).await?.expect("no state 2"))//TODO error?
        }.boxed()
    }

    fn get_state_info<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<StageStateInfo, ThoError>> {
        StageWithState::get_state_info(self.base.as_ref(), tho_device.tho_device_idx())
    }

    fn reset_state(&self, info: StageStateInfo) -> BoxFuture<'_, Result<(), ThoError>> {
        StageWithState::reset_state(self.base.as_ref(), info)
    }

    fn get_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id: String) -> BoxFuture<'a, Result<Tensor, ThoError>> {
        StageWithState::get_state_tensor(self.base.as_ref(), tho_device.tho_device_idx(), tensor_id)
    }

    fn set_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id: String, t:Tensor) -> BoxFuture<'a, Result<(), ThoError>> {
        StageWithState::set_state_tensor(self.base.as_ref(), tho_device.tho_device_idx(), tensor_id, t)
    }

    fn do_run_with_state<'a>(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, state: &mut Self::S, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
        debug!("SyncStageIpcWrapper::do_run_with_state: input={input:?}");

        let worker_ctx = WokerRunCtx::new(run_ctx.tho_device_idx());//TODO fixit
        let stage_ctx = St::Ctx::from_worker_ctx(&worker_ctx);

        let a = (self.input_ref)(input.as_ref())?;
        let out = Arc::clone(&self.base).do_run_with_state(&a.ref_to_tensor(all_cuda_tensors, false)?, state, &stage_ctx)?;

        debug!("SyncStageIpcWrapper::do_run_with_state: out={out:?}");

        Ok(Box::new(out.encode_from_worker(all_cuda_tensors, tho_device)?))
    }
}

///object-safe trait for running stages in CudaWorker
pub trait SyncStageIpcWrapper: Send + Sync + Debug {
    fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
        None
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="SyncStageIpcWrapper::do_run_response"))]
    fn do_run(&self, _input: &Box<dyn WorkerInput>, _all_cuda_tensors: &AllTensors, _tho_device: &ThoDevice, _run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
        // A wrapper may support only one dispatch capability.
        Err(ThoError::new("worker wrapper does not support borrowed execution"))
    }
    fn consume(&self, _input: Box<dyn WorkerInput>, _all_cuda_tensors: &AllTensors, _tho_device: &ThoDevice, _run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>{
        Err(ThoError::new("worker wrapper does not support consuming execution"))
    }
    
    fn purge(&self, _tho_device: ThoDevice) -> BoxFuture<'_, Result<(), ThoError>>//move to SyncStageIpcWrapperWithState
    {
        async move {
            Ok(())
        }.boxed()
    }
}

pub trait SyncStageIpcWrapperWithState: SyncStageIpcWrapper {
    // The stage mutex grants exclusive access; the state itself need not be Sync.
    type S: Debug + Send;

    fn pre_run<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<tokio::sync::MappedMutexGuard<'a, Self::S>, ThoError>>;

    fn get_state_info<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<StageStateInfo, ThoError>>;

    fn reset_state(&self, _info: StageStateInfo) -> BoxFuture<'_, Result<(), ThoError>> {
        Box::pin(async { Err(ThoError::new("stage does not support checkpoint restoration")) })
    }
    fn get_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id:String) -> BoxFuture<'a, Result<Tensor, ThoError>>;
    fn set_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id:String, t:Tensor) -> BoxFuture<'a, Result<(), ThoError>>;

    fn do_run_with_state<'a>(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, state: &mut Self::S, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>;
}

pub trait ErasedWithState: Sync + Debug {

    fn do_run_with_state<'a>(&'a self, input: &'a Box<dyn WorkerInput>, all_cuda_tensors: &'a AllTensors, tho_device: &'a ThoDevice, handle_ctx: &'a mut dyn HandleCtx, run_ctx: &'a WokerRunCtx) -> BoxFuture<'a, Result<Box<dyn WorkerOutput>, ThoError>>;
    fn get_state_info<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<StageStateInfo, ThoError>>;
    fn reset_state(&self, _info: StageStateInfo) -> BoxFuture<'_, Result<(), ThoError>> {
        Box::pin(async { Err(ThoError::new("stage does not support checkpoint restoration")) })
    }
    fn get_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id:String) -> BoxFuture<'a, Result<Tensor, ThoError>>;
    fn set_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id:String, t:Tensor) -> BoxFuture<'a, Result<(), ThoError>>;
}

// Stateless execution shares the same error/panic boundary as stateful and local
// execution, including synchronization after failed submissions.
pub(crate) fn run_worker_stage(
    stage: &dyn SyncStageIpcWrapper, input: Box<dyn WorkerInput>, consume: bool,
    all_cuda_tensors: &AllTensors, tho_device: &ThoDevice, run_ctx: &WokerRunCtx,
    handle_ctx: &mut dyn HandleCtx,
) -> Result<Box<dyn WorkerOutput>, ThoError> {
    let _guard = handle_ctx.sync_ctx()?;
    run_cuda_stage(|| {
        if consume { stage.consume(input, all_cuda_tensors, tho_device, run_ctx) }
        else { stage.do_run(&input, all_cuda_tensors, tho_device, run_ctx) }
    }, || handle_ctx.on_after_run())
}

async fn run_worker_stage_with_state<T: SyncStageIpcWrapperWithState>(
    stage: &T, input: &Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors,
    tho_device: &ThoDevice, run_ctx: &WokerRunCtx, handle_ctx: &mut dyn HandleCtx,
) -> Result<Box<dyn WorkerOutput>, ThoError> {
    let mut state = stage.pre_run(tho_device).await?;
    // Create the thread-affine guard only after the last await. Keep ownership
    // of state here while submission and synchronization run, even on unwind.
    let _guard = handle_ctx.sync_ctx()?;
    run_cuda_stage(
        || stage.do_run_with_state(input, all_cuda_tensors, &mut *state, tho_device, run_ctx),
        || handle_ctx.on_after_run(),
    )
}

impl<T> ErasedWithState for T
where
    T: SyncStageIpcWrapperWithState + 'static,
{

    fn do_run_with_state<'a>(&'a self, input: &'a Box<dyn WorkerInput>, all_cuda_tensors: &'a AllTensors, tho_device: &'a ThoDevice, handle_ctx: &'a mut dyn HandleCtx, run_ctx: &'a WokerRunCtx) -> BoxFuture<'a, Result<Box<dyn WorkerOutput>, ThoError>> {
        async move {
            run_worker_stage_with_state(self, input, all_cuda_tensors, tho_device, run_ctx, handle_ctx).await
        }.boxed()
    }

    fn get_state_info<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<StageStateInfo, ThoError>> {
        async move {
            SyncStageIpcWrapperWithState::get_state_info(self, tho_device).await
        }.boxed()
    }

    fn reset_state(&self, info: StageStateInfo) -> BoxFuture<'_, Result<(), ThoError>> {
        SyncStageIpcWrapperWithState::reset_state(self, info)
    }

    fn get_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id: String) -> BoxFuture<'a, Result<Tensor, ThoError>> {
        async move {
            SyncStageIpcWrapperWithState::get_state_tensor(self, tho_device, tensor_id).await
        }.boxed()
    }

    fn set_state_tensor<'a>(&'a self, tho_device: &'a ThoDevice, tensor_id: String, t:Tensor) -> BoxFuture<'a, Result<(), ThoError>> {
        async move {
            SyncStageIpcWrapperWithState::set_state_tensor(self, tho_device, tensor_id, t).await
        }.boxed()
    }
}

#[cfg(test)]
pub(crate) mod mock_stateful_worker {
    use super::*;

    mockall::mock! {
        #[derive(Debug)]
        pub StatefulWorker {
            pub fn state(&self) -> &Arc<tokio::sync::Mutex<usize>>;
            pub fn execute(&self, state: &mut usize) -> Result<Box<dyn WorkerOutput>, ThoError>;
        }
    }

    impl SyncStageIpcWrapper for MockStatefulWorker {
        fn as_with_state(&self) -> Option<&dyn ErasedWithState> { Some(self) }
        fn do_run(&self, _: &Box<dyn WorkerInput>, _: &AllTensors, _: &ThoDevice, _: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
            self.execute(&mut 0)
        }
        fn consume(&self, _: Box<dyn WorkerInput>, _: &AllTensors, _: &ThoDevice, _: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
            self.execute(&mut 0)
        }
    }

    impl SyncStageIpcWrapperWithState for MockStatefulWorker {
        type S = usize;
        fn pre_run<'a>(&'a self, _: &'a ThoDevice) -> BoxFuture<'a, Result<MappedMutexGuard<'a, usize>, ThoError>> {
            async move { Ok(MutexGuard::map(self.state().lock().await, |state| state)) }.boxed()
        }
        fn do_run_with_state<'a>(&self, _: &Box<dyn WorkerInput>, _: &AllTensors, state: &mut usize, _: &ThoDevice, _: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
            self.execute(state)
        }
        fn get_state_info<'a>(&'a self, _: &'a ThoDevice) -> BoxFuture<'a, Result<StageStateInfo, ThoError>> { unreachable!("execution must not export state") }
        fn get_state_tensor<'a>(&'a self, _: &'a ThoDevice, _: String) -> BoxFuture<'a, Result<Tensor, ThoError>> { unreachable!("execution must not export state") }
        fn set_state_tensor<'a>(&'a self, _: &'a ThoDevice, _: String, _: Tensor) -> BoxFuture<'a, Result<(), ThoError>> { unreachable!("execution must not restore state") }
    }

    pub struct MockStatefulWorkerBuilder {
        pub state: Arc<tokio::sync::Mutex<usize>>,
        pub panics: bool,
        pub fails: bool,
    }

    impl MockStatefulWorkerBuilder {
        pub fn build(self) -> MockStatefulWorker {
            let mut stage = MockStatefulWorker::new();
            stage.expect_state().return_const(self.state);
            stage.expect_execute().times(1).return_once(move |state| {
                // Simulate work submitted before an error/panic, without CUDA.
                *state += 1;
                if self.panics { std::panic::panic_any("worker submission panic"); }
                if self.fails { return Err(ThoError::oom()); }
                Ok(Box::new(crate::ipc::worker_msg::DebugOutput::new(&vec![])))
            });
            stage
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::mock_stateful_worker::MockStatefulWorkerBuilder;
    use crate::copy_ctx::MockHandleCtx;
    use crate::ipc::worker_msg::CudaWorkerId;

    #[tokio::test]
    async fn test_stateless_worker_synchronizes_errors_and_panics() {
        let tensors = AllTensors::new(CudaWorkerId { device_idx: 1, generation: thoosa_shared::CudaDeviceGeneration(0) });
        let device: ThoDevice = thoosa_common::tho_device::ThoDevice::from(1usize);
        let ctx = WokerRunCtx::new(1);
        for consume in [false, true] {
            for outcome in 0..3 {
                for fatal_sync in [false, true] {
                    let stage = MockStatefulWorkerBuilder { state: Arc::new(tokio::sync::Mutex::new(0)), panics: outcome == 2, fails: outcome == 1 }.build();
                    let mut handle = MockHandleCtx::new();
                    handle.expect_sync_ctx().times(1).return_once(|| Ok(Box::new(())));
                    handle.expect_on_after_run().times(1).return_once(move || {
                        if fatal_sync { Err(ThoError::Cuda { action: thoosa_shared::CudaAction::NeedsRestart, msg: "stateless sync failure".into() }) } else { Ok(()) }
                    });
                    // Exercise the actual worker dispatch boundary for both
                    // borrowed and consuming endpoints, without CUDA timing.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_worker_stage(&stage, Box::new(()), consume, &tensors, &device, &ctx, &mut handle)));
                    if fatal_sync { assert!(result.unwrap().unwrap_err().needs_cuda_restart()); }
                    else if outcome == 2 { assert!(result.is_err()); }
                    else if outcome == 1 { assert!(result.unwrap().unwrap_err().is_oom()); }
                    else { assert!(result.unwrap().is_ok()); }
                }
            }
        }
    }

    #[tokio::test]
    async fn test_worker_state_guard_survives_synchronization_on_success_error_and_panic() {
        let tensors = AllTensors::new(CudaWorkerId { device_idx: 1, generation: thoosa_shared::CudaDeviceGeneration(0) });
        let device: ThoDevice = thoosa_common::tho_device::ThoDevice::from(1usize);
        let input: Box<dyn WorkerInput> = Box::new(());
        let ctx = WokerRunCtx::new(1);
        for outcome in 0..3 {
            for sync_fails in [false, true] {
                let state = Arc::new(tokio::sync::Mutex::new(0));
                let stage = MockStatefulWorkerBuilder { state: Arc::clone(&state), panics: outcome == 2, fails: outcome == 1 }.build();
                let mut handle = MockHandleCtx::new();
                handle.expect_sync_ctx().times(1).return_once(|| Ok(Box::new(())));
                let during_sync = Arc::clone(&state);
                handle.expect_on_after_run().times(1).return_once(move || {
                    // This is the execution body used by the erased IPC wrapper;
                    // a real mutex detects an early unlock even after failure.
                    assert!(during_sync.try_lock().is_err(), "worker state unlocked before stream synchronization");
                    if sync_fails { Err(ThoError::Cuda { action: thoosa_shared::CudaAction::NeedsRestart, msg: "worker synchronization".into() }) } else { Ok(()) }
                });
                let result = std::panic::AssertUnwindSafe(run_worker_stage_with_state(&stage, &input, &tensors, &device, &ctx, &mut handle)).catch_unwind().await;
                assert_eq!(*state.try_lock().expect("worker state must unlock after synchronization"), 1);
                if sync_fails {
                    assert!(result.unwrap().unwrap_err().needs_cuda_restart());
                } else if outcome == 2 {
                    assert_eq!(result.unwrap_err().downcast_ref::<&str>(), Some(&"worker submission panic"));
                } else if outcome == 1 {
                    assert!(result.unwrap().unwrap_err().is_oom());
                } else {
                    assert!(result.unwrap().is_ok());
                }
            }
        }
    }
}
