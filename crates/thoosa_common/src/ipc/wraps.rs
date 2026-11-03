use crate::ipc::state_transfer::{StateTransfer, CheckpointRestore};
use crate::runner::sync_stage::{ConsumingStage, PrepareStage};
use crate::copy_ctx::{HandleCtx, run_cuda_stage};
use crate::ipc::cuda_wokrer::{AllTensors, WokerRunCtx, WorkerStageInput, WorkerStageOutput};
use crate::ipc::serialization::json::IpcCodec;
use crate::ipc::worker_msg::WorkerInput;
use crate::runner::purge::CachePurger;
use crate::tho_device::ThoDevice;
use futures_util::{FutureExt};
use futures_util::future::BoxFuture;
use log::debug;
use std::any::Any;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::marker::PhantomData;
use std::sync::Arc;
use thoosa_shared::ThoError;
use thoosa_shared::traits::{RunCtx, Stage};
use thoosa_common::runner::sync_stage::{ConsumeResult, StageWithState, SyncStage};
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
    S: Stage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput + 'static,
{
    base:Arc<S>,
    pd:PhantomData<Co>,
    execution: fn(&Self) -> WorkerExecution<'_>,
    inpf:InpF<I>
}

impl <Co, S, I, O> Debug for WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: Stage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput + 'static,

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
    S: Stage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput + 'static,
{
    pub fn new(base: Arc<S>, pd: PhantomData<Co>) -> Self where S: SyncStage {
        Self::with_execution(base, pd, |wrapper| WorkerExecution::Borrowed(wrapper))
    }

    pub fn new_consuming(base: Arc<S>, pd: PhantomData<Co>) -> Self where S: ConsumingStage {
        Self::with_execution(base, pd, |wrapper| WorkerExecution::Consuming(wrapper))
    }

    pub fn new_both(base: Arc<S>, pd: PhantomData<Co>) -> Self where S: SyncStage + ConsumingStage {
        Self::with_execution(base, pd, |wrapper| WorkerExecution::Both(wrapper, wrapper))
    }

    fn with_execution(base: Arc<S>, pd: PhantomData<Co>, execution: fn(&Self) -> WorkerExecution<'_>) -> Self {
        Self {
            base, pd, execution,
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
    S: Stage + Debug + Send + Sync + 'static,
    S: Stage<In<'static> = I, Out = O>,
    S::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput+ 'static,
{
    fn execution(&self) -> WorkerExecution<'_> { (self.execution)(self) }
}

impl <Co, S, I, O>WorkerKernel for WorkerWrapper<Co, S, I, O>
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

impl <Co, S, I, O>WorkerConsume for WorkerWrapper<Co, S, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    S: ConsumingStage + Debug + Send + Sync + 'static,
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
    St: StageWithState + CachePurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput + 'static,
{
    base:Arc<St>,
    pd:PhantomData<Co>,
    checkpoint: Option<Arc<dyn CheckpointRestore>>,
    transfer: Option<Arc<dyn StateTransfer>>,
    input_ref: for<'a> fn(&'a dyn WorkerInput) -> Result<&'a I, ThoError>,
}

impl <Co, St, I, O> WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
// SyncStageIpcWrapper is Sync and stores the stage in Arc, so the stage
// must be both Send and Sync at this IPC boundary.
    St: StageWithState + CachePurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput + 'static,
{
    pub fn new(base: Arc<St>, pd: PhantomData<Co>) -> Self where St: StateTransfer {
        let transfer = Some(Arc::clone(&base) as Arc<dyn StateTransfer>);
        Self { base, pd, checkpoint: None, transfer, input_ref: borrowed_worker_input::<I, I> }
    }

    /// Execution-only state need not implement tensor transport. It cannot be
    /// exported by the residency protocol; its owner supplies its lifecycle.
    pub fn new_execution_only(base: Arc<St>, pd: PhantomData<Co>) -> Self {
        Self { base, pd, checkpoint: None, transfer: None, input_ref: borrowed_worker_input::<I, I> }
    }

    /// Accept a crate-owned WorkerInput wrapper that borrows the shared stage input.
    /// The proxy's input cloner must send W; only I materializes tensor references.
    pub fn new_with_input<W>(base: Arc<St>, pd: PhantomData<Co>) -> Self
    where
        W: WorkerInput + Borrow<I>, St: StateTransfer,
    {
        let transfer = Some(Arc::clone(&base) as Arc<dyn StateTransfer>);
        Self { base, pd, checkpoint: None, transfer, input_ref: borrowed_worker_input::<I, W> }
    }
    /// Checkpoint restoration is installed only for stages that implement it.
    pub fn new_with_checkpoint_input<W>(base: Arc<St>, pd: PhantomData<Co>) -> Self
    where W: WorkerInput + Borrow<I>, St: CheckpointRestore,
    {
        let checkpoint = Some(Arc::clone(&base) as Arc<dyn CheckpointRestore>);
        let transfer = Some(Arc::clone(&base) as Arc<dyn StateTransfer>);
        Self { base, pd, checkpoint, transfer, input_ref: borrowed_worker_input::<I, W> }
    }

}

impl <Co, St, I, O>SyncStageIpcWrapper for WorkerWrapperWithState<Co, St, I, O>
where
    Co: IpcCodec,
    St: StageWithState + CachePurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput + WorkerOutput + 'static,
{
    fn execution(&self) -> WorkerExecution<'_> { WorkerExecution::Stateful(self) }

    fn checkpoint_restore(&self) -> Option<&dyn CheckpointRestore> { self.checkpoint.as_deref() }

    fn state_transfer(&self) -> Option<&dyn StateTransfer> { self.transfer.as_deref() }

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
    St: StageWithState + CachePurger + Debug + Send + Sync + 'static,
    St: Stage<In<'static> = I, Out = O>,
    St::Ctx: WorkerWrapperCtx,
    I: WorkerStageInput + 'static,
    O: WorkerStageOutput +WorkerOutput + 'static,
{
    type S = <St as PrepareStage>::S;

    fn pre_run<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<tokio::sync::MappedMutexGuard<'a, Self::S>, ThoError>> {
        async {
            PrepareStage::pre_run(self.base.as_ref(), tho_device.tho_device_idx()).await
        }.boxed()
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

/// The wire request is validated against installed capabilities before invoking
/// a kernel. An endpoint cannot accidentally inherit an unsupported operation.
pub enum WorkerExecution<'a> {
    Borrowed(&'a dyn WorkerKernel),
    Consuming(&'a dyn WorkerConsume),
    Both(&'a dyn WorkerKernel, &'a dyn WorkerConsume),
    Stateful(&'a dyn ErasedWithState),
}

impl WorkerExecution<'_> {
    pub(crate) fn validate_request(&self, consume: bool) -> Result<(), ThoError> {
        match (self, consume) {
            (Self::Borrowed(_) | Self::Stateful(_), false) | (Self::Consuming(_), true) | (Self::Both(..), _) => Ok(()),
            _ => Err(ThoError::new("request does not match the installed worker execution capability")),
        }
    }
}

pub trait WorkerKernel: Send + Sync + Debug {
    fn do_run(&self, input: &Box<dyn WorkerInput>, tensors: &AllTensors, device: &ThoDevice, ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>;
}

pub trait WorkerConsume: Send + Sync + Debug {
    fn consume(&self, input: Box<dyn WorkerInput>, tensors: &AllTensors, device: &ThoDevice, ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>;
}

pub trait SyncStageIpcWrapper: Send + Sync + Debug {
    fn execution(&self) -> WorkerExecution<'_>;

    fn as_with_state(&self) -> Option<&dyn ErasedWithState> {
        match self.execution() { WorkerExecution::Stateful(state) => Some(state), _ => None }
    }

    fn checkpoint_restore(&self) -> Option<&dyn CheckpointRestore> { None }
    fn state_transfer(&self) -> Option<&dyn StateTransfer> { None }

    fn purge(&self, _device: ThoDevice) -> BoxFuture<'_, Result<(), ThoError>> {
        async { Ok(()) }.boxed()
    }
}

pub trait SyncStageIpcWrapperWithState: SyncStageIpcWrapper {
    // The stage mutex grants exclusive access; the state itself need not be Sync.
    type S: Debug + Send;

    fn pre_run<'a>(&'a self, tho_device: &'a ThoDevice) -> BoxFuture<'a, Result<tokio::sync::MappedMutexGuard<'a, Self::S>, ThoError>>;







    fn do_run_with_state<'a>(&self, input: &Box<dyn WorkerInput>, all_cuda_tensors: &AllTensors, state: &mut Self::S, tho_device: &ThoDevice, run_ctx: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError>;
}

pub trait ErasedWithState: Sync + Debug {

    fn do_run_with_state<'a>(&'a self, input: &'a Box<dyn WorkerInput>, all_cuda_tensors: &'a AllTensors, tho_device: &'a ThoDevice, handle_ctx: &'a mut dyn HandleCtx, run_ctx: &'a WokerRunCtx) -> BoxFuture<'a, Result<Box<dyn WorkerOutput>, ThoError>>;




}

// Stateless execution shares the same error/panic boundary as stateful and local
// execution, including synchronization after failed submissions.
pub(crate) fn run_worker_stage(
    execution: WorkerExecution<'_>, input: Box<dyn WorkerInput>, consume: bool,
    all_cuda_tensors: &AllTensors, tho_device: &ThoDevice, run_ctx: &WokerRunCtx,
    handle_ctx: &mut dyn HandleCtx,
) -> Result<Box<dyn WorkerOutput>, ThoError> {
    // Reject incompatible wire requests before acquiring a stream or touching
    // input tensors. Stateful execution uses its own preparation boundary.
    execution.validate_request(consume)?;
    let _guard = handle_ctx.sync_ctx()?;
    run_cuda_stage(|| {
        match (execution, consume) {
            (WorkerExecution::Borrowed(kernel) | WorkerExecution::Both(kernel, _), false) => kernel.do_run(&input, all_cuda_tensors, tho_device, run_ctx),
            (WorkerExecution::Consuming(kernel) | WorkerExecution::Both(_, kernel), true) => kernel.consume(input, all_cuda_tensors, tho_device, run_ctx),
            _ => Err(ThoError::new("request does not match the installed worker execution capability")),
        }
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
        fn execution(&self) -> WorkerExecution<'_> { WorkerExecution::Stateful(self) }
    }

    impl WorkerKernel for MockStatefulWorker {
        fn do_run(&self, _: &Box<dyn WorkerInput>, _: &AllTensors, _: &ThoDevice, _: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> { self.execute(&mut 0) }
    }

    impl WorkerConsume for MockStatefulWorker {
        fn consume(&self, _: Box<dyn WorkerInput>, _: &AllTensors, _: &ThoDevice, _: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> { self.execute(&mut 0) }
    }

    impl SyncStageIpcWrapperWithState for MockStatefulWorker {
        type S = usize;
        fn pre_run<'a>(&'a self, _: &'a ThoDevice) -> BoxFuture<'a, Result<MappedMutexGuard<'a, usize>, ThoError>> {
            async move { Ok(MutexGuard::map(self.state().lock().await, |state| state)) }.boxed()
        }
        fn do_run_with_state<'a>(&self, _: &Box<dyn WorkerInput>, _: &AllTensors, state: &mut usize, _: &ThoDevice, _: &WokerRunCtx) -> Result<Box<dyn WorkerOutput>, ThoError> {
            self.execute(state)
        }

    }

    pub struct MockStatefulWorkerBuilder {
        pub state: Arc<tokio::sync::Mutex<usize>>,
        pub panics: bool,
        pub fails: bool,
    }

    impl MockStatefulWorkerBuilder {
        pub fn build_without_execution() -> MockStatefulWorker {
            let mut stage = MockStatefulWorker::new();
            stage.expect_execute().times(0);
            stage
        }

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

    #[test]
    fn test_worker_rejects_mismatched_execution_capability_before_stream_setup() {
        let stage = MockStatefulWorkerBuilder::build_without_execution();
        let tensors = AllTensors::new(CudaWorkerId { device_idx: 1, generation: thoosa_shared::CudaDeviceGeneration(0) });
        let device = ThoDevice::from(1usize);
        let ctx = WokerRunCtx::new(1);
        for consume in [false, true] {
            let execution = if consume { WorkerExecution::Borrowed(&stage) } else { WorkerExecution::Consuming(&stage) };
            let mut handle = MockHandleCtx::new();
            handle.expect_sync_ctx().times(0);
            handle.expect_on_after_run().times(0);
            let error = run_worker_stage(execution, Box::new(()), consume, &tensors, &device, &ctx, &mut handle).unwrap_err();
            assert!(!error.is_oom() && !error.needs_cuda_recovery());
        }
    }

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
                        // Use the constructor so injected CUDA errors capture their backtrace.
                        if fatal_sync { Err(thoosa_shared::tho_error::cuda_error(thoosa_shared::CudaAction::NeedsRestart, "stateless sync failure".into())) } else { Ok(()) }
                    });
                    // Exercise the actual worker dispatch boundary for both
                    // borrowed and consuming endpoints, without CUDA timing.
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_worker_stage(WorkerExecution::Both(&stage, &stage), Box::new(()), consume, &tensors, &device, &ctx, &mut handle)));
                    if fatal_sync { assert!(result.unwrap().unwrap_err().needs_cuda_recovery()); }
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
                    if sync_fails { Err(thoosa_shared::tho_error::cuda_error(thoosa_shared::CudaAction::NeedsRestart, "worker synchronization".into())) } else { Ok(()) }
                });
                let result = std::panic::AssertUnwindSafe(run_worker_stage_with_state(&stage, &input, &tensors, &device, &ctx, &mut handle)).catch_unwind().await;
                assert_eq!(*state.try_lock().expect("worker state must unlock after synchronization"), 1);
                if sync_fails {
                    assert!(result.unwrap().unwrap_err().needs_cuda_recovery());
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
