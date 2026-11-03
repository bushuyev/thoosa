#[cfg(test)]
use thoosa_common::runner::execution::{StageExecution, ExecutionContext, execute_stateful};
#[cfg(test)]
use thoosa_common::runner::sync_stage::PrepareStage;
use crate::cuda_utils::{CopyBufferPool, PinnedHostBuffer, copy_d2h, copy_d2h_to_ptr, copy_h2d, copy_h2d_from_ptr, cuda_mem_for_device};
#[cfg(test)]
use crate::runner::common::cachable::ConsumeData;
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use anyhow::Context as _;
use async_trait::async_trait;
use cust::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
#[cfg(test)]
use futures_util::future::BoxFuture;
use log::{debug, trace, warn};
use std::backtrace::Backtrace;
#[cfg(test)]
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
#[cfg(test)]
use std::future::Future;
use std::sync::Arc;
use std::time::Duration;
use tch::Kind::Int64;
use tch::{Kind, Tensor};
use thoosa_shared::tracked_await::track_await;
#[cfg(test)]
use thoosa_common::runner::sync_stage::ConsumeResult;
#[cfg(test)]
use thoosa_shared::traits::RunCtx;
#[cfg(test)]
use thoosa_shared::traits::Stage;
use thoosa_shared::{CudaActivityGate, CudaActivityScope, CudaDeviceGeneration, ThoError};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tokio::sync::oneshot::Sender;
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::time::timeout;
use tub::{Guard, Pool};


enum CudaLocalActorMessage {
    Copy{ src:ThoTensor, dst_device:ThoDevice, activity: CudaActivityScope, respond_to: Sender<Result<Tensor, ThoError>>},
    // Tensor is Send but not Sync; keep the queued message (and SendError) Sync.
    CopyBatchToCpu { sources: parking_lot::Mutex<Vec<Tensor>>, activity: CudaActivityScope, respond_to: Sender<Result<Vec<Tensor>, ThoError>> },

    GetExecStream {
        activity: CudaActivityScope,
        respond_to: oneshot::Sender<Result<LocalHandleCtx, ThoError>>
    },
    Restart {
        authority: thoosa_shared::CudaActivityPauseTicket,
        respond_to: oneshot::Sender<Result<(), ThoError>>
    }

}

pub trait SyncCtx {}

impl SyncCtx for (){}

#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait HandleCtx:Send {

    fn sync_ctx(&self) ->Result<Box<dyn SyncCtx>, ThoError> {
        Ok(Box::new(()))
    }

     fn on_after_run(&mut self) ->Result<(), ThoError> {
         Ok(())
     }

}

#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait CudaActorHandle: Debug + Send + Sync {
    async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) -> Result<ThoTensor, ThoError>;

    // Backend work is called by GlobalCtx's coordinator hook after draining.
    // It must neither request a new cycle nor change admission state.
    async fn restart_backend(&self, authority: thoosa_shared::CudaActivityPauseTicket) -> Result<(), ThoError>;

    #[cfg(test)]
    async fn restart(&self) -> Result<(), ThoError>;//shortcut bypassing SyncStageRunner
}
// Only the local backend can lend a CUDA stream or copy process-local tensors.
#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait LocalCudaExecution: Debug + Send + Sync {
    async fn init_handle_ctx(&self) -> Result<Box<dyn HandleCtx>, ThoError>;
    async fn copy_batch_to_cpu(&self, sources: Vec<Tensor>) -> Result<Vec<Tensor>, ThoError>;
}

// Catch submission panics inside the wrapper's guard lifetime. CUDA work can
// already be queued when submission errors or panics, so always synchronize.
pub(crate) fn run_cuda_stage<T>(run: impl FnOnce() -> Result<T, ThoError>, synchronize: impl FnOnce() -> Result<(), ThoError>) -> Result<T, ThoError> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(run));
    let sync = synchronize();
    finish_cuda_run(finish_cuda_panic(result, &sync)?, sync)
}

pub(crate) fn finish_cuda_panic<T>(result: std::thread::Result<T>, sync: &Result<(), ThoError>) -> Result<T, ThoError> {
    match result {
        Ok(result) => Ok(result),
        Err(panic) => match sync {
            Ok(()) => std::panic::resume_unwind(panic),
            // Fatal CUDA errors must still reach recovery. A nonfatal sync
            // error must not turn a host panic into a retryable OOM instead.
            Err(error) if error.needs_cuda_recovery() || error.is_cuda_recovery_failed() => {
                let message = panic.downcast_ref::<String>().map(String::as_str).or_else(|| panic.downcast_ref::<&str>().copied()).unwrap_or("non-string panic payload");
                Err(error.clone().context(format!("stage panicked before stream synchronization: {message}")))
            }
            Err(error) => {
                warn!("stream synchronization also failed while unwinding a stage: {error:?}");
                std::panic::resume_unwind(panic)
            }
        },
    }
}

// All stage entry points must retain a fatal execution error even if stream
// synchronization also fails, and must propagate a fatal synchronization error.
pub(crate) fn finish_cuda_run<T>(result: Result<T, ThoError>, sync: Result<(), ThoError>) -> Result<T, ThoError> {
    match (result, sync) {
        (Err(original), sync) => Err(finish_cuda_error(original, sync)),
        (_, Err(error)) => Err(error),
        (Ok(output), Ok(())) => Ok(output),
    }
}

// Consume failures retain their owned input, but use the same error precedence
// as borrowed execution so the runner sees fatal stream errors through either API.
pub(crate) fn finish_cuda_error(original: ThoError, sync: Result<(), ThoError>) -> ThoError {
    match sync {
        Err(sync) => {
            warn!("finish_cuda_error: original={original:?} sync={sync:?}");

            if sync.is_cuda_recovery_failed()
                ||
                (
                    sync.needs_cuda_recovery() && !original.is_cuda_recovery_failed() && !original.needs_cuda_recovery()
                )
            {
                sync.context(format!("stage also failed: {original}"))
            } else {
                original.context(format!("stream synchronization also failed: {sync}"))
            }
        }
        Ok(()) => original,
    }
}

// Report in the actor-owned task, including when the awaiting caller disappeared.
// The captured lease generation prevents a late copy error restarting a successor.
pub(crate) fn report_cuda_copy_result<T>(result: Result<T, ThoError>, gate: &Arc<CudaActivityGate>, generation: CudaDeviceGeneration) -> Result<T, ThoError> {
    result.map_err(|error| {
        if error.needs_cuda_recovery() || error.is_cuda_recovery_failed() {
            GlobalCtx::global()
                .get_runner(gate.device_idx())
                .request_cuda_reset_for_generation(generation, error)
                .err()
                .unwrap_or(ThoError::CudaBeingRecovered)
        } else {
            error
        }
    })
}

struct CudaLocalActor {
    gate: Arc<CudaActivityGate>,
    tho_device_idx:usize,
    receiver: mpsc::Receiver<CudaLocalActorMessage>,
    copy_ctx:Arc<RwLock<CopyCtx>>,
}

impl CudaLocalActor {
    pub fn new(gate: Arc<CudaActivityGate>, receiver: mpsc::Receiver<CudaLocalActorMessage>) -> Result<Self, ThoError> {
        let tho_device_idx = gate.device_idx();
        debug!("CudaActor::new: tho_device_idx={tho_device_idx}");
        assert!(tho_device_idx > 0);

        Ok(Self {
            tho_device_idx,
            receiver,
            copy_ctx: Arc::new(RwLock::new(CopyCtx::with_gate(Arc::clone(&gate), 0.2, 32)?)),
            gate,
        })
    }

     fn handle_message(&mut self, msg: CudaLocalActorMessage) {

        match msg {
            CudaLocalActorMessage::CopyBatchToCpu { sources, activity, respond_to } => {
                if let Err(error) = activity.check_operation(&self.gate) {
                    let _ = respond_to.send(Err(error));
                    return;
                }
                let copy_ctx = Arc::clone(&self.copy_ctx);
                let gate = Arc::clone(&self.gate);
                let generation = activity.generation(&gate).expect("batch copy device lease");
                // Accepted copies keep their owners and activity even if the
                // checkpoint caller disappears before receiving the batch.
                tokio::spawn(activity.run(async move {
                    let result = copy_ctx.read().await.copy_batch_to_cpu(sources.into_inner()).await;
                    let _ = respond_to.send(report_cuda_copy_result(result, &gate, generation));
                }));
            }
            CudaLocalActorMessage::Copy{ src, dst_device, activity, respond_to } => {
                if let Err(error) = activity.check_operation(&self.gate) {
                    let _ = respond_to.send(
                        Err(error)
                    );
                    return;
                }

                let copy_ctx = Arc::clone(&self.copy_ctx);
                let gate = Arc::clone(&self.gate);
                let generation = activity.generation(&gate).expect("copy device lease");
                tokio::spawn(activity.run(async move {
                    debug!("CudaActor::handle_message/Copy: src={src:?}");
                    // Move the allocation into the low-level transfer; retaining it
                    // only in this async task is insufficient during shutdown.
                    let result = copy_ctx.read().await.copy(src.t(), dst_device.tho_device_idx()).await;

                    let result = report_cuda_copy_result(result, &gate, generation);
                    let _ = respond_to.send(result);
                 }));
            }

            CudaLocalActorMessage::Restart { authority, respond_to } => {
                if let Err(error) = self.gate.check_recovery_authority(authority, true) {
                    let _ = respond_to.send(Err(error));
                    return;
                }
                let copy_ctx = Arc::clone(&self.copy_ctx);
                let tho_device_idx = self.tho_device_idx;
                tokio::spawn(async move {
                    let result = Self::do_restart(copy_ctx, tho_device_idx).await;
                    let _ = respond_to.send(result);
                });
            }
            CudaLocalActorMessage::GetExecStream { activity, respond_to } => {
                if let Err(error) = activity.check_operation(&self.gate) {
                    let _ = respond_to.send(
                        Err(error)
                    );
                    return;
                }
                let copy_ctx = Arc::clone(&self.copy_ctx);
                tokio::spawn(async move {
                    // The response owns the lease too: caller cancellation must
                    // not let recovery replace a stream still being acquired.
                    let result = match copy_ctx.read().await.acquire_exec_stream().await {
                        Ok(stream) => Ok(LocalHandleCtx { stream, _activity: activity }),
                        Err(error) => Err(error),
                    };
                    respond_to.send(result)
                });
            }
        }
    }

    async fn do_restart(copy_ctx: Arc<RwLock<CopyCtx>>, tho_device_idx: usize) -> Result<(), ThoError> {
        // Cache cleanup and admission transitions belong to the coordinator.
        let device: ThoDevice = tho_device_idx.into();
        let mut copy_ctx = timeout(Duration::from_secs(30), copy_ctx.write_owned()).await?;
        tokio::task::spawn_blocking(move || {
            // CUDA context binding is thread-local; do not hold it across await.
            let mut cu_ctx = ThoCudaContext::new(i32::try_from(device.gpu_ordinal()?)?)?;
            cu_ctx.restart()?;
            copy_ctx.restart()
        }).await?
    }

    async fn run(mut actor: CudaLocalActor) {
        while let Some(msg) = actor.receiver.recv().await {
            actor.handle_message(msg);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaLocalActorHandle {
    gate: Arc<CudaActivityGate>,
    sender: mpsc::Sender<CudaLocalActorMessage>,
}


impl CudaLocalActorHandle {
    pub fn new(gate: Arc<CudaActivityGate>) -> Result<Self, ThoError> {
        let tho_device_id = gate.device_idx();
        debug!("CudaLocalActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let actor = CudaLocalActor::new(Arc::clone(&gate), receiver)?;
        tokio::spawn(CudaLocalActor::run(actor));

        Ok(Self {
            gate,
            sender,
        })
    }

    async fn copy_tensor(&self, src:ThoTensor, dst_device:ThoDevice, activity: CudaActivityScope) -> anyhow::Result<Tensor> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaLocalActorMessage::Copy {
                src,
                dst_device,
                activity,
                respond_to,
            }
        ).await.context("copy_tensor: CUDA actor channel closed")?;

        Ok(recv.await.context("copy_tensor: CUDA actor response channel closed")??)
    }

    async fn restart_inner(&self, authority: thoosa_shared::CudaActivityPauseTicket) -> anyhow::Result<()> {
        self.gate.check_recovery_authority(authority, true)?;
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaLocalActorMessage::Restart {
                authority,
                respond_to,
            }
        ).await.context("restart_inner: CUDA actor channel closed")?;

        Ok(recv.await.context("restart_inner: CUDA actor response channel closed")??)
    }

    async fn acquire_exec_stream(&self) -> anyhow::Result<LocalHandleCtx> {
        let activity = CudaActivityScope::for_operation(&self.gate)?;
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaLocalActorMessage::GetExecStream {
                activity,
                respond_to,
            }
        ).await.context("acquire_exec_stream: CUDA actor channel closed")?;

        Ok(recv.await.context("acquire_exec_stream: CUDA actor response channel closed")??)
    }

}

// A nonfatal CUDA error can be reported by synchronization without making the
// context unusable. Try to drain the stream once more before quarantining it.
// Even a successful drain must not publish the failed operation's output.
pub(crate) struct StreamCompletion {
    pub(crate) result: Result<(), ThoError>,
    pub(crate) confirmed: bool,
}

pub(crate) fn complete_cuda_stream(result: Result<(), ThoError>, retry: impl FnOnce() -> Result<(), ThoError>) -> StreamCompletion {
    let error = match result {
        Ok(()) => return StreamCompletion { result: Ok(()), confirmed: true },
        Err(error) => error,
    };
    if error.needs_cuda_recovery() || error.is_cuda_recovery_failed() {
        return StreamCompletion { result: Err(error), confirmed: false };
    }
    match retry() {
        Ok(()) => StreamCompletion { result: Err(error), confirmed: true },
        Err(recovery) => {
            let error = if recovery.needs_cuda_recovery() || recovery.is_cuda_recovery_failed() {
                finish_cuda_error(error, Err(recovery))
            } else {
                // Last resort: completion is still unknown, so ordinary reuse
                // and OOM retries are unsafe. Keep both original classifications;
                // this is a failed drain, not a fabricated fatal CUDA code.
                ThoError::CudaRecoveryFailed(Arc::new(thoosa_shared::CudaRecoveryFailure { original: error, recovery: Some(recovery) }))
                    .context("CUDA stream could not be drained after synchronization failed")
            };
            StreamCompletion { result: Err(error), confirmed: false }
        }
    }
}

// Only unconfirmed completion closes the pool. Wake queued acquisitions too,
// so quarantine cannot strand callers waiting for an unavailable stream.
pub(crate) struct ExecutionStreamPool {
    pool: Pool<ThoCudaStream>,
    failure: tokio::sync::watch::Sender<Option<ThoError>>,
    quarantined: parking_lot::Mutex<Vec<Guard<ThoCudaStream>>>,
}

impl ExecutionStreamPool {
    pub(crate) fn new(streams: Vec<ThoCudaStream>) -> Arc<Self> {
        Arc::new(Self { pool: Pool::from_vec(streams), failure: tokio::sync::watch::channel(None).0, quarantined: parking_lot::Mutex::new(Vec::new()) })
    }

    pub(crate) fn check(&self) -> Result<(), ThoError> {
        match self.failure.borrow().as_ref() { Some(error) => Err(error.clone()), None => Ok(()) }
    }

    pub(crate) async fn acquire(self: &Arc<Self>) -> Result<ExecutionStream, ThoError> {
        let mut failure = self.failure.subscribe();
        self.check()?;
        let lease = tokio::select! {
            biased;
            _ = failure.changed() => return Err(failure.borrow().clone().expect("poisoned execution pool")),
            lease = self.pool.acquire() => lease,
        };
        self.check()?;
        Ok(ExecutionStream { lease: Some(lease), pool: Arc::clone(self) })
    }
}

pub struct ExecutionStream {
    lease: Option<Guard<ThoCudaStream>>,
    pool: Arc<ExecutionStreamPool>,
}

impl ExecutionStream {
    fn complete(&mut self, result: Result<(), ThoError>) -> Result<(), ThoError> {
        let completion = complete_cuda_stream(result, || self.lease.as_ref().expect("execution lease").synchronize());
        if !completion.confirmed {
            let error = completion.result.as_ref().unwrap_err();
            self.pool.failure.send_if_modified(|failure| {
                if failure.is_some() { false } else { *failure = Some(error.clone()); true }
            });
            if let Some(lease) = self.lease.take() { self.pool.quarantined.lock().push(lease); }
        }
        completion.result
    }
}

impl HandleCtx for ExecutionStream {
    fn sync_ctx(&self) -> Result<Box<dyn SyncCtx>, ThoError> {
        self.pool.check()?;
        Ok(Box::new(ThoStreamGuard::new(self.lease.as_ref().expect("execution lease"))?))
    }

    fn on_after_run(&mut self) -> Result<(), ThoError> {
        let result = self.lease.as_ref().expect("execution lease").synchronize();
        self.complete(result)
    }
}

struct LocalHandleCtx {
    stream: ExecutionStream,
    _activity: CudaActivityScope,
}

impl SyncCtx for ThoStreamGuard {}
#[async_trait]
impl HandleCtx for LocalHandleCtx {

    fn sync_ctx(&self) -> Result<Box<dyn SyncCtx>, ThoError> {
        self.stream.sync_ctx()
    }

    fn on_after_run(&mut self) -> Result<(), ThoError> {
        self.stream.on_after_run()
    }
}

#[async_trait]
impl CudaActorHandle for CudaLocalActorHandle {


    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="CudaLocalActorHandle::copy"))]
    async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) -> Result<ThoTensor, ThoError> {
        trace!("CudaLocalActorHandle::copy: src={src:?} dst_device={dst_device:?} backtrace={}", Backtrace::capture().to_string());

        // Register before acquiring copy resources; children share this scope so
        // a pause drains the whole copy instead of blocking its remaining chunks.
        let activity = CudaActivityScope::for_operation(&self.gate)?;
        Ok(self.copy_tensor(src, dst_device, activity).await?.into())
    }

    #[cfg(test)]
    async fn restart(&self) -> Result<(), ThoError> {
        GlobalCtx::global().restart_cuda_device(self.gate.device_idx()).await
    }

    async fn restart_backend(&self, authority: thoosa_shared::CudaActivityPauseTicket) -> Result<(), ThoError> {
        Ok(self.restart_inner(authority).await?)
    }





}


pub struct CopyCtx {
    gate: Arc<CudaActivityGate>,
    cuda_device: usize,

    pub(crate) buffs: usize,
    pub(crate) buf_size: usize,
    exec_stream_pool: Arc<ExecutionStreamPool>,
    bufs_pool_d2h: Arc<CopyBufferPool>,
    bufs_pool_h2d: Arc<CopyBufferPool>,
}

impl Debug for CopyCtx {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CopyCtx")
            // .field("cu_ctx", &self.cu_ctx)
            .field("buffs", &self.buffs)
            .field("buf_size", &self.buf_size)
                .field("bufs_pool.remaining_capacity", &self.bufs_pool_h2d.remaining_capacity())
            .finish()
    }
}

impl CopyCtx {

    pub fn make_instance_for_cuda_device(cuda_device: usize, mem_k: f64, buffs: usize) -> Result<CopyCtx, ThoError> {
        // Standalone contexts (including IPC workers) own their admission state.
        // Local actors instead inject the gate shared with their stage runner.
        Self::with_gate(Arc::new(CudaActivityGate::new(cuda_device + 1)), mem_k, buffs)
    }

    fn with_gate(gate: Arc<CudaActivityGate>, mem_k: f64, buffs: usize) -> Result<CopyCtx, ThoError> {
        let cuda_device = gate.device_idx() - 1;

        debug!("make_instance_for_cuda_device: cuda_device={cuda_device} mem_k={mem_k} buffs={buffs}");


        let buf_size = Self::clc_buff_size(cuda_device, mem_k, buffs, cuda_mem_for_device(cuda_device).1);

        let exec_stream_pool = {
            let exec_streams = (0..buffs)
                .map(|_| ThoCudaStream::from_pool(cuda_device as i32, true))
                .collect::<Result<Vec<_>, _>>()?;

                ExecutionStreamPool::new(exec_streams)
        };

        Ok(Self {
            gate,
            cuda_device,
            buffs,
            buf_size,
            exec_stream_pool,
            bufs_pool_d2h: Self::build_pool(buffs /2, buf_size, cuda_device, 0)?,
            bufs_pool_h2d: Self::build_pool(buffs /2, buf_size, cuda_device, CU_MEMHOSTALLOC_WRITECOMBINED)?,
        })
    }


    fn build_pool(
        num_buffs: usize,
        buf_size: usize,
        cuda_device:usize,
        flags: u32
    ) -> Result<Arc<CopyBufferPool>, ThoError> {
        let buffers = (0..num_buffs).map(|_| {
            Ok((
                PinnedHostBuffer::new(buf_size, flags)?,
                Arc::new(ThoCudaStream::from_pool(cuda_device as i32, true)?),//TODO check if streams do not repeat
            ))
        }).collect::<Result<Vec<_>, ThoError>>()?;
        Ok(Arc::new(CopyBufferPool::new(buffers)))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub fn rebuild(&mut self, mem_k: f64, buffs: usize) -> Result<(), ThoError> {
        assert_eq!(self.bufs_pool_h2d.as_ref().remaining_capacity(), self.buffs);
        // self
            // .cu_ctx
            // .bind_to_thread()
            // .expect("failed to bind CUDA context to thread");

        drop(Arc::into_inner(std::mem::replace(
            &mut self.bufs_pool_h2d,
            Arc::new(CopyBufferPool::new(vec![(
                PinnedHostBuffer::new(1, 0).unwrap(),
                Arc::new(ThoCudaStream::from_pool(self.cuda_device as i32, true)?),
            )])),
        )));

        self.buffs = buffs;
        self.buf_size = Self::clc_buff_size(self.cuda_device, mem_k, buffs, cuda_mem_for_device(self.cuda_device).1);
        self.bufs_pool_h2d = Self::build_pool(buffs, self.buf_size, self.cuda_device, CU_MEMHOSTALLOC_WRITECOMBINED)?;
        Ok(())
    }

    pub async fn acquire_exec_stream(&self) -> Result<ExecutionStream, ThoError> {
        track_await!(self.exec_stream_pool.acquire()).await
    }

    pub fn restart(&mut self) -> Result<(), ThoError> {
        // Build all replacements before publishing any pool. Both copy
        // directions retain the same capacity as initial construction.
        let cuda_context = ThoCudaContext::new(self.cuda_device as i32)?;
        cuda_context.bind_to_thread()?;
        let h2d = Self::build_pool(self.buffs / 2, self.buf_size, self.cuda_device, CU_MEMHOSTALLOC_WRITECOMBINED)?;
        let d2h = Self::build_pool(self.buffs / 2, self.buf_size, self.cuda_device, 0)?;
        let exec_streams = (0..32)
            .map(|_| ThoCudaStream::from_pool(self.cuda_device as i32, false))
            .collect::<Result<Vec<_>, _>>()?;
        self.bufs_pool_h2d = h2d;
        self.bufs_pool_d2h = d2h;
        self.exec_stream_pool = ExecutionStreamPool::new(exec_streams);

        Ok(())
    }

    fn clc_buff_size(_cuda_device: usize, mem_k: f64, buffs: usize, total_mem: usize) -> usize {

        let init_type = Int64;

        let max_elt_size = init_type.elt_size_in_bytes();
        let elts_in_mem = (total_mem as f64 * mem_k) as usize / max_elt_size; //123456; //25000; //123456; //

        debug!("CopyCtx::clc_buff_size: elts_in_mem={elts_in_mem} total_mem={total_mem} max_elt_size={max_elt_size}");

        let elts_in_buff = elts_in_mem / buffs;

        debug!("CopyCtx::clc_buff_size: elts_in_buff={elts_in_buff} buffs={buffs}");

        let buf_size = elts_in_buff * max_elt_size;
        buf_size
    }

    /// The transfer takes ownership of the source and of its new destination.
    /// Already-running blocking chunks retain both if this future is cancelled.
    pub async fn copy(&self, source: Tensor, dst_tho_device: usize) -> Result<Tensor, ThoError> {
        if dst_tho_device == 0 {
            copy_d2h(&self.gate, source, self.buf_size, &self.bufs_pool_d2h).await
        } else {
            copy_h2d(&self.gate, source, self.buf_size, &self.bufs_pool_h2d, dst_tho_device).await
        }
    }

    pub(crate) async fn copy_batch_to_cpu(&self, sources: Vec<Tensor>) -> Result<Vec<Tensor>, ThoError> {
        crate::cuda_utils::copy_d2h_batch(&self.gate, sources, self.buf_size, &self.bufs_pool_d2h).await
    }

    /// # Safety
    /// The caller owns disjoint writable destination ranges and retains them
    /// until reply or worker exit; the owner keeps their mapping alive as well.
    pub(crate) async unsafe fn copy_batch_to_cpu_ptrs(&self, sources: Vec<(Tensor, usize)>, owner: Arc<impl Send + Sync + 'static>) -> Result<(), ThoError> {
        unsafe { crate::cuda_utils::copy_d2h_batch_to_ptrs(&self.gate, sources, self.buf_size, &self.bufs_pool_d2h, owner).await }
    }

    /// # Safety
    /// source_owner must retain the readable CPU range implied by size/kind
    /// without concurrent mutation, even after this future is cancelled. The IPC
    /// peer must retain its allocation until reply or worker exit.
    pub(crate) async unsafe fn copy_from_ptr(&self, source: usize, size: &[i64], dst_device: usize, kind: Kind, source_owner: impl Send + 'static) -> Result<Tensor, ThoError> {
        // SAFETY: ownership and range obligations are forwarded to the chunk engine.
        unsafe { copy_h2d_from_ptr(&self.gate, kind, self.buf_size, &self.bufs_pool_h2d, dst_device, source, size, source_owner).await }
    }

    /// # Safety
    /// destination_owner must retain exclusive access to a writable CPU range
    /// covering source's bytes until it is dropped, including after cancellation.
    /// The IPC peer must retain its allocation until reply or worker exit.
    pub(crate) async unsafe fn copy_to_cpu_ptr(&self, source: Tensor, destination: usize, destination_owner: impl Send + 'static) -> Result<(), ThoError> {
        // SAFETY: ownership and range obligations are forwarded to the chunk engine.
        unsafe { copy_d2h_to_ptr(&self.gate, source, self.buf_size, &self.bufs_pool_d2h, destination, destination_owner).await }
    }

}

#[cfg(test)]
mod mock_stateful_stage {
    use super::*;
    use crate::runner::common::MockRunCtx;
    use crate::runner::oom_handler::test_types::{TestInput, TestOutput};
    use futures_util::FutureExt;
    use thoosa_common::runner::sync_stage::{StageWithState};

    mockall::mock! {
        #[derive(Debug)]
        pub StatefulStage {
            pub fn state(&self) -> &Arc<tokio::sync::Mutex<usize>>;
            pub fn execute(&self, input: &TestInput, state: &mut usize, ctx: &MockRunCtx) -> Result<TestOutput, ThoError>;
        }
        impl Stage for StatefulStage {
            type In<'a> = TestInput;
            type Out = TestOutput;
            type Ctx = MockRunCtx;
        }
    }

    impl StageExecution for MockStatefulStage {
    fn execute<'a, 'i: 'a, I, IFut>(self: Arc<Self>, input: IFut, execution: &'a ExecutionContext<'_>, ctx: &'a Self::Ctx) -> futures_util::future::BoxFuture<'a, Result<Self::Out, ThoError>>
    where Self: 'a, Self::In<'i>: Sync, Self::Out: Send, I: Borrow<Self::In<'i>> + Send + 'a, IFut: Future<Output = Result<I, ThoError>> + Send + 'a {
        Box::pin(execute_stateful(self, input, execution, ctx))
    }
}

    impl PrepareStage for MockStatefulStage {
        type S = usize;
        fn pre_run(&self, _: usize) -> BoxFuture<'_, Result<tokio::sync::MappedMutexGuard<'_, usize>, ThoError>> {
            async move { Ok(tokio::sync::MutexGuard::map(self.state().lock().await, |state| state)) }.boxed()
        }
}

impl StageWithState for MockStatefulStage {
        fn do_run_with_state<'f>(self: Arc<Self>, input: &Self::In<'f>, state: &mut usize, ctx: &MockRunCtx) -> Result<TestOutput, ThoError> {
            MockStatefulStage::execute(&self, input, state, ctx)
        }
}

    pub struct MockStatefulStageBuilder {
        pub state: Arc<tokio::sync::Mutex<usize>>,
        pub panics: bool,
        pub result: Result<TestOutput, ThoError>,
    }

    impl MockStatefulStageBuilder {
        pub fn build(self) -> Arc<MockStatefulStage> {
            let mut stage = MockStatefulStage::new();
            stage.expect_state().return_const(self.state);
            stage.expect_execute().times(1).return_once(move |_, state, _| {
                // Mutating before failure models a submission that has already
                // begun using state when it errors or unwinds.
                *state += 1;
                if self.panics { std::panic::panic_any("stateful submission panic"); }
                self.result
            });
            Arc::new(stage)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_submission_panic_is_not_retried_as_a_synchronization_oom() {
        // Synchronization still runs, but its OOM cannot make an arbitrary
        // partially completed host operation safe to replay.
        let mut synchronized = false;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_cuda_stage::<()>(|| std::panic::panic_any("submission panic"), || {
                synchronized = true;
                Err(ThoError::oom())
            })
        }));
        assert!(synchronized);
        assert_eq!(result.unwrap_err().downcast_ref::<&str>(), Some(&"submission panic"));
    }

    struct MockLocalCudaExecutionBuilder {
        sync_result: Result<(), ThoError>,
    }

    impl MockLocalCudaExecutionBuilder {
        fn build(self) -> Box<dyn LocalCudaExecution> {
            let mut ctx = MockHandleCtx::new();
            ctx.expect_sync_ctx().times(1).return_once(|| Ok(Box::new(())));
            ctx.expect_on_after_run().times(1).return_once(move || self.sync_result);
            let mut actor = MockLocalCudaExecution::new();
            actor.expect_init_handle_ctx().times(1).return_once(move || Ok(Box::new(ctx)));
            Box::new(actor)
        }
    }

    #[tokio::test]
    async fn test_state_guard_survives_synchronization_on_success_error_and_panic() {
        use super::mock_stateful_stage::MockStatefulStageBuilder;
        use crate::runner::common::cachable::DevicesWrap;
        use futures_util::FutureExt;
        use thoosa_shared::traits::RangeSelector;
        let _global = crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        for selected in [false, true] {
            for outcome in 0..3 {
                for sync_fails in [false, true] {
                    let state = Arc::new(tokio::sync::Mutex::new(0));
                    let stage = MockStatefulStageBuilder {
                        state: Arc::clone(&state), panics: outcome == 2,
                        result: if outcome == 1 { Err(ThoError::oom()) } else { Ok(TestOutput::new(7.)) },
                    }.build();
                    let mut handle = MockHandleCtx::new();
                    handle.expect_sync_ctx().times(1).return_once(|| Ok(Box::new(())));
                    let during_sync = Arc::clone(&state);
                    handle.expect_on_after_run().times(1).return_once(move || {
                        // A competing exporter/trainer must remain unable to
                        // acquire the real mutex until synchronization returns.
                        assert!(during_sync.try_lock().is_err(), "state unlocked before stream synchronization");
                        if sync_fails { Err(fatal_consume_error()) } else { Ok(()) }
                    });
                    let mut actor = MockLocalCudaExecution::new();
                    actor.expect_init_handle_ctx().times(1).return_once(move || Ok(Box::new(handle)));
                    let actor: Box<dyn LocalCudaExecution> = Box::new(actor);
                    let input = Arc::new(DevicesWrap::wrap_cpu(TestInput::new(1)));
                    let ctx = crate::runner::common::mock_run_ctx::build().with_device_idx(0).finish();
                    let result = std::panic::AssertUnwindSafe(async {
                        if selected {
                            ExecutionContext::Local(actor.as_ref()).run_stage_with_select(stage, input, None::<RangeSelector>, 0, &ctx).await
                        } else {
                            ExecutionContext::Local(actor.as_ref()).run_stage(stage, input, 0, &ctx).await
                        }
                    }).catch_unwind().await;
                    assert_eq!(*state.try_lock().expect("state must unlock after synchronization"), 1);
                    if sync_fails {
                        assert!(result.unwrap().unwrap_err().needs_cuda_recovery());
                    } else if outcome == 2 {
                        assert_eq!(result.unwrap_err().downcast_ref::<&str>(), Some(&"stateful submission panic"));
                    } else if outcome == 1 {
                        assert!(result.unwrap().unwrap_err().is_oom());
                    } else {
                        assert_eq!(result.unwrap().unwrap().v.0, 7.);
                    }
                }
            }
        }
    }

    use crate::runner::oom_handler::{mock_runnable::MockRunnable, test_types::{TestInput, TestOutput}};

    // Adapt the existing owned test data through the actual ConsumeData callback
    // contract, including a suspension during materialization.
    impl ConsumeData for TestInput {
        type O = Self;

        fn consume<R: Send, F, P>(self, _to_device_idx: usize, run_ctx: P, do_run: F) -> impl Future<Output=ConsumeResult<Self::O, R>> + Send
        where
            F: FnOnce(Self, P) -> ConsumeResult<Self, R> + Send,
            P: RunCtx,
        {
            async move {
                tokio::task::yield_now().await;
                do_run(self, run_ctx)
            }
        }

        fn burp(input: Self) -> impl Future<Output = Self> + Send {
            async move { input }
        }
    }

    mockall::mock! {
        pub StreamScope {
            fn release(&mut self);
        }
    }
    impl SyncCtx for MockStreamScope {}
    impl Drop for MockStreamScope {
        fn drop(&mut self) { self.release(); }
    }

    #[derive(Clone, Copy)]
    enum SetupFailure { Handle, Stream }

    struct MockConsumeCallBuilder {
        panics: bool,
        execution: Result<TestOutput, ThoError>,
        synchronization: Result<(), ThoError>,
        setup_failure: Option<SetupFailure>,
    }

    impl MockConsumeCallBuilder {
        fn build(self) -> (Arc<MockRunnable>, Box<dyn LocalCudaExecution>) {
            let mut sequence = mockall::Sequence::new();
            let mut actor = MockLocalCudaExecution::new();
            let init = actor.expect_init_handle_ctx().times(1).in_sequence(&mut sequence);
            let mut stage = MockRunnable::new();
            if matches!(self.setup_failure, Some(SetupFailure::Handle)) {
                init.return_once(|| Err(ThoError::oom()));
                stage.expect_consume().times(0);
                return (Arc::new(stage), Box::new(actor));
            }
            let mut ctx = MockHandleCtx::new();
            let install = ctx.expect_sync_ctx().times(1).in_sequence(&mut sequence);
            if matches!(self.setup_failure, Some(SetupFailure::Stream)) {
                install.return_once(|| Err(ThoError::oom()));
                stage.expect_consume().times(0);
            } else {
                // A separate liveness check verifies the guard survives synchronization.
                let alive = Arc::new(std::sync::atomic::AtomicBool::new(true));
                let dropped = Arc::clone(&alive);
                let mut guard = MockStreamScope::new();
                guard.expect_release().times(1).return_once(move || { dropped.store(false, std::sync::atomic::Ordering::SeqCst); });
                install.return_once(move || Ok(Box::new(guard)));
                stage.expect_consume().times(1).in_sequence(&mut sequence).return_once(move |input, _| {
                    if self.panics { std::panic::panic_any("consume panic"); }
                    match self.execution {
                        Ok(output) => ConsumeResult::Ok(output),
                        Err(error) => ConsumeResult::Failed { input, error },
                    }
                });
                ctx.expect_on_after_run().times(1).in_sequence(&mut sequence).return_once(move || {
                    assert!(alive.load(std::sync::atomic::Ordering::SeqCst), "stream guard must remain installed during synchronization");
                    self.synchronization
                });
            }
            init.return_once(move || Ok(Box::new(ctx)));
            (Arc::new(stage), Box::new(actor))
        }
    }

    fn fatal_consume_error() -> ThoError {
        // Use the constructor so injected CUDA errors capture their backtrace.
        thoosa_shared::tho_error::cuda_error(thoosa_shared::CudaAction::NeedsRestart, "consume stream illegal address".into())
    }

    #[tokio::test]
    async fn test_consume_panic_synchronizes_before_unwinding() {
        use futures_util::FutureExt;
        for fatal_sync in [false, true] {
            // Builder expectations require synchronization while the installed
            // stream guard is still alive, including the unwinding callback.
            let (stage, actor) = MockConsumeCallBuilder {
                panics: true, execution: Ok(TestOutput::new(7.)), setup_failure: None,
                synchronization: if fatal_sync { Err(fatal_consume_error()) } else { Ok(()) },
            }.build();
            let ctx = crate::runner::common::mock_run_ctx::build().finish();
            let result = std::panic::AssertUnwindSafe(ExecutionContext::Local(actor.as_ref()).consume_stage(stage, TestInput::new(1), 0, ctx)).catch_unwind().await;
            if fatal_sync { assert!(result.unwrap().unwrap_err().needs_cuda_recovery()); }
            else { assert_eq!(result.unwrap_err().downcast_ref::<&str>(), Some(&"consume panic")); }
        }
    }

    #[tokio::test]
    async fn test_failed_execution_stream_is_quarantined_and_wakes_waiters() -> Result<(), ThoError> {
        let context = ThoCudaContext::new(0)?;
        context.bind_to_thread()?;
        let pool = ExecutionStreamPool::new(vec![ThoCudaStream::from_pool(0, false)?, ThoCudaStream::from_pool(0, false)?]);
        let mut failed = pool.acquire().await?;
        let healthy = pool.acquire().await?;
        let mut waiter = Box::pin(pool.acquire());
        assert!(futures::poll!(waiter.as_mut()).is_pending());
        // Inject an uncertain completion result without poisoning real CUDA.
        assert!(failed.complete(Err(ThoError::from_cuda_error("injected synchronization failure", 700))).unwrap_err().needs_cuda_recovery());
        assert!(matches!(timeout(Duration::from_secs(3), waiter).await.unwrap(), Err(error) if error.needs_cuda_recovery()));
        drop(failed);
        assert_eq!(pool.quarantined.lock().len(), 1);
        assert_eq!(pool.pool.remaining_capacity(), 0, "failed stream returned to ordinary reuse");
        assert!(healthy.sync_ctx().is_err(), "a previously acquired stream must not begin new work in a poisoned pool");
        drop(healthy);
        assert!(pool.acquire().await.is_err(), "pool must stay closed even when healthy leases return");
        Ok(())
    }

    #[test]
    fn test_stream_drain_preserves_cuda_classification() {
        // Recoverable synchronization errors remain operation errors after a
        // successful drain; only an actual fatal code requests CUDA recovery.
        for code in [2, 1, 400] {
            let completion = complete_cuda_stream(Err(ThoError::from_cuda_error("first sync", code)), || Ok(()));
            assert!(completion.confirmed);
            let error = completion.result.unwrap_err();
            assert!(!error.needs_cuda_recovery() && !error.is_cuda_recovery_failed());
            assert_eq!(error.is_oom(), code == 2);
        }
        let completion = complete_cuda_stream(Err(ThoError::from_cuda_error("first sync", 700)), || panic!("fatal CUDA errors must not be retried"));
        assert!(!completion.confirmed && completion.result.unwrap_err().needs_cuda_recovery());
        let completion = complete_cuda_stream(Err(ThoError::oom()), || Err(ThoError::from_cuda_error("drain", 700)));
        assert!(!completion.confirmed && completion.result.unwrap_err().needs_cuda_recovery());
        let completion = complete_cuda_stream(Err(ThoError::oom()), || Err(ThoError::from_cuda_error("drain", 400)));
        assert!(!completion.confirmed);
        let error = completion.result.unwrap_err();
        assert!(!error.needs_cuda_recovery() && !error.is_retryable_oom());
        let ThoError::CudaRecoveryFailed(failure) = error.what() else { panic!("failed drain must retain its typed errors") };
        assert!(failure.original.is_oom());
        assert!(!failure.recovery.as_ref().unwrap().needs_cuda_recovery());
    }

    #[tokio::test]
    async fn test_nonfatal_execution_sync_error_keeps_stream_pool_usable() -> Result<(), ThoError> {
        let context = ThoCudaContext::new(0)?;
        context.bind_to_thread()?;
        let pool = ExecutionStreamPool::new(vec![ThoCudaStream::from_pool(0, false)?]);
        let mut stream = pool.acquire().await?;
        let mut waiter = Box::pin(pool.acquire());
        assert!(futures::poll!(waiter.as_mut()).is_pending());
        // The injected error takes the same second-drain path as a real error,
        // while CUDA itself stays healthy for subsequent execution.
        let error = stream.complete(Err(ThoError::oom())).unwrap_err();
        assert!(error.is_oom() && !error.needs_cuda_recovery());
        assert!(pool.check().is_ok() && pool.quarantined.lock().is_empty());
        drop(stream);
        let mut next = timeout(Duration::from_secs(3), waiter).await.unwrap()?;
        let _guard = next.sync_ctx()?;
        next.on_after_run()?;
        Ok(())
    }

    #[tokio::test]
    async fn test_consume_stream_lifecycle_and_error_precedence() {
        // Exercise the real actor entry point, rather than just the error-merging helper.
        for execution in [Ok(TestOutput::new(7.)), Err(ThoError::oom()), Err(fatal_consume_error())] {
            for synchronization in [Ok(()), Err(ThoError::oom()), Err(fatal_consume_error())] {
                let preserves_input = execution.is_err();
                let fatal = execution.as_ref().err().is_some_and(ThoError::needs_cuda_recovery) || synchronization.as_ref().err().is_some_and(ThoError::needs_cuda_recovery);
                let failed = execution.is_err() || synchronization.is_err();
                let (stage, actor) = MockConsumeCallBuilder { panics: false, execution: execution.clone(), synchronization, setup_failure: None }.build();
                let input = TestInput::new(3);
                let original_allocation = input.data.0.as_ptr();
                let ctx = crate::runner::common::mock_run_ctx::build().with_device_idx(1).finish();
                let result = ExecutionContext::Local(actor.as_ref()).consume_stage(stage, input, 1, ctx).await;
                let error = match result {
                    Ok(ConsumeResult::Failed { input, error }) => {
                        assert!(preserves_input);
                        assert_eq!(input.data.0.as_ptr(), original_allocation, "failure must return the exact owned input allocation");
                        assert_eq!(input.data.0, vec![0., 1., 2.]);
                        error
                    }
                    Ok(ConsumeResult::Ok(output)) => {
                        assert!(!failed);
                        assert_eq!(output.v.0, 7.);
                        continue;
                    }
                    Err(error) => {
                        assert!(!preserves_input, "a failed consume must retain its input even when synchronization fails");
                        error
                    }
                };
                assert!(failed);
                assert_eq!(error.needs_cuda_recovery(), fatal);
                assert_eq!(error.is_oom(), !fatal);
            }
        }
    }

    #[tokio::test]
    async fn test_consume_setup_failure_preserves_owned_input() {
        for setup_failure in [SetupFailure::Handle, SetupFailure::Stream] {
            let (stage, actor) = MockConsumeCallBuilder { panics: false, execution: Ok(TestOutput::new(7.)), synchronization: Ok(()), setup_failure: Some(setup_failure) }.build();
            let input = TestInput::new(3);
            let original_allocation = input.data.0.as_ptr();
            let ctx = crate::runner::common::mock_run_ctx::build().with_device_idx(1).finish();
            let result = ExecutionContext::Local(actor.as_ref()).consume_stage(stage, input, 1, ctx).await.unwrap();
            let ConsumeResult::Failed { input, error } = result else { panic!("setup failure must preserve the input") };
            assert_eq!(input.data.0.as_ptr(), original_allocation);
            assert!(error.is_oom());
        }
    }

    #[tokio::test]
    async fn test_stage_entry_points_preserve_fatal_execution_and_sync_errors() {
        use crate::runner::common::cachable::DevicesWrap;
        use crate::runner::oom_handler::{mock_runnable, test_types::{TestInput, TestOutput}};
        use thoosa_shared::traits::RangeSelector;
        let _ctx = crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder::install_global_ctx_mock(false, false);
        for selected in [false, true] {
            for fatal_sync in [false, true] {
                let fatal = || thoosa_shared::tho_error::cuda_error(thoosa_shared::CudaAction::NeedsRestart, "illegal address".into());
                let (execution, sync) = if fatal_sync { (Ok(TestOutput::new(1.)), Err(fatal())) } else { (Err(fatal()), Err(ThoError::oom())) };
                let (stage, _) = mock_runnable::build([execution]);
                let actor = MockLocalCudaExecutionBuilder { sync_result: sync }.build();
                let input = Arc::new(DevicesWrap::wrap_cpu(TestInput::new(1)));
                let ctx = crate::runner::common::mock_run_ctx::build().with_device_idx(0).finish();
                let result = if selected {
                    ExecutionContext::Local(actor.as_ref()).run_stage_with_select(Arc::new(stage), input, None::<RangeSelector>, 0, &ctx).await
                } else {
                    ExecutionContext::Local(actor.as_ref()).run_stage(Arc::new(stage), input, 0, &ctx).await
                };
                assert!(result.unwrap_err().needs_cuda_recovery(), "both entry points must surface the fatal error");
            }
        }
    }

    #[tokio::test]
    async fn test_cancelled_local_copy_keeps_activity_in_queued_request() {
        let gate = Arc::new(CudaActivityGate::new(1));
        let (sender, mut receiver) = mpsc::channel(1);
        let handle = CudaLocalActorHandle { gate: Arc::clone(&gate), sender };
        let src = ThoTensor::new((), Tensor::f_from_slice(&[1f32]).unwrap()).unwrap();
        let mut copy = Box::pin(handle.copy(src, 1usize.into()));
        assert!(futures::poll!(copy.as_mut()).is_pending());
        let request = receiver.recv().await.unwrap();
        drop(copy);
        let pause = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        let mut idle = Box::pin(gate.wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending());
        let CudaLocalActorMessage::Copy { activity, .. } = request else { panic!("expected copy") };
        // Actor dispatch must admit this queued operation during draining.
        activity.check_operation(&gate).unwrap();
        activity.run(async { CudaActivityScope::for_operation(&gate).unwrap(); }).await;
        timeout(Duration::from_secs(2), idle).await.unwrap();
        gate.finish_pause(pause);
    }

    #[tokio::test]
    async fn test_cancelled_stream_acquisition_keeps_activity_in_request() {
        let gate = Arc::new(CudaActivityGate::new(1));
        let (sender, mut receiver) = mpsc::channel(1);
        let handle = CudaLocalActorHandle { gate: Arc::clone(&gate), sender };
        let mut acquire = Box::pin(handle.init_handle_ctx());
        assert!(futures::poll!(acquire.as_mut()).is_pending());
        let request = receiver.recv().await.unwrap();
        drop(acquire);
        let pause = gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        let mut idle = Box::pin(gate.wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending());
        drop(request);
        timeout(Duration::from_secs(2), idle).await.unwrap();
        gate.finish_pause(pause);
    }

    #[tokio::test]
    async fn test_local_copy_admission_is_device_local() {
        let failed_gate = Arc::new(CudaActivityGate::new(1));
        let healthy_gate = Arc::new(CudaActivityGate::new(2));
        let (failed_tx, mut failed_rx) = mpsc::channel(1);
        let (healthy_tx, mut healthy_rx) = mpsc::channel(1);
        let failed = CudaLocalActorHandle { gate: Arc::clone(&failed_gate), sender: failed_tx };
        let healthy = CudaLocalActorHandle { gate: healthy_gate, sender: healthy_tx };
        let pause = failed_gate.request_pause(ThoError::CudaBeingRecovered).unwrap();
        for terminal in [false, true] {
            if terminal { failed_gate.fail_pause(pause, None); }
            let src = || ThoTensor::new((), Tensor::f_from_slice(&[1f32]).unwrap()).unwrap();
            let error = failed.copy(src(), 1usize.into()).await.unwrap_err();
            assert_eq!(error.is_cuda_recovery_failed(), terminal);
            assert!(failed_rx.try_recv().is_err());
            // Exercise the actual handle's admission and enqueue without a GPU.
            // A different device must still be able to publish its copy request.
            let mut copy = Box::pin(healthy.copy(src(), 2usize.into()));
            assert!(futures::poll!(copy.as_mut()).is_pending());
            assert!(matches!(healthy_rx.try_recv().unwrap(), CudaLocalActorMessage::Copy { .. }));
        }
    }
}

#[async_trait]
impl LocalCudaExecution for CudaLocalActorHandle {
    async fn copy_batch_to_cpu(&self, sources: Vec<Tensor>) -> Result<Vec<Tensor>, ThoError> {
        let activity = CudaActivityScope::for_operation(&self.gate)?;
        let (respond_to, receive) = oneshot::channel();
        self.sender.send(CudaLocalActorMessage::CopyBatchToCpu { sources: parking_lot::Mutex::new(sources), activity, respond_to }).await.context("batch copy: CUDA actor channel closed")?;
        receive.await.context("batch copy: CUDA actor response channel closed")?
    }
    async fn init_handle_ctx(&self) -> Result<Box<dyn HandleCtx>, ThoError> {

        Ok(Box::new(self.acquire_exec_stream().await?))
    }
}
