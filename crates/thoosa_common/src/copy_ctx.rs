use crate::cuda_utils::{PinnedHostBuffer, copy_d2h, copy_d2h_from_ptr, copy_d2h_to_ptr, copy_h2d, copy_h2d_from_ptr, cuda_mem_for_device};
use crate::dims::Dims;
use crate::runner::common::cachable::{BorrowRef, ConsumeData, SelectRef};
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
use anyhow::Context as _;
use async_trait::async_trait;
use cust::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
use futures_util::future::BoxFuture;
use log::{debug, info, trace};
use std::any::Any;
use std::backtrace::Backtrace;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::io::{Seek, SeekFrom, Write};
use std::sync::Arc;
use std::sync::atomic::Ordering::{AcqRel, Acquire};
use std::sync::atomic::AtomicU64;
use std::time::Duration;
use tch::Kind::Int64;
use tch::{Kind, Tensor};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::ConsumeResult;
use thoosa_shared::traits::{Named, ThoSelector};
use thoosa_shared::traits::{RunCtx, Stage, SyncStage};
use thoosa_shared::{CudaActivityGate, CudaActivityScope, CudaDeviceGeneration, ThoError};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tokio::sync::oneshot::Sender;
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::time::timeout;
use tracing::error;
use tub::{Guard, Pool};
use crate::ipc::worker_msg::{CudaWorkerMsgOut, OpResult};

static NEXT_IPC_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

enum CudaLocalActorMessage {
    Copy{ src:ThoTensor, dst_device:ThoDevice, activity: CudaActivityScope, respond_to: oneshot::Sender<Result<Tensor, ThoError>>},

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
pub trait CudaActorHandle: Any/*TODO remove when is_ipc replaced with Enum*/ + Debug + Send + Sync {
    async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) -> Result<ThoTensor, ThoError>;

    #[cfg(test)]
    async fn restart(&self) -> Result<(), ThoError>;

    // Backend work is called by GlobalCtx's coordinator hook after draining.
    // It must neither request a new cycle nor change admission state.
    async fn restart_backend(&self, authority: thoosa_shared::CudaActivityPauseTicket) -> Result<(), ThoError>;

    async fn retire_backend(&self, _authority: thoosa_shared::CudaActivityPauseTicket) -> Result<(), ThoError> {
        Err(ThoError::new("this CUDA backend cannot terminate active work"))
    }

    async fn shutdown(&self) -> Result<(), ThoError> {
        Ok(())
    }

    async fn purge_stage_state(&self, stage_id: u32, endpoint_name: String, worker_id: crate::ipc::worker_msg::CudaWorkerId) -> Result<CudaWorkerMsgOut, ThoError> {
        unreachable!()
    }

    async fn drop_endpoint(&self, _stage_id: u32, _endpoint_name: String, _worker_id: crate::ipc::worker_msg::CudaWorkerId) -> Result<CudaWorkerMsgOut, ThoError> {
        unreachable!()
    }

    async fn drop_tho_tensor(&self, key:u64, worker_id: crate::ipc::worker_msg::CudaWorkerId)-> Result<(), ThoError>
    {
        Ok(())
    }


    async fn init_handle_ctx(&self) ->Result<Box<dyn HandleCtx>, ThoError>;

    async fn debug_worker(&self) -> Result<CudaWorkerMsgOut, ThoError>;
}

impl dyn CudaActorHandle + '_ {

    pub async fn consume_stage<'z, S, W>(
        &self,
        stage: Arc<S>,
        input: W,
        to_device_idx: usize,
        ctx: S::Ctx,
    ) -> ConsumeResult<S::In<'z>, S::Out>
    where
        S: SyncStage + Send + Sync + 'static,
        S::Out: Send,
        W: ConsumeData<O = S::In<'z>> + Debug,
        S::In<'z>: 'static,
    {
        debug!("CudaActorHandle::consume_stage: to_device_idx={to_device_idx} input={input:?}");
        input.consume(to_device_idx, ctx, None, |input, ctx| stage.consume(input, ctx)).await
    }


    /// invoked from oom_executor, dump_to_cpu can't happen between pre_run, borrow_to and do_run  awaits
    pub async fn run_stage<S, W>(
        &self,
        stage: Arc<S>,
        input: W,
        to_device_idx: usize,
        ctx:&S::Ctx
    ) -> Result<S::Out, ThoError>
    where
        S: SyncStage,
        W: Debug + Sync,
        for<'a> W: BorrowRef + 'a,
        for<'a> <W as BorrowRef>::Out<'a>: Borrow<S::In<'a>>,
    {
        debug!("CudaActorHandle::run_stage: to_device_idx={to_device_idx} input={input:?}");

        let state = if let Some(with_state) = stage.as_with_state() {
            // Select stateful execution before preparation so only stateful stages run pre_run.
            Some(with_state.pre_run(to_device_idx).await?)
        } else {
            None
        };

        debug!("CudaActorHandle::run_stage: stage={stage:?} with_state={}", state.is_some());

        let borrowed = BorrowRef::borrow_ref(&input, to_device_idx).await?;

        let mut handle_ctx = self.init_handle_ctx().await?;

        let sync_ctx = handle_ctx.sync_ctx()?;

        let out =  match state {
            None => {
                Arc::clone(&stage).do_run(borrowed.borrow(), ctx)
            }
            Some(state) => {
                state.do_run_with_state(Arc::clone(&stage), borrowed.borrow(), ctx)
            }
        };


        finish_cuda_run(out, handle_ctx.on_after_run())
    }

    pub async fn run_stage_with_select<S, W, SL>(
        &self,
        stage: Arc<S>,
        input: W,
        selector: Option<SL>,
        to_device_idx: usize,
        ctx:&S::Ctx
    ) -> Result<S::Out, ThoError>
    where
        S: SyncStage,
        SL: ThoSelector + Clone + Debug + Send + 'static,
        W: Debug + Sync,
        for<'a> W: SelectRef<SL> + 'a,
        for<'a> <W as BorrowRef>::Out<'a>: Borrow<S::In<'a>> + Debug,
    {
        debug!("CudaActorHandle::run_stage_with_select: to_device_idx={to_device_idx} input={input:?} selector={selector:?}");

        let state = if let Some(with_state) = stage.as_with_state() {
            // Keep selector-based execution consistent with the primary stateful runner path.
            Some(with_state.pre_run(to_device_idx).await?)
        } else {
            None
        };
        let has_selector = selector.is_some();
        let borrowed = SelectRef::select_ref(&input, to_device_idx, selector).await?;

        debug!("CudaActorHandle::run_stage_with_select: to_device_idx={to_device_idx} borrowed={borrowed:?}");

        let mut handle_ctx = self.init_handle_ctx().await?;

        let sync_ctx = handle_ctx.sync_ctx()?;

        let out = match state {
            Some(state) => state.do_run_with_state(Arc::clone(&stage), borrowed.borrow(), ctx),
            None => Arc::clone(&stage).do_run(borrowed.borrow(), ctx),
        };
        debug!("CudaActorHandle::run_stage_with_select: to_device_idx={to_device_idx} out={out:?}");

        let sync_result  = handle_ctx.on_after_run();
        drop(sync_ctx);
        drop(borrowed);

        debug!("CudaActorHandle::run_stage_with_select: to_device_idx={to_device_idx} sync_result={sync_result:?}");


        finish_cuda_run(out, sync_result)
    }

}

// Both stage entry points must retain a fatal execution error even if stream
// synchronization also fails, and must propagate a fatal synchronization error.
fn finish_cuda_run<T>(result: Result<T, ThoError>, sync: Result<(), ThoError>) -> Result<T, ThoError> {
    match (result, sync) {
        (Err(original), Err(sync)) => {
            if sync.is_cuda_recovery_failed() || (!original.is_cuda_recovery_failed() && sync.needs_cuda_ctx_reset() && !original.needs_cuda_ctx_reset()) {
                Err(sync.context(format!("stage also failed: {original}")))
            } else {
                Err(original.context(format!("stream synchronization also failed: {sync}")))
            }
        }
        (Err(error), _) | (_, Err(error)) => Err(error),
        (Ok(output), Ok(())) => Ok(output),
    }
}

// Report in the actor-owned task, including when the awaiting caller disappeared.
// The captured lease generation prevents a late copy error restarting a successor.
pub(crate) fn report_cuda_copy_result<T>(result: Result<T, ThoError>, gate: &Arc<CudaActivityGate>, generation: CudaDeviceGeneration) -> Result<T, ThoError> {
    result.map_err(|error| {
        if error.needs_cuda_ctx_reset() {
            GlobalCtx::global().get_runner(gate.device_idx()).request_cuda_reset_for_generation(generation, error)
                .err().unwrap_or(ThoError::ContextReset)
        } else { error }
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
                    let src_ptr = src.t_ref().data_ptr() as usize;
                    let src_size = src.t_ref().size();
                    let src_kind = src.t_ref().kind();
                    let _src_keepalive = src;

                    let result = copy_ctx.read().await.copy_from_ptr(
                        src_ptr,
                        &src_size,
                        dst_device.tho_device_idx(),
                        src_kind,
                    ).await;

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
                    let stream = copy_ctx.read().await.acquire_exec_stream().await;
                    respond_to.send(Ok(LocalHandleCtx { stream, _activity: activity }))
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
            let mut cu_ctx = ThoCudaContext::new(device.gpu_ordinal() as i32)?;
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
    tho_device_idx: usize,
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
            tho_device_idx: tho_device_id,
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
        ).await.context("CUDA actor channel closed")?;

        Ok(recv.await.context("CUDA actor response channel closed")??)
    }

    async fn restart_inner(&self, authority: thoosa_shared::CudaActivityPauseTicket) -> anyhow::Result<()> {
        self.gate.check_recovery_authority(authority, true)?;
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaLocalActorMessage::Restart {
                authority,
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        Ok(recv.await.context("CUDA actor response channel closed")??)
    }

    async fn acquire_exec_stream(&self) -> anyhow::Result<LocalHandleCtx> {
        let activity = CudaActivityScope::for_operation(&self.gate)?;
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaLocalActorMessage::GetExecStream {
                activity,
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        Ok(recv.await.context("CUDA actor response channel closed")??)
    }

}

struct LocalHandleCtx {
    stream: Guard<ThoCudaStream>,
    _activity: CudaActivityScope,
}

impl SyncCtx for ThoStreamGuard {}
#[async_trait]
impl HandleCtx for LocalHandleCtx {

    fn sync_ctx(&self) -> Result<Box<dyn SyncCtx>, ThoError> {
        Ok(Box::new(ThoStreamGuard::new(&self.stream)?))
    }

    fn on_after_run(&mut self) -> Result<(), ThoError> {
        self.stream.synchronize()
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
        GlobalCtx::global().restart_cuda_device(self.tho_device_idx).await
    }

    async fn restart_backend(&self, authority: thoosa_shared::CudaActivityPauseTicket) -> Result<(), ThoError> {
        Ok(self.restart_inner(authority).await?)
    }


    async fn init_handle_ctx(&self) -> Result<Box<dyn HandleCtx>, ThoError> {

        Ok(Box::new(self.acquire_exec_stream().await?))
    }

    async fn debug_worker(&self) -> Result<CudaWorkerMsgOut, ThoError> {
       Ok(CudaWorkerMsgOut{ worker_id: crate::ipc::worker_msg::CudaWorkerId { device_idx: self.tho_device_idx, generation: self.gate.device_generation() }, request_id: 0, status: OpResult::Ok(None) })
    }
}


pub struct CopyCtx {
    gate: Arc<CudaActivityGate>,
    cuda_device: usize,

    pub(crate) buffs: usize,
    pub(crate) buf_size: usize,
    exec_stream_pool: Arc<Pool<ThoCudaStream>>,
    bufs_pool_d2h: Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
    bufs_pool_h2d: Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>,
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
            let exec_streams = (0..32)
                .map(|_| ThoCudaStream::from_pool(cuda_device as i32, false))
                .collect::<Result<Vec<_>, _>>()?;

                Arc::new(Pool::from_vec(exec_streams))
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
    ) -> Result<Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>, ThoError> {
        let buffers = (0..num_buffs).map(|_| {
            Ok((
                PinnedHostBuffer::new(buf_size, flags)?,
                Arc::new(ThoCudaStream::from_pool(cuda_device as i32, true)?),//TODO check if streams do not repeat
            ))
        }).collect::<Result<Vec<_>, ThoError>>()?;
        Ok(Arc::new(Pool::from_vec(buffers)))
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
            Arc::new(Pool::from_vec(vec![(
                PinnedHostBuffer::new(1, 0).unwrap(),
                Arc::new(ThoCudaStream::from_pool(self.cuda_device as i32, true)?),
            )])),
        )));

        self.buffs = buffs;
        self.buf_size = Self::clc_buff_size(self.cuda_device, mem_k, buffs, cuda_mem_for_device(self.cuda_device).1);
        self.bufs_pool_h2d = Self::build_pool(buffs, self.buf_size, self.cuda_device, CU_MEMHOSTALLOC_WRITECOMBINED)?;
        Ok(())
    }

    pub async fn acquire_exec_stream(&self) -> Guard<ThoCudaStream> {
        track_await(self.exec_stream_pool.as_ref()
            .acquire())
            .await
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
        self.exec_stream_pool = Arc::new(Pool::from_vec(exec_streams));

        Ok(())
    }

    fn clc_buff_size(cuda_device: usize, mem_k: f64, buffs: usize, total_mem: usize) -> usize {

        let init_type = Int64;

        let max_elt_size = init_type.elt_size_in_bytes();
        let elts_in_mem = (total_mem as f64 * mem_k) as usize / max_elt_size; //123456; //25000; //123456; //

        debug!("CopyCtx::clc_buff_size: elts_in_mem={elts_in_mem} total_mem={total_mem} max_elt_size={max_elt_size}");

        let elts_in_buff = elts_in_mem / buffs;

        debug!("CopyCtx::clc_buff_size: elts_in_buff={elts_in_buff} buffs={buffs}");

        let buf_size = elts_in_buff * max_elt_size;
        buf_size
    }

    pub async fn copy<R: AsRef<Tensor>>(&self, src_t: &R, dst_tho_device: usize, kind: Kind) -> Result<Tensor, ThoError> {
        if dst_tho_device == 0 {
            track_await(copy_d2h(&self.gate, kind, src_t, self.buf_size, &self.bufs_pool_h2d)).await
        } else {
            track_await(copy_h2d(&self.gate, kind, src_t, self.buf_size, &self.bufs_pool_h2d, dst_tho_device)).await
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="copy_from_ptr"))]
    pub async fn copy_from_ptr(
        &self,
        src_ptr: usize,
        src_size: &[i64],
        dst_tho_device_idx: usize,
        kind: Kind,
    ) -> Result<Tensor, ThoError> {

        if dst_tho_device_idx == 0 {
            track_await(copy_d2h_from_ptr(
                &self.gate, kind, self.buf_size,
                &self.bufs_pool_d2h, src_ptr, src_size
            )).await

        } else {
            track_await(copy_h2d_from_ptr(
                &self.gate, kind,
                self.buf_size,
                &self.bufs_pool_h2d,
                dst_tho_device_idx,
                src_ptr,
                src_size,
            )).await
        }
    }

    /// Copies CUDA storage into an already allocated CPU range, which lets an IPC worker fill a tensor owned by the parent shared allocator.
    pub async fn copy_to_cpu_ptr(&self, src_ptr: usize, src_size: &[i64], kind: Kind, dst_ptr: usize) -> Result<(), ThoError> {
        track_await(copy_d2h_to_ptr(&self.gate, kind, self.buf_size, &self.bufs_pool_d2h, src_ptr, src_size, dst_ptr)).await
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockCudaActorHandleBuilder {
        sync_result: Result<(), ThoError>,
    }

    impl MockCudaActorHandleBuilder {
        fn build(self) -> Box<dyn CudaActorHandle> {
            let mut ctx = MockHandleCtx::new();
            ctx.expect_sync_ctx().times(1).return_once(|| Ok(Box::new(())));
            ctx.expect_on_after_run().times(1).return_once(move || self.sync_result);
            let mut actor = MockCudaActorHandle::new();
            actor.expect_init_handle_ctx().times(1).return_once(move || Ok(Box::new(ctx)));
            Box::new(actor)
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
                let fatal = || ThoError::Cuda { action: thoosa_shared::CudaAction::NeedsRestart, msg: "illegal address".into() };
                let (execution, sync) = if fatal_sync { (Ok(TestOutput::new(1.)), Err(fatal())) } else { (Err(fatal()), Err(ThoError::oom())) };
                let (stage, _) = mock_runnable::build([execution]);
                let actor = MockCudaActorHandleBuilder { sync_result: sync }.build();
                let input = Arc::new(DevicesWrap::wrap_cpu(TestInput::new(1)));
                let ctx = crate::runner::common::mock_run_ctx::build().with_device_idx(0).finish();
                let result = if selected {
                    actor.run_stage_with_select(Arc::new(stage), input, None::<RangeSelector>, 0, &ctx).await
                } else {
                    actor.run_stage(Arc::new(stage), input, 0, &ctx).await
                };
                assert!(result.unwrap_err().needs_cuda_ctx_reset(), "both entry points must surface the fatal error");
            }
        }
    }

    #[tokio::test]
    async fn test_cancelled_local_copy_keeps_activity_in_queued_request() {
        let gate = Arc::new(CudaActivityGate::new(1));
        let (sender, mut receiver) = mpsc::channel(1);
        let handle = CudaLocalActorHandle { gate: Arc::clone(&gate), tho_device_idx: 1, sender };
        let src = ThoTensor::new((), Tensor::f_from_slice(&[1f32]).unwrap()).unwrap();
        let mut copy = Box::pin(handle.copy(src, 1usize.into()));
        assert!(futures::poll!(copy.as_mut()).is_pending());
        let request = receiver.recv().await.unwrap();
        drop(copy);
        let pause = gate.request_pause(ThoError::ContextReset).unwrap();
        let mut idle = Box::pin(gate.wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending());
        let CudaLocalActorMessage::Copy { activity, .. } = request else { panic!("expected copy") };
        // Actor dispatch must admit this queued operation during draining.
        activity.check_operation(&gate).unwrap();
        activity.run(async { CudaActivityScope::for_operation(&gate).unwrap(); }).await;
        tokio::time::timeout(Duration::from_secs(2), idle).await.unwrap();
        gate.finish_pause(pause);
    }

    #[tokio::test]
    async fn test_cancelled_stream_acquisition_keeps_activity_in_request() {
        let gate = Arc::new(CudaActivityGate::new(1));
        let (sender, mut receiver) = mpsc::channel(1);
        let handle = CudaLocalActorHandle { gate: Arc::clone(&gate), tho_device_idx: 1, sender };
        let mut acquire = Box::pin(handle.init_handle_ctx());
        assert!(futures::poll!(acquire.as_mut()).is_pending());
        let request = receiver.recv().await.unwrap();
        drop(acquire);
        let pause = gate.request_pause(ThoError::ContextReset).unwrap();
        let mut idle = Box::pin(gate.wait_until_idle());
        assert!(futures::poll!(idle.as_mut()).is_pending());
        drop(request);
        tokio::time::timeout(Duration::from_secs(2), idle).await.unwrap();
        gate.finish_pause(pause);
    }

    #[tokio::test]
    async fn test_local_copy_admission_is_device_local() {
        let failed_gate = Arc::new(CudaActivityGate::new(1));
        let healthy_gate = Arc::new(CudaActivityGate::new(2));
        let (failed_tx, mut failed_rx) = mpsc::channel(1);
        let (healthy_tx, mut healthy_rx) = mpsc::channel(1);
        let failed = CudaLocalActorHandle { gate: Arc::clone(&failed_gate), tho_device_idx: 1, sender: failed_tx };
        let healthy = CudaLocalActorHandle { gate: healthy_gate, tho_device_idx: 2, sender: healthy_tx };
        let pause = failed_gate.request_pause(ThoError::ContextReset).unwrap();
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
