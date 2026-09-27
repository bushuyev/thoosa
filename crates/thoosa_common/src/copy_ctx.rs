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
use thoosa_shared::{ThoError, cuda_activity_gate};
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use tokio::sync::oneshot::Sender;
use tokio::sync::{RwLock, mpsc, oneshot};
use tokio::time::timeout;
use tracing::error;
use tub::{Guard, Pool};
use crate::ipc::worker_msg::{CudaWorkerMsgOut, OpResult};

static NEXT_IPC_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

pub enum CudaLocalActorMessage {
    Copy{ src:ThoTensor, dst_device:ThoDevice, respond_to: oneshot::Sender<Result<Tensor, ThoError>>},

    GetExecStream {
        respond_to: oneshot::Sender<Result<Guard<ThoCudaStream>, ThoError>>
    },
    Restart {
        respond_to: oneshot::Sender<Result<(), ThoError>>
    }

}

pub trait SyncCtx {}

impl SyncCtx for (){}

#[async_trait]
pub trait HandleCtx:Send {

    fn sync_ctx(&self) ->Result<Box<dyn SyncCtx>, ThoError> {
        Ok(Box::new(()))
    }

     fn on_after_run(&mut self) ->Result<(), ThoError> {
         Ok(())
     }

}


#[async_trait]
pub trait CudaActorHandle: Any/*TODO remove when is_ipc replaced with Enum*/ + Debug + Send + Sync {
    async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) -> Result<ThoTensor, ThoError>;

    async fn restart(&self) -> Result<(), ThoError>;

    // Backend work is called by GlobalCtx's coordinator hook after draining.
    // It must neither request a new cycle nor change admission state.
    async fn restart_backend(&self) -> Result<(), ThoError>;

    async fn shutdown(&self) -> Result<(), ThoError> {
        Ok(())
    }

    async fn purge_stage_state(&self, stage_id: u32, endpoint_name: String) -> Result<CudaWorkerMsgOut, ThoError> {
        unreachable!()
    }

    async fn drop_endpoint(&self, _stage_id: u32, _endpoint_name: String) -> Result<CudaWorkerMsgOut, ThoError> {
        unreachable!()
    }

    async fn drop_tho_tensor(&self, key:u64)-> Result<(), ThoError>
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


        handle_ctx.on_after_run()?;

        out
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


        out
    }

}

struct CudaLocalActor {
    tho_device_idx:usize,
    receiver: mpsc::Receiver<CudaLocalActorMessage>,
    copy_ctx:Arc<RwLock<CopyCtx>>,
}

impl CudaLocalActor {
    pub fn new(tho_device_idx:usize, receiver: mpsc::Receiver<CudaLocalActorMessage>) -> Result<Self, ThoError> {
        debug!("CudaActor::new: tho_device_idx={tho_device_idx}");
        assert!(tho_device_idx > 0);

        Ok(Self {
            tho_device_idx,
            receiver,
            copy_ctx: Arc::new(RwLock::new(CopyCtx::make_instance_for_cuda_device(tho_device_idx - 1, 0.2, 32)?)),
        })
    }

     fn handle_message(&mut self, msg: CudaLocalActorMessage) {

        match msg {
            CudaLocalActorMessage::Copy{ src, dst_device, respond_to } => {
                if let Err(error) = cuda_activity_gate().check_admission() {
                    let _ = respond_to.send(
                        Err(error)
                    );
                    return;
                }

                let copy_ctx = Arc::clone(&self.copy_ctx);
                let tho_device_idx = self.tho_device_idx;
                let generation = cuda_activity_gate().device_generation(tho_device_idx);
                tokio::spawn(async move {
                    debug!("CudaActor::handle_message/Copy: src={src:?}");
                    let src_ptr = src.t_ref().data_ptr() as usize;
                    let src_size = src.t_ref().size();
                    let src_kind = src.t_ref().kind();
                    let _src_keepalive = src;

                    let mut result = copy_ctx.read().await.copy_from_ptr(
                        src_ptr,
                        &src_size,
                        dst_device.tho_device_idx(),
                        src_kind,
                    ).await;

                    if let Err(error) = &result && error.needs_cuda_ctx_reset() {
                        // Report synchronously and release the caller. Recovery is
                        // independent and may need cache locks held by that caller.
                        if let Err(error) = GlobalCtx::global().get_executor(tho_device_idx)
                            .request_cuda_reset_for_generation(generation, error.clone()) {
                            result = Err(error);
                        }
                    }

                    let _ = respond_to.send(
                        result
                     );
                 });
            }

            CudaLocalActorMessage::Restart { respond_to } => {
                let copy_ctx = Arc::clone(&self.copy_ctx);
                let tho_device_idx = self.tho_device_idx;
                tokio::spawn(async move {
                    let result = Self::do_restart(copy_ctx, tho_device_idx).await;
                    let _ = respond_to.send(result);
                });
            }
            CudaLocalActorMessage::GetExecStream { respond_to } => {
                if let Err(error) = cuda_activity_gate().check_admission() {
                    let _ = respond_to.send(
                        Err(error)
                    );
                    return;
                }
                let copy_ctx = Arc::clone(&self.copy_ctx);
                tokio::spawn(async move {
                    respond_to.send(Ok(copy_ctx.read().await.acquire_exec_stream().await))
                });
            }
        }
    }

    async fn do_restart(copy_ctx: Arc<RwLock<CopyCtx>>, tho_device_idx: usize) -> Result<(), ThoError> {
        // Cache cleanup and admission transitions belong to the coordinator.
        let device: ThoDevice = tho_device_idx.into();
        let mut cu_ctx = ThoCudaContext::new(device.gpu_ordinal() as i32)?;
        cu_ctx.restart()?;
        timeout(Duration::from_millis(1000), copy_ctx.write()).await?.restart()
    }

    async fn run(mut actor: CudaLocalActor) {
        while let Some(msg) = actor.receiver.recv().await {
            actor.handle_message(msg);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaLocalActorHandle {
    tho_device_idx: usize,
    sender: mpsc::Sender<CudaLocalActorMessage>,
}


impl CudaLocalActorHandle {
    pub fn new(tho_device_id:usize) -> Result<Self, ThoError> {
        debug!("CudaLocalActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let actor = CudaLocalActor::new(tho_device_id, receiver)?;
        tokio::spawn(CudaLocalActor::run(actor));

        Ok(Self {
            tho_device_idx: tho_device_id,
            sender,
        })
    }

    async fn copy_tensor(&self, src:ThoTensor, dst_device:ThoDevice) -> anyhow::Result<Tensor> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaLocalActorMessage::Copy {
                src,
                dst_device,
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        Ok(recv.await.context("CUDA actor response channel closed")??)
    }

    async fn restart_inner(&self) -> anyhow::Result<()> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaLocalActorMessage::Restart {
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        Ok(recv.await.context("CUDA actor response channel closed")??)
    }

    //TODO these are 'exec' streams, not for copy
    async fn acquire_exec_stream(&self) -> anyhow::Result<Guard<ThoCudaStream>> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaLocalActorMessage::GetExecStream {
                respond_to,
            }
        ).await.context("CUDA actor channel closed")?;

        Ok(recv.await.context("CUDA actor response channel closed")??)
    }

}

struct LocalHandleCtx {
    stream: Guard<ThoCudaStream>,
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
        Ok(thoosa_shared::CudaActivityScope::for_operation()?.run(self.copy_tensor(src, dst_device)).await?.into())
    }

    async fn restart(&self) -> Result<(), ThoError> {
        GlobalCtx::global().restart_cuda_device(self.tho_device_idx).await
    }

    async fn restart_backend(&self) -> Result<(), ThoError> {
        Ok(self.restart_inner().await?)
    }


    async fn init_handle_ctx(&self) -> Result<Box<dyn HandleCtx>, ThoError> {

        let stream = self.acquire_exec_stream().await?;


        Ok(Box::new(LocalHandleCtx { stream }))
    }

    async fn debug_worker(&self) -> Result<CudaWorkerMsgOut, ThoError> {
       Ok(CudaWorkerMsgOut{ request_id: 0, status: OpResult::Ok(None) })
    }
}


pub struct CopyCtx {
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

        debug!("make_instance_for_cuda_device: cuda_device={cuda_device} mem_k={mem_k} buffs={buffs}");


        let buf_size = Self::clc_buff_size(cuda_device, mem_k, buffs, cuda_mem_for_device(cuda_device).1);

        let exec_stream_pool = {
            let exec_streams = (0..32)
                .map(|_| ThoCudaStream::from_pool(cuda_device as i32, false))
                .collect::<Result<Vec<_>, _>>()?;

                Arc::new(Pool::from_vec(exec_streams))
        };

        Ok(Self {
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
        info!("CopyCtx::reset_cuda_context: about to reset cu_ctx.reset");

        let cuda_context = ThoCudaContext::new(self.cuda_device as i32)?;
        cuda_context.bind_to_thread()?;


        info!("CopyCtx::reset_cuda_context: cu_ctx.reset done");

        self.bufs_pool_h2d = Self::build_pool(self.buffs, self.buf_size, self.cuda_device, CU_MEMHOSTALLOC_WRITECOMBINED)?;
        info!("CopyCtx::reset_cuda_context: bufs_pool rebuilt");

        info!("CopyCtx::reset_cuda_context: ipc_bufs_pool rebuilt");

        let exec_streams = (0..32)
            .map(|_| ThoCudaStream::from_pool(self.cuda_device as i32, false))
            .collect::<Result<Vec<_>, _>>()?;
        self.exec_stream_pool = Arc::new(Pool::from_vec(exec_streams));

        info!("CopyCtx::reset_cuda_context: exec_stream_pool rebuilt");

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
            track_await(copy_d2h(kind, src_t, self.buf_size, &self.bufs_pool_h2d)).await
        } else {
            track_await(copy_h2d(kind, src_t, self.buf_size, &self.bufs_pool_h2d, dst_tho_device)).await
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
        let t = if dst_tho_device_idx == 0 {
            //TODO opt to .to if tensor_mem_size < buf_size {
            track_await(copy_d2h_from_ptr(
                kind, self.buf_size,
                &self.bufs_pool_d2h, src_ptr, src_size
            )).await
        } else {
            track_await(copy_h2d_from_ptr(
                kind,
                self.buf_size,
                &self.bufs_pool_h2d,
                dst_tho_device_idx,
                src_ptr,
                src_size,
            )).await
        };

        // println!("copy_from_ptr: copied t.size={:?} dst_device={dst_device} t=\n{t}", t.size());

        t
    }

    /// Copies CUDA storage into an already allocated CPU range, which lets an IPC worker fill a tensor owned by the parent shared allocator.
    pub async fn copy_to_cpu_ptr(&self, src_ptr: usize, src_size: &[i64], kind: Kind, dst_ptr: usize) -> Result<(), ThoError> {
        track_await(copy_d2h_to_ptr(kind, self.buf_size, &self.bufs_pool_d2h, src_ptr, src_size, dst_ptr)).await
    }

}
