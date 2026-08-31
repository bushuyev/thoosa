use crate::cuda_utils::{PinnedHostBuffer, copy_d2h, copy_d2h_from_ptr, copy_d2h_to_ptr, copy_h2d, copy_h2d_from_ptr, cuda_mem_for_device, move_d2h, move_h2d};
use crate::dims::Dims;
use crate::ipc::ipc_cuda_utils::SharedMemfdBuffer;
use crate::runner::common::cachable::{BorrowRef, ConsumeData, SelectRef};
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
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
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};
use std::sync::atomic::{AtomicBool, AtomicU64};
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
use crate::ipc::worker_msg::CudaWorkerMsgOut;

static NEXT_IPC_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

pub enum CudaActorMessage {
    Copy{ src:ThoTensor, dst_device:ThoDevice, respond_to: oneshot::Sender<Result<Tensor, ThoError>>},
    SyncStream {
        stream:Guard<ThoCudaStream>,
        respond_to: oneshot::Sender<Result<(), ThoError>>
    },
    GetExecStream {
        respond_to: oneshot::Sender<Result<Guard<ThoCudaStream>, ThoError>>
    },
    Reset {
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

    async fn reset(&self) -> Result<(), ThoError>;

    async fn shutdown(&self) -> Result<(), ThoError> {
        Ok(())
    }

    async fn purge_stage_state(&self, stage_id: u32, endpoint_name: String) -> Result<CudaWorkerMsgOut, ThoError> {
        todo!()//TODO purge local states too
    }

    async fn drop_endpoint(&self, _stage_id: u32, _endpoint_name: String) -> Result<CudaWorkerMsgOut, ThoError> {
        todo!()
    }

    async fn drop_tho_tensor(&self, t:ThoTensor)-> Result<(), ThoError>
    {
        Ok(())
    }


    async fn init_handle_ctx(&self) ->Result<Box<dyn HandleCtx>, ThoError>;
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

        let borrowed = BorrowRef::borrow_to(&input, to_device_idx).await?;

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
        for<'a> <W as BorrowRef>::Out<'a>: Borrow<S::In<'a>>,
    {
        debug!("CudaActorHandle::run_stage_with_select: to_device_idx={to_device_idx} input={input:?} selector={selector:?}");

        let state = if let Some(with_state) = stage.as_with_state() {
            // Keep selector-based execution consistent with the primary stateful runner path.
            Some(with_state.pre_run(to_device_idx).await?)
        } else {
            None
        };

        let borrowed = SelectRef::borrow_to(&input, to_device_idx, selector).await?;

        let mut handle_ctx = self.init_handle_ctx().await?;

        let sync_ctx = handle_ctx.sync_ctx()?;

        let out = match state {
            Some(state) => state.do_run_with_state(Arc::clone(&stage), borrowed.borrow(), ctx),
            None => Arc::clone(&stage).do_run(borrowed.borrow(), ctx),
        };

        handle_ctx.on_after_run()?;

        out
    }

}

struct CudaLocalActor {
    tho_device_idx:usize,
    receiver: mpsc::Receiver<CudaActorMessage>,
    copy_ctx:Arc<RwLock<CopyCtx>>,
    is_resetting:Arc<AtomicBool>,
}

impl CudaLocalActor {
    pub fn new(tho_device_idx:usize, receiver: mpsc::Receiver<CudaActorMessage>) -> Result<Self, ThoError> {
        debug!("CudaActor::new: tho_device_idx={tho_device_idx}");

        Ok(Self {
            tho_device_idx,
            receiver,
            copy_ctx: Arc::new(RwLock::new(CopyCtx::make_instance_for_cuda_device(tho_device_idx - 1, 0.2, 32)?)),
            is_resetting: Arc::new(AtomicBool::new(false)),
        })
    }

     fn handle_message(&mut self, msg: CudaActorMessage) {

        match msg {
            CudaActorMessage::Copy{ src, dst_device, respond_to } => {
                if cuda_activity_gate().is_paused() {
                    let _ = respond_to.send(
                        Err(ThoError::new("CudaActor is offline"))
                    );
                    return;
                }

                let copy_ctx = Arc::clone(&self.copy_ctx);
                let is_resetting = Arc::clone(&self.is_resetting);
                let tho_device_idx = self.tho_device_idx;
                tokio::spawn(async move {
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

                    if let Err(e) = &result  && e.needs_cuda_ctx_reset() {
                        if let Err(e) = Self::do_reset(copy_ctx, is_resetting, tho_device_idx, "Copy").await {
                            error!("CudaActor::handle_message, Copy:, e={e:?}");
                        }
                    }

                    let _ = respond_to.send(
                        result
                     );
                 });
            }
            CudaActorMessage::SyncStream { stream, respond_to } => {
                if cuda_activity_gate().is_paused() {
                    let _ = respond_to.send(
                        Err(ThoError::new("CudaActor is offline"))
                    );
                    return;
                }


                let copy_ctx = Arc::clone(&self.copy_ctx);
                let is_resetting = Arc::clone(&self.is_resetting);
                let tho_device_idx = self.tho_device_idx;
                tokio::spawn(async move {
                    let result = match tokio::task::spawn_blocking(move || {
                        stream.synchronize()

                    }).await {
                        Ok(result) => result,
                        Err(e) => Err(ThoError::new(format!("CudaActor failed to join stream synchronization task: {e}"))),
                    };

                    if let Err(e) = &result  && e.needs_cuda_ctx_reset() {
                        if let Err(e) =Self::do_reset(copy_ctx, is_resetting, tho_device_idx, "SyncStream").await {
                            error!("CudaActor::handle_message, SyncStream:, e={e:?}");
                        }
                    }

                    let _ = respond_to.send(result);

                    Ok::<(), ThoError>(())
                });
            }
            CudaActorMessage::Reset { respond_to } => {
                if self.is_resetting.load(Relaxed) {
                    let _ = respond_to.send(
                        Err(ThoError::new("CudaActor is resetting"))
                    );
                    return;
                }

                let copy_ctx = Arc::clone(&self.copy_ctx);
                let is_resetting = Arc::clone(&self.is_resetting);
                let tho_device_idx = self.tho_device_idx;
                tokio::spawn(async move {
                    Self::do_reset(copy_ctx, is_resetting, tho_device_idx, "Reset").await.unwrap();

                    let _ = respond_to.send(Ok(()));
                });
            }
            CudaActorMessage::GetExecStream { respond_to } => {
                if cuda_activity_gate().is_paused() {
                    let _ = respond_to.send(
                        Err(ThoError::new("CudaActor is offline"))
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

    async fn do_reset(copy_ctx: Arc<RwLock<CopyCtx>>, is_resetting:Arc<AtomicBool>,  tho_device_idx:usize, from:&str) -> Result<(), ThoError> {
        if is_resetting.load(Relaxed) {
            return Ok(());
        }

        is_resetting.store(true, Relaxed);
        let ticket = cuda_activity_gate().request_pause();


        match {
            track_await(GlobalCtx::global()
                .drop_device(tho_device_idx, "drop"))
                .await?;

            info!("CudaActor::handle_message/{from}, Reset: drop_device done");

            timeout(Duration::from_millis(1000), copy_ctx.write()).await?.reset()
        } {
            Ok(_) => {
                is_resetting.store(false, Relaxed);
                let is_owner = ticket.is_owner();

                cuda_activity_gate().finish_pause(ticket);//TODO can be done only here so

                info!("CudaActor::handle_message/{from}: is_owner={is_owner} Reset ok");
                Ok(())
            }
            Err(e) => {
                info!("CudaActor::handle_message/{from}, Reset: failed: e={e:?}");
                panic!("reset failed");//TODO shutdown
            }
        }
    }

    async fn run(mut actor: CudaLocalActor) {
        while let Some(msg) = actor.receiver.recv().await {
            actor.handle_message(msg);
        }
    }
}

#[derive(Debug, Clone)]
pub struct CudaLocalActorHandle {
    sender: mpsc::Sender<CudaActorMessage>,
}


impl CudaLocalActorHandle {
    pub fn new(tho_device_id:usize) -> Result<Self, ThoError> {
        debug!("CudaLocalActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let actor = CudaLocalActor::new(tho_device_id, receiver)?;
        tokio::spawn(CudaLocalActor::run(actor));

        Ok(Self {
            sender,
        })
    }

    async fn copy_tensor(&self, src:ThoTensor, dst_device:ThoDevice) ->Result<Tensor, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaActorMessage::Copy {
                src,
                dst_device,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    async fn sync_stream(&self, stream:Guard<ThoCudaStream>) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaActorMessage::SyncStream {
                stream,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    async fn reset_inner(&self) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaActorMessage::Reset {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    //TODO these are 'exec' streams, not for copy
    async fn acquire_exec_stream(&self) -> Result<Guard<ThoCudaStream>, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaActorMessage::GetExecStream {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
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

        Ok(self.copy_tensor(src, dst_device).await?.into())
    }

    async fn reset(&self) -> Result<(), ThoError> {
        self.reset_inner().await
    }


    async fn init_handle_ctx(&self) -> Result<Box<dyn HandleCtx>, ThoError> {

        let stream = self.acquire_exec_stream().await?;


        Ok(Box::new(LocalHandleCtx { stream }))
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

    pub fn reset(&mut self) -> Result<(), ThoError> {
        info!("CopyCtx::reset_cuda_context: about to reset cu_ctx.reset");


        let cuda_context = ThoCudaContext::new(self.cuda_device as i32)?;
        cuda_context.bind_to_thread()?;
        let mut cu_ctx = ThoCudaContext::new(self.cuda_device as i32)?;
        cu_ctx.reset()?;

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

    pub async fn move_t(&self, src_t: Tensor, dst_device: usize, kind: Kind) -> Result<Tensor, ThoError> {
        if dst_device == 0 {
            track_await(move_d2h(kind, src_t, self.buf_size, &self.bufs_pool_h2d)).await
        } else {
            track_await(move_h2d(kind, src_t, self.buf_size, &self.bufs_pool_h2d, dst_device)).await
        }
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

    pub fn buf_size(&self) -> usize {
        self.buf_size
    }


}
