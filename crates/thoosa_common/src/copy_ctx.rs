use crate::cuda_utils::{copy_d2h, copy_d2h_from_ptr, copy_h2d, copy_h2d_from_ptr, cuda_mem_for_device, move_d2h, move_h2d, PinnedHostBuffer};
use crate::ipc::ipc_cuda_utils::{SharedMemfdBuffer, ipc_copy_d2h_from_ptr, ipc_copy_h2d_from_ptr};
use async_trait::async_trait;
use thoosa_shims::context::ThoCudaContext;
use thoosa_shims::stream::{ThoCudaStream, ThoStreamGuard};
use log::{debug, info};
use std::any::Any;
use std::borrow::Borrow;
use std::fmt::{Debug, Formatter};
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64};
use std::sync::atomic::Ordering::{AcqRel, Acquire, Relaxed};
use std::time::Duration;
use cust::sys::CU_MEMHOSTALLOC_WRITECOMBINED;
use futures_util::future::BoxFuture;
use tch::Kind::Int64;
use tch::{Kind, Tensor};
use tokio::sync::{mpsc, oneshot, RwLock};
use tokio::sync::oneshot::Sender;
use tokio::time::timeout;
use tracing::error;
use tub::{Guard, Pool};
use thoosa_shared::{cuda_activity_gate, ThoError};
use thoosa_shared::tracked_await::track_await;
use thoosa_shared::traits::Named;
use crate::dims::Dims;
use crate::runner::common::cachable::{BorrowRef, WithDataRef, WithSelectDataRef};
use crate::runner::common::RunCtx;
use crate::runner::common::runnable::{Stage, SyncStage};
use crate::runner::global_ctx::GlobalCtx;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;

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

pub enum ExecSyncStageActionResult<T> {
    Sync { stream: Option<Guard<ThoCudaStream>>, out: T },
    NoSync(T),
}

    pub enum GuardOrNot {
    Local(Guard<ThoCudaStream>),
    IPC
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
pub trait CudaActorHandle: Debug + Send + Sync {
    async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) -> Result<ThoTensor, ThoError>;

    async fn reset(&self) -> Result<(), ThoError>;

    async fn shutdown(&self) -> Result<(), ThoError> {
        Ok(())
    }


    fn exec_sync_stage_boxed<'a>(
        &'a self,
        action: Box<dyn FnOnce(Option<Guard<ThoCudaStream>>) -> BoxFuture<'a, Result<ExecSyncStageActionResult<Box<dyn Any + Send>>, ThoError>> + Send + 'a>,
    ) -> BoxFuture<'a, Result<Box<dyn Any + Send>, ThoError>>;

    async fn install_endpoint(&self, init_msg:String)-> Result<(), ThoError> {
        Ok(())
    }

    async fn invoke_endpoint(&self, _input:String)-> Result<String, ThoError> {
        Ok("unimplemented".to_string())
    }

    async fn drop_tho_tensor(&self, t:ThoTensor)-> Result<(), ThoError> {
        Ok(())
    }

    async fn guard(&self) -> Result<GuardOrNot, ThoError>;

    async fn init_handle_ctx(&self) ->Result<Box<dyn HandleCtx>, ThoError>;
}

impl dyn CudaActorHandle + '_ {
    pub fn exec_sync_stage<'a, R, F, Fut>(&'a self, action: F) -> BoxFuture<'a, Result<R, ThoError>>
    where
        R: Send + 'static,
        F: FnOnce(Option<Guard<ThoCudaStream>>) -> Fut + Send + 'a,
        Fut: Future<Output = Result<ExecSyncStageActionResult<R>, ThoError>> + Send + 'a,
    {
        Box::pin(async move {
            let out = self.exec_sync_stage_boxed(Box::new(move |stream| {
                Box::pin(async move {
                    match action(stream).await? {
                        ExecSyncStageActionResult::Sync { stream, out } => Ok(ExecSyncStageActionResult::Sync { stream, out: Box::new(out) as Box<dyn Any + Send> }),
                        ExecSyncStageActionResult::NoSync(out) => Ok(ExecSyncStageActionResult::NoSync(Box::new(out) as Box<dyn Any + Send>)),
                    }
                })
            })).await?;

            out.downcast::<R>()
                .map(|out| *out)
                .map_err(|_| ThoError::new("CudaActorHandle::exec_sync_stage result type mismatch"))
        })
    }


    pub async fn run_stage<S, W, SL>(
        &self,
        stage: Arc<S>,
        input: W,
        selector: Option<SL>,
        ctx:&S::Ctx
    ) -> Result<S::Out, ThoError>
    where
        S: SyncStage,
        SL: Clone + Debug + Send + 'static,
        W: Sync,
        for<'a> W: BorrowRef<SL> + 'a,
        for<'a> <W as BorrowRef<SL>>::Out<'a>: Borrow<S::In<'a>>,
    {


        let borrowed = input.borrow_to(0, selector).await?;

        let mut handle_ctx = self.init_handle_ctx().await?;

        let sync_ctx = handle_ctx.sync_ctx()?;

        let out = stage.do_run(borrowed.borrow(), ctx);

        handle_ctx.on_after_run()?;

        out
    }

}


struct CudaActor {
    tho_device_idx:usize,
    receiver: mpsc::Receiver<CudaActorMessage>,
    copy_ctx:Arc<RwLock<CopyCtx>>,
    is_resetting:Arc<AtomicBool>,
}

impl CudaActor {
    pub fn new(tho_device_idx:usize, receiver: mpsc::Receiver<CudaActorMessage>) -> Self {
        debug!("CudaActor::new: tho_device_idx={tho_device_idx}");

        Self {
            tho_device_idx,
            receiver,
            copy_ctx: Arc::new(RwLock::new(CopyCtx::make_instance_for_cuda_device(tho_device_idx - 1, 0.2, 16))),
            is_resetting: Arc::new(AtomicBool::new(false)),
        }
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
                        dst_device.ordinal(),
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

    async fn run(mut actor: CudaActor) {
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
    pub fn new(tho_device_id:usize) -> Self {
        debug!("CudaLocalActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let actor = CudaActor::new(tho_device_id, receiver);
        tokio::spawn(CudaActor::run(actor));

        Self {
            sender,
        }
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
    async fn copy(&self, src:ThoTensor, dst_device:ThoDevice) -> Result<ThoTensor, ThoError> {
        Ok(self.copy_tensor(src, dst_device).await?.into())
    }

    async fn reset(&self) -> Result<(), ThoError> {
        self.reset_inner().await
    }

    fn exec_sync_stage_boxed<'a>(
        &'a self,
        action: Box<dyn FnOnce(Option<Guard<ThoCudaStream>>) -> BoxFuture<'a, Result<ExecSyncStageActionResult<Box<dyn Any + Send>>, ThoError>> + Send + 'a>,
    ) -> BoxFuture<'a, Result<Box<dyn Any + Send>, ThoError>> {
        Box::pin(async move {
            let stream = self.acquire_exec_stream().await?;
            match action(Some(stream)).await? {
                ExecSyncStageActionResult::Sync { stream, out } => {
                    let stream = stream.ok_or_else(|| ThoError::new("CudaLocalActorHandle::exec_sync_stage_boxed requested sync without stream"))?;
                    self.sync_stream(stream).await?;
                    Ok(out)
                }
                ExecSyncStageActionResult::NoSync(out) => Ok(out),
            }
        })
    }

    async fn guard(&self) -> Result<GuardOrNot, ThoError> {
        let stream = self.acquire_exec_stream().await?;

        Ok(GuardOrNot::Local(stream))
    }

    async fn init_handle_ctx(&self) -> Result<Box<dyn HandleCtx>, ThoError> {

        let stream = self.acquire_exec_stream().await?;


        Ok(Box::new(LocalHandleCtx { stream }))
    }

}

//TODO to be renamed to StreamCtx?
pub struct CopyCtx {
    pub(crate) cu_ctx: ThoCudaContext,
    pub(crate) buffs: usize,
    pub(crate) buf_size: usize,
    exec_stream_pool: Option<Arc<Pool<ThoCudaStream>>>,
    bufs_pool_d2h: Option<Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>>,
    bufs_pool_h2d: Option<Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>>>,
    pub(crate)  ipc_bufs_pool: Option<Arc<Pool<(SharedMemfdBuffer, Arc<ThoCudaStream>)>>>,
}

impl Debug for CopyCtx {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CopyCtx")
            // .field("cu_ctx", &self.cu_ctx)
            .field("buffs", &self.buffs)
            .field("buf_size", &self.buf_size)
                .field("bufs_pool.remaining_capacity", &self.bufs_pool_h2d.as_ref().expect("bufs_pool").remaining_capacity())
                .field("ipc_bufs_pool.remaining_capacity", &self.ipc_bufs_pool.as_ref().expect("ipc_bufs_pool").remaining_capacity())
            .finish()
    }
}

impl CopyCtx {
    pub fn new(cuda_device: usize, mem_k: f64, buffs: usize) -> Self {
        debug!("CopyCtx::new: cuda_device={cuda_device} mem_k={mem_k} buffs={buffs}");

        let buf_size = Self::clc_buff_size(cuda_device, mem_k, buffs);
        Self::new_with_buf_size(cuda_device, buffs, buf_size)
    }

    fn new_with_buf_size(cuda_device: usize, buffs: usize, buf_size: usize) -> Self {
        Self::new_with_buf_size_inner(cuda_device, buffs, buf_size, false)
    }

    fn new_with_buf_size_inner(cuda_device: usize, buffs: usize, buf_size: usize, is_ipc: bool) -> Self {
        debug!("CopyCtx::new_with_buf_size: cuda_device={cuda_device} buffs={buffs} buf_size={buf_size}");

        let cuda_context = ThoCudaContext::new(cuda_device as i32).unwrap();
        cuda_context.bind_to_thread().unwrap();
        // let cu_ctx = ThoCudaContext::new(cuda_device as i32).expect("failed to create CUDA context");
        //TODO all state from CopyCtx should be moved into ActorHandles

        Self {
            cu_ctx: cuda_context,
            buffs,
            buf_size,
            exec_stream_pool: (!is_ipc).then(||Arc::new(Pool::from_iter((0..32).map(|_| ThoCudaStream::from_pool(cuda_device as i32, false).unwrap())))),//TODO unwrap)),
            bufs_pool_d2h: (!is_ipc).then(||Self::build_pool(buffs/2, buf_size, cuda_device, 0)),//TODO unwrap)),
            bufs_pool_h2d: (!is_ipc).then(||Self::build_pool(buffs/2, buf_size, cuda_device, CU_MEMHOSTALLOC_WRITECOMBINED)),
            ipc_bufs_pool: is_ipc.then( || Self::build_ipc_pool(buffs, buf_size, cuda_device)),
        }
    }

    #[cfg(test)]
    pub(crate) fn make_instance_with_buffer_size(cuda_device: usize, buffs: usize, buf_size: usize) -> Self {
        Self::new_with_buf_size_inner(cuda_device, buffs, buf_size, true)
    }

    fn build_ipc_pool(
        num_buffs: usize,
        buf_size: usize,
        cuda_device: usize,
    ) -> Arc<Pool<(SharedMemfdBuffer, Arc<ThoCudaStream>)>> {
        Arc::new(Pool::from_iter((0..num_buffs).map(|_| {
            let buffer_id = NEXT_IPC_BUFFER_ID.fetch_add(1, Relaxed);
            (
                SharedMemfdBuffer::create(buffer_id, buf_size).unwrap(),
                Arc::new(ThoCudaStream::from_pool(cuda_device as i32, true).unwrap()),
            )
        })))
    }

    fn build_pool(
        num_buffs: usize,
        buf_size: usize,
        //cu_ctx: &Arc<ThoCudaContext>,
        cuda_device:usize,
        flags: u32
    ) -> Arc<Pool<(PinnedHostBuffer, Arc<ThoCudaStream>)>> {

        Arc::new(Pool::from_iter((0..num_buffs).map(|_| {
            (
                PinnedHostBuffer::new(buf_size, flags).unwrap(),
                Arc::new(ThoCudaStream::from_pool(cuda_device as i32, true).unwrap()), //TODO unwrap
            )
        })))
    }

    #[cfg(test)]
    #[allow(dead_code)]
    pub fn rebuild(&mut self, mem_k: f64, buffs: usize) {
        assert_eq!(self.bufs_pool_h2d.as_ref().expect("no bufs_pool").remaining_capacity(), self.buffs);
        assert_eq!(self.ipc_bufs_pool.as_ref().expect("no ipc_bufs_pool").remaining_capacity(), self.buffs);
        // self
            // .cu_ctx
            // .bind_to_thread()
            // .expect("failed to bind CUDA context to thread");

        drop(Arc::into_inner(std::mem::replace(
            &mut self.bufs_pool_h2d,
            Some(Arc::new(Pool::from_vec(vec![(
                PinnedHostBuffer::new(1, 0).unwrap(),
                Arc::new(ThoCudaStream::from_pool(0/*self.cu_ctx.ordinal()*/ as i32, true).unwrap()),
            )]))),
        ).unwrap()));

        drop(Arc::into_inner(std::mem::replace(
            &mut self.ipc_bufs_pool,
            Some(Arc::new(Pool::from_vec(vec![(
                SharedMemfdBuffer::create(NEXT_IPC_BUFFER_ID.fetch_add(1, Relaxed), 1).unwrap(),
                Arc::new(ThoCudaStream::from_pool(0, true).unwrap()),
            )]))),
        ).unwrap()));

        self.buffs = buffs;
        self.buf_size = Self::clc_buff_size(0/*self.cu_ctx.ordinal()*/, mem_k, buffs);
        self.bufs_pool_h2d = Some(Self::build_pool(buffs, self.buf_size, 0, CU_MEMHOSTALLOC_WRITECOMBINED/*&self.cu_ctx*/));
        self.ipc_bufs_pool = Some(Self::build_ipc_pool(buffs, self.buf_size, 0/*&self.cu_ctx*/));
    }

    //TODO these are 'exec' streams, not for copy
    pub async fn acquire_exec_stream(&self) -> Guard<ThoCudaStream> {
        track_await(self.exec_stream_pool.as_ref().expect("no exec_stream_pool")
            .acquire())
            .await
    }

    pub fn reset(&mut self) -> Result<(), ThoError> {
        info!("CopyCtx::reset_cuda_context: about to reset cu_ctx.reset");

        drop(self.bufs_pool_h2d.take());
        drop(self.ipc_bufs_pool.take());
        drop(self.exec_stream_pool.take());

        self.cu_ctx.reset()?;
        info!("CopyCtx::reset_cuda_context: cu_ctx.reset done");

        self.bufs_pool_h2d = Some(Self::build_pool(self.buffs, self.buf_size, self.cu_ctx.device as usize, CU_MEMHOSTALLOC_WRITECOMBINED));
        info!("CopyCtx::reset_cuda_context: bufs_pool rebuilt");

        self.ipc_bufs_pool = Some(Self::build_ipc_pool(self.buffs, self.buf_size, self.cu_ctx.device as usize));
        info!("CopyCtx::reset_cuda_context: ipc_bufs_pool rebuilt");

        self.exec_stream_pool = Some(Arc::new(Pool::from_iter((0..32).map(|_| ThoCudaStream::from_pool(self.cu_ctx.device, false).unwrap()))));

        info!("CopyCtx::reset_cuda_context: exec_stream_pool rebuilt");

        Ok(())
    }

    pub fn make_instance(mem_k: f64, buffs: usize) -> CopyCtx {
        Self::make_instance_for_cuda_device(0, mem_k, buffs)
    }

    pub fn make_instance_for_cuda_device(cuda_device: usize, mem_k: f64, buffs: usize) -> CopyCtx {

        debug!("make_instance_for_cuda_device: mem_k={mem_k} buffs={buffs}");

        CopyCtx::new(cuda_device, mem_k, buffs)
    }

    pub(crate) fn make_ipc_instance_for_cuda_device(cuda_device: usize, mem_k: f64, buffs: usize) -> CopyCtx {
        debug!("make_ipc_instance_for_cuda_device: mem_k={mem_k} buffs={buffs}");

        let buf_size = Self::clc_buff_size(cuda_device, mem_k, buffs);
        Self::new_with_buf_size_inner(cuda_device, buffs, buf_size, true)
    }

    fn clc_buff_size(cuda_device: usize, mem_k: f64, buffs: usize) -> usize {
        let (_free_mem, total_mem) = cuda_mem_for_device(cuda_device);

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
            track_await(move_d2h(kind, src_t, self.buf_size, &self.bufs_pool_h2d.as_ref().expect("no bufs_pool"))).await
        } else {
            track_await(move_h2d(kind, src_t, self.buf_size, &self.bufs_pool_h2d.as_ref().expect("no bufs_pool"), dst_device)).await
        }
    }

    pub async fn copy<R: AsRef<Tensor>>(&self, src_t: &R, dst_tho_device: usize, kind: Kind) -> Result<Tensor, ThoError> {
        if dst_tho_device == 0 {
            track_await(copy_d2h(kind, src_t, self.buf_size, &self.bufs_pool_h2d.as_ref().expect("no bufs_pool"))).await
        } else {
            track_await(copy_h2d(kind, src_t, self.buf_size, &self.bufs_pool_h2d.as_ref().expect("no bufs_pool"), dst_tho_device)).await
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(agg_perf="split_padded_motif_scores"))]
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
                &self.bufs_pool_d2h.as_ref().expect("no bufs_pool"), src_ptr, src_size
            )).await
        } else {
            track_await(copy_h2d_from_ptr(
                kind,
                self.buf_size,
                &self.bufs_pool_h2d.as_ref().expect("no bufs_pool"),
                dst_tho_device_idx,
                src_ptr,
                src_size,
            )).await
        };

        // println!("copy_from_ptr: copied t.size={:?} dst_device={dst_device} t=\n{t}", t.size());

        t
    }

    pub async fn ipc_copy_from_ptr(
        &self,
        src_ptr: usize,
        src_size: &[i64],
        dst_tho_device_idx: usize,
        kind: Kind,
    ) -> Result<Tensor, ThoError> {
        if dst_tho_device_idx == 0 {
            track_await(ipc_copy_d2h_from_ptr(
                kind,
                self.buf_size,
                self.ipc_bufs_pool.as_ref().expect("no ipc_bufs_pool"),
                src_ptr,
                src_size,
            )).await
        } else {
            track_await(ipc_copy_h2d_from_ptr(
                kind,
                self.buf_size,
                self.ipc_bufs_pool.as_ref().expect("no ipc_bufs_pool"),
                dst_tho_device_idx,
                src_ptr,
                src_size,
            )).await
        }
    }

    pub fn buf_size(&self) -> usize {
        self.buf_size
    }


}
