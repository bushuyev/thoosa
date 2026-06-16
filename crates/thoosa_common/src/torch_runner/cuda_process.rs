use std::collections::HashMap;
use std::fmt::Debug;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use dashmap::DashMap;
use log::{debug, error};
use tch::Device::Cuda;
use tch::Kind::Float;
use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
use tch::Tensor;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task::JoinHandle;
use tub::Guard;
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::{mono_key, ThoError};
use thoosa_shared::traits::{Data, Named};
use thoosa_shims::stream::ThoCudaStream;
use crate::copy_ctx::CudaActorMessage;
use crate::dims::Dims;
use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster};
use crate::runner::common::cachable::DevicesWrap;
use crate::runner::common::RunCtx;
use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::tho_device::ThoDevice;
use crate::tho_tensor::ThoTensor;
#[cfg(test)]
use crate::torch_runner::tests::{ClcTask, LoadedTensor};

const IPC_SOCK_ENV: &str = "THOOSA_WORKER_IPC_SOCK";
const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

struct CudaWorker {
    all_cuda_tensors:DashMap<u64, Tensor>,
    handle:JoinHandle<Result<(), ThoError>>,
}

struct WorkerReply {
    id: u64,
    payload: String,
}


impl CudaWorker {
    pub fn new() -> Result<Self, ThoError> {
        let handle = tokio::spawn(async move {
            Self::run_worker().await?;

            Ok(())
        });

        Ok(Self {
            all_cuda_tensors:DashMap::new(),
            handle
        })
    }


    pub async fn run_worker() -> Result<(), ThoError> {
        let sock_path = std::env::var_os(IPC_SOCK_ENV)
            .ok_or_else(|| ThoError::new("missing THOOSA_WORKER_IPC_SOCK"))?;

        debug!("running worker: sock_path={sock_path:?}");

        let stream = UnixStream::connect(PathBuf::from(sock_path)).await?;

        debug!("run_worker: worker connected");

        let (mut reader, mut writer) = stream.into_split();

        write_text_frame(&mut writer, "worker-ready").await?;

        let (reply_tx, reply_rx) = mpsc::channel::<WorkerReply>(1024);

        let writer_task = tokio::spawn(worker_response_writer_loop(writer, reply_rx));

        loop {
            let frame = read_text_frame(&mut reader).await?;
            let (id, payload) = parse_wire_msg(&frame)?;

            debug!("run_worker: worker received: id={id}, payload={payload:?}");

            if payload == "shutdown" {
                reply_tx
                    .send(WorkerReply {
                        id,
                        payload: "shutdown-ok".to_string(),
                    })
                    .await
                    .map_err(|_| ThoError::new("worker response writer is closed"))?;

                debug!("run_worker: worker got shutdown command");
                break;
            }

            let reply_tx = reply_tx.clone();

            tokio::spawn(async move {
                let reply = handle_worker_request(payload).await;

                debug!("run_worker: reply={reply}");

                let _ = reply_tx
                    .send(WorkerReply { id, payload: reply })
                    .await;
            });
        }

        drop(reply_tx);

        writer_task
            .await
            .map_err(|e| ThoError::new(format!("worker writer task join error: {e}")))??;

        debug!("worker finished");

        Ok(())
    }

}


async fn worker_response_writer_loop<W>(
    mut writer: W,
    mut reply_rx: mpsc::Receiver<WorkerReply>,
) -> Result<(), ThoError>
where
    W: AsyncWrite + Unpin,
{
    while let Some(reply) = reply_rx.recv().await {
        let frame = encode_wire_msg(reply.id, &reply.payload);
        write_text_frame(&mut writer, &frame).await?;
    }

    Ok(())
}


async fn handle_worker_request(payload: String) -> String {
    debug!("handle_worker_request: payload={payload:?}");

    match payload.as_str() {
        "ping" => "pong".to_string(),

        "status" => "worker-status: alive".to_string(),

        msg if msg.starts_with("copy: ") => {
            let text = &msg["copy: ".len()..];

            format!("copy: {text} done")
        }


        other => {
            format!("unknown-command {other:?}")
        }
    }
}


#[derive(Debug)]
pub struct IPCTensor {//reference to a tensor kept in worker process started in CudaWorker::run_worker
    id:String,
}


impl IPCTensor {
    pub fn new<S:Into<String>>(id:S) -> Self {
        Self { id :id.into() }
    }
}

pub enum CudaIPCActorMessage {
    Copy{ src:Tensor, dst_device:ThoDevice, respond_to: oneshot::Sender<Result<IPCTensor, ThoError>>},
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

struct OutgoingRequest {
    id: u64,
    payload: String,
}


#[derive(Debug)]
pub struct CudaIPCActor {
    tx: mpsc::Sender<OutgoingRequest>,
    receiver: mpsc::Receiver<CudaIPCActorMessage>,
    child:Child,

    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    next_id: Arc<AtomicU64>,
}

impl CudaIPCActor {
    pub async fn new(tho_device_id:usize, cuda_worker_args: Vec<&str>, receiver: mpsc::Receiver<CudaIPCActorMessage>) -> Result<Self, ThoError> {

        let (mut child, stream) = Self::spawn_worker(cuda_worker_args).await?;

        let (reader, writer) = stream.into_split();

        let (tx, rx) = mpsc::channel::<OutgoingRequest>(1024);

        let pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>> =
            Arc::new(Mutex::new(HashMap::new()));

        tokio::spawn(writer_loop(writer, rx, Arc::clone(&pending)));
        tokio::spawn(reader_loop(reader, Arc::clone(&pending)));


        Ok(Self {
            tx,
            receiver,
            child,
            pending,
            next_id: Arc::new(AtomicU64::new(1)),
        })
    }

    async fn spawn_worker(args: Vec<&str>) -> Result<(Child, UnixStream), ThoError> {
        let exe = std::env::current_exe()?;

        debug!("spawn_worker, exe={exe:?}");

        let mut command = &mut Command::new(exe);

        for arg in args {
            command = command.arg(arg);
        }

        let dir = tempfile::tempdir()?;
        let sock_path = dir.path().join("worker.sock");

        debug!("spawn_worker, dir={dir:?} sock_path={sock_path:?}");

        let mut child = command

            .env(IPC_SOCK_ENV, &sock_path)
            .stdin(Stdio::null())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .spawn()?;

        let listener = UnixListener::bind(&sock_path)?;

        debug!("spawn_worker, listener={listener:?}");

        let accepted = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            listener.accept(),
        )
            .await;

        debug!("spawn_worker, accepted={accepted:?}");

        let (mut stream, _addr) = match accepted {
            Ok(Ok(pair)) => pair,
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => match child.try_wait()? {
                Some(status) => {
                    return Err(ThoError::new(format!(
                        "worker exited before connecting: status={status}"
                    )));
                }
                None => {
                    let _ = child.kill().await;
                    return Err(ThoError::new("worker did not connect within timeout"));
                }
            },
        };

        debug!("spawn_worker, _addr={_addr:?}");

        let ready = read_text_frame(&mut stream).await?;
        debug!("main received: {ready:?}");
        assert_eq!(ready, "worker-ready");

        // The socket file can be removed after accept. The connected UnixStream
        // remains valid.
        drop(dir);

        Ok((child, stream))
    }

    fn handle_message(&mut self, msg: CudaIPCActorMessage) {
        match msg {
            CudaIPCActorMessage::Copy { src, dst_device, respond_to } => {

                let id = self.next_id.fetch_add(1, Ordering::Relaxed);
                let (reply_tx, reply_rx) = oneshot::channel();
                let payload = format!("copy: {src:?}");
                let pending = Arc::clone(&self.pending);
                let self_tx = self.tx.clone();

                tokio::spawn(async move {
                    {
                        let mut pending = pending.lock().await;

                        let old = pending.insert(id, reply_tx);
                        debug_assert!(old.is_none());
                    }

                    let send_result = self_tx.send(OutgoingRequest { id, payload }).await;

                    if send_result.is_err() {
                        let mut pending = pending.lock().await;
                        pending.remove(&id);

                        return Err(ThoError::new("worker IPC writer task is closed"));
                    }


                    let id = reply_rx
                        .await
                        .map_err(|_| ThoError::new(format!("response channel for request {id} was dropped")))??;

                    respond_to.send(Ok(IPCTensor::new(id.clone()))).map_err(|e| ThoError::new(format!("response channel for request {id} got {e:?}")))?;

                    Ok(())
                });


            }
            CudaIPCActorMessage::SyncStream { .. } => {}
            CudaIPCActorMessage::GetExecStream { .. } => {}
            CudaIPCActorMessage::Reset { .. } => {}
        }
    }

    async fn run(mut actor: CudaIPCActor) {
        while let Some(msg) = actor.receiver.recv().await {
            actor.handle_message(msg);
        }
    }
}

async fn writer_loop<W>(
    mut writer: W,
    mut rx: mpsc::Receiver<OutgoingRequest>,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
) where
    W: AsyncWrite + Unpin,
{
    while let Some(req) = rx.recv().await {
        let frame = encode_wire_msg(req.id, &req.payload);

        if let Err(e) = write_text_frame(&mut writer, &frame).await {
            error!("worker IPC writer failed: {e}");

            fail_one_pending(
                &pending,
                req.id,
                format!("failed to write request {}: {e}", req.id),
            )
                .await;

            fail_all_pending(&pending, format!("worker IPC writer stopped: {e}")).await;
            break;
        }
    }

    debug!("worker IPC writer loop finished");
}

async fn reader_loop<R>(
    mut reader: R,
    pending: Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
) where
    R: AsyncRead + Unpin,
{
    loop {
        let frame = match read_text_frame(&mut reader).await {
            Ok(frame) => frame,
            Err(e) => {
                debug!("worker IPC reader stopped: {e}");

                fail_all_pending(&pending, format!("worker IPC reader stopped: {e}")).await;
                break;
            }
        };

        let (id, payload) = match parse_wire_msg(&frame) {
            Ok(parsed) => parsed,
            Err(e) => {
                error!("bad worker response frame {frame:?}: {e}");
                continue;
            }
        };

        let reply_tx = {
            let mut pending = pending.lock().await;
            pending.remove(&id)
        };

        match reply_tx {
            Some(reply_tx) => {
                let _ = reply_tx.send(Ok(payload));
            }
            None => {
                error!("received response for unknown request id={id}, payload={payload:?}");
            }
        }
    }

    debug!("worker IPC reader loop finished");
}

fn parse_wire_msg(frame: &str) -> Result<(u64, String), ThoError> {
    let Some((id, payload)) = frame.split_once('\t') else {
        return Err(ThoError::new(format!(
            "bad wire frame without request id separator: {frame:?}"
        )));
    };

    let id = id
        .parse::<u64>()
        .map_err(|e| ThoError::new(format!("bad request id {id:?}: {e}")))?;

    Ok((id, payload.to_string()))
}


async fn fail_one_pending(
    pending: &Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    id: u64,
    msg: String,
) {
    let reply_tx = {
        let mut pending = pending.lock().await;
        pending.remove(&id)
    };

    if let Some(reply_tx) = reply_tx {
        let _ = reply_tx.send(Err(ThoError::new(msg)));
    }
}

async fn fail_all_pending(
    pending: &Arc<Mutex<HashMap<u64, oneshot::Sender<Result<String, ThoError>>>>>,
    msg: String,
) {
    let pending_items = {
        let mut pending = pending.lock().await;
        pending.drain().collect::<Vec<_>>()
    };

    for (_id, reply_tx) in pending_items {
        let _ = reply_tx.send(Err(ThoError::new(msg.clone())));
    }
}

fn encode_wire_msg(id: u64, payload: &str) -> String {
    format!("{id}\t{payload}")
}


#[derive(Debug, Clone)]
pub struct CudaIPCActorHandle {
    sender: mpsc::Sender<CudaIPCActorMessage>,
}


impl CudaIPCActorHandle {
    pub async fn new(tho_device_id:usize, cuda_worker_args: Vec<&str>) -> Result<Self, ThoError> {
        debug!("CudaActorHandle::new: tho_device_id={tho_device_id}");

        let (sender, receiver) = mpsc::channel(8);
        let actor = CudaIPCActor::new(tho_device_id, cuda_worker_args, receiver).await?;
        tokio::spawn(CudaIPCActor::run(actor));

        Ok(Self {
            sender,
        })
    }

    pub async fn copy(&self, src:Tensor, dst_device:ThoDevice) ->Result<IPCTensor, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Copy {
                src,
                dst_device,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    pub async fn sync_stream(&self, stream:Guard<ThoCudaStream>) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::SyncStream {
                stream,
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    pub async fn reset(&self) -> Result<(), ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::Reset {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

    //TODO these are 'exec' streams, not for copy
    pub async fn acquire_exec_stream(&self) -> Result<Guard<ThoCudaStream>, ThoError> {
        let (respond_to, recv) = oneshot::channel();
        self.sender.send(
            CudaIPCActorMessage::GetExecStream {
                respond_to,
            }
        ).await.map_err(|e|ThoError::new(format!("e={e:?}")))?;//TODO

        recv.await.map_err(|e|ThoError::new(format!("e={e:?}")))?//TODO
    }

}

async fn write_text_frame<W>(
    writer: &mut W,
    text: &str,
) -> Result<(), ThoError>
where
    W: AsyncWrite + Unpin,
{
    write_frame(writer, text.as_bytes()).await?;
    Ok(())
}

async fn read_text_frame<R>(
    reader: &mut R,
) -> Result<String, ThoError>
where
    R: AsyncRead + Unpin,
{
    let payload = read_frame(reader).await?;

    String::from_utf8(payload)
        .map_err(|e| ThoError::new(format!("received non-utf8 frame: {e}")))
}

async fn write_frame<W>(
    writer: &mut W,
    payload: &[u8],
) -> std::io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    if payload.len() > u32::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("frame too large: {} bytes", payload.len()),
        ));
    }

    let len = payload.len() as u32;

    writer.write_all(&len.to_le_bytes()).await?;
    writer.write_all(payload).await?;
    writer.flush().await?;

    Ok(())
}

async fn read_frame<R>(
    reader: &mut R,
) -> std::io::Result<Vec<u8>>
where
    R: AsyncRead + Unpin,
{
    let mut len_buf = [0u8; 4];

    reader.read_exact(&mut len_buf).await?;

    let len = u32::from_le_bytes(len_buf) as usize;

    if len > MAX_FRAME_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame too large: {len} bytes"),
        ));
    }

    let mut payload = vec![0u8; len];

    reader.read_exact(&mut payload).await?;

    Ok(payload)
}


#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::process::Stdio;
    use std::sync::Arc;
    use log::debug;
    use tch::{Device, Tensor};
    use tch::Device::Cuda;
    use tch::Kind::Float;
    use tch::kind::{FLOAT_CPU, FLOAT_CUDA};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{UnixListener, UnixStream};
    use tokio::process::{Child, Command};
    use thoosa_shared::comp_id::SynKey;
    use thoosa_shared::{mono_key, ThoError};
    use thoosa_shared::tracked_await::track_await;
    use crate::runner::adjusters::max_parallel::{MaxParallelAdjustContextImpl, MaxParallelAdjuster};
    use crate::runner::common::cachable::DevicesWrap;
    use crate::runner::common::RunCtx;
    use crate::runner::global_ctx::GlobalCtx;
    use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
    use crate::runner::oom_handler::{Adjusters, Contexts, NoopInputContext, NoopRunnableContext, OOMHandler};
    use crate::runner::run_ctx_2::RunCtxImpl;
    use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, OOMRunnerConfig};
    use crate::tho_device::ThoDevice;
    use crate::tho_tensor::ThoTensor;
    use crate::torch_runner::cuda_process::{read_text_frame, write_text_frame, CudaWorker, CudaIPCActor, IPC_SOCK_ENV};
    use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
    use crate::torch_runner::tests::{ClcTask, ClcTaskOutput, LoadAndClcTask, LoadAndClcTaskId, LoadedTensor};



    macro_rules! pure_module_path {
        () => {{
            let p = module_path!();

            match p.split_once("::") {
                Some((_crate_name, rest)) => rest,
                None => "",
            }
        }};
    }

    async fn spawn_test_worker(test_name: &str) -> Result<(Child, UnixStream), ThoError> {

        let module = pure_module_path!();

        let test_name = if module.is_empty() {
            test_name.to_string()
        } else {
            format!("{module}::{test_name}")
        };
        let args:Vec<&str> = vec![&test_name, "--exact", "--nocapture"];

        CudaIPCActor::spawn_worker(args).await
    }


    async fn run_main(test_name: &str) -> Result<(), ThoError> {
        debug!("run_main: test_name={test_name:?}");
        let module = pure_module_path!();

        let test_name = if module.is_empty() {
            test_name.to_string()
        } else {
            format!("{module}::{test_name}")
        };
        let args:Vec<&str> = vec![test_name.clone().leak(), "--exact", "--nocapture"];

        use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
        RealGlobalCtx::init_devices_2(2, args).await?;
        let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

        // let (mut child, mut stream) = spawn_test_worker(test_name).await?;
        debug!("run_main: test_name={test_name:?}, spawn_worker ok");

        let tho_device_idx = 1;

        let main_process_cpu_tensor = Tensor::rand(&[10, 1000], FLOAT_CPU);


        let cuda_actor = GlobalCtx::global().cuda_ipc_actor(tho_device_idx);
        let copy = track_await(cuda_actor
            .copy(
                main_process_cpu_tensor,
                tho_device_idx.into(),
            )).await?;

        debug!("run_main: copy={copy:?}");

        // let ready = read_text_frame(&mut stream).await?;
        // debug!("main received: {ready:?}");
        // assert_eq!(ready, "worker-ready");
        //
        // write_text_frame(&mut stream, "ping").await?;
        // let reply = read_text_frame(&mut stream).await?;
        // debug!("main received: {reply:?}");
        // assert_eq!(reply, "pong");
        //
        // write_text_frame(&mut stream, "status").await?;
        // let reply = read_text_frame(&mut stream).await?;
        // debug!("main received: {reply:?}");
        // assert_eq!(reply, "worker-status: alive");
        //
        // write_text_frame(&mut stream, "echo hello from main").await?;
        // let reply = read_text_frame(&mut stream).await?;
        // debug!("main received: {reply:?}");
        // assert_eq!(reply, "echo-reply hello from main");
        //
        // write_text_frame(&mut stream, "shutdown").await?;
        // let reply = read_text_frame(&mut stream).await?;
        // debug!("main received: {reply:?}");
        // assert_eq!(reply, "shutdown-ok");


        /////


        /*
        let device_idx = 1;
        const MUL_K: f64 = 2.;

        let size = [3, 10];
        let load_and_mut = Arc::new(LoadAndClcTask::new(
            LoadAndClcTaskId("load_and_mul_id"),
            MUL_K,
            vec![],
            size.into(),
            ClcOOMRunner::new(
                OOMRunnerConfig::MaxParallel { max: 20, step: 10 },
                Box::new(move |id| {
                    SingleDimTensorOutputBuilder::<SynKey, ClcTaskOutput>::new(
                        id,
                        1,
                        &size,
                        (Float, Device::Cuda(0)),
                        BuildAs::Append,
                    ).expect("single dim output builder")
                }),
            ),
            Float,
            Device::Cuda(0),
            false,
        ));

        run_mul_task_invocations(
            Arc::clone(&load_and_mut),
            &["resource_1", "resource_2", "resource_3"],
            device_idx,
            |_device_idx, _cache_area, _name| RunCtxImpl::new(1, "noname"),
            move |res, out| {
                crate::torch_runner::tests::mul_check(MUL_K, size.into(), res, out, 0.1, 0.1);
            },
        ).await?;*/




        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_cuda_process() {
        debug!("test_cuda_process, IPC_SOCK_ENV={:?}",std::env::var_os(IPC_SOCK_ENV));

        if std::env::var_os(IPC_SOCK_ENV).is_some() {
            // CudaProcess::new().unwrap();//will be called from run_cli //crates/thoosa_main/src/cli.rs:339, i.e. if app is started as "worker"
            CudaWorker::run_worker().await.unwrap()
        } else {
            run_main("test_cuda_process").await.unwrap();
        }
    }

}
