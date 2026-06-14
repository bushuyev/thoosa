use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use log::debug;
use tch::Device;
use tch::Device::Cuda;
use tch::Kind::Float;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use thoosa_shared::comp_id::SynKey;
use thoosa_shared::{mono_key, ThoError};
use crate::runner::global_ctx::mock_global_ctx::MockGlobalCtxBuilder;
use crate::runner::run_ctx_2::RunCtxImpl;
use crate::runner::test::{run_mul_task_invocations, ClcOOMRunner, OOMRunnerConfig};
use crate::torch_runner::output_builders::{BuildAs, SingleDimTensorOutputBuilder};
use crate::torch_runner::tests::{ClcTask, ClcTaskOutput, LoadAndClcTask, LoadAndClcTaskId};

const IPC_SOCK_ENV: &str = "THOOSA_WORKER_IPC_SOCK";

const MAX_FRAME_LEN: usize = 16 * 1024 * 1024;

macro_rules! pure_module_path {
    () => {{
        let p = module_path!();

        match p.split_once("::") {
            Some((_crate_name, rest)) => rest,
            None => "",
        }
    }};
}

async fn spawn_worker(test_name: &str) -> Result<(Child, UnixStream), ThoError> {
    let exe = std::env::current_exe()?;

    let dir = tempfile::tempdir()?;
    let sock_path = dir.path().join("worker.sock");

    debug!("spawn_worker, exe={exe:?} dir={dir:?} sock_path={sock_path:?}");

    let listener = UnixListener::bind(&sock_path)?;

    debug!("spawn_worker, listener={listener:?}");

    let module = pure_module_path!();

    let test_name = if module.is_empty() {
        test_name.to_string()
    } else {
        format!("{module}::{test_name}")
    };

    debug!("spawn_worker, test_name={test_name:?}");

    let mut child = Command::new(exe)
        .arg(test_name)
        .arg("--exact")
        .arg("--nocapture")
        .env(IPC_SOCK_ENV, &sock_path)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .spawn()?;

    let accepted = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        listener.accept(),
    )
        .await;

    debug!("spawn_worker, accepted={accepted:?}");

    let (stream, _addr) = match accepted {
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

    // The socket file can be removed after accept. The connected UnixStream
    // remains valid.
    drop(dir);

    Ok((child, stream))
}

async fn run_worker() -> Result<(), ThoError> {
    let sock_path = std::env::var_os(IPC_SOCK_ENV)
        .ok_or_else(|| ThoError::new("missing THOOSA_WORKER_IPC_SOCK"))?;

    let mut stream = UnixStream::connect(PathBuf::from(sock_path)).await?;

    debug!("worker connected");

    write_text_frame(&mut stream, "worker-ready").await?;

    let task = ClcTask::new(
        SynKey::new([mono_key!(0, "mul_task_0")]),
        1.0,
        vec![],
        1000,
        Float,
        Cuda(0),
        false,
    );

    loop {
        let msg = read_text_frame(&mut stream).await?;

        debug!("worker received: {msg:?}");

        match msg.as_str() {
            "ping" => {
                write_text_frame(&mut stream, "pong").await?;
            }

            "status" => {
                write_text_frame(&mut stream, "worker-status: alive").await?;
            }

            "shutdown" => {
                write_text_frame(&mut stream, "shutdown-ok").await?;
                debug!("worker shutting down after shutdown command");
                break;
            }

            msg if msg.starts_with("echo ") => {
                let payload = &msg["echo ".len()..];
                let reply = format!("echo-reply {payload}");
                write_text_frame(&mut stream, &reply).await?;
            }

            other => {
                let reply = format!("unknown-command {other:?}");
                write_text_frame(&mut stream, &reply).await?;
            }
        }
    }

    debug!("worker finished");

    Ok(())
}

async fn run_main(test_name: &str) -> Result<(), ThoError> {
    let (mut child, mut stream) = spawn_worker(test_name).await?;


    let ready = read_text_frame(&mut stream).await?;
    debug!("main received: {ready:?}");
    assert_eq!(ready, "worker-ready");

    write_text_frame(&mut stream, "ping").await?;
    let reply = read_text_frame(&mut stream).await?;
    debug!("main received: {reply:?}");
    assert_eq!(reply, "pong");

    write_text_frame(&mut stream, "status").await?;
    let reply = read_text_frame(&mut stream).await?;
    debug!("main received: {reply:?}");
    assert_eq!(reply, "worker-status: alive");

    write_text_frame(&mut stream, "echo hello from main").await?;
    let reply = read_text_frame(&mut stream).await?;
    debug!("main received: {reply:?}");
    assert_eq!(reply, "echo-reply hello from main");

    write_text_frame(&mut stream, "shutdown").await?;
    let reply = read_text_frame(&mut stream).await?;
    debug!("main received: {reply:?}");
    assert_eq!(reply, "shutdown-ok");


    /////
    use crate::runner::global_ctx::GlobalCtx as RealGlobalCtx;
    RealGlobalCtx::init_devices(2);
    let _ctx = MockGlobalCtxBuilder::install_global_ctx_mock(true);

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
    )
        .await?;
    /////


    drop(stream);

    let status = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        child.wait(),
    )
        .await
        .map_err(|_| ThoError::new("worker did not exit after shutdown"))??;

    debug!("worker exited with status={status}");

    assert!(status.success());

    Ok(())
}

async fn write_text_frame(
    stream: &mut UnixStream,
    text: &str,
) -> Result<(), ThoError> {
    write_frame(stream, text.as_bytes()).await?;
    Ok(())
}

async fn read_text_frame(
    stream: &mut UnixStream,
) -> Result<String, ThoError> {
    let payload = read_frame(stream).await?;

    String::from_utf8(payload)
        .map_err(|e| ThoError::new(format!("received non-utf8 frame: {e}")))
}

async fn write_frame(
    w: &mut UnixStream,
    payload: &[u8],
) -> std::io::Result<()> {
    if payload.len() > u32::MAX as usize {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("frame too large: {} bytes", payload.len()),
        ));
    }

    let len = payload.len() as u32;

    w.write_all(&len.to_le_bytes()).await?;
    w.write_all(payload).await?;
    w.flush().await?;

    Ok(())
}

async fn read_frame(
    r: &mut UnixStream,
) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];

    r.read_exact(&mut len_buf).await?;

    let len = u32::from_le_bytes(len_buf) as usize;

    if len > MAX_FRAME_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("frame too large: {len} bytes"),
        ));
    }

    let mut payload = vec![0u8; len];

    r.read_exact(&mut payload).await?;

    Ok(payload)
}



#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cuda_process() {
    debug!(
        "test_cuda_process, IPC_SOCK_ENV={:?}",
        std::env::var_os(IPC_SOCK_ENV)
    );

    if std::env::var_os(IPC_SOCK_ENV).is_some() {
        run_worker().await.unwrap();
    } else {
        run_main("test_cuda_process").await.unwrap();
    }
}
