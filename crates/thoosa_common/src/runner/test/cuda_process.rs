use std::path::PathBuf;
use std::process::Stdio;
use log::debug;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::process::{Child, Command};
use thoosa_shared::ThoError;

const IPC_SOCK_ENV: &str = "THOOSA_WORKER_IPC_SOCK";

async fn spawn_worker() -> Result<(Child, UnixStream), ThoError> {
    let exe = std::env::current_exe()?;

    let dir = tempfile::tempdir()?;
    let sock_path = dir.path().join("worker.sock");

    debug!("spawn_worker, exe={exe:?} dir={dir:?} sock_path={sock_path:?}");

    let listener = UnixListener::bind(&sock_path)?;

    debug!("spawn_worker, listener={listener:?}");

    let test_name = concat!(module_path!(), "::test_cuda_process");

    debug!("spawn_worker, test_name={test_name:?}");

    let mut child = Command::new(exe)
        // .arg(test_name)
        .arg("runner::test::cuda_process::test_cuda_process")
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
        Err(_) => {
            match child.try_wait()? {
                Some(status) => {
                    return Err(ThoError::new(format!(
                        "worker exited before connecting: status={status}"
                    )));
                }
                None => {
                    let _ = child.kill().await;
                    return Err(ThoError::new(
                        "worker did not connect within timeout",
                    ));
                }
            }
        }
    };

    debug!("spawn_worker, _addr={_addr:?}");

    drop(dir);

    Ok((child, stream))
}
async fn run_worker() -> Result<(), ThoError> {
    let sock_path = std::env::var_os(IPC_SOCK_ENV)
        .ok_or_else(|| ThoError::new("missing THOOSA_WORKER_IPC_SOCK"))?;

    let mut stream = UnixStream::connect(PathBuf::from(sock_path)).await?;

    debug!("worker connected");

    stream.write_all(b"hello from worker\n").await?;

    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf).await?;

    debug!("worker received: {:?}", std::str::from_utf8(&buf[..n]).unwrap_or("<non-utf8>"));

    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_cuda_process() {
    debug!("test_0, IPC_SOCK_ENV={:?}", std::env::var_os(IPC_SOCK_ENV));
    if std::env::var_os(IPC_SOCK_ENV).is_some() {
        run_worker().await.unwrap();
    } else {
        run_main().await.unwrap();
    }
}
async fn run_main() -> Result<(), ThoError> {
    let (mut child, mut stream) = spawn_worker().await?;

    let mut buf = [0u8; 1024];
    let n = stream.read(&mut buf).await?;

    debug!("main received: {:?}", std::str::from_utf8(&buf[..n]).unwrap_or("<non-utf8>"));

    stream.write_all(b"hello from main\n").await?;

    let status = child.wait().await?;
    assert!(status.success());

    Ok(())
}

async fn write_frame(
    w: &mut UnixStream,
    payload: &[u8],
) -> std::io::Result<()> {
    let len = payload.len() as u32;
    w.write_all(&len.to_le_bytes()).await?;
    w.write_all(payload).await?;
    Ok(())
}

async fn read_frame(
    r: &mut UnixStream,
) -> std::io::Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;

    let len = u32::from_le_bytes(len_buf) as usize;
    let mut payload = vec![0u8; len];

    r.read_exact(&mut payload).await?;
    Ok(payload)
}