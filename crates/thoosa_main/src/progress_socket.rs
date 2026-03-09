use interprocess::local_socket::{
    tokio::{prelude::*, Stream},
    GenericFilePath, GenericNamespaced, ListenerOptions, Name, ToFsName, ToNsName,
};
use log::{debug, warn};
use std::io;
use std::time::Duration;
use thoosa_shared::collect_futures_eta::{collect_eta_subscribe_progress, CollectEtaProgressState, COLLECT_ETA_PROGRESS_SOCKET_ID};
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::sync::watch;
use tokio::task::JoinHandle;

type SocketResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub(crate) struct CollectEtaProgressSocketServer {
    name: Name<'static>,
    shutdown_tx: watch::Sender<bool>,
    task: Option<JoinHandle<()>>,
}

impl CollectEtaProgressSocketServer {
    pub(crate) fn start_default() -> io::Result<Self> {
        Self::start_with_socket_id(COLLECT_ETA_PROGRESS_SOCKET_ID, collect_eta_subscribe_progress())
    }

    fn start_with_socket_id(
        socket_id: &str,
        snapshot_rx: watch::Receiver<CollectEtaProgressState>,
    ) -> io::Result<Self> {
        let mut last_err = None;
        for name in socket_names(socket_id)? {
            match Self::start_with_name(name.clone(), snapshot_rx.clone()) {
                Ok(server) => return Ok(server),
                Err(err) if should_try_fallback(&err) => {
                    debug!("collect ETA socket bind fallback for {socket_id}: {err}");
                    last_err = Some(err);
                }
                Err(err) => return Err(err),
            }
        }

        Err(last_err.unwrap_or_else(|| io::Error::other("no local socket names available")))
    }

    fn start_with_name(
        name: Name<'static>,
        snapshot_rx: watch::Receiver<CollectEtaProgressState>,
    ) -> io::Result<Self> {
        let listener = ListenerOptions::new()
            .name(name.clone())
            .try_overwrite(true)
            .max_spin_time(Duration::from_millis(100))
            .create_tokio()?;
        let (shutdown_tx, mut shutdown_rx) = watch::channel(false);
        let task = tokio::spawn(async move {
            loop {
                tokio::select! {
                    changed = shutdown_rx.changed() => {
                        if changed.is_err() || *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    accepted = listener.accept() => {
                        match accepted {
                            Ok(stream) => {
                                let rx = snapshot_rx.clone();
                                tokio::spawn(async move {
                                    if let Err(err) = serve_one_client(stream, rx).await {
                                        debug!("collect ETA socket client disconnected: {err}");
                                    }
                                });
                            }
                            Err(err) => {
                                warn!("collect ETA socket accept failed: {err}");
                                break;
                            }
                        }
                    }
                }
            }
        });

        Ok(Self {
            name,
            shutdown_tx,
            task: Some(task),
        })
    }

    #[cfg(test)]
    fn name(&self) -> Name<'_> {
        self.name.borrow()
    }

    pub(crate) async fn shutdown(mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }
}

impl Drop for CollectEtaProgressSocketServer {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

fn socket_names(socket_id: &str) -> io::Result<Vec<Name<'static>>> {
    let mut names = Vec::with_capacity(2);
    if GenericNamespaced::is_supported() {
        names.push(format!("{socket_id}.sock").to_ns_name::<GenericNamespaced>()?);
    }
    if cfg!(windows) {
        names.push(format!(r"\\.\pipe\{socket_id}").to_fs_name::<GenericFilePath>()?);
    } else {
        names.push(format!("/tmp/{socket_id}.sock").to_fs_name::<GenericFilePath>()?);
    }
    Ok(names)
}

fn should_try_fallback(err: &io::Error) -> bool {
    matches!(err.kind(), io::ErrorKind::PermissionDenied | io::ErrorKind::AddrNotAvailable | io::ErrorKind::InvalidInput)
}

async fn serve_one_client(
    stream: Stream,
    mut rx: watch::Receiver<CollectEtaProgressState>,
) -> SocketResult<()> {
    let mut writer = &stream;
    let first = rx.borrow_and_update().clone();
    send_snapshot(&mut writer, &first).await?;

    loop {
        if rx.changed().await.is_err() {
            break;
        }

        let snapshot = rx.borrow_and_update().clone();
        send_snapshot(&mut writer, &snapshot).await?;
    }

    Ok(())
}

async fn send_snapshot<W>(writer: &mut W, snapshot: &CollectEtaProgressState) -> SocketResult<()>
where
    W: AsyncWrite + Unpin,
{
    let mut bytes = serde_json::to_vec(snapshot)?;
    bytes.push(b'\n');
    writer.write_all(&bytes).await?;
    writer.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{CollectEtaProgressSocketServer, SocketResult};
    use interprocess::local_socket::tokio::{prelude::*, Stream};
    use std::process;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;
    use thoosa_shared::collect_futures_eta::{CollectEtaProgressInvocation, CollectEtaProgressState};
    use tokio::io::{AsyncBufReadExt, BufReader};
    use tokio::sync::watch;
    use tokio::time::timeout;

    static NEXT_TEST_SOCKET_ID: AtomicU64 = AtomicU64::new(1);

    fn next_test_socket_id() -> String {
        let suffix = NEXT_TEST_SOCKET_ID.fetch_add(1, Ordering::Relaxed);
        format!("thoosa-progress-test-{}-{suffix}", process::id())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn test_client_receives_progress_snapshots() -> SocketResult<()> {
        let (snapshot_tx, snapshot_rx) = watch::channel(CollectEtaProgressState::default());
        let server = CollectEtaProgressSocketServer::start_with_socket_id(&next_test_socket_id(), snapshot_rx)?;
        let stream = timeout(Duration::from_secs(2), Stream::connect(server.name())).await??;
        let mut lines = BufReader::new(stream).lines();

        let first = timeout(Duration::from_secs(2), lines.next_line()).await??.expect("first line");
        let first: CollectEtaProgressState = serde_json::from_str(&first)?;
        assert!(first.invocations.is_empty());

        snapshot_tx.send(CollectEtaProgressState {
            invocations: vec![CollectEtaProgressInvocation {
                descr: "build_features".to_string(),
                total: 10,
                completed: 3,
            }],
        })?;

        let second = timeout(Duration::from_secs(2), lines.next_line()).await??.expect("second line");
        let second: CollectEtaProgressState = serde_json::from_str(&second)?;
        assert_eq!(
            second.invocations,
            vec![CollectEtaProgressInvocation {
                descr: "build_features".to_string(),
                total: 10,
                completed: 3,
            }]
        );

        snapshot_tx.send(CollectEtaProgressState {
            invocations: vec![
                CollectEtaProgressInvocation {
                    descr: "build_features".to_string(),
                    total: 10,
                    completed: 7,
                },
                CollectEtaProgressInvocation {
                    descr: "train_net".to_string(),
                    total: 4,
                    completed: 1,
                },
            ],
        })?;

        let third = timeout(Duration::from_secs(2), lines.next_line()).await??.expect("third line");
        let third: CollectEtaProgressState = serde_json::from_str(&third)?;
        assert_eq!(
            third.invocations,
            vec![
                CollectEtaProgressInvocation {
                    descr: "build_features".to_string(),
                    total: 10,
                    completed: 7,
                },
                CollectEtaProgressInvocation {
                    descr: "train_net".to_string(),
                    total: 4,
                    completed: 1,
                },
            ]
        );

        server.shutdown().await;
        Ok(())
    }
}
