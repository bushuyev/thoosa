use interprocess::local_socket::{
    tokio::{prelude::*, Stream},
    GenericFilePath, GenericNamespaced, ListenerOptions, Name, ToFsName, ToNsName,
};
use log::{debug, warn};
use std::io;
use std::time::Duration;
use thoosa_shared::collect_futures_eta::{collect_eta_set_device_stats, collect_eta_subscribe_progress, CollectEtaProgressState, COLLECT_ETA_PROGRESS_SOCKET_ID};
use thoosa_shared::device_stats::ThoDeviceStats;
use thoosa_shims::device_stats::get_device_stats;
use torch_sys::cuda::atc_cuda_device_count;
use tokio::io::{AsyncWrite, AsyncWriteExt};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tracing::Instrument;
use thoosa_shared::tracked_await::track_await;

type SocketResult<T> = Result<T, Box<dyn std::error::Error + Send + Sync>>;

pub(crate) struct CollectEtaProgressSocketServer {
    name: Name<'static>,
    shutdown_tx: watch::Sender<bool>,
    task: Option<JoinHandle<()>>,
    stats_task: Option<JoinHandle<()>>,
}

impl CollectEtaProgressSocketServer {
    pub(crate) fn start_default() -> io::Result<Self> {
        let mut server = Self::start_with_socket_id(COLLECT_ETA_PROGRESS_SOCKET_ID, collect_eta_subscribe_progress())?;
        server.stats_task = Some(server.spawn_device_stats_task());
        Ok(server)
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
                                    if let Err(err) = serve_one_client(stream, rx).instrument(tracing::debug_span!("start_with_name/1", file = file!(), line = line!())).await {
                                        debug!("collect ETA socket client disconnected: {err}");
                                    }
                                }.instrument(tracing::info_span!(
                                    "progress_socket::client",
                                    kind = "spawn",
                                    file = file!(),
                                    line = line!(),
                                    task = "serve collect ETA progress socket client",
                                )));
                            }
                            Err(err) => {
                                warn!("collect ETA socket accept failed: {err}");
                                break;
                            }
                        }
                    }
                }
            }
        }.instrument(tracing::info_span!(
            "progress_socket::server",
            kind = "spawn",
            file = file!(),
            line = line!(),
            task = "accept collect ETA progress socket clients",
        )));

        Ok(Self {
            name,
            shutdown_tx,
            task: Some(task),
            stats_task: None,
        })
    }

    fn spawn_device_stats_task(&self) -> JoinHandle<()> {
        let mut shutdown_rx = self.shutdown_tx.subscribe();
        tokio::spawn(async move {
            loop {
                collect_eta_set_device_stats(read_device_stats_snapshot());
                tokio::select! {
                    changed = shutdown_rx.changed() => {
                        if changed.is_err() || *shutdown_rx.borrow() {
                            break;
                        }
                    }
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {}
                }
            }
        }.instrument(tracing::info_span!(
            "progress_socket::device_stats",
            kind = "spawn",
            file = file!(),
            line = line!(),
            task = "publish device stats snapshots",
        )))
    }

    #[cfg(test)]
    fn name(&self) -> Name<'_> {
        self.name.borrow()
    }

    pub(crate) async fn shutdown(mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(task) = self.task.take() {
            let _ = track_await("shutdown/1", task).instrument(tracing::debug_span!("shutdown/1", file = file!(), line = line!())).await;
        }
        if let Some(stats_task) = self.stats_task.take() {
            let _ = track_await("shutdown/2", stats_task).instrument(tracing::debug_span!("shutdown/2", file = file!(), line = line!())).await;
        }
    }
}

impl Drop for CollectEtaProgressSocketServer {
    fn drop(&mut self) {
        let _ = self.shutdown_tx.send(true);
        if let Some(task) = self.task.take() {
            task.abort();
        }
        if let Some(stats_task) = self.stats_task.take() {
            stats_task.abort();
        }
    }
}

fn read_device_stats_snapshot() -> Option<ThoDeviceStats> {
    let device_count = unsafe { atc_cuda_device_count() };
    if device_count < 1 {
        return None;
    }

    let mut total = ThoDeviceStats::default();
    for device in 0..device_count {
        match get_device_stats(device) {
            Ok(device_stats) => {
                total = total.saturating_add(device_stats);
            }
            Err(err) => {
                debug!("failed to read device stats for device={device}: {}", err.msg);
                return None;
            }
        }
    }

    Some(total)
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
    track_await("serve_one_client/1", send_snapshot(&mut writer, &first)).instrument(tracing::debug_span!("serve_one_client/1", file = file!(), line = line!())).await?;

    loop {
        if track_await("serve_one_client/2", rx.changed()).instrument(tracing::debug_span!("serve_one_client/2", file = file!(), line = line!())).await.is_err() {
            break;
        }

        let snapshot = rx.borrow_and_update().clone();
        track_await("serve_one_client/3", send_snapshot(&mut writer, &snapshot)).instrument(tracing::debug_span!("serve_one_client/3", file = file!(), line = line!())).await?;
    }

    Ok(())
}

async fn send_snapshot<W>(writer: &mut W, snapshot: &CollectEtaProgressState) -> SocketResult<()>
where
    W: AsyncWrite + Unpin,
{
    let mut bytes = serde_json::to_vec(snapshot)?;
    bytes.push(b'\n');
    track_await("send_snapshot/1", writer.write_all(&bytes)).instrument(tracing::debug_span!("send_snapshot/1", file = file!(), line = line!())).await?;
    track_await("send_snapshot/2", writer.flush()).instrument(tracing::debug_span!("send_snapshot/2", file = file!(), line = line!())).await?;
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
            started_at_unix_ms: 111,
            oom_all_failed: 1,
            oom_success: 0,
            device_stats: None,
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
        assert_eq!(second.started_at_unix_ms, 111);
        assert_eq!(second.oom_all_failed, 1);

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
            started_at_unix_ms: 222,
            oom_all_failed: 2,
            oom_success: 0,
            device_stats: None,
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
        assert_eq!(third.started_at_unix_ms, 222);
        assert_eq!(third.oom_all_failed, 2);

        server.shutdown().await;
        Ok(())
    }
}
